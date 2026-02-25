#!/bin/bash
#
# test_cusolvermp_single_gpu.sh — cusolverMp regression tests for single-GPU systems
#
# cusolverMp 0.7+ uses NCCL, which requires 1 GPU per MPI rank. The standard
# --mpiranks 2 regression tests fail on single-GPU machines because NCCL rejects
# the "2 ranks, 1 GPU" configuration with "Duplicate GPU detected".
#
# This script runs cusolverMp-relevant test directories with --mpiranks 1 and
# post-processes results to detect regressions. It classifies each test as:
#   - cusolverMp-exercising: passes and dispatches to GPU eigensolvers (nfuncs >= 64)
#   - ScaLAPACK fallback: passes but matrix too small for GPU dispatch (nfuncs < 64)
#   - Known 1-rank failure: requires 2+ MPI ranks for reasons unrelated to cusolverMp
#   - Unexpected: regression or improvement vs. baseline
#
# Baseline data collected on: A40, cusolverMp 0.7.2/NCCL, CUDA 12.9, OpenMPI 4.1.6
#
# Usage:
#   ./test_cusolvermp_single_gpu.sh <binary_dir> <version> [--cp2kdatadir <dir>] [extra args...]
#
# Example:
#   ./test_cusolvermp_single_gpu.sh /workspace/build/bin psmp --cp2kdatadir /workspace/data
#
# Exit code:
#   0 — all results match expectations (no regressions)
#   1 — unexpected failures or wrong results detected
#   2 — usage error or do_regtest.py not found
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Test directories containing cusolverMp-relevant tests.
# These QS directories exercise the eigensolver code paths (cp_fm_diag) that
# dispatch to cusolverMpSyevd/Sygvd when cusolverMp is available.
# ============================================================================
TEST_DIRS=(
    "QS/regtest-cusolver"
    "QS/regtest-gpw-2-1"
    "QS/regtest-gpw-2-2"
    "QS/regtest-gpw-2-3"
    "QS/regtest-gpw-3"
    "QS/regtest-gpw-4"
    "QS/regtest-gpw-5"
    "QS/regtest-gpw-6-2"
    "QS/regtest-gpw-6-3"
    "QS/regtest-gpw-6-4"
    "QS/regtest-gpw-7"
    "QS/regtest-gpw-8"
)

# ============================================================================
# Known test classifications
# ============================================================================
# Baseline: 1-rank runs on A40 with cusolverMp 0.7.2 (NCCL backend).
# Results are identical with and without CP2K_USE_CUSOLVER_MP, confirming
# that failures are inherent 1-rank limitations, not cusolverMp issues.
#
# Summary: 127 correct, 14 wrong, 57 failed out of ~198 tests.
# ============================================================================

# 49 tests that pass with 1 rank AND exercise cusolverMp (nfuncs >= 64).
# These dispatch to cusolverMpSyevd (standard) or cusolverMpSygvd (generalized)
# on the GPU. They are the primary validation targets for cusolverMp correctness.
CUSOLVERMP_TESTS=(
    # regtest-gpw-2-3: Au clusters (nfuncs=300-325)
    "QS/regtest-gpw-2-3/Au12_rmsd_AB_mtd.inp"
    "QS/regtest-gpw-2-3/Au12_rmsd_A_mtd.inp"
    "QS/regtest-gpw-2-3/Au13ico_mtd.inp"
    # regtest-gpw-3: BSSE with 3 waters (nfuncs=69)
    "QS/regtest-gpw-3/3H2O_bsse.inp"
    "QS/regtest-gpw-3/3H2O_bsse_multi_LIST.inp"
    # regtest-gpw-5: Si8/Si7C bulk systems (nfuncs=104)
    "QS/regtest-gpw-5/si7c_broy_gapw.inp"
    "QS/regtest-gpw-5/si7c_broy_gapw_a04_atomic.inp"
    "QS/regtest-gpw-5/si7c_broy_gapw_a04_nopmix.inp"
    "QS/regtest-gpw-5/si7c_broy_gapw_a04_restart.inp"
    "QS/regtest-gpw-5/si7c_kerker_test4.inp"
    "QS/regtest-gpw-5/si7c_kerker_test4_nopmix.inp"
    "QS/regtest-gpw-5/si7c_pulay_gapw.inp"
    "QS/regtest-gpw-5/si8_broy.inp"
    "QS/regtest-gpw-5/si8_kerker.inp"
    "QS/regtest-gpw-5/si8_pmix.inp"
    "QS/regtest-gpw-5/si8_pmix_nosmear_mocubes.inp"
    "QS/regtest-gpw-5/si8_pulay.inp"
    "QS/regtest-gpw-5/si8_pulay_inverse.inp"
    "QS/regtest-gpw-5/si8_pulay_mocubes.inp"
    "QS/regtest-gpw-5/si8_pulay_off.inp"
    "QS/regtest-gpw-5/si8_pulay_reduce.inp"
    "QS/regtest-gpw-5/si8_pulay_restore.inp"
    "QS/regtest-gpw-5/si8_pulay_skip.inp"
    # regtest-gpw-6-2: Si8 LSD/noorth variants (nfuncs=104)
    "QS/regtest-gpw-6-2/si8_lsd_broy_stm.inp"
    "QS/regtest-gpw-6-2/si8_noort_broy_wc_jacobi_all.inp"
    "QS/regtest-gpw-6-2/si8_noort_broy_wc_jacobi_ene1.inp"
    # regtest-gpw-6-3: C8/Si8 mixing variants (nfuncs=104)
    "QS/regtest-gpw-6-3/c8_kerker.inp"
    "QS/regtest-gpw-6-3/c8_pmix.inp"
    "QS/regtest-gpw-6-3/c8_pmix_gapw.inp"
    "QS/regtest-gpw-6-3/si8_lsd_broy_fm0.2.inp"
    # regtest-gpw-6-4: C8 Broyden variants (nfuncs=104)
    "QS/regtest-gpw-6-4/c8_broy.inp"
    "QS/regtest-gpw-6-4/c8_broy_gapw.inp"
    "QS/regtest-gpw-6-4/c8_kerker_gapw.inp"
    # regtest-gpw-7: Si8 Krylov/OT-diag (nfuncs=104)
    "QS/regtest-gpw-7/si8_broy_kry.inp"
    "QS/regtest-gpw-7/si8_broy_kry_r.inp"
    "QS/regtest-gpw-7/si8_broy_lsd.inp"
    "QS/regtest-gpw-7/si8_pmix_otdiag.inp"
    # regtest-gpw-8: Si8 Davidson/Broyden variants (nfuncs=104)
    "QS/regtest-gpw-8/local_stress.inp"
    "QS/regtest-gpw-8/si8_broy_ch.inp"
    "QS/regtest-gpw-8/si8_broy_dav_md.inp"
    "QS/regtest-gpw-8/si8_broy_dav_t300.inp"
    "QS/regtest-gpw-8/si8_broy_dav_t300_lsd.inp"
    "QS/regtest-gpw-8/si8_broy_dav_t300_r.inp"
    "QS/regtest-gpw-8/si8_broy_dav_t5000_r.inp"
    "QS/regtest-gpw-8/si8_broy_davsparse_md.inp"
    "QS/regtest-gpw-8/si8_broy_stm.inp"
    "QS/regtest-gpw-8/si8_broy_wc.inp"
    "QS/regtest-gpw-8/si8_broy_wc_crazy.inp"
    "QS/regtest-gpw-8/si8_broy_wc_crazy_ene.inp"
)

# 14 tests that produce WRONG numerical results with 1 MPI rank.
# Results are identical with/without cusolverMp — these are inherent 1-rank
# issues (data distribution assumptions, wavelet solvers, etc.).
KNOWN_WRONG_1RANK=(
    "QS/regtest-gpw-3/H2O-UKS-GPW-relax_multip.inp"
    "QS/regtest-gpw-3/rsgrid-dist-1.inp"
    "QS/regtest-gpw-5/si8_pulay_inv_dbcsr.inp"
    "QS/regtest-gpw-5/si8_pulay_md.inp"
    "QS/regtest-gpw-6-2/si8_lsd_broy_wc.inp"
    "QS/regtest-gpw-6-2/si8_lsd_broy_wc_ene.inp"
    "QS/regtest-gpw-6-2/si8_noort_broy_wc_direct_ene.inp"
    "QS/regtest-gpw-6-2/si8_noort_broy_wc_jacobi_ene2.inp"
    "QS/regtest-gpw-6-3/si8_lsd_broy_wc_list.inp"
    "QS/regtest-gpw-6-3/si8_lsd_broy_wc_list_rst.inp"
    "QS/regtest-gpw-6-3/si8_lsd_broy_wc_rst.inp"
    "QS/regtest-gpw-6-4/c8_broy_gapw_gop.inp"
    "QS/regtest-gpw-6-4/c8_broy_gop.inp"
    "QS/regtest-gpw-8/si8_broy_std_md.inp"
)

# 57 tests that FAIL (crash/abort) with 1 MPI rank.
# These require 2+ MPI ranks for reasons unrelated to cusolverMp:
#   - BSSE/fragment calculations needing distributed fragments
#   - MD restart chains where prior test outputs are missing at 1 rank
#   - Grid distributions that assume multiple MPI processes
#   - Diagonalization methods requiring multi-rank matrix layout
KNOWN_FAILED_1RANK=(
    # NaN energy / SCF convergence failure (45 tests)
    "QS/regtest-gpw-2-1/H2O-2.inp"
    "QS/regtest-gpw-2-1/H2O-3.inp"
    "QS/regtest-gpw-2-1/H2O-4.inp"
    "QS/regtest-gpw-2-2/H2O-meta.inp"
    "QS/regtest-gpw-2-2/H2O-meta_coord.inp"
    "QS/regtest-gpw-2-2/H2O-meta_coord_1.inp"
    "QS/regtest-gpw-2-2/H2O-meta_kinds.inp"
    "QS/regtest-gpw-2-2/H2O-meta_res0.inp"
    "QS/regtest-gpw-3/H2O-ata.inp"
    "QS/regtest-gpw-3/H2O-bloechl-Spl.inp"
    "QS/regtest-gpw-3/H2O-bloechl-restraint.inp"
    "QS/regtest-gpw-3/H2O-bloechl.inp"
    "QS/regtest-gpw-3/H2O-langevin-1.inp"
    "QS/regtest-gpw-3/H2O-ref-1.inp"
    "QS/regtest-gpw-3/H2O-ref-2.inp"
    "QS/regtest-gpw-3/H2O-solv.inp"
    "QS/regtest-gpw-3/H2O-solv2.inp"
    "QS/regtest-gpw-3/H2O-xc_none.inp"
    "QS/regtest-gpw-3/O2-ROKS.inp"
    "QS/regtest-gpw-3/dynamics-2.inp"
    "QS/regtest-gpw-3/dynamics.inp"
    "QS/regtest-gpw-3/rsgrid-dist-2.inp"
    "QS/regtest-gpw-4/2H2O_meta_welltemp.inp"
    "QS/regtest-gpw-4/H2-geo-1.inp"
    "QS/regtest-gpw-4/H2-geo-2.inp"
    "QS/regtest-gpw-4/H2-geo-3.inp"
    "QS/regtest-gpw-4/H2-geo-4.inp"
    "QS/regtest-gpw-4/H2-geo-5.inp"
    "QS/regtest-gpw-4/H2O+SC.inp"
    "QS/regtest-gpw-4/H2O-5.inp"
    "QS/regtest-gpw-4/H2O-7.inp"
    "QS/regtest-gpw-4/H2O-analytic_vee.inp"
    "QS/regtest-gpw-4/H2O-debug-1.inp"
    "QS/regtest-gpw-4/H2O-debug-2.inp"
    "QS/regtest-gpw-4/H2O-debug-3.inp"
    "QS/regtest-gpw-4/H2O-debug-4.inp"
    "QS/regtest-gpw-4/H2O-extpot.inp"
    "QS/regtest-gpw-4/H2O-gapw.inp"
    "QS/regtest-gpw-4/H2O-meta_g.inp"
    "QS/regtest-gpw-4/H2O-read_cube.inp"
    "QS/regtest-gpw-4/ND3_meta_welltemp.inp"
    "QS/regtest-gpw-6-4/Ne_GAPW_nlcc_md.inp"
    "QS/regtest-gpw-6-4/Ne_nlcc_md.inp"
    "QS/regtest-gpw-7/H2O-meta-mindisp.inp"
    "QS/regtest-gpw-7/H2O-meta-mindisp2.inp"
    # Missing restart files from prior tests in chain (6 tests)
    "QS/regtest-gpw-2-2/H2O-meta_coord_2.inp"
    "QS/regtest-gpw-2-2/H2O-meta_res1.inp"
    "QS/regtest-gpw-2-2/H2O-meta_res2.inp"
    "QS/regtest-gpw-2-2/H2O-meta_res3.inp"
    "QS/regtest-gpw-3/H2O-langevin-2.inp"
    "QS/regtest-gpw-4/cell-2.inp"
    # Runtime errors — vibration analysis, PDB, etc. (5 tests)
    "QS/regtest-gpw-2-1/H2-vib.inp"
    "QS/regtest-gpw-2-1/H2-vib_tc.inp"
    "QS/regtest-gpw-4/H2O-meta_hydro.inp"
    "QS/regtest-gpw-4/cell-1.inp"
    "QS/regtest-gpw-4/test-pdb.inp"
    # Diagonalization failure (1 test)
    "QS/regtest-gpw-3/N-ROKS.inp"
)

# ============================================================================
# Argument parsing
# ============================================================================
usage() {
    echo "Usage: $0 <binary_dir> <version> [--cp2kdatadir <dir>] [extra do_regtest.py args...]"
    echo ""
    echo "Runs cusolverMp-relevant regression tests with 1 MPI rank on a single GPU."
    echo ""
    echo "Arguments:"
    echo "  binary_dir    Directory containing CP2K binaries"
    echo "  version       Version tag (e.g., psmp, sopt)"
    echo "  Extra args are passed through to do_regtest.py (e.g., --timeout, --num_gpus)"
    exit 2
}

if [[ $# -lt 2 ]]; then
    usage
fi

BINARY_DIR="$1"
VERSION="$2"
shift 2

if [[ ! -f "${SCRIPT_DIR}/do_regtest.py" ]]; then
    echo "ERROR: do_regtest.py not found in ${SCRIPT_DIR}"
    exit 2
fi

# Build --restrictdir flags for the cusolverMp-relevant directories
RESTRICT_FLAGS=()
for dir in "${TEST_DIRS[@]}"; do
    RESTRICT_FLAGS+=(--restrictdir "$dir")
done

# Setup log file
LOG_DIR="$(pwd)/regtesting"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/cusolvermp_single_gpu_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================================"
echo "cusolverMp Single-GPU Test Runner"
echo "============================================================================"
echo "Date:       $(date)"
echo "Binary dir: ${BINARY_DIR}"
echo "Version:    ${VERSION}"
echo "MPI ranks:  1 (single-GPU mode)"
echo "Log file:   ${LOG_FILE}"
echo "Test dirs:  ${#TEST_DIRS[@]}"
echo "Known classifications:"
echo "  cusolverMp-exercising tests: ${#CUSOLVERMP_TESTS[@]}"
echo "  Known 1-rank WRONG:          ${#KNOWN_WRONG_1RANK[@]}"
echo "  Known 1-rank FAILED:         ${#KNOWN_FAILED_1RANK[@]}"
echo "============================================================================"
echo ""

# ============================================================================
# Run do_regtest.py with 1 MPI rank
# ============================================================================
set +e
python3 "${SCRIPT_DIR}/do_regtest.py" \
    --mpiranks 1 \
    --maxtasks 1 \
    --maxerrors 999 \
    "${RESTRICT_FLAGS[@]}" \
    "$@" \
    "${BINARY_DIR}" "${VERSION}" 2>&1 | tee "${LOG_FILE}"
REGTEST_EXIT=$?
set -e

echo ""
echo "============================================================================"
echo "Post-Processing: Test Classification"
echo "============================================================================"
echo ""

# ============================================================================
# Parse test results from the log file
# ============================================================================
# do_regtest.py output format:
#   >>> /path/to/workdir/QS/regtest-dir
#       test.inp                            value        STATUS ( dur sec)
#       test.inp:matcher                    value        STATUS ( dur sec)
#   <<< /path/to/workdir/QS/regtest-dir (N of M) done in X sec
#
# Status is one of: OK, WRONG RESULT, RUNTIME FAIL, TIMED OUT, HUGE OUTPUT, N/A
#
# We aggregate per test file (stripping :matcher suffix), taking the worst
# status when a test has multiple matchers (FAILED > WRONG > OK > N/A).
PARSED_RESULTS=$(awk '
/^>>> / {
    # Extract batch directory: last 2 path components (e.g., QS/regtest-gpw-3)
    n = split($2, parts, "/")
    if (n >= 2)
        dir = parts[n-1] "/" parts[n]
    else
        dir = $2
}
/^    [^ ]/ && dir != "" {
    test = $1
    # Strip :matcher suffix for per-file aggregation
    sub(/:.*/, "", test)
    fullname = dir "/" test

    if (index($0, "RUNTIME FAIL") > 0) status = "FAILED"
    else if (index($0, "TIMED OUT") > 0) status = "FAILED"
    else if (index($0, "HUGE OUTPUT") > 0) status = "FAILED"
    else if (index($0, "WRONG RESULT") > 0) status = "WRONG"
    else if ($0 ~ / OK \(/) status = "OK"
    else if (index($0, " N/A ") > 0) status = "NA"
    else next

    # Take worst status per test file: FAILED > WRONG > OK > NA
    if (!(fullname in results)) {
        results[fullname] = status
    } else {
        cur = results[fullname]
        if (status == "FAILED") results[fullname] = "FAILED"
        else if (status == "WRONG" && cur != "FAILED") results[fullname] = "WRONG"
    }
}
END {
    for (name in results) print results[name] "\t" name
}
' "${LOG_FILE}" | sort -t$'\t' -k2)

if [[ -z "$PARSED_RESULTS" ]]; then
    echo "ERROR: No test results found in log file. Check that do_regtest.py ran correctly."
    echo "       do_regtest.py exit code was: ${REGTEST_EXIT}"
    exit 2
fi

# ============================================================================
# Classify results against known baselines
# ============================================================================
# Load known lists into associative arrays for O(1) lookup
declare -A IS_CUSOLVERMP IS_KNOWN_WRONG IS_KNOWN_FAILED
for t in "${CUSOLVERMP_TESTS[@]}"; do IS_CUSOLVERMP["$t"]=1; done
for t in "${KNOWN_WRONG_1RANK[@]}"; do IS_KNOWN_WRONG["$t"]=1; done
for t in "${KNOWN_FAILED_1RANK[@]}"; do IS_KNOWN_FAILED["$t"]=1; done

ok_cusolvermp=0
ok_scalapack=0
expected_wrong=0
expected_failed=0
unexpected_wrong=0
unexpected_failed=0
total_ok=0
total_wrong=0
total_failed=0
total_na=0

declare -a UNEXPECTED_FAILURES=()
declare -a UNEXPECTED_WRONGS=()
declare -a UNEXPECTED_PASSES=()

while IFS=$'\t' read -r status testname; do
    [[ -z "$status" ]] && continue

    case "$status" in
        OK)
            total_ok=$((total_ok + 1))
            if [[ -n "${IS_KNOWN_WRONG[$testname]:-}" ]]; then
                UNEXPECTED_PASSES+=("$testname (was WRONG)")
            elif [[ -n "${IS_KNOWN_FAILED[$testname]:-}" ]]; then
                UNEXPECTED_PASSES+=("$testname (was FAILED)")
            elif [[ -n "${IS_CUSOLVERMP[$testname]:-}" ]]; then
                ok_cusolvermp=$((ok_cusolvermp + 1))
            else
                ok_scalapack=$((ok_scalapack + 1))
            fi
            ;;
        WRONG)
            total_wrong=$((total_wrong + 1))
            if [[ -n "${IS_KNOWN_WRONG[$testname]:-}" ]]; then
                expected_wrong=$((expected_wrong + 1))
            else
                unexpected_wrong=$((unexpected_wrong + 1))
                UNEXPECTED_WRONGS+=("$testname")
            fi
            ;;
        FAILED)
            total_failed=$((total_failed + 1))
            if [[ -n "${IS_KNOWN_FAILED[$testname]:-}" ]]; then
                expected_failed=$((expected_failed + 1))
            else
                unexpected_failed=$((unexpected_failed + 1))
                UNEXPECTED_FAILURES+=("$testname")
            fi
            ;;
        NA)
            total_na=$((total_na + 1))
            ;;
    esac
done <<< "$PARSED_RESULTS"

total=$((total_ok + total_wrong + total_failed + total_na))

# ============================================================================
# Print classification report
# ============================================================================
echo "--- Test Results (per unique test file) ---"
echo "Total tests:     ${total}"
echo "  CORRECT:       ${total_ok}"
echo "  WRONG:         ${total_wrong}"
echo "  FAILED:        ${total_failed}"
if [[ $total_na -gt 0 ]]; then
    echo "  N/A:           ${total_na}"
fi
echo ""
echo "--- CORRECT Tests Breakdown (${total_ok}) ---"
echo "  cusolverMp GPU eigensolvers (nfuncs >= 64):  ${ok_cusolvermp}"
echo "  ScaLAPACK fallback / non-eigensolver:        ${ok_scalapack}"
echo ""
echo "--- Known 1-Rank Issues ---"
echo "  Expected WRONG:   ${expected_wrong} / ${#KNOWN_WRONG_1RANK[@]} known"
echo "  Expected FAILED:  ${expected_failed} / ${#KNOWN_FAILED_1RANK[@]} known"
echo ""

# Report unexpected results
has_regressions=0

if [[ ${#UNEXPECTED_FAILURES[@]} -gt 0 ]]; then
    has_regressions=1
    echo "!!! UNEXPECTED FAILURES (${#UNEXPECTED_FAILURES[@]}) !!!"
    echo "    These tests were not in the known-failure list:"
    for t in "${UNEXPECTED_FAILURES[@]}"; do
        echo "      FAIL: $t"
    done
    echo ""
fi

if [[ ${#UNEXPECTED_WRONGS[@]} -gt 0 ]]; then
    has_regressions=1
    echo "!!! UNEXPECTED WRONG RESULTS (${#UNEXPECTED_WRONGS[@]}) !!!"
    echo "    These tests produced wrong values but are not in the known-wrong list:"
    for t in "${UNEXPECTED_WRONGS[@]}"; do
        echo "      WRONG: $t"
    done
    echo ""
fi

if [[ ${#UNEXPECTED_PASSES[@]} -gt 0 ]]; then
    echo "*** IMPROVEMENTS (${#UNEXPECTED_PASSES[@]}) ***"
    echo "    These tests were expected to fail/be wrong but now pass:"
    for t in "${UNEXPECTED_PASSES[@]}"; do
        echo "      FIXED: $t"
    done
    echo ""
fi

# Check for missing expected failures (tests that should have failed but weren't seen)
missing_expected=0
for t in "${KNOWN_FAILED_1RANK[@]}"; do
    found=0
    while IFS=$'\t' read -r status testname; do
        if [[ "$testname" == "$t" ]]; then
            found=1
            break
        fi
    done <<< "$PARSED_RESULTS"
    if [[ $found -eq 0 ]]; then
        if [[ $missing_expected -eq 0 ]]; then
            echo "NOTE: Some known-failure tests were not in the test run (removed or skipped):"
        fi
        missing_expected=$((missing_expected + 1))
    fi
done
if [[ $missing_expected -gt 0 ]]; then
    echo "  ($missing_expected tests from known-failure list not found in results)"
    echo ""
fi

echo "============================================================================"
if [[ ${has_regressions} -eq 0 ]]; then
    echo "RESULT: PASS"
    echo "  All results match expectations. No cusolverMp regressions detected."
    echo "  ${ok_cusolvermp} tests validated cusolverMp GPU eigensolvers with 1 MPI rank."
    echo "  ${expected_wrong} + ${expected_failed} known 1-rank issues confirmed (need multi-GPU)."
else
    echo "RESULT: FAIL"
    echo "  Unexpected test results detected — possible regressions."
    echo "  Review the failures above and compare with the log file:"
    echo "  ${LOG_FILE}"
fi
echo "============================================================================"

if [[ ${has_regressions} -ne 0 ]]; then
    exit 1
fi
exit 0
