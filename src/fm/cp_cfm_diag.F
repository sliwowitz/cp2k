!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief used for collecting diagonalization schemes available for cp_cfm_type
!> \note
!>      first version : only one routine right now
!> \author Joost VandeVondele (2003-09)
! **************************************************************************************************
MODULE cp_cfm_diag
   USE cp_cfm_cholesky, ONLY: cp_cfm_cholesky_decompose
   USE cp_cfm_basic_linalg, ONLY: cp_cfm_gemm, &
                                  cp_cfm_column_scale, &
                                  cp_cfm_scale, &
                                  cp_cfm_triangular_invert, &
                                  cp_cfm_triangular_multiply
   USE cp_cfm_types, ONLY: cp_cfm_get_info, &
                           cp_cfm_set_element, &
                           cp_cfm_to_cfm, &
                           cp_cfm_type
#if defined(__DLAF)
   USE cp_cfm_dlaf_api, ONLY: cp_cfm_diag_gen_dlaf, &
                              cp_cfm_diag_dlaf
   USE cp_dlaf_utils_api, ONLY: cp_dlaf_initialize, cp_dlaf_create_grid
   USE cp_fm_diag, ONLY: diag_type, dlaf_neigvec_min, FM_DIAG_TYPE_DLAF
#endif
   USE kinds, ONLY: dp
#if defined (__HAS_IEEE_EXCEPTIONS)
   USE ieee_exceptions, ONLY: ieee_get_halting_mode, &
                              ieee_set_halting_mode, &
                              IEEE_ALL
#endif

#include "../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'cp_cfm_diag'

   PUBLIC :: cp_cfm_heevd, cp_cfm_geeig, cp_cfm_geeig_canon

CONTAINS

! **************************************************************************************************
!> \brief Perform a diagonalisation of a complex matrix
!> \param matrix ...
!> \param eigenvectors ...
!> \param eigenvalues ...
!> \par History
!>      12.2024 Added DLA-Future support [Rocco Meli]
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE cp_cfm_heevd(matrix, eigenvectors, eigenvalues)

      TYPE(cp_cfm_type), INTENT(IN)                      :: matrix, eigenvectors
      REAL(KIND=dp), DIMENSION(:), INTENT(OUT)           :: eigenvalues

      CHARACTER(len=*), PARAMETER                        :: routineN = 'cp_cfm_heevd'

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

#if defined(__DLAF)
      IF (diag_type == FM_DIAG_TYPE_DLAF .AND. matrix%matrix_struct%nrow_global >= dlaf_neigvec_min) THEN
         ! Initialize DLA-Future on-demand; if already initialized, does nothing
         CALL cp_dlaf_initialize()

         ! Create DLAF grid from BLACS context; if already present, does nothing
         CALL cp_dlaf_create_grid(matrix%matrix_struct%context%get_handle())

         CALL cp_cfm_diag_dlaf(matrix, eigenvectors, eigenvalues)
      ELSE
#endif
         CALL cp_cfm_heevd_base(matrix, eigenvectors, eigenvalues)
#if defined(__DLAF)
      END IF
#endif

      CALL timestop(handle)

   END SUBROUTINE cp_cfm_heevd

! **************************************************************************************************
!> \brief Perform a diagonalisation of a complex matrix
!> \param matrix ...
!> \param eigenvectors ...
!> \param eigenvalues ...
!> \par History
!>      - (De)Allocation checks updated (15.02.2011,MK)
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE cp_cfm_heevd_base(matrix, eigenvectors, eigenvalues)

      TYPE(cp_cfm_type), INTENT(IN)            :: matrix, eigenvectors
      REAL(KIND=dp), DIMENSION(:), INTENT(OUT) :: eigenvalues

      CHARACTER(len=*), PARAMETER :: routineN = 'cp_cfm_heevd_base'

      COMPLEX(KIND=dp), DIMENSION(:), POINTER  :: work
      COMPLEX(KIND=dp), DIMENSION(:, :), &
         POINTER                               :: m
      INTEGER                                  :: handle, info, liwork, &
                                                  lrwork, lwork, n
      INTEGER, DIMENSION(:), POINTER           :: iwork
      REAL(KIND=dp), DIMENSION(:), POINTER     :: rwork
#if defined(__parallel)
      INTEGER, DIMENSION(9)                    :: descm, descv
      COMPLEX(KIND=dp), DIMENSION(:, :), &
         POINTER                               :: v
#if defined (__HAS_IEEE_EXCEPTIONS)
      LOGICAL, DIMENSION(5)                    :: halt
#endif
#endif

      CALL timeset(routineN, handle)

      n = matrix%matrix_struct%nrow_global
      m => matrix%local_data
      ALLOCATE (iwork(1), rwork(1), work(1))
      ! work space query
      lwork = -1
      lrwork = -1
      liwork = -1

#if defined(__parallel)
      v => eigenvectors%local_data
      descm(:) = matrix%matrix_struct%descriptor(:)
      descv(:) = eigenvectors%matrix_struct%descriptor(:)
      CALL pzheevd('V', 'U', n, m(1, 1), 1, 1, descm, eigenvalues(1), v(1, 1), 1, 1, descv, &
                   work(1), lwork, rwork(1), lrwork, iwork(1), liwork, info)
      ! The work space query for lwork does not return always sufficiently large values.
      ! Let's add some margin to avoid crashes.
      lwork = CEILING(REAL(work(1), KIND=dp)) + 1000
      ! needed to correct for a bug in scalapack, unclear how much the right number is
      lrwork = CEILING(rwork(1)) + 1000000
      liwork = iwork(1)
#else
      CALL zheevd('V', 'U', n, m(1, 1), SIZE(m, 1), eigenvalues(1), &
                  work(1), lwork, rwork(1), lrwork, iwork(1), liwork, info)
      lwork = CEILING(REAL(work(1), KIND=dp))
      lrwork = CEILING(rwork(1))
      liwork = iwork(1)
#endif

      DEALLOCATE (iwork, rwork, work)
      ALLOCATE (iwork(liwork), rwork(lrwork), work(lwork))

#if defined(__parallel)
! Scalapack takes advantage of IEEE754 exceptions for speedup.
! Therefore, we disable floating point traps temporarily.
#if defined (__HAS_IEEE_EXCEPTIONS)
      CALL ieee_get_halting_mode(IEEE_ALL, halt)
      CALL ieee_set_halting_mode(IEEE_ALL, .FALSE.)
#endif

      CALL pzheevd('V', 'U', n, m(1, 1), 1, 1, descm, eigenvalues(1), v(1, 1), 1, 1, descv, &
                   work(1), lwork, rwork(1), lrwork, iwork(1), liwork, info)

#if defined (__HAS_IEEE_EXCEPTIONS)
      CALL ieee_set_halting_mode(IEEE_ALL, halt)
#endif
#else
      CALL zheevd('V', 'U', n, m(1, 1), SIZE(m, 1), eigenvalues(1), &
                  work(1), lwork, rwork(1), lrwork, iwork(1), liwork, info)
      eigenvectors%local_data = matrix%local_data
#endif

      DEALLOCATE (iwork, rwork, work)
      IF (info /= 0) &
         CPABORT("Diagonalisation of a complex matrix failed")

      CALL timestop(handle)

   END SUBROUTINE cp_cfm_heevd_base

! **************************************************************************************************
!> \brief General Eigenvalue Problem  AX = BXE
!>        Single option version: Cholesky decomposition of B
!> \param amatrix ...
!> \param bmatrix ...
!> \param eigenvectors ...
!> \param eigenvalues ...
!> \param work ...
!> \par History
!>      12.2024 Added DLA-Future support [Rocco Meli]
! **************************************************************************************************
   SUBROUTINE cp_cfm_geeig(amatrix, bmatrix, eigenvectors, eigenvalues, work)

      TYPE(cp_cfm_type), INTENT(IN)                      :: amatrix, bmatrix, eigenvectors
      REAL(KIND=dp), DIMENSION(:)                        :: eigenvalues
      TYPE(cp_cfm_type), INTENT(IN)                      :: work

      CHARACTER(len=*), PARAMETER                        :: routineN = 'cp_cfm_geeig'

      INTEGER                                            :: handle, nao, nmo
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: evals

      CALL timeset(routineN, handle)

      CALL cp_cfm_get_info(amatrix, nrow_global=nao)
      ALLOCATE (evals(nao))
      nmo = SIZE(eigenvalues)

#if defined(__DLAF)
      IF (diag_type == FM_DIAG_TYPE_DLAF .AND. amatrix%matrix_struct%nrow_global >= dlaf_neigvec_min) THEN
         ! Initialize DLA-Future on-demand; if already initialized, does nothing
         CALL cp_dlaf_initialize()

         ! Create DLAF grid from BLACS context; if already present, does nothing
         CALL cp_dlaf_create_grid(amatrix%matrix_struct%context%get_handle())
         CALL cp_dlaf_create_grid(bmatrix%matrix_struct%context%get_handle())
         CALL cp_dlaf_create_grid(eigenvectors%matrix_struct%context%get_handle())

         ! Use DLA-Future generalized eigenvalue solver for large matrices
         CALL cp_cfm_diag_gen_dlaf(amatrix, bmatrix, work, evals)
      ELSE
#endif
         ! Cholesky decompose S=U(T)U
         CALL cp_cfm_cholesky_decompose(bmatrix)
         ! Invert to get U^(-1)
         CALL cp_cfm_triangular_invert(bmatrix)
         ! Reduce to get U^(-T) * H * U^(-1)
         CALL cp_cfm_triangular_multiply(bmatrix, amatrix, side="R")
         CALL cp_cfm_triangular_multiply(bmatrix, amatrix, transa_tr="C")
         ! Diagonalize
         CALL cp_cfm_heevd(matrix=amatrix, eigenvectors=work, eigenvalues=evals)
         ! Restore vectors C = U^(-1) * C*
         CALL cp_cfm_triangular_multiply(bmatrix, work)
#if defined(__DLAF)
      END IF
#endif

      CALL cp_cfm_to_cfm(work, eigenvectors, nmo)
      eigenvalues(1:nmo) = evals(1:nmo)

      DEALLOCATE (evals)

      CALL timestop(handle)

   END SUBROUTINE cp_cfm_geeig

! **************************************************************************************************
!> \brief General Eigenvalue Problem  AX = BXE
!>        Use canonical orthogonalization
!> \param amatrix ...
!> \param bmatrix ...
!> \param eigenvectors ...
!> \param eigenvalues ...
!> \param work ...
!> \param epseig ...
! **************************************************************************************************
   SUBROUTINE cp_cfm_geeig_canon(amatrix, bmatrix, eigenvectors, eigenvalues, work, epseig)

      TYPE(cp_cfm_type), INTENT(IN)                      :: amatrix, bmatrix, eigenvectors
      REAL(KIND=dp), DIMENSION(:), INTENT(OUT)           :: eigenvalues
      TYPE(cp_cfm_type), INTENT(IN)                      :: work
      REAL(KIND=dp), INTENT(IN)                          :: epseig

      CHARACTER(len=*), PARAMETER :: routineN = 'cp_cfm_geeig_canon'
      COMPLEX(KIND=dp), PARAMETER :: cone = CMPLX(1.0_dp, 0.0_dp, KIND=dp), &
         czero = CMPLX(0.0_dp, 0.0_dp, KIND=dp)

      COMPLEX(KIND=dp), ALLOCATABLE, DIMENSION(:)        :: cevals
      INTEGER                                            :: handle, i, icol, irow, nao, nc, ncol, &
                                                            nmo, nx
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: evals

      CALL timeset(routineN, handle)

      ! Test sizes
      CALL cp_cfm_get_info(amatrix, nrow_global=nao)
      nmo = SIZE(eigenvalues)
      ALLOCATE (evals(nao), cevals(nao))

      ! Diagonalize -S matrix, this way the NULL space is at the end of the spectrum
      CALL cp_cfm_scale(-cone, bmatrix)
      CALL cp_cfm_heevd(bmatrix, work, evals)
      evals(:) = -evals(:)
      nc = nao
      DO i = 1, nao
         IF (evals(i) < epseig) THEN
            nc = i - 1
            EXIT
         END IF
      END DO
      CPASSERT(nc /= 0)

      IF (nc /= nao) THEN
         IF (nc < nmo) THEN
            ! Copy NULL space definition to last vectors of eigenvectors (if needed)
            ncol = nmo - nc
            CALL cp_cfm_to_cfm(work, eigenvectors, ncol, nc + 1, nc + 1)
         END IF
         ! Set NULL space in eigenvector matrix of S to zero
         DO icol = nc + 1, nao
            DO irow = 1, nao
               CALL cp_cfm_set_element(work, irow, icol, czero)
            END DO
         END DO
         ! Set small eigenvalues to a dummy save value
         evals(nc + 1:nao) = 1.0_dp
      END IF
      ! Calculate U*s**(-1/2)
      cevals(:) = CMPLX(1.0_dp/SQRT(evals(:)), 0.0_dp, KIND=dp)
      CALL cp_cfm_column_scale(work, cevals)
      ! Reduce to get U^(-C) * H * U^(-1)
      CALL cp_cfm_gemm("C", "N", nao, nao, nao, cone, work, amatrix, czero, bmatrix)
      CALL cp_cfm_gemm("N", "N", nao, nao, nao, cone, bmatrix, work, czero, amatrix)
      IF (nc /= nao) THEN
         ! set diagonal values to save large value
         DO icol = nc + 1, nao
            CALL cp_cfm_set_element(amatrix, icol, icol, CMPLX(10000.0_dp, 0.0_dp, KIND=dp))
         END DO
      END IF
      ! Diagonalize
      CALL cp_cfm_heevd(amatrix, bmatrix, evals)
      eigenvalues(1:nmo) = evals(1:nmo)
      nx = MIN(nc, nmo)
      ! Restore vectors C = U^(-1) * C*
      CALL cp_cfm_gemm("N", "N", nao, nx, nc, cone, work, bmatrix, czero, eigenvectors)

      DEALLOCATE (evals)

      CALL timestop(handle)

   END SUBROUTINE cp_cfm_geeig_canon

END MODULE cp_cfm_diag
