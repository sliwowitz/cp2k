/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#if defined(__CUSOLVERMP)

#include "../offload/offload_library.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverMp.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#if defined(__CUSOLVERMP_NCCL)
#include <nccl.h>
#else
#include <cal.h>
#endif

/*******************************************************************************
 * \brief Run given CUDA command and upon failure abort with a nice message.
 * \author Ole Schuett
 ******************************************************************************/
#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t status = cmd;                                                  \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "ERROR: %s %s %d\n", cudaGetErrorString(status),         \
              __FILE__, __LINE__);                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#if defined(__CUSOLVERMP_NCCL)
/*******************************************************************************
 * \brief Run given NCCL command and upon failure abort with a nice message.
 * \author Jiri Vyskocil
 ******************************************************************************/
#define NCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t status = cmd;                                                 \
    if (status != ncclSuccess) {                                               \
      fprintf(stderr, "ERROR: %s %s %d\n", ncclGetErrorString(status),         \
              __FILE__, __LINE__);                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#else
/*******************************************************************************
 * \brief Decode given cal error.
 * \author Ole Schuett
 ******************************************************************************/
static char *calGetErrorString(calError_t status) {
  switch (status) {
  case CAL_OK:
    return "CAL_OK";
  case CAL_ERROR:
    return "CAL_ERROR";
  case CAL_ERROR_INVALID_PARAMETER:
    return "CAL_ERROR_INVALID_PARAMETER";
  case CAL_ERROR_INTERNAL:
    return "CAL_ERROR_INTERNAL";
  case CAL_ERROR_CUDA:
    return "CAL_ERROR_CUDA";
  case CAL_ERROR_UCC:
    return "CAL_ERROR_UCC";
  case CAL_ERROR_NOT_SUPPORTED:
    return "CAL_ERROR_NOT_SUPPORTED";
  case CAL_ERROR_INPROGRESS:
    return "CAL_ERROR_INPROGRESS";
  default:
    return "CAL UNKNOWN ERROR";
  }
}

/*******************************************************************************
 * \brief Run given cal command and upon failure abort with a nice message.
 * \author Ole Schuett
 ******************************************************************************/
#define CAL_CHECK(cmd)                                                         \
  do {                                                                         \
    calError_t status = cmd;                                                   \
    if (status != CAL_OK) {                                                    \
      fprintf(stderr, "ERROR: %s %s %d\n", calGetErrorString(status),          \
              __FILE__, __LINE__);                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

/*******************************************************************************
 * \brief Callback for cal library to initiate an allgather operation.
 * \author Ole Schuett
 ******************************************************************************/
static calError_t allgather(void *src_buf, void *recv_buf, size_t size,
                            void *data, void **req) {
  const MPI_Comm comm = *(MPI_Comm *)data;
  MPI_Request *request = malloc(sizeof(MPI_Request));
  *req = request;
  const int status = MPI_Iallgather(src_buf, size, MPI_BYTE, recv_buf, size,
                                    MPI_BYTE, comm, request);
  return (status == MPI_SUCCESS) ? CAL_OK : CAL_ERROR;
}

/*******************************************************************************
 * \brief Callback for cal library to test if a request has completed.
 * \author Ole Schuett
 ******************************************************************************/
static calError_t req_test(void *req) {
  MPI_Request *request = (MPI_Request *)(req);
  int completed;
  const int status = MPI_Test(request, &completed, MPI_STATUS_IGNORE);
  if (status != MPI_SUCCESS) {
    return CAL_ERROR;
  }
  return completed ? CAL_OK : CAL_ERROR_INPROGRESS;
}

/*******************************************************************************
 * \brief Callback for cal library to free a request.
 * \author Ole Schuett
 ******************************************************************************/
static calError_t req_free(void *req) {
  free(req);
  return CAL_OK;
}
#endif /* __CUSOLVERMP_NCCL */

/*******************************************************************************
 * \brief Decode given cusolver error.
 * \author Ole Schuett
 ******************************************************************************/
static char *cusolverGetErrorString(cusolverStatus_t status) {
  switch (status) {
  case CUSOLVER_STATUS_SUCCESS:
    return "CUSOLVER_STATUS_SUCCESS";
  case CUSOLVER_STATUS_NOT_INITIALIZED:
    return "CUSOLVER_STATUS_NOT_INITIALIZED";
  case CUSOLVER_STATUS_ALLOC_FAILED:
    return "CUSOLVER_STATUS_ALLOC_FAILED";
  case CUSOLVER_STATUS_INVALID_VALUE:
    return "CUSOLVER_STATUS_INVALID_VALUE";
  case CUSOLVER_STATUS_ARCH_MISMATCH:
    return "CUSOLVER_STATUS_ARCH_MISMATCH";
  case CUSOLVER_STATUS_MAPPING_ERROR:
    return "CUSOLVER_STATUS_MAPPING_ERROR";
  case CUSOLVER_STATUS_EXECUTION_FAILED:
    return "CUSOLVER_STATUS_EXECUTION_FAILED";
  case CUSOLVER_STATUS_INTERNAL_ERROR:
    return "CUSOLVER_STATUS_INTERNAL_ERROR";
  case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  case CUSOLVER_STATUS_NOT_SUPPORTED:
    return "CUSOLVER_STATUS_NOT_SUPPORTED";
  case CUSOLVER_STATUS_ZERO_PIVOT:
    return "CUSOLVER_STATUS_ZERO_PIVOT";
  case CUSOLVER_STATUS_INVALID_LICENSE:
    return "CUSOLVER_STATUS_INVALID_LICENSE";
  case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:
    return "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
  case CUSOLVER_STATUS_IRS_PARAMS_INVALID:
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
  case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC:
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";
  case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE:
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";
  case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER:
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";
  case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:
    return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
  case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:
    return "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";
  case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:
    return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
  case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES:
    return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES";
  case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:
    return "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";
  case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED:
    return "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";
  case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR:
    return "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";
  case CUSOLVER_STATUS_INVALID_WORKSPACE:
    return "CUSOLVER_STATUS_INVALID_WORKSPACE";
  default:
    return "CUSOLVER UNKNOWN ERROR";
  }
}

/*******************************************************************************
 * \brief Run given cusolver command and upon failure abort with a nice message.
 * \author Ole Schuett
 ******************************************************************************/
#define CUSOLVER_CHECK(cmd)                                                    \
  do {                                                                         \
    cusolverStatus_t status = cmd;                                             \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "ERROR: %s %s %d\n", cusolverGetErrorString(status),     \
              __FILE__, __LINE__);                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

/*******************************************************************************
 * \brief Driver routine to diagonalize a matrix with the cuSOLVERMp library.
 * \author Ole Schuett
 ******************************************************************************/
void cp_fm_diag_cusolver(const int fortran_comm, const int matrix_desc[9],
                         const int nprow, const int npcol, const int myprow,
                         const int mypcol, const int n, const double *matrix,
                         double *eigenvectors, double *eigenvalues) {

  offload_activate_chosen_device();
  const int local_device = offload_get_chosen_device();

  MPI_Comm comm = MPI_Comm_f2c(fortran_comm);
  int rank, nranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);

#if defined(__CUSOLVERMP_NCCL)
  // Create NCCL communicator.
  ncclUniqueId nccl_id;
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&nccl_id));
  }
  MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, comm);

  ncclComm_t nccl_comm;
  NCCL_CHECK(ncclCommInitRank(&nccl_comm, nranks, nccl_id, rank));
#else
  // Create CAL communicator.
  cal_comm_t cal_comm = NULL;
  cal_comm_create_params_t params;
  params.allgather = &allgather;
  params.req_test = &req_test;
  params.req_free = &req_free;
  params.data = &comm;
  params.rank = rank;
  params.nranks = nranks;
  params.local_device = local_device;
  CAL_CHECK(cal_comm_create(params, &cal_comm));
#endif

  // Create various handles.
  cudaStream_t stream = NULL;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cusolverMpHandle_t cusolvermp_handle = NULL;
  CUSOLVER_CHECK(cusolverMpCreate(&cusolvermp_handle, local_device, stream));

  cusolverMpGrid_t grid = NULL;
#if defined(__CUSOLVERMP_NCCL)
  CUSOLVER_CHECK(cusolverMpCreateDeviceGrid(cusolvermp_handle, &grid, nccl_comm,
                                            nprow, npcol,
                                            CUSOLVERMP_GRID_MAPPING_ROW_MAJOR));
#else
  CUSOLVER_CHECK(cusolverMpCreateDeviceGrid(cusolvermp_handle, &grid, cal_comm,
                                            nprow, npcol,
                                            CUSOLVERMP_GRID_MAPPING_ROW_MAJOR));
#endif
  const int mb = matrix_desc[4];
  const int nb = matrix_desc[5];
  const int rsrc = matrix_desc[6];
  const int csrc = matrix_desc[7];
  const int ldA = matrix_desc[8];
  assert(rsrc == csrc);
  assert(ldA >= 1);

  const int np = cusolverMpNUMROC(n, mb, myprow, rsrc, nprow);
  const int nq = cusolverMpNUMROC(n, nb, mypcol, csrc, npcol);
  assert(np == ldA);

  const cublasFillMode_t fill_mode = CUBLAS_FILL_MODE_UPPER;
  const cudaDataType_t data_type = CUDA_R_64F; // double

  cusolverMpMatrixDescriptor_t cusolvermp_matrix_desc = NULL;
  CUSOLVER_CHECK(cusolverMpCreateMatrixDesc(
      &cusolvermp_matrix_desc, grid, data_type, n, n, mb, nb, rsrc, csrc, np));

  // Allocate workspaces.
  size_t work_dev_size, work_host_size;
  void *DUMMY = (void *)1; // Workaround to avoid crash when passing NULL.
  CUSOLVER_CHECK(cusolverMpSyevd_bufferSize(
      cusolvermp_handle, "V", fill_mode, n, DUMMY, 1, 1, cusolvermp_matrix_desc,
      NULL, NULL, 1, 1, cusolvermp_matrix_desc, data_type, &work_dev_size,
      &work_host_size));

  double *work_dev = NULL;
  CUDA_CHECK(cudaMalloc((void **)&work_dev, work_dev_size));

  double *work_host = NULL;
  CUDA_CHECK(cudaMallocHost((void **)&work_host, work_host_size));

  // Upload input matrix.
  const size_t matrix_local_size = ldA * nq * sizeof(double);
  double *matrix_dev = NULL;
  CUDA_CHECK(cudaMalloc((void **)&matrix_dev, matrix_local_size));
  CUDA_CHECK(cudaMemcpyAsync(matrix_dev, matrix, matrix_local_size,
                             cudaMemcpyHostToDevice, stream));

  // Allocate result buffers.
  int *info_dev = NULL;
  CUDA_CHECK(cudaMalloc((void **)&info_dev, sizeof(int)));

  double *eigenvectors_dev = NULL;
  CUDA_CHECK(cudaMalloc((void **)&eigenvectors_dev, matrix_local_size));

  double *eigenvalues_dev = NULL;
  CUDA_CHECK(cudaMalloc((void **)&eigenvalues_dev, n * sizeof(double)));

  // Call solver.
  CUSOLVER_CHECK(
      cusolverMpSyevd(cusolvermp_handle, "V", fill_mode, n, matrix_dev, 1, 1,
                      cusolvermp_matrix_desc, eigenvalues_dev, eigenvectors_dev,
                      1, 1, cusolvermp_matrix_desc, data_type, work_dev,
                      work_dev_size, work_host, work_host_size, info_dev));

  // Wait for solver to finish.
  CUDA_CHECK(cudaStreamSynchronize(stream));
#if !defined(__CUSOLVERMP_NCCL)
  CAL_CHECK(cal_stream_sync(cal_comm, stream));
#endif

  // Check info.
  int info = -1;
  CUDA_CHECK(cudaMemcpy(&info, info_dev, sizeof(int), cudaMemcpyDeviceToHost));
  assert(info == 0);

  // Download results.
  CUDA_CHECK(cudaMemcpyAsync(eigenvectors, eigenvectors_dev, matrix_local_size,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(eigenvalues, eigenvalues_dev, n * sizeof(double),
                             cudaMemcpyDeviceToHost, stream));

  // Wait for download to finish.
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Free buffers.
  CUDA_CHECK(cudaFree(matrix_dev));
  CUDA_CHECK(cudaFree(info_dev));
  CUDA_CHECK(cudaFree(eigenvectors_dev));
  CUDA_CHECK(cudaFree(eigenvalues_dev));
  CUDA_CHECK(cudaFree(work_dev));
  CUDA_CHECK(cudaFreeHost(work_host));

  // Destroy handles.
  CUSOLVER_CHECK(cusolverMpDestroyMatrixDesc(cusolvermp_matrix_desc));
  CUSOLVER_CHECK(cusolverMpDestroyGrid(grid));
  CUSOLVER_CHECK(cusolverMpDestroy(cusolvermp_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
#if defined(__CUSOLVERMP_NCCL)
  NCCL_CHECK(ncclCommDestroy(nccl_comm));
#else
  CAL_CHECK(cal_comm_destroy(cal_comm));
#endif

  // Sync MPI ranks to include load imbalance in total timings.
  MPI_Barrier(comm);
}

/*******************************************************************************
 * \brief Cached GPU buffers for the SyGvd generalized eigensolver.
 *        Reused across SCF steps to avoid costly cudaMalloc/cudaFree per call.
 ******************************************************************************/
static double *sygvd_dev_A = NULL;
static double *sygvd_dev_B = NULL;
static double *sygvd_dev_Z = NULL;
static double *sygvd_eigenvalues_dev = NULL;
static int *sygvd_info_dev = NULL;
static void *sygvd_work_dev = NULL;
static void *sygvd_work_host = NULL;
static size_t sygvd_matrix_local_size = 0;
static int sygvd_n = 0;
static size_t sygvd_work_dev_size = 0;
static size_t sygvd_work_host_size = 0;
// Note: B matrix cannot be cached on GPU because cusolverMpSygvd overwrites it.

/*******************************************************************************
 * \brief Free cached GPU buffers when they need to be reallocated or at exit.
 ******************************************************************************/
static void sygvd_free_buffers(void) {
  if (sygvd_dev_A) { cudaFree(sygvd_dev_A); sygvd_dev_A = NULL; }
  if (sygvd_dev_B) { cudaFree(sygvd_dev_B); sygvd_dev_B = NULL; }
  if (sygvd_dev_Z) { cudaFree(sygvd_dev_Z); sygvd_dev_Z = NULL; }
  if (sygvd_eigenvalues_dev) { cudaFree(sygvd_eigenvalues_dev); sygvd_eigenvalues_dev = NULL; }
  if (sygvd_info_dev) { cudaFree(sygvd_info_dev); sygvd_info_dev = NULL; }
  if (sygvd_work_dev) { cudaFree(sygvd_work_dev); sygvd_work_dev = NULL; }
  if (sygvd_work_host) { cudaFreeHost(sygvd_work_host); sygvd_work_host = NULL; }
  sygvd_matrix_local_size = 0;
  sygvd_n = 0;
  sygvd_work_dev_size = 0;
  sygvd_work_host_size = 0;
  // B matrix tracking removed: solver overwrites B in-place.
}

/*******************************************************************************
 * \brief Driver routine to solve A*x = lambda*B*x with cuSOLVERMp sygvd.
 * \author Jiri Vyskocil
 ******************************************************************************/
void cp_fm_diag_cusolver_sygvd(const int fortran_comm,
                               const int a_matrix_desc[9],
                               const int b_matrix_desc[9], const int nprow,
                               const int npcol, const int myprow,
                               const int mypcol, const int n,
                               const double *aMatrix, const double *bMatrix,
                               double *eigenvectors, double *eigenvalues) {

  offload_activate_chosen_device();
  const int local_device = offload_get_chosen_device();

  MPI_Comm comm = MPI_Comm_f2c(fortran_comm);
  int rank, nranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);

#if defined(__CUSOLVERMP_NCCL)
  // Create NCCL communicator.
  ncclUniqueId nccl_id;
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&nccl_id));
  }
  MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, comm);

  ncclComm_t nccl_comm;
  NCCL_CHECK(ncclCommInitRank(&nccl_comm, nranks, nccl_id, rank));
#else
  // Create CAL communicator
  cal_comm_t cal_comm = NULL;
  cal_comm_create_params_t params;
  params.allgather = &allgather;
  params.req_test = &req_test;
  params.req_free = &req_free;
  params.data = &comm;
  params.rank = rank;
  params.nranks = nranks;
  params.local_device = local_device;
  CAL_CHECK(cal_comm_create(params, &cal_comm));
#endif

  // Create CUDA stream and cuSOLVER handle
  cudaStream_t stream = NULL;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cusolverMpHandle_t cusolvermp_handle = NULL;
  CUSOLVER_CHECK(cusolverMpCreate(&cusolvermp_handle, local_device, stream));

  // Define grid for device computation
  cusolverMpGrid_t grid = NULL;
#if defined(__CUSOLVERMP_NCCL)
  CUSOLVER_CHECK(cusolverMpCreateDeviceGrid(cusolvermp_handle, &grid, nccl_comm,
                                            nprow, npcol,
                                            CUSOLVERMP_GRID_MAPPING_ROW_MAJOR));
#else
  CUSOLVER_CHECK(cusolverMpCreateDeviceGrid(cusolvermp_handle, &grid, cal_comm,
                                            nprow, npcol,
                                            CUSOLVERMP_GRID_MAPPING_ROW_MAJOR));
#endif

  // Matrix descriptors for A, B, and Z
  const int mb_a = a_matrix_desc[4];
  const int nb_a = a_matrix_desc[5];
  const int rsrc_a = a_matrix_desc[6];
  const int csrc_a = a_matrix_desc[7];
  const int ldA = a_matrix_desc[8];

  const int mb_b = b_matrix_desc[4];
  const int nb_b = b_matrix_desc[5];
  const int rsrc_b = b_matrix_desc[6];
  const int csrc_b = b_matrix_desc[7];
  const int ldB = b_matrix_desc[8];

  // Ensure consistency in block sizes, sources, and leading dimensions
  assert(mb_a == mb_b && nb_a == nb_b);
  assert(rsrc_a == rsrc_b && csrc_a == csrc_b);
  assert(ldA == ldB);

  const int np_a = cusolverMpNUMROC(n, mb_a, myprow, rsrc_a, nprow);
  const int nq_a = cusolverMpNUMROC(n, nb_a, mypcol, csrc_a, npcol);
  (void)np_a;

  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  const cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  const cudaDataType_t data_type = CUDA_R_64F;

  // Create matrix descriptors
  cusolverMpMatrixDescriptor_t descrA = NULL;
  cusolverMpMatrixDescriptor_t descrB = NULL;
  cusolverMpMatrixDescriptor_t descrZ = NULL;

  CUSOLVER_CHECK(cusolverMpCreateMatrixDesc(&descrA, grid, data_type, n, n,
                                            mb_a, nb_a, rsrc_a, csrc_a, ldA));
  CUSOLVER_CHECK(cusolverMpCreateMatrixDesc(&descrB, grid, data_type, n, n,
                                            mb_b, nb_b, rsrc_b, csrc_b, ldB));
  CUSOLVER_CHECK(cusolverMpCreateMatrixDesc(&descrZ, grid, data_type, n, n,
                                            mb_a, nb_a, rsrc_a, csrc_a, ldA));

  // Compute local buffer size
  const size_t matrix_local_size = ldA * nq_a * sizeof(double);

  // Reallocate GPU buffers only if matrix dimensions changed
  if (matrix_local_size != sygvd_matrix_local_size || n != sygvd_n) {
    sygvd_free_buffers();
    sygvd_matrix_local_size = matrix_local_size;
    sygvd_n = n;
    CUDA_CHECK(cudaMalloc((void **)&sygvd_dev_A, matrix_local_size));
    CUDA_CHECK(cudaMalloc((void **)&sygvd_dev_B, matrix_local_size));
    CUDA_CHECK(cudaMalloc((void **)&sygvd_dev_Z, matrix_local_size));
    CUDA_CHECK(cudaMalloc((void **)&sygvd_eigenvalues_dev, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&sygvd_info_dev, sizeof(int)));
  }

  // Copy A matrix (KS) from host to device every call
  CUDA_CHECK(cudaMemcpyAsync(sygvd_dev_A, aMatrix, matrix_local_size,
                             cudaMemcpyHostToDevice, stream));

  // Copy B matrix (overlap S) every call.
  // cusolverMpSygvd overwrites B in-place (Cholesky factorization),
  // so we must re-upload the original B each time.
  CUDA_CHECK(cudaMemcpyAsync(sygvd_dev_B, bMatrix, matrix_local_size,
                             cudaMemcpyHostToDevice, stream));

  // Query workspace size and reallocate if needed
  size_t new_work_dev_size = 0, new_work_host_size = 0;
  const int64_t ia = 1, ja = 1, ib = 1, jb = 1, iz = 1, jz = 1;
  const int64_t m = (int64_t)n;

  cusolverStatus_t status_bufsize = cusolverMpSygvd_bufferSize(
      cusolvermp_handle, itype, jobz, uplo, m, ia, ja, descrA, ib, jb, descrB,
      iz, jz, descrZ, data_type, &new_work_dev_size, &new_work_host_size);
  if (status_bufsize != CUSOLVER_STATUS_SUCCESS) {
    fprintf(stderr, "ERROR: cusolverMpSygvd_bufferSize failed with status=%d\n",
            (int)status_bufsize);
    abort();
  }

  if (new_work_dev_size > sygvd_work_dev_size) {
    if (sygvd_work_dev) { cudaFree(sygvd_work_dev); }
    CUDA_CHECK(cudaMalloc(&sygvd_work_dev, new_work_dev_size));
    sygvd_work_dev_size = new_work_dev_size;
  }
  if (new_work_host_size > sygvd_work_host_size) {
    if (sygvd_work_host) { cudaFreeHost(sygvd_work_host); }
    CUDA_CHECK(cudaMallocHost(&sygvd_work_host, new_work_host_size));
    sygvd_work_host_size = new_work_host_size;
  }

  // Reset info
  CUDA_CHECK(cudaMemset(sygvd_info_dev, 0, sizeof(int)));

  // Call cusolverMpSygvd
  cusolverStatus_t status_sygvd = cusolverMpSygvd(
      cusolvermp_handle, itype, jobz, uplo, m, sygvd_dev_A, ia, ja, descrA,
      sygvd_dev_B, ib, jb, descrB, sygvd_eigenvalues_dev, sygvd_dev_Z, iz, jz,
      descrZ, data_type, sygvd_work_dev, sygvd_work_dev_size, sygvd_work_host,
      sygvd_work_host_size, sygvd_info_dev);
  if (status_sygvd != CUSOLVER_STATUS_SUCCESS) {
    fprintf(stderr, "ERROR: cusolverMpSygvd failed with status=%d\n",
            (int)status_sygvd);
    abort();
  }

  // Wait for computation to finish
  CUDA_CHECK(cudaStreamSynchronize(stream));
#if !defined(__CUSOLVERMP_NCCL)
  CAL_CHECK(cal_stream_sync(cal_comm, stream));
#endif

  // Check info
  int info;
  CUDA_CHECK(cudaMemcpy(&info, sygvd_info_dev, sizeof(int), cudaMemcpyDeviceToHost));
  if (info != 0) {
    fprintf(stderr, "ERROR: cusolverMpSygvd failed with info = %d\n", info);
    abort();
  }

  // Copy results back to host
  CUDA_CHECK(cudaMemcpyAsync(eigenvectors, sygvd_dev_Z, matrix_local_size,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(eigenvalues, sygvd_eigenvalues_dev,
                             n * sizeof(double), cudaMemcpyDeviceToHost,
                             stream));

  // Wait for copy to finish
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Destroy per-call handles (NCCL comm/handle caching deferred to later)
  CUSOLVER_CHECK(cusolverMpDestroyMatrixDesc(descrA));
  CUSOLVER_CHECK(cusolverMpDestroyMatrixDesc(descrB));
  CUSOLVER_CHECK(cusolverMpDestroyMatrixDesc(descrZ));
  CUSOLVER_CHECK(cusolverMpDestroyGrid(grid));
  CUSOLVER_CHECK(cusolverMpDestroy(cusolvermp_handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
#if defined(__CUSOLVERMP_NCCL)
  NCCL_CHECK(ncclCommDestroy(nccl_comm));
#else
  CAL_CHECK(cal_comm_destroy(cal_comm));
#endif

  MPI_Barrier(comm); // Synchronize MPI ranks
}
#endif

// EOF
