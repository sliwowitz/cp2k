#include "grid_fft_cufftmp.h"
const size_t nullptr = 0; //TODO: Workaround cufftmp bug.
#include <cufftMp.h>
#include <mpi.h>
#include <complex.h>
#include <cuda_runtime_api.h>

#include "grid_fft_grid_layout.h"
#include "common/grid_common.h"
#include "grid_fft_grid.h"
#include "grid_fft_lib.h"
#include "grid_fft_reorder.h"


void extract_point(int result[3], const int pool[3][3][2], int proc_id, int location) {
  result[0] = pool[proc_id][0][location];
  result[1] = pool[proc_id][1][location];
  result[2] = pool[proc_id][2][location];
}

void make_strides(int strides[3], const int lower[3], const int upper[3]) {
  strides[0] = (upper[1] - lower[1]) * (upper[2] - lower[2]);
  strides[1] = (upper[2] - lower[2]);
  strides[2] = 1;
}

void fft_3d_fw_cufftmp(double *grid_rs,
                       double complex *grid_gs,
                       const int npts_global[3],
                       const int (*proc2local_rs)[3][2],
                       const int (*proc2local_gs)[3][2],
                       const grid_mpi_comm comm,
                       const grid_mpi_comm sub_comm[2])
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int ndevices;
  cudaGetDeviceCount(&ndevices);
  cudaSetDevice(rank % ndevices);  //TODO: Set the correct device


  const int my_process = grid_mpi_comm_rank(comm);

  int real_lower[3], real_upper[3], complex_lower[3], complex_upper[3];
  extract_point(real_lower, proc2local_rs, my_process, 0);
  extract_point(real_upper, proc2local_rs, my_process, 1);
  extract_point(complex_lower, proc2local_gs, my_process, 0);
  extract_point(complex_upper, proc2local_gs, my_process, 1);

  int real_strides[3], complex_strides[3];
  make_strides(real_strides, real_lower, real_upper);
  make_strides(complex_strides, complex_lower, complex_upper);

  cufftHandle plan_r2c;
  size_t work_size;

  cufftMpMakePlanDecomposition(plan_r2c, 3, npts_global, real_lower, real_upper, real_strides,
                              complex_lower, complex_upper, complex_strides,
                              CUFFT_R2C, &comm, CUFFT_COMM_MPI, &work_size);

  // Allocate a multi-GPU descriptor
  cudaLibXtDesc *desc;
  cufftXtMalloc(plan_r2c, &desc, CUFFT_XT_FORMAT_INPLACE);

  // Copy data from the CPU to the GPU.
  // The CPU data is distributed according to CUFFT_XT_FORMAT_DISTRIBUTED_INPUT
  cufftXtMemcpy(plan_r2c, desc, grid_rs, CUFFT_COPY_HOST_TO_DEVICE);

  cufftXtExecDescriptor(plan_r2c, desc, desc, CUFFT_FORWARD);


  //TODO: Arithmetic operations on the GPU


  cufftHandle plan_c2r;
  cufftMpMakePlanDecomposition(plan_c2r, 3, npts_global, complex_lower, complex_upper, complex_strides,
                              real_lower, real_upper, real_strides,
                              CUFFT_C2R, &comm, CUFFT_COMM_MPI, &work_size);

  cufftXtExecDescriptor(plan_c2r, desc, desc, CUFFT_INVERSE);

  cufftXtMemcpy(plan_c2r, grid_rs, desc, CUFFT_COPY_DEVICE_TO_HOST);


  // Cleanup

}