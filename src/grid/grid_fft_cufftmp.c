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
  strides[1] = upper[2] - lower[2];
  strides[2] = 1;
}

void cufft_grid_copy_to_multigrid_single(
    const grid_multigrid *multigrid,
    const double *grid,
    const grid_mpi_comm comm,
    const int (*proc2local)[3][2])
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int ndevices;
  cudaGetDeviceCount(&ndevices);
  cudaSetDevice(rank % ndevices);  //TODO: Set the correct device

  const int my_process = grid_mpi_comm_rank(comm);

  // Forward FFT the fine grid using cuFFTMp which also redistributes the data
  // The input must exclude halos
  const int npts_global[3] = { multigrid->npts_global[0][0],
                               multigrid->npts_global[0][1],
                               multigrid->npts_global[0][2] };
  const long int total_number_of_elements =
      (long int)multigrid->npts_global[0][0] *
      (long int)multigrid->npts_global[0][1] *
      (long int)multigrid->npts_global[0][2];

  struct fft_box_t box_real;
  struct fft_box_t box_complex;

  extract_point(box_real.lower, proc2local, grid_mpi_comm_rank(comm), 0);
  extract_point(box_real.upper, proc2local, grid_mpi_comm_rank(comm), 1);
  make_strides(box_real.strides, box_real.lower, box_real.upper);

  extract_point(box_complex.lower,
                multigrid->fft_gs_grids[0].fft_grid_layout->proc2local_gs,
                grid_mpi_comm_rank(comm), 0);
  extract_point(box_complex.upper,
                multigrid->fft_gs_grids[0].fft_grid_layout->proc2local_gs,
                grid_mpi_comm_rank(comm), 1);
  make_strides(box_complex.strides, box_complex.lower, box_complex.upper);

  // Allocate a multi-GPU descriptor
  cudaLibXtDesc *gpu_fine_grid;

  cufft_fwd(grid,
            gpu_fine_grid,
            npts_global,
            &box_real,
            &box_complex,
            comm);

  // Prepare the coarse grids. In the end, these need halos (in real space).
  // In k space, we can do without halos, but either cuFFTMp needs to output
  // into inner parts of real-space grids with halos, or we need to copy the
  // cuFFTMp output into the correct grid ourselves.

  // Copy from the fine grid to the coarse grids, then backward FFT
  // all grids using cuFFTMp. Since it also redistributes the data,
  // we must make sure the coarse grids in the real space are at the correct
  // location corresponding to the fine grid. The output should be written into
  // the inner part of the grid, and we need to find out how to set the halos.
  for (int level = 1; level < multigrid->nlevels; level++) {
    // Copy the data to the coarse grid
    //TODO: Both grids should be allocated via cufftXtMalloc!
    grid_copy_to_coarse_grid(gpu_fine_grid,
                             &multigrid->fft_gs_grids[level]);
    cufft_bck(&multigrid->fft_gs_grids[level],
              &multigrid->fft_rs_grids[level],
              npts_global,  // Should be only the coarse grid size
              multigrid->fft_gs_grids[level].fft_grid_layout->proc2local_gs,
              multigrid->fft_rs_grids[level].fft_grid_layout->proc2local_rs,
              comm);

    //TODO: Normalize the data in the coarse grid
    const double factor = (double)total_number_of_elements /
                          ((double)multigrid->npts_global[level][0]) /
                          ((double)multigrid->npts_global[level][1]) /
                          ((double)multigrid->npts_global[level][2]);
    const int(*my_bounds)[2] =
        multigrid->fft_rs_grids[level]
            .fft_grid_layout->proc2local_rs[grid_mpi_comm_rank(
                multigrid->fft_rs_grids[level].fft_grid_layout->comm)];
    for (int index = 0; index < (my_bounds[0][1] - my_bounds[0][0] + 1) *
                                    (my_bounds[1][1] - my_bounds[1][0] + 1) *
                                    (my_bounds[2][1] - my_bounds[2][0] + 1);
         index++)
      multigrid->fft_rs_grids[level].data[index] *= factor;

    //TODO: Communicate the halos in real space.
  }
}

void cufft_grid_copy_from_multigrid_single(
    const grid_multigrid *multigrid,
    double *grid,
    const grid_mpi_comm comm,
    const int (*proc2local)[3][2]) {
}

void cufft_fwd(const double *cpu_data,
               cudaLibXtDesc *gpu_data,
               const int npts_global[3],
               struct fft_box_t *box_real,
               struct fft_box_t *box_complex,
               const grid_mpi_comm comm) {

  cufftHandle plan_r2c;
  size_t work_size;

  cufftMpMakePlanDecomposition(plan_r2c, 3, npts_global,
      box_real->lower, box_real->upper, box_real->strides,
      box_complex->lower, box_complex->upper, box_complex->strides,
      CUFFT_R2C, &comm, CUFFT_COMM_MPI, &work_size);

  cufftXtMalloc(plan_r2c, &gpu_data, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT);
  // Copy the real data to the device
  cufftXtMemcpy(plan_r2c, gpu_data, cpu_data, CUFFT_COPY_HOST_TO_DEVICE);

  cufftXtExecDescriptor(plan_r2c, gpu_data, gpu_data, CUFFT_FORWARD);
}

void cufft_bck(double *grid_rs,
               cudaLibXtDesc *complex_scratch_space,
               const int npts_global[3],
               struct fft_box_t *box_real,
               struct fft_box_t *box_complex,
               const grid_mpi_comm comm)
{
  cufftHandle plan_c2r;
  size_t work_size;

  cufftMpMakePlanDecomposition(plan_c2r, 3, npts_global,
      box_complex->lower, box_complex->upper, box_complex->strides,
      box_real->lower, box_real->upper, box_real->strides,
      CUFFT_C2R, &comm, CUFFT_COMM_MPI, &work_size);

  cufftXtExecDescriptor(plan_c2r, complex_scratch_space, complex_scratch_space, CUFFT_INVERSE);
  cufftXtMemcpy(plan_c2r, grid_rs, complex_scratch_space, CUFFT_COPY_DEVICE_TO_HOST);
}