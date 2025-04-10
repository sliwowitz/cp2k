#pragma once

#include "common/grid_mpi.h"
#include <complex.h>
#include <stdbool.h>

#include "grid_multigrid.h"

struct fft_box_t {
  int lower[3];
  int upper[3];
  int strides[3];
};

void cufftmp_grid_copy_to_multigrid_single(
  const grid_multigrid *multigrid,
  const double *grid,
  const grid_mpi_comm comm,
  const int (*proc2local)[3][2]);

void cufftmp_grid_copy_from_multigrid_single(
  const grid_multigrid *multigrid,
  double *grid,
  const grid_mpi_comm comm,
  const int (*proc2local)[3][2]);

void cufft_fwd(
    double *grid_rs,
    double complex *grid_gs,
    const int npts_global[3],
    struct fft_box_t *box_real,
    struct fft_box_t *box_complex,
    const grid_mpi_comm comm
    );

void cufft_bck(
    double *grid_rs,
    double complex *grid_gs,
    const int npts_global[3],
    struct fft_box_t *box_real,
    struct fft_box_t *box_complex,
    const grid_mpi_comm comm
    );