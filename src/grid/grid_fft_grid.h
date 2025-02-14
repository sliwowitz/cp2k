/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_GRID_H
#define GRID_FFT_GRID_H

#include "common/grid_mpi.h"

#include <complex.h>
#include <stdbool.h>

/*******************************************************************************
 * \brief Container to represent fft grids to merge the multigrids.
 * \author Frederick Stein
 ******************************************************************************/
typedef struct {
  // Global number of points
  int npts_global[3];
  // New communicator
  grid_mpi_comm comm;
  int proc_grid[2];
  int periodic[2];
  int proc_coords[2];
  // distributions for each FFT step (real space/mixed-space 1 (rs), mixed space
  // 1/mixed space 2 (ms), mixed-space 2/g-space (gs)) first index is for the
  // process, the second for the coordinate, the third for start (0) / end(1)
  // proc2local_rs is also used for the distribution of the data in realspace
  // (that's why it is called "rs") proc2local_gs is also used for the
  // distribution of the data in reciprocal (g)-space (that's why it is called
  // "gs") in blocked mode (usually the finest grid)
  int (*proc2local_rs)[3][2]; // Order: (x, y, z)
  int (*proc2local_ms)[3][2]; // Order: (z, x, y)
  int (*proc2local_gs)[3][2]; // Order: (y, z, x)
  // Actual data
  double *grid_rs;
  double complex *grid_gs;
  // buffers for different purposes
} grid_fft_grid;

void grid_free_fft_grid(grid_fft_grid *fft_grid);

void grid_create_fft_grid(grid_fft_grid **fft_grid, const grid_mpi_comm comm,
                          const int npts_global[3]);

#endif

// EOF
