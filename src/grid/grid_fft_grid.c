/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_grid.h"
#include "common/grid_common.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void grid_free_fft_grid(grid_fft_grid *fft_grid) {
  if (fft_grid != NULL) {
    grid_mpi_comm_free(&fft_grid->comm);
    free(fft_grid->proc2local_rs);
    free(fft_grid->proc2local_ms);
    free(fft_grid->proc2local_gs);
    free(fft_grid->grid_rs);
    free(fft_grid->grid_rs_complex);
    free(fft_grid->grid_ms);
    free(fft_grid->grid_gs);
    free(fft_grid);
  }
}

void grid_create_fft_grid(grid_fft_grid **fft_grid, const grid_mpi_comm comm,
                          const int npts_global[3]) {
  grid_fft_grid *my_fft_grid = NULL;
  if (*fft_grid != NULL) {
    my_fft_grid = *fft_grid;
    grid_free_fft_grid(my_fft_grid);
  } else {
    my_fft_grid = malloc(sizeof(grid_fft_grid));
  }

  const int number_of_processes = grid_mpi_comm_size(comm);

  if (npts_global[0] < number_of_processes) {
    // We only distribute in two directions if necessary to reduce communication
    grid_mpi_dims_create(number_of_processes, 2, &my_fft_grid->proc_grid[0]);
  } else {
    my_fft_grid->proc_grid[0] = number_of_processes;
    my_fft_grid->proc_grid[1] = 1;
  }

  memcpy(my_fft_grid->npts_global, npts_global, 3 * sizeof(int));

  my_fft_grid->periodic[0] = 1;
  my_fft_grid->periodic[1] = 1;
  grid_mpi_cart_create(comm, 2, my_fft_grid->proc_grid, my_fft_grid->periodic,
                       true, &my_fft_grid->comm);

  const int my_process = grid_mpi_comm_rank(my_fft_grid->comm);

  grid_mpi_cart_get(my_fft_grid->comm, 2, my_fft_grid->proc_grid,
                    my_fft_grid->periodic, my_fft_grid->proc_coords);

  my_fft_grid->proc2local_rs = malloc(number_of_processes * sizeof(int[3][2]));
  my_fft_grid->proc2local_ms = malloc(number_of_processes * sizeof(int[3][2]));
  my_fft_grid->proc2local_gs = malloc(number_of_processes * sizeof(int[3][2]));
  for (int proc = 0; proc < number_of_processes; proc++) {
    int proc_coords[2];
    grid_mpi_cart_coords(my_fft_grid->comm, proc, 2, proc_coords);
    // Determine the bounds in real space
    my_fft_grid->proc2local_rs[proc][0][0] =
        proc_coords[0] * npts_global[0] / my_fft_grid->proc_grid[0];
    my_fft_grid->proc2local_rs[proc][0][1] =
        (proc_coords[0] + 1) * npts_global[0] / my_fft_grid->proc_grid[0] - 1;
    my_fft_grid->proc2local_rs[proc][1][0] =
        proc_coords[1] * npts_global[1] / my_fft_grid->proc_grid[1];
    my_fft_grid->proc2local_rs[proc][1][1] =
        (proc_coords[1] + 1) * npts_global[1] / my_fft_grid->proc_grid[1] - 1;
    my_fft_grid->proc2local_rs[proc][2][0] = 0;
    my_fft_grid->proc2local_rs[proc][2][1] = npts_global[2] - 1;
    // Determine the bounds in mixed space: we keep the distribution in the
    // first direction to reduce communication
    my_fft_grid->proc2local_ms[proc][0][0] =
        proc_coords[0] * npts_global[0] / my_fft_grid->proc_grid[0];
    my_fft_grid->proc2local_ms[proc][0][1] =
        (proc_coords[0] + 1) * npts_global[0] / my_fft_grid->proc_grid[0] - 1;
    my_fft_grid->proc2local_ms[proc][1][0] = 0;
    my_fft_grid->proc2local_ms[proc][1][1] = npts_global[1] - 1;
    my_fft_grid->proc2local_ms[proc][2][0] =
        proc_coords[1] * npts_global[2] / my_fft_grid->proc_grid[1];
    my_fft_grid->proc2local_ms[proc][2][1] =
        (proc_coords[1] + 1) * npts_global[2] / my_fft_grid->proc_grid[1] - 1;
    // Determine the bounds in mixed space: we keep the distribution in the
    // third direction to reduce communication
    my_fft_grid->proc2local_gs[proc][0][0] = 0;
    my_fft_grid->proc2local_gs[proc][0][1] = npts_global[0] - 1;
    my_fft_grid->proc2local_gs[proc][1][0] =
        proc_coords[0] * npts_global[1] / my_fft_grid->proc_grid[0];
    my_fft_grid->proc2local_gs[proc][1][1] =
        (proc_coords[0] + 1) * npts_global[1] / my_fft_grid->proc_grid[0] - 1;
    my_fft_grid->proc2local_gs[proc][2][0] =
        proc_coords[1] * npts_global[2] / my_fft_grid->proc_grid[1];
    my_fft_grid->proc2local_gs[proc][2][1] =
        (proc_coords[1] + 1) * npts_global[2] / my_fft_grid->proc_grid[1] - 1;
  }

  my_fft_grid->grid_rs =
      calloc((my_fft_grid->proc2local_rs[my_process][0][1] -
              my_fft_grid->proc2local_rs[my_process][0][0] + 1) *
                 (my_fft_grid->proc2local_rs[my_process][1][1] -
                  my_fft_grid->proc2local_rs[my_process][1][0] + 1) *
                 (my_fft_grid->proc2local_rs[my_process][2][1] -
                  my_fft_grid->proc2local_rs[my_process][2][0] + 1),
             sizeof(double));

  my_fft_grid->grid_rs_complex =
      calloc((my_fft_grid->proc2local_rs[my_process][0][1] -
              my_fft_grid->proc2local_rs[my_process][0][0] + 1) *
                 (my_fft_grid->proc2local_rs[my_process][1][1] -
                  my_fft_grid->proc2local_rs[my_process][1][0] + 1) *
                 (my_fft_grid->proc2local_rs[my_process][2][1] -
                  my_fft_grid->proc2local_rs[my_process][2][0] + 1),
             sizeof(double complex));
  my_fft_grid->grid_ms =
      calloc((my_fft_grid->proc2local_ms[my_process][0][1] -
              my_fft_grid->proc2local_ms[my_process][0][0] + 1) *
                 (my_fft_grid->proc2local_ms[my_process][1][1] -
                  my_fft_grid->proc2local_ms[my_process][1][0] + 1) *
                 (my_fft_grid->proc2local_ms[my_process][2][1] -
                  my_fft_grid->proc2local_ms[my_process][2][0] + 1),
             sizeof(double complex));
  my_fft_grid->grid_gs =
      calloc((my_fft_grid->proc2local_gs[my_process][0][1] -
              my_fft_grid->proc2local_gs[my_process][0][0] + 1) *
                 (my_fft_grid->proc2local_gs[my_process][1][1] -
                  my_fft_grid->proc2local_gs[my_process][1][0] + 1) *
                 (my_fft_grid->proc2local_gs[my_process][2][1] -
                  my_fft_grid->proc2local_gs[my_process][2][0] + 1),
             sizeof(double complex));

  *fft_grid = my_fft_grid;
}

// EOF
