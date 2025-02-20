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
    free(fft_grid->yz_to_process);
    free(fft_grid->ray_number_to_yz);
    free(fft_grid->rays_per_process);
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

  my_fft_grid->ray_distribution = false;
  my_fft_grid->npts_gs_local =
      (my_fft_grid->proc2local_gs[my_process][0][1] -
       my_fft_grid->proc2local_gs[my_process][0][0] + 1) *
      (my_fft_grid->proc2local_gs[my_process][1][1] -
       my_fft_grid->proc2local_gs[my_process][1][0] + 1) *
      (my_fft_grid->proc2local_gs[my_process][2][1] -
       my_fft_grid->proc2local_gs[my_process][2][0] + 1);

  my_fft_grid->yz_to_process = NULL;
  my_fft_grid->ray_number_to_yz = NULL;
  my_fft_grid->rays_per_process = NULL;

  *fft_grid = my_fft_grid;
}

void grid_create_fft_grid_from_reference(grid_fft_grid **fft_grid,
                                         const int npts_global[3],
                                         const grid_fft_grid *fft_grid_ref) {
  assert(fft_grid_ref != NULL &&
         "Grid creation from reference grid requires a valid reference grid!");
  // Current restriction of the code.
  assert(!fft_grid_ref->ray_distribution &&
         "The reference grid has to have a blocked distribution!");
  // We will use the reference grid to collect the data from other grids. To
  // prevent loss of accuracy, we require the new grid to be coarser than or as
  // coarse as the reference grid.
  assert(npts_global[0] <= fft_grid_ref->npts_global[0] &&
         npts_global[1] <= fft_grid_ref->npts_global[1] &&
         npts_global[2] <= fft_grid_ref->npts_global[2] &&
         "The new grid cannot have more grid points in any direction than the "
         "reference grid!");

  grid_fft_grid *my_fft_grid = NULL;
  if (*fft_grid != NULL) {
    my_fft_grid = *fft_grid;
    grid_free_fft_grid(my_fft_grid);
  } else {
    my_fft_grid = malloc(sizeof(grid_fft_grid));
  }

  const int number_of_processes = grid_mpi_comm_size(fft_grid_ref->comm);

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
  grid_mpi_cart_create(fft_grid_ref->comm, 2, my_fft_grid->proc_grid,
                       my_fft_grid->periodic, false, &my_fft_grid->comm);

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
  // Here, they need a different size then in the blocked case as we will only
  // carry the data from our local rays
  my_fft_grid->grid_gs =
      calloc((my_fft_grid->proc2local_gs[my_process][0][1] -
              my_fft_grid->proc2local_gs[my_process][0][0] + 1) *
                 (my_fft_grid->proc2local_gs[my_process][1][1] -
                  my_fft_grid->proc2local_gs[my_process][1][0] + 1) *
                 (my_fft_grid->proc2local_gs[my_process][2][1] -
                  my_fft_grid->proc2local_gs[my_process][2][0] + 1),
             sizeof(double complex));

  my_fft_grid->ray_distribution = true;

  // Assign the (yz)-rays of the reference grid which are also on the current
  // grid to each process
  my_fft_grid->yz_to_process =
      malloc(npts_global[1] * npts_global[2] * sizeof(int));
  my_fft_grid->rays_per_process = calloc(number_of_processes, sizeof(int));
  for (int index_y = 0; index_y < npts_global[1]; index_y++) {
    for (int index_z = 0; index_z < npts_global[2]; index_z++) {
      my_fft_grid->yz_to_process[index_y * npts_global[2] + index_z] = -1;
    }
  }
  for (int process = 0; process < number_of_processes; process++) {
    for (int index_y = fft_grid_ref->proc2local_gs[process][1][0];
         index_y <= fft_grid_ref->proc2local_gs[process][1][1]; index_y++) {
      // The right half of the indices is shifted
      const int index_y_shifted = convert_c_index_to_shifted_index(
          index_y, fft_grid_ref->npts_global[1]);
      // Compare the shifted index with the allowed subset of shifted indices of
      // the new grid The allowed set is given by -(n-1)//2...n//2 (these are
      // always n elements)
      if (is_on_grid(index_y_shifted, npts_global[1]))
        continue;
      for (int index_z = fft_grid_ref->proc2local_gs[process][2][0];
           index_z <= fft_grid_ref->proc2local_gs[process][2][1]; index_z++) {
        // The right half of the indices is shifted
        const int index_z_shifted = convert_c_index_to_shifted_index(
            index_z, fft_grid_ref->npts_global[2]);
        // Same check for z-coordinate
        if (is_on_grid(index_z_shifted, npts_global[2]))
          continue;
        my_fft_grid->yz_to_process[index_y * npts_global[2] + index_z] =
            process;
        my_fft_grid->rays_per_process[process]++;
      }
    }
  }
  my_fft_grid->npts_gs_local =
      npts_global[0] * my_fft_grid->rays_per_process[my_process];

  // Create the map of yz index to the yz coordinates and the z-values required
  // for the mixed space
  my_fft_grid->ray_number_to_yz =
      malloc(my_fft_grid->rays_per_process[my_process] * sizeof(int[2]));
  int ray_index = 0;
  for (int index_y = 0; index_y < npts_global[1]; index_y++) {
    for (int index_z = 0; index_z < npts_global[2]; index_z++) {
      if (my_fft_grid->yz_to_process[index_y * npts_global[2] + index_z] ==
          my_process) {
        my_fft_grid->ray_number_to_yz[ray_index][0] = index_y;
        my_fft_grid->ray_number_to_yz[ray_index][1] = index_z;
        ray_index++;
      }
    }
  }

  *fft_grid = my_fft_grid;
}

// EOF
