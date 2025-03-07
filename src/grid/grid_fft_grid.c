/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_grid.h"
#include "common/grid_common.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Could be reformulated with Lapack or calculated
// For orthorhombic cells, this is at the order of 3*eps(multiplication)+6*eps(addition)
// For non-orthorhombic cells, this depends on the cell shape
const double max_rel_error_for_equivalence_g_squared = 1e-12;

typedef struct {
    double value;
    int index;
} double_index_pair;

int compare_double(const void *a, const void *b) {
  const double a_value = ((const double_index_pair*)a)->value;
  const double b_value = ((const double_index_pair*)b)->value;
  return (a_value > b_value ? 1 : (a_value < b_value ? -1 : 0));
}

int compare_shell(const void *a, const void *b) {
  for (int index = 0; index < 3; index++) {
    const int a_value = ((const int*)a)[index];
    const int b_value = ((const int*)b)[index];
    if (a_value > b_value + max_rel_error_for_equivalence_g_squared) {
      return 1;
    } else if (a_value + max_rel_error_for_equivalence_g_squared < b_value) {
      return -1;
    }
  }
  return 0;
}

void sort_shell(int (*shell)[3], const int shell_size) {
  qsort(shell, shell_size, sizeof(int[3]), compare_shell);
}

void sort_g_vectors(grid_fft_grid *my_fft_grid) {
  assert(my_fft_grid != NULL);
  assert(my_fft_grid->npts_gs_local >= 0);

  const double two_pi = 2.0*acos(-1.0);

  int local_index2g_squared[my_fft_grid->npts_gs_local];
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    double length_g_squared = 0.0;
    for (int dir = 0; dir < 3; dir++) {
      double length_g_dir = 0.0;
      for (int dir2 = 0; dir2 < 3; dir2++) {
        length_g_dir +=
          my_fft_grid->index_to_g[index][dir] * my_fft_grid->dh_inv[dir2][dir];
      }
      length_g_dir *= two_pi;
      length_g_squared += length_g_dir * length_g_dir;
    }
    local_index2g_squared[index] = length_g_squared;
  }

  // Sort the indices according to the length of the vectors
  double_index_pair g_square_index_pair[my_fft_grid->npts_gs_local];
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    g_square_index_pair[index].value = local_index2g_squared[index];
    g_square_index_pair[index].index = index;
  }
  qsort(g_square_index_pair, my_fft_grid->npts_gs_local, sizeof(double_index_pair), compare_double);

  // Apply the sorting to the index_to_g array
  {
    int index_to_g_sorted[my_fft_grid->npts_gs_local][3];
    for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
        memcpy(index_to_g_sorted[index], my_fft_grid->index_to_g[g_square_index_pair[index].index], 3 * sizeof(int));
        local_index2g_squared[index] = g_square_index_pair[index].value;
    }
    memcpy(my_fft_grid->index_to_g, &index_to_g_sorted[0][0], my_fft_grid->npts_gs_local * sizeof(int[3]));
  }

  // Sort the vectors with the same length according to the x-, then y-, then z-coordinate
  {
    double last_g_squared = g_square_index_pair[0].value;
    int start_index = 0;
    for (int end_index = 1; end_index < my_fft_grid->npts_gs_local; end_index++) {
      if (fabs(g_square_index_pair[end_index].value - last_g_squared) > fmax(g_square_index_pair[end_index].value, last_g_squared)*max_rel_error_for_equivalence_g_squared) {
        // If the length of the current vector is different from the previous one, sort the vectors with the same length
        // according to the x-, then y-, then z-coordinate
        sort_shell(my_fft_grid->index_to_g + start_index, end_index - start_index);
        start_index = end_index;
        last_g_squared = g_square_index_pair[end_index].value;
      }
    }
    // At the end, we need to sort the last shell
    sort_shell(my_fft_grid->index_to_g + start_index, my_fft_grid->npts_gs_local - start_index);
  }
}

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
    free(fft_grid->ray_to_yz);
    free(fft_grid->rays_per_process);
    free(fft_grid->index_to_g);
    free(fft_grid->local_index_to_ref_grid);
    free(fft_grid);
  }
}

void grid_create_fft_grid(grid_fft_grid **fft_grid, const grid_mpi_comm comm,
                          const int npts_global[3], const double dh_inv[3][3]) {
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
  memcpy(my_fft_grid->dh_inv, dh_inv, 3 * 3 * sizeof(double));

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
  my_fft_grid->ray_to_yz = NULL;
  my_fft_grid->rays_per_process = NULL;
  my_fft_grid->index_to_g = calloc(my_fft_grid->npts_gs_local, sizeof(int[3]));
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    int index_g[3];
    index_g[0] = index % (my_fft_grid->proc2local_gs[my_process][0][1] -
                          my_fft_grid->proc2local_gs[my_process][0][0] + 1);
    index_g[1] = (index / (my_fft_grid->proc2local_gs[my_process][0][1] -
                           my_fft_grid->proc2local_gs[my_process][0][0] + 1)) %
                 (my_fft_grid->proc2local_gs[my_process][1][1] -
                  my_fft_grid->proc2local_gs[my_process][1][0] + 1);
    index_g[2] = index /
                 ((my_fft_grid->proc2local_gs[my_process][0][1] -
                   my_fft_grid->proc2local_gs[my_process][0][0] + 1) *
                  (my_fft_grid->proc2local_gs[my_process][1][1] -
                   my_fft_grid->proc2local_gs[my_process][1][0] + 1));
    for (int dir = 0; dir < 3; dir++)
      my_fft_grid->index_to_g[index][dir] =
          index_g[dir] + my_fft_grid->proc2local_gs[my_process][dir][0];
  }

  my_fft_grid->local_index_to_ref_grid = calloc(
      my_fft_grid->npts_gs_local, sizeof(int));

  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    my_fft_grid->local_index_to_ref_grid[index] = index;
  }

  sort_g_vectors(my_fft_grid);

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
  grid_mpi_cart_create(fft_grid_ref->comm, 2, &my_fft_grid->proc_grid[0],
                       &my_fft_grid->periodic[0], false, &my_fft_grid->comm);

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

  my_fft_grid->ray_distribution = true;

  // Assign the (yz)-rays of the reference grid which are also on the current
  // grid to each process
  my_fft_grid->yz_to_process =
      malloc(npts_global[1] * npts_global[2] * sizeof(int));
  // Count the number of rays on each process
  my_fft_grid->rays_per_process = calloc(number_of_processes, sizeof(int));
  memset(my_fft_grid->yz_to_process, -1,
         npts_global[1] * npts_global[2] * sizeof(int));
  int total_number_of_rays = 0;
  for (int process = 0; process < number_of_processes; process++) {
    for (int index_y = fft_grid_ref->proc2local_gs[process][1][0];
         index_y <= fft_grid_ref->proc2local_gs[process][1][1]; index_y++) {
      // The right half of the indices is shifted
      const int index_y_shifted = convert_c_index_to_shifted_index(
          index_y, fft_grid_ref->npts_global[1]);
      // Compare the shifted index with the allowed subset of shifted indices of
      // the new grid The allowed set is given by -(n-1)//2...n//2 (these are
      // always n elements)
      if (!is_on_grid(index_y_shifted, npts_global[1]))
        continue;
      const int index_y_new =
          convert_shifted_index_to_c_index(index_y_shifted, npts_global[1]);
      for (int index_z = fft_grid_ref->proc2local_gs[process][2][0];
           index_z <= fft_grid_ref->proc2local_gs[process][2][1]; index_z++) {
        // The right half of the indices is shifted
        const int index_z_shifted = convert_c_index_to_shifted_index(
            index_z, fft_grid_ref->npts_global[2]);
        // Same check for z-coordinate
        if (!is_on_grid(index_z_shifted, npts_global[2]))
          continue;
        const int index_z_new =
            convert_shifted_index_to_c_index(index_z_shifted, npts_global[2]);
        assert(index_y_new * npts_global[2] + index_z_new >= 0);
        assert(npts_global[1] * npts_global[2] >
               index_y_new * npts_global[2] + index_z_new);
        assert(my_fft_grid->yz_to_process[index_y_new * npts_global[2] +
                                          index_z_new] < 0);
        my_fft_grid->yz_to_process[index_y_new * npts_global[2] + index_z_new] =
            process;
        my_fft_grid->rays_per_process[process]++;
        total_number_of_rays++;
      }
    }
  }
  my_fft_grid->npts_gs_local =
      npts_global[0] * my_fft_grid->rays_per_process[my_process];

  int *ray_offsets = calloc(number_of_processes, sizeof(int));
  int *ray_index_per_process = calloc(number_of_processes, sizeof(int));
  for (int process = 1; process < number_of_processes; process++) {
    ray_offsets[process] =
        ray_offsets[process - 1] + my_fft_grid->rays_per_process[process - 1];
  }
  assert(ray_offsets[number_of_processes - 1] +
             my_fft_grid->rays_per_process[number_of_processes - 1] ==
         total_number_of_rays);

  // Create the map of yz index to the yz coordinates and the z-values required
  // for the mixed space
  my_fft_grid->ray_to_yz = malloc(total_number_of_rays * sizeof(int[2]));
  for (int ray = 0; ray < total_number_of_rays; ray++) {
    my_fft_grid->ray_to_yz[ray][0] = -1;
    my_fft_grid->ray_to_yz[ray][1] = -1;
  }
  for (int index_y = 0; index_y < fft_grid_ref->npts_global[1]; index_y++) {
    const int index_y_shifted =
        convert_c_index_to_shifted_index(index_y, fft_grid_ref->npts_global[1]);
    if (!is_on_grid(index_y_shifted, npts_global[1]))
      continue;
    const int index_y_new =
        convert_shifted_index_to_c_index(index_y_shifted, npts_global[1]);
    assert(index_y_new >= 0);
    assert(index_y_new < npts_global[1]);
    for (int index_z = 0; index_z < fft_grid_ref->npts_global[2]; index_z++) {
      const int index_z_shifted = convert_c_index_to_shifted_index(
          index_z, fft_grid_ref->npts_global[2]);
      // Same check for z-coordinate
      if (!is_on_grid(index_z_shifted, npts_global[2]))
        continue;
      const int index_z_new =
          convert_shifted_index_to_c_index(index_z_shifted, npts_global[2]);
      assert(index_z_new >= 0);
      assert(index_z_new < npts_global[2]);
      const int current_process =
          my_fft_grid
              ->yz_to_process[index_y_new * npts_global[2] + index_z_new];
      assert(current_process >= 0);
      const int current_ray_index = ray_index_per_process[current_process];
      assert(current_ray_index <
             my_fft_grid->rays_per_process[current_process]);
      const int current_ray_offset = ray_offsets[current_process];
      assert(current_ray_offset < total_number_of_rays);
      assert(current_ray_index <
             my_fft_grid->rays_per_process[current_process]);
      assert(current_ray_offset + current_ray_index >= 0);
      assert(current_ray_offset + current_ray_index < total_number_of_rays);
      assert(my_fft_grid->ray_to_yz[current_ray_offset + current_ray_index][0] <
             0);
      my_fft_grid->ray_to_yz[current_ray_offset + current_ray_index][0] =
          index_y_new;
      assert(my_fft_grid->ray_to_yz[current_ray_offset + current_ray_index][1] <
             0);
      my_fft_grid->ray_to_yz[current_ray_offset + current_ray_index][1] =
          index_z_new;
      ray_index_per_process[current_process]++;
      assert(ray_index_per_process[current_process] <=
             my_fft_grid->rays_per_process[current_process]);
    }
  }
  for (int process = 0; process < number_of_processes; process++) {
    assert(ray_index_per_process[process] ==
               my_fft_grid->rays_per_process[process] &&
           "The number of rays does not match the expected number of rays!");
    const int current_ray_offset = ray_offsets[process];
    for (int ray_index = 0; ray_index < my_fft_grid->rays_per_process[process];
         ray_index++) {
      assert(my_fft_grid->ray_to_yz[current_ray_offset + ray_index][0] >= 0 &&
             my_fft_grid->ray_to_yz[current_ray_offset + ray_index][1] >= 0 &&
             "The ray has to be assigned to a valid yz index!");
      assert(my_fft_grid->ray_to_yz[current_ray_offset + ray_index][0] <
                 npts_global[1] &&
             my_fft_grid->ray_to_yz[current_ray_offset + ray_index][1] <
                 npts_global[2] &&
             "The ray has to be assigned to a valid yz index!");
    }
  }

  free(ray_offsets);
  free(ray_index_per_process);

  // Here, they need a different size then in the blocked case as we will only
  // carry the data from our local rays
  my_fft_grid->grid_gs =
      calloc(my_fft_grid->npts_gs_local, sizeof(double complex));

      my_fft_grid->index_to_g = calloc(my_fft_grid->npts_gs_local, sizeof(int[3]));
      my_fft_grid->local_index_to_ref_grid = calloc(
          my_fft_grid->npts_gs_local, sizeof(int));

    sort_g_vectors(my_fft_grid);

  *fft_grid = my_fft_grid;
}

// EOF
