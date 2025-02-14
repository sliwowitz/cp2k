/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_test.h"

#include "common/grid_common.h"
#include "common/grid_mpi.h"
#include "grid_fft.h"
#include "grid_fft_grid.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

inline double norm_vector(const double complex *vector, const int size) {
  double norm = 0.0;
  for (int i = 0; i < size; i++)
    norm += cabs(vector[i]);
  return sqrt(norm);
}

inline double norm_vector_double(const double *vector, const int size) {
  double norm = 0.0;
  for (int i = 0; i < size; i++)
    norm += fabs(vector[i]);
  return sqrt(norm);
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_local() {
  // Check a few fft sizes
  const int fft_sizes[3] = {16, 18, 20};

  const double pi = acos(-1);

  int max_size = fft_sizes[0];
  for (int dir = 1; dir < 3; dir++)
    max_size = imax(max_size, fft_sizes[dir]);

  double complex *input_array =
      malloc(max_size * max_size * sizeof(double complex));
  double complex *output_array =
      malloc(max_size * max_size * sizeof(double complex));

  double error = 0.0;
  for (int dir = 0; dir < 3; dir++) {
    const int current_size = fft_sizes[dir];
    memset(input_array, 0,
           current_size * current_size * sizeof(double complex));

    for (int number_of_fft = 0; number_of_fft < current_size; number_of_fft++) {
      input_array[number_of_fft * current_size + number_of_fft] = 1.0;
    }

    fft_1d_fw(input_array, output_array, current_size, current_size);

    for (int number_of_fft = 0; number_of_fft < current_size; number_of_fft++) {
      for (int index = 0; index < current_size; index++) {
        error = fmax(
            error,
            cabs(output_array[number_of_fft * current_size + index] -
                 cexp(-2.0 * I * pi * number_of_fft * index / current_size)));
      }
    }
  }

  free(input_array);
  free(output_array);

  if (error > 1e-12) {
    printf("\nThe low-level FFTs do not work properly: %f!\n", error);
    return 1;
  } else {
    printf("\nThe 1D FFTs do work properly!\n");
    return 0;
  }
}

/*******************************************************************************
 * \brief Function to test the local transposition operation.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_transpose() {
  // Check a few fft sizes
  const int fft_sizes[2] = {16, 18};

  int max_size = fft_sizes[0] * fft_sizes[1];

  double complex *input_array = calloc(max_size, sizeof(double complex));
  double complex *output_array = calloc(max_size, sizeof(double complex));

  for (int index_1 = 0; index_1 < fft_sizes[0]; index_1++) {
    for (int index_2 = 0; index_2 < fft_sizes[1]; index_2++) {
      input_array[index_1 * fft_sizes[1] + index_2] =
          1.0 * index_1 - index_2 * I;
    }
  }

  transpose_local(input_array, output_array, fft_sizes[1], fft_sizes[0]);

  double error = 0.0;

  for (int index_1 = 0; index_1 < fft_sizes[0]; index_1++) {
    for (int index_2 = 0; index_2 < fft_sizes[1]; index_2++) {
      error = fmax(error, cabs(output_array[index_2 * fft_sizes[0] + index_1] -
                               (1.0 * index_1 - index_2 * I)));
    }
  }

  free(input_array);
  free(output_array);

  if (error > 1e-12) {
    printf("\nThe low-level transpose does not work properly: %f!\n", error);
    return 1;
  } else {
    printf("\nThe local transpose does work properly!\n");
    return 0;
  }
}

/*******************************************************************************
 * \brief Function to test the parallel FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_parallel() {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const int my_process = grid_mpi_comm_rank(comm);

  // Use an asymmetric cell to check correctness of indices
  const int npts_global[3] = {2, 4, 8};

  const double pi = acos(-1);
  (void)pi;

  grid_fft_grid *fft_grid = NULL;
  grid_create_fft_grid(&fft_grid, comm, npts_global);

  const int(*my_bounds_rs)[2] = fft_grid->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1] - my_bounds_rs[dir][0] + 1;
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int(*my_bounds_gs)[2] = fft_grid->proc2local_gs[my_process];
  int my_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_gs[dir] = my_bounds_gs[dir][1] - my_bounds_gs[dir][0] + 1;

  double error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        memset(fft_grid->grid_rs, 0, my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] && nx <= my_bounds_rs[0][1] &&
            ny >= my_bounds_rs[1][0] && ny <= my_bounds_rs[1][1] &&
            nz >= my_bounds_rs[2][0] && nz <= my_bounds_rs[2][1])
          fft_grid->grid_rs[(nx - my_bounds_rs[0][0]) * my_sizes_rs[1] *
                                my_sizes_rs[2] +
                            (ny - my_bounds_rs[1][0]) * my_sizes_rs[2] +
                            (nz - my_bounds_rs[2][0])] = 1.0;

        fft_3d_fw_blocked(fft_grid->grid_rs, fft_grid->grid_gs,
                          fft_grid->npts_global, fft_grid->proc2local_rs,
                          fft_grid->proc2local_ms, fft_grid->proc2local_gs,
                          fft_grid->comm);

        for (int mx = 0; mx < my_sizes_gs[0]; mx++) {
          for (int my = 0; my < my_sizes_gs[1]; my++) {
            for (int mz = 0; mz < my_sizes_gs[2]; mz++) {
              const double complex my_value =
                  fft_grid->grid_gs[my * my_sizes_gs[0] * my_sizes_gs[2] +
                                    mz * my_sizes_gs[0] + mx];
              const double complex ref_value = cexp(
                  -2.0 * I * pi *
                  (((double)mx + my_bounds_gs[0][0]) * nx / npts_global[0] +
                   ((double)my + my_bounds_gs[1][0]) * ny / npts_global[1] +
                   ((double)mz + my_bounds_gs[2][0]) * nz / npts_global[2]));
              double current_error = cabs(my_value - ref_value);
              error = fmax(error, current_error);
            }
          }
        }
      }
    }
  }
  grid_mpi_max_double(&error, 1, comm);

  grid_free_fft_grid(fft_grid);

  if (error > 1e-12) {
    printf("\nThe 3D FFTs do not work properly: %f!\n", error);
    return 1;
  } else {
    printf("\nThe 3D FFTs do work properly!\n");
    return 0;
  }
}

// EOF
