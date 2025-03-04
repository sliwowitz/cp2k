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
int fft_test_local_low(const int fft_size, const int number_of_ffts) {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

  int errors = 0;

  const double pi = acos(-1);

  double complex *input_array =
      calloc(fft_size * number_of_ffts, sizeof(double complex));
  double complex *output_array =
      calloc(fft_size * number_of_ffts, sizeof(double complex));

  double max_error = 0.0;
  // Check the forward FFT
  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    input_array[(number_of_fft % fft_size) * number_of_ffts + number_of_fft] =
        1.0;
  }

  fft_1d_fw(input_array, output_array, fft_size, number_of_ffts);

  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    for (int index = 0; index < fft_size; index++) {
      max_error =
          fmax(max_error, cabs(output_array[number_of_fft * fft_size + index] -
                               cexp(-2.0 * I * pi * (number_of_fft % fft_size) *
                                    index / fft_size)));
    }
  }

  if (max_error > 1.0e-12) {
    if (my_process == 0)
      printf("The 1D-FFT does not work properly (%i %i): %f!\n", fft_size,
             number_of_ffts, max_error);
    errors++;
  }

  // Check the backward FFT
  memset(input_array, 0, fft_size * number_of_ffts * sizeof(double complex));

  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    input_array[number_of_fft * fft_size + number_of_fft % fft_size] = 1.0;
  }

  fft_1d_bw(input_array, output_array, fft_size, number_of_ffts);

  max_error = 0.0;
  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    for (int index = 0; index < fft_size; index++) {
      max_error = fmax(
          max_error, cabs(output_array[index * number_of_ffts + number_of_fft] -
                          cexp(2.0 * I * pi * (number_of_fft % fft_size) *
                               index / fft_size)));
    }
  }

  free(input_array);
  free(output_array);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The low-level FFTs do not work properly (%i %i): %f!\n", fft_size,
             number_of_ffts, max_error);
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 1D FFT does work properly (%i %i)!\n", fft_size,
           number_of_ffts);
  return errors;
}

int fft_test_local() {
  int errors = 0;

  errors += fft_test_local_low(1, 1);
  errors += fft_test_local_low(2, 1);
  errors += fft_test_local_low(3, 1);
  errors += fft_test_local_low(4, 1);
  errors += fft_test_local_low(5, 1);
  errors += fft_test_local_low(16, 1);
  errors += fft_test_local_low(18, 1);
  errors += fft_test_local_low(20, 1);
  errors += fft_test_local_low(1, 1);
  errors += fft_test_local_low(2, 2);
  errors += fft_test_local_low(3, 3);
  errors += fft_test_local_low(4, 4);
  errors += fft_test_local_low(5, 5);
  errors += fft_test_local_low(16, 16);
  errors += fft_test_local_low(18, 18);
  errors += fft_test_local_low(20, 20);
  errors += fft_test_local_low(16, 360);
  errors += fft_test_local_low(18, 320);
  errors += fft_test_local_low(20, 288);

  return errors;
}

/*******************************************************************************
 * \brief Function to test the local transposition operation.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_transpose() {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);
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
    if (my_process == 0)
      printf("The low-level transpose does not work properly: %f!\n", error);
    return 1;
  } else {
    if (my_process == 0)
      printf("The local transpose does work properly!\n");
    return 0;
  }
}

double fft_test_transpose_ray(const int npts_global[3],
                              const int npts_global_ref[3]) {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const int my_process = grid_mpi_comm_rank(comm);

  int errors = 0;

  double max_error = 0.0;

  // Build the reference grid
  grid_fft_grid *ref_grid = NULL;
  grid_create_fft_grid(&ref_grid, comm, npts_global_ref);

  // Test ray transpositiond,
  grid_fft_grid *fft_grid_ray = NULL;
  grid_create_fft_grid_from_reference(&fft_grid_ray, npts_global, ref_grid);

  int my_bounds_ms_ray[3][2];
  memcpy(my_bounds_ms_ray, fft_grid_ray->proc2local_ms[my_process],
         sizeof(int[3][2]));
  int my_sizes_ms_ray[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_ms_ray[dir] =
        my_bounds_ms_ray[dir][1] - my_bounds_ms_ray[dir][0] + 1;

  for (int index_x = 0; index_x < my_sizes_ms_ray[0]; index_x++) {
    for (int index_y = 0; index_y < my_sizes_ms_ray[1]; index_y++) {
      for (int index_z = 0; index_z < my_sizes_ms_ray[2]; index_z++) {
        fft_grid_ray
            ->grid_ms[index_x * my_sizes_ms_ray[1] * my_sizes_ms_ray[2] +
                      index_z * my_sizes_ms_ray[1] + index_y] =
            ((index_y + my_bounds_ms_ray[1][0]) * fft_grid_ray->npts_global[2] +
             (index_z + my_bounds_ms_ray[2][0])) +
            I * (index_x + my_bounds_ms_ray[0][0]);
      }
    }
  }

  transpose_xz_to_yz_ray(fft_grid_ray->grid_ms, fft_grid_ray->grid_gs,
                         fft_grid_ray->npts_global, fft_grid_ray->proc2local_ms,
                         fft_grid_ray->yz_to_process,
                         fft_grid_ray->rays_per_process,
                         fft_grid_ray->ray_to_yz, fft_grid_ray->comm);

  max_error = 0.0;
  int ray_index_offset = 0;
  for (int process = 0; process < my_process; process++)
    ray_index_offset += fft_grid_ray->rays_per_process[process];
  for (int yz_ray = 0; yz_ray < fft_grid_ray->rays_per_process[my_process];
       yz_ray++) {
    const int index_y = fft_grid_ray->ray_to_yz[ray_index_offset + yz_ray][0];
    const int index_z = fft_grid_ray->ray_to_yz[ray_index_offset + yz_ray][1];
    for (int index_x = 0; index_x < fft_grid_ray->npts_global[0]; index_x++) {
      const double complex my_value =
          fft_grid_ray
              ->grid_gs[index_x * fft_grid_ray->rays_per_process[my_process] +
                        yz_ray];
      const double complex ref_value =
          (index_y * npts_global[2] + index_z) + I * index_x;
      double current_error = cabs(my_value - ref_value);
      max_error = fmax(max_error, current_error);
    }
  }

  if (max_error > 1e-12) {
    if (my_process == 0) {
      printf("The transpose xz_to_yz_ray does not work properly: %f!\n",
             max_error);
    }
    errors++;
  }

  memset(fft_grid_ray->grid_gs, 0,
         fft_grid_ray->npts_global[0] *
             fft_grid_ray->rays_per_process[my_process] *
             sizeof(double complex));
  memset(fft_grid_ray->grid_ms, 0,
         product3(my_sizes_ms_ray) * sizeof(double complex));

  for (int yz_ray = 0; yz_ray < fft_grid_ray->rays_per_process[my_process];
       yz_ray++) {
    const int index_y = fft_grid_ray->ray_to_yz[ray_index_offset + yz_ray][0];
    const int index_z = fft_grid_ray->ray_to_yz[ray_index_offset + yz_ray][1];
    for (int index_x = 0; index_x < fft_grid_ray->npts_global[0]; index_x++) {
      fft_grid_ray
          ->grid_gs[index_x * fft_grid_ray->rays_per_process[my_process] +
                    yz_ray] =
          (index_y * fft_grid_ray->npts_global[2] + index_z) + I * index_x;
    }
  }
  transpose_yz_to_xz_ray(fft_grid_ray->grid_gs, fft_grid_ray->grid_ms,
                         fft_grid_ray->npts_global, fft_grid_ray->yz_to_process,
                         fft_grid_ray->proc2local_ms,
                         fft_grid_ray->rays_per_process,
                         fft_grid_ray->ray_to_yz, fft_grid_ray->comm);

  max_error = 0.0;
  for (int index_y = 0; index_y < my_sizes_ms_ray[1]; index_y++) {
    for (int index_z = 0; index_z < my_sizes_ms_ray[2]; index_z++) {
      // Check whether there is a ray with the given index pair
      if (fft_grid_ray->yz_to_process[(index_y + my_bounds_ms_ray[1][0]) *
                                          fft_grid_ray->npts_global[2] +
                                      (index_z + my_bounds_ms_ray[2][0])] >=
          0) {
        for (int index_x = 0; index_x < my_sizes_ms_ray[0]; index_x++) {
          const double complex my_value =
              fft_grid_ray
                  ->grid_ms[index_x * my_sizes_ms_ray[1] * my_sizes_ms_ray[2] +
                            index_z * my_sizes_ms_ray[1] + index_y];
          const double complex ref_value =
              ((index_y + my_bounds_ms_ray[1][0]) *
                   fft_grid_ray->npts_global[2] +
               (index_z + my_bounds_ms_ray[2][0])) +
              I * (index_x + my_bounds_ms_ray[0][0]);
          double current_error = cabs(my_value - ref_value);
          max_error = fmax(max_error, current_error);
        }
      } else {
        for (int index_x = 0; index_x < fft_grid_ray->npts_global[0];
             index_x++) {
          const double complex my_value =
              fft_grid_ray
                  ->grid_ms[index_x * fft_grid_ray->npts_global[1] *
                                fft_grid_ray->npts_global[2] +
                            index_z * fft_grid_ray->npts_global[1] + index_y];
          // The value is assumed to be zero if absent
          const double complex ref_value = 0.0;
          double current_error = cabs(my_value - ref_value);
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }

  grid_free_fft_grid(fft_grid_ray);
  grid_free_fft_grid(ref_grid);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The transpose yz_to_xz_ray does not work properly: %f!\n",
             max_error);
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The transpose from the ray distribution works properly "
           "(sizes: %i %i %i)!\n",
           npts_global[0], npts_global[1], npts_global[2]);

  return errors;
}

/*******************************************************************************
 * \brief Function to test the parallel transposition operation.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_transpose_blocked(const int npts_global[3]) {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const int my_process = grid_mpi_comm_rank(comm);

  int errors = 0;

  grid_fft_grid *fft_grid = NULL;
  grid_create_fft_grid(&fft_grid, comm, npts_global);

  const int(*my_bounds_rs)[2] = fft_grid->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1] - my_bounds_rs[dir][0] + 1;
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int(*my_bounds_ms)[2] = fft_grid->proc2local_ms[my_process];
  int my_sizes_ms[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_ms[dir] = my_bounds_ms[dir][1] - my_bounds_ms[dir][0] + 1;
  const int my_number_of_elements_ms = product3(my_sizes_ms);

  const int(*my_bounds_gs)[2] = fft_grid->proc2local_gs[my_process];
  int my_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_gs[dir] = my_bounds_gs[dir][1] - my_bounds_gs[dir][0] + 1;

  // Collect the maximum error
  double max_error = 0.0;

  // Check forward RS->MS FFTs
  for (int nx = 0; nx < my_sizes_rs[0]; nx++) {
    for (int ny = 0; ny < my_sizes_rs[1]; ny++) {
      for (int nz = 0; nz < my_sizes_rs[2]; nz++) {
        fft_grid->grid_rs_complex[ny * my_sizes_rs[0] * my_sizes_rs[2] +
                                  nx * my_sizes_rs[2] + nz] =
            ((nx + my_bounds_rs[0][0]) * npts_global[1] +
             (ny + my_bounds_rs[1][0])) +
            I * (nz + my_bounds_rs[2][0]);
      }
    }
  }

  transpose_xy_to_xz_blocked(fft_grid->grid_rs_complex, fft_grid->grid_ms,
                             npts_global, fft_grid->proc2local_rs,
                             fft_grid->proc2local_ms, fft_grid->comm);

  for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
    for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
      for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
        const double complex my_value =
            fft_grid->grid_ms[ny * my_sizes_ms[0] * my_sizes_ms[2] +
                              nx * my_sizes_ms[2] + nz];
        const double complex ref_value =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
        double current_error = cabs(my_value - ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The transpose xy_to_xz_blocked does not work properly (%i %i "
             "%i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
    for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
      for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
        fft_grid->grid_ms[ny * my_sizes_ms[0] * my_sizes_ms[2] +
                          nx * my_sizes_ms[2] + nz] =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
      }
    }
  }
  memset(fft_grid->grid_rs_complex, 0,
         my_number_of_elements_rs * sizeof(double complex));

  // Check the reverse direction
  transpose_xz_to_xy_blocked(fft_grid->grid_ms, fft_grid->grid_rs_complex,
                             npts_global, fft_grid->proc2local_ms,
                             fft_grid->proc2local_rs, fft_grid->comm);

  // Check forward RS->MS FFTs
  max_error = 0.0;
  for (int nx = 0; nx < my_sizes_rs[0]; nx++) {
    for (int ny = 0; ny < my_sizes_rs[1]; ny++) {
      for (int nz = 0; nz < my_sizes_rs[2]; nz++) {
        const double complex my_value =
            fft_grid->grid_rs_complex[ny * my_sizes_rs[0] * my_sizes_rs[2] +
                                      nx * my_sizes_rs[2] + nz];
        const double complex ref_value =
            ((nx + my_bounds_rs[0][0]) * npts_global[1] +
             (ny + my_bounds_rs[1][0])) +
            I * (nz + my_bounds_rs[2][0]);
        double current_error = cabs(my_value - ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The transpose xz_to_xy_blocked does not work properly (%i %i "
             "%i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
    for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
      for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
        fft_grid->grid_ms[nx * my_sizes_ms[1] * my_sizes_ms[2] +
                          nz * my_sizes_ms[1] + ny] =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
      }
    }
  }

  // Check the MS/GS direction
  transpose_xz_to_yz_blocked(fft_grid->grid_ms, fft_grid->grid_gs, npts_global,
                             fft_grid->proc2local_ms, fft_grid->proc2local_gs,
                             fft_grid->comm);

  // Check forward RS->MS FFTs
  max_error = 0.0;
  for (int nx = 0; nx < my_sizes_gs[0]; nx++) {
    for (int ny = 0; ny < my_sizes_gs[1]; ny++) {
      for (int nz = 0; nz < my_sizes_gs[2]; nz++) {
        const double complex my_value =
            fft_grid->grid_gs[nx * my_sizes_gs[1] * my_sizes_gs[2] +
                              nz * my_sizes_gs[1] + ny];
        const double complex ref_value =
            ((nx + my_bounds_gs[0][0]) * npts_global[1] +
             (ny + my_bounds_gs[1][0])) +
            I * (nz + my_bounds_gs[2][0]);
        double current_error = cabs(my_value - ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The transpose xz_to_yz_blocked does not work properly (%i %i "
             "%i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  for (int nx = 0; nx < my_sizes_gs[0]; nx++) {
    for (int ny = 0; ny < my_sizes_gs[1]; ny++) {
      for (int nz = 0; nz < my_sizes_gs[2]; nz++) {
        fft_grid->grid_gs[nx * my_sizes_gs[1] * my_sizes_gs[2] +
                          nz * my_sizes_gs[1] + ny] =
            ((nx + my_bounds_gs[0][0]) * npts_global[1] +
             (ny + my_bounds_gs[1][0])) +
            I * (nz + my_bounds_gs[2][0]);
      }
    }
  }
  memset(fft_grid->grid_ms, 0, my_number_of_elements_ms);

  // Check the MS/GS direction
  transpose_yz_to_xz_blocked(fft_grid->grid_gs, fft_grid->grid_ms, npts_global,
                             fft_grid->proc2local_gs, fft_grid->proc2local_ms,
                             fft_grid->comm);

  // Check forward RS->MS FFTs
  max_error = 0.0;
  for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
    for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
      for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
        const double complex my_value =
            fft_grid->grid_ms[nx * my_sizes_ms[1] * my_sizes_ms[2] +
                              nz * my_sizes_ms[1] + ny];
        const double complex ref_value =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
        double current_error = cabs(my_value - ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The transpose yz_to_xz_blocked does not work properly (%i %i "
             "%i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  grid_free_fft_grid(fft_grid);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the parallel transposition operation.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_transpose_parallel() {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const int my_process = grid_mpi_comm_rank(comm);

  int errors = 0;

  // Grid sizes to be checked
  const int npts_global[3] = {2, 4, 8};
  const int npts_global_small[3] = {2, 3, 5};
  const int npts_global_reverse[3] = {8, 4, 2};
  const int npts_global_small_reverse[3] = {5, 3, 2};

  // Check the blocked layout
  errors += fft_test_transpose_blocked(npts_global);
  errors += fft_test_transpose_blocked(npts_global_small);
  errors += fft_test_transpose_blocked(npts_global_reverse);
  errors += fft_test_transpose_blocked(npts_global_small_reverse);

  // Check the ray layout with the same grid sizes
  errors += fft_test_transpose_ray(npts_global, npts_global);
  errors += fft_test_transpose_ray(npts_global_small, npts_global_small);
  errors += fft_test_transpose_ray(npts_global_reverse, npts_global_reverse);
  errors += fft_test_transpose_ray(npts_global_small_reverse,
                                   npts_global_small_reverse);

  // Check the ray layout with different grid sizes
  errors += fft_test_transpose_ray(npts_global_small, npts_global);
  errors +=
      fft_test_transpose_ray(npts_global_small_reverse, npts_global_reverse);

  if (errors == 0 && my_process == 0)
    printf("\n The parallel transposition routines work properly!\n");
  return errors;
}

/*******************************************************************************
 * \brief Function to test the parallel FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_blocked(const int npts_global[3]) {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const int my_process = grid_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);

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
  const int my_number_of_elements_gs = product3(my_sizes_gs);

  // Check forward 3D FFTs
  double max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        memset(fft_grid->grid_rs, 0, my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] && nx <= my_bounds_rs[0][1] &&
            ny >= my_bounds_rs[1][0] && ny <= my_bounds_rs[1][1] &&
            nz >= my_bounds_rs[2][0] && nz <= my_bounds_rs[2][1])
          fft_grid->grid_rs[(nz - my_bounds_rs[2][0]) * my_sizes_rs[0] *
                                my_sizes_rs[1] +
                            (ny - my_bounds_rs[1][0]) * my_sizes_rs[0] +
                            (nx - my_bounds_rs[0][0])] = 1.0;

        fft_3d_fw_blocked(fft_grid->grid_rs, fft_grid->grid_gs,
                          fft_grid->npts_global, fft_grid->proc2local_rs,
                          fft_grid->proc2local_ms, fft_grid->proc2local_gs,
                          fft_grid->comm);

        for (int mx = 0; mx < my_sizes_gs[0]; mx++) {
          for (int my = 0; my < my_sizes_gs[1]; my++) {
            for (int mz = 0; mz < my_sizes_gs[2]; mz++) {
              const double complex my_value =
                  fft_grid->grid_gs[mz * my_sizes_gs[0] * my_sizes_gs[1] +
                                    my * my_sizes_gs[0] + mx];
              const double complex ref_value = cexp(
                  -2.0 * I * pi *
                  (((double)mx + my_bounds_gs[0][0]) * nx / npts_global[0] +
                   ((double)my + my_bounds_gs[1][0]) * ny / npts_global[1] +
                   ((double)mz + my_bounds_gs[2][0]) * nz / npts_global[2]));
              double current_error = cabs(my_value - ref_value);
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The 3D forward FFTs with blocked layout do not work properly (%i "
             "%i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  // Check backwards 3D FFTs
  max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        memset(fft_grid->grid_gs, 0,
               my_number_of_elements_gs * sizeof(double complex));

        if (nx >= my_bounds_gs[0][0] && nx <= my_bounds_gs[0][1] &&
            ny >= my_bounds_gs[1][0] && ny <= my_bounds_gs[1][1] &&
            nz >= my_bounds_gs[2][0] && nz <= my_bounds_gs[2][1])
          fft_grid->grid_gs[(nz - my_bounds_gs[2][0]) * my_sizes_gs[0] *
                                my_sizes_gs[1] +
                            (ny - my_bounds_gs[1][0]) * my_sizes_gs[0] +
                            (nx - my_bounds_gs[0][0])] = 1.0;

        fft_3d_bw_blocked(fft_grid->grid_gs, fft_grid->grid_rs,
                          fft_grid->npts_global, fft_grid->proc2local_rs,
                          fft_grid->proc2local_ms, fft_grid->proc2local_gs,
                          fft_grid->comm);

        for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
          for (int my = 0; my < my_sizes_rs[1]; my++) {
            for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
              const double my_value =
                  fft_grid->grid_rs[mz * my_sizes_rs[0] * my_sizes_rs[1] +
                                    my * my_sizes_rs[0] + mx];
              const double ref_value = cos(
                  2.0 * pi *
                  (((double)mx + my_bounds_rs[0][0]) * nx / npts_global[0] +
                   ((double)my + my_bounds_rs[1][0]) * ny / npts_global[1] +
                   ((double)mz + my_bounds_rs[2][0]) * nz / npts_global[2]));
              double current_error = fabs(my_value - ref_value);
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  grid_free_fft_grid(fft_grid);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The backwards 3D FFTs with blocked layout do not work properly "
             "(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 3D FFTs with blocked layout do work properly!\n");
  return errors;
}

/*******************************************************************************
 * \brief Function to test the parallel FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_ray(const int npts_global[3], const int npts_global_ref[3]) {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const int my_process = grid_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);

  grid_fft_grid *ref_grid = NULL;
  grid_create_fft_grid(&ref_grid, comm, npts_global_ref);

  grid_fft_grid *fft_grid = NULL;
  grid_create_fft_grid_from_reference(&fft_grid, npts_global, ref_grid);

  const int(*my_bounds_rs)[2] = fft_grid->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1] - my_bounds_rs[dir][0] + 1;
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int my_number_of_elements_gs =
      fft_grid->rays_per_process[my_process] * npts_global[0];

  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += fft_grid->rays_per_process[process];

  // Check forward 3D FFTs
  double max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        memset(fft_grid->grid_rs, 0, my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] && nx <= my_bounds_rs[0][1] &&
            ny >= my_bounds_rs[1][0] && ny <= my_bounds_rs[1][1] &&
            nz >= my_bounds_rs[2][0] && nz <= my_bounds_rs[2][1])
          fft_grid->grid_rs[(nz - my_bounds_rs[2][0]) * my_sizes_rs[0] *
                                my_sizes_rs[1] +
                            (ny - my_bounds_rs[1][0]) * my_sizes_rs[0] +
                            (nx - my_bounds_rs[0][0])] = 1.0;

        fft_3d_fw_ray(fft_grid->grid_rs, fft_grid->grid_gs,
                      fft_grid->npts_global, fft_grid->proc2local_rs,
                      fft_grid->proc2local_ms, fft_grid->yz_to_process,
                      fft_grid->rays_per_process, fft_grid->ray_to_yz,
                      fft_grid->comm);

        for (int index_x = 0; index_x < npts_global[0]; index_x++) {
          for (int yz_ray = 0; yz_ray < fft_grid->rays_per_process[my_process];
               yz_ray++) {
            const int index_y = fft_grid->ray_to_yz[my_ray_offset + yz_ray][0];
            const int index_z = fft_grid->ray_to_yz[my_ray_offset + yz_ray][1];
            const double complex my_value =
                fft_grid->grid_gs[yz_ray * npts_global[0] + index_x];
            const double complex ref_value =
                cexp(-2.0 * I * pi *
                     (((double)index_x) * nx / npts_global[0] +
                      ((double)index_y) * ny / npts_global[1] +
                      ((double)index_z) * nz / npts_global[2]));
            double current_error = cabs(my_value - ref_value);
            max_error = fmax(max_error, current_error);
          }
        }
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The 3D forward FFT with ray layout does not work properly (%i %i "
             "%i)/(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], max_error);
    errors++;
  }

  // Check backwards 3D FFTs
  int total_number_of_gs_elements = 0;
  for (int process = 0; process < grid_mpi_comm_size(comm); process++)
    total_number_of_gs_elements += fft_grid->rays_per_process[process];
  max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int nyz = 0; nyz < total_number_of_gs_elements; nyz++) {
      const int ny = fft_grid->ray_to_yz[nyz][0];
      const int nz = fft_grid->ray_to_yz[nyz][1];
      memset(fft_grid->grid_gs, 0,
             my_number_of_elements_gs * sizeof(double complex));

      if (nyz >= my_ray_offset &&
          nyz < my_ray_offset + fft_grid->rays_per_process[my_process]) {
        fft_grid->grid_gs[(nyz - my_ray_offset) * npts_global[0] + nx] = 1.0;
      }

      fft_3d_bw_ray(fft_grid->grid_gs, fft_grid->grid_rs, fft_grid->npts_global,
                    fft_grid->proc2local_rs, fft_grid->proc2local_ms,
                    fft_grid->yz_to_process, fft_grid->rays_per_process,
                    fft_grid->ray_to_yz, fft_grid->comm);

      for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
        for (int my = 0; my < my_sizes_rs[1]; my++) {
          for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
            const double my_value =
                fft_grid->grid_rs[mz * my_sizes_rs[0] * my_sizes_rs[1] +
                                  my * my_sizes_rs[0] + mx];
            const double ref_value =
                cos(2.0 * pi *
                    (((double)mx + my_bounds_rs[0][0]) * nx / npts_global[0] +
                     ((double)my + my_bounds_rs[1][0]) * ny / npts_global[1] +
                     ((double)mz + my_bounds_rs[2][0]) * nz / npts_global[2]));
            double current_error = fabs(my_value - ref_value);
            max_error = fmax(max_error, current_error);
          }
        }
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  grid_free_fft_grid(fft_grid);
  grid_free_fft_grid(ref_grid);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The backwards 3D FFT with ray layout does not work properly (%i "
             "%i %i)/(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], max_error);
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 3D FFT with ray layout does work properly (%i %i %i)/(%i %i "
           "%i)!\n",
           npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
           npts_global_ref[1], npts_global_ref[2]);
  return errors;
}

int fft_test_3d() {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

  int errors = 0;

  // Grid sizes to be checked
  const int npts_global[3] = {2, 4, 8};
  const int npts_global_small[3] = {2, 3, 5};
  const int npts_global_reverse[3] = {8, 4, 2};
  const int npts_global_small_reverse[3] = {5, 3, 2};

  // Check the blocked layout
  errors += fft_test_3d_blocked(npts_global);
  errors += fft_test_3d_blocked(npts_global_small);
  errors += fft_test_3d_blocked(npts_global_reverse);
  errors += fft_test_3d_blocked(npts_global_small_reverse);

  // Check the ray layout with the same grid sizes
  errors += fft_test_3d_ray(npts_global, npts_global);
  errors += fft_test_3d_ray(npts_global_small, npts_global_small);
  errors += fft_test_3d_ray(npts_global_reverse, npts_global_reverse);
  errors +=
      fft_test_3d_ray(npts_global_small_reverse, npts_global_small_reverse);

  // Check the ray layout with different grid sizes
  // errors += fft_test_3d_ray(npts_global_small, npts_global);
  // errors += fft_test_3d_ray(npts_global_small_reverse, npts_global_reverse);

  if (errors == 0 && my_process == 0)
    printf("\n The 3D FFT routines work properly!\n");
  return errors;
}

// EOF
