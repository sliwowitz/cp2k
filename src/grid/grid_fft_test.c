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
#include "grid_fft_methods.h"

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

  fft_1d_fw_local(input_array, output_array, fft_size, number_of_ffts);

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

  fft_1d_bw_local(input_array, output_array, fft_size, number_of_ffts);

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
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  // Build the reference grid
  grid_fft_grid_layout *ref_grid_layout = NULL;
  grid_create_fft_grid_layout(&ref_grid_layout, comm, npts_global_ref, dh_inv);

  // Test ray transpositiond,
  grid_fft_grid_layout *fft_grid_ray_layout = NULL;
  grid_create_fft_grid_layout_from_reference(&fft_grid_ray_layout, npts_global,
                                             ref_grid_layout);

  int my_bounds_ms_ray[3][2];
  memcpy(my_bounds_ms_ray, fft_grid_ray_layout->proc2local_ms[my_process],
         sizeof(int[3][2]));
  int my_sizes_ms_ray[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_ms_ray[dir] =
        my_bounds_ms_ray[dir][1] - my_bounds_ms_ray[dir][0] + 1;

  for (int index_x = 0; index_x < my_sizes_ms_ray[0]; index_x++) {
    for (int index_y = 0; index_y < my_sizes_ms_ray[1]; index_y++) {
      for (int index_z = 0; index_z < my_sizes_ms_ray[2]; index_z++) {
        fft_grid_ray_layout
            ->grid_ms[index_x * my_sizes_ms_ray[1] * my_sizes_ms_ray[2] +
                      index_z * my_sizes_ms_ray[1] + index_y] =
            ((index_y + my_bounds_ms_ray[1][0]) *
                 fft_grid_ray_layout->npts_global[2] +
             (index_z + my_bounds_ms_ray[2][0])) +
            I * (index_x + my_bounds_ms_ray[0][0]);
      }
    }
  }

  collect_x_and_distribute_y_ray(
      fft_grid_ray_layout->grid_ms, fft_grid_ray_layout->grid_gs,
      fft_grid_ray_layout->npts_global, fft_grid_ray_layout->proc2local_ms,
      fft_grid_ray_layout->yz_to_process, fft_grid_ray_layout->rays_per_process,
      fft_grid_ray_layout->ray_to_yz, fft_grid_ray_layout->comm);

  max_error = 0.0;
  int ray_index_offset = 0;
  for (int process = 0; process < my_process; process++)
    ray_index_offset += fft_grid_ray_layout->rays_per_process[process];
  for (int yz_ray = 0;
       yz_ray < fft_grid_ray_layout->rays_per_process[my_process]; yz_ray++) {
    const int index_y =
        fft_grid_ray_layout->ray_to_yz[ray_index_offset + yz_ray][0];
    const int index_z =
        fft_grid_ray_layout->ray_to_yz[ray_index_offset + yz_ray][1];
    for (int index_x = 0; index_x < fft_grid_ray_layout->npts_global[0];
         index_x++) {
      const double complex my_value =
          fft_grid_ray_layout
              ->grid_gs[index_x *
                            fft_grid_ray_layout->rays_per_process[my_process] +
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

  memset(fft_grid_ray_layout->grid_gs, 0,
         fft_grid_ray_layout->npts_global[0] *
             fft_grid_ray_layout->rays_per_process[my_process] *
             sizeof(double complex));
  memset(fft_grid_ray_layout->grid_ms, 0,
         product3(my_sizes_ms_ray) * sizeof(double complex));

  for (int yz_ray = 0;
       yz_ray < fft_grid_ray_layout->rays_per_process[my_process]; yz_ray++) {
    const int index_y =
        fft_grid_ray_layout->ray_to_yz[ray_index_offset + yz_ray][0];
    const int index_z =
        fft_grid_ray_layout->ray_to_yz[ray_index_offset + yz_ray][1];
    for (int index_x = 0; index_x < fft_grid_ray_layout->npts_global[0];
         index_x++) {
      fft_grid_ray_layout
          ->grid_gs[index_x *
                        fft_grid_ray_layout->rays_per_process[my_process] +
                    yz_ray] =
          (index_y * fft_grid_ray_layout->npts_global[2] + index_z) +
          I * index_x;
    }
  }
  collect_y_and_distribute_x_ray(
      fft_grid_ray_layout->grid_gs, fft_grid_ray_layout->grid_ms,
      fft_grid_ray_layout->npts_global, fft_grid_ray_layout->yz_to_process,
      fft_grid_ray_layout->proc2local_ms, fft_grid_ray_layout->rays_per_process,
      fft_grid_ray_layout->ray_to_yz, fft_grid_ray_layout->comm);

  max_error = 0.0;
  for (int index_y = 0; index_y < my_sizes_ms_ray[1]; index_y++) {
    for (int index_z = 0; index_z < my_sizes_ms_ray[2]; index_z++) {
      // Check whether there is a ray with the given index pair
      if (fft_grid_ray_layout
              ->yz_to_process[(index_y + my_bounds_ms_ray[1][0]) *
                                  fft_grid_ray_layout->npts_global[2] +
                              (index_z + my_bounds_ms_ray[2][0])] >= 0) {
        for (int index_x = 0; index_x < my_sizes_ms_ray[0]; index_x++) {
          const double complex my_value =
              fft_grid_ray_layout
                  ->grid_ms[index_x * my_sizes_ms_ray[1] * my_sizes_ms_ray[2] +
                            index_z * my_sizes_ms_ray[1] + index_y];
          const double complex ref_value =
              ((index_y + my_bounds_ms_ray[1][0]) *
                   fft_grid_ray_layout->npts_global[2] +
               (index_z + my_bounds_ms_ray[2][0])) +
              I * (index_x + my_bounds_ms_ray[0][0]);
          double current_error = cabs(my_value - ref_value);
          max_error = fmax(max_error, current_error);
        }
      } else {
        for (int index_x = 0; index_x < fft_grid_ray_layout->npts_global[0];
             index_x++) {
          const double complex my_value =
              fft_grid_ray_layout
                  ->grid_ms[index_x * fft_grid_ray_layout->npts_global[1] *
                                fft_grid_ray_layout->npts_global[2] +
                            index_z * fft_grid_ray_layout->npts_global[1] +
                            index_y];
          // The value is assumed to be zero if absent
          const double complex ref_value = 0.0;
          double current_error = cabs(my_value - ref_value);
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }

  grid_free_fft_grid_layout(fft_grid_ray_layout);
  grid_free_fft_grid_layout(ref_grid_layout);

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
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  grid_fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout(&fft_grid_layout, comm, npts_global, dh_inv);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1] - my_bounds_rs[dir][0] + 1;
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int(*my_bounds_ms)[2] = fft_grid_layout->proc2local_ms[my_process];
  int my_sizes_ms[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_ms[dir] = my_bounds_ms[dir][1] - my_bounds_ms[dir][0] + 1;
  const int my_number_of_elements_ms = product3(my_sizes_ms);

  const int(*my_bounds_gs)[2] = fft_grid_layout->proc2local_gs[my_process];
  int my_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_gs[dir] = my_bounds_gs[dir][1] - my_bounds_gs[dir][0] + 1;

  // Collect the maximum error
  double max_error = 0.0;

  // Check forward RS->MS FFTs
  for (int nx = 0; nx < my_sizes_rs[0]; nx++) {
    for (int ny = 0; ny < my_sizes_rs[1]; ny++) {
      for (int nz = 0; nz < my_sizes_rs[2]; nz++) {
        fft_grid_layout->grid_rs_complex[ny * my_sizes_rs[0] * my_sizes_rs[2] +
                                         nx * my_sizes_rs[2] + nz] =
            ((nx + my_bounds_rs[0][0]) * npts_global[1] +
             (ny + my_bounds_rs[1][0])) +
            I * (nz + my_bounds_rs[2][0]);
      }
    }
  }

  collect_y_and_distribute_z_blocked(
      fft_grid_layout->grid_rs_complex, fft_grid_layout->grid_ms, npts_global,
      fft_grid_layout->proc2local_rs, fft_grid_layout->proc2local_ms,
      fft_grid_layout->comm);

  for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
    for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
      for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
        const double complex my_value =
            fft_grid_layout->grid_ms[ny * my_sizes_ms[0] * my_sizes_ms[2] +
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
        fft_grid_layout->grid_ms[ny * my_sizes_ms[0] * my_sizes_ms[2] +
                                 nx * my_sizes_ms[2] + nz] =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
      }
    }
  }
  memset(fft_grid_layout->grid_rs_complex, 0,
         my_number_of_elements_rs * sizeof(double complex));

  // Check the reverse direction
  collect_z_and_distribute_y_blocked(
      fft_grid_layout->grid_ms, fft_grid_layout->grid_rs_complex, npts_global,
      fft_grid_layout->proc2local_ms, fft_grid_layout->proc2local_rs,
      fft_grid_layout->comm);

  // Check forward RS->MS FFTs
  max_error = 0.0;
  for (int nx = 0; nx < my_sizes_rs[0]; nx++) {
    for (int ny = 0; ny < my_sizes_rs[1]; ny++) {
      for (int nz = 0; nz < my_sizes_rs[2]; nz++) {
        const double complex my_value =
            fft_grid_layout
                ->grid_rs_complex[ny * my_sizes_rs[0] * my_sizes_rs[2] +
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
        fft_grid_layout->grid_ms[nx * my_sizes_ms[1] * my_sizes_ms[2] +
                                 nz * my_sizes_ms[1] + ny] =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
      }
    }
  }

  // Check the MS/GS direction
  collect_y_and_distribute_x_blocked(
      fft_grid_layout->grid_ms, fft_grid_layout->grid_gs, npts_global,
      fft_grid_layout->proc2local_ms, fft_grid_layout->proc2local_gs,
      fft_grid_layout->comm);

  // Check forward RS->MS FFTs
  max_error = 0.0;
  for (int nx = 0; nx < my_sizes_gs[0]; nx++) {
    for (int ny = 0; ny < my_sizes_gs[1]; ny++) {
      for (int nz = 0; nz < my_sizes_gs[2]; nz++) {
        const double complex my_value =
            fft_grid_layout->grid_gs[nx * my_sizes_gs[1] * my_sizes_gs[2] +
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
        fft_grid_layout->grid_gs[nx * my_sizes_gs[1] * my_sizes_gs[2] +
                                 nz * my_sizes_gs[1] + ny] =
            ((nx + my_bounds_gs[0][0]) * npts_global[1] +
             (ny + my_bounds_gs[1][0])) +
            I * (nz + my_bounds_gs[2][0]);
      }
    }
  }
  memset(fft_grid_layout->grid_ms, 0, my_number_of_elements_ms);

  // Check the MS/GS direction
  collect_y_and_distribute_x_blocked(
      fft_grid_layout->grid_gs, fft_grid_layout->grid_ms, npts_global,
      fft_grid_layout->proc2local_gs, fft_grid_layout->proc2local_ms,
      fft_grid_layout->comm);

  // Check forward RS->MS FFTs
  max_error = 0.0;
  for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
    for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
      for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
        const double complex my_value =
            fft_grid_layout->grid_ms[nx * my_sizes_ms[1] * my_sizes_ms[2] +
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

  grid_free_fft_grid_layout(fft_grid_layout);
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
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  grid_fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout(&fft_grid_layout, comm, npts_global, dh_inv);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1] - my_bounds_rs[dir][0] + 1;
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int(*my_bounds_gs)[2] = fft_grid_layout->proc2local_gs[my_process];
  int my_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_gs[dir] = my_bounds_gs[dir][1] - my_bounds_gs[dir][0] + 1;
  const int my_number_of_elements_gs = product3(my_sizes_gs);

  grid_fft_real_rs_grid grid_rs;
  grid_create_real_rs_grid(&grid_rs, fft_grid_layout);
  grid_fft_complex_gs_grid grid_gs;
  grid_create_complex_gs_grid(&grid_gs, fft_grid_layout);

  // Check forward 3D FFTs
  double max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        memset(fft_grid_layout->grid_rs, 0,
               my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] && nx <= my_bounds_rs[0][1] &&
            ny >= my_bounds_rs[1][0] && ny <= my_bounds_rs[1][1] &&
            nz >= my_bounds_rs[2][0] && nz <= my_bounds_rs[2][1])
          fft_grid_layout->grid_rs[(nz - my_bounds_rs[2][0]) * my_sizes_rs[0] *
                                       my_sizes_rs[1] +
                                   (ny - my_bounds_rs[1][0]) * my_sizes_rs[0] +
                                   (nx - my_bounds_rs[0][0])] = 1.0;

        fft_3d_fw_blocked(
            fft_grid_layout->grid_rs, fft_grid_layout->grid_gs,
            fft_grid_layout->npts_global, fft_grid_layout->proc2local_rs,
            fft_grid_layout->proc2local_ms, fft_grid_layout->proc2local_gs,
            fft_grid_layout->comm);

        for (int mx = 0; mx < my_sizes_gs[0]; mx++) {
          for (int my = 0; my < my_sizes_gs[1]; my++) {
            for (int mz = 0; mz < my_sizes_gs[2]; mz++) {
              const double complex my_value =
                  fft_grid_layout
                      ->grid_gs[mz * my_sizes_gs[0] * my_sizes_gs[1] +
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
      printf(
          "The 3D forward FFT with blocked layout does not work properly (%i "
          "%i %i): %f!\n",
          npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  // Check forward 3D FFTs
  max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {

        memset(grid_rs.data, 0, my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] && nx <= my_bounds_rs[0][1] &&
            ny >= my_bounds_rs[1][0] && ny <= my_bounds_rs[1][1] &&
            nz >= my_bounds_rs[2][0] && nz <= my_bounds_rs[2][1])
          grid_rs.data[(nz - my_bounds_rs[2][0]) * my_sizes_rs[0] *
                           my_sizes_rs[1] +
                       (ny - my_bounds_rs[1][0]) * my_sizes_rs[0] +
                       (nx - my_bounds_rs[0][0])] = 1.0;

        fft_3d_fw(&grid_rs, &grid_gs);

        for (int index = 0; index < my_number_of_elements_gs; index++) {
          const int mx = fft_grid_layout->index_to_g[index][0];
          const int my = fft_grid_layout->index_to_g[index][1];
          const int mz = fft_grid_layout->index_to_g[index][2];
          const double complex my_value = grid_gs.data[index];
          const double complex ref_value =
              cexp(-2.0 * I * pi *
                   (((double)mx) * nx / npts_global[0] +
                    ((double)my) * ny / npts_global[1] +
                    ((double)mz) * nz / npts_global[2]));
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("%i ERROR %i %i %i/%i %i %i: (%f %f) (%f %f)\n", my_process,
                   nx, ny, nz, mx, my, mz, creal(my_value), cimag(my_value),
                   creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The 3D forward FFT (blocked) to ordered layout does not work "
             "properly (%i "
             "%i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  // Check backwards 3D FFTs
  max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        memset(fft_grid_layout->grid_gs, 0,
               my_number_of_elements_gs * sizeof(double complex));

        if (nx >= my_bounds_gs[0][0] && nx <= my_bounds_gs[0][1] &&
            ny >= my_bounds_gs[1][0] && ny <= my_bounds_gs[1][1] &&
            nz >= my_bounds_gs[2][0] && nz <= my_bounds_gs[2][1])
          fft_grid_layout->grid_gs[(nz - my_bounds_gs[2][0]) * my_sizes_gs[0] *
                                       my_sizes_gs[1] +
                                   (ny - my_bounds_gs[1][0]) * my_sizes_gs[0] +
                                   (nx - my_bounds_gs[0][0])] = 1.0;

        fft_3d_bw_blocked(
            fft_grid_layout->grid_gs, fft_grid_layout->grid_rs,
            fft_grid_layout->npts_global, fft_grid_layout->proc2local_rs,
            fft_grid_layout->proc2local_ms, fft_grid_layout->proc2local_gs,
            fft_grid_layout->comm);

        for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
          for (int my = 0; my < my_sizes_rs[1]; my++) {
            for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
              const double my_value =
                  fft_grid_layout
                      ->grid_rs[mz * my_sizes_rs[0] * my_sizes_rs[1] +
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

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The backwards 3D FFT with blocked layout does not work properly "
             "(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  // Check backwards 3D FFTs
  max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        memset(grid_gs.data, 0,
               my_number_of_elements_gs * sizeof(double complex));

        for (int index = 0; index < my_number_of_elements_gs; index++) {
          if (nx == fft_grid_layout->index_to_g[index][0] &&
              ny == fft_grid_layout->index_to_g[index][1] &&
              nz == fft_grid_layout->index_to_g[index][2]) {
            grid_gs.data[index] = 1.0;
            break;
          }
        }

        fft_3d_bw(&grid_gs, &grid_rs);

        for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
          for (int my = 0; my < my_sizes_rs[1]; my++) {
            for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
              const double my_value =
                  grid_rs.data[mz * my_sizes_rs[0] * my_sizes_rs[1] +
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

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The backwards 3D FFT (blocked) to ordered layout does not work "
             "properly "
             "(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  grid_free_real_rs_grid(&grid_rs);
  grid_free_complex_gs_grid(&grid_gs);
  grid_free_fft_grid_layout(fft_grid_layout);

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
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  grid_fft_grid_layout *ref_grid_layout = NULL;
  grid_create_fft_grid_layout(&ref_grid_layout, comm, npts_global_ref, dh_inv);

  grid_fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout_from_reference(&fft_grid_layout, npts_global,
                                             ref_grid_layout);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1] - my_bounds_rs[dir][0] + 1;
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int my_number_of_elements_gs =
      fft_grid_layout->rays_per_process[my_process] * npts_global[0];

  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += fft_grid_layout->rays_per_process[process];

  // Check forward 3D FFTs
  double max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        memset(fft_grid_layout->grid_rs, 0,
               my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] && nx <= my_bounds_rs[0][1] &&
            ny >= my_bounds_rs[1][0] && ny <= my_bounds_rs[1][1] &&
            nz >= my_bounds_rs[2][0] && nz <= my_bounds_rs[2][1])
          fft_grid_layout->grid_rs[(nz - my_bounds_rs[2][0]) * my_sizes_rs[0] *
                                       my_sizes_rs[1] +
                                   (ny - my_bounds_rs[1][0]) * my_sizes_rs[0] +
                                   (nx - my_bounds_rs[0][0])] = 1.0;

        fft_3d_fw_ray(
            fft_grid_layout->grid_rs, fft_grid_layout->grid_gs,
            fft_grid_layout->npts_global, fft_grid_layout->proc2local_rs,
            fft_grid_layout->proc2local_ms, fft_grid_layout->yz_to_process,
            fft_grid_layout->rays_per_process, fft_grid_layout->ray_to_yz,
            fft_grid_layout->comm);

        for (int index_x = 0; index_x < npts_global[0]; index_x++) {
          for (int yz_ray = 0;
               yz_ray < fft_grid_layout->rays_per_process[my_process];
               yz_ray++) {
            const int index_y =
                fft_grid_layout->ray_to_yz[my_ray_offset + yz_ray][0];
            const int index_z =
                fft_grid_layout->ray_to_yz[my_ray_offset + yz_ray][1];
            const double complex my_value =
                fft_grid_layout->grid_gs[yz_ray * npts_global[0] + index_x];
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
    total_number_of_gs_elements += fft_grid_layout->rays_per_process[process];
  max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int nyz = 0; nyz < total_number_of_gs_elements; nyz++) {
      const int ny = fft_grid_layout->ray_to_yz[nyz][0];
      const int nz = fft_grid_layout->ray_to_yz[nyz][1];
      memset(fft_grid_layout->grid_gs, 0,
             my_number_of_elements_gs * sizeof(double complex));

      if (nyz >= my_ray_offset &&
          nyz < my_ray_offset + fft_grid_layout->rays_per_process[my_process]) {
        fft_grid_layout->grid_gs[(nyz - my_ray_offset) * npts_global[0] + nx] =
            1.0;
      }

      fft_3d_bw_ray(
          fft_grid_layout->grid_gs, fft_grid_layout->grid_rs,
          fft_grid_layout->npts_global, fft_grid_layout->proc2local_rs,
          fft_grid_layout->proc2local_ms, fft_grid_layout->yz_to_process,
          fft_grid_layout->rays_per_process, fft_grid_layout->ray_to_yz,
          fft_grid_layout->comm);

      for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
        for (int my = 0; my < my_sizes_rs[1]; my++) {
          for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
            const double my_value =
                fft_grid_layout->grid_rs[mz * my_sizes_rs[0] * my_sizes_rs[1] +
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

  grid_free_fft_grid_layout(fft_grid_layout);
  grid_free_fft_grid_layout(ref_grid_layout);

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
  const int npts_global_x[3] = {8, 1, 1};
  const int npts_global_y[3] = {1, 8, 1};
  const int npts_global_z[3] = {1, 1, 8};
  const int npts_global_yz[3] = {1, 8, 8};
  const int npts_global_xz[3] = {8, 1, 8};
  const int npts_global_xy[3] = {8, 8, 1};
  const int npts_global[3] = {2, 4, 8};
  const int npts_global_small[3] = {2, 3, 5};
  const int npts_global_reverse[3] = {8, 4, 2};
  const int npts_global_small_reverse[3] = {5, 3, 2};

  // Check the blocked layout
  errors += fft_test_3d_blocked(npts_global);
  errors += fft_test_3d_blocked(npts_global_small);
  errors += fft_test_3d_blocked(npts_global_reverse);
  errors += fft_test_3d_blocked(npts_global_small_reverse);
  errors += fft_test_3d_blocked(npts_global_x);
  errors += fft_test_3d_blocked(npts_global_y);
  errors += fft_test_3d_blocked(npts_global_z);
  errors += fft_test_3d_blocked(npts_global_yz);
  errors += fft_test_3d_blocked(npts_global_xz);
  errors += fft_test_3d_blocked(npts_global_xy);

  // Check the ray layout with the same grid sizes
  errors += fft_test_3d_ray(npts_global_y, npts_global_yz);
  errors += fft_test_3d_ray(npts_global_z, npts_global_yz);
  errors += fft_test_3d_ray(npts_global_x, npts_global_xz);
  errors += fft_test_3d_ray(npts_global_z, npts_global_xz);
  errors += fft_test_3d_ray(npts_global_x, npts_global_xy);
  errors += fft_test_3d_ray(npts_global_y, npts_global_xy);
  errors += fft_test_3d_ray(npts_global, npts_global);
  errors += fft_test_3d_ray(npts_global_small, npts_global_small);
  errors += fft_test_3d_ray(npts_global_reverse, npts_global_reverse);
  errors +=
      fft_test_3d_ray(npts_global_small_reverse, npts_global_small_reverse);

  if (errors == 0 && my_process == 0)
    printf("\n The 3D FFT routines work properly!\n");
  return errors;
}

int fft_test_add_copy_low(const int npts_global_fine[3],
                          const int npts_global_coarse[3]) {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  grid_fft_grid_layout *fft_grid_fine_layout = NULL;
  grid_create_fft_grid_layout(&fft_grid_fine_layout, comm, npts_global_fine,
                              dh_inv);

  grid_fft_grid_layout *fft_grid_coarse_layout = NULL;
  grid_create_fft_grid_layout_from_reference(
      &fft_grid_coarse_layout, npts_global_coarse, fft_grid_fine_layout);

  int errors = 0;

  grid_free_fft_grid_layout(fft_grid_fine_layout);
  grid_free_fft_grid_layout(fft_grid_coarse_layout);

  return errors;
}

/*******************************************************************************
 * \brief Function to test the addition/copy between different grids.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_add_copy() {
  int errors = 0;

  const int npts_globals[][3] = {{2, 4, 8}, {2, 3, 5}};

  errors += fft_test_add_copy_low(npts_globals[0], npts_globals[1]);

  return errors;
}

// EOF
