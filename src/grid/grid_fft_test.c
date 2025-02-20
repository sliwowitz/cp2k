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
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

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
  // Check the forward FFT
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

  // Check the backward FFT
  for (int dir = 0; dir < 3; dir++) {
    const int current_size = fft_sizes[dir];
    memset(input_array, 0,
           current_size * current_size * sizeof(double complex));

    for (int number_of_fft = 0; number_of_fft < current_size; number_of_fft++) {
      input_array[number_of_fft * current_size + number_of_fft] = 1.0;
    }

    fft_1d_bw(input_array, output_array, current_size, current_size);

    for (int number_of_fft = 0; number_of_fft < current_size; number_of_fft++) {
      for (int index = 0; index < current_size; index++) {
        error = fmax(
            error,
            cabs(output_array[number_of_fft * current_size + index] -
                 cexp(2.0 * I * pi * number_of_fft * index / current_size)));
      }
    }
  }

  free(input_array);
  free(output_array);

  if (error > 1e-12) {
    if (my_process == 0)
      printf("\nThe low-level FFTs do not work properly: %f!\n", error);
    return 1;
  } else {
    if (my_process == 0)
      printf("\nThe 1D FFTs do work properly!\n");
    return 0;
  }
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
      printf("\nThe low-level transpose does not work properly: %f!\n", error);
    return 1;
  } else {
    if (my_process == 0)
      printf("\nThe local transpose does work properly!\n");
    return 0;
  }
}

/*******************************************************************************
 * \brief Function to test the parallel transposition operation.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_transpose_parallel() {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const int my_process = grid_mpi_comm_rank(comm);

  // Use an asymmetric cell to check correctness of indices
  const int npts_global[3] = {2, 4, 8};

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
  const int my_number_of_elements_gs = product3(my_sizes_gs);
  (void)my_number_of_elements_gs;

  // Collect the maximum error
  double max_error = 0.0;

  // Check forward RS->MS FFTs
  for (int nx = 0; nx < my_sizes_rs[0]; nx++) {
    for (int ny = 0; ny < my_sizes_rs[1]; ny++) {
      for (int nz = 0; nz < my_sizes_rs[2]; nz++) {
        fft_grid->grid_rs_complex[nx * my_sizes_rs[1] * my_sizes_rs[2] +
                                  ny * my_sizes_rs[2] + nz] =
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
            fft_grid->grid_ms[nz * my_sizes_ms[0] * my_sizes_ms[1] +
                              nx * my_sizes_ms[1] + ny];
        const double complex ref_value =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
        double current_error = cabs(my_value - ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  }

  if (max_error > 1e-12) {
    grid_free_fft_grid(fft_grid);
    if (my_process == 0)
      printf("\nThe transpose xy_to_xz_blocked does not work properly: %f!\n",
             max_error);
    return 1;
  }

  memset(fft_grid->grid_rs_complex, 0,
         my_number_of_elements_rs * sizeof(double complex));

  // Check the reverse direction
  transpose_xz_to_xy_blocked(fft_grid->grid_ms, fft_grid->grid_rs_complex,
                             npts_global, fft_grid->proc2local_ms,
                             fft_grid->proc2local_rs, fft_grid->comm);

  // Check forward RS->MS FFTs
  for (int nx = 0; nx < my_sizes_rs[0]; nx++) {
    for (int ny = 0; ny < my_sizes_rs[1]; ny++) {
      for (int nz = 0; nz < my_sizes_rs[2]; nz++) {
        const double complex my_value =
            fft_grid->grid_rs_complex[nx * my_sizes_rs[1] * my_sizes_rs[2] +
                                      ny * my_sizes_rs[2] + nz];
        const double complex ref_value =
            ((nx + my_bounds_rs[0][0]) * npts_global[1] +
             (ny + my_bounds_rs[1][0])) +
            I * (nz + my_bounds_rs[2][0]);
        double current_error = cabs(my_value - ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  }

  if (max_error > 1e-12) {
    grid_free_fft_grid(fft_grid);
    if (my_process == 0)
      printf("\nThe transpose xz_to_xy_blocked does not work properly: %f!\n",
             max_error);
    return 1;
  }

  // Check the MS/GS direction
  transpose_xz_to_yz_blocked(fft_grid->grid_ms, fft_grid->grid_gs, npts_global,
                             fft_grid->proc2local_ms, fft_grid->proc2local_gs,
                             fft_grid->comm);

  // Check forward RS->MS FFTs
  for (int nx = 0; nx < my_sizes_gs[0]; nx++) {
    for (int ny = 0; ny < my_sizes_gs[1]; ny++) {
      for (int nz = 0; nz < my_sizes_gs[2]; nz++) {
        const double complex my_value =
            fft_grid->grid_gs[ny * my_sizes_gs[0] * my_sizes_gs[2] +
                              nz * my_sizes_gs[0] + nx];
        const double complex ref_value =
            ((nx + my_bounds_gs[0][0]) * npts_global[1] +
             (ny + my_bounds_gs[1][0])) +
            I * (nz + my_bounds_gs[2][0]);
        double current_error = cabs(my_value - ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  }

  if (max_error > 1e-12) {
    grid_free_fft_grid(fft_grid);
    if (my_process == 0)
      printf("\nThe transpose xz_to_yz_blocked does not work properly: %f!\n",
             max_error);
    return 1;
  }

  memset(fft_grid->grid_ms, 0, my_number_of_elements_ms);

  // Check the MS/GS direction
  transpose_yz_to_xz_blocked(fft_grid->grid_gs, fft_grid->grid_ms, npts_global,
                             fft_grid->proc2local_gs, fft_grid->proc2local_ms,
                             fft_grid->comm);

  // Check forward RS->MS FFTs
  for (int nx = 0; nx < my_sizes_ms[0]; nx++) {
    for (int ny = 0; ny < my_sizes_ms[1]; ny++) {
      for (int nz = 0; nz < my_sizes_ms[2]; nz++) {
        const double complex my_value =
            fft_grid->grid_ms[nz * my_sizes_ms[0] * my_sizes_ms[1] +
                              nx * my_sizes_ms[1] + ny];
        const double complex ref_value =
            ((nx + my_bounds_ms[0][0]) * npts_global[1] +
             (ny + my_bounds_ms[1][0])) +
            I * (nz + my_bounds_ms[2][0]);
        double current_error = cabs(my_value - ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  }

  if (max_error > 1e-12) {
    grid_free_fft_grid(fft_grid);
    if (my_process == 0)
      printf("\nThe transpose yz_to_xz_blocked does not work properly: %f!\n",
             max_error);
    return 1;
  }

  // Test ray transpositiond,
  grid_fft_grid *fft_grid_ray = NULL;
  grid_create_fft_grid_from_reference(&fft_grid_ray, npts_global, fft_grid);

  int my_bounds_ms_ray[3][2];
  memcpy(my_bounds_ms_ray, fft_grid_ray->proc2local_ms[my_process],
         sizeof(int[3][2]));
  int my_sizes_ms_ray[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_ms_ray[dir] =
        my_bounds_ms_ray[dir][1] - my_bounds_ms_ray[dir][0] + 1;

  int my_bounds_gs_ray[3][2];
  memcpy(my_bounds_gs_ray, fft_grid_ray->proc2local_gs[my_process],
         sizeof(int[3][2]));
  int my_sizes_gs_ray[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_gs_ray[dir] =
        my_bounds_gs_ray[dir][1] - my_bounds_gs_ray[dir][0] + 1;

  for (int index_x = 0; index_x < my_sizes_ms_ray[0]; index_x++) {
    for (int index_y = 0; index_y < my_sizes_ms_ray[1]; index_y++) {
      for (int index_z = 0; index_z < my_sizes_ms_ray[2]; index_z++) {
        fft_grid_ray
            ->grid_ms[index_z * my_sizes_ms_ray[0] * my_sizes_ms_ray[1] +
                      index_x * my_sizes_ms_ray[1] + index_y] =
            ((index_y + my_bounds_ms_ray[1][0]) * npts_global[2] +
             (index_z + my_bounds_ms_ray[2][0])) +
            I * (index_x + my_bounds_ms_ray[0][0]);
      }
    }
  }

  transpose_xz_to_yz_ray(fft_grid_ray->grid_ms, fft_grid_ray->grid_gs,
                         fft_grid_ray->npts_global, fft_grid_ray->proc2local_ms,
                         fft_grid_ray->yz_to_process, fft_grid_ray->comm);

  for (int yz_ray = 0; yz_ray < fft_grid_ray->rays_per_process[my_process];
       yz_ray++) {
    const int index_y = fft_grid_ray->ray_number_to_yz[yz_ray][0];
    const int index_z = fft_grid_ray->ray_number_to_yz[yz_ray][1];
    for (int index_x = 0; index_x < npts_global[0]; index_x++) {
      const double complex my_value =
          fft_grid_ray->grid_gs[yz_ray * npts_global[0] + index_x];
      const double complex ref_value =
          (index_y * npts_global[2] + index_z) + I * index_x;
      double current_error = cabs(my_value - ref_value);
      if (current_error > 0.1)
        fprintf(stderr, "%i %i %i: (%f, %f) (%f, %f)\n", index_x, index_y,
                index_z, creal(my_value), cimag(my_value), creal(ref_value),
                cimag(ref_value));
      max_error = fmax(max_error, current_error);
    }
  }

  if (max_error > 1e-12) {
    grid_free_fft_grid(fft_grid_ray);
    grid_free_fft_grid(fft_grid);
    if (my_process == 0)
      printf("\nThe transpose xz_to_yz_blocked does not work properly: %f!\n",
             max_error);
    return 1;
  }

  memset(fft_grid_ray->grid_gs, 0,
         product3(my_sizes_gs_ray) * sizeof(double complex));

  for (int yz_ray = 0; yz_ray < fft_grid_ray->rays_per_process[my_process];
       yz_ray++) {
    const int index_y = fft_grid_ray->ray_number_to_yz[yz_ray][0];
    const int index_z = fft_grid_ray->ray_number_to_yz[yz_ray][1];
    for (int index_x = 0; index_x < npts_global[0]; index_x++) {
      fft_grid_ray->grid_gs[yz_ray * npts_global[0] + index_x] =
          (index_y * npts_global[2] + index_z) + I * index_x;
    }
  }
  transpose_yz_to_xz_ray(fft_grid_ray->grid_gs, fft_grid_ray->grid_ms,
                         fft_grid_ray->npts_global, fft_grid_ray->yz_to_process,
                         fft_grid_ray->proc2local_ms, fft_grid_ray->comm);

  for (int index_y = 0; index_y < my_sizes_ms_ray[1]; index_y++) {
    for (int index_z = 0; index_z < my_sizes_ms_ray[2]; index_z++) {
      // Check whether there is a ray with the given index pair
      bool found = false;
      for (int yz_ray = 0; yz_ray < npts_global[1] * npts_global[2]; yz_ray++) {
        if (index_y == fft_grid_ray->ray_number_to_yz[yz_ray][0] &&
            index_z == fft_grid_ray->ray_number_to_yz[yz_ray][1]) {
          found = true;
          break;
        }
      }
      if (found) {
        for (int index_x = 0; index_x < npts_global[0]; index_x++) {
          const double complex my_value =
              fft_grid_ray->grid_ms[index_z * npts_global[0] * npts_global[1] +
                                    index_x * npts_global[1] + index_y];
          const double complex ref_value =
              ((index_y + my_bounds_ms_ray[1][0]) * npts_global[2] +
               (index_z + my_bounds_ms_ray[2][0])) +
              I * (index_z + my_bounds_ms_ray[0][0]);
          double current_error = cabs(my_value - ref_value);
          if (current_error > 0.1)
            fprintf(stderr, "yz_to_xz: %i %i %i: (%f, %f) (%f, %f)\n", index_x,
                    index_y, index_z, creal(my_value), cimag(my_value),
                    creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      } else {
        for (int index_x = 0; index_x < npts_global[0]; index_x++) {
          const double complex my_value =
              fft_grid_ray->grid_ms[index_z * npts_global[0] * npts_global[1] +
                                    index_x * npts_global[1] + index_y];
          // The value is assumed to be zero if absent
          const double complex ref_value = 0.0;
          double current_error = cabs(my_value - ref_value);
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }

  grid_free_fft_grid(fft_grid_ray);
  grid_free_fft_grid(fft_grid);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("\nThe transpose_yz_to_xz_ray does not work properly: %f!\n",
             max_error);
    return 1;
  }

  if (my_process == 0)
    printf("\n The parallel transposition routines work properly!\n");
  return 0;
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
  const int my_number_of_elements_gs = product3(my_sizes_gs);

  // Check forward 3D FFTs
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

  if (error > 1e-12) {
    grid_free_fft_grid(fft_grid);
    if (my_process == 0)
      printf("\nThe 3D forward FFTs do not work properly: %f!\n", error);
    return 1;
  }

  // Check backwards 3D FFTs
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        memset(fft_grid->grid_gs, 0,
               my_number_of_elements_gs * sizeof(double complex));

        if (nx >= my_bounds_gs[0][0] && nx <= my_bounds_gs[0][1] &&
            ny >= my_bounds_gs[1][0] && ny <= my_bounds_gs[1][1] &&
            nz >= my_bounds_gs[2][0] && nz <= my_bounds_gs[2][1])
          fft_grid->grid_gs[(ny - my_bounds_gs[1][0]) * my_sizes_gs[0] *
                                my_sizes_gs[2] +
                            (nz - my_bounds_gs[2][0]) * my_sizes_gs[0] +
                            (nx - my_bounds_gs[0][0])] = 1.0;

        fft_3d_bw_blocked(fft_grid->grid_gs, fft_grid->grid_rs,
                          fft_grid->npts_global, fft_grid->proc2local_rs,
                          fft_grid->proc2local_ms, fft_grid->proc2local_gs,
                          fft_grid->comm);

        for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
          for (int my = 0; my < my_sizes_rs[1]; my++) {
            for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
              const double my_value =
                  fft_grid->grid_rs[mx * my_sizes_rs[1] * my_sizes_rs[2] +
                                    my * my_sizes_rs[2] + mz];
              const double ref_value = creal(cexp(
                  2.0 * I * pi *
                  (((double)mx + my_bounds_rs[0][0]) * nx / npts_global[0] +
                   ((double)my + my_bounds_rs[1][0]) * ny / npts_global[1] +
                   ((double)mz + my_bounds_rs[2][0]) * nz / npts_global[2])));
              double current_error = fabs(my_value - ref_value);
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
    if (my_process == 0)
      printf("\nThe 3D FFTs do not work properly: %f!\n", error);
    return 1;
  } else {
    if (my_process == 0)
      printf("\nThe 3D FFTs do work properly!\n");
    return 0;
  }
}

// EOF
