/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft.h"
#include "common/grid_common.h"
#include "common/grid_mpi.h"

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

inline double norm_vector(const double complex *vector, const int size,
                          const grid_mpi_comm comm) {
  double norm = 0.0;
  for (int i = 0; i < size; i++)
    norm += creal(vector[i]) * creal(vector[i]) +
            cimag(vector[i]) * cimag(vector[i]);
  grid_mpi_sum_double(&norm, 1, comm);
  return sqrt(norm);
}

inline double norm_vector_double(const double *vector, const int size,
                                 const grid_mpi_comm comm) {
  double norm = 0.0;
  for (int i = 0; i < size; i++)
    norm += vector[i] * vector[i];
  grid_mpi_sum_double(&norm, 1, comm);
  return sqrt(norm);
}

/*******************************************************************************
 * \brief Naive implementation of FFT. To be replaced by a real FFT library
 *later. \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw(const double complex *grid_rs, double complex *grid_gs,
               const int fft_size, const int number_of_ffts) {
  const double pi = acos(-1.0);
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_rs, grid_gs, fft_size, number_of_ffts, pi)
  for (int fft = 0; fft < number_of_ffts; fft++) {
    for (int index_out = 0; index_out < fft_size; index_out++) {
      double complex tmp = 0.0;
      for (int index_in = 0; index_in < fft_size; index_in++) {
        tmp += grid_rs[fft * fft_size + index_in] *
               cexp(-2.0 * I * pi * index_out * index_in / fft_size);
      }
      grid_gs[fft * fft_size + index_out] = tmp;
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT. To be replaced by a real FFT library
 *later. \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw(const double complex *grid_gs, double complex *grid_rs,
               const int fft_size, const int number_of_ffts) {
  const double pi = acos(-1.0);
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_rs, grid_gs, fft_size, number_of_ffts, pi)
  for (int fft = 0; fft < number_of_ffts; fft++) {
    for (int index_out = 0; index_out < fft_size; index_out++) {
      double complex tmp = 0.0;
      for (int index_in = 0; index_in < fft_size; index_in++) {
        tmp += grid_gs[fft * fft_size + index_in] *
               cexp(2.0 * I * pi * index_out * index_in / fft_size);
      }
      grid_rs[fft * fft_size + index_out] = tmp;
    }
  }
}

void transpose_local(double complex *grid, double complex *grid_transposed,
                     const int number_of_columns_grid,
                     const int number_of_rows_grid) {
#pragma omp parallel for collapse(2) default(none)                             \
    shared(grid, grid_transposed, number_of_columns_grid, number_of_rows_grid)
  for (int column_index = 0; column_index < number_of_columns_grid;
       column_index++) {
    for (int row_index = 0; row_index < number_of_rows_grid; row_index++) {
      grid_transposed[column_index * number_of_rows_grid + row_index] =
          grid[row_index * number_of_columns_grid + column_index];
    }
  }
}

void fft_3d_fw(double complex *grid_rs, double complex *grid_gs,
               const int npts_global[3]) {

  // Perform the first FFT along z
  fft_1d_fw(grid_rs, grid_gs, npts_global[2], npts_global[0] * npts_global[1]);

  // Perform first transposition (x, y, z) -> (z, x, y)
  transpose_local(grid_gs, grid_rs, npts_global[2],
                  npts_global[0] * npts_global[1]);

  // Perform the second FFT along y
  fft_1d_fw(grid_rs, grid_gs, npts_global[1], npts_global[0] * npts_global[2]);

  // Perform second transpose (z, x, y) -> (y, z, x)
  transpose_local(grid_gs, grid_rs, npts_global[1],
                  npts_global[0] * npts_global[2]);

  // Perform the third FFT along x
  fft_1d_fw(grid_rs, grid_gs, npts_global[0], npts_global[1] * npts_global[2]);
}

void fft_3d_bw(double complex *grid_gs, double complex *grid_rs,
               const int npts_global[3]) {

  // Perform the first FFT along x
  fft_1d_bw(grid_gs, grid_rs, npts_global[0], npts_global[1] * npts_global[2]);

  // Perform first transposition (y, z, x) -> (z, x, y)
  transpose_local(grid_rs, grid_gs, npts_global[0] * npts_global[2],
                  npts_global[1]);

  // Perform the second FFT along y
  fft_1d_fw(grid_gs, grid_rs, npts_global[1], npts_global[0] * npts_global[2]);

  // Perform second transpose (z, x, y) -> (x, y, z)
  transpose_local(grid_rs, grid_gs, npts_global[0] * npts_global[1],
                  npts_global[2]);

  // Perform the third FFT along z
  fft_1d_fw(grid_gs, grid_rs, npts_global[2], npts_global[0] * npts_global[1]);
}

/*******************************************************************************
 * \brief Performs a transposition of (x,y,z)->(z,x,y).
 * \author Frederick Stein
 ******************************************************************************/
void transpose_xy_to_xz_blocked(const double complex *grid,
                                double complex *transposed,
                                const int npts_global[3],
                                const int (*proc2local)[3][2],
                                const int (*proc2local_transposed)[3][2],
                                const grid_mpi_comm comm) {
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);
  (void)npts_global;

  int max_number_of_elements = 0;
  for (int process = 0; process < number_of_processes; process++) {
    max_number_of_elements =
        imax(max_number_of_elements,
             (proc2local[process][0][1] - proc2local[process][0][0] + 1) *
                 (proc2local[process][1][1] - proc2local[process][1][0] + 1) *
                 (proc2local[process][2][1] - proc2local[process][2][0] + 1));
  }
  double complex *recv_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  grid_mpi_request recv_request, send_request;

  const int my_number_of_elements =
      (proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1) *
      (proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1) *
      (proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1);

  int my_original_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_original_sizes[dir] =
        proc2local[my_process][dir][1] - proc2local[my_process][dir][0] + 1;

  int my_transposed_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_transposed_sizes[dir] = proc2local_transposed[my_process][dir][1] -
                               proc2local_transposed[my_process][dir][0] + 1;

  // Copy and transpose the data
  for (int index_x = imax(proc2local[my_process][0][0],
                          proc2local_transposed[my_process][0][0]);
       index_x <= imin(proc2local[my_process][0][1],
                       proc2local_transposed[my_process][0][1]);
       index_x++) {
    for (int index_y = imax(proc2local[my_process][1][0],
                            proc2local_transposed[my_process][1][0]);
         index_y <= imin(proc2local[my_process][1][1],
                         proc2local_transposed[my_process][1][1]);
         index_y++) {
      for (int index_z = imax(proc2local[my_process][2][0],
                              proc2local_transposed[my_process][2][0]);
           index_z <= imin(proc2local[my_process][2][1],
                           proc2local_transposed[my_process][2][1]);
           index_z++) {
        transposed[(index_z - proc2local_transposed[my_process][2][0]) *
                       my_transposed_sizes[0] * my_transposed_sizes[1] +
                   (index_x - proc2local_transposed[my_process][0][0]) *
                       my_transposed_sizes[1] +
                   (index_y - proc2local_transposed[my_process][1][0])] =
            grid[(index_x - proc2local[my_process][0][0]) *
                     my_original_sizes[1] * my_original_sizes[2] +
                 (index_y - proc2local[my_process][1][0]) *
                     my_original_sizes[2] +
                 (index_z - proc2local[my_process][2][0])];
      }
    }
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    int recv_sizes[3];
    for (int dir = 0; dir < 3; dir++)
      recv_sizes[dir] = proc2local[recv_process][dir][1] -
                        proc2local[recv_process][dir][0] + 1;

    // Post receive request
    grid_mpi_irecv_double_complex(recv_buffer, product3(recv_sizes),
                                  recv_process, 1, comm, &recv_request);

    // Post send request
    grid_mpi_isend_double_complex(grid, my_number_of_elements, send_process, 1,
                                  comm, &send_request);

    // Wait for the receive process and copy the data
    grid_mpi_wait(&recv_request);

    // Copy and transpose the data
    for (int index_x = imax(proc2local[recv_process][0][0],
                            proc2local_transposed[my_process][0][0]);
         index_x <= imin(proc2local[recv_process][0][1],
                         proc2local_transposed[my_process][0][1]);
         index_x++) {
      for (int index_y = imax(proc2local[recv_process][1][0],
                              proc2local_transposed[my_process][1][0]);
           index_y <= imin(proc2local[recv_process][1][1],
                           proc2local_transposed[my_process][1][1]);
           index_y++) {
        for (int index_z = imax(proc2local[recv_process][2][0],
                                proc2local_transposed[my_process][2][0]);
             index_z <= imin(proc2local[recv_process][2][1],
                             proc2local_transposed[my_process][2][1]);
             index_z++) {
          transposed[(index_z - proc2local_transposed[my_process][2][0]) *
                         my_transposed_sizes[0] * my_transposed_sizes[1] +
                     (index_x - proc2local_transposed[my_process][0][0]) *
                         my_transposed_sizes[1] +
                     (index_y - proc2local_transposed[my_process][1][0])] =
              recv_buffer[(index_x - proc2local[recv_process][0][0]) *
                              recv_sizes[1] * recv_sizes[2] +
                          (index_y - proc2local[recv_process][1][0]) *
                              recv_sizes[2] +
                          (index_z - proc2local[recv_process][2][0])];
        }
      }
    }

    // Wait for the send request
    grid_mpi_wait(&send_request);
  }

  free(recv_buffer);
}

/*******************************************************************************
 * \brief Performs a transposition of (x,y,z)->(z,x,y).
 * \author Frederick Stein
 ******************************************************************************/
void transpose_xz_to_xy_blocked(const double complex *grid,
                                double complex *transposed,
                                const int npts_global[3],
                                const int (*proc2local)[3][2],
                                const int (*proc2local_transposed)[3][2],
                                const grid_mpi_comm comm) {
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);
  (void)npts_global;

  int max_number_of_elements = 0;
  for (int process = 0; process < number_of_processes; process++) {
    max_number_of_elements =
        imax(max_number_of_elements,
             (proc2local[process][0][1] - proc2local[process][0][0] + 1) *
                 (proc2local[process][1][1] - proc2local[process][1][0] + 1) *
                 (proc2local[process][2][1] - proc2local[process][2][0] + 1));
  }
  double complex *recv_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  grid_mpi_request recv_request, send_request;

  const int my_number_of_elements =
      (proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1) *
      (proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1) *
      (proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1);

  int my_original_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_original_sizes[dir] =
        proc2local[my_process][dir][1] - proc2local[my_process][dir][0] + 1;

  int my_transposed_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_transposed_sizes[dir] = proc2local_transposed[my_process][dir][1] -
                               proc2local_transposed[my_process][dir][0] + 1;

  // Copy and transpose the data
  for (int index_x = imax(proc2local[my_process][0][0],
                          proc2local_transposed[my_process][0][0]);
       index_x <= imin(proc2local[my_process][0][1],
                       proc2local_transposed[my_process][0][1]);
       index_x++) {
    for (int index_y = imax(proc2local[my_process][1][0],
                            proc2local_transposed[my_process][1][0]);
         index_y <= imin(proc2local[my_process][1][1],
                         proc2local_transposed[my_process][1][1]);
         index_y++) {
      for (int index_z = imax(proc2local[my_process][2][0],
                              proc2local_transposed[my_process][2][0]);
           index_z <= imin(proc2local[my_process][2][1],
                           proc2local_transposed[my_process][2][1]);
           index_z++) {
        transposed[(index_x - proc2local_transposed[my_process][0][0]) *
                       my_transposed_sizes[1] * my_transposed_sizes[2] +
                   (index_y - proc2local_transposed[my_process][1][0]) *
                       my_transposed_sizes[2] +
                   (index_z - proc2local_transposed[my_process][2][0])] =
            grid[(index_z - proc2local[my_process][2][0]) *
                     my_original_sizes[0] * my_original_sizes[1] +
                 (index_x - proc2local[my_process][0][0]) *
                     my_original_sizes[1] +
                 (index_y - proc2local[my_process][1][0])];
      }
    }
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    int recv_sizes[3];
    for (int dir = 0; dir < 3; dir++)
      recv_sizes[dir] = proc2local[recv_process][dir][1] -
                        proc2local[recv_process][dir][0] + 1;

    // Post receive request
    grid_mpi_irecv_double_complex(recv_buffer, product3(recv_sizes),
                                  recv_process, 1, comm, &recv_request);

    // Post send request
    grid_mpi_isend_double_complex(grid, my_number_of_elements, send_process, 1,
                                  comm, &send_request);

    // Wait for the receive process and copy the data
    grid_mpi_wait(&recv_request);

    // Copy and transpose the data
    for (int index_x = imax(proc2local[recv_process][0][0],
                            proc2local_transposed[my_process][0][0]);
         index_x <= imin(proc2local[recv_process][0][1],
                         proc2local_transposed[my_process][0][1]);
         index_x++) {
      for (int index_y = imax(proc2local[recv_process][1][0],
                              proc2local_transposed[my_process][1][0]);
           index_y <= imin(proc2local[recv_process][1][1],
                           proc2local_transposed[my_process][1][1]);
           index_y++) {
        for (int index_z = imax(proc2local[recv_process][2][0],
                                proc2local_transposed[my_process][2][0]);
             index_z <= imin(proc2local[recv_process][2][1],
                             proc2local_transposed[my_process][2][1]);
             index_z++) {
          transposed[(index_x - proc2local_transposed[my_process][0][0]) *
                         my_transposed_sizes[1] * my_transposed_sizes[2] +
                     (index_y - proc2local_transposed[my_process][1][0]) *
                         my_transposed_sizes[2] +
                     (index_z - proc2local_transposed[my_process][2][0])] =
              recv_buffer[(index_z - proc2local[recv_process][2][0]) *
                              recv_sizes[0] * recv_sizes[1] +
                          (index_x - proc2local[recv_process][0][0]) *
                              recv_sizes[1] +
                          (index_y - proc2local[recv_process][1][0])];
        }
      }
    }

    // Wait for the send request
    grid_mpi_wait(&send_request);
  }

  free(recv_buffer);
}

/*******************************************************************************
 * \brief Performs a transposition of (z,x,y)->(y,z,x).
 * \author Frederick Stein
 ******************************************************************************/
void transpose_xz_to_yz_blocked(const double complex *grid,
                                double complex *transposed,
                                const int npts_global[3],
                                const int (*proc2local)[3][2],
                                const int (*proc2local_transposed)[3][2],
                                const grid_mpi_comm comm) {
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);
  (void)npts_global;

  int max_number_of_elements = 0;
  for (int process = 0; process < number_of_processes; process++) {
    max_number_of_elements =
        imax(max_number_of_elements,
             (proc2local[process][0][1] - proc2local[process][0][0] + 1) *
                 (proc2local[process][1][1] - proc2local[process][1][0] + 1) *
                 (proc2local[process][2][1] - proc2local[process][2][0] + 1));
  }
  double complex *recv_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  grid_mpi_request recv_request, send_request;

  const int my_number_of_elements =
      (proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1) *
      (proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1) *
      (proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1);

  int my_original_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_original_sizes[dir] =
        proc2local[my_process][dir][1] - proc2local[my_process][dir][0] + 1;

  int my_transposed_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_transposed_sizes[dir] = proc2local_transposed[my_process][dir][1] -
                               proc2local_transposed[my_process][dir][0] + 1;

  // Copy and transpose the data
  for (int index_x = imax(proc2local[my_process][0][0],
                          proc2local_transposed[my_process][0][0]);
       index_x <= imin(proc2local[my_process][0][1],
                       proc2local_transposed[my_process][0][1]);
       index_x++) {
    for (int index_y = imax(proc2local[my_process][1][0],
                            proc2local_transposed[my_process][1][0]);
         index_y <= imin(proc2local[my_process][1][1],
                         proc2local_transposed[my_process][1][1]);
         index_y++) {
      for (int index_z = imax(proc2local[my_process][2][0],
                              proc2local_transposed[my_process][2][0]);
           index_z <= imin(proc2local[my_process][2][1],
                           proc2local_transposed[my_process][2][1]);
           index_z++) {
        transposed[(index_y - proc2local_transposed[my_process][1][0]) *
                       my_transposed_sizes[0] * my_transposed_sizes[2] +
                   (index_z - proc2local_transposed[my_process][2][0]) *
                       my_transposed_sizes[0] +
                   (index_x - proc2local_transposed[my_process][0][0])] =
            grid[(index_z - proc2local[my_process][2][0]) *
                     my_original_sizes[0] * my_original_sizes[1] +
                 (index_x - proc2local[my_process][0][0]) *
                     my_original_sizes[1] +
                 (index_y - proc2local[my_process][1][0])];
      }
    }
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    int recv_sizes[3];
    for (int dir = 0; dir < 3; dir++)
      recv_sizes[dir] = proc2local[recv_process][dir][1] -
                        proc2local[recv_process][dir][0] + 1;

    // Post receive request
    grid_mpi_irecv_double_complex(recv_buffer, product3(recv_sizes),
                                  recv_process, 1, comm, &recv_request);

    // Post send request
    grid_mpi_isend_double_complex(grid, my_number_of_elements, send_process, 1,
                                  comm, &send_request);

    // Wait for the receive process and copy the data
    grid_mpi_wait(&recv_request);

    // Copy and transpose the data
    for (int index_x = imax(proc2local[recv_process][0][0],
                            proc2local_transposed[my_process][0][0]);
         index_x <= imin(proc2local[recv_process][0][1],
                         proc2local_transposed[my_process][0][1]);
         index_x++) {
      for (int index_y = imax(proc2local[recv_process][1][0],
                              proc2local_transposed[my_process][1][0]);
           index_y <= imin(proc2local[recv_process][1][1],
                           proc2local_transposed[my_process][1][1]);
           index_y++) {
        for (int index_z = imax(proc2local[recv_process][2][0],
                                proc2local_transposed[my_process][2][0]);
             index_z <= imin(proc2local[recv_process][2][1],
                             proc2local_transposed[my_process][2][1]);
             index_z++) {
          transposed[(index_y - proc2local_transposed[my_process][1][0]) *
                         my_transposed_sizes[0] * my_transposed_sizes[2] +
                     (index_z - proc2local_transposed[my_process][2][0]) *
                         my_transposed_sizes[0] +
                     (index_x - proc2local_transposed[my_process][0][0])] =
              recv_buffer[(index_z - proc2local[recv_process][2][0]) *
                              recv_sizes[0] * recv_sizes[1] +
                          (index_x - proc2local[recv_process][0][0]) *
                              recv_sizes[1] +
                          (index_y - proc2local[recv_process][1][0])];
        }
      }
    }

    // Wait for the send request
    grid_mpi_wait(&send_request);
  }

  free(recv_buffer);
}

/*******************************************************************************
 * \brief Performs a transposition of (z,x,y)->(y,z,x).
 * \author Frederick Stein
 ******************************************************************************/
void transpose_yz_to_xz_blocked(const double complex *grid,
                                double complex *transposed,
                                const int npts_global[3],
                                const int (*proc2local)[3][2],
                                const int (*proc2local_transposed)[3][2],
                                const grid_mpi_comm comm) {
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);
  (void)npts_global;

  int max_number_of_elements = 0;
  for (int process = 0; process < number_of_processes; process++) {
    max_number_of_elements =
        imax(max_number_of_elements,
             (proc2local[process][0][1] - proc2local[process][0][0] + 1) *
                 (proc2local[process][1][1] - proc2local[process][1][0] + 1) *
                 (proc2local[process][2][1] - proc2local[process][2][0] + 1));
  }
  double complex *recv_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  grid_mpi_request recv_request, send_request;

  const int my_number_of_elements =
      (proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1) *
      (proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1) *
      (proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1);

  int my_original_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_original_sizes[dir] =
        proc2local[my_process][dir][1] - proc2local[my_process][dir][0] + 1;

  int my_transposed_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_transposed_sizes[dir] = proc2local_transposed[my_process][dir][1] -
                               proc2local_transposed[my_process][dir][0] + 1;

  // Copy and transpose the data
  for (int index_x = imax(proc2local[my_process][0][0],
                          proc2local_transposed[my_process][0][0]);
       index_x <= imin(proc2local[my_process][0][1],
                       proc2local_transposed[my_process][0][1]);
       index_x++) {
    for (int index_y = imax(proc2local[my_process][1][0],
                            proc2local_transposed[my_process][1][0]);
         index_y <= imin(proc2local[my_process][1][1],
                         proc2local_transposed[my_process][1][1]);
         index_y++) {
      for (int index_z = imax(proc2local[my_process][2][0],
                              proc2local_transposed[my_process][2][0]);
           index_z <= imin(proc2local[my_process][2][1],
                           proc2local_transposed[my_process][2][1]);
           index_z++) {
        transposed[(index_z - proc2local_transposed[my_process][2][0]) *
                       my_transposed_sizes[0] * my_transposed_sizes[1] +
                   (index_x - proc2local_transposed[my_process][0][0]) *
                       my_transposed_sizes[1] +
                   (index_y - proc2local_transposed[my_process][1][0])] =
            grid[(index_y - proc2local[my_process][1][0]) *
                     my_original_sizes[0] * my_original_sizes[2] +
                 (index_z - proc2local[my_process][2][0]) *
                     my_original_sizes[0] +
                 (index_x - proc2local[my_process][0][0])];
      }
    }
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    int recv_sizes[3];
    for (int dir = 0; dir < 3; dir++)
      recv_sizes[dir] = proc2local[recv_process][dir][1] -
                        proc2local[recv_process][dir][0] + 1;

    // Post receive request
    grid_mpi_irecv_double_complex(recv_buffer, product3(recv_sizes),
                                  recv_process, 1, comm, &recv_request);

    // Post send request
    grid_mpi_isend_double_complex(grid, my_number_of_elements, send_process, 1,
                                  comm, &send_request);

    // Wait for the receive process and copy the data
    grid_mpi_wait(&recv_request);

    // Copy and transpose the data
    for (int index_x = imax(proc2local[recv_process][0][0],
                            proc2local_transposed[my_process][0][0]);
         index_x <= imin(proc2local[recv_process][0][1],
                         proc2local_transposed[my_process][0][1]);
         index_x++) {
      for (int index_y = imax(proc2local[recv_process][1][0],
                              proc2local_transposed[my_process][1][0]);
           index_y <= imin(proc2local[recv_process][1][1],
                           proc2local_transposed[my_process][1][1]);
           index_y++) {
        for (int index_z = imax(proc2local[recv_process][2][0],
                                proc2local_transposed[my_process][2][0]);
             index_z <= imin(proc2local[recv_process][2][1],
                             proc2local_transposed[my_process][2][1]);
             index_z++) {
          transposed[(index_z - proc2local_transposed[my_process][2][0]) *
                         my_transposed_sizes[0] * my_transposed_sizes[1] +
                     (index_x - proc2local_transposed[my_process][0][0]) *
                         my_transposed_sizes[1] +
                     (index_y - proc2local_transposed[my_process][1][0])] =
              recv_buffer[(index_y - proc2local[recv_process][1][0]) *
                              recv_sizes[0] * recv_sizes[2] +
                          (index_z - proc2local[recv_process][2][0]) *
                              recv_sizes[0] +
                          (index_x - proc2local[recv_process][0][0])];
        }
      }
    }

    // Wait for the send request
    grid_mpi_wait(&send_request);
  }

  free(recv_buffer);
}

/*******************************************************************************
 * \brief Performs a transposition of (z,x,y)->(y,z,x).
 * \author Frederick Stein
 ******************************************************************************/
void transpose_xz_to_yz_ray(const double complex *grid,
                            double complex *transposed,
                            const int npts_global[3],
                            const int (*proc2local)[3][2],
                            const int *yz_to_process, const int *number_of_rays,
                            const int (*ray_to_yz)[2],
                            const grid_mpi_comm comm) {
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);
  (void)npts_global;
  (void)yz_to_process;
  (void)grid;
  (void)transposed;
  (void)number_of_rays;
  (void)ray_to_yz;

  int max_number_of_elements = 0;
  for (int process = 0; process < number_of_processes; process++) {
    max_number_of_elements =
        imax(max_number_of_elements,
             (proc2local[process][0][1] - proc2local[process][0][0] + 1) *
                 (proc2local[process][1][1] - proc2local[process][1][0] + 1) *
                 (proc2local[process][2][1] - proc2local[process][2][0] + 1));
  }
  double complex *recv_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  grid_mpi_request recv_request, send_request;
  (void)recv_request;
  (void)send_request;

  int my_original_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_original_sizes[dir] =
        proc2local[my_process][dir][1] - proc2local[my_process][dir][0] + 1;
  const int my_number_of_elements = product3(my_original_sizes);
  (void)my_number_of_elements;

  // Copy and transpose the local data
  int number_of_received_rays = 0;
  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += number_of_rays[process];
  for (int yz_ray = 0; yz_ray < number_of_rays[my_process]; yz_ray++) {
    const int index_y = ray_to_yz[my_ray_offset + yz_ray][0];
    const int index_z = ray_to_yz[my_ray_offset + yz_ray][1];

    if (index_y < proc2local[my_process][1][0] ||
        index_y > proc2local[my_process][1][1])
      continue;
    if (index_z < proc2local[my_process][2][0] ||
        index_z > proc2local[my_process][2][1])
      continue;

    printf("%i DEBUG %i %i\n", my_process, yz_ray, npts_global[0]);
    assert(npts_global[0] >= 0);
    transposed[yz_ray * npts_global[0]] = 0.0;

    // Copy the data
    for (int index_x = proc2local[my_process][0][0];
         index_x <= proc2local[my_process][0][1]; index_x++) {
      assert(yz_ray * npts_global[0] + index_x >= 0);
      assert(yz_ray * npts_global[0] + index_x <
             npts_global[0] * number_of_rays[my_process]);
      transposed[yz_ray * npts_global[0] + index_x] =
          grid[(index_z - proc2local[my_process][2][0]) * my_original_sizes[0] *
                   my_original_sizes[1] +
               (index_x - proc2local[my_process][0][0]) * my_original_sizes[1] +
               (index_y - proc2local[my_process][1][0])];
      printf("%i %i %i: (%f %f)\n", index_x, index_y, index_z,
             creal(transposed[yz_ray * npts_global[0] + index_x]),
             cimag(transposed[yz_ray * npts_global[0] + index_x]));
    }
    printf("%i Copy in xz_to_yz (process %i) %i %i to %i\n", my_process,
           my_process, index_y, index_z, yz_ray);
    number_of_received_rays++;
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    int recv_sizes[3];
    for (int dir = 0; dir < 3; dir++)
      recv_sizes[dir] = proc2local[recv_process][dir][1] -
                        proc2local[recv_process][dir][0] + 1;

    // Post receive request
    grid_mpi_irecv_double_complex(recv_buffer, product3(recv_sizes),
                                  recv_process, 1, comm, &recv_request);

    // Post send request
    grid_mpi_isend_double_complex(grid, my_number_of_elements, send_process, 1,
                                  comm, &send_request);

    // Wait for the receive process and copy the data
    grid_mpi_wait(&recv_request);

    // Copy and transpose the data
    for (int yz_ray = 0; yz_ray < number_of_rays[my_process]; yz_ray++) {
      const int index_y = ray_to_yz[my_ray_offset + yz_ray][0];
      const int index_z = ray_to_yz[my_ray_offset + yz_ray][1];

      if (index_y < proc2local[recv_process][1][0] ||
          index_y > proc2local[recv_process][1][1])
        continue;
      if (index_z < proc2local[recv_process][2][0] ||
          index_z > proc2local[recv_process][2][1])
        continue;

      // Copy the data
      for (int index_x = proc2local[recv_process][0][0];
           index_x <= proc2local[recv_process][0][1]; index_x++) {
        assert(yz_ray * npts_global[0] + index_x >= 0);
        assert(yz_ray * npts_global[0] + index_x <
               npts_global[0] * number_of_rays[my_process]);
        transposed[yz_ray * npts_global[0] + index_x] =
            recv_buffer[(index_z - proc2local[recv_process][2][0]) *
                            recv_sizes[0] * recv_sizes[1] +
                        (index_x - proc2local[recv_process][0][0]) *
                            recv_sizes[1] +
                        (index_y - proc2local[recv_process][1][0])];
      }
      printf("%i: Copy in xz_to_yz (process %i) %i %i to %i\n", my_process,
             recv_process, index_y, index_z, yz_ray);
      number_of_received_rays++;
    }

    // Wait for the send request
    grid_mpi_wait(&send_request);
    printf("%i sent %i elements to %i\n", my_process, my_number_of_elements,
           send_process);
  }

  printf("Process %i: Received %i rays\n", my_process, number_of_received_rays);
  fflush(stdout);
  grid_mpi_barrier(comm);
  // assert(number_of_received_rays == number_of_rays[my_process]);

  free(recv_buffer);
}

/*******************************************************************************
 * \brief Performs a transposition of (z,x,y)->(y,z,x).
 * \author Frederick Stein
 ******************************************************************************/
void transpose_yz_to_xz_ray(const double complex *grid,
                            double complex *transposed,
                            const int npts_global[3], const int *yz_to_process,
                            const int (*proc2local_transposed)[3][2],
                            const int *number_of_rays,
                            const int (*ray_to_yz)[2],
                            const grid_mpi_comm comm) {
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);
  (void)npts_global;
  (void)yz_to_process;

  int max_number_of_elements = 0;
  for (int process = 0; process < number_of_processes; process++)
    max_number_of_elements =
        imax(max_number_of_elements, number_of_rays[process]);
  double complex *recv_buffer =
      malloc(max_number_of_elements * npts_global[0] * sizeof(double complex));
  grid_mpi_request recv_request, send_request;

  const int my_number_of_elements = npts_global[0] * number_of_rays[my_process];

  int my_transposed_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_transposed_sizes[dir] = proc2local_transposed[my_process][dir][1] -
                               proc2local_transposed[my_process][dir][0] + 1;

  memset(transposed, 0, product3(my_transposed_sizes) * sizeof(double complex));

  // Copy and transpose the local data
  int number_of_received_rays = 0;
  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += number_of_rays[process];
  for (int yz_ray = 0; yz_ray < number_of_rays[my_process]; yz_ray++) {
    const int index_y = ray_to_yz[my_ray_offset + yz_ray][0];
    const int index_z = ray_to_yz[my_ray_offset + yz_ray][1];

    // Check whether we carry that ray after the transposition
    if (index_y < proc2local_transposed[my_process][1][0] ||
        index_y > proc2local_transposed[my_process][1][1])
      continue;
    if (index_z < proc2local_transposed[my_process][2][0] ||
        index_z > proc2local_transposed[my_process][2][1])
      continue;

    // printf("%i yz_ray %i: %i %i\n", my_process, yz_ray, index_y, index_z);

    // Copy the data
    for (int index_x = proc2local_transposed[my_process][0][0];
         index_x <= proc2local_transposed[my_process][0][1]; index_x++) {
      transposed[(index_z - proc2local_transposed[my_process][2][0]) *
                     my_transposed_sizes[0] * my_transposed_sizes[1] +
                 (index_x - proc2local_transposed[my_process][0][0]) *
                     my_transposed_sizes[1] +
                 (index_y - proc2local_transposed[my_process][1][0])] =
          grid[yz_ray * npts_global[0] + index_x];
      printf("Copy in yz_to_xz (%i) %i to %i %i: (%f %f)\n", index_x,
             my_ray_offset + yz_ray, index_y, index_z,
             creal(grid[yz_ray * npts_global[0] + index_x]),
             cimag(grid[yz_ray * npts_global[0] + index_x]));
    }
    number_of_received_rays++;
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    const int number_of_yz_rays = number_of_rays[recv_process];

    // Post receive request
    grid_mpi_irecv_double_complex(recv_buffer,
                                  npts_global[0] * number_of_yz_rays,
                                  recv_process, 1, comm, &recv_request);

    // Post send request
    grid_mpi_isend_double_complex(grid, my_number_of_elements, send_process, 1,
                                  comm, &send_request);

    int recv_ray_offset = 0;
    for (int process = 0; process < recv_process; process++)
      recv_ray_offset += number_of_rays[process];

    // Wait for the receive process and copy the data
    grid_mpi_wait(&recv_request);

    // Copy and transpose the data
    for (int yz_ray = 0; yz_ray < number_of_rays[recv_process]; yz_ray++) {
      const int index_y = ray_to_yz[recv_ray_offset + yz_ray][0];
      const int index_z = ray_to_yz[recv_ray_offset + yz_ray][1];

      if (index_y < proc2local_transposed[my_process][1][0] ||
          index_y > proc2local_transposed[my_process][1][1])
        continue;
      if (index_z < proc2local_transposed[my_process][2][0] ||
          index_z > proc2local_transposed[my_process][2][1])
        continue;

      // Copy the data
      for (int index_x = proc2local_transposed[my_process][0][0];
           index_x <= proc2local_transposed[my_process][0][1]; index_x++) {
        transposed[(index_z - proc2local_transposed[my_process][2][0]) *
                       my_transposed_sizes[0] * my_transposed_sizes[1] +
                   (index_x - proc2local_transposed[my_process][0][0]) *
                       my_transposed_sizes[1] +
                   (index_y - proc2local_transposed[my_process][1][0])] =
            recv_buffer[yz_ray * npts_global[0] + index_x];
        printf("Copy in yz_to_xz (%i) %i to %i %i: (%f %f)\n", index_x,
               recv_ray_offset + yz_ray, index_y, index_z,
               creal(recv_buffer[yz_ray * npts_global[0] + index_x]),
               cimag(recv_buffer[yz_ray * npts_global[0] + index_x]));
      }
      number_of_received_rays++;
    }

    // Wait for the send request
    grid_mpi_wait(&send_request);
  }
  grid_mpi_sum_int(&number_of_received_rays, 1, comm);
  int total_number_of_rays = 0;
  for (int process = 0; process < number_of_processes; process++)
    total_number_of_rays += number_of_rays[process];

  printf("Process %i: Received %i from %i rays\n", my_process,
         number_of_received_rays, total_number_of_rays);
  fflush(stdout);
  grid_mpi_barrier(comm);
  // assert(number_of_received_rays == total_number_of_rays);

  free(recv_buffer);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_blocked(double *grid_rs, double complex *grid_gs,
                       const int npts_global[3],
                       const int (*proc2local_rs)[3][2],
                       const int (*proc2local_ms)[3][2],
                       const int (*proc2local_gs)[3][2],
                       const grid_mpi_comm comm) {
  const int my_process = grid_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  const int number_of_elements_rs = product3(fft_sizes_rs);
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  const int number_of_elements_ms = product3(fft_sizes_ms);
  int fft_sizes_gs[3] = {
      proc2local_gs[my_process][0][1] - proc2local_gs[my_process][0][0] + 1,
      proc2local_gs[my_process][1][1] - proc2local_gs[my_process][1][0] + 1,
      proc2local_gs[my_process][2][1] - proc2local_gs[my_process][2][0] + 1};
  const int number_of_elements_gs = product3(fft_sizes_gs);
  const int size_of_buffer =
      imax(imax(number_of_elements_rs, number_of_elements_ms),
           number_of_elements_gs);
  double complex *grid_buffer_1 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));
  double complex *grid_buffer_2 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));

  // Copy real array to complex buffer
  for (int i = 0; i < number_of_elements_rs; i++) {
    grid_buffer_1[i] = grid_rs[i];
  }

  if (grid_mpi_comm_size(comm) > 1) {
    // Perform the first FFT
    fft_1d_fw(grid_buffer_1, grid_buffer_2, fft_sizes_rs[2],
              fft_sizes_rs[0] * fft_sizes_rs[1]);

    // Perform transpose
    transpose_xy_to_xz_blocked(grid_buffer_2, grid_buffer_1, npts_global,
                               proc2local_rs, proc2local_ms, comm);

    // Perform the second FFT
    fft_1d_fw(grid_buffer_1, grid_buffer_2, fft_sizes_ms[1],
              fft_sizes_ms[0] * fft_sizes_ms[2]);

    // Perform second transpose
    transpose_xz_to_yz_blocked(grid_buffer_2, grid_buffer_1, npts_global,
                               proc2local_ms, proc2local_gs, comm);

    // Perform the third FFT
    fft_1d_fw(grid_buffer_1, grid_gs, fft_sizes_gs[0],
              fft_sizes_gs[1] * fft_sizes_gs[2]);
  } else {
    fft_3d_fw(grid_buffer_1, grid_gs, npts_global);
  }

  free(grid_buffer_1);
  free(grid_buffer_2);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_blocked(double complex *grid_gs, double *grid_rs,
                       const int npts_global[3],
                       const int (*proc2local_rs)[3][2],
                       const int (*proc2local_ms)[3][2],
                       const int (*proc2local_gs)[3][2],
                       const grid_mpi_comm comm) {
  const int my_process = grid_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  const int number_of_elements_rs = product3(fft_sizes_rs);
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  const int number_of_elements_ms = product3(fft_sizes_ms);
  int fft_sizes_gs[3] = {
      proc2local_gs[my_process][0][1] - proc2local_gs[my_process][0][0] + 1,
      proc2local_gs[my_process][1][1] - proc2local_gs[my_process][1][0] + 1,
      proc2local_gs[my_process][2][1] - proc2local_gs[my_process][2][0] + 1};
  const int number_of_elements_gs = product3(fft_sizes_gs);
  const int size_of_buffer =
      imax(imax(number_of_elements_rs, number_of_elements_ms),
           number_of_elements_gs);
  double complex *grid_buffer_1 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));
  double complex *grid_buffer_2 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));

  if (grid_mpi_comm_size(comm) > 1) {
    // Perform the first FFT
    fft_1d_bw((const double complex *)grid_gs, grid_buffer_1, fft_sizes_gs[0],
              fft_sizes_gs[1] * fft_sizes_gs[2]);

    // Perform transpose
    transpose_yz_to_xz_blocked(grid_buffer_1, grid_buffer_2, npts_global,
                               proc2local_gs, proc2local_ms, comm);

    // Perform the second FFT
    fft_1d_bw((const double complex *)grid_buffer_2, grid_buffer_1,
              fft_sizes_ms[1], fft_sizes_ms[0] * fft_sizes_ms[2]);

    // Perform second transpose
    transpose_xz_to_xy_blocked(grid_buffer_1, grid_buffer_2, npts_global,
                               proc2local_ms, proc2local_rs, comm);

    // Perform the third FFT
    fft_1d_bw((const double complex *)grid_buffer_2, grid_buffer_1,
              fft_sizes_rs[2], fft_sizes_rs[0] * fft_sizes_rs[1]);
  } else {
    fft_3d_bw(grid_gs, grid_buffer_1, npts_global);
  }

  // Copy real array to complex buffer
  for (int i = 0; i < number_of_elements_rs; i++) {
    grid_rs[i] = creal(grid_buffer_1[i]);
  }
  fflush(stdout);
  fflush(stderr);
  grid_mpi_barrier(comm);

  free(grid_buffer_1);
  free(grid_buffer_2);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_ray(double *grid_rs, double complex *grid_gs,
                   const int npts_global[3], const int (*proc2local_rs)[3][2],
                   const int (*proc2local_ms)[3][2], const int *yz_to_process,
                   const int *rays_per_process, const int (*ray_to_yz)[2],
                   const grid_mpi_comm comm) {
  const int my_process = grid_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  const int number_of_elements_rs = product3(fft_sizes_rs);
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  const int number_of_elements_ms = product3(fft_sizes_ms);
  int number_of_local_yz_rays = 0;
  for (int yz_ray = 0; yz_ray < npts_global[1] * npts_global[2]; yz_ray++) {
    if (yz_to_process[yz_ray] == my_process)
      number_of_local_yz_rays++;
  }
  const int number_of_elements_gs = number_of_local_yz_rays * npts_global[0];
  const int size_of_buffer =
      imax(imax(number_of_elements_rs, number_of_elements_ms),
           number_of_elements_gs);
  double complex *grid_buffer_1 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));
  double complex *grid_buffer_2 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));

  // Copy real array to complex buffer
  for (int i = 0; i < number_of_elements_rs; i++) {
    grid_buffer_1[i] = grid_rs[i];
  }

  if (grid_mpi_comm_size(comm) > 1) {
    // Perform the first FFT
    fft_1d_fw(grid_buffer_1, grid_buffer_2, npts_global[2],
              fft_sizes_rs[0] * fft_sizes_rs[1]);

    // Perform transpose
    transpose_xy_to_xz_blocked(grid_buffer_2, grid_buffer_1, npts_global,
                               proc2local_rs, proc2local_ms, comm);

    // Perform the second FFT
    fft_1d_fw(grid_buffer_1, grid_buffer_2, npts_global[1],
              fft_sizes_ms[0] * fft_sizes_ms[2]);

    // Perform second transpose
    transpose_xz_to_yz_ray(grid_buffer_2, grid_buffer_1, npts_global,
                           proc2local_ms, yz_to_process, rays_per_process,
                           ray_to_yz, comm);

    // Perform the third FFT
    fft_1d_fw(grid_buffer_1, grid_gs, npts_global[0], number_of_local_yz_rays);
  } else {
    fft_3d_fw(grid_buffer_1, grid_buffer_2, npts_global);
    // Copy to the new format
    // Maybe, a 2D FFT, redistribution to rays and final FFT is faster
    int ray_index = 0;
    for (int index_y = 0; index_y < npts_global[1]; index_y++) {
      for (int index_z = 0; index_z < npts_global[2]; index_z++) {
        if (yz_to_process[index_y * npts_global[2] + index_z] == 0) {
          memcpy(&grid_gs[ray_index * npts_global[0]],
                 &grid_buffer_2[index_y * npts_global[0] * npts_global[2] +
                                index_z * npts_global[0]],
                 npts_global[0] * sizeof(double complex));
          ray_index++;
        }
      }
    }
  }

  free(grid_buffer_1);
  free(grid_buffer_2);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_ray(double complex *grid_gs, double *grid_rs,
                   const int npts_global[3], const int (*proc2local_rs)[3][2],
                   const int (*proc2local_ms)[3][2], const int *yz_to_process,
                   const int *rays_per_process, const int (*ray_to_yz)[2],
                   const grid_mpi_comm comm) {
  const int my_process = grid_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  const int number_of_elements_rs = product3(fft_sizes_rs);
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  const int number_of_elements_ms = product3(fft_sizes_ms);
  int number_of_local_yz_rays = 0;
  for (int yz_ray = 0; yz_ray < npts_global[1] * npts_global[2]; yz_ray++) {
    if (yz_to_process[yz_ray] == my_process)
      number_of_local_yz_rays++;
  }
  const int number_of_elements_gs = number_of_local_yz_rays * npts_global[0];
  const int size_of_buffer =
      imax(imax(number_of_elements_rs, number_of_elements_ms),
           number_of_elements_gs);
  double complex *grid_buffer_1 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));
  double complex *grid_buffer_2 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));

  if (grid_mpi_comm_size(comm) > 1) {
    // Perform the first FFT
    fft_1d_bw((const double complex *)grid_gs, grid_buffer_1, npts_global[0],
              number_of_local_yz_rays);

    // Perform transpose
    transpose_yz_to_xz_ray(grid_buffer_1, grid_buffer_2, npts_global,
                           yz_to_process, proc2local_ms, rays_per_process,
                           ray_to_yz, comm);

    // Perform the second FFT
    fft_1d_bw((const double complex *)grid_buffer_2, grid_buffer_1,
              npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2]);

    // Perform second transpose
    transpose_xz_to_xy_blocked(grid_buffer_1, grid_buffer_2, npts_global,
                               proc2local_ms, proc2local_rs, comm);

    // Perform the third FFT
    fft_1d_bw((const double complex *)grid_buffer_2, grid_buffer_1,
              npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1]);
  } else {
    // Copy to the new format
    // Maybe, the order 1D FFT, redistribution to blocks and 2D FFT is faster
    memset(grid_buffer_1, 0, product3(npts_global));
    for (int yz_ray = 0; yz_ray < rays_per_process[my_process]; yz_ray++) {
      const int index_y = ray_to_yz[yz_ray][0];
      const int index_z = ray_to_yz[yz_ray][1];

      memcpy(&grid_buffer_1[index_y * npts_global[0] * npts_global[2] +
                            index_z * npts_global[0]],
             &grid_gs[yz_ray * npts_global[0]],
             npts_global[0] * sizeof(double complex));
    }
    fft_3d_bw(grid_buffer_1, grid_buffer_2, npts_global);
  }

  // Copy real array to complex buffer
  for (int i = 0; i < number_of_elements_rs; i++) {
    grid_rs[i] = creal(grid_buffer_1[i]);
  }

  free(grid_buffer_1);
  free(grid_buffer_2);
}

// EOF
