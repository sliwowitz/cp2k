/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_reorder.h"
#include "common/grid_common.h"
#include "common/grid_mpi.h"

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Performs a transposition of (x,y,z)->(z,x,y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_z_blocked(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]) {
  const int my_process = grid_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  grid_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      npts_global[2]};
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      npts_global[1],
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes[0] == my_sizes_transposed[0]);

  int *send_displacements = calloc(dims[1], sizeof(int));
  int *recv_displacements = calloc(dims[1], sizeof(int));
  int *send_counts = calloc(dims[1], sizeof(int));
  int *recv_counts = calloc(dims[1], sizeof(int));
  double complex *send_buffer =
      calloc(product3(my_sizes), sizeof(double complex));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[1]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    int rank;
    grid_mpi_cart_rank(comm, (const int[2]){proc_coord[0], process}, &rank);
    const int current_send_count = my_sizes[0] * my_sizes[1] *
                                   (proc2local_transposed[rank][2][1] -
                                    proc2local_transposed[rank][2][0] + 1);
    send_counts[process] = current_send_count;
    send_offset += current_send_count;
    const int current_recv_count =
        my_sizes_transposed[0] *
        (proc2local[rank][1][1] - proc2local[rank][1][0] + 1) *
        my_sizes_transposed[2];
    recv_counts[process] = current_recv_count;
    recv_offset += current_recv_count;
// Copy the data to the send buffer
#pragma omp parallel for collapse(2) default(none)                             \
    shared(my_sizes, my_sizes_transposed, proc2local, proc2local_transposed,   \
               send_buffer, grid, send_displacements, process, rank)
    for (int index_y = 0; index_y <= my_sizes[1]; index_y++) {
      for (int index_x = 0; index_x <= my_sizes[0]; index_x++) {
        memcpy(send_buffer + send_displacements[process] +
                   (index_y * my_sizes[0] + index_x) *
                       (proc2local_transposed[rank][2][1] -
                        proc2local_transposed[rank][2][0] + 1),
               grid + (index_y * my_sizes[0] + index_x) * my_sizes[2] +
                   proc2local_transposed[rank][2][0],
               (proc2local_transposed[rank][2][1] -
                proc2local_transposed[rank][2][0] + 1) *
                   sizeof(double complex));
      }
    }
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == product3(my_sizes_transposed));

  // Use collective MPI communication
  grid_mpi_alltoallv_double_complex(send_buffer, send_counts,
                                    send_displacements, transposed, recv_counts,
                                    recv_displacements, sub_comm[1]);

  free(send_buffer);
  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
}

/*******************************************************************************
 * \brief Performs a transposition of (x,y,z)->(z,x,y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_z_and_distribute_y_blocked(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]) {
  const int my_process = grid_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  grid_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      npts_global[1],
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      npts_global[2]};
  assert(my_sizes[0] == my_sizes_transposed[0]);

  int *send_displacements = calloc(dims[1], sizeof(int));
  int *recv_displacements = calloc(dims[1], sizeof(int));
  int *send_counts = calloc(dims[1], sizeof(int));
  int *recv_counts = calloc(dims[1], sizeof(int));
  double complex *recv_buffer =
      calloc(product3(my_sizes_transposed), sizeof(double complex));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[1]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    int rank;
    grid_mpi_cart_rank(comm, (const int[2]){proc_coord[0], process}, &rank);
    const int current_send_count =
        my_sizes[0] * (proc2local[rank][1][1] - proc2local[rank][1][0] + 1) *
        my_sizes[2];
    send_counts[process] = current_send_count;
    send_offset += current_send_count;
    const int current_recv_count = my_sizes_transposed[0] *
                                   my_sizes_transposed[1] *
                                   (proc2local_transposed[rank][2][1] -
                                    proc2local_transposed[rank][2][0] + 1);
    recv_counts[process] = current_recv_count;
    recv_offset += current_recv_count;
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == product3(my_sizes_transposed));

  // Use collective MPI communication
  grid_mpi_alltoallv_double_complex(grid, send_counts, send_displacements,
                                    recv_buffer, recv_counts,
                                    recv_displacements, sub_comm[1]);

  for (int process = 0; process < dims[1]; process++) {
    int rank;
    grid_mpi_cart_rank(comm, (const int[2]){proc_coord[0], process}, &rank);
// Copy the data to the output array
#pragma omp parallel for collapse(2) default(none)                             \
    shared(my_sizes, my_sizes_transposed, proc2local, proc2local_transposed,   \
               recv_buffer, transposed, recv_displacements, process, rank)
    for (int index_y = 0; index_y <= my_sizes[1]; index_y++) {
      for (int index_x = 0; index_x <= my_sizes[0]; index_x++) {
        memcpy(transposed +
                   (index_y * my_sizes_transposed[0] + index_x) *
                       my_sizes_transposed[2] +
                   proc2local[rank][2][0],
               recv_buffer + recv_displacements[process] +
                   (index_y * my_sizes[0] + index_x) *
                       (proc2local[rank][2][1] - proc2local[rank][2][0] + 1),
               (proc2local[rank][2][1] - proc2local[rank][2][0] + 1) *
                   sizeof(double complex));
      }
    }
  }

  free(recv_buffer);
  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
}

/*******************************************************************************
 * \brief Performs a transposition of (z,x,y)->(y,z,x).
 * \author Frederick Stein
 ******************************************************************************/
void collect_x_and_distribute_y_blocked(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]) {
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);
  (void)npts_global;
  (void)sub_comm;

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
                   (index_z - proc2local_transposed[my_process][2][0]) *
                       my_transposed_sizes[1] +
                   (index_y - proc2local_transposed[my_process][1][0])] =
            grid[(index_x - proc2local[my_process][0][0]) *
                     my_original_sizes[1] * my_original_sizes[2] +
                 (index_z - proc2local[my_process][2][0]) *
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
                     (index_z - proc2local_transposed[my_process][2][0]) *
                         my_transposed_sizes[1] +
                     (index_y - proc2local_transposed[my_process][1][0])] =
              recv_buffer[(index_x - proc2local[recv_process][0][0]) *
                              recv_sizes[1] * recv_sizes[2] +
                          (index_z - proc2local[recv_process][2][0]) *
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
void collect_y_and_distribute_x_blocked(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]) {
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);
  (void)npts_global;
  (void)sub_comm;

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
                   (index_z - proc2local_transposed[my_process][2][0]) *
                       my_transposed_sizes[1] +
                   (index_y - proc2local_transposed[my_process][1][0])] =
            grid[(index_x - proc2local[my_process][0][0]) *
                     my_original_sizes[1] * my_original_sizes[2] +
                 (index_z - proc2local[my_process][2][0]) *
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
                     (index_z - proc2local_transposed[my_process][2][0]) *
                         my_transposed_sizes[1] +
                     (index_y - proc2local_transposed[my_process][1][0])] =
              recv_buffer[(index_x - proc2local[recv_process][0][0]) *
                              recv_sizes[1] * recv_sizes[2] +
                          (index_z - proc2local[recv_process][2][0]) *
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
void collect_x_and_distribute_y_ray(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int *yz_to_process, const int *number_of_rays,
    const int (*ray_to_yz)[2], const grid_mpi_comm comm) {
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

    // Copy the data
    for (int index_x = proc2local[my_process][0][0];
         index_x <= proc2local[my_process][0][1]; index_x++) {
      transposed[index_x * number_of_rays[my_process] + yz_ray] =
          grid[(index_x - proc2local[my_process][0][0]) * my_original_sizes[1] *
                   my_original_sizes[2] +
               (index_z - proc2local[my_process][2][0]) * my_original_sizes[1] +
               (index_y - proc2local[my_process][1][0])];
    }
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
        transposed[index_x * number_of_rays[my_process] + yz_ray] =
            recv_buffer[(index_x - proc2local[recv_process][0][0]) *
                            recv_sizes[1] * recv_sizes[2] +
                        (index_z - proc2local[recv_process][2][0]) *
                            recv_sizes[1] +
                        (index_y - proc2local[recv_process][1][0])];
      }
      number_of_received_rays++;
    }

    // Wait for the send request
    grid_mpi_wait(&send_request);
  }

  // assert(number_of_received_rays == number_of_rays[my_process]);

  free(recv_buffer);
}

/*******************************************************************************
 * \brief Performs a transposition of (z,x,y)->(y,z,x).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_x_ray(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int *yz_to_process,
    const int (*proc2local_transposed)[3][2], const int *number_of_rays,
    const int (*ray_to_yz)[2], const grid_mpi_comm comm) {
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

    // Copy the data
    for (int index_x = proc2local_transposed[my_process][0][0];
         index_x <= proc2local_transposed[my_process][0][1]; index_x++) {
      transposed[(index_x - proc2local_transposed[my_process][0][0]) *
                     my_transposed_sizes[1] * my_transposed_sizes[2] +
                 (index_z - proc2local_transposed[my_process][2][0]) *
                     my_transposed_sizes[1] +
                 (index_y - proc2local_transposed[my_process][1][0])] =
          grid[index_x * number_of_rays[my_process] + yz_ray];
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
        transposed[(index_x - proc2local_transposed[my_process][0][0]) *
                       my_transposed_sizes[1] * my_transposed_sizes[2] +
                   (index_z - proc2local_transposed[my_process][2][0]) *
                       my_transposed_sizes[1] +
                   (index_y - proc2local_transposed[my_process][1][0])] =
            recv_buffer[index_x * number_of_rays[recv_process] + yz_ray];
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

  // assert(number_of_received_rays == total_number_of_rays);

  free(recv_buffer);
}

// EOF
