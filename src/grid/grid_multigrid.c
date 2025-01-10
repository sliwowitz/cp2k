/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_multigrid.h"
#include "common/grid_common.h"
#include "common/grid_library.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool grid_get_multigrid_orthorhombic(const grid_multigrid *multigrid) {
  assert(multigrid != NULL);
  return multigrid->orthorhombic;
}

int grid_get_multigrid_nlevels(const grid_multigrid *multigrid) {
  assert(multigrid != NULL);
  return multigrid->nlevels;
}

void grid_get_multigrid_npts_global(const grid_multigrid *multigrid,
                                    int *nlevels, int **npts_global) {
  assert(multigrid != NULL);
  *nlevels = multigrid->nlevels;
  *npts_global = (int *)multigrid->npts_global;
}

void grid_get_multigrid_npts_local(const grid_multigrid *multigrid,
                                   int *nlevels, int **npts_local) {
  assert(multigrid != NULL);
  *nlevels = multigrid->nlevels;
  *npts_local = (int *)multigrid->npts_local;
}

void grid_get_multigrid_shift_local(const grid_multigrid *multigrid,
                                    int *nlevels, int **shift_local) {
  assert(multigrid != NULL);
  *nlevels = multigrid->nlevels;
  *shift_local = (int *)multigrid->shift_local;
}

void grid_get_multigrid_border_width(const grid_multigrid *multigrid,
                                     int *nlevels, int **border_width) {
  assert(multigrid != NULL);
  *nlevels = multigrid->nlevels;
  *border_width = (int *)multigrid->border_width;
}

void grid_get_multigrid_dh(const grid_multigrid *multigrid, int *nlevels,
                           double **dh) {
  assert(multigrid != NULL);
  *nlevels = multigrid->nlevels;
  *dh = (double *)multigrid->dh;
}

void grid_get_multigrid_dh_inv(const grid_multigrid *multigrid, int *nlevels,
                               double **dh_inv) {
  assert(multigrid != NULL);
  *nlevels = multigrid->nlevels;
  *dh_inv = (double *)multigrid->dh_inv;
}

grid_mpi_fint grid_get_multigrid_fortran_comm(const grid_multigrid *multigrid) {
  assert(multigrid != NULL);
  return grid_mpi_comm_c2f(multigrid->comm);
}

grid_mpi_comm grid_get_multigrid_comm(const grid_multigrid *multigrid) {
  assert(multigrid != NULL);
  return multigrid->comm;
}

void grid_copy_to_multigrid(const grid_multigrid *multigrid,
                            const offload_buffer **grids) {
  for (int level = 0; level < multigrid->nlevels; level++) {
    memcpy(offload_get_buffer_host_pointer(multigrid->grids[level]),
           offload_get_buffer_host_pointer((offload_buffer *)grids[level]),
           sizeof(double) * multigrid->npts_local[level][0] *
               multigrid->npts_local[level][1] *
               multigrid->npts_local[level][2]);
  }
}

void grid_copy_from_multigrid(const grid_multigrid *multigrid,
                              offload_buffer **grids) {
  for (int level = 0; level < multigrid->nlevels; level++) {
    memcpy(offload_get_buffer_host_pointer(grids[level]),
           offload_get_buffer_host_pointer(multigrid->grids[level]),
           sizeof(double) * multigrid->npts_local[level][0] *
               multigrid->npts_local[level][1] *
               multigrid->npts_local[level][2]);
  }
}

void grid_copy_to_multigrid_single(const grid_multigrid *multigrid,
                                   const double *grid, const int level) {
  memcpy(offload_get_buffer_host_pointer(multigrid->grids[level]), grid,
         sizeof(double) * multigrid->npts_local[level][0] *
             multigrid->npts_local[level][1] * multigrid->npts_local[level][2]);
}

void grid_copy_from_multigrid_single(const grid_multigrid *multigrid,
                                     double *grid, const int level) {
  memcpy(grid, offload_get_buffer_host_pointer(multigrid->grids[level]),
         sizeof(double) * multigrid->npts_local[level][0] *
             multigrid->npts_local[level][1] * multigrid->npts_local[level][2]);
}

void grid_copy_to_multigrid_single_f(const grid_multigrid *multigrid,
                                     const double *grid, const int level) {
  grid_copy_to_multigrid_single(multigrid, grid, level - 1);
}

void grid_copy_from_multigrid_single_f(const grid_multigrid *multigrid,
                                       double *grid, const int level) {
  grid_copy_from_multigrid_single(multigrid, grid, level - 1);
}

void grid_copy_to_multigrid_serial(double *grid_rs, const double *grid_pw,
                                   const int npts_rs[3],
                                   const int border_width[3]) {
  if (border_width[0] == 0 && border_width[1] == 0 && border_width[2] == 0) {
    memcpy(grid_rs, grid_pw,
           npts_rs[0] * npts_rs[1] * npts_rs[2] * sizeof(double));
  } else {
    int npts_pw[3];
    for (int dir = 0; dir < 3; dir++)
      npts_pw[dir] = npts_rs[dir] - 2 * border_width[dir];
    (void)npts_pw;
    for (int iz = 0; iz < npts_rs[2]; iz++) {
      //
      int iz_pw = iz - border_width[2];
      if (iz < border_width[2])
        iz_pw += npts_pw[2];
      if (iz >= npts_pw[2] + border_width[2])
        iz_pw -= npts_pw[2];
      for (int iy = 0; iy < npts_rs[1]; iy++) {
        //
        int iy_pw = iy - border_width[1];
        if (iy < border_width[1])
          iy_pw += npts_pw[1];
        if (iy >= npts_pw[1] + border_width[1])
          iy_pw -= npts_pw[1];
        for (int ix = 0; ix < npts_rs[0]; ix++) {
          //
          int ix_pw = ix - border_width[0];
          if (ix < border_width[0])
            ix_pw += npts_pw[0];
          if (ix >= npts_pw[0] + border_width[0])
            ix_pw -= npts_pw[0];
          grid_rs[iz * npts_rs[0] * npts_rs[1] + iy * npts_rs[0] + ix] =
              grid_pw[iz_pw * npts_pw[1] * npts_pw[2] + iy_pw * npts_pw[0] +
                      ix_pw];
        }
      }
    }
  }
}

void grid_copy_to_multigrid_replicated(
    double *grid_rs, const double *grid_pw, const int npts_rs[3],
    const int border_width[3], const grid_mpi_comm comm,
    const int proc2local[grid_mpi_comm_size(comm)][3][2]) {

  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);

  memset(grid_rs, 0, npts_rs[0]*npts_rs[1]*npts_rs[2]*sizeof(double));

  // Determine the maximum number of grid points on a single rank
  int maximum_number_of_elements = 0;
  for (int process = 0; process < number_of_processes; process++) {
    const int current_number_of_elements =
        (proc2local[process][0][1] - proc2local[process][0][0] + 1) *
        (proc2local[process][1][1] - proc2local[process][1][0] + 1) *
        (proc2local[process][2][1] - proc2local[process][2][0] + 1);
    maximum_number_of_elements =
        (current_number_of_elements > maximum_number_of_elements
             ? current_number_of_elements
             : maximum_number_of_elements);
    for (int dir = 0; dir < 3; dir++) {
      assert(proc2local[process][dir][0] >= 0);
      assert(proc2local[process][dir][1]-proc2local[process][dir][0]+1>=0);
      }
  }

  // Allocate communication buffers
  double *send_buffer = calloc(maximum_number_of_elements, sizeof(double));
  double *recv_buffer = calloc(maximum_number_of_elements, sizeof(double));

  // Initialize send buffer with local data
  const int my_number_of_elements =
      (proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1) *
      (proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1) *
      (proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1);

  memcpy(send_buffer, grid_pw, my_number_of_elements * sizeof(double));

  // We send and receive from our direct neighbor only
  const int send_process_static = modulo(my_process + 1, number_of_processes);
  const int recv_process_static = modulo(my_process - 1, number_of_processes);

  int send_size[3], recv_size[3];
  grid_mpi_request send_request = grid_mpi_request_null;
  grid_mpi_request recv_request = grid_mpi_request_null;

  // Pass local data to each process
  for (int process_shift = 0; process_shift < number_of_processes;
       process_shift++) {

    // Determine the process whose data we receive
    const int recv_process =
        modulo(my_process - process_shift - 1, number_of_processes);
    for(int dir = 0; dir < 3; dir++) recv_size[dir] = proc2local[recv_process][dir][1] - proc2local[recv_process][dir][0] + 1;

    if (process_shift < number_of_processes) {
      grid_mpi_irecv_double(recv_buffer, product3(recv_size),
                            recv_process_static, process_shift, comm, &recv_request);
    }

    // Determine the process whose data we send
    const int send_process =
        modulo(my_process - process_shift, number_of_processes);
    for(int dir = 0; dir < 3; dir++) send_size[dir] = proc2local[send_process][dir][1] - proc2local[send_process][dir][0] + 1;

    if (process_shift < number_of_processes) {
      if (process_shift > 1)
        grid_mpi_wait(&send_request);
      grid_mpi_isend_double(send_buffer, product3(send_size),
                            send_process_static, process_shift, comm, &send_request);
    }

    // Unpack recv
    grid_mpi_wait(&recv_request);
    for (int iz = 0; iz < recv_size[2]; iz++) {
      for (int iy = 0; iy < recv_size[1]; iy++) {
        for (int ix = 0; ix < recv_size[0]; ix++) {
          grid_rs[(iz + border_width[2] + proc2local[recv_process][2][0]) * npts_rs[0] * npts_rs[1] +
                  (iy + border_width[1] + proc2local[recv_process][1][0]) * npts_rs[0] +
                  (ix + border_width[0] + proc2local[recv_process][0][0])] =
              recv_buffer[iz * recv_size[0] * recv_size[1] + iy * recv_size[0] + ix];
        }
      }
    }

    grid_mpi_wait(&send_request);

    // Swap buffers
    double *temp_pointer = send_buffer;
    send_buffer = recv_buffer;
    recv_buffer = temp_pointer;
  }

  // Deal with bounds
  if (border_width[0] != 0 || border_width[1] != 0 || border_width[2] != 0) {
    int shifts[3];
    for (int dir = 0; dir < 3; dir++) {shifts[dir] = npts_rs[dir]-2*border_width[dir];};
    for (int iz = 0; iz < npts_rs[2]; iz++) {
      int iz_orig = iz;
      if (iz < border_width[2])
        iz_orig += shifts[2];
      if (iz >= npts_rs[2] - border_width[2])
        iz_orig -= shifts[2];
      for (int iy = 0; iy < npts_rs[1]; iy++) {
        int iy_orig = iy;
        if (iy < border_width[1])
          iy_orig += shifts[1];
        if (iy >= npts_rs[1] - border_width[1])
          iy_orig -= shifts[1];
        for (int ix = 0; ix < npts_rs[0]; ix++) {
          int ix_orig = ix;
          if (ix < border_width[0]) ix_orig += shifts[0];
          if (ix >= npts_rs[0] - border_width[0])
            ix_orig -= shifts[0];
          grid_rs[iz * npts_rs[0] * npts_rs[1] + iy * npts_rs[0] + ix] =
              grid_rs[iz_orig * npts_rs[0] * npts_rs[1] +
                      iy_orig * npts_rs[0] + ix_orig];
        }
      }
    }
  }
  free(send_buffer);
  free(recv_buffer);
}

void grid_copy_to_multigrid_distributed(
    double *grid_rs, const double *grid_pw, const grid_mpi_comm comm_pw,
    const grid_mpi_comm comm_rs, const int npts_global[3], 
    const int proc2local_rs[grid_mpi_comm_size(comm_rs)][3][2], 
    const int proc2local_pw[grid_mpi_comm_size(comm_pw)][3][2],
    const int border_width[3]) {
  const int number_of_processes = grid_mpi_comm_size(comm_rs);
  const int my_process_rs = grid_mpi_comm_rank(comm_rs);
  const int my_process_pw = grid_mpi_comm_rank(comm_pw);
  
  assert(grid_rs != NULL);
  assert(grid_pw != NULL);
  assert(!grid_mpi_comm_is_unequal(comm_pw, comm_rs));
  for (int process = 0; process < number_of_processes; process++) {
    for (int dir = 0; dir < 3; dir++) {
      assert(proc2local_rs[process][dir][0] >= -border_width[dir] && "The inner part of the RS grid cannot be lower than zero!");
      assert(proc2local_rs[process][dir][1] < npts_global[dir]+border_width[dir] && "The inner part of the RS grid contains too many points!");
      assert(proc2local_rs[process][dir][1] - proc2local_rs[process][dir][0]+1 >= 0 && "The number of points on the RS grid on one processor cannot be negative!");
      assert(proc2local_pw[process][dir][0] >= 0 && "The PW grid is only allowed to have nonnegative indices!");
      assert(proc2local_pw[process][dir][1] < npts_global[dir] && "The PW grid cannot have points outside of the inner RS grid!");
      assert(proc2local_pw[process][dir][1] - proc2local_pw[process][dir][0]+1 >= 0 && "The number of points on the PW grid on one processor cannot be negative!");
    }
  }
  for (int dir = 0; dir < 3; dir++) {
    assert(border_width[dir] >= 0 && "The number of points on the boundary cannot be negative!");
    assert(npts_global[dir] >= 0 && "Global number of points cannot be negative!");
  }

  (void)npts_global;

  // Prepare the intermediate buffer
  int my_bounds_rs_inner[3][2];
  int my_bounds_rs[3][2];
  int my_sizes_rs_inner[3];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++) {
    my_bounds_rs_inner[dir][0] = proc2local_rs[my_process_rs][dir][0]+border_width[dir];
    my_bounds_rs_inner[dir][1] = proc2local_rs[my_process_rs][dir][1]-border_width[dir];
    my_bounds_rs[dir][0] = proc2local_rs[my_process_rs][dir][0];
    my_bounds_rs[dir][1] = proc2local_rs[my_process_rs][dir][1];
    my_sizes_rs_inner[dir] = proc2local_rs[my_process_rs][dir][1]-proc2local_rs[my_process_rs][dir][0]+1-2*border_width[dir];
    my_sizes_rs[dir] = proc2local_rs[my_process_rs][dir][1]-proc2local_rs[my_process_rs][dir][0]+1;
  }
  const int my_number_of_inner_elements_rs = product3(my_sizes_rs_inner);
  const int my_number_of_elements_rs = product3(my_sizes_rs);
  double * grid_rs_inner = calloc(my_number_of_inner_elements_rs, sizeof(double));

  int received_elements = 0;

  // Step A: Collect the inner local block
  {
    // A1) Count number of elements to be received to determine the buffer sizes
    int max_number_of_elements_pw = 0;
    for (int process = 0; process < number_of_processes; process++) {
      max_number_of_elements_pw = imax(max_number_of_elements_pw, (proc2local_pw[process][0][1]-proc2local_pw[process][0][0]+1)*(proc2local_pw[process][1][1]-proc2local_pw[process][1][0]+1)*(proc2local_pw[process][2][1]-proc2local_pw[process][2][0]+1));
    }
    const int my_number_of_elements_pw = (proc2local_pw[my_process_pw][0][1]-proc2local_pw[my_process_pw][0][0]+1)*(proc2local_pw[my_process_pw][1][1]-proc2local_pw[my_process_pw][1][0]+1)*(proc2local_pw[my_process_pw][2][1]-proc2local_pw[my_process_pw][2][0]+1);

    double * recv_buffer = calloc(max_number_of_elements_pw, sizeof(double));

    // A2) Send around local data of the PW grid and copy it to our local buffer
    for (int process_shift = 0; process_shift < number_of_processes; process_shift++) {
      const int send_process = modulo(my_process_pw+process_shift,number_of_processes);
      const int recv_process = modulo(my_process_pw-process_shift,number_of_processes);

      int recv_size[3];
      for (int dir = 0; dir < 3; dir++) recv_size[dir] = proc2local_pw[recv_process][dir][1]-proc2local_pw[recv_process][dir][0]+1;

      if (process_shift == 0) {
        memcpy(recv_buffer, grid_pw, my_number_of_elements_pw*sizeof(double));
      } else {
        grid_mpi_sendrecv_double(grid_pw, my_number_of_elements_pw, send_process, process_shift, recv_buffer, product3(recv_size), recv_process, process_shift, comm_pw);
      }

      for (int iz = imax(0, proc2local_pw[recv_process][2][0]-my_bounds_rs_inner[2][0]); iz <= imin(my_sizes_rs_inner[2]-1, proc2local_pw[recv_process][2][1]-my_bounds_rs_inner[2][0]); iz++) {
        for (int iy = imax(0, proc2local_pw[recv_process][1][0]-my_bounds_rs_inner[1][0]); iy <= imin(my_sizes_rs_inner[1]-1, proc2local_pw[recv_process][1][1]-my_bounds_rs_inner[1][0]); iy++) {
          for (int ix = imax(0, proc2local_pw[recv_process][0][0]-my_bounds_rs_inner[0][0]); ix <= imin(my_sizes_rs_inner[0]-1, proc2local_pw[recv_process][0][1]-my_bounds_rs_inner[0][0]); ix++) {
            assert(iz*my_sizes_rs_inner[0]*my_sizes_rs_inner[1]+iy*my_sizes_rs_inner[0]+ix < my_number_of_inner_elements_rs && "Too large index for grid_rs_inner");
            assert((iz+my_bounds_rs_inner[2][0]-proc2local_pw[recv_process][2][0])*recv_size[0]*recv_size[1]+(iy+my_bounds_rs_inner[1][0]-proc2local_pw[recv_process][1][0])*recv_size[0]+(ix+my_bounds_rs_inner[0][0]-proc2local_pw[recv_process][0][0]) < max_number_of_elements_pw && "Too large index for recv_buffer");
            grid_rs_inner[iz*my_sizes_rs_inner[0]*my_sizes_rs_inner[1]+iy*my_sizes_rs_inner[0]+ix] += recv_buffer[(iz+my_bounds_rs_inner[2][0]-proc2local_pw[recv_process][2][0])*recv_size[0]*recv_size[1]+(iy+my_bounds_rs_inner[1][0]-proc2local_pw[recv_process][1][0])*recv_size[0]+(ix+my_bounds_rs_inner[0][0]-proc2local_pw[recv_process][0][0])];
            received_elements++;
          }
        }
      }
    }

    // Cleanup
    free(recv_buffer);
  }

  assert(received_elements == my_number_of_inner_elements_rs && "Not elements of the inner part of the RS grid were received");

  // B) Distribute inner local block the everyone
  {
    int max_number_of_elements_rs = 0;
    for (int process = 0; process < number_of_processes; process++) {
      max_number_of_elements_rs = imax(max_number_of_elements_rs, (proc2local_rs[process][0][1]-proc2local_rs[process][0][0]+1)*(proc2local_rs[process][1][1]-proc2local_rs[process][1][0]+1)*(proc2local_rs[process][2][1]-proc2local_rs[process][2][0]+1));
    }

    double * recv_buffer = calloc(max_number_of_elements_rs, sizeof(double));

    received_elements = 0;

    // A2) Send around local data of the PW grid and copy it to our local buffer
    for (int process_shift = 0; process_shift < number_of_processes; process_shift++) {
      const int send_process = modulo(my_process_rs+process_shift,number_of_processes);
      const int recv_process = modulo(my_process_rs-process_shift,number_of_processes);

      int recv_size[3];
      for (int dir = 0; dir < 3; dir++) recv_size[dir] = proc2local_rs[recv_process][dir][1]-proc2local_rs[recv_process][dir][0]+1-2*border_width[dir];

      if (process_shift == 0) {
        memcpy(recv_buffer, grid_rs_inner, my_number_of_inner_elements_rs*sizeof(double));
      } else {
        grid_mpi_sendrecv_double(grid_rs_inner, my_number_of_inner_elements_rs, send_process, process_shift, recv_buffer, product3(recv_size), recv_process, process_shift, comm_rs);
      }

      // Do not forget the boundary outside of the main bound
      for (int iz = 0; iz < my_sizes_rs[2]; iz++) {
        const int iz_orig = modulo(iz+my_bounds_rs[2][0], npts_global[2])-proc2local_rs[recv_process][2][0]-border_width[2];
        if (iz_orig < 0 || iz_orig >= recv_size[2]) continue;
        for (int iy = 0; iy < my_sizes_rs[1]; iy++) {
          const int iy_orig = modulo(iy+my_bounds_rs[1][0], npts_global[1])-proc2local_rs[recv_process][1][0]-border_width[1];
          if (iy_orig < 0 || iy_orig >= recv_size[1]) continue;
          for (int ix = 0; ix < my_sizes_rs[0]; ix++) {
            const int ix_orig = modulo(ix+my_bounds_rs[0][0], npts_global[0])-proc2local_rs[recv_process][0][0]-border_width[0];
            if (ix_orig < 0 || ix_orig >= recv_size[0]) continue;
            grid_rs[iz*my_sizes_rs[0]*my_sizes_rs[1]+iy*my_sizes_rs[0]+ix] = recv_buffer[iz_orig*recv_size[0]*recv_size[1]+iy_orig*recv_size[0]+ix_orig];
            received_elements++;
          }
        }
      }
    }
    free(recv_buffer);

    assert(received_elements == my_number_of_elements_rs && "Not all elements of the RS grid and its boundary were received!");
  }

  free(grid_rs_inner);
}

void grid_copy_to_multigrid_general(
    const grid_multigrid *multigrid, const double *grids[multigrid->nlevels],
    const grid_mpi_comm comm[multigrid->nlevels], const int *proc2local) {
  (void)grids;
  (void)proc2local;
  for (int level = 0; level < multigrid->nlevels; level++) {
    assert(!grid_mpi_comm_is_unequal(multigrid->comm, comm[level]));
    if (grid_mpi_comm_size(comm[level]) == 1) {
      grid_copy_to_multigrid_serial(multigrid->grids[level]->host_buffer,
                                    grids[level], multigrid->npts_local[level],
                                    multigrid->border_width[level]);
    } else {
      // The parallel case, we need to distinguish replicated grids from
      // distributed grids
      if (multigrid->pgrid_dims[level][0] * multigrid->pgrid_dims[level][1] *
              multigrid->pgrid_dims[level][2] ==
          1) {
        grid_copy_to_multigrid_replicated(
            multigrid->grids[level]->host_buffer, grids[level],
            multigrid->npts_local[level], multigrid->border_width[level],
            comm[level],
            (const int(*)[3][2]) &
                proc2local[level * grid_mpi_comm_size(comm[level]) * 6]);
      } else {
        grid_copy_to_multigrid_distributed(
          multigrid->grids[level]->host_buffer, grids[level],
        multigrid->comm, comm[level], multigrid->npts_global[level],
        (const int (*)[3][2])&multigrid->proc2local[6*level*grid_mpi_comm_size(multigrid->comm)],
        (const int (*)[3][2])&proc2local[6*level*grid_mpi_comm_size(comm[0])], multigrid->border_width[level]);
      }
    }
  }
}

void grid_copy_to_multigrid_general_f(
    const grid_multigrid *multigrid, const double *grids[multigrid->nlevels],
    const grid_mpi_fint fortran_comm[multigrid->nlevels],
    const int *proc2local) {
  grid_mpi_comm comm[multigrid->nlevels];
  for (int level = 0; level < multigrid->nlevels; level++) {
    comm[level] = grid_mpi_comm_f2c(fortran_comm[level]);
  }

  grid_copy_to_multigrid_general(multigrid, grids, comm, proc2local);
}

void grid_copy_to_multigrid_general_single(const grid_multigrid *multigrid,
                                           const int level, const double *grid,
                                           const grid_mpi_comm comm,
                                           const int *proc2local) {
  assert(multigrid != NULL);
  assert(!grid_mpi_comm_is_unequal(multigrid->comm, comm));
  assert(grid != NULL);
  if (grid_mpi_comm_size(comm) == 1) {
    grid_copy_to_multigrid_serial(multigrid->grids[level]->host_buffer, grid,
                                  multigrid->npts_local[level],
                                  multigrid->border_width[level]);
  } else {
    // The parallel case, we need to distinguish replicated grids from
    // distributed grids
    if (multigrid->pgrid_dims[level][0] * multigrid->pgrid_dims[level][1] *
            multigrid->pgrid_dims[level][2] ==
        1) {
      grid_copy_to_multigrid_replicated(multigrid->grids[level]->host_buffer,
                                        grid, multigrid->npts_local[level],
                                        multigrid->border_width[level], comm,
                                        (const int(*)[3][2])proc2local);
    } else {
        grid_copy_to_multigrid_distributed(
          multigrid->grids[level]->host_buffer, grid,
        multigrid->comm, comm, multigrid->npts_global[level],
        (const int (*)[3][2])&multigrid->proc2local[6*level*grid_mpi_comm_size(multigrid->comm)],
        (const int (*)[3][2])proc2local, multigrid->border_width[level]);
    }
  }
}

void grid_copy_to_multigrid_general_single_f(const grid_multigrid *multigrid,
                                             const int level,
                                             const double *grid,
                                             const grid_mpi_fint fortran_comm,
                                             const int *proc2local) {
  grid_copy_to_multigrid_general_single(
      multigrid, level - 1, grid, grid_mpi_comm_f2c(fortran_comm), proc2local);
}

void grid_copy_from_multigrid_serial(const double *grid_rs, double *grid_pw,
                                     const int npts_rs[3],
                                     const int border_width[3]) {
  if (border_width[0] == 0 && border_width[1] == 0 && border_width[2] == 0) {
    memcpy(grid_pw, grid_rs,
           npts_rs[0] * npts_rs[1] * npts_rs[2] * sizeof(double));
  } else {
    int npts_pw[3];
    for (int dir = 0; dir < 3; dir++)
      npts_pw[dir] = npts_rs[dir] - 2 * border_width[dir];
    (void)npts_pw;
    for (int iz = border_width[2]; iz < npts_rs[2] - border_width[2]; iz++) {
      //
      int iz_pw = iz - border_width[2];
      for (int iy = border_width[1]; iy < npts_rs[1] - border_width[1]; iy++) {
        //
        int iy_pw = iy - border_width[1];
        for (int ix = border_width[0]; ix < npts_rs[0] - border_width[0];
             ix++) {
          //
          int ix_pw = ix - border_width[0];
          grid_pw[iz_pw * npts_pw[1] * npts_pw[2] + iy_pw * npts_pw[0] +
                  ix_pw] =
              grid_rs[iz * npts_rs[0] * npts_rs[1] + iy * npts_rs[0] + ix];
        }
      }
    }
  }
}

void grid_copy_from_multigrid_replicated(
    const double *grid_rs, double *grid_pw, const int npts_rs[3],
    const int border_width[3], const grid_mpi_comm comm,
    const int proc2local[grid_mpi_comm_size(comm)][3][2]) {

  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);

  // Determine the maximum number of grid points on a single rank
  int maximum_number_of_elements = 0;
  for (int process = 0; process < number_of_processes; process++) {
    const int current_number_of_elements =
        (proc2local[process][0][1] - proc2local[process][0][0] + 1) *
        (proc2local[process][1][1] - proc2local[process][1][0] + 1) *
        (proc2local[process][2][1] - proc2local[process][2][0] + 1);
    maximum_number_of_elements = imax(current_number_of_elements, maximum_number_of_elements);
  }
  const int my_number_of_elements =
      (proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1) *
      (proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1) *
      (proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1);

  // Allocate communication buffers
  double *send_buffer = calloc(maximum_number_of_elements, sizeof(double));
  double *recv_buffer = calloc(maximum_number_of_elements, sizeof(double));

  // We actually send to the next neighbor
  const int process_to_send_to = modulo(my_process - 1, number_of_processes);
  const int process_to_recv_from = modulo(my_process + 1, number_of_processes);

  grid_mpi_request recv_request = grid_mpi_request_null;
  grid_mpi_request send_request = grid_mpi_request_null;

  int send_size[3], recv_size[3];

  // Send the data of the ip-th neighbor
  for (int process_shift = 1; process_shift <= number_of_processes;
       process_shift++) {
    // Load the send buffer for the next process to which the will be finally sent (not necessarily the process to which we actually send)
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    for(int dir = 0; dir < 3; dir++) send_size[dir] = proc2local[send_process][dir][1] - proc2local[send_process][dir][0] + 1;

    // Wait until the sendbuffer (former recvbuffer) was sent (not required in the very first iteration)
    if (process_shift > 1)
      grid_mpi_wait(&recv_request);

    // Pack send_buffer
    for (int iz = 0; iz < send_size[2]; iz++) {
      for (int iy = 0; iy < send_size[1]; iy++) {
        for (int ix = 0; ix < send_size[0]; ix++) {
          send_buffer[iz * send_size[0] * send_size[1] + iy * send_size[0] + ix] +=
              grid_rs[(iz + proc2local[send_process][2][0] + border_width[2]) * npts_rs[0] * npts_rs[1] +
                      (iy + proc2local[send_process][1][0] + border_width[1]) * npts_rs[0] +
                      (ix + proc2local[send_process][0][0] + border_width[0])];
        }
      }
    }

    if (process_shift == number_of_processes)
      break;

    // Load the recv buffer for the process to which the next data is sent
    const int recv_process =
        modulo(my_process + process_shift + 1, number_of_processes);
    for(int dir = 0; dir < 3; dir++) recv_size[dir] = proc2local[recv_process][dir][1] - proc2local[recv_process][dir][0] + 1;

    // Communicate buffers
    if (process_shift > 1)
      grid_mpi_wait(&send_request);
    grid_mpi_irecv_double(recv_buffer, product3(recv_size),
                          process_to_recv_from, process_shift, comm, &recv_request);
    grid_mpi_isend_double(send_buffer, product3(send_size),
                          process_to_send_to, process_shift, comm, &send_request);
    double *temp_pointer = send_buffer;
    send_buffer = recv_buffer;
    recv_buffer = temp_pointer;
  }

  grid_mpi_wait(&send_request);

  // Copy the final received data for yourself to the result
  memcpy(grid_pw, send_buffer, my_number_of_elements * sizeof(double));

  free(send_buffer);
  free(recv_buffer);
}

void grid_copy_from_multigrid_distributed(
    const double *grid_rs, double *grid_pw, const grid_mpi_comm comm_pw,
    const grid_mpi_comm comm_rs, const int npts_global[3], 
    const int proc2local_rs[grid_mpi_comm_size(comm_rs)][3][2], 
    const int proc2local_pw[grid_mpi_comm_size(comm_pw)][3][2],
    const int border_width[3]) {
  const int number_of_processes = grid_mpi_comm_size(comm_rs);
  const int my_process_rs = grid_mpi_comm_rank(comm_rs);
  const int my_process_pw = grid_mpi_comm_rank(comm_pw);
  
  assert(grid_rs != NULL);
  assert(grid_pw != NULL);
  assert(!grid_mpi_comm_is_unequal(comm_pw, comm_rs));
  for (int process = 0; process < number_of_processes; process++) {
    for (int dir = 0; dir < 3; dir++) {
      assert(proc2local_rs[process][dir][0] >= -border_width[dir] && "The inner part of the RS grid cannot be lower than zero!");
      assert(proc2local_rs[process][dir][1] < npts_global[dir]+border_width[dir] && "The inner part of the RS grid contains too many points!");
      assert(proc2local_rs[process][dir][1] - proc2local_rs[process][dir][0]+1 >= 0 && "The number of points on the RS grid on one processor cannot be negative!");
      assert(proc2local_pw[process][dir][0] >= 0 && "The PW grid is only allowed to have nonnegative indices!");
      assert(proc2local_pw[process][dir][1] < npts_global[dir] && "The PW grid cannot have points outside of the inner RS grid!");
      assert(proc2local_pw[process][dir][1] - proc2local_pw[process][dir][0]+1 >= 0 && "The number of points on the PW grid on one processor cannot be negative!");
    }
  }
  for (int dir = 0; dir < 3; dir++) {
    assert(border_width[dir] >= 0 && "The number of points on the boundary cannot be negative!");
    assert(npts_global[dir] >= 0 && "Global number of points cannot be negative!");
  }

  (void)grid_rs;
  (void)grid_pw;
  (void)comm_rs;
  (void)comm_pw;
  (void)proc2local_rs;
  (void)proc2local_pw;
  (void)npts_global;
  (void)border_width;
  (void)number_of_processes;
  (void)my_process_pw;
  (void)my_process_rs;
}

void grid_copy_from_multigrid_general(
    const grid_multigrid *multigrid, double *grids[multigrid->nlevels],
    const grid_mpi_comm comm[multigrid->nlevels], const int *proc2local) {
  for (int level = 0; level < multigrid->nlevels; level++) {
    assert(!grid_mpi_comm_is_unequal(multigrid->comm, comm[level]));
    if (grid_mpi_comm_size(comm[level]) == 1) {
      grid_copy_from_multigrid_serial(
          multigrid->grids[level]->host_buffer, grids[level],
          multigrid->npts_local[level], multigrid->border_width[level]);
    } else {
      // The parallel case, we need to distinguish replicated grids from
      // distributed grids
      if (multigrid->pgrid_dims[level][0] * multigrid->pgrid_dims[level][1] *
              multigrid->pgrid_dims[level][2] ==
          1) {
        grid_copy_from_multigrid_replicated(
            multigrid->grids[level]->host_buffer, grids[level],
            multigrid->npts_local[level], multigrid->border_width[level],
            comm[level],
            (const int(*)[3][2]) &
                proc2local[level * grid_mpi_comm_size(comm[level]) * 6]);
      } else {
        grid_copy_from_multigrid_distributed(
          multigrid->grids[level]->host_buffer, grids[level],
        multigrid->comm, comm[level], multigrid->npts_global[level],
        (const int (*)[3][2])&multigrid->proc2local[6*level*grid_mpi_comm_size(multigrid->comm)],
        (const int (*)[3][2])&proc2local[6*level*grid_mpi_comm_size(multigrid->comm)], multigrid->border_width[level]);
      }
    }
  }
}

void grid_copy_from_multigrid_general_f(
    const grid_multigrid *multigrid, double *grids[multigrid->nlevels],
    const grid_mpi_fint fortran_comm[multigrid->nlevels],
    const int *proc2local) {
  grid_mpi_comm comm[multigrid->nlevels];
  for (int level = 0; level < multigrid->nlevels; level++)
    comm[level] = grid_mpi_comm_f2c(fortran_comm[level]);
  grid_copy_from_multigrid_general(multigrid, grids, comm, proc2local);
}

void grid_copy_from_multigrid_general_single(const grid_multigrid *multigrid,
                                             const int level, double *grid,
                                             const grid_mpi_comm comm,
                                             const int *proc2local) {
  assert(multigrid != NULL);
  assert(!grid_mpi_comm_is_unequal(multigrid->comm, comm));
  assert(grid != NULL);
  if (grid_mpi_comm_size(comm) == 1) {
    grid_copy_from_multigrid_serial(multigrid->grids[level]->host_buffer, grid,
                                    multigrid->npts_local[level],
                                    multigrid->border_width[level]);
  } else {
    // The parallel case, we need to distinguish replicated grids from
    // distributed grids
    if (multigrid->pgrid_dims[level][0] * multigrid->pgrid_dims[level][1] *
            multigrid->pgrid_dims[level][2] ==
        1) {
      grid_copy_from_multigrid_replicated(multigrid->grids[level]->host_buffer,
                                          grid, multigrid->npts_local[level],
                                          multigrid->border_width[level], comm,
                                          (const int(*)[3][2])proc2local);
    } else {
        grid_copy_from_multigrid_distributed(
          multigrid->grids[level]->host_buffer, grid,
        multigrid->comm, comm, multigrid->npts_global[level],
        (const int (*)[3][2])&multigrid->proc2local[6*level*grid_mpi_comm_size(multigrid->comm)],
        (const int (*)[3][2])proc2local, multigrid->border_width[level]);
    }
  }
}

void grid_copy_from_multigrid_general_single_f(const grid_multigrid *multigrid,
                                               const int level, double *grid,
                                               const grid_mpi_fint fortran_comm,
                                               const int *proc2local) {
  grid_copy_from_multigrid_general_single(
      multigrid, level - 1, grid, grid_mpi_comm_f2c(fortran_comm), proc2local);
}

/*******************************************************************************
 * \brief Allocates a multigrid which is passed to task list-based and
 *pgf_product-based routines.
 *
 * \param orthorhombic     Whether simulation box is orthorhombic.
 * \param nlevels          Number of grid levels.
 * \param npts_global     Number of global grid points in each direction.
 * \param npts_local      Number of local grid points in each direction.
 * \param shift_local     Number of points local grid is shifted wrt global grid
 * \param border_width    Width of halo region in grid points in each direction.
 * \param dh              Incremental grid matrix.
 * \param dh_inv          Inverse incremental grid matrix.
 *
 * \param multigrid        Handle to the created multigrid.
 *
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_multigrid_f(
    const bool orthorhombic, const int nlevels,
    const int npts_global[nlevels][3], const int npts_local[nlevels][3],
    const int shift_local[nlevels][3], const int border_width[nlevels][3],
    const double dh[nlevels][3][3], const double dh_inv[nlevels][3][3],
    const int grid_dims[nlevels][3], const grid_mpi_fint fortran_comm,
    grid_multigrid **multigrid_out) {
  grid_create_multigrid(orthorhombic, nlevels, npts_global, npts_local,
                        shift_local, border_width, dh, dh_inv, grid_dims,
                        grid_mpi_comm_f2c(fortran_comm), multigrid_out);
}

/*******************************************************************************
 * \brief Allocates a multigrid which is passed to task list-based and
 *pgf_product-based routines.
 *
 * \param orthorhombic     Whether simulation box is orthorhombic.
 * \param nlevels          Number of grid levels.
 * \param npts_global     Number of global grid points in each direction.
 * \param npts_local      Number of local grid points in each direction.
 * \param shift_local     Number of points local grid is shifted wrt global grid
 * \param border_width    Width of halo region in grid points in each direction.
 * \param dh              Incremental grid matrix.
 * \param dh_inv          Inverse incremental grid matrix.
 *
 * \param multigrid        Handle to the created multigrid.
 *
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_multigrid(
    const bool orthorhombic, const int nlevels,
    const int npts_global[nlevels][3], const int npts_local[nlevels][3],
    const int shift_local[nlevels][3], const int border_width[nlevels][3],
    const double dh[nlevels][3][3], const double dh_inv[nlevels][3][3],
    const int pgrid_dims[nlevels][3], const grid_mpi_comm comm,
    grid_multigrid **multigrid_out) {

  const grid_library_config config = grid_library_get_config();

  grid_multigrid *multigrid = NULL;

  assert(multigrid_out != NULL);
  for (int level = 0; level < nlevels; level++) {
    assert(pgrid_dims[level][0] * pgrid_dims[level][1] * pgrid_dims[level][2] ==
               grid_mpi_comm_size(comm) ||
           (pgrid_dims[level][0] == 1 && pgrid_dims[level][1] == 1 &&
            pgrid_dims[level][2] == 1));
  }

  const int num_int = 3 * nlevels;
  const int num_double = 9 * nlevels;

  if (*multigrid_out != NULL) {
    multigrid = *multigrid_out;
    if (nlevels != multigrid->nlevels) {
      multigrid->npts_global =
          realloc(multigrid->npts_global, num_int * sizeof(int));
      multigrid->npts_local =
          realloc(multigrid->npts_local, num_int * sizeof(int));
      multigrid->shift_local =
          realloc(multigrid->shift_local, num_int * sizeof(int));
      multigrid->border_width =
          realloc(multigrid->border_width, num_int * sizeof(int));
      multigrid->dh = realloc(multigrid->dh, num_double * sizeof(double));
      multigrid->dh_inv =
          realloc(multigrid->dh_inv, num_double * sizeof(double));
      multigrid->pgrid_dims =
          realloc(multigrid->pgrid_dims, num_int * sizeof(int));

      for (int level = 0; level < multigrid->nlevels; level++) {
        offload_free_buffer(multigrid->grids[level]);
      }
      multigrid->grids =
          realloc(multigrid->grids, nlevels * sizeof(offload_buffer *));
      if (nlevels > multigrid->nlevels) {
        memset(multigrid->grids[multigrid->nlevels], 0,
               (nlevels - multigrid->nlevels) * sizeof(offload_buffer *));
      }
    }
    // Always free the old communicator
    grid_mpi_comm_free(&multigrid->comm);
    multigrid->proc2local =
        realloc(multigrid->proc2local,
                nlevels * grid_mpi_comm_size(comm) * 6 * sizeof(int));
  } else {
    multigrid = calloc(1, sizeof(grid_multigrid));
    multigrid->npts_global = calloc(num_int, sizeof(int));
    multigrid->npts_local = calloc(num_int, sizeof(int));
    multigrid->shift_local = calloc(num_int, sizeof(int));
    multigrid->border_width = calloc(num_int, sizeof(int));
    multigrid->dh = calloc(num_double, sizeof(double));
    multigrid->dh_inv = calloc(num_double, sizeof(double));
    multigrid->grids = calloc(nlevels, sizeof(offload_buffer *));
    multigrid->pgrid_dims = calloc(num_int, sizeof(int));
    multigrid->proc2local =
        calloc(nlevels * grid_mpi_comm_size(comm) * 6, sizeof(int));

    // Resolve AUTO to a concrete backend.
    if (config.backend == GRID_BACKEND_AUTO) {
#if defined(__OFFLOAD_HIP) && !defined(__NO_OFFLOAD_GRID)
      multigrid->backend = GRID_BACKEND_HIP;
#elif defined(__OFFLOAD) && !defined(__NO_OFFLOAD_GRID)
      multigrid->backend = GRID_BACKEND_GPU;
#else
      multigrid->backend = GRID_BACKEND_CPU;
#endif
    } else {
      multigrid->backend = config.backend;
    }
  }

  for (int level = 0; level < nlevels; level++) {
    offload_create_buffer(npts_local[level][0] * npts_local[level][1] *
                              npts_local[level][2],
                          &multigrid->grids[level]);
  }

  multigrid->nlevels = nlevels;
  multigrid->orthorhombic = orthorhombic;
  memcpy(multigrid->npts_global, npts_global, num_int * sizeof(int));
  memcpy(multigrid->npts_local, npts_local, num_int * sizeof(int));
  memcpy(multigrid->shift_local, shift_local, num_int * sizeof(int));
  memcpy(multigrid->border_width, border_width, num_int * sizeof(int));
  memcpy(multigrid->dh, dh, num_double * sizeof(double));
  memcpy(multigrid->dh_inv, dh_inv, num_double * sizeof(double));
  memcpy(multigrid->pgrid_dims, pgrid_dims, num_int * sizeof(int));
  grid_mpi_comm_dup(comm, &multigrid->comm);

  for (int level = 0; level < nlevels; level++) {
    int local_bounds[3][2];
    for (int dir = 0; dir < 3; dir++) {
      local_bounds[dir][0] = shift_local[level][dir];
      local_bounds[dir][1] =
          shift_local[level][dir] + npts_local[level][dir] - 1;
    }
    grid_mpi_allgather_int(
        &local_bounds[0][0], 6,
        &multigrid->proc2local[level * 6 * grid_mpi_comm_size(comm)], comm);
  }

  grid_ref_create_multigrid(orthorhombic, nlevels, npts_global, npts_local,
                            shift_local, border_width, dh, dh_inv, comm,
                            &(multigrid->ref));

  // We need it for collocation/integration of pgf_products
  grid_cpu_create_multigrid(orthorhombic, nlevels, npts_global, npts_local,
                            shift_local, border_width, dh, dh_inv, comm,
                            &(multigrid->cpu));

  switch (multigrid->backend) {
  case GRID_BACKEND_REF:
    break;
  case GRID_BACKEND_CPU:
    break;
  }

  for (int level = 0; level < nlevels; level++) {
    memset(offload_get_buffer_host_pointer(multigrid->grids[level]), 0,
           npts_local[level][0] * npts_local[level][1] * npts_local[level][2] *
               sizeof(double));
  }

  *multigrid_out = multigrid;

  grid_mpi_barrier(multigrid->comm);
}

/*******************************************************************************
 * \brief Deallocates given multigrid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_multigrid(grid_multigrid *multigrid) {
  if (multigrid != NULL) {
    if (multigrid->npts_global != NULL)
      free(multigrid->npts_global);
    if (multigrid->npts_local != NULL)
      free(multigrid->npts_local);
    if (multigrid->shift_local != NULL)
      free(multigrid->shift_local);
    if (multigrid->border_width != NULL)
      free(multigrid->border_width);
    if (multigrid->dh != NULL)
      free(multigrid->dh);
    if (multigrid->dh_inv != NULL)
      free(multigrid->dh_inv);
    if (multigrid->grids != NULL) {
      for (int level = 0; level < multigrid->nlevels; level++) {
        offload_free_buffer(multigrid->grids[level]);
      }
      free(multigrid->grids);
    }
    if (multigrid->pgrid_dims != NULL)
      free(multigrid->pgrid_dims);
    if (multigrid->proc2local != NULL)
      free(multigrid->proc2local);
    grid_mpi_comm_free(&multigrid->comm);
    grid_ref_free_multigrid(multigrid->ref);
    grid_cpu_free_multigrid(multigrid->cpu);
    free(multigrid);
  }
}

// EOF
