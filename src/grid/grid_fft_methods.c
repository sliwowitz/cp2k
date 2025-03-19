/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_methods.h"
#include "grid_fft.h"
#include "grid_fft_grid.h"

#include <stdio.h>
#include <string.h>

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a high-level FFT grid.
 * \param grid_rs real-valued data in real space, ordered according to
 *fft_grid->proc2local_rs \param grid_gs complex data in reciprocal space,
 *ordered according to fft_grid->index_to_g \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw(const grid_fft_grid *fft_grid, const double *grid_rs,
               double complex *grid_gs) {
  const int my_process = grid_mpi_comm_rank(fft_grid->comm);
  int local_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_rs[dir] = fft_grid->proc2local_rs[my_process][dir][1] -
                          fft_grid->proc2local_rs[my_process][dir][0] + 1;
  }
  memcpy(fft_grid->grid_rs, grid_rs,
         local_sizes_rs[0] * local_sizes_rs[1] * local_sizes_rs[2] *
             sizeof(double));
  if (fft_grid->ray_distribution) {
    fft_3d_fw_ray(fft_grid->grid_rs, fft_grid->grid_gs, fft_grid->npts_global,
                  fft_grid->proc2local_rs, fft_grid->proc2local_ms,
                  fft_grid->yz_to_process, fft_grid->rays_per_process,
                  fft_grid->ray_to_yz, fft_grid->comm);
    int(*my_ray_to_yz)[2] = fft_grid->ray_to_yz;
    for (int process = 0; process < my_process; process++) {
      my_ray_to_yz += fft_grid->rays_per_process[process];
    }
    for (int index = 0; index < fft_grid->npts_gs_local; index++) {
      int *index_g = fft_grid->index_to_g[index];
      for (int yz_ray = 0; yz_ray < fft_grid->rays_per_process[my_process];
           yz_ray++) {
        if (my_ray_to_yz[yz_ray][0] == index_g[1] &&
            my_ray_to_yz[yz_ray][1] == index_g[2]) {
          grid_gs[index] =
              fft_grid->grid_gs[yz_ray * fft_grid->npts_global[0] + index_g[0]];
          break;
        }
      }
    }
  } else {
    fft_3d_fw_blocked(fft_grid->grid_rs, fft_grid->grid_gs,
                      fft_grid->npts_global, fft_grid->proc2local_rs,
                      fft_grid->proc2local_ms, fft_grid->proc2local_gs,
                      fft_grid->comm);
    int local_sizes_gs[3];
    for (int dir = 0; dir < 3; dir++) {
      local_sizes_gs[dir] = fft_grid->proc2local_gs[my_process][dir][1] -
                            fft_grid->proc2local_gs[my_process][dir][0] + 1;
    }
    for (int index = 0; index < fft_grid->npts_gs_local; index++) {
      int *index_g = fft_grid->index_to_g[index];
      grid_gs[index] =
          fft_grid->grid_gs
              [(index_g[2] - fft_grid->proc2local_gs[my_process][2][0]) *
                   local_sizes_gs[0] * local_sizes_gs[1] +
               (index_g[1] - fft_grid->proc2local_gs[my_process][1][0]) *
                   local_sizes_gs[0] +
               (index_g[0] - fft_grid->proc2local_gs[my_process][0][0])];
    }
  }
  fflush(stdout);
  grid_mpi_barrier(fft_grid->comm);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a high-level FFT grid.
 * \param fft_grid FFT grid object
 * \param grid_gs complex data in reciprocal space, ordered according to
 *fft_grid->index_to_g \param grid_rs real-valued data in real space, ordered
 *according to fft_grid->proc2local_rs \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw(const grid_fft_grid *fft_grid, const double complex *grid_gs,
               double *grid_rs) {
  const int my_process = grid_mpi_comm_rank(fft_grid->comm);
  int local_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_rs[dir] = fft_grid->proc2local_rs[my_process][dir][1] -
                          fft_grid->proc2local_rs[my_process][dir][0] + 1;
  }
  if (fft_grid->ray_distribution) {
    int(*my_ray_to_yz)[2] = fft_grid->ray_to_yz;
    for (int process = 0; process < my_process; process++) {
      my_ray_to_yz += fft_grid->rays_per_process[process];
    }
    for (int index = 0; index < fft_grid->npts_gs_local; index++) {
      int *index_g = fft_grid->index_to_g[index];
      for (int yz_ray = 0; yz_ray < fft_grid->rays_per_process[my_process];
           yz_ray++) {
        if (my_ray_to_yz[yz_ray][0] == index_g[1] &&
            my_ray_to_yz[yz_ray][1] == index_g[2]) {
          fft_grid->grid_gs[yz_ray * fft_grid->npts_global[0] + index_g[0]] =
              grid_gs[index];
          break;
        }
      }
    }
    fft_3d_bw_ray(fft_grid->grid_gs, fft_grid->grid_rs, fft_grid->npts_global,
                  fft_grid->proc2local_rs, fft_grid->proc2local_ms,
                  fft_grid->yz_to_process, fft_grid->rays_per_process,
                  fft_grid->ray_to_yz, fft_grid->comm);
  } else {
    int local_sizes_gs[3];
    for (int dir = 0; dir < 3; dir++) {
      local_sizes_gs[dir] = fft_grid->proc2local_gs[my_process][dir][1] -
                            fft_grid->proc2local_gs[my_process][dir][0] + 1;
    }
    for (int index = 0; index < fft_grid->npts_gs_local; index++) {
      int *index_g = fft_grid->index_to_g[index];
      fft_grid
          ->grid_gs[(index_g[2] - fft_grid->proc2local_gs[my_process][2][0]) *
                        local_sizes_gs[0] * local_sizes_gs[1] +
                    (index_g[1] - fft_grid->proc2local_gs[my_process][1][0]) *
                        local_sizes_gs[0] +
                    (index_g[0] - fft_grid->proc2local_gs[my_process][0][0])] =
          grid_gs[index];
    }
    fft_3d_bw_blocked(fft_grid->grid_gs, fft_grid->grid_rs,
                      fft_grid->npts_global, fft_grid->proc2local_rs,
                      fft_grid->proc2local_ms, fft_grid->proc2local_gs,
                      fft_grid->comm);
  }
  memcpy(grid_rs, fft_grid->grid_rs,
         local_sizes_rs[0] * local_sizes_rs[1] * local_sizes_rs[2] *
             sizeof(double));
}

// EOF
