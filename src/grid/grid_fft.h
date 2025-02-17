/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_H
#define GRID_FFT_H

#include "common/grid_mpi.h"

#include <complex.h>

/*******************************************************************************
 * \brief 1D Forward FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw(const double complex *grid_rs, double complex *grid_gs,
               const int fft_size, const int number_of_ffts);

/*******************************************************************************
 * \brief 1D Backward FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw(const double complex *grid_gs, double complex *grid_rs,
               const int fft_size, const int number_of_ffts);

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_blocked(double *grid_rs, double complex *grid_gs,
                       const int npts_global[3], int (*proc2local_rs)[3][2],
                       int (*proc2local_ms)[3][2], int (*proc2local_gs)[3][2],
                       const grid_mpi_comm comm);

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_blocked(double complex *grid_gs, double *grid_rs,
                       const int npts_global[3], int (*proc2local_rs)[3][2],
                       int (*proc2local_ms)[3][2], int (*proc2local_gs)[3][2],
                       const grid_mpi_comm comm);

void transpose_local(double complex *grid, double complex *grid_transposed,
                     const int number_of_columns_grid,
                     const int number_of_rows_grid);

void transpose_xy_to_xz_blocked(const double complex *grid,
                                double complex *transposed,
                                const int npts_global[3],
                                const int (*proc2local)[3][2],
                                const int (*proc2local_transposed)[3][2],
                                const grid_mpi_comm comm);

void transpose_xz_to_xy_blocked(const double complex *grid,
                                double complex *transposed,
                                const int npts_global[3],
                                const int (*proc2local)[3][2],
                                const int (*proc2local_transposed)[3][2],
                                const grid_mpi_comm comm);

void transpose_xz_to_yz_blocked(const double complex *grid,
                                double complex *transposed,
                                const int npts_global[3],
                                const int (*proc2local)[3][2],
                                const int (*proc2local_transposed)[3][2],
                                const grid_mpi_comm comm);

void transpose_yz_to_xz_blocked(const double complex *grid,
                                double complex *transposed,
                                const int npts_global[3],
                                const int (*proc2local)[3][2],
                                const int (*proc2local_transposed)[3][2],
                                const grid_mpi_comm comm);

#endif /* GRID_FFT_H */

// EOF
