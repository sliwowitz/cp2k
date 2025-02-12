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
void fft_1d_fw(double complex *grid_rs, double complex *grid_gs,
               const int fft_size, const int number_of_ffts);

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_blocked(double *grid_rs, double complex *grid_gs,
                       const int npts_global[3], int (*proc2local_rs)[3][2],
                       int (*proc2local_ms)[3][2], int (*proc2local_gs)[3][2],
                       const grid_mpi_comm comm);

#endif /* GRID_FFT_H */

// EOF
