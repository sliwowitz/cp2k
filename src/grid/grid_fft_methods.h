/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_METHODS_H
#define GRID_FFT_METHODS_H

#include "grid_fft_grid.h"

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a high-level FFT grid.
 * \param grid_rs real-valued data in real space, ordered according to
 *fft_grid->proc2local_rs \param grid_gs complex data in reciprocal space,
 *ordered according to fft_grid->index_to_g \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw(const grid_fft_real_rs_grid *grid_rs,
               const grid_fft_complex_gs_grid *grid_gs);

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a high-level FFT grid.
 * \param fft_grid FFT grid object
 * \param grid_gs complex data in reciprocal space, ordered according to
 *fft_grid->index_to_g \param grid_rs real-valued data in real space, ordered
 *according to fft_grid->proc2local_rs \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw(const grid_fft_complex_gs_grid *grid_gs,
               const grid_fft_real_rs_grid *grid_rs);

#endif /* GRID_FFT_METHODS_H */

// EOF
