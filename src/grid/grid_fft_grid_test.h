/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_GRID_TEST_H
#define GRID_FFT_GRID_TEST_H

/*******************************************************************************
 * \brief Function to test the 3D FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d();

/*******************************************************************************
 * \brief Function to test the addition/copy between different grids.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_add_copy();

#endif /* GRID_FFT_GRID_TEST_H */

// EOF
