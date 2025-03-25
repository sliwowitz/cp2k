/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_lib.h"
#include "grid_fft_lib_fftw.h"
#include "grid_fft_lib_ref.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

grid_fft_lib grid_fft_lib_choice = GRID_FFT_LIB_REF;

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_lib(const grid_fft_lib lib) {
  grid_fft_lib_choice = lib;
  fft_ref_init_lib();
  fft_fftw_init_lib();
}

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_finalize_lib() {
  fft_ref_finalize_lib();
  fft_fftw_finalize_lib();
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_double(const int length, double **buffer) {
  if (grid_fft_lib_choice == GRID_FFT_LIB_REF) {
    fft_ref_allocate_double(length, buffer);
  } else if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
    fft_fftw_allocate_double(length, buffer);
  } else {
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_complex(const int length, double complex **buffer) {
  if (grid_fft_lib_choice == GRID_FFT_LIB_REF) {
    fft_ref_allocate_complex(length, buffer);
  } else if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
    fft_fftw_allocate_complex(length, buffer);
  } else {
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_double(double *buffer) {
  if (grid_fft_lib_choice == GRID_FFT_LIB_REF) {
    fft_ref_free_double(buffer);
  } else if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
    fft_fftw_free_double(buffer);
  } else {
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_complex(double complex *buffer) {
  if (grid_fft_lib_choice == GRID_FFT_LIB_REF) {
    fft_ref_free_complex(buffer);
  } else if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
    fft_fftw_free_complex(buffer);
  } else {
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw_local(const double complex *grid_rs, double complex *grid_gs,
                     const int fft_size, const int number_of_ffts) {
  fft_ref_1d_fw_local(grid_rs, grid_gs, fft_size, number_of_ffts);
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local(const double complex *grid_gs, double complex *grid_rs,
                     const int fft_size, const int number_of_ffts) {
  fft_ref_1d_bw_local(grid_gs, grid_rs, fft_size, number_of_ffts);
}

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void transpose_local(double complex *grid, double complex *grid_transposed,
                     const int number_of_columns_grid,
                     const int number_of_rows_grid) {
  fft_ref_transpose_local(grid, grid_transposed, number_of_columns_grid,
                          number_of_rows_grid);
}

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_local(double complex *grid_rs, double complex *grid_gs,
                     const int size_of_first_fft, const int size_of_second_fft,
                     const int number_of_ffts) {
  fft_ref_2d_fw_local(grid_rs, grid_gs, size_of_first_fft, size_of_second_fft,
                      number_of_ffts);
}

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_local(double complex *grid_gs, double complex *grid_rs,
                     const int size_of_first_fft, const int size_of_second_fft,
                     const int number_of_ffts) {

  fft_ref_2d_bw_local(grid_gs, grid_rs, size_of_first_fft, size_of_second_fft,
                      number_of_ffts);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_local(double complex *grid_rs, double complex *grid_gs,
                     const int fft_size[3]) {

  fft_ref_3d_fw_local(grid_rs, grid_gs, fft_size);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_local(double complex *grid_gs, double complex *grid_rs,
                     const int fft_size[3]) {

  fft_ref_3d_bw_local(grid_gs, grid_rs, fft_size);
}

// EOF
