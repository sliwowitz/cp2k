/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_lib.h"

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(__FFTW3)
#include <fftw3.h>
#endif

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_lib() {
#if defined(__FFTW3)
  fftw_init_threads();
#endif
}

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_finalize_lib() {
#if defined(__FFTW3)
  fftw_cleanup();
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_double(const int length, double **buffer) {
  assert(buffer != NULL);
  assert(*buffer == NULL);
#if defined(__FFTW3)
  *buffer = fftw_alloc_real(length);
#else
  *buffer = malloc(length * sizeof(double));
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_complex(const int length, double complex **buffer) {
  assert(buffer != NULL);
  assert(*buffer == NULL);
#if defined(__FFTW3)
  *buffer = fftw_alloc_complex(length);
#else
  *buffer = malloc(length * sizeof(double complex));
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_double(double *buffer) {
#if defined(__FFTW3)
  fftw_free(buffer);
#else
  free(buffer);
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_complex(double complex *buffer) {
#if defined(__FFTW3)
  fftw_free(buffer);
#else
  free(buffer);
#endif
}

// EOF
