/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_test.h"

#include "common/grid_mpi.h"
#include "grid_fft.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Function to test the FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_1() {
  const grid_mpi_comm comm = grid_mpi_comm_world;

  // Use an asymmetric cell to check correctness of indices
  const int npts_global[3] = {16, 18, 20};

  int max_size = npts_global[0];
  for (int dir = 1; dir < 3; dir++)
    max_size = imax(max_size, npts_global[dir]);

  double complex *input_array =
      malloc(max_size * max_size * sizeof(double complex));
  double complex *output_array =
      malloc(max_size * max_size * sizeof(double complex));

  double error = 0.0;
  for (int dir = 0; dir < 3; dir++) {
    const int current_size = npts_global[dir];
    memset(input_array, 0, current_size * sizeof(double complex));

    for (int number_of_fft = 0; number_of_fft < current_size; number_of_fft++) {
      input_array[number_of_fft * current_size + number_of_fft] = 1.0;
    }

    fft_1d_fw(input_array, output_array, current_size, current_size);

    const double pi = acos(-1);

    for (int number_of_fft = 0; number_of_fft < current_size; number_of_fft++) {
      for (int index = 0; index < current_size; index++) {
        error = max(
            error,
            cabs(output_array[number_of_fft * current_size + number_of_fft] -
                 cexp(-2.0 * I * pi * number_of_fft * index / current_size)));
      }
    }
  }

  if (error > 1e-12) {
    printf("\nThe low-level FFTs do not work properly!\n");
    return 1;
  } else {
    return 0;
  }
}

// EOF
