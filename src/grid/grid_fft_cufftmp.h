#pragma once

#include "common/grid_mpi.h"
#include <complex.h>
#include <stdbool.h>

void fft_3d_fw_cufftmp(
    double *grid_rs,
    double complex *grid_gs,
    const int npts_global[3],
    const int (*proc2local_rs)[3][2],
    const int (*proc2local_gs)[3][2],
    const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]
    );