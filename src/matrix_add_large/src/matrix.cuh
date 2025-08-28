#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

void kernel_matrix_add_g1d_b1d(const float* a, const float* b, float* c, int row_size, int col_size);
void kernel_matrix_add_g2d_b1d(const float* a, const float* b, float* c, int row_size, int col_size);
void kernel_matrix_add_g2d_b2d(const float* a, const float* b, float* c, int row_size, int col_size);

__global__ void matrix_add_g1d_b1d(const float* a, const float* b, float* c, int row_size, int col_size);
__global__ void matrix_add_g2d_b1d(const float* a, const float* b, float* c, int row_size, int col_size);
__global__ void matrix_add_g2d_b2d(const float* a, const float* b, float* c, int row_size, int col_size);

#endif  // MATRIX_CUH