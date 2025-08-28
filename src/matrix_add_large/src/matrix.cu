#include "matrix.cuh"

void kernel_matrix_add_g1d_b1d(const float* a, const float* b, float* c, int row_size, int col_size) {
  dim3 blockDim(32);
  dim3 gridDim(ceil((float)col_size / blockDim.x));
  matrix_add_g1d_b1d<<<gridDim, blockDim>>>(a, b, c, row_size, col_size);
}

void kernel_matrix_add_g2d_b1d(const float* a, const float* b, float* c, int row_size, int col_size) {
  dim3 blockDim(32);
  dim3 gridDim(ceil((float)col_size / blockDim.x), row_size);
  matrix_add_g2d_b1d<<<gridDim, blockDim>>>(a, b, c, row_size, col_size);
}

void kernel_matrix_add_g2d_b2d(const float* a, const float* b, float* c, int row_size, int col_size) {
  dim3 blockDim(32, 32);
  dim3 gridDim(ceil((float)col_size / blockDim.x), ceil((float)row_size / blockDim.y));
  matrix_add_g2d_b2d<<<gridDim, blockDim>>>(a, b, c, row_size, col_size);
}

__global__ void matrix_add_g1d_b1d(const float* a, const float* b, float* c, int row_size, int col_size) {
  unsigned int col = threadIdx.x + (blockIdx.x * blockDim.x);  // col
  if (col < col_size) {
    for (int row = 0; row < row_size; row++) {
      int index = (row * col_size) + col;
      c[index] = a[index] + b[index];
    }
  }
}

__global__ void matrix_add_g2d_b1d(const float* a, const float* b, float* c, int row_size, int col_size) {
  unsigned int col = threadIdx.x + (blockIdx.x * blockDim.x);  // col
  unsigned int row = blockIdx.y;                               // row
  unsigned int index = (row * col_size) + col;

  if (col < col_size && row < row_size) c[index] = a[index] + b[index];
}

__global__ void matrix_add_g2d_b2d(const float* a, const float* b, float* c, int row_size, int col_size) {
  unsigned int col = threadIdx.x + (blockIdx.x * blockDim.x);
  unsigned int row = threadIdx.y + (blockIdx.y * blockDim.y);
  unsigned int index = (row * col_size) + col;

  if (col < col_size && row < row_size) c[index] = a[index] + b[index];
}