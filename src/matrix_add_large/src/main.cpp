#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "matrix.cuh"

#define ROW_SIZE (8192)
#define COL_SIZE (8192)
#define MATRIX_SIZE (ROW_SIZE * COL_SIZE)

void validation(float *&h_c, float *&h_hc);

void checkCudaError(cudaError_t err, const char *msg);
void allocateDeviceMemory(float **d_ptr, size_t size, const char *name);
void freeDeviceMemory(void *d_ptr, const char *name);
void cleanDeviceMemory(void *d_ptr, size_t size, const char *name);
void copyToDevice(float *d_dst, const float *h_src, size_t size, const char *msg);
void copyToHost(float *h_dst, const float *d_src, size_t size, const char *msg);

int main() {
  auto *h_a = (float *)malloc(sizeof(float) * MATRIX_SIZE);
  auto *h_b = (float *)malloc(sizeof(float) * MATRIX_SIZE);
  auto *h_c_g1d_b1d = (float *)malloc(sizeof(float) * MATRIX_SIZE);
  auto *h_c_g2d_b1d = (float *)malloc(sizeof(float) * MATRIX_SIZE);
  auto *h_c_g2d_b2d = (float *)malloc(sizeof(float) * MATRIX_SIZE);
  auto *h_hc = (float *)malloc(sizeof(float) * MATRIX_SIZE);

  if (h_a == nullptr || h_b == nullptr || h_c_g1d_b1d == nullptr || h_c_g2d_b1d == nullptr || h_c_g2d_b2d == nullptr || h_hc == nullptr) {
    printf("fail to allocate host memory\n");
    exit(EXIT_FAILURE);
  }

  float *d_a;
  float *d_b;
  float *d_c;
  allocateDeviceMemory(&d_a, sizeof(float) * MATRIX_SIZE, "d_a");
  allocateDeviceMemory(&d_b, sizeof(float) * MATRIX_SIZE, "d_b");
  allocateDeviceMemory(&d_c, sizeof(float) * MATRIX_SIZE, "d_c");
  cleanDeviceMemory(d_a, sizeof(float) * MATRIX_SIZE, "d_a");
  cleanDeviceMemory(d_b, sizeof(float) * MATRIX_SIZE, "d_b");
  cleanDeviceMemory(d_c, sizeof(float) * MATRIX_SIZE, "d_c");

  for (int i = 0; i < MATRIX_SIZE; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  auto start = std::chrono::steady_clock::now();
  for (int row = 0; row < ROW_SIZE; row++) {
    for (int col = 0; col < COL_SIZE; col++) {
      size_t i = (row * COL_SIZE) + col;
      h_hc[i] = h_a[i] + h_b[i];
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  printf("CPU matrix add: %f (ms)\n", duration.count() / 1000.0);

  copyToDevice(d_a, h_a, sizeof(float) * MATRIX_SIZE, "failed to copy h_a to d_a");
  copyToDevice(d_b, h_b, sizeof(float) * MATRIX_SIZE, "failed to copy h_b to d_b");

  // Kernels
  start = std::chrono::steady_clock::now();
  kernel_matrix_add_g1d_b1d(d_a, d_b, d_c, ROW_SIZE, COL_SIZE);
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  printf("GPU matrix add (g1d_b1d): %f (ms) : ", duration.count() / 1000.0);
  copyToHost(h_c_g1d_b1d, d_c, sizeof(float) * MATRIX_SIZE, "failed to copy d_c to h_c");
  validation(h_c_g1d_b1d, h_hc);

  start = std::chrono::steady_clock::now();
  kernel_matrix_add_g2d_b1d(d_a, d_b, d_c, ROW_SIZE, COL_SIZE);
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  printf("GPU matrix add (g2d_b1d): %f (ms) : ", duration.count() / 1000.0);
  copyToHost(h_c_g2d_b1d, d_c, sizeof(float) * MATRIX_SIZE, "failed to copy d_c to h_c");
  validation(h_c_g2d_b1d, h_hc);

  start = std::chrono::steady_clock::now();
  kernel_matrix_add_g2d_b2d(d_a, d_b, d_c, ROW_SIZE, COL_SIZE);
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  printf("GPU matrix add (g2d_b2d): %f (ms) : ", duration.count() / 1000.0);
  copyToHost(h_c_g2d_b2d, d_c, sizeof(float) * MATRIX_SIZE, "failed to copy d_c to h_c");
  validation(h_c_g2d_b2d, h_hc);

  freeDeviceMemory(d_a, "d_a");
  freeDeviceMemory(d_b, "d_b");
  freeDeviceMemory(d_c, "d_c");

  free(h_a);
  free(h_b);
  free(h_c_g1d_b1d);
  free(h_c_g2d_b1d);
  free(h_c_g2d_b2d);
  free(h_hc);

  return 0;
}

void validation(float *&h_c, float *&h_hc) {
  bool is_correct = true;
  for (int i = 0; i < MATRIX_SIZE; i++) {
    if (h_hc[i] != h_c[i]) {
      is_correct = false;
      break;
    }
  }

  if (is_correct) {
    printf("ok\n");
  } else {
    printf("fail\n");
  }
}

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void allocateDeviceMemory(float **d_ptr, size_t size, const char *name) {
  cudaError_t err = cudaMalloc((void **)d_ptr, size);
  std::string msg = std::string("Failed to allocate device memory for ") + name;
  checkCudaError(err, msg.c_str());
}

void freeDeviceMemory(void *d_ptr, const char *name) {
  cudaError_t err = cudaFree(d_ptr);
  std::string msg = std::string("Failed to free device memory for ") + name;
  checkCudaError(err, msg.c_str());
}

void cleanDeviceMemory(void *d_ptr, size_t size, const char *name) {
  cudaError_t err = cudaMemset(d_ptr, 0, size);
  std::string msg = std::string("Failed to clean memory for ") + name;
  checkCudaError(err, msg.c_str());
}

void copyToDevice(float *d_dst, const float *h_src, size_t size, const char *msg) {
  cudaError_t err = cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
  checkCudaError(err, msg);
}

void copyToHost(float *h_dst, const float *d_src, size_t size, const char *msg) {
  cudaError_t err = cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
  checkCudaError(err, msg);
}