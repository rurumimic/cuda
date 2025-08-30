#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "index.cuh"

/* C = A * B
 * C = [M x N]
 * A = [M x K]
 * B = [K x N]
 */

#define SIZE_M (512 * 2)
#define SIZE_N (512 * 4)
#define SIZE_K (512 * 2)
#define THREADS_PER_BLOCK 1024

void checkCudaError(cudaError_t err, const char *msg);
void displayDeviceMemory();
void allocateDeviceMemory(float **d_ptr, size_t size, const char *name);
void freeDeviceMemory(void *d_ptr, const char *name);
void cleanDeviceMemory(void *d_ptr, size_t size, const char *name);
void copyToDevice(float *d_dst, const float *h_src, size_t size, const char *msg);
void copyToHost(float *h_dst, const float *d_src, size_t size, const char *msg);

__global__ void matrix_mul(const float *a, const float *b, float *c, int size_m, int size_n, int size_k) {
  unsigned int col = (blockDim.x * blockIdx.x) + threadIdx.x;
  unsigned int row = (blockDim.y * blockIdx.y) + threadIdx.y;
  unsigned int i = (row * size_n) + col;

  if (row >= size_m || col >= size_n) {
    return;
  }

  c[i] = 0.0F;

  for (int k = 0; k < size_k; k++) {
    c[i] += (a[(row * size_k) + k] * b[(size_n * k) + col]);
  }
}

int main(int argc, char *argv[]) {
  size_t size_a = SIZE_M * SIZE_K * sizeof(float);
  size_t size_b = SIZE_K * SIZE_N * sizeof(float);
  size_t size_c = SIZE_M * SIZE_N * sizeof(float);
  printf("Matrix multiplication: C = A * B\n");
  printf("A [%d x %d]\n", SIZE_M, SIZE_K);
  printf("B [%d x %d]\n", SIZE_K, SIZE_N);
  printf("C [%d x %d]\n", SIZE_M, SIZE_N);
  printf("\n");

  printf("Allocate Host memory\n");
  auto *h_a = (float *)malloc(size_a);
  auto *h_b = (float *)malloc(size_b);
  auto *h_c = (float *)malloc(size_c);
  auto *h_hc = (float *)malloc(size_c);
  if (h_a == nullptr || h_b == nullptr || h_c == nullptr || h_hc == nullptr) {
    fprintf(stderr, "Failed to allocate host matrix\n");
    exit(EXIT_FAILURE);
  }

  printf("Initialize Host matrix\n");
  for (int i = 0; i < (SIZE_M * SIZE_K); i++) {
    h_a[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
  }
  for (int i = 0; i < (SIZE_K * SIZE_N); i++) {
    h_b[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
  }
  printf("Matrix A[0][0] = %f\n", h_a[0]);
  printf("\n");

  printf("Matrix mul on Host\n");
  auto start = std::chrono::steady_clock::now();
  for (int x = 0; x < SIZE_N; x++) {
    for (int y = 0; y < SIZE_M; y++) {
      unsigned int i = (y * SIZE_N) + x;
      h_hc[i] = 0.0F;
      for (int k = 0; k < SIZE_K; k++) {
        h_hc[i] += (h_a[(y * SIZE_K) + k] * h_b[(SIZE_N * k) + x]);
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  printf("Host matrix mul duration: %ld s\n", secs.count());
  printf("\n");

  displayDeviceMemory();

  printf("Allocate Device memory\n");
  float *d_a;
  float *d_b;
  float *d_c;
  allocateDeviceMemory(&d_a, size_a, "d_a");
  allocateDeviceMemory(&d_b, size_b, "d_b");
  allocateDeviceMemory(&d_c, size_c, "d_c");

  printf("Cleaning Device memory\n");
  cleanDeviceMemory(d_a, size_a, "d_a");
  cleanDeviceMemory(d_b, size_b, "d_b");
  cleanDeviceMemory(d_c, size_c, "d_c");
  printf("\n");

  displayDeviceMemory();

  printf("Copy: Host to Device\n");
  start = std::chrono::steady_clock::now();
  copyToDevice(d_a, h_a, size_a, "Failed to copy h_a to Device");
  copyToDevice(d_b, h_b, size_b, "Failed to copy h_b to Device");
  end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  printf("Copy duration: %ld µs\n", duration.count());
  printf("\n");

  dim3 blockDim(32, 32);
  dim3 gridDim((ceil((float)SIZE_N / blockDim.x)), (ceil((float)SIZE_M / blockDim.y)));
  printf("Block Dim: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
  printf("Grid Dim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

  printf("Launch matrix_mul kernel\n");
  start = std::chrono::steady_clock::now();
  matrix_mul<<<gridDim, blockDim>>>(d_a, d_b, d_c, SIZE_M, SIZE_N, SIZE_K);
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  checkCudaError(cudaGetLastError(), "Failed to launch kernel");
  printf("Kernel execution duration: %ld ms\n", millis.count());
  printf("\n");

  printf("Copy: Device to Host\n");
  start = std::chrono::steady_clock::now();
  copyToHost(h_c, d_c, size_c, "Failed to copy d_c to Host");
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  printf("Copy duration: %ld µs\n", duration.count());
  printf("\n");

  printf("Verify results\n");
  for (int i = 0; i < SIZE_M * SIZE_N; i++) {
    if (h_hc[i] != h_c[i]) {
      printf("h_hc[%d] = %f, h_c[%d] = %f\n", i, h_hc[i], i, h_c[i]);
      fprintf(stderr, "Result verification failed at %d\n", i);
      exit(EXIT_FAILURE);
    }
  }
  printf("Result verification: OK\n");
  printf("\n");

  displayDeviceMemory();

  printf("Free Device memory\n");
  freeDeviceMemory(d_a, "d_a");
  freeDeviceMemory(d_b, "d_b");
  freeDeviceMemory(d_c, "d_c");
  printf("\n");

  displayDeviceMemory();

  printf("Free Host memory\n");
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_hc);

  return EXIT_SUCCESS;
}

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void displayDeviceMemory() {
  size_t free;
  size_t total;

  cudaMemGetInfo(&free, &total);

  printf("-----Device memory-----\n");
  printf("free:  %zu bytes\n", free);
  printf("total: %zu bytes\n\n", total);
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
