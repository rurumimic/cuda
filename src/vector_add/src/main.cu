#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#define LENGTH 50000
#define THREADS_PER_BLOCK 256

constexpr double kEpsilon = 1e-5;

void checkCudaError(cudaError_t err, const char *msg);
void checkDeviceMemory();
void allocateDeviceMemory(float **d_ptr, size_t size, const char *name);
void freeDeviceMemory(void *d_ptr, const char *name);
void cleanDeviceMemory(void *d_ptr, size_t size, const char *name);
void copyToDevice(float *d_dst, const float *h_src, size_t size, const char *msg);
void copyToHost(float *h_dst, const float *d_src, size_t size, const char *msg);

__global__ void vector_add(const float *a, const float *b, float *c, int length) {
  int i = (blockDim.x * blockIdx.x) + threadIdx.x;

  if (i < length) {
    c[i] = a[i] + b[i] + 0.0F;
  }
}

int main(int argc, char *argv[]) {
  size_t size = LENGTH * sizeof(float);
  printf("Vector length: %d\n", LENGTH);

  printf("Allocate host memory\n");
  auto *h_a = (float *)malloc(size);
  auto *h_b = (float *)malloc(size);
  auto *h_c = (float *)malloc(size);

  if (h_a == nullptr || h_b == nullptr || h_c == nullptr) {
    fprintf(stderr, "Failed to allocate host vectors\n");
    exit(EXIT_FAILURE);
  }

  printf("Initialize host vectors\n");
  for (int i = 0; i < LENGTH; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  checkDeviceMemory();
  printf("Allocate device memory\n");

  float *d_a;
  float *d_b;
  float *d_c;
  allocateDeviceMemory(&d_a, size, "d_a");
  allocateDeviceMemory(&d_b, size, "d_b");
  allocateDeviceMemory(&d_c, size, "d_c");

  checkDeviceMemory();

  printf("Copy: host to device\n");

  copyToDevice(d_a, h_a, size, "Failed to copy h_a to device");
  copyToDevice(d_b, h_b, size, "Failed to copy h_b to device");

  printf("\n");

  int blocksPerGrid = (LENGTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  printf("CUDA kernel: %d blocks x %d threads\n", blocksPerGrid, THREADS_PER_BLOCK);

  printf("Launch vector_add kernel\n");
  vector_add<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, LENGTH);
  checkCudaError(cudaGetLastError(), "Failed to launch kernel");

  printf("Copy: device to host\n");
  copyToHost(h_c, d_c, size, "Failed to copy d_c to host");

  printf("Verify results\n");
  for (int i = 0; i < LENGTH; i++) {
    if (fabs(h_a[i] + h_b[i] - h_c[i]) > kEpsilon) {
      fprintf(stderr, "Result verification failed at %d\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Result verification: OK\n");
  printf("\n");

  printf("Cleaning device memory\n");
  cleanDeviceMemory(d_a, size, "d_a");
  cleanDeviceMemory(d_b, size, "d_b");
  cleanDeviceMemory(d_c, size, "d_c");

  checkDeviceMemory();

  printf("Free device memory\n");

  freeDeviceMemory(d_a, "d_a");
  freeDeviceMemory(d_b, "d_b");
  freeDeviceMemory(d_c, "d_c");

  checkDeviceMemory();

  free(h_a);
  free(h_b);
  free(h_c);

  printf("Program completed successfully.\n");
  return 0;
}

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void checkDeviceMemory() {
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
