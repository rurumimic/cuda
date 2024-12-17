#include <cuda_runtime.h>
#include <stdio.h>

#include <string>

#define LENGTH 50000
#define THREADS_PER_BLOCK 256

__global__ void vector_add(const float *a, const float *b, float *c,
                           int length) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < length) {
    c[i] = a[i] + b[i] + 0.0f;
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
  checkCudaError(
      err,
      (std::string("Failed to allocate device memory for ") + name).c_str());
}

void freeDeviceMemory(void *d_ptr, const char *name) {
  cudaError_t err = cudaFree(d_ptr);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device memory for %s: %s\n", name,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void copyToDevice(float *d_dst, const float *h_src, size_t size,
                  const char *msg) {
  checkCudaError(cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice), msg);
}

void copyToHost(float *h_dst, const float *d_src, size_t size,
                const char *msg) {
  checkCudaError(cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost), msg);
}

int main(int argc, char *argv[]) {
  size_t size = LENGTH * sizeof(float);
  printf("Vector length: %d\n", LENGTH);

  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_c = (float *)malloc(size);

  if (h_a == NULL || h_b == NULL || h_c == NULL) {
    fprintf(stderr, "Failed to allocate host vectors\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < LENGTH; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  float *d_a, *d_b, *d_c;
  allocateDeviceMemory(&d_a, size, "d_a");
  allocateDeviceMemory(&d_b, size, "d_b");
  allocateDeviceMemory(&d_c, size, "d_c");

  printf("Copy: host to device\n");

  copyToDevice(d_a, h_a, size, "Failed to copy h_a to device");
  copyToDevice(d_b, h_b, size, "Failed to copy h_b to device");

  int blocksPerGrid = (LENGTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  printf("CUDA kernel: %d blocks x %d threads\n", blocksPerGrid,
         THREADS_PER_BLOCK);
  vector_add<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, LENGTH);
  checkCudaError(cudaGetLastError(), "Failed to launch kernel");

  printf("Copy: device to host\n");
  copyToDevice(h_c, d_c, size, "Failed to copy d_c to host");

  for (int i = 0; i < LENGTH; i++) {
    if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at %d\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Result verification: OK\n");

  freeDeviceMemory(d_a, "d_a");
  freeDeviceMemory(d_b, "d_b");
  freeDeviceMemory(d_c, "d_c");

  free(h_a);
  free(h_b);
  free(h_c);

  printf("Program completed successfully.\n");
  return 0;
}
