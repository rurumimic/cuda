#include <cstdio>

#include "kernel.h"

__global__ void helloCUDA() { printf("Hello CUDA from GPU!\n"); }

void launchHelloCUDA() {
  helloCUDA<<<1, 1>>>();
  checkCudaError(cudaGetLastError(), "Failed to launch kernel");
}

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
