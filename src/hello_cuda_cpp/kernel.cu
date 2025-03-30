#include <stdio.h>

#include "kernel.h"

__global__ void helloCUDA(void) { printf("Hello CUDA from GPU!\n"); }

void launchHelloCUDA(void) {
  helloCUDA<<<1, 1>>>();
  checkCudaError(cudaGetLastError(), "Failed to launch kernel");
}

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
