#include <cuda_runtime.h>
#include <stdio.h>

void checkCudaError(cudaError_t err, const char *msg);

__global__ void helloUDA(void) { printf("Hello CUDA from GPU!\n"); }

int main(int argc, char *argv[]) {
  printf("Hello CUDA from CPU!\n");

  helloUDA<<<1, 1>>>();
  checkCudaError(cudaGetLastError(), "Failed to launch kernel");

  printf("Program completed successfully.\n");
  return 0;
}

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
