#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(const float *a, const float *b, float *c,
                           int length) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < length) {
    c[i] = a[i] + b[i] + 0.0f;
  }
}

int main(int argc, char *argv[]) {
  cudaError_t err = cudaSuccess;

  int LENGTH = 50000;
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

  float *d_a = NULL;
  err = cudaMalloc((void **)&d_a, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *d_b = NULL;
  err = cudaMalloc((void **)&d_b, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *d_c = NULL;
  err = cudaMalloc((void **)&d_c, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("Copy: host to device\n");

  err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector from host to device: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector from host to device: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (LENGTH + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel: %d blocks x %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, LENGTH);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("Copy: device to host\n");
  err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector from device to host: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < LENGTH; i++) {
    if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at %d\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("OK\n");

  err = cudaFree(d_a);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_b);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_c);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  free(h_a);
  free(h_b);
  free(h_c);

  printf("End\n");
  return 0;
}
