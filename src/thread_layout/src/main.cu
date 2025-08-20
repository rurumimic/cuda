#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void displayIndex() {
  printf("Thread ID: (%d, %d, %d), Block ID: (%d, %d, %d), Block Dim: (%d, %d, %d), Grid Dim: (%d, %d, %d)\n",
         threadIdx.x, threadIdx.y, threadIdx.z,
         blockIdx.x, blockIdx.y, blockIdx.z,
         blockDim.x, blockDim.y, blockDim.z,
         gridDim.x, gridDim.y, gridDim.z);
}

void example1() {
  dim3 dimBlock(3, 1, 1); // dim3 dimBlock(3);
  dim3 dimGrid(2, 1, 1); // dim3 dimGrid(2);

  printf("dimGrid: (%d, %d, %d), dimBlock: (%d, %d, %d)\n",
         dimGrid.x, dimGrid.y, dimGrid.z,
         dimBlock.x, dimBlock.y, dimBlock.z);

  displayIndex<<<dimGrid, dimBlock>>>();
  cudaDeviceSynchronize();
}

void example2() {
  dim3 dimBlock(3, 2, 1);
  dim3 dimGrid(2, 4, 1);

  printf("dimGrid: (%d, %d, %d), dimBlock: (%d, %d, %d)\n",
         dimGrid.x, dimGrid.y, dimGrid.z,
         dimBlock.x, dimBlock.y, dimBlock.z);

  displayIndex<<<dimGrid, dimBlock>>>();
  cudaDeviceSynchronize();
}


int main(int argc, char *argv[]) {
  printf("CUDA Kernel Launch Example\n");
  printf("Example 1:\n");
  example1();

  printf("Example 2:\n");
  example2();

  return EXIT_SUCCESS;
}


