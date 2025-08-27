#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "index.cuh"

#define GRID_1D_TID ((NUM_THREADS_IN_BLOCK * BID_X) + TID_IN_BLOCK)
#define GRID_2D_TID ((NUM_THREADS_IN_BLOCK * GDIM_X * BID_Y) + GRID_1D_TID)
#define GRID_3D_TID ((NUM_THREADS_IN_BLOCK * GDIM_X * GDIM_Y * BID_Z) + GRID_2D_TID)

__device__ __forceinline__ unsigned long long tid_in_block_u64() {
  const auto tx = static_cast<unsigned long long>(threadIdx.x);
  const auto ty = static_cast<unsigned long long>(threadIdx.y);
  const auto tz = static_cast<unsigned long long>(threadIdx.z);
  const auto bx = static_cast<unsigned long long>(blockDim.x);
  const auto by = static_cast<unsigned long long>(blockDim.y);

  // t = tx + bx * (ty + by * tz)
  const unsigned long long inner = (ty + (by * tz));
  return (tx + (bx * inner));
}

__device__ __forceinline__ unsigned long long threads_per_block_u64() {
  const auto bx = static_cast<unsigned long long>(blockDim.x);
  const auto by = static_cast<unsigned long long>(blockDim.y);
  const auto bz = static_cast<unsigned long long>(blockDim.z);
  return (bx * by * bz);
}

__device__ __forceinline__ unsigned long long block_id_linear_u64() {
  const auto bx = static_cast<unsigned long long>(blockIdx.x);
  const auto by = static_cast<unsigned long long>(blockIdx.y);
  const auto bz = static_cast<unsigned long long>(blockIdx.z);
  const auto GX = static_cast<unsigned long long>(gridDim.x);
  const auto GY = static_cast<unsigned long long>(gridDim.y);

  // B = bx + GX * (by + GY * bz)
  const unsigned long long inner = (by + (GY * bz));
  return (bx + (GX * inner));
}

__device__ __forceinline__ unsigned long long global_tid_u64() {
  const auto tpb = threads_per_block_u64();
  const auto b = block_id_linear_u64();
  const auto tin = tid_in_block_u64();
  // GLOBAL_TID = tin + b * tpb
  return (tin + (b * tpb));
}

// overflow is possible
__device__ __forceinline__ unsigned int global_tid_u32() { return static_cast<unsigned int>(global_tid_u64()); }

__global__ void displayIndex1() {
  printf("Thread ID: (%d, %d, %d), Block ID: (%d, %d, %d), Block Dim: (%d, %d, %d), Grid Dim: (%d, %d, %d)\n", threadIdx.x, threadIdx.y,
         threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

__global__ void displayIndex2() {
  printf("Thread ID: (%d, %d, %d), Block ID: (%d, %d, %d), Block Dim: (%d, %d, %d), Grid Dim: (%d, %d, %d)\n", TID_X, TID_Y, TID_Z, BID_X,
         BID_Y, BID_Z, BDIM_X, BDIM_Y, BDIM_Z, GDIM_X, GDIM_Y, GDIM_Z);
}

__global__ void displayIndex() {
  const unsigned long long gid = global_tid_u64();

  printf("GID: %3lld, Thread ID: (%d, %d, %d), Block ID: (%d, %d, %d), Block Dim: (%d, %d, %d), Grid Dim: (%d, %d, %d)\n", gid, TID_X,
         TID_Y, TID_Z, BID_X, BID_Y, BID_Z, BDIM_X, BDIM_Y, BDIM_Z, GDIM_X, GDIM_Y, GDIM_Z);
}

void example1() {
  dim3 dimBlock(3, 1, 1);  // dim3 dimBlock(3);
  dim3 dimGrid(2, 1, 1);   // dim3 dimGrid(2);

  printf("dimGrid: (%d, %d, %d), dimBlock: (%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

  displayIndex<<<dimGrid, dimBlock>>>();
  cudaDeviceSynchronize();
}

void example2() {
  dim3 dimBlock(3, 2, 1);
  dim3 dimGrid(2, 4, 1);

  printf("dimGrid: (%d, %d, %d), dimBlock: (%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

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
