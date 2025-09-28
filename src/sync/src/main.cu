#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

// #include "index.cuh"

/* C = A * B
 * C = [M x N]
 * A = [M x K]
 * B = [K x N]
 */

#define BLOCK_SIZE 64
#define SIZE_JOBS (BLOCK_SIZE * 2)
#define WARP_SIZE 32

constexpr double kEpsilon = 1e-2;

void checkCudaError(cudaError_t err, const char *msg);
void displayDeviceMemory();
void allocateDeviceMemory(float **d_ptr, size_t size, const char *name);
void freeDeviceMemory(void *d_ptr, const char *name);
void cleanDeviceMemory(void *d_ptr, size_t size, const char *name);
void copyToDevice(float *d_dst, const float *h_src, size_t size, const char *msg);
void copyToHost(float *h_dst, const float *d_src, size_t size, const char *msg);

__global__ void sync_even(int *even, int *odd) {
  __shared__ int lock;

  if (threadIdx.x % 2 == 0) {
    atomicAdd(&lock, 1); // even threads increment the lock
    while (lock < (blockDim.x / 2)) {}; // wait for all even threads to finish
    atomicAdd(even, 1); // work for even threads
  } else {
    // odd threads wait for even threads to finish
    atomicAdd(odd, 1); // just to ensure memory visibility
  }

  // work all threads

  __syncthreads();
}

__global__ void sync_warp(int *global_count) {
  int thread_id = (blockDim.x * blockIdx.x) + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;

  __shared__ int master_threads[BLOCK_SIZE/WARP_SIZE]; // assuming BLOCK_SIZE is multiple of WARP_SIZE
  __shared__ int warp_count[BLOCK_SIZE/WARP_SIZE];
  __shared__ int block_count;

  if (threadIdx.x == 0) { // first thread of the block
    block_count = 0;
  }

  __syncthreads();

  if (threadIdx.x % WARP_SIZE == 0) { // first thread of each warp
    master_threads[warp_id] = thread_id;
    warp_count[warp_id] = 0; // initialize warp_count
  }

  __syncwarp();
  atomicAdd(&warp_count[warp_id], 1); // warp-level synchronization
  __syncwarp();

  if (threadIdx.x % WARP_SIZE == 0) { // grid-level synchronization
    atomicAdd(&block_count, warp_count[warp_id]);
  }

  __syncthreads();
  if (threadIdx.x == 0) { // grid-level synchronization
    atomicAdd(global_count, block_count);
  }

  printf("Thread %d in warp %d, master thread is %d\n", thread_id, warp_id, master_threads[warp_id]);
}

__global__ void sync_block(int *global_count) {
  int thread_id = (blockDim.x * blockIdx.x) + threadIdx.x;
  int block_id = blockIdx.x;

  __shared__ int master_threads[SIZE_JOBS / BLOCK_SIZE]; // assuming BLOCK_SIZE is multiple of WARP_SIZE
  __shared__ int block_count;

  if (threadIdx.x == 0) { // first thread of each block
    master_threads[blockIdx.x] = thread_id;
    block_count = 0;
  }

  __syncthreads();
  atomicAdd(&block_count, 1); // block-level synchronization
  __syncthreads();

  if (threadIdx.x == 0) { // grid-level synchronization
    atomicAdd(global_count, block_count);
  }

  printf("Thread %d in block %d, master thread is %d\n", thread_id, block_id, master_threads[block_id]);
}

int main(int argc, char *argv[]) {
  printf("Synchroization\n");

  dim3 blockDim(BLOCK_SIZE, 1, 1);
  dim3 gridDim((SIZE_JOBS + blockDim.x - 1) / blockDim.x);
  printf("Block Dim: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
  printf("Grid Dim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

  // grid synchronization is not natively supported in CUDA
  // need to use cooperative groups or launch multiple kernels
  // here we just use two separate kernel launches for demonstration

  int count;
  int *d_global_count;

  cudaMalloc((void **)&d_global_count, sizeof(int));
  cudaMemset(d_global_count, 0, sizeof(int));

  printf("Launch sync_block kernel\n");
  auto start = std::chrono::steady_clock::now();
  sync_block<<<gridDim, blockDim>>>(d_global_count);
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  checkCudaError(cudaGetLastError(), "Failed to launch kernel");
  /**
    * Thread 64 in block 1, master thread is 64
    * ...
    * Thread 128 in block 2, master thread is 128
    * ...
    * Thread 0 in block 0, master thread is 0
    * ...
    * Thread 192 in block 3, master thread is 192
    * ...
    */
  cudaMemcpy(&count, d_global_count, sizeof(int), cudaMemcpyDeviceToHost);
  printf("count with block synchronization: %d\n", count);
  printf("Kernel execution duration: %ld μs\n", duration.count());

  count = 0;
  cudaMemset(d_global_count, 0, sizeof(int));
  printf("Launch sync_warp kernel\n");
  start = std::chrono::steady_clock::now();
  sync_warp<<<1, BLOCK_SIZE>>>(d_global_count);
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  checkCudaError(cudaGetLastError(), "Failed to launch kernel");
  cudaMemcpy(&count, d_global_count, sizeof(int), cudaMemcpyDeviceToHost);
  printf("count with warp synchronization: %d\n", count);
  printf("Kernel execution duration: %ld μs\n", duration.count());
  /**
    * Thread 0 in warp 0, master thread is 0
    * Thread 1 in warp 0, master thread is 0
    * ...
    * Thread 31 in warp 0, master thread is 0
    * Thread 32 in warp 1, master thread is 32
    * ...
    * Thread 63 in warp 1, master thread is 32
    */

  int even = 0;
  int odd = 0;
  int *d_even;
  int *d_odd;

  cudaMalloc((void **)&d_even, sizeof(int));
  cudaMalloc((void **)&d_odd, sizeof(int));
  cudaMemset(d_even, 0, sizeof(int));
  cudaMemset(d_odd, 0, sizeof(int));

  printf("Launch sync_even kernel\n");
  start = std::chrono::steady_clock::now();
  sync_even<<<gridDim, blockDim>>>(d_even, d_odd);
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  checkCudaError(cudaGetLastError(), "Failed to launch kernel");
  cudaMemcpy(&even, d_even, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&odd, d_odd, sizeof(int), cudaMemcpyDeviceToHost);
  printf("even: %d, odd: %d\n", even, odd); // even: 64, odd: 64
  printf("Kernel execution duration: %ld μs\n", duration.count());

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
