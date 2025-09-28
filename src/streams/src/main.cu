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

#define BLOCK_SIZE 1024
#define NUM_BLOCKS (128 * 1024)
#define ARRAY_SIZE (1024 * NUM_BLOCKS)
#define STREAMS 4
#define WORK_LOAD 256

constexpr double kEpsilon = 1e-2;

void checkCudaError(cudaError_t err, const char *msg);
void displayDeviceMemory();
void allocateDeviceMemory(float **d_ptr, size_t size, const char *name);
void freeDeviceMemory(void *d_ptr, const char *name);
void cleanDeviceMemory(void *d_ptr, size_t size, const char *name);
void copyToDevice(float *d_dst, const float *h_src, size_t size, const char *msg);
void copyToHost(float *h_dst, const float *d_src, size_t size, const char *msg);

__global__ void kernel(const int *_in, int *_out) {
  int thread_id = (blockDim.x * blockIdx.x) + threadIdx.x;

  int temp = 0;
  int in = _in[thread_id];
  for (int i = 0; i < WORK_LOAD; i++) {
    temp = (temp + in * 5) % 10;
  }

  _out[thread_id] = temp;
}

int main(int argc, char *argv[]) {
  printf("Multi streams\n");

  int *in = nullptr;
  int *out = nullptr;
  int *d_in = nullptr;
  int *d_out = nullptr;

  cudaMallocHost(&in, sizeof(int) * ARRAY_SIZE);
  cudaMallocHost(&out, sizeof(int) * ARRAY_SIZE);
  cudaMalloc((void **)&d_in, sizeof(int) * ARRAY_SIZE);
  cudaMalloc((void **)&d_out, sizeof(int) * ARRAY_SIZE);
  cudaMemset(in, 0, sizeof(int) * ARRAY_SIZE);
  cudaMemset(out, 0, sizeof(int) * ARRAY_SIZE);
  cudaMemset(d_in, 0, sizeof(int) * ARRAY_SIZE);
  cudaMemset(d_out, 0, sizeof(int) * ARRAY_SIZE);

  for (int i = 0; i < ARRAY_SIZE; i++) {
    in[i] = i;
  }

  printf("Launch kernel\n");
  auto single_start = std::chrono::steady_clock::now();
  auto start = std::chrono::steady_clock::now();
  cudaMemcpy(d_in, in, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("Memcpy H2D duration: %ld ms\n", duration.count());

  start = std::chrono::steady_clock::now();
  kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_in, d_out);
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("Kernel launch duration: %ld ms\n", duration.count());

  start = std::chrono::steady_clock::now();
  cudaMemcpy(out, d_out, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  auto single_end = std::chrono::steady_clock::now();
  auto single_duration = std::chrono::duration_cast<std::chrono::milliseconds>(single_end - single_start);
  checkCudaError(cudaGetLastError(), "Failed to launch kernel");
  printf("Kernel execution duration: %ld ms\n", duration.count());
  printf("Single Stream duration: %ld ms\n", single_duration.count());

  cudaStream_t streams[STREAMS];
  cudaEvent_t event_start[STREAMS];
  cudaEvent_t event_end[STREAMS];

  for (int i = 0; i < STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
    cudaEventCreate(&event_start[i]);
    cudaEventCreate(&event_end[i]);
  }

  int chunk_size = ARRAY_SIZE / STREAMS;

  printf("Launch Multi Streams kernel\n");
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < STREAMS; i++) {
    cudaEventRecord(event_start[i], streams[i]);

    int offset = i * chunk_size;
    cudaMemcpyAsync(d_in + offset, in + offset, sizeof(int) * chunk_size, cudaMemcpyHostToDevice, streams[i]);
  }

  for (int i = 0; i < STREAMS; i++) {
    int offset = i * chunk_size;
    kernel<<<NUM_BLOCKS / STREAMS, BLOCK_SIZE, 0, streams[i]>>>(d_in + offset, d_out + offset);
  }

  for (int i = 0; i < STREAMS; i++) {
    int offset = i * chunk_size;
    cudaMemcpyAsync(out + offset, d_out +offset, sizeof(int) * chunk_size, cudaMemcpyDeviceToHost, streams[i]);

    cudaEventRecord(event_end[i], streams[i]);
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("Multi streams duration: %ld ms\n", duration.count());

  for (int i = 0; i < STREAMS; i++) {
    float millis = 0.0F;
    cudaEventElapsedTime(&millis, event_start[i], event_end[i]);
    printf("Stream %d duration: %f ms\n", i, millis);
  }

  for (int i = 0; i < STREAMS; i++) {
    cudaStreamDestroy(streams[i]);
    cudaEventDestroy(event_start[i]);
    cudaEventDestroy(event_end[i]);
  }

  cudaFreeHost(in);
  cudaFreeHost(out);
  cudaFree(d_in);
  cudaFree(d_out);

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
