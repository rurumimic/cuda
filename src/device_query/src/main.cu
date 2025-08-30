#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char *argv[]) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess) {
    fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  if (device_count == 0) {
    printf("No CUDA-capable devices found.\n");
    return EXIT_SUCCESS;
  }

  int driver_version = 0;
  int runtime_version = 0;

  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);

  printf("CUDA Driver Version: %d.%d\n", driver_version / 1000, (driver_version % 100) / 10);
  printf("CUDA Runtime Version: %d.%d\n", runtime_version / 1000, (runtime_version % 100) / 10);
  printf("\n");
  printf("Number of CUDA-capable devices: %d\n", device_count);

  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, i);

    printf("\n");
    printf("Device %d: %s\n", i, device_prop.name);
    printf("  Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
    printf("  Total global memory: %.2f GB\n", static_cast<float>(device_prop.totalGlobalMem) / (1 << 30));
    printf("  Multiprocessors: %d\n", device_prop.multiProcessorCount);
    printf("  CUDA Cores: %d\n", device_prop.multiProcessorCount * 128); // Assuming 128 cores per SM
    printf("  Clock rate: %.2f GHz\n", device_prop.clockRate / 1e6);
    printf("  Memory clock rate: %.2f GHz\n", device_prop.memoryClockRate / 1e6);
    printf("  Memory bus width: %d bits\n", device_prop.memoryBusWidth);
    printf("  L2 cache size: %d KB\n", device_prop.l2CacheSize / 1024);
    printf("  Max threads per block: %d\n", device_prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", device_prop.maxThreadsPerMultiProcessor);
    printf("  Max grid size: %d x %d x %d\n", device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
    printf("  Max block dimensions: %d x %d x %d\n", device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
    printf("  Max blocks per multiprocessor: %d\n", device_prop.maxBlocksPerMultiProcessor);
    printf("  Warp size: %d\n", device_prop.warpSize);
  }

  return EXIT_SUCCESS;
}

