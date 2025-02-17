#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

void launchHelloCUDA();
void checkCudaError(cudaError_t err, const char *msg);

#endif  // KERNEL_H
