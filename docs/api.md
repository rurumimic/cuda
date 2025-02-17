# Driver API, Runtime API

- docs: [Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
- docs: [Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)

## Comparison: CUDA Runtime API vs. Driver API

| Feature             | **Runtime API**        | **Driver API**         |
|---------------------|----------------------|----------------------|
| **Ease of Use**     | Simple and intuitive | More complex |
| **Context Management** | Automatic management | Explicit management required |
| **Kernel Execution** | `kernel<<<blocks, threads>>>()` | `cuLaunchKernel()` |
| **Memory Management** | `cudaMalloc`, `cudaMemcpy` | `cuMemAlloc`, `cuMemcpyHtoD` |
| **Flexibility**     | Limited (automated)  | High (fine-grained control) |
| **Library Compatibility** | Directly works with cuBLAS, cuFFT | May require extra handling in some cases |

- **Use Runtime API for most cases.**
- **Use Driver API when precise control is needed.**

