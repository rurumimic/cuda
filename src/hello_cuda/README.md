# Hello CUDA

## Build

```bash
make
nvcc -o hello_cuda main.cu
```

## Run

```bash
./hello_cuda

Hello CUDA from CPU!
Hello CUDA from GPU!
Program completed successfully.
```

errors:

```bash
Failed to launch kernel: CUDA driver version is insufficient for CUDA runtime version
Failed to launch kernel: no CUDA-capable device is detected
```

---

## Syntax

### CUDA Function Qualifiers

| Qualifier      | Caller (Who Calls)  | Execution Space (Where It Runs) | Return Type Support |
|----------------|---------------------|---------------------------------|---------------------|
| `__host__`     | **CPU**             | **CPU**                         | ✅ Allowed |
| `__device__`   | **GPU**             | **GPU**                         | ✅ Allowed |
| `__global__`   | **CPU**             | **GPU (Kernel Function)**       | ❌ `void` only |
| `__host__ __device__`   | **CPU** & **GPU**            | **CPU** & **GPU**       |  ✅ Allowed |

#### examples

```cu
__host__ void cpuFunction() {
    printf("This runs on the CPU.\n");
}

__host__ __device__ int square(int x) {
    return x * x;
}

__global__ void kernel() {
    int result = square(5);  // Runs on GPU
}

kernel<<<1, 1>>>();
```

### Execution Configuration

```cpp
helloCUDA<<<1, 1>>>(args); // 1 block, 1 thread
helloCUDA<<<(1, 1, 1), (1, 1, 1)>>>(args); // 1 block, 1 thread

helloCUDA<<<(Grid Dimension), (Block Dimension)>>>(args);
```

