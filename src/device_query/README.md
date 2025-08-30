# Device Query

- NVIDIA/cuda-samples: [Samples/1_Utilities/deviceQuery/deviceQuery.cpp](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/deviceQuery.cpp)

## Build & Run

recommend: justfile + cmake + ninja

### Just

```bash
just debug run
```

### Run

```bash
CUDA Driver Version: 12.8
CUDA Runtime Version: 12.8

Number of CUDA-capable devices: 1

Device 0: NVIDIA GeForce RTX 3070
  Compute capability: 8.6
  Total global memory: 7.63 GB
  Multiprocessors: 46
  CUDA Cores: 5888
  Clock rate: 1.84 GHz
  Memory clock rate: 7.00 GHz
  Memory bus width: 256 bits
  L2 cache size: 4096 KB
  Max threads per block: 1024
  Max threads per multiprocessor: 1536
  Max grid size: 2147483647 x 65535 x 65535
  Max block dimensions: 1024 x 1024 x 64
  Max blocks per multiprocessor: 16
  Warp size: 32
```

