# Ampere

- nvidia: [ampere architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- [s21730-inside-the-nvidia-ampere-architecture.pdf](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21730-inside-the-nvidia-ampere-architecture.pdf)
- [GA102 architecture.pdf](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)

## Nsight

- [nsight compute](https://developer.nvidia.com/nsight-compute)
- [nsight systems](https://developer.nvidia.com/nsight-systems)

### Nsight Compute

```bash
cd src/vector_add
sudo /usr/local/cuda/bin/ncu ./build/debug/vector_add
```

### Nsight Systems

```bash
nsys profile ./build/debug/vector_add
```

## CUDA Device Query

```bash
git clone https://github.com/NVIDIA/cuda-samples
cd cuda-samples/Samples/1_Utilities/deviceQuery
```

build:

```bash
cmake -S . -B build -G Ninja
ninja -C build
```

run:

```bash
./build/deviceQuery
```

### NVIDIA GeForce RTX 3070

- 46 SM x 128 CUDA Cores/MP = 5888 CUDA Cores
  - SM: Streaming Multiprocessor = MP
- Warp = 32 threads
- Max 1536 threads/MP
  - 1536 threads / 32 threads = 48 warps/MP
- Max 1024 threads/block
  - 1024 threads / 32 threads = 32 warps/block
  - 512 threads block * 3 = 1536 threads/MP
    - run 3 blocks in parallel on each SM
- Max (x,y,z) dimension size of a thread block: (1024, 1024, 64)
- Max (x,y,z) dimension size of a grid size: (2147483647, 65535, 65535)
- Total 49152 B shared memory/block
- Total 102400 B shared memory/MP

```cpp
int blocksPerGrid = (LENGTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
vector_add<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, LENGTH);
```

result:

```bash
 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 3070"
  CUDA Driver Version / Runtime Version          12.8 / 12.8
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 7817 MBytes (8197111808 bytes)
  (046) Multiprocessors, (128) CUDA Cores/MP:    5888 CUDA Cores
  GPU Max Clock rate:                            1845 MHz (1.85 GHz)
  Memory Clock rate:                             7001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 43 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.8, CUDA Runtime Version = 12.8, NumDevs = 1
Result = PASS
```

- blocksPerGrid: ``
