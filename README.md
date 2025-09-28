# CUDA

- nvidia developer
  - [cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
  - [gpu compute capability](https://developer.nvidia.com/cuda-gpus)
- docs
  - [quick start](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
  - [support compiler](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#host-compiler-support-policy)
  - [best practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
  - [cuda c++ programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- source
  - [samples](https://developer.nvidia.com/cuda-code-samples)
  - github: [nvidia/cuda-samples](https://github.com/nvidia/cuda-samples)
- repos
  - [cutlass](https://github.com/NVIDIA/cutlass)

---

## GPU Compute Capability

- [gpu compute capability](https://developer.nvidia.com/cuda-gpus)

```bash
nvidia-smi --query-gpu=compute_cap --format=csv

compute_cap
8.6
```

---

## Code

### Samples

```bash
git clone https://github.com/NVIDIA/cuda-samples.git
```

#### c++11_cuda

- Introduction: [c++11_cuda](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/c++11_cuda)

```bash
cd Samples/0_Introduction/c++11_cuda
```

##### Compile

```bash
make HOST_COMPILER=clang++ SMS="86" dbg=1
make HOST_COMPILER=g++ SMS="86" dbg=1
make HOST_COMPILER=g++-13 SMS="86" dbg=1
```

#### Run

```bash
./c++11_cuda

GPU Device 0: "Ampere" with compute capability 8.6

Read 3223503 byte corpus from ./warandpeace.txt
counted 107310 instances of 'x', 'y', 'z', or 'w' in "./warandpeace.txt"
```

---

## Docs

- [install](docs/install.md)
- [clang](docs/clang.md): format
- [api](docs/api.md): driver, runtime
- [huggingface](docs/huggingface.md)
  - [text embeddings inference](docs/text.embeddings.inference.md)
- [docker](docs/docker.md)
- nvidia
  - [triton](docs/triton.md)
  - [libnvidia-container](docs/libnvidia.container.md)
  - [dynamo](docs/dynamo.md)
  - [tensorRT](docs/tensorrt.md), src/[tensorrt](src/tensorrt/README.md)
- [leetgpu](docs/leetgpu.md)

---

## Code

- Hello CUDA: [hello_cuda](src/hello_cuda/README.md), [hello_cuda with C++](src/hello_cuda_cpp/README.md)
- Thread: [thread_layout](src/thread_layout/README.md)
- Device: [device_query](src/device_query/README.md)
- Vector: [vector_add](src/vector_add/README.md)
- Matrix
  - add: [matrix_add](src/matrix_add/README.md), [matrix_add_large](src/matrix_add_large/README.md)
  - mul: [matrix_mul](src/matrix_mul/README.md), [matrix_mul_shared_memory](src/matrix_mul_shared_memory/README.md), [matrix_mul_shared_memory_large](src/matrix_mul_shared_memory_large/README.md)
- TensorRT: [tensorrt](src/tensorrt/README.md)
- Sync: [sync](src/sync/README.md), [streams + event](src/streams/README.md)

---

## Ref

- [CUDA Books archive](https://developer.nvidia.com/cuda-books-archive)
- book: [Programming Massively Parallel Processors](https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638)
- book: [CUDA Programming](https://github.com/bluekds/CUDA_Programming)
- book: [The Art of HPC](https://theartofhpc.com/)
- youtube: [CUDA Programming Course – High-Performance Computing with GPUs](https://www.youtube.com/watch?v=86FAWCzIe_4)
- youtube: [GPU MODE](https://www.youtube.com/@GPUMODE)
- [GPU Glossary](https://modal.com/gpu-glossary)
- UIUC: [Introduction to Parallel Programming with CUDA](https://newfrontiers.illinois.edu/news-and-events/introduction-to-parallel-programming-with-cuda/)

