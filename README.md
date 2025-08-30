# CUDA

- nvidia developer
  - [cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
  - [gpu compute capability](https://developer.nvidia.com/cuda-gpus)
- docs
  - [quick start](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
  - [support compiler](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#host-compiler-support-policy)
  - [best practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- source
  - [samples](https://developer.nvidia.com/cuda-code-samples)
  - github: [nvidia/cuda-samples](https://github.com/nvidia/cuda-samples)

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

---

## Code

- Hello CUDA: [hello_cuda](src/hello_cuda/README.md), [hello_cuda with C++](src/hello_cuda_cpp/README.md)
- Thread: [thread_layout](src/thread_layout/README.md)
- Device: [device_query](src/device_query/README.md)
- Vector: [vector_add](src/vector_add/README.md)
- Matrix: [matrix_add](src/matrix_add/README.md), [matrix_add_large](src/matrix_add_large/README.md)
- TensorRT: [tensorrt](src/tensorrt/README.md)

