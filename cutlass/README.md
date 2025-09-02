# cutlass

CUDA Templates for Linear Algebra Subroutines

- github: [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
- docs
  - [c++ quickstart](https://docs.nvidia.com/cutlass/media/docs/cpp/quickstart.html): [quickstart.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/quickstart.md)
  - [with clang](https://docs.nvidia.com/cutlass/media/docs/cpp/build/building_with_clang_as_host_compiler.html)
- install: [cudnn](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html)

## Compile

### Requirements

Install clang 19:

```bash
sudo /usr/bin/bash -c "$(curl -fsSL https://apt.llvm.org/llvm.sh)" -- 19 all
```

Install dependencies:

```bash
sudo apt-get install \
cmake ninja-build pkg-config libgtk-3-dev liblzma-dev libstdc++-12-dev
```

Install cuDNN 9 for CUDA 12:

```bash
sudo apt-get install zlib1g
sudo apt-get install cudnn9-cuda-12
```

### Clone the repo

```bash
git clone https://github.com/NVIDIA/cutlass
```

### Build the source

```bash
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
```

```bash
mkdir build
```

#### Configure with cmake

options:

- `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`

```bash
cmake -S . -B build \
-DCUTLASS_NVCC_ARCHS=86 \
-DCUTLASS_ENABLE_TESTS=OFF \
-DCUTLASS_UNITY_BUILD_ENABLED=OFF \
-DCUTLASS_ENABLE_CUBLAS=ON \
-DCUTLASS_ENABLE_CUDNN=ON \
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
-DCMAKE_CXX_COMPILER=/usr/bin/clang++-19 \
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++-19 \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
```

#### Build with cmake

```bash
cmake --build build --parallel 12
```

```bash
[ 99%] Linking CUDA executable 69_hopper_int4_bf16_grouped_gemm
[ 99%] Built target 69_hopper_int4_bf16_grouped_gemm
90 warnings generated.
[100%] Linking CXX executable cutlass_profiler
[100%] Built target cutlass_profiler
[100%] Linking CUDA executable 86_blackwell_mixed_dtype_gemm
[100%] Built target 86_blackwell_mixed_dtype_gemm
[100%] Linking CUDA executable 88_hopper_fmha
[100%] Built target 88_hopper_fmha
```

#### .clangd

- global [~/.config/clangd/config.yaml](config.yaml)
- local [.clangd](.clangd)

---

## Run

### cutlass_profiler

```bash
./build/tools/profiler/cutlass_profiler --version
./build/tools/profiler/cutlass_profiler --help
```

#### Examples

GEMM:

```bash
./build/tools/profiler/cutlass_profiler --kernels=sgemm --m=4352 --n=4096 --k=4096
```

convolution:

```bash
./build/tools/profiler/cutlass_profiler --kernels=s1688fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --pad_h=1 --pad_w=1
```

2-D convolution operators:

```bash
./build/tools/profiler/cutlass_profiler --operation=conv2d --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3
```

