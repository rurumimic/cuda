# LeetGPU

- [LeetGPU](https://leetgpu.com/)
  - [CLI](https://leetgpu.com/cli)

## CLI

```bash
# sh
curl -fsSL https://cli.leetgpu.com/install.sh | sh

# zsh
curl -fsSL https://cli.leetgpu.com/install.sh | zsh
```

### List GPUs

```bash
leetgpu list-gpus

NVIDIA GTX TITAN X
NVIDIA GV100
NVIDIA QV100
NVIDIA TITAN V
NVIDIA RTX 2060 Super
NVIDIA RTX 3070
```

### Run CUDA on remote GPU

- [hello_cuda/src/main.cu](../src/hello_cuda/src/main.cu)

```bash
leetgpu run main.cu

Running in FUNCTIONAL mode...
Compiling...
Executing...
Hello CUDA from CPU!
Program completed successfully.
Hello CUDA from GPU!
Exit status: 0
```

```bash
leetgpu run main.cu --mode cycle-accurate

Running NVIDIA GTX TITAN X in CYCLE ACCURATE mode...
Compiling...
Executing...
Hello CUDA from CPU!
Hello CUDA from GPU!
Program completed successfully.
GPU Execution Time: 3.79 microseconds
Exit status: 0
```

