# CUDA

- nvidia developer
  - [cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
  - [gpu compute capability](https://developer.nvidia.com/cuda-gpus)
- docs
  - [quick start](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
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

##### Compile

```bash
make HOST_COMPILER=g++ SMS="86" dbg=1
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

