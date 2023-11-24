# Install

- cuda: [downloads](https://developer.nvidia.com/cuda-downloads)
- docs
  - [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux)

## Ubuntu 22.04

- CUDA Toolkit 12.3 Update 1

### Base Installer

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-3
```

```bash
cuda-cccl-12-3 cuda-command-line-tools-12-3 cuda-compiler-12-3 cuda-crt-12-3 cuda-cudart-12-3 cuda-cudart-dev-12-3 cuda-cuobjdump-12-3 cuda-cupti-12-3 cuda-cupti-dev-12-3 cuda-cuxxfilt-12-3 cuda-documentation-12-3 cuda-driver-dev-12-3 cuda-gdb-12-3 cuda-libraries-12-3 cuda-libraries-dev-12-3 cuda-nsight-12-3 cuda-nsight-compute-12-3 cuda-nsight-systems-12-3 cuda-nvcc-12-3 cuda-nvdisasm-12-3 cuda-nvml-dev-12-3 cuda-nvprof-12-3 cuda-nvprune-12-3 cuda-nvrtc-12-3 cuda-nvrtc-dev-12-3 cuda-nvtx-12-3 cuda-nvvm-12-3 cuda-nvvp-12-3 cuda-opencl-12-3 cuda-opencl-dev-12-3 cuda-profiler-api-12-3 cuda-sanitizer-12-3 cuda-toolkit-12-3 cuda-toolkit-12-3-config-common cuda-toolkit-12-config-common cuda-toolkit-config-common cuda-tools-12-3 cuda-visual-tools-12-3 default-jre gds-tools-12-3 libcublas-12-3 libcublas-dev-12-3 libcufft-12-3 libcufft-dev-12-3 libcufile-12-3 libcufile-dev-12-3 libcurand-12-3 libcurand-dev-12-3 libcusolver-12-3 libcusolver-dev-12-3 libcusparse-12-3 libcusparse-dev-12-3 libnpp-12-3 libnpp-dev-12-3 libnvjitlink-12-3 libnvjitlink-dev-12-3 libnvjpeg-12-3 libnvjpeg-dev-12-3 libtinfo5 nsight-compute-2023.3.1 nsight-systems-2023.3.3
```

```bash
Install 61 Packages

 Total download size  2.8 GB
 Disk space required  6.4 GB
```

### Driver Installer

To install the open kernel module flavor:

```bash
sudo apt-get install -y nvidia-kernel-open-545
sudo apt-get install -y cuda-drivers-545
```

### Env

```bash
export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### libstdc++-12-dev

```bash
sudo apt install -y libstdc++-12-dev
```

