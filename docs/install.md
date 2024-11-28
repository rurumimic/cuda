# Install

- cuda: [downloads](https://developer.nvidia.com/cuda-downloads)
- docs
  - [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux)

---

## Ubuntu 22.04

- CUDA Toolkit 12.3 Update 1

### CUDA Toolkit Installer

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

---

## Arch Linux

- wiki: [gpgpu](https://wiki.archlinux.org/title/GPGPU)
- arch packages: [cuda](https://archlinux.org/packages/extra/x86_64/cuda/)

### CUDA Toolkit Installer

```bash
sudo pacman -Syu cuda
```

```bash
Packages (8) alsa-ucm-conf-1.2.13-2  alsa-utils-1.2.13-2  gcc13-13.3.1+r432+gfc8bd63119c0-1 gcc13-libs-13.3.1+r432+gfc8bd63119c0-1  github-cli-2.63.0-1  libzip-1.11.2-1 opencl-nvidia-565.57.01-2  cuda-12.6.3-1
```

```bash
Total Download Size:   1897.05 MiB
Total Installed Size:  5150.12 MiB
Net Upgrade Size:      5105.97 MiB
```

```bash
- The cuda binaries are in /opt/cuda/bin/
- The cuda samples are in /opt/cuda/samples/
- The cuda docs are in /opt/cuda/doc/
- You need to source /etc/profile or restart your session in order for the CUDA
  binaries to appear in your $PATH
- The default host compiler for nvcc is set by the $NVCC_CCBIN environment
  variable in /etc/profile.d/cuda.sh
- The default host compiler for nvcc is no longer configured using symlinks in
  /opt/cuda/bin/ but by the $NVCC_CCBIN environment variable in
  /etc/profile.d/cuda.sh.  You need to source /etc/profile or restart your
  session for it to be available in your environment.  Additionally, you may
  need to clear the build cache of your projects where the old path may have
  been recorded.
- When you uninstall an old, unrequired version of GCC that was previously
  required by cuda for the default host compiler ($NVCC_CCBIN), you may need
  to source /etc/profile or restart your session.  Additionally, you may need
  to clear the build cache of your projects where the old path may be still
  recorded.

Optional dependencies for cuda
    gdb: for cuda-gdb
    glu: required for some profiling tools in CUPTI
    nvidia-utils: for NVIDIA drivers (not needed in CDI containers) [installed]
    rdma-core: for GPUDirect Storage (libcufile_rdma.so)
```

### Driver Installer

```bash
sudo pacman -Syu nvidia
```

### Env

```bash
export PATH=/opt/cuda/bin/${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/opt/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

