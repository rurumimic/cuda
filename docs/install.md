# Install

- cuda: [downloads](https://developer.nvidia.com/cuda-downloads)
  - [12.9](https://developer.nvidia.com/cuda-12-9-0-download-archive)
- docs
  - [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux)
  - [container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

---

## Check Versions

```bash
nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_
```

```bash
nvidia-smi --query-gpu=compute_cap --format=csv

compute_cap
8.6
```

---

## Ubuntu 24.04

- CUDA Toolkit 13.0
- ubuntu: [nvidia-drivers-installation](https://ubuntu.com/server/docs/nvidia-drivers-installation)

```bash
cat /proc/driver/nvidia/version
sudo ubuntu-drivers list
sudo ubuntu-drivers install
```

### CUDA Toolkit Installer

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```

<details>
    <summary>installing...</summary>

```bash
==============================================================================
 Installing
==============================================================================
  Package:                        Version:                               Size:
  ca-certificates-java            20240118                               12 KB
  cuda-cccl-12-8                  12.8.55-1                             966 KB
  cuda-command-line-tools-12-8    12.8.0-1                                3 KB
  cuda-compiler-12-8              12.8.0-1                                3 KB
  cuda-crt-12-8                   12.8.61-1                              81 KB
  cuda-cudart-12-8                12.8.57-1                             170 KB
  cuda-cudart-dev-12-8            12.8.57-1                             1.2 MB
  cuda-cuobjdump-12-8             12.8.55-1                             210 KB
  cuda-cupti-12-8                 12.8.57-1                            11.4 MB
  cuda-cupti-dev-12-8             12.8.57-1                             4.1 MB
  cuda-cuxxfilt-12-8              12.8.55-1                             191 KB
  cuda-documentation-12-8         12.8.55-1                              50 KB
  cuda-driver-dev-12-8            12.8.57-1                              29 KB
  cuda-gdb-12-8                   12.8.55-1                            25.1 MB
  cuda-libraries-12-8             12.8.0-1                                3 KB
  cuda-libraries-dev-12-8         12.8.0-1                                3 KB
  cuda-nsight-12-8                12.8.55-1                           118.7 MB
  cuda-nsight-compute-12-8        12.8.0-1                                4 KB
  cuda-nsight-systems-12-8        12.8.0-1                                3 KB
  cuda-nvcc-12-8                  12.8.61-1                            36.1 MB
  cuda-nvdisasm-12-8              12.8.55-1                             5.1 MB
  cuda-nvml-dev-12-8              12.8.55-1                             127 KB
  cuda-nvprof-12-8                12.8.57-1                             2.4 MB
  cuda-nvprune-12-8               12.8.55-1                              61 KB
  cuda-nvrtc-12-8                 12.8.61-1                            62.3 MB
  cuda-nvrtc-dev-12-8             12.8.61-1                            53.1 MB
  cuda-nvtx-12-8                  12.8.55-1                              52 KB
  cuda-nvvm-12-8                  12.8.61-1                            43.8 MB
  cuda-nvvp-12-8                  12.8.57-1                           114.6 MB
  cuda-opencl-12-8                12.8.55-1                              24 KB
  cuda-opencl-dev-12-8            12.8.55-1                              89 KB
  cuda-profiler-api-12-8          12.8.55-1                              19 KB
  cuda-sanitizer-12-8             12.8.55-1                            10.3 MB
  cuda-toolkit-12-8               12.8.0-1                                3 KB
  cuda-toolkit-12-8-config-comm…  12.8.57-1                              16 KB
  cuda-toolkit-12-config-common   12.8.57-1                              16 KB
  cuda-toolkit-config-common      12.8.57-1                              16 KB
  cuda-tools-12-8                 12.8.0-1                                2 KB
  cuda-visual-tools-12-8          12.8.0-1                                3 KB
  default-jre                     2:1.21-75+exp1                     922 Bytes
  default-jre-headless            2:1.21-75+exp1                          3 KB
  fonts-dejavu-extra              2.37-8                                1.9 MB
  gds-tools-12-8                  1.13.0.11-1                          39.0 MB
  java-common                     0.75+exp1                               7 KB
  libatk-wrapper-java             0.40.0-3build2                         54 KB
  libatk-wrapper-java-jni         0.40.0-3build2                         46 KB
  libcublas-12-8                  12.8.3.14-1                         474.4 MB
  libcublas-dev-12-8              12.8.3.14-1                         497.9 MB
  libcufft-12-8                   11.3.3.41-1                         150.5 MB
  libcufft-dev-12-8               11.3.3.41-1                         301.8 MB
  libcufile-12-8                  1.13.0.11-1                           885 KB
  libcufile-dev-12-8              1.13.0.11-1                           2.7 MB
  libcurand-12-8                  10.3.9.55-1                          44.7 MB
  libcurand-dev-12-8              10.3.9.55-1                          44.8 MB
  libcusolver-12-8                11.7.2.55-1                         159.5 MB
  libcusolver-dev-12-8            11.7.2.55-1                         105.7 MB
  libcusparse-12-8                12.5.7.53-1                         165.5 MB
  libcusparse-dev-12-8            12.5.7.53-1                         170.8 MB
  libnpp-12-8                     12.3.3.65-1                         131.0 MB
  libnpp-dev-12-8                 12.3.3.65-1                         129.5 MB
  libnvfatbin-12-8                12.8.55-1                             743 KB
  libnvfatbin-dev-12-8            12.8.55-1                             619 KB
  libnvjitlink-12-8               12.8.61-1                            28.3 MB
  libnvjitlink-dev-12-8           12.8.61-1                            26.1 MB
  libnvjpeg-12-8                  12.3.5.57-1                           2.9 MB
  libnvjpeg-dev-12-8              12.3.5.57-1                           2.6 MB
  libxcb-cursor0                  0.1.4-1build1                          11 KB
  libxcb-xinerama0                1.15-1ubuntu2                           5 KB
  libxcb-xinput0                  1.15-1ubuntu2                          33 KB
  nsight-compute-2025.1.0         2025.1.0.14-1                       294.6 MB
  nsight-systems-2024.6.2         2024.6.2.225-246235244400v0         374.0 MB
  openjdk-21-jre                  21.0.6+7-1~24.04.1                    227 KB
  openjdk-21-jre-headless         21.0.6+7-1~24.04.1                   46.4 MB
```

</details>

```bash
Install 73 Packages

 Total download size  3.3 GB
 Disk space required  6.7 GB
```

### Driver Installer

To install the open kernel module flavor:

```bash
# open kernel module flavor
sudo apt-get install -y nvidia-open

# legacy kernel module flavor
# sudo apt-get install -y cuda-drivers
```

### Env

```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```bash
sudo apt-get update
```

```bash
sudo apt-get install -y nvidia-container-toolkit
```

<details>
    <summary>installing...</summary>

```bash
=================================================================
 Installing
=================================================================
  Package:                            Version:            Size:
  libnvidia-container-tools           1.17.4-1            22 KB
  libnvidia-container1                1.17.4-1           926 KB
  nvidia-container-toolkit            1.17.4-1           1.2 MB
  nvidia-container-toolkit-base       1.17.4-1           3.7 MB

=================================================================
 Summary
=================================================================
 Install 4 Packages

 Total download size   5.8 MB
 Disk space required  27.7 MB
```

</details>

#### Configure Docker

add in `daemon.json`:

```bash
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/daemon.json
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
```

```json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

```bash
sudo systemctl restart docker
```

option:

```bash
sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups=false --in-place
```

```bash
cat /etc/nvidia-container-runtime/config.toml

[nvidia-container-cli]
#no-cgroups = false
```

#### Test a workload

```bash
docker run --rm --runtime=nvidia --gpus all \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
  ubuntu nvidia-smi
```

```bash
docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

```bash
docker run --rm -it --gpus=all \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

```bash
docker run --rm -it --gpus '"device=0"' nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

---

## Ubuntu 22.04

- CUDA Toolkit 12.8

### CUDA Toolkit Installer

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

<details>
    <summary>installing...</summary>

```bash
===========================================================================
 Installing
===========================================================================
  Package:                         Version:                           Size:
  cuda-cccl-12-8                   12.8.55-1                         966 KB
  cuda-command-line-tools-12-8     12.8.0-1                            3 KB
  cuda-compiler-12-8               12.8.0-1                            3 KB
  cuda-crt-12-8                    12.8.61-1                          81 KB
  cuda-cudart-12-8                 12.8.57-1                         170 KB
  cuda-cudart-dev-12-8             12.8.57-1                         1.2 MB
  cuda-cuobjdump-12-8              12.8.55-1                         210 KB
  cuda-cupti-12-8                  12.8.57-1                        11.4 MB
  cuda-cupti-dev-12-8              12.8.57-1                         4.1 MB
  cuda-cuxxfilt-12-8               12.8.55-1                         191 KB
  cuda-documentation-12-8          12.8.55-1                          50 KB
  cuda-driver-dev-12-8             12.8.57-1                          29 KB
  cuda-gdb-12-8                    12.8.55-1                        25.1 MB
  cuda-libraries-12-8              12.8.0-1                            3 KB
  cuda-libraries-dev-12-8          12.8.0-1                            3 KB
  cuda-nsight-12-8                 12.8.55-1                       118.7 MB
  cuda-nsight-compute-12-8         12.8.0-1                            4 KB
  cuda-nsight-systems-12-8         12.8.0-1                            3 KB
  cuda-nvcc-12-8                   12.8.61-1                        36.1 MB
  cuda-nvdisasm-12-8               12.8.55-1                         5.1 MB
  cuda-nvml-dev-12-8               12.8.55-1                         127 KB
  cuda-nvprof-12-8                 12.8.57-1                         2.4 MB
  cuda-nvprune-12-8                12.8.55-1                          61 KB
  cuda-nvrtc-12-8                  12.8.61-1                        62.3 MB
  cuda-nvrtc-dev-12-8              12.8.61-1                        53.1 MB
  cuda-nvtx-12-8                   12.8.55-1                          52 KB
  cuda-nvvm-12-8                   12.8.61-1                        43.8 MB
  cuda-nvvp-12-8                   12.8.57-1                       114.6 MB
  cuda-opencl-12-8                 12.8.55-1                          24 KB
  cuda-opencl-dev-12-8             12.8.55-1                          89 KB
  cuda-profiler-api-12-8           12.8.55-1                          19 KB
  cuda-sanitizer-12-8              12.8.55-1                        10.3 MB
  cuda-toolkit-12-8                12.8.0-1                            3 KB
  cuda-toolkit-12-8-config-common  12.8.57-1                          16 KB
  cuda-toolkit-12-config-common    12.8.57-1                          16 KB
  cuda-toolkit-config-common       12.8.57-1                          16 KB
  cuda-tools-12-8                  12.8.0-1                            2 KB
  cuda-visual-tools-12-8           12.8.0-1                            3 KB
  default-jre                      2:1.11-72build2                896 Bytes
  default-jre-headless             2:1.11-72build2                     3 KB
  fonts-dejavu-extra               2.37-2build1                      2.0 MB
  gds-tools-12-8                   1.13.0.11-1                      39.0 MB
  libatk-wrapper-java              0.38.0-5build1                     53 KB
  libatk-wrapper-java-jni          0.38.0-5build1                     49 KB
  libcublas-12-8                   12.8.3.14-1                     474.4 MB
  libcublas-dev-12-8               12.8.3.14-1                     497.9 MB
  libcufft-12-8                    11.3.3.41-1                     150.5 MB
  libcufft-dev-12-8                11.3.3.41-1                     301.8 MB
  libcufile-12-8                   1.13.0.11-1                       885 KB
  libcufile-dev-12-8               1.13.0.11-1                       2.7 MB
  libcurand-12-8                   10.3.9.55-1                      44.7 MB
  libcurand-dev-12-8               10.3.9.55-1                      44.8 MB
  libcusolver-12-8                 11.7.2.55-1                     159.5 MB
  libcusolver-dev-12-8             11.7.2.55-1                     105.7 MB
  libcusparse-12-8                 12.5.7.53-1                     165.5 MB
  libcusparse-dev-12-8             12.5.7.53-1                     170.8 MB
  libnpp-12-8                      12.3.3.65-1                     131.0 MB
  libnpp-dev-12-8                  12.3.3.65-1                     129.5 MB
  libnvfatbin-12-8                 12.8.55-1                         743 KB
  libnvfatbin-dev-12-8             12.8.55-1                         619 KB
  libnvjitlink-12-8                12.8.61-1                        28.3 MB
  libnvjitlink-dev-12-8            12.8.61-1                        26.1 MB
  libnvjpeg-12-8                   12.3.5.57-1                       2.9 MB
  libnvjpeg-dev-12-8               12.3.5.57-1                       2.6 MB
  libxcb-cursor0                   0.1.1-4ubuntu1                     11 KB
  nsight-compute-2025.1.0          2025.1.0.14-1                   294.6 MB
  nsight-systems-2024.6.2          2024.6.2.225-246235244400v0     374.0 MB
  openjdk-11-jre                   11.0.26+4-1ubuntu1~22.04          214 KB
```

</details>

```bash
 Install 68 Packages

 Total download size  3.6 GB
 Disk space required  8.4 GB
```

### Driver Installer

To install the open kernel module flavor:

```bash
# open kernel module flavor
sudo apt-get install -y nvidia-open

# legacy kernel module flavor
# sudo apt-get install -y cuda-drivers
```

### Env

```bash
export PATH=/usr/local/cuda-12/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
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

