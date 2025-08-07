# Docker

- github: [Open Container Initiative](https://github.com/opencontainers)
  - [opencontainers/runc](https://github.com/opencontainers/runc)

## Install Docker Engine

- docs: [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

## Containerd

- containerd
  - [runtime/v2](https://github.com/containerd/containerd/blob/main/core/runtime/v2/README.md)

## AI accelerators

- medium: [How Docker Runs Machine Learning on NVIDIA GPUs, AWS Inferentia, and Other Hardware AI Accelerators](https://medium.com/towards-data-science/how-docker-runs-machine-learning-on-nvidia-gpus-aws-inferentia-and-other-hardware-ai-accelerators-e076c6eb7802) by Shashank Prasanna
- youtube: [How does Docker run machine learning on AI accelerators (NVIDIA GPUs, AWS Inferentia)](https://www.youtube.com/watch?v=TEIPEPY44g8) by
Shashank Prasanna
- aws blog: [Why use Docker containers for machine learning development?](https://aws.amazon.com/ko/blogs/opensource/why-use-docker-containers-for-machine-learning-development/) by Shashank Prasanna

### ML software stack

![ml software stack by Shashank Prasanna](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*fC_UswDIFJ0BDWwI)

- Docker Container
  - Typical software stack
    - My Code
    - Tensorflow, PyTorch, Frameworks + Library Dependencies
    - Python
    - CPU ML libraries
  - **Hardware Accelator**
    - **AI accelerator ML libraries**
    - **AI accelerator drivers**
- OS
  - **AI accelerator drivers**: with matching versions
  - OS Kernel
  - Host OS
- Heterogeneous Hardware
  - CPU
  - **AI Accelerator**

#### Challenges

- Duplicating drivers = bloated VMs and containers
- Hardware driver versions must match
- Not portable (whole point of containers). difficult to scale
- Very brittle solution

### Container Runtimes

- docker: [alternative runtimes](https://docs.docker.com/engine/daemon/alternative-runtimes/)

#### runc/libcontainer/process_linux.go

- runc: [libcontainer/process_linux.go#L754](https://github.com/opencontainers/runc/blob/701516b57a55a6462eeea42bc5ce7e3f103d20da/libcontainer/process_linux.go#L754)

```go
func (p *initProcess) start() (retErr error) {
	ierr := parseSync(p.comm.syncSockParent, func(sync *syncT) error {
		switch sync.Type {
		case procHooks:
			if p.config.Config.HasHook(configs.Prestart, configs.CreateRuntime) {
				if err := hooks.Run(configs.Prestart, s); err != nil {
					return err
				}
```

---

## Nvidia

- docs: [container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html)

![libnvidia-container](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/_images/runtime-architecture.png)

### Configs

- `/etc/docker/daemon.json`
- `/etc/nvidia-container-runtime/config.toml`

#### /etc/docker/daemon.json

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

#### in a tritonserver image

```bash
docker run --rm -it --gpus all nvcr.io/nvidia/tritonserver:25.07-py3 bash
```

```bash
ls -Fl /dev | grep nvidia

crw-rw-rw- 1 root root 511,   0 Aug  7 13:00 nvidia-uvm
crw-rw-rw- 1 root root 511,   1 Aug  7 13:00 nvidia-uvm-tools
crw-rw-rw- 1 root root 195,   0 Aug  7 13:00 nvidia0
crw-rw-rw- 1 root root 195, 255 Aug  7 13:00 nvidiactl
```

```bash
nvidia-smi

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.15              Driver Version: 570.86.15      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070        Off |   00000000:2B:00.0  On |                  N/A |
|  0%   50C    P3             49W /  270W |    1256MiB /   8192MiB |     21%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

---

## Neuron

- AWS docs: [Tutorial Docker Neuron OCI Hook Setup](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/containers/tutorials/tutorial-oci-hook.html)
- github: [awslabs/oci-add-hooks](https://github.com/awslabs/oci-add-hooks)

