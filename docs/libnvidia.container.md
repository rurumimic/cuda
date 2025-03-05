# libnvidia-container

- docs: [container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html)

![libnvidia-container](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/_images/runtime-architecture.png)

```bash
git clone https://github.com/NVIDIA/libnvidia-container
cd libnvidia-container
```

## Build

### Build a package

- support versions: [mk/docker.mk#L36-L41](https://github.com/NVIDIA/libnvidia-container/blob/95d3e86522976061e856724867ebcaf75c4e9b60/mk/docker.mk#L36-L41)

### ubuntu20.04

```bash
make ubuntu20.04
```

```bash
dist
└── ubuntu20.04
    └── amd64
        ├── libnvidia-container1_1.17.4+2.g95d3e865-1_amd64.deb
        ├── libnvidia-container1-dbg_1.17.4+2.g95d3e865-1_amd64.deb
        ├── libnvidia-container-dev_1.17.4+2.g95d3e865-1_amd64.deb
        └── libnvidia-container-tools_1.17.4+2.g95d3e865-1_amd64.deb
```

### Build from source

```bash
sudo apt-get install -y libntirpc-dev
```

### Ubuntu

```bash
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
        apt-utils \
        bmake \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        devscripts \
        dh-make \
        fakeroot \
        git \
        libcap-dev \
        libelf-dev \
        libseccomp-dev \
        lintian \
        lsb-release \
        m4 \
        pkg-config \
        xz-utils \
        gzip
```

```bash
curl -L https://go.dev/dl/go1.23.6.linux-amd64.tar.gz | sudo tar -C /usr/local -xz
export GOPATH=/go
export PATH=$GOPATH/bin:/usr/local/go/bin:$PATH
```

```bash
export WITH_NVCGO=yes
export WITH_LIBELF=no
export WITH_TIRPC=no
export WITH_SECCOMP=yes
```

```bash
make distclean
make
```

```bash
mkdir dist
export DIST_DIR=$PWD/dist
make dist
```

#### (option) compile_commands.json

```bash
apt-get install -y bear
git config --global --add safe.directory $PWD/libnvidia-container
bear make
```

```bash
cp compile_commands.json compile_commands.json.bk
# sed -i "s|/libnvidia-container|$PWD|g" compile_commands.json
sed -i '/"-fplan9-extensions",/d' compile_commands.json
```

###### .clangd

```yaml
CompileFlags:
  Add:
    - -std=gnu11
    - -I./deps/usr/local/include
    - -I/usr/include/ntirpc
Diagnostics:
  Suppress:
    - -Wimplicit-function-declaration
```

---

## Source

- src/cli/main.c: int main - load_libnvc()
- src/cli/libnvc.c: int load_libnvc() - load_libnvc_v1()
  - static int load_libnvc_v1() - load_libnvc_func(init)
- src/cli/libnvc.h: struct libnvc
  - libnvc_entry(container_new)
  - libnvc_entry(driver_new)
  - libnvc_entry(init)
- src/nvc.c: int nvc_init()
  - driver_init()
    - src/driver.c: rpc_init()
      - src/rpc.c - setup_service() - svc_run()
  - nvcgo_init()
    - src/nvcgo.c: rpc_init()

