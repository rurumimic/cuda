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
export CC=gcc
export CXX=g++
```

- gcc: Ubuntu 11.4.0-1ubuntu1~22.04
- g++: Ubuntu 11.4.0-1ubuntu1~22.04

```bash
export WITH_NVCGO=yes
export WITH_LIBELF=yes
export WITH_TIRPC=yes
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

# bear v2
bear make
# bear v3
bear -- make
```

for clangd:

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
    # - -I/usr/include/ntirpc
    - -fms-extensions
    - -include
    - stdbool.h
Diagnostics:
  Suppress:
    - -Wimplicit-function-declaration
    - -Wpragma_pop_visibility_mismatch
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

### src/cli/libnvc.c

```cpp
static int
load_libnvc_v1(void)
{
        #define load_libnvc_func(func) \
            libnvc.func = nvc_##func

        load_libnvc_func(config_free);
        load_libnvc_func(config_new);
        load_libnvc_func(container_config_free);
        load_libnvc_func(container_config_new);
        load_libnvc_func(container_free);
        load_libnvc_func(container_new);
        load_libnvc_func(context_free);
        load_libnvc_func(context_new);
        load_libnvc_func(device_info_free);
        load_libnvc_func(device_info_new);
        load_libnvc_func(device_mount);
        load_libnvc_func(driver_info_free);
        load_libnvc_func(driver_info_new);
        load_libnvc_func(driver_mount);
        load_libnvc_func(error);
        load_libnvc_func(init);
        load_libnvc_func(ldcache_update);
        load_libnvc_func(shutdown);
        load_libnvc_func(version);
        load_libnvc_func(nvcaps_style);
        load_libnvc_func(nvcaps_device_from_proc_path);
        load_libnvc_func(mig_device_access_caps_mount);
        load_libnvc_func(mig_config_global_caps_mount);
        load_libnvc_func(mig_monitor_global_caps_mount);
        load_libnvc_func(device_mig_caps_mount);
        load_libnvc_func(imex_channel_mount);

        return (0);
}
```

### src/nvc.h

```c
struct nvc_device_node {
  char *path;
  dev_t id;
};

struct nvc_driver_info {
  char *nvrm_version;
  char *cuda_version;
  char **bins;
  size_t nbins;
  char **libs;
  size_t nlibs;
  char **libs32;
  size_t nlibs32;
  char **ipcs;
  size_t nipcs;
  struct nvc_device_node *devs;
  size_t ndevs;
  char **firmwares;
  size_t nfirmwares;
};

struct nvc_mig_device {
  struct nvc_device *parent;
  char *uuid;
  unsigned int gi;
  unsigned int ci;
  char *gi_caps_path;
  char *ci_caps_path;
};

struct nvc_mig_device_info {
  struct nvc_mig_device *devices;
  size_t ndevices;
};

struct nvc_device {
  char *model;
  char *uuid;
  char *busid;
  char *arch;
  char *brand;
  struct nvc_device_node node;
  bool mig_capable;
  char *mig_caps_path;
  struct nvc_mig_device_info mig_devices;
};

struct nvc_device_info {
  struct nvc_device *gpus;
  size_t ngpus;
};

struct nvc_container_config {
  pid_t pid;
  char *rootfs;
  char *bins_dir;
  char *libs_dir;
  char *libs32_dir;
  char *cudart_dir;
  char *ldconfig;
};

const struct nvc_version *nvc_version(void);
struct nvc_context *nvc_context_new(void);
void nvc_context_free(struct nvc_context *);
struct nvc_config *nvc_config_new(void);
void nvc_config_free(struct nvc_config *);
int nvc_init(struct nvc_context *, const struct nvc_config *, const char *);
int nvc_shutdown(struct nvc_context *);
struct nvc_container_config *nvc_container_config_new(pid_t, const char *);
void nvc_container_config_free(struct nvc_container_config *);
struct nvc_container *nvc_container_new(struct nvc_context *, const struct nvc_container_config *, const char *);
void nvc_container_free(struct nvc_container *);
struct nvc_driver_info *nvc_driver_info_new(struct nvc_context *, const char *);
void nvc_driver_info_free(struct nvc_driver_info *);
struct nvc_device_info *nvc_device_info_new(struct nvc_context *, const char *);
void nvc_device_info_free(struct nvc_device_info *);
int nvc_nvcaps_style(void);
int nvc_nvcaps_device_from_proc_path(struct nvc_context *, const char *, struct nvc_device_node *);
int nvc_driver_mount(struct nvc_context *, const struct nvc_container *, const struct nvc_driver_info *);
int nvc_device_mount(struct nvc_context *, const struct nvc_container *, const struct nvc_device *);
int nvc_mig_device_access_caps_mount(struct nvc_context *, const struct nvc_container *, const struct nvc_mig_device *);
int nvc_mig_config_global_caps_mount(struct nvc_context *, const struct nvc_container *);
int nvc_mig_monitor_global_caps_mount(struct nvc_context *, const struct nvc_container *);
int nvc_device_mig_caps_mount(struct nvc_context *, const struct nvc_container *, const struct nvc_device *);
int nvc_imex_channel_mount(struct nvc_context *, const struct nvc_container *, const struct nvc_imex_channel *);
int nvc_ldcache_update(struct nvc_context *, const struct nvc_container *);
const char *nvc_error(struct nvc_context *);
```

### src/nvc.c

```cpp
int
nvc_init(struct nvc_context *ctx, const struct nvc_config *cfg, const char *opts)
{
        if (driver_init(&ctx->err, &ctx->dxcore, ctx->cfg.root, ctx->cfg.uid, ctx->cfg.gid) < 0)
                goto fail;

        #ifdef WITH_NVCGO
        if (nvcgo_init(&ctx->err) < 0)
                goto fail;
        #endif
}
```

### src/driver.c

```cpp
int
driver_init(struct error *err, struct dxcore_context *dxcore, const char *root, uid_t uid, gid_t gid)
{
        rpc_prog = (struct rpc_prog){
                .name = "driver",
                .id = DRIVER_PROGRAM,
                .version = DRIVER_VERSION,
                .dispatch = driver_program_1,
        };

        *ctx = (struct driver){
                .rpc = {0},
                .root = {0},
                .nvml_path = SONAME_LIBNVML,
                .uid = uid,
                .gid = gid,
                .nvml_dl = NULL,
        };

        if (rpc_init(err, &ctx->rpc, &rpc_prog) < 0)
                goto fail;

        ret = call_rpc(err, &ctx->rpc, &res, driver_init_1);
}
```

### src/rpc.c

```cpp
int
rpc_init(struct error *err, struct rpc *rpc, struct rpc_prog *prog)
{
        *rpc = (struct rpc){false, {-1, -1}, -1, NULL, NULL, *prog};

        pid = getpid();
        if (socketpair(PF_LOCAL, SOCK_STREAM|SOCK_CLOEXEC, 0, rpc->fd) < 0 || (rpc->pid = fork()) < 0) {
                error_set(err, "%s rpc service process creation failed", rpc->prog.name);
                goto fail;
        }

        if (rpc->pid == 0)
                setup_service(err, rpc, pid);

        if (setup_client(err, rpc) < 0)
                goto fail;
}
```

```cpp
static noreturn void
setup_service(struct error *err, struct rpc *rpc, pid_t ppid)
{
        svc_run();
}
```
