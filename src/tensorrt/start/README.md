# Start TensorRT

- download: [tensorrt](https://developer.nvidia.com/tensorrt/download/10x)

```bash
uv venv
source .venv/bin/activate
uv pip install -U pip wheel
```

## Install

### trtexec

```bash
sudo dpkg -i ./nv-tensorrt-local-repo-ubuntu2404-10.9.0-cuda-12.8_1.0-1_amd64.deb
cd /var/nv-tensorrt-local-repo-ubuntu2404-10.9.0-cuda-12.8
sudo cp nv-tensorrt-local-53871640-keyring.gpg /usr/share/keyrings
sudo apt update
sudo apt install tensorrt
```

```bash
/usr/src/tensorrt/bin/trtexec --help
```

```bash
export PATH="$PATH:/usr/src/tensorrt/bin"
```

### python package

```bash
uv pip install -U tensorrt

 + nvidia-cuda-runtime-cu12==12.8.90
 + tensorrt==10.9.0.34
 + tensorrt-cu12==10.9.0.34
 + tensorrt-cu12-bindings==10.9.0.34
 + tensorrt-cu12-libs==10.9.0.34
```

- torch, torchvision

### Test TensorRT

```bash
pytest
```

## Model

```bash
wget https://download.onnxruntime.ai/onnx/models/resnet50.tar.gz
tar xzf resnet50.tar.gz
```

```bash
export PATH="$PATH:/usr/src/tensorrt/bin"
trtexec --onnx=resnet50/model.onnx --saveEngine=resnet50/model.trt
```

<details>
    <summary>trtexec --onnx=resnet50/model.onnx --saveEngine=resnet50/model.trt</summary>

```log
&&&& RUNNING TensorRT.trtexec [TensorRT v100900] [b34] # trtexec --onnx=resnet50/model.onnx --saveEngine=resnet50/model.trt
[04/07/2025-23:59:33] [I] TF32 is enabled by default. Add --noTF32 flag to further improve accuracy with some performance cost.
[04/07/2025-23:59:33] [I] === Model Options ===
[04/07/2025-23:59:33] [I] Format: ONNX
[04/07/2025-23:59:33] [I] Model: resnet50/model.onnx
[04/07/2025-23:59:33] [I] Output:
[04/07/2025-23:59:33] [I] === Build Options ===
[04/07/2025-23:59:33] [I] Memory Pools: workspace: default, dlaSRAM: default, dlaLocalDRAM: default, dlaGlobalDRAM: default, tacticSharedMem: default
[04/07/2025-23:59:33] [I] avgTiming: 8
[04/07/2025-23:59:33] [I] Precision: FP32
[04/07/2025-23:59:33] [I] LayerPrecisions:
[04/07/2025-23:59:33] [I] Layer Device Types:
[04/07/2025-23:59:33] [I] Calibration:
[04/07/2025-23:59:33] [I] Refit: Disabled
[04/07/2025-23:59:33] [I] Strip weights: Disabled
[04/07/2025-23:59:33] [I] Version Compatible: Disabled
[04/07/2025-23:59:33] [I] ONNX Plugin InstanceNorm: Disabled
[04/07/2025-23:59:33] [I] TensorRT runtime: full
[04/07/2025-23:59:33] [I] Lean DLL Path:
[04/07/2025-23:59:33] [I] Tempfile Controls: { in_memory: allow, temporary: allow }
[04/07/2025-23:59:33] [I] Exclude Lean Runtime: Disabled
[04/07/2025-23:59:33] [I] Sparsity: Disabled
[04/07/2025-23:59:33] [I] Safe mode: Disabled
[04/07/2025-23:59:33] [I] Build DLA standalone loadable: Disabled
[04/07/2025-23:59:33] [I] Allow GPU fallback for DLA: Disabled
[04/07/2025-23:59:33] [I] DirectIO mode: Disabled
[04/07/2025-23:59:33] [I] Restricted mode: Disabled
[04/07/2025-23:59:33] [I] Skip inference: Disabled
[04/07/2025-23:59:33] [I] Save engine: resnet50/model.trt
[04/07/2025-23:59:33] [I] Load engine:
[04/07/2025-23:59:33] [I] Profiling verbosity: 0
[04/07/2025-23:59:33] [I] Tactic sources: Using default tactic sources
[04/07/2025-23:59:33] [I] timingCacheMode: local
[04/07/2025-23:59:33] [I] timingCacheFile:
[04/07/2025-23:59:33] [I] Enable Compilation Cache: Enabled
[04/07/2025-23:59:33] [I] Enable Monitor Memory: Disabled
[04/07/2025-23:59:33] [I] errorOnTimingCacheMiss: Disabled
[04/07/2025-23:59:33] [I] Preview Features: Use default preview flags.
[04/07/2025-23:59:33] [I] MaxAuxStreams: -1
[04/07/2025-23:59:33] [I] BuilderOptimizationLevel: -1
[04/07/2025-23:59:33] [I] MaxTactics: -1
[04/07/2025-23:59:33] [I] Calibration Profile Index: 0
[04/07/2025-23:59:33] [I] Weight Streaming: Disabled
[04/07/2025-23:59:33] [I] Runtime Platform: Same As Build
[04/07/2025-23:59:33] [I] Debug Tensors:
[04/07/2025-23:59:33] [I] Input(s)s format: fp32:CHW
[04/07/2025-23:59:33] [I] Output(s)s format: fp32:CHW
[04/07/2025-23:59:33] [I] Input build shapes: model
[04/07/2025-23:59:33] [I] Input calibration shapes: model
[04/07/2025-23:59:33] [I] === System Options ===
[04/07/2025-23:59:33] [I] Device: 0
[04/07/2025-23:59:33] [I] DLACore:
[04/07/2025-23:59:33] [I] Plugins:
[04/07/2025-23:59:33] [I] setPluginsToSerialize:
[04/07/2025-23:59:33] [I] dynamicPlugins:
[04/07/2025-23:59:33] [I] ignoreParsedPluginLibs: 0
[04/07/2025-23:59:33] [I]
[04/07/2025-23:59:33] [I] === Inference Options ===
[04/07/2025-23:59:33] [I] Batch: Explicit
[04/07/2025-23:59:33] [I] Input inference shapes: model
[04/07/2025-23:59:33] [I] Iterations: 10
[04/07/2025-23:59:33] [I] Duration: 3s (+ 200ms warm up)
[04/07/2025-23:59:33] [I] Sleep time: 0ms
[04/07/2025-23:59:33] [I] Idle time: 0ms
[04/07/2025-23:59:33] [I] Inference Streams: 1
[04/07/2025-23:59:33] [I] ExposeDMA: Disabled
[04/07/2025-23:59:33] [I] Data transfers: Enabled
[04/07/2025-23:59:33] [I] Spin-wait: Disabled
[04/07/2025-23:59:33] [I] Multithreading: Disabled
[04/07/2025-23:59:33] [I] CUDA Graph: Disabled
[04/07/2025-23:59:33] [I] Separate profiling: Disabled
[04/07/2025-23:59:33] [I] Time Deserialize: Disabled
[04/07/2025-23:59:33] [I] Time Refit: Disabled
[04/07/2025-23:59:33] [I] NVTX verbosity: 0
[04/07/2025-23:59:33] [I] Persistent Cache Ratio: 0
[04/07/2025-23:59:33] [I] Optimization Profile Index: 0
[04/07/2025-23:59:33] [I] Weight Streaming Budget: 100.000000%
[04/07/2025-23:59:33] [I] Inputs:
[04/07/2025-23:59:33] [I] Debug Tensor Save Destinations:
[04/07/2025-23:59:33] [I] === Reporting Options ===
[04/07/2025-23:59:33] [I] Verbose: Disabled
[04/07/2025-23:59:33] [I] Averages: 10 inferences
[04/07/2025-23:59:33] [I] Percentiles: 90,95,99
[04/07/2025-23:59:33] [I] Dump refittable layers:Disabled
[04/07/2025-23:59:33] [I] Dump output: Disabled
[04/07/2025-23:59:33] [I] Profile: Disabled
[04/07/2025-23:59:33] [I] Export timing to JSON file:
[04/07/2025-23:59:33] [I] Export output to JSON file:
[04/07/2025-23:59:33] [I] Export profile to JSON file:
[04/07/2025-23:59:33] [I]
[04/07/2025-23:59:33] [I] === Device Information ===
[04/07/2025-23:59:33] [I] Available Devices:
[04/07/2025-23:59:33] [I]   Device 0: "NVIDIA GeForce RTX 3070" UUID: GPU-84a8068b-81a5-7df1-8d22-a523bb848827
[04/07/2025-23:59:33] [I] Selected Device: NVIDIA GeForce RTX 3070
[04/07/2025-23:59:33] [I] Selected Device ID: 0
[04/07/2025-23:59:33] [I] Selected Device UUID: GPU-84a8068b-81a5-7df1-8d22-a523bb848827
[04/07/2025-23:59:33] [I] Compute Capability: 8.6
[04/07/2025-23:59:33] [I] SMs: 46
[04/07/2025-23:59:33] [I] Device Global Memory: 7817 MiB
[04/07/2025-23:59:33] [I] Shared Memory per SM: 100 KiB
[04/07/2025-23:59:33] [I] Memory Bus Width: 256 bits (ECC disabled)
[04/07/2025-23:59:33] [I] Application Compute Clock Rate: 1.845 GHz
[04/07/2025-23:59:33] [I] Application Memory Clock Rate: 7.001 GHz
[04/07/2025-23:59:33] [I]
[04/07/2025-23:59:33] [I] Note: The application clock rates do not reflect the actual clock rates that the GPU is currently running at.
[04/07/2025-23:59:33] [I]
[04/07/2025-23:59:33] [I] TensorRT version: 10.9.0
[04/07/2025-23:59:33] [I] Loading standard plugins
[04/07/2025-23:59:33] [I] [TRT] [MemUsageChange] Init CUDA: CPU +2, GPU +0, now: CPU 26, GPU 1693 (MiB)
[04/07/2025-23:59:35] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +2646, GPU +414, now: CPU 2873, GPU 2107 (MiB)
[04/07/2025-23:59:35] [I] Start parsing network model.
[04/07/2025-23:59:36] [I] [TRT] ----------------------------------------------------------------
[04/07/2025-23:59:36] [I] [TRT] Input filename:   resnet50/model.onnx
[04/07/2025-23:59:36] [I] [TRT] ONNX IR version:  0.0.3
[04/07/2025-23:59:36] [I] [TRT] Opset version:    9
[04/07/2025-23:59:36] [I] [TRT] Producer name:    onnx-caffe2
[04/07/2025-23:59:36] [I] [TRT] Producer version:
[04/07/2025-23:59:36] [I] [TRT] Domain:
[04/07/2025-23:59:36] [I] [TRT] Model version:    0
[04/07/2025-23:59:36] [I] [TRT] Doc string:
[04/07/2025-23:59:36] [I] [TRT] ----------------------------------------------------------------
[04/07/2025-23:59:36] [I] Finished parsing network model. Parse time: 0.141199
[04/07/2025-23:59:36] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
[04/07/2025-23:59:45] [I] [TRT] Compiler backend is used during engine build.
[04/07/2025-23:59:46] [I] [TRT] Detected 1 inputs and 1 output network tensors.
[04/07/2025-23:59:46] [I] [TRT] Total Host Persistent Memory: 353104 bytes
[04/07/2025-23:59:46] [I] [TRT] Total Device Persistent Memory: 6656 bytes
[04/07/2025-23:59:46] [I] [TRT] Max Scratch Memory: 4096 bytes
[04/07/2025-23:59:46] [I] [TRT] [BlockAssignment] Started assigning block shifts. This will take 84 steps to complete.
[04/07/2025-23:59:46] [I] [TRT] [BlockAssignment] Algorithm ShiftNTopDown took 0.666687ms to assign 5 blocks to 84 nodes requiring 8229376 bytes.
[04/07/2025-23:59:46] [I] [TRT] Total Activation Memory: 8228864 bytes
[04/07/2025-23:59:46] [I] [TRT] Total Weights Memory: 103529216 bytes
[04/07/2025-23:59:46] [I] [TRT] Compiler backend is used during engine execution.
[04/07/2025-23:59:46] [I] [TRT] Engine generation completed in 10.6392 seconds.
[04/07/2025-23:59:46] [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 9 MiB, GPU 105 MiB
[04/07/2025-23:59:46] [I] Engine built in 10.7204 sec.
[04/07/2025-23:59:46] [I] Created engine with size: 101.811 MiB
[04/07/2025-23:59:47] [I] [TRT] Loaded engine size: 101 MiB
[04/07/2025-23:59:47] [I] Engine deserialized in 0.10331 sec.
[04/07/2025-23:59:47] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +8, now: CPU 0, GPU 106 (MiB)
[04/07/2025-23:59:47] [I] Setting persistentCacheLimit to 0 bytes.
[04/07/2025-23:59:47] [I] Created execution context with device memory size: 7.84766 MiB
[04/07/2025-23:59:47] [I] Using random values for input gpu_0/data_0
[04/07/2025-23:59:47] [I] Input binding for gpu_0/data_0 with dimensions 1x3x224x224 is created.
[04/07/2025-23:59:47] [I] Output binding for gpu_0/softmax_1 with dimensions 1x1000 is created.
[04/07/2025-23:59:47] [I] Starting inference
[04/07/2025-23:59:50] [I] Warmup completed 135 queries over 200 ms
[04/07/2025-23:59:50] [I] Timing trace has 2077 queries over 3.0039 s
[04/07/2025-23:59:50] [I]
[04/07/2025-23:59:50] [I] === Trace details ===
[04/07/2025-23:59:50] [I] Trace averages of 10 runs:
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.4891 ms - Host latency: 1.52709 ms (enqueue 0.251123 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39653 ms - Host latency: 1.43655 ms (enqueue 0.260228 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41783 ms - Host latency: 1.45835 ms (enqueue 0.263342 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39848 ms - Host latency: 1.43856 ms (enqueue 0.256186 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39766 ms - Host latency: 1.43816 ms (enqueue 0.264752 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39796 ms - Host latency: 1.43805 ms (enqueue 0.259625 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51767 ms - Host latency: 1.55604 ms (enqueue 0.254953 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.42765 ms - Host latency: 1.46621 ms (enqueue 0.260324 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.45132 ms - Host latency: 1.49118 ms (enqueue 0.260025 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41548 ms - Host latency: 1.45574 ms (enqueue 0.258871 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39848 ms - Host latency: 1.43913 ms (enqueue 0.28558 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39826 ms - Host latency: 1.43875 ms (enqueue 0.275851 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.59529 ms - Host latency: 1.63992 ms (enqueue 0.285071 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41824 ms - Host latency: 1.45816 ms (enqueue 0.251263 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39674 ms - Host latency: 1.43644 ms (enqueue 0.237393 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39766 ms - Host latency: 1.43887 ms (enqueue 0.273593 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39776 ms - Host latency: 1.4381 ms (enqueue 0.266827 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39684 ms - Host latency: 1.43743 ms (enqueue 0.265686 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39633 ms - Host latency: 1.43637 ms (enqueue 0.26517 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.60727 ms - Host latency: 1.64435 ms (enqueue 0.258243 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.42192 ms - Host latency: 1.46216 ms (enqueue 0.263129 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39786 ms - Host latency: 1.43795 ms (enqueue 0.273987 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39807 ms - Host latency: 1.43802 ms (enqueue 0.254218 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51091 ms - Host latency: 1.54827 ms (enqueue 0.255109 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.48931 ms - Host latency: 1.52958 ms (enqueue 0.259381 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41874 ms - Host latency: 1.45943 ms (enqueue 0.267255 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39796 ms - Host latency: 1.4389 ms (enqueue 0.275568 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39797 ms - Host latency: 1.43846 ms (enqueue 0.273212 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39736 ms - Host latency: 1.44348 ms (enqueue 0.289429 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39663 ms - Host latency: 1.43678 ms (enqueue 0.27218 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51398 ms - Host latency: 1.55261 ms (enqueue 0.264923 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.60757 ms - Host latency: 1.6451 ms (enqueue 0.259595 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.49678 ms - Host latency: 1.53575 ms (enqueue 0.23844 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41855 ms - Host latency: 1.46136 ms (enqueue 0.265533 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39765 ms - Host latency: 1.4377 ms (enqueue 0.256342 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39776 ms - Host latency: 1.43796 ms (enqueue 0.26806 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39704 ms - Host latency: 1.44072 ms (enqueue 0.299896 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.3989 ms - Host latency: 1.43915 ms (enqueue 0.266034 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51285 ms - Host latency: 1.55051 ms (enqueue 0.268677 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.48766 ms - Host latency: 1.52848 ms (enqueue 0.266431 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41876 ms - Host latency: 1.45892 ms (enqueue 0.246436 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51439 ms - Host latency: 1.55232 ms (enqueue 0.251184 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.49023 ms - Host latency: 1.53116 ms (enqueue 0.250848 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41681 ms - Host latency: 1.45688 ms (enqueue 0.239685 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39776 ms - Host latency: 1.43852 ms (enqueue 0.281403 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39918 ms - Host latency: 1.43895 ms (enqueue 0.239844 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39755 ms - Host latency: 1.43835 ms (enqueue 0.258282 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39817 ms - Host latency: 1.43876 ms (enqueue 0.261426 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51542 ms - Host latency: 1.55091 ms (enqueue 0.24621 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.48634 ms - Host latency: 1.52724 ms (enqueue 0.2659 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41844 ms - Host latency: 1.45833 ms (enqueue 0.271661 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39786 ms - Host latency: 1.43818 ms (enqueue 0.269 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39796 ms - Host latency: 1.43864 ms (enqueue 0.249707 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51491 ms - Host latency: 1.55446 ms (enqueue 0.243127 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.48776 ms - Host latency: 1.52781 ms (enqueue 0.273218 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41937 ms - Host latency: 1.45986 ms (enqueue 0.273901 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39901 ms - Host latency: 1.43918 ms (enqueue 0.266827 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39868 ms - Host latency: 1.43901 ms (enqueue 0.261279 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39899 ms - Host latency: 1.43906 ms (enqueue 0.256604 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39868 ms - Host latency: 1.44149 ms (enqueue 0.254382 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.5156 ms - Host latency: 1.55928 ms (enqueue 0.267566 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.6126 ms - Host latency: 1.65004 ms (enqueue 0.266577 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.49259 ms - Host latency: 1.52968 ms (enqueue 0.263403 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41886 ms - Host latency: 1.45896 ms (enqueue 0.270618 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39745 ms - Host latency: 1.43776 ms (enqueue 0.26217 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39716 ms - Host latency: 1.44006 ms (enqueue 0.27345 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39829 ms - Host latency: 1.43865 ms (enqueue 0.268127 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39692 ms - Host latency: 1.43712 ms (enqueue 0.258496 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51317 ms - Host latency: 1.55419 ms (enqueue 0.264465 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.60441 ms - Host latency: 1.64198 ms (enqueue 0.26416 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.49604 ms - Host latency: 1.53514 ms (enqueue 0.281665 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41946 ms - Host latency: 1.4608 ms (enqueue 0.272852 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39744 ms - Host latency: 1.4394 ms (enqueue 0.276208 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39694 ms - Host latency: 1.43741 ms (enqueue 0.274622 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39775 ms - Host latency: 1.44044 ms (enqueue 0.291846 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39692 ms - Host latency: 1.43735 ms (enqueue 0.257837 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39774 ms - Host latency: 1.43811 ms (enqueue 0.255554 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.60602 ms - Host latency: 1.64258 ms (enqueue 0.277661 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41765 ms - Host latency: 1.46001 ms (enqueue 0.274731 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51705 ms - Host latency: 1.5545 ms (enqueue 0.265247 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.48713 ms - Host latency: 1.52827 ms (enqueue 0.281665 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41887 ms - Host latency: 1.45917 ms (enqueue 0.252271 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39663 ms - Host latency: 1.43693 ms (enqueue 0.279529 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39745 ms - Host latency: 1.43748 ms (enqueue 0.269702 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39734 ms - Host latency: 1.43796 ms (enqueue 0.268018 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39692 ms - Host latency: 1.43973 ms (enqueue 0.276819 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51327 ms - Host latency: 1.55073 ms (enqueue 0.264697 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.48961 ms - Host latency: 1.52843 ms (enqueue 0.266833 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.42069 ms - Host latency: 1.46387 ms (enqueue 0.266907 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.40034 ms - Host latency: 1.44044 ms (enqueue 0.254236 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39869 ms - Host latency: 1.43865 ms (enqueue 0.241956 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51095 ms - Host latency: 1.55474 ms (enqueue 0.277454 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.48992 ms - Host latency: 1.52684 ms (enqueue 0.263709 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41888 ms - Host latency: 1.45972 ms (enqueue 0.273816 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39766 ms - Host latency: 1.43766 ms (enqueue 0.247705 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39847 ms - Host latency: 1.43877 ms (enqueue 0.244727 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39716 ms - Host latency: 1.43804 ms (enqueue 0.264612 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.3983 ms - Host latency: 1.43843 ms (enqueue 0.241687 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.54338 ms - Host latency: 1.58596 ms (enqueue 0.251465 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.45787 ms - Host latency: 1.49584 ms (enqueue 0.264465 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.4166 ms - Host latency: 1.45674 ms (enqueue 0.241809 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39784 ms - Host latency: 1.43854 ms (enqueue 0.258228 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.60305 ms - Host latency: 1.64102 ms (enqueue 0.244861 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41301 ms - Host latency: 1.45293 ms (enqueue 0.260413 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39818 ms - Host latency: 1.43932 ms (enqueue 0.291357 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39735 ms - Host latency: 1.43773 ms (enqueue 0.243884 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39836 ms - Host latency: 1.43845 ms (enqueue 0.248975 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.3991 ms - Host latency: 1.43967 ms (enqueue 0.263281 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39696 ms - Host latency: 1.43707 ms (enqueue 0.253943 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.60255 ms - Host latency: 1.64027 ms (enqueue 0.254199 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.61874 ms - Host latency: 1.65547 ms (enqueue 0.273206 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41979 ms - Host latency: 1.45985 ms (enqueue 0.236743 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39753 ms - Host latency: 1.43859 ms (enqueue 0.256873 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39889 ms - Host latency: 1.43951 ms (enqueue 0.264661 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.399 ms - Host latency: 1.43854 ms (enqueue 0.239648 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39786 ms - Host latency: 1.43788 ms (enqueue 0.253906 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.3994 ms - Host latency: 1.43917 ms (enqueue 0.245361 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51521 ms - Host latency: 1.55132 ms (enqueue 0.234412 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.58975 ms - Host latency: 1.62897 ms (enqueue 0.243262 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.49218 ms - Host latency: 1.5302 ms (enqueue 0.240015 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41926 ms - Host latency: 1.45967 ms (enqueue 0.274976 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39724 ms - Host latency: 1.43816 ms (enqueue 0.257227 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39683 ms - Host latency: 1.43722 ms (enqueue 0.242773 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39744 ms - Host latency: 1.43755 ms (enqueue 0.241394 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39705 ms - Host latency: 1.43756 ms (enqueue 0.268994 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51573 ms - Host latency: 1.55934 ms (enqueue 0.288989 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.49026 ms - Host latency: 1.52806 ms (enqueue 0.284131 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41855 ms - Host latency: 1.45879 ms (enqueue 0.260181 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.54863 ms - Host latency: 1.58684 ms (enqueue 0.257568 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.47769 ms - Host latency: 1.51704 ms (enqueue 0.265723 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39795 ms - Host latency: 1.43838 ms (enqueue 0.26167 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39817 ms - Host latency: 1.43887 ms (enqueue 0.262158 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39788 ms - Host latency: 1.43936 ms (enqueue 0.266943 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39866 ms - Host latency: 1.43894 ms (enqueue 0.251929 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39688 ms - Host latency: 1.43672 ms (enqueue 0.239453 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.53728 ms - Host latency: 1.57666 ms (enqueue 0.262134 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.60894 ms - Host latency: 1.64541 ms (enqueue 0.263892 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.46062 ms - Host latency: 1.50054 ms (enqueue 0.279834 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41455 ms - Host latency: 1.45459 ms (enqueue 0.258423 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39851 ms - Host latency: 1.43875 ms (enqueue 0.253589 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39722 ms - Host latency: 1.45071 ms (enqueue 0.32854 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39771 ms - Host latency: 1.43828 ms (enqueue 0.280469 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.3979 ms - Host latency: 1.43794 ms (enqueue 0.274365 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.5418 ms - Host latency: 1.58442 ms (enqueue 0.292847 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.60674 ms - Host latency: 1.64597 ms (enqueue 0.272656 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.45664 ms - Host latency: 1.49587 ms (enqueue 0.269556 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41362 ms - Host latency: 1.45503 ms (enqueue 0.29353 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39705 ms - Host latency: 1.43726 ms (enqueue 0.246826 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.3969 ms - Host latency: 1.43721 ms (enqueue 0.256152 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.3968 ms - Host latency: 1.43735 ms (enqueue 0.280396 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39856 ms - Host latency: 1.4395 ms (enqueue 0.266455 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.53665 ms - Host latency: 1.57678 ms (enqueue 0.269141 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.45425 ms - Host latency: 1.49221 ms (enqueue 0.27002 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41294 ms - Host latency: 1.45308 ms (enqueue 0.262793 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51541 ms - Host latency: 1.55151 ms (enqueue 0.257983 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.49077 ms - Host latency: 1.53196 ms (enqueue 0.278687 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41946 ms - Host latency: 1.46016 ms (enqueue 0.26748 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39751 ms - Host latency: 1.43784 ms (enqueue 0.265576 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39805 ms - Host latency: 1.43906 ms (enqueue 0.280322 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39641 ms - Host latency: 1.43594 ms (enqueue 0.24231 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.3989 ms - Host latency: 1.43889 ms (enqueue 0.238257 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51326 ms - Host latency: 1.55112 ms (enqueue 0.254956 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.48975 ms - Host latency: 1.53013 ms (enqueue 0.280908 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41665 ms - Host latency: 1.45742 ms (enqueue 0.280225 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39846 ms - Host latency: 1.43909 ms (enqueue 0.271021 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39785 ms - Host latency: 1.4415 ms (enqueue 0.27085 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.55098 ms - Host latency: 1.58816 ms (enqueue 0.276221 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.45171 ms - Host latency: 1.49172 ms (enqueue 0.246899 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41628 ms - Host latency: 1.4563 ms (enqueue 0.257349 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39639 ms - Host latency: 1.43689 ms (enqueue 0.275391 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39797 ms - Host latency: 1.43813 ms (enqueue 0.276099 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.3979 ms - Host latency: 1.43787 ms (enqueue 0.280298 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39839 ms - Host latency: 1.44434 ms (enqueue 0.303223 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.54819 ms - Host latency: 1.58506 ms (enqueue 0.263672 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.48721 ms - Host latency: 1.52734 ms (enqueue 0.279028 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39817 ms - Host latency: 1.44133 ms (enqueue 0.285132 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.5074 ms - Host latency: 1.54724 ms (enqueue 0.253735 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.48062 ms - Host latency: 1.52168 ms (enqueue 0.282886 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41702 ms - Host latency: 1.46011 ms (enqueue 0.277222 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39741 ms - Host latency: 1.43743 ms (enqueue 0.271924 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.3981 ms - Host latency: 1.43801 ms (enqueue 0.282178 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39727 ms - Host latency: 1.43823 ms (enqueue 0.275171 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39697 ms - Host latency: 1.4375 ms (enqueue 0.274658 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51843 ms - Host latency: 1.55872 ms (enqueue 0.27439 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.48767 ms - Host latency: 1.52468 ms (enqueue 0.266675 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.61423 ms - Host latency: 1.65046 ms (enqueue 0.261108 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41375 ms - Host latency: 1.45488 ms (enqueue 0.276172 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39878 ms - Host latency: 1.43872 ms (enqueue 0.240332 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39924 ms - Host latency: 1.43936 ms (enqueue 0.240894 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39695 ms - Host latency: 1.43735 ms (enqueue 0.24873 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39697 ms - Host latency: 1.43708 ms (enqueue 0.244531 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39749 ms - Host latency: 1.4385 ms (enqueue 0.258008 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.60178 ms - Host latency: 1.64031 ms (enqueue 0.244165 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.62153 ms - Host latency: 1.66177 ms (enqueue 0.282959 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.41606 ms - Host latency: 1.45669 ms (enqueue 0.288867 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39805 ms - Host latency: 1.43813 ms (enqueue 0.27041 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39727 ms - Host latency: 1.43713 ms (enqueue 0.25249 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39778 ms - Host latency: 1.43821 ms (enqueue 0.264209 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39927 ms - Host latency: 1.43909 ms (enqueue 0.244946 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39753 ms - Host latency: 1.43792 ms (enqueue 0.249609 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.51482 ms - Host latency: 1.55137 ms (enqueue 0.259521 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.48801 ms - Host latency: 1.5281 ms (enqueue 0.245288 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.53228 ms - Host latency: 1.57319 ms (enqueue 0.267065 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.49263 ms - Host latency: 1.52993 ms (enqueue 0.249365 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.42485 ms - Host latency: 1.46484 ms (enqueue 0.240869 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39883 ms - Host latency: 1.43892 ms (enqueue 0.244702 ms)
[04/07/2025-23:59:50] [I] Average on 10 runs - GPU latency: 1.39878 ms - Host latency: 1.43972 ms (enqueue 0.261963 ms)
[04/07/2025-23:59:50] [I]
[04/07/2025-23:59:50] [I] === Performance summary ===
[04/07/2025-23:59:50] [I] Throughput: 691.435 qps
[04/07/2025-23:59:50] [I] Latency: min = 1.4209 ms, max = 3.047 ms, mean = 1.48372 ms, median = 1.43823 ms, percentile(90%) = 1.45624 ms, percentile(95%) = 1.71796 ms, percentile(99%) = 2.61597 ms
[04/07/2025-23:59:50] [I] Enqueue Time: min = 0.208801 ms, max = 0.461304 ms, mean = 0.263621 ms, median = 0.259033 ms, percentile(90%) = 0.293823 ms, percentile(95%) = 0.311768 ms, percentile(99%) = 0.402832 ms
[04/07/2025-23:59:50] [I] H2D Latency: min = 0.0258789 ms, max = 0.0687256 ms, mean = 0.0362824 ms, median = 0.0361328 ms, percentile(90%) = 0.0378418 ms, percentile(95%) = 0.0385742 ms, percentile(99%) = 0.0616455 ms
[04/07/2025-23:59:50] [I] GPU Compute Time: min = 1.38855 ms, max = 3.00439 ms, mean = 1.44357 ms, median = 1.39775 ms, percentile(90%) = 1.40308 ms, percentile(95%) = 1.67834 ms, percentile(99%) = 2.57642 ms
[04/07/2025-23:59:50] [I] D2H Latency: min = 0.00286865 ms, max = 0.00561523 ms, mean = 0.00387057 ms, median = 0.00366211 ms, percentile(90%) = 0.00488281 ms, percentile(95%) = 0.00500488 ms, percentile(99%) = 0.00537109 ms
[04/07/2025-23:59:50] [I] Total Host Walltime: 3.0039 s
[04/07/2025-23:59:50] [I] Total GPU Compute Time: 2.9983 s
[04/07/2025-23:59:50] [W] * GPU compute time is unstable, with coefficient of variance = 13.1794%.
[04/07/2025-23:59:50] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[04/07/2025-23:59:50] [I] Explanations of the performance metrics are printed in the verbose logs.
[04/07/2025-23:59:50] [I]
&&&& PASSED TensorRT.trtexec [TensorRT v100900] [b34] # trtexec --onnx=resnet50/model.onnx --saveEngine=resnet50/model.trt
```

</details>

```bash

```
