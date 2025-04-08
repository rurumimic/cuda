# Start TensorRT

- download: [tensorrt](https://developer.nvidia.com/tensorrt/download/10x)
- tutorial
  - [tutorial-runtime.ipynb](https://github.com/NVIDIA/TensorRT/blob/main/quickstart/SemanticSegmentation/tutorial-runtime.ipynb)

```bash
uv venv
source .venv/bin/activate
uv pip install -U pip wheel
```

---

## Run a TensorRT Model

```bash
python src/segmentation_tutorial.py
```

```log
Running TensorRT inference for FCN-ResNet101
Reading engine from file models/fcn-resnet101.engine
Reading input image from file images/input.ppm
Writing output image to file images/output.ppm
[04/08/2025-23:57:00] [TRT] [E] [defaultAllocator.cpp::deallocate::85] Error Code 1: Cuda Runtime (invalid argument)
[04/08/2025-23:57:00] [TRT] [E] [defaultAllocator.cpp::deallocate::85] Error Code 1: Cuda Runtime (invalid argument)
[04/08/2025-23:57:00] [TRT] [E] [cudaDriverHelpers.cpp::operator()::107] Error Code 1: Cuda Driver (invalid resource handle)
[04/08/2025-23:57:00] [TRT] [E] [cudaDriverHelpers.cpp::operator()::107] Error Code 1: Cuda Driver (invalid resource handle)
[04/08/2025-23:57:00] [TRT] [E] [cudaDriverHelpers.cpp::operator()::107] Error Code 1: Cuda Driver (invalid resource handle)
```

- [images/input.ppm](images/input.ppm)
- [images/output.ppm](images/output.ppm)

---

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
export PATH="$PATH:/usr/src/tensorrt/bin"
trtexec --help
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

---

## Model

- NVIDIA/TensorRT v10.9: [quickstart/SemanticSegmentation/export.py](https://github.com/NVIDIA/TensorRT/blob/release/10.9/quickstart/SemanticSegmentation/export.py)

pull a docker image:

```bash
docker pull nvcr.io/nvidia/pytorch:20.12-py3
```

```bash
docker images | grep pytorch

nvcr.io/nvidia/pytorch    20.12-py3    ad0f29ddeb63    4 years ago    14.2GB
```

clone TensorRT:

```bash
git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT/quickstart
```

export the model to ONNX format:

```bash
docker run --rm -it --gpus all -p 8888:8888 -v `pwd`:/workspace -w /workspace/SemanticSegmentation nvcr.io/nvidia/pytorch:20.12-py3 bash
python3 export.py
```

<details>
    <summary>python3 export.py</summary>

```bash
=============
== PyTorch ==
=============

NVIDIA Release 20.12 (build 17950526)
PyTorch Version 1.8.0a0+1606899

Container image Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

Copyright (c) 2014-2020 Facebook Inc.
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
Copyright (c) 2015      Google Inc.
Copyright (c) 2015      Yangqing Jia
Copyright (c) 2013-2016 The Caffe contributors
All rights reserved.

NVIDIA Deep Learning Profiler (dlprof) Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
NVIDIA modifications are covered by the license terms that apply to the underlying project or file.
Failed to detect NVIDIA driver version.

NOTE: MOFED driver for multi-node communication was not detected.
      Multi-node communication performance may be reduced.

NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be
   insufficient for PyTorch.  NVIDIA recommends the use of the following flags:
   nvidia-docker run --ipc=host ...

root@fdf1b5ed6c45:/workspace/SemanticSegmentation# python3 export.py
Exporting ppm image input.ppm
Downloading: "https://github.com/pytorch/vision/archive/v0.6.0.zip" to /root/.cache/torch/hub/v0.6.0.zip
Downloading: "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth" to /root/.cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth
100%|█████ 170M/170M [00:15<00:00, 11.6MB/s]
Downloading: "https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth" to /root/.cache/torch/hub/checkpoints/fcn_resnet101_coco-7ecb50ca.pth
100%|█████ 208M/208M [00:17<00:00, 12.3MB/s]
Exporting ONNX model fcn-resnet101.onnx

root@fdf1b5ed6c45:/workspace/SemanticSegmentation# exit
```

</details>

```bash
TensorRT/quickstart/
└── SemanticSegmentation/
    ├── export.py
    ├── fcn-resnet101.onnx
    ├── input.ppm
    ├── Makefile
    ├── tutorial-runtime.cpp
    └── tutorial-runtime.ipynb
```

build a TensorRT engine from ONNX using the trtexec tool:

```bash
export PATH="$PATH:/usr/src/tensorrt/bin"
trtexec --onnx=fcn-resnet101.onnx --saveEngine=fcn-resnet101.engine --optShapes=input:1x3x1026x1282
```

<details>
    <summary>trtexec --onnx=fcn-resnet101.onnx --saveEngine=fcn-resnet101.engine --optShapes=input:1x3x1026x1282</summary>

```log
&&&& RUNNING TensorRT.trtexec [TensorRT v100900] [b34] # trtexec --onnx=fcn-resnet101.onnx --saveEngine=fcn-resnet101.engine --optShapes=input:1x3x1026x1282
[04/08/2025-23:13:59] [W] optShapes is being broadcasted to minShapes for tensor input
[04/08/2025-23:13:59] [W] optShapes is being broadcasted to maxShapes for tensor input
[04/08/2025-23:13:59] [I] TF32 is enabled by default. Add --noTF32 flag to further improve accuracy with some performance cost.
[04/08/2025-23:13:59] [I] === Model Options ===
[04/08/2025-23:13:59] [I] Format: ONNX
[04/08/2025-23:13:59] [I] Model: fcn-resnet101.onnx
[04/08/2025-23:13:59] [I] Output:
[04/08/2025-23:13:59] [I] === Build Options ===
[04/08/2025-23:13:59] [I] Memory Pools: workspace: default, dlaSRAM: default, dlaLocalDRAM: default, dlaGlobalDRAM: default, tacticSharedMem: default
[04/08/2025-23:13:59] [I] avgTiming: 8
[04/08/2025-23:13:59] [I] Precision: FP32
[04/08/2025-23:13:59] [I] LayerPrecisions:
[04/08/2025-23:13:59] [I] Layer Device Types:
[04/08/2025-23:13:59] [I] Calibration:
[04/08/2025-23:13:59] [I] Refit: Disabled
[04/08/2025-23:13:59] [I] Strip weights: Disabled
[04/08/2025-23:13:59] [I] Version Compatible: Disabled
[04/08/2025-23:13:59] [I] ONNX Plugin InstanceNorm: Disabled
[04/08/2025-23:13:59] [I] TensorRT runtime: full
[04/08/2025-23:13:59] [I] Lean DLL Path:
[04/08/2025-23:13:59] [I] Tempfile Controls: { in_memory: allow, temporary: allow }
[04/08/2025-23:13:59] [I] Exclude Lean Runtime: Disabled
[04/08/2025-23:13:59] [I] Sparsity: Disabled
[04/08/2025-23:13:59] [I] Safe mode: Disabled
[04/08/2025-23:13:59] [I] Build DLA standalone loadable: Disabled
[04/08/2025-23:13:59] [I] Allow GPU fallback for DLA: Disabled
[04/08/2025-23:13:59] [I] DirectIO mode: Disabled
[04/08/2025-23:13:59] [I] Restricted mode: Disabled
[04/08/2025-23:13:59] [I] Skip inference: Disabled
[04/08/2025-23:13:59] [I] Save engine: fcn-resnet101.engine
[04/08/2025-23:13:59] [I] Load engine:
[04/08/2025-23:13:59] [I] Profiling verbosity: 0
[04/08/2025-23:13:59] [I] Tactic sources: Using default tactic sources
[04/08/2025-23:13:59] [I] timingCacheMode: local
[04/08/2025-23:13:59] [I] timingCacheFile:
[04/08/2025-23:13:59] [I] Enable Compilation Cache: Enabled
[04/08/2025-23:13:59] [I] Enable Monitor Memory: Disabled
[04/08/2025-23:13:59] [I] errorOnTimingCacheMiss: Disabled
[04/08/2025-23:13:59] [I] Preview Features: Use default preview flags.
[04/08/2025-23:13:59] [I] MaxAuxStreams: -1
[04/08/2025-23:13:59] [I] BuilderOptimizationLevel: -1
[04/08/2025-23:13:59] [I] MaxTactics: -1
[04/08/2025-23:13:59] [I] Calibration Profile Index: 0
[04/08/2025-23:13:59] [I] Weight Streaming: Disabled
[04/08/2025-23:13:59] [I] Runtime Platform: Same As Build
[04/08/2025-23:13:59] [I] Debug Tensors:
[04/08/2025-23:13:59] [I] Input(s)s format: fp32:CHW
[04/08/2025-23:13:59] [I] Output(s)s format: fp32:CHW
[04/08/2025-23:13:59] [I] Input build shape (profile 0): input=1x3x1026x1282+1x3x1026x1282+1x3x1026x1282
[04/08/2025-23:13:59] [I] Input calibration shapes: model
[04/08/2025-23:13:59] [I] === System Options ===
[04/08/2025-23:13:59] [I] Device: 0
[04/08/2025-23:13:59] [I] DLACore:
[04/08/2025-23:13:59] [I] Plugins:
[04/08/2025-23:13:59] [I] setPluginsToSerialize:
[04/08/2025-23:13:59] [I] dynamicPlugins:
[04/08/2025-23:13:59] [I] ignoreParsedPluginLibs: 0
[04/08/2025-23:13:59] [I]
[04/08/2025-23:13:59] [I] === Inference Options ===
[04/08/2025-23:13:59] [I] Batch: Explicit
[04/08/2025-23:13:59] [I] Input inference shape : input=1x3x1026x1282
[04/08/2025-23:13:59] [I] Iterations: 10
[04/08/2025-23:13:59] [I] Duration: 3s (+ 200ms warm up)
[04/08/2025-23:13:59] [I] Sleep time: 0ms
[04/08/2025-23:13:59] [I] Idle time: 0ms
[04/08/2025-23:13:59] [I] Inference Streams: 1
[04/08/2025-23:13:59] [I] ExposeDMA: Disabled
[04/08/2025-23:13:59] [I] Data transfers: Enabled
[04/08/2025-23:13:59] [I] Spin-wait: Disabled
[04/08/2025-23:13:59] [I] Multithreading: Disabled
[04/08/2025-23:13:59] [I] CUDA Graph: Disabled
[04/08/2025-23:13:59] [I] Separate profiling: Disabled
[04/08/2025-23:13:59] [I] Time Deserialize: Disabled
[04/08/2025-23:13:59] [I] Time Refit: Disabled
[04/08/2025-23:13:59] [I] NVTX verbosity: 0
[04/08/2025-23:13:59] [I] Persistent Cache Ratio: 0
[04/08/2025-23:13:59] [I] Optimization Profile Index: 0
[04/08/2025-23:13:59] [I] Weight Streaming Budget: 100.000000%
[04/08/2025-23:13:59] [I] Inputs:
[04/08/2025-23:13:59] [I] Debug Tensor Save Destinations:
[04/08/2025-23:13:59] [I] === Reporting Options ===
[04/08/2025-23:13:59] [I] Verbose: Disabled
[04/08/2025-23:13:59] [I] Averages: 10 inferences
[04/08/2025-23:13:59] [I] Percentiles: 90,95,99
[04/08/2025-23:13:59] [I] Dump refittable layers:Disabled
[04/08/2025-23:13:59] [I] Dump output: Disabled
[04/08/2025-23:13:59] [I] Profile: Disabled
[04/08/2025-23:13:59] [I] Export timing to JSON file:
[04/08/2025-23:13:59] [I] Export output to JSON file:
[04/08/2025-23:13:59] [I] Export profile to JSON file:
[04/08/2025-23:13:59] [I]
[04/08/2025-23:13:59] [I] === Device Information ===
[04/08/2025-23:13:59] [I] Available Devices:
[04/08/2025-23:13:59] [I]   Device 0: "NVIDIA GeForce RTX 3070" UUID: GPU-84a8068b-81a5-7df1-8d22-a523bb848827
[04/08/2025-23:14:00] [I] Selected Device: NVIDIA GeForce RTX 3070
[04/08/2025-23:14:00] [I] Selected Device ID: 0
[04/08/2025-23:14:00] [I] Selected Device UUID: GPU-84a8068b-81a5-7df1-8d22-a523bb848827
[04/08/2025-23:14:00] [I] Compute Capability: 8.6
[04/08/2025-23:14:00] [I] SMs: 46
[04/08/2025-23:14:00] [I] Device Global Memory: 7817 MiB
[04/08/2025-23:14:00] [I] Shared Memory per SM: 100 KiB
[04/08/2025-23:14:00] [I] Memory Bus Width: 256 bits (ECC disabled)
[04/08/2025-23:14:00] [I] Application Compute Clock Rate: 1.845 GHz
[04/08/2025-23:14:00] [I] Application Memory Clock Rate: 7.001 GHz
[04/08/2025-23:14:00] [I]
[04/08/2025-23:14:00] [I] Note: The application clock rates do not reflect the actual clock rates that the GPU is currently running at.
[04/08/2025-23:14:00] [I]
[04/08/2025-23:14:00] [I] TensorRT version: 10.9.0
[04/08/2025-23:14:00] [I] Loading standard plugins
[04/08/2025-23:14:00] [I] [TRT] [MemUsageChange] Init CUDA: CPU +2, GPU +0, now: CPU 26, GPU 1519 (MiB)
[04/08/2025-23:14:02] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +2648, GPU +414, now: CPU 2875, GPU 1933 (MiB)
[04/08/2025-23:14:02] [I] Start parsing network model.
[04/08/2025-23:14:02] [I] [TRT] ----------------------------------------------------------------
[04/08/2025-23:14:02] [I] [TRT] Input filename:   fcn-resnet101.onnx
[04/08/2025-23:14:02] [I] [TRT] ONNX IR version:  0.0.6
[04/08/2025-23:14:02] [I] [TRT] Opset version:    12
[04/08/2025-23:14:02] [I] [TRT] Producer name:    pytorch
[04/08/2025-23:14:02] [I] [TRT] Producer version: 1.8
[04/08/2025-23:14:02] [I] [TRT] Domain:
[04/08/2025-23:14:02] [I] [TRT] Model version:    0
[04/08/2025-23:14:02] [I] [TRT] Doc string:
[04/08/2025-23:14:02] [I] [TRT] ----------------------------------------------------------------
[04/08/2025-23:14:02] [W] [TRT] ModelImporter.cpp:804: Make sure output output has Int64 binding.
[04/08/2025-23:14:02] [I] Finished parsing network model. Parse time: 0.274727
[04/08/2025-23:14:02] [I] Set shape of input tensor input for optimization profile 0 to: MIN=1x3x1026x1282 OPT=1x3x1026x1282 MAX=1x3x1026x1282
[04/08/2025-23:14:02] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
[04/08/2025-23:15:45] [I] [TRT] Compiler backend is used during engine build.
[04/08/2025-23:15:46] [I] [TRT] Detected 1 inputs and 1 output network tensors.
[04/08/2025-23:15:47] [I] [TRT] Total Host Persistent Memory: 729520 bytes
[04/08/2025-23:15:47] [I] [TRT] Total Device Persistent Memory: 124928 bytes
[04/08/2025-23:15:47] [I] [TRT] Max Scratch Memory: 42735104 bytes
[04/08/2025-23:15:47] [I] [TRT] [BlockAssignment] Started assigning block shifts. This will take 197 steps to complete.
[04/08/2025-23:15:47] [I] [TRT] [BlockAssignment] Algorithm ShiftNTopDown took 3.03527ms to assign 5 blocks to 197 nodes requiring 467884544 bytes.
[04/08/2025-23:15:47] [I] [TRT] Total Activation Memory: 467884032 bytes
[04/08/2025-23:15:47] [I] [TRT] Total Weights Memory: 238797908 bytes
[04/08/2025-23:15:47] [I] [TRT] Compiler backend is used during engine execution.
[04/08/2025-23:15:47] [I] [TRT] Engine generation completed in 104.604 seconds.
[04/08/2025-23:15:47] [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 64 MiB, GPU 1392 MiB
[04/08/2025-23:15:47] [I] Engine built in 104.774 sec.
[04/08/2025-23:15:47] [I] Created engine with size: 231.019 MiB
[04/08/2025-23:15:48] [I] [TRT] Loaded engine size: 231 MiB
[04/08/2025-23:15:48] [I] Engine deserialized in 0.21022 sec.
[04/08/2025-23:15:48] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +447, now: CPU 1, GPU 674 (MiB)
[04/08/2025-23:15:48] [I] Setting persistentCacheLimit to 0 bytes.
[04/08/2025-23:15:48] [I] Created execution context with device memory size: 446.209 MiB
[04/08/2025-23:15:48] [I] Using random values for input input
[04/08/2025-23:15:48] [I] Input binding for input with dimensions 1x3x1026x1282 is created.
[04/08/2025-23:15:48] [I] Output binding for output with dimensions 1x1x1026x1282 is created.
[04/08/2025-23:15:48] [I] Starting inference
[04/08/2025-23:15:51] [I] Warmup completed 2 queries over 200 ms
[04/08/2025-23:15:51] [I] Timing trace has 29 queries over 3.34555 s
[04/08/2025-23:15:51] [I]
[04/08/2025-23:15:51] [I] === Trace details ===
[04/08/2025-23:15:51] [I] Trace averages of 10 runs:
[04/08/2025-23:15:51] [I] Average on 10 runs - GPU latency: 111.218 ms - Host latency: 112.617 ms (enqueue 0.685264 ms)
[04/08/2025-23:15:51] [I] Average on 10 runs - GPU latency: 111.351 ms - Host latency: 112.752 ms (enqueue 0.753516 ms)
[04/08/2025-23:15:51] [I]
[04/08/2025-23:15:51] [I] === Performance summary ===
[04/08/2025-23:15:51] [I] Throughput: 8.66824 qps
[04/08/2025-23:15:51] [I] Latency: min = 110.946 ms, max = 115.703 ms, mean = 112.938 ms, median = 112.567 ms, percentile(90%) = 114.427 ms, percentile(95%) = 115.415 ms, percentile(99%) = 115.703 ms
[04/08/2025-23:15:51] [I] Enqueue Time: min = 0.495506 ms, max = 0.953369 ms, mean = 0.733638 ms, median = 0.734863 ms, percentile(90%) = 0.802246 ms, percentile(95%) = 0.835693 ms, percentile(99%) = 0.953369 ms
[04/08/2025-23:15:51] [I] H2D Latency: min = 0.612305 ms, max = 0.961426 ms, mean = 0.771373 ms, median = 0.769043 ms, percentile(90%) = 0.873108 ms, percentile(95%) = 0.873779 ms, percentile(99%) = 0.961426 ms
[04/08/2025-23:15:51] [I] GPU Compute Time: min = 109.522 ms, max = 114.185 ms, mean = 111.557 ms, median = 111.208 ms, percentile(90%) = 113.02 ms, percentile(95%) = 114.115 ms, percentile(99%) = 114.185 ms
[04/08/2025-23:15:51] [I] D2H Latency: min = 0.548096 ms, max = 0.763855 ms, mean = 0.61034 ms, median = 0.609924 ms, percentile(90%) = 0.709229 ms, percentile(95%) = 0.71228 ms, percentile(99%) = 0.763855 ms
[04/08/2025-23:15:51] [I] Total Host Walltime: 3.34555 s
[04/08/2025-23:15:51] [I] Total GPU Compute Time: 3.23514 s
[04/08/2025-23:15:51] [W] * GPU compute time is unstable, with coefficient of variance = 1.08662%.
[04/08/2025-23:15:51] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[04/08/2025-23:15:51] [I] Explanations of the performance metrics are printed in the verbose logs.
[04/08/2025-23:15:51] [I]
&&&& PASSED TensorRT.trtexec [TensorRT v100900] [b34] # trtexec --onnx=fcn-resnet101.onnx --saveEngine=fcn-resnet101.engine --optShapes=input:1x3x1026x1282
```

</details>

```bash
TensorRT/quickstart/
└── SemanticSegmentation/
    ├── export.py
    ├── fcn-resnet101.engine
    ├── fcn-resnet101.onnx
    ├── input.ppm
    ├── Makefile
    ├── tutorial-runtime.cpp
    └── tutorial-runtime.ipynb
```

move an image and the models to the `models` folder:

```bash
start/
├── images
│   └── input.ppm
├── models
│   ├── fcn-resnet101.engine
│   └── fcn-resnet101.onnx
├── pyproject.toml
├── README.md
├── src/
├── TensorRT/
├── tests/
└── uv.lock
```

