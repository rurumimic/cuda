# Hello CUDA with C++

## Build

```bash
make

clang++  -c main.cpp -o main.o -I/usr/local/cuda/include
nvcc -c kernel.cu -o kernel.o
clang++  -o hello_cuda main.o kernel.o -L/usr/local/cuda/lib64 -lcudart
```

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 12
```

## Run

```bash
./hello_cuda
```

