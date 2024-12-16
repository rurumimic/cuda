# Vector Add

## Build

```bash
make
nvcc -o vector_add  main.cu
```

## Run

```bash
./vector_add

Vector length: 50000
Copy: host to device
CUDA kernel: 196 blocks x 256 threads
Copy: device to host
OK
End
```

