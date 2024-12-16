# Vector Add

## Files

```bash
tree -afhp

vector_Add/
├── [rw- 7.3K]  ./.clang-format
├── [rw-   12]  ./.gitignore
├── [rw- 3.4K]  ./main.cu
├── [rw-  135]  ./Makefile
├── [rw-   78]  ./README.md
└── [rwx 982K]  ./vector_add

1 directory, 6 files
```

## Build

```bash
make
nvcc -o vector_add  main.cu
```

## Run

```bash
./vector_add
```

Output:

```bash
Vector length: 50000
Copy: host to device
CUDA kernel: 196 blocks x 256 threads
Copy: device to host
OK
End
```

