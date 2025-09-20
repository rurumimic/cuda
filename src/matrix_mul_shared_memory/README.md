# Metrix Mul Shared Memory

## Build & Run

recommend: justfile + cmake + ninja

### Just

```bash
just debug run
```

### Run

```bash
Matrix multiplication: C = A * B
A [32 x 192]
B [192 x 32]
C [32 x 32]

Allocate Host memory
Initialize Host matrix
Matrix A[0][0] = 3.860000

Matrix mul on Host
Host matrix mul duration: 841 μs

Allocate Device memory
Copy: Host to Device
Copy duration: 35 µs

Block Dim: (32, 32, 1)
Grid Dim: (1, 1, 1)
Launch matrix_mul kernel
Kernel execution duration: 273 μs

Launch matrix_mul_shared kernel
Kernel execution duration: 246 μs

Copy: Device to Host
Copy duration: 12 µs

Verify results
Result verification: OK

Free Device memory

Free Host memory
```

