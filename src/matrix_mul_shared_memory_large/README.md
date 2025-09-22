# Metrix Mul Shared Memory Large

## Build & Run

recommend: justfile + cmake + ninja

### Just

```bash
just debug run
```

### Run

- Naive: 34ms
- Shared: 49ms

```bash
Matrix multiplication: C = A * B
A [1024 x 1024]
B [1024 x 2048]
C [1024 x 2048]

Allocate Host memory
Initialize Host matrix
Matrix A[0][0] = 3.860000

Matrix mul on Host
Host matrix mul duration: 7701 ms

Allocate Device memory
Copy: Host to Device
Copy duration: 1 ms

Block Dim: (16, 16, 1)
Grid Dim: (128, 64, 1)
Launch matrix_mul kernel
Kernel execution duration: 34 ms

Launch matrix_mul_shared kernel
Kernel execution duration: 49 ms

Copy: Device to Host
Copy duration: 4 µs

Verify results
Result verification: OK

Free Device memory

Free Host memory
```

