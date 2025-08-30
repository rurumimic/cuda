# Metrix Mul

## Build & Run

recommend: justfile + cmake + ninja

### Just

```bash
just debug run
```

### Run

```bash
Matrix multiplication: C = A * B
A [1024 x 1024]
B [1024 x 2048]
C [1024 x 2048]

Allocate Host memory
Initialize Host matrix
Matrix A[0][0] = 3.860000

Matrix mul on Host
Host matrix mul duration: 8 s

Allocate Device memory
Copy: Host to Device
Copy duration: 1287 µs

Block Dim: (32, 32, 1)
Grid Dim: (64, 32, 1)
Launch matrix_mul kernel
Kernel execution duration: 47 ms

Copy: Device to Host
Copy duration: 4535 µs

Verify results
Result verification: OK

Free Device memory

Free Host memory
```

