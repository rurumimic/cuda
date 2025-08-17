# Vector Add

## Build & Run

recommend: justfile + cmake + ninja

### Just

```bash
just
```

```bash
just run
# or
./build/debug/vector_add
```

### Make

```bash
make
# or
nvcc -o vector_add src/main.cu
```

```bash
./vector_add
```

### CMake


```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 12
```

```bash
./build/vector_add
```

## Run

```bash
./vector_add
```

### Steps

#### Vector Length

```bash
Vector length: 500000
```

#### Init Vectors on Host

```bash
Allocate Host memory
Initialize Host vectors
```

#### Vector Add on Host

```bash
Vector add on Host
Host vector add duration: 1446 µs
```

#### Init Device Memory

```bash
-----Device memory-----
free:  7034699776 bytes
total: 8197111808 bytes

Allocate Device memory
Cleaning Device memory

-----Device memory-----
free:  7028408320 bytes
total: 8197111808 bytes
```

#### Copy Data from Host to Device

```bash
Copy: Host to Device
Copy duration: 406 µs
```

#### Launch Kernel

```bash
CUDA kernel: 1954 blocks x 256 threads
Launch vector_add kernel
Kernel execution duration: 78 µs
```

#### Copy Data from Device to Host

```bash
Copy: Device to Host
Copy duration: 1092 µs
```

#### Verify Results

```bash
Verify results
Result verification: OK
```

#### Free Device Memory

```bash
-----Device memory-----
free:  7028408320 bytes
total: 8197111808 bytes

Free Device memory

-----Device memory-----
free:  7034699776 bytes
total: 8197111808 bytes
```

#### Free Host Memory

```bash
Free Host memory
```

---

## Figure

```text

               CPU (Host)
─────────────────────────────────────────────────────
 1. Allocate Memory (Host)
 ┌─────────────────────────────────────────────────┐
 │ h_a[LENGTH]                                     │
 │ h_b[LENGTH]                                     │
 │ h_c[LENGTH]                                     │
 └─────────────────────────────────────────────────┘
          ↓
 2. Allocate Memory (Device)
 ┌─────────────────────────────────────────────────┐
 │ d_a[LENGTH] (on GPU)                            │
 │ d_b[LENGTH] (on GPU)                            │
 │ d_c[LENGTH] (on GPU)                            │
 └─────────────────────────────────────────────────┘
          ↓
 3. Copy Data (Host → Device)
 ┌─────────────────────────────────────────────────┐
 │ memcpy(h_a → d_a)                               │
 │ memcpy(h_b → d_b)                               │
 └─────────────────────────────────────────────────┘
          ↓
 4. Kernel Launch
 ┌────────────────────────────────────────────────────────────────────┐
 │ CUDA Grid Configuration                                            │
 │ ┌──────────────────┬──────────────────┬──────────────────┬───┐     │
 │ │ Block 0          │ Block 1          │ Block 2          │...│     │
 │ │ ┌───┬───┬───┬─   │ ┌───┬───┬───┬─   │ ┌───┬───┬───┬─   │   │     │
 │ │ │ T │ T │ T │    │ │ T │ T │ T │    │ │ T │ T │ T │    │   │     │
 │ │ │ h │ h │ h │    │ │ h │ h │ h │    │ │ h │ h │ h │    │   │     │
 │ │ │ r │ r │ r │    │ │ r │ r │ r │    │ │ r │ r │ r │    │   │     │
 │ │ │ e │ e │ e │    │ │ e │ e │ e │    │ │ e │ e │ e │    │   │     │
 │ │ │ a │ a │ a │    │ │ a │ a │ a │    │ │ a │ a │ a │    │   │     │
 │ │ │ d │ d │ d │    │ │ d │ d │ d │    │ │ d │ d │ d │    │   │     │
 │ │ └───┴───┴───┴─   │ └───┴───┴───┴─   │ └───┴───┴───┴─   │   │     │
 │ └──────────────────┴──────────────────┴──────────────────┴───┘     │
 │ Each thread computes: c[i] = a[i] + b[i]                           │
 └────────────────────────────────────────────────────────────────────┘
          ↓
 5. Copy Data (Device → Host)
 ┌─────────────────────────────────────────────────┐
 │ memcpy(d_c → h_c)                               │
 └─────────────────────────────────────────────────┘
          ↓
 6. Result Verification
 ┌─────────────────────────────────────────────────┐
 │ Check if h_c[i] == h_a[i] + h_b[i]              │
 └─────────────────────────────────────────────────┘
          ↓
 7. Free Memory
 ┌─────────────────────────────────────────────────┐
 │ Free d_a, d_b, d_c on GPU                       │
 │ Free h_a, h_b, h_c on CPU                       │
 └─────────────────────────────────────────────────┘
```

