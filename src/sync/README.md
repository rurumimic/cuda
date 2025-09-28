# Synchronization

## Build & Run

recommend: justfile + cmake + ninja

### Just

```bash
just debug run
```

### Run

```bash
Synchroization
Block Dim: (64, 1, 1)
Grid Dim: (2, 1, 1)
Launch sync_block kernel
Thread 65 in block 1, master thread is 64
Thread 66 in block 1, master thread is 64
Thread 67 in block 1, master thread is 64
Thread 68 in block 1, master thread is 64
Thread 69 in block 1, master thread is 64
Thread 70 in block 1, master thread is 64
Thread 71 in block 1, master thread is 64
Thread 72 in block 1, master thread is 64
Thread 73 in block 1, master thread is 64
Thread 74 in block 1, master thread is 64
Thread 75 in block 1, master thread is 64
Thread 76 in block 1, master thread is 64
Thread 77 in block 1, master thread is 64
Thread 78 in block 1, master thread is 64
Thread 79 in block 1, master thread is 64
Thread 80 in block 1, master thread is 64
Thread 81 in block 1, master thread is 64
Thread 82 in block 1, master thread is 64
Thread 83 in block 1, master thread is 64
Thread 84 in block 1, master thread is 64
Thread 85 in block 1, master thread is 64
Thread 86 in block 1, master thread is 64
Thread 87 in block 1, master thread is 64
Thread 88 in block 1, master thread is 64
Thread 89 in block 1, master thread is 64
Thread 90 in block 1, master thread is 64
Thread 91 in block 1, master thread is 64
Thread 92 in block 1, master thread is 64
Thread 93 in block 1, master thread is 64
Thread 94 in block 1, master thread is 64
Thread 95 in block 1, master thread is 64
Thread 1 in block 0, master thread is 0
Thread 2 in block 0, master thread is 0
Thread 3 in block 0, master thread is 0
Thread 4 in block 0, master thread is 0
Thread 5 in block 0, master thread is 0
Thread 6 in block 0, master thread is 0
Thread 7 in block 0, master thread is 0
Thread 8 in block 0, master thread is 0
Thread 9 in block 0, master thread is 0
Thread 10 in block 0, master thread is 0
Thread 11 in block 0, master thread is 0
Thread 12 in block 0, master thread is 0
Thread 13 in block 0, master thread is 0
Thread 14 in block 0, master thread is 0
Thread 15 in block 0, master thread is 0
Thread 16 in block 0, master thread is 0
Thread 17 in block 0, master thread is 0
Thread 18 in block 0, master thread is 0
Thread 19 in block 0, master thread is 0
Thread 20 in block 0, master thread is 0
Thread 21 in block 0, master thread is 0
Thread 22 in block 0, master thread is 0
Thread 23 in block 0, master thread is 0
Thread 24 in block 0, master thread is 0
Thread 25 in block 0, master thread is 0
Thread 26 in block 0, master thread is 0
Thread 27 in block 0, master thread is 0
Thread 28 in block 0, master thread is 0
Thread 29 in block 0, master thread is 0
Thread 30 in block 0, master thread is 0
Thread 31 in block 0, master thread is 0
Thread 96 in block 1, master thread is 64
Thread 97 in block 1, master thread is 64
Thread 98 in block 1, master thread is 64
Thread 99 in block 1, master thread is 64
Thread 100 in block 1, master thread is 64
Thread 101 in block 1, master thread is 64
Thread 102 in block 1, master thread is 64
Thread 103 in block 1, master thread is 64
Thread 104 in block 1, master thread is 64
Thread 105 in block 1, master thread is 64
Thread 106 in block 1, master thread is 64
Thread 107 in block 1, master thread is 64
Thread 108 in block 1, master thread is 64
Thread 109 in block 1, master thread is 64
Thread 110 in block 1, master thread is 64
Thread 111 in block 1, master thread is 64
Thread 112 in block 1, master thread is 64
Thread 113 in block 1, master thread is 64
Thread 114 in block 1, master thread is 64
Thread 115 in block 1, master thread is 64
Thread 116 in block 1, master thread is 64
Thread 117 in block 1, master thread is 64
Thread 118 in block 1, master thread is 64
Thread 119 in block 1, master thread is 64
Thread 120 in block 1, master thread is 64
Thread 121 in block 1, master thread is 64
Thread 122 in block 1, master thread is 64
Thread 123 in block 1, master thread is 64
Thread 124 in block 1, master thread is 64
Thread 125 in block 1, master thread is 64
Thread 126 in block 1, master thread is 64
Thread 127 in block 1, master thread is 64
Thread 32 in block 0, master thread is 0
Thread 33 in block 0, master thread is 0
Thread 34 in block 0, master thread is 0
Thread 35 in block 0, master thread is 0
Thread 36 in block 0, master thread is 0
Thread 37 in block 0, master thread is 0
Thread 38 in block 0, master thread is 0
Thread 39 in block 0, master thread is 0
Thread 40 in block 0, master thread is 0
Thread 41 in block 0, master thread is 0
Thread 42 in block 0, master thread is 0
Thread 43 in block 0, master thread is 0
Thread 44 in block 0, master thread is 0
Thread 45 in block 0, master thread is 0
Thread 46 in block 0, master thread is 0
Thread 47 in block 0, master thread is 0
Thread 48 in block 0, master thread is 0
Thread 49 in block 0, master thread is 0
Thread 50 in block 0, master thread is 0
Thread 51 in block 0, master thread is 0
Thread 52 in block 0, master thread is 0
Thread 53 in block 0, master thread is 0
Thread 54 in block 0, master thread is 0
Thread 55 in block 0, master thread is 0
Thread 56 in block 0, master thread is 0
Thread 57 in block 0, master thread is 0
Thread 58 in block 0, master thread is 0
Thread 59 in block 0, master thread is 0
Thread 60 in block 0, master thread is 0
Thread 61 in block 0, master thread is 0
Thread 62 in block 0, master thread is 0
Thread 63 in block 0, master thread is 0
Thread 64 in block 1, master thread is 64
Thread 0 in block 0, master thread is 0
count with block synchronization: 128
Kernel execution duration: 5730 μs
Launch sync_warp kernel
Thread 1 in warp 0, master thread is 0
Thread 2 in warp 0, master thread is 0
Thread 3 in warp 0, master thread is 0
Thread 4 in warp 0, master thread is 0
Thread 5 in warp 0, master thread is 0
Thread 6 in warp 0, master thread is 0
Thread 7 in warp 0, master thread is 0
Thread 8 in warp 0, master thread is 0
Thread 9 in warp 0, master thread is 0
Thread 10 in warp 0, master thread is 0
Thread 11 in warp 0, master thread is 0
Thread 12 in warp 0, master thread is 0
Thread 13 in warp 0, master thread is 0
Thread 14 in warp 0, master thread is 0
Thread 15 in warp 0, master thread is 0
Thread 16 in warp 0, master thread is 0
Thread 17 in warp 0, master thread is 0
Thread 18 in warp 0, master thread is 0
Thread 19 in warp 0, master thread is 0
Thread 20 in warp 0, master thread is 0
Thread 21 in warp 0, master thread is 0
Thread 22 in warp 0, master thread is 0
Thread 23 in warp 0, master thread is 0
Thread 24 in warp 0, master thread is 0
Thread 25 in warp 0, master thread is 0
Thread 26 in warp 0, master thread is 0
Thread 27 in warp 0, master thread is 0
Thread 28 in warp 0, master thread is 0
Thread 29 in warp 0, master thread is 0
Thread 30 in warp 0, master thread is 0
Thread 31 in warp 0, master thread is 0
Thread 32 in warp 1, master thread is 32
Thread 33 in warp 1, master thread is 32
Thread 34 in warp 1, master thread is 32
Thread 35 in warp 1, master thread is 32
Thread 36 in warp 1, master thread is 32
Thread 37 in warp 1, master thread is 32
Thread 38 in warp 1, master thread is 32
Thread 39 in warp 1, master thread is 32
Thread 40 in warp 1, master thread is 32
Thread 41 in warp 1, master thread is 32
Thread 42 in warp 1, master thread is 32
Thread 43 in warp 1, master thread is 32
Thread 44 in warp 1, master thread is 32
Thread 45 in warp 1, master thread is 32
Thread 46 in warp 1, master thread is 32
Thread 47 in warp 1, master thread is 32
Thread 48 in warp 1, master thread is 32
Thread 49 in warp 1, master thread is 32
Thread 50 in warp 1, master thread is 32
Thread 51 in warp 1, master thread is 32
Thread 52 in warp 1, master thread is 32
Thread 53 in warp 1, master thread is 32
Thread 54 in warp 1, master thread is 32
Thread 55 in warp 1, master thread is 32
Thread 56 in warp 1, master thread is 32
Thread 57 in warp 1, master thread is 32
Thread 58 in warp 1, master thread is 32
Thread 59 in warp 1, master thread is 32
Thread 60 in warp 1, master thread is 32
Thread 61 in warp 1, master thread is 32
Thread 62 in warp 1, master thread is 32
Thread 63 in warp 1, master thread is 32
Thread 0 in warp 0, master thread is 0
count with warp synchronization: 64
Kernel execution duration: 337 μs
Launch sync_even kernel
even: 64, odd: 64
Kernel execution duration: 24 μs
```

