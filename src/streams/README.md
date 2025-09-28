# Streams + Events

## Build & Run

recommend: justfile + cmake + ninja

### Just

```bash
just debug run
```

### Run

```bash
Multi streams

Launch kernel
Memcpy H2D duration: 23 ms
Kernel launch duration: 470 ms
Kernel execution duration: 23 ms
Single Stream duration: 517 ms

Launch Multi Streams kernel
Multi streams duration: 465 ms
Stream 0 duration: 128.026917 ms
Stream 1 duration: 240.854919 ms
Stream 2 duration: 353.537628 ms
Stream 3 duration: 465.855835 ms
```

