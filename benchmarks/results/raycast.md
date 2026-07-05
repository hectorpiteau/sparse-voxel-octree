# Raycast Benchmark Results

Append rows emitted by `svo_raycast_benchmark` when intentionally recording a performance snapshot.

| Date | GPU | Build | Max Depth | Nodes | Leaves | Rays | Hit Rate | H2D Transfer ms | CUDA Kernel ms | Rays/s | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-07-05 | NVIDIA GeForce RTX 3090 | local CUDA 12.8 | 6 | 1563 | 1152 | 1048576 | 0.23524 | 2.36285 | 7.89146 | 1.32875e+08 | Initial correctness-first stack traversal baseline |
