# Raycast Benchmark Results

Append rows or JSONL records emitted by `svo_raycast_benchmark` when intentionally recording a performance snapshot.

Example:

```bash
./build-cuda/svo_raycast_benchmark --scene sparse_random --grid-size 64 --branching both --seed 20260712 --density 0.035 --iterations 20 --count 1048576 --profile --jsonl benchmarks/results/raycast.jsonl
```

| Date | GPU | Build | Max Depth | Nodes | Leaves | Rays | Hit Rate | H2D Transfer ms | CUDA Kernel ms | Rays/s | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-07-05 | NVIDIA GeForce RTX 3090 | local CUDA 12.8 | 6 | 1563 | 1152 | 1048576 | 0.23524 | 2.36285 | 7.89146 | 1.32875e+08 | Initial correctness-first stack traversal baseline |
