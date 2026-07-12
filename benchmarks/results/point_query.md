# Point Query Benchmark Results

Append rows or JSONL records emitted by `svo_point_query_benchmark` when intentionally recording a performance snapshot.

Example:

```bash
./build-cuda/svo_point_query_benchmark --scene sparse_random --grid-size 64 --branching both --seed 20260712 --density 0.035 --iterations 20 --count 1048576 --profile --jsonl benchmarks/results/point_query.jsonl
```

| Date | GPU | CPU | Build | Scene | Points | CPU Query ms | H2D Transfer ms | CUDA Kernel ms | D2H Transfer ms | Total CUDA ms | Speedup | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
