# Render Benchmark Results

Append rows or JSONL records emitted by `svo_render_benchmark` when intentionally recording a performance snapshot.

Examples:

```bash
./build-cuda/svo_render_benchmark --operation forward --scene sparse_random --grid-size 64 --branching both --seed 20260712 --density 0.035 --iterations 20 --count 262144 --profile --jsonl benchmarks/results/render.jsonl
./build-cuda/svo_render_benchmark --operation backward --scene sparse_random --grid-size 64 --branching both --seed 20260712 --density 0.035 --iterations 20 --count 262144 --profile --jsonl benchmarks/results/render.jsonl
```

| Date | GPU | Build | Operation | Max Depth | Nodes | Leaves | Pixels | CPU Reference ms | H2D Transfer ms | Forward Kernel ms | Backward Kernel ms | D2H Transfer ms | Pixels/s | FPS | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
