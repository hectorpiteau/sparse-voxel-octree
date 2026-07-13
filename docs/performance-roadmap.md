# Performance Roadmap

This document is the decision record for Milestone 17.5. It defines what the
next optimization work is allowed to measure, change, and defer.

The current decision is:

- Milestone 18 profiles the current implementation and applies low-risk
  optimizations only.
- Milestone 19 has started with measured `Wide4` local DDA/HDDA traversal for
  raycast, forward render, and backward render.
- Dense bricks, compact descriptors, block-local addressing, contour data,
  compressed attributes, and serialization stay deferred unless profiling proves
  one of them is the immediate bottleneck.

## Benchmark Scenes

All benchmark scenes must be deterministic. Recorded benchmark rows should
include `scene`, `scene_version`, `grid_size`, `branching`, and any seed or
density used by the generator.

Default scene version: `1`.

| Scene | Purpose | Default settings |
| --- | --- | --- |
| `empty` | Measures launch/setup overhead and all-miss traversal. | grid sizes `64`, `128`, `256` |
| `single_voxel` | Measures trivial hit/miss behavior and overhead on tiny topology. | one centered occupied voxel |
| `dense_cube` | Measures dense local traversal and compositing cost. | centered cube occupying about one quarter of each axis |
| `sparse_random` | Measures irregular sparse occupancy without changing between runs. | fixed seed `20260712`, density printed in output |
| `sphere` | Matches the current render/viewer intuition. | filled sparse sphere, smooth payloads |
| `shell` | Stresses thin geometry, boundaries, and near-miss rays. | thin sphere shell |
| `viewer_large` | Measures interactive rendering behavior. | sphere-like scene at grid sizes `64`, `128`, `256` |

`sparse_random` should be generated from `(scene_version, grid_size, density,
seed)`. A saved coordinate fixture is not required for v1; if the generation
algorithm changes, increment `scene_version` or add a canonical fixture.

## Metrics

Milestone 18 should standardize benchmark output around human-readable terminal
summaries plus JSONL rows for machine comparison.

Current C++ benchmark entry points:

```bash
./build-cuda/svo_point_query_benchmark --scene sparse_random --grid-size 64 --branching both --seed 20260712 --density 0.035 --iterations 20 --count 1048576 --jsonl benchmarks/results/local.jsonl
./build-cuda/svo_raycast_benchmark --scene sparse_random --grid-size 64 --branching both --seed 20260712 --density 0.035 --iterations 20 --count 1048576 --jsonl benchmarks/results/local.jsonl
./build-cuda/svo_render_benchmark --operation both --scene sparse_random --grid-size 64 --branching both --render-strategy direct --seed 20260712 --density 0.035 --iterations 20 --count 262144 --jsonl benchmarks/results/local.jsonl
./build-cuda/svo_render_benchmark --operation both --scene sparse_random --grid-size 64 --branching both --render-strategy intervals --seed 20260712 --density 0.035 --iterations 20 --count 262144 --jsonl benchmarks/results/local.jsonl
```

Common flags:

- `--scene`: `empty`, `single_voxel`, `dense_cube`, `sphere`, or `sparse_random`.
- `--grid-size`: power-of-two resolution, with `64`, `128`, and `256` as the standard comparison set.
- `--branching`: `octree8`, `wide4`, or `both`.
- `--render-strategy`: `direct`, `intervals`, or `auto` for render benchmarks.
- `--seed`: deterministic scene/ray/point seed. Default seed is `20260712`.
- `--density`: sparse-random occupancy probability inside the centered benchmark region.
- `--iterations`: timed kernel iterations.
- `--count`: point, ray, or pixel count.
- `--jsonl`: append machine-readable benchmark rows.
- `--profile`: enable aggregate traversal counters.

Render also accepts `--operation forward`, `--operation backward`, or `--operation both`.
The realtime viewer accepts `--profile` to show frame, render,
transfer/readback, tonemap, display, scene size, backend, and branching data in
the overlay.

Required metadata:

- date/time
- Git commit when available
- CPU/GPU name when available
- build type
- CUDA enabled flag and CUDA runtime/toolkit version when available
- operation: query, raycast, render forward, render backward, viewer frame
- scene metadata: scene, scene version, grid size, branching, seed, density
- topology metadata: max depth, nodes, leaves

Required timings:

- CPU reference wall time when applicable.
- CUDA H2D transfer time.
- CUDA kernel time.
- CUDA D2H/readback time.
- Total wall time in Python/viewer paths.
- FPS and frame time in realtime viewer paths.

Implemented profiling counters for Milestone 18:

- nodes visited
- child candidates tested
- leaf segments or samples composited
- early opacity termination count
- stack pushes/pops
- maximum stack depth
- memory allocated or allocation count on hot paths

Profiling counters are disabled by default and runtime-gated through explicit
benchmark/viewer options. CUDA counters use aggregate atomics only when a
non-null stats pointer is provided.

## Milestone 18 Scope

Allowed:

- Reproducible C++ and Python benchmark entry points.
- Existing benchmark output cleanup and JSONL emission.
- Optional profiling counters.
- `DeviceBuffer` reuse or destructive allocation APIs.
- Reducing avoidable per-call allocations.
- Hidden synchronization review.
- Python GIL release review.
- CUDA launch/block-size tuning.
- Local memory coalescing and stack/register-pressure cleanup when no descriptor
  redesign is required.

Not allowed:

- No new brick or tile leaf layout.
- No compact descriptor redesign.
- No relative child pointers.
- No block-local addressing.
- No serialization format.
- No compact interval rendering rewrite.
- No topology-changing acceleration unless a new plan promotes it explicitly.

Milestone 18 is successful when benchmark output is reproducible, before/after
numbers are documented, and the data is sufficient to confirm or revise the
Milestone 19 starting point.

## Milestone 19 Status

Implemented primary path:

- `Wide4` raycast, forward render, and backward render now use local DDA/HDDA
  traversal.
- Each visited wide node steps through its local `4 x 4 x 4` child grid in ray
  order.
- Occupancy is tested with bit masks before child rank/payload rank is computed.
- Popcount/rank is used only after an active child cell is known.
- The Wide4 CUDA render paths no longer allocate or sort per-node
  `candidates[64]` arrays.
- Octree8 traversal remains unchanged.

Implemented secondary path:

- CUDA rendering now has an opt-in compact interval strategy:
  `render_strategy="intervals"`.
- Interval mode runs a count/scan/emit traversal prepass, stores compact
  CUDA-resident interval arrays, composites forward from those intervals, and
  reuses saved interval/aux buffers for Torch backward.
- `render_strategy="auto"` currently maps to direct traversal. It should stay
  conservative until benchmark data supports a scene-dependent heuristic.
- CPU interval rendering is intentionally unsupported; CPU rendering remains
  direct/reference-only.

Interval validation notes:

- Interval records stay compact: `ray_index` and `payload_index` are
  `uint32_t`, `leaf_id` is `int32_t`, and interval distances plus forward aux
  values are `float`.
- Per-ray `counts` and prefix-sum `offsets` use `uint32_t`. This keeps scan
  bandwidth and per-ray workspace compact for realtime/GPU-memory-first use.
  A post-scan overflow check rejects interval mode if the total emitted
  interval stream exceeds `uint32_t` capacity.
- Interval build invalidates saved forward aux data. Interval forward marks aux
  data valid only after launching the compositing kernel. Interval backward
  rejects buffers that have not run interval forward.
- The count/scan stage intentionally synchronizes to read the total interval
  count and overflow status before allocating the interval stream. The emit
  stage intentionally synchronizes to read overflow status before returning.
- Temporary overflow counters in renderer host code are owned by CUDA
  `DeviceBuffer` RAII.
- Sanitizer-clean C++ and Python interval runs are required before closing this
  milestone.
- Current local validation has C++ interval `memcheck` clean. `initcheck` still
  needs to be rerun from a stable CUDA-visible shell because the local attempt
  failed before the first instrumented CUDA API call. Python interval sanitizer
  runs need separate care around Torch CUDA visibility and Torch's caching
  allocator. Prefer the narrow smoke target below before using full pytest
  under `compute-sanitizer`:

  ```bash
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck --leak-check full ./.venv/bin/python scripts/check_interval_sanitizer.py
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool initcheck ./.venv/bin/python scripts/check_interval_sanitizer.py
  ```

Still open:

- Record larger before/after benchmark snapshots for representative scenes and
  grid sizes.
- Compare direct traversal against interval prepass/compositing for forward and
  backward separately before making interval mode automatic.

Optional path:

- A coarse occupancy or macro-cell accelerator is considered only if profiling
  shows tree entry through empty space dominates large scenes.

Deferred:

- Dense brick/tile leaves.
- Relative child pointers.
- Block-local addressing.
- Optional contour data.
- Optional compressed attributes.
- Serialization.
