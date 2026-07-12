# Performance Roadmap

This document is the decision record for Milestone 17.5. It defines what the
next optimization work is allowed to measure, change, and defer.

The current decision is:

- Milestone 18 profiles the current implementation and applies low-risk
  optimizations only.
- Milestone 19 starts with measured `Wide4` local DDA/HDDA traversal.
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

Future profiling counters for Milestone 18:

- nodes visited
- child candidates tested
- leaf segments or samples composited
- early opacity termination count
- stack pushes/pops
- maximum stack depth
- memory allocated or allocation count on hot paths

Profiling counters must be disabled by default and compile/runtime gated so
normal kernels do not pay overhead.

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

## Milestone 19 Direction

Primary path:

- Implement `Wide4` local DDA/HDDA traversal.
- Step through each local `4 x 4 x 4` child grid in ray order.
- Use bit masks to test occupancy.
- Use popcount/rank only after an active child cell is known.
- Avoid per-node child candidate arrays in the hot path.
- Keep the previous wide traversal available for debug comparison until the new
  traversal is proven stable.

Secondary path:

- Compact render intervals are considered only if Milestone 18 shows traversal
  is still a major render bottleneck after DDA.
- Interval membership remains non-differentiable; gradients flow through payload
  values only.

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
