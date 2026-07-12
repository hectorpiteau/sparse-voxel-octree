# TODO and Milestones

This file is the implementation roadmap.

The project should be built in milestones. Each milestone should leave the repository in a working, tested state. Do not skip tests to move faster. Do not optimize layout before a correct reference implementation exists.

---

## Milestone 0 — Repository foundation

Goal: create a buildable repository skeleton with CI-ready structure.

### Tasks

- [x] Create repository layout.
- [x] Add `README.md`.
- [x] Add `TODO.md`.
- [x] Add `LICENSE`.
- [x] Add `.gitignore`.
- [x] Add `.python-version`.
- [x] Add `pyproject.toml`.
- [x] Add `uv.lock`.
- [x] Add top-level `CMakeLists.txt`.
- [x] Add empty Python package directory `svo/`.
- [x] Add `svo/__init__.py`.
- [x] Add `include/svo/` headers.
- [x] Add `src/` for C++ implementation.
- [x] Add `cuda/` for CUDA kernels.
- [x] Add `python/` for bindings.
- [x] Add `tests/python/`.
- [x] Add `tests/cpp/`.
- [x] Add `examples/python/`.
- [x] Add `examples/cpp/`.

### Acceptance criteria

- [x] `uv sync --extra test` works.
- [x] `uv run python -c "import svo"` works.
- [x] `cmake -S . -B build` works.
- [x] `cmake --build build` works.
- [x] `uv run pytest tests/python -q` runs at least one passing test.
- [x] `ctest --test-dir build` runs at least one passing test.

---

## Milestone 1 — Core C++ types

Goal: define the public C++ API and the paper-style descriptor-backed CPU-side topology.

### Tasks

- [x] Add GLM as the core C++ math dependency.
- [x] Use `glm::ivec3` for voxel coordinates.
- [x] Use `glm::vec3` for points, directions, and bounds.
- [x] Add `svo::Device`.
- [x] Add `svo::BuildOptions`.
- [x] Add `svo::QueryOptions`.
- [x] Add `svo::RenderOptions`.
- [x] Add `svo::Error` / exception helpers.
- [x] Add `svo::NodeDescriptor` bit-packed layout.
- [x] Add `svo::Octree` class.
- [x] Add validation helpers.

Required descriptor layout direction:

```cpp
struct NodeDescriptor {
    uint64_t bits;
};
```

Notes:

- [x] Do not add bespoke `float3`, `VoxelCoord`, or `Aabb` structs in the C++ core.
- [x] Expose bitwise accessors for child masks and addressing fields.
- [x] Use the descriptor-backed layout as the actual octree representation, not as a later optimization pass.

### Acceptance criteria

- [x] C++ tests compile.
- [x] `sizeof(NodeDescriptor)` is stable and tested.
- [x] Empty tree validation works.
- [x] Invalid tree validation fails with a useful error.
- [x] Public headers can be included from an external C++ target.

---

## Milestone 2 — CPU octree builder

Goal: build a correct sparse octree on CPU from occupied voxel coordinates.

### Tasks

- [x] Implement `Octree::from_voxels_cpu`.
- [x] Validate input coordinate dtype/range.
- [x] Reject negative coordinates.
- [x] Reject coordinates outside `[0, 2^max_depth)`.
- [x] Deduplicate occupied coordinates.
- [x] Build nodes top-down.
- [x] Assign stable leaf IDs.
- [x] Store `leaf_id -> payload_index`.
- [x] Store root bounds.
- [x] Add tree statistics.
- [x] Add serialization-friendly internal arrays.

### Tests

- [x] Empty input.
- [x] One voxel.
- [x] Two sibling voxels.
- [x] Voxels in different branches.
- [x] Duplicate coordinates.
- [x] Out-of-range coordinates.
- [x] Maximum depth edge case.
- [x] Deterministic build output.

### Acceptance criteria

- [x] CPU builder passes all tests.
- [x] Tree validation passes after every successful build.
- [x] Leaf count equals number of unique occupied coordinates.
- [x] Payload indices are stable and documented.

---

## Validation gate — 64^3 sphere occupancy

Goal: validate the CPU builder and CPU point query against a structured 64x64x64 sparse scene before expanding the API further.

### Tasks

- [x] Build a depth-6 octree from occupied voxels inside a sphere embedded in a `64 x 64 x 64` grid.
- [x] Treat occupied sphere leaves as value `1`.
- [x] Treat query misses outside the sphere as value `0`.
- [x] Check representative inside and outside query locations.
- [x] Compare query results against a direct dense-grid reference for the sphere.

### Acceptance criteria

- [x] The sphere test passes in C++.
- [x] Queried points inside the sphere resolve to occupied leaves with value `1`.
- [x] Queried points outside the sphere resolve to misses and value `0`.
- [x] The octree leaf count matches the direct occupied-voxel count for the sphere.

---

## Milestone 3 — CPU point query

Goal: query points against the CPU octree.

### Tasks

- [x] Implement `query_points_cpu`.
- [x] Define coordinate convention.
- [x] Define behavior outside root bounds.
- [x] Return `-1` for miss.
- [x] Return leaf ID or payload index for hit.
- [x] Add batch query API.

Coordinate convention proposal:

```text
Root bounds are [0, 1]^3 by default.
Voxel coordinates are integer cells in [0, 2^max_depth)^3.
A point maps to floor(point * 2^max_depth).
```

### Tests

- [x] Query center of occupied voxel.
- [x] Query empty voxel.
- [x] Query outside root.
- [x] Query exactly at `0`.
- [x] Query near `1`.
- [x] Query boundary behavior.
- [x] Random occupied coordinates vs direct hash reference.
- [x] 64^3 sphere occupancy regression.

### Acceptance criteria

- [x] CPU point query agrees with direct coordinate hash reference.
- [x] The 64^3 sphere occupancy regression passes.

---

## Milestone 4 — Python bindings

Goal: expose the core CPU octree API to Python.

### Tasks

- [x] Add `pybind11` or `nanobind`.
- [x] Bind `Octree`.
- [x] Bind `Octree.from_voxels`.
- [x] Bind `Octree.query`.
- [x] Bind properties:
  - [x] `max_depth`
  - [x] `num_nodes`
  - [x] `num_leaves`
  - [x] `device`
  - [x] `root_bounds`
- [x] Support NumPy input for CPU build.
- [x] Support NumPy input for CPU query.
- [x] Add clear dtype and shape errors.
- [x] Add docstrings.

### Acceptance criteria

- [x] `uv sync` builds extension.
- [x] `import svo` works.
- [x] Python CPU examples work.
- [x] Bad input errors are readable.

---

## Milestone 5 — Device buffer abstraction

Goal: add safe CPU/GPU memory management.

### Tasks

- [x] Add `DeviceBuffer<T>`.
- [x] Support CPU allocation.
- [x] Support CUDA allocation when enabled.
- [x] Add copy CPU to CUDA.
- [x] Add copy CUDA to CPU.
- [x] Add move semantics.
- [x] Disable unsafe copying.
- [x] Add stream-aware copy APIs.
- [x] Add memory size reporting.
- [x] Add debug bounds metadata.

### Tests

- [x] Allocate/free CPU.
- [x] Allocate/free CUDA.
- [x] Copy CPU to CUDA to CPU.
- [x] Move buffer.
- [x] Zero-size buffer.
- [x] Large buffer smoke test.

### Acceptance criteria

- [x] No leaks under sanitizer where available.
- [x] CUDA buffer tests pass on GPU runner.
- [x] CPU-only build still works.
- [ ] Python-facing CUDA APIs avoid implicit host synchronization except when explicitly requested (manual check: review binding code for `cudaDeviceSynchronize`, default-stream synchronization, blocking tensor transfers, and document any intentional sync).
- [ ] Python bindings accept existing GPU memory without unnecessary host roundtrips.
- [ ] Stream ownership is explicit: callers can pass/use the current stream, and async copies document lifetime requirements (manual check: review API docs and call sites for stream/lifetime ownership rules).
- [ ] Host-to-device copies support pinned host memory for large transfers or clearly document when pageable memory is used (manual check: review transfer APIs and docs for pinned/pageable behavior).
- [ ] Device allocations are amortized or reusable on hot paths; per-call allocation is avoided in query/render loops (manual check: profile allocation counts and review hot-path code).
- [x] CPU fallback paths remain available and do not import or require CUDA/Torch unless the CUDA/Torch layer is enabled.
- [ ] Errors from CUDA calls preserve the original CUDA error string and include the operation name.
- [ ] Python bindings release the GIL around long-running C++/CUDA work (manual check: review pybind wrappers for `py::gil_scoped_release` around blocking or launch-heavy calls).
- [ ] Tensor/array inputs validate dtype, shape, contiguity, device, and lifetime before launching kernels.
- [ ] GPU outputs have deterministic ownership semantics and do not expose dangling views (manual check: review returned tensor/array ownership and lifetime relationships).
- [x] Benchmarks cover transfer latency separately from kernel latency.
- [x] Tests include asynchronous copy behavior and stream ordering.
- [x] Tests include CUDA-disabled misuse errors.
- [ ] Tests include mixed CPU/GPU misuse errors once APIs expose meaningful mixed-device operations.

---

## Pre-Milestone 6 cleanup — DeviceBuffer review follow-ups

Goal: address small code cleanliness and architecture issues before adding CUDA point query code.

### Tasks

- [x] Make `DeviceBuffer::allocate` exception-safe.
  - Allocate CPU storage or CUDA memory into temporary state first.
  - Commit `device_`, `size_`, `cpu_storage_`, and `cuda_data_` only after allocation succeeds.
  - Preserve the existing buffer when a new allocation fails, or clearly document/reset behavior if preserving is intentionally not supported.
  - Add a focused test if practical, especially for CPU allocation failure or a small injected failure path.
- [x] Stop relying on transitive CUDA includes in CUDA tests.
  - Add an explicit `#if SVO_ENABLE_CUDA` guarded include of `<cuda_runtime_api.h>` in `tests/cpp/test_device_buffer.cpp`.
  - Keep `DeviceBuffer.hpp` free to hide or move CUDA implementation details later without breaking tests accidentally.
- [x] Replace manual CUDA stream cleanup in tests with a tiny RAII helper.
  - Wrap `cudaStreamCreate` / `cudaStreamDestroy` in a local test-only helper under `#if SVO_ENABLE_CUDA`.
  - Make early `require()` failures after stream creation avoid leaking the stream during test exits where possible.
  - Keep the helper private to the test file unless another CUDA test needs it.
- [x] Clarify or complete the mixed CPU/GPU misuse checklist item.
  - Current coverage verifies CUDA-disabled misuse and async stream ordering, but not true mixed CPU/GPU misuse cases.
  - Either split the TODO checkbox into separate completed/open items, or add concrete tests for mixed-device misuse once APIs expose enough surface to make that meaningful.
  - Keep TODO wording precise so validation status does not overstate coverage.

## Milestone 6 — CUDA point query

Goal: implement batched point queries on GPU.

### Tasks

- [x] Add CUDA kernel `query_points_cuda`.
- [x] Add C++ launcher.
- [x] Add stream support.
- [x] Add Python binding for CUDA-owned octree topology and CPU NumPy point batches.
- [ ] Add Python binding accepting Torch CUDA tensors, if Torch layer is enabled.
- [ ] Add Python binding accepting raw CUDA arrays later if desired.
- [ ] Return `torch.int64` or `torch.int32` leaf IDs in the future Torch path.
- [x] Compare CUDA result with CPU reference.

### Tests

- [x] Small deterministic tree.
- [x] Random tree.
- [x] Large batch of points.
- [x] Misses.
- [x] Boundary points.
- [x] Non-contiguous Python point inputs should fail clearly.
- [x] Wrong dtype should fail clearly.
- [x] CPU vs CUDA agreement.

### Acceptance criteria

- [x] CUDA query matches CPU reference.
- [x] Query performance is measured with a benchmark script.
- [x] No CUDA errors under normal use.

---

## Milestone 7 — Ray and camera support

Goal: define rays and camera ray generation.

### Tasks

- [x] Implement GLM-based camera helpers.
- [x] Support pinhole camera.
- [x] Support look-at constructor.
- [x] Support intrinsics matrix input.
- [x] Implement CPU ray generation.
- [x] Implement CUDA ray generation.
- [x] Use `glm::vec3` origin/direction arrays or tensors for ray batches.
- [x] Add tests for ray directions.

### Tests

- [x] Center pixel points toward target.
- [x] Camera basis is orthonormal.
- [x] FOV affects ray spread.
- [x] CPU and CUDA ray generation match.
- [x] Invalid camera inputs fail.

### Acceptance criteria

- [x] Python can create a camera.
- [x] Python can generate rays.
- [x] C++ can generate rays.

---

## Milestone 8 — CPU raycast

Goal: implement a simple correct CPU raycast through the octree.

### Tasks

- [x] Implement ray/AABB intersection.
- [x] Implement dense-grid reference raycast for tests.
- [x] Implement CPU octree raycast.
- [x] Return hit/miss.
- [x] Return hit leaf ID.
- [x] Return hit distance `t`.
- [x] Return hit position.
- [x] Return hit voxel scale/depth.
- [x] Add Python binding.

### Tests

- [x] Empty tree miss.
- [x] Single voxel hit.
- [x] Single voxel miss.
- [x] Cube hit.
- [x] Ray starts inside root.
- [x] Ray starts inside occupied voxel.
- [x] Axis-aligned rays.
- [x] Diagonal rays.
- [x] Boundary rays.
- [x] Compare against dense-grid reference.

### Acceptance criteria

- [x] CPU raycast is correct for small trees.
- [x] Raycast behavior is documented.

---

## Milestone 9 — CUDA raycast

Goal: implement GPU ray traversal.

### Tasks

- [x] Implement CUDA raycast kernel.
- [x] Use simple stack-based traversal first.
- [x] Add C++ launcher.
- [x] Add Python binding.
- [x] Add output device buffers:
  - [x] hit mask
  - [x] leaf ID
  - [x] t
  - [x] position
  - [x] depth
- [x] Compare with CPU reference.
- [x] Add benchmark.

### Tests

- [x] CPU vs CUDA raycast on tiny scenes.
- [x] CPU vs CUDA raycast on random sparse scenes.
- [x] Many rays.
- [x] Miss-heavy rays.
- [x] Hit-heavy rays.
- [x] Rays near boundaries.
- [x] Compute sanitizer run locally; keep GPU CI sanitizer enabled when CI exists.

### Acceptance criteria

- [x] CUDA raycast matches CPU reference within tolerance.
- [x] Kernel handles large ray batches.
- [x] No invalid memory accesses under sanitizer.

---

## Milestone 10 — Payload indexing and gather

Goal: use the octree as an index into arbitrary payload tensors.

### Tasks

- [x] Define leaf ID vs payload index behavior.
- [x] Add `leaf_payload_indices`.
- [x] Add `query_payload_indices`.
- [x] Add gather helper for CPU arrays.
- [x] Add gather helper for Torch tensors.
- [x] Add shape rules.
- [x] Add dtype-agnostic support where possible.

### Tests

- [x] One int per voxel.
- [x] RGB float per voxel.
- [x] Feature vector per voxel.
- [x] Miss handling.
- [x] Payload index remapping.
- [x] Torch gather agrees with manual indexing.

### Acceptance criteria

- [x] Users can store arbitrary payload outside the octree.
- [x] Query outputs can index payload tensors directly.

---



## Milestone 10.1 — C++20 migration

Goal: move the C++/CUDA core from C++17 to C++20 before adding heavier Torch CUDA interop.

### Tasks

- [x] Update CMake C++ standard from 17 to 20.
- [x] Update target compile features from `cxx_std_17` to `cxx_std_20`.
- [x] Update README and developer docs that mention C++17.
- [x] Document minimum supported compiler and CUDA expectations for C++20 builds.
- [x] Keep CUDA device code conservative; avoid C++20 standard library features in kernels unless verified with `nvcc`.
- [x] Verify CPU-only builds still work.
- [x] Verify CUDA builds still work.
- [x] Verify Python extension builds through `scikit-build-core`.
- [x] Check public headers still compile from an external target.

### Tests

- [x] CPU CMake configure/build passes.
- [x] CPU C++ test suite passes.
- [x] CUDA CMake configure/build passes.
- [x] CUDA C++ test suite passes.
- [x] Python package editable install passes.
- [x] Python test suite passes.
- [x] Wheel/source build still succeeds if practical.

### Acceptance criteria

- [x] The project consistently builds as C++20.
- [x] README no longer advertises C++17 as the target standard.
- [x] CPU, CUDA, and Python build/test paths still pass.
- [x] Any compiler/CUDA version assumptions are documented.

---


## Milestone 10.2 — C++20 idiom modernization audit

Goal: modernize useful C++20 idioms after the standard migration without changing behavior or making CUDA device code fragile.

### Notes

- [x] No deprecated C++17-era constructs were found in the current core scan (`std::auto_ptr`, `std::result_of`, old iterator adapters, raw owning `new/delete`, or `typedef`).
- [x] Keep changes pragmatic; do not refactor stable code just to use newer syntax.
- [x] Keep CUDA kernels conservative unless each C++20 feature is verified with `nvcc`.

### Candidate follow-ups

- [x] Replace CPU-side manual bit counting helpers with `std::popcount` from `<bit>` where it improves clarity.
- [x] Keep CUDA-side bit counting explicit or use CUDA intrinsics such as `__popc`; do not assume `std::popcount` is device-safe without testing.
- [ ] Consider `std::span` overloads for non-owning C++ batch inputs and buffer copy APIs to reduce `std::vector` coupling.
- [x] Replace local math constants such as camera pi with `std::numbers::pi_v<float>` where host-only.
- [ ] Review sort/unique builder code for possible `std::ranges` use only if readability improves and compiler support remains clean.
- [ ] Consolidate duplicated CUDA RAII test/benchmark helpers if it reduces maintenance without affecting public API.
- [ ] Review repeated `static_cast` size/index conversions and add small checked conversion helpers only where they reduce real risk.
- [ ] Consider lightweight concepts or `requires` only for public templates that need clearer diagnostics; avoid template complexity early.

### Tests

- [x] CPU C++ tests pass after each modernization batch.
- [x] CUDA C++ tests pass for any touched CUDA-adjacent code.
- [x] Python tests pass if public APIs or bindings are touched.
- [x] Public header compile test remains clean under C++20.

### Acceptance criteria

- [ ] Modernizations are behavior-preserving.
- [ ] No CUDA device code adopts unsupported C++20 library features.
- [ ] API additions, if any, are backwards-compatible.
- [ ] Code is clearer or safer after each change; purely cosmetic rewrites are avoided.

---

## Milestone 10.5 — Torch CUDA tensor interop

Goal: let Python users keep query/raycast inputs, outputs, and payload gathering on CUDA tensors without implicit host roundtrips.

### Tasks

- [x] Add optional Torch build/runtime integration without making Torch mandatory for CPU-only users.
- [x] Accept Torch CUDA point tensors for CUDA point query.
- [x] Accept Torch CUDA ray origin/direction tensors for CUDA raycast.
- [x] Return CUDA tensor outputs for CUDA query/raycast when inputs are CUDA tensors.
- [x] Support `torch.int32` payload/leaf ID outputs by default, with documented dtype behavior.
- [x] Make `svo.gather_payload` work on CUDA payload tensors and CUDA index tensors without device transfers.
- [x] Avoid CPU-GPU transfers on CUDA tensor hot paths; only explicit CPU-return APIs may copy results to host.
- [x] Reuse CUDA-resident topology and caller-provided CUDA tensors instead of staging through NumPy/CPU buffers.
- [x] Respect the current PyTorch CUDA stream or expose explicit stream behavior.
- [x] Avoid per-call topology transfers; require/use a CUDA-owned octree for hot paths.
- [x] Validate dtype, shape, contiguity, device, and lifetime before launching kernels.
- [x] Document synchronization behavior and avoid implicit host synchronization except for explicit CPU-return APIs.

### Tests

- [x] CUDA Torch query returns CUDA tensors and matches CPU reference after explicit `.cpu()`.
- [x] CUDA Torch raycast returns CUDA tensors and matches CPU/CUDA NumPy reference after explicit `.cpu()`.
- [x] CUDA Torch payload gather matches manual masked indexing on CUDA tensors.
- [x] Miss handling fills `-1` entries without indexing the last payload row.
- [x] Mixed-device inputs fail clearly.
- [x] Non-contiguous tensors fail clearly or are explicitly copied only when requested.
- [x] Current-stream behavior is tested with a non-default stream.
- [x] Transfer behavior is checked: CUDA tensor query/raycast/gather paths do not perform implicit host copies.
- [x] No hidden host synchronization in hot paths, verified by code review and profiler/manual check.

### Acceptance criteria

- [x] Users can run `cuda_tree.query(torch_points_cuda)` and receive CUDA tensor IDs.
- [x] Users can run `svo.gather_payload(torch_payload_cuda, ids_cuda)` entirely on GPU.
- [x] Query/raycast + gather can be chained into downstream Torch CUDA operations without CPU copies.
- [x] CPU-GPU transfers are avoided or reduced to documented explicit API boundaries.
- [x] CPU-only installs still import and run without Torch.

---

## Milestone 11 — Trilinear interpolation

Goal: add differentiable sampling of sparse voxel payloads.

### Tasks

- [x] Decide cell-centered vs corner-valued vs brick-valued layout.
- [x] Implement CPU trilinear reference.
- [x] Implement CUDA trilinear interpolation.
- [x] Add forward API.
- [x] Add backward API for payload gradients.
- [x] Add Torch autograd wrapper.
- [x] Support feature dimension `C`.

Implemented layout:

```text
leaf_payload: [payload_rows] or [payload_rows, C]
leaf_id -> payload_index
samples blend the 8 neighboring max-depth leaf-center values
missing neighbors use fill_value, default 0
```

### Tests

- [x] Constant field returns constant.
- [x] Linear field interpolates exactly.
- [x] CPU vs CUDA interpolation.
- [x] Torch `gradcheck` for payload values.
- [x] Miss behavior.
- [x] Boundary behavior.

### Acceptance criteria

- [x] Interpolation is differentiable with respect to payload tensors.
- [x] Gradients match finite differences for stable cases.

---

## Milestone 12 — Forward renderer

Goal: render an image from a camera using the octree and payload tensors.

### Tasks

- [x] Implement CPU reference renderer for tiny scenes.
- [x] Implement CUDA forward renderer.
- [x] Generate rays from camera.
- [x] Traverse octree.
- [x] Sample density/color along rays.
- [x] Implement alpha compositing.
- [x] Return image/RGB.
- [x] Return depth.
- [x] Return opacity.
- [ ] Return optional aux buffers for backward.

Implemented scope:

- Rays-first API: `svo.render_volume(tree, origins, directions, sigma, color, ...)`.
- Payload layout: `sigma` shape `(P,)`, `color` shape `(P, 3)`, float32.
- CUDA Torch path requires CUDA-owned octree and CUDA tensors to avoid hidden CPU/GPU transfers.
- Aux buffers and autograd are deferred to Milestone 13.

Volume rendering convention:

```text
alpha_i = 1 - exp(-sigma_i * delta_i)
T_i = product_{j<i}(1 - alpha_j)
rgb = sum_i T_i * alpha_i * color_i
opacity = 1 - product_i(1 - alpha_i)
```

### Tests

- [x] Empty scene renders transparent/background.
- [x] Constant density cube.
- [x] Constant color cube.
- [x] Depth sanity.
- [x] CPU vs CUDA on tiny image/ray batches.
- [x] Deterministic output.
- [x] No NaNs for hit outputs.

### Acceptance criteria

- [x] Python can call `svo.render_volume(...)` and receive RGB/depth/opacity arrays or tensors.
- [x] CPU reference and CUDA agree for small scenes.
- [x] Forward render example works.

---

## Milestone 13 — Backward renderer

Goal: integrate custom CUDA backward with PyTorch autograd.

### Tasks

- [x] Define saved forward state.
- [x] Implement compositing backward.
- [x] Recompute renderer traversal in backward.
- [x] Scatter-add gradients into density.
- [x] Scatter-add gradients into color.
- [ ] Support feature gradients if features are rendered.
- [x] Add `torch.autograd.Function`.
- [x] Add `torch.nn.Module` wrapper.
- [x] Add finite-difference gradient tests.

Implemented scope:

- CUDA Torch backward for `sigma` and `color` payloads.
- Gradients from RGB and opacity outputs; depth remains forward-only.
- Backward recomputes traversal and uses atomic adds into payload gradients.
- Renderer remains float32-only; formal float64 `gradcheck` is deferred.

### Tests

- [x] Finite-difference color gradient check.
- [x] Finite-difference density gradient check.
- [ ] Gradcheck features, if implemented.
- [x] Loss backward smoke test.
- [x] Optimization loop reduces loss on tiny target.
- [x] No race conditions under compute-sanitizer for covered cases.
- [x] Atomic scatter behavior is documented by tests as numerically tolerant.

### Acceptance criteria

- [x] `loss.backward()` works for CUDA Torch `sigma` and `color`.
- [x] Custom PyTorch losses over RGB and opacity work.
- [x] Gradients are correct for stable tiny scenes within float32 finite-difference tolerances.

---

## Milestone 14 — Wide 4x4x4 sparse nodes

Goal: add a sparse wide-tree topology option with 64 children per node to reduce traversal depth and benchmark it against the current 2x2x2 octree layout.

Design direction:

- Start with a tree-level 4x4x4 branching mode, not arbitrary per-node branching.
- Keep topology separate from payload buffers.
- Keep gradient/compositing math independent from descriptor decoding where practical.
- Preserve the current 8-way octree path until the wide path is validated.

### Tasks

- [x] Define the public/API representation for branching mode or a dedicated wide-tree type.
- [x] Design a 64-child descriptor format with child occupancy, leaf flags, child base, and payload base.
- [x] Implement CPU builder for 4x4x4 sparse nodes.
- [x] Implement CPU query traversal for wide nodes.
- [x] Implement CUDA query traversal for wide nodes.
- [x] Implement CPU raycast traversal for wide nodes.
- [x] Implement CUDA raycast traversal for wide nodes.
- [x] Implement CPU forward render traversal for wide nodes.
- [x] Implement CUDA forward render traversal for wide nodes.
- [x] Update renderer backward traversal after Milestone 13 exists.
- [x] Add benchmarks comparing 8-way vs 64-way traversal at equal voxel resolution.
- [x] Document memory/layout tradeoffs and when to choose each topology.

### Tests

- [x] Builder output validation for empty, single voxel, dense small cube, and sparse random scenes.
- [x] CPU query parity between 8-way and 64-way trees for equivalent occupied coordinates.
- [x] CUDA query parity with CPU wide query.
- [x] CPU/CUDA raycast parity for wide nodes.
- [x] CPU/CUDA forward render parity for wide nodes.
- [x] Backward renderer parity/gradcheck after wide traversal is wired into Milestone 13 code.
- [ ] Edge cases at 4x4x4 child boundaries.

### Acceptance criteria

- [ ] Wide nodes reduce traversal depth for equivalent voxel resolution.
- [ ] Wide-node CPU and CUDA results match existing 8-way behavior for equivalent scenes.
- [ ] Benchmarks show whether wide nodes improve query/raycast/render performance enough to justify the memory tradeoff.
- [ ] Existing 8-way tests and APIs remain working.

---

## Milestone 15 — Packaging

Goal: build installable Python wheels and source distributions.

### Tasks

- [x] Finalize `pyproject.toml`.
- [x] Add package metadata.
- [x] Add versioning with `setuptools_scm` or equivalent.
- [x] Add `python -m svo.info`.
- [x] Add source distribution build.
- [x] Add wheel build.
- [x] Add Torch/CUDA compatibility matrix for public Python package support.
- [x] Make PyPI wheels Python-runtime-only; keep C++ headers/static library/CMake package in source/CMake installs.
- [ ] Add `uv publish` workflow.
- [x] Add package install tests.
- [x] Add README badges later if desired.

### Acceptance criteria

- [x] `uv build` succeeds.
- [x] Wheel installs in a clean virtual environment.
- [x] `import svo` works from installed wheel.
- [x] Basic CPU tests pass from installed wheel.
- [ ] CUDA tests pass from installed wheel on GPU runner.
- [x] Compatibility matrix documents supported Python, PyTorch, CUDA toolkit/runtime, NVIDIA driver, OS/architecture, and wheel variant combinations.
- [x] Matrix distinguishes tested, expected-supported, and unsupported/deferred combinations.

---

## Milestone 16 — CI

Goal: continuous integration for correctness and packaging.

### Tasks

- [x] Add GitHub Actions workflow for CPU tests.
- [x] Add workflow for wheel builds.
- [x] Add workflow for linting.
- [x] Add optional self-hosted GPU runner workflow.
- [x] Add TestPyPI publish workflow.
- [x] Add PyPI publish workflow on tags.
- [x] Cache build dependencies.
- [x] Upload wheel artifacts.

### Acceptance criteria

- [x] PRs run CPU tests.
- [x] PRs build at least one wheel.
- [ ] GPU runner runs CUDA tests after a self-hosted CUDA runner with labels `self-hosted`, `linux`, `x64`, and `cuda` is configured.
- [x] Tag builds release wheels.
- [ ] TestPyPI release can be installed after the GitHub `testpypi` environment and TestPyPI Trusted Publisher are configured.

---

## Milestone 17 — Documentation and examples

Goal: make the project usable by new developers.

### Tasks

- [x] Add installation guide.
- [x] Add C++ usage guide.
- [x] Add Python usage guide.
- [x] Add PyTorch rendering guide.
- [x] Add differentiability notes.
- [x] Add architecture document.
- [x] Add data layout document.
- [x] Add testing guide.
- [x] Add packaging guide.
- [x] Add Torch/CUDA compatibility matrix to installation and packaging docs.
- [x] Add troubleshooting guide.

### Docsify / GitHub Pages

- [x] Add a Docsify shell for the Markdown documentation.
- [x] Add `docs/_sidebar.md` with the documentation navigation tree.
- [x] Add `docs/_coverpage.md` or a simple landing page using the existing project logo.
- [x] Add `docs/.nojekyll` so GitHub Pages serves Docsify assets correctly.
- [x] Add local preview instructions, for example `npx docsify-cli serve docs`.
- [x] Add a GitHub Actions workflow or Pages configuration note to publish `docs/` to GitHub Pages.
- [x] Verify internal links and images render correctly in both GitHub Markdown and Docsify.
- [x] Keep Docsify optional: no runtime package dependency and no impact on Python/C++ builds.

### Examples

- [x] Build tree from voxel coordinates.
- [x] Query points.
- [x] Raycast from camera.
- [x] Gather per-voxel features.
- [x] Render density/color payloads.
- [ ] Render density/color bricks once brick payload layout exists.
- [ ] Optimize payload with PyTorch loss.
- [ ] Compare CPU and CUDA outputs.

### Acceptance criteria

- [ ] New contributor can run examples.
- [x] Documentation can be previewed locally through Docsify.
- [x] Documentation can be published through GitHub Pages.
- [x] Coding agent has enough context to modify the project safely.
- [x] Documentation describes what is differentiable and what is not.

Note: the first Markdown documentation split is in place under `docs/`. The
remaining open example items should become runnable scripts, not only reference
snippets in documentation.

Docsify estimate: low to medium effort. A basic local Docsify site over the
existing Markdown files is about half a day. A clean GitHub Pages setup with
navigation, link/image cleanup, preview instructions, and CI/publish validation
is closer to one day. The main risk is link compatibility between GitHub's
Markdown renderer and Docsify routing, not code complexity.

Docsify implementation note: `scripts/generate_docsify.py` owns the generated
Docsify shell files. GitHub Pages deployment is configured in
`.github/workflows/docs.yml`; repository Pages settings still need to allow
deployment from GitHub Actions.

---

## Milestone 17.5 — Rendering acceleration roadmap

Goal: remove ambiguity before coding more performance work. This milestone is a
short design/triage milestone, not an implementation milestone.

Context:

- The existing `Octree8` and `Wide4` paths are correct, but wide branching alone is not guaranteed to improve rendering speed.
- Public high-performance sparse volume systems usually combine hierarchy with measured traversal choices, DDA/HDDA stepping, empty-space skipping, compact ray intervals, and sometimes dense brick/tile leaves.
- Topology and interval membership stay discrete. Payload values such as density, color, and features remain differentiable.

### Decisions

- [x] Define representative benchmark scenes: empty, single voxel, dense cube, deterministic sparse random, sphere, shell, and larger viewer-style scenes.
- [x] Define target metrics: kernel time, wall time, FPS, nodes visited, stack traffic, leaf segments composited, memory allocated, and CPU-GPU transfers.
- [x] Decide which metrics must be available from C++ benchmarks, Python scripts, and the realtime viewer.
- [x] Decide which optimizations are allowed in Milestone 18 because they do not change topology/layout semantics.
- [x] Decide which rendering acceleration path starts Milestone 19 after profiling data exists.
- [x] Keep dense brick leaves, relative pointers, block-local addressing, contour data, compressed attributes, and serialization explicitly out of Milestones 18 and 19 unless profiling proves they are the immediate bottleneck.

### Candidate Acceleration Ideas

- [x] Wide4 local DDA/HDDA traversal instead of scanning or sorting many child AABBs.
- [x] Compact interval generation before forward/backward rendering.
- [x] Optional coarse occupancy or macro-cell accelerator for empty-space skipping.
- [x] Dense brick/tile leaves for later layout work.
- [x] Advanced descriptor compression for later memory-layout work.

### Acceptance criteria

- [x] Milestone 18 has a concrete profiling and low-risk optimization scope.
- [x] Milestone 19 has one primary rendering acceleration target and clear fallback criteria.
- [x] Deferred layout work is captured without blocking the immediate performance pass.

Decision record: see `docs/performance-roadmap.md`. `sparse_random` uses fixed
seed `20260712` by default and benchmark output should include seed, density,
grid size, scene version, and branching mode.

---

## Milestone 18 — Profiling and low-risk optimization pass

Goal: measure the current implementation and apply optimizations that do not
change public semantics, topology layout, or differentiability behavior.

### Profiling and Benchmarks

- [x] Add reproducible C++/Python benchmark entry points for point query, raycast, forward render, backward render, and realtime-viewer-style rendering.
- [x] Report GPU kernel time separately from allocation, Python overhead, and CPU-GPU transfer time.
- [x] Add optional CPU/CUDA profiling counters for traversal and rendering, disabled by default.
- [x] Count nodes visited per ray/query.
- [x] Count child candidates tested per ray.
- [x] Count leaf segments/samples composited per ray.
- [x] Count early opacity termination events.
- [x] Count stack pushes/pops and maximum stack depth.
- [x] Expose aggregated profiling output in C++ benchmarks.
- [x] Expose optional profiling output in Python for debug/viewer use.
- [x] Keep profiling runtime-gated so atomics/counter writes are disabled unless a stats pointer is provided.
- [ ] Add a compile-time `SVO_ENABLE_TRAVERSAL_STATS` option to compile out stats instrumentation for lean release/performance builds.
  - Prefer preserving public option fields and treating stats pointers as ignored when disabled, unless an API/ABI decision says otherwise.
  - Make benchmark `--profile` report clearly when traversal stats were compiled out.
  - Validate both enabled and disabled builds so stats code does not become an always-on maintenance tax.

### Low-Risk Optimizations

- [x] Add explicit destructive/reuse allocation APIs for `DeviceBuffer`.
  - Consider `reset`, `release_and_allocate`, or reusable capacity-style APIs for hot paths where double allocation is too expensive.
  - Keep the current `allocate` strong-guarantee behavior as the safe default.
  - Document whether destructive allocation preserves old contents, old capacity, stream ordering, and failure state.
- [ ] Reduce avoidable per-call allocations in query, raycast, render, and viewer loops.
- [ ] Review hidden host synchronization in Python/CUDA hot paths.
- [ ] Review GIL release around long-running C++/CUDA work.
- [ ] Improve memory coalescing where it does not require a descriptor redesign.
- [ ] Reduce stack traffic and register pressure where profiling identifies it as a bottleneck.
- [ ] Tune launch/block sizes for query, raycast, forward render, and backward render.
- [ ] Keep `Octree8` and `Wide4` output parity unchanged.

### Explicitly Out of Scope

- No new brick/tile leaf layout.
- No relative child pointer or block-local descriptor redesign.
- No serialization format.
- No interval-rendering rewrite.
- No topology-changing optimization unless promoted by a new plan.

### Acceptance criteria

- [x] Benchmarks are reproducible from documented commands.
- [x] Benchmark output gives enough data to choose the Milestone 19 acceleration target.
- [ ] Existing CPU, CUDA, and Torch rendering tests still pass.
- [ ] Performance changes are documented with before/after numbers.
- [ ] Any optimization that is scene-dependent is labeled as such.

---

## Milestone 19 — Rendering acceleration implementation

Goal: implement the first measured structural acceleration for rendering, based
on Milestone 18 profiling results. This milestone is about traversal/rendering
speed, not general sparse-layout compression.

### Primary Path — Wide4 Local DDA / HDDA

- [ ] Implement CPU reference DDA traversal through `Wide4` child grids.
- [ ] Step through the local `4 x 4 x 4` child grid in ray order.
- [ ] Use bit masks to check whether the current child cell is occupied.
- [ ] Use popcount/rank only after a hit child cell is known to be active.
- [ ] Avoid per-node child candidate arrays in the hot path.
- [ ] Add CUDA implementation with fixed-size local state and low register pressure.
- [ ] Preserve the existing 8-way path unless a safe equivalent DDA traversal is added.
- [ ] Keep the previous wide traversal available for debug comparison until confidence is high.

### Secondary Path — Compact Render Intervals

- [ ] Add only if Milestone 18 shows traversal is still a major render bottleneck after DDA.
- [ ] Add a render prepass that traverses rays and emits compact occupied intervals.
- [ ] Store interval records as structure-of-arrays for CUDA-friendly access:
  - [ ] `ray_index`
  - [ ] `t_start`
  - [ ] `t_end`
  - [ ] `leaf_id`
  - [ ] `payload_index`
- [ ] Add prefix-sum/compaction logic for variable interval counts per ray.
- [ ] Add capacity handling and clear overflow reporting.
- [ ] Add a forward-render kernel that consumes intervals and composites color/opacity/depth.
- [ ] Add a backward-render path that either consumes stored intervals or recomputes intervals according to the chosen memory/performance tradeoff.
- [ ] Keep interval generation non-differentiable; gradients flow through payload values, not topology or interval membership.
- [ ] Separate render-only and training modes when their memory/performance tradeoffs differ.

### Optional Path — Coarse Occupancy Accelerator

- [ ] Add only if profiling shows tree entry through empty space dominates large scenes.
- [ ] Evaluate top-level macro resolutions such as `16^3`, `32^3`, and `64^3`.
- [ ] Traverse macro cells with DDA and enter the tree only for occupied macro cells.
- [ ] Keep CUDA-resident occupancy data to avoid CPU-GPU transfers.
- [ ] Ensure this accelerator is optional and does not change query/render semantics.

### Deferred Layout Work

- Dense brick/tile leaves are deferred to a later sparse-layout milestone.
- Relative child pointers are deferred to a later compact-descriptor milestone.
- Block-local addressing is deferred to a later compact-descriptor milestone.
- Optional contour data and compressed attributes are deferred.
- Serialization is deferred unless release planning pulls it earlier.

### Tests

- [ ] CPU DDA traversal matches existing wide traversal on empty, dense, sparse random, sphere, shell, and boundary scenes.
- [ ] CUDA DDA traversal matches CPU DDA traversal within existing tolerances.
- [ ] Forward rendering with DDA matches existing direct traversal for color, opacity, and depth.
- [ ] Backward rendering with DDA matches existing finite-difference tests for density and color.
- [ ] Compact interval generation, if implemented, matches direct traversal compositing.
- [ ] Interval overflow paths, if implemented, fail clearly and do not corrupt outputs.
- [ ] Optional coarse occupancy acceleration, if implemented, preserves exact render/query results relative to the non-accelerated path.

### Benchmarks

- [ ] Compare old wide traversal vs wide DDA traversal.
- [ ] Compare direct traversal rendering vs interval prepass + interval rendering if intervals are implemented.
- [ ] Compare with and without coarse occupancy acceleration if it is implemented.
- [ ] Benchmark forward render and backward render separately.
- [ ] Include realtime viewer FPS measurements for representative small, medium, and large scenes.

### Acceptance criteria

- [ ] Default rendering remains correct and tested on CPU and CUDA.
- [ ] Differentiable rendering still supports PyTorch optimization loops.
- [ ] Benchmarks show whether the acceleration helps, hurts, or is scene-dependent.
- [ ] Render-only and training modes are clearly separated where their performance/memory tradeoffs differ.
- [ ] Deferred sparse-layout work is not mixed into the rendering acceleration milestone.

---

## Milestone 20 — Release 0.1

Goal: publish a minimal useful release.

### Required features

- [ ] CPU octree builder.
- [ ] CPU point query.
- [ ] Python bindings.
- [ ] Basic C++ API.
- [ ] Source distribution.
- [ ] Linux wheel.
- [ ] Tests.
- [ ] Documentation.

### Optional features

- [ ] CUDA point query.
- [ ] CUDA raycast.
- [ ] Torch tensor support.

### Release checklist

- [ ] Version bumped.
- [ ] Changelog updated.
- [ ] README accurate.
- [ ] TODO updated.
- [ ] Tests pass.
- [ ] Wheels built.
- [ ] TestPyPI install tested.
- [ ] PyPI publish completed.
- [ ] Git tag pushed.

---

## Milestone 21 — Release 0.2

Goal: publish first CUDA-focused release.

### Required features

- [ ] CUDA point query.
- [ ] CUDA raycast.
- [ ] Torch tensor input/output.
- [ ] GPU CI tests.
- [ ] Linux CUDA wheel.
- [ ] Installation diagnostics.

### Optional features

- [ ] Windows CUDA wheel.
- [ ] Basic renderer forward.

---

## Milestone 22 — Release 0.3

Goal: publish first differentiable rendering release.

### Required features

- [ ] Brick payload layout.
- [ ] Trilinear interpolation.
- [ ] Forward renderer.
- [ ] Backward renderer.
- [ ] PyTorch autograd integration.
- [ ] Gradcheck tests.
- [ ] Optimization example.

### Acceptance criteria

- [ ] User can optimize density/color payloads from image loss.
- [ ] Custom PyTorch losses work.
- [ ] Documentation clearly explains differentiability limitations.

---

## Coding standards

### C++

- Use C++20.
- Prefer RAII.
- Avoid raw owning pointers.
- Use explicit error handling.
- Keep public headers clean.
- Avoid unnecessary template complexity early.

### CUDA

- Check CUDA errors.
- Keep launchers in C++.
- Keep kernels small and testable.
- Use stream-aware APIs.
- Avoid hidden synchronization unless documented.
- Add comments for memory layout assumptions.

### Python

- Keep wrappers thin.
- Validate dtypes and shapes.
- Give useful error messages.
- Avoid implicit CPU/GPU copies unless documented.
- Do not hide expensive operations.

### Tests

- Every feature needs tests.
- Every CUDA kernel needs CPU reference coverage.
- Every public Python API needs at least one test.
- Every bug fix should add a regression test.

---

## Future features, not scheduled

These are useful directions but are not assigned to a milestone yet. Do not treat them as required scope for the next implementation milestone unless explicitly promoted into the roadmap.

- [ ] Add corner-valued trilinear interpolation for workloads where values live on sparse grid corners instead of leaf centers.
- [ ] Add brick-based interpolation for per-leaf dense feature bricks, with explicit boundary stitching behavior.
- [ ] Add hierarchical/coarse fallback interpolation for samples near sparse edges where one or more neighboring leaves are missing.
- [ ] Compare interpolation layouts in documentation: leaf-centered, corner-valued, brick-based, and hierarchical fallback.
- [ ] Add dense brick/tile leaves for production sparse rendering after the first measured rendering acceleration pass.
- [ ] Add relative child pointers for compact descriptors.
- [ ] Add block-local addressing for page/block-oriented node storage.
- [ ] Add optional contour data.
- [ ] Add optional compressed attributes.
- [ ] Add serialization format and tests.

---

## Open design decisions

These should be resolved before or during implementation.

- [ ] Project name on PyPI.
- [ ] Whether PyTorch support is in the main package or separate package.
- [ ] Initial CUDA version target.
- [ ] Initial supported compute capabilities.
- [ ] Which additional interpolation layouts should be supported after the leaf-centered baseline.
- [ ] Serialization format.
- [ ] Whether to support dynamic updates in early versions.
- [ ] Whether to support camera gradients in first renderer version.
- [ ] Deterministic vs fast scatter-add in backward.
