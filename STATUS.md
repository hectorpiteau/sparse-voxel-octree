# Project Status

Last updated: 2026-06-30

This repository is currently developed on a machine without a CUDA-capable GPU. CPU/C++/Python packaging work has been verified locally. CUDA paths are scaffolded or guarded, but GPU behavior still needs validation on a CUDA machine.

## Current State

- Milestones 0-4 are implemented: repository foundation, core C++ types, CPU octree builder, CPU point query, Python bindings, and the sphere slice example.
- Milestone 5 is partially complete: `DeviceBuffer<T>` exists with CPU behavior verified locally and CUDA code paths guarded by `SVO_ENABLE_CUDA`.
- Core C++ uses GLM types. Do not introduce custom vector structs.
- Topology uses the bit-packed `NodeDescriptor` structure. This is the intended paper-style packed integer structure.
- Torch should remain only in Python-facing/Torch integration layers, not in the core C++ API.
- `pybind11` is no longer vendored. It is a Python build dependency in `pyproject.toml`.
- `third_party/glm` remains vendored because GLM is part of the public C++ API.
- `SVO_ENABLE_PYTHON` defaults to `OFF` for C++ users. Python wheel builds explicitly enable it through scikit-build config.

## Important Files

- `include/svo/Octree.hpp`: public octree types, `Device`, `NodeDescriptor`, `Octree`.
- `src/Builder.cpp`: CPU octree construction from occupied voxel coordinates.
- `src/Query.cpp`: CPU point query implementation.
- `include/svo/DeviceBuffer.hpp`: milestone 5 memory abstraction.
- `tests/cpp/test_device_buffer.cpp`: CPU tests plus CUDA-guarded tests.
- `python/bindings.cpp`: pybind11 Python bindings.
- `examples/python/sphere_slice.py`: 64x64x64 sphere octree slice visualization.
- `TODO.md`: milestone plan and current checklist.

## Local Verification Passed

Commands run successfully on the non-CUDA development machine:

```bash
cmake -S . -B build-m5 -DSVO_ENABLE_CUDA=OFF -DSVO_BUILD_TESTS=ON
cmake --build build-m5
ctest --test-dir build-m5 --output-on-failure
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/python -q
UV_CACHE_DIR=/tmp/uv-cache uv build
```

Results:

- C++ tests: 5/5 passed.
- Python tests: 5 passed.
- Python sdist/wheel build: passed.
- Sphere slice example previously rendered successfully to `/tmp/sphere_slice.png`.

## Current Dirty Worktree Notes

Expected local changes include:

- Modified `CMakeLists.txt`.
- Modified `pyproject.toml`.
- Modified `TODO.md`.
- Modified `tests/cpp/test_public_headers.cpp`.
- Added `include/svo/DeviceBuffer.hpp`.
- Added `tests/cpp/test_device_buffer.cpp`.

There may also be the earlier pybind11-vendoring cleanup depending on what has been committed before transfer.

## Milestone 5 Remaining Work On CUDA Machine

Run and fix as needed:

```bash
cmake -S . -B build-cuda -DSVO_ENABLE_CUDA=ON -DSVO_BUILD_TESTS=ON
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
```

Specific checks:

- Confirm `DeviceBuffer<T>` CUDA allocation/free works.
- Confirm CPU to CUDA to CPU roundtrip test passes.
- Confirm `SVO_ENABLE_CUDA=ON` correctly links `CUDA::cudart`.
- Check that the public header compiles in both CPU-only and CUDA-enabled builds.
- Run sanitizer or `compute-sanitizer` where available.
- Review whether `DeviceBuffer<T>` should remain header-only or move CUDA implementation into `.cu/.cpp` to reduce CUDA header exposure for installed users.

## Milestone 5 Best-Practice Acceptance Checks

The TODO contains a detailed checklist for Python/CUDA module quality. Important points:

- Avoid implicit host synchronization in Python-facing CUDA APIs unless explicitly requested.
- Accept existing GPU memory without host roundtrips.
- Make stream ownership explicit and document async copy lifetime rules.
- Avoid per-call device allocation in query/render hot paths.
- Preserve CPU fallback without importing CUDA/Torch unless those layers are enabled.
- Preserve CUDA operation names and original CUDA error strings.
- Release the GIL around long-running C++/CUDA work.
- Validate dtype, shape, contiguity, device, and lifetime before kernel launch.
- Benchmark transfer latency separately from kernel latency.

## Recommended Next Steps

1. Verify milestone 5 on the CUDA machine with `SVO_ENABLE_CUDA=ON`.
2. If CUDA build fails because `DeviceBuffer.hpp` includes CUDA runtime headers from a public header, consider splitting implementation into a private CUDA translation unit while preserving the public template API.
3. Once milestone 5 passes on GPU, update `TODO.md` to mark CUDA allocation/copy and GPU runner acceptance complete.
4. Start milestone 6: CUDA point query.
5. Implement `query_points_cuda` by mirroring the CPU traversal result exactly, then compare against CPU reference tests on deterministic and random trees.
6. Add stream-aware launcher APIs before exposing Python GPU bindings.
7. Delay Torch integration until the raw C++/CUDA query path is correct and tested.

## Design Constraints To Preserve

- Keep GLM in the core.
- Keep Torch only in Python/Torch-facing integration.
- Do not invent new vector math structures.
- Keep octree topology separate from payload buffers.
- Use the packed `NodeDescriptor` for topology traversal.
- Keep Python bindings thin; important algorithms belong in C++/CUDA.
- Keep C++-only installation easy and independent from Python/pybind11.
