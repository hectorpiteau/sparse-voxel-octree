# Project Status

Last updated: 2026-07-01

This repository has now been validated on a CUDA-capable machine with an NVIDIA GeForce RTX 3090. CPU/C++/Python packaging work and the current CUDA `DeviceBuffer<T>` paths have been verified locally.

## Current State

- Milestones 0-4 are implemented: repository foundation, core C++ types, CPU octree builder, CPU point query, Python bindings, and the sphere slice example.
- Milestone 5 is partially complete: `DeviceBuffer<T>` exists with CPU behavior and CUDA allocation/copy behavior verified locally. CUDA code paths remain guarded by `SVO_ENABLE_CUDA`.
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

Commands run successfully on the CUDA development machine:

```bash
nvidia-smi
nvcc --version
cmake -S . -B build-cuda -DSVO_ENABLE_CUDA=ON -DSVO_BUILD_TESTS=ON
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
compute-sanitizer --tool memcheck ./build-cuda/svo_device_buffer_test
cmake -S . -B build-cpu -DSVO_ENABLE_CUDA=OFF -DSVO_BUILD_TESTS=ON
cmake --build build-cpu
ctest --test-dir build-cpu --output-on-failure
UV_CACHE_DIR=/tmp/uv-cache uv run --extra test pytest tests/python -q
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import svo; print(svo.__version__)"
UV_CACHE_DIR=/tmp/uv-cache uv build
```

Results:

- GPU: NVIDIA GeForce RTX 3090, driver 580.142, CUDA runtime 13.0, nvcc 12.8.61.
- CUDA C++ tests: 5/5 passed.
- CPU-only C++ tests: 5/5 passed.
- CUDA `DeviceBuffer<T>` memcheck: 0 errors.
- Python tests: 5 passed.
- Python import check: reported version `0.1.0`.
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

## Milestone 5 CUDA Validation

Completed on the CUDA machine:

- `DeviceBuffer<T>` CUDA allocation/free works.
- CPU to CUDA to CPU roundtrip test passes.
- CUDA zero-size, move ownership, partial copy, and async stream-ordering tests pass.
- `SVO_ENABLE_CUDA=ON` links `CUDA::cudart` (`libcudart.so.12`).
- Public headers compile in both CPU-only and CUDA-enabled builds.
- `compute-sanitizer --tool memcheck ./build-cuda/svo_device_buffer_test` reports 0 errors.

Remaining design review:

- Decide whether `DeviceBuffer<T>` should remain header-only or move CUDA implementation into `.cu/.cpp` to reduce CUDA header exposure for installed users.

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

1. Decide whether to keep `DeviceBuffer<T>` header-only or move CUDA implementation into a private `.cu/.cpp` translation unit.
2. Start milestone 6: CUDA point query.
3. Implement `query_points_cuda` by mirroring the CPU traversal result exactly, then compare against CPU reference tests on deterministic and random trees.
4. Add stream-aware launcher APIs before exposing Python GPU bindings.
5. Delay Torch integration until the raw C++/CUDA query path is correct and tested.

## Design Constraints To Preserve

- Keep GLM in the core.
- Keep Torch only in Python/Torch-facing integration.
- Do not invent new vector math structures.
- Keep octree topology separate from payload buffers.
- Use the packed `NodeDescriptor` for topology traversal.
- Keep Python bindings thin; important algorithms belong in C++/CUDA.
- Keep C++-only installation easy and independent from Python/pybind11.
