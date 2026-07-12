# Installation

## Requirements

- CMake `>= 3.24`.
- C++20-capable compiler.
- Python `>= 3.10`.
- `uv` for local development.
- CUDA 12.x toolkit for CUDA source builds.
- Optional: PyTorch for CUDA tensor interop and autograd rendering.

Recommended compiler baseline for C++20 builds: GCC 11+, Clang 14+, MSVC 2022,
or an equivalent compiler supported by the selected CUDA toolkit.

## Local Development Setup

```bash
uv sync --extra test
uv run python -c "import svo; print(svo.__version__)"
```

Optional extras:

```bash
uv sync --extra viewer
uv sync --extra lint
```

## CPU Wheel Build

The default Python package build is CPU-first and produces a runtime-only wheel:

```bash
uv build
uv run python scripts/check_package.py
```

The wheel contains the Python package and native extension. It intentionally does
not include C++ headers, GLM headers, CMake package files, or the static C++
library.

## CMake Build

```bash
cmake -S . -B build -DSVO_ENABLE_CUDA=OFF -DSVO_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

## CUDA Source Build

```bash
cmake -S . -B build-cuda -DSVO_ENABLE_CUDA=ON -DSVO_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release
ctest --test-dir build-cuda --output-on-failure
```

Python editable install with CUDA enabled:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python --reinstall --no-cache -e . -C cmake.define.SVO_ENABLE_CUDA=ON
```

Example CUDA architecture configuration:

```bash
cmake -S . -B build-cuda \
  -DSVO_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"
```

## Diagnostics

```bash
python -m svo.info
```

Expected fields include package version, C++ core version, extension path, CUDA
compiled flag, Python/platform details, NumPy version, Torch availability, Torch
CUDA version, and CUDA device count.

## Compatibility Matrix

| Platform | Python | CUDA | Torch | Wheel variant | Status |
| --- | --- | --- | --- | --- | --- |
| Linux x86_64 | 3.10 | none | optional CPU | CPU wheel | tested locally |
| Linux x86_64 | 3.11, 3.12 | none | optional CPU | CPU wheel | expected-supported |
| Linux x86_64 | 3.10-3.12 | local CUDA 12.x toolkit | optional CUDA Torch matching local runtime | source build | expected-supported |
| Linux x86_64 | 3.10-3.12 | CUDA 12.x runtime | CUDA Torch | CUDA wheel | deferred |
| Windows x86_64 | 3.10-3.12 | none or CUDA 12.x | optional | wheel | deferred |
| macOS | 3.10-3.12 | unsupported | optional CPU | CPU wheel | deferred |
