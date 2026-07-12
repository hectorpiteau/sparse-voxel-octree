# Testing

Testing is part of the project contract. CUDA features should have CPU reference
coverage unless explicitly documented otherwise.

## Python Tests

```bash
./.venv/bin/python -m pytest tests/python -q
```

or:

```bash
uv run --extra test python -m pytest tests/python -q
```

## C++ CPU Tests

```bash
cmake -S . -B build-cpu -DSVO_ENABLE_CUDA=OFF -DSVO_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release -j2
ctest --test-dir build-cpu --output-on-failure
```

## CUDA Tests

```bash
cmake -S . -B build-cuda -DSVO_ENABLE_CUDA=ON -DSVO_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release -j2
ctest --test-dir build-cuda --output-on-failure
```

Python CUDA extension install:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python --reinstall --no-cache -e . -C cmake.define.SVO_ENABLE_CUDA=ON
./.venv/bin/python -m pytest tests/python -q
```

## Lint

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra lint ruff check .
```

## Package Smoke

```bash
UV_CACHE_DIR=/tmp/uv-cache uv build
./.venv/bin/python scripts/check_package.py
```

## Numerical Rules

For differentiable rendering tests:

- Avoid rays exactly on voxel boundaries.
- Use tiny deterministic scenes.
- Prefer finite differences or CPU references.
- Keep image sizes small.
- Use tolerances that account for CUDA atomics and traversal ordering.

Recommended scenes:

- Empty tree.
- Single active voxel.
- Solid cube.
- Sphere.
- Constant color field.
- Linear color field for interpolation tests.
