# Sparse Voxel Octree CUDA

[![CI](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/ci.yml)
[![Docs](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/docs.yml)
[![GPU CI](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/gpu.yml/badge.svg?branch=main)](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/gpu.yml)
[![Publish](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/publish.yml/badge.svg)](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/publish.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/hectorpiteau/sparse-voxel-octree/blob/main/pyproject.toml)
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue)](https://github.com/hectorpiteau/sparse-voxel-octree/blob/main/CMakeLists.txt)
[![CUDA](https://img.shields.io/badge/CUDA-12.x%20source%20builds-76B900)](docs/installation.md#compatibility-matrix)
[![PyPI](https://img.shields.io/badge/PyPI-planned-lightgrey)](docs/packaging.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Documentation: <https://hectorpiteau.github.io/sparse-voxel-octree/>

Sparse voxel octree library with a C++20/CUDA core, pybind11 Python bindings,
NumPy CPU APIs, CUDA traversal/rendering, and optional PyTorch CUDA autograd
integration.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/d6cd63fe-c4c8-4abd-92f6-e7ad734972bd">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/a5d8303a-4519-4af8-853d-fab9667387aa">

  <img alt="Sparse Voxel Octree CUDA logo" src="https://github.com/user-attachments/assets/a5d8303a-4519-4af8-853d-fab9667387aa">
</picture>

## Contents

- [Project status](#project-status)
- [What the project does](#what-the-project-does)
- [Quickstart](#quickstart)
- [Documentation](#documentation)
- [Examples](#examples)
- [Development](#development)
- [Contributing](#contributing)
- [Packaging and CI](#packaging-and-ci)
- [References](#references)
- [License](#license)

---

## Project status

The project is a working pre-alpha implementation. The CPU reference path,
CUDA traversal/rendering kernels, Python package build, and CI workflows exist.
The API is still allowed to change before `1.0.0`, but changes should now be
classified as patch/minor/major according to API compatibility.

Current capabilities:

- C++20 core with GLM math types.
- pybind11 Python extension built with CMake and `scikit-build-core`.
- CPU octree construction from voxel coordinates.
- CPU and CUDA point query, camera ray generation, raycast, trilinear interpolation, and forward volume rendering.
- CUDA/PyTorch tensor interop for query, raycast, payload gathering, interpolation, and differentiable volume rendering.
- `svo.VolumeRenderer`, a small `torch.nn.Module` wrapper around the CUDA renderer.
- Tree-level branching modes: classic 8-way octree and experimental 4x4x4 wide nodes with local DDA raycast/render traversal.
- CPU-first runtime-only Python wheels, source distributions, package smoke tests, and GitHub Actions CI.

Still in progress:

- Public CUDA wheels.
- Serialization.
- Further production rendering acceleration beyond the current Wide4 DDA traversal.
- Advanced sparse layouts such as DDA interval generation, brick leaves, and VDB-style hierarchy.

## What the project does

This library provides a sparse spatial index over occupied voxel coordinates.
The tree stores topology and leaf-to-payload indirection; application data lives
in external arrays or tensors. That keeps the same octree useful for labels,
density/color fields, learned features, TSDF values, semantic IDs, or
application-specific payloads.

Typical use cases:

- Build an octree from sparse occupied voxels.
- Query which leaf or payload index contains a point.
- Raycast through sparse voxel topology.
- Render density/color payloads from camera rays.
- Use CUDA-resident tensors without avoidable CPU-GPU transfers.
- Optimize density/color payload tensors with PyTorch autograd.

The design is inspired by Laine and Karras' sparse voxel octree work, especially
the separation between topology, traversal, and attributes, adapted for a modern
C++/CUDA + Python + PyTorch workflow.

## Quickstart

Set up the local development environment:

```bash
uv sync --extra test
```

Build a tree and query it from Python:

```python
import numpy as np
import svo

coords = np.array([[0, 0, 0], [1, 2, 3], [4, 4, 4]], dtype=np.int32)
tree = svo.Octree.from_voxels(coords, max_depth=4, branching="octree8")

points = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]], dtype=np.float32)
leaf_ids = tree.query(points)
```

Use `branching="wide4"` for the experimental 4x4x4 topology. Wide trees require
an even `max_depth`; see [Data layout](docs/data-layout.md) for the descriptor
format and tradeoffs.

Run package diagnostics:

```bash
uv run python -m svo.info
```

## Documentation

The README is intentionally a short entry point. The full documentation is
available online at <https://hectorpiteau.github.io/sparse-voxel-octree/> and
as Markdown sources under `docs/`.

| Topic | Document |
| --- | --- |
| Documentation map | [docs/index.md](docs/index.md) |
| Installation, CUDA source builds, compatibility matrix | [docs/installation.md](docs/installation.md) |
| First Python workflow | [docs/quickstart.md](docs/quickstart.md) |
| Python API | [docs/python-api.md](docs/python-api.md) |
| C++ API | [docs/cpp-api.md](docs/cpp-api.md) |
| PyTorch rendering and autograd | [docs/pytorch-rendering.md](docs/pytorch-rendering.md) |
| Architecture and contributor guidance | [docs/architecture.md](docs/architecture.md) |
| Octree and wide-node data layout | [docs/data-layout.md](docs/data-layout.md) |
| Differentiability scope | [docs/differentiability.md](docs/differentiability.md) |
| Testing and numerical checks | [docs/testing.md](docs/testing.md) |
| Packaging, versioning, CI, release flow | [docs/packaging.md](docs/packaging.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Examples | [docs/examples.md](docs/examples.md) |
| Documentation workflow and Docsify preview | [docs/documentation.md](docs/documentation.md) |

Regenerate the Docsify shell after adding, removing, or renaming pages:

```bash
./.venv/bin/python scripts/generate_docsify.py
```

Preview the documentation site locally:

```bash
npx docsify-cli serve docs
```

## Examples

Render a sparse sphere with sinusoidal color payloads:

```bash
./.venv/bin/python examples/python/forward_render.py --device auto --output docs/assets/forward_render.png
```

![Forward render of a colored sparse sphere](docs/assets/forward_render.png)

Open the lightweight real-time viewer:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra viewer
./.venv/bin/python examples/python/realtime_viewer.py --device auto
./.venv/bin/python examples/python/realtime_viewer.py --device cuda --branching wide4
```

The viewer orbits around `(0, 0, 0)` with left-click drag, zooms with the mouse
wheel, resets with `R`, and exits with `Q` or `Esc`. FPS and frame time are
displayed in the window.

More examples are listed in [docs/examples.md](docs/examples.md).

## Development

Run the common local checks:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra lint ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run --extra test pytest tests/python -q
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure
```

Build and validate the Python package:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv build
./.venv/bin/python scripts/check_package.py
```

See [docs/testing.md](docs/testing.md) for CPU/CUDA test policy and
[docs/architecture.md](docs/architecture.md) for contributor-facing design
rules.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. It
covers project design constraints, test and documentation expectations, and the
information every PR should provide so reviewers can understand why the change
matters.

## Packaging and CI

The project is CPU-wheel-first. Wheels are runtime-only and should contain the
Python package plus native extension, not C++ headers, GLM headers, CMake
package files, or the static library. C++ users should build from source with
CMake.

CI runs Linux CPU lint, Python tests, CMake tests, and package checks. Optional
CUDA validation is handled by a self-hosted GPU workflow. Publishing uses
Trusted Publishing with `vX.Y.Z` tags generated after an explicit patch/minor/
major API compatibility decision.

See [docs/packaging.md](docs/packaging.md) for release details.

---

## License

This project is licensed under the MIT License.

Before copying code from any external repository, verify license compatibility.

---

## References

- Samuli Laine and Tero Karras, *Efficient Sparse Voxel Octrees – Analysis, Extensions, and Implementation*, NVIDIA Technical Report NVR-2010-001, 2010.
- Amanatides and Woo, *A Fast Voxel Traversal Algorithm for Ray Tracing*, 1987.
- PyTorch C++/CUDA extension documentation.
- uv documentation.
- scikit-build-core documentation.
