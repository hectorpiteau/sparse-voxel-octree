# Sparse Voxel Octree CUDA

[![CI](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/ci.yml)
[![GPU CI](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/gpu.yml/badge.svg?branch=main)](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/gpu.yml)
[![Publish](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/publish.yml/badge.svg)](https://github.com/hectorpiteau/sparse-voxel-octree/actions/workflows/publish.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/hectorpiteau/sparse-voxel-octree/blob/main/pyproject.toml)
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue)](https://github.com/hectorpiteau/sparse-voxel-octree/blob/main/CMakeLists.txt)
[![CUDA](https://img.shields.io/badge/CUDA-12.x%20source%20builds-76B900)](https://github.com/hectorpiteau/sparse-voxel-octree#compatibility-matrix)
[![PyPI](https://img.shields.io/badge/PyPI-planned-lightgrey)](https://github.com/hectorpiteau/sparse-voxel-octree#packaging-plan)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
- [Core idea](#core-idea)
- [Branching modes](#branching-modes)
- [Forward renderer example](#forward-renderer-example)
- [Real-time viewer](#real-time-viewer)
- [CPU raycast behavior](#cpu-raycast-behavior)
- [Design principles for coding agents](#design-principles-for-coding-agents)
- [Repository layout](#repository-layout)
- [C++ API sketch](#c-api-sketch)
- [Python API sketch](#python-api-sketch)
- [PyTorch rendering API sketch](#pytorch-rendering-api-sketch)
- [Differentiable rendering design](#differentiable-rendering-design)
- [Data layout phases](#data-layout-phases)
- [Building from source](#building-from-source)
- [Packaging plan](#packaging-plan)
- [CI and release workflow](#ci-and-release-workflow)
- [Testing](#testing)
- [Diagnostics](#diagnostics)
- [Non-goals for the first version](#non-goals-for-the-first-version)
- [References](#references)

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
- Tree-level branching modes: classic 8-way octree and experimental 4x4x4 wide nodes.
- CPU-first runtime-only Python wheels, source distributions, package smoke tests, and GitHub Actions CI.

Still in progress:

- Public CUDA wheels.
- Documentation split into dedicated pages.
- Serialization.
- Production rendering acceleration beyond the current traversal kernels.
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

---

## Core idea

The octree should be treated primarily as a sparse spatial index.

The octree answers questions like:

```text
Which leaf contains this point?
Which voxel does this ray hit?
Which brick or payload index corresponds to this leaf?
Which active cells should be sampled along this camera ray?
```

The actual voxel payload should live in separate buffers or tensors.

## Branching modes

`Octree` supports two tree-level topology layouts:

- `branching="octree8"` is the default 2x2x2 octree path. It uses the original compact 64-bit node descriptor and supports all current features, including trilinear interpolation.
- `branching="wide4"` uses 4x4x4 nodes with 64-bit child and leaf masks plus compact child/payload spans. It consumes two coordinate bits per axis per level, so `max_depth` must be even. Query, raycast, forward rendering, and CUDA/Torch backward rendering dispatch to dedicated wide kernels.

Example:

```python
tree = svo.Octree.from_voxels(coords, max_depth=8, branching="wide4")
assert tree.branching == "wide4"
```

Choose `wide4` when lower traversal depth matters and the scene is sparse enough that wider node descriptors pay for themselves. Keep `octree8` when memory density is the priority, when `max_depth` is odd, or when using interpolation features that have not been generalized to wide nodes yet.

## Forward renderer example

The forward renderer can render external density/color payloads indexed by the octree. This example builds a sparse sphere, colors it with smooth sinusoidal payloads, generates camera rays, and writes a PNG without extra visualization dependencies. Use `--device auto` to use CUDA Torch rendering when available and CPU otherwise.

```bash
./.venv/bin/python examples/python/forward_render.py --device auto --output docs/assets/forward_render.png
```

![Forward render of a colored sparse sphere](docs/assets/forward_render.png)

## Real-time viewer

`examples/python/realtime_viewer.py` opens a small uncapped pygame viewer for quick renderer performance checks. It builds the same style of sparse sphere in the center of the scene, uses CUDA Torch rendering when available with `--device auto`, and falls back to CPU otherwise.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra viewer
./.venv/bin/python examples/python/realtime_viewer.py --device auto
./.venv/bin/python examples/python/realtime_viewer.py --device cuda --branching wide4
```

Controls: left-click drag orbits the camera around `(0, 0, 0)`, the mouse wheel zooms, `R` resets the camera, and `Q` or `Esc` exits. FPS and frame time are displayed in the top-left corner. The display loop is not frame-capped; GPU-to-CPU readback is still required to show the rendered image in the pygame window.

## CPU raycast behavior

`Octree.raycast(origins, directions)` accepts NumPy ray batches shaped `(N, 3)` or `(H, W, 3)` and returns `(hit_mask, leaf_ids, t, positions, depths)` with matching leading dimensions. Directions are normalized internally, so `t` is distance along the normalized ray direction and hit positions are `origin + t * normalized_direction`.

Misses use explicit sentinels: `hit_mask=false`, `leaf_id=-1`, `depth=-1`, `t=inf`, and `position=(nan, nan, nan)`. Boundary ties are deterministic: nearest `t` wins, then smaller leaf id.

Example payloads:

```text
payload_int:       [num_leaves]
payload_rgb:       [num_leaves, 3]
payload_density:   [num_leaves, 1]
payload_features:  [num_leaves, C]
payload_bricks:    [num_bricks, B, B, B, C]
```

This makes the same octree usable for rendering, simulation, ML features, point queries, ray queries, and differentiable volume rendering.

---

## Design principles for coding agents

Coding agents working on this repository should follow these rules.

### 1. Keep the core implementation in C++/CUDA

Do not put important algorithms only in Python.

Python should be a thin usability layer around the C++/CUDA implementation.

Correct layering:

```text
CUDA kernels
    traversal, query, renderer forward/backward, interpolation, gather/scatter helpers

C++ library
    memory ownership, octree object, builder, validation, API types

Python bindings
    ergonomic classes, NumPy/Torch conversion, error messages

Torch wrapper
    torch.autograd.Function and torch.nn.Module wrappers
```

### 2. Keep topology separate from payload

The octree topology must not assume a fixed payload type.

Good:

```cpp
uint32_t payload_index;
uint64_t payload_byte_offset;
```

Avoid hardcoding:

```cpp
float density;
glm::vec3 color;
```

The octree should be able to index arbitrary external payload buffers.

### 3. Prefer explicit memory ownership

C++ objects should clearly own or borrow buffers.

Use separate types for:

- Owned host memory.
- Owned device memory.
- Borrowed device pointers.
- Borrowed Torch tensors.
- Borrowed NumPy arrays.

Do not silently keep dangling pointers to Python-owned data unless the Python binding also keeps the owner object alive.

### 4. Prioritize a simple correct layout first

Do not invent a temporary node layout that will later be thrown away.

Start with a paper-style bit-packed descriptor as the actual octree node representation, but keep the surrounding algorithms simple and testable.

```cpp
struct NodeDescriptor {
    uint64_t bits;
};
```

Use bitwise accessors to extract masks, child addressing information, and payload indirection fields. Delay page streaming, contour data, and other advanced compression layers until after correctness and tests exist.

### 5. Reuse established math types

Do not add custom vector math structs such as `float3`, `VoxelCoord`, or bespoke AABB types in the C++ core.

Use:

- `glm::ivec3` for integer voxel coordinates.
- `glm::vec3` for points, directions, colors, and bounds.
- `glm::mat*` where matrix math is needed.

Torch-specific tensor and vector types should stay in the Torch integration layer, not the standalone core C++ API.

### 6. Every CUDA kernel needs a CPU reference test

For each CUDA operation, add a small CPU reference implementation and compare outputs.

Required comparisons:

- Point query CPU vs CUDA.
- Raycast CPU vs CUDA.
- Forward render CPU vs CUDA for tiny scenes.
- Interpolation CPU vs CUDA.
- Backward gradients vs numerical or PyTorch reference.

### 7. Avoid global state

No global current tree, global CUDA stream, or global device assumptions.

APIs should accept device, stream, and context where needed.

### 8. Make failure modes explicit

Bad inputs should fail with useful errors.

Examples:

- Non-contiguous tensor.
- Wrong dtype.
- CPU tensor passed to CUDA-only function.
- Unsupported device.
- Mismatched payload shape.
- Query points outside root bounds.
- Tree built with invalid coordinates.
- CUDA architecture not supported.

### 9. Keep differentiability scoped

The octree traversal and topology decisions are discrete. The project should not claim that octree topology is differentiable.

Differentiable parts:

- Trilinear interpolation of payload values.
- Volume compositing.
- Color and density payloads.
- Feature payloads.
- Camera parameters, if explicitly implemented.
- Sample positions inside fixed cells, if explicitly implemented.

Non-differentiable or piecewise-discontinuous parts:

- Leaf selection.
- Occupancy thresholding.
- Topology creation.
- Tree rebuilds.
- Ray/cell boundary crossing decisions.

---

## Repository layout

```text
sparse-voxel-octree/
  CMakeLists.txt
  pyproject.toml
  README.md
  TODO.md
  LICENSE

  include/
    svo/
      Octree.hpp
      Builder.hpp
      Camera.hpp
      Renderer.hpp
      Query.hpp
      Raycast.hpp
      Interpolation.hpp
      Math.hpp
      DeviceBuffer.hpp
      Error.hpp
      Version.hpp

  src/
      Octree.cpp
      Builder.cpp
      Camera.cpp
      Renderer.cpp
      Query.cpp
      Raycast.cpp
      Interpolation.cpp
      Version.cpp

  cuda/
    generate_rays.cu
    query_points.cu
    raycast.cu
    renderer.cu
    interpolation.cu

  python/
    bindings.cpp

  svo/
    __init__.py
    info.py
    interpolation.py
    payload.py
    rendering.py

  scripts/
    check_package.py

  tests/
    cpp/
      test_builder.cpp
      test_query.cpp
      test_raycast.cpp
      test_renderer.cpp
      test_camera.cpp

    python/
      test_import.py
      test_build_tree.py
      test_query_cpu.py
      test_raycast.py
      test_rendering.py
      test_torch_cuda_interop.py

  examples/
    python/
      point_query.py
      forward_render.py
      realtime_viewer.py
      sphere_slice.py

    cpp/
      query_example.cpp
```

---

## C++ API sketch

The C++ API should be usable without Python.

```cpp
#include <svo/Octree.hpp>
#include <svo/Builder.hpp>
#include <svo/Query.hpp>

#include <cstdint>
#include <vector>

int main() {
    std::vector<glm::ivec3> coords = {
        {0, 0, 0},
        {1, 2, 3},
        {4, 4, 4},
    };

    svo::BuildOptions options;
    options.max_depth = 8;
    options.device = svo::Device::CPU;

    svo::Octree tree = svo::build_octree_cpu(coords, options);

    std::vector<glm::vec3> points = {
        {0.1f, 0.1f, 0.1f},
        {0.5f, 0.5f, 0.5f},
    };

    std::vector<int32_t> leaf_ids = svo::query_points(tree, points);

    return 0;
}
```

The C++ API should expose enough information for applications to use the octree as a spatial index.

Important methods:

```cpp
class Octree {
public:
    int max_depth() const;
    int64_t num_nodes() const;
    int64_t num_leaves() const;
    std::array<glm::vec3, 2> root_bounds() const;
    BranchingMode branching() const;

    Device device() const;

    const std::vector<NodeDescriptor>& nodes() const;
    const std::vector<WideNodeDescriptor>& wide_nodes() const;
    const std::vector<uint32_t>& leaf_payload_indices() const;

    void validate() const;
};
```

---

## Python API sketch

The Python API is meant to be simple enough for interactive use.

```python
import numpy as np
import svo

coords = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int32)
tree = svo.Octree.from_voxels(coords, max_depth=4, device="cpu")

points = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
leaf_ids = tree.query(points)
```

CUDA query/raycast hot paths use a CUDA-owned tree and CUDA Torch tensors:

```python
import torch

cuda_tree = tree.to("cuda")
points_cuda = torch.rand(100_000, 3, device="cuda")
leaf_ids_cuda = cuda_tree.query(points_cuda)

features = torch.randn(tree.num_leaves, 16, device="cuda")
sampled = svo.gather_payload(features, leaf_ids_cuda, fill_value=0.0)
```

---

## PyTorch rendering API sketch

The differentiable renderer is exposed as a normal PyTorch module.

```python
import torch
import svo

tree = svo.Octree.from_voxels(coords, max_depth=10)
cuda_tree = tree.to("cuda")

renderer = svo.VolumeRenderer(cuda_tree)

sigma = torch.nn.Parameter(
    torch.ones(tree.num_leaves, device="cuda")
)
color = torch.nn.Parameter(
    torch.rand(tree.num_leaves, 3, device="cuda")
)

camera = svo.Camera.look_at(
    origin=[0.0, 0.0, 3.0],
    target=[0.0, 0.0, 0.0],
    up=[0.0, 1.0, 0.0],
    width=800,
    height=800,
    vertical_fov_y_degrees=60.0,
)
origins_np, directions_np = camera.generate_rays()
origins = torch.as_tensor(origins_np, device="cuda")
directions = torch.as_tensor(directions_np, device="cuda")

rgb, depth, opacity = renderer(origins, directions, sigma, color)

loss = rgb.mean() + 0.001 * opacity.mean()
loss.backward()
```

The renderer allows custom losses in normal PyTorch.

```python
pred, _depth, opacity = renderer(origins, directions, sigma, color)

loss = (
    torch.nn.functional.mse_loss(pred, target)
    + 0.001 * sigma.abs().mean()
    + 0.01 * opacity.mean()
)

loss.backward()
```

---

## Differentiable rendering design

The renderer uses the octree as an acceleration structure and differentiates through the continuous parts of the computation.

Forward path:

```text
camera config
    -> generate rays
    -> traverse sparse octree
    -> find active leaves or bricks
    -> sample density/color/features
    -> trilinear interpolation
    -> alpha compositing
    -> image/depth/opacity
```

Backward path:

```text
image gradient
    -> compositing backward
    -> interpolation backward
    -> scatter-add into density/color/feature tensors
    -> optional gradient wrt camera parameters
```

The first differentiable implementation should support gradients with respect to:

- Color payload.
- Density payload.
- Feature payload, if features are rendered or passed to a shading function.

Later versions may support gradients with respect to:

- Camera extrinsics.
- Camera intrinsics.
- Sample positions.
- Deformation fields.

Topology gradients are out of scope for the initial project.

---

## Data layout phases

### Phase 1: bit-packed node descriptor

Use a paper-style node descriptor from the beginning:

```cpp
struct NodeDescriptor {
    uint64_t bits;
};
```

Properties:

- Single source of truth for octree topology.
- Bitwise extraction works on CPU and CUDA.
- Closer to the paper's representation.
- Avoids a rewrite from an unrelated temporary node struct.

### Phase 2: linear octree with Morton ordering

Add Morton sorting for construction and better spatial locality.

Useful concepts:

- Morton code per occupied voxel.
- Radix sort.
- Prefix tree construction.
- Linearized node arrays.
- Compact child spans.

### Phase 3: payload indirection and bricks

Add leaf-to-payload mapping.

```text
leaf_id -> payload_index
leaf_id -> brick_index
leaf_id -> brick local transform
```

Brick payloads are recommended for differentiable interpolation.

```text
density_bricks: [num_bricks, B, B, B, 1]
color_bricks:   [num_bricks, B, B, B, 3]
feature_bricks: [num_bricks, B, B, B, C]
```

### Phase 4: advanced descriptor and streaming features

After correctness and tests are stable, extend the base descriptor system with more advanced sparse voxel octree ideas.

Possible optimizations:

- 64-bit descriptor format.
- Relative child pointers.
- Page/block-local offsets.
- Contour payloads.
- Compressed attributes.
- Optional block streaming.

---

## Building from source

Requirements:

- CMake >= 3.24.
- C++20-capable compiler.
- CUDA 12.x toolkit, for CUDA builds.
- Python >= 3.10.
- `uv`.
- Optional: PyTorch for CUDA tensor interop and autograd rendering.

Recommended compiler baseline for C++20 builds: GCC 11+, Clang 14+, MSVC 2022,
or an equivalent compiler supported by the selected CUDA toolkit. CUDA device
code should stay conservative with C++20 standard library features unless a
specific use is verified with `nvcc`.

Set up the project environment:

```bash
uv sync --extra test
```

Run commands inside the project environment:

```bash
uv run python -c "import svo; print(svo.__version__)"
```

Build distributions for PyPI:

```bash
uv build
```

Validate built package artifacts and install the wheel in a clean temporary
environment:

```bash
uv run python scripts/check_package.py
```

Build with CMake directly:

```bash
cmake -S . -B build -DSVO_ENABLE_CUDA=ON -DSVO_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

Example CUDA architecture configuration:

```bash
cmake -S . -B build \
  -DSVO_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"
```

---

## Packaging plan

The Python package should be built using:

- `uv`
- `scikit-build-core`
- CMake
- `pybind11`
- GitHub Actions for CPU wheel builds and release publishing

Initial wheels should target a small stable matrix:

```text
Linux x86_64, Python 3.10-3.12, CPU-only wheel first
Linux x86_64, Python 3.10-3.12, CUDA 12.x source builds
Windows x86_64, Python 3.10-3.12, CUDA 12.x wheels later
macOS CPU-only wheels later
```

Do not attempt to support every CUDA version immediately.

The project should publish a source distribution so advanced users can compile for their own CUDA version and GPU architecture.

### Compatibility matrix

| Platform | Python | CUDA | Torch | Wheel variant | Status |
| --- | --- | --- | --- | --- | --- |
| Linux x86_64 | 3.10 | none | optional CPU | CPU wheel | tested locally |
| Linux x86_64 | 3.11, 3.12 | none | optional CPU | CPU wheel | expected-supported |
| Linux x86_64 | 3.10-3.12 | local CUDA 12.x toolkit | optional CUDA Torch matching local runtime | source build | expected-supported |
| Linux x86_64 | 3.10-3.12 | CUDA 12.x runtime | CUDA Torch | CUDA wheel | deferred to CI/release milestone |
| Windows x86_64 | 3.10-3.12 | none or CUDA 12.x | optional | wheel | deferred |
| macOS | 3.10-3.12 | unsupported | optional CPU | CPU wheel | deferred |

The PyPI wheel is a Python runtime artifact. It contains the Python package and
native extension, but not C++ headers, GLM headers, CMake package files, or the
static C++ library. C++ users should build/install from source with CMake.

CPU wheel install:

```bash
pip install sparse-voxel-octree
python -m svo.info
```

CUDA source build:

```bash
pip install . -Ccmake.define.SVO_ENABLE_CUDA=ON
python -m svo.info
```

Public CUDA wheel variants are intentionally deferred until the CI and release
matrix can test Python, CUDA toolkit/runtime, NVIDIA driver, OS, architecture,
and Torch compatibility together.

---

## CI and release workflow

Pull requests run the CPU-focused validation set:

- Ruff lint checks for Python/package scripts.
- Python tests on Python 3.10, 3.11, and 3.12.
- CPU CMake configure/build/test.
- CPU wheel and source distribution build, package artifact checks, and clean wheel install smoke.

CUDA validation is available through the `GPU CI` workflow. It runs manually or
when a pull request has the `gpu-ci` label, and expects a self-hosted runner with
the labels `self-hosted`, `linux`, `x64`, and `cuda`.

Publishing uses PyPI Trusted Publishing. Configure GitHub environments named
`testpypi` and `pypi`, then add this repository as a trusted publisher on
TestPyPI/PyPI for the corresponding workflow and environment.

Release checklist:

1. Review whether the change breaks the public Python or C++ API contract.
2. Choose the next version: patch for fixes/docs/tests/perf, minor for backward-compatible features or pre-1.0 breaking changes, major for post-1.0 breaking changes.
3. Run `uv run --extra lint ruff check .`.
4. Run `uv build` and `uv run python scripts/check_package.py`.
5. Run Python and CMake tests.
6. Create and push an annotated tag like `v0.1.0`.
7. Use the publish workflow manually for TestPyPI, or let a `v*` tag publish to PyPI after environment approval.

---

## Testing

Testing is a core part of this project.

Run Python tests:

```bash
uv run pytest tests/python -q
```

Run C++ tests:

```bash
ctest --test-dir build --output-on-failure
```

Run CUDA sanitizer tests when a GPU runner is available:

```bash
uv run compute-sanitizer python -m pytest tests/python -q
```

Recommended test categories:

1. Import tests.
2. CPU tree construction tests.
3. CUDA tree construction tests.
4. CPU vs CUDA point query tests.
5. CPU vs CUDA raycast tests.
6. Tiny-scene render tests.
7. PyTorch autograd tests.
8. Gradcheck tests.
9. Serialization tests.
10. Packaging tests from built wheels.

---

## Numerical testing rules

For differentiable rendering, avoid tests where rays lie exactly on voxel boundaries.

Use tiny stable scenes:

- Single active voxel.
- Solid cube.
- Plane.
- Empty scene.
- One brick with known density.
- Constant color field.
- Linear color field, useful for interpolation tests.

For gradient tests:

- Use double precision where possible.
- Use small image sizes.
- Use deterministic camera rays.
- Avoid discontinuity boundaries.
- Compare against CPU reference or finite differences.

---

## Diagnostics

The package should expose a diagnostic command:

```bash
python -m svo.info
```

Expected output:

```text
svo version: 0.1.0
C++ core version: 0.1.0
extension path: ...
CUDA extension loaded: yes
python version: 3.10.20
python executable: ...
platform: ...
machine: x86_64
numpy version: ...
torch available: yes
torch version: ...
torch CUDA available: yes
torch CUDA version: ...
torch CUDA device count: 1
torch CUDA device 0: NVIDIA ...
torch CUDA interop available: yes
```

This is important for debugging user installation issues.

---

## Non-goals for the first version

The first version should not attempt to solve everything.

Out of scope initially:

- Full block/page streaming system around the descriptor format.
- Differentiable topology construction.
- Dynamic tree mutation during autograd.
- Multi-GPU training.
- Out-of-core streaming.
- Production-quality real-time renderer.
- Full material system.
- Vulkan/OpenGL/DirectX interop.
- AMD GPU support.

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
