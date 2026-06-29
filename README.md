# Sparse Voxel Octree CUDA

A C++/CUDA sparse voxel octree library with Python bindings and optional PyTorch autograd integration.

This project is intended to become both:

1. A reusable C++/CUDA package for applications that need fast sparse spatial indexing, ray traversal, voxel queries, and rendering.
2. A Python package that exposes the same core functionality through an ergonomic API usable with NumPy, PyTorch, and other Python tools.

The long-term goal is to provide a compact sparse voxel octree acceleration structure that can act as a spatial index into external payload tensors or buffers. This allows users to store arbitrary per-voxel data such as integer labels, densities, RGB colors, normals, learned features, TSDF values, semantic IDs, or application-specific structs.

The design is inspired by the NVIDIA sparse voxel octree work by Laine and Karras, especially the separation between octree topology, traversal, and attribute attachments. This project adapts that idea for a modern C++/CUDA + Python + PyTorch workflow.

---

## Project status

This project is in the planning and early implementation stage.

The first implementation should prioritize correctness, clean APIs, and testability while still using a paper-style bit-packed node descriptor for octree topology. More advanced features such as block/page streaming and additional compression layers can be added later once the basic package, tests, and bindings are stable.

Initial target:

- C++17 / CUDA core.
- GLM for C++ vector and matrix math in the core library.
- Python bindings using `pybind11` or `nanobind`.
- Build system based on CMake and `scikit-build-core`.
- `uv` as the default Python project and packaging frontend.
- Optional PyTorch CUDA extension with custom forward and backward kernels.
- CPU reference implementation for tests.
- CUDA implementation for production workloads.

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
    traversal, query, render_forward, render_backward, gather, scatter

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

## Planned architecture

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
      Renderer.hpp
      Query.hpp
      Math.hpp
      TensorView.hpp
      DeviceBuffer.hpp
      Error.hpp

  src/
    Octree.cpp
    Builder.cpp
    Renderer.cpp
    Query.cpp
    DeviceBuffer.cpp

  cuda/
    query_points.cu
    raycast.cu
    render_forward.cu
    render_backward.cu
    interpolate.cu
    gather.cu
    scatter.cu

  python/
    bindings.cpp
    torch_bindings.cpp

  svo/
    __init__.py
    torch.py
    info.py

  tests/
    cpp/
      test_octree.cpp
      test_builder.cpp
      test_raycast.cpp

    python/
      test_import.py
      test_build_tree.py
      test_query_cpu.py
      test_query_cuda.py
      test_raycast.py
      test_render_forward.py
      test_gradcheck.py

  examples/
    python/
      point_query.py
      torch_render.py
      optimize_density.py

    cpp/
      query_example.cpp
      raycast_example.cpp
```

---

## C++ API sketch

The C++ API should be usable without Python.

```cpp
#include <svo/Octree.hpp>
#include <svo/Builder.hpp>
#include <svo/Query.hpp>

int main() {
    std::vector<glm::ivec3> coords = {
        {0, 0, 0},
        {1, 2, 3},
        {4, 4, 4},
    };

    svo::BuildOptions options;
    options.max_depth = 8;
    options.device = svo::Device::CUDA;

    svo::Octree tree = svo::Octree::from_voxels(coords, options);

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

    Device device() const;

    const DeviceBuffer<NodeDescriptor>& nodes() const;
    const DeviceBuffer<uint32_t>& leaf_payload_indices() const;

    void to_cuda(int device_id = 0);
    void to_cpu();

    void validate() const;
};
```

---

## Python API sketch

The Python API should be simple enough for interactive use.

```python
import torch
import svo

coords = torch.tensor(
    [[0, 0, 0], [1, 2, 3], [4, 4, 4]],
    dtype=torch.int32,
    device="cuda",
)

tree = svo.Octree.from_voxels(coords, max_depth=8)

points = torch.rand(100_000, 3, device="cuda")
leaf_ids = tree.query(points)

features = torch.randn(tree.num_leaves, 16, device="cuda")
sampled = features[leaf_ids.clamp_min(0)]
```

The API should also allow CPU NumPy usage for small data and tests.

```python
import numpy as np
import svo

coords = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int32)
tree = svo.Octree.from_voxels(coords, max_depth=4, device="cpu")

points = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
leaf_ids = tree.query(points)
```

---

## PyTorch rendering API sketch

The differentiable renderer should be exposed as a normal PyTorch module.

```python
import torch
import svo.torch as svo_torch

tree = svo_torch.Octree.from_voxels(coords, max_depth=10)

renderer = svo_torch.OctreeRenderer(
    tree,
    brick_size=8,
    step_mode="adaptive",
)

density = torch.nn.Parameter(
    torch.zeros(tree.num_bricks, 8, 8, 8, 1, device="cuda")
)

color = torch.nn.Parameter(
    torch.rand(tree.num_bricks, 8, 8, 8, 3, device="cuda")
)

camera = svo_torch.Camera.look_at(
    eye=[0.0, 0.0, 3.0],
    target=[0.0, 0.0, 0.0],
    up=[0.0, 1.0, 0.0],
    fov_degrees=60.0,
)

image, aux = renderer(
    density=density,
    color=color,
    camera=camera,
    width=800,
    height=800,
)

loss = image.mean()
loss.backward()
```

The renderer should allow custom losses in normal PyTorch.

```python
pred, aux = renderer(density, color, camera, width=W, height=H)

loss = (
    torch.nn.functional.mse_loss(pred, target)
    + 0.001 * density.abs().mean()
)

loss.backward()
```

---

## Differentiable rendering design

The renderer should use the octree as an acceleration structure and differentiate through the continuous parts of the computation.

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
- C++17 compiler.
- CUDA toolkit, for CUDA builds.
- Python >= 3.9.
- `uv`.
- Optional: PyTorch for `svo.torch`.

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
- `pybind11` or `nanobind`
- `cibuildwheel` for CI wheel builds

Initial wheels should target a small stable matrix:

```text
Linux x86_64, Python 3.10-3.12, CUDA 12.x
Windows x86_64, Python 3.10-3.12, CUDA 12.x
macOS CPU-only, optional later
```

Do not attempt to support every CUDA version immediately.

The project should publish a source distribution so advanced users can compile for their own CUDA version and GPU architecture.

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
CUDA extension loaded: yes
compiled CUDA version: 12.x
available CUDA devices: 1
device 0: NVIDIA ...
compute capability: ...
torch available: yes
torch CUDA version: ...
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

Choose a permissive license such as MIT or Apache-2.0 unless the project needs stricter terms.

Before copying code from any external repository, verify license compatibility.

---

## References

- Samuli Laine and Tero Karras, *Efficient Sparse Voxel Octrees – Analysis, Extensions, and Implementation*, NVIDIA Technical Report NVR-2010-001, 2010.
- Amanatides and Woo, *A Fast Voxel Traversal Algorithm for Ray Tracing*, 1987.
- PyTorch C++/CUDA extension documentation.
- uv documentation.
- scikit-build-core documentation.
- cibuildwheel documentation.
