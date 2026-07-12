# Architecture

## Core Idea

The octree is primarily a sparse spatial index. It answers questions like:

```text
Which leaf contains this point?
Which voxel does this ray hit?
Which payload index corresponds to this leaf?
Which active cells should be sampled along this camera ray?
```

Application payloads live outside the topology in NumPy arrays, Torch tensors, or
user-managed buffers.

## Layering

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

Important algorithms should live in C++/CUDA, not only in Python.

## Topology and Payload Separation

The topology stores child masks, leaf masks, child spans, and payload indices.
Payload values are external:

```text
payload_int:       [num_leaves]
payload_rgb:       [num_leaves, 3]
payload_density:   [num_leaves]
payload_features:  [num_leaves, C]
payload_bricks:    [num_bricks, B, B, B, C]
```

This keeps one octree usable for labels, rendering, simulation, features, TSDFs,
and differentiable optimization.

## Repository Layout

```text
include/svo/   public C++ headers
src/           CPU C++ implementation
cuda/          CUDA kernels
python/        pybind11 bindings
svo/           Python package helpers
tests/         C++ and Python tests
examples/      Python and C++ examples
scripts/       package validation scripts
docs/          Markdown documentation
```

## Memory Ownership

C++ objects should clearly own or borrow buffers. Python bindings must not keep
dangling pointers to Python-owned memory unless they also keep the owning Python
object alive.

Important ownership categories:

- Owned host memory.
- Owned device memory.
- Borrowed device pointers.
- Borrowed Torch tensors.
- Borrowed NumPy arrays.

## Failure Modes

Bad inputs should fail with useful errors:

- Non-contiguous tensor or array.
- Wrong dtype.
- CPU tensor passed to CUDA-only function.
- Unsupported device.
- Mismatched payload shape.
- Invalid voxel coordinates.
- CUDA architecture or CUDA build unavailable.
