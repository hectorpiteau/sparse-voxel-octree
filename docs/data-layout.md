# Data Layout

## Coordinate Convention

Voxel coordinates are integer cells in `[0, 2^max_depth)` for each axis.
Normalized query/render coordinates use root bounds `[0, 1]^3` by default.

## Octree8 Descriptor

The default topology uses a compact 64-bit node descriptor:

```cpp
class NodeDescriptor {
    uint64_t bits;
};
```

Packed fields:

- 8-bit child mask.
- 8-bit leaf mask.
- 24-bit child base.
- 24-bit payload base.

Bitwise accessors are used on CPU and CUDA.

## Wide4 Descriptor

`branching="wide4"` uses a 4x4x4 child grid per node:

```cpp
class WideNodeDescriptor {
    uint64_t child_mask;
    uint64_t leaf_mask;
    uint32_t child_base;
    uint32_t payload_base;
};
```

The child index is:

```text
x2 | (y2 << 2) | (z2 << 4)
```

Each component is in `[0, 3]`. `wide4` consumes two coordinate bits per axis per
level, so `max_depth` must be even.

## Payload Indirection

Leaves map to user payloads through `leaf_payload_indices`:

```text
leaf_id -> payload_index
```

This allows users to compact, reorder, or share payload buffers independently of
topology.

## Current and Future Layout Phases

Current:

- Bit-packed octree8 descriptors.
- Wide4 descriptors for 64-child nodes.
- Compact child and leaf spans.
- CPU/CUDA traversal dispatch by branching mode.

Planned/future:

- Morton ordering for construction and locality.
- Compact interval generation before rendering.
- Brick leaves for dense local payloads.
- Homogeneous tile values.
- Relative child pointers.
- Page/block-local offsets.
- Optional compressed attributes.
- Serialization format.
