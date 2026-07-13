# Python API

The Python package is a thin pybind11 layer over the C++/CUDA core. CPU paths use
NumPy arrays. CUDA hot paths use a CUDA-owned octree and CUDA Torch tensors.

## Public Imports

```python
import svo

svo.Octree
svo.Camera
svo.CameraIntrinsics
svo.CameraConvention
svo.BranchingMode
svo.VolumeRenderer
svo.render_volume
svo.sample_trilinear
svo.gather_payload
svo.cuda_enabled
svo.build_info
```

## Build

```python
tree = svo.Octree.from_voxels(coords, max_depth=8, device="cpu")
```

Optional payload remapping:

```python
tree = svo.Octree.from_voxels(
    coords,
    max_depth=8,
    payload_indices=payload_indices,
)
```

Branching modes:

```python
octree8 = svo.Octree.from_voxels(coords, max_depth=8, branching="octree8")
wide4 = svo.Octree.from_voxels(coords, max_depth=8, branching="wide4")
```

`wide4` requires an even `max_depth`.

## Query

```python
leaf_ids = tree.query(points)
payload_indices = tree.query(points, return_payload_indices=True)
```

Input shape is `(N, 3)` with dtype `float32` or `float64`.

## CUDA-Owned Tree

```python
cuda_tree = tree.to("cuda")
```

CUDA query with Torch tensors:

```python
points = torch.rand(100_000, 3, device="cuda")
ids = cuda_tree.query(points)
```

CUDA outputs stay on GPU for Torch input.

## Raycast

```python
hit_mask, leaf_ids, t, positions, depths = tree.raycast(origins, directions)
```

`origins` and `directions` can be shaped `(N, 3)` or `(H, W, 3)`.

## Payload Gather

```python
features = torch.randn(tree.num_leaves, 16, device="cuda")
sampled = svo.gather_payload(features, leaf_ids_cuda, fill_value=0.0)
```

`gather_payload` supports NumPy and Torch tensors and uses `-1` as miss.

## Rendering

```python
rgb, depth, opacity = svo.render_volume(
    tree,
    origins,
    directions,
    sigma,
    color,
    render_strategy="direct",
)
```

CPU rendering uses NumPy. CUDA autograd rendering uses Torch CUDA tensors with a
CUDA-owned tree. `render_strategy="intervals"` is an experimental CUDA Torch
path that saves compact interval buffers for backward reuse; CPU interval
rendering is not implemented. `render_strategy="auto"` currently maps to
`"direct"`.
