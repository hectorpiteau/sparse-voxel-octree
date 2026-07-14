# PyTorch Rendering

The PyTorch rendering path is designed for CUDA-resident optimization loops. A
CPU octree is built first, then copied into a CUDA-owned octree with
`tree.to("cuda")`.

## VolumeRenderer

```python
import torch
import svo

tree = svo.Octree.from_voxels(coords, max_depth=10)
cuda_tree = tree.to("cuda")

renderer = svo.VolumeRenderer(cuda_tree)

sigma = torch.nn.Parameter(torch.ones(tree.num_leaves, device="cuda"))
color = torch.nn.Parameter(torch.rand(tree.num_leaves, 3, device="cuda"))

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

## Custom Losses

```python
pred, _depth, opacity = renderer(origins, directions, sigma, color)

loss = (
    torch.nn.functional.mse_loss(pred, target)
    + 0.001 * sigma.abs().mean()
    + 0.01 * opacity.mean()
)
loss.backward()
```

## Gradient Scope

Current CUDA autograd supports gradients for density (`sigma`) and RGB color
payloads. Depth is forward-only. Topology, leaf selection, occupancy, and ray
boundary decisions are discrete and are not differentiable.

## Adaptive Topology

Adaptive reconstruction uses a discrete rebuild phase between optimization
segments. A typical loop is:

1. Optimize `sigma` and `color` with `VolumeRenderer`.
2. Pause under `torch.no_grad()`.
3. Call `svo.refine_octree(...)` with density or leaf contribution scores.
4. Replace `cuda_tree`, `sigma`, `color`, and recreate `VolumeRenderer`.
5. Continue optimization.

This follows the PlenOctrees/Plenoxels style: continuous payload values are
differentiable, but split/prune/merge topology decisions are thresholded
structure updates. V1 supports `Octree8` only and requires one payload row per
leaf.

## Render Strategies

The default strategy is direct traversal:

```python
rgb, depth, opacity = svo.render_volume(
    cuda_tree,
    origins,
    directions,
    sigma,
    color,
    render_strategy="direct",
)
```

CUDA Torch rendering also has an experimental interval strategy:

```python
renderer = svo.VolumeRenderer(cuda_tree, render_strategy="intervals")
rgb, depth, opacity = renderer(origins, directions, sigma, color)
```

Interval mode runs a non-differentiable traversal prepass that emits compact
CUDA-resident leaf intervals, composites forward from those intervals, and
reuses the saved interval buffers in backward. Gradients still flow only through
`sigma` and `color`. CPU rendering intentionally does not implement interval
mode. `render_strategy="auto"` currently maps to direct traversal until
benchmark data supports a heuristic.

## Transfer Policy

For hot paths, keep the tree and tensors on CUDA:

```python
cuda_tree = tree.to("cuda")
ids = cuda_tree.query(points_cuda)
features = svo.gather_payload(features_cuda, ids, fill_value=0.0)
rgb, depth, opacity = renderer(origins_cuda, directions_cuda, sigma, color)
```

Avoid converting CUDA tensors to CPU unless explicitly inspecting or saving
results.
