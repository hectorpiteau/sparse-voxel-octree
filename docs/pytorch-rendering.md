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
