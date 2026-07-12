# Differentiability

The octree is an acceleration structure with discrete topology decisions. The
project differentiates through continuous payload computations, not through
topology creation or leaf selection.

## Differentiable

- Trilinear interpolation of payload values.
- Volume compositing.
- Density payloads.
- RGB color payloads.
- Feature payloads when exposed through supported rendering/sampling paths.

## Not Differentiable

- Occupancy thresholding.
- Topology construction.
- Tree rebuilds.
- Leaf selection.
- Ray/cell boundary crossing decisions.
- Payload index selection.

## Current Renderer Backward

The CUDA Torch renderer supports gradients for:

- `sigma`: density payload.
- `color`: RGB payload.

Depth is forward-only. Gradients are scattered back into payload tensors.

## Practical Guidance

Use a fixed topology for an optimization loop, then optimize payload values:

```python
sigma = torch.nn.Parameter(torch.ones(tree.num_leaves, device="cuda"))
color = torch.nn.Parameter(torch.rand(tree.num_leaves, 3, device="cuda"))
rgb, depth, opacity = renderer(origins, directions, sigma, color)
loss = rgb.mean()
loss.backward()
```

If occupancy or topology must change, rebuild the tree outside autograd.
