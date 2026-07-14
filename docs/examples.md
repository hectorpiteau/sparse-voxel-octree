# Examples

This page lists runnable examples that currently exist in the repository. More
specialized scripts, such as payload optimization and explicit CPU/CUDA parity
examples, are still planned.

## Point Query

```bash
./.venv/bin/python examples/python/point_query.py
```

Builds a small tree and queries points against it.

## Sphere Slice

```bash
./.venv/bin/python examples/python/sphere_slice.py
```

Builds a sparse sphere and queries a 2D slice.

## Forward Render

```bash
./.venv/bin/python examples/python/forward_render.py --device auto --output docs/assets/forward_render.png
```

Builds a sparse sphere, creates smooth density/color payloads, renders camera
rays, prints timing, and writes a PNG.

![Forward render of a colored sparse sphere](assets/forward_render.png)

## Realtime Viewer

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra viewer
./.venv/bin/python examples/python/realtime_viewer.py --device auto
```

Useful options:

```bash
./.venv/bin/python examples/python/realtime_viewer.py --device cuda
./.venv/bin/python examples/python/realtime_viewer.py --device cuda --branching wide4
./.venv/bin/python examples/python/realtime_viewer.py --device cuda --scene outputs/nerf_octree/model.svo
```

Controls:

- Left-click drag: orbit around `(0, 0, 0)`.
- Mouse wheel: zoom.
- `R`: reset camera.
- `Q` or `Esc`: exit.

## NeRF Octree Reconstruction

```bash
./.venv/bin/python examples/python/reconstruct_nerf_octree.py \
  --data data/nerf \
  --save-svo outputs/nerf_octree/model.svo
```

Optimizes density/color payloads from calibrated Blender/NeRF images, refines
the Octree8 topology between optimization phases, writes debug PNGs, and can
save the final scene as a `.svo` file for the viewer or benchmarks.

## Planned Examples

- Optimize density/color payloads with a PyTorch loss.
- Compare CPU and CUDA outputs in one short script.
- Render brick payloads after the brick layout is implemented.
