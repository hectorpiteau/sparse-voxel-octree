# Quickstart

## Build an Octree

```python
import numpy as np
import svo

coords = np.array([[0, 0, 0], [1, 2, 3], [7, 7, 7]], dtype=np.int32)
tree = svo.Octree.from_voxels(coords, max_depth=3)

print(tree.num_leaves)
print(tree.num_nodes)
```

Voxel coordinates are integer cells in `[0, 2^max_depth)` for each axis.

## Query Points

```python
points = np.array(
    [
        [0.0625, 0.0625, 0.0625],
        [0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)

leaf_ids = tree.query(points)
payload_ids = tree.query(points, return_payload_indices=True)
```

Misses return `-1`.

## Raycast

```python
origins = np.array([[-1.0, 0.1, 0.1]], dtype=np.float32)
directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

hit_mask, leaf_ids, t, positions, depths = tree.raycast(origins, directions)
```

Raycast misses use explicit sentinels: `hit_mask=false`, `leaf_id=-1`,
`depth=-1`, `t=inf`, and `position=(nan, nan, nan)`.

## Camera Rays

```python
camera = svo.Camera.look_at(
    origin=[1.5, 1.0, 1.5],
    target=[0.5, 0.5, 0.5],
    up=[0.0, 1.0, 0.0],
    width=320,
    height=240,
    vertical_fov_y_degrees=45.0,
)
origins, directions = camera.generate_rays()
```

## Forward Render

```python
sigma = np.ones(tree.num_leaves, dtype=np.float32)
color = np.ones((tree.num_leaves, 3), dtype=np.float32)

rgb, depth, opacity = svo.render_volume(tree, origins, directions, sigma, color)
```

For a complete PNG-producing example:

```bash
./.venv/bin/python examples/python/forward_render.py --device auto --output docs/assets/forward_render.png
```

![Forward render of a colored sparse sphere](assets/forward_render.png)

## Realtime Viewer

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra viewer
./.venv/bin/python examples/python/realtime_viewer.py --device auto
./.venv/bin/python examples/python/realtime_viewer.py --device cuda --branching wide4
```

Controls: left-click drag orbits around `(0, 0, 0)`, mouse wheel zooms, `R`
resets the camera, and `Q` or `Esc` exits.
