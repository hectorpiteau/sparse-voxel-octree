# SVO File Format

The project uses a small custom binary `.svo` file for saved scenes. It is
intended for local experiments, realtime viewer inspection, and benchmark input.

V1 is deliberately plain:

- uncompressed binary data
- little-endian host format
- one octree topology
- optional named payload arrays
- `float32` payload arrays only

The file stores compact runtime topology directly: branching mode, max depth,
root bounds, node descriptors, leaf payload indices, and leaf specs. It does not
store original input voxels or optimizer state.

## Python Usage

Save a tree and payload arrays:

```python
svo.save("model.svo", tree, {"sigma": sigma, "color": color})
```

Load on CPU:

```python
loaded = svo.load("model.svo")
tree = loaded.tree
sigma = loaded.payloads["sigma"]
color = loaded.payloads["color"]
```

Load on CUDA:

```python
loaded = svo.load("model.svo", device="cuda")
cuda_tree = loaded.cuda_tree
sigma = loaded.payloads["sigma"]
color = loaded.payloads["color"]
```

CUDA loading requires a CUDA-enabled `svo` build and CUDA-enabled PyTorch.

## Examples And Benchmarks

The reconstruction example can save the final optimized scene:

```bash
./.venv/bin/python examples/python/reconstruct_nerf_octree.py \
  --data data/nerf \
  --save-svo outputs/nerf_octree/model.svo
```

The realtime viewer can load that scene:

```bash
./.venv/bin/python examples/python/realtime_viewer.py \
  --device cuda \
  --scene outputs/nerf_octree/model.svo
```

Render and raycast benchmarks can consume saved scenes:

```bash
./build-cuda/svo_render_benchmark --scene-file outputs/nerf_octree/model.svo --operation forward
./build-cuda/svo_raycast_benchmark --scene-file outputs/nerf_octree/model.svo
```

Render benchmarks require `sigma` with shape `(P,)` and `color` with shape
`(P, 3)`, where `P` is the payload row count.

## Current Limits

- The format is versioned but not promised stable before `1.0.0`.
- Compression, memory mapping, checksums, optimizer state, and arbitrary payload
  dtypes are deferred.
- Saving CUDA tensors copies payload data to CPU because file I/O is host-side.
