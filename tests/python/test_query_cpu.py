from __future__ import annotations

import numpy as np
import pytest

import svo


def test_query_cpu_hits_and_misses() -> None:
    coords = np.array([[0, 0, 0], [3, 3, 3]], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=2)

    points = np.array(
        [
            [0.125, 0.125, 0.125],
            [0.875, 0.875, 0.875],
            [0.375, 0.375, 0.375],
            [1.0, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    leaf_ids = tree.query(points)
    payload_indices = tree.query(points, return_payload_indices=True)

    assert leaf_ids.dtype == np.int32
    assert payload_indices.dtype == np.int32
    assert leaf_ids.tolist() == [0, 1, -1, -1]
    assert payload_indices.tolist() == [0, 1, -1, -1]


def test_query_cpu_sphere_regression() -> None:
    grid_size = 64
    radius = 18.0
    center = np.array([32.0, 32.0, 32.0], dtype=np.float32)

    coords = []
    for z in range(grid_size):
        for y in range(grid_size):
            for x in range(grid_size):
                voxel_center = np.array([x + 0.5, y + 0.5, z + 0.5], dtype=np.float32)
                if np.sum((voxel_center - center) ** 2) <= radius**2:
                    coords.append((x, y, z))
    coords_array = np.array(coords, dtype=np.int32)

    tree = svo.Octree.from_voxels(coords_array, max_depth=6)

    inside = np.array([[32.5 / 64.0, 32.5 / 64.0, 32.5 / 64.0]], dtype=np.float32)
    outside = np.array([[0.5 / 64.0, 0.5 / 64.0, 0.5 / 64.0]], dtype=np.float32)

    assert tree.num_leaves == len(coords)
    assert tree.query(inside).tolist()[0] >= 0
    assert tree.query(outside).tolist()[0] == -1


def test_bad_inputs_raise_clear_errors() -> None:
    with pytest.raises(TypeError, match="coords must have dtype int32 or int64"):
        svo.Octree.from_voxels(np.zeros((1, 3), dtype=np.float32), max_depth=1)

    with pytest.raises(ValueError, match="coords must have shape"):
        svo.Octree.from_voxels(np.zeros((3,), dtype=np.int32), max_depth=1)

    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=1)

    with pytest.raises(TypeError, match="points must have dtype float32 or float64"):
        tree.query(np.zeros((1, 3), dtype=np.int32))

    with pytest.raises(ValueError, match="points must be C-contiguous"):
        tree.query(np.zeros((3, 2), dtype=np.float32).T)

    with pytest.raises(ValueError, match="currently supports only device='cpu'"):
        svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=1, device="cuda")


def test_cuda_api_is_explicit_for_current_build() -> None:
    coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
    points = np.array(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [1.0, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    tree = svo.Octree.from_voxels(coords, max_depth=1)

    assert tree.to("cpu").device == "cpu"

    if svo.cuda_enabled():
        cuda_tree = tree.to("cuda")
        assert cuda_tree.device == "cuda"
        assert cuda_tree.max_depth == tree.max_depth
        assert cuda_tree.num_nodes == tree.num_nodes
        assert cuda_tree.num_leaves == tree.num_leaves
        assert cuda_tree.query(points).tolist() == tree.query(points).tolist()
        assert tree.query_cuda(points).tolist() == tree.query(points).tolist()
        assert cuda_tree.to("cpu").device == "cpu"

        with pytest.raises(TypeError, match="points must have dtype float32 or float64"):
            cuda_tree.query(np.zeros((1, 3), dtype=np.int32))

        with pytest.raises(ValueError, match="points must be C-contiguous"):
            cuda_tree.query(np.zeros((3, 2), dtype=np.float32).T)
    else:
        with pytest.raises(TypeError, match="SVO_ENABLE_CUDA=ON"):
            tree.to("cuda")

        with pytest.raises(TypeError, match="SVO_ENABLE_CUDA=ON"):
            tree.query_cuda(points)
