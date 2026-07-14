from __future__ import annotations

import numpy as np
import pytest

import svo


def test_from_leaf_specs_builds_variable_depth_tree() -> None:
    coord_min = np.array([[0, 0, 0], [4, 4, 4]], dtype=np.int32)
    depths = np.array([1, 3], dtype=np.int32)
    payload_indices = np.array([0, 1], dtype=np.int32)

    tree = svo.Octree.from_leaf_specs(coord_min, depths, payload_indices, max_depth=3)

    assert tree.num_leaves == 2
    specs = tree.leaf_specs
    np.testing.assert_array_equal(specs[1], depths)
    assert tree.query(np.array([[0.1, 0.1, 0.1], [0.56, 0.56, 0.56]], dtype=np.float32)).tolist() == [0, 1]


def test_full_grid_initializes_coarse_leaves() -> None:
    tree = svo.Octree.full_grid(max_depth=3, leaf_depth=1)

    assert tree.max_depth == 3
    assert tree.num_leaves == 8
    _coord_min, depths, payload_indices = tree.leaf_specs
    np.testing.assert_array_equal(depths, np.full(8, 1, dtype=np.int32))
    np.testing.assert_array_equal(payload_indices, np.arange(8, dtype=np.int32))


def test_full_grid_rejects_root_leaf_with_deeper_split_budget() -> None:
    with pytest.raises(ValueError, match="leaf_depth=0"):
        svo.Octree.full_grid(max_depth=3, leaf_depth=0)


def test_refine_octree_splits_and_copies_payloads() -> None:
    tree = svo.Octree.full_grid(max_depth=3, leaf_depth=1)
    sigma = np.zeros(tree.num_leaves, dtype=np.float32)
    color = np.zeros((tree.num_leaves, 3), dtype=np.float32)
    sigma[0] = 2.0
    color[0] = [1.0, 0.5, 0.25]

    result = svo.refine_octree(
        tree,
        sigma,
        color,
        split_threshold=1.0,
        prune_threshold=-1.0,
        max_leaf_growth=2.0,
    )

    assert result.tree.num_leaves == 15
    assert result.stats["split_leaves"] == 1
    assert result.old_to_new_leaf[0] == 0
    np.testing.assert_allclose(result.sigma[:8], np.full(8, 2.0, dtype=np.float32))
    np.testing.assert_allclose(result.color[:8], np.tile(color[0], (8, 1)))


def test_refine_octree_prunes_low_density_leaves() -> None:
    tree = svo.Octree.full_grid(max_depth=3, leaf_depth=1)
    sigma = np.zeros(tree.num_leaves, dtype=np.float32)
    color = np.zeros((tree.num_leaves, 3), dtype=np.float32)

    result = svo.refine_octree(
        tree,
        sigma,
        color,
        split_threshold=1.0,
        prune_threshold=0.1,
        min_depth=1,
    )

    assert result.tree.num_leaves == 0
    assert result.sigma.shape == (0,)
    assert result.color.shape == (0, 3)
    assert np.all(result.old_to_new_leaf == -1)


def test_refine_octree_merges_sibling_groups() -> None:
    tree = svo.Octree.full_grid(max_depth=3, leaf_depth=2)
    sigma = np.full(tree.num_leaves, 0.25, dtype=np.float32)
    color = np.ones((tree.num_leaves, 3), dtype=np.float32)

    result = svo.refine_octree(
        tree,
        sigma,
        color,
        split_threshold=1.0,
        prune_threshold=-1.0,
        merge_threshold=0.5,
        min_depth=1,
    )

    assert result.tree.num_leaves == 8
    assert result.stats["merged_groups"] == 8
    np.testing.assert_allclose(result.sigma, np.full(8, 0.25, dtype=np.float32))


def test_refine_octree_rejects_excessive_growth() -> None:
    tree = svo.Octree.full_grid(max_depth=4, leaf_depth=2)
    sigma = np.ones(tree.num_leaves, dtype=np.float32)
    color = np.zeros((tree.num_leaves, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="max_leaf_growth"):
        svo.refine_octree(
            tree,
            sigma,
            color,
            split_threshold=0.5,
            prune_threshold=-1.0,
            max_leaf_growth=2.0,
        )


def test_refine_octree_rejects_wide4() -> None:
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=2, branching="wide4")
    sigma = np.ones(tree.num_leaves, dtype=np.float32)
    color = np.ones((tree.num_leaves, 3), dtype=np.float32)

    with pytest.raises(NotImplementedError, match="octree8"):
        svo.refine_octree(tree, sigma, color, split_threshold=1.0, prune_threshold=0.0)


def test_refine_octree_torch_cuda_remaps_payloads_on_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not svo.cuda_enabled() or not torch.cuda.is_available():
        pytest.skip("CUDA extension and CUDA Torch are required")

    tree = svo.Octree.full_grid(max_depth=3, leaf_depth=1)
    cuda_tree = tree.to("cuda")
    sigma = torch.zeros(tree.num_leaves, device="cuda")
    color = torch.zeros(tree.num_leaves, 3, device="cuda")
    sigma[0] = 2.0
    color[0] = torch.tensor([1.0, 0.5, 0.25], device="cuda")

    result = svo.refine_octree(
        cuda_tree,
        sigma,
        color,
        split_threshold=1.0,
        prune_threshold=-1.0,
        max_leaf_growth=2.0,
    )

    assert result.cuda_tree is not None
    assert result.cuda_tree.device == "cuda"
    assert result.sigma.is_cuda
    assert result.color.is_cuda
    torch.testing.assert_close(result.sigma[:8], torch.full((8,), 2.0, device="cuda"))
