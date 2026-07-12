from __future__ import annotations

import numpy as np
import pytest

import svo


def test_build_tree_cpu_from_numpy() -> None:
    coords = np.array([[0, 0, 0], [3, 3, 3], [3, 3, 3]], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=2, device="cpu")

    assert tree.max_depth == 2
    assert tree.num_leaves == 2
    assert tree.num_nodes >= 1
    assert tree.device == "cpu"
    assert tree.branching == "octree8"
    np.testing.assert_allclose(tree.root_bounds, np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))


def test_build_tree_wide4_from_numpy() -> None:
    coords = np.array([[0, 0, 0], [15, 15, 15]], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=4, branching="wide4")
    octree = svo.Octree.from_voxels(coords, max_depth=4)

    assert tree.branching == "wide4"
    assert tree.max_depth == 4
    assert tree.num_nodes < octree.num_nodes
    assert tree.num_leaves == 2


def test_build_tree_wide4_rejects_odd_depth() -> None:
    with pytest.raises(svo.ValidationError, match="even max_depth"):
        svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=3, branching="wide4")


def test_build_tree_rejects_bad_branching() -> None:
    with pytest.raises(ValueError, match="branching"):
        svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=2, branching="wide8")
