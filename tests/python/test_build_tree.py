from __future__ import annotations

import numpy as np

import svo


def test_build_tree_cpu_from_numpy() -> None:
    coords = np.array([[0, 0, 0], [3, 3, 3], [3, 3, 3]], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=2, device="cpu")

    assert tree.max_depth == 2
    assert tree.num_leaves == 2
    assert tree.num_nodes >= 1
    assert tree.device == "cpu"
    np.testing.assert_allclose(tree.root_bounds, np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))
