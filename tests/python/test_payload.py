from __future__ import annotations

import numpy as np
import pytest

import svo


def test_custom_payload_indices_query_and_exposed_mapping() -> None:
    coords = np.array([[3, 3, 3], [0, 0, 0], [2, 1, 0]], dtype=np.int32)
    payload_indices = np.array([42, 7, 99], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=2, payload_indices=payload_indices)

    points = np.array(
        [
            [(0 + 0.5) / 4, (0 + 0.5) / 4, (0 + 0.5) / 4],
            [(2 + 0.5) / 4, (1 + 0.5) / 4, (0 + 0.5) / 4],
            [(3 + 0.5) / 4, (3 + 0.5) / 4, (3 + 0.5) / 4],
            [(1 + 0.5) / 4, (1 + 0.5) / 4, (1 + 0.5) / 4],
        ],
        dtype=np.float32,
    )

    assert tree.leaf_payload_indices.dtype == np.int32
    assert tree.leaf_payload_indices.tolist() == [7, 99, 42]
    assert tree.query(points).tolist() == [0, 1, 2, -1]
    assert tree.query(points, return_payload_indices=True).tolist() == [7, 99, 42, -1]
    assert tree.query_payload_indices(points).tolist() == [7, 99, 42, -1]

    if svo.cuda_enabled():
        cuda_tree = tree.to("cuda")
        assert cuda_tree.leaf_payload_indices.tolist() == [7, 99, 42]
        assert cuda_tree.query_payload_indices(points).tolist() == [7, 99, 42, -1]
        assert tree.query_cuda(points, return_payload_indices=True).tolist() == [7, 99, 42, -1]


def test_payload_indices_validation() -> None:
    coords = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int32)

    tree = svo.Octree.from_voxels(coords, max_depth=1, payload_indices=np.array([5, 5], dtype=np.int32))
    assert tree.leaf_payload_indices.tolist() == [5]

    with pytest.raises(svo.ValidationError, match="matching payload indices"):
        svo.Octree.from_voxels(coords, max_depth=1, payload_indices=np.array([5, 6], dtype=np.int32))

    with pytest.raises(ValueError, match="same length"):
        svo.Octree.from_voxels(
            np.array([[0, 0, 0]], dtype=np.int32),
            max_depth=1,
            payload_indices=np.array([1, 2], dtype=np.int32),
        )

    with pytest.raises(ValueError, match="non-negative"):
        svo.Octree.from_voxels(
            np.array([[0, 0, 0]], dtype=np.int32),
            max_depth=1,
            payload_indices=np.array([-1], dtype=np.int32),
        )


def test_gather_payload_numpy_scalar_rgb_and_features() -> None:
    indices = np.array([[2, -1], [0, 1]], dtype=np.int32)

    scalar_payload = np.array([10, 11, 12], dtype=np.int32)
    gathered_scalars = svo.gather_payload(scalar_payload, indices, fill_value=-5)
    assert gathered_scalars.dtype == np.int32
    assert gathered_scalars.shape == (2, 2)
    assert gathered_scalars.tolist() == [[12, -5], [10, 11]]

    rgb_payload = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    gathered_rgb = svo.gather_payload(rgb_payload, indices, fill_value=np.nan)
    assert gathered_rgb.dtype == np.float32
    assert gathered_rgb.shape == (2, 2, 3)
    np.testing.assert_allclose(gathered_rgb[0, 0], [0.0, 0.0, 1.0])
    assert np.isnan(gathered_rgb[0, 1]).all()

    feature_payload = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
    gathered_features = svo.gather_payload(feature_payload, np.array([1, -1], dtype=np.int64))
    assert gathered_features.shape == (2, 2, 2)
    np.testing.assert_allclose(gathered_features[0], feature_payload[1])
    np.testing.assert_allclose(gathered_features[1], np.zeros((2, 2), dtype=np.float32))


def test_gather_payload_numpy_rejects_bad_indices() -> None:
    payload = np.array([10, 11, 12], dtype=np.int32)

    with pytest.raises(IndexError, match="inside the payload row range"):
        svo.gather_payload(payload, np.array([-2], dtype=np.int32))

    with pytest.raises(IndexError, match="inside the payload row range"):
        svo.gather_payload(payload, np.array([3], dtype=np.int32))

    with pytest.raises(TypeError, match="integer dtype"):
        svo.gather_payload(payload, np.array([0.0], dtype=np.float32))


def test_gather_payload_torch_matches_manual_masked_indexing() -> None:
    torch = pytest.importorskip("torch")

    payload = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[2, -1], [0, 1]], dtype=torch.int32)
    gathered = svo.gather_payload(payload, indices, fill_value=-1.0)

    expected = torch.full((2, 2, 2), -1.0, dtype=torch.float32)
    valid = indices != -1
    expected[valid] = payload[indices[valid].long()]

    assert gathered.dtype == payload.dtype
    assert tuple(gathered.shape) == (2, 2, 2)
    assert torch.equal(gathered, expected)
