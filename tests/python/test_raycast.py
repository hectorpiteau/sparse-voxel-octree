from __future__ import annotations

import numpy as np
import pytest

import svo


def test_raycast_flat_hits_and_misses() -> None:
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32), max_depth=1)
    origins = np.array(
        [
            [-1.0, 0.25, 0.25],
            [-1.0, 0.75, 0.75],
            [0.75, 0.25, 0.25],
        ],
        dtype=np.float32,
    )
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    hit_mask, leaf_ids, t, positions, depths = tree.raycast(origins, directions)

    assert hit_mask.dtype == np.bool_
    assert leaf_ids.dtype == np.int32
    assert t.dtype == np.float32
    assert positions.dtype == np.float32
    assert depths.dtype == np.int32
    assert hit_mask.shape == (3,)
    assert positions.shape == (3, 3)

    assert hit_mask.tolist() == [True, False, True]
    assert leaf_ids.tolist() == [0, -1, 1]
    np.testing.assert_allclose(t[[0, 2]], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(positions[0], [0.0, 0.25, 0.25], atol=1e-6)
    np.testing.assert_allclose(positions[2], [0.75, 0.25, 0.25], atol=1e-6)
    assert depths.tolist() == [1, -1, 1]
    assert np.isinf(t[1])
    assert np.isnan(positions[1]).all()


def test_raycast_image_shape_and_camera_rays() -> None:
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=1)
    camera = svo.Camera.look_at(
        origin=[-1.0, 0.25, 0.25],
        target=[0.0, 0.25, 0.25],
        up=[0.0, 1.0, 0.0],
        width=3,
        height=3,
        vertical_fov_y_degrees=45.0,
    )
    origins, directions = camera.generate_rays()

    hit_mask, leaf_ids, t, positions, depths = tree.raycast(origins, directions)

    assert hit_mask.shape == (3, 3)
    assert leaf_ids.shape == (3, 3)
    assert t.shape == (3, 3)
    assert positions.shape == (3, 3, 3)
    assert depths.shape == (3, 3)
    assert hit_mask[1, 1]
    assert leaf_ids[1, 1] == 0
    assert depths[1, 1] == 1
    np.testing.assert_allclose(t[1, 1], 1.0, atol=1e-6)
    np.testing.assert_allclose(positions[1, 1], [0.0, 0.25, 0.25], atol=1e-6)


def test_raycast_float64_and_payload_indices() -> None:
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32), max_depth=1)
    origins = np.array([[-1.0, 0.25, 0.25]], dtype=np.float64)
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    hit_mask, leaf_ids, t, positions, depths = tree.raycast(
        origins,
        directions,
        return_payload_indices=True,
    )

    assert hit_mask.tolist() == [True]
    assert leaf_ids.tolist() == [0]
    np.testing.assert_allclose(t, [1.0], atol=1e-6)
    np.testing.assert_allclose(positions[0], [0.0, 0.25, 0.25], atol=1e-6)
    assert depths.tolist() == [1]


def test_raycast_bad_inputs_raise_clear_errors() -> None:
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=1)
    origins = np.zeros((1, 3), dtype=np.float32)
    directions = np.ones((1, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="origins must have dtype float32 or float64"):
        tree.raycast(np.zeros((1, 3), dtype=np.int32), directions)

    with pytest.raises(ValueError, match="origins must have shape"):
        tree.raycast(np.zeros((3,), dtype=np.float32), directions)

    with pytest.raises(ValueError, match="same shape"):
        tree.raycast(np.zeros((2, 3), dtype=np.float32), directions)

    with pytest.raises(ValueError, match="directions must be C-contiguous"):
        tree.raycast(origins, np.zeros((3, 2), dtype=np.float32).T)

    with pytest.raises(svo.ValidationError, match="directions must be non-zero"):
        tree.raycast(origins, np.zeros((1, 3), dtype=np.float32))

    with pytest.raises(svo.ValidationError, match="origins must be finite"):
        bad_origins = origins.copy()
        bad_origins[0, 0] = np.inf
        tree.raycast(bad_origins, directions)
