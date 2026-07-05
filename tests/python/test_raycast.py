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



def _assert_raycast_outputs_equal(actual, expected) -> None:
    for actual_array, expected_array in zip(actual, expected):
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[2], expected[2], atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(actual[3], expected[3], atol=1e-5, equal_nan=True)
    np.testing.assert_array_equal(actual[4], expected[4])


def test_raycast_cuda_api_matches_cpu_when_available() -> None:
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=1)
    origins = np.array(
        [
            [-1.0, 0.25, 0.25],
            [-1.0, 0.25, 0.75],
            [0.75, 0.25, 0.25],
            [-1.0, -1.0, -1.0],
            [-1.0, 0.5, 0.25],
        ],
        dtype=np.float32,
    )
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    cpu_results = tree.raycast(origins, directions)

    if svo.cuda_enabled():
        cuda_tree = tree.to("cuda")
        _assert_raycast_outputs_equal(cuda_tree.raycast(origins, directions), cpu_results)
        _assert_raycast_outputs_equal(tree.raycast_cuda(origins, directions), cpu_results)
        _assert_raycast_outputs_equal(
            cuda_tree.raycast(origins, directions, return_payload_indices=True),
            tree.raycast(origins, directions, return_payload_indices=True),
        )
    else:
        with pytest.raises(TypeError, match="SVO_ENABLE_CUDA=ON"):
            tree.raycast_cuda(origins, directions)


def test_raycast_cuda_image_shape_matches_cpu_when_available() -> None:
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
    cpu_results = tree.raycast(origins, directions)

    if svo.cuda_enabled():
        cuda_tree = tree.to("cuda")
        cuda_results = cuda_tree.raycast(origins, directions)
        _assert_raycast_outputs_equal(cuda_results, cpu_results)
        assert cuda_results[0].shape == (3, 3)
        assert cuda_results[3].shape == (3, 3, 3)
    else:
        with pytest.raises(TypeError, match="SVO_ENABLE_CUDA=ON"):
            tree.raycast_cuda(origins, directions)


def test_raycast_cuda_bad_inputs_match_cpu_validation_when_available() -> None:
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=1)
    origins = np.zeros((1, 3), dtype=np.float32)
    directions = np.ones((1, 3), dtype=np.float32)

    if not svo.cuda_enabled():
        return

    cuda_tree = tree.to("cuda")
    with pytest.raises(TypeError, match="origins must have dtype float32 or float64"):
        cuda_tree.raycast(np.zeros((1, 3), dtype=np.int32), directions)

    with pytest.raises(ValueError, match="same shape"):
        cuda_tree.raycast(np.zeros((2, 3), dtype=np.float32), directions)

    with pytest.raises(ValueError, match="directions must be C-contiguous"):
        cuda_tree.raycast(origins, np.zeros((3, 2), dtype=np.float32).T)
