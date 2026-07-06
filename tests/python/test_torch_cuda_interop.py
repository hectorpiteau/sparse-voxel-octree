from __future__ import annotations

import numpy as np
import pytest

import svo


def _torch_cuda():
    torch = pytest.importorskip("torch")
    if not svo.cuda_enabled():
        pytest.skip("SVO extension was not built with CUDA")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is not available")
    return torch


def test_cuda_tree_query_accepts_torch_cuda_tensors() -> None:
    torch = _torch_cuda()
    coords = np.array([[0, 0, 0], [3, 3, 3], [2, 1, 0]], dtype=np.int32)
    payload_indices = np.array([7, 99, 42], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=2, payload_indices=payload_indices)
    points_np = np.array(
        [
            [0.125, 0.125, 0.125],
            [0.875, 0.875, 0.875],
            [0.625, 0.375, 0.125],
            [1.0, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    with torch.cuda.device(0):
        cuda_tree = tree.to("cuda")
        points = torch.tensor(points_np, device="cuda", dtype=torch.float32)
        leaf_ids = cuda_tree.query(points)
        payload_ids = cuda_tree.query_payload_indices(points)

    assert leaf_ids.device.type == "cuda"
    assert leaf_ids.dtype == torch.int32
    assert tuple(leaf_ids.shape) == (4,)
    assert payload_ids.device.type == "cuda"
    assert payload_ids.dtype == torch.int32
    np.testing.assert_array_equal(leaf_ids.cpu().numpy(), tree.query(points_np))
    np.testing.assert_array_equal(payload_ids.cpu().numpy(), tree.query_payload_indices(points_np))


def test_cuda_tree_rejects_bad_torch_query_inputs() -> None:
    torch = _torch_cuda()
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=1)
    cuda_tree = tree.to("cuda")

    with pytest.raises(TypeError, match="CUDA Torch tensor"):
        cuda_tree.query(torch.zeros((1, 3), dtype=torch.float32))

    with pytest.raises(TypeError, match="torch.float32"):
        cuda_tree.query(torch.zeros((1, 3), device="cuda", dtype=torch.float64))

    with pytest.raises(ValueError, match="contiguous"):
        cuda_tree.query(torch.zeros((3, 2), device="cuda", dtype=torch.float32).t())

    with pytest.raises(ValueError, match="shape"):
        cuda_tree.query(torch.zeros((3,), device="cuda", dtype=torch.float32))


def test_cuda_tree_raycast_accepts_torch_cuda_tensors() -> None:
    torch = _torch_cuda()
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=1)
    origins_np = np.array(
        [
            [-1.0, 0.25, 0.25],
            [-1.0, 0.25, 0.75],
            [0.75, 0.25, 0.25],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    directions_np = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    cpu_results = tree.raycast(origins_np, directions_np)

    cuda_tree = tree.to("cuda")
    origins = torch.tensor(origins_np, device="cuda")
    directions = torch.tensor(directions_np, device="cuda")
    hit_mask, leaf_ids, t, positions, depths = cuda_tree.raycast(origins, directions)

    assert hit_mask.dtype == torch.bool
    assert leaf_ids.dtype == torch.int32
    assert t.dtype == torch.float32
    assert positions.dtype == torch.float32
    assert depths.dtype == torch.int32
    assert all(tensor.device.type == "cuda" for tensor in (hit_mask, leaf_ids, t, positions, depths))
    assert tuple(positions.shape) == (4, 3)
    np.testing.assert_array_equal(hit_mask.cpu().numpy(), cpu_results[0])
    np.testing.assert_array_equal(leaf_ids.cpu().numpy(), cpu_results[1])
    np.testing.assert_allclose(t.cpu().numpy(), cpu_results[2], atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(positions.cpu().numpy(), cpu_results[3], atol=1e-5, equal_nan=True)
    np.testing.assert_array_equal(depths.cpu().numpy(), cpu_results[4])


def test_cuda_tree_raycast_preserves_image_shape_for_torch_tensors() -> None:
    torch = _torch_cuda()
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=1)
    camera = svo.Camera.look_at(
        origin=[-1.0, 0.25, 0.25],
        target=[0.0, 0.25, 0.25],
        up=[0.0, 1.0, 0.0],
        width=3,
        height=3,
        vertical_fov_y_degrees=45.0,
    )
    origins_np, directions_np = camera.generate_rays()
    cpu_results = tree.raycast(origins_np, directions_np)

    cuda_tree = tree.to("cuda")
    origins = torch.tensor(origins_np, device="cuda")
    directions = torch.tensor(directions_np, device="cuda")
    cuda_results = cuda_tree.raycast(origins, directions)

    assert tuple(cuda_results[0].shape) == (3, 3)
    assert tuple(cuda_results[3].shape) == (3, 3, 3)
    np.testing.assert_array_equal(cuda_results[0].cpu().numpy(), cpu_results[0])
    np.testing.assert_array_equal(cuda_results[1].cpu().numpy(), cpu_results[1])
    np.testing.assert_allclose(cuda_results[2].cpu().numpy(), cpu_results[2], atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(cuda_results[3].cpu().numpy(), cpu_results[3], atol=1e-5, equal_nan=True)
    np.testing.assert_array_equal(cuda_results[4].cpu().numpy(), cpu_results[4])


def test_torch_cuda_query_and_payload_gather_use_current_stream() -> None:
    torch = _torch_cuda()
    coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
    payload_indices = np.array([1, 0], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=1, payload_indices=payload_indices)
    cuda_tree = tree.to("cuda")
    points = torch.tensor(
        [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [1.0, 0.5, 0.5]],
        device="cuda",
        dtype=torch.float32,
    )
    payload = torch.tensor([[10.0, 11.0], [20.0, 21.0]], device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        indices = cuda_tree.query_payload_indices(points)
        gathered = svo.gather_payload(payload, indices, fill_value=-1.0)

    stream.synchronize()
    assert indices.device.type == "cuda"
    assert gathered.device.type == "cuda"
    np.testing.assert_array_equal(indices.cpu().numpy(), np.array([1, 0, -1], dtype=np.int32))
    np.testing.assert_allclose(
        gathered.cpu().numpy(),
        np.array([[20.0, 21.0], [10.0, 11.0], [-1.0, -1.0]], dtype=np.float32),
    )
