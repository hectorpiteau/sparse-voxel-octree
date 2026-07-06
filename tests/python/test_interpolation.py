from __future__ import annotations

import numpy as np
import pytest

import svo


def _dense_depth_one_tree() -> svo.Octree:
    coords = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int32,
    )
    return svo.Octree.from_voxels(coords, max_depth=1)


def _torch_cuda():
    torch = pytest.importorskip("torch")
    if not svo.cuda_enabled():
        pytest.skip("SVO extension was not built with CUDA")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is not available")
    return torch


def test_sample_trilinear_numpy_scalar_payload() -> None:
    tree = _dense_depth_one_tree()
    payload = np.arange(8, dtype=np.float32)
    points = np.array(
        [
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    sampled = svo.sample_trilinear(tree, points, payload)

    assert sampled.dtype == np.float32
    assert sampled.shape == (2,)
    np.testing.assert_allclose(sampled, [0.0, 3.5], atol=1e-6)


def test_sample_trilinear_numpy_feature_payload_and_fill() -> None:
    coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    payload_indices = np.array([1, 0], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=1, payload_indices=payload_indices)
    payload = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    points = np.array(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    sampled = svo.sample_trilinear(tree, points, payload, fill_value=0.0)

    assert sampled.dtype == np.float64
    assert sampled.shape == (4, 2)
    np.testing.assert_allclose(sampled[0], [30.0, 40.0], atol=1e-12)
    np.testing.assert_allclose(sampled[1], [10.0, 20.0], atol=1e-12)
    np.testing.assert_allclose(sampled[2], [3.75, 5.0], atol=1e-12)
    np.testing.assert_allclose(sampled[3], [0.0, 0.0], atol=1e-12)


def test_sample_trilinear_rejects_bad_numpy_inputs() -> None:
    tree = _dense_depth_one_tree()
    with pytest.raises(ValueError, match="points must have shape"):
        svo.sample_trilinear(tree, np.zeros((3,), dtype=np.float32), np.zeros((8,), dtype=np.float32))
    with pytest.raises(TypeError, match="payload must have dtype"):
        svo.sample_trilinear(tree, np.zeros((1, 3), dtype=np.float32), np.zeros((8,), dtype=np.int32))
    with pytest.raises(svo.ValidationError, match="payload row range"):
        svo.sample_trilinear(tree, np.zeros((1, 3), dtype=np.float32), np.zeros((2,), dtype=np.float32))


def test_sample_trilinear_torch_cuda_forward_and_backward() -> None:
    torch = _torch_cuda()
    tree = _dense_depth_one_tree()
    cuda_tree = tree.to("cuda")
    points = torch.tensor([[0.5, 0.5, 0.5]], device="cuda", dtype=torch.float32)
    payload = torch.arange(8, device="cuda", dtype=torch.float64, requires_grad=True)

    sampled = svo.sample_trilinear(cuda_tree, points, payload)
    sampled.sum().backward()

    assert sampled.device.type == "cuda"
    assert sampled.dtype == torch.float64
    assert tuple(sampled.shape) == (1,)
    np.testing.assert_allclose(sampled.detach().cpu().numpy(), [3.5], atol=1e-12)
    np.testing.assert_allclose(payload.grad.detach().cpu().numpy(), np.full((8,), 0.125), atol=1e-12)


def test_sample_trilinear_torch_cuda_gradcheck_payload() -> None:
    torch = _torch_cuda()
    tree = _dense_depth_one_tree()
    cuda_tree = tree.to("cuda")
    points = torch.tensor([[0.5, 0.5, 0.5]], device="cuda", dtype=torch.float32)
    payload = torch.arange(8, device="cuda", dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda candidate_payload: svo.sample_trilinear(cuda_tree, points, candidate_payload),
        (payload,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_sample_trilinear_torch_cuda_current_stream() -> None:
    torch = _torch_cuda()
    tree = _dense_depth_one_tree()
    cuda_tree = tree.to("cuda")
    points = torch.tensor([[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]], device="cuda", dtype=torch.float32)
    payload = torch.arange(8, device="cuda", dtype=torch.float32, requires_grad=True)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        sampled = svo.sample_trilinear(cuda_tree, points, payload)
        loss = sampled.sum()
        loss.backward()

    stream.synchronize()
    np.testing.assert_allclose(sampled.detach().cpu().numpy(), [0.0, 3.5], atol=1e-6)
    assert payload.grad is not None
