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


def test_render_volume_cpu_numpy_flat_outputs() -> None:
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=0)
    origins = np.array([[-1.0, 0.5, 0.5]], dtype=np.float32)
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    sigma = np.array([2.0], dtype=np.float32)
    color = np.array([[0.8, 0.2, 0.1]], dtype=np.float32)

    rgb, depth, opacity = svo.render_volume(tree, origins, directions, sigma, color)

    alpha = np.float32(1.0 - np.exp(-2.0))
    assert rgb.shape == (1, 3)
    assert depth.shape == (1,)
    assert opacity.shape == (1,)
    np.testing.assert_allclose(rgb, alpha * color, atol=1e-6)
    np.testing.assert_allclose(opacity, np.array([alpha], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(depth, np.array([1.5], dtype=np.float32), atol=1e-6)


def test_render_volume_cpu_numpy_image_outputs_and_background() -> None:
    tree = svo.Octree.from_voxels(np.empty((0, 3), dtype=np.int32), max_depth=1)
    origins = np.zeros((2, 3, 3), dtype=np.float32)
    directions = np.zeros((2, 3, 3), dtype=np.float32)
    directions[..., 0] = 1.0
    sigma = np.empty((0,), dtype=np.float32)
    color = np.empty((0, 3), dtype=np.float32)

    rgb, depth, opacity = svo.render_volume(tree, origins, directions, sigma, color, background_color=(0.1, 0.2, 0.3))

    assert rgb.shape == (2, 3, 3)
    assert depth.shape == (2, 3)
    assert opacity.shape == (2, 3)
    np.testing.assert_allclose(rgb, np.full((2, 3, 3), [0.1, 0.2, 0.3], dtype=np.float32), atol=1e-6)
    np.testing.assert_array_equal(opacity, np.zeros((2, 3), dtype=np.float32))
    assert np.isposinf(depth).all()


def test_render_volume_rejects_bad_numpy_inputs() -> None:
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=0)
    origins = np.array([[-1.0, 0.5, 0.5]], dtype=np.float32)
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    sigma = np.array([1.0], dtype=np.float32)
    color = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="float32"):
        svo.render_volume(tree, origins.astype(np.float64), directions, sigma, color)
    with pytest.raises(ValueError, match="same shape"):
        svo.render_volume(tree, origins, np.zeros((2, 3), dtype=np.float32), sigma, color)
    with pytest.raises(ValueError, match="color"):
        svo.render_volume(tree, origins, directions, sigma, np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(NotImplementedError, match="backward"):
        svo.render_volume(tree, origins, directions, sigma, color, store_aux=True)


def test_render_volume_torch_cuda_matches_cpu_when_available() -> None:
    torch = _torch_cuda()
    coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=1)
    origins_np = np.array([[-1.0, 0.25, 0.25], [-1.0, 0.75, 0.75]], dtype=np.float32)
    directions_np = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    sigma_np = np.array([1.0, 2.0], dtype=np.float32)
    color_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    expected = svo.render_volume(tree, origins_np, directions_np, sigma_np, color_np, background_color=(0.01, 0.02, 0.03))

    cuda_tree = tree.to("cuda")
    origins = torch.tensor(origins_np, device="cuda")
    directions = torch.tensor(directions_np, device="cuda")
    sigma = torch.tensor(sigma_np, device="cuda")
    color = torch.tensor(color_np, device="cuda")
    rgb, depth, opacity = svo.render_volume(cuda_tree, origins, directions, sigma, color, background_color=(0.01, 0.02, 0.03))

    assert rgb.device.type == "cuda"
    assert depth.device.type == "cuda"
    assert opacity.device.type == "cuda"
    assert rgb.dtype == torch.float32
    assert tuple(rgb.shape) == (2, 3)
    np.testing.assert_allclose(rgb.cpu().numpy(), expected[0], atol=2e-5)
    np.testing.assert_allclose(depth.cpu().numpy(), expected[1], atol=2e-5)
    np.testing.assert_allclose(opacity.cpu().numpy(), expected[2], atol=2e-5)


def test_render_volume_torch_cuda_preserves_image_shape_when_available() -> None:
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
    sigma_np = np.array([1.0], dtype=np.float32)
    color_np = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    expected = svo.render_volume(tree, origins_np, directions_np, sigma_np, color_np)

    cuda_tree = tree.to("cuda")
    rgb, depth, opacity = svo.render_volume(
        cuda_tree,
        torch.tensor(origins_np, device="cuda"),
        torch.tensor(directions_np, device="cuda"),
        torch.tensor(sigma_np, device="cuda"),
        torch.tensor(color_np, device="cuda"),
    )

    assert tuple(rgb.shape) == (3, 3, 3)
    assert tuple(depth.shape) == (3, 3)
    assert tuple(opacity.shape) == (3, 3)
    np.testing.assert_allclose(rgb.cpu().numpy(), expected[0], atol=2e-5)
    np.testing.assert_allclose(depth.cpu().numpy(), expected[1], atol=2e-5, equal_nan=True)
    np.testing.assert_allclose(opacity.cpu().numpy(), expected[2], atol=2e-5)
