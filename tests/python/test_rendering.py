from __future__ import annotations

import gc

import numpy as np
import pytest

import svo


@pytest.fixture(autouse=True)
def _cleanup_torch_cuda_after_rendering_test():
    yield
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


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


def test_render_volume_cpu_wide4_matches_octree8() -> None:
    coords = np.array([[0, 0, 0], [8, 8, 8], [15, 15, 15]], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=4)
    wide = svo.Octree.from_voxels(coords, max_depth=4, branching="wide4")
    origins = np.array(
        [
            [-1.0, 0.03125, 0.03125],
            [-1.0, 8.5 / 16.0, 8.5 / 16.0],
            [2.0, 15.5 / 16.0, 15.5 / 16.0],
            [-1.0, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    sigma = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    color = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    expected = svo.render_volume(tree, origins, directions, sigma, color)
    actual = svo.render_volume(wide, origins, directions, sigma, color)
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_allclose(actual_array, expected_array, atol=2e-5, equal_nan=True)


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
    with pytest.raises(NotImplementedError, match="Torch CUDA"):
        svo.render_volume(tree, origins, directions, sigma, color, render_strategy="intervals")
    with pytest.raises(ValueError, match="render_strategy"):
        svo.render_volume(tree, origins, directions, sigma, color, render_strategy="unknown")


def test_volume_renderer_is_exported() -> None:
    assert hasattr(svo, "VolumeRenderer")


def test_volume_renderer_numpy_matches_function_when_torch_available() -> None:
    pytest.importorskip("torch")
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=0)
    origins = np.array([[-1.0, 0.5, 0.5]], dtype=np.float32)
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    sigma = np.array([1.0], dtype=np.float32)
    color = np.array([[0.2, 0.4, 0.8]], dtype=np.float32)
    renderer = svo.VolumeRenderer(tree, background_color=(0.01, 0.02, 0.03))

    actual = renderer(origins, directions, sigma, color)
    expected = svo.render_volume(tree, origins, directions, sigma, color, background_color=(0.01, 0.02, 0.03))

    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_allclose(actual_array, expected_array, atol=1e-6)


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


def test_render_volume_torch_cuda_intervals_match_direct_when_available() -> None:
    torch = _torch_cuda()
    coords = np.array([[0, 0, 0], [1, 0, 0], [3, 3, 3]], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=2)
    cuda_tree = tree.to("cuda")
    origins_np = np.array([[-1.0, 0.125, 0.125], [-1.0, 0.375, 0.375], [2.0, 0.875, 0.875]], dtype=np.float32)
    directions_np = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    sigma_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    color_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    origins = torch.tensor(origins_np, device="cuda")
    directions = torch.tensor(directions_np, device="cuda")
    sigma = torch.tensor(sigma_np, device="cuda")
    color = torch.tensor(color_np, device="cuda")
    direct = svo.render_volume(cuda_tree, origins, directions, sigma, color, render_strategy="direct")
    auto = svo.render_volume(cuda_tree, origins, directions, sigma, color, render_strategy="auto")
    intervals = svo.render_volume(cuda_tree, origins, directions, sigma, color, render_strategy="intervals")

    for actual, expected in zip(auto, direct):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(intervals, direct):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)


def test_render_volume_torch_cuda_wide4_matches_cpu_when_available() -> None:
    torch = _torch_cuda()
    coords = np.array([[0, 0, 0], [4, 4, 4], [15, 15, 15]], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=4, branching="wide4")
    origins_np = np.array(
        [[-1.0, 0.03125, 0.03125], [-1.0, 0.28125, 0.28125], [2.0, 0.96875, 0.96875]],
        dtype=np.float32,
    )
    directions_np = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    sigma_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    color_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    expected = svo.render_volume(tree, origins_np, directions_np, sigma_np, color_np)

    cuda_tree = tree.to("cuda")
    sigma = torch.tensor(sigma_np, device="cuda", dtype=torch.float32, requires_grad=True)
    color = torch.tensor(color_np, device="cuda", dtype=torch.float32, requires_grad=True)
    rgb, depth, opacity = svo.render_volume(
        cuda_tree,
        torch.tensor(origins_np, device="cuda"),
        torch.tensor(directions_np, device="cuda"),
        sigma,
        color,
    )
    (rgb.sum() + opacity.sum()).backward()

    assert cuda_tree.branching == "wide4"
    np.testing.assert_allclose(rgb.detach().cpu().numpy(), expected[0], atol=2e-5)
    np.testing.assert_allclose(depth.cpu().numpy(), expected[1], atol=2e-5)
    np.testing.assert_allclose(opacity.detach().cpu().numpy(), expected[2], atol=2e-5)
    assert sigma.grad is not None
    assert color.grad is not None
    assert torch.isfinite(sigma.grad).all()
    assert torch.isfinite(color.grad).all()


def test_render_volume_torch_cuda_wide4_intervals_match_direct_when_available() -> None:
    torch = _torch_cuda()
    coords = np.array([[0, 0, 0], [4, 4, 4], [15, 15, 15]], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=4, branching="wide4")
    cuda_tree = tree.to("cuda")
    origins_np = np.array(
        [[-1.0, 0.03125, 0.03125], [-1.0, 0.28125, 0.28125], [2.0, 0.96875, 0.96875]],
        dtype=np.float32,
    )
    directions_np = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    sigma_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    color_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    origins = torch.tensor(origins_np, device="cuda")
    directions = torch.tensor(directions_np, device="cuda")
    sigma = torch.tensor(sigma_np, device="cuda")
    color = torch.tensor(color_np, device="cuda")
    direct = svo.render_volume(cuda_tree, origins, directions, sigma, color)
    intervals = svo.render_volume(cuda_tree, origins, directions, sigma, color, render_strategy="intervals")

    for actual, expected in zip(intervals, direct):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)


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


def test_render_volume_torch_cuda_backward_matches_single_leaf_analytic_gradient() -> None:
    torch = _torch_cuda()
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=0)
    cuda_tree = tree.to("cuda")
    origins = torch.tensor([[-1.0, 0.5, 0.5]], device="cuda", dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    sigma = torch.tensor([2.0], device="cuda", dtype=torch.float32, requires_grad=True)
    color = torch.tensor([[0.8, 0.2, 0.1]], device="cuda", dtype=torch.float32, requires_grad=True)

    rgb, depth, opacity = svo.render_volume(cuda_tree, origins, directions, sigma, color)
    loss = rgb.sum() + opacity.sum()
    loss.backward()

    alpha = np.float32(1.0 - np.exp(-2.0))
    expected_sigma = np.float32((1.0 + 0.8 + 0.2 + 0.1) * (1.0 - alpha))
    assert depth.requires_grad is False
    np.testing.assert_allclose(sigma.grad.detach().cpu().numpy(), [expected_sigma], atol=2e-5)
    np.testing.assert_allclose(color.grad.detach().cpu().numpy(), np.full((1, 3), alpha, dtype=np.float32), atol=2e-5)


def test_render_volume_torch_cuda_intervals_backward_matches_direct_when_available() -> None:
    torch = _torch_cuda()
    coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    tree = svo.Octree.from_voxels(coords, max_depth=1)
    cuda_tree = tree.to("cuda")
    origins = torch.tensor([[-1.0, 0.25, 0.25], [-1.0, 0.75, 0.75]], device="cuda", dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    sigma_direct = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32, requires_grad=True)
    color_direct = torch.tensor([[1.0, 0.2, 0.0], [0.0, 1.0, 0.4]], device="cuda", dtype=torch.float32, requires_grad=True)
    sigma_intervals = sigma_direct.detach().clone().requires_grad_(True)
    color_intervals = color_direct.detach().clone().requires_grad_(True)

    rgb, _depth, opacity = svo.render_volume(cuda_tree, origins, directions, sigma_direct, color_direct)
    (0.7 * rgb[..., 0].sum() + 0.2 * rgb[..., 1].sum() + 0.4 * opacity.sum()).backward()
    rgb_i, _depth_i, opacity_i = svo.render_volume(
        cuda_tree,
        origins,
        directions,
        sigma_intervals,
        color_intervals,
        render_strategy="intervals",
    )
    (0.7 * rgb_i[..., 0].sum() + 0.2 * rgb_i[..., 1].sum() + 0.4 * opacity_i.sum()).backward()

    torch.testing.assert_close(sigma_intervals.grad, sigma_direct.grad, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(color_intervals.grad, color_direct.grad, atol=2e-5, rtol=1e-5)


def test_render_volume_torch_cuda_backward_finite_difference_when_available() -> None:
    torch = _torch_cuda()
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=0)
    cuda_tree = tree.to("cuda")
    origins_np = np.array([[-1.0, 0.5, 0.5]], dtype=np.float32)
    directions_np = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    sigma_np = np.array([1.4], dtype=np.float32)
    color_np = np.array([[0.3, 0.6, 0.9]], dtype=np.float32)

    origins = torch.tensor(origins_np, device="cuda", dtype=torch.float32)
    directions = torch.tensor(directions_np, device="cuda", dtype=torch.float32)
    sigma = torch.tensor(sigma_np, device="cuda", dtype=torch.float32, requires_grad=True)
    color = torch.tensor(color_np, device="cuda", dtype=torch.float32, requires_grad=True)
    rgb, _depth, opacity = svo.render_volume(cuda_tree, origins, directions, sigma, color)
    (0.7 * rgb[..., 0].sum() + 0.2 * rgb[..., 1].sum() + 0.4 * opacity.sum()).backward()

    def cpu_loss(candidate_sigma: np.ndarray, candidate_color: np.ndarray) -> float:
        rgb_np, _depth_np, opacity_np = svo.render_volume(tree, origins_np, directions_np, candidate_sigma, candidate_color)
        return float(0.7 * rgb_np[..., 0].sum() + 0.2 * rgb_np[..., 1].sum() + 0.4 * opacity_np.sum())

    eps = np.float32(1.0e-3)
    sigma_plus = sigma_np.copy()
    sigma_minus = sigma_np.copy()
    sigma_plus[0] += eps
    sigma_minus[0] -= eps
    expected_sigma = (cpu_loss(sigma_plus, color_np) - cpu_loss(sigma_minus, color_np)) / float(2.0 * eps)

    color_plus = color_np.copy()
    color_minus = color_np.copy()
    color_plus[0, 1] += eps
    color_minus[0, 1] -= eps
    expected_color_g = (cpu_loss(sigma_np, color_plus) - cpu_loss(sigma_np, color_minus)) / float(2.0 * eps)

    np.testing.assert_allclose(sigma.grad.detach().cpu().numpy()[0], expected_sigma, atol=3e-3, rtol=2e-2)
    np.testing.assert_allclose(color.grad.detach().cpu().numpy()[0, 1], expected_color_g, atol=3e-3, rtol=2e-2)


def test_render_volume_torch_cuda_wide4_backward_finite_difference_when_available() -> None:
    torch = _torch_cuda()
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=0, branching="wide4")
    cuda_tree = tree.to("cuda")
    origins_np = np.array([[-1.0, 0.5, 0.5]], dtype=np.float32)
    directions_np = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    sigma_np = np.array([1.4], dtype=np.float32)
    color_np = np.array([[0.3, 0.6, 0.9]], dtype=np.float32)

    origins = torch.tensor(origins_np, device="cuda", dtype=torch.float32)
    directions = torch.tensor(directions_np, device="cuda", dtype=torch.float32)
    sigma = torch.tensor(sigma_np, device="cuda", dtype=torch.float32, requires_grad=True)
    color = torch.tensor(color_np, device="cuda", dtype=torch.float32, requires_grad=True)
    rgb, _depth, opacity = svo.render_volume(cuda_tree, origins, directions, sigma, color)
    (0.7 * rgb[..., 0].sum() + 0.2 * rgb[..., 1].sum() + 0.4 * opacity.sum()).backward()

    def cpu_loss(candidate_sigma: np.ndarray, candidate_color: np.ndarray) -> float:
        rgb_np, _depth_np, opacity_np = svo.render_volume(tree, origins_np, directions_np, candidate_sigma, candidate_color)
        return float(0.7 * rgb_np[..., 0].sum() + 0.2 * rgb_np[..., 1].sum() + 0.4 * opacity_np.sum())

    eps = np.float32(1.0e-3)
    sigma_plus = sigma_np.copy()
    sigma_minus = sigma_np.copy()
    sigma_plus[0] += eps
    sigma_minus[0] -= eps
    expected_sigma = (cpu_loss(sigma_plus, color_np) - cpu_loss(sigma_minus, color_np)) / float(2.0 * eps)

    color_plus = color_np.copy()
    color_minus = color_np.copy()
    color_plus[0, 1] += eps
    color_minus[0, 1] -= eps
    expected_color_g = (cpu_loss(sigma_np, color_plus) - cpu_loss(sigma_np, color_minus)) / float(2.0 * eps)

    np.testing.assert_allclose(sigma.grad.detach().cpu().numpy()[0], expected_sigma, atol=3e-3, rtol=2e-2)
    np.testing.assert_allclose(color.grad.detach().cpu().numpy()[0, 1], expected_color_g, atol=3e-3, rtol=2e-2)


def test_render_volume_depth_is_forward_only_when_available() -> None:
    torch = _torch_cuda()
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=0)
    cuda_tree = tree.to("cuda")
    origins = torch.tensor([[-1.0, 0.5, 0.5]], device="cuda", dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    sigma = torch.tensor([1.0], device="cuda", dtype=torch.float32, requires_grad=True)
    color = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)

    _rgb, depth, _opacity = svo.render_volume(cuda_tree, origins, directions, sigma, color)

    assert depth.requires_grad is False
    with pytest.raises(RuntimeError, match="does not require grad"):
        depth.sum().backward()


def test_render_volume_torch_cuda_backward_current_stream_when_available() -> None:
    torch = _torch_cuda()
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32), max_depth=1)
    cuda_tree = tree.to("cuda")
    origins = torch.tensor([[-1.0, 0.25, 0.25]], device="cuda", dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    sigma = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32, requires_grad=True)
    color = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        rgb, _depth, opacity = svo.render_volume(cuda_tree, origins, directions, sigma, color)
        loss = rgb.sum() + opacity.sum()
        loss.backward()

    stream.synchronize()
    assert sigma.grad is not None
    assert color.grad is not None
    assert torch.isfinite(sigma.grad).all()
    assert torch.isfinite(color.grad).all()


def test_render_volume_torch_cuda_intervals_backward_current_stream_when_available() -> None:
    torch = _torch_cuda()
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32), max_depth=1)
    cuda_tree = tree.to("cuda")
    origins = torch.tensor([[-1.0, 0.25, 0.25]], device="cuda", dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    sigma = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32, requires_grad=True)
    color = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        rgb, _depth, opacity = svo.render_volume(
            cuda_tree,
            origins,
            directions,
            sigma,
            color,
            render_strategy="intervals",
        )
        loss = rgb.sum() + opacity.sum()
        loss.backward()

    stream.synchronize()
    assert sigma.grad is not None
    assert color.grad is not None
    assert torch.isfinite(sigma.grad).all()
    assert torch.isfinite(color.grad).all()


def test_render_volume_torch_cuda_intervals_repeated_backward_when_available() -> None:
    torch = _torch_cuda()
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32), max_depth=1)
    cuda_tree = tree.to("cuda")
    origins = torch.tensor([[-1.0, 0.25, 0.25], [-1.0, 0.75, 0.75]], device="cuda", dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    sigma = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32, requires_grad=True)
    color = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)

    for _ in range(2):
        sigma.grad = None
        color.grad = None
        rgb, _depth, opacity = svo.render_volume(
            cuda_tree,
            origins,
            directions,
            sigma,
            color,
            render_strategy="intervals",
        )
        (rgb.sum() + opacity.sum()).backward()
        assert sigma.grad is not None
        assert color.grad is not None
        assert torch.isfinite(sigma.grad).all()
        assert torch.isfinite(color.grad).all()


def test_volume_renderer_torch_cuda_matches_function_when_available() -> None:
    torch = _torch_cuda()
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=1)
    camera = svo.Camera.look_at(
        origin=[-1.0, 0.25, 0.25],
        target=[0.0, 0.25, 0.25],
        up=[0.0, 1.0, 0.0],
        width=2,
        height=2,
        vertical_fov_y_degrees=35.0,
    )
    origins_np, directions_np = camera.generate_rays()
    cuda_tree = tree.to("cuda")
    origins = torch.tensor(origins_np, device="cuda", dtype=torch.float32)
    directions = torch.tensor(directions_np, device="cuda", dtype=torch.float32)
    sigma = torch.tensor([1.2], device="cuda", dtype=torch.float32)
    color = torch.tensor([[0.7, 0.2, 0.1]], device="cuda", dtype=torch.float32)
    renderer = svo.VolumeRenderer(
        cuda_tree,
        background_color=(0.01, 0.02, 0.03),
        early_stop_transmittance=2.0e-3,
        render_strategy="intervals",
    )

    assert isinstance(renderer, torch.nn.Module)
    actual = renderer(origins, directions, sigma, color)
    expected = svo.render_volume(
        cuda_tree,
        origins,
        directions,
        sigma,
        color,
        background_color=(0.01, 0.02, 0.03),
        early_stop_transmittance=2.0e-3,
        render_strategy="intervals",
    )

    assert tuple(actual[0].shape) == (2, 2, 3)
    assert tuple(actual[1].shape) == (2, 2)
    assert tuple(actual[2].shape) == (2, 2)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_volume_renderer_torch_cuda_backward_when_available() -> None:
    torch = _torch_cuda()
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32), max_depth=1)
    cuda_tree = tree.to("cuda")
    origins = torch.tensor([[-1.0, 0.25, 0.25], [-1.0, 0.75, 0.75]], device="cuda", dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    sigma = torch.nn.Parameter(torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32))
    color = torch.nn.Parameter(torch.tensor([[0.2, 0.4, 0.8], [0.8, 0.4, 0.2]], device="cuda", dtype=torch.float32))
    renderer = svo.VolumeRenderer(cuda_tree, render_strategy="intervals")

    rgb, depth, opacity = renderer(origins, directions, sigma, color)
    loss = rgb.sum() + opacity.sum()
    loss.backward()

    assert depth.requires_grad is False
    assert sigma.grad is not None
    assert color.grad is not None
    assert torch.isfinite(sigma.grad).all()
    assert torch.isfinite(color.grad).all()


def test_volume_renderer_torch_cuda_tiny_optimization_reduces_loss_when_available() -> None:
    torch = _torch_cuda()
    tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=0)
    cuda_tree = tree.to("cuda")
    origins = torch.tensor([[-1.0, 0.5, 0.5]], device="cuda", dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    renderer = svo.VolumeRenderer(cuda_tree)

    with torch.no_grad():
        target_rgb, _target_depth, target_opacity = renderer(
            origins,
            directions,
            torch.tensor([1.5], device="cuda", dtype=torch.float32),
            torch.tensor([[0.8, 0.25, 0.1]], device="cuda", dtype=torch.float32),
        )

    sigma = torch.nn.Parameter(torch.tensor([0.4], device="cuda", dtype=torch.float32))
    color = torch.nn.Parameter(torch.tensor([[0.1, 0.7, 0.4]], device="cuda", dtype=torch.float32))
    optimizer = torch.optim.Adam([sigma, color], lr=0.05)

    def loss_value() -> torch.Tensor:
        rgb, _depth, opacity = renderer(origins, directions, sigma, color)
        return torch.nn.functional.mse_loss(rgb, target_rgb) + torch.nn.functional.mse_loss(opacity, target_opacity)

    initial_loss = loss_value().detach()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        optimizer.step()
    final_loss = loss_value().detach()

    assert final_loss < initial_loss * 0.7
