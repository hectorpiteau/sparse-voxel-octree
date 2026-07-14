from __future__ import annotations

import numpy as np
import pytest

import svo


def _make_tree() -> svo.Octree:
    coords = np.array([[0, 0, 0], [3, 3, 3], [1, 2, 0]], dtype=np.int32)
    payload_indices = np.array([2, 0, 1], dtype=np.int32)
    root_bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    return svo.Octree.from_voxels(coords, max_depth=2, payload_indices=payload_indices, root_bounds=root_bounds)


def test_save_load_roundtrips_tree_and_payloads(tmp_path) -> None:
    tree = _make_tree()
    sigma = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    color = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    path = tmp_path / "scene.svo"

    svo.save(path, tree, {"sigma": sigma, "color": color})
    loaded = svo.load(path)

    assert loaded.cuda_tree is None
    assert loaded.tree.max_depth == tree.max_depth
    assert loaded.tree.branching == tree.branching
    np.testing.assert_allclose(loaded.tree.root_bounds, tree.root_bounds)
    np.testing.assert_array_equal(loaded.tree.leaf_payload_indices, tree.leaf_payload_indices)
    np.testing.assert_allclose(loaded.payloads["sigma"], sigma)
    np.testing.assert_allclose(loaded.payloads["color"], color)


def test_loaded_scene_renders_like_original(tmp_path) -> None:
    tree = _make_tree()
    sigma = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    color = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    origins = np.array([[-2.0, -0.75, -0.75], [-2.0, 0.75, 0.75]], dtype=np.float32)
    directions = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    path = tmp_path / "renderable.svo"
    svo.save(path, tree, {"sigma": sigma, "color": color})
    loaded = svo.load(path)

    expected = svo.render_volume(tree, origins, directions, sigma, color)
    actual = svo.render_volume(loaded.tree, origins, directions, loaded.payloads["sigma"], loaded.payloads["color"])
    for expected_array, actual_array in zip(expected, actual, strict=True):
        np.testing.assert_allclose(actual_array, expected_array)


def test_save_rejects_payload_row_mismatch(tmp_path) -> None:
    tree = _make_tree()
    with pytest.raises(ValueError, match="payload row count"):
        svo.save(tmp_path / "bad.svo", tree, {"sigma": np.ones(2, dtype=np.float32)})


def test_save_rejects_non_float32_payload(tmp_path) -> None:
    tree = _make_tree()
    with pytest.raises(TypeError, match="float32"):
        svo.save(tmp_path / "bad_dtype.svo", tree, {"sigma": np.ones(3, dtype=np.float64)})


def test_load_cuda_returns_cuda_payloads_when_available(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    if not svo.cuda_enabled() or not torch.cuda.is_available():
        pytest.skip("CUDA extension and CUDA Torch are required")

    tree = _make_tree()
    cuda_tree = tree.to("cuda")
    sigma = torch.tensor([0.5, 1.5, 2.5], device="cuda")
    color = torch.eye(3, device="cuda")
    path = tmp_path / "cuda_scene.svo"

    svo.save(path, cuda_tree, {"sigma": sigma, "color": color})
    loaded = svo.load(path, device="cuda")

    assert loaded.cuda_tree is not None
    assert loaded.cuda_tree.device == "cuda"
    assert loaded.payloads["sigma"].is_cuda
    assert loaded.payloads["color"].is_cuda
    torch.testing.assert_close(loaded.payloads["sigma"], sigma)
    torch.testing.assert_close(loaded.payloads["color"], color)
