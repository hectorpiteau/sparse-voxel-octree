from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_example_module():
    path = Path(__file__).resolve().parents[2] / "examples/python/reconstruct_nerf_octree.py"
    spec = importlib.util.spec_from_file_location("reconstruct_nerf_octree", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_png_roundtrip_and_dataset_loader(tmp_path: Path) -> None:
    module = _load_example_module()
    data_dir = tmp_path / "nerf"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)

    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    module.write_png_rgb(train_dir / "r_0.png", rgb)
    rgba = module.read_png_rgba(train_dir / "r_0.png")
    assert rgba.shape == (2, 2, 4)
    np.testing.assert_allclose(rgba[..., :3], rgb.astype(np.float32) / 255.0)
    np.testing.assert_allclose(rgba[..., 3], np.ones((2, 2), dtype=np.float32))

    transforms = {
        "camera_angle_x": 0.7,
        "frames": [
            {
                "file_path": "./train/r_0",
                "transform_matrix": np.eye(4, dtype=np.float32).tolist(),
            }
        ],
    }
    (data_dir / "transforms_train.json").write_text(json.dumps(transforms), encoding="utf-8")

    dataset = module.load_nerf_dataset(data_dir, "train")
    assert dataset.images.shape == (1, 2, 2, 3)
    assert dataset.origins.shape == (1, 4, 3)
    assert dataset.directions.shape == (1, 4, 3)
    np.testing.assert_allclose(dataset.images[0], rgb.astype(np.float32) / 255.0)


def test_psnr_helper() -> None:
    module = _load_example_module()
    assert module.psnr_from_mse(1.0) == 0.0
    assert module.psnr_from_mse(0.01) == 20.0


def test_make_root_bounds_centers_scene() -> None:
    module = _load_example_module()
    bounds = module.make_root_bounds((0.0, 0.0, 0.0), 3.0)
    np.testing.assert_allclose(
        bounds,
        np.array([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=np.float32),
    )
