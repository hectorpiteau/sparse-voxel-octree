from __future__ import annotations

import numpy as np
import pytest

import svo


def test_look_at_generate_rays_shape_and_center_direction() -> None:
    camera = svo.Camera.look_at(
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        target=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        width=5,
        height=5,
        vertical_fov_y_degrees=60.0,
    )

    origins, directions = camera.generate_rays()

    assert origins.shape == (5, 5, 3)
    assert directions.shape == (5, 5, 3)
    assert origins.dtype == np.float32
    assert directions.dtype == np.float32
    np.testing.assert_allclose(origins[2, 2], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(directions[2, 2], [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=2), 1.0, atol=1e-6)


def test_from_intrinsics_and_conventions() -> None:
    intrinsics = svo.CameraIntrinsics(width=3, height=3, fx=2.0, fy=2.0, cx=1.5, cy=1.5)
    opengl = svo.Camera.from_intrinsics(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        intrinsics,
        svo.CameraConvention.OpenGL,
    )
    cv = svo.Camera.from_intrinsics(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        intrinsics,
        svo.CameraConvention.ComputerVision,
    )

    _, gl_directions = opengl.generate_rays()
    _, cv_directions = cv.generate_rays()

    np.testing.assert_allclose(gl_directions[1, 1], cv_directions[1, 1], atol=1e-6)
    assert gl_directions[0, 1, 1] == pytest.approx(-cv_directions[0, 1, 1], abs=1e-6)


def test_camera_bad_inputs_raise_clear_errors() -> None:
    with pytest.raises(svo.ValidationError, match="width and height"):
        svo.Camera.look_at([0, 0, 0], [0, 0, -1], [0, 1, 0], 0, 3, 60.0)

    with pytest.raises(svo.ValidationError, match="vertical_fov_y_degrees"):
        svo.Camera.look_at([0, 0, 0], [0, 0, -1], [0, 1, 0], 3, 3, 180.0)

    with pytest.raises(svo.ValidationError, match="forward vector"):
        svo.Camera.look_at([0, 0, 0], [0, 0, 0], [0, 1, 0], 3, 3, 60.0)

    with pytest.raises(svo.ValidationError, match="up vector"):
        svo.Camera.look_at([0, 0, 0], [0, 0, -1], [0, 0, -1], 3, 3, 60.0)

    with pytest.raises(svo.ValidationError, match="fx and fy"):
        bad = svo.CameraIntrinsics(width=3, height=3, fx=0.0, fy=2.0, cx=1.5, cy=1.5)
        svo.Camera.from_intrinsics([0, 0, 0], [0, 0, -1], [0, 1, 0], bad)

    with pytest.raises(ValueError, match="origin must have shape"):
        svo.Camera.look_at([0, 0], [0, 0, -1], [0, 1, 0], 3, 3, 60.0)
