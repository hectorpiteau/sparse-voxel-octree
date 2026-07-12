"""Validate built sparse-voxel-octree package artifacts."""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _latest(pattern: str, dist_dir: Path) -> Path:
    matches = [Path(path) for path in glob.glob(str(dist_dir / pattern))]
    if not matches:
        raise FileNotFoundError(f"no artifact matching {pattern!r} in {dist_dir}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def _check_wheel_contents(wheel: Path) -> None:
    forbidden_prefixes = ("include/", "lib/", "share/")
    required_suffixes = ("svo/__init__.py", "svo/info.py", "svo/rendering.py")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    forbidden = [name for name in names if name.startswith(forbidden_prefixes)]
    if forbidden:
        preview = "\n".join(forbidden[:20])
        raise AssertionError(f"wheel contains developer install files:\n{preview}")

    for suffix in required_suffixes:
        if suffix not in names:
            raise AssertionError(f"wheel is missing {suffix}")
    if not any(name.startswith("svo/_svo") and name.endswith((".so", ".pyd")) for name in names):
        raise AssertionError("wheel is missing the native svo extension")


def _check_sdist_contents(sdist: Path) -> None:
    required_suffixes = [
        "CMakeLists.txt",
        "pyproject.toml",
        "include/svo/Octree.hpp",
        "python/bindings.cpp",
        "src/Octree.cpp",
        "cuda/query_points.cu",
        "tests/python/test_import.py",
        "tests/cpp/test_smoke.cpp",
        "examples/python/point_query.py",
        "docs/assets/forward_render.png",
        "third_party/glm/glm/glm.hpp",
    ]
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()

    for suffix in required_suffixes:
        if not any(name.endswith(suffix) for name in names):
            raise AssertionError(f"sdist is missing {suffix}")


def _venv_python(directory: Path) -> Path:
    if os.name == "nt":
        return directory / "Scripts" / "python.exe"
    return directory / "bin" / "python"


def _uv_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    return env


def _create_venv(env_dir: Path) -> None:
    uv = shutil.which("uv")
    if uv is not None:
        subprocess.run(
            [uv, "venv", str(env_dir), "--python", sys.executable],
            check=True,
            cwd=ROOT,
            env=_uv_env(),
        )
        return

    venv.EnvBuilder(with_pip=True).create(env_dir)


def _install_wheel(venv_python: Path, wheel: Path) -> None:
    uv = shutil.which("uv")
    if uv is not None:
        subprocess.run(
            [uv, "pip", "install", "--python", str(venv_python), str(wheel)],
            check=True,
            cwd=ROOT,
            env=_uv_env(),
        )
        return

    subprocess.run([str(venv_python), "-m", "pip", "install", str(wheel)], check=True, cwd=ROOT)


def _run_installed_smoke(venv_python: Path, cwd: Path) -> None:
    smoke = r"""
import subprocess
import sys

import numpy as np
import svo

assert isinstance(svo.__version__, str)
assert svo.__version__
info = svo.build_info()
assert isinstance(info["cuda_enabled"], bool)
assert isinstance(info["core_version"], str)
assert info["extension_path"]

coords = np.array([[0, 0, 0], [3, 3, 3]], dtype=np.int32)
tree = svo.Octree.from_voxels(coords, max_depth=2)
points = np.array(
    [[0.125, 0.125, 0.125], [0.875, 0.875, 0.875], [0.5, 0.5, 0.5]],
    dtype=np.float32,
)
assert tree.query(points).tolist() == [0, 1, -1]

camera = svo.Camera.look_at(
    origin=[0.0, 0.0, 0.0],
    target=[0.0, 0.0, -1.0],
    up=[0.0, 1.0, 0.0],
    width=3,
    height=3,
    vertical_fov_y_degrees=60.0,
)
origins, directions = camera.generate_rays()
assert origins.shape == (3, 3, 3)
assert directions.shape == (3, 3, 3)

render_tree = svo.Octree.from_voxels(np.array([[0, 0, 0]], dtype=np.int32), max_depth=0)
rgb, depth, opacity = svo.render_volume(
    render_tree,
    np.array([[-1.0, 0.5, 0.5]], dtype=np.float32),
    np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    np.array([1.0], dtype=np.float32),
    np.array([[0.5, 0.25, 0.125]], dtype=np.float32),
)
assert rgb.shape == (1, 3)
assert depth.shape == (1,)
assert opacity.shape == (1,)

result = subprocess.run(
    [sys.executable, "-m", "svo.info"],
    check=True,
    text=True,
    capture_output=True,
)
assert "svo version:" in result.stdout
assert "C++ core version:" in result.stdout
assert "extension path:" in result.stdout
"""
    subprocess.run([str(venv_python), "-c", textwrap.dedent(smoke)], check=True, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--skip-install", action="store_true", help="Only inspect wheel and sdist contents.")
    args = parser.parse_args()

    wheel = _latest("sparse_voxel_octree-*.whl", args.dist_dir)
    sdist = _latest("sparse_voxel_octree-*.tar.gz", args.dist_dir)

    _check_wheel_contents(wheel)
    _check_sdist_contents(sdist)

    if not args.skip_install:
        with tempfile.TemporaryDirectory(prefix="svo-package-") as tmp:
            tmpdir = Path(tmp)
            env_dir = tmpdir / "venv"
            _create_venv(env_dir)
            python = _venv_python(env_dir)
            _install_wheel(python, wheel)
            _run_installed_smoke(python, tmpdir)

    print(f"package artifacts ok: {wheel.name}, {sdist.name}")


if __name__ == "__main__":
    main()
