"""Python package scaffold for sparse voxel octree bindings."""

__version__ = "0.1.0"

from ._svo import Camera, CameraConvention, CameraIntrinsics, Octree, ValidationError, cuda_enabled
from .payload import gather_payload
from .interpolation import sample_trilinear
from .rendering import render_volume

__all__ = [
    "__version__",
    "Camera",
    "CameraConvention",
    "CameraIntrinsics",
    "Octree",
    "ValidationError",
    "build_info",
    "gather_payload",
    "sample_trilinear",
    "render_volume",
    "cuda_enabled",
]


def build_info():
    from .info import build_info as _build_info

    return _build_info()


def cuda_enabled() -> bool:
    from .info import cuda_enabled as _cuda_enabled

    return _cuda_enabled()
