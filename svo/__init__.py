"""Python bindings for sparse-voxel-octree."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sparse-voxel-octree")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from ._svo import BranchingMode, Camera, CameraConvention, CameraIntrinsics, Octree, ValidationError
from .interpolation import sample_trilinear
from .payload import gather_payload
from .refinement import RefinementResult, refine_octree
from .rendering import VolumeRenderer, render_volume
from .serialization import LoadedSvo, load, save

__all__ = [
    "__version__",
    "BranchingMode",
    "Camera",
    "CameraConvention",
    "CameraIntrinsics",
    "Octree",
    "LoadedSvo",
    "RefinementResult",
    "ValidationError",
    "VolumeRenderer",
    "build_info",
    "gather_payload",
    "load",
    "sample_trilinear",
    "save",
    "refine_octree",
    "render_volume",
    "cuda_enabled",
]


def build_info():
    from .info import build_info as _build_info

    return _build_info()


def cuda_enabled() -> bool:
    from .info import cuda_enabled as _cuda_enabled

    return _cuda_enabled()
