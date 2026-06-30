"""Python package scaffold for sparse voxel octree bindings."""

__version__ = "0.1.0"

from ._svo import Octree, ValidationError, cuda_enabled

__all__ = ["__version__", "Octree", "ValidationError", "build_info", "cuda_enabled"]


def build_info():
    from .info import build_info as _build_info

    return _build_info()


def cuda_enabled() -> bool:
    from .info import cuda_enabled as _cuda_enabled

    return _cuda_enabled()
