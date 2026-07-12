"""Runtime diagnostics for sparse-voxel-octree installations."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any

from . import __version__
from . import _svo
from ._svo import cuda_enabled


def _optional_version(module_name: str) -> str | None:
    if importlib.util.find_spec(module_name) is None:
        return None
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _torch_info() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {
            "torch_available": False,
            "torch_version": None,
            "torch_cuda_available": False,
            "torch_cuda_version": None,
            "torch_cuda_device_count": 0,
            "torch_cuda_devices": [],
        }
    try:
        import torch
    except Exception as error:
        return {
            "torch_available": False,
            "torch_import_error": str(error),
            "torch_version": None,
            "torch_cuda_available": False,
            "torch_cuda_version": None,
            "torch_cuda_device_count": 0,
            "torch_cuda_devices": [],
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices = [torch.cuda.get_device_name(index) for index in range(device_count)]
    return {
        "torch_available": True,
        "torch_version": str(torch.__version__),
        "torch_cuda_available": cuda_available,
        "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "torch_cuda_device_count": device_count,
        "torch_cuda_devices": devices,
    }


def build_info() -> dict[str, object]:
    torch = _torch_info()
    return {
        "version": __version__,
        "core_version": _svo._core_version(),
        "extension_path": str(Path(_svo.__file__).resolve()),
        "cuda_enabled": cuda_enabled(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": _optional_version("numpy"),
        **torch,
        "torch_cuda_interop_available": cuda_enabled() and bool(torch["torch_cuda_available"]),
    }


def _yes_no(value: object) -> str:
    return "yes" if bool(value) else "no"


def main() -> None:
    info = build_info()
    print(f"svo version: {info['version']}")
    print(f"C++ core version: {info['core_version']}")
    print(f"extension path: {info['extension_path']}")
    print(f"CUDA extension loaded: {_yes_no(info['cuda_enabled'])}")
    print(f"python version: {info['python_version']}")
    print(f"python executable: {info['python_executable']}")
    print(f"platform: {info['platform']}")
    print(f"machine: {info['machine']}")
    print(f"numpy version: {info['numpy_version']}")
    print(f"torch available: {_yes_no(info['torch_available'])}")
    print(f"torch version: {info.get('torch_version')}")
    print(f"torch CUDA available: {_yes_no(info['torch_cuda_available'])}")
    print(f"torch CUDA version: {info['torch_cuda_version']}")
    print(f"torch CUDA device count: {info['torch_cuda_device_count']}")
    for index, name in enumerate(info["torch_cuda_devices"]):
        print(f"torch CUDA device {index}: {name}")
    if "torch_import_error" in info:
        print(f"torch import error: {info['torch_import_error']}")
    print(f"torch CUDA interop available: {_yes_no(info['torch_cuda_interop_available'])}")


if __name__ == "__main__":
    main()
