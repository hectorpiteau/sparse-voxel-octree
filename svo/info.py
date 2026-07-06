"""Runtime diagnostics for the milestone 0 scaffold."""

from __future__ import annotations

import importlib.util

from . import __version__
from ._svo import cuda_enabled


def _torch_cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def build_info() -> dict[str, object]:
    torch_available = importlib.util.find_spec("torch") is not None
    torch_cuda_available = _torch_cuda_available()
    return {
        "version": __version__,
        "cuda_enabled": cuda_enabled(),
        "torch_available": torch_available,
        "torch_cuda_available": torch_cuda_available,
        "torch_cuda_interop_available": cuda_enabled() and torch_cuda_available,
    }


def main() -> None:
    info = build_info()
    print(f"svo version: {info['version']}")
    print(f"CUDA extension loaded: {'yes' if info['cuda_enabled'] else 'no'}")
    print(f"torch available: {'yes' if info['torch_available'] else 'no'}")
    print(f"torch CUDA available: {'yes' if info['torch_cuda_available'] else 'no'}")
    print(f"torch CUDA interop available: {'yes' if info['torch_cuda_interop_available'] else 'no'}")


if __name__ == "__main__":
    main()
