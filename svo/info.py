"""Runtime diagnostics for the milestone 0 scaffold."""

from __future__ import annotations

import importlib.util

from . import __version__
from ._svo import cuda_enabled


def build_info() -> dict[str, object]:
    return {
        "version": __version__,
        "cuda_enabled": cuda_enabled(),
        "torch_available": importlib.util.find_spec("torch") is not None,
    }


def main() -> None:
    info = build_info()
    print(f"svo version: {info['version']}")
    print(f"CUDA extension loaded: {'yes' if info['cuda_enabled'] else 'no'}")
    print(f"torch available: {'yes' if info['torch_available'] else 'no'}")


if __name__ == "__main__":
    main()
