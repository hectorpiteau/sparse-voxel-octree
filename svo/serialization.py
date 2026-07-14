"""SVO scene serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._svo import _load_svo, _save_svo


@dataclass(frozen=True)
class LoadedSvo:
    tree: Any
    payloads: dict[str, Any]
    cuda_tree: Any | None = None


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return torch.is_tensor(value)


def _payload_row_count(tree: Any) -> int:
    indices = np.asarray(tree.leaf_payload_indices, dtype=np.int64)
    if indices.size == 0:
        return 0
    return int(indices.max()) + 1


def _payload_to_numpy(name: str, value: Any, expected_rows: int) -> np.ndarray:
    if _is_torch_tensor(value):
        array = value.detach().to("cpu").numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"payload '{name}' must have at least one dimension")
    if array.dtype != np.float32:
        raise TypeError(f"payload '{name}' must have dtype float32")
    if array.shape[0] != expected_rows:
        raise ValueError(f"payload '{name}' first dimension must match payload row count {expected_rows}")
    return np.ascontiguousarray(array)


def save(path: str | Path, tree: Any, payloads: Mapping[str, Any] | None = None) -> None:
    """Save an octree and optional float32 payload arrays to a `.svo` file."""

    tree_cpu = tree.to("cpu") if getattr(tree, "device", None) == "cuda" else tree
    expected_rows = _payload_row_count(tree_cpu)
    payload_dict: dict[str, np.ndarray] = {}
    if payloads is not None:
        for name, value in payloads.items():
            if not isinstance(name, str) or not name:
                raise ValueError("payload names must be non-empty strings")
            payload_dict[name] = _payload_to_numpy(name, value, expected_rows)
    _save_svo(str(path), tree_cpu, payload_dict)


def load(path: str | Path, *, device: str = "cpu") -> LoadedSvo:
    """Load a `.svo` file.

    Args:
        path: File produced by :func:`save`.
        device: `"cpu"` returns NumPy payloads and a CPU tree. `"cuda"` also
            uploads the tree and payloads to CUDA Torch tensors.
    """

    tree, payloads = _load_svo(str(path))
    if device == "cpu":
        return LoadedSvo(tree=tree, payloads=dict(payloads), cuda_tree=None)
    if device != "cuda":
        raise ValueError("device must be 'cpu' or 'cuda'")

    import svo

    if not svo.cuda_enabled():
        raise RuntimeError("loading a .svo file on CUDA requires SVO_ENABLE_CUDA=ON")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("loading a .svo file on CUDA requires PyTorch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("loading a .svo file on CUDA requires torch.cuda")

    cuda_tree = tree.to("cuda")
    cuda_payloads = {
        name: torch.as_tensor(np.ascontiguousarray(array), device="cuda") for name, array in dict(payloads).items()
    }
    return LoadedSvo(tree=tree, payloads=cuda_payloads, cuda_tree=cuda_tree)
