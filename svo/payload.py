"""Payload gathering helpers for sparse voxel octree query results."""

from __future__ import annotations

from typing import Any

import numpy as np


def _gather_numpy(payload: np.ndarray, indices: Any, fill_value: Any) -> np.ndarray:
    if payload.ndim < 1:
        raise ValueError("payload must have shape (P,) or (P, ...)")

    index_array = np.asarray(indices)
    if not np.issubdtype(index_array.dtype, np.integer):
        raise TypeError("indices must have an integer dtype")

    output_shape = index_array.shape + payload.shape[1:]
    output = np.empty(output_shape, dtype=payload.dtype)
    output[...] = np.asarray(fill_value, dtype=payload.dtype)

    flat_indices = index_array.reshape(-1)
    flat_output = output.reshape((flat_indices.size,) + payload.shape[1:])
    if flat_indices.size == 0:
        return output

    invalid = (flat_indices < -1) | (flat_indices >= payload.shape[0])
    if bool(np.any(invalid)):
        raise IndexError("indices must be -1 or inside the payload row range")

    valid = flat_indices != -1
    if bool(np.any(valid)):
        flat_output[valid] = payload[flat_indices[valid]]

    return output


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return torch.is_tensor(value)


def _gather_torch(payload: Any, indices: Any, fill_value: Any) -> Any:
    import torch

    if not torch.is_tensor(indices):
        raise TypeError("indices must be a Torch tensor when payload is a Torch tensor")
    if payload.ndim < 1:
        raise ValueError("payload must have shape (P,) or (P, ...)")
    if indices.device != payload.device:
        raise ValueError("indices must be on the same device as payload")
    if indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("indices must have a signed integer dtype")

    output_shape = tuple(indices.shape) + tuple(payload.shape[1:])
    output = torch.empty(output_shape, dtype=payload.dtype, device=payload.device)
    output[...] = torch.as_tensor(fill_value, dtype=payload.dtype, device=payload.device)

    flat_indices = indices.reshape(-1)
    flat_output = output.reshape((flat_indices.numel(),) + tuple(payload.shape[1:]))
    if flat_indices.numel() == 0:
        return output

    invalid = (flat_indices < -1) | (flat_indices >= payload.shape[0])
    if bool(torch.any(invalid).item()):
        raise IndexError("indices must be -1 or inside the payload row range")

    valid = flat_indices != -1
    if bool(torch.any(valid).item()):
        flat_output[valid] = payload[flat_indices[valid].long()]

    return output


def gather_payload(payload: Any, indices: Any, fill_value: Any = 0) -> Any:
    """Gather external payload rows by query/raycast payload indices.

    Misses encoded as -1 are filled with ``fill_value`` instead of indexing the
    final payload row. NumPy payloads accept NumPy-like integer indices. Torch
    payloads require Torch integer indices on the same device.
    """

    if isinstance(payload, np.ndarray):
        return _gather_numpy(payload, indices, fill_value)
    if _is_torch_tensor(payload):
        return _gather_torch(payload, indices, fill_value)
    raise TypeError("payload must be a NumPy array or Torch tensor")
