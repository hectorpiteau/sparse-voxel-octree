"""Forward volume rendering helpers."""

from __future__ import annotations

from math import inf
from typing import Any

import numpy as np

from ._svo import _render_volume_cpu


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return torch.is_tensor(value)


def render_volume(
    tree: Any,
    origins: Any,
    directions: Any,
    sigma: Any,
    color: Any,
    *,
    near: float = 0.0,
    far: float | None = None,
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    early_stop_transmittance: float = 1.0e-4,
    store_aux: bool = False,
    enable_empty_space_skipping: bool = True,
) -> tuple[Any, Any, Any]:
    """Render leaf sigma/color payloads along ray batches.

    NumPy inputs run the CPU reference path and return NumPy arrays. CUDA Torch
    inputs run on a CUDA-owned octree from ``tree.to("cuda")`` and return CUDA
    Torch tensors. Shapes are ``(N, 3)`` or ``(H, W, 3)`` for rays, ``(P,)``
    for ``sigma``, and ``(P, 3)`` for ``color``.
    """

    if store_aux:
        raise NotImplementedError("renderer aux buffers are reserved for the renderer backward milestone")

    far_plane = inf if far is None else float(far)
    if isinstance(origins, np.ndarray) or isinstance(directions, np.ndarray) or isinstance(sigma, np.ndarray) or isinstance(color, np.ndarray):
        if not all(isinstance(value, np.ndarray) for value in (origins, directions, sigma, color)):
            raise TypeError("origins, directions, sigma, and color must all be NumPy arrays for CPU rendering")
        if hasattr(tree, "_render_volume_torch"):
            raise TypeError("NumPy rendering requires a CPU octree; use Torch CUDA tensors with CudaOctree to stay on GPU")
        return _render_volume_cpu(
            tree,
            origins,
            directions,
            sigma,
            color,
            float(near),
            far_plane,
            background_color,
            float(early_stop_transmittance),
            False,
            bool(enable_empty_space_skipping),
        )

    if _is_torch_tensor(origins) or _is_torch_tensor(directions) or _is_torch_tensor(sigma) or _is_torch_tensor(color):
        import torch

        if not all(torch.is_tensor(value) for value in (origins, directions, sigma, color)):
            raise TypeError("origins, directions, sigma, and color must all be Torch tensors for CUDA rendering")
        if not hasattr(tree, "_render_volume_torch"):
            raise TypeError("Torch rendering requires a CUDA-owned octree from tree.to('cuda')")
        return tree._render_volume_torch(
            origins,
            directions,
            sigma,
            color,
            float(near),
            far_plane,
            background_color,
            float(early_stop_transmittance),
            False,
            bool(enable_empty_space_skipping),
        )

    raise TypeError("renderer inputs must be all NumPy arrays or all Torch tensors")
