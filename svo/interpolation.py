"""Leaf-centered trilinear interpolation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._svo import _sample_trilinear_cpu


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return torch.is_tensor(value)


class _TrilinearSampleFunction:
    @staticmethod
    def apply(tree: Any, points: Any, payload: Any, fill_value: float) -> Any:
        import torch

        class _Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, points_tensor, payload_tensor):
                ctx.tree = tree
                ctx.fill_value = float(fill_value)
                ctx.save_for_backward(points_tensor, payload_tensor)
                return tree._sample_trilinear_torch(points_tensor, payload_tensor, ctx.fill_value)

            @staticmethod
            def backward(ctx, grad_outputs):
                points_tensor, payload_tensor = ctx.saved_tensors
                grad_payload = None
                if ctx.needs_input_grad[1]:
                    grad_payload = ctx.tree._sample_trilinear_backward_torch(
                        points_tensor,
                        payload_tensor,
                        grad_outputs.contiguous(),
                        ctx.fill_value,
                    )
                return None, grad_payload

        return _Function.apply(points, payload)


def sample_trilinear(tree: Any, points: Any, payload: Any, fill_value: float = 0.0) -> Any:
    """Sample leaf payload values with trilinear interpolation.

    NumPy inputs run the CPU reference path. CUDA Torch inputs run on the
    CUDA-resident octree and support autograd for ``payload`` only. Points use
    shape ``(..., 3)``. Payload uses shape ``(P,)`` or ``(P, C)``. Missing
    neighboring leaves and out-of-root samples contribute ``fill_value``.
    """

    if isinstance(points, np.ndarray) and isinstance(payload, np.ndarray):
        return _sample_trilinear_cpu(tree, points, payload, float(fill_value))

    if _is_torch_tensor(points) or _is_torch_tensor(payload):
        import torch

        if not torch.is_tensor(points) or not torch.is_tensor(payload):
            raise TypeError("points and payload must both be Torch tensors for Torch interpolation")
        if not hasattr(tree, "_sample_trilinear_torch"):
            raise TypeError("Torch interpolation requires a CUDA-owned octree from tree.to('cuda')")
        return _TrilinearSampleFunction.apply(tree, points, payload, float(fill_value))

    raise TypeError("points and payload must both be NumPy arrays or both be Torch tensors")
