"""Forward volume rendering helpers."""

from __future__ import annotations

from math import inf
from typing import Any

import numpy as np

from ._svo import _render_volume_cpu

try:
    import torch as _torch
except ModuleNotFoundError:
    _torch = None
    _TorchModuleBase = object
else:
    _TorchModuleBase = _torch.nn.Module


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return torch.is_tensor(value)


def _normalize_render_strategy(render_strategy: str) -> str:
    if not isinstance(render_strategy, str):
        raise TypeError("render_strategy must be a string")
    strategy = render_strategy.lower()
    if strategy == "auto":
        return "direct"
    if strategy not in {"direct", "intervals"}:
        raise ValueError("render_strategy must be 'direct', 'intervals', or 'auto'")
    return strategy


class _RenderVolumeFunction:
    @staticmethod
    def apply(
        tree: Any,
        origins: Any,
        directions: Any,
        sigma: Any,
        color: Any,
        near: float,
        far: float,
        background_color: tuple[float, float, float],
        early_stop_transmittance: float,
        enable_empty_space_skipping: bool,
        render_strategy: str,
    ) -> tuple[Any, Any, Any]:
        import torch

        class _Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, origins_tensor, directions_tensor, sigma_tensor, color_tensor):
                interval_buffer = None
                needs_payload_backward = bool(sigma_tensor.requires_grad or color_tensor.requires_grad)
                if render_strategy == "intervals":
                    rgb, depth, opacity, interval_buffer = tree._render_volume_intervals_torch(
                        origins_tensor,
                        directions_tensor,
                        sigma_tensor,
                        color_tensor,
                        near,
                        far,
                        background_color,
                        early_stop_transmittance,
                        False,
                        enable_empty_space_skipping,
                    )
                else:
                    rgb, depth, opacity = tree._render_volume_torch(
                        origins_tensor,
                        directions_tensor,
                        sigma_tensor,
                        color_tensor,
                        near,
                        far,
                        background_color,
                        early_stop_transmittance,
                        False,
                        enable_empty_space_skipping,
                    )
                ctx.tree = tree if needs_payload_backward else None
                ctx.near = near
                ctx.far = far
                ctx.background_color = background_color
                ctx.early_stop_transmittance = early_stop_transmittance
                ctx.enable_empty_space_skipping = enable_empty_space_skipping
                ctx.render_strategy = render_strategy
                ctx.interval_buffer = interval_buffer if needs_payload_backward else None
                if needs_payload_backward:
                    ctx.save_for_backward(origins_tensor, directions_tensor, sigma_tensor, color_tensor)
                ctx.mark_non_differentiable(depth)
                return rgb, depth, opacity

            @staticmethod
            def backward(ctx, grad_rgb, grad_depth, grad_opacity):
                del grad_depth
                if not (ctx.needs_input_grad[2] or ctx.needs_input_grad[3]):
                    ctx.tree = None
                    ctx.interval_buffer = None
                    return None, None, None, None
                origins_tensor, directions_tensor, sigma_tensor, color_tensor = ctx.saved_tensors
                if grad_rgb is None:
                    grad_rgb = torch.zeros(
                        (*origins_tensor.shape[:-1], 3),
                        device=origins_tensor.device,
                        dtype=torch.float32,
                    )
                if grad_opacity is None:
                    grad_opacity = torch.zeros(
                        origins_tensor.shape[:-1],
                        device=origins_tensor.device,
                        dtype=torch.float32,
                    )

                grad_sigma = None
                grad_color = None
                if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
                    if ctx.render_strategy == "intervals":
                        grad_sigma_tensor, grad_color_tensor = ctx.tree._render_volume_backward_intervals_torch(
                            origins_tensor,
                            directions_tensor,
                            sigma_tensor,
                            color_tensor,
                            grad_rgb.contiguous(),
                            grad_opacity.contiguous(),
                            ctx.interval_buffer,
                            ctx.near,
                            ctx.far,
                            ctx.background_color,
                            ctx.early_stop_transmittance,
                            False,
                            ctx.enable_empty_space_skipping,
                        )
                    else:
                        grad_sigma_tensor, grad_color_tensor = ctx.tree._render_volume_backward_torch(
                            origins_tensor,
                            directions_tensor,
                            sigma_tensor,
                            color_tensor,
                            grad_rgb.contiguous(),
                            grad_opacity.contiguous(),
                            ctx.near,
                            ctx.far,
                            ctx.background_color,
                            ctx.early_stop_transmittance,
                            False,
                            ctx.enable_empty_space_skipping,
                        )
                    if ctx.needs_input_grad[2]:
                        grad_sigma = grad_sigma_tensor
                    if ctx.needs_input_grad[3]:
                        grad_color = grad_color_tensor
                ctx.tree = None
                ctx.interval_buffer = None
                return None, None, grad_sigma, grad_color

        return _Function.apply(origins, directions, sigma, color)


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
    render_strategy: str = "direct",
) -> tuple[Any, Any, Any]:
    """Render leaf sigma/color payloads along ray batches.

    NumPy inputs run the CPU reference path and return NumPy arrays. CUDA Torch
    inputs run on a CUDA-owned octree from ``tree.to("cuda")`` and return CUDA
    Torch tensors. Torch CUDA rendering supports autograd for ``sigma`` and
    ``color``; depth is forward-only. Shapes are ``(N, 3)`` or ``(H, W, 3)``
    for rays, ``(P,)`` for ``sigma``, and ``(P, 3)`` for ``color``.
    """

    if store_aux:
        raise NotImplementedError("renderer aux buffers are not exposed; backward recomputes traversal")

    strategy = _normalize_render_strategy(render_strategy)
    far_plane = inf if far is None else float(far)
    if isinstance(origins, np.ndarray) or isinstance(directions, np.ndarray) or isinstance(sigma, np.ndarray) or isinstance(color, np.ndarray):
        if not all(isinstance(value, np.ndarray) for value in (origins, directions, sigma, color)):
            raise TypeError("origins, directions, sigma, and color must all be NumPy arrays for CPU rendering")
        if strategy == "intervals":
            raise NotImplementedError("render_strategy='intervals' is only implemented for Torch CUDA rendering")
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
        return _RenderVolumeFunction.apply(
            tree,
            origins,
            directions,
            sigma,
            color,
            float(near),
            far_plane,
            background_color,
            float(early_stop_transmittance),
            bool(enable_empty_space_skipping),
            strategy,
        )

    raise TypeError("renderer inputs must be all NumPy arrays or all Torch tensors")


class VolumeRenderer(_TorchModuleBase):
    """Small stateless ``torch.nn.Module`` wrapper around ``render_volume``."""

    def __init__(
        self,
        tree: Any,
        *,
        near: float = 0.0,
        far: float | None = None,
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        early_stop_transmittance: float = 1.0e-4,
        enable_empty_space_skipping: bool = True,
        render_strategy: str = "direct",
    ) -> None:
        if _torch is None:
            raise ModuleNotFoundError("VolumeRenderer requires PyTorch; install torch to use the renderer module")
        super().__init__()
        if len(background_color) != 3:
            raise ValueError("background_color must contain exactly three values")
        self.tree = tree
        self.near = float(near)
        self.far = None if far is None else float(far)
        self.background_color = tuple(float(value) for value in background_color)
        self.early_stop_transmittance = float(early_stop_transmittance)
        self.enable_empty_space_skipping = bool(enable_empty_space_skipping)
        self.render_strategy = _normalize_render_strategy(render_strategy)

    def forward(self, origins: Any, directions: Any, sigma: Any, color: Any) -> tuple[Any, Any, Any]:
        return render_volume(
            self.tree,
            origins,
            directions,
            sigma,
            color,
            near=self.near,
            far=self.far,
            background_color=self.background_color,
            early_stop_transmittance=self.early_stop_transmittance,
            enable_empty_space_skipping=self.enable_empty_space_skipping,
            render_strategy=self.render_strategy,
        )

    def extra_repr(self) -> str:
        far = "None" if self.far is None else f"{self.far:g}"
        return (
            f"near={self.near:g}, far={far}, background_color={self.background_color}, "
            f"early_stop_transmittance={self.early_stop_transmittance:g}, "
            f"enable_empty_space_skipping={self.enable_empty_space_skipping}, "
            f"render_strategy='{self.render_strategy}'"
        )
