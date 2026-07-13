from __future__ import annotations

import gc

import numpy as np

import svo


def _cleanup(*objects: object) -> None:
    for obj in objects:
        release = getattr(obj, "_release", None)
        if release is not None:
            release()
    del objects
    gc.collect()

    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def _run_case(branching: str) -> None:
    import torch

    coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    max_depth = 1
    yz = 0.25
    if branching == "wide4":
        coords = np.array([[0, 0, 0], [4, 0, 0]], dtype=np.int32)
        max_depth = 4
        yz = 0.5 / 16.0

    tree = svo.Octree.from_voxels(coords, max_depth=max_depth, branching=branching)
    cuda_tree = tree.to("cuda")
    origins = torch.tensor([[-1.0, yz, yz], [-1.0, yz, yz]], device="cuda", dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    sigma = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32, requires_grad=True)
    color = torch.tensor([[1.0, 0.2, 0.0], [0.0, 1.0, 0.4]], device="cuda", dtype=torch.float32, requires_grad=True)

    rgb, _depth, opacity = svo.render_volume(
        cuda_tree,
        origins,
        directions,
        sigma,
        color,
        render_strategy="intervals",
    )
    loss = 0.7 * rgb[..., 0].sum() + 0.2 * rgb[..., 1].sum() + 0.4 * opacity.sum()
    loss.backward()
    torch.cuda.synchronize()

    if sigma.grad is None or color.grad is None:
        raise RuntimeError("interval backward did not produce gradients")
    if not torch.isfinite(sigma.grad).all() or not torch.isfinite(color.grad).all():
        raise RuntimeError("interval backward produced non-finite gradients")

    _cleanup(cuda_tree, tree, origins, directions, sigma, color, rgb, opacity, loss)


def main() -> None:
    import torch

    if not svo.cuda_enabled():
        raise RuntimeError("SVO extension is not built with CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is not available")

    _run_case("octree8")
    _run_case("wide4")
    _cleanup()
    print("interval sanitizer smoke passed")


if __name__ == "__main__":
    main()
