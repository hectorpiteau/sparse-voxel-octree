"""Reconstruct a Blender/NeRF scene with an adaptive sparse voxel octree.

This is a small CUDA-first experiment script. It optimizes view-independent
density and RGB payloads from photometric RGB loss, periodically rebuilds the
Octree8 topology from density thresholds, and saves PNG checkpoints.

Example:
    ./.venv/bin/python examples/python/reconstruct_nerf_octree.py --data data/nerf --steps 2000
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import svo

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class NerfFrame:
    image_path: Path
    transform_matrix: np.ndarray


@dataclass(frozen=True)
class NerfDataset:
    images: np.ndarray
    origins: np.ndarray
    directions: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class LoadedFrame:
    image: np.ndarray
    origins: np.ndarray
    directions: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class CudaNerfDataset:
    images: Any
    origins: Any
    directions: Any
    width: int
    height: int
    image_count: int
    pixels_per_image: int


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", checksum)


def _paeth_predictor(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    dist_left = abs(estimate - left)
    dist_above = abs(estimate - above)
    dist_upper_left = abs(estimate - upper_left)
    if dist_left <= dist_above and dist_left <= dist_upper_left:
        return left
    if dist_above <= dist_upper_left:
        return above
    return upper_left


def read_png_rgba(path: Path) -> np.ndarray:
    """Read a non-interlaced 8-bit RGB/RGBA PNG as float32 RGBA in [0, 1]."""
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path} is not a PNG file")

    offset = len(PNG_SIGNATURE)
    width = height = color_type = bit_depth = interlace = None
    compressed_parts: list[bytes] = []
    while offset < len(data):
        chunk_size = struct.unpack(">I", data[offset : offset + 4])[0]
        tag = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + chunk_size]
        offset += 12 + chunk_size
        if tag == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace = struct.unpack(">IIBBBBB", payload)
        elif tag == b"IDAT":
            compressed_parts.append(payload)
        elif tag == b"IEND":
            break

    if width is None or height is None or color_type is None or bit_depth is None or interlace is None:
        raise ValueError(f"{path} is missing PNG header data")
    if bit_depth != 8 or color_type not in (2, 6) or interlace != 0:
        raise ValueError("only non-interlaced 8-bit RGB/RGBA PNGs are supported")

    channels = 4 if color_type == 6 else 3
    row_bytes = width * channels
    raw = zlib.decompress(b"".join(compressed_parts))
    expected = height * (row_bytes + 1)
    if len(raw) != expected:
        raise ValueError(f"{path} has unexpected decompressed PNG size")

    rows = np.empty((height, row_bytes), dtype=np.uint8)
    previous = np.zeros(row_bytes, dtype=np.uint8)
    cursor = 0
    for row_index in range(height):
        filter_type = raw[cursor]
        cursor += 1
        encoded = np.frombuffer(raw[cursor : cursor + row_bytes], dtype=np.uint8).copy()
        cursor += row_bytes
        decoded = np.empty_like(encoded)
        for byte_index in range(row_bytes):
            left = decoded[byte_index - channels] if byte_index >= channels else np.uint8(0)
            above = previous[byte_index]
            upper_left = previous[byte_index - channels] if byte_index >= channels else np.uint8(0)
            encoded_value = int(encoded[byte_index])
            if filter_type == 0:
                value = encoded_value
            elif filter_type == 1:
                value = encoded_value + int(left)
            elif filter_type == 2:
                value = encoded_value + int(above)
            elif filter_type == 3:
                value = encoded_value + ((int(left) + int(above)) // 2)
            elif filter_type == 4:
                value = encoded_value + _paeth_predictor(int(left), int(above), int(upper_left))
            else:
                raise ValueError(f"{path} uses unsupported PNG filter {filter_type}")
            decoded[byte_index] = value & 0xFF
        rows[row_index] = decoded
        previous = decoded

    image = rows.reshape(height, width, channels).astype(np.float32) / 255.0
    if channels == 3:
        alpha = np.ones((height, width, 1), dtype=np.float32)
        image = np.concatenate([image, alpha], axis=-1)
    return image


def write_png_rgb(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a uint8 RGB array")

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _channels = image.shape
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = PNG_SIGNATURE
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw_rows, level=9))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def composite_rgba_white(image: np.ndarray) -> np.ndarray:
    rgb = image[..., :3]
    alpha = image[..., 3:4]
    return rgb * alpha + (1.0 - alpha)


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1.0e-12))


def make_root_bounds(root_center: tuple[float, float, float] | list[float], root_size: float) -> np.ndarray:
    center = np.asarray(root_center, dtype=np.float32)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("root_center must contain three finite values")
    if not np.isfinite(root_size) or root_size <= 0.0:
        raise ValueError("root_size must be finite and positive")
    half_extent = np.float32(root_size * 0.5)
    return np.stack([center - half_extent, center + half_extent]).astype(np.float32)


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def load_transforms(data_dir: Path, split: str) -> tuple[float, list[NerfFrame]]:
    transform_path = data_dir / f"transforms_{split}.json"
    with transform_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    camera_angle_x = float(metadata["camera_angle_x"])
    frames = []
    for frame in metadata["frames"]:
        relative_path = Path(frame["file_path"])
        image_path = data_dir / relative_path
        if image_path.suffix == "":
            image_path = image_path.with_suffix(".png")
        frames.append(
            NerfFrame(
                image_path=image_path,
                transform_matrix=np.asarray(frame["transform_matrix"], dtype=np.float32),
            )
        )
    return camera_angle_x, frames


def camera_rays_from_blender(
    transform_matrix: np.ndarray,
    *,
    width: int,
    height: int,
    camera_angle_x: float,
) -> tuple[np.ndarray, np.ndarray]:
    c2w = np.asarray(transform_matrix, dtype=np.float32)
    origin = c2w[:3, 3]
    forward = -c2w[:3, 2]
    up = c2w[:3, 1]
    target = origin + forward
    focal = 0.5 * float(width) / math.tan(0.5 * float(camera_angle_x))
    intrinsics = svo.CameraIntrinsics(width, height, focal, focal, 0.5 * width, 0.5 * height)
    camera = svo.Camera.from_intrinsics(
        origin=origin,
        target=target,
        up=up,
        intrinsics=intrinsics,
        convention=svo.CameraConvention.OpenGL,
    )
    return camera.generate_rays()



def load_frame(frame: NerfFrame, camera_angle_x: float) -> LoadedFrame:
    rgba = read_png_rgba(frame.image_path)
    rgb = composite_rgba_white(rgba)
    height, width = rgb.shape[:2]
    ray_origins, ray_directions = camera_rays_from_blender(
        frame.transform_matrix,
        width=int(width),
        height=int(height),
        camera_angle_x=camera_angle_x,
    )
    return LoadedFrame(
        image=rgb.astype(np.float32),
        origins=ray_origins.reshape(-1, 3).astype(np.float32),
        directions=ray_directions.reshape(-1, 3).astype(np.float32),
        width=int(width),
        height=int(height),
    )


def _load_frame_worker(args: tuple[NerfFrame, float]) -> LoadedFrame:
    frame, camera_angle_x = args
    return load_frame(frame, camera_angle_x)


def _default_load_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def load_frames_parallel(
    frames: list[NerfFrame],
    camera_angle_x: float,
    *,
    workers: int,
    backend: str,
) -> list[LoadedFrame]:
    if workers <= 1 or len(frames) <= 1:
        return [load_frame(frame, camera_angle_x) for frame in frames]

    if backend == "threads":
        executor_type = concurrent.futures.ThreadPoolExecutor
    elif backend == "processes":
        executor_type = concurrent.futures.ProcessPoolExecutor
    else:
        raise ValueError("load_backend must be 'threads' or 'processes'")

    worker_args = [(frame, camera_angle_x) for frame in frames]
    with executor_type(max_workers=workers) as executor:
        return list(executor.map(_load_frame_worker, worker_args))


def load_nerf_dataset(
    data_dir: Path,
    split: str,
    max_images: int | None = None,
    load_workers: int | None = None,
    load_backend: str = "threads",
) -> NerfDataset:
    camera_angle_x, frames = load_transforms(data_dir, split)
    if max_images is not None:
        frames = frames[:max_images]
    if not frames:
        raise ValueError("dataset split contains no frames")

    workers = _default_load_workers() if load_workers is None else int(load_workers)
    load_start = time.perf_counter()
    loaded_frames = load_frames_parallel(frames, camera_angle_x, workers=workers, backend=load_backend)
    print(
        f"loaded {len(loaded_frames)} frames with {load_backend}/{max(1, workers)} "
        f"in {(time.perf_counter() - load_start) * 1000.0:.3f} ms"
    )

    width = loaded_frames[0].width
    height = loaded_frames[0].height
    for frame in loaded_frames:
        if frame.width != width or frame.height != height:
            raise ValueError("all training images must share the same resolution")

    return NerfDataset(
        images=np.stack([frame.image for frame in loaded_frames]).astype(np.float32),
        origins=np.stack([frame.origins for frame in loaded_frames]).astype(np.float32),
        directions=np.stack([frame.directions for frame in loaded_frames]).astype(np.float32),
        width=width,
        height=height,
    )


def require_cuda_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("this reconstruction example requires PyTorch") from error
    if not svo.cuda_enabled():
        raise RuntimeError("this reconstruction example requires SVO_ENABLE_CUDA=ON")
    if not torch.cuda.is_available():
        raise RuntimeError("this reconstruction example requires torch.cuda")
    return torch


def dataset_to_cuda(torch: Any, dataset: NerfDataset, device: str) -> CudaNerfDataset:
    image_count = dataset.images.shape[0]
    pixels_per_image = dataset.width * dataset.height
    return CudaNerfDataset(
        images=torch.as_tensor(dataset.images.reshape(image_count, pixels_per_image, 3), device=device),
        origins=torch.as_tensor(dataset.origins, device=device),
        directions=torch.as_tensor(dataset.directions, device=device),
        width=dataset.width,
        height=dataset.height,
        image_count=image_count,
        pixels_per_image=pixels_per_image,
    )


def make_parameters(torch: Any, leaf_count: int, device: str) -> tuple[Any, Any]:
    raw_sigma = torch.full((leaf_count,), -4.0, device=device, dtype=torch.float32, requires_grad=True)
    raw_color = torch.zeros((leaf_count, 3), device=device, dtype=torch.float32, requires_grad=True)
    return raw_sigma, raw_color


def positive_sigma(torch: Any, raw_sigma: Any) -> Any:
    return torch.nn.functional.softplus(raw_sigma)


def bounded_color(torch: Any, raw_color: Any) -> Any:
    return torch.sigmoid(raw_color)


def render_payloads(torch: Any, raw_sigma: Any, raw_color: Any) -> tuple[Any, Any]:
    return positive_sigma(torch, raw_sigma), bounded_color(torch, raw_color)


def make_optimizer(torch: Any, raw_sigma: Any, raw_color: Any, learning_rate: float) -> Any:
    return torch.optim.Adam([raw_sigma, raw_color], lr=learning_rate)


def sample_training_rays(torch: Any, dataset: CudaNerfDataset, rays_per_step: int, device: str) -> tuple[Any, Any, Any]:
    image_indices = torch.randint(dataset.image_count, (rays_per_step,), device=device)
    pixel_indices = torch.randint(dataset.pixels_per_image, (rays_per_step,), device=device)

    origins = dataset.origins[image_indices, pixel_indices]
    directions = dataset.directions[image_indices, pixel_indices]
    target = dataset.images[image_indices, pixel_indices]
    return origins.contiguous(), directions.contiguous(), target.contiguous()


def render_debug_image(
    torch: Any,
    cuda_tree: Any,
    dataset: CudaNerfDataset,
    raw_sigma: Any,
    raw_color: Any,
    *,
    image_index: int,
    device: str,
) -> np.ndarray:
    origins = dataset.origins[image_index]
    directions = dataset.directions[image_index]
    sigma, color = render_payloads(torch, raw_sigma, raw_color)
    with torch.no_grad():
        rgb, _depth, _opacity = svo.render_volume(
            cuda_tree,
            origins,
            directions,
            sigma,
            color,
            background_color=(1.0, 1.0, 1.0),
        )
    return rgb.reshape(dataset.height, dataset.width, 3).detach().cpu().numpy()


def refine_training_state(
    torch: Any,
    cuda_tree: Any,
    raw_sigma: Any,
    raw_color: Any,
    *,
    learning_rate: float,
    split_threshold: float,
    prune_threshold: float,
    merge_threshold: float | None,
    min_depth: int,
    max_leaf_growth: float,
) -> tuple[Any, Any, Any, Any, dict[str, int]]:
    with torch.no_grad():
        sigma, color = render_payloads(torch, raw_sigma, raw_color)
        result = svo.refine_octree(
            cuda_tree,
            sigma,
            color,
            split_threshold=split_threshold,
            prune_threshold=prune_threshold,
            merge_threshold=merge_threshold,
            min_depth=min_depth,
            leaf_scores=sigma.detach(),
            max_leaf_growth=max_leaf_growth,
        )
        # Convert constrained payload values back to unconstrained optimizer parameters.
        new_sigma = torch.clamp(result.sigma.detach(), min=1.0e-6)
        new_color = torch.clamp(result.color.detach(), min=1.0e-6, max=1.0 - 1.0e-6)
        raw_sigma = torch.log(torch.expm1(new_sigma)).detach().requires_grad_(True)
        raw_color = torch.logit(new_color).detach().requires_grad_(True)
        optimizer = make_optimizer(torch, raw_sigma, raw_color, learning_rate)
    return result.cuda_tree, raw_sigma, raw_color, optimizer, result.stats


def train(args: argparse.Namespace) -> None:
    print("start train")
    torch = require_cuda_torch()
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("start loading nerf dataset...")
    dataset_cpu = load_nerf_dataset(
        args.data,
        args.split,
        max_images=args.max_images,
        load_workers=args.load_workers,
        load_backend=args.load_backend,
    )
    print("dataset loaded on cpu")
    if args.debug_image >= dataset_cpu.images.shape[0]:
        raise ValueError("debug-image must be smaller than the number of loaded images")
    dataset = dataset_to_cuda(torch, dataset_cpu, device)
    print("dataset on cuda")

    root_bounds = make_root_bounds(args.root_center, args.root_size)
    tree = svo.Octree.full_grid(max_depth=args.max_depth, leaf_depth=args.leaf_depth, root_bounds=root_bounds)
    print(f"svo created with root_bounds={root_bounds.tolist()}")

    cuda_tree = tree.to("cuda")
    raw_sigma, raw_color = make_parameters(torch, cuda_tree.num_leaves, device)
    optimizer = make_optimizer(torch, raw_sigma, raw_color, args.learning_rate)

    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"loaded {dataset.images.shape[0]} images at {dataset.width}x{dataset.height}; "
        f"initial leaves={cuda_tree.num_leaves} nodes={cuda_tree.num_nodes}"
    )

    for step in range(1, args.steps + 1):
        origins, directions, target = sample_training_rays(torch, dataset, args.rays_per_step, device)
        sigma, color = render_payloads(torch, raw_sigma, raw_color)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        render_start = time.perf_counter()
        pred, _depth, _opacity = svo.render_volume(
            cuda_tree,
            origins,
            directions,
            sigma,
            color,
            background_color=(1.0, 1.0, 1.0),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        render_ms = (time.perf_counter() - render_start) * 1000.0

        loss = torch.nn.functional.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        mse = float(loss.detach().item())
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step:05d} loss={mse:.6f} psnr={psnr_from_mse(mse):.2f}dB "
                f"leaves={cuda_tree.num_leaves} nodes={cuda_tree.num_nodes} render_ms={render_ms:.3f}"
            )

        if args.save_every > 0 and (step == 1 or step % args.save_every == 0):
            image = render_debug_image(
                torch,
                cuda_tree,
                dataset,
                raw_sigma,
                raw_color,
                image_index=args.debug_image,
                device=device,
            )
            write_png_rgb(args.output / f"step_{step:06d}.png", image_to_uint8(image))

        if args.refine_every > 0 and step >= args.refine_warmup and step % args.refine_every == 0:
            cuda_tree, raw_sigma, raw_color, optimizer, stats = refine_training_state(
                torch,
                cuda_tree,
                raw_sigma,
                raw_color,
                learning_rate=args.learning_rate,
                split_threshold=args.split_threshold,
                prune_threshold=args.prune_threshold,
                merge_threshold=args.merge_threshold,
                min_depth=args.min_depth,
                max_leaf_growth=args.max_leaf_growth,
            )
            print(
                "refine "
                f"old={stats['old_leaves']} new={stats['new_leaves']} "
                f"split={stats['split_leaves']} prune={stats['pruned_leaves']} merge={stats['merged_groups']}"
            )

    if args.save_svo is not None:
        with torch.no_grad():
            sigma, color = render_payloads(torch, raw_sigma, raw_color)
            args.save_svo.parent.mkdir(parents=True, exist_ok=True)
            svo.save(args.save_svo, cuda_tree, {"sigma": sigma, "color": color})
        print(f"saved SVO scene to {args.save_svo}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/nerf"), help="Directory containing transforms_*.json.")
    parser.add_argument("--split", default="train", help="Dataset split name.")
    parser.add_argument("--output", type=Path, default=Path("outputs/nerf_octree"), help="Output checkpoint PNG directory.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--rays-per-step", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=5.0e-2)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--leaf-depth", type=int, default=3)
    parser.add_argument(
        "--root-center",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="World-space center of the octree root bounds.",
    )
    parser.add_argument(
        "--root-size",
        type=float,
        default=3.0,
        help="World-space side length of the cubic octree root bounds.",
    )
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--save-svo", type=Path, default=None, help="Optional final .svo scene output path.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--refine-every", type=int, default=500)
    parser.add_argument("--refine-warmup", type=int, default=500)
    parser.add_argument("--split-threshold", type=float, default=0.5)
    parser.add_argument("--prune-threshold", type=float, default=0.01)
    parser.add_argument("--merge-threshold", type=float, default=0.02)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-leaf-growth", type=float, default=2.0)
    parser.add_argument("--debug-image", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick experiments/tests.")
    parser.add_argument("--load-workers", type=int, default=None, help="Parallel dataset loading workers.")
    parser.add_argument(
        "--load-backend",
        choices=("threads", "processes"),
        default="threads",
        help="Use threads for low overhead or processes to bypass the GIL in the pure-Python PNG decoder.",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.rays_per_step <= 0:
        raise ValueError("rays-per-step must be positive")
    if args.debug_image < 0:
        raise ValueError("debug-image must be non-negative")
    if args.load_workers is not None and args.load_workers <= 0:
        raise ValueError("load-workers must be positive")
    train(args)


if __name__ == "__main__":
    main()
