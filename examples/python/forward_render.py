"""Render a colored sparse sphere with smooth sinusoidal payloads.

The octree stores occupied voxels inside a sphere. Density is strongest near the
sphere center, and RGB payloads vary smoothly with sine functions over position.

Example:
    ./.venv/bin/python examples/python/forward_render.py --output docs/assets/forward_render.png
"""

from __future__ import annotations

import argparse
import math
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import svo


@dataclass(frozen=True)
class SphereVolumeConfig:
    grid_size: int = 64
    radius: float = 0.34
    width: int = 360
    height: int = 260

    @property
    def max_depth(self) -> int:
        return int(math.log2(self.grid_size))


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", checksum)


def write_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB uint8 image without extra visualization dependencies."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a uint8 RGB array")

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = image.shape
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw_rows, level=9))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def build_sphere_payloads(config: SphereVolumeConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = config.grid_size
    inv_grid = 1.0 / float(grid)
    center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    coords: list[tuple[int, int, int]] = []
    sigma_values: list[float] = []
    color_values: list[tuple[float, float, float]] = []

    for z_index in range(grid):
        z = (z_index + 0.5) * inv_grid
        for y_index in range(grid):
            y = (y_index + 0.5) * inv_grid
            for x_index in range(grid):
                x = (x_index + 0.5) * inv_grid
                position = np.array([x, y, z], dtype=np.float32)
                radius_fraction = float(np.linalg.norm(position - center) / config.radius)
                if radius_fraction > 1.0:
                    continue

                coords.append((x_index, y_index, z_index))
                core = max(0.0, 1.0 - radius_fraction)
                sigma_values.append(4.0 + 24.0 * core * core)
                color_values.append(
                    (
                        0.50 + 0.50 * math.sin(2.0 * math.pi * (1.20 * x + 0.15 * y)),
                        0.50 + 0.50 * math.sin(2.0 * math.pi * (1.10 * y + 0.20 * z) + 2.0),
                        0.50 + 0.50 * math.sin(2.0 * math.pi * (1.05 * z + 0.18 * x) + 4.0),
                    )
                )

    coords_array = np.asarray(coords, dtype=np.int32)
    payload_indices = np.arange(len(coords_array), dtype=np.int32)
    sigma = np.asarray(sigma_values, dtype=np.float32)
    color = np.asarray(color_values, dtype=np.float32)
    return coords_array, payload_indices, sigma, color


def render_sphere(config: SphereVolumeConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, svo.Octree, float, float]:
    build_start = time.perf_counter()
    coords, payload_indices, sigma, color = build_sphere_payloads(config)
    tree = svo.Octree.from_voxels(
        coords,
        max_depth=config.max_depth,
        payload_indices=payload_indices,
    )
    build_seconds = time.perf_counter() - build_start

    camera = svo.Camera.look_at(
        origin=[1.45, 0.95, 1.45],
        target=[0.50, 0.50, 0.50],
        up=[0.0, 1.0, 0.0],
        width=config.width,
        height=config.height,
        vertical_fov_y_degrees=35.0,
    )
    origins, directions = camera.generate_rays()
    render_start = time.perf_counter()
    rgb, depth, opacity = svo.render_volume(
        tree,
        origins,
        directions,
        sigma,
        color,
        background_color=(0.018, 0.022, 0.030),
        early_stop_transmittance=2.0e-3,
    )
    render_seconds = time.perf_counter() - render_start
    return rgb, depth, opacity, tree, build_seconds, render_seconds


def tonemap(rgb: np.ndarray, depth: np.ndarray, opacity: np.ndarray) -> np.ndarray:
    image = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    finite_depth = np.isfinite(depth)
    if finite_depth.any():
        near = float(np.percentile(depth[finite_depth], 5.0))
        far = float(np.percentile(depth[finite_depth], 95.0))
        depth_norm = np.zeros_like(depth, dtype=np.float32)
        depth_norm[finite_depth] = np.clip((depth[finite_depth] - near) / max(far - near, 1.0e-6), 0.0, 1.0)
        image *= (0.78 + 0.22 * (1.0 - depth_norm))[..., None]

    image = image * (0.72 + 0.28 * np.clip(opacity, 0.0, 1.0))[..., None]
    image = np.clip(image, 0.0, 1.0) ** (1.0 / 2.2)
    return np.round(image * 255.0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=64, help="Dense construction grid. Must be a power of two.")
    parser.add_argument("--radius", type=float, default=0.34, help="Sphere radius in normalized root coordinates.")
    parser.add_argument("--width", type=int, default=360, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=260, help="Render height in pixels.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/forward_render.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid_size <= 0 or args.grid_size & (args.grid_size - 1):
        raise ValueError("grid-size must be a positive power of two")
    if args.radius <= 0.0:
        raise ValueError("radius must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")

    config = SphereVolumeConfig(
        grid_size=args.grid_size,
        radius=args.radius,
        width=args.width,
        height=args.height,
    )
    rgb, depth, opacity, tree, build_seconds, render_seconds = render_sphere(config)
    write_start = time.perf_counter()
    image = tonemap(rgb, depth, opacity)
    write_png(args.output, image)
    write_seconds = time.perf_counter() - write_start

    print(f"Built {tree.num_leaves} occupied leaves at depth {tree.max_depth} in {build_seconds * 1000.0:.2f} ms")
    print(f"Forward render: {render_seconds * 1000.0:.2f} ms for {args.width}x{args.height} rays")
    print(f"Tonemap + PNG write: {write_seconds * 1000.0:.2f} ms")
    print(f"Saved render to {args.output}")


if __name__ == "__main__":
    main()
