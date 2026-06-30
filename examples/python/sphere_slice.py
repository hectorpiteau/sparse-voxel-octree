"""Render an axis-aligned occupancy slice of the milestone 4 sphere octree.

Example:
    UV_CACHE_DIR=/tmp/uv-cache uv sync --extra viz
    UV_CACHE_DIR=/tmp/uv-cache uv run python examples/python/sphere_slice.py \
        --axis z --position 0.5 --output /tmp/sphere_slice.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

import svo


@dataclass(frozen=True)
class SphereConfig:
    grid_size: int = 64
    radius: float = 18.0
    center: tuple[float, float, float] = (32.0, 32.0, 32.0)

    @property
    def max_depth(self) -> int:
        return int(np.log2(self.grid_size))


def sphere_voxel_coordinates(config: SphereConfig) -> np.ndarray:
    """Return occupied voxel coordinates for a sphere embedded in a dense grid."""
    coords: list[tuple[int, int, int]] = []
    center = np.asarray(config.center, dtype=np.float32)

    for z in range(config.grid_size):
        for y in range(config.grid_size):
            for x in range(config.grid_size):
                voxel_center = np.array([x + 0.5, y + 0.5, z + 0.5], dtype=np.float32)
                if np.sum((voxel_center - center) ** 2) <= config.radius**2:
                    coords.append((x, y, z))

    return np.asarray(coords, dtype=np.int32)


def build_sphere_octree(config: SphereConfig) -> svo.Octree:
    coords = sphere_voxel_coordinates(config)
    return svo.Octree.from_voxels(coords, max_depth=config.max_depth, device="cpu")


def slice_index_from_position(grid_size: int, position: float) -> int:
    if not (0.0 <= position < 1.0):
        raise ValueError("position must be in the range [0.0, 1.0)")
    return min(int(position * grid_size), grid_size - 1)


def slice_points(axis: str, slice_index: int, grid_size: int) -> tuple[np.ndarray, tuple[str, str]]:
    """Create query points for an axis-aligned slice through [0, 1)^3."""
    inv_grid = 1.0 / grid_size
    axis_labels = {
        "x": ("y", "z"),
        "y": ("x", "z"),
        "z": ("x", "y"),
    }
    labels = axis_labels[axis]

    points = np.zeros((grid_size * grid_size, 3), dtype=np.float32)
    fixed = (slice_index + 0.5) * inv_grid

    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            coords = {
                "x": (col + 0.5) * inv_grid,
                "y": (row + 0.5) * inv_grid,
                "z": fixed,
            }
            if axis == "x":
                coords["x"] = fixed
            elif axis == "y":
                coords["y"] = fixed
                coords["z"] = (row + 0.5) * inv_grid
                coords["x"] = (col + 0.5) * inv_grid
            else:
                coords["x"] = (col + 0.5) * inv_grid
                coords["y"] = (row + 0.5) * inv_grid

            points[index] = [coords["x"], coords["y"], coords["z"]]
            index += 1

    return points, labels


def occupancy_slice(tree: svo.Octree, axis: str, position: float, grid_size: int) -> tuple[np.ndarray, int, tuple[str, str]]:
    slice_index = slice_index_from_position(grid_size, position)
    points, labels = slice_points(axis, slice_index, grid_size)
    hits = tree.query(points)
    occupancy = (hits >= 0).astype(np.uint8).reshape(grid_size, grid_size)
    return occupancy, slice_index, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z", help="Slice direction.")
    parser.add_argument(
        "--position",
        type=float,
        default=0.5,
        help="Normalized slice position in [0, 1).",
    )
    parser.add_argument("--grid-size", type=int, default=64, help="Dense grid resolution. Must be a power of two.")
    parser.add_argument("--radius", type=float, default=18.0, help="Sphere radius in voxel units.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional image path. If omitted, the figure is shown interactively.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "matplotlib is required for this example; install it with `uv sync --extra viz`"
        ) from error

    args = parse_args()
    if args.grid_size <= 0 or args.grid_size & (args.grid_size - 1):
        raise ValueError("grid-size must be a positive power of two")

    center = tuple(float(args.grid_size) / 2.0 for _ in range(3))
    config = SphereConfig(grid_size=args.grid_size, radius=args.radius, center=center)

    print("Building sphere octree...")
    tree = build_sphere_octree(config)
    print(tree)

    occupancy, slice_index, labels = occupancy_slice(tree, args.axis, args.position, config.grid_size)

    figure, axis = plt.subplots(figsize=(6, 6))
    image = axis.imshow(
        occupancy,
        origin="lower",
        interpolation="nearest",
        cmap="gray_r",
        extent=(0, config.grid_size, 0, config.grid_size),
    )
    axis.set_title(f"Sphere occupancy slice: {args.axis}={slice_index} / {config.grid_size - 1}")
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    figure.colorbar(image, ax=axis, label="occupancy (1=inside)")
    figure.tight_layout()

    if args.output:
        figure.savefig(args.output, dpi=160)
        print(f"Saved slice to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
