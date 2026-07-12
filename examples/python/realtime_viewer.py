"""Interactive SVO forward-render viewer.

Controls:
    Left mouse drag: orbit around the model at target (0, 0, 0)
    Mouse wheel: zoom in/out
    R: reset camera
    Escape or Q: quit

Install viewer dependency:
    UV_CACHE_DIR=/tmp/uv-cache uv sync --extra viewer

Run:
    ./.venv/bin/python examples/python/realtime_viewer.py --device auto
    ./.venv/bin/python examples/python/realtime_viewer.py --device cuda --branching wide4
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

import svo

RenderDevice = Literal["auto", "cpu", "cuda"]
BranchingMode = Literal["octree8", "wide4"]


@dataclass(frozen=True)
class ScenePayload:
    tree: svo.Octree
    sigma_np: np.ndarray
    color_np: np.ndarray
    num_leaves: int
    num_nodes: int
    branching: str


@dataclass
class CameraOrbit:
    yaw: float = 0.75
    pitch: float = 0.28
    distance: float = 2.7

    def reset(self) -> None:
        self.yaw = 0.75
        self.pitch = 0.28
        self.distance = 2.7

    def origin(self) -> list[float]:
        cos_pitch = math.cos(self.pitch)
        return [
            self.distance * cos_pitch * math.cos(self.yaw),
            self.distance * math.sin(self.pitch),
            self.distance * cos_pitch * math.sin(self.yaw),
        ]


def torch_cuda_available() -> bool:
    if not svo.cuda_enabled():
        return False
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def cuda_unavailable_reasons() -> list[str]:
    reasons: list[str] = []
    if not svo.cuda_enabled():
        reasons.append("the installed `svo` extension was built without SVO_ENABLE_CUDA=ON")
    try:
        import torch
    except ModuleNotFoundError:
        reasons.append("PyTorch is not installed in this virtual environment")
    else:
        if not torch.cuda.is_available():
            reasons.append("PyTorch is installed but `torch.cuda.is_available()` is false")
    return reasons


def resolve_device(requested: RenderDevice) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        reasons = cuda_unavailable_reasons()
        if reasons:
            details = "; ".join(reasons)
            raise RuntimeError(f"--device cuda requested, but CUDA Torch rendering is not available: {details}")
        return "cuda"
    return "cuda" if torch_cuda_available() else "cpu"


def build_sphere_scene(grid_size: int, radius: float, branching: BranchingMode) -> ScenePayload:
    inv_grid = 1.0 / float(grid_size)
    coords: list[tuple[int, int, int]] = []
    sigma_values: list[float] = []
    color_values: list[tuple[float, float, float]] = []

    for z_index in range(grid_size):
        z = -1.0 + (z_index + 0.5) * 2.0 * inv_grid
        for y_index in range(grid_size):
            y = -1.0 + (y_index + 0.5) * 2.0 * inv_grid
            for x_index in range(grid_size):
                x = -1.0 + (x_index + 0.5) * 2.0 * inv_grid
                distance = math.sqrt(x * x + y * y + z * z)
                if distance > radius:
                    continue

                coords.append((x_index, y_index, z_index))
                core = max(0.0, 1.0 - distance / radius)
                sigma_values.append(5.0 + 26.0 * core * core)
                color_values.append(
                    (
                        0.50 + 0.50 * math.sin(2.3 * x + 1.1 * y),
                        0.50 + 0.50 * math.sin(2.1 * y + 1.4 * z + 2.0),
                        0.50 + 0.50 * math.sin(2.2 * z + 1.2 * x + 4.0),
                    )
                )

    coords_np = np.asarray(coords, dtype=np.int32)
    payload_indices = np.arange(len(coords_np), dtype=np.int32)
    root_bounds = np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    tree = svo.Octree.from_voxels(
        coords_np,
        max_depth=int(math.log2(grid_size)),
        root_bounds=root_bounds,
        payload_indices=payload_indices,
        branching=branching,
    )
    return ScenePayload(
        tree=tree,
        sigma_np=np.asarray(sigma_values, dtype=np.float32),
        color_np=np.asarray(color_values, dtype=np.float32),
        num_leaves=len(coords_np),
        num_nodes=tree.num_nodes,
        branching=tree.branching,
    )


def tonemap(rgb: np.ndarray, opacity: np.ndarray) -> np.ndarray:
    image = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    image *= (0.70 + 0.30 * np.clip(opacity, 0.0, 1.0))[..., None]
    image = np.clip(image, 0.0, 1.0) ** (1.0 / 2.2)
    return np.round(image * 255.0).astype(np.uint8)


def generate_rays(width: int, height: int, orbit: CameraOrbit, fov_y: float) -> tuple[np.ndarray, np.ndarray]:
    camera = svo.Camera.look_at(
        origin=orbit.origin(),
        target=[0.0, 0.0, 0.0],
        up=[0.0, 1.0, 0.0],
        width=width,
        height=height,
        vertical_fov_y_degrees=fov_y,
    )
    return camera.generate_rays()


def render_cpu(
    scene: ScenePayload,
    origins: np.ndarray,
    directions: np.ndarray,
    early_stop_transmittance: float,
) -> tuple[np.ndarray, np.ndarray]:
    rgb, _depth, opacity = svo.render_volume(
        scene.tree,
        origins,
        directions,
        scene.sigma_np,
        scene.color_np,
        background_color=(0.015, 0.018, 0.026),
        early_stop_transmittance=early_stop_transmittance,
    )
    return rgb, opacity


def prepare_cuda_scene(scene: ScenePayload) -> tuple[Any, Any, Any]:
    import torch

    cuda_tree = scene.tree.to("cuda")
    sigma = torch.as_tensor(scene.sigma_np, device="cuda")
    color = torch.as_tensor(scene.color_np, device="cuda")
    return cuda_tree, sigma, color


def render_cuda(
    cuda_tree: Any,
    sigma: Any,
    color: Any,
    origins: np.ndarray,
    directions: np.ndarray,
    early_stop_transmittance: float,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    with torch.no_grad():
        origins_t = torch.as_tensor(origins, device="cuda")
        directions_t = torch.as_tensor(directions, device="cuda")
        rgb_t, _depth_t, opacity_t = svo.render_volume(
            cuda_tree,
            origins_t,
            directions_t,
            sigma,
            color,
            background_color=(0.015, 0.018, 0.026),
            early_stop_transmittance=early_stop_transmittance,
        )
        return rgb_t.detach().cpu().numpy(), opacity_t.detach().cpu().numpy()


def draw_overlay(
    pygame: Any,
    surface: Any,
    font: Any,
    fps: float,
    frame_ms: float,
    device: str,
    scene: ScenePayload,
) -> None:
    lines = [
        f"{fps:6.1f} FPS  {frame_ms:5.2f} ms",
        f"backend: {device}   branching: {scene.branching}   nodes: {scene.num_nodes}   leaves: {scene.num_leaves}",
        "drag: orbit   wheel: zoom   R: reset   Q/Esc: quit",
    ]
    pad = 8
    line_height = font.get_linesize()
    box = pygame.Rect(8, 8, 520, pad * 2 + line_height * len(lines))
    overlay = pygame.Surface((box.width, box.height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    surface.blit(overlay, box.topleft)
    for index, line in enumerate(lines):
        text = font.render(line, True, (235, 240, 245))
        surface.blit(text, (box.x + pad, box.y + pad + index * line_height))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320, help="Viewer render width.")
    parser.add_argument("--height", type=int, default=240, help="Viewer render height.")
    parser.add_argument("--grid-size", type=int, default=64, help="Sphere grid size. Must be a power of two.")
    parser.add_argument("--radius", type=float, default=0.68, help="Sphere radius in root coordinates.")
    parser.add_argument("--fov-y", type=float, default=38.0, help="Vertical camera FOV in degrees.")
    parser.add_argument(
        "--early-stop-transmittance",
        type=float,
        default=2.0e-3,
        help="Stop rays once remaining transmittance falls below this value.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Render backend. auto uses CUDA Torch when available, otherwise CPU.",
    )
    parser.add_argument(
        "--branching",
        choices=("octree8", "wide4"),
        default="octree8",
        help="Tree topology to build. wide4 requires an even log2(grid-size).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    if args.grid_size <= 0 or args.grid_size & (args.grid_size - 1):
        raise ValueError("grid-size must be a positive power of two")
    if args.radius <= 0.0:
        raise ValueError("radius must be positive")
    max_depth = int(math.log2(args.grid_size))
    if args.branching == "wide4" and (max_depth % 2) != 0:
        raise ValueError("wide4 requires an even log2(grid-size); use e.g. --grid-size 16, 64, or 256")

    device = resolve_device(args.device)

    try:
        import pygame
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "pygame is required for the real-time viewer; install it with `uv sync --extra viewer`"
        ) from error

    scene = build_sphere_scene(args.grid_size, args.radius, args.branching)
    cuda_resources = prepare_cuda_scene(scene) if device == "cuda" else None

    pygame.init()
    pygame.display.set_caption("SVO real-time forward renderer")
    screen = pygame.display.set_mode((args.width, args.height))
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    orbit = CameraOrbit()
    dragging = False
    last_mouse = (0, 0)
    fps_ema = 0.0
    frame_ms_ema = 0.0
    running = True

    while running:
        frame_start = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    orbit.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:
                    orbit.distance = max(1.1, orbit.distance * 0.90)
                elif event.button == 5:
                    orbit.distance = min(6.0, orbit.distance * 1.10)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]
                last_mouse = event.pos
                orbit.yaw += dx * 0.007
                orbit.pitch = float(np.clip(orbit.pitch + dy * 0.007, -1.35, 1.35))
            elif event.type == pygame.MOUSEWHEEL:
                orbit.distance = float(np.clip(orbit.distance * math.exp(-0.10 * event.y), 1.1, 6.0))

        origins, directions = generate_rays(args.width, args.height, orbit, args.fov_y)
        if device == "cuda":
            assert cuda_resources is not None
            rgb, opacity = render_cuda(*cuda_resources, origins, directions, args.early_stop_transmittance)
        else:
            rgb, opacity = render_cpu(scene, origins, directions, args.early_stop_transmittance)
        frame = tonemap(rgb, opacity)

        pygame.surfarray.blit_array(screen, np.swapaxes(frame, 0, 1))
        frame_seconds = time.perf_counter() - frame_start
        fps = 1.0 / max(frame_seconds, 1.0e-9)
        fps_ema = fps if fps_ema == 0.0 else 0.90 * fps_ema + 0.10 * fps
        frame_ms = frame_seconds * 1000.0
        frame_ms_ema = frame_ms if frame_ms_ema == 0.0 else 0.90 * frame_ms_ema + 0.10 * frame_ms
        draw_overlay(pygame, screen, font, fps_ema, frame_ms_ema, device, scene)
        pygame.display.flip()
        clock.tick(0)

    pygame.quit()


if __name__ == "__main__":
    main()
