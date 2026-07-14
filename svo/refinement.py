"""Adaptive octree rebuild/refinement helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RefinementResult:
    tree: Any
    cuda_tree: Any | None
    sigma: Any
    color: Any
    old_to_new_leaf: np.ndarray
    new_from_old_leaf: np.ndarray
    stats: dict[str, int]


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return torch.is_tensor(value)


def _validate_identity_payloads(tree: Any) -> np.ndarray:
    payload_indices = np.asarray(tree.leaf_payload_indices, dtype=np.int64)
    expected = np.arange(payload_indices.shape[0], dtype=np.int64)
    if not np.array_equal(payload_indices, expected):
        raise ValueError("adaptive refinement requires one payload row per leaf and identity leaf_payload_indices")
    return payload_indices


def _leaf_scores_cpu(tree: Any, sigma: Any, leaf_scores: Any | None) -> np.ndarray:
    num_leaves = int(tree.num_leaves)
    if leaf_scores is not None:
        if _is_torch_tensor(leaf_scores):
            values = leaf_scores.detach().to("cpu").numpy()
        else:
            values = np.asarray(leaf_scores)
    elif _is_torch_tensor(sigma):
        values = sigma.detach().clamp_min(0).to("cpu").numpy()
    else:
        values = np.maximum(np.asarray(sigma), 0.0)

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.shape[0] != num_leaves:
        raise ValueError("leaf_scores must have shape (num_leaves,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("leaf_scores must contain only finite values")
    return values


def _child_coord(coord: np.ndarray, child_size: int, child_index: int) -> np.ndarray:
    return coord + np.array(
        [
            child_size if (child_index & 1) else 0,
            child_size if (child_index & 2) else 0,
            child_size if (child_index & 4) else 0,
        ],
        dtype=np.int32,
    )


def _build_refined_specs(
    tree: Any,
    scores: np.ndarray,
    *,
    split_threshold: float,
    prune_threshold: float,
    merge_threshold: float | None,
    min_depth: int,
    max_depth: int,
    max_leaf_growth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, list[int]], dict[str, int]]:
    coord_min, depths, _payload_indices = tree.leaf_specs
    coord_min = np.asarray(coord_min, dtype=np.int32)
    depths = np.asarray(depths, dtype=np.int32)
    old_count = int(depths.shape[0])

    if old_count != int(tree.num_leaves):
        raise ValueError("tree.leaf_specs must contain one row per leaf")
    if not (0 <= min_depth <= max_depth <= int(tree.max_depth)):
        raise ValueError("depth limits must satisfy 0 <= min_depth <= max_depth <= tree.max_depth")
    if not np.isfinite(split_threshold) or not np.isfinite(prune_threshold):
        raise ValueError("split_threshold and prune_threshold must be finite")
    if split_threshold <= prune_threshold:
        raise ValueError("split_threshold must be greater than prune_threshold")
    if merge_threshold is not None and not np.isfinite(merge_threshold):
        raise ValueError("merge_threshold must be finite when provided")
    if max_leaf_growth <= 0.0 or not np.isfinite(max_leaf_growth):
        raise ValueError("max_leaf_growth must be finite and positive")

    split = (scores >= float(split_threshold)) & (depths < max_depth)
    prune = (scores <= float(prune_threshold)) & (depths >= min_depth) & ~split

    merge_groups: dict[tuple[int, int, int, int], list[int]] = {}
    if merge_threshold is not None:
        for old_leaf, (coord, depth) in enumerate(zip(coord_min, depths, strict=True)):
            if split[old_leaf] or prune[old_leaf] or depth <= min_depth or depth <= 0:
                continue
            cell_size = 1 << (int(tree.max_depth) - int(depth))
            parent_size = cell_size * 2
            parent_coord = (coord // parent_size) * parent_size
            key = (int(depth) - 1, int(parent_coord[0]), int(parent_coord[1]), int(parent_coord[2]))
            merge_groups.setdefault(key, []).append(old_leaf)

    merge_for_old: dict[int, list[int]] = {}
    for group in merge_groups.values():
        if len(group) != 8:
            continue
        if not np.all(scores[np.asarray(group, dtype=np.int64)] <= float(merge_threshold)):
            continue
        offsets = set()
        depth = int(depths[group[0]])
        cell_size = 1 << (int(tree.max_depth) - depth)
        parent_size = cell_size * 2
        parent_coord = (coord_min[group[0]] // parent_size) * parent_size
        for old_leaf in group:
            offset = (coord_min[old_leaf] - parent_coord) // cell_size
            offsets.add(int(offset[0]) | (int(offset[1]) << 1) | (int(offset[2]) << 2))
        if offsets == set(range(8)):
            ordered_group = sorted(group)
            for old_leaf in ordered_group:
                merge_for_old[old_leaf] = ordered_group

    new_coords: list[np.ndarray] = []
    new_depths: list[int] = []
    new_from_old: list[int] = []
    old_to_new = np.full(old_count, -1, dtype=np.int32)
    merge_rows: dict[int, list[int]] = {}
    split_count = 0
    prune_count = 0
    merge_count = 0

    for old_leaf in range(old_count):
        if split[old_leaf]:
            parent_depth = int(depths[old_leaf])
            child_depth = parent_depth + 1
            child_size = 1 << (int(tree.max_depth) - child_depth)
            first_new = len(new_coords)
            for child_index in range(8):
                new_coords.append(_child_coord(coord_min[old_leaf], child_size, child_index))
                new_depths.append(child_depth)
                new_from_old.append(old_leaf)
            old_to_new[old_leaf] = first_new
            split_count += 1
            continue

        group = merge_for_old.get(old_leaf)
        if group is not None:
            representative = group[0]
            if old_leaf != representative:
                continue
            depth = int(depths[old_leaf])
            cell_size = 1 << (int(tree.max_depth) - depth)
            parent_size = cell_size * 2
            parent_coord = (coord_min[old_leaf] // parent_size) * parent_size
            new_index = len(new_coords)
            new_coords.append(parent_coord.astype(np.int32))
            new_depths.append(depth - 1)
            new_from_old.append(representative)
            merge_rows[new_index] = group
            for merged_leaf in group:
                old_to_new[merged_leaf] = new_index
            merge_count += 1
            continue

        if prune[old_leaf]:
            prune_count += 1
            continue

        new_index = len(new_coords)
        new_coords.append(coord_min[old_leaf].copy())
        new_depths.append(int(depths[old_leaf]))
        new_from_old.append(old_leaf)
        old_to_new[old_leaf] = new_index

    new_count = len(new_coords)
    if new_count > int(np.ceil(old_count * float(max_leaf_growth))):
        raise ValueError("refinement would exceed max_leaf_growth")

    if new_count == 0:
        new_coord_array = np.empty((0, 3), dtype=np.int32)
        new_depth_array = np.empty((0,), dtype=np.int32)
    else:
        new_coord_array = np.stack(new_coords).astype(np.int32, copy=False)
        new_depth_array = np.asarray(new_depths, dtype=np.int32)
    new_payload_indices = np.arange(new_count, dtype=np.int32)
    new_from_old_array = np.asarray(new_from_old, dtype=np.int32)
    stats = {
        "old_leaves": old_count,
        "new_leaves": new_count,
        "split_leaves": split_count,
        "pruned_leaves": prune_count,
        "merged_groups": merge_count,
    }
    return new_coord_array, new_depth_array, new_payload_indices, old_to_new, new_from_old_array, merge_rows, stats


def _remap_numpy_payloads(
    sigma: Any,
    color: Any,
    scores: np.ndarray,
    new_from_old: np.ndarray,
    merge_rows: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    sigma_array = np.asarray(sigma)
    color_array = np.asarray(color)
    if sigma_array.ndim != 1:
        raise ValueError("sigma must have shape (num_leaves,)")
    if color_array.ndim != 2 or color_array.shape[1] != 3:
        raise ValueError("color must have shape (num_leaves, 3)")
    new_sigma = sigma_array[new_from_old].copy()
    new_color = color_array[new_from_old].copy()
    for new_index, old_indices in merge_rows.items():
        old_array = np.asarray(old_indices, dtype=np.int64)
        weights = np.maximum(scores[old_array], 0.0)
        if float(weights.sum()) > 0.0:
            normalized = weights / weights.sum()
            new_sigma[new_index] = np.sum(sigma_array[old_array] * normalized)
            new_color[new_index] = np.sum(color_array[old_array] * normalized[:, None], axis=0)
        else:
            new_sigma[new_index] = np.mean(sigma_array[old_array])
            new_color[new_index] = np.mean(color_array[old_array], axis=0)
    return new_sigma, new_color


def _remap_torch_payloads(
    sigma: Any,
    color: Any,
    scores: np.ndarray,
    new_from_old: np.ndarray,
    merge_rows: dict[int, list[int]],
) -> tuple[Any, Any]:
    import torch

    if sigma.ndim != 1:
        raise ValueError("sigma must have shape (num_leaves,)")
    if color.ndim != 2 or color.shape[1] != 3:
        raise ValueError("color must have shape (num_leaves, 3)")
    if sigma.device != color.device:
        raise ValueError("sigma and color must be on the same device")
    with torch.no_grad():
        source = torch.as_tensor(new_from_old.astype(np.int64), device=sigma.device, dtype=torch.long)
        new_sigma = sigma.detach().index_select(0, source).clone()
        new_color = color.detach().index_select(0, source).clone()
        score_tensor = torch.as_tensor(scores, device=sigma.device, dtype=sigma.dtype)
        for new_index, old_indices in merge_rows.items():
            old_tensor = torch.as_tensor(old_indices, device=sigma.device, dtype=torch.long)
            weights = torch.clamp_min(score_tensor.index_select(0, old_tensor), 0)
            weight_sum = weights.sum()
            if bool((weight_sum > 0).item()):
                normalized = weights / weight_sum
                new_sigma[new_index] = (sigma.detach().index_select(0, old_tensor) * normalized).sum()
                new_color[new_index] = (color.detach().index_select(0, old_tensor) * normalized[:, None]).sum(dim=0)
            else:
                new_sigma[new_index] = sigma.detach().index_select(0, old_tensor).mean()
                new_color[new_index] = color.detach().index_select(0, old_tensor).mean(dim=0)
    return new_sigma, new_color


def refine_octree(
    tree_or_cuda_tree: Any,
    sigma: Any,
    color: Any,
    *,
    split_threshold: float,
    prune_threshold: float,
    merge_threshold: float | None = None,
    min_depth: int = 0,
    max_depth: int | None = None,
    leaf_scores: Any | None = None,
    max_leaf_growth: float = 2.0,
) -> RefinementResult:
    """Rebuild an adaptive octree from split/prune/merge leaf decisions."""

    if getattr(tree_or_cuda_tree, "branching", None) != "octree8":
        raise NotImplementedError("adaptive refinement currently supports only octree8 trees")
    _validate_identity_payloads(tree_or_cuda_tree)

    num_leaves = int(tree_or_cuda_tree.num_leaves)
    if _is_torch_tensor(sigma) or _is_torch_tensor(color):
        import torch

        if not (torch.is_tensor(sigma) and torch.is_tensor(color)):
            raise TypeError("sigma and color must both be Torch tensors or both be NumPy arrays")
        if not hasattr(tree_or_cuda_tree, "_render_volume_torch"):
            raise TypeError("Torch refinement requires a CUDA-owned octree")
        if not sigma.is_cuda or not color.is_cuda:
            raise TypeError("Torch refinement requires CUDA tensors")
        if sigma.shape[0] != num_leaves or color.shape[0] != num_leaves:
            raise ValueError("sigma and color must have one row per leaf")
        tensor_mode = True
    else:
        if hasattr(tree_or_cuda_tree, "_render_volume_torch"):
            raise TypeError("NumPy refinement requires a CPU octree")
        sigma_array = np.asarray(sigma)
        color_array = np.asarray(color)
        if sigma_array.shape[0] != num_leaves or color_array.shape[0] != num_leaves:
            raise ValueError("sigma and color must have one row per leaf")
        tensor_mode = False

    scores = _leaf_scores_cpu(tree_or_cuda_tree, sigma, leaf_scores)
    target_max_depth = int(tree_or_cuda_tree.max_depth) if max_depth is None else int(max_depth)
    (
        new_coords,
        new_depths,
        new_payload_indices,
        old_to_new,
        new_from_old,
        merge_rows,
        stats,
    ) = _build_refined_specs(
        tree_or_cuda_tree,
        scores,
        split_threshold=float(split_threshold),
        prune_threshold=float(prune_threshold),
        merge_threshold=None if merge_threshold is None else float(merge_threshold),
        min_depth=int(min_depth),
        max_depth=target_max_depth,
        max_leaf_growth=float(max_leaf_growth),
    )

    cpu_tree = tree_or_cuda_tree.to("cpu") if hasattr(tree_or_cuda_tree, "_render_volume_torch") else tree_or_cuda_tree
    new_tree = type(cpu_tree).from_leaf_specs(
        new_coords,
        new_depths,
        new_payload_indices,
        max_depth=int(tree_or_cuda_tree.max_depth),
        root_bounds=tree_or_cuda_tree.root_bounds,
        branching="octree8",
    )

    if tensor_mode:
        new_sigma, new_color = _remap_torch_payloads(sigma, color, scores, new_from_old, merge_rows)
        cuda_tree = new_tree.to("cuda")
    else:
        new_sigma, new_color = _remap_numpy_payloads(sigma, color, scores, new_from_old, merge_rows)
        cuda_tree = None

    return RefinementResult(
        tree=new_tree,
        cuda_tree=cuda_tree,
        sigma=new_sigma,
        color=new_color,
        old_to_new_leaf=old_to_new,
        new_from_old_leaf=new_from_old,
        stats=stats,
    )
