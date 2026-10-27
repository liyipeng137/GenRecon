"""CPU-only pose selection with explicit, approximate chunk-frustum coverage.

Coverage uses the same nine-point projected bounding box as GenRecon's existing
visibility test. It does not test occlusion or claim observed surface coverage.
"""
from __future__ import annotations

import numpy as np


def _poses(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite [N,4,4] matrices")
    if not np.allclose(value[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"{name} must use homogeneous coordinates")
    return value


def chunk_projection_areas(c2w_blender, intrinsics, world_to_chunks):
    """Nine-point projected areas, one row per actual crop camera.

    Cameras use Blender axes and normalized pinhole K. Chunk transforms map
    world coordinates to the normalized cube and may include scale.
    """
    poses = _poses(c2w_blender, "c2w_blender")
    chunks = _poses(world_to_chunks, "world_to_chunks")
    intr = np.asarray(intrinsics, dtype=np.float64)
    if intr.shape != (len(poses), 3, 3) or not np.isfinite(intr).all():
        raise ValueError("intrinsics must be finite [N,3,3] matrices")
    corners = np.array([[x, y, z, 1.] for x in (-.5, .5)
                        for y in (-.5, .5) for z in (-.5, .5)] + [[0., 0., 0., 1.]])
    flip = np.diag([1., -1., -1., 1.])
    result = np.zeros((len(poses), len(chunks)), dtype=np.float64)
    for column, chunk in enumerate(chunks):
        extrinsics = flip @ np.linalg.inv(chunk @ poses)
        camera_points = np.einsum("nij,pj->npi", extrinsics[:, :3], corners)
        projected = np.einsum("nij,npj->npi", intr, camera_points)
        valid = ((camera_points[..., 2] > 0) & np.isfinite(projected).all(-1)
                 & (np.abs(projected[..., 2]) > 1e-8))
        safe_z = np.where(valid, projected[..., 2], 1.)
        uv = np.clip(projected[..., :2] / safe_z[..., None], 0., 1.)
        lower = np.where(valid[..., None], uv, np.inf).min(axis=1)
        upper = np.where(valid[..., None], uv, -np.inf).max(axis=1)
        extent = np.maximum(upper - lower, 0.)
        result[:, column] = np.where(valid.any(axis=1), extent[:, 0] * extent[:, 1], 0.)
    return result


def coverage_report(coverage_areas, indices, *, min_area=.4, min_views_per_chunk=3):
    visible = np.asarray(coverage_areas) >= min_area
    available = visible.sum(axis=0)
    chosen = visible[np.asarray(indices, dtype=int)].sum(axis=0)
    return {
        "visibility_metric": "nine_point_crop_frustum_bbox_area; no occlusion test",
        "available_views_per_chunk": available.tolist(),
        "selected_views_per_chunk": chosen.tolist(),
        "uncovered_chunk_indices": np.flatnonzero(chosen == 0).tolist(),
        "unobservable_chunk_indices": np.flatnonzero(available == 0).tolist(),
        "undersupported_chunk_indices": np.flatnonzero(chosen < min_views_per_chunk).tolist(),
        "target_views_per_chunk": min_views_per_chunk,
    }


def select_pose_coverage(c2w, intrinsics, coverage_areas, budget, *,
                         position_scale, position_threshold=.05,
                         rotation_threshold_degrees=10., intrinsics_threshold=.01,
                         min_views_per_chunk=3, min_area=.4):
    """Greedy coverage followed by pose diversity, with a hard original-image cap.

    A near duplicate must be close in position, full SO(3) rotation AND K. Such
    a frame can still be rescued when it is needed for an uncovered chunk.
    Returning fewer than the budget is intentional when only duplicates remain.
    Results are deterministic; the returned indices retain input ordering.
    """
    poses = _poses(c2w, "c2w")
    n = len(poses)
    if n == 0:
        raise ValueError("At least one camera is required")
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if isinstance(min_views_per_chunk, (bool, np.bool_)) or not isinstance(min_views_per_chunk, (int, np.integer)) or min_views_per_chunk < 1:
        raise ValueError("min_views_per_chunk must be a positive integer")
    for name, value in (("position_scale", position_scale), ("position_threshold", position_threshold),
                        ("rotation_threshold_degrees", rotation_threshold_degrees),
                        ("intrinsics_threshold", intrinsics_threshold), ("min_area", min_area)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if rotation_threshold_degrees > 180 or min_area > 1:
        raise ValueError("rotation threshold must be <=180 and min_area <=1")
    intr = np.asarray(intrinsics, dtype=np.float64)
    if intr.shape != (n, 3, 3) or not np.isfinite(intr).all() or np.any(intr[:, (0, 1), (0, 1)] <= 0):
        raise ValueError("intrinsics must be finite [N,3,3] pinhole matrices with positive focal lengths")
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotations), 1., atol=1e-4):
        raise ValueError("camera rotations must be proper orthonormal matrices")
    areas = np.asarray(coverage_areas, dtype=np.float64)
    if areas.ndim != 2 or areas.shape[0] != n or not np.isfinite(areas).all() or np.any(areas < 0) or np.any(areas > 1 + 1e-8):
        raise ValueError("coverage_areas must be finite [N,K] values in [0,1]")
    visible = areas >= min_area
    available = visible.sum(axis=0)
    candidates = visible.any(axis=1) if areas.shape[1] else np.ones(n, dtype=bool)
    fallback = not candidates.any()
    if fallback:
        candidates[:] = True
    rarity = 1. / np.maximum(available, 1)
    counts = np.zeros(areas.shape[1], dtype=int)
    chosen_mask = np.zeros(n, dtype=bool)
    duplicates = np.zeros(n, dtype=bool)
    nearest = np.full(n, np.inf)
    positions = poses[:, :3, 3]
    # Focal changes are relative; principal point/skew changes are normalized
    # image fractions. Different fields of view are not collapsed by pose alone.
    k_features = np.column_stack([np.log(intr[:, 0, 0]), np.log(intr[:, 1, 1]),
                                  intr[:, 0, 1], intr[:, 0, 2], intr[:, 1, 2]])
    order, reasons, rescues = [], [], []
    for _ in range(min(int(budget), n)):
        new_gain = (visible * (counts == 0)).sum(axis=1)
        support_gain = (visible * (counts < min_views_per_chunk) * rarity).sum(axis=1)
        eligible = candidates & ~chosen_mask & (~duplicates | (new_gain > 0))
        ids = np.flatnonzero(eligible)
        if not len(ids):
            break
        novelty = nearest if order else np.zeros(n)
        # Round only tie-break scores to make rigid/scale invariant decisions
        # insensitive to harmless floating point differences near exact ties.
        ranking = np.lexsort((ids, -np.round(areas[ids].sum(axis=1), 10),
                              -np.round(novelty[ids], 10),
                              -np.round(support_gain[ids], 10), -new_gain[ids]))
        pick = int(ids[ranking[0]])
        rescue = bool(duplicates[pick])
        reason = ("coverage_rescue" if rescue else "new_chunk" if new_gain[pick] else
                  "chunk_support" if support_gain[pick] > 0 else "pose_diversity")
        order.append(pick)
        reasons.append(reason)
        if rescue:
            rescues.append(pick)
        chosen_mask[pick] = True
        counts += visible[pick]
        distance = np.linalg.norm(positions - positions[pick], axis=1) / position_scale
        cosine = np.clip((np.einsum("nij,ij->n", rotations, rotations[pick]) - 1.) / 2., -1., 1.)
        angle = np.arccos(cosine)
        k_close = np.max(np.abs(k_features - k_features[pick]), axis=1) <= intrinsics_threshold
        duplicates |= ((distance <= position_threshold + 1e-10)
                       & (angle <= np.deg2rad(rotation_threshold_degrees) + 1e-10) & k_close)
        nearest = np.minimum(nearest, np.hypot(distance, angle / np.pi))
    selected = sorted(order)
    report = coverage_report(areas, selected, min_area=min_area,
                             min_views_per_chunk=min_views_per_chunk)
    report.update({
        "strategy": "pose_coverage", "input_camera_count": n,
        "candidate_camera_count": int(candidates.sum()), "budget_original_images": int(budget),
        "selected_original_images": len(selected), "selected_indices": selected,
        "selection_order": order, "selection_reasons": reasons,
        "coverage_rescue_indices": rescues,
        "pose_redundant_unselected_count": int((candidates & ~chosen_mask & duplicates).sum()),
        "all_invisible_fallback": fallback,
        "position_scale_scene_units": float(position_scale),
        "position_threshold_chunk_fraction": float(position_threshold),
        "rotation_threshold_degrees": float(rotation_threshold_degrees),
        "intrinsics_threshold": float(intrinsics_threshold),
        "min_frustum_area": float(min_area),
        "budget_exhausted": len(selected) == int(budget),
        "all_observable_chunks_covered": bool(np.all((available == 0) | (counts > 0))),
    })
    return selected, report
