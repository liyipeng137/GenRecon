#!/usr/bin/env python3
"""Compare actual CPU frame selectors on fixed chunks; no model or GPU inference.

Frustum area, view-direction diversity, and COLMAP-track retention are selection
proxies. None measures generated mesh quality or resolves surface occlusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.get_images import IphoneImageSelecter


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tracks(colmap, names, transforms):
    """Read existing text export, retaining real track membership and coordinates."""
    name_to_index = {name: i for i, name in enumerate(names)}
    image_to_index = {}
    with (colmap / "images.txt").open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            row = line.split(maxsplit=9)
            image_to_index[int(row[0])] = name_to_index[Path(row[9].strip()).name]
            next(handle)  # Original image observations: the following line.
    points, observation_point, observation_image = [], [], []
    with (colmap / "points3D.txt").open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            row = line.split()
            point_index = len(points)
            points.append([float(value) for value in row[1:4]])
            for image_id in map(int, row[8::2]):
                observation_point.append(point_index)
                observation_image.append(image_to_index[image_id])
    xyz = np.asarray(points)
    homogeneous = np.column_stack([xyz, np.ones(len(xyz))])
    membership = []
    for transform in transforms:
        local = homogeneous @ transform.numpy().T
        membership.append((np.abs(local[:, :3]) <= .5 + 1e-7).all(axis=1))
    return (np.asarray(observation_point, dtype=np.int64),
            np.asarray(observation_image, dtype=np.int64), np.asarray(membership), len(xyz))


def track_report(chosen, track_data, count):
    point_ids, image_ids, membership, point_count = track_data
    mask = np.zeros(count, dtype=bool)
    mask[chosen] = True
    support = np.bincount(point_ids[mask[image_ids]], minlength=point_count)
    return {
        "metric": "existing_real_COLMAP_sparse_points_with_selected_image_tracks; texture_biased_proxy",
        "point_count": point_count,
        "selected_track_observations": int(mask[image_ids].sum()),
        "points_supported_by_at_least_one_selected_image": int((support >= 1).sum()),
        "points_supported_by_at_least_two_selected_images": int((support >= 2).sum()),
        "fraction_at_least_one": float((support >= 1).mean()),
        "fraction_at_least_two": float((support >= 2).mean()),
        "per_chunk": [{
            "chunk": i, "sparse_points": int(member.sum()),
            "fraction_at_least_one": float((support[member] >= 1).mean()) if member.any() else None,
            "fraction_at_least_two": float((support[member] >= 2).mean()) if member.any() else None,
        } for i, member in enumerate(membership)],
    }


def pose_report(chosen, poses, intrinsics, visibility, position_scale, position_threshold, rotation_threshold):
    selected = poses[chosen]
    positions, rotations = selected[:, :3, 3], selected[:, :3, :3]
    distance = np.linalg.norm(positions[:, None] - positions[None], axis=2)
    cosine = (np.einsum("aij,bij->ab", rotations, rotations) - 1.) / 2.
    angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    intr = intrinsics[chosen]
    features = np.column_stack([np.log(intr[:, 0, 0]), np.log(intr[:, 1, 1]),
                                intr[:, 0, 1], intr[:, 0, 2], intr[:, 1, 2]])
    k_close = np.max(np.abs(features[:, None] - features[None]), axis=2) <= .01
    duplicate = ((distance <= position_scale * position_threshold + 1e-9)
                 & (angle <= rotation_threshold + 1e-5) & k_close)
    np.fill_diagonal(duplicate, False)
    forward = -selected[:, :3, 2]  # Blender camera looks along -Z.
    yaw = np.arctan2(forward[:, 1], forward[:, 0])
    yaw_bin = np.floor((yaw + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    pitch = np.degrees(np.arctan2(forward[:, 2], np.linalg.norm(forward[:, :2], axis=1)))
    pitch_bin = np.digitize(pitch, [-20., 20.])
    selected_visibility = visibility[chosen]
    directions = []
    for chunk in range(visibility.shape[1]):
        active = selected_visibility[:, chunk]
        directions.append({
            "chunk": chunk, "visible_original_images": int(active.sum()),
            "yaw_eighth_turn_histogram": np.bincount(yaw_bin[active], minlength=8).tolist(),
            "pitch_down_horizontal_up_histogram": np.bincount(pitch_bin[active], minlength=3).tolist(),
            "occupied_yaw_bins": int(len(np.unique(yaw_bin[active]))),
            "occupied_yaw_pitch_bins": int(len(np.unique(yaw_bin[active] * 3 + pitch_bin[active]))),
        })
    return {
        "position_threshold_scene_units": position_scale * position_threshold,
        "rotation_threshold_degrees": rotation_threshold,
        "intrinsics_threshold": .01,
        "near_duplicate_selected_pairs": int(duplicate.sum() // 2),
        "selected_images_with_near_duplicate": int(duplicate.any(axis=1).sum()),
        "per_chunk_directions": directions,
        "direction_metric_caveat": "optical-axis bins among crop-frustum-accepted cameras; no true surface visibility or roll diversity claim",
    }


def plot_selection(output, poses, chunk_records, cases, budgets, filename="camera_pose_selection.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    positions = poses[:, :3, 3]
    forward = -poses[:, :3, 2]
    fig, axes = plt.subplots(len(budgets), 2, figsize=(13, 6.4 * len(budgets)), squeeze=False)
    span = max(float(np.ptp(positions[:, :2], axis=0).max()), 1e-6)
    for row, budget in enumerate(budgets):
        for column, strategy in enumerate(("uniform", "pose_coverage")):
            ax = axes[row, column]
            case = next(case for case in cases if case["budget"] == budget and case["strategy"] == strategy)
            chosen = np.asarray(case["selected_indices"])
            ax.plot(positions[:, 0], positions[:, 1], lw=.5, alpha=.25, color="#697586")
            ax.scatter(positions[:, 0], positions[:, 1], s=5, color="#c0c7d1", label="all input poses")
            for record in chunk_records:
                inv = np.array(record["M_chunk_to_original"])
                side = float(inv[0, 0]);center = inv[:3, 3]
                ax.add_patch(Rectangle(center[:2] - side / 2, side, side,
                                       fill=False, color="#a4aab3", lw=.6, alpha=.5))
            color = np.linspace(0, 1, len(poses))[chosen]
            ax.scatter(positions[chosen, 0], positions[chosen, 1], s=15, c=color, cmap="viridis", zorder=3)
            arrow = forward[chosen, :2] * (.04 * span)
            ax.quiver(positions[chosen, 0], positions[chosen, 1], arrow[:, 0], arrow[:, 1],
                      angles="xy", scale_units="xy", scale=1, width=.002, color="#324b6b", alpha=.7)
            counts = case["actual_crop_visible_originals_per_chunk"]
            yaw_bins = [item["occupied_yaw_bins"] for item in case["pose"]["per_chunk_directions"]]
            ax.set_title(f"{strategy}: cap {budget}, selected {len(chosen)}\n"
                         f"chunk frustum views {min(counts)}–{max(counts)}; near-duplicate pairs {case['pose']['near_duplicate_selected_pairs']}; "
                         f"mean yaw bins {np.mean(yaw_bins):.2f}")
            ax.set_xlabel("upright world X (scene units)");ax.set_ylabel("upright world Y (scene units)")
            ax.set_aspect("equal", adjustable="box");ax.grid(alpha=.15)
    fig.suptitle("Fixed full-scene chunks: selected camera centres and optical axes\n"
                 "Colours follow original capture order; frustum/direction statistics are selection proxies, not mesh quality", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .965 if len(budgets) > 1 else .88))
    fig.savefig(output / filename, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--chunk-transforms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 96, 128, 529])
    parser.add_argument("--position-threshold", type=float, default=.05)
    parser.add_argument("--rotation-threshold", type=float, default=10.)
    parser.add_argument("--min-views-per-chunk", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if any(budget < 1 for budget in args.budgets):
        parser.error("budgets must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    chunk_record = json.loads(args.chunk_transforms.read_text())["chunks"]
    transforms = [torch.tensor(record["M_original_to_chunk"], dtype=torch.float32) for record in chunk_record]
    settings = dict(pose_position_threshold=args.position_threshold,
                    pose_rotation_threshold=args.rotation_threshold,
                    min_views_per_chunk=args.min_views_per_chunk)
    loader = IphoneImageSelecter(view_selection="uniform", **settings)
    cameras = loader._get_cameras(args.data_dir / "colmap/cameras.txt")
    paths = [camera["img_path"] for camera in cameras]
    names = [Path(path).name for path in paths]
    poses = np.stack([camera["c2w_blender"].numpy() for camera in cameras]).astype(np.float64)
    intrinsics = np.stack([camera["intrinsics"].numpy() for camera in cameras]).astype(np.float64)
    areas = loader._camera_chunk_areas(cameras, transforms)
    visibility = areas >= .4
    position_scale = float(np.median([abs(np.linalg.det(t[:3, :3].double().numpy())) ** (-1. / 3.)
                                     for t in transforms]))
    tracks = load_tracks(args.data_dir / "colmap", names, transforms)
    cases = []
    for budget in args.budgets:
        for strategy in ("uniform", "pose_coverage"):
            selector = IphoneImageSelecter(view_selection=strategy, **settings)
            start = time.perf_counter()
            selected = selector._sample_scene_cameras(cameras, budget, random.Random(args.seed), transforms)
            elapsed = time.perf_counter() - start
            selection = dict(selector._selection_report)
            chosen = selection["selected_indices"]
            selected_paths = [camera["img_path"] for camera in selected]
            if selected_paths != [paths[index] for index in chosen]:
                raise AssertionError("Selector report and actual returned camera pool differ")
            case = {
                "strategy": strategy, "budget": budget, "selected_original_images": len(selected),
                "selected_crop_views": len(selector._prepare_scene_pool(selected)),
                "selected_indices": chosen, "selection_cpu_seconds": elapsed,
                "actual_crop_visible_originals_per_chunk": visibility[chosen].sum(axis=0).tolist(),
                "uncovered_chunks": np.flatnonzero(~visibility[chosen].any(axis=0)).tolist(),
                "pose": pose_report(chosen, poses, intrinsics, visibility, position_scale,
                                    args.position_threshold, args.rotation_threshold),
                "tracks": track_report(chosen, tracks, len(cameras)),
            }
            target = args.output / f"{strategy}_{budget}"
            target.mkdir(exist_ok=True)
            write_json(target / "view_selection.json", selection)
            write_json(target / "selection_metrics.json", case)
            write_json(target / "selected_paths.json", selected_paths)
            (target / "selected_paths.txt").write_text("\n".join(selected_paths) + "\n")
            cases.append(case)
    sources = [Path(__file__), Path(__file__).parents[1] / "inference/get_images.py",
               Path(__file__).parents[1] / "inference/view_selection.py"]
    report = {
        "status": "actual_cpu_selectors_benchmarked", "gpu_used": False,
        "models_loaded": False, "full_resolution_rgb_decoded": False,
        "data_dir": str(args.data_dir.resolve()), "fixed_chunk_transforms": str(args.chunk_transforms.resolve()),
        "fixed_chunk_transforms_sha256": digest(args.chunk_transforms),
        "input_original_images": len(cameras), "fixed_chunks": len(transforms),
        "position_scale_scene_units": position_scale,
        "available_actual_crop_visible_originals_per_chunk": visibility.sum(axis=0).tolist(),
        "configuration": settings, "seed": args.seed,
        "source_sha256": {str(path): digest(path) for path in sources},
        "metric_limits": ["Nine-point projected crop-frustum bounding boxes do not test depth occlusion.",
                          "Camera optical-axis bins are a view-direction proxy, not true observed surface coverage.",
                          "COLMAP-track coverage measures retention of existing sparse feature observations, not generated mesh accuracy."],
        "cases": cases,
    }
    write_json(args.output / "report.json", report)
    plot_selection(args.output, poses, chunk_record, cases, args.budgets)
    if 128 in args.budgets:
        plot_selection(args.output, poses, chunk_record, cases, [128], "camera_pose_selection_128.png")
    concise = []
    for case in cases:
        counts = case["actual_crop_visible_originals_per_chunk"]
        concise.append({"strategy": case["strategy"], "budget": case["budget"],
                        "selected": case["selected_original_images"], "crops": case["selected_crop_views"],
                        "cpu_seconds": round(case["selection_cpu_seconds"], 3),
                        "frustum_count_min_max": [min(counts), max(counts)],
                        "duplicate_pairs": case["pose"]["near_duplicate_selected_pairs"],
                        "mean_yaw_bins": round(float(np.mean([item["occupied_yaw_bins"]
                                                       for item in case["pose"]["per_chunk_directions"]])), 3),
                        "sparse_point_retention_at_least_one": round(case["tracks"]["fraction_at_least_one"], 5),
                        "sparse_point_retention_at_least_two": round(case["tracks"]["fraction_at_least_two"], 5)})
    write_json(args.output / "summary.json", concise)
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
