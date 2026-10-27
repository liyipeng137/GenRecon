#!/usr/bin/env python3
"""Compare two saved CPU raycast inspections on the same eight cameras.

These are cross-run geometry differences, not ground-truth errors. No-hit rays
measure model intersections and do not by themselves establish scene holes.
The original inspector did not serialize extrinsics: matching image IDs, K,
coordinate transforms and pixels is checked, while both inspections must have
been rendered using the same camera dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_VIEWS = tuple(f"frame_{index:06d}.png" for index in (0, 75, 151, 226, 302, 377, 453, 528))
DEFAULT_FOCUS = tuple(f"frame_{index:06d}.png" for index in (75, 302, 453))
INTERPRETATION = (
    "Cross-run comparison on matched camera rays, not ground-truth reconstruction error. "
    "No-hit changes measure model intersection coverage, not verified scene holes. "
    "Independent overlapping reconstructions were not evaluated; partitioning one mesh "
    "cannot provide evidence of independent overlap quality."
)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def percentiles(values):
    return np.percentile(values, [50, 90, 95]).tolist() if values.size else None


def compare_arrays(baseline_depth, baseline_hit, baseline_normals,
                   candidate_depth, candidate_hit, candidate_normals):
    shape = baseline_depth.shape
    if (len(shape) != 2 or candidate_depth.shape != shape or baseline_hit.shape != shape or
            candidate_hit.shape != shape or baseline_normals.shape != (*shape, 3) or
            candidate_normals.shape != (*shape, 3)):
        raise ValueError("Raycast array shapes disagree")
    if baseline_hit.dtype != np.bool_ or candidate_hit.dtype != np.bool_:
        raise ValueError("Ray-hit masks must be boolean arrays")
    for depth, hit, normal in ((baseline_depth, baseline_hit, baseline_normals),
                               (candidate_depth, candidate_hit, candidate_normals)):
        if not np.isfinite(depth[hit]).all() or not (depth[hit] > 0).all() or not np.isfinite(normal[hit]).all():
            raise ValueError("Invalid camera-Z or normals on a valid ray")
    common = baseline_hit & candidate_hit
    gained, lost = ~baseline_hit & candidate_hit, baseline_hit & ~candidate_hit
    difference = candidate_depth[common].astype(np.float64) - baseline_depth[common]
    relative = np.abs(difference) / np.maximum((candidate_depth[common] + baseline_depth[common]) * .5, 1e-12)
    norm_a, norm_b = np.linalg.norm(baseline_normals, axis=-1), np.linalg.norm(candidate_normals, axis=-1)
    normal_valid = common & (norm_a > 1e-8) & (norm_b > 1e-8)
    dots = np.sum(baseline_normals[normal_valid].astype(np.float64) * candidate_normals[normal_valid], axis=-1)
    dots /= norm_a[normal_valid].astype(np.float64) * norm_b[normal_valid]
    dots = np.clip(dots, -1, 1)
    normal_angles = np.degrees(np.arccos(np.abs(dots)))
    return {
        "pixel_count": int(common.size), "common_hit_count": int(common.sum()),
        "baseline_no_hit_count": int((~baseline_hit).sum()),
        "candidate_no_hit_count": int((~candidate_hit).sum()),
        "baseline_no_hit_fraction": float((~baseline_hit).mean()),
        "candidate_no_hit_fraction": float((~candidate_hit).mean()),
        "no_hit_change_percentage_points_candidate_minus_baseline": float(100 * ((~candidate_hit).mean() - (~baseline_hit).mean())),
        "candidate_only_hit_count": int(gained.sum()), "baseline_only_hit_count": int(lost.sum()),
        "both_no_hit_count": int((~baseline_hit & ~candidate_hit).sum()),
        "common_hit_absolute_camera_z_difference_p50_p90_p95_scene_units": percentiles(np.abs(difference)),
        "common_hit_relative_camera_z_difference_p50_p90_p95": percentiles(relative),
        "common_hit_signed_camera_z_difference_p50_candidate_minus_baseline": float(np.median(difference)) if difference.size else None,
        "common_hit_unoriented_normal_angle_p50_p90_p95_degrees": percentiles(normal_angles),
        "common_hit_opposite_normal_fraction_dot_lt_minus09": float((dots < -.9).mean()) if dots.size else None,
    }


def matched_view_metadata(baseline_report, candidate_report, view_names):
    if not view_names or len(set(view_names)) != len(view_names):
        raise ValueError("Camera view names must be nonempty and unique")
    if baseline_report.get("coordinate_transform_sha256") != candidate_report.get("coordinate_transform_sha256"):
        raise ValueError("Inspection coordinate transforms differ")
    if not baseline_report.get("coordinate_transform_sha256"):
        raise ValueError("Inspection coordinate-transform provenance is missing")
    if baseline_report.get("preview_protocol") != candidate_report.get("preview_protocol"):
        raise ValueError("Inspection raycast protocols differ")
    maps = [{row["image_name"]: row for row in report["previews"]}
            for report in (baseline_report, candidate_report)]
    if any(len(mapping) != len(report["previews"]) for mapping, report in zip(maps, (baseline_report, candidate_report))):
        raise ValueError("Inspection contains duplicate camera view records")
    for name in view_names:
        if name not in maps[0] or name not in maps[1]:
            raise ValueError(f"Required camera view missing: {name}")
        a, b = maps[0][name], maps[1][name]
        for key in ("image_id", "width", "height"):
            if a[key] != b[key]:
                raise ValueError(f"Camera {key} differs: {name}")
        if not a.get("half_integer_pixel_centres") or not b.get("half_integer_pixel_centres"):
            raise ValueError(f"Expected half-integer camera rays: {name}")
        if np.shape(a["K_scaled"]) != (3, 3) or np.shape(b["K_scaled"]) != (3, 3) or not np.allclose(a["K_scaled"], b["K_scaled"], atol=1e-10, rtol=0):
            raise ValueError(f"Camera K differs: {name}")
    return maps


def load_view(root, name):
    folder = root / "views" / Path(name).stem
    result = {"depth": np.load(folder / "camera_z.npy"),
              "hit": np.load(folder / "ray_hit_mask.npy"),
              "normal": np.load(folder / "camera_normals.npy")}
    for key, filename in (("rgb", "real_rgb.png"), ("gray", "gray_geometry.png")):
        with Image.open(folder / filename) as image:
            result[key] = np.asarray(image.convert("RGB"))
    return result


def render_row(name, a, b, depth_range, cell_width):
    from matplotlib import colormaps

    lo, hi = depth_range
    colors = []
    for view in (a, b):
        color = (colormaps["turbo"](np.clip((view["depth"] - lo) / (hi - lo), 0, 1))[..., :3] * 255).astype(np.uint8)
        color[~view["hit"]] = [24, 28, 34]
        colors.append(color)
    hit_change = np.full((*a["hit"].shape, 3), [220, 224, 228], dtype=np.uint8)
    hit_change[~a["hit"] & ~b["hit"]] = [24, 28, 34]
    hit_change[a["hit"] & ~b["hit"]] = [230, 120, 45]
    hit_change[~a["hit"] & b["hit"]] = [30, 190, 180]
    arrays = [a["rgb"], a["gray"], b["gray"], *colors, hit_change]
    labels = ["Real RGB", "Baseline geometry", "Candidate geometry", "Baseline camera-Z", "Candidate camera-Z", "Ray hit change"]
    height = round(a["depth"].shape[0] * cell_width / a["depth"].shape[1])
    row = Image.new("RGB", (6 * cell_width, height + 75), "#11151a")
    draw = ImageDraw.Draw(row)
    for column, (array, label) in enumerate(zip(arrays, labels)):
        row.paste(Image.fromarray(array).resize((cell_width, height), Image.Resampling.LANCZOS), (column * cell_width, 35))
        draw.text((column * cell_width + 5, 5), label, fill="white")
        draw.text((column * cell_width + 5, 19), name, fill="#b5c1cf")
    draw.text((5, height + 41), f"Z colors shared: {lo:.4g}..{hi:.4g} scene units. Hit change: teal=candidate only; orange=baseline only; dark=neither.", fill="#c8d1dc")
    draw.text((5, height + 57), "Cross-run differences, not GT error. Gray is geometry only. Missing rays do not by themselves prove holes.", fill="#c8d1dc")
    return row


def save_sheet(rows, path):
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "#11151a")
    position = 0
    for row in rows:
        sheet.paste(row, (0, position))
        position += row.height
    sheet.save(path, quality=92)


def compare_inspections(baseline, candidate, output, *, view_names=DEFAULT_VIEWS, focus_names=DEFAULT_FOCUS,
                        cell_width=180, camera_data_dir=None):
    start = time.perf_counter()
    baseline, candidate, output = Path(baseline).resolve(), Path(candidate).resolve(), Path(output).resolve()
    report_paths = [root / "inspection_report.json" for root in (baseline, candidate)]
    reports = [json.loads(path.read_text()) for path in report_paths]
    maps = matched_view_metadata(*reports, view_names)
    if not set(focus_names).issubset(view_names):
        raise ValueError("Focus cameras must be included in compared views")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty comparison output: {output}")
    # Validate and calculate before producing previews. At most eight small ray
    # arrays are retained; no meshes, adjacency structures or GPU are loaded.
    pairs, metrics, all_depths = {}, [], []
    for name in view_names:
        a, b = load_view(baseline, name), load_view(candidate, name)
        if not np.array_equal(a["rgb"], b["rgb"]):
            raise ValueError(f"Real RGB pixels differ: {name}")
        expected_shape = (maps[0][name]["height"], maps[0][name]["width"])
        if a["depth"].shape != expected_shape or a["rgb"].shape != (*expected_shape, 3) or any(view["gray"].shape != (*expected_shape, 3) for view in (a, b)):
            raise ValueError(f"Preview dimensions disagree with camera metadata: {name}")
        metric = compare_arrays(a["depth"], a["hit"], a["normal"], b["depth"], b["hit"], b["normal"])
        metric.update(image_name=name, image_id=maps[0][name]["image_id"], width=expected_shape[1], height=expected_shape[0], K_scaled=maps[0][name]["K_scaled"])
        metrics.append(metric)
        pairs[name] = (a, b)
        all_depths.extend(view["depth"][view["hit"]] for view in (a, b))
    depths = np.concatenate(all_depths)
    lo, hi = np.percentile(depths, [2, 98]) if depths.size else (0., 1.)
    if hi <= lo:
        hi = lo + max(abs(lo) * .01, 1e-6)
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, (a, b) in pairs.items():
        row = render_row(name, a, b, (lo, hi), cell_width)
        row.save(output / (Path(name).stem + "_comparison.jpg"), quality=92)
        rows[name] = row
    save_sheet(list(rows.values()), output / "comparison_contact_sheet.jpg")
    if focus_names:
        save_sheet([rows[name] for name in focus_names], output / "focus_075_302_453.jpg")
    totals = {key: sum(row[key] for row in metrics) for key in (
        "pixel_count", "common_hit_count", "baseline_no_hit_count", "candidate_no_hit_count",
        "candidate_only_hit_count", "baseline_only_hit_count", "both_no_hit_count")}
    totals["baseline_no_hit_fraction"] = totals["baseline_no_hit_count"] / totals["pixel_count"]
    totals["candidate_no_hit_fraction"] = totals["candidate_no_hit_count"] / totals["pixel_count"]
    totals["no_hit_change_percentage_points_candidate_minus_baseline"] = 100 * (totals["candidate_no_hit_fraction"] - totals["baseline_no_hit_fraction"])
    camera_provenance = None
    if camera_data_dir is not None:
        camera_data_dir = Path(camera_data_dir).resolve()
        camera_provenance = {"path": str(camera_data_dir), "role": "Declared common camera source for both inspections; poses are not serialized in the original inspection reports.",
                             "txt_files": {str(path.relative_to(camera_data_dir)): sha256(path)
                                           for path in sorted(camera_data_dir.rglob("*.txt"))
                                           if path.name in ("images.txt", "cameras.txt")}}
    same_mesh = bool(reports[0].get("raw_sha256") and reports[0].get("raw_sha256") == reports[1].get("raw_sha256"))
    report = {"status": "matched_camera_cross_run_comparison_complete", "interpretation": INTERPRETATION,
              "independent_overlap_quality_evaluated": False, "same_raw_mesh_sha256": same_mesh,
              "same_mesh_interpretation": "Zero differences for the same mesh are only a consistency check, not evidence of independent overlap quality." if same_mesh else None,
              "baseline_inspection": str(baseline), "candidate_inspection": str(candidate),
              "inspection_report_sha256": [sha256(path) for path in report_paths],
              "raw_mesh_sha256": [report.get("raw_sha256") for report in reports],
              "coordinate_transform_sha256": reports[0]["coordinate_transform_sha256"],
              "matched_camera_checks": "Exact image IDs/dimensions/RGB pixels, K within 1e-10, same transform hash/protocol, half-integer pixel centres. Camera pose equivalence relies on rendering both inspections from the same declared camera dataset.",
              "camera_data_provenance": camera_provenance, "view_names": list(view_names), "focus_names": list(focus_names),
              "depth_shared_color_range_scene_units": [float(lo), float(hi)],
              "totals_over_requested_camera_rays": totals, "views": metrics,
              "contact_sheet": str(output / "comparison_contact_sheet.jpg"),
              "focus_contact_sheet": str(output / "focus_075_302_453.jpg") if focus_names else None,
              "elapsed_seconds": time.perf_counter() - start,
              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
              "comparison_script_sha256": sha256(__file__)}
    (output / "comparison_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-inspection", type=Path, required=True)
    parser.add_argument("--candidate-inspection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-data-dir", type=Path)
    parser.add_argument("--views", nargs="+", default=DEFAULT_VIEWS)
    parser.add_argument("--focus-views", nargs="*", default=DEFAULT_FOCUS)
    parser.add_argument("--cell-width", type=int, default=180)
    args = parser.parse_args()
    if args.cell_width <= 0:
        parser.error("--cell-width must be positive")
    result = compare_inspections(args.baseline_inspection, args.candidate_inspection, args.output,
                                 view_names=args.views, focus_names=args.focus_views,
                                 cell_width=args.cell_width, camera_data_dir=args.camera_data_dir)
    print(json.dumps({key: result[key] for key in ("status", "totals_over_requested_camera_rays", "contact_sheet", "focus_contact_sheet", "elapsed_seconds", "peak_rss_gib")}, indent=2))


if __name__ == "__main__":
    main()
