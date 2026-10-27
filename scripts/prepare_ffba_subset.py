#!/usr/bin/env python3
"""Prepare a reciprocal COLMAP subset and an explicit, reversible upright frame.

Only the selected source RGBs, their trusted depth files, and their reciprocal BA
observations determine the output. Source calibration was fitted upstream; this
script does not claim a new end-to-end SfM holdout or an absolute metric scale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def unpack(handle, fmt):
    return struct.unpack("<" + fmt, handle.read(struct.calcsize("<" + fmt)))


def read_model(model):
    cameras, images, points = {}, {}, {}
    with (model / "cameras.bin").open("rb") as handle:
        for _ in range(unpack(handle, "Q")[0]):
            camera_id, model_id, width, height = unpack(handle, "iiQQ")
            name, count = CAMERA_MODELS[model_id]
            params = np.array(unpack(handle, "d" * count), dtype=np.float64)
            cameras[camera_id] = dict(id=camera_id, model=name, width=width, height=height, params=params)
    with (model / "images.bin").open("rb") as handle:
        for _ in range(unpack(handle, "Q")[0]):
            data = unpack(handle, "i" + "d" * 7 + "i")
            name = bytearray()
            while (char := handle.read(1)) != b"\0":
                if not char:
                    raise EOFError("Missing COLMAP image name terminator")
                name.extend(char)
            count = unpack(handle, "Q")[0]
            observations = np.frombuffer(handle.read(count * 24), dtype=[("xy", "<f8", (2,)), ("pid", "<i8")]).copy()
            qvec = np.array(data[1:5], dtype=np.float64)
            rot = Rotation.from_quat(qvec[[1, 2, 3, 0]]).as_matrix()
            images[data[0]] = dict(id=data[0], name=name.decode("utf8"), qvec=qvec,
                                   tvec=np.array(data[5:8]), camera_id=data[8], R=rot,
                                   xys=observations["xy"], pids=observations["pid"])
    with (model / "points3D.bin").open("rb") as handle:
        for _ in range(unpack(handle, "Q")[0]):
            data = unpack(handle, "QdddBBBd")
            count = unpack(handle, "Q")[0]
            tracks = np.frombuffer(handle.read(count * 8), dtype="<i4").reshape(-1, 2).copy()
            points[data[0]] = dict(id=data[0], xyz=np.array(data[1:4]), rgb=data[4:7], error=data[7], tracks=tracks)
    return cameras, images, points


def intrinsic(camera):
    if camera["model"] == "SIMPLE_PINHOLE":
        f, cx, cy = camera["params"]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.]])
    if camera["model"] == "PINHOLE":
        fx, fy, cx, cy = camera["params"]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    raise ValueError("Trusted camera-Z backprojection currently requires undistorted pinhole input")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unit(vector):
    return vector / np.linalg.norm(vector)


def collect_depth_geometry(source, cameras, selected, rng):
    all_points, all_normals, frame_ids, camera_centers, files, diagnostics = [], [], [], [], [], []
    for image in selected:
        path = source / "depth" / Path(image["name"]).with_suffix(".npy")
        if not path.is_file():
            continue
        depth = np.load(path, allow_pickle=False)
        camera = cameras[image["camera_id"]]
        if depth.shape != (camera["height"], camera["width"]):
            raise ValueError(f"Depth shape differs from calibrated camera: {path}")
        K = intrinsic(camera)
        yy, xx = np.meshgrid(np.arange(12, depth.shape[0] - 12, 16),
                             np.arange(12, depth.shape[1] - 12, 16), indexing="ij")
        yy, xx = yy.ravel(), xx.ravel()
        offsets = [(0, 0), (-5, 0), (5, 0), (0, -5), (0, 5)]
        zs = np.stack([depth[yy + dy, xx + dx] for dx, dy in offsets], axis=1).astype(np.float64)
        valid = np.isfinite(zs).all(1) & (zs > 0).all(1)
        valid &= (np.abs(zs[:, 1:] / np.maximum(zs[:, :1], 1e-12) - 1) < .08).all(1)
        xx, yy, zs = xx[valid], yy[valid], zs[valid]
        if not len(zs):
            continue
        # FFBA/COLMAP K is half-integer: array centre (x+.5,y+.5).
        pcam = np.stack([np.stack([(xx + dx + .5 - K[0, 2]) / K[0, 0] * zs[:, j],
                                   (yy + dy + .5 - K[1, 2]) / K[1, 1] * zs[:, j],
                                   zs[:, j]], axis=1)
                         for j, (dx, dy) in enumerate(offsets)], axis=1)
        normals = np.cross(pcam[:, 2] - pcam[:, 1], pcam[:, 4] - pcam[:, 3])
        lengths = np.linalg.norm(normals, axis=1)
        ok = lengths > 1e-12
        pcam, normals = pcam[ok], normals[ok] / lengths[ok, None]
        if len(pcam) > 1800:
            pick = rng.choice(len(pcam), 1800, replace=False)
            pcam, normals = pcam[pick], normals[pick]
        center = -image["R"].T @ image["tvec"]
        all_points.append(pcam[:, 0] @ image["R"] + center)
        all_normals.append(normals @ image["R"])
        frame_ids.extend([image["id"]] * len(pcam))
        camera_centers.extend([center] * len(pcam))
        files.append(path)
        diagnostics.append(dict(image_name=image["name"], samples=len(pcam), median_z=float(np.median(zs[:, 0]))))
    if not all_points:
        raise ValueError("No selected trusted depth geometry available to establish upright frame")
    return (np.concatenate(all_points), np.concatenate(all_normals), np.array(frame_ids),
            np.array(camera_centers), files, diagnostics)


def upright_transform(points, normals, frame_ids, observed_centers, selected, rng):
    camera_ups = np.array([-image["R"][1] for image in selected])
    up_reference = unit(np.median(camera_ups, axis=0))
    camera_centers = np.array([-image["R"].T @ image["tvec"] for image in selected])
    sample_ranges = np.linalg.norm(points - observed_centers, axis=1)
    scene_range = float(np.median(sample_ranges))
    threshold = .0125 * scene_range
    normal_alignment = np.abs(normals @ up_reference)
    relative_height = np.einsum("ij,j->i", points - observed_centers, up_reference)
    floor_candidate = (normal_alignment > np.cos(np.radians(30))) & (relative_height < -.08 * scene_range)
    floor_points, floor_normals, floor_frames = points[floor_candidate], normals[floor_candidate], frame_ids[floor_candidate]
    if len(floor_points) < 1000:
        raise ValueError(f"Insufficient horizontal below-camera floor candidates: {len(floor_points)}")
    search_indices = rng.choice(len(floor_points), min(16000, len(floor_points)), replace=False)
    search_points, search_normals = floor_points[search_indices], floor_normals[search_indices]
    best, best_mask, best_score = None, None, (-1, -1)
    for _ in range(512):
        triplet = search_points[rng.choice(len(search_points), 3, replace=False)]
        normal = np.cross(triplet[1] - triplet[0], triplet[2] - triplet[0])
        if np.linalg.norm(normal) < 1e-8:
            continue
        normal = unit(normal)
        if normal @ up_reference < 0:
            normal = -normal
        if normal @ up_reference < np.cos(np.radians(25)):
            continue
        offset = -float(normal @ triplet[0])
        inlier = np.abs(search_points @ normal + offset) < threshold
        inlier &= np.abs(search_normals @ normal) > np.cos(np.radians(20))
        inlier_frames, counts = np.unique(floor_frames[search_indices][inlier], return_counts=True)
        score = (int((counts >= 10).sum()), int(inlier.sum()))
        if score > best_score:
            best, best_mask, best_score = (normal, offset), inlier, score
    if best is None:
        raise ValueError("No defensible floor plane found")
    normal, offset = best
    for _ in range(3):
        inlier = (np.abs(floor_points @ normal + offset) < threshold)
        inlier &= np.abs(floor_normals @ normal) > np.cos(np.radians(20))
        centroid = floor_points[inlier].mean(0)
        covariance = (floor_points[inlier] - centroid).T @ (floor_points[inlier] - centroid)
        normal = np.linalg.eigh(covariance)[1][:, 0]
        if normal @ up_reference < 0:
            normal = -normal
        offset = -float(normal @ centroid)
    inlier = (np.abs(floor_points @ normal + offset) < threshold)
    inlier &= np.abs(floor_normals @ normal) > np.cos(np.radians(20))
    supported_frames, counts = np.unique(floor_frames[inlier], return_counts=True)
    supported_frames = supported_frames[counts >= 10]
    residuals = np.abs(floor_points[inlier] @ normal + offset)
    angle = float(np.degrees(np.arccos(np.clip(normal @ up_reference, -1, 1))))
    if len(supported_frames) < 10 or int(inlier.sum()) < 1000 or angle > 25:
        raise ValueError(f"Floor orientation uncertain: frames={len(supported_frames)}, points={inlier.sum()}, camera-up angle={angle}")
    # Minimal rotation aligns plane normal to +Z; no arbitrary yaw change.
    target = np.array([0., 0., 1.])
    axis = np.cross(normal, target)
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-12:
        rotate = np.eye(3) if normal @ target > 0 else Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    else:
        rotate = Rotation.from_rotvec(unit(axis) * np.arctan2(axis_length, normal @ target)).as_matrix()
    # Put the projected median selected camera centre on the floor at world origin.
    origin = np.median(camera_centers, axis=0)
    origin -= (origin @ normal + offset) * normal
    transform = np.eye(4)
    transform[:3, :3] = rotate
    transform[:3, 3] = -rotate @ origin
    angles = np.degrees(np.arccos(np.clip(camera_ups @ up_reference, -1, 1)))
    plane_heights = camera_centers @ normal + offset
    report = dict(method="selected_trusted_depth_floor_plane_ransac_camera_up_disambiguation",
                  scene_units="unaltered COLMAP scene units; no metre assumption", seed=20261027,
                  camera_up_median=up_reference.tolist(), camera_up_deviation_degrees_p50_p90_max=np.percentile(angles, [50, 90, 100]).tolist(),
                  floor_plane_normal_original=normal.tolist(), floor_plane_offset_original=offset,
                  floor_candidate_count=len(floor_points), floor_inlier_count=int(inlier.sum()),
                  floor_supported_real_frame_count=len(supported_frames), floor_supported_real_frame_ids=supported_frames.tolist(),
                  floor_camera_up_angle_degrees=angle, median_observed_range_scene_units=scene_range,
                  plane_inlier_threshold_scene_units=threshold,
                  floor_residual_p50_p90_max_scene_units=np.percentile(residuals, [50, 90, 100]).tolist(),
                  selected_camera_heights_p05_p50_p95_scene_units=np.percentile(plane_heights, [5, 50, 95]).tolist(),
                  normal_selection="within 30deg of camera-up axis, below each source camera by >0.08 median observed range",
                  trust_gate="at least 1000 inliers, >=10 real frames each >=10 samples, floor normal <=25deg from median camera-up",
                  origin_original=origin.tolist(), rotation_policy="minimal rotation floor normal to +Z; yaw not separately optimized")
    return transform, report, floor_candidate, normal, offset


def write_contact(source, selected, output):
    thumb_w, thumb_h, label = 168, 229, 22
    sheet = Image.new("RGB", (10 * thumb_w, 6 * (thumb_h + label)), "#151515")
    draw = ImageDraw.Draw(sheet)
    for idx, image in enumerate(selected):
        with Image.open(source / "images" / image["name"]) as rgb:
            rgb.thumbnail((thumb_w, thumb_h))
            x, y = (idx % 10) * thumb_w, (idx // 10) * (thumb_h + label)
            sheet.paste(rgb, (x, y + label))
            draw.text((x + 3, y + 4), image["name"], fill="white")
    sheet.save(output / "first60_contact_sheet.jpg", quality=90)


def diagnostic_plot(points, normals, candidate, normal, offset, transform, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = points @ transform[:3, :3].T + transform[:3, 3]
    floor = candidate & (np.abs(points @ normal + offset) < .025 * np.median(np.linalg.norm(points - points.mean(0), axis=1)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, indices, labels in zip(axes, [(0, 1), (0, 2), (1, 2)], [("x", "y"), ("x", "z"), ("y", "z")]):
        ax.scatter(p[::5, indices[0]], p[::5, indices[1]], s=.25, c="#8797a8", alpha=.4)
        ax.scatter(p[floor, indices[0]], p[floor, indices[1]], s=.4, c="#c25832", alpha=.55, label="depth floor support")
        ax.set_xlabel(labels[0] + " (scene units)"); ax.set_ylabel(labels[1] + " (scene units)")
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.2)
    axes[1].axhline(0, color="black", lw=.8)
    axes[2].axhline(0, color="black", lw=.8)
    fig.suptitle("Only first 60 real frames / 43 available trusted depth maps; rigid upright transform")
    fig.tight_layout(); fig.savefig(output / "upright_depth_geometry.png", dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--min-selected-track", type=int, default=2)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cameras, images, points = read_model(source / "sparse" / "0")
    selected = sorted(images.values(), key=lambda image: image["name"])[:args.count]
    if len(selected) != args.count:
        raise ValueError("Not enough source registered images")
    selected_by_id = {image["id"]: image for image in selected}
    rng = np.random.default_rng(20261027)
    geometry = collect_depth_geometry(source, cameras, selected, rng)
    xyz, normals, frame_ids, observed_centers, depth_files, depth_diagnostics = geometry
    transform, orientation, candidate, normal, offset = upright_transform(xyz, normals, frame_ids, observed_centers, selected, rng)
    inverse = np.linalg.inv(transform)
    transform_record = dict(original_to_genrecon=transform.tolist(), genrecon_to_original=inverse.tolist(),
                            coordinate_convention="COLMAP world / OpenCV camera axes; output world is floor-aligned +Z up",
                            length_scale_factor=1., provenance=orientation)
    (output / "world_transform.json").write_text(json.dumps(transform_record, indent=2, allow_nan=False) + "\n")
    kept = {}
    bad_reciprocal = 0
    for pid, point in points.items():
        tracks = []
        for image_id, index in point["tracks"]:
            if int(image_id) not in selected_by_id:
                continue
            image = selected_by_id[int(image_id)]
            if index < 0 or index >= len(image["pids"]) or int(image["pids"][index]) != pid:
                bad_reciprocal += 1
                continue
            tracks.append((int(image_id), int(index)))
        tracks = sorted(set(tracks))
        if len({image_id for image_id, _ in tracks}) >= args.min_selected_track:
            kept[pid] = dict(**point, selected_tracks=tracks,
                             transformed_xyz=transform[:3, :3] @ point["xyz"] + transform[:3, 3])
    if not kept:
        raise ValueError("No reciprocal subset-supported BA points")
    rgb, colmap = output / "rgb", output / "colmap"
    rgb.mkdir(); colmap.mkdir()
    used_cameras = sorted({image["camera_id"] for image in selected})
    with (colmap / "cameras.txt").open("w") as handle:
        handle.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        for cid in used_cameras:
            cam = cameras[cid]
            handle.write(f'{cid} {cam["model"]} {cam["width"]} {cam["height"]} ' + " ".join(f"{x:.17g}" for x in cam["params"]) + "\n")
    reciprocal_count, projection_error, camera_error, dimensions = 0, [], [], []
    selected_sources = []
    with (colmap / "images.txt").open("w") as handle:
        handle.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as (X,Y,POINT3D_ID)\n")
        for image in selected:
            src = source / "images" / image["name"]
            if Path(image["name"]).name != image["name"]:
                raise ValueError("This standalone GenRecon input requires unique flat image names")
            cam = cameras[image["camera_id"]]
            with Image.open(src) as im:
                if im.size != (cam["width"], cam["height"]):
                    raise ValueError(f"RGB dimensions mismatch: {src}")
            dimensions.append(dict(image_name=image["name"], width=cam["width"], height=cam["height"]))
            (rgb / image["name"]).symlink_to(src)
            selected_sources.append(dict(image_name=image["name"], image_id=image["id"], rgb_sha256=digest(src),
                                         source_pose_qvec=image["qvec"].tolist(), source_pose_tvec=image["tvec"].tolist()))
            Rnew = image["R"] @ transform[:3, :3].T
            tnew = image["tvec"] - Rnew @ transform[:3, 3]
            quat = Rotation.from_matrix(Rnew).as_quat()[[3, 0, 1, 2]]
            if quat[0] < 0:
                quat = -quat
            handle.write(f'{image["id"]} ' + " ".join(f"{x:.17g}" for x in np.r_[quat, tnew]) + f' {image["camera_id"]} {image["name"]}\n')
            obs = []
            for index, (xy, pid) in enumerate(zip(image["xys"], image["pids"])):
                valid = int(pid) in kept and (image["id"], index) in kept[int(pid)]["selected_tracks"]
                obs.append(f'{xy[0]:.17g} {xy[1]:.17g} {int(pid) if valid else -1}')
                if valid:
                    reciprocal_count += 1
                    X = points[int(pid)]["xyz"]
                    Y = kept[int(pid)]["transformed_xyz"]
                    old_camera = image["R"] @ X + image["tvec"]
                    new_camera = Rnew @ Y + tnew
                    camera_error.append(np.max(np.abs(old_camera - new_camera)))
                    if old_camera[2] > 0:
                        K = intrinsic(cam)
                        old_uv, new_uv = K @ old_camera, K @ new_camera
                        projection_error.append(np.max(np.abs(old_uv[:2] / old_uv[2] - new_uv[:2] / new_uv[2])))
            handle.write(" ".join(obs) + "\n")
    with (colmap / "points3D.txt").open("w") as handle:
        handle.write("# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID,POINT2D_IDX)\n")
        for pid in sorted(kept):
            point = kept[pid]
            handle.write(f'{pid} ' + " ".join(f"{x:.17g}" for x in point["transformed_xyz"]) + " " +
                         " ".join(str(int(x)) for x in point["rgb"]) + f' {point["error"]:.17g} ' +
                         " ".join(f"{image_id} {index}" for image_id, index in point["selected_tracks"]) + "\n")
    max_projection = float(max(projection_error, default=0))
    max_camera = float(max(camera_error, default=0))
    if max_projection > 1e-7 or max_camera > 1e-9 or not np.allclose(transform @ inverse, np.eye(4), atol=1e-12):
        raise AssertionError("Global rigid transform changed camera projection")
    points_new = np.array([point["transformed_xyz"] for point in kept.values()])
    write_contact(source, selected, output)
    diagnostic_plot(xyz, normals, candidate, normal, offset, transform, output)
    assets = [source / "sparse" / "0" / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    report = dict(schema_version=1, status="prepared_and_numerically_verified", source_root=str(source),
                  output_root=str(output), selection="lexicographic registered RGB name, first count",
                  count=len(selected), first_image=selected[0]["name"], last_image=selected[-1]["name"],
                  selected_names=[image["name"] for image in selected], trusted_depth_count=len(depth_files),
                  used_depth_inputs=[dict(path=str(path), sha256=digest(path)) for path in depth_files],
                  depth_diagnostics=depth_diagnostics, minimum_distinct_selected_track_images=args.min_selected_track,
                  retained_ba_point_count=len(kept), retained_reciprocal_observation_count=reciprocal_count,
                  source_selected_nonreciprocal_tracks_dropped=bad_reciprocal,
                  no_unselected_image_records_or_tracks=True, source_model_total_images=len(images),
                  source_model_total_points=len(points), source_model_sha256={str(path): digest(path) for path in assets},
                  source_camera_calibration_scope="upstream FFBA used full capture; subset contains only first60 records/tracks",
                  no_generated_rgb_or_depth=True, no_source_asset_modified=True,
                  scale_factor=1., upright=orientation, transform_file="world_transform.json",
                  numeric_validation=dict(max_projection_change_pixels=max_projection, max_camera_xyz_change=max_camera,
                                          rigid_determinant=float(np.linalg.det(transform[:3, :3])),
                                          inverse_error=float(np.max(np.abs(transform @ inverse - np.eye(4)))),
                                          checked_projection_count=len(projection_error), checked_rgb_dimensions=dimensions),
                  retained_ba_bounds_genrecon=dict(min=points_new.min(0).tolist(), max=points_new.max(0).tolist(),
                                                 percentile005_995=np.percentile(points_new, [.5, 99.5], axis=0).tolist()),
                  source_rgb=selected_sources,
                  files={str(path.relative_to(output)): dict(sha256=digest(path), size_bytes=path.stat().st_size)
                         for path in [colmap / "cameras.txt", colmap / "images.txt", colmap / "points3D.txt", output / "world_transform.json"]},
                  preparation_script_sha256=digest(Path(__file__).resolve()))
    (output / "subset_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "count", "first_image", "last_image", "trusted_depth_count",
                                                  "retained_ba_point_count", "retained_reciprocal_observation_count", "numeric_validation")}, indent=2))


if __name__ == "__main__":
    main()
