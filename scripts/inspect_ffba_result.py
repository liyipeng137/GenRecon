#!/usr/bin/env python3
"""Export a GenRecon mesh to FFBA coordinates and make CPU geometry previews.

The preview casts rays using scaled COLMAP K and half-integer pixel centres.
Missing ray hits are undefined model intersections, not a proof of scene holes.
No generated texture or photometric quality metric is used for the gray mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_cameras(data_dir):
    cameras, images = {}, []
    with (data_dir / "colmap" / "cameras.txt").open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            tokens = line.split()
            cid, model, width, height = int(tokens[0]), tokens[1], int(tokens[2]), int(tokens[3])
            params = np.array(tokens[4:], dtype=float)
            if model == "SIMPLE_PINHOLE":
                focal, cx, cy = params
                fx = fy = focal
            elif model == "PINHOLE":
                fx, fy, cx, cy = params
            else:
                raise ValueError(f"Preview requires undistorted pinhole K, got {model}")
            cameras[cid] = dict(width=width, height=height,
                                K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=float))
    # A standard images.txt has one header followed by one POINTS2D line.
    with (data_dir / "colmap" / "images.txt").open() as handle:
        while line := handle.readline():
            if not line.strip() or line.startswith("#"):
                continue
            tokens = line.split(maxsplit=9)
            if len(tokens) != 10:
                raise ValueError("Malformed COLMAP image header")
            qvec = np.array(tokens[1:5], dtype=float)
            images.append(dict(id=int(tokens[0]), name=tokens[9].strip(), camera_id=int(tokens[8]),
                               R=Rotation.from_quat(qvec[[1, 2, 3, 0]]).as_matrix(),
                               tvec=np.array(tokens[5:8], dtype=float)))
            if not handle.readline():
                raise ValueError("Missing COLMAP POINTS2D line")
    return cameras, sorted(images, key=lambda image: image["name"])


def xyz_values(data):
    return np.column_stack([data[key] for key in ("x", "y", "z")]).astype(np.float64)


def triangles_from_ply(ply):
    if "face" not in ply:
        raise ValueError("PLY has no faces")
    names = ply["face"].data.dtype.names
    key = "vertex_indices" if "vertex_indices" in names else "vertex_index" if "vertex_index" in names else None
    if key is None:
        raise ValueError("PLY face indices missing")
    values = ply["face"].data[key]
    sizes = np.fromiter((len(row) for row in values), dtype=np.int64, count=len(values))
    if not len(sizes) or not np.all(sizes == 3):
        raise ValueError("GenRecon inspection currently requires a nonempty triangle mesh")
    return np.stack(values).astype(np.int64), key


def transform_ply(ply, inverse, output_path):
    elements = []
    transformed_normals = []
    for element in ply.elements:
        values = element.data.copy()
        keys = values.dtype.names or ()
        if element.name == "vertex":
            xyz = xyz_values(values)
            transformed = xyz @ inverse[:3, :3].T + inverse[:3, 3]
            for column, key in enumerate(("x", "y", "z")):
                values[key] = transformed[:, column]
        if all(key in keys for key in ("nx", "ny", "nz")):
            normals = np.column_stack([values[key] for key in ("nx", "ny", "nz")]).astype(float)
            normals = normals @ inverse[:3, :3].T
            for column, key in enumerate(("nx", "ny", "nz")):
                values[key] = normals[:, column]
            transformed_normals.append(element.name)
        elements.append(PlyElement(element.name, element.properties, values.size,
                                   comments=list(element.comments)))
        elements[-1].data = values
    result = PlyData(elements, text=ply.text, byte_order=ply.byte_order,
                     comments=[*ply.comments, "Rigid inverse transform to original FFBA world; raw mesh preserved"],
                     obj_info=list(ply.obj_info))
    result.write(str(output_path))
    readback = PlyData.read(str(output_path))
    expected = xyz_values(ply["vertex"].data) @ inverse[:3, :3].T + inverse[:3, 3]
    error = float(np.max(np.abs(xyz_values(readback["vertex"].data) - expected)))
    scale = max(1., float(np.max(np.abs(expected))))
    if error > scale * 2e-6:
        raise ValueError("Export coordinate roundtrip error is unexpectedly large")
    for original, exported in zip(ply.elements, readback.elements):
        if original.name != exported.name or original.data.dtype.names != exported.data.dtype.names:
            raise ValueError("PLY export changed element/property layout")
        for key in original.data.dtype.names:
            if original.name == "vertex" and key in ("x", "y", "z"):
                continue
            if original.name in transformed_normals and key in ("nx", "ny", "nz"):
                continue
            left, right = original.data[key], exported.data[key]
            if left.dtype.kind == "O":
                same = all(np.array_equal(a, b) for a, b in zip(left, right))
            else:
                same = np.array_equal(left, right)
            if not same:
                raise ValueError(f"PLY export changed attribute {original.name}.{key}")
    return error


def mesh_checks(ply, vertices, triangles):
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite vertex positions")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("Face indices out of vertex range")
    numeric_finite = {}
    for element in ply.elements:
        for key in element.data.dtype.names:
            field = element.data[key]
            if field.dtype.kind in "fc":
                numeric_finite[f"{element.name}.{key}"] = bool(np.isfinite(field).all())
    if not all(numeric_finite.values()):
        raise ValueError("Nonfinite PLY numeric property")
    v1, v2 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]], vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    double_area = np.linalg.norm(np.cross(v1, v2), axis=1)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    labels, counts, areas = mesh.cluster_connected_triangles()
    counts, areas = np.asarray(counts), np.asarray(areas)
    order = np.argsort(counts)[::-1]
    info = dict(vertex_count=len(vertices), triangle_count=len(triangles),
                finite_vertices=True, in_range_face_indices=True, finite_numeric_attributes=numeric_finite,
                bbox_zup=dict(min=vertices.min(0).tolist(), max=vertices.max(0).tolist()),
                degenerate_triangles=int((double_area <= 1e-14).sum()),
                repeated_vertex_index_triangles=int(((triangles[:, 0] == triangles[:, 1]) |
                                                     (triangles[:, 0] == triangles[:, 2]) |
                                                     (triangles[:, 1] == triangles[:, 2])).sum()),
                unreferenced_vertices=int(len(vertices) - len(np.unique(triangles))),
                triangle_edge_connected_component_count=len(counts),
                largest_components=[dict(id=int(i), triangles=int(counts[i]), area_scene_units_squared=float(areas[i]))
                                    for i in order[:20]],
                raw_filtered_or_smoothed=False,
                ply_properties={element.name: list(element.data.dtype.names) for element in ply.elements})
    return mesh, info


def cast_view(scene, image, camera, width):
    width = min(width, camera["width"])
    height = int(round(camera["height"] * width / camera["width"]))
    K = np.diag([width / camera["width"], height / camera["height"], 1.]) @ camera["K"]
    yy, xx = np.meshgrid(np.arange(height) + .5, np.arange(width) + .5, indexing="ij")
    camera_rays = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)], axis=-1)
    world_rays = camera_rays @ image["R"]
    center = -image["R"].T @ image["tvec"]
    rays = np.concatenate([np.broadcast_to(center, world_rays.shape), world_rays], axis=-1)
    cast = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32), device=o3d.core.Device("CPU:0")))
    # Rays have camera-z component 1, so t_hit itself is camera-Z, not range.
    depth = cast["t_hit"].numpy().astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    normals = cast["primitive_normals"].numpy().astype(np.float32)
    camera_normals = normals @ image["R"].T
    direction = world_rays / np.linalg.norm(world_rays, axis=-1, keepdims=True)
    diffuse = np.abs(np.einsum("ijk,ijk->ij", normals, direction))
    shade = np.clip(.22 + .70 * diffuse, 0, 1)
    gray = np.broadcast_to((shade[..., None] * 255).astype(np.uint8), (*shade.shape, 3)).copy()
    gray[~valid] = [24, 28, 34]
    normal_rgb = np.clip((camera_normals + 1) * 127.5, 0, 255).astype(np.uint8)
    normal_rgb[~valid] = [24, 28, 34]
    depth[~valid] = 0
    return dict(width=width, height=height, K=K, gray=gray, normals=normal_rgb,
                depth=depth, valid=valid, camera_normals=camera_normals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--view-count", type=int, default=8)
    parser.add_argument("--views", help="Optional comma-separated selected RGB names")
    parser.add_argument("--contact-cell-width", type=int, default=240)
    args = parser.parse_args()
    raw = args.result_dir.resolve() / "mesh.ply"
    if not raw.is_file():
        raise FileNotFoundError(raw)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty inspection output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw_sha = digest(raw)
    ply = PlyData.read(str(raw))
    vertices = xyz_values(ply["vertex"].data)
    triangles, _ = triangles_from_ply(ply)
    legacy, checks = mesh_checks(ply, vertices, triangles)
    transform_path = args.data_dir / "world_transform.json"
    transformation = json.loads(transform_path.read_text())
    inverse = np.array(transformation["genrecon_to_original"], dtype=float)
    forward = np.array(transformation["original_to_genrecon"], dtype=float)
    if not np.allclose(forward @ inverse, np.eye(4), atol=1e-9) or not np.allclose(inverse[:3, :3].T @ inverse[:3, :3], np.eye(3), atol=1e-9):
        raise ValueError("Expected a consistent rigid FFBA/world transform")
    restored = output / "mesh_ffba_original.ply"
    export_error = transform_ply(ply, inverse, restored)
    cameras, available = load_cameras(args.data_dir)
    if args.views:
        by_name = {image["name"]: image for image in available}
        selected = [by_name[name.strip()] for name in args.views.split(",")]
    else:
        indices = np.linspace(0, len(available) - 1, min(args.view_count, len(available))).round().astype(int)
        selected = [available[index] for index in indices]
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    previews = []
    for image in selected:
        view = cast_view(scene, image, cameras[image["camera_id"]], args.width)
        rgb_path = args.source_dataset / "images" / image["name"]
        if not rgb_path.is_file():
            rgb_path = args.data_dir / "rgb" / image["name"]
        with Image.open(rgb_path) as rgb:
            camera = cameras[image["camera_id"]]
            if rgb.size != (camera["width"], camera["height"]):
                raise ValueError(f"Source RGB size differs from subset K: {image['name']}")
            view["rgb"] = np.asarray(rgb.convert("RGB").resize((view["width"], view["height"]), Image.Resampling.LANCZOS))
        view["image"] = image
        previews.append(view)
    nonempty = [view["depth"][view["valid"]] for view in previews if view["valid"].any()]
    if nonempty:
        depth_min, depth_max = np.percentile(np.concatenate(nonempty), [2, 98])
        if depth_max <= depth_min:
            depth_max = depth_min + max(1e-6, abs(depth_min) * .01)
    else:
        depth_min, depth_max = 0., 1.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    contact_rows, metrics = [], []
    labels = ["Real RGB", "Gray mesh geometry", "Camera-Z (scene units)", "Camera-space normal"]
    for view in previews:
        image = view["image"]
        prefix = Path(image["name"]).stem
        view_dir = output / "views" / prefix
        view_dir.mkdir(parents=True)
        depth_normalized = np.clip((view["depth"] - depth_min) / (depth_max - depth_min), 0, 1)
        depth_rgb = (colormaps["turbo"](depth_normalized)[..., :3] * 255).astype(np.uint8)
        depth_rgb[~view["valid"]] = [24, 28, 34]
        for filename, value in (("real_rgb.png", view["rgb"]), ("gray_geometry.png", view["gray"]),
                                ("depth.png", depth_rgb), ("normal.png", view["normals"])):
            Image.fromarray(value).save(view_dir / filename)
        np.save(view_dir / "camera_z.npy", view["depth"])
        np.save(view_dir / "ray_hit_mask.npy", view["valid"])
        np.save(view_dir / "camera_normals.npy", view["camera_normals"])
        width, height, label_height, footer_height = view["width"], view["height"], 40, 32
        row = Image.new("RGB", (width * 4, height + label_height + footer_height), "#11151a")
        draw = ImageDraw.Draw(row)
        for col, (label, value) in enumerate(zip(labels, [view["rgb"], view["gray"], depth_rgb, view["normals"]])):
            row.paste(Image.fromarray(value), (col * width, label_height))
            draw.text((col * width + 7, 7), label, fill="white")
            draw.text((col * width + 7, 23), image["name"], fill="#b5c1cf")
        hit_fraction = float(view["valid"].mean())
        note = f"Geometry preview. Ray hits: {hit_fraction:.1%}; undefined rays do not establish real scene holes. Z colors {depth_min:.4g}..{depth_max:.4g} scene units."
        draw.text((7, label_height + height + 9), note, fill="#c8d1dc")
        row.save(view_dir / "preview.jpg", quality=92)
        cell_width = args.contact_cell_width
        contact_rows.append(row.resize((cell_width * 4, round(row.height * cell_width / width)), Image.Resampling.LANCZOS))
        metrics.append(dict(image_name=image["name"], image_id=image["id"], width=width, height=height,
                            K_scaled=view["K"].tolist(), half_integer_pixel_centres=True,
                            ray_hit_fraction=hit_fraction,
                            camera_z_p05_p50_p95_scene_units=(np.percentile(view["depth"][view["valid"]], [5, 50, 95]).tolist()
                                                                         if view["valid"].any() else None),
                            preview=str((view_dir / "preview.jpg").resolve())))
    contact = Image.new("RGB", (contact_rows[0].width, sum(row.height for row in contact_rows)), "#11151a")
    offset = 0
    for row in contact_rows:
        contact.paste(row, (0, offset)); offset += row.height
    contact_path = output / "geometry_contact_sheet.jpg"
    contact.save(contact_path, quality=92)
    restored_xyz = xyz_values(PlyData.read(str(restored))["vertex"].data)
    if digest(raw) != raw_sha:
        raise ValueError("Raw GenRecon mesh changed during CPU inspection")
    report = dict(status="cpu_mesh_inspection_complete", raw_mesh=str(raw), raw_sha256=raw_sha,
                  raw_preserved=True, raw_mesh_is_genrecon_zup=True,
                  ffba_mesh=str(restored), ffba_mesh_sha256=digest(restored),
                  ffba_bbox=dict(min=restored_xyz.min(0).tolist(), max=restored_xyz.max(0).tolist()),
                  inverse_export_max_coordinate_roundtrip_error=export_error,
                  all_noncoordinate_ply_properties_preserved=True, transform_sha256=digest(transform_path),
                  source_dataset=str(args.source_dataset.resolve()), data_dir=str(args.data_dir.resolve()),
                  mesh_checks=checks, previews=metrics, contact_sheet=str(contact_path),
                  depth_color_range_scene_units=[float(depth_min), float(depth_max)],
                  preview_protocol="Open3D CPU double-sided first triangle hit, no mesh filtering; scaled source K; pixel centres x+.5/y+.5; t_hit is camera-Z because camera ray z=1",
                  interpretation="Gray is geometry-only headlight shading. Ray-hit fraction is model intersection coverage, not true geometry coverage; missing hits do not alone prove holes.",
                  inspection_script_sha256=digest(Path(__file__).resolve()))
    (output / "inspection_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "raw_mesh", "ffba_mesh", "contact_sheet")}, indent=2))


if __name__ == "__main__":
    main()
