#!/usr/bin/env python3
"""Bounded-memory CPU inspection for GenRecon binary triangle PLYs.

All source triangles are retained. Component clustering is explicitly deferred.
Vertex records are transformed in blocks and the original face bytes are copied
unchanged. Raycasting uses the complete raw mesh and half-integer COLMAP rays.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


SCALARS = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
           "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
           "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
           "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8"}


def sha256(path, offset=0):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        for block in iter(lambda: handle.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def map_ply(path):
    header_lines, vertex_props, face_props = [], [], []
    current, vertex_count, face_count = None, None, None
    with Path(path).open("rb") as handle:
        for _ in range(1000):
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError("Incomplete PLY header")
            header_lines.append(raw_line)
            fields = raw_line.decode("ascii").strip().split()
            if not fields:
                continue
            if fields[0] == "format" and fields[1] != "binary_little_endian":
                raise ValueError("Large inspector requires binary_little_endian PLY")
            if fields[0] == "element":
                current = fields[1]
                if current == "vertex":
                    vertex_count = int(fields[2])
                elif current == "face":
                    face_count = int(fields[2])
                else:
                    raise ValueError("Unknown PLY element; use the generic inspector")
            elif fields[0] == "property":
                (vertex_props if current == "vertex" else face_props).append(fields[1:])
            elif fields[0] == "end_header":
                break
        else:
            raise ValueError("Unreasonably long PLY header")
    header = b"".join(header_lines)
    if vertex_count is None or face_count is None or vertex_count <= 0 or face_count <= 0:
        raise ValueError("Expected a nonempty vertex/face mesh")
    if any(len(prop) != 2 or prop[0] not in SCALARS for prop in vertex_props):
        raise ValueError("Only scalar vertex properties are supported")
    if face_props not in ([["list", "uchar", "int", "vertex_indices"]],
                          [["list", "uchar", "int", "vertex_index"]]):
        raise ValueError("Expected only uchar-count/int32 triangle face lists")
    dtype = np.dtype([(prop[1], SCALARS[prop[0]]) for prop in vertex_props])
    if not all(key in dtype.names for key in ("x", "y", "z")):
        raise ValueError("Vertex coordinates missing")
    vertex_offset = len(header)
    face_offset = vertex_offset + vertex_count * dtype.itemsize
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    expected_size = face_offset + face_count * face_dtype.itemsize
    if expected_size != Path(path).stat().st_size:
        raise ValueError("PLY length disagrees with fixed triangle records; no variable polygons are accepted")
    return dict(header=header, vertex_count=vertex_count, face_count=face_count,
                vertices=np.memmap(path, dtype=dtype, mode="r", offset=vertex_offset, shape=(vertex_count,)),
                faces=np.memmap(path, dtype=face_dtype, mode="r", offset=face_offset, shape=(face_count,)),
                face_offset=face_offset, vertex_offset=vertex_offset)


def xyz(records, dtype=np.float64):
    return np.column_stack([records[key] for key in ("x", "y", "z")]).astype(dtype, copy=False)


def validate_and_export(raw, source, inverse, target):
    n_vertices, n_faces = source["vertex_count"], source["face_count"]
    vertices = np.empty((n_vertices, 3), dtype=np.float32)
    referenced = np.zeros(n_vertices, dtype=bool)
    source_bbox = [np.full(3, np.inf), np.full(3, -np.inf)]
    restored_bbox = [np.full(3, np.inf), np.full(3, -np.inf)]
    normal_keys = ("nx", "ny", "nz")
    has_normals = all(key in source["vertices"].dtype.names for key in normal_keys)
    with target.open("wb") as out:
        out.write(source["header"])
        for start in range(0, n_vertices, 500_000):
            stop = min(start + 500_000, n_vertices)
            records = source["vertices"][start:stop]
            for key in records.dtype.names:
                if records[key].dtype.kind in "fc" and not np.isfinite(records[key]).all():
                    raise ValueError(f"Nonfinite vertex property {key}")
            coords = xyz(records)
            vertices[start:stop] = coords
            source_bbox[0] = np.minimum(source_bbox[0], coords.min(0))
            source_bbox[1] = np.maximum(source_bbox[1], coords.max(0))
            restored = coords @ inverse[:3, :3].T + inverse[:3, 3]
            restored_bbox[0] = np.minimum(restored_bbox[0], restored.min(0))
            restored_bbox[1] = np.maximum(restored_bbox[1], restored.max(0))
            changed = records.copy()
            for col, key in enumerate(("x", "y", "z")):
                changed[key] = restored[:, col]
            if has_normals:
                ns = np.column_stack([records[key] for key in normal_keys]) @ inverse[:3, :3].T
                for col, key in enumerate(normal_keys):
                    changed[key] = ns[:, col]
            out.write(changed.tobytes())
        with Path(raw).open("rb") as original:
            original.seek(source["face_offset"])
            for block in iter(lambda: original.read(8 << 20), b""):
                out.write(block)
    exported = map_ply(target)
    max_export_error = 0.
    unchanged_fields = [key for key in source["vertices"].dtype.names
                        if key not in ("x", "y", "z") and not (has_normals and key in normal_keys)]
    for start in range(0, n_vertices, 500_000):
        stop = min(start + 500_000, n_vertices)
        left, right = source["vertices"][start:stop], exported["vertices"][start:stop]
        expected = xyz(left) @ inverse[:3, :3].T + inverse[:3, 3]
        max_export_error = max(max_export_error, float(np.max(np.abs(xyz(right) - expected))))
        for key in unchanged_fields:
            if not np.array_equal(left[key], right[key]):
                raise ValueError(f"Inverse export modified {key}")
    source_face_hash = sha256(raw, source["face_offset"])
    if source_face_hash != sha256(target, exported["face_offset"]):
        raise ValueError("Face records changed during inverse export")
    max_index, min_index, degenerate, repeated, area = -1, n_vertices, 0, 0, 0.
    faces = np.empty((n_faces, 3), dtype=np.uint32)
    for start in range(0, n_faces, 250_000):
        stop = min(start + 250_000, n_faces)
        records = source["faces"][start:stop]
        if not np.all(records["count"] == 3):
            raise ValueError("Non-triangle face in fixed record PLY")
        indices = records["indices"]
        smallest, largest = int(indices.min()), int(indices.max())
        if smallest < 0 or largest >= n_vertices:
            raise ValueError("Face index outside vertex range")
        min_index, max_index = min(min_index, smallest), max(max_index, largest)
        faces[start:stop] = indices
        referenced[indices] = True
        repeated += int(((indices[:, 0] == indices[:, 1]) | (indices[:, 0] == indices[:, 2]) |
                         (indices[:, 1] == indices[:, 2])).sum())
        edges1 = vertices[indices[:, 1]].astype(np.float64) - vertices[indices[:, 0]]
        edges2 = vertices[indices[:, 2]].astype(np.float64) - vertices[indices[:, 0]]
        double_area = np.linalg.norm(np.cross(edges1, edges2), axis=1)
        degenerate += int((double_area <= 1e-14).sum())
        area += float(double_area.sum() * .5)
    report = dict(vertex_count=n_vertices, triangle_count=n_faces, finite_vertices_and_numeric_vertex_attributes=True,
                  all_triangle_records_count_3=True, in_range_face_indices=True, min_face_index=min_index,
                  max_face_index=max_index, degenerate_triangles=degenerate,
                  repeated_vertex_index_triangles=repeated, unreferenced_vertices=int((~referenced).sum()),
                  total_area_scene_units_squared=area,
                  bbox_zup=dict(min=source_bbox[0].tolist(), max=source_bbox[1].tolist()),
                  bbox_ffba_original=dict(min=restored_bbox[0].tolist(), max=restored_bbox[1].tolist()),
                  source_face_bytes_sha256=source_face_hash, inverse_export_max_coordinate_roundtrip_error=max_export_error,
                  all_noncoordinate_pbr_properties_preserved=True, all_raw_face_records_byte_identical=True,
                  triangle_edge_connected_component_count=None,
                  component_analysis="Deferred to a separate optional heavy audit; no connected component filtering performed",
                  mesh_decimated_or_filtered=False)
    return vertices, faces, report


def cast_view(scene, image, camera, width, threads):
    width = min(width, camera["width"])
    height = int(round(camera["height"] * width / camera["width"]))
    K = np.diag([width / camera["width"], height / camera["height"], 1.]) @ camera["K"]
    yy, xx = np.meshgrid(np.arange(height) + .5, np.arange(width) + .5, indexing="ij")
    camera_rays = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)], axis=-1)
    world_rays = camera_rays @ image["R"]
    center = -image["R"].T @ image["tvec"]
    rays = np.concatenate([np.broadcast_to(center, world_rays.shape), world_rays], axis=-1)
    cast = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32), device=o3d.core.Device("CPU:0")), nthreads=threads)
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
    for flag in ("result-dir", "data-dir", "source-dataset", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--view-count", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--contact-cell-width", type=int, default=240)
    parser.add_argument("--generic-inspector", type=Path, default=Path(__file__).resolve().with_name("inspect_ffba_result.py"))
    args = parser.parse_args()
    if min(args.width, args.view_count, args.threads, args.contact_cell_width) <= 0:
        raise ValueError("Preview dimensions/count/threads must be positive")
    start = time.perf_counter()
    raw, output = args.result_dir.resolve() / "mesh.ply", args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw_hash = sha256(raw)
    mapped = map_ply(raw)
    transform_file = args.data_dir / "world_transform.json"
    record = json.loads(transform_file.read_text())
    inverse, forward = np.array(record["genrecon_to_original"]), np.array(record["original_to_genrecon"])
    if not np.allclose(forward @ inverse, np.eye(4), atol=1e-9) or not np.allclose(inverse[:3, :3].T @ inverse[:3, :3], np.eye(3), atol=1e-9):
        raise ValueError("Expected consistent rigid transform")
    restored = output / "mesh_ffba_original.ply"
    vertices, faces, checks = validate_and_export(raw, mapped, inverse, restored)
    print(json.dumps({"stage": "streaming_validate_export", "seconds": time.perf_counter() - start,
                      "vertices": len(vertices), "faces": len(faces)}), flush=True)
    export_done = time.perf_counter()
    spec = importlib.util.spec_from_file_location("generic_inspection", args.generic_inspector)
    generic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generic)
    cameras, available = generic.load_cameras(args.data_dir)
    chosen = np.linspace(0, len(available) - 1, min(args.view_count, len(available))).round().astype(int)
    selected = [available[i] for i in chosen]
    scene = o3d.t.geometry.RaycastingScene(nthreads=args.threads)
    scene.add_triangles(o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CPU:0")),
                        o3d.core.Tensor(faces, dtype=o3d.core.Dtype.UInt32, device=o3d.core.Device("CPU:0")))
    del vertices, faces
    previews = []
    for image in selected:
        view = cast_view(scene, image, cameras[image["camera_id"]], args.width, args.threads)
        rgb_path = args.source_dataset / "images" / image["name"]
        if not rgb_path.is_file():
            rgb_path = args.data_dir / "rgb" / image["name"]
        with Image.open(rgb_path) as rgb:
            camera = cameras[image["camera_id"]]
            if rgb.size != (camera["width"], camera["height"]):
                raise ValueError("Source RGB dimensions disagree with camera K")
            view["rgb"] = np.asarray(rgb.convert("RGB").resize((view["width"], view["height"]), Image.Resampling.LANCZOS))
        view["image"] = image
        previews.append(view)
        print(json.dumps({"stage": "cpu_raycast", "image": image["name"], "hit_fraction": float(view["valid"].mean())}), flush=True)
    renders_done = time.perf_counter()
    depth_arrays = [v["depth"][v["valid"]] for v in previews if v["valid"].any()]
    lo, hi = np.percentile(np.concatenate(depth_arrays), [2, 98]) if depth_arrays else (0., 1.)
    if hi <= lo:
        hi = lo + max(1e-6, abs(lo) * .01)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    labels = ["Real RGB", "Gray mesh geometry", "Camera-Z (scene units)", "Camera-space normal"]
    rows, metrics = [], []
    for view in previews:
        image = view["image"]
        vd = output / "views" / Path(image["name"]).stem
        vd.mkdir(parents=True)
        color = (colormaps["turbo"](np.clip((view["depth"] - lo) / (hi - lo), 0, 1))[..., :3] * 255).astype(np.uint8)
        color[~view["valid"]] = [24, 28, 34]
        arrays = [view["rgb"], view["gray"], color, view["normals"]]
        for filename, value in zip(("real_rgb.png", "gray_geometry.png", "depth.png", "normal.png"), arrays):
            Image.fromarray(value).save(vd / filename)
        for filename, value in (("camera_z.npy", view["depth"]), ("ray_hit_mask.npy", view["valid"]),
                                ("camera_normals.npy", view["camera_normals"])):
            np.save(vd / filename, value)
        w, h = view["width"], view["height"]
        row = Image.new("RGB", (4 * w, h + 72), "#11151a")
        draw = ImageDraw.Draw(row)
        for col, (label, value) in enumerate(zip(labels, arrays)):
            row.paste(Image.fromarray(value), (col * w, 40))
            draw.text((col * w + 7, 7), label, fill="white")
            draw.text((col * w + 7, 23), image["name"], fill="#b5c1cf")
        draw.text((7, h + 49), f"All raw triangles, geometry only. Ray hits {view['valid'].mean():.1%}; undefined rays do not prove holes. Z colors {lo:.4g}..{hi:.4g} scene units.", fill="#c8d1dc")
        row.save(vd / "preview.jpg", quality=92)
        cw = args.contact_cell_width
        rows.append(row.resize((4 * cw, round(row.height * cw / w)), Image.Resampling.LANCZOS))
        metrics.append(dict(image_name=image["name"], image_id=image["id"], width=w, height=h,
                            K_scaled=view["K"].tolist(), ray_hit_fraction=float(view["valid"].mean()),
                            half_integer_pixel_centres=True,
                            camera_z_p05_p50_p95_scene_units=(np.percentile(view["depth"][view["valid"]], [5, 50, 95]).tolist()
                                                                         if view["valid"].any() else None),
                            preview=str(vd / "preview.jpg")))
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "#11151a")
    yy = 0
    for row in rows:
        sheet.paste(row, (0, yy)); yy += row.height
    sheet.save(output / "geometry_contact_sheet.jpg", quality=92)
    if sha256(raw) != raw_hash:
        raise ValueError("Raw mesh changed during CPU inspection")
    report = dict(status="cpu_mesh_inspection_complete", inspector_variant="streaming_binary_ply_complete_geometry_no_component_clustering",
                  raw_mesh=str(raw), raw_sha256=raw_hash, raw_preserved=True,
                  ffba_mesh=str(restored), ffba_mesh_sha256=sha256(restored), mesh_checks=checks,
                  previews=metrics, contact_sheet=str(output / "geometry_contact_sheet.jpg"),
                  depth_color_range_scene_units=[float(lo), float(hi)], thread_limit=args.threads,
                  preview_protocol="Open3D CPU with every raw triangle; scaled COLMAP K; x+.5/y+.5 pixel centres; camera ray z=1 so t_hit=camera-Z; geometry-only headlight shading",
                  interpretation="Ray hit fraction is model intersection coverage, not true geometry coverage. Undefined rays alone do not prove scene holes. Connected components were not computed or filtered.",
                  coordinate_transform_sha256=sha256(transform_file), inspection_script_sha256=sha256(Path(__file__)),
                  generic_camera_loader_sha256=sha256(args.generic_inspector),
                  elapsed_seconds=dict(streaming_validation_inverse_export=export_done-start,
                                       raycast_scene_and_selected_views=renders_done-export_done,
                                       total=time.perf_counter()-start),
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2))
    (output / "inspection_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "ffba_mesh", "contact_sheet", "elapsed_seconds", "peak_rss_gib")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
