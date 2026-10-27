#!/usr/bin/env python3
"""CPU baseline for independent GenRecon meshes in one fixed global frame.

No ICP, pose fitting, smoothing, hole filling or generated texture is applied.
The global chunk partition selects faces before concatenation. A small optional
quantized weld can remove coincident vertices; it cannot reconcile inconsistent
surfaces and this process does not guarantee a watertight mesh.

Manifest schema (relative paths resolve against the manifest directory)::

  {"coordinate_frame": "global_zup", "data_dir": "/.../data",
   "chunks": [{"id": 0, "center": [0,0,0], "size": 1.0}, ...],
   "groups": [{"id": "g00", "mesh_path": "g00/mesh.ply",
                "chunk_ids": [0,1], "core_chunk_ids": [0],
                "selected_image_names": ["frame_000000.png", ...],
                "mesh_is_core_owned": false,
                "vertex_owner_points_path": "g00/vertex_owner_points.npy",
                "context_mesh_path": "g00/mesh_context.ply"}, ...]}

Optional vertex_owner_points_path stores fixed lattice vertex anchors [V,3], or
face_owner_points_path stores fixed face barycentres [F,3], in this global frame.
Otherwise face positions are used (and reported as a less
stable fallback). mesh_is_core_owned accepts meshes already filtered with the
model's fixed lattice rule. For meaningful overlap comparisons, context_mesh_path
must then retain generated context faces; core-only meshes cannot provide this.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np


BATCH = 250_000


def import_helper(name):
    path = Path(__file__).resolve().with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_path(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def normalize_manifest(manifest, base):
    if manifest.get("coordinate_frame", "global_zup") != "global_zup":
        raise ValueError("All meshes/cameras/chunks must use the same global_zup frame")
    chunks = manifest.get("chunks")
    if chunks is None:
        chunk_path = manifest.get("chunk_transforms_path", manifest.get("chunks_path"))
        if not chunk_path:
            raise ValueError("Manifest must contain chunks or chunk_transforms_path")
        chunks = json.loads(resolve_path(base, chunk_path).read_text())["chunks"]
    normalized = []
    for c in chunks:
        cid = int(c.get("id", c.get("index")))
        center = np.asarray(c.get("center", c.get("crop_center")), dtype=float)
        size = c.get("size", c.get("chunk_size"))
        if size is None:
            matrix = np.asarray(c["M_chunk_to_original"], dtype=float)
            linear = matrix[:3, :3]
            scale = np.diag(linear)
            if not np.allclose(linear, np.diag(scale), atol=1e-8):
                raise ValueError("Only axis-aligned global chunk cubes are supported")
            size = scale
        size = np.broadcast_to(np.asarray(size, dtype=float), (3,)).copy()
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("Invalid chunk center/size")
        normalized.append(dict(id=cid, center=center, size=size))
    normalized.sort(key=lambda c: c["id"])
    ids = [c["id"] for c in normalized]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Chunk ids must be unique and nonempty")
    groups = []
    owner = {}
    for i, raw in enumerate(manifest["groups"]):
        g = dict(raw)
        g["id"] = str(g.get("id", f"group_{i:03d}"))
        if not g["id"] or Path(g["id"]).name != g["id"]:
            raise ValueError("Group ids must be simple nonempty filenames")
        g["chunk_ids"] = [int(v) for v in g["chunk_ids"]]
        g["core_chunk_ids"] = [int(v) for v in g.get("core_chunk_ids", g["chunk_ids"])]
        if not set(g["core_chunk_ids"]).issubset(g["chunk_ids"]) or not set(g["chunk_ids"]).issubset(ids):
            raise ValueError("Core chunks must be a subset of valid context chunks")
        for cid in g["core_chunk_ids"]:
            if cid in owner:
                raise ValueError(f"Core chunk {cid} is assigned more than once")
            owner[cid] = g["id"]
        for key in ("mesh_path", "context_mesh_path", "face_owner_points_path", "vertex_owner_points_path"):
            if key in g:
                g[key] = resolve_path(base, g[key])
        if not g["mesh_path"].is_file():
            raise FileNotFoundError(g["mesh_path"])
        groups.append(g)
    if len({g["id"] for g in groups}) != len(groups):
        raise ValueError("Group ids must be unique")
    if set(owner) != set(ids):
        raise ValueError("Core groups must cover every global chunk exactly once")
    result = dict(manifest, chunks=normalized, groups=groups)
    for key in ("data_dir", "source_dataset"):
        if key in result:
            result[key] = resolve_path(base, result[key])
    return result


def exclusive_chunk_owner(points, chunks, batch_size=BATCH, tie_tolerance=1e-7):
    """Nearest centre among *covering* cubes; low chunk id wins exact ties.

    Distances within tie_tolerance scene units choose the lower chunk id.
    Cubes are half-open, matching the pipeline ownership rule. No point outside
    the global union is silently assigned to a nearest noncovering chunk.
    """
    p = np.asarray(points)
    result = np.full(len(p), -1, dtype=np.int64)
    ordered = sorted(chunks, key=lambda c: c["id"])
    for start in range(0, len(p), batch_size):
        block = p[start:start + batch_size]
        best = np.full(len(block), np.inf)
        local = result[start:start + len(block)]
        for c in ordered:
            delta = block - c["center"]
            covered = np.all((delta >= -c["size"] / 2) & (delta < c["size"] / 2), axis=1)
            distance = np.linalg.norm(delta, axis=1)
            take = covered & (distance < best - tie_tolerance)
            best[take] = distance[take]
            local[take] = c["id"]
    return result


def inside_chunks(points, chunks, chunk_ids):
    result = np.zeros(len(points), dtype=bool)
    for c in chunks:
        if c["id"] in chunk_ids:
            delta = points - c["center"]
            result |= np.all((delta >= -c["size"] / 2) & (delta < c["size"] / 2), axis=1)
    return result


def write_ply(path, vertices, faces):
    """Write xyz-only binary triangle PLY without constructing object arrays."""
    if len(vertices) > np.iinfo(np.int32).max:
        raise ValueError("PLY int32 face-index limit exceeded")
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment Independent local generation; common fixed global frame\n"
              f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    with Path(path).open("wb") as out:
        out.write(header.encode("ascii"))
        for start in range(0, len(vertices), BATCH):
            out.write(np.asarray(vertices[start:start + BATCH], dtype="<f4").tobytes())
        for start in range(0, len(faces), BATCH):
            block = faces[start:start + BATCH]
            records = np.empty(len(block), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
            records["count"] = 3
            records["indices"] = block
            out.write(records.tobytes())


def selected_group_arrays(group, chunks, output, mapper):
    """Stream ownership selection, compact references, and spool one group."""
    mapped = mapper.map_ply(group["mesh_path"])
    nv, nf = mapped["vertex_count"], mapped["face_count"]
    referenced = np.zeros(nv, dtype=bool)
    point_path = group.get("face_owner_points_path")
    owner_points = np.load(point_path, mmap_mode="r") if point_path else None
    if owner_points is not None and owner_points.shape != (nf, 3):
        raise ValueError("Fixed-lattice ownership points must have shape [face_count,3]")
    anchor_path = group.get("vertex_owner_points_path")
    vertex_anchors = np.load(anchor_path, mmap_mode="r") if anchor_path else None
    if vertex_anchors is not None and vertex_anchors.shape != (nv, 3):
        raise ValueError("Fixed-lattice vertex anchors must have shape [vertex_count,3]")
    core_owned = bool(group.get("mesh_is_core_owned", False))
    mode = ("already_fixed_lattice_core_owned" if core_owned else
            "fixed_lattice_vertex_anchors" if vertex_anchors is not None else
            "fixed_lattice_points" if owner_points is not None else "predicted_face_centroid_fallback")
    spool = output / (group["id"] + "_faces.bin")
    kept = 0
    bbox_lo, bbox_hi = np.full(3, np.inf), np.full(3, -np.inf)
    with spool.open("wb") as handle:
        for start in range(0, nf, BATCH):
            records = mapped["faces"][start:start + BATCH]
            if not np.all(records["count"] == 3):
                raise ValueError("Non-triangle face")
            face = np.asarray(records["indices"], dtype=np.int64)
            if face.min() < 0 or face.max() >= nv:
                raise ValueError("Face references outside vertex range")
            if core_owned:
                take = np.ones(len(face), dtype=bool)
            else:
                if vertex_anchors is not None:
                    points = np.asarray(vertex_anchors[face], dtype=np.float64).mean(axis=1)
                elif owner_points is not None:
                    points = np.asarray(owner_points[start:start + len(face)])
                else:
                    points = np.stack([mapped["vertices"][key][face].mean(1) for key in ("x", "y", "z")], axis=1)
                if not np.isfinite(points).all():
                    raise ValueError("Nonfinite ownership coordinates")
                take = np.isin(exclusive_chunk_owner(points, chunks), group["core_chunk_ids"])
            selected = face[take]
            referenced[selected] = True
            handle.write(selected.astype("<i8", copy=False).tobytes())
            kept += len(selected)
    if not kept:
        raise ValueError(f"No faces remain for group {group['id']}")
    used = np.flatnonzero(referenced)
    remap = np.full(nv, -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    vertex_file, face_file = output / (group["id"] + "_vertices.npy"), output / (group["id"] + "_faces.npy")
    vertices = np.lib.format.open_memmap(vertex_file, mode="w+", dtype=np.float32, shape=(len(used), 3))
    for start in range(0, len(used), BATCH):
        coords = np.column_stack([mapped["vertices"][key][used[start:start + BATCH]] for key in ("x", "y", "z")])
        if not np.isfinite(coords).all():
            raise ValueError("Nonfinite mesh vertices")
        vertices[start:start + len(coords)] = coords
        bbox_lo, bbox_hi = np.minimum(bbox_lo, coords.min(0)), np.maximum(bbox_hi, coords.max(0))
    faces = np.lib.format.open_memmap(face_file, mode="w+", dtype=np.int64, shape=(kept, 3))
    source = np.memmap(spool, mode="r", dtype="<i8", shape=(kept, 3))
    for start in range(0, kept, BATCH):
        faces[start:start + BATCH] = remap[source[start:start + BATCH]]
    vertices.flush()
    faces.flush()
    del source
    spool.unlink()
    return vertex_file, face_file, dict(id=group["id"], raw_path=str(group["mesh_path"]), raw_sha256=sha256(group["mesh_path"]),
        original_vertices=nv, original_faces=nf, selected_vertices=len(used), selected_faces=kept,
        dropped_faces=nf - kept, ownership_mode=mode, core_chunk_ids=group["core_chunk_ids"],
        bbox=dict(min=bbox_lo.tolist(), max=bbox_hi.tolist()))


def weld_and_clean(vertices, faces, tolerance=0., memory_limit_gib=12.):
    """Deterministic first-vertex quantized weld; no coordinate averaging.

    Positive tolerance is the quantization cell side, not a Euclidean radius.
    Equal-cell vertices can be at most sqrt(3)*tolerance apart. At zero tolerance
    only exactly equal float32 positions weld. The caller can disable all welding
    with a negative value. Original group PLYs are never modified.
    """
    estimated = (len(vertices) * 150 + len(faces) * 100) / 2 ** 30
    info = dict(requested_quantization_cell_side=float(tolerance), estimated_temporary_ram_gib=estimated,
                memory_limit_gib=memory_limit_gib, source_vertex_count=len(vertices), source_triangle_count=len(faces))
    if tolerance < 0 or estimated > memory_limit_gib:
        info.update(applied=False, reason="explicitly_disabled" if tolerance < 0 else "estimated_ram_exceeds_limit",
                    final_vertices=len(vertices), final_faces=len(faces))
        return vertices, faces, info
    if tolerance > 0:
        scaled = np.asarray(vertices, dtype=np.float64) / tolerance
        if not np.isfinite(scaled).all() or np.abs(scaled).max() > np.iinfo(np.int64).max / 2:
            raise ValueError("Weld tolerance cannot safely represent quantized coordinates")
        keys = np.rint(scaled).astype(np.int64)
        del scaled
    else:
        keys = np.asarray(vertices)
    _, representatives, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    del keys
    welded = np.asarray(vertices[representatives], dtype=np.float32)
    remapped = inverse[faces]
    max_shift = 0.
    for start in range(0, len(vertices), BATCH):
        delta = np.asarray(vertices[start:start + BATCH]) - welded[inverse[start:start + BATCH]]
        max_shift = max(max_shift, float(np.linalg.norm(delta, axis=1).max(initial=0.)))
    distinct = (remapped[:, 0] != remapped[:, 1]) & (remapped[:, 1] != remapped[:, 2]) & (remapped[:, 0] != remapped[:, 2])
    nondegenerate = np.zeros(len(remapped), dtype=bool)
    for start in range(0, len(remapped), BATCH):
        f = remapped[start:start + BATCH]
        area2 = np.linalg.norm(np.cross(welded[f[:, 1]] - welded[f[:, 0]], welded[f[:, 2]] - welded[f[:, 0]]), axis=1)
        nondegenerate[start:start + len(f)] = area2 > 1e-14
    valid = distinct & nondegenerate
    clean = remapped[valid]
    _, keep = np.unique(np.sort(clean, axis=1), axis=0, return_index=True)
    clean = clean[np.sort(keep)]
    if not len(clean):
        raise ValueError("No nondegenerate faces remain after weld/cleanup")
    referenced = np.unique(clean)
    compact = np.full(len(welded), -1, dtype=np.int64)
    compact[referenced] = np.arange(len(referenced))
    cleaned_vertices, cleaned_faces = welded[referenced], compact[clean]
    info.update(applied=True, maximum_vertex_displacement=max_shift,
                welded_vertex_count_before_compaction=len(welded), removed_degenerate_faces=int((~valid).sum()),
                removed_duplicate_faces=int(valid.sum() - len(clean)), final_vertices=len(cleaned_vertices),
                final_faces=len(cleaned_faces), coordinate_representative="first_input_vertex; no averaging")
    return cleaned_vertices, cleaned_faces, info


def summarize(values):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    if not a.size:
        return dict(count=0, median=None, p90=None, p95=None, max=None, mean=None)
    return dict(count=int(a.size), median=float(np.median(a)), p90=float(np.quantile(a, .9)),
                p95=float(np.quantile(a, .95)), max=float(a.max()), mean=float(a.mean()))


def compare_raycast_hits(left, right, left_inside, right_inside):
    """Compare common first-hit surfaces only; this is not a GT quality metric."""
    common = left["valid"] & right["valid"] & left_inside & right_inside
    dz = np.abs(left["depth"] - right["depth"])
    relative = dz / np.maximum(np.minimum(left["depth"], right["depth"]), 1e-8)
    dot = np.sum(left["camera_normals"] * right["camera_normals"], axis=-1)
    angle = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))
    disagreement = common & (relative > .02)
    parallel = disagreement & (angle < 15.)
    count = int(common.sum())
    return dict(common_overlap_pixels=count, common_depth_absolute=summarize(dz[common]),
                common_depth_relative=summarize(relative[common]), normal_unsigned_degrees=summarize(angle[common]),
                relative_depth_disagreement_gt_2pct_fraction=float(disagreement.sum() / count) if count else None,
                parallel_depth_disagreement_candidate_fraction=float(parallel.sum() / count) if count else None,
                left_only_hit_pixels=int((left["valid"] & ~right["valid"] & left_inside).sum()),
                right_only_hit_pixels=int((right["valid"] & ~left["valid"] & right_inside).sum())), common, relative, angle


def shared_names(group, images):
    names = group.get("selected_image_names", group.get("image_names"))
    if names is not None:
        return set(names)
    ids = set(group.get("selected_image_ids", group.get("image_ids", [])))
    return {im["name"] for im in images if im["id"] in ids}


def make_scene(path, mapper, threads):
    import open3d as o3d
    mapped = mapper.map_ply(path)
    vertices = np.column_stack([mapped["vertices"][key] for key in ("x", "y", "z")]).astype(np.float32)
    faces = np.asarray(mapped["faces"]["indices"], dtype=np.uint32).copy()
    if not np.isfinite(vertices).all() or faces.max() >= len(vertices):
        raise ValueError("Invalid context mesh for raycast")
    scene = o3d.t.geometry.RaycastingScene(nthreads=threads)
    scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
    return scene


def hit_points(view, image):
    h, w = view["depth"].shape
    yy, xx = np.meshgrid(np.arange(h) + .5, np.arange(w) + .5, indexing="ij")
    K = view["K"]
    rays_camera = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)], -1)
    center = -image["R"].T @ image["tvec"]
    return center + (rays_camera @ image["R"]) * view["depth"][..., None]


def groups_spatially_overlap(a, b, chunks):
    """True only when context cubes share positive volume, not just a face."""
    ca = [c for c in chunks if c["id"] in a["chunk_ids"]]
    cb = [c for c in chunks if c["id"] in b["chunk_ids"]]
    return any(np.all(np.maximum(x["center"] - x["size"] / 2, y["center"] - y["size"] / 2) <
                         np.minimum(x["center"] + x["size"] / 2, y["center"] + y["size"] / 2))
               for x in ca for y in cb)


def overlap_audit(manifest, output, mapper, width=320, max_views=3, threads=4):
    from PIL import Image, ImageDraw
    generic = import_helper("inspect_ffba_result")
    cameras, images = generic.load_cameras(manifest["data_dir"])
    image_by_name = {im["name"]: im for im in images}
    groups, chunks = manifest["groups"], manifest["chunks"]
    reports, previews = [], []
    for ai, a in enumerate(groups):
        for b in groups[ai + 1:]:
            if not groups_spatially_overlap(a, b, chunks):
                continue
            names = sorted(shared_names(a, images) & shared_names(b, images))
            if not names:
                reports.append(dict(groups=[a["id"], b["id"]], status="no_shared_input_camera"))
                continue
            chosen = [names[i] for i in np.unique(np.linspace(0, len(names) - 1, min(max_views, len(names))).round().astype(int))]
            pa, pb = a.get("context_mesh_path", a["mesh_path"]), b.get("context_mesh_path", b["mesh_path"])
            if ((a.get("mesh_is_core_owned") and "context_mesh_path" not in a) or
                    (b.get("mesh_is_core_owned") and "context_mesh_path" not in b)):
                reports.append(dict(groups=[a["id"], b["id"]], status="context_mesh_missing_for_core_owned_mesh", shared_cameras=len(names)))
                continue
            sa, sb = make_scene(pa, mapper, threads), make_scene(pb, mapper, threads)
            pair = dict(groups=[a["id"], b["id"]], status="ok", shared_cameras=len(names), views=[])
            pair_output = output / f"{a['id']}__{b['id']}"
            pair_output.mkdir()
            for name in chosen:
                image = image_by_name[name]
                cam = cameras[image["camera_id"]]
                va, vb = mapper.cast_view(sa, image, cam, width, threads), mapper.cast_view(sb, image, cam, width, threads)
                masks = []
                for view in (va, vb):
                    points = hit_points(view, image).reshape(-1, 3)
                    mask = inside_chunks(points, chunks, a["chunk_ids"]) & inside_chunks(points, chunks, b["chunk_ids"])
                    masks.append(mask.reshape(view["valid"].shape))
                metrics, common, relative, angles = compare_raycast_hits(va, vb, *masks)
                pair["views"].append(dict(image=name, **metrics))
                stem = Path(name).stem
                np.savez_compressed(pair_output / (stem + ".npz"), depth_left=va["depth"], depth_right=vb["depth"],
                                    common_overlap_mask=common, depth_relative=relative, normal_unsigned_degrees=angles)
                heat = np.zeros((*common.shape, 3), dtype=np.uint8)
                heat[:] = [24, 28, 34]
                level = np.minimum(relative / .1, 1.)
                heat[common] = np.stack([255 * level, 210 * (1 - level), np.zeros_like(level)], -1)[common].astype(np.uint8)
                cell_w = width
                panels = [(va["gray"], a["id"]), (vb["gray"], b["id"]), (heat, "overlap depth difference: green 0%, red >=10%")]
                row = Image.new("RGB", (cell_w * 3, common.shape[0] + 38), "#181c22")
                draw = ImageDraw.Draw(row)
                for i, (arr, label) in enumerate(panels):
                    row.paste(Image.fromarray(arr), (i * cell_w, 38))
                    draw.text((i * cell_w + 5, 4), label, fill="white")
                draw.text((5, 20), name, fill="#aaccff")
                row.save(pair_output / (stem + ".jpg"), quality=90)
                previews.append(row)
            reports.append(pair)
            del sa, sb
            gc.collect()
    preview_total = len(previews)
    if previews:
        selected_rows = np.unique(np.linspace(0, len(previews) - 1, min(24, len(previews))).round().astype(int))
        previews = [previews[i] for i in selected_rows]
        contact = Image.new("RGB", (previews[0].width, sum(p.height for p in previews)), "#181c22")
        y = 0
        for row in previews:
            contact.paste(row, (0, y))
            y += row.height
        contact.save(output / "overlap_preview.jpg", quality=90)
    return dict(pairs=reports, contact_sheet_total_available_rows=preview_total,
                contact_sheet_rows=min(24, preview_total), definitions={
        "common_overlap": "Both meshes have a first ray hit; both hits lie in both groups' context chunk unions",
        "relative_depth": "abs(z_left-z_right)/min(z_left,z_right), COLMAP camera-Z",
        "normal_angle": "Unsigned angle; opposite winding is treated as equivalent",
        "parallel_depth_disagreement_candidate": "Relative depth difference >2% and unsigned normals <15 degrees; a possible duplicate sheet, not proof",
        "limitations": "No ground truth, visibility equivalence, watertightness or mesh accuracy claim; occlusion and different surfaces can also disagree"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weld-tolerance", type=float, default=0., help="0 exact positions; positive quantization cell side; negative disables")
    parser.add_argument("--weld-memory-limit-gib", type=float, default=12.)
    parser.add_argument("--raw-union", action="store_true")
    parser.add_argument("--skip-overlap-audit", action="store_true")
    parser.add_argument("--preview-width", type=int, default=320)
    parser.add_argument("--max-shared-views-per-pair", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    if min(args.weld_memory_limit_gib, args.preview_width, args.max_shared_views_per_pair, args.threads) <= 0:
        raise ValueError("Memory limit, preview dimensions/count and threads must be positive")
    start = time.perf_counter()
    manifest_path, output = args.manifest.resolve(), args.output.resolve()
    manifest = normalize_manifest(json.loads(manifest_path.read_text()), manifest_path.parent)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    work = output / ".work"
    work.mkdir()
    mapper = import_helper("inspect_ffba_large_result")
    parts, reports = [], []
    for group in manifest["groups"]:
        vf, ff, report = selected_group_arrays(group, manifest["chunks"], work, mapper)
        parts.append((vf, ff))
        reports.append(report)
        print(json.dumps(dict(stage="select_group", **report)), flush=True)
    total_v, total_f = sum(r["selected_vertices"] for r in reports), sum(r["selected_faces"] for r in reports)
    vertices = np.lib.format.open_memmap(work / "merged_vertices.npy", mode="w+", dtype=np.float32, shape=(total_v, 3))
    faces = np.lib.format.open_memmap(work / "merged_faces.npy", mode="w+", dtype=np.int64, shape=(total_f, 3))
    voff = foff = 0
    for vf, ff in parts:
        v, f = np.load(vf, mmap_mode="r"), np.load(ff, mmap_mode="r")
        for start0 in range(0, len(v), BATCH):
            vertices[voff + start0:voff + min(start0 + BATCH, len(v))] = v[start0:start0 + BATCH]
        for start0 in range(0, len(f), BATCH):
            faces[foff + start0:foff + min(start0 + BATCH, len(f))] = f[start0:start0 + BATCH] + voff
        voff += len(v)
        foff += len(f)
        del v, f
    vertices.flush()
    faces.flush()
    raw_owned = output / "mesh_owned_unwelded.ply"
    write_ply(raw_owned, vertices, faces)
    cleaned_v, cleaned_f, weld = weld_and_clean(vertices, faces, args.weld_tolerance, args.weld_memory_limit_gib)
    final = output / "mesh.ply"
    write_ply(final, cleaned_v, cleaned_f)
    if args.raw_union:
        raw_parts, offsets, rv, rf = [], [], 0, 0
        for group in manifest["groups"]:
            mesh = mapper.map_ply(group["mesh_path"])
            raw_parts.append(mesh)
            offsets.append(rv)
            rv += mesh["vertex_count"]
            rf += mesh["face_count"]
        uv = np.lib.format.open_memmap(work / "raw_vertices.npy", mode="w+", dtype=np.float32, shape=(rv, 3))
        uf = np.lib.format.open_memmap(work / "raw_faces.npy", mode="w+", dtype=np.int64, shape=(rf, 3))
        offset_f = 0
        for mesh, offset_v in zip(raw_parts, offsets):
            for start0 in range(0, mesh["vertex_count"], BATCH):
                rows = mesh["vertices"][start0:start0 + BATCH]
                coords = np.column_stack([rows[key] for key in ("x", "y", "z")])
                uv[offset_v + start0:offset_v + start0 + len(coords)] = coords
            for start0 in range(0, mesh["face_count"], BATCH):
                f = mesh["faces"]["indices"][start0:start0 + BATCH]
                uf[offset_f + start0:offset_f + start0 + len(f)] = f + offset_v
            offset_f += mesh["face_count"]
        write_ply(output / "mesh_raw_union.ply", uv, uf)
        del uv, uf, raw_parts
    del vertices, faces, cleaned_v, cleaned_f
    gc.collect()
    audit = None
    if not args.skip_overlap_audit:
        if "data_dir" not in manifest:
            raise ValueError("data_dir required unless --skip-overlap-audit is used")
        audit_output = output / "overlap"
        audit_output.mkdir()
        audit = overlap_audit(manifest, audit_output, mapper, args.preview_width, args.max_shared_views_per_pair, args.threads)
        (audit_output / "overlap_report.json").write_text(json.dumps(audit, indent=2) + "\n")
    for group, report in zip(manifest["groups"], reports):
        if sha256(group["mesh_path"]) != report["raw_sha256"]:
            raise ValueError("Raw local mesh changed during fusion")
    result = dict(status="complete", coordinate_frame="global_zup", mesh=str(final), mesh_sha256=sha256(final),
                  groups=reports, global_chunk_count=len(manifest["chunks"]), core_chunk_partition_exact=True,
                  owner_distance_tie_tolerance_scene_units=1e-7,
                  weld=weld, overlap_audit=audit, raw_meshes_preserved=True,
                  no_icp_or_pose_adjustment=True, texture_exported=False, watertightness_guaranteed=False,
                  manifest_path=str(manifest_path), manifest_sha256=sha256(manifest_path),
                  seconds=time.perf_counter() - start, maximum_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
    (output / "fusion_report.json").write_text(json.dumps(result, indent=2) + "\n")
    if not args.keep_work:
        shutil.rmtree(work)
    print(json.dumps({k: result[k] for k in ("status", "mesh", "seconds", "maximum_rss_gib")}), flush=True)


if __name__ == "__main__":
    main()
