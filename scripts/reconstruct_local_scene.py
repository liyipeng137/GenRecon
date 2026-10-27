#!/usr/bin/env python3
"""Independent local geometry baseline with a frozen global image pool.

One globally defined chunk is generated at a time. Every image assigned by the
layout is used, without another image budget. DINO crops are encoded once into
a CPU/disk cache. Geometry and fixed lattice anchors are exported in the shared
upright world frame; no per-chunk pose fitting or normalization is performed.
Run under scripts/run_profiled.py to acquire the shared GPU lease.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_layout(layout, selected_paths):
    names = [Path(p).name for p in selected_paths]
    if not names or len(set(names)) != len(names):
        raise ValueError("Frozen input names must be unique and nonempty")
    selected = set(names)
    assigned = set()
    ids = set()
    for chunk in layout["chunks"]:
        cid = chunk["id"]
        if cid in ids:
            raise ValueError("Duplicate chunk id")
        ids.add(cid)
        pool = chunk["pool_image_names"]
        if not pool or len(set(pool)) != len(pool) or not set(pool) <= selected:
            raise ValueError(f"Chunk {cid}: invalid or non-frozen local image pool")
        assigned.update(pool)
        forward = np.asarray(chunk["M_original_to_chunk"], dtype=np.float64)
        inverse = np.asarray(chunk["M_chunk_to_original"], dtype=np.float64)
        side = float(chunk["size"])
        if side <= 0 or not np.allclose(inverse[:3, :3], np.eye(3) * side, atol=1e-10):
            raise ValueError("All chunk transforms must preserve the global axis frame")
        if not np.allclose(forward @ inverse, np.eye(4), atol=1e-10):
            raise ValueError("Chunk transform is not invertible")
        if not np.allclose(inverse[:3, 3], chunk["center"], atol=1e-10):
            raise ValueError("Chunk center and inverse transform disagree")
    if assigned != selected:
        raise ValueError(f"Frozen images omitted from all local pools: {sorted(selected-assigned)}")
    return names


def crop_indices(crops, image_names):
    wanted = set(image_names)
    indices = [i for i, c in enumerate(crops) if Path(c["img_path"]).name in wanted]
    if {Path(crops[i]["img_path"]).name for i in indices} != wanted:
        raise ValueError("Some requested local images have no encoded crop")
    return indices


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def read_crop_512(crop):
    with Image.open(crop["img_path"]) as im:
        box = crop.get("crop_box")
        if box is not None:
            left, top, size = box
            im = im.crop((left, top, left + size, top + size))
        rgb = np.asarray(im.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)).copy()
        result = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        if "A" in im.getbands():
            # Matches the native image loader's bilinear alpha interpolation.
            import torch
            alpha = torch.from_numpy(np.array(im.getchannel("A"), dtype=np.float32)) / 255.
            alpha = torch.nn.functional.interpolate(alpha[None, None], size=(512, 512),
                                                    mode="bilinear", align_corners=False)
            result *= alpha[0, 0].numpy()[None]
        return result


def write_mesh(path, vertices, faces):
    """Binary xyz-only PLY, preserving both vertex and face order."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Invalid mesh vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Empty or invalid triangle mesh")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Triangle references an invalid vertex")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    with Path(path).open("wb") as f:
        f.write(header.encode())
        for start in range(0, len(vertices), 250_000):
            np.asarray(vertices[start:start+250_000], dtype="<f4").tofile(f)
        for start in range(0, len(faces), 250_000):
            part = faces[start:start+250_000]
            packed = np.empty(len(part), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
            packed["count"] = 3
            packed["indices"] = part
            packed.tofile(f)


def encode_cache(pipeline, crops, root, identity, load_batch_size):
    import torch
    metadata_path, feature_path = root / "feature_cache.json", root / "scene_features_512.npy"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity:
            raise ValueError("Existing feature cache belongs to different inputs/settings")
        features = np.load(feature_path, mmap_mode="r")
        if list(features.shape) != metadata["shape"] or features.dtype != np.float32:
            raise ValueError("Feature cache shape/dtype changed")
        if digest_file(feature_path) != metadata["sha256"]:
            raise ValueError("Feature cache contents changed")
        return features, metadata
    temp_path = feature_path.with_suffix(".partial.npy")
    model = pipeline.image_cond_model
    model.image_size = 512
    model.to(pipeline.device)
    start = time.monotonic()
    features = None
    try:
        with torch.no_grad():
            for lo in range(0, len(crops), load_batch_size):
                images = torch.from_numpy(np.stack([read_crop_512(c) for c in crops[lo:lo+load_batch_size]]))
                tokens = model.encode_batched(images.unsqueeze(0), output_device="cpu")[0].detach().float().numpy()
                if features is None:
                    features = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float32,
                                                         shape=(len(crops), *tokens.shape[1:]))
                features[lo:lo+len(tokens)] = tokens
                del images, tokens
                print(f"[feature_cache] {min(lo+load_batch_size,len(crops))}/{len(crops)} crops", flush=True)
    finally:
        model.cpu()
        torch.cuda.empty_cache()
    features.flush()
    shape = list(features.shape)
    del features
    temp_path.replace(feature_path)
    metadata = dict(identity=identity, shape=shape, dtype="float32", seconds=time.monotonic()-start,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    feature_path=str(feature_path), sha256=digest_file(feature_path))
    write_json(metadata_path, metadata)
    return np.load(feature_path, mmap_mode="r"), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--selected-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("/kiri/models/genrecon"))
    parser.add_argument("--pipeline-config", default="configs/pipelines/texture.json")
    parser.add_argument("--proj-batch-voxels", type=int, default=256)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--load-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--occ-threshold", type=float, default=-1.)
    args = parser.parse_args()
    for value in (args.proj_batch_voxels, args.image_batch_size, args.load_batch_size):
        if value < 1:
            parser.error("Batch sizes must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    layout = json.loads(args.layout.read_text())
    selected = json.loads(args.selected_paths.read_text())
    names = validate_layout(layout, selected)
    import torch
    from inference.get_images import IphoneImageSelecter, _to_chunk0_extrinsics
    from inference.view_selection import chunk_projection_areas
    from genrecon.pipelines.full_scene_images_to_3d import FullSceneImagesTo3DPipeline
    from genrecon.pipelines.types import SelectedImages

    selector = IphoneImageSelecter(center_crop=False)
    cameras = selector._get_cameras(args.data_dir / "colmap" / "cameras.txt")
    by_name = {Path(c["img_path"]).name: c for c in cameras}
    if not set(names) <= by_name.keys():
        raise ValueError("Frozen images absent from scene COLMAP")
    crops = selector._prepare_scene_pool([by_name[n] for n in names])
    settings = dict(seed=args.seed, occ_threshold=args.occ_threshold,
                    proj_batch_voxels=args.proj_batch_voxels, image_batch_size=args.image_batch_size,
                    layout_sha256=digest_file(args.layout), selected_sha256=digest_file(args.selected_paths),
                    pipeline_config_sha256=digest_file(args.pipeline_config),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    images=[dict(name=n, size=Path(by_name[n]["img_path"]).stat().st_size,
                                 sha256=digest_file(by_name[n]["img_path"])) for n in names],
                    checkpoints={k:dict(path=str((args.weights/k).resolve()), size=(args.weights/k).stat().st_size,
                                        mtime_ns=(args.weights/k).stat().st_mtime_ns)
                                 for k in ("sparse_structure.pt", "shape_slat.pt")})
    identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    previous = root / "settings.json"
    if previous.exists() and json.loads(previous.read_text())["identity"] != identity:
        raise ValueError("Refusing to resume outputs created with different inputs/settings")
    write_json(previous, dict(identity=identity, settings=settings))
    manifest = dict(coordinate_frame="global_zup", data_dir=str(args.data_dir.resolve()),
                    layout_path=str(args.layout.resolve()), frozen_image_names=names,
                    chunks=[{k:c[k] for k in ("id", "center", "size")} for c in layout["chunks"]],
                    groups=[], method="independent_local_geometry", identity=identity,
                    noise_policy="independent seed+chunk_id; no overlap latent synchronization",
                    fill_holes=False, resolution=512)
    if args.source_dataset:
        manifest["source_dataset"] = str(args.source_dataset.resolve())
    start_all = time.monotonic()
    pipeline = FullSceneImagesTo3DPipeline.from_finetuned(
        stage_models={"sparse_structure_flow_model":str(args.weights/"sparse_structure.pt"),
                      "shape_slat_flow_model_512":str(args.weights/"shape_slat.pt")},
        pipeline_config_file=args.pipeline_config)
    pipeline.image_cond_model.batch_size = args.image_batch_size
    pipeline.proj_batch_voxels = args.proj_batch_voxels
    pipeline.to(torch.device("cuda"))
    write_json(root / "checkpoint_load_report.json", pipeline._checkpoint_load_report)
    features, cache_report = encode_cache(pipeline, crops, root, identity, args.load_batch_size)
    intr = torch.stack([c["intrinsics"] for c in crops])
    for chunk in layout["chunks"]:
        cid = int(chunk["id"])
        group_dir = root / "groups" / f"chunk_{cid:03d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        indices = crop_indices(crops, chunk["pool_image_names"])
        matrix = torch.tensor(chunk["M_original_to_chunk"], dtype=torch.float32)
        areas = chunk_projection_areas(
            np.stack([crops[i]["c2w_blender"].numpy() for i in indices]),
            intr[indices].numpy(), np.asarray(chunk["M_original_to_chunk"])[None])[:, 0]
        # Deterministic 2D cross-attention anchor, chosen from this exact pool.
        # Maximum native projected area; input order breaks ties.
        anchor_global = indices[int(np.argmax(areas))]
        record_path = group_dir / "result.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("identity") != identity or record.get("status") != "complete":
                raise ValueError("Incomplete or incompatible group result")
            for key, hashkey in (("mesh_path", "mesh_sha256"), ("vertex_owner_points_path", "anchors_sha256")):
                if digest_file(record[key]) != record[hashkey]:
                    raise ValueError("Existing group output changed")
        else:
            begin = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            ext = torch.stack([_to_chunk0_extrinsics(crops[i]["c2w_blender"], matrix) for i in indices])
            # Cached tokens replace RGB loading in run(); zero-stride metadata
            # placeholders carry the expected view count without allocating RGB.
            dummy = torch.zeros(1).expand(len(indices), 3, 512, 512)
            empty_1024 = torch.empty(0)
            sel = SelectedImages(scene_images_1024=empty_1024, scene_images_512=dummy,
                                 scene_intrinsics=intr[indices], scene_extrinsics_c0=ext,
                                 cond2d_images_1024=[], cond2d_images_512=[dummy[0]],
                                 cond2d_intrinsics=[intr[anchor_global]],
                                 cond2d_extrinsics_c0=[_to_chunk0_extrinsics(crops[anchor_global]["c2w_blender"],matrix)],
                                 chunk_indices=[0])
            # Advanced indexing already creates an independent writable array.
            local_features = torch.from_numpy(features[indices]).unsqueeze(0)
            anchor_features = local_features[:, [indices.index(anchor_global)]]
            print(f"[local_group] chunk={cid}, originals={len(chunk['pool_image_names'])}, crops={len(indices)}", flush=True)
            with torch.no_grad():
                mesh, coords = pipeline.run(sel, [torch.zeros(3)], seed=args.seed+cid, pipeline_type="512",
                                            geometry_only=True, fill_holes=False, occ_threshold=args.occ_threshold,
                                            scene_features_512=local_features, cond2d_features_512=anchor_features)
            torch.cuda.synchronize()
            phase_peak = torch.cuda.max_memory_allocated()
            if not hasattr(mesh, "lattice_vertices") or len(mesh.lattice_vertices) != len(mesh.vertices):
                raise ValueError("Decoder did not preserve fixed vertex ownership anchors")
            vertices = transform_points(mesh.vertices.cpu().numpy(), chunk["M_chunk_to_original"])
            anchors = transform_points(mesh.lattice_vertices.cpu().numpy(), chunk["M_chunk_to_original"])
            faces = mesh.faces.cpu().numpy()
            mesh_path, anchors_path = group_dir / "mesh.ply", group_dir / "vertex_owner_points.npy"
            write_mesh(mesh_path, vertices, faces)
            np.save(anchors_path, anchors)
            record = dict(status="complete", identity=identity, id=f"chunk_{cid:03d}", chunk_ids=[cid],core_chunk_ids=[cid],
                          mesh_path=str(mesh_path), vertex_owner_points_path=str(anchors_path),
                          selected_image_names=chunk["pool_image_names"], crop_indices=indices,
                          crop_count=len(indices), seed=args.seed+cid,
                          anchor=dict(image_name=Path(crops[anchor_global]["img_path"]).name,
                                      crop_box=crops[anchor_global].get("crop_box"), projected_area=float(areas.max())),
                          vertices=len(vertices), faces=len(faces), occupancy_voxels=[len(c) for c in coords],
                          seconds=time.monotonic()-begin, cuda_shape_decode_peak_allocated_bytes=phase_peak,
                          mesh_sha256=digest_file(mesh_path), anchors_sha256=digest_file(anchors_path))
            write_json(record_path, record)
            del mesh, coords, local_features, anchor_features, vertices, anchors, faces, sel, dummy
            gc.collect()
            torch.cuda.empty_cache()
        manifest["groups"].append(record)
        write_json(root / "manifest.partial.json", manifest)
        print(f"[local_group] completed {cid}, faces={record['faces']}, seconds={record['seconds']:.1f}", flush=True)
    used = set().union(*(set(g["selected_image_names"]) for g in manifest["groups"]))
    if used != set(names):
        raise ValueError("Completed groups did not use every frozen image")
    manifest["coverage"] = dict(frozen_original_images=len(names), used_original_images=len(used),
                                used_crop_views=len(crops), image_memberships=sum(len(g["selected_image_names"]) for g in manifest["groups"]),
                                missing_image_names=[], extra_image_names=[])
    manifest["seconds_this_invocation"] = time.monotonic()-start_all
    manifest["feature_cache"] = cache_report
    write_json(root / "manifest.json", manifest)
    print(json.dumps(dict(status="complete",groups=len(manifest["groups"]),coverage=manifest["coverage"],
                          seconds=manifest["seconds_this_invocation"])), flush=True)


if __name__ == "__main__":
    main()
