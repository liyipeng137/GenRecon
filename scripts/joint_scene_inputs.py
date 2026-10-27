"""Frozen local inputs and coherent CPU noise for joint scene inference.

No model or CUDA imports are required. Chunk translations are center offsets
in units of one chunk side; their voxel offsets must lie on the requested
resolution's lattice. Randomness comes only from CPU ``torch.randn`` and can
be controlled by ``torch.manual_seed`` or an explicit CPU Generator.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _cpu_tensor(value, *, dtype):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=dtype)
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        return torch.stack([item.detach().cpu() for item in value]).to(dtype=dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype, device="cpu")


def global_grid_layout(relative_translations, resolution):
    """Return zero-based offsets[K,3], global shape, and minimum translation.

    Local cell ``xyz`` of chunk i has global integer cell
    ``xyz + offsets[i]`` and chunk-0 coordinates
    ``(global_cell + .5) / resolution + minimum_translation - .5``.
    Reject fractional-cell translations instead of silently rounding distinct
    world locations onto the same voxel.
    """
    resolution = _positive_integer(resolution, "resolution")
    relative = _cpu_tensor(relative_translations, dtype=torch.float64)
    if relative.ndim != 2 or relative.shape[1] != 3 or not len(relative) or not torch.isfinite(relative).all():
        raise ValueError("relative_translations must be finite nonempty [K,3]")
    scaled = relative * resolution
    rounded = scaled.round()
    if not torch.allclose(scaled, rounded, rtol=0, atol=1e-5):
        raise ValueError("Chunk translations do not align to the requested voxel lattice")
    absolute = rounded.long()
    minimum = absolute.min(dim=0).values
    offsets = absolute - minimum
    shape = tuple((offsets.max(dim=0).values + resolution).tolist())
    return offsets, shape, minimum.double() / resolution


def _random_arguments(dtype, generator):
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Noise dtype must be a floating-point torch dtype")
    if generator is not None and str(generator.device) != "cpu":
        raise ValueError("Shared noise requires a CPU Generator")
    return dict(device="cpu", dtype=dtype, generator=generator)


def shared_dense_noise(relative_translations, resolution, channels, *, batch_size=1,
                       dtype=torch.float32, generator=None):
    """Draw one global dense noise field and clone each chunk's subcube."""
    channels = _positive_integer(channels, "channels")
    batch_size = _positive_integer(batch_size, "batch_size")
    offsets, shape, _ = global_grid_layout(relative_translations, resolution)
    field = torch.randn(batch_size, channels, *shape, **_random_arguments(dtype, generator))
    result = []
    for offset in offsets.tolist():
        x, y, z = offset
        result.append(field[:, :, x:x + resolution, y:y + resolution, z:z + resolution].clone())
    return result


def shared_sparse_noise(coords_list, relative_translations, resolution, channels, *,
                        dtype=torch.float32, generator=None):
    """One Gaussian vector per global active (batch,x,y,z) cell.

    ``coords_list`` contains integer [N_i,4] tensors in per-chunk coordinates.
    Returns CPU feature tensors [N_i,channels] in exactly the input row order;
    callers can attach these to their original SparseTensor coordinates.
    """
    channels = _positive_integer(channels, "channels")
    offsets, _, _ = global_grid_layout(relative_translations, resolution)
    if len(coords_list) != len(offsets):
        raise ValueError("One coordinate tensor is required per chunk")
    globals_ = []
    lengths = []
    for coords, offset in zip(coords_list, offsets):
        raw = _cpu_tensor(coords, dtype=torch.float64)
        if raw.ndim != 2 or raw.shape[1] != 4 or not torch.isfinite(raw).all() or not torch.equal(raw, raw.round()):
            raise ValueError("Sparse coordinates must be finite integer [N,4]")
        values = raw.long()
        if (values[:, 0] < 0).any() or (values[:, 1:] < 0).any() or (values[:, 1:] >= resolution).any():
            raise ValueError("Sparse batch indices must be nonnegative and xyz must be within the chunk")
        global_coords = values.clone()
        global_coords[:, 1:] += offset
        globals_.append(global_coords)
        lengths.append(len(values))
    pooled = torch.cat(globals_, dim=0)
    unique, inverse = torch.unique(pooled, dim=0, sorted=True, return_inverse=True)
    field = torch.randn(len(unique), channels, **_random_arguments(dtype, generator))
    rows = field[inverse]
    return [part.clone() for part in rows.split(lengths)]


def load_joint_scene_inputs(layout_path, baseline_manifest_path, feature_cache_path=None):
    """Verify frozen layout/pools and mmap the exact baseline DINO cache.

    This validates cache SHA256 against both baseline manifest and cache sidecar,
    without materializing the multi-GB array in RAM. Runtime crop geometry must
    subsequently be bound with ``validate_crop_bindings`` before inference.
    """
    layout_path, baseline_manifest_path = Path(layout_path), Path(baseline_manifest_path)
    layout, baseline = _json(layout_path), _json(baseline_manifest_path)
    names = baseline["frozen_image_names"]
    if not names or len(names) != len(set(names)):
        raise ValueError("Frozen original names must be nonempty and unique")
    selected_paths = _json(layout["selection"]["paths_file"])
    if [Path(path).name for path in selected_paths] != names:
        raise ValueError("Frozen original path order differs from baseline")
    if layout["selection"]["frozen_global_original_image_count"] != len(names):
        raise ValueError("Frozen original image count differs from layout")
    if baseline.get("coordinate_frame") != "global_zup":
        raise ValueError("Baseline must use the same global Z-up frame")
    original_layout = Path(baseline["layout_path"])
    if _digest(original_layout) != _digest(layout_path):
        raise ValueError("Layout bytes differ from the baseline layout")
    baseline_chunks = {chunk["id"]: chunk for chunk in baseline["chunks"]}
    group_map = {}
    for group in baseline["groups"]:
        ids = group["chunk_ids"]
        if len(ids) != 1 or ids[0] in group_map or group.get("status") != "complete":
            raise ValueError("Expected one complete independent baseline group per chunk")
        group_map[ids[0]] = group
    chunks = []
    seen = set()
    size = float(layout["chunk_size_world"])
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Invalid global chunk size")
    reference = np.asarray(layout["chunks"][0]["center"], dtype=np.float64)
    for chunk in layout["chunks"]:
        cid = chunk["id"]
        if cid in seen or cid not in group_map or cid not in baseline_chunks:
            raise ValueError("Layout and baseline chunk ids do not match uniquely")
        seen.add(cid)
        group = group_map[cid]
        if group["selected_image_names"] != chunk["pool_image_names"]:
            raise ValueError(f"Chunk {cid}: local image pool/order changed")
        pool = chunk["pool_image_names"]
        if not pool or len(set(pool)) != len(pool) or not set(pool) <= set(names):
            raise ValueError(f"Chunk {cid}: invalid frozen local pool")
        center = np.asarray(chunk["center"], dtype=np.float64)
        relative = np.asarray(chunk["relative_translation"], dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all() or not np.allclose(relative, (center-reference)/size, atol=1e-9):
            raise ValueError(f"Chunk {cid}: relative translation disagrees with global centers")
        forward = np.asarray(chunk["M_original_to_chunk"], dtype=np.float64)
        inverse = np.asarray(chunk["M_chunk_to_original"], dtype=np.float64)
        expected = np.eye(4)
        expected[:3, :3] *= size
        expected[:3, 3] = center
        if not np.allclose(inverse, expected, atol=1e-9) or not np.allclose(forward @ inverse, np.eye(4), atol=1e-9):
            raise ValueError(f"Chunk {cid}: local/world transform changed")
        old = baseline_chunks[cid]
        if not np.isclose(chunk["size"], size) or not np.isclose(old["size"], size) or not np.allclose(old["center"], center, atol=1e-9):
            raise ValueError(f"Chunk {cid}: baseline center/size differs")
        indices = group["crop_indices"]
        if not indices or indices != sorted(set(indices)) or group["crop_count"] != len(indices):
            raise ValueError(f"Chunk {cid}: invalid baseline crop indices")
        if group["anchor"]["image_name"] not in pool:
            raise ValueError(f"Chunk {cid}: baseline anchor is outside the pool")
        chunks.append(dict(id=cid, layout=copy.deepcopy(chunk), crop_indices=list(indices),
                           relative_translation=relative.tolist(), anchor=copy.deepcopy(group["anchor"]),
                           baseline_group=copy.deepcopy(group)))
    if seen != set(group_map) or seen != set(baseline_chunks):
        raise ValueError("Baseline contains a different set of chunks")
    frozen_union = set().union(*(set(chunk["layout"]["pool_image_names"]) for chunk in chunks))
    if frozen_union != set(names):
        raise ValueError("Some frozen originals are absent from every chunk pool")
    relative_translations = [chunk["relative_translation"] for chunk in chunks]
    for resolution in (16, 32):
        global_grid_layout(relative_translations, resolution)
    feature_info = baseline["feature_cache"]
    feature_path = Path(feature_cache_path or feature_info["feature_path"])
    sidecar_path = feature_path.parent / "feature_cache.json"
    sidecar = _json(sidecar_path)
    if sidecar["identity"] != baseline["identity"] or feature_info["identity"] != baseline["identity"]:
        raise ValueError("Feature cache identity does not match baseline")
    for key in ("shape", "dtype", "sha256"):
        if sidecar[key] != feature_info[key]:
            raise ValueError(f"Feature cache sidecar/manifest disagree: {key}")
    actual_hash = _digest(feature_path)
    if actual_hash != feature_info["sha256"]:
        raise ValueError("Feature cache SHA256 differs from baseline")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if list(features.shape) != feature_info["shape"] or str(features.dtype) != feature_info["dtype"] or features.ndim != 3:
        raise ValueError("Feature cache shape/dtype differs from baseline")
    for chunk in chunks:
        if min(chunk["crop_indices"]) < 0 or max(chunk["crop_indices"]) >= len(features):
            raise ValueError("Baseline crop indices lie outside the feature cache")
    return dict(layout=layout, manifest=baseline, chunks=chunks, feature_cache=features,
                feature_cache_path=str(feature_path), selected_paths=selected_paths,
                frozen_image_names=list(names), relative_translations=relative_translations,
                validation=dict(feature_cache_sha256=actual_hash, layout_sha256=_digest(layout_path),
                                frozen_original_count=len(names), chunk_count=len(chunks),
                                crop_bindings_validated=False))


def validate_crop_bindings(inputs, crops):
    """Bind baseline 2D anchors to runtime crops without changing any pool.

    Mutates only ``inputs['chunks']`` (adds ``anchor_global_index``) and its
    validation flag; returns inputs for convenient driver composition.
    """
    if len(crops) != len(inputs["feature_cache"]):
        raise ValueError("Runtime crop count differs from baseline feature cache")
    names = [Path(crop["img_path"]).name for crop in crops]
    original_order = []
    for name in names:
        if not original_order or original_order[-1] != name:
            original_order.append(name)
    if original_order != inputs["frozen_image_names"]:
        raise ValueError("Runtime crop original order differs from frozen path order")
    frozen_paths = {Path(path).name: Path(path).resolve() for path in inputs["selected_paths"]}
    if any(Path(crop["img_path"]).resolve() != frozen_paths[name] for crop, name in zip(crops, names)):
        raise ValueError("Runtime crop image paths differ from the frozen original paths")
    key_to_index = {}
    for index, crop in enumerate(crops):
        box = crop.get("crop_box")
        key = (names[index], tuple(box) if box is not None else None)
        if key in key_to_index:
            raise ValueError("Duplicate runtime crop name/box binding")
        key_to_index[key] = index
    for chunk in inputs["chunks"]:
        pool = set(chunk["layout"]["pool_image_names"])
        expected = [index for index, name in enumerate(names) if name in pool]
        if expected != chunk["crop_indices"]:
            raise ValueError(f"Chunk {chunk['id']}: runtime crop pool/order differs from baseline")
        anchor = chunk["anchor"]
        box = anchor.get("crop_box")
        key = (anchor["image_name"], tuple(box) if box is not None else None)
        if key not in key_to_index or key_to_index[key] not in expected:
            raise ValueError(f"Chunk {chunk['id']}: baseline anchor crop is unavailable")
        chunk["anchor_global_index"] = key_to_index[key]
    inputs["validation"]["crop_bindings_validated"] = True
    return inputs
