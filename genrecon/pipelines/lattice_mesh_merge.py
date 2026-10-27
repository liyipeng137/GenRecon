"""CPU merge of independently decoded pieces on one fixed fine lattice.

This mirrors the grouped shape decoder's final lattice-key deduplication. It
only remaps face indices: predicted coordinates are never averaged and faces
are never removed, reordered, welded geometrically, or reoriented.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch


_BATCH = 250_000


def _cpu_array(value, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must already be on CPU")
        value = value.detach().numpy()
    return np.asarray(value)


def merge_cpu_lattice_parts(
    v_parts: Sequence,
    f_parts: Sequence,
    voxel_coords_parts: Sequence,
    aabb,
    grid_size_fine,
):
    """Return CPU ``(vertices float32, faces int32, anchors float32, report)``.

    Faces in each part index that part's local vertices. Voxel coordinates are
    integer ``[V, 3]`` positions in the shared fine lattice, not local group
    coordinates. Keys use ``GY=grid_size_fine[1]+1`` and likewise for ``GZ``.
    Output vertices follow ascending lattice keys; the first input row at each
    key supplies its predicted position, exactly as native stable-first merge.
    Face order follows part order followed by each part's original row order.

    Only one global inverse *vertex* map is built. Faces are remapped in bounded
    batches, without materializing a global int64 ``[F, 3]`` face array.
    """
    started = time.perf_counter()
    if not v_parts or len(v_parts) != len(f_parts) or len(v_parts) != len(voxel_coords_parts):
        raise ValueError("Vertex, face and lattice parts must have matching nonempty part counts")
    bounds = _cpu_array(aabb, "aabb").astype(np.float32, copy=False)
    grid = _cpu_array(grid_size_fine, "grid_size_fine")
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
        raise ValueError("aabb must contain finite increasing bounds of shape [2,3]")
    if grid.shape != (3,) or not np.issubdtype(grid.dtype, np.integer) or not (grid > 0).all():
        raise ValueError("grid_size_fine must contain three positive integers")
    gx, gy, gz = (int(value) for value in grid)
    key_y, key_z = gy + 1, gz + 1
    if (gx + 1) * key_y * key_z - 1 > np.iinfo(np.int64).max:
        raise ValueError("Fine lattice packed keys exceed int64 range")

    vertices, faces, coords = [], [], []
    counts, face_counts = [], []
    for index, (vp, fp, cp) in enumerate(zip(v_parts, f_parts, voxel_coords_parts)):
        v = _cpu_array(vp, f"vertices[{index}]")
        f = _cpu_array(fp, f"faces[{index}]")
        c = _cpu_array(cp, f"voxel_coords[{index}]")
        if v.ndim != 2 or v.shape[1] != 3 or not np.issubdtype(v.dtype, np.floating):
            raise ValueError("Vertices must be floating point arrays of shape [V,3]")
        if c.shape != v.shape or not np.issubdtype(c.dtype, np.integer):
            raise ValueError("Lattice coordinates must be integer arrays matching [V,3]")
        if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
            raise ValueError("Faces must be integer arrays of shape [F,3]")
        for start in range(0, len(v), _BATCH):
            cv = c[start:start + _BATCH]
            if not np.isfinite(v[start:start + _BATCH]).all():
                raise ValueError("Nonfinite mesh vertex")
            if (cv < 0).any() or (cv > grid).any():
                raise ValueError("Lattice coordinates must lie in the shared nonnegative fine grid")
        for start in range(0, len(f), _BATCH):
            ff = f[start:start + _BATCH]
            if ff.min() < 0 or ff.max() >= len(v):
                raise ValueError("Face references an invalid local vertex")
        vertices.append(v)
        faces.append(f)
        coords.append(c)
        counts.append(len(v))
        face_counts.append(len(f))
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    total_vertices, total_faces = int(offsets[-1]), sum(face_counts)
    keys = np.empty(total_vertices, dtype=np.int64)
    for index, c in enumerate(coords):
        for start in range(0, len(c), _BATCH):
            block = c[start:start + _BATCH].astype(np.int64, copy=False)
            position = int(offsets[index]) + start
            keys[position:position + len(block)] = block[:, 0] * (key_y * key_z) + block[:, 1] * key_z + block[:, 2]
    unique_keys, representatives, inverse = np.unique(keys, return_index=True, return_inverse=True)
    unique_count = len(unique_keys)
    if unique_count > np.iinfo(np.int32).max:
        raise ValueError("Merged vertex count exceeds the int32 index limit")
    del keys, unique_keys
    result_v = np.empty((unique_count, 3), dtype=np.float32)
    anchors = np.empty((unique_count, 3), dtype=np.float32)
    result_f = np.empty((total_faces, 3), dtype=np.int32)
    spacing = (bounds[1] - bounds[0]) / grid.astype(np.float32)
    for start in range(0, unique_count, _BATCH):
        source = representatives[start:start + _BATCH]
        source_parts = np.searchsorted(offsets[1:], source, side="right")
        for index in np.unique(source_parts):
            take = np.flatnonzero(source_parts == index)
            local_rows = source[take] - offsets[index]
            result_v[start + take] = vertices[index][local_rows]
            anchors[start + take] = (coords[index][local_rows].astype(np.float32) + np.float32(.5)) * spacing + bounds[0]

    face_offset = 0
    cross_group_keys = np.zeros(unique_count, dtype=bool)
    comparison_count = 0
    distance_sum = 0.
    distance_max = 0.
    for index, f in enumerate(faces):
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        local_inverse = inverse[lo:hi]
        for start in range(0, len(f), _BATCH):
            block = f[start:start + _BATCH]
            result_f[face_offset + start:face_offset + start + len(block)] = local_inverse[block]
        face_offset += len(f)
        # Compare only predictions from a group different from the key's first
        # representative group. Within-group duplicate rows do not count as a
        # cross-group measurement. Each later-group row is one comparison.
        for start in range(0, hi - lo, _BATCH):
            destination = local_inverse[start:start + _BATCH]
            first_rows = representatives[destination]
            other_group = (first_rows < lo) | (first_rows >= hi)
            if not other_group.any():
                continue
            selected = destination[other_group]
            cross_group_keys[selected] = True
            # Native Mesh stores float32 coordinates. Compare those positions,
            # then accumulate the distance metric at float64 precision.
            predicted = vertices[index][start:start + len(destination)][other_group].astype(np.float32)
            difference = predicted.astype(np.float64) - result_v[selected].astype(np.float64)
            distances = np.linalg.norm(difference, axis=1)
            comparison_count += len(distances)
            distance_sum += float(distances.sum(dtype=np.float64))
            distance_max = max(distance_max, float(distances.max()))
    report = dict(
        input_parts=len(v_parts), input_vertices=total_vertices, output_vertices=unique_count,
        input_faces=total_faces, output_faces=total_faces,
        removed_duplicate_lattice_vertices=total_vertices - unique_count,
        representative="first input part/row at each key; vertices ordered by ascending packed lattice key",
        face_order="input part order, then original face row order; indices remapped only",
        faces_removed=0,
        cross_group_shared_lattice_keys=int(cross_group_keys.sum()),
        cross_group_prediction_distance_to_first_representative=dict(
            compared_vertex_predictions=comparison_count,
            mean=distance_sum / comparison_count if comparison_count else None,
            max=distance_max if comparison_count else None,
            coordinates="normalized joint mesh frame",
            interpretation="Decoder predictions at identical lattice keys; not errors against ground truth",
        ),
        lattice_key_gy=key_y, lattice_key_gz=key_z,
        face_remap_batch_size=_BATCH,
        global_int64_face_array_materialized=False,
        seconds=time.perf_counter() - started,
    )
    return torch.from_numpy(result_v), torch.from_numpy(result_f), torch.from_numpy(anchors), report
