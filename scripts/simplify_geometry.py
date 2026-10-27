#!/usr/bin/env python3
"""Export an optional coarse, untextured initializer from a completed mesh.

Run in an isolated process under run_profiled.py: CuMesh owns CUDA allocations
outside PyTorch's allocator. The original mesh is never changed; simplification
does not repair disagreement between independently generated local surfaces.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import resource
import time

import numpy as np
import torch
import cumesh

from inspect_ffba_large_result import map_ply, xyz
from reconstruct_local_scene import digest_file, write_json, write_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=2_000_000)
    args = parser.parse_args()
    if args.target_faces < 1:
        parser.error("target-faces must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    source_hash = digest_file(args.mesh)
    mapped = map_ply(args.mesh)
    vertices = xyz(mapped["vertices"], np.float32)
    faces = np.array(mapped["faces"]["indices"], dtype=np.int32, order="C")
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Invalid input mesh")
    source_counts = dict(vertices=len(vertices), faces=len(faces))
    mesh = cumesh.CuMesh()
    vg = torch.from_numpy(vertices).cuda()
    fg = torch.from_numpy(faces).cuda()
    mesh.init(vg, fg)
    del vg, fg, vertices, faces
    torch.cuda.empty_cache()
    print(f"[simplify] {source_counts} -> target {args.target_faces} faces", flush=True)
    simplify_start = time.monotonic()
    mesh.simplify(args.target_faces, verbose=True)
    torch.cuda.synchronize()
    simplify_seconds = time.monotonic() - simplify_start
    vg, fg = mesh.read()
    vertices, faces = vg.cpu().numpy(), fg.cpu().numpy()
    # CuMesh can leave vertices no longer referenced by any surviving face.
    # A downstream initializer may consume the complete vertex array, so do
    # not export these inactive points as if they belonged to the surface.
    referenced, inverse = np.unique(faces, return_inverse=True)
    removed_unreferenced = len(vertices) - len(referenced)
    vertices = vertices[referenced]
    faces = inverse.reshape(-1, 3).astype(np.int32)
    target = output / "mesh.ply"
    write_mesh(target, vertices, faces)
    degenerate_indices = int(np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0, axis=1).sum())
    if degenerate_indices:
        raise ValueError(f"Simplified mesh has {degenerate_indices} collapsed triangles")
    if digest_file(args.mesh) != source_hash:
        raise ValueError("Source mesh changed")
    result = dict(status="complete", source_mesh=str(args.mesh.resolve()), source_sha256=source_hash,
                  source_counts=source_counts, target_faces=args.target_faces,
                  mesh=str(target), sha256=digest_file(target), vertices=len(vertices), faces=len(faces),
                  degenerate_index_faces=degenerate_indices, geometry_only=True,
                  removed_unreferenced_vertices=removed_unreferenced,
                  method="CuMesh QEM on completed globally owned mesh; no remeshing or texture",
                  source_preserved=True, seam_repair_performed=False,
                  simplify_seconds=simplify_seconds, seconds=time.monotonic()-start,
                  maximum_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                  memory_measurement="CuMesh CUDA allocations require process/device sampling; PyTorch peak is incomplete")
    write_json(output / "simplification_report.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
