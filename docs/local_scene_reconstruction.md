# Independent local geometry baseline

`scripts/reconstruct_local_scene.py` processes a frozen global selection of
images using an explicit, common upright layout. It generates one chunk at a
time with SS + shape only, using every original image assigned to that chunk.
There is no additional local image cap. It rejects layouts that omit any image
from all local pools. Non-square iPhone frames use the native two-crop policy.

DINO features are encoded once in small batches into an on-disk CPU cache.
Local camera matrices and corresponding feature rows are gathered in identical
order. Both flow models are loaded once; projection is batched over voxels.
The 512 pipeline uses no 1024 RGB images or texture models. The 2D anchor uses
the maximum native projected chunk area, with input order breaking ties.

Each chunk uses seed `base_seed + chunk_id`. These runs are independent: shared
images, coordinates and lattice alignment do **not** synchronize generated
occupancy, shape latents, or predicted vertices between chunks. No ICP or
per-chunk scale fitting is applied.

The decoder exports fixed voxel-center anchors in the same order as vertices.
Hole filling is disabled to preserve that correspondence. Outputs include raw
global-frame PLY meshes, float64 world anchors, per-group conditioning lists,
timings, hashes and a manifest proving the final global image union. Existing
completed groups can be resumed only if settings, source identity and output
hashes match. CUDA peaks stored per group cover the final shape decode scope;
`run_profiled.py` separately samples whole-process GPU memory once per second.

```bash
python scripts/run_profiled.py --output RUN/attempt_01 -- \
  python scripts/reconstruct_local_scene.py \
  --layout LAYOUT.json --selected-paths FROZEN_PATHS.json \
  --data-dir SCENE --source-dataset ORIGINAL_FFBA_SCENE \
  --output RUN/attempt_01/result --proj-batch-voxels 256

python scripts/fuse_local_meshes.py \
  --manifest RUN/attempt_01/result/manifest.json --output RUN/fusion

python scripts/inspect_ffba_large_result.py \
  --result-dir RUN/fusion --data-dir SCENE \
  --source-dataset ORIGINAL_FFBA_SCENE --output RUN/inspection --threads 4

# Optional derived initializer; keep the complete fusion for quality checks.
python scripts/run_profiled.py --output RUN/coarse_profile -- \
  timeout 240s python scripts/simplify_geometry.py \
  --mesh RUN/fusion/mesh.ply --output RUN/coarse --target-faces 2000000
```

Fusion assigns each fixed triangle barycenter to the nearest **covering** chunk
center, with stable lower-ID ties, then joins the owned triangles. By default
only exactly coincident predicted positions are welded. Different predictions
at the same lattice coordinate remain distinct; the baseline does not claim
watertight seams. Raw local meshes remain available for overlap diagnostics.

The overlap audit compares adjacent raw meshes along shared real camera rays,
restricted to their common spatial volume. It reports depth and unsigned-normal
disagreement and possible double surfaces. This measures local consistency,
not ground-truth reconstruction accuracy. The final inspector also preserves
all triangles when applying the inverse FFBA world transform.

Audit cameras are ranked by rays intersecting the overlap volume, then a bounded
low-resolution raw-mesh probe selects views with visible common surfaces and
different poses. Depth/normal error values do not influence this selection.
Empty probes, full-resolution common-pixel counts and every candidate score are
retained in the audit report. Missing or outside-overlap hits are excluded from
common-depth statistics; these statistics alone cannot establish seam quality.

`simplify_geometry.py` optionally applies CuMesh QEM to the completed fusion in
a separate GPU process, exporting a smaller geometry-only PLY. It preserves the
complete source and does not fix local seam disagreements. CuMesh CUDA memory
is measured by the device profiler, since it is allocated outside PyTorch.
