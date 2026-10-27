# Pose and chunk-coverage frame selection

The default selector is now `pose_coverage`. It changes the RGB conditioning
pool only; the full calibration, sparse point cloud, chunk layout, and model
weights are retained. Use `--view_selection uniform` for the previous
visibility-filtered linspace image indices. The historical first60 script
explicitly requests `uniform` to preserve that experiment's input selection.

## Selection rule

1. Compute each original image's maximum projected chunk area over its actual
   square crops, including the principal-point centre-crop path.
2. Prefer frames that cover currently uncovered chunks; then try to reach the
   soft per-chunk support target, weighted toward sparsely supported chunks.
3. Fill the remaining budget with diverse camera positions and full rotations.
   Near duplicates must be within 0.05 chunk side and 10 degrees, with similar
   normalized K. A close-pose frame can be rescued for a still-uncovered chunk.
4. Stop at the hard original-image budget, or earlier if only duplicates remain.
   Return frames in input order, with deterministic tie-breaking.

Two crops from the same original are retained: their poses are equal but their
fields of view differ. The per-chunk 2D anchor remains a seeded random visible
crop from the selected pool, with the original closest-camera fallback.
Visibility now uses the K of the actual processed crop for centre crops too.

The nine-point projected bounding-box metric is inherited from GenRecon. It
does not check depth occlusion and is not a measure of true surface coverage.
The minimum-support target is soft; budgets or missing candidates can leave
deficits. These are reported rather than silently exceeding the budget.

## Controls and output

```text
--view_selection pose_coverage|uniform   default: pose_coverage
--num_imgs_per_scene 128                hard cap on ORIGINAL images
--pose_position_threshold 0.05          translation / chunk side
--pose_rotation_threshold 10            full SO(3) angle, degrees
--min_views_per_chunk 3                 soft support target
```

The defaults are an initial experiment setting, not a calibrated optimum.
`view_selection.json` records original paths/indices, crop count, selection
reasons, available/selected views per chunk, uncovered/unobservable chunks,
thresholds, and the actual per-chunk 2D anchors. The seed still controls anchor
sampling; changing the global selected pool may change these anchors even at a
fixed seed, so mesh-quality comparisons must account for that confound.

Implementation: `inference/view_selection.py` is pure NumPy and needs no model
weights or GPU. `inference/get_images.py` integrates the selector before image
loading and crop expansion. It reuses crop geometry for both visibility and
image loading so principal-point crops are not evaluated with the wrong K.

## 2026-10-27 CPU validation on the 529-frame capture

Source: `/kiri/529_prior_pose_and_k_moge3_20261021`.
Output: `/kiri/output/scene_reconstruction/genrecon_pose_selection_20261027`.
The isolated text COLMAP export keeps all 529 cameras, 105,555 BA points, and
739,004 observations. The first60 upright rigid transform is reused, with
projection invariance checked to 1.55e-11 pixels. All selector comparisons use
the same 16 spatial chunks. No source dataset file is modified.

| Original-image cap | Uniform / pose selected | Near-duplicate pairs, uniform / pose | Mean chunk yaw bins, uniform / pose |
| --- | --- | --- | --- |
| 64 | 64 / 64 | 0 / 0 | 6.31 / 6.94 |
| 96 | 96 / 96 | 3 / 0 | 6.69 / 7.19 |
| 128 | 128 / 128 | 4 / 0 | 7.00 / 7.38 |
| 529 | 529 / 429 | 118 / 0 | 7.44 / 7.44 |

All compared selections have at least one crop-frustum-visible original per
chunk. At cap 128, the minimum selected originals per chunk changes from 19 to
21. At cap 529, the pose selector keeps 99.55% of existing sparse BA points with
at least one selected track and 93.90% with at least two tracks.

There are tradeoffs at smaller budgets: cap128 sparse-point one-track retention
is 76.92% uniform versus 76.32% pose; two-track retention is 35.99% versus 33.12%.
The current result demonstrates deduplication and direction diversity, not an
improvement in generated mesh quality. Sparse tracks are texture-biased proxies.
The fixed full-scene chunk layout also has a larger height estimate than the
first60 subset; these CPU results do not validate full-scene geometry quality.

The benchmark uses the actual `IphoneImageSelecter` methods, reads image headers
for crop geometry, and does not load DINO or other model weights. Rerun it with
a new output directory:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /kiri/envs/genrecon/bin/python scripts/benchmark_view_selection.py \
  --data-dir /kiri/output/scene_reconstruction/genrecon_pose_selection_20261027/data_all529 \
  --chunk-transforms /kiri/output/scene_reconstruction/genrecon_pose_selection_20261027/protocol/chunk_probe/chunk_transforms.json \
  --output /path/to/new/selection_benchmark --budgets 64 96 128 529
```

`selection_benchmark/report.json` contains exact metrics and source hashes;
`camera_pose_selection_128.png` compares selected positions and optical axes.

All 38 CPU tests passed, including 20 new selector/crop integration checks.
They exercise duplicates, turning/roll, differing K, rare-coverage rescue,
hard budgets and deficits, deterministic rigid/scale-invariant selection,
actual crop K/visibility, legacy uniform indices/RNG compatibility, JSON output,
and preservation of both crops after original-image selection.

## Full inference integration smoke test

`inference_first60_pose32/` runs the complete 512 pipeline with the existing
first60 dataset and a 32-original-image pose-coverage budget. It selects 32
originals / 64 crops, retains all 8 fixed chunks, and has 3–32 crop-frustum-visible
originals per chunk. Sampling, geometry/texture decoding and saving all finish
successfully in 202.43 seconds. The runtime commit is `6840922`, with no source
changes during execution.

The largest reported per-stage PyTorch allocated peak is 6.98 GiB. The 1-second
device-memory samples report 5562 MiB; these samples can miss brief peaks and
should not be substituted for the phase peak. This smoke test validates the
new selector's integration, not mesh quality or 529-frame generation capacity.
No additional GLB baking is needed for this check.

The raw mesh and CPU geometry/FFBA-coordinate checks are kept under that run,
along with `result/view_selection.json`, complete argv, source hashes, and logs.
