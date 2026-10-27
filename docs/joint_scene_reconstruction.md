# Joint full-scene geometry with CPU state

The [429-image experiment report](joint_scene_429_results.md) records measured
resource use, matched-view comparisons, artifacts and the remaining ceiling defects.
This joint output is the [adopted baseline](current_baseline.md) for future work.

`scripts/reconstruct_joint_scene.py` runs joint generation using the original
independent experiment's frozen layout and conditioning.
It verifies the original 23 chunks, all 429 image hashes, 858 crop bindings,
per-chunk image pools, exact 2D anchors, camera files, checkpoint metadata,
sampler configuration and cached DINO features before loading models.

The GPU executes one chunk at a time. CPU holds conditions and denoising state.
All chunks complete an Euler step before native MultiDiffusion overlap averaging
is applied on CPU. Six numerical tests compare each synchronized step against
the original sampler, including CFG and sparse boundary-sensitive aggregation.
Rebuilding sparse tensors during transfers releases device-specific caches.

Shared initial noise is drawn once per global voxel, using CPU RNG seed 42 for
sparse structure and 43 for shape. The independent baseline used separate
chunk seeds; coherent noise is part of this joint method, so a single run does
not isolate every component of the change or establish multi-seed robustness.

Sparse-structure latents decode on one common canvas, producing shared occupancy.
Shape latents then synchronize at every denoising step. The shape decoder merges
the common coarse lattice, runs spatial groups with neighboring latent context,
and keeps faces according to the existing exclusive global ownership rule.
Defaults are one chunk per group, coarse halo 16, and a soft 20k inflated-voxel
budget. A single-chunk leaf can exceed that budget; actual counts are printed.

Geometry-only decoding discards texture subdivision context. Final lattice-key
deduplication runs on CPU with the same first representative, key ordering and
face ordering as the original GPU merge. Seven tests compare the merge numerically.
It exports diagnostics for predictions at shared lattice keys before choosing
the representative; these diagnostics are not ground-truth accuracy metrics.

The following example compares a new joint candidate to the adopted joint run.
`OBSERVATIONS` is the old independent experiment used only for fixed input pools,
anchors and the feature cache; `REFERENCE` is the adopted quality reference.
Use a fresh `RUN` directory, never the reference output directory.

```bash
OBSERVATIONS=/kiri/output/scene_reconstruction/genrecon_local_mesh_20261027
REFERENCE=/kiri/output/scene_reconstruction/genrecon_joint_429_20261027
FULL_NORMALIZED_SCENE=/kiri/output/scene_reconstruction/genrecon_pose_selection_20261027/data_all529
ORIGINAL_FFBA_SCENE=/kiri/529_prior_pose_and_k_moge3_20261021
RUN=/kiri/output/scene_reconstruction/genrecon_candidate_001

python scripts/run_profiled.py --output "$RUN/attempt_01" -- \
  python scripts/reconstruct_joint_scene.py \
  --layout "$OBSERVATIONS/planning/layout.json" \
  --baseline-manifest "$OBSERVATIONS/attempt_01/result/manifest.json" \
  --data-dir "$FULL_NORMALIZED_SCENE" --output "$RUN/result" \
  --proj-batch-voxels 256 --decode-overlap 16 \
  --decode-max-chunks 1 --decode-max-voxels 20000

python scripts/inspect_ffba_large_result.py \
  --result-dir "$RUN/result" --data-dir "$FULL_NORMALIZED_SCENE" \
  --source-dataset "$ORIGINAL_FFBA_SCENE" --output "$RUN/inspection" --threads 4

python scripts/compare_scene_runs.py \
  --baseline-inspection "$REFERENCE/inspection" \
  --candidate-inspection "$RUN/inspection" \
  --camera-data-dir "$FULL_NORMALIZED_SCENE" --output "$RUN/comparison"
```

Sampling checkpoints and conditions are hash checked and CPU resident. Changing
sampling code or inputs invalidates resume. Decode budgets can change without
resampling; decoded output metadata records them separately. Every GPU attempt
should have a fresh profiler directory while resuming the same result directory.
Freeze tracked source during each profiler run.

Use the full 529-camera evaluation dataset for both inspections to retain the
same eight camera views (0/75/151/226/302/377/453/528). The comparison checks
dimensions, K, camera IDs, image identity and coordinate transform metadata.
It measures gained/lost ray hits and surface changes; these are cross-run
differences, not ground-truth reconstruction accuracy. Shared final geometry
must not be cut into copies and reported as independent overlap agreement.
