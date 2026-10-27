# Joint full-scene geometry with CPU state

`scripts/reconstruct_joint_scene.py` compares joint generation against the
independent 429-image baseline without changing the layout or conditioning.
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

```bash
python scripts/run_profiled.py --output RUN/attempt_01 -- \
  python scripts/reconstruct_joint_scene.py \
  --layout BASELINE/planning/layout.json \
  --baseline-manifest BASELINE/attempt_01/result/manifest.json \
  --data-dir FULL_NORMALIZED_SCENE --output RUN/result \
  --proj-batch-voxels 256 --decode-overlap 16 \
  --decode-max-chunks 1 --decode-max-voxels 20000

python scripts/inspect_ffba_large_result.py \
  --result-dir RUN/result --data-dir FULL_NORMALIZED_SCENE \
  --source-dataset ORIGINAL_FFBA_SCENE --output RUN/inspection --threads 4

python scripts/compare_scene_runs.py \
  --baseline-inspection BASELINE/inspection \
  --candidate-inspection RUN/inspection \
  --camera-data-dir FULL_NORMALIZED_SCENE --output RUN/comparison
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
