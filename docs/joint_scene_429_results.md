# Joint 429-image reconstruction, 2026-10-27

**Adopted baseline:** after inspecting the output, the user selected this joint
run as the formal basis for further development because it is visually closer
to the real scene. Future candidates should be compared against this run's full
mesh. See [the current baseline record](current_baseline.md).

The joint scheme completed on one RTX 4090 at the original 512 resolution.
Some regular chunk artifacts improve, especially the false stepped wall above
the doorway in frame 453. Metrics are mixed and ceiling voids remain or increase
in frames 75 and 302. These limitations remain part of the adopted reference;
synchronized generation alone has not resolved all reconstruction defects.

## Frozen inputs and implementation

Generation used commit `9cbaac21abf64de1745c20b54af718f565b35324` on the
user fork, branch `codex/first60-4090`. Tracked source stayed unchanged during
both profiled GPU runs.

- The same 23 spatial chunks, side 2.8 and stride 2.1 in scene units.
- All 429 selected original images, the same 858 crops, per-chunk image pools
  (14–202 originals), exact anchor images/crops, camera calibration and poses.
- The same model checkpoints and 512 shape decoder; 12 sparse-structure and
  12 shape sampling steps. No training, texture generation or hole filling.
- The original cached DINO features were hash checked and reused. Their
  extraction is excluded from this run's runtime. The baseline extraction took
  85.14 seconds; adding that number is not a measured cold-cache rerun.

Conditions and sampling state reside on CPU. GPU evaluates one chunk at a time;
all chunks finish each Euler step before overlap averaging. Both sparse structure
and shape use shared global noise and synchronize each step. Sparse structure
decodes on one common canvas. Shape decodes from a common coarse lattice, with
one chunk per group, a 16-cell coarse halo and exclusive global face ownership.
The 20,000-voxel group budget is soft: the largest actual inflated group had
26,908 input cells and still fit. Geometry-only output discards texture context;
the CPU merge deduplicates shared fine lattice keys without averaging positions.

This changes coherent-noise policy as part of the joint algorithm: sparse
structure seed 42, shape seed 43, versus separate chunk seeds in the independent
baseline. One run cannot isolate each component or establish seed robustness.
See [the method and commands](joint_scene_reconstruction.md).

## Runtime and output

| Measurement | Result |
| --- | ---: |
| Profiled reconstruction wall time | 342.91 s |
| Device memory, 1 Hz sampled peak | 13,506 MiB / 13.19 GiB |
| Process RSS, sampled peak | 18.08 GiB |
| Full output vertices / triangles | 28,258,066 / 60,775,997 |
| Optional simplified vertices / triangles | 812,671 / 1,976,794 |
| Simplification profiled wall time | 18.52 s |

The reconstruction runtime includes loading, conditioning, sampling, decoding
and export, but excludes image feature extraction and the subsequent inspection
and simplification. The independent baseline's generation used about 4.84 GiB
device memory; the joint decoder uses more memory. Baseline fusion and its heavy
pairwise audit were separate operations, so their combined timing is not a
controlled speed benchmark against this run.

| Reconstruction phase | Seconds |
| --- | ---: |
| Model loading | 31.79 |
| Sparse-structure conditions, all chunks | 18.19 |
| Sparse-structure sampling and synchronization | 77.86 |
| Shared occupancy decoding | 0.40 |
| Shape conditions, all chunks | 25.80 |
| Shape sampling and synchronization | 124.20 |
| Grouped shared shape decoding, including CPU merge | 38.77 |
| Geometry export | 2.81 |

Phase timers omit startup, frozen-input checks and minor driver overhead.
The full mesh has finite vertices, valid indices, zero degenerate triangles,
zero repeated-index triangles and zero unused vertices. These checks do not
establish watertightness, connectedness or geometric accuracy.

## Matched-view comparison

Both full meshes were rendered with all triangles at the same eight cameras:
0, 75, 151, 226, 302, 377, 453, 528. The comparison checks image identity,
calibration and coordinate-transform metadata against the common camera dataset.
Images are geometry-only shading at 500 × 680, using camera-Z ray depths.

| Observation | Independent local meshes → joint scheme |
| --- | --- |
| Frame 453 doorway | Large hanging stepped wall is substantially reduced; wall/ceiling junction resembles the RGB more closely |
| Frame 75 ceiling | Walls appear smoother, but missing ceiling patches increase; local floor artifacts remain |
| Frame 302 ceiling | Long regular seam changes, with larger irregular ceiling voids and a vertical split |
| Eight-view rays with no mesh hit | 1.4915% → 1.7521%, +0.2606 percentage points |

There are 36,335 newly covered rays and 43,423 lost rays out of 2,720,000.
No-hit fractions describe coverage from these cameras, not a ground-truth hole
count: some rays can correctly pass through openings or beyond the scene.
The preview `before_after_focus.jpg` uses full meshes, not simplified meshes.

Trusted BA-aligned MoGe camera-Z priors are available for six evaluation views:
0, 75, 151, 226, 377, 528. Frames 302 and 453 are excluded because their depth
alignment was not admitted. On the common prior-valid pixels hit by both meshes,
mean absolute relative depth residual changes from **8.1307% to 7.2735%**;
P90 changes from 14.7808% to 11.9954%. Five of six view means improve; frame 75
worsens. No scale or Sim3 fitting was performed for this evaluation. These are
residuals against a model prior, not ground-truth accuracy. The six-view subset
does not evaluate the important ceiling regression at 302 or all artifacts at 453.

## What the diagnostics establish

The final CPU lattice merge removes 106,883 duplicate vertices and **zero faces**.
At 106,775 shared lattice keys, 106,883 later group predictions before merging
have mean distance to their first representative of 8.27e-7 and maximum 0.001236
in normalized chunk coordinates
(multiply by 2.8 for scene units). This measures decoder consistency at the same
keys; it does not prove complete topology or accuracy and is not comparable to
the old independently generated overlap audit by simply cutting this mesh up.

For ceiling rays that changed from hit to no hit:

- Frame 75, top 130 rows, trusted-prior reference: 4,947 pixels; 79.40% of
  prior surface points already fall in occupied coarse cells and 100% have
  occupied support in a Chebyshev ±1-cell neighborhood.
- Frame 302, top 350 rows, baseline mesh as reference: 26,243 pixels; 100% of
  baseline surface points fall in occupied coarse cells. This reference is
  the previous prediction, not an independently trusted depth observation.

The coarse shape cell side is 0.0875 scene units. These checks make absent
coarse occupancy alone an insufficient explanation for these voids. They do not
identify a confirmed bug. The next diagnostic target is learned subdivision,
intersection/connectivity prediction and group ownership before the final merge.

Chunk 19 contributes no owned surface. Its 510 shared occupied cells within
the chunk cube all belong to neighbors under the global ownership rule.
Inspection finds very little admitted depth support in its exclusive corner;
the old 821,870 faces there are not evidence of correctness. Real surface loss
in this weakly observed region remains possible; details are saved separately.

## Optional simplified mesh

QEM simplifies the full output to about two million triangles and exports a
35.45 MB mesh in the original FFBA coordinate frame. Over the eight views,
common-hit relative depth differences from the full mesh have P90 0.1113% and
P99 0.9154%. This is simplification fidelity, not reconstruction accuracy.
Errors are concentrated: frame 453 has a 46.65% P99 relative difference near
thin/occluding geometry, despite its P90 being 0.3260%. Use the full mesh for
the reconstruction comparison; the small version is a convenient inspection
and initialization candidate, with that local simplification limitation.

## Artifacts and validation

Experiment root:
`/kiri/output/scene_reconstruction/genrecon_joint_429_20261027`

| Relative path under experiment root | Contents |
| --- | --- |
| `results.json` | Compact artifact and metric index |
| `attempt_01/run.json`, `console.log` | Reconstruction profiler and console |
| `result/settings.json`, `input_protocol.json` | Frozen-input and sampling identity |
| `result/phases.json`, `sampling_steps.json` | Phase and per-step records |
| `result/mesh.ply` | Full geometry, global Z-up frame at original FFBA scale |
| `inspection/mesh_ffba_original.ply` | Full geometry in original FFBA coordinates, 1.129 GB |
| `coarse_inspection/mesh_ffba_original.ply` | Optional simplified geometry in original FFBA coordinates |
| `before_after_focus.jpg` | RGB / old / new, frame 302 above frame 453 |
| `comparison/comparison_report.json` | Matched eight-view comparison |
| `observed_depth_comparison/observed_depth_report.json` | Six-view admitted-prior residuals |
| `ceiling_occupancy_diagnostic.json`, `.py` | Reproducible CPU occupancy diagnostic |
| `chunk19_diagnostic.json` | Empty-owned-region investigation |
| `coarse_inspection/comparison_to_full.json` | Simplification fidelity, including per-view tails |
| `tests_console.log` | 97 passing CPU tests |

The 97 tests include nonzero numerical dense/sparse sampling with CFG and
boundary-sensitive overlap, CPU/native lattice merge agreement, exact crop
bindings and comparison validation. CPU equality tests do not imply bitwise
equivalence across GPU kernels or validate learned geometry against truth.

Full global Z-up mesh SHA256:
`9cbbb837871a164c72272f544b810d037c323c99524937ca44ea34b6265f2ce1`.

Simplified global Z-up mesh SHA256:
`d51b0c40559552ab6108a63b63706b572e4e95dee026b26ff55fbe0e66bcbe3f`.
