# Full-scene local-mesh baseline: 529 source frames, 429 frozen inputs

The independent local baseline processed every one of the 429 previously
selected images. It completed on the 24 GB RTX 4090 without an OOM. The resulting
mesh shows substantial disagreement between independently generated chunks;
known global camera poses and exclusive ownership do not enforce consistent
generated geometry. It is a rough initialization candidate, not evidence that
simple local fusion already meets the reconstruction quality requirement.

Experiment root:
`/kiri/output/scene_reconstruction/genrecon_local_mesh_20261027`

## Inputs and protocol

- Full normalized scene: `../genrecon_pose_selection_20261027/data_all529`.
- Frozen selection: 429 original RGB images, expanded to 858 native iPhone crops.
- Common upright world and invertible original FFBA transform; scene units are
  retained, with no assumption that they are metres.
- 23 global cubes, side 2.8, stride 2.1, overlap 0.7; fixed floor -0.08, top 2.72.
  The layout contains 99.9849% of sampled trusted-depth points. Trusted depths
  helped audit the layout; generated geometry was conditioned on RGB and poses.
- Each cube receives all frozen images passing native crop-frustum support:
  14–202 original images per cube. There is no extra local image cap.
- All 429 images enter at least one local conditioning pool; 2,087 repeated
  image memberships across cubes. These are conditioning inputs, not a claim
  that every image contributes nonzero features at every generated point.
- Global DINO features cached once on CPU/disk; projection batches contain
  256 voxels, DINO batches 2 crops. SS + shape only, resolution 512, occupancy
  threshold -1, base seed 42 plus chunk ID. No texture model or 1024 RGB stack.
- One independently generated cube at a time, without cross-cube latent
  synchronization. Fixed lattice anchors determine triangle ownership. No ICP,
  scale fitting, hole filling or smoothing is used in the baseline.

## Resources and timing

| Stage | Time | GPU peak, sampled each second | CPU memory |
| --- | ---: | ---: | ---: |
| All 23 independent generations, including loading/cache/export | 474.42 s | 4,956 MiB (4.84 GiB) | sampled RSS 22.54 GiB |
| DINO cache, included above | 85.14 s | included above | included above |
| Per-cube runs combined, included above | 340.96 s | included above | included above |
| Fusion with exact cleanup and full overlap diagnostics | 553.80 s | CPU | peak RSS 13.40 GiB |
| Fusion with cleanup/diagnostics disabled | 157.99 s | CPU | peak RSS 2.23 GiB |
| Final coarse mesh process, including startup | 19.54 s | 10,802 MiB (10.55 GiB) | worker peak RSS 3.10 GiB |
| Complete mesh validation, inverse export and eight views | 51.29 s | CPU | peak RSS 8.31 GiB |

The fast fusion is a separate warm-cache run, concurrent with the tail of the
full diagnostic pass. These are measured stage times, not one measured
end-to-end fast-path invocation. Exact cleanup found **zero** duplicate vertices,
duplicate faces or degenerate faces, and moved no vertices. The fast mesh is
byte-identical to the original owned/unwelded mesh. Thus omitting exact cleanup
preserves this baseline's geometry. Diagnostics remain available separately.

CuMesh memory comes from CUDA allocations outside PyTorch. Device sampling
captures those allocations. The timeout wrapper's sampled parent RSS is not the
worker RSS; the table uses the simplification worker's own resource report.

## Geometry and consistency

Raw local outputs total 48,799,747 vertices / 105,105,508 faces. Ownership fusion
retains 30,846,645 vertices / 66,075,989 faces. The complete mesh has finite
vertices, valid indices, zero degenerate faces and zero unreferenced vertices.
Its inverse FFBA export preserves every face byte; maximum coordinate roundtrip
error is below 4.8e-7 scene units. Topology is not guaranteed watertight.

The raw overlap audit examines 66 spatially overlapping chunk pairs. A bounded
low-resolution probe selects shared camera views by visible common surface
area and pose diversity, without using depth/normal error to choose views.
65 pairs have common surface evidence; 193 of 196 view comparisons are nonempty,
using 115 distinct cameras and 7,459,178 repeated common pixels.

| Raw overlap consistency | Result |
| --- | ---: |
| Relative camera-Z difference, median | 3.56% |
| Relative camera-Z difference, P90 / P95 | 21.43% / 34.89% |
| Common pixels with relative depth difference >2% | 63.22% |
| Unsigned normal difference, median / P90 | 1.63° / 54.69° |
| Effective pairs with median depth difference >2% / >10% | 37/65 / 7/65 |

Relative depth is `abs(z_a-z_b)/min(z_a,z_b)`. These are consistency statistics,
**not ground-truth reconstruction accuracy**. Cameras/pixels repeat between
pair comparisons. Only both-valid first hits inside both context volumes enter
these statistics; missing and outside-overlap surfaces are excluded. Different
visible surfaces and occlusion can also cause disagreements.

The complete fused preview provides additional visible evidence: frames 75 and
302 show strips with no mesh hit on a continuous real ceiling; frame 453 shows
doorway/ceiling steps and anomalous planar surfaces. These problems already
exist before simplification. A no-hit strip alone does not measure a topological
hole count. The independent review and per-pair/view metrics are in
`audit_summary.json`.

## Coarse initializer artifact

The final CuMesh QEM export has 825,537 referenced vertices / 1,958,361 faces.
Inactive vertices left by CuMesh are removed before export. The approximately
35 MB untextured PLY in original FFBA coordinates is:

`coarse_final_inspection/mesh_ffba_original.ply`

It has finite coordinates, valid indices, zero degenerate faces and zero
unreferenced vertices. Against the complete fusion on the same eight camera
grids, relative depth difference P90 is 0.0742%, P99 0.661%; 0.681% of common
pixels differ by more than 1%. This checks simplification fidelity only.
Simplification does not repair inconsistent chunks. The initializer has not yet
been tested in a downstream GGGS optimization run.

Useful files under the experiment root:

- `results.json`: compact resource, coverage and artifact index.
- `attempt_01/run.json` and `attempt_01/result/manifest.json`: generation
  provenance, per-cube usage and outputs (generation commit `6989793`).
- `fusion/fusion_report.json`: complete ownership/cleanup/diagnostic results
  (fusion implementation commit `dbfd000`).
- `fusion_fast/equivalence_to_audited_fusion.json`: fast-path equivalence.
- `coarse_final_profile/run.json`: final simplification provenance (commit
  `2baf8f8`); no source changes occurred during either GPU profile.
- `inspection/geometry_contact_sheet.jpg`: all eight complete mesh comparisons.
- `fusion_vs_coarse_preview.jpg`: selected real / full / coarse comparisons.
- `coarse_final_inspection/comparison_to_complete_fusion.json`: coarse fidelity.
- `tests_console.log`: all 68 CPU tests passed.

## Implication for the next experiment

Local generation removes the immediate input-capacity obstacle at this scene
size. The next uncertainty is cross-chunk shape consistency. Simple ownership
fusion alone is visibly insufficient to establish a reliable full-scene mesh.

The next focused comparison should retain the exact global layout and 429-image
pool, then use native MultiDiffusion overlap synchronization with cached image
features, bounded projection batches and grouped shape decoding. Add CPU state
swapping if its measured working set requires it. This isolates the benefit of
joint generation before investing in a larger streaming implementation.

See [local_scene_reconstruction.md](local_scene_reconstruction.md) for CLI usage.
