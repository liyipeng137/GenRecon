# Adopted reconstruction baseline

The user adopted **`joint-429-512-20261027`** as the formal reconstruction
baseline after visually comparing it to the captured scene. All further changes
should start from this implementation and compare their output to this joint run.
The previous independent-chunk experiment remains a historical comparison.

| Identity | Value |
| --- | --- |
| Frozen version tag | `baseline-joint-429-512-20261027` |
| Generation source commit | `9cbaac21abf64de1745c20b54af718f565b35324` |
| Original result documentation commit | `5be8191` |
| Machine-readable reference | [`configs/baselines/joint_429_512_20261027.json`](../configs/baselines/joint_429_512_20261027.json) |
| Fixed input | 429 original images, 858 crops, 23 chunks |
| Resolution / sampling steps | 512 / 12 sparse-structure + 12 shape |
| Shared noise seeds | sparse structure 42, shape 43 |
| Decoder groups / halo / soft voxel budget | 1 chunk / 16 coarse cells / 20,000 |
| Full reference geometry | 28,258,066 vertices, 60,775,997 triangles |

The tag includes the adoption documentation and registry. The generation source
commit identifies the code that actually produced the reference. Promotion does
not regenerate or modify its geometry. See [the method](joint_scene_reconstruction.md)
and [the measured results](joint_scene_429_results.md).

## Reference artifacts

Local artifact root:
`/kiri/output/scene_reconstruction/genrecon_joint_429_20261027`

The full mesh and inspection are the quality reference. The two-million-face
version is an optional initialization/inspection derivative with local
simplification errors; it does not replace the full comparison reference.

| Relative path under artifact root | Use |
| --- | --- |
| `result/mesh.ply` | Full geometry, global Z-up coordinates at original FFBA scale |
| `inspection/mesh_ffba_original.ply` | Full geometry, original FFBA coordinates |
| `inspection/` | Fixed eight-view full-mesh renders, depth and hit masks |
| `coarse_inspection/mesh_ffba_original.ply` | Optional simplified derivative |
| `before_after_focus.jpg` | RGB / independent / joint visual comparison |
| `result/input_protocol.json`, `settings.json` | Frozen observations and generation identity |

Large meshes, camera data, caches and weights remain in external storage. Git
records paths, hashes and provenance in the registry. Relocating an artifact
should preserve its recorded hash. Keep the reference tree unchanged; in
particular, do not pass its `result` directory as a candidate `--output`, since
the reconstruction driver's resume path writes metadata and decoded meshes.

## Future comparison protocol

The [joint workflow commands](joint_scene_reconstruction.md) now compare new
candidates against this joint reference. The two similarly named CLI arguments
have different purposes:

- `reconstruct_joint_scene.py --baseline-manifest` continues to use
  `genrecon_local_mesh_20261027/attempt_01/result/manifest.json`. It supplies
  frozen input pools, anchors and the DINO cache, not the quality reference.
- `compare_scene_runs.py --baseline-inspection` uses
  `genrecon_joint_429_20261027/inspection`. This is the adopted quality reference.

Use the same full 529-camera evaluation dataset and cameras
0/75/151/226/302/377/453/528. Verify its `colmap/cameras.txt`, `colmap/images.txt`
and `world_transform.json` hashes against the registry before a controlled
comparison: the original inspection reports do not serialize full camera R/t.
Keep the 429 observations, layout and calibration fixed when attributing changes
to the algorithm. Record input changes explicitly when an experiment requires
different inputs. Always use a fresh candidate output directory.

Compare images and geometry alongside coverage and prior-depth residuals.
The user's adoption reflects the closer visual match; it is not a claim that
every metric improves. The increased ceiling voids at 75/302, weakly observed
corner, single-seed evaluation, and simplified-mesh errors remain documented
limitations and targets for later work. Model checkpoint identity in this run
was verified by path, size and mtime, not by checkpoint content hashes.
