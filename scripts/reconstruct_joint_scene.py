#!/usr/bin/env python3
"""Joint geometry with frozen local observations and CPU MultiDiffusion state.

Reuse the independent baseline's exact layout, image pools, 2D anchors and DINO
tokens. One chunk at a time executes on the GPU; all chunks synchronize after
each complete denoising step. Shared occupancy and shape are decoded in a common
lattice with inflated spatial context and exclusive output ownership.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from reconstruct_local_scene import digest_file, transform_points, write_json, write_mesh
from joint_scene_inputs import load_joint_scene_inputs, validate_crop_bindings, shared_dense_noise, shared_sparse_noise


def check_frozen_inputs(inputs, data_dir, weights, pipeline_config):
    """Refuse changes to observations, calibration, model weights or config."""
    baseline = inputs["manifest"]
    baseline_root = Path(baseline["feature_cache"]["feature_path"]).parent
    settings = json.loads((baseline_root / "settings.json").read_text())["settings"]
    if digest_file(pipeline_config) != settings["pipeline_config_sha256"]:
        raise ValueError("Pipeline sampler configuration differs from the baseline")
    if inputs["validation"]["layout_sha256"] != settings["layout_sha256"]:
        raise ValueError("Layout differs from baseline snapshot")
    for checkpoint, old in settings["checkpoints"].items():
        path = weights / checkpoint
        if str(path.resolve()) != old["path"] or path.stat().st_size != old["size"] or path.stat().st_mtime_ns != old["mtime_ns"]:
            raise ValueError(f"Checkpoint changed: {checkpoint}")
    result_record = baseline_root.parents[1] / "results.json"
    camera_hashes = json.loads(result_record.read_text())["camera_file_sha256_captured_after_run"]
    for name, value in camera_hashes.items():
        if digest_file(data_dir / "colmap" / name) != value:
            raise ValueError(f"Camera geometry changed: {name}")
    if json.loads((data_dir / "world_transform.json").read_text()) != inputs["layout"]["world_transform"]:
        raise ValueError("World transform differs from fixed layout")
    old_images = {image["name"]: image for image in settings["images"]}
    for name, path in zip(inputs["frozen_image_names"], inputs["selected_paths"]):
        if digest_file(path) != old_images[name]["sha256"]:
            raise ValueError(f"Frozen image changed: {name}")
    return dict(camera_geometry_sha256=camera_hashes, original_image_hashes_verified=True,
                checkpoint_metadata_verified=True, pipeline_config_sha256=settings["pipeline_config_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", required=True, type=Path)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=Path("/kiri/models/genrecon"))
    parser.add_argument("--pipeline-config", type=Path, default=Path("configs/pipelines/texture.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--occ-threshold", type=float, default=-1.)
    parser.add_argument("--proj-batch-voxels", type=int, default=256)
    parser.add_argument("--decode-overlap", type=int, default=16)
    parser.add_argument("--decode-max-voxels", type=int, default=20000)
    parser.add_argument("--decode-max-chunks", type=int, default=1)
    args = parser.parse_args()
    if min(args.proj_batch_voxels, args.decode_max_voxels, args.decode_max_chunks) < 1 or args.decode_overlap < 0:
        parser.error("Invalid projection/decode budget")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    from inference.get_images import IphoneImageSelecter, _to_chunk0_extrinsics
    from genrecon.modules.sparse import SparseTensor
    from genrecon.pipelines.full_scene_images_to_3d import FullSceneImagesTo3DPipeline
    from genrecon.pipelines.samplers.cpu_multi_diff_orchestrator import CpuMultiDiffusionOrchestrator, move_sampling_tree

    inputs = load_joint_scene_inputs(args.layout, args.baseline_manifest)
    frozen_checks = check_frozen_inputs(inputs, args.data_dir, args.weights, args.pipeline_config)
    selector = IphoneImageSelecter(center_crop=False)
    all_cameras = selector._get_cameras(args.data_dir / "colmap" / "cameras.txt")
    by_name = {Path(c["img_path"]).name: c for c in all_cameras}
    crops = selector._prepare_scene_pool([by_name[name] for name in inputs["frozen_image_names"]])
    validate_crop_bindings(inputs, crops)
    features = inputs["feature_cache"]
    rel_t = [torch.tensor(c["relative_translation"], dtype=torch.float32) for c in inputs["chunks"]]
    baseline_settings = dict(seed=args.seed, occ_threshold=args.occ_threshold,
                             proj_batch_voxels=args.proj_batch_voxels,
                             baseline_manifest_sha256=digest_file(args.baseline_manifest),
                             inputs=inputs["validation"], frozen_checks=frozen_checks,
                             sampling_source_sha256={path:digest_file(path) for path in (
                                 "scripts/reconstruct_joint_scene.py", "scripts/joint_scene_inputs.py",
                                 "inference/get_images.py", "genrecon/pipelines/full_scene_images_to_3d.py",
                                 "genrecon/pipelines/samplers/cpu_multi_diff_orchestrator.py",
                                 "genrecon/pipelines/samplers/multi_diff_orchestrator.py",
                                 "genrecon/modules/cond_3D/projection.py",
                                 "genrecon/modules/cond_3D/sparse_aggregation_net.py",
                                 "genrecon/modules/cond_3D/aggregation_net.py")})
    identity = hashlib.sha256(json.dumps(baseline_settings, sort_keys=True).encode()).hexdigest()
    prior_settings = root / "settings.json"
    if prior_settings.exists() and json.loads(prior_settings.read_text())["sampling_identity"] != identity:
        raise ValueError("Refusing to resume different sampling inputs/settings")
    write_json(prior_settings, dict(sampling_identity=identity, settings=baseline_settings,
                                   git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                   decode=dict(overlap=args.decode_overlap, max_voxels=args.decode_max_voxels,
                                               max_chunks=args.decode_max_chunks, output_device="cpu")))
    write_json(root / "input_protocol.json", dict(**inputs["validation"], **frozen_checks,
               chunks=[dict(id=c["id"], originals=c["layout"]["pool_image_names"], crop_indices=c["crop_indices"],
                            anchor=c["anchor"], anchor_global_index=c["anchor_global_index"], relative_translation=c["relative_translation"])
                       for c in inputs["chunks"]],
               noise=dict(policy="one CPU random draw per global voxel; synchronize every complete step",
                          sparse_structure_seed=args.seed, shape_seed=args.seed+1),
               conditioning="exact independent baseline local pools, 2D anchors and cached DINO tokens"))

    phases = json.loads((root / "phases.json").read_text()) if (root / "phases.json").exists() else []
    sampling_steps = json.loads((root / "sampling_steps.json").read_text()) if (root / "sampling_steps.json").exists() else []
    start_all = time.monotonic()

    @contextmanager
    def phase(name):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        completed = False
        print(f"[phase] {name}", flush=True)
        try:
            yield
            completed = True
        finally:
            torch.cuda.synchronize()
            record = dict(name=name, seconds=time.monotonic()-start,
                          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          status="complete" if completed else "failed")
            phases.append(record)
            write_json(root / "phases.json", phases)
            print(f"[phase_complete] {record}", flush=True)

    def save_checkpoint(name, value):
        path = root / (name + ".pt")
        temp = root / (name + ".partial.pt")
        torch.save(value, temp)
        temp.replace(path)
        write_json(root / (name + ".json"), dict(sampling_identity=identity, sha256=digest_file(path)))

    def load_checkpoint(name):
        path, sidecar = root / (name + ".pt"), root / (name + ".json")
        if not sidecar.exists():
            return None
        info = json.loads(sidecar.read_text())
        if info["sampling_identity"] != identity or digest_file(path) != info["sha256"]:
            raise ValueError(f"Incompatible/damaged stage checkpoint: {name}")
        print(f"[resume] {name}", flush=True)
        # These are locally produced, hash-checked CPU tensor/SparseTensor trees.
        return torch.load(path, map_location="cpu", weights_only=False)

    def callback(stage):
        begin = time.monotonic()
        def record(step, time_value, states):
            tensor_values = [x if isinstance(x, torch.Tensor) else x.feats for x in states]
            if not all(t.device.type == "cpu" and torch.isfinite(t).all().item() for t in tensor_values):
                raise ValueError("Nonfinite or non-CPU synchronized state")
            row = dict(stage=stage, step=step+1, t=time_value, elapsed_seconds=time.monotonic()-begin,
                       cpu_resident_chunks=len(states), maximum_absolute_state=max(float(t.abs().max()) for t in tensor_values))
            sampling_steps.append(row)
            write_json(root / "sampling_steps.json", sampling_steps)
            print(f"[synchronized_step] {row}", flush=True)
        return record

    with phase("load_models"):
        pipeline = FullSceneImagesTo3DPipeline.from_finetuned(
            stage_models={"sparse_structure_flow_model":str(args.weights/"sparse_structure.pt"),
                          "shape_slat_flow_model_512":str(args.weights/"shape_slat.pt")},
            pipeline_config_file=str(args.pipeline_config))
        pipeline.proj_batch_voxels = args.proj_batch_voxels
        pipeline.to(torch.device("cuda"))
        write_json(root / "checkpoint_load_report.json", pipeline._checkpoint_load_report)
    ss_model = pipeline.models["sparse_structure_flow_model"]
    shape_model = pipeline.models["shape_slat_flow_model_512"]
    intr = torch.stack([c["intrinsics"] for c in crops])

    def make_conditions(kind, coords_list=None):
        model = ss_model if kind == "ss" else shape_model
        result = []
        for i, chunk in enumerate(inputs["chunks"]):
            with phase(f"{kind}_condition_chunk_{chunk['id']:03d}"):
                indices = chunk["crop_indices"]
                local_features = torch.from_numpy(features[indices]).unsqueeze(0)
                anchor_features = local_features[:, [indices.index(chunk["anchor_global_index"])]]
                matrix = torch.tensor(chunk["layout"]["M_original_to_chunk"], dtype=torch.float32)
                ext = torch.stack([_to_chunk0_extrinsics(crops[j]["c2w_blender"], matrix) for j in indices])
                ext, k = pipeline._prep_extrinsics_intrinsics(ext, intr[indices], model.dtype)
                arguments = dict(flow_model=model, scene_feats=local_features.to(model.dtype),
                                 cond2d_feats=anchor_features.to(model.dtype),scene_ext_c0=ext,scene_intr=k,
                                 rel_t=[torch.zeros(3)], R=model.resolution)
                if kind == "ss":
                    condition = pipeline._build_global_dense_cond(**arguments)[0]
                else:
                    condition = pipeline._build_global_sparse_cond(**arguments, coords_list=[coords_list[i].cuda()])[0]
                result.append(move_sampling_tree(condition, "cpu"))
                del condition, arguments, local_features, anchor_features, ext, k
            gc.collect()
            torch.cuda.empty_cache()
        return result

    with torch.no_grad():
        coords_list = load_checkpoint("shared_coords")
        if coords_list is None:
            states = load_checkpoint("shared_ss_latents")
            if states is None:
                conditions = load_checkpoint("ss_conditions")
                if conditions is None:
                    conditions = make_conditions("ss")
                    save_checkpoint("ss_conditions", conditions)
                generator = torch.Generator(device="cpu").manual_seed(args.seed)
                noise = shared_dense_noise(rel_t, ss_model.resolution, ss_model.in_channels,
                                           dtype=ss_model.dtype, generator=generator)
                with phase("sample_ss_cpu_state_global_sync"):
                    ss_model.to(pipeline.device)
                    with torch.autocast("cuda", dtype=ss_model.dtype):
                        states = CpuMultiDiffusionOrchestrator(pipeline.sparse_structure_sampler).sample(
                            ss_model, noise, [c["cond"] for c in conditions], [c["neg_cond"] for c in conditions], rel_t,
                            **pipeline.sparse_structure_sampler_params, execution_device=pipeline.device,
                            initialize_noise=False, step_callback=callback("ss"), tqdm_desc="Joint sparse structure")
                    ss_model.cpu()
                save_checkpoint("shared_ss_latents", states)
                del conditions, noise
            gc.collect()
            torch.cuda.empty_cache()
            with phase("decode_shared_ss_occupancy"):
                occupancy = pipeline.joint_decode_sparse_structure([s.cuda() for s in states], rel_t, 32)
                coords_list = [c.cpu() for c in pipeline.extract_coords_from_occ_logits(occupancy, 32, shape_model.resolution,
                                                                                     threshold=args.occ_threshold)]
                if any(len(c) == 0 for c in coords_list):
                    raise ValueError("Shared occupancy left an empty chunk")
            save_checkpoint("shared_coords", coords_list)
            del states, occupancy
            gc.collect()
            torch.cuda.empty_cache()

        shape_slats = load_checkpoint("shared_shape_slats")
        if shape_slats is None:
            conditions = load_checkpoint("shape_conditions")
            if conditions is None:
                conditions = make_conditions("shape", coords_list)
                save_checkpoint("shape_conditions", conditions)
            # An independent stage RNG makes checkpoint resume deterministic.
            generator = torch.Generator(device="cpu").manual_seed(args.seed+1)
            feature_noise = shared_sparse_noise(coords_list, rel_t, shape_model.resolution, shape_model.in_channels,
                                                dtype=torch.float32, generator=generator)
            noise = [SparseTensor(feats=f, coords=c) for f,c in zip(feature_noise,coords_list)]
            del feature_noise
            with phase("sample_shape_cpu_state_global_sync"):
                shape_model.to(pipeline.device)
                with torch.autocast("cuda", dtype=shape_model.dtype):
                    states = CpuMultiDiffusionOrchestrator(pipeline.shape_slat_sampler).sample(
                        shape_model, noise, [c["cond"] for c in conditions], [c["neg_cond"] for c in conditions], rel_t,
                        **pipeline.shape_slat_sampler_params, execution_device=pipeline.device,
                        initialize_noise=False, step_callback=callback("shape"), tqdm_desc="Joint shape SLat")
                shape_model.cpu()
            std = torch.tensor(pipeline.shape_slat_normalization["std"])[None]
            mean = torch.tensor(pipeline.shape_slat_normalization["mean"])[None]
            shape_slats = [move_sampling_tree(s * std + mean, "cpu") for s in states]
            save_checkpoint("shared_shape_slats", shape_slats)
            del states, conditions, noise
            gc.collect()
            torch.cuda.empty_cache()

        with phase("grouped_shared_shape_decode"):
            mesh, aabb, grid, ctx, meta = pipeline.joint_decode_shape_slats(
                [move_sampling_tree(s, pipeline.device) for s in shape_slats], 512, rel_t,
                max_chunks_per_group=args.decode_max_chunks, max_inflated_voxels=args.decode_max_voxels,
                overlap_r_in=args.decode_overlap, output_device="cpu", retain_subdivision_context=False)
            del ctx, meta, shape_slats
        with phase("export_global_geometry"):
            vertices = transform_points(mesh.vertices.numpy(), inputs["chunks"][0]["layout"]["M_chunk_to_original"])
            write_mesh(root / "mesh.ply", vertices, mesh.faces.numpy())
            if hasattr(mesh, "merge_report"):
                write_json(root / "decode_merge_report.json", mesh.merge_report)
            counts = dict(vertices=len(vertices), faces=len(mesh.faces))
            del mesh, vertices
    coverage = inputs["manifest"]["coverage"]
    write_json(root / "result.json", dict(status="complete", method="joint_cpu_state_multidiffusion_grouped_decode",
               sampling_identity=identity, coverage=coverage, local_input_pools_and_anchors_unchanged=True,
               resolution=512, decode_halo_coarse_voxels=args.decode_overlap,
               geometry_only=True, fill_holes=False, mesh=str(root/"mesh.ply"), mesh_sha256=digest_file(root/"mesh.ply"),
               mesh_counts=counts, occupancy_voxels_per_chunk=[len(c) for c in coords_list],
               noise_policy=dict(shared_per_global_voxel=True, ss_seed=args.seed, shape_seed=args.seed+1),
               phases=phases, seconds=time.monotonic()-start_all,
               diagnostic_scope="Shared-state/decoder consistency is not ground-truth reconstruction accuracy"))
    print(f"[complete] {counts}, {time.monotonic()-start_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
