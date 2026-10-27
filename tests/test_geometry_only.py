"""CPU checks for the geometry-only execution/export contract, without CUDA imports."""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def definition(path, name, namespace, class_name=None):
    source = ast.parse((ROOT / path).read_text())
    body = source.body
    if class_name:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    node = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[name]


class SparseStub:
    def __init__(self, feats, coords):
        self.feats, self.coords = feats, coords

    def replace(self, feats):
        return SparseStub(feats, self.coords)

    def __mul__(self, x):
        return self.replace(self.feats * x)

    def __add__(self, x):
        return self.replace(self.feats + x)

    def __sub__(self, x):
        return self.replace(self.feats - x)

    def __truediv__(self, x):
        return self.replace(self.feats / x)


class MeshStub:
    def __init__(self, events=None):
        self.vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        self.faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        self.events = events
        self.lattice_vertices = self.vertices.clone()

    def fill_holes(self):
        self.events.append("fill_holes")


class TexturedMeshStub(MeshStub):
    def __init__(self, vertices, faces, **kwargs):
        self.vertices, self.faces = vertices, faces
        self.__dict__.update(kwargs)


class GeometryOnlyTest(unittest.TestCase):
    def test_component_validation_does_not_require_texture_for_geometry(self):
        check = definition("genrecon/pipelines/images_to_3d.py", "_require_run_components", {}, "ImagesTo3DPipeline")
        required = []
        instance = SimpleNamespace(_require_components=lambda keys, attrs: required.append((keys, attrs)))
        check(instance, "512", geometry_only=True)
        keys, attrs = required.pop()
        self.assertIn("shape_slat_flow_model_512", keys)
        self.assertFalse(any("tex" in key for key in (*keys, *attrs)))
        check(instance, "512")
        keys, attrs = required.pop()
        self.assertIn("tex_slat_flow_model_512", keys)
        self.assertIn("tex_slat_decoder", keys)
        self.assertIn("tex_slat_sampler", attrs)
        with self.assertRaisesRegex(ValueError, "Invalid pipeline"):
            check(instance, "unknown", geometry_only=True)

    def make_run(self, geometry_only, sel_overrides=None, **run_kwargs):
        events = []
        mesh = MeshStub(events)
        shape_coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
        model = SimpleNamespace(dtype=torch.float32, resolution=32, in_channels=2)
        model.to = lambda device: events.append("model_to_device")
        model.cpu = lambda: events.append("model_to_cpu")
        models = {"sparse_structure_flow_model": model, "shape_slat_flow_model_512": model}
        if not geometry_only:
            models["tex_slat_flow_model_512"] = model
        args = dict(torch=torch, nullcontext=nullcontext, SparseTensor=SparseStub,
                    _vram_peak=lambda label: nullcontext(), MeshWithVoxel=TexturedMeshStub)

        class Orchestrator:
            def __init__(self, sampler):
                self.sampler = sampler

            def sample(self, model, noise, *args, **kwargs):
                events.append(self.sampler)
                if self.sampler == "ss_sampling":
                    return [torch.zeros(1)]
                return [SparseStub(torch.ones(1, 1), shape_coords)]

        args["MultiDiffusionOrchestrator"] = Orchestrator
        run = definition("genrecon/pipelines/full_scene_images_to_3d.py", "run", args, "FullSceneImagesTo3DPipeline")

        def encode(images, res):
            events.append(("encode", res))
            return torch.zeros(1, len(images), 1, 1)

        def dense_cond(model, scene, cond, *args, **kwargs):
            events.append(("dense_features", scene.flatten().tolist(), cond.flatten().tolist()))
            return [{"cond": None, "neg_cond": None}]

        instance = SimpleNamespace(
            default_pipeline_type="512", device=torch.device("cpu"), low_vram=True,
            models=models, sparse_structure_sampler="ss_sampling", shape_slat_sampler="shape_sampling",
            sparse_structure_sampler_params={}, shape_slat_sampler_params={},
            shape_slat_normalization={"std": [1.], "mean": [0.]},
            _require_run_components=lambda pipeline, **kwargs: events.append(("require", kwargs)),
            _encode_scene=encode,
            _prep_extrinsics_intrinsics=lambda ext, intr, dtype: (ext, intr),
            _build_global_dense_cond=dense_cond,
            _build_global_sparse_cond=lambda *args, **kwargs: [{"cond": None, "neg_cond": None}],
            _make_sparse_structure_noise=lambda *args: [torch.zeros(1)],
            joint_decode_sparse_structure=lambda *args: [torch.zeros(1)],
            extract_coords_from_occ_logits=lambda *args, **kwargs: [shape_coords],
            joint_decode_shape_slats=lambda *args, **kwargs: (
                mesh, torch.zeros(2, 3), torch.ones(3, dtype=torch.int64), object(), {}),
        )
        if not geometry_only:
            instance.tex_slat_sampler = "tex_sampling"
            instance.tex_slat_sampler_params = {}
            instance.tex_slat_normalization = {"std": [1.], "mean": [0.]}
            instance.pbr_attr_layout = {}
            instance.joint_decode_tex_slats = lambda *args: SparseStub(torch.ones(1, 6), shape_coords)
        sel = SimpleNamespace(chunk_indices=[0], scene_images_512=torch.zeros(1, 3, 2, 2),
                              cond2d_images_512=[torch.zeros(3, 2, 2)],
                              scene_extrinsics_c0=torch.eye(4)[None], scene_intrinsics=torch.eye(3)[None])
        if sel_overrides:
            sel.__dict__.update(sel_overrides)
        result, coords = run(instance, sel, [torch.zeros(3)], geometry_only=geometry_only, **run_kwargs)
        return result, coords, mesh, shape_coords, events

    def test_geometry_execution_with_no_texture_models_or_attributes(self):
        result, coords, mesh, expected_coords, events = self.make_run(True)
        self.assertIs(result, mesh)
        self.assertIs(coords[0], expected_coords)
        self.assertIn("ss_sampling", events)
        self.assertIn("shape_sampling", events)
        self.assertNotIn("tex_sampling", events)
        self.assertEqual(events[-1], "fill_holes")
        self.assertEqual(events[0], ("require", {"geometry_only": True}))
        self.assertFalse(hasattr(result, "lattice_vertices"))

    def test_cached_features_bypass_encoding_and_remain_unchanged(self):
        scene = torch.full((1, 1, 1, 1), 3., requires_grad=True)
        cond = torch.full((1, 1, 1, 1), 7., requires_grad=True)
        result, _, _, _, events = self.make_run(True, scene_features_512=scene,
                                               cond2d_features_512=cond, fill_holes=False)
        self.assertFalse(any(isinstance(event, tuple) and event[0] == "encode" for event in events))
        self.assertIn(("dense_features", [3.], [7.]), events)
        torch.testing.assert_close(scene, torch.full_like(scene, 3.))
        torch.testing.assert_close(cond, torch.full_like(cond, 7.))
        self.assertTrue(scene.requires_grad)
        self.assertIsNone(scene.grad)
        self.assertIsNone(cond.grad)

    def test_partial_cache_encodes_only_the_missing_input(self):
        for field, expected_values in (("scene_features_512", ([3.], [0.])),
                                       ("cond2d_features_512", ([0.], [3.]))):
            with self.subTest(field=field):
                *_, events = self.make_run(True, **{field: torch.full((1, 1, 1, 1), 3.)})
                self.assertEqual(events.count(("encode", 512)), 1)
                self.assertIn(("dense_features", *expected_values), events)

    def test_skip_hole_filling_preserves_lattice_vertex_correspondence(self):
        result, _, mesh, _, events = self.make_run(True, fill_holes=False)
        self.assertNotIn("fill_holes", events)
        self.assertIs(result.lattice_vertices, mesh.lattice_vertices)
        torch.testing.assert_close(result.vertices, result.lattice_vertices)

    def test_invalid_caches_and_hole_flag_are_rejected(self):
        invalid = [
            ("scene_features_512", [1.], TypeError),
            ("scene_features_512", torch.zeros(1, 1, 1), ValueError),
            ("scene_features_512", torch.zeros(1, 2, 1, 1), ValueError),
            ("cond2d_features_512", torch.zeros(2, 1, 1, 1), ValueError),
            ("scene_features_512", torch.zeros(1, 1, 0, 1), ValueError),
            ("scene_features_512", torch.ones(1, 1, 1, 1, dtype=torch.int64), ValueError),
            ("scene_features_512", torch.full((1, 1, 1, 1), float("nan")), ValueError),
            ("cond2d_features_512", torch.full((1, 1, 1, 1), float("inf")), ValueError),
            ("fill_holes", 0, ValueError),
        ]
        for key, value, error in invalid:
            with self.subTest(key=key, value=str(value)), self.assertRaises(error):
                self.make_run(True, **{key: value})
        with self.assertRaisesRegex(ValueError, "identical token/channel dimensions"):
            self.make_run(True, scene_features_512=torch.zeros(1, 1, 2, 1),
                          cond2d_features_512=torch.zeros(1, 1, 1, 1))
        for field, shape in (("scene_extrinsics_c0", (2, 4, 4)), ("scene_intrinsics", (2, 3, 3))):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "extrinsics and intrinsics"):
                self.make_run(True, scene_features_512=torch.zeros(1, 1, 1, 1),
                              sel_overrides={field: torch.zeros(shape)})
        with self.assertRaisesRegex(ValueError, "one view per selected chunk"):
            self.make_run(True, cond2d_features_512=torch.zeros(1, 1, 1, 1),
                          sel_overrides={"chunk_indices": [0, 1]})

    def test_existing_textured_execution_still_runs_texture(self):
        result, coords, mesh, expected_coords, events = self.make_run(False)
        self.assertIsInstance(result, TexturedMeshStub)
        self.assertIn("tex_sampling", events)
        self.assertEqual(events[-1], "fill_holes")
        self.assertIs(result.vertices, mesh.vertices)
        self.assertIs(coords[0], expected_coords)

    def test_plain_mesh_export_applies_transform_without_querying_attributes(self):
        ns = dict(torch=torch, np=np, trimesh=trimesh, MeshWithVoxel=TexturedMeshStub)
        definition("inference/transform_to_original.py", "_apply_transform", ns)
        save = definition("inference/transform_to_original.py", "save_mesh_to_original", ns)
        mesh = MeshStub()
        transform = torch.eye(4)
        transform[:3, :3] *= 2
        transform[:3, 3] = torch.tensor([3., -2., 1.])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.ply"
            save(target, mesh, transform)
            loaded = trimesh.load(target, process=False)
        np.testing.assert_allclose(loaded.vertices, mesh.vertices.numpy() * 2 + [3, -2, 1])
        np.testing.assert_array_equal(loaded.faces, mesh.faces.numpy())

    def test_textured_export_preserves_existing_pbr_path(self):
        writes = []
        ns = dict(torch=torch, np=np, trimesh=trimesh, MeshWithVoxel=TexturedMeshStub,
                  write_pbr_ply=lambda *args: writes.append(args))
        definition("inference/transform_to_original.py", "_apply_transform", ns)
        save = definition("inference/transform_to_original.py", "save_mesh_to_original", ns)
        plain = MeshStub()
        mesh = TexturedMeshStub(plain.vertices, plain.faces,
                                layout={"base_color": slice(0, 3), "metallic": slice(3, 4),
                                        "roughness": slice(4, 5), "alpha": slice(5, 6)})
        mesh.query_vertex_attrs = lambda: torch.tensor([[0.2, 0.4, 0.6, 0., 1., 1.]]).repeat(3, 1)
        save("ignored.ply", mesh, torch.eye(4))
        self.assertEqual(len(writes), 1)
        np.testing.assert_array_equal(writes[0][3], np.tile([51, 102, 153], (3, 1)))

    def test_runtime_assets_only_load_requested_decoders(self):
        loads = []
        ns = dict(image_feature_extractor=SimpleNamespace(Image=lambda: object()),
                  PBR_ATTR_LAYOUT={},
                  get_requested_stage_families=lambda keys: {"sparse_structure_flow_model", "shape_slat_flow_model"},
                  load_pretrained_model_from_args=lambda args, key, source: loads.append(key) or object(),
                  samplers=SimpleNamespace(Sampler=lambda: object()))
        init = definition("genrecon/pipelines/setup_utils.py", "init_runtime_assets", ns)
        with (ROOT / "configs/pipelines/texture.json").open() as source:
            config = json.load(source)["args"]
        config["image_cond_model"] = {"name": "Image", "args": {}}
        config["sparse_structure_sampler"] = {"name": "Sampler", "args": {}, "params": {}}
        config["shape_slat_sampler"] = {"name": "Sampler", "args": {}, "params": {}}
        instance = SimpleNamespace()
        init(instance, config, {"sparse_structure_flow_model", "shape_slat_flow_model_512"}, "unused")
        self.assertEqual(loads, ["sparse_structure_decoder", "shape_slat_decoder"])
        self.assertIsNone(instance.tex_slat_sampler)
        self.assertIsNone(instance.tex_slat_normalization)

    def test_cli_accepts_geometry_flag_without_texture_checkpoint(self):
        ns = dict(argparse=argparse, MODES={"Iphone": None}, __doc__="test")
        parser = definition("reconstruct_scene.py", "build_parser", ns)()
        args = parser.parse_args(["--mode", "Iphone", "--path", "input", "--output_path", "output",
                                  "--ss_ckpt", "ss.pt", "--shape_ckpt", "shape.pt", "--geometry_only"])
        self.assertTrue(args.geometry_only)
        self.assertIsNone(args.tex_ckpt)


if __name__ == "__main__":
    unittest.main()
