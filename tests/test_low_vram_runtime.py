"""CPU checks for scheduling/math changes; no model downloads or CUDA setup.

The production modules also import optional sparse/CUDA backends. Loading the
actual definitions via AST keeps these checks usable before those extensions
are built, while exercising the implementation rather than a copied version.
Run with: python -m unittest discover -s tests -p 'test_low_vram_runtime.py'
"""

import ast
import json
import os
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
from typing import List, Optional, Union
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def definitions(relative_path, names, namespace):
    path = ROOT / relative_path
    source = ast.parse(path.read_text())
    nodes = []
    for node in source.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            nodes.append(node)
    namespace["__file__"] = str(path)
    tree = ast.Module(body=nodes, type_ignores=[])
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def class_method(relative_path, class_name, method_name, namespace):
    path = ROOT / relative_path
    source = ast.parse(path.read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


class FeatureBatchingTest(unittest.TestCase):
    def make_extractor(self, batch_size):
        ns = definitions(
            "genrecon/modules/image_feature_extractor.py",
            {"DinoV3FeatureExtractor"},
            dict(torch=torch, F=F, np=np, Image=Image, Union=Union, List=List, Optional=Optional),
        )
        extractor = ns["DinoV3FeatureExtractor"].__new__(ns["DinoV3FeatureExtractor"])
        extractor.batch_size = batch_size
        extractor.image_size = 4
        extractor.model = SimpleNamespace(
            embeddings=SimpleNamespace(patch_embeddings=SimpleNamespace(weight=torch.empty(1)))
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        extractor.transform = lambda batch: (batch - mean) / std
        extractor.calls = []

        def encode(self, batch):
            self.calls.append(batch.shape[0])
            return batch.flatten(2).transpose(1, 2).contiguous()

        extractor.extract_features = MethodType(encode, extractor)
        return extractor

    def test_all_120_crops_encoded_in_order_with_small_batches(self):
        image = torch.arange(120 * 3 * 4 * 4).reshape(1, 120, 3, 4, 4).float() / 10000
        batched = self.make_extractor(2)
        unbatched = self.make_extractor(None)
        result = batched.encode_batched(image, output_device="cpu")
        expected = unbatched(image)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        self.assertEqual(batched.calls, [2] * 60)
        self.assertEqual(result.shape, (1, 120, 16, 3))
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.dtype, torch.float32)

    def test_batched_scene_dimensions_and_pil_tail_preserved(self):
        image = torch.rand(2, 3, 3, 4, 4)
        batched, unbatched = self.make_extractor(4), self.make_extractor(None)
        torch.testing.assert_close(batched(image), unbatched(image), atol=0, rtol=0)
        self.assertEqual(batched.calls, [4, 2])
        images = [Image.new("RGB", (7, 5), color=(i * 50, 20, 40)) for i in range(5)]
        batched, unbatched = self.make_extractor(2), self.make_extractor(None)
        torch.testing.assert_close(batched(images), unbatched(images), atol=0, rtol=0)
        self.assertEqual(batched.calls, [2, 2, 1])

    def test_invalid_batch_and_empty_input_fail(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.make_extractor(size)(torch.zeros(1, 1, 3, 4, 4))
        with self.assertRaises(ValueError):
            self.make_extractor(2)([])


class SparseStub:
    def __init__(self, feats, coords=None):
        self.feats, self.coords = feats, coords

    def replace(self, feats, coords=None):
        return SparseStub(feats, self.coords if coords is None else coords)


class SparseAggregationTest(unittest.TestCase):
    def setUp(self):
        ns = definitions(
            "genrecon/modules/cond_3D/sparse_aggregation_net.py",
            {"MultiViewFeatAggregator"},
            dict(torch=torch, nn=nn, F=F, sp=SimpleNamespace(SparseTensor=SparseStub)),
        )
        torch.manual_seed(17)
        self.model = ns["MultiViewFeatAggregator"](7).double()
        # The published architecture starts its final layer at zero. Nonzero
        # trained-like weights are essential to test both MLP branches.
        for parameter in self.model.parameters():
            nn.init.normal_(parameter, std=0.15)

    def reference(self, feats, mask):
        mf = mask.unsqueeze(-1)
        count = mf.sum(1, keepdim=True).clamp_min(1)
        mu = (feats * mf).sum(1, keepdim=True) / count
        var = ((feats - mu).square() * mf).sum(1, keepdim=True) / count
        inputs = torch.cat((feats, mu.expand_as(feats), var.expand_as(feats)), dim=-1)
        prime = self.model.feature_mlp(inputs)
        logits = self.model.weight_mlp(inputs).masked_fill(~mf, torch.finfo(feats.dtype).min)
        weights = logits.softmax(1)
        weights = torch.where((~mask).all(1, keepdim=True).unsqueeze(-1), 0, weights)
        return (mu + (prime * weights).sum(1, keepdim=True)).squeeze(1)

    def test_split_linear_outputs_and_gradients_match_concatenated_math(self):
        feats = torch.randn(9, 5, 7, dtype=torch.float64, requires_grad=True)
        mask = torch.rand(9, 5) > 0.35
        mask[0] = False
        expected = self.reference(feats, mask)
        expected.square().sum().backward()
        feature_grad = feats.grad.clone()
        parameter_grads = [p.grad.clone() for p in self.model.parameters()]
        feats.grad = None
        self.model.zero_grad()
        actual = self.model(SparseStub(feats), mask).feats
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual.square().sum().backward()
        torch.testing.assert_close(feats.grad, feature_grad, atol=1e-12, rtol=1e-12)
        for parameter, expected_grad in zip(self.model.parameters(), parameter_grads):
            torch.testing.assert_close(parameter.grad, expected_grad, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.equal(actual[0], torch.zeros_like(actual[0])))

    def test_low_precision_output_dtype_and_invalid_voxels(self):
        mask = torch.rand(9, 5) > 0.35
        mask[0] = False
        for dtype, atol in ((torch.bfloat16, 0.02), (torch.float16, 0.003)):
            with self.subTest(dtype=dtype):
                self.model.to(dtype=dtype)
                feats = torch.randn(9, 5, 7).to(dtype)
                actual = self.model(SparseStub(feats), mask).feats
                expected = self.reference(feats, mask)
                self.assertEqual(actual.dtype, dtype)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.equal(actual[0], torch.zeros_like(actual[0])))
                torch.testing.assert_close(actual, expected, atol=atol, rtol=0.02)


class PublishedCheckpointConfigTest(unittest.TestCase):
    def setUp(self):
        self.ns = definitions(
            "genrecon/pipelines/setup_utils.py",
            {"PUBLISHED_CHECKPOINT_CONFIGS", "resolve_train_config_path", "load_train_config"},
            dict(os=os, json=json, Path=Path),
        )

    def test_three_official_standalone_names_use_correct_repo_architectures(self):
        expected = {
            "sparse_structure.pt": ("SparseStructureFlowModel", 8),
            "shape_slat.pt": ("ElasticSLatFlowModel", 32),
            "texture_slat.pt": ("ElasticSLatFlowModel", 64),
        }
        with tempfile.TemporaryDirectory() as root:
            for name, (model_name, in_channels) in expected.items():
                with self.subTest(name=name):
                    cfg = self.ns["load_train_config"](str(Path(root) / "weights" / name))
                    self.assertEqual(cfg["models"]["denoiser"]["name"], model_name)
                    self.assertEqual(cfg["models"]["denoiser"]["args"]["in_channels"], in_channels)

    def test_existing_run_and_explicit_config_take_priority(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            run = root / "run"
            (run / "ckpts").mkdir(parents=True)
            (run / "config.json").write_text('{"origin": "run"}')
            explicit = root / "other.json"
            explicit.write_text('{"origin": "explicit"}')
            checkpoint = str(run / "ckpts" / "shape_slat.pt")
            self.assertEqual(self.ns["load_train_config"](checkpoint)["origin"], "run")
            self.assertEqual(self.ns["load_train_config"](checkpoint, str(explicit))["origin"], "explicit")

    def test_unknown_names_and_missing_explicit_config_do_not_silently_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                self.ns["load_train_config"](str(Path(root) / "weights" / "custom.pt"))
            with self.assertRaises(FileNotFoundError):
                self.ns["load_train_config"](str(Path(root) / "shape_slat.pt"), str(Path(root) / "missing.json"))


class SparseStructureResolutionTest(unittest.TestCase):
    def test_released_latent_grid_passes_model_contract_and_decodes_to_32_grid(self):
        cfg = json.loads((ROOT / "configs/gen/ss_flow_img/genrecon.json").read_text())
        args = cfg["models"]["denoiser"]["args"]
        model = SimpleNamespace(
            resolution=args["resolution"], in_channels=args["in_channels"], dtype=torch.float32,
            input_layer=nn.Linear(args["in_channels"], 12),
            out_layer=nn.Linear(12, args["out_channels"]),
            pe_mode="rope", share_mod=False, proj_linears=None, blocks=[],
            t_embedder=lambda t: torch.zeros(len(t), 12),
        )
        pipeline = SimpleNamespace(device=torch.device("cpu"), low_vram=False)
        make_noise = class_method(
            "genrecon/pipelines/full_scene_images_to_3d.py", "FullSceneImagesTo3DPipeline",
            "_make_sparse_structure_noise", dict(torch=torch),
        )
        noise = make_noise(pipeline, model, 2)
        self.assertEqual(noise[0].shape, (1, 8, 16, 16, 16))
        forward = class_method(
            "genrecon/models/sparse_structure_flow.py", "SparseStructureFlowModel", "forward",
            dict(torch=torch, F=F, manual_cast=lambda x, dtype: x.to(dtype)),
        )
        cond = {"cond_2D": torch.zeros(1, 2, 3), "cond_3D": torch.zeros(1, model.resolution**3, 3)}
        self.assertEqual(forward(model, noise[0], torch.zeros(1), cond).shape, noise[0].shape)
        # Before the fix, FullScene used decoded occupancy res=32 here and
        # immediately failed this assertion in the actual flow forward.
        with self.assertRaisesRegex(AssertionError, "Input shape mismatch"):
            forward(model, torch.zeros(1, 8, 32, 32, 32), torch.zeros(1), cond)

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = nn.Parameter(torch.ones(()))

            def forward(self, z):
                return F.interpolate(z.mean(1, keepdim=True), scale_factor=4, mode="nearest")

        pipeline.models = {"sparse_structure_decoder": Decoder()}
        decode = class_method(
            "genrecon/pipelines/full_scene_images_to_3d.py", "FullSceneImagesTo3DPipeline",
            "joint_decode_sparse_structure", dict(torch=torch),
        )
        occupancy = decode(pipeline, noise, [torch.zeros(3), torch.tensor([0.5, 0., 0.])], 32)
        self.assertEqual([tuple(x.shape) for x in occupancy], [(1, 1, 32, 32, 32)] * 2)


if __name__ == "__main__":
    unittest.main()
