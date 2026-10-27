"""CPU geometry, frozen-conditioning and exact shared-noise checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.joint_scene_inputs import (
    global_grid_layout,
    load_joint_scene_inputs,
    shared_dense_noise,
    shared_sparse_noise,
    validate_crop_bindings,
)


class SharedNoiseTest(unittest.TestCase):
    def test_three_quarter_stride_gives_exact_ss_overlap_and_world_coordinates(self):
        translations = [[0, 0, 0], [.75, 0, 0], [0, .75, 0]]
        offsets, shape, minimum = global_grid_layout(translations, 16)
        self.assertEqual(offsets.tolist(), [[0, 0, 0], [12, 0, 0], [0, 12, 0]])
        self.assertEqual(shape, (28, 28, 16))
        torch.manual_seed(42)
        noise = shared_dense_noise(translations, 16, 8, dtype=torch.float16)
        self.assertTrue(torch.equal(noise[0][:, :, 12:], noise[1][:, :, :4]))
        self.assertTrue(torch.equal(noise[0][:, :, :, 12:], noise[2][:, :, :, :4]))
        self.assertTrue(all(value.dtype == torch.float16 and value.device.type == "cpu" for value in noise))
        local_a = torch.tensor([13, 2, 5])
        local_b = torch.tensor([1, 2, 5])
        world_a = (local_a.double() + .5) / 16 + torch.tensor(translations[0]) - .5
        global_b = (local_b + offsets[1]).double()
        world_b = (global_b + .5) / 16 + minimum - .5
        torch.testing.assert_close(world_a, world_b, atol=0, rtol=0)

    def test_dense_noise_is_one_global_draw_and_independent_clones(self):
        translations = [[-.75, 0, 0], [0, 0, 0]]
        torch.manual_seed(42)
        expected = torch.randn(2, 3, 28, 16, 16)
        torch.manual_seed(42)
        actual = shared_dense_noise(translations, 16, 3, batch_size=2)
        self.assertTrue(torch.equal(actual[0], expected[:, :, :16]))
        self.assertTrue(torch.equal(actual[1], expected[:, :, 12:28]))
        unchanged = actual[1][0, 0, 0, 0, 0].clone()
        actual[0][0, 0, 12, 0, 0] += 1
        self.assertTrue(torch.equal(actual[1][0, 0, 0, 0, 0], unchanged))

    def test_explicit_generator_does_not_advance_default_rng(self):
        torch.manual_seed(17)
        state = torch.get_rng_state().clone()
        a = shared_dense_noise([[0, 0, 0]], 16, 2, generator=torch.Generator().manual_seed(42))
        b = shared_dense_noise([[0, 0, 0]], 16, 2, generator=torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_sparse_shape_noise_matches_shared_cells_with_preserved_row_order(self):
        coords = [torch.tensor([[0, 24, 4, 5], [0, 31, 6, 7], [1, 24, 4, 5], [0, 0, 0, 0]]),
                  torch.tensor([[0, 7, 6, 7], [0, 0, 4, 5], [1, 0, 4, 5], [0, 31, 8, 8]])]
        original = [value.clone() for value in coords]
        torch.manual_seed(42)
        actual = shared_sparse_noise(coords, [[0, 0, 0], [.75, 0, 0]], 32, 32)
        self.assertTrue(torch.equal(actual[0][0], actual[1][1]))
        self.assertTrue(torch.equal(actual[0][1], actual[1][0]))
        self.assertTrue(torch.equal(actual[0][2], actual[1][2]))
        self.assertFalse(torch.equal(actual[0][0], actual[0][2]))
        self.assertFalse(torch.equal(actual[0][3], actual[1][3]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(coords, original)))
        for value in actual:
            self.assertEqual(value.shape, (4, 32))
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(value.device.type, "cpu")
        # Independent reference: one draw per sorted global batch/xyz key.
        global_coords = torch.cat([coords[0], coords[1] + torch.tensor([0, 24, 0, 0])])
        unique, inverse = torch.unique(global_coords, dim=0, sorted=True, return_inverse=True)
        torch.manual_seed(42)
        expected = torch.randn(len(unique), 32)[inverse]
        self.assertTrue(torch.equal(torch.cat(actual), expected))

    def test_sparse_empty_chunks_and_duplicate_rows(self):
        coords = [torch.empty((0, 4), dtype=torch.long), torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])]
        result = shared_sparse_noise(coords, [[0, 0, 0], [.75, 0, 0]], 32, 4)
        self.assertEqual(result[0].shape, (0, 4))
        self.assertTrue(torch.equal(result[1][0], result[1][1]))
        empty = shared_sparse_noise([coords[0], coords[0]], [[0, 0, 0], [.75, 0, 0]], 32, 4)
        self.assertEqual([tuple(part.shape) for part in empty], [(0, 4), (0, 4)])

    def test_fractional_lattice_and_invalid_coordinates_raise(self):
        with self.assertRaisesRegex(ValueError, "voxel lattice"):
            global_grid_layout([[0, 0, 0], [.76, 0, 0]], 16)
        for bad in ([], [[0, 0]], [[np.nan, 0, 0]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                global_grid_layout(bad, 16)
        for bad in ([[0, -1, 0, 0]], [[0, 32, 0, 0]], [[0, .5, 0, 0]], [[-1, 1, 1, 1]]):
            with self.subTest(coords=bad), self.assertRaises(ValueError):
                shared_sparse_noise([bad], [[0, 0, 0]], 32, 4)


class FrozenConditioningTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.layout_path = self.root / "layout.json"
        self.manifest_path = self.root / "manifest.json"
        self.feature_path = self.root / "scene_features_512.npy"
        self.names = ["a.png", "b.png", "c.png"]
        selected = self.root / "selected.json"
        selected.write_text(json.dumps([str(self.root / name) for name in self.names]))
        chunks = []
        groups = []
        for cid, center, pool, indices, anchor in (
            (0, [0, 0, 1.4], self.names[:2], [0, 1, 2, 3], ("b.png", [0, 0, 8])),
            (1, [2.1, 0, 1.4], self.names[1:], [2, 3, 4, 5], ("c.png", [0, 4, 8])),
        ):
            inverse = np.eye(4)
            inverse[:3, :3] *= 2.8
            inverse[:3, 3] = center
            chunks.append(dict(id=cid, center=center, size=2.8, relative_translation=[cid * .75, 0, 0],
                               pool_image_names=pool, M_chunk_to_original=inverse.tolist(),
                               M_original_to_chunk=np.linalg.inv(inverse).tolist()))
            groups.append(dict(status="complete", chunk_ids=[cid], selected_image_names=pool,
                               crop_indices=indices, crop_count=len(indices),
                               anchor=dict(image_name=anchor[0], crop_box=anchor[1], projected_area=1.)))
        self.layout = dict(chunk_size_world=2.8, selection=dict(paths_file=str(selected),
                           frozen_global_original_image_count=3), chunks=chunks)
        self.layout_path.write_text(json.dumps(self.layout))
        features = np.arange(6 * 3 * 4, dtype=np.float32).reshape(6, 3, 4)
        np.save(self.feature_path, features)
        feature_info = dict(identity="frozen", shape=list(features.shape), dtype="float32",
                            feature_path=str(self.feature_path),
                            sha256=hashlib.sha256(self.feature_path.read_bytes()).hexdigest())
        (self.root / "feature_cache.json").write_text(json.dumps(feature_info))
        self.manifest = dict(identity="frozen", coordinate_frame="global_zup", layout_path=str(self.layout_path),
                             frozen_image_names=self.names, feature_cache=feature_info,
                             chunks=[dict(id=c["id"], center=c["center"], size=2.8) for c in chunks], groups=groups)
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.crops = [dict(img_path=str(self.root / name), crop_box=box)
                      for name in self.names for box in ((0, 0, 8), (0, 4, 8))]

    def tearDown(self):
        self.temporary.cleanup()

    def load(self):
        return load_joint_scene_inputs(self.layout_path, self.manifest_path)

    def rewrite_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def test_exact_pool_anchor_and_readonly_mmap_are_preserved(self):
        inputs = self.load()
        self.assertIsInstance(inputs["feature_cache"], np.memmap)
        self.assertFalse(inputs["feature_cache"].flags.writeable)
        self.assertEqual([chunk["crop_indices"] for chunk in inputs["chunks"]], [[0, 1, 2, 3], [2, 3, 4, 5]])
        self.assertFalse(inputs["validation"]["crop_bindings_validated"])
        returned = validate_crop_bindings(inputs, self.crops)
        self.assertIs(returned, inputs)
        self.assertEqual([chunk["anchor_global_index"] for chunk in inputs["chunks"]], [2, 5])
        self.assertTrue(inputs["validation"]["crop_bindings_validated"])
        self.assertEqual([chunk["anchor"] for chunk in inputs["chunks"]],
                         [group["anchor"] for group in self.manifest["groups"]])
        np.testing.assert_array_equal(inputs["feature_cache"][2], np.arange(24, 36).reshape(3, 4))

    def test_modified_cache_bytes_are_rejected(self):
        feature = np.load(self.feature_path)
        feature[0, 0, 0] += 1
        np.save(self.feature_path, feature)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.load()

    def test_local_pool_and_frozen_path_order_changes_are_rejected(self):
        self.manifest["groups"][0]["selected_image_names"] = ["b.png", "a.png"]
        self.rewrite_manifest()
        with self.assertRaisesRegex(ValueError, "pool/order"):
            self.load()
        self.manifest["groups"][0]["selected_image_names"] = ["a.png", "b.png"]
        self.rewrite_manifest()
        selected = Path(self.layout["selection"]["paths_file"])
        paths = json.loads(selected.read_text())
        selected.write_text(json.dumps(paths[::-1]))
        with self.assertRaisesRegex(ValueError, "path order"):
            self.load()

    def test_runtime_crop_order_anchor_box_and_index_changes_are_rejected(self):
        original = self.load()
        variants = []
        swapped = copy.deepcopy(self.crops)
        swapped[0], swapped[2] = swapped[2], swapped[0]
        variants.append(swapped)
        changed_box = copy.deepcopy(self.crops)
        changed_box[5]["crop_box"] = (0, 5, 8)
        variants.append(changed_box)
        changed_path = copy.deepcopy(self.crops)
        changed_path[0]["img_path"] = str(self.root / "another_directory" / "a.png")
        variants.append(changed_path)
        for crops in variants:
            with self.subTest(crops=crops), self.assertRaises(ValueError):
                validate_crop_bindings(copy.deepcopy(original), crops)
        original["chunks"][0]["crop_indices"] = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "pool/order"):
            validate_crop_bindings(original, self.crops)


if __name__ == "__main__":
    unittest.main()
