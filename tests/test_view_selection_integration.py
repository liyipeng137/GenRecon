"""Actual image-selector integration, with small local PNGs and CPU tensors."""

import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from inference.get_images import IphoneImageSelecter
from inference.view_selection import chunk_projection_areas


FLIP = torch.diag(torch.tensor([1., -1., -1., 1.]))


class ImageSelectionIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def camera(self, name, *, size=(144, 96), position=(0, 0, -1), intrinsic=None):
        width, height = size
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        pixels[..., 0] = np.linspace(0, 255, width).astype(np.uint8)
        pixels[..., 1] = np.linspace(0, 255, height).astype(np.uint8)[:, None]
        pixels[..., 2] = 80
        image_path = self.root / name
        Image.fromarray(pixels).save(image_path)
        c2w_cv = torch.eye(4)
        c2w_cv[:3, 3] = torch.tensor(position)
        return {
            "img_path": str(image_path), "c2w_blender": c2w_cv @ FLIP,
            "intrinsics": torch.tensor(intrinsic if intrinsic is not None else
                                       [[.8, 0, .5], [0, .8, .5], [0, 0, 1.]], dtype=torch.float32),
        }

    def test_two_crops_share_pose_and_are_retained_after_original_deduplication(self):
        camera = self.camera("wide.png")
        duplicate = {**camera, "img_path": str(self.root / "repeat.png")}
        Image.open(camera["img_path"]).save(duplicate["img_path"])
        selector = IphoneImageSelecter(view_selection="pose_coverage", min_views_per_chunk=1)
        raw = selector._sample_scene_cameras([camera, duplicate], 4, random.Random(8), [torch.eye(4)])
        self.assertEqual(len(raw), 1)
        crops = selector._prepare_scene_pool(raw)
        self.assertEqual(len(crops), 2)
        torch.testing.assert_close(crops[0]["c2w_blender"], crops[1]["c2w_blender"])
        self.assertNotEqual(crops[0]["crop_box"], crops[1]["crop_box"])
        self.assertFalse(torch.equal(crops[0]["intrinsics"], crops[1]["intrinsics"]))

    def test_parent_coverage_is_maximum_of_actual_crop_areas(self):
        camera = self.camera("wide.png", size=(300, 100), position=(0, 0, -2),
                             intrinsic=[[2 / 3, 0, .5], [0, 2, .5], [0, 0, 1.]])
        selector = IphoneImageSelecter()
        crops = selector._prepare_scene_pool([camera])
        crop_areas = chunk_projection_areas(
            np.stack([crop["c2w_blender"].numpy() for crop in crops]),
            np.stack([crop["intrinsics"].numpy() for crop in crops]), np.eye(4)[None])
        actual = selector._camera_chunk_areas([camera], [torch.eye(4)])
        np.testing.assert_allclose(actual[0], crop_areas.max(axis=0), atol=1e-12)
        raw = chunk_projection_areas(camera["c2w_blender"].numpy()[None],
                                     camera["intrinsics"].numpy()[None], np.eye(4)[None])
        self.assertGreater(raw[0, 0], .4)
        self.assertLess(actual[0, 0], .4)

    def test_center_crop_coverage_uses_exact_loaded_intrinsics(self):
        camera = self.camera("portrait.png", size=(80, 160), position=(0, 0, -1),
                             intrinsic=[[.8, 0, .4], [0, .8, .6], [0, 0, 1.]])
        selector = IphoneImageSelecter(center_crop=True)
        areas = selector._camera_chunk_areas([camera], [torch.eye(4)])
        processed_k, images_512, _ = selector._process_images([camera["img_path"]],
                                                            camera["intrinsics"][None])
        # Largest square around the principal point: S=64, left=0, top=64.
        expected = torch.tensor([[1., 0, .5], [0, 2., .5], [0, 0, 1.]])
        torch.testing.assert_close(processed_k[0], expected, atol=2e-7, rtol=0)
        projected = chunk_projection_areas(camera["c2w_blender"].numpy()[None],
                                           processed_k.numpy(), np.eye(4)[None])
        np.testing.assert_allclose(areas, projected, atol=1e-12)
        self.assertEqual(images_512.shape, (1, 3, 512, 512))
        # Neither visibility probing nor loading mutates the source K.
        self.assertAlmostEqual(float(camera["intrinsics"][0, 2]), .4, places=6)

    def test_uniform_retains_historical_filter_and_linspace_choices(self):
        cameras = [self.camera(f"frame_{i}.png", position=(x, 0, -1.5))
                   for i, x in enumerate(np.linspace(-2, 2, 9))]
        selector = IphoneImageSelecter(view_selection="uniform")
        distant = torch.eye(4)
        distant[0, 3] = 100
        for transforms in (None, [torch.eye(4)], [distant]):
            for budget in (1, 3, 12):
                with self.subTest(transforms=transforms, budget=budget):
                    old_pool = list(cameras)
                    if transforms is not None:
                        visible = [camera for camera in cameras if any(
                            selector._check_acceptance(FLIP @ torch.linalg.inv(t @ camera["c2w_blender"]),
                                                       camera["intrinsics"]) for t in transforms)]
                        if visible:
                            old_pool = visible
                    if len(old_pool) > budget:
                        indices = np.linspace(0, len(old_pool) - 1, budget).round().astype(int)
                        expected = [old_pool[i] for i in indices]
                    else:
                        expected = old_pool
                    rng = random.Random(11)
                    before_rng = rng.getstate()
                    actual = selector._sample_scene_cameras(cameras, budget, rng, transforms)
                    self.assertEqual([c["img_path"] for c in actual], [c["img_path"] for c in expected])
                    self.assertEqual(rng.getstate(), before_rng)
                    for selected, previous in zip(actual, expected):
                        torch.testing.assert_close(selected["intrinsics"], previous["intrinsics"], atol=0, rtol=0)
                        torch.testing.assert_close(selected["c2w_blender"], previous["c2w_blender"], atol=0, rtol=0)

    def test_get_images_obeys_original_budget_and_cond2d_stays_in_pool(self):
        cameras = [self.camera(f"frame_{i}.png", position=(i * .2, 0, -1)) for i in range(3)]
        selector = IphoneImageSelecter(view_selection="pose_coverage", min_views_per_chunk=1)
        selector._get_cameras = lambda _: cameras
        second = torch.eye(4)
        second[0, 3] = -.4
        output = self.root / "result"
        selected = selector.get_images([torch.eye(4), second], self.root, 2, output, seed=19)
        report = json.loads((output / "view_selection.json").read_text(),
                            parse_constant=lambda value: self.fail(f"Non-finite JSON value: {value}"))
        dump = json.loads((output / "cameras.json").read_text())
        self.assertEqual(report["selected_original_images"], 2)
        self.assertEqual(report["selected_crop_views"], 4)
        self.assertEqual(len(set(camera["img_path"] for camera in dump["scene"])), 2)
        self.assertEqual(selected.scene_images_512.shape, (4, 3, 512, 512))
        self.assertEqual(selected.scene_images_1024.shape, (4, 3, 1024, 1024))
        self.assertEqual(selected.scene_intrinsics.shape, (4, 3, 3))
        self.assertEqual(selected.chunk_indices, [0, 1])
        for tensor in (selected.scene_images_512, selected.scene_images_1024,
                       selected.scene_intrinsics, selected.scene_extrinsics_c0):
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertEqual(tensor.device.type, "cpu")
        self.assertGreaterEqual(float(selected.scene_images_512.min()), 0)
        self.assertLessEqual(float(selected.scene_images_512.max()), 1)
        for chunk in dump["chunks"]:
            condition = chunk["cond2d_view"]
            matches = [view for view in dump["scene"] if view["img_path"] == condition["img_path"]
                       and np.allclose(view["intrinsics"], condition["intrinsics"], atol=1e-7)
                       and np.allclose(view["extrinsics_c0"], condition["extrinsics_c0"], atol=1e-7)]
            self.assertTrue(matches)
        self.assertTrue(all(item["image_path"] in report["selected_image_paths"]
                            for item in report["cond2d_views"]))
        json.dumps(report, allow_nan=False)

    def test_center_crop_cond2d_uses_visible_actual_crop_instead_of_raw_frame(self):
        intrinsic = [[2 / 3, 0, .5], [0, 2, .5], [0, 0, 1.]]
        # A sees chunk 0 in the wide original, but its center square misses
        # enough of it to fail acceptance. B has the opposite acceptance.
        # A remains useful for chunk 1, so both survive global selection.
        camera_a = self.camera("camera_a.png", size=(300, 100), position=(.65, 0, -2),
                               intrinsic=intrinsic)
        camera_b = self.camera("camera_b.png", size=(300, 100), position=(0, 0, -3),
                               intrinsic=intrinsic)
        selector = IphoneImageSelecter(center_crop=True, min_views_per_chunk=1)
        selector._get_cameras = lambda _: [camera_a, camera_b]
        second = torch.eye(4)
        second[0, 3] = -.65
        raw_areas = chunk_projection_areas(
            np.stack([camera["c2w_blender"].numpy() for camera in (camera_a, camera_b)]),
            np.stack([camera["intrinsics"].numpy() for camera in (camera_a, camera_b)]), np.eye(4)[None])
        crop_areas = selector._camera_chunk_areas([camera_a, camera_b], [torch.eye(4)])
        self.assertGreater(raw_areas[0, 0], .4)
        self.assertLess(crop_areas[0, 0], .4)
        self.assertLess(raw_areas[1, 0], .4)
        self.assertGreater(crop_areas[1, 0], .4)
        output = self.root / "center_result"
        selected = selector.get_images([torch.eye(4), second], self.root, 2, output, seed=0)
        report = json.loads((output / "view_selection.json").read_text())
        self.assertEqual(report["selected_original_images"], 2)
        self.assertEqual(report["selected_crop_views"], 2)
        self.assertEqual(Path(report["cond2d_views"][0]["image_path"]).name, "camera_b.png")
        self.assertTrue(report["cond2d_views"][0]["frustum_visible"])
        expected = torch.tensor([[2., 0, .5], [0, 2., .5], [0, 0, 1.]])
        torch.testing.assert_close(selected.scene_intrinsics, expected.repeat(2, 1, 1))

    def test_excluding_every_frame_has_clear_error(self):
        camera = self.camera("only.png")
        selector = IphoneImageSelecter()
        selector._get_cameras = lambda _: [camera]
        with self.assertRaisesRegex(ValueError, "No cameras remain"):
            selector.get_images([torch.eye(4)], self.root, 2, self.root / "empty",
                                exclude_images=["only.png"])


if __name__ == "__main__":
    unittest.main()
