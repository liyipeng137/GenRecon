"""CPU contract checks for independent local-scene input/cache/export handling."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import trimesh

from inference.get_images import IphoneImageSelecter, _to_chunk0_extrinsics
from scripts.reconstruct_local_scene import (
    crop_indices, read_crop_512, transform_points, validate_layout, write_mesh,
)


class LocalSceneDriverTest(unittest.TestCase):
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

    def camera(self, name, width=144, height=96, rgba=False):
        rng = np.random.default_rng(4)
        pixels = rng.integers(0, 256, size=(height, width, 4 if rgba else 3), dtype=np.uint8)
        image_path = self.root / name
        Image.fromarray(pixels).save(image_path)
        return dict(img_path=str(image_path), c2w_blender=torch.eye(4),
                    intrinsics=torch.tensor([[.8, 0., .4], [0., .7, .6], [0., 0., 1.]]))

    def test_streamed_crops_match_native_loader_for_actual_rgb_and_alpha_pixels(self):
        selector = IphoneImageSelecter(center_crop=False)
        for width, height, rgba in ((144, 96, False), (80, 160, False), (70, 70, False),
                                    (144, 96, True), (80, 160, True), (70, 70, True)):
            with self.subTest(width=width, height=height, rgba=rgba):
                camera = self.camera(f"im_{width}_{height}_{rgba}.png", width, height, rgba)
                crops = selector._prepare_scene_pool([camera])
                intrinsics = torch.stack([c["intrinsics"] for c in crops])
                native_k, native_512, _ = selector._process_images(
                    [c["img_path"] for c in crops], intrinsics,
                    crop_boxes=[c["crop_box"] for c in crops])
                streamed = np.stack([read_crop_512(c) for c in crops])
                np.testing.assert_array_equal(streamed, native_512.numpy())
                torch.testing.assert_close(native_k, intrinsics, atol=0, rtol=0)

    def test_crop_indices_retain_global_order_and_all_crops_for_overlapping_pools(self):
        cameras = [self.camera(f"frame_{i:06d}.png", width=100, height=180) for i in range(3)]
        crops = IphoneImageSelecter()._prepare_scene_pool(cameras)
        selected = crop_indices(crops, ["frame_000002.png", "frame_000000.png"])
        self.assertEqual(selected, [0, 1, 4, 5])
        shared = crop_indices(crops, ["frame_000001.png", "frame_000002.png"])
        self.assertEqual(shared, [2, 3, 4, 5])
        tokens = np.arange(6 * 4 * 3).reshape(6, 4, 3)
        local = torch.from_numpy(np.asarray(tokens[selected]).copy()).unsqueeze(0)
        anchor_global = 5
        anchor = local[:, [selected.index(anchor_global)]]
        np.testing.assert_array_equal(anchor[0, 0].numpy(), tokens[anchor_global])
        with self.assertRaisesRegex(ValueError, "no encoded crop"):
            crop_indices(crops, ["missing.png"])

    def layout(self):
        names = [f"frame_{i:06d}.png" for i in range(429)]
        chunks = []
        for cid, (center, pool) in enumerate((([1., 2., 1.4], names[:250]),
                                             ([2.5, 2., 1.4], names[200:]))):
            c2o = np.eye(4)
            c2o[:3, :3] *= 2.8
            c2o[:3, 3] = center
            chunks.append(dict(id=cid, center=center, size=2.8,
                               M_chunk_to_original=c2o.tolist(),
                               M_original_to_chunk=np.linalg.inv(c2o).tolist(), pool_image_names=pool))
        return dict(chunks=chunks), [str(self.root / name) for name in names]

    def test_layout_requires_every_one_of_429_frozen_images_and_shared_pose_frame(self):
        layout, paths = self.layout()
        self.assertEqual(validate_layout(layout, paths), [Path(path).name for path in paths])
        bad = copy.deepcopy(layout)
        bad["chunks"][1]["pool_image_names"].pop()
        with self.assertRaisesRegex(ValueError, "omitted"):
            validate_layout(bad, paths)
        bad = copy.deepcopy(layout)
        bad["chunks"][0]["pool_image_names"].append("new.png")
        with self.assertRaisesRegex(ValueError, "non-frozen"):
            validate_layout(bad, paths)
        bad = copy.deepcopy(layout)
        bad["chunks"][0]["M_chunk_to_original"][0][0] = 2.9
        with self.assertRaisesRegex(ValueError, "global axis"):
            validate_layout(bad, paths)
        bad = copy.deepcopy(layout)
        bad["chunks"][0]["M_original_to_chunk"][0][3] += 1
        with self.assertRaisesRegex(ValueError, "not invertible"):
            validate_layout(bad, paths)

    def test_world_local_round_trip_and_native_projection_are_invariant(self):
        layout, _ = self.layout()
        rng = np.random.default_rng(12)
        points = rng.normal(size=(20, 3))
        c2w_cv = torch.eye(4, dtype=torch.float32)
        theta = .7
        c2w_cv[:3, :3] = torch.tensor([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.],
                                      [-np.sin(theta), 0., np.cos(theta)]])
        c2w_cv[:3, 3] = torch.tensor([-.4, .8, -2.])
        c2w_blender = c2w_cv @ torch.diag(torch.tensor([1., -1., -1., 1.], dtype=torch.float32))
        world_ext = torch.linalg.inv(c2w_cv)
        for chunk in layout["chunks"]:
            local = transform_points(points, chunk["M_original_to_chunk"])
            restored = transform_points(local, chunk["M_chunk_to_original"])
            np.testing.assert_allclose(restored, points, atol=1e-14)
            ext = _to_chunk0_extrinsics(c2w_blender, torch.tensor(chunk["M_original_to_chunk"]))
            local_camera = transform_points(local, ext.numpy())
            world_camera = transform_points(points, world_ext.numpy())
            np.testing.assert_allclose(local_camera, world_camera, atol=1e-6)

    def test_binary_ply_preserves_vertex_face_and_anchor_order(self):
        vertices = np.array([[4., 3., 2.], [-2., 0., 1.], [0., 4., 0.], [4., 3., 2.]], dtype=np.float32)
        faces = np.array([[3, 2, 1], [0, 1, 2]], dtype=np.int32)
        anchors = np.arange(12).reshape(4, 3) / 7.
        target = self.root / "mesh.ply"
        write_mesh(target, vertices, faces)
        np.save(self.root / "anchors.npy", anchors)
        mesh = trimesh.load(target, process=False)
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)
        np.testing.assert_array_equal(np.load(self.root / "anchors.npy"), anchors)
        # Duplicate vertex rows remain independent; the decoder's per-vertex
        # ownership anchors can therefore still be indexed by exported faces.
        self.assertEqual(len(mesh.vertices), 4)
        np.testing.assert_array_equal(anchors[mesh.faces], anchors[faces])
        with self.assertRaisesRegex(ValueError, "invalid vertex"):
            write_mesh(target, vertices, np.array([[0, 1, 4]]))
        with self.assertRaisesRegex(ValueError, "Invalid mesh vertices"):
            write_mesh(target, np.full((4, 3), np.nan), faces)


if __name__ == "__main__":
    unittest.main()
