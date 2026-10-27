"""CPU checks for pose deduplication and chunk-aware view selection.

These tests use synthetic camera geometry rather than model outputs. The
projection reference is the actual Torch visibility implementation, loaded
without importing the dataset's optional CUDA dependencies.
"""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("genrecon_view_selection", ROOT / "inference/view_selection.py")
SELECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTION)


def rotation(axis, degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def poses(positions, rotations=None):
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(positions), axis=0)
    matrices[:, :3, 3] = positions
    if rotations is not None:
        matrices[:, :3, :3] = rotations
    return matrices


def intrinsics(count):
    return np.repeat(np.array([[[0.8, 0, 0.5], [0, 0.8, 0.5], [0, 0, 1.0]]]), count, axis=0)


def torch_projection_reference():
    source = ast.parse((ROOT / "genrecon/datasets/components.py").read_text())
    function = copy.deepcopy(next(node for node in source.body if isinstance(node, ast.FunctionDef)
                                  and node.name == "chunk_visible_in_camera"))
    # Expose the existing area calculation instead of its thresholded boolean.
    function.body[-1] = ast.Return(value=ast.Call(func=ast.Name(id="float", ctx=ast.Load()),
                                               args=[ast.Name(id="area", ctx=ast.Load())], keywords=[]))
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(tree, str(ROOT / "genrecon/datasets/components.py"), "exec"), namespace)
    return namespace["chunk_visible_in_camera"]


class PoseCoverageSelectionTest(unittest.TestCase):
    def select(self, camera_poses, areas, budget, camera_intrinsics=None, **kwargs):
        if camera_intrinsics is None:
            camera_intrinsics = intrinsics(len(camera_poses))
        indices, report = SELECTION.select_pose_coverage(
            camera_poses, camera_intrinsics, np.asarray(areas, dtype=np.float64), budget,
            position_scale=kwargs.pop("position_scale", 1.0), **kwargs,
        )
        self.assertEqual(indices, sorted(set(indices)))
        self.assertLessEqual(len(indices), budget)
        self.assertTrue(all(0 <= index < len(camera_poses) for index in indices))
        self.assertIsInstance(report, dict)
        json.dumps(report, allow_nan=False)
        return indices, report

    def test_dense_stationary_repeats_do_not_fill_the_budget(self):
        positions = [[center + jitter * 0.0001, 0, 0] for center in (0, 1, 3) for jitter in range(12)]
        camera_poses = poses(positions)
        selected, _ = self.select(camera_poses, np.ones((36, 1)), 20,
                                  position_threshold=0.1, min_views_per_chunk=1)
        self.assertEqual(len(selected), 3)
        self.assertEqual({int(round(camera_poses[i, 0, 3])) for i in selected}, {0, 1, 3})

    def test_turning_and_roll_at_one_position_are_not_duplicates(self):
        camera_poses = poses([[0, 0, 0]] * 4,
                             [np.eye(3), rotation("y", 60), rotation("z", 60), rotation("z", 60.2)])
        selected, _ = self.select(camera_poses, np.empty((4, 0)), 8, rotation_threshold_degrees=10)
        self.assertEqual(len(selected), 3)
        self.assertIn(0, selected)
        self.assertIn(1, selected)
        self.assertEqual(len(set(selected) & {2, 3}), 1)

    def test_near_identical_pose_can_be_rescued_for_a_missing_chunk(self):
        camera_poses = poses([[0, 0, 0], [0.0001, 0, 0]])
        selected, report = self.select(camera_poses, [[1, 1, 0], [0, 0, 1]], 2,
                                       position_threshold=0.1, min_views_per_chunk=1)
        self.assertEqual(selected, [0, 1])
        self.assertTrue(np.all(np.array([[1, 1, 0], [0, 0, 1]])[selected].sum(axis=0) > 0))
        self.assertEqual(report["coverage_rescue_indices"], [1])
        self.assertEqual(report["uncovered_chunk_indices"], [])

    def test_distinct_intrinsics_preserve_distinct_observations(self):
        camera_poses = poses([[0, 0, 0]] * 3)
        camera_intrinsics = intrinsics(3)
        camera_intrinsics[1, 0, 0] += 0.2
        camera_intrinsics[2, 0, 0] += 0.0001
        selected, _ = self.select(camera_poses, np.empty((3, 0)), 3,
                                  camera_intrinsics=camera_intrinsics, intrinsics_threshold=0.01)
        self.assertEqual(len(selected), 2)
        self.assertIn(1, selected)

    def test_soft_per_chunk_targets_keep_distinct_secondary_views(self):
        camera_poses = poses([[i * 0.5, 0, 0] for i in range(4)])
        selected, _ = self.select(camera_poses, [[1, 0], [0, 1], [1, 0], [0, 1]], 4,
                                  min_views_per_chunk=2)
        self.assertEqual(selected, [0, 1, 2, 3])

    def test_budget_can_leave_deficits_but_is_never_exceeded(self):
        camera_poses = poses([[i, 0, 0] for i in range(4)])
        areas = np.column_stack([np.eye(4), np.zeros(4)])
        selected, report = self.select(camera_poses, areas, 2, min_views_per_chunk=3)
        self.assertEqual(len(selected), 2)
        self.assertEqual(int(np.count_nonzero((areas[selected] >= 0.4).any(axis=0))), 2)
        self.assertEqual(report["unobservable_chunk_indices"], [4])
        self.assertEqual(len(report["uncovered_chunk_indices"]), 3)
        self.assertEqual(report["undersupported_chunk_indices"], list(range(5)))
        self.assertEqual(report["selected_views_per_chunk"],
                         (areas[selected] >= 0.4).sum(axis=0).tolist())
        self.assertTrue(report["budget_exhausted"])
        self.assertFalse(report["all_observable_chunks_covered"])

    def test_all_invisible_inputs_fall_back_to_pose_diversity(self):
        camera_poses = poses([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2.0001, 0, 0]])
        selected, report = self.select(camera_poses, np.zeros((4, 3)), 8)
        self.assertEqual(len(selected), 3)
        self.assertTrue(report["all_invisible_fallback"])
        self.assertEqual(report["selected_views_per_chunk"], [0, 0, 0])

    def test_world_rigid_transform_and_scale_do_not_change_selection(self):
        camera_poses = poses([[0, 0, 0], [0.001, 0, 0], [0.8, 0.2, 0], [2, 0, 0], [2, 0, 0]],
                             [np.eye(3), np.eye(3), rotation("y", 30), np.eye(3), rotation("z", 50)])
        areas = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1]], dtype=np.float64)
        expected, _ = self.select(camera_poses, areas, 4, position_scale=2)
        world_rotation = rotation("x", 37) @ rotation("z", -23)
        transformed = camera_poses.copy()
        transformed[:, :3, :3] = world_rotation @ camera_poses[:, :3, :3]
        transformed[:, :3, 3] = 7.3 * (camera_poses[:, :3, 3] @ world_rotation.T) + [100, -50, 20]
        actual, _ = self.select(transformed, areas, 4, position_scale=14.6)
        self.assertEqual(actual, expected)

    def test_selection_is_deterministic_and_outputs_input_order(self):
        camera_poses = poses([[i * 0.2, (i % 3) * 0.1, 0] for i in range(20)])
        areas = np.random.default_rng(82).uniform(0, 1, (20, 6))
        first, first_report = self.select(camera_poses, areas, 7)
        for _ in range(3):
            actual, report = self.select(camera_poses, areas, 7)
            self.assertEqual(actual, first)
            self.assertEqual(report, first_report)

    def test_invalid_geometry_and_empty_or_nonpositive_requests_raise(self):
        base = poses([[0, 0, 0], [1, 0, 0]])
        for budget in (0, -1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                SELECTION.select_pose_coverage(base, intrinsics(2), np.ones((2, 1)), budget,
                                               position_scale=1)
        with self.assertRaises(ValueError):
            SELECTION.select_pose_coverage(np.empty((0, 4, 4)), np.empty((0, 3, 3)),
                                           np.empty((0, 1)), 1, position_scale=1)
        for scale in (0, -1, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                SELECTION.select_pose_coverage(base, intrinsics(2), np.ones((2, 1)), 1,
                                               position_scale=scale)
        for poses_bad, intrinsics_bad, areas_bad in (
            (base[:1], intrinsics(2), np.ones((2, 1))),
            (base, intrinsics(2), np.ones((1, 1))),
            (np.full((2, 4, 4), np.nan), intrinsics(2), np.ones((2, 1))),
            (base, intrinsics(2), np.full((2, 1), np.nan)),
        ):
            with self.subTest(shapes=(poses_bad.shape, intrinsics_bad.shape, areas_bad.shape)), \
                    self.assertRaises(ValueError):
                SELECTION.select_pose_coverage(poses_bad, intrinsics_bad, areas_bad, 1,
                                               position_scale=1)


class ChunkProjectionAreaTest(unittest.TestCase):
    def test_visibility_must_use_actual_crop_intrinsics(self):
        # A 3:1 frame can see the chunk centrally, while its two extreme square
        # crops barely include it. Original-image visibility cannot stand in
        # for the actual images the model will receive.
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        camera = poses([[0, 0, -2]]) @ flip
        original = np.array([[[2 / 3, 0, 0.5], [0, 2, 0.5], [0, 0, 1]]])
        crop_intrinsics = np.repeat(original, 2, axis=0)
        crop_intrinsics[:, 0, 0] = 2
        crop_intrinsics[:, 0, 2] = [1.5, -0.5]
        raw_area = SELECTION.chunk_projection_areas(camera, original, np.eye(4)[None])
        crop_areas = SELECTION.chunk_projection_areas(np.repeat(camera, 2, axis=0),
                                                     crop_intrinsics, np.eye(4)[None])
        self.assertGreater(raw_area[0, 0], 0.4)
        self.assertTrue(np.all(crop_areas[:, 0] < 0.4))

    def test_numpy_areas_match_existing_torch_projection_and_thresholds(self):
        reference = torch_projection_reference()
        random = np.random.default_rng(147)
        c2w_cv = poses([[*random.uniform(-0.7, 0.7, 2), random.uniform(-3, -0.8)] for _ in range(12)],
                       [rotation("x", random.uniform(-25, 25)) @ rotation("y", random.uniform(-25, 25))
                        @ rotation("z", random.uniform(-180, 180)) for _ in range(12)])
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        c2w_blender = c2w_cv @ flip
        camera_intrinsics = intrinsics(12)
        camera_intrinsics[:, 0, 0] = random.uniform(0.4, 1.5, 12)
        camera_intrinsics[:, 1, 1] = random.uniform(0.4, 1.5, 12)
        camera_intrinsics[:, :2, 2] = random.uniform(-0.2, 1.2, (12, 2))
        chunks = np.repeat(np.eye(4)[None], 4, axis=0)
        for i in range(4):
            scale = random.uniform(0.6, 1.7)
            chunks[i, :3, :3] /= scale
            chunks[i, :3, 3] = -random.uniform(-0.5, 0.5, 3) / scale
        actual = SELECTION.chunk_projection_areas(c2w_blender, camera_intrinsics, chunks)
        self.assertEqual(actual.shape, (12, 4))
        self.assertEqual(actual.dtype, np.dtype(np.float64))
        expected = np.empty_like(actual)
        for i in range(12):
            for j in range(4):
                ext = flip @ np.linalg.inv(chunks[j] @ c2w_blender[i])
                expected[i, j] = reference(torch.from_numpy(ext).float(),
                                            torch.from_numpy(camera_intrinsics[i]).float())
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        for threshold in (0.01, 0.4, 0.8):
            np.testing.assert_array_equal(actual >= threshold, expected >= threshold)

    def test_chunk_behind_camera_has_zero_area(self):
        camera = poses([[0, 0, -2]])
        # Identity Blender camera looks down world -Z, away from this chunk.
        actual = SELECTION.chunk_projection_areas(camera, intrinsics(1), np.eye(4)[None])
        np.testing.assert_array_equal(actual, [[0.0]])


if __name__ == "__main__":
    unittest.main()
