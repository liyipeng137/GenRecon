"""Meaningful CPU fixtures for matched-ray cross-run diagnostics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_scene_runs.py"
SPEC = importlib.util.spec_from_file_location("compare_scene_runs", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def fixture_inspection(root, *, candidate=False, raw_hash=None):
    root.mkdir()
    previews = []
    for index, name in enumerate(compare.DEFAULT_VIEWS):
        folder = root / "views" / Path(name).stem
        folder.mkdir(parents=True)
        depth = np.full((3, 4), 2.2 if candidate else 2., dtype=np.float32)
        hit = np.ones((3, 4), dtype=bool)
        hit[0, 0 if not candidate else 1] = False
        depth[~hit] = 0
        normals = np.zeros((3, 4, 3), dtype=np.float32)
        normals[..., 2] = -1. if candidate else 1.
        rgb = np.full((3, 4, 3), 30 + index, dtype=np.uint8)
        Image.fromarray(rgb).save(folder / "real_rgb.png")
        Image.fromarray(rgb + 20).save(folder / "gray_geometry.png")
        np.save(folder / "camera_z.npy", depth)
        np.save(folder / "ray_hit_mask.npy", hit)
        np.save(folder / "camera_normals.npy", normals)
        previews.append({"image_name": name, "image_id": index + 1, "width": 4, "height": 3,
                         "K_scaled": [[3., 0., 2.], [0., 3., 1.5], [0., 0., 1.]],
                         "half_integer_pixel_centres": True})
    report = {"coordinate_transform_sha256": "common_transform", "preview_protocol": "half_integer_fixture",
              "raw_sha256": raw_hash or ("candidate" if candidate else "baseline"), "previews": previews}
    (root / "inspection_report.json").write_text(json.dumps(report))
    return report


class CompareSceneRunsTest(unittest.TestCase):
    def test_cli_equivalent_fixture_records_gained_lost_hits_and_sign_free_normals(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            fixture_inspection(temp / "baseline")
            fixture_inspection(temp / "candidate", candidate=True)
            report = compare.compare_inspections(temp / "baseline", temp / "candidate", temp / "comparison", cell_width=150)
            totals = report["totals_over_requested_camera_rays"]
            self.assertEqual(totals["pixel_count"], 96)
            self.assertEqual(totals["common_hit_count"], 80)
            self.assertEqual(totals["candidate_only_hit_count"], 8)
            self.assertEqual(totals["baseline_only_hit_count"], 8)
            self.assertEqual(totals["no_hit_change_percentage_points_candidate_minus_baseline"], 0.)
            metric = report["views"][0]
            np.testing.assert_allclose(metric["common_hit_absolute_camera_z_difference_p50_p90_p95_scene_units"], .2, atol=1e-6)
            np.testing.assert_allclose(metric["common_hit_unoriented_normal_angle_p50_p90_p95_degrees"], 0., atol=1e-6)
            self.assertEqual(metric["common_hit_opposite_normal_fraction_dot_lt_minus09"], 1.)
            self.assertFalse(report["independent_overlap_quality_evaluated"])
            self.assertTrue((temp / "comparison/focus_075_302_453.jpg").is_file())
            self.assertEqual(len(report["views"]), 8)

    def test_same_mesh_zero_difference_only_reports_consistency(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            fixture_inspection(temp / "baseline", raw_hash="same_mesh")
            fixture_inspection(temp / "candidate", raw_hash="same_mesh")
            report = compare.compare_inspections(temp / "baseline", temp / "candidate", temp / "comparison", cell_width=80)
            self.assertTrue(report["same_raw_mesh_sha256"])
            self.assertIn("not evidence", report["same_mesh_interpretation"])
            self.assertFalse(report["independent_overlap_quality_evaluated"])
            self.assertEqual(report["views"][0]["common_hit_absolute_camera_z_difference_p50_p90_p95_scene_units"], [0., 0., 0.])

    def test_camera_mismatch_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            a = fixture_inspection(temp / "baseline")
            b = fixture_inspection(temp / "candidate", candidate=True)
            for key, value in (("K_scaled", [[4., 0., 2.], [0., 3., 1.5], [0., 0., 1.]]),
                               ("half_integer_pixel_centres", False), ("image_id", 900)):
                original = b["previews"][0][key]
                b["previews"][0][key] = value
                with self.assertRaises(ValueError):
                    compare.matched_view_metadata(a, b, compare.DEFAULT_VIEWS)
                b["previews"][0][key] = original
            b["coordinate_transform_sha256"] = "different_transform"
            with self.assertRaisesRegex(ValueError, "coordinate transforms"):
                compare.matched_view_metadata(a, b, compare.DEFAULT_VIEWS)

    def test_different_source_rgb_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            fixture_inspection(temp / "baseline")
            fixture_inspection(temp / "candidate", candidate=True)
            Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(temp / "candidate/views/frame_000075/real_rgb.png")
            with self.assertRaisesRegex(ValueError, "RGB pixels differ"):
                compare.compare_inspections(temp / "baseline", temp / "candidate", temp / "comparison")
            self.assertFalse((temp / "comparison").exists())

    def test_no_common_rays_gives_null_depth_and_normal_metrics(self):
        depth = np.ones((2, 2), dtype=np.float32)
        hit = np.array([[True, False], [True, False]])
        normals = np.zeros((2, 2, 3), dtype=np.float32)
        normals[..., 2] = 1
        result = compare.compare_arrays(depth, hit, normals, depth, ~hit, normals)
        self.assertEqual(result["common_hit_count"], 0)
        self.assertIsNone(result["common_hit_absolute_camera_z_difference_p50_p90_p95_scene_units"])
        self.assertIsNone(result["common_hit_unoriented_normal_angle_p50_p90_p95_degrees"])
        self.assertEqual(result["candidate_only_hit_count"], 2)
        self.assertEqual(result["baseline_only_hit_count"], 2)

    def test_invalid_valid_ray_and_nonboolean_mask_are_rejected(self):
        depth = np.ones((2, 2), dtype=np.float32)
        hit = np.ones((2, 2), dtype=bool)
        normals = np.ones((2, 2, 3), dtype=np.float32)
        invalid = depth.copy()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid camera-Z"):
            compare.compare_arrays(depth, hit, normals, invalid, hit, normals)
        with self.assertRaisesRegex(ValueError, "boolean"):
            compare.compare_arrays(depth, hit.astype(np.uint8), normals, depth, hit, normals)


if __name__ == "__main__":
    unittest.main()
