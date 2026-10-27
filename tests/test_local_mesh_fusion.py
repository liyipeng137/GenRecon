"""CPU verification of fixed-frame local fusion and overlap disagreement audit."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fuse_local_meshes", ROOT / "scripts/fuse_local_meshes.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def chunks():
    return [dict(id=0, center=np.array([0., 0., 2.]), size=np.full(3, 4.)),
            dict(id=1, center=np.array([1., 0., 2.]), size=np.full(3, 4.))]


def grid(z):
    x, y = np.meshgrid(np.array([-2., -1., 0., .5, 1., 2., 3.]), np.array([-2., 0., 2.]), indexing="xy")
    vertices = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)]).astype(np.float32)
    faces = []
    for row in range(2):
        for col in range(6):
            i = row * 7 + col
            faces.extend([[i, i + 1, i + 8], [i, i + 8, i + 7]])
    return vertices, np.asarray(faces, dtype=np.int64)


class OwnershipTests(unittest.TestCase):
    def test_tie_cover_and_half_open_limits(self):
        points = np.array([[-2., 0, 2], [.49, 0, 2], [.5, 0, 2], [.51, 0, 2], [3., 0, 2], [0, 2., 2]])
        np.testing.assert_array_equal(MOD.exclusive_chunk_owner(points, chunks(), batch_size=2), [0, 0, 0, 1, -1, -1])

    def test_near_ties_choose_lower_chunk_id(self):
        points = np.array([[.5 + 1e-10, 0, 2.], [.5 + 1e-5, 0, 2.]])
        np.testing.assert_array_equal(MOD.exclusive_chunk_owner(points, chunks()), [0, 1])

    def test_noncovering_nearest_center_cannot_claim(self):
        cs = [dict(id=0, center=np.array([-.51, 0, 0]), size=np.ones(3)),
              dict(id=7, center=np.array([.49, .49, .49]), size=np.ones(3))]
        self.assertEqual(MOD.exclusive_chunk_owner(np.zeros((1, 3)), cs)[0], 7)

    def test_fixed_points_preserve_owner_despite_prediction_shift(self):
        fixed = np.array([[.49, 0, 2.]])
        self.assertEqual(MOD.exclusive_chunk_owner(fixed, chunks())[0], 0)
        self.assertEqual(MOD.exclusive_chunk_owner(fixed + [.03, 0, 0], chunks())[0], 1)

    def test_batch_order_independence(self):
        p = np.random.default_rng(7).uniform([-3, -3, -1], [4, 3, 5], size=(300, 3))
        np.testing.assert_array_equal(MOD.exclusive_chunk_owner(p, chunks(), 1), MOD.exclusive_chunk_owner(p, chunks(), 1000))
        self.assertEqual(len(MOD.exclusive_chunk_owner(np.empty((0, 3)), chunks())), 0)


class WeldTests(unittest.TestCase):
    def test_exact_duplicates_weld_and_remove_duplicate_and_degenerate_faces(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        f = np.array([[0, 1, 2], [3, 2, 1], [0, 3, 1]], dtype=np.int64)
        vc, fc, report = MOD.weld_and_clean(v, f, tolerance=0)
        self.assertEqual((len(vc), len(fc)), (3, 1))
        self.assertEqual(report["removed_duplicate_faces"], 1)
        self.assertEqual(report["removed_degenerate_faces"], 1)
        self.assertEqual(report["maximum_vertex_displacement"], 0.)

    def test_weld_respects_memory_ceiling_and_disable(self):
        v, f = grid(2.)
        vo, fo, info = MOD.weld_and_clean(v, f, memory_limit_gib=1e-12)
        self.assertIs(vo, v)
        self.assertIs(fo, f)
        self.assertEqual(info["reason"], "estimated_ram_exceeds_limit")
        self.assertFalse(MOD.weld_and_clean(v, f, tolerance=-1)[2]["applied"])

    def test_quantized_weld_uses_original_coordinate_representative(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [.0001, 0, 0]], dtype=np.float32)
        f = np.array([[0, 1, 2], [3, 1, 2]], dtype=np.int64)
        vc, fc, report = MOD.weld_and_clean(v, f, tolerance=.001)
        self.assertEqual((len(vc), len(fc)), (3, 1))
        self.assertTrue(any(np.array_equal(row, v[0]) for row in vc))
        self.assertAlmostEqual(report["maximum_vertex_displacement"], .0001, places=8)


class AuditTests(unittest.TestCase):
    def test_relative_depth_and_unsigned_normals(self):
        shape = (2, 3)
        a = dict(depth=np.full(shape, 2.), valid=np.ones(shape, bool), camera_normals=np.broadcast_to([0., 0., 1.], (*shape, 3)))
        b = dict(depth=np.full(shape, 2.06), valid=np.ones(shape, bool), camera_normals=np.broadcast_to([0., 0., -1.], (*shape, 3)))
        info, common, rel, angle = MOD.compare_raycast_hits(a, b, a["valid"], a["valid"])
        self.assertEqual(info["common_overlap_pixels"], 6)
        self.assertAlmostEqual(info["common_depth_absolute"]["median"], .06)
        self.assertAlmostEqual(info["common_depth_relative"]["median"], .03)
        self.assertEqual(info["parallel_depth_disagreement_candidate_fraction"], 1.)
        self.assertEqual(info["normal_unsigned_degrees"]["max"], 0.)

    def test_cli_fixed_lattice_selection_preview_and_raw_preservation(self):
        with tempfile.TemporaryDirectory(prefix="genrecon-local-fusion-test-") as tmp:
            base = Path(tmp)
            (base / "data/colmap").mkdir(parents=True)
            (base / "data/colmap/cameras.txt").write_text("1 PINHOLE 80 60 70 70 40 30\n")
            (base / "data/colmap/images.txt").write_text("1 1 0 0 0 0 0 0 1 frame_000000.png\n\n")
            va, fa = grid(2.)
            vb, fb = grid(2.06)
            MOD.write_ply(base / "a.ply", va, fa)
            MOD.write_ply(base / "b.ply", vb, fb)
            raw_hashes = [MOD.sha256(base / f"{s}.ply") for s in "ab"]
            owner = va[fa].mean(1)
            np.save(base / "points.npy", owner)
            np.save(base / "anchors.npy", va.astype(np.float64))
            manifest = dict(coordinate_frame="global_zup", data_dir="data",
                chunks=[dict(id=c["id"], center=c["center"].tolist(), size=c["size"].tolist()) for c in chunks()],
                groups=[dict(id=s, mesh_path=f"{s}.ply", chunk_ids=[0, 1], core_chunk_ids=[i],
                    vertex_owner_points_path="anchors.npy", selected_image_names=["frame_000000.png"])
                    for i, s in enumerate("ab")])
            (base / "manifest.json").write_text(json.dumps(manifest))
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
            completed = subprocess.run([sys.executable, str(ROOT / "scripts/fuse_local_meshes.py"),
                "--manifest", str(base / "manifest.json"), "--output", str(base / "fusion"),
                "--raw-union", "--preview-width", "80", "--threads", "2"],
                env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads((base / "fusion/fusion_report.json").read_text())
            self.assertTrue(report["raw_meshes_preserved"])
            self.assertTrue(all(g["ownership_mode"] == "fixed_lattice_vertex_anchors" for g in report["groups"]))
            self.assertTrue(report["no_icp_or_pose_adjustment"])
            self.assertFalse(report["watertightness_guaranteed"])
            self.assertEqual(sum(g["selected_faces"] for g in report["groups"]), len(fa))
            self.assertEqual(report["weld"]["final_faces"], len(fa))
            for s, digest in zip("ab", raw_hashes):
                self.assertEqual(MOD.sha256(base / f"{s}.ply"), digest)
            audit = report["overlap_audit"]["pairs"][0]
            metrics = audit["views"][0]
            self.assertGreater(metrics["common_overlap_pixels"], 1000)
            self.assertAlmostEqual(metrics["common_depth_relative"]["median"], .03, places=5)
            self.assertEqual(metrics["parallel_depth_disagreement_candidate_fraction"], 1.)
            self.assertTrue((base / "fusion/overlap/overlap_preview.jpg").is_file())
            self.assertTrue((base / "fusion/mesh_raw_union.ply").is_file())


if __name__ == "__main__":
    unittest.main()
