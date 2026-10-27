"""CPU fixture for complete-geometry streaming export and COLMAP raycasting.

Run unittest discovery, or retain the small fixture and reports with:
  python tests/test_inspect_ffba_large.py --output /absolute/new/fixture_dir
No model weights, CUDA, or production mesh are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_fixture(base):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    if any(base.iterdir()):
        raise FileExistsError(f"Refusing a nonempty fixture directory: {base}")
    (base / "result").mkdir()
    (base / "data/colmap").mkdir(parents=True)
    (base / "data/rgb").mkdir()
    (base / "source/images").mkdir(parents=True)
    rotation = Rotation.from_euler("xyz", [55, 12, -8], degrees=True).as_matrix()
    shift = np.array([1.5, -.4, .7])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, shift
    original = np.array([[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]], dtype=float)
    original[:, 2] = 2 + .2 * original[:, 0] + .1 * original[:, 1]
    vertices = original @ rotation.T + shift
    records = np.empty(4, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                 ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                 ("metallic", "u1"), ("roughness", "u1"), ("alpha", "u1")])
    for col, key in enumerate(("x", "y", "z")):
        records[key] = vertices[:, col]
    for j, key in enumerate(("red", "green", "blue", "metallic", "roughness", "alpha")):
        records[key] = np.array([30, 60, 90, 120], dtype=np.uint8) + j
    faces = np.empty(2, dtype=[("vertex_indices", "O")])
    faces["vertex_indices"] = [np.array([0, 1, 2], dtype=np.int32), np.array([0, 2, 3], dtype=np.int32)]
    raw = base / "result/mesh.ply"
    PlyData([PlyElement.describe(records, "vertex"), PlyElement.describe(faces, "face")],
            text=False, byte_order="<").write(str(raw))
    raw_hash = digest(raw)
    (base / "data/world_transform.json").write_text(json.dumps({
        "original_to_genrecon": transform.tolist(), "genrecon_to_original": np.linalg.inv(transform).tolist()}))
    (base / "data/colmap/cameras.txt").write_text("1 PINHOLE 160 120 120 120 80 60\n")
    r_cam = rotation.T
    t_cam = -r_cam @ shift
    q_cam = Rotation.from_matrix(r_cam).as_quat()[[3, 0, 1, 2]]
    (base / "data/colmap/images.txt").write_text(
        "1 " + " ".join(f"{v:.17g}" for v in np.r_[q_cam, t_cam]) + " 1 frame_000000.png\n\n")
    image = base / "source/images/frame_000000.png"
    Image.new("RGB", (160, 120), "#aaccee").save(image)
    (base / "data/rgb/frame_000000.png").symlink_to(image.resolve())
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
    for variant in ("generic", "large"):
        script = ROOT / "scripts" / ("inspect_ffba_result.py" if variant == "generic" else "inspect_ffba_large_result.py")
        command = [sys.executable, str(script), "--result-dir", str(base / "result"),
                   "--data-dir", str(base / "data"), "--source-dataset", str(base / "source"),
                   "--output", str(base / variant), "--width", "160", "--view-count", "1"]
        if variant == "large":
            command += ["--threads", "2"]
        completed = subprocess.run(command, env=env, capture_output=True, text=True)
        (base / (variant + ".log")).write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise AssertionError(f"{variant} inspector failed: {completed.stdout}\n{completed.stderr}")
    matches = {}
    for name in ("camera_z.npy", "ray_hit_mask.npy", "camera_normals.npy"):
        old = np.load(base / "generic/views/frame_000000" / name)
        new = np.load(base / "large/views/frame_000000" / name)
        matches[name] = bool(np.array_equal(old, new))
        if not matches[name]:
            raise AssertionError(f"Preview arrays differ: {name}")
    generic = PlyData.read(str(base / "generic/mesh_ffba_original.ply"))
    large = PlyData.read(str(base / "large/mesh_ffba_original.ply"))
    for key in generic["vertex"].data.dtype.names:
        if not np.array_equal(generic["vertex"].data[key], large["vertex"].data[key]):
            raise AssertionError(f"Inverse coordinate/PBR export differs: {key}")
        if key not in ("x", "y", "z") and not np.array_equal(large["vertex"].data[key], records[key]):
            raise AssertionError(f"PBR property changed: {key}")
    for left, right in zip(generic["face"].data["vertex_indices"], large["face"].data["vertex_indices"]):
        if not np.array_equal(left, right):
            raise AssertionError("Face indices changed")
    y, x = np.meshgrid(np.arange(120) + .5, np.arange(160) + .5, indexing="ij")
    expected_depth = 2 / (1 - .2 * (x - 80) / 120 - .1 * (y - 60) / 120)
    depth = np.load(base / "large/views/frame_000000/camera_z.npy")
    normals = np.load(base / "large/views/frame_000000/camera_normals.npy")
    valid = np.load(base / "large/views/frame_000000/ray_hit_mask.npy")
    expected_normal = np.array([-.2, -.1, 1.])
    expected_normal /= np.linalg.norm(expected_normal)
    depth_error = float(np.max(np.abs(depth - expected_depth)))
    normal_error = float(np.max(np.abs(normals - expected_normal)))
    if not valid.all() or depth_error > 3e-6 or normal_error > 2e-6:
        raise AssertionError("Analytical half-integer sloped plane check failed")
    new_report = json.loads((base / "large/inspection_report.json").read_text())
    old_report = json.loads((base / "generic/inspection_report.json").read_text())
    if digest(raw) != raw_hash or old_report["raw_sha256"] != new_report["raw_sha256"]:
        raise AssertionError("Raw fixture mesh changed")
    if not new_report["mesh_checks"]["all_raw_face_records_byte_identical"]:
        raise AssertionError("Face byte preservation check failed")
    report = dict(status="pass", fixture="Rigidly transformed plane z=2+0.2*x+0.1*y; two complete triangles",
                  same_rays_and_coords_as_generic=True, same_preview_arrays=matches,
                  unchanged_custom_pbr_properties=True, unchanged_all_face_records=True,
                  raw_hash_preserved=True, analytical_half_integer_depth_max_error=depth_error,
                  analytical_camera_normal_max_error=normal_error,
                  generic_inspector_sha256=digest(ROOT / "scripts/inspect_ffba_result.py"),
                  large_inspector_sha256=digest(ROOT / "scripts/inspect_ffba_large_result.py"))
    (base / "fixture_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class LargeInspectorFixtureTest(unittest.TestCase):
    def test_complete_geometry_rays_inverse_coordinates_and_pbr(self):
        with tempfile.TemporaryDirectory(prefix="genrecon-inspector-fixture-") as temp:
            self.assertEqual(run_fixture(Path(temp) / "fixture")["status"], "pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args, remaining = parser.parse_known_args()
    if args.output:
        if remaining:
            parser.error("Unexpected arguments with --output")
        print(json.dumps(run_fixture(args.output.resolve()), indent=2))
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
