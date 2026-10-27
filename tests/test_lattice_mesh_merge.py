"""Numerical comparison with native grouped decoder's final merge algorithm."""
from __future__ import annotations

from unittest import mock
import unittest

import numpy as np
import torch

import genrecon.pipelines.lattice_mesh_merge as cpu_merge


def native_merge(vertices, faces, coords, aabb, grid):
    offsets = np.cumsum([0] + [len(value) for value in vertices[:-1]])
    all_vertices = torch.cat(vertices)
    all_faces = torch.cat([face + int(offset) for face, offset in zip(faces, offsets)])
    all_coords = torch.cat(coords)
    gy, gz = int(grid[1]) + 1, int(grid[2]) + 1
    keys = all_coords[:, 0].long() * (gy * gz) + all_coords[:, 1].long() * gz + all_coords[:, 2].long()
    _, inverse = torch.unique(keys, return_inverse=True)
    order = torch.argsort(inverse, stable=True)
    inverse_sorted = inverse[order]
    keep = torch.ones(len(inverse), dtype=torch.bool)
    keep[1:] = inverse_sorted[1:] != inverse_sorted[:-1]
    representative = order[keep]
    result_v = all_vertices[representative].float()
    result_f = inverse[all_faces.long()].int()
    anchors = ((all_coords[representative].float() + .5) * ((aabb[1] - aabb[0]) / grid) + aabb[0])
    return result_v, result_f, anchors


class LatticeMeshMergeTest(unittest.TestCase):
    def randomized_parts(self):
        torch.manual_seed(983)
        grid = torch.tensor([11, 7, 9], dtype=torch.int32)
        all_coords = torch.cartesian_prod(torch.arange(12), torch.arange(8), torch.arange(10)).int()
        vertices, faces, coords = [], [], []
        for index, count in enumerate([63, 51, 44]):
            cp = all_coords[torch.randperm(len(all_coords))[:count]]
            # Force common keys, but give each local decoder distinct predictions.
            cp[:9] = all_coords[torch.arange(9) * 17]
            vp = cp.float() * .1 + torch.randn(count, 3) * .004 + index * .003
            fp = torch.randint(0, count, (count * 2, 3), dtype=torch.int32)
            vertices.append(vp)
            faces.append(fp)
            coords.append(cp)
        aabb = torch.tensor([[-1.25, -.5, .2], [2.1, 3.5, 1.7]], dtype=torch.float32)
        return vertices, faces, coords, aabb, grid

    def test_exact_native_representatives_faces_and_anchors_for_random_overlapping_parts(self):
        inputs = self.randomized_parts()
        expected = native_merge(*inputs)
        actual = cpu_merge.merge_cpu_lattice_parts(*inputs)
        for left, right in zip(expected, actual[:3]):
            torch.testing.assert_close(left, right, atol=0., rtol=0.)
            self.assertEqual(right.device.type, "cpu")
        self.assertEqual(actual[0].dtype, torch.float32)
        self.assertEqual(actual[1].dtype, torch.int32)
        self.assertEqual(actual[2].dtype, torch.float32)
        report = actual[3]
        self.assertEqual(report["input_faces"], report["output_faces"])
        self.assertGreaterEqual(report["cross_group_shared_lattice_keys"], 9)
        self.assertGreater(report["cross_group_prediction_distance_to_first_representative"]["max"], 0.)
        self.assertFalse(report["global_int64_face_array_materialized"])

    def test_first_group_coordinate_wins_and_cross_group_disagreement_is_not_averaged(self):
        coords = [torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int32)] * 2
        first = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        second = first + torch.tensor([3., 4., 0.])
        faces = [torch.tensor([[2, 0, 1]], dtype=torch.int32), torch.tensor([[0, 2, 1]], dtype=torch.int32)]
        vertices, remapped_faces, _, report = cpu_merge.merge_cpu_lattice_parts(
            [first, second], faces, coords, torch.tensor([[0., 0., 0.], [1., 1., 1.]]), torch.tensor([2, 2, 2]))
        # Ascending packed keys sort z first, y second and x third.
        torch.testing.assert_close(vertices, first[torch.tensor([2, 1, 0])], atol=0., rtol=0.)
        torch.testing.assert_close(remapped_faces, torch.tensor([[0, 2, 1], [2, 0, 1]], dtype=torch.int32))
        stats = report["cross_group_prediction_distance_to_first_representative"]
        self.assertEqual(report["cross_group_shared_lattice_keys"], 3)
        self.assertEqual(stats["compared_vertex_predictions"], 3)
        self.assertEqual(stats["mean"], 5.)
        self.assertEqual(stats["max"], 5.)
        self.assertIn("not errors against ground truth", stats["interpretation"])

    def test_within_group_duplicates_degenerate_faces_and_unreferenced_vertices_match_native(self):
        cp = torch.tensor([[2, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0]], dtype=torch.int32)
        vp = torch.tensor([[1., 0., 0.], [3., 0., 0.], [8., 0., 0.], [7., 1., 0.]])
        fp = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32)
        inputs = ([vp], [fp], [cp], torch.tensor([[0., 0., 0.], [1., 1., 1.]]), torch.tensor([3, 3, 3]))
        expected = native_merge(*inputs)
        actual = cpu_merge.merge_cpu_lattice_parts(*inputs)
        for left, right in zip(expected, actual[:3]):
            torch.testing.assert_close(left, right, atol=0., rtol=0.)
        self.assertEqual(len(actual[0]), 3)  # includes the unreferenced zero-key vertex
        self.assertEqual(len(actual[1]), 2)  # duplicate/degenerate faces preserved
        self.assertEqual(actual[3]["cross_group_shared_lattice_keys"], 0)
        self.assertIsNone(actual[3]["cross_group_prediction_distance_to_first_representative"]["mean"])

    def test_small_batch_remapping_keeps_original_face_and_part_order(self):
        inputs = self.randomized_parts()
        expected = native_merge(*inputs)
        with mock.patch.object(cpu_merge, "_BATCH", 3):
            actual = cpu_merge.merge_cpu_lattice_parts(*inputs)
        for left, right in zip(expected, actual[:3]):
            torch.testing.assert_close(left, right, atol=0., rtol=0.)
        self.assertEqual(actual[3]["face_remap_batch_size"], 3)

    def test_numpy_noncontiguous_inputs_and_inclusive_grid_edge_keys(self):
        coords = np.array([[0, 2, 0], [0, 0, 2], [2, 2, 2], [0, 1, 0]], dtype=np.int32)
        vertices = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
        faces = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.int64)
        aabb = np.array([[0., 0., 0.], [2., 2., 2.]], dtype=np.float32)
        grid = np.array([2, 2, 2], dtype=np.int32)
        actual = cpu_merge.merge_cpu_lattice_parts([vertices], [faces], [coords], aabb, grid)
        expected = native_merge([torch.from_numpy(vertices)], [torch.from_numpy(faces)],
                                [torch.from_numpy(coords)], torch.from_numpy(aabb), torch.from_numpy(grid))
        for left, right in zip(expected, actual[:3]):
            torch.testing.assert_close(left, right, atol=0., rtol=0.)
        self.assertEqual(len(actual[0]), 4)

    def test_invalid_lattices_indices_and_nonfinite_vertices_are_rejected(self):
        vertices = [torch.zeros(3, 3)]
        faces = [torch.tensor([[0, 1, 2]], dtype=torch.int32)]
        coords = [torch.tensor([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.int32)]
        bounds = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
        grid = torch.tensor([2, 2, 2], dtype=torch.int32)
        bad_cases = [
            ([], [], [], bounds, grid),
            (vertices, [], coords, bounds, grid),
            (vertices, [torch.tensor([[0, 1, 3]])], coords, bounds, grid),
            ([torch.full((3, 3), float("nan"))], faces, coords, bounds, grid),
            (vertices, faces, [coords[0] - 1], bounds, grid),
            (vertices, faces, [coords[0] + 3], bounds, grid),
            (vertices, faces, [coords[0].float()], bounds, grid),
            (vertices, faces, coords, bounds, grid.float()),
            (vertices, faces, coords, bounds, torch.tensor([2**30, 2**30, 2**30])),
        ]
        for inputs in bad_cases:
            with self.subTest(inputs=str(inputs)[:80]), self.assertRaises(ValueError):
                cpu_merge.merge_cpu_lattice_parts(*inputs)

    def test_empty_part_does_not_change_representative_selection(self):
        inputs = self.randomized_parts()
        expected = cpu_merge.merge_cpu_lattice_parts(*inputs)
        vp, fp, cp, bounds, grid = inputs
        actual = cpu_merge.merge_cpu_lattice_parts(
            [torch.empty(0, 3), *vp], [torch.empty(0, 3, dtype=torch.int32), *fp],
            [torch.empty(0, 3, dtype=torch.int32), *cp], bounds, grid)
        for left, right in zip(expected[:3], actual[:3]):
            torch.testing.assert_close(left, right, atol=0., rtol=0.)


if __name__ == "__main__":
    unittest.main()
