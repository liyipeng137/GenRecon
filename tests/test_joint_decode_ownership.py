"""CPU checks for exclusive chunk ownership before shape/texture merging.

Load the actual helper definitions without importing optional CUDA backends.
Run: python -m unittest discover -s tests -p 'test_joint_decode_ownership.py'
"""
import ast
from pathlib import Path
from typing import List
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


class SparseFixture:
    def __init__(self, feats, coords):
        self.feats, self.coords = feats, coords


def helpers():
    path = ROOT / "genrecon/pipelines/joint_decode.py"
    names = {"_exclusive_chunk_owner", "_owned_by_group", "_filter_sparse_to_owned", "_face_owned_by_group"}
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(torch=torch, SparseTensor=SparseFixture, List=List)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ExclusiveChunkOwnershipTest(unittest.TestCase):
    def setUp(self):
        self.impl = helpers()

    def test_real_overlap_stride_and_exact_tie(self):
        origins = torch.tensor([[0., 0, 0], [20., 0, 0]])  # 0.625 chunk stride
        points = torch.tensor([[19., 16, 16], [25., 16, 16], [26., 16, 16],
                               [27., 16, 16], [32., 16, 16], [51.99, 16, 16]])
        expected = torch.tensor([0, 0, 0, 1, 1, 1])
        torch.testing.assert_close(self.impl["_exclusive_chunk_owner"](points, origins, 32), expected)

    def test_gap_and_half_open_outer_boundaries_are_unowned(self):
        origins = torch.tensor([[0., 0, 0], [40., 0, 0]])
        points = torch.tensor([[0., 0, 0], [31.99, 16, 16], [32., 16, 16],
                               [39.99, 16, 16], [40., 16, 16], [72., 16, 16],
                               [4., 32, 16], [4., 16, -0.01]])
        expected = torch.tensor([0, 0, -1, -1, 1, -1, -1, -1])
        torch.testing.assert_close(self.impl["_exclusive_chunk_owner"](points, origins, 32), expected)

    def test_nearest_noncovering_centre_is_not_allowed_to_claim_point(self):
        # Chunk 0's centre is closer, but the point lies just outside its cube.
        origins = torch.tensor([[-1.01, -.5, -.5], [-.01, -.01, -.01]])
        owner = self.impl["_exclusive_chunk_owner"](torch.tensor([[0., 0, 0]]), origins, 1)
        self.assertEqual(owner.item(), 1)

    def test_multiple_chunk_groups_partition_union_once_without_filling_gap(self):
        origins = torch.tensor([[0., 0, 0], [20., 0, 0], [72., 0, 0]])
        points = torch.tensor([[1., 16, 16], [26., 16, 16], [40., 16, 16],
                               [60., 16, 16], [80., 16, 16], [104., 16, 16]])
        masks = [self.impl["_owned_by_group"](points, origins, 32, group) for group in ([2, 0], [1])]
        torch.testing.assert_close(masks[0], torch.tensor([True, True, False, False, True, False]))
        torch.testing.assert_close(masks[1], torch.tensor([False, False, True, False, False, False]))
        torch.testing.assert_close(torch.stack(masks).sum(0), torch.tensor([1, 1, 1, 0, 1, 0]))

    def test_texture_uses_voxel_centres_and_concatenates_without_duplicate_coords(self):
        origins = torch.tensor([[0., 0, 0], [.625, 0, 0]])
        coords = torch.tensor([[0, x, 7, 7] for x in range(28)], dtype=torch.int32)
        source = SparseFixture(feats=torch.arange(28.)[:, None], coords=coords)
        parts = [self.impl["_filter_sparse_to_owned"](source, origins, 1, 16, [i]) for i in range(2)]
        # Fine x=12 has centre .78125 (chunk 0); x=13 centre .84375 (chunk 1).
        self.assertEqual(parts[0].coords[-1, 1].item(), 12)
        self.assertEqual(parts[1].coords[0, 1].item(), 13)
        merged = torch.cat([part.coords for part in parts])
        self.assertEqual(len(merged), 26)  # x=26,27 are outside both cubes
        self.assertEqual(len(torch.unique(merged, dim=0)), len(merged))
        for part in parts:
            torch.testing.assert_close(part.feats[:, 0], part.coords[:, 1].float())

    def test_face_centroids_and_texture_centres_share_same_owner_rule(self):
        origins = torch.tensor([[0., 0, 0], [20., 0, 0], [0., 22, 0], [20., 22, 0]])
        fine_coords = torch.tensor([[0, 12, 12, 12], [0, 400, 200, 200],
                                    [0, 420, 400, 200], [0, 300, 550, 100]], dtype=torch.int32)
        source = SparseFixture(torch.arange(4.)[:, None], fine_coords)
        coarse_centres = (fine_coords[:, 1:].float() + .5) / 16
        for group in ([0, 2], [1, 3]):
            face_mask = self.impl["_owned_by_group"](coarse_centres, origins, 32, group)
            tex = self.impl["_filter_sparse_to_owned"](source, origins, 32, 16, group)
            torch.testing.assert_close(tex.coords, fine_coords[face_mask])

    def test_group_vertex_prediction_perturbation_cannot_change_face_owner(self):
        origins = torch.tensor([[0., 0, 0], [20., 0, 0]])
        lattice = torch.tensor([[415, 255, 256], [415, 256, 256], [416, 256, 256]], dtype=torch.int32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        centre = (lattice.float() + .5).mean(0) / 16
        # The same voxel keys can predict slightly different vertex positions
        # in separate groups. Their geometric centroids straddle x=26.
        predictions_a = (lattice.float() + .5) / 16 - torch.tensor([.03, 0, 0])
        predictions_b = (lattice.float() + .5) / 16 + torch.tensor([.03, 0, 0])
        owner = self.impl["_exclusive_chunk_owner"]
        self.assertEqual(owner(predictions_a.mean(0, keepdim=True), origins, 32).item(), 0)
        self.assertEqual(owner(predictions_b.mean(0, keepdim=True), origins, 32).item(), 1)
        self.assertLess(centre[0].item(), 26)
        # The actual face helper takes fixed voxel keys, so both decoded copies
        # keep the face only in group 0 despite those prediction differences.
        face_owner = self.impl["_face_owned_by_group"]
        self.assertTrue(face_owner(lattice, faces, origins, 32, 16, [0]).item())
        self.assertFalse(face_owner(lattice, faces, origins, 32, 16, [1]).item())
        self.assertTrue(face_owner(lattice, faces, origins, 32, 16, [1, 0]).item())

    def test_bounded_batches_and_empty_points(self):
        generator = torch.Generator().manual_seed(4)
        points = torch.rand((137, 3), generator=generator) * 100 - 10
        origins = torch.tensor([[0., 0, 0], [20., 22, 0], [40., 0, 0], [72., 0, 0]])
        owner = self.impl["_exclusive_chunk_owner"]
        torch.testing.assert_close(owner(points, origins, 32, batch_size=1),
                                   owner(points, origins, 32, batch_size=1000))
        self.assertEqual(owner(torch.empty((0, 3)), origins, 32).numel(), 0)


if __name__ == "__main__":
    unittest.main()
