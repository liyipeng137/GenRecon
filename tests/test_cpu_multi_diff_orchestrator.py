"""CPU numerical checks against the native sampler at every synchronized step."""
from __future__ import annotations

from fractions import Fraction
import unittest

import torch
from torch import nn

from genrecon.modules.sparse.basic import SparseTensor
from genrecon.pipelines.samplers.cpu_multi_diff_orchestrator import (
    CpuMultiDiffusionOrchestrator, move_sampling_tree,
)
from genrecon.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from genrecon.pipelines.samplers.multi_diff_orchestrator import MultiDiffusionOrchestrator


def features(value):
    return value if isinstance(value, torch.Tensor) else value.feats


def snapshot(values):
    return [features(value).detach().clone() for value in values]


class TracedNative(MultiDiffusionOrchestrator):
    def __init__(self, sampler):
        super().__init__(sampler)
        self.history = []

    def _aggregate_overlaps(self, *args, **kwargs):
        result = super()._aggregate_overlaps(*args, **kwargs)
        self.history.append(snapshot(result))
        return result

    def _aggregate_overlaps_boundary_sensitive(self, *args, **kwargs):
        result = super()._aggregate_overlaps_boundary_sensitive(*args, **kwargs)
        self.history.append(snapshot(result))
        return result


class NonzeroModel(nn.Module):
    """Small nonzero predictor with real learned layers and conditional inputs."""
    def __init__(self, channels=5):
        super().__init__()
        self.resolution = 4
        self.linear1 = nn.Linear(channels, channels * 2)
        self.linear2 = nn.Linear(channels * 2, channels)
        self.calls = []

    @property
    def device(self):
        return self.linear1.weight.device

    def forward(self, x, t, cond, concat_cond=None, gain=1.):
        self.calls.append(float(t[0]))
        sparse = isinstance(x, SparseTensor)
        values = x.feats if sparse else x.movedim(1, -1)
        values = self.linear2(torch.tanh(self.linear1(values)))
        values = values + cond["bias"] + .03 * cond["nested"]["tokens"].mean()
        values = values + .07 * cond["nested"]["sparse"].feats.mean()
        values = values + t[0] / 9000.
        if concat_cond is not None:
            values = values + .11 * (concat_cond.feats if sparse else concat_cond.movedim(1, -1))
        values = torch.tanh(values) * gain
        if sparse:
            # A real model may fill coordinate/attention caches on the device.
            x.register_spatial_cache("model_scratch", torch.ones(7, device=x.device))
            return x.replace(values)
        return values.movedim(-1, 1)


class CpuMultiDiffusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def inputs(self, sparse):
        torch.manual_seed(78)
        model = NonzeroModel().eval()
        # Check that neither model path is a zero-initialized identity fixture.
        self.assertGreater(sum(float(p.detach().abs().sum()) for p in model.parameters()), 1.)
        full = torch.cartesian_prod(torch.arange(4), torch.arange(4), torch.arange(4)).int()
        noise, cond, neg, extra = [], [], [], []
        for index in range(3):
            xyz = full if index == 0 else full[torch.randperm(len(full))[:55-index*7]]
            coords = torch.cat([torch.zeros(len(xyz), 1, dtype=torch.int32), xyz], dim=1)
            cond_sparse = SparseTensor(feats=torch.randn(len(coords), 5), coords=coords)
            # Deliberately cache non-data state to catch SparseTensor.to/cpu use.
            cond_sparse.register_spatial_cache("must_not_transfer", torch.randn(3))
            if sparse:
                current = SparseTensor(feats=torch.randn(len(coords), 5), coords=coords)
                concat = SparseTensor(feats=torch.randn(len(coords), 5), coords=coords)
            else:
                current = torch.randn(1, 5, 4, 4, 4)
                concat = torch.randn_like(current)
            noise.append(current)
            cond.append(dict(bias=torch.randn(5), nested=dict(tokens=torch.randn(2, 3), sparse=cond_sparse)))
            neg.append(dict(bias=torch.randn(5), nested=dict(tokens=torch.randn(2, 3), sparse=cond_sparse)))
            extra.append(dict(concat_cond=concat, gain=.6 + index * .15))
        translations = [torch.tensor([-.5, 0., 0.]), torch.zeros(3), torch.tensor([.25, .5, 0.])]
        return model, noise, cond, neg, translations, extra

    def compare(self, sparse, boundary=False, initialized=False, guidance_rescale=.35):
        model, noise, cond, neg, translations, extra = self.inputs(sparse)
        native = TracedNative(FlowEulerGuidanceIntervalSampler(1e-5))
        offloaded = CpuMultiDiffusionOrchestrator(FlowEulerGuidanceIntervalSampler(1e-5))
        options = dict(steps=5, rescale_t=3., guidance_strength=2.3, guidance_interval=(.3, .95),
                       guidance_rescale=guidance_rescale, verbose=False, chunk_kwargs=extra,
                       boundary_sensitive=boundary, boundary_width=1)
        if initialized:
            torch.manual_seed(128)
            noise = native._initialize_noise(noise, translations, model.resolution)
            # Native always initializes. Override only its initial draw for the
            # comparison of the public pre-coordinated-input option.
            native._initialize_noise = lambda noise, *args: list(noise)
        torch.manual_seed(345)
        expected = native.sample(model, noise, cond, neg, translations, **options)
        history, times = [], []

        def record(step, time, chunks):
            self.assertEqual(step, len(history))
            self.assertTrue(all(features(value).device.type == "cpu" for value in chunks))
            if sparse:
                self.assertTrue(all(value._spatial_cache == {} for value in chunks))
            history.append(snapshot(chunks))
            times.append(time)

        torch.manual_seed(345)
        actual = offloaded.sample(model, noise, cond, neg, translations,
                                  initialize_noise=not initialized, execution_device="cpu",
                                  step_callback=record, **options)
        self.assertEqual(len(history), 5)
        self.assertEqual(times[-1], 0.)
        self.assertTrue(all(a > b for a, b in zip(times[:-1], times[1:])))
        for expected_step, actual_step in zip(native.history, history):
            for left, right in zip(expected_step, actual_step):
                torch.testing.assert_close(left, right, atol=0., rtol=0.)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(features(left), features(right), atol=0., rtol=0.)
            if sparse:
                torch.testing.assert_close(left.coords, right.coords, atol=0., rtol=0.)

    def test_dense_matches_native_at_every_step_with_real_cfg_and_rescale(self):
        self.compare(False)

    def test_sparse_matches_native_at_every_step_with_real_cfg_and_rescale(self):
        self.compare(True)

    def test_boundary_sensitive_sparse_matches_native_at_every_step(self):
        self.compare(True, boundary=True)

    def test_precoordinated_dense_and_sparse_values_are_preserved(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                self.compare(sparse, initialized=True)

    def test_sparse_offload_discards_backend_caches_and_preserves_data_scale_dtype(self):
        value = SparseTensor(feats=torch.randn(3, 5, dtype=torch.bfloat16),
                             coords=torch.tensor([[0, 1, 1, 1], [0, 1, 2, 1], [0, 2, 1, 1]], dtype=torch.int32),
                             scale=(Fraction(2), Fraction(2), Fraction(2)))
        value.register_spatial_cache("sentinel", object())
        copied = move_sampling_tree(dict(value=value, nested=[None, (2, "x")]), "cpu")
        clone = copied["value"]
        self.assertIsNot(clone, value)
        self.assertEqual(clone._spatial_cache, {})
        self.assertEqual(clone._scale, value._scale)
        self.assertEqual(clone.dtype, value.dtype)
        torch.testing.assert_close(clone.feats, value.feats)
        torch.testing.assert_close(clone.coords, value.coords)
        self.assertEqual(copied["nested"], [None, (2, "x")])
        self.assertIsNotNone(value.get_spatial_cache("sentinel"))

    def test_invalid_counts_steps_and_t_schedule_are_rejected(self):
        model, noise, cond, neg, translations, extra = self.inputs(False)
        sampler = CpuMultiDiffusionOrchestrator(FlowEulerGuidanceIntervalSampler(1e-5))
        for kwargs in (dict(steps=0), dict(steps=True), dict(steps=1.5), dict(rescale_t=0.),
                       dict(rescale_t=float("nan")), dict(initialize_noise=0), dict(chunk_kwargs=extra[:1])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sampler.sample(model, noise, cond, neg, translations, verbose=False, **kwargs)
        with self.assertRaisesRegex(ValueError, "matching nonempty"):
            sampler.sample(model, [], [], [], [], verbose=False)
        with self.assertRaisesRegex(ValueError, "matching nonempty"):
            sampler.sample(model, noise, cond[:1], neg, translations, verbose=False)
        with self.assertRaises(TypeError):
            move_sampling_tree(object(), "cpu")


if __name__ == "__main__":
    unittest.main()
