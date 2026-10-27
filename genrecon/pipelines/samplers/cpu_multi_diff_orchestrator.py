"""MultiDiffusion with CPU-resident state and one execution-device chunk.

The schedule, Euler update, coordinated noise, and overlap averaging follow
``MultiDiffusionOrchestrator``. Moving state between devices does not change
which chunks contribute at a step: *all* chunks advance from the preceding
synchronized state, then the complete step is averaged on CPU.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Any

import numpy as np
import torch
import tqdm

from ...modules.sparse.basic import SparseTensor
from .multi_diff_orchestrator import MultiDiffusionOrchestrator


def move_sampling_tree(value: Any, device: str | torch.device) -> Any:
    """Move tensor leaves, rebuilding sparse tensors without backend caches.

    ``SparseTensor.to`` preserves its spatial cache, which can retain CUDA
    attention/kernel indices in an otherwise CPU-resident condition or state.
    Rebuilding from the data leaves also ensures each device uses its own maps.
    """
    if isinstance(value, SparseTensor):
        return SparseTensor(
            feats=value.feats.detach().to(device=device),
            coords=value.coords.detach().to(device=device),
            shape=value.shape,
            scale=value._scale,
        )
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device)
    if isinstance(value, dict):
        return {key: move_sampling_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_sampling_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_sampling_tree(item, device) for item in value)
    if value is None or isinstance(value, (str, bool, int, float, torch.dtype, torch.device)):
        return value
    raise TypeError(f"Unsupported sampling state leaf: {type(value).__name__}")


class CpuMultiDiffusionOrchestrator(MultiDiffusionOrchestrator):
    """Keep condition/state lists on CPU; run one chunk at a time on the model.

    The model must already be on ``execution_device`` and in evaluation mode.
    Callers can retain their usual autocast context around ``sample``. Each
    chunk's device copies and sparse backend caches are released before the
    next chunk. The returned state, and callback state, are always on CPU.

    Pass CPU conditions for bounded residency; caller-owned device objects
    naturally remain alive while the caller retains them. Sparse coordinate
    sets must be fixed during the Euler sampling phase, as in the native path.
    """

    @torch.no_grad()
    def sample(
        self,
        model,
        noise: Sequence[Any],
        cond: Sequence[dict],
        neg_cond: Sequence[dict],
        relative_transl: Sequence[torch.Tensor],
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: tuple[float, float] = (0.0, 1.0),
        guidance_rescale: float = 0.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        chunk_kwargs: Sequence[dict] | None = None,
        boundary_sensitive: bool = False,
        boundary_width: int = 2,
        *,
        execution_device: str | torch.device | None = None,
        initialize_noise: bool = True,
        step_callback: Callable[[int, float, list[Any]], None] | None = None,
    ) -> list[Any]:
        """Sample using native MultiDiffusion math with CPU overlap reduction.

        ``initialize_noise=True`` draws one CPU noise value per global voxel,
        preserving native coordinated-noise semantics. CPU and CUDA random
        generators need not produce the same seeded stream. For controlled
        comparisons, supply identical pre-coordinated ``noise`` and set
        ``initialize_noise=False``; that path preserves its values exactly.

        ``step_callback(step_index, t_prev, chunks)`` runs after each complete
        overlap synchronization, with zero-based indices. It must not mutate
        ``chunks``; clone any data retained for later inspection.
        """
        count = len(noise)
        if count < 1 or any(len(values) != count for values in (cond, neg_cond, relative_transl)):
            raise ValueError("noise, cond, neg_cond and relative_transl need matching nonempty chunk counts")
        if chunk_kwargs is not None and len(chunk_kwargs) != count:
            raise ValueError("chunk_kwargs must have one entry per chunk")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(rescale_t) or rescale_t <= 0:
            raise ValueError("rescale_t must be finite and positive")
        if not isinstance(initialize_noise, bool):
            raise ValueError("initialize_noise must be a boolean")
        if execution_device is None:
            execution_device = getattr(model, "device", None)
            if execution_device is None:
                execution_device = next(model.parameters()).device
        execution_device = torch.device(execution_device)
        relative_cpu = [translation.detach().to(device="cpu") for translation in relative_transl]
        chunks = [move_sampling_tree(value, "cpu") for value in noise]
        cond_cpu = [move_sampling_tree(value, "cpu") for value in cond]
        neg_cond_cpu = [move_sampling_tree(value, "cpu") for value in neg_cond]
        kwargs_cpu = [move_sampling_tree(value, "cpu") for value in chunk_kwargs] if chunk_kwargs is not None else None
        if initialize_noise:
            chunks = self._initialize_noise(chunks, relative_cpu, model.resolution)

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        for step, (t, t_prev) in enumerate(tqdm.tqdm(t_pairs, desc=tqdm_desc, disable=not verbose)):
            for index in range(count):
                current = move_sampling_tree(chunks[index], execution_device)
                positive = move_sampling_tree(cond_cpu[index], execution_device)
                negative = move_sampling_tree(neg_cond_cpu[index], execution_device)
                extra = move_sampling_tree(kwargs_cpu[index], execution_device) if kwargs_cpu is not None else {}
                velocity = self.sampler._inference_model(
                    model, current, t, cond=positive, neg_cond=negative,
                    guidance_strength=guidance_strength,
                    guidance_interval=guidance_interval,
                    guidance_rescale=guidance_rescale,
                    **extra,
                )
                updated = current - (t - t_prev) * velocity
                chunks[index] = move_sampling_tree(updated, "cpu")
                del current, positive, negative, extra, velocity, updated
            if boundary_sensitive:
                chunks = self._aggregate_overlaps_boundary_sensitive(
                    chunks, relative_cpu, model.resolution, boundary_width=boundary_width)
            else:
                chunks = self._aggregate_overlaps(chunks, relative_cpu, model.resolution)
            if step_callback is not None:
                step_callback(step, t_prev, chunks)
        return chunks
