"""Embedding perturbation used by the CAPE generation adapter.

Plant et al. first map a text to a contextual BERT representation, min--max
normalise that representation to [0, 1], and add coordinate-wise
``Laplace(1 / epsilon)`` noise.  CAPE then trains task and private-attribute
heads adversarially.  WikiText-2 has no corresponding private attribute, so
the training portion cannot be reproduced on this data.  This module isolates
the paper's DP embedding operation for a clearly labelled frozen-LLaMA
sequence adapter.

The adapter normalises each *active LLaMA embedding sequence* separately so
that padding does not affect the private representation.  This is necessary
for causal-LM evaluation, but differs from the paper's mean-pooled BERT
vector.  In particular, ``epsilon`` below denotes the paper's noise-scale
parameter, not an end-to-end DP claim for this sequence adaptation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class CAPEConfig:
    """Paper DP-noise parameters retained by the LLaMA sequence adapter."""

    epsilon: float
    seed: int = 42

    def validate(self) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number")


def laplace_noise(
    shape: Tuple[int, ...],
    *,
    scale: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample i.i.d. Laplace noise using a deterministic inverse CDF."""

    uniform = torch.rand(shape, device=device, dtype=torch.float32, generator=generator) - 0.5
    # ``torch.rand`` can land sufficiently close to 1 that subtracting 0.5
    # rounds to *exactly* 0.5 in float32.  The inverse CDF would then evaluate
    # log(0) and introduce an artificial infinity.  This only removes the
    # non-representable endpoint of a continuous Laplace distribution; it does
    # not clip any finite perturbation.
    magnitude = uniform.abs().clamp_max(0.5 - torch.finfo(uniform.dtype).eps)
    return -float(scale) * uniform.sign() * torch.log1p(-2.0 * magnitude)


def normalize_and_perturb_sequence(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    config: CAPEConfig,
    *,
    seed: int,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Apply CAPE's min--max plus Laplace operation to one active sequence.

    Args:
        embeddings: ``[length, hidden]`` LLaMA input embeddings on CPU.
        attention_mask: active-token indicator of shape ``[length]``.
        seed: sample-specific seed so PPL and ROUGE see identical noise.

    Returns:
        A sequence of perturbed embeddings and auditable perturbation stats.
    """

    config.validate()
    if embeddings.ndim != 2 or attention_mask.ndim != 1 or embeddings.shape[0] != attention_mask.shape[0]:
        raise ValueError("Expected embeddings [length, hidden] and a matching one-dimensional attention mask")
    if embeddings.device.type != "cpu":
        raise ValueError("The dataset adapter generates deterministic CAPE noise on CPU")

    active = attention_mask.to(dtype=torch.bool, device="cpu")
    active_count = int(active.sum().item())
    output = embeddings.float().clone()
    if not active_count:
        return output, {
            "active_tokens": 0.0,
            "privatized_coordinates": 0.0,
            "normalization_range": 0.0,
            "laplace_scale": 1.0 / config.epsilon,
        }

    values = embeddings[active].float()
    minimum = values.min()
    maximum = values.max()
    value_range = (maximum - minimum).clamp_min(torch.finfo(values.dtype).eps)
    normalized = (values - minimum) / value_range

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = laplace_noise(
        tuple(normalized.shape),
        scale=1.0 / config.epsilon,
        generator=generator,
        device=torch.device("cpu"),
    )
    output[active] = normalized + noise
    return output, {
        "active_tokens": float(active_count),
        "privatized_coordinates": float(normalized.numel()),
        "normalization_range": float(value_range.item()),
        "laplace_scale": 1.0 / config.epsilon,
    }
