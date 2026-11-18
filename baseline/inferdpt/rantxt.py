"""RANTEXT, the perturbation module of InferDPT.

This clean-room implementation follows Algorithm 2 and Eqs. (11)--(22) of
Tong et al. (TDSC 2025).  The authors publish GPT-4's cl100k vocabulary and
text-embedding-ada-002 vectors as inputs.  For a local LLaMA comparison the
same algorithm is applied to a public, LLaMA-token vocabulary and its input
embedding table; this representation substitution is intentionally exposed in
the result metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import torch


@dataclass(frozen=True)
class RANTextConfig:
    """Parameters of the paper's random-adjacency exponential mechanism."""

    epsilon: float
    source_chunk_size: int = 64
    seed: int = 42

    def validate(self) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number")
        if self.source_chunk_size < 1:
            raise ValueError("source_chunk_size must be positive")


class RANTextSanitizer:
    """On-demand RANTEXT sampling over a supplied model-token vocabulary.

    For every input token in ``candidate_token_ids`` we draw the paper's
    coordinate-wise Laplace vector, form a random adjacency list by the
    induced Euclidean radius, then sample from that list with the exponential
    mechanism.  Tokens outside the advertised vocabulary are discarded, just
    as Algorithm 2 specifies; callers receive an explicit coverage count.
    """

    def __init__(
        self,
        candidate_token_ids: Sequence[int],
        config: RANTextConfig,
        *,
        vocab_size: int,
    ) -> None:
        config.validate()
        unique = tuple(dict.fromkeys(int(item) for item in candidate_token_ids))
        if not unique:
            raise ValueError("candidate_token_ids is empty")
        if min(unique) < 0 or max(unique) >= vocab_size:
            raise ValueError("candidate_token_ids contains an out-of-range ID")
        self.candidate_token_ids = unique
        self.config = config
        self.vocab_size = int(vocab_size)
        self._lookup_cpu = torch.zeros(self.vocab_size, dtype=torch.bool)
        self._lookup_cpu[list(unique)] = True

    @staticmethod
    def _noise_denominator(epsilon: float) -> float:
        """Paper Eq. (13), with the continuous epsilon=2 boundary fix.

        The public release branches on ``epsilon < 2`` but then divides by
        zero for exactly 2.0.  Eq. (13)'s first branch is the finite limiting
        choice at that boundary, so it is used for ``epsilon <= 2``.
        """

        if epsilon <= 2.0:
            return epsilon
        argument = epsilon * 19.064721649556482 - 38.1294334077209
        if argument <= 0:
            return epsilon
        return 0.01658160142016071 * math.log(argument) + 9.311083811697406

    @staticmethod
    def _laplace_noise(
        shape: Tuple[int, int],
        scale: torch.Tensor,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        """Draw independent coordinate-wise Laplace noise reproducibly."""

        uniform = torch.rand(shape, device=device, generator=generator) - 0.5
        # Inverse CDF of Laplace(0, scale), avoiding an extra distribution
        # object and retaining the CUDA generator used for token sampling.  A
        # float32 draw close enough to one can round to the inverse-CDF
        # endpoint after subtracting 0.5; omit only that non-finite endpoint.
        magnitude = uniform.abs().clamp_max(0.5 - torch.finfo(uniform.dtype).eps)
        return -scale.unsqueeze(0) * uniform.sign() * torch.log1p(-2.0 * magnitude)

    def sanitize_token_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embedding_weight: torch.Tensor,
        *,
        seed: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Return sampled IDs, their retained mask, and coverage statistics.

        ``retained_mask`` is false for an inactive input or for a token outside
        the advertised RANTEXT vocabulary.  The latter is intentionally
        discarded to reproduce Algorithm 2 rather than silently retained.
        """

        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")
        if embedding_weight.ndim != 2 or embedding_weight.shape[0] != self.vocab_size:
            raise ValueError("embedding weight does not match the tokenizer vocabulary")
        device = embedding_weight.device
        ids_cpu = input_ids.to(dtype=torch.long, device="cpu")
        active_cpu = attention_mask.to(dtype=torch.bool, device="cpu")
        flat_ids = ids_cpu.reshape(-1).to(device)
        flat_active = active_cpu.reshape(-1).to(device)
        lookup = self._lookup_cpu.to(device=device, non_blocking=True)
        flat_known = lookup.index_select(0, flat_ids)
        retained = flat_active & flat_known
        positions = retained.nonzero(as_tuple=False).flatten()
        source_ids = flat_ids.index_select(0, positions)
        output = flat_ids.clone()

        candidate_ids = torch.tensor(self.candidate_token_ids, dtype=torch.long, device=device)
        candidate_vectors = embedding_weight.index_select(0, candidate_ids).float()
        candidate_norms = (candidate_vectors * candidate_vectors).sum(dim=1)
        # Δφ in Eq. (12): per-coordinate maximum range over the published V.
        sensitivity = candidate_vectors.max(dim=0).values - candidate_vectors.min(dim=0).values
        denominator = self._noise_denominator(self.config.epsilon)
        scale = sensitivity / denominator
        generator = torch.Generator(device=device)
        generator.manual_seed(int(self.config.seed if seed is None else seed))

        adjacency_total = 0
        adjacency_min = len(self.candidate_token_ids)
        adjacency_max = 0
        for start in range(0, int(source_ids.numel()), self.config.source_chunk_size):
            stop = min(start + self.config.source_chunk_size, int(source_ids.numel()))
            source_chunk = source_ids[start:stop]
            source_vectors = embedding_weight.index_select(0, source_chunk).float()
            source_norms = (source_vectors * source_vectors).sum(dim=1, keepdim=True)
            noise = self._laplace_noise(
                (int(source_chunk.numel()), int(source_vectors.shape[1])), scale,
                generator=generator,
                device=device,
            )
            threshold = noise.norm(dim=1).clamp_min_(1e-12)
            distance_sq = (
                source_norms + candidate_norms.unsqueeze(0) - 2.0 * (source_vectors @ candidate_vectors.T)
            ).clamp_min_(0.0)
            distances = distance_sq.sqrt_()
            adjacency = distances < threshold.unsqueeze(1)
            # A token is always a candidate for itself, but this defensive
            # fallback retains a valid distribution under numerical underflow.
            missing = ~adjacency.any(dim=1)
            if missing.any():
                nearest = distances.argmin(dim=1)
                adjacency[missing, nearest[missing]] = True
            utility = 1.0 - distances / threshold.unsqueeze(1)
            logits = (self.config.epsilon / 2.0) * utility
            logits = logits.masked_fill(~adjacency, -torch.inf)
            probabilities = torch.softmax(logits, dim=1)
            sampled_indices = torch.multinomial(probabilities, 1, replacement=True, generator=generator).squeeze(1)
            output.index_copy_(0, positions[start:stop], candidate_ids.index_select(0, sampled_indices))
            adjacency_sizes = adjacency.sum(dim=1)
            adjacency_total += int(adjacency_sizes.sum().item())
            adjacency_min = min(adjacency_min, int(adjacency_sizes.min().item()))
            adjacency_max = max(adjacency_max, int(adjacency_sizes.max().item()))

        sampled = output.reshape_as(ids_cpu).cpu().to(torch.long).contiguous()
        retained_cpu = retained.reshape_as(active_cpu).cpu().to(torch.bool).contiguous()
        active_count = int(active_cpu.sum().item())
        sampled_count = int(retained_cpu.sum().item())
        changed = int((retained_cpu & sampled.ne(ids_cpu)).sum().item())
        return sampled, retained_cpu, {
            "active_tokens": active_count,
            "in_vocabulary_tokens": sampled_count,
            "discarded_out_of_vocabulary_tokens": active_count - sampled_count,
            "coverage_fraction": float(sampled_count / active_count) if active_count else 0.0,
            "sampled_tokens": sampled_count,
            "realized_replacements": changed,
            "realized_replacement_fraction": float(changed / sampled_count) if sampled_count else 0.0,
            "adjacency_mean_size": float(adjacency_total / sampled_count) if sampled_count else 0.0,
            "adjacency_min_size": float(adjacency_min if sampled_count else 0),
            "adjacency_max_size": float(adjacency_max),
            "laplace_denominator_z": float(denominator),
        }
