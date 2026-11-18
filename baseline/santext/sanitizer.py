"""SanText mechanisms and a LLaMA-token comparison adapter.

This module is a clean-room implementation of Yue et al. (ACL-IJCNLP 2021).
For a source token ``x`` and candidate output ``y``, SanText samples with

    P[y | x] = softmax_y(-epsilon * ||phi(x) - phi(y)||_2 / 2).

The authors' released implementation materialises the full probability
matrix.  A LLaMA vocabulary makes that matrix unnecessarily large for a
generation-only evaluation, so this adapter evaluates exactly the same rows
on demand in GPU chunks.  This changes neither the distribution nor its
per-token sampling semantics.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import torch


@dataclass(frozen=True)
class SanTextConfig:
    """Paper parameters for SanText or SanText+.

    ``sensitive_fraction`` is ``w`` in the paper.  For SanText+ the release
    chooses the least-frequent observed tokens as sensitive and keeps a
    non-sensitive token unchanged with probability ``1 - p``.  The defaults
    (``w=0.9, p=0.3``) are the settings reported for Table 1 of the paper;
    the public repository's CLI defaults are deliberately not substituted.
    """

    variant: str = "santext"
    epsilon: float = 1.0
    sensitive_fraction: float = 0.9
    non_sensitive_sanitize_probability: float = 0.3
    source_chunk_size: int = 96
    seed: int = 42

    def validate(self) -> None:
        if self.variant not in {"santext", "santext_plus"}:
            raise ValueError("variant must be 'santext' or 'santext_plus'")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number")
        if not 0.0 < self.sensitive_fraction <= 1.0:
            raise ValueError("sensitive_fraction must be in (0, 1]")
        if not 0.0 < self.non_sensitive_sanitize_probability <= 1.0:
            raise ValueError("non_sensitive_sanitize_probability must be in (0, 1]")
        if self.source_chunk_size < 1:
            raise ValueError("source_chunk_size must be positive")


@dataclass(frozen=True)
class SanTextVocabulary:
    """Observed token vocabulary and the paper/release sensitive-token split."""

    source_token_ids: Tuple[int, ...]
    sensitive_token_ids: Tuple[int, ...]
    total_observed_tokens: int
    metadata: Mapping[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_token_ids": list(self.source_token_ids),
            "sensitive_token_ids": list(self.sensitive_token_ids),
            "total_observed_tokens": self.total_observed_tokens,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SanTextVocabulary":
        return cls(
            source_token_ids=tuple(int(item) for item in payload["source_token_ids"]),
            sensitive_token_ids=tuple(int(item) for item in payload["sensitive_token_ids"]),
            total_observed_tokens=int(payload["total_observed_tokens"]),
            metadata=dict(payload.get("metadata", {})),
        )


def build_vocabulary(
    token_counts: Counter,
    config: SanTextConfig,
    *,
    metadata: Mapping[str, object],
) -> SanTextVocabulary:
    """Build ``V`` and ``V_S`` using the authors' frequency rule.

    ``Counter.most_common`` has the same descending-frequency, first-seen tie
    order used by the public SanText code.  Its release selects
    ``words[-int(w*len(words))-1:]``; the extra boundary item is retained here
    for exact release compatibility.
    """

    config.validate()
    ordered = [int(token_id) for token_id, _ in token_counts.most_common()]
    if not ordered:
        raise ValueError("Cannot construct a SanText vocabulary from zero tokens")
    sensitive_count = int(config.sensitive_fraction * len(ordered))
    start = max(0, len(ordered) - sensitive_count - 1)
    sensitive = ordered[start:]
    if not sensitive:
        raise ValueError("SanText+ requires at least one sensitive output token")
    return SanTextVocabulary(
        source_token_ids=tuple(ordered),
        sensitive_token_ids=tuple(sensitive),
        total_observed_tokens=int(sum(token_counts.values())),
        metadata=dict(metadata),
    )


class SanTextTokenSanitizer:
    """Exact SanText rows sampled on demand over a LLaMA-token vocabulary.

    The adaptation uses LLaMA BPE IDs in place of the paper's words.  Its
    source/output vocabulary is built from the public WikiText-2 train and
    validation splits, just as the original implementation builds a task
    vocabulary from train/dev.  Evaluation IDs outside that vocabulary are
    sampled uniformly from ``V`` (the public SanText+ OOV behaviour), so every
    active non-special evaluation token is processed.
    """

    def __init__(
        self,
        vocabulary: SanTextVocabulary,
        config: SanTextConfig,
        *,
        vocab_size: int,
        special_token_ids: Iterable[int],
    ) -> None:
        config.validate()
        if not vocabulary.source_token_ids:
            raise ValueError("SanText vocabulary is empty")
        self.vocabulary = vocabulary
        self.config = config
        self.vocab_size = int(vocab_size)
        self.source_token_ids = tuple(int(item) for item in vocabulary.source_token_ids)
        self.sensitive_token_ids = tuple(int(item) for item in vocabulary.sensitive_token_ids)
        if min(self.source_token_ids) < 0 or max(self.source_token_ids) >= self.vocab_size:
            raise ValueError("SanText vocabulary has an out-of-range token ID")
        if not set(self.sensitive_token_ids).issubset(self.source_token_ids):
            raise ValueError("Sensitive tokens must be a subset of source vocabulary")

        self._source_lookup_cpu = torch.zeros(self.vocab_size, dtype=torch.bool)
        self._source_lookup_cpu[list(self.source_token_ids)] = True
        self._sensitive_lookup_cpu = torch.zeros(self.vocab_size, dtype=torch.bool)
        self._sensitive_lookup_cpu[list(self.sensitive_token_ids)] = True
        self.special_token_ids = frozenset(int(item) for item in special_token_ids)

    def _lookup_on(self, lookup_cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
        return lookup_cpu.to(device=device, non_blocking=True)

    @staticmethod
    def _make_generator(device: torch.device, seed: int) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        return generator

    def _sample_exponential(
        self,
        output: torch.Tensor,
        *,
        sampled_positions: torch.Tensor,
        sampled_ids: torch.Tensor,
        embedding_weight: torch.Tensor,
        candidate_ids: torch.Tensor,
        generator: torch.Generator,
    ) -> None:
        """Sample the exact exponential-mechanism row for each source ID."""

        if sampled_positions.numel() == 0:
            return
        device = embedding_weight.device
        # Float32 distances are important: bfloat16 makes nearby LLaMA token
        # embeddings artificially tied and changes the intended mechanism.
        candidate_vectors = embedding_weight.index_select(0, candidate_ids).float()
        candidate_norms = (candidate_vectors * candidate_vectors).sum(dim=1)

        order = torch.argsort(sampled_ids)
        sorted_ids = sampled_ids.index_select(0, order)
        sorted_positions = sampled_positions.index_select(0, order)
        unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(dim=0)], dim=0
        )

        for start in range(0, int(unique_ids.numel()), self.config.source_chunk_size):
            stop = min(start + self.config.source_chunk_size, int(unique_ids.numel()))
            source_chunk = unique_ids[start:stop]
            source_vectors = embedding_weight.index_select(0, source_chunk).float()
            source_norms = (source_vectors * source_vectors).sum(dim=1, keepdim=True)
            distance_sq = (source_norms + candidate_norms.unsqueeze(0) - 2.0 * (source_vectors @ candidate_vectors.T)).clamp_min_(0.0)
            probabilities = torch.softmax(-0.5 * self.config.epsilon * distance_sq.sqrt_(), dim=1)

            for local_row in range(stop - start):
                row = start + local_row
                left = int(offsets[row].item())
                right = int(offsets[row + 1].item())
                positions = sorted_positions[left:right]
                picked = torch.multinomial(
                    probabilities[local_row],
                    num_samples=int(positions.numel()),
                    replacement=True,
                    generator=generator,
                )
                output.index_copy_(0, positions, candidate_ids.index_select(0, picked))

    def sanitize_token_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embedding_weight: torch.Tensor,
        *,
        seed: int,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Sanitize every active non-special ID and return CPU token IDs/counts."""

        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")
        if embedding_weight.ndim != 2 or embedding_weight.shape[0] != self.vocab_size:
            raise ValueError("embedding_weight does not match the tokenizer vocabulary")
        device = embedding_weight.device
        ids_cpu = input_ids.to(dtype=torch.long, device="cpu")
        mask_cpu = attention_mask.to(dtype=torch.bool, device="cpu")
        special_cpu = torch.zeros_like(mask_cpu)
        for token_id in self.special_token_ids:
            special_cpu |= ids_cpu.eq(token_id)
        eligible_cpu = mask_cpu & ~special_cpu
        flat_ids = ids_cpu.reshape(-1).to(device)
        flat_eligible = eligible_cpu.reshape(-1).to(device)
        output = flat_ids.clone()
        eligible_positions = flat_eligible.nonzero(as_tuple=False).flatten()
        eligible_ids = flat_ids.index_select(0, eligible_positions)
        source_lookup = self._lookup_on(self._source_lookup_cpu, device)
        sensitive_lookup = self._lookup_on(self._sensitive_lookup_cpu, device)
        known = source_lookup.index_select(0, eligible_ids)
        generator = self._make_generator(device, seed)

        source_ids = torch.tensor(self.source_token_ids, dtype=torch.long, device=device)
        if self.config.variant == "santext":
            candidate_ids = source_ids
            exponential_mask = known
            preserved_common = 0
        else:
            candidate_ids = torch.tensor(self.sensitive_token_ids, dtype=torch.long, device=device)
            sensitive = sensitive_lookup.index_select(0, eligible_ids)
            common = known & ~sensitive
            random_values = torch.rand(int(eligible_ids.numel()), device=device, generator=generator)
            exponential_mask = sensitive | (common & (random_values <= self.config.non_sensitive_sanitize_probability))
            preserved_common = int((common & ~exponential_mask).sum().item())

        exponential_positions = eligible_positions[exponential_mask]
        exponential_ids = eligible_ids[exponential_mask]
        self._sample_exponential(
            output,
            sampled_positions=exponential_positions,
            sampled_ids=exponential_ids,
            embedding_weight=embedding_weight,
            candidate_ids=candidate_ids,
            generator=generator,
        )

        # The public SanText+ release replaces an OOV input uniformly from the
        # task vocabulary.  We apply that fallback to either variant to avoid
        # silently leaving a test BPE token untouched.
        oov_mask = ~known
        oov_positions = eligible_positions[oov_mask]
        if oov_positions.numel():
            uniform_indices = torch.randint(
                len(self.source_token_ids),
                size=(int(oov_positions.numel()),),
                device=device,
                generator=generator,
            )
            output.index_copy_(0, oov_positions, source_ids.index_select(0, uniform_indices))

        sanitized = output.reshape_as(ids_cpu).cpu()
        replacement_mask = eligible_cpu & sanitized.ne(ids_cpu)
        counts = {
            "active_tokens": int(mask_cpu.sum().item()),
            "eligible_non_special_tokens": int(eligible_cpu.sum().item()),
            # ``special_cpu`` is intentionally formed on the padded token
            # table so it can be combined with ``mask_cpu`` above.  Report
            # only active special IDs, however: counting right-padding here
            # makes the audit field appear inconsistent with active_tokens
            # even though padding is never sanitized.
            "special_tokens_preserved": int((mask_cpu & special_cpu).sum().item()),
            "in_vocab_tokens": int(known.sum().item()),
            "oov_tokens_uniformly_sampled": int(oov_mask.sum().item()),
            "exponential_sampled_tokens": int(exponential_mask.sum().item()),
            "preserved_non_sensitive_tokens": preserved_common,
            "mapped": int(eligible_cpu.sum().item()),
            "realized_replacements": int(replacement_mask.sum().item()),
        }
        return sanitized.to(torch.long).contiguous(), counts
