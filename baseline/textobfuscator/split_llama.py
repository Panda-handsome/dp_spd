"""LLaMA client/server split and TextObfuscator-style perturbation.

Zhou et al. (ACL Findings 2023) split a 12-layer RoBERTa after the first
three client layers, perturb every token hidden state at that boundary with
Laplace scale ``1 / epsilon``, and train the client/server jointly with
prototype losses.  A frozen causal LLaMA on WikiText-2 has neither the
classification task nor labels required by that training.  The code here
therefore implements only the directly transferable inference mechanism:
split after configurable LLaMA layers and perturb every active boundary
hidden-state coordinate.  It deliberately does *not* call this full
TextObfuscator or an end-to-end DP guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import create_causal_mask


@dataclass(frozen=True)
class TextObfuscatorConfig:
    """Paper noise parameters used by the frozen split adapter."""

    epsilon: float
    client_layers: int = 3
    seed: int = 42

    def validate(self, total_layers: int) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number")
        if not 1 <= self.client_layers < total_layers:
            raise ValueError(
                f"client_layers must be in [1, {total_layers - 1}], got {self.client_layers}"
            )


def _laplace_noise(
    shape: Tuple[int, ...], *, scale: float, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    """Deterministic i.i.d. Laplace noise, guarded only at float endpoint."""

    uniform = torch.rand(shape, device=device, dtype=torch.float32, generator=generator) - 0.5
    # In float32 a value close enough to one can round to exactly 0.5 after
    # subtraction.  Excluding that non-representable continuous endpoint
    # avoids a spurious ``log(0)`` infinity without clipping finite noise.
    magnitude = uniform.abs().clamp_max(0.5 - torch.finfo(uniform.dtype).eps)
    return -float(scale) * uniform.sign() * torch.log1p(-2.0 * magnitude)


class LlamaTextObfuscatorSplit:
    """Run a frozen ``LlamaForCausalLM`` through an explicit 3/remaining split."""

    def __init__(self, model, config: TextObfuscatorConfig) -> None:
        self.model = model
        self.config = config
        self.layers = model.model.layers
        config.validate(len(self.layers))
        self.hidden_size = int(model.config.hidden_size)

    def _mask_and_positions(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, sequence_length = inputs_embeds.shape[:2]
        position_ids = torch.arange(sequence_length, device=inputs_embeds.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        causal_mask = create_causal_mask(
            config=self.model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = self.model.model.rotary_emb(inputs_embeds, position_ids=position_ids)
        return causal_mask, position_ids, position_embeddings

    def client_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return all-token hidden states at the client/server boundary."""

        hidden_states = self.model.model.embed_tokens(input_ids)
        causal_mask, position_ids, position_embeddings = self._mask_and_positions(hidden_states, attention_mask)
        for layer in self.layers[: self.config.client_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
        return hidden_states

    def perturb(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_indices: torch.Tensor,
        *,
        privacy_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Perturb every active private token at the client/server boundary.

        ``privacy_length=None`` means all active sequence tokens (PPL).  For
        ROUGE it is the 32-token private prefix; autoregressively produced
        tokens are model outputs and are not re-privatized.  Per-example PRNG
        seeds give PPL and ROUGE exactly the same noise on their common prefix.
        """

        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must match hidden-state batch and sequence dimensions")
        if sample_indices.numel() != hidden_states.shape[0]:
            raise ValueError("sample_indices must contain one id per batch element")

        output = hidden_states.clone()
        sequence_length = hidden_states.shape[1]
        private_end = sequence_length if privacy_length is None else min(int(privacy_length), sequence_length)
        if private_end <= 0:
            return output
        scale = 1.0 / self.config.epsilon
        token_positions = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
        private_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        private_mask = private_mask & (token_positions < private_end)
        for row, dataset_index in enumerate(sample_indices.detach().cpu().tolist()):
            active = private_mask[row]
            count = int(active.sum().item())
            if count == 0:
                continue
            generator = torch.Generator(device=hidden_states.device)
            # The stride isolates PRNG streams even when adjacent sample ids
            # are evaluated in different batches or processes.
            generator.manual_seed(int(self.config.seed + 1_000_003 * int(dataset_index)))
            noise = _laplace_noise(
                (count, self.hidden_size), scale=scale, generator=generator, device=hidden_states.device
            ).to(dtype=hidden_states.dtype)
            output[row, active] = output[row, active] + noise
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_indices: torch.Tensor,
        *,
        privacy_length: Optional[int] = None,
        perturb: bool = True,
    ) -> torch.Tensor:
        """Return LLaMA logits after optional boundary perturbation."""

        hidden_states = self.forward_hidden(
            input_ids,
            attention_mask,
            sample_indices,
            privacy_length=privacy_length,
            perturb=perturb,
        )
        return self.model.lm_head(hidden_states)

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_indices: torch.Tensor,
        *,
        privacy_length: Optional[int] = None,
        perturb: bool = True,
    ) -> torch.Tensor:
        """Return final normalized hidden states without constructing logits.

        The IMDb benchmark trains a linear probe on the frozen LLaMA feature.
        This method intentionally shares the exact split/noise computation of
        :meth:`forward`, while avoiding an otherwise unused vocabulary-logits
        allocation for each batch.
        """

        hidden_states = self.client_hidden(input_ids, attention_mask)
        if perturb:
            hidden_states = self.perturb(
                hidden_states, attention_mask, sample_indices, privacy_length=privacy_length
            )
        # The same RoPE positions and causal mask are valid above and below
        # the split because all layers process the same un-cached sequence.
        causal_mask, position_ids, position_embeddings = self._mask_and_positions(hidden_states, attention_mask)
        for layer in self.layers[self.config.client_layers :]:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
        return self.model.model.norm(hidden_states)

    def forward_cached(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_indices: torch.Tensor,
        *,
        past_key_values: Optional[DynamicCache] = None,
        perturb: bool = True,
    ) -> tuple[torch.Tensor, DynamicCache]:
        """Cached split forward used for autoregressive generation.

        The first call consumes the whole private prompt and receives boundary
        noise.  Each later call consumes one model-generated token with the
        same layer-wise KV cache and deliberately receives no new noise.  This
        is the intermediate-state equivalent of normal ``generate`` using a
        noised prefix, rather than repeatedly recomputing the prompt.
        """

        cache = past_key_values if past_key_values is not None else DynamicCache(config=self.model.config)
        hidden_states = self.model.model.embed_tokens(input_ids)
        past_seen_tokens = cache.get_seq_length()
        sequence_length = hidden_states.shape[1]
        position_ids = torch.arange(sequence_length, device=hidden_states.device, dtype=torch.long)
        position_ids = (position_ids + past_seen_tokens).unsqueeze(0).expand(hidden_states.shape[0], -1)
        causal_mask = create_causal_mask(
            config=self.model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=cache,
            position_ids=position_ids,
        )
        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids=position_ids)
        for layer in self.layers[: self.config.client_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
        # ``past_seen_tokens == 0`` is exactly the private prompt pass.  A
        # continuation token is generated by the model and therefore should
        # not be transmitted as a new private client token.
        if perturb and past_seen_tokens == 0:
            hidden_states = self.perturb(hidden_states, attention_mask, sample_indices, privacy_length=None)
        for layer in self.layers[self.config.client_layers :]:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
        return self.model.lm_head(self.model.model.norm(hidden_states)), cache

    @torch.no_grad()
    def assert_clean_equivalence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
        """Verify that explicit split routing is identical before adding noise."""

        sample_indices = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        reference = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        split_logits = self.forward(
            input_ids, attention_mask, sample_indices, privacy_length=None, perturb=False
        )
        delta = (reference.float() - split_logits.float()).abs()
        max_abs = float(delta.max().item())
        mean_abs = float(delta.mean().item())
        # bfloat16 model math can differ by a rounding unit across transformer
        # releases, but an actual layer-routing mistake is many orders larger.
        if not torch.allclose(reference.float(), split_logits.float(), rtol=2e-3, atol=2e-3):
            raise RuntimeError(
                "The explicit LLaMA split does not reproduce the clean model "
                f"(max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g})."
            )
        return {"clean_split_max_abs_logit_delta": max_abs, "clean_split_mean_abs_logit_delta": mean_abs}
