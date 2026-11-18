"""Explicit Qwen3 client/server splits used by generation baseline adapters.

The existing baseline implementations split LLaMA internals.  Qwen3 uses a
different causal-mask interface, so reusing those classes would silently
produce incorrect logits.  This module mirrors ``Qwen3Model.forward`` on
Transformers 4.51+ / 5.x and exposes a checked split boundary for the two
methods that perturb contextual states (DP-Forward and TextObfuscator).

This is intentionally an inference adapter: neither original baseline was
trained for CNN/DailyMail summarisation with Qwen3.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from baseline.dp_forward.noise import DPForwardConfig, clip_and_add_official_release_noise
from baseline.textobfuscator.split_llama import TextObfuscatorConfig, _laplace_noise


class Qwen3SplitBase:
    """Faithful explicit execution of a contiguous Qwen3 decoder-layer split."""

    def __init__(self, model, client_layers: int) -> None:
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise TypeError("Expected a HuggingFace Qwen3ForCausalLM-compatible model")
        self.model = model
        self.layers = model.model.layers
        self.client_layers = int(client_layers)
        if not 1 <= self.client_layers < len(self.layers):
            raise ValueError(
                f"client_layers must be in [1, {len(self.layers) - 1}], got {self.client_layers}"
            )
        self.hidden_size = int(model.config.hidden_size)

    def _prepare(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, cache=None):
        past_seen_tokens = cache.get_seq_length() if cache is not None else 0
        sequence_length = int(inputs_embeds.shape[1])
        position_ids = torch.arange(sequence_length, device=inputs_embeds.device, dtype=torch.long)
        position_ids = (position_ids + past_seen_tokens).unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        mask_kwargs = {
            "config": self.model.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "position_ids": position_ids,
        }
        masks = {"full_attention": create_causal_mask(**mask_kwargs)}
        if getattr(self.model.model, "has_sliding_layers", False):
            masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        position_embeddings = self.model.model.rotary_emb(inputs_embeds, position_ids)
        return masks, position_ids, position_embeddings

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        start: int,
        end: int,
        masks,
        position_ids: torch.Tensor,
        position_embeddings,
        *,
        cache=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        for layer_index in range(start, end):
            layer = self.layers[layer_index]
            layer_type = self.model.config.layer_types[layer_index]
            hidden_states = layer(
                hidden_states,
                attention_mask=masks[layer_type],
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
        return hidden_states

    def client_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.model.embed_tokens(input_ids)
        masks, position_ids, position_embeddings = self._prepare(hidden_states, attention_mask)
        return self._run_layers(
            hidden_states,
            0,
            self.client_layers,
            masks,
            position_ids,
            position_embeddings,
        )

    def _finish(self, hidden_states: torch.Tensor, masks, position_ids, position_embeddings, *, cache=None, use_cache=False):
        hidden_states = self._run_layers(
            hidden_states,
            self.client_layers,
            len(self.layers),
            masks,
            position_ids,
            position_embeddings,
            cache=cache,
            use_cache=use_cache,
        )
        return self.model.model.norm(hidden_states)

    @torch.no_grad()
    def assert_clean_equivalence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
        reference = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        sample_indices = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        split = self.forward(input_ids, attention_mask, sample_indices, perturb=False)
        delta = (reference.float() - split.float()).abs()
        maximum = float(delta.max().item())
        mean = float(delta.mean().item())
        if not torch.allclose(reference.float(), split.float(), rtol=2e-3, atol=2e-3):
            raise RuntimeError(
                "Qwen3 explicit split does not reproduce the clean model "
                f"(max_abs={maximum:.6g}, mean_abs={mean:.6g})."
            )
        return {
            "clean_split_max_abs_logit_delta": maximum,
            "clean_split_mean_abs_logit_delta": mean,
        }


class Qwen3DPForwardSplit(Qwen3SplitBase):
    """Qwen3 deep-boundary route with the DP-Forward release operator."""

    def __init__(self, model, config: DPForwardConfig) -> None:
        config.validate(len(model.model.layers))
        self.config = config
        super().__init__(model, client_layers=config.noise_layer + 1)

    def forward(self, input_ids, attention_mask, sample_indices, *, perturb: bool = True) -> torch.Tensor:
        hidden_states = self.model.model.embed_tokens(input_ids)
        masks, position_ids, position_embeddings = self._prepare(hidden_states, attention_mask)
        hidden_states = self._run_layers(
            hidden_states, 0, self.client_layers, masks, position_ids, position_embeddings
        )
        if perturb:
            hidden_states, _ = clip_and_add_official_release_noise(
                hidden_states, attention_mask, self.config, sample_indices
            )
        hidden_states = self._finish(hidden_states, masks, position_ids, position_embeddings)
        return self.model.lm_head(hidden_states)

    def forward_cached(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_indices: torch.Tensor,
        *,
        past_key_values: Optional[DynamicCache] = None,
        perturb: bool = True,
    ) -> tuple[torch.Tensor, DynamicCache]:
        cache = past_key_values if past_key_values is not None else DynamicCache(config=self.model.config)
        is_private_prompt = cache.get_seq_length() == 0
        hidden_states = self.model.model.embed_tokens(input_ids)
        masks, position_ids, position_embeddings = self._prepare(hidden_states, attention_mask, cache)
        hidden_states = self._run_layers(
            hidden_states,
            0,
            self.client_layers,
            masks,
            position_ids,
            position_embeddings,
            cache=cache,
            use_cache=True,
        )
        if perturb and is_private_prompt:
            hidden_states, _ = clip_and_add_official_release_noise(
                hidden_states, attention_mask, self.config, sample_indices
            )
        hidden_states = self._finish(
            hidden_states, masks, position_ids, position_embeddings, cache=cache, use_cache=True
        )
        return self.model.lm_head(hidden_states), cache


class Qwen3TextObfuscatorSplit(Qwen3SplitBase):
    """Qwen3 split with TextObfuscator-style boundary Laplace noise."""

    def __init__(self, model, config: TextObfuscatorConfig) -> None:
        config.validate(len(model.model.layers))
        self.config = config
        super().__init__(model, client_layers=config.client_layers)

    def perturb(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_indices: torch.Tensor,
        *,
        privacy_length: int | None = None,
    ) -> torch.Tensor:
        output = hidden_states.clone()
        sequence_length = int(hidden_states.shape[1])
        private_end = sequence_length if privacy_length is None else min(int(privacy_length), sequence_length)
        positions = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
        private_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool) & (positions < private_end)
        for row, sample_index in enumerate(sample_indices.detach().cpu().tolist()):
            active = private_mask[row]
            count = int(active.sum().item())
            if count == 0:
                continue
            generator = torch.Generator(device=hidden_states.device)
            generator.manual_seed(int(self.config.seed + 1_000_003 * int(sample_index)))
            noise = _laplace_noise(
                (count, self.hidden_size),
                scale=1.0 / self.config.epsilon,
                generator=generator,
                device=hidden_states.device,
            ).to(dtype=hidden_states.dtype)
            output[row, active] = output[row, active] + noise
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_indices: torch.Tensor,
        *,
        privacy_length: int | None = None,
        perturb: bool = True,
    ) -> torch.Tensor:
        hidden_states = self.model.model.embed_tokens(input_ids)
        masks, position_ids, position_embeddings = self._prepare(hidden_states, attention_mask)
        hidden_states = self._run_layers(
            hidden_states, 0, self.client_layers, masks, position_ids, position_embeddings
        )
        if perturb:
            hidden_states = self.perturb(
                hidden_states, attention_mask, sample_indices, privacy_length=privacy_length
            )
        hidden_states = self._finish(hidden_states, masks, position_ids, position_embeddings)
        return self.model.lm_head(hidden_states)

    def forward_cached(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_indices: torch.Tensor,
        *,
        past_key_values: Optional[DynamicCache] = None,
        perturb: bool = True,
    ) -> tuple[torch.Tensor, DynamicCache]:
        cache = past_key_values if past_key_values is not None else DynamicCache(config=self.model.config)
        is_private_prompt = cache.get_seq_length() == 0
        hidden_states = self.model.model.embed_tokens(input_ids)
        masks, position_ids, position_embeddings = self._prepare(hidden_states, attention_mask, cache)
        hidden_states = self._run_layers(
            hidden_states,
            0,
            self.client_layers,
            masks,
            position_ids,
            position_embeddings,
            cache=cache,
            use_cache=True,
        )
        if perturb and is_private_prompt:
            hidden_states = self.perturb(hidden_states, attention_mask, sample_indices)
        hidden_states = self._finish(
            hidden_states, masks, position_ids, position_embeddings, cache=cache, use_cache=True
        )
        return self.model.lm_head(hidden_states), cache
