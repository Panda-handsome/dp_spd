"""Frozen LLaMA routing with a DP-Forward perturbation boundary."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import create_causal_mask

from baseline.dp_forward.noise import DPForwardConfig, clip_and_add_official_release_noise


class LlamaDPForwardSplit:
    """Run first ``noise_layer + 1`` LLaMA layers, perturb, then finish."""

    def __init__(self, model, config: DPForwardConfig) -> None:
        self.model = model
        self.config = config
        self.layers = model.model.layers
        config.validate(len(self.layers))
        self.client_layer_count = config.noise_layer + 1
        self.hidden_size = int(model.config.hidden_size)

    def _mask_positions(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, cache=None):
        past_seen = cache.get_seq_length() if cache is not None else 0
        batch, sequence = hidden_states.shape[:2]
        positions = torch.arange(sequence, device=hidden_states.device, dtype=torch.long) + past_seen
        positions = positions.unsqueeze(0).expand(batch, -1)
        causal = create_causal_mask(
            config=self.model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=cache,
            position_ids=positions,
        )
        rotary = self.model.model.rotary_emb(hidden_states, position_ids=positions)
        return causal, positions, rotary

    def _run_layers(self, hidden_states, start: int, end: int, attention_mask, cache=None, use_cache=False):
        causal, positions, rotary = self._mask_positions(hidden_states, attention_mask, cache)
        return self._run_prepared_layers(hidden_states, start, end, causal, positions, rotary, cache, use_cache)

    def _run_prepared_layers(self, hidden_states, start: int, end: int, causal, positions, rotary, cache=None, use_cache=False):
        for layer in self.layers[start:end]:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal,
                position_ids=positions,
                past_key_values=cache,
                use_cache=use_cache,
                position_embeddings=rotary,
            )
        return hidden_states

    def client_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model.model.embed_tokens(input_ids)
        return self._run_layers(hidden, 0, self.client_layer_count, attention_mask)

    def forward(self, input_ids, attention_mask, sample_indices, *, perturb: bool = True) -> torch.Tensor:
        hidden = self.forward_hidden(input_ids, attention_mask, sample_indices, perturb=perturb)
        return self.model.lm_head(hidden)

    def forward_hidden(self, input_ids, attention_mask, sample_indices, *, perturb: bool = True) -> torch.Tensor:
        """Return final normalized hidden states without materializing logits.

        This is used by the IMDb frozen-feature probe.  It follows exactly the
        same client/noise/server route as :meth:`forward`; exposing it avoids a
        32k-vocabulary logits tensor solely to obtain the final representation.
        """

        hidden = self.client_hidden(input_ids, attention_mask)
        if perturb:
            hidden, _ = clip_and_add_official_release_noise(hidden, attention_mask, self.config, sample_indices)
        hidden = self._run_layers(hidden, self.client_layer_count, len(self.layers), attention_mask)
        return self.model.model.norm(hidden)

    def forward_cached(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_indices: torch.Tensor,
        *,
        past_key_values: Optional[DynamicCache] = None,
        perturb: bool = True,
    ) -> tuple[torch.Tensor, DynamicCache]:
        """Cached inference: perturb only the initial private prompt pass."""

        cache = past_key_values if past_key_values is not None else DynamicCache(config=self.model.config)
        is_private_prompt = cache.get_seq_length() == 0
        hidden = self.model.model.embed_tokens(input_ids)
        # Prepare once before *any* layer updates the DynamicCache.  Reusing
        # these positions/mask below mirrors LlamaModel.forward, where every
        # layer in a pass sees the same cache length and RoPE positions.
        causal, positions, rotary = self._mask_positions(hidden, attention_mask, cache)
        hidden = self._run_prepared_layers(
            hidden, 0, self.client_layer_count, causal, positions, rotary, cache=cache, use_cache=True
        )
        if perturb and is_private_prompt:
            hidden, _ = clip_and_add_official_release_noise(hidden, attention_mask, self.config, sample_indices)
        hidden = self._run_prepared_layers(
            hidden, self.client_layer_count, len(self.layers), causal, positions, rotary, cache=cache, use_cache=True
        )
        return self.model.lm_head(self.model.model.norm(hidden)), cache

    @torch.no_grad()
    def assert_clean_equivalence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
        sample_ids = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        reference = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        split = self.forward(input_ids, attention_mask, sample_ids, perturb=False)
        delta = (reference.float() - split.float()).abs()
        maximum, mean = float(delta.max().item()), float(delta.mean().item())
        if not torch.allclose(reference.float(), split.float(), rtol=2e-3, atol=2e-3):
            raise RuntimeError(f"Clean DP-Forward split mismatch: max_abs={maximum:.6g}, mean_abs={mean:.6g}")
        return {"clean_split_max_abs_logit_delta": maximum, "clean_split_mean_abs_logit_delta": mean}
