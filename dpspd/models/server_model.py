# -*- coding: utf-8 -*-
# Server-side model wrapper. The server consumes client embeddings and does not add noise.

import torch
from torch import nn, Tensor
from transformers import PreTrainedModel
from typing import Optional, Dict, Any, Tuple, List

# ----------------------------
# Utilities
# ----------------------------
def get_input_embeddings_tensor(input_ids: Tensor, base_model: PreTrainedModel, device: torch.device) -> Tensor:
    """
    Obtain token embeddings from the base HF model.
    Used only if client does not provide embeddings directly.
    """
    emb_layer = base_model.get_input_embeddings()
    if emb_layer is None:
        raise ValueError("Base model must expose input embeddings via get_input_embeddings()")
    emb_layer = emb_layer.to(device)
    return emb_layer(input_ids.to(device))


# ----------------------------
# Server-side model wrapper
# ----------------------------

class PrivacyAwareModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel, args):
        super().__init__()
        self.base_model = base_model
        self.args = args
        self.base_model.config.output_hidden_states = True

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        embeddings: Optional[Tensor] = None,
        noisy_embeddings: Optional[Tensor] = None,
        prefix_hashes: Optional[Tensor] = None,  # [B]
        **kwargs
    ):
        # Preferred DP-SPD path: the client has already clipped/noised the
        # embeddings and sends them as noisy_embeddings. input_ids is a fallback
        # for non-private or debugging runs.
        if noisy_embeddings is not None:
            inputs_embeds = noisy_embeddings
        elif embeddings is not None:
            inputs_embeds = embeddings
        else:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Without prefix hashes there is no safe grouping opportunity, so the
        # wrapper behaves like a normal HF model invoked through inputs_embeds.
        if prefix_hashes is None or inputs_embeds.size(0) <= 1:
            return self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

        # prefix_hashes groups the real query with its dummy candidates. Within
        # each group, shared prefix computation can be reused before processing
        # the different suffixes.
        unique_hashes, inverse_indices = torch.unique(prefix_hashes, return_inverse=True)

        B, L, D = inputs_embeds.shape
        all_hidden_states = torch.zeros(B, L, self.base_model.config.hidden_size, device=inputs_embeds.device)

        for i, h_val in enumerate(unique_hashes):
            group_indices = (inverse_indices == i).nonzero(as_tuple=True)[0]
            if len(group_indices) == 0: continue

            rep_idx = group_indices[0]
            rep_embeds = inputs_embeds[rep_idx]
            rep_mask = attention_mask[rep_idx]

            # Locate the first token position where two candidates in the same
            # group differ. Everything before it is the reusable prefix.
            prefix_len = 0
            if len(group_indices) > 1:
                diff = (inputs_embeds[group_indices[0]] - inputs_embeds[group_indices[1]]).abs().sum(dim=-1)
                diff_indices = (diff > 1e-6).nonzero()
                if len(diff_indices) > 0:
                    prefix_len = diff_indices[0].item()
                else:
                    prefix_len = L
            else:
                prefix_len = L

            if prefix_len == 0:
                 # No shared prefix was detected; fall back to a regular batched
                 # forward for this group.
                 group_embeds = inputs_embeds[group_indices]
                 group_mask = attention_mask[group_indices]
                 out = self.base_model(inputs_embeds=group_embeds, attention_mask=group_mask)
                 hidden = out.hidden_states[-1] if out.hidden_states else out.last_hidden_state
                 all_hidden_states[group_indices] = hidden
                 continue

            prefix_embeds = rep_embeds[:prefix_len].unsqueeze(0)
            prefix_mask = rep_mask[:prefix_len].unsqueeze(0)

            with torch.no_grad():
                # Cache the prefix once. The suffix pass expands the cached KV
                # tensors across all candidates in this group.
                prefix_out = self.base_model(
                    inputs_embeds=prefix_embeds,
                    attention_mask=prefix_mask,
                    use_cache=True
                )

            prefix_hidden = prefix_out.hidden_states[-1] if prefix_out.hidden_states else prefix_out.last_hidden_state

            if prefix_len == L:
                expanded_hidden = prefix_hidden.expand(len(group_indices), -1, -1)
                all_hidden_states[group_indices] = expanded_hidden
                continue

            past_key_values = prefix_out.past_key_values

            group_embeds = inputs_embeds[group_indices]
            suffix_embeds = group_embeds[:, prefix_len:, :]
            suffix_attn_mask = attention_mask[group_indices]

            # Run only the non-shared suffix with expanded past_key_values. This
            # is where grouped serving saves compute for real+dummy batches.
            suffix_out = self.base_model(
                inputs_embeds=suffix_embeds,
                attention_mask=suffix_attn_mask,
                past_key_values=self._expand_past_kv(past_key_values, len(group_indices)),
                use_cache=True
            )

            prefix_hidden_expanded = prefix_hidden.expand(len(group_indices), -1, -1)
            suffix_hidden = suffix_out.hidden_states[-1] if suffix_out.hidden_states else suffix_out.last_hidden_state

            full_hidden = torch.cat([prefix_hidden_expanded, suffix_hidden], dim=1)
            all_hidden_states[group_indices] = full_hidden

        from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=all_hidden_states,
            hidden_states=(all_hidden_states,)
        )

    def _expand_past_kv(self, past_key_values, batch_size):
        # HF cache tensors are shaped with batch as the first dimension; expand
        # the single-prefix cache to all candidates in the group.
        expanded_past = []
        for layer_past in past_key_values:
            key, value = layer_past
            key = key.expand(batch_size, -1, -1, -1)
            value = value.expand(batch_size, -1, -1, -1)
            expanded_past.append((key, value))
        return tuple(expanded_past)

# ----------------------------
# Classification and Regression heads
# ----------------------------
class clsModel(nn.Module):
    """
    Lightweight classification head that returns logits.
    """
    def __init__(self, input_dim: int, n_class: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, n_class),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class regModel(nn.Module):
    """
    Lightweight regression head that returns one scalar per example.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
