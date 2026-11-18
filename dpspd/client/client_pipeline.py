# -*- coding: utf-8 -*-
# Client-side DP-SPD data pipeline.

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Sequence
from dataclasses import dataclass
import math
import time
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import Dataset


@dataclass
class LDPNoiseConfig:
    mechanism: str = "Laplace"
    epsilon_private: float = 1.0
    epsilon_non_sensitive: float = 1.0
    clip_norm: Optional[float] = None
    seed: Optional[int] = 42

@dataclass
class DummyConfig:
    m_private_tokens: int = 1
    sensitive_ratio: float = 1.0
    n_dummies_per_text: int = 3
    fixed_private_positions: Optional[List[int]] = None
    dummy_mode: str = "token"
    dummy_corpus: Optional[List[str]] = None
    pos_mapping_path: Optional[str] = None

@dataclass
class BatchConfig:
    max_length: int
    device: torch.device
    hidden_size: int
    # When True, the first shared prefix receives a stable group id so the
    # server can reuse prefix computation across the real item and its dummies.
    preserve_prefix_hash: bool = True

class PrivacyDataset(Dataset):
    def __init__(self,
                 texts: List[str],
                 tokenizer,
                 batch_cfg: BatchConfig,
                 dummy_cfg: DummyConfig,
                 ldp_cfg: LDPNoiseConfig,
                 seed: Optional[int] = 42,
                 shuffle_mode: str = "global",
                 pretrained_embeddings: Optional[torch.nn.Module] = None,
                 base_model: Optional[torch.nn.Module] = None
                 ):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.batch_cfg = batch_cfg
        self.dummy_cfg = dummy_cfg
        self.ldp_cfg = ldp_cfg
        self.seed = seed if seed is not None else 42
        self.special_token_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        self.pad_token_id = self._resolve_pad_token_id()

        # Optional POS mapping restricts token-level dummy/private selection to
        # positions whose token ids are known in the saved mapping.
        if self.dummy_cfg.dummy_mode == "token" and self.dummy_cfg.pos_mapping_path:
            try:
                mapping = torch.load(self.dummy_cfg.pos_mapping_path)
                self.id_to_pos = mapping['id_to_pos']
                self.pos_to_ids = mapping['pos_to_ids']
            except FileNotFoundError:
                self.id_to_pos = {}
                self.pos_to_ids = {}
        else:
            self.id_to_pos = {}
            self.pos_to_ids = {}

        # Sentence dummies sample replacement texts from a corpus; using the
        # input set as a fallback keeps the dataset self-contained.
        if self.dummy_cfg.dummy_mode == "sentence" and self.dummy_cfg.dummy_corpus is None:
            self.dummy_cfg.dummy_corpus = texts

        self.device = torch.device('cpu')

        # The client builds noisy embeddings on CPU. If the caller supplies the
        # model embedding layer, clone it so later model moves do not affect this
        # deterministic preprocessing path.
        if pretrained_embeddings is not None:
            cloned_weight = pretrained_embeddings.weight.detach().cpu().clone()
            self.emb_layer = torch.nn.Embedding.from_pretrained(cloned_weight)
        else:
            self.emb_layer = torch.nn.Embedding(tokenizer.vocab_size, batch_cfg.hidden_size).to(self.device)

        if getattr(dummy_cfg, 'm_private_tokens', 0) > 0 and batch_cfg.max_length > 0:
             calculated_ratio = dummy_cfg.m_private_tokens / batch_cfg.max_length
             self.sensitive_ratio = max(0.0, min(1.0, calculated_ratio))
        else:
             self.sensitive_ratio = dummy_cfg.sensitive_ratio

        self.generation_cache = []
        # Keep client-side continuation-generation timing separate from the
        # rest of dataset construction.  This is intentionally available to
        # callers because proxy-generator scaling experiments must distinguish
        # LM decoding cost from CPU-side DP noise/projection work.
        self.generation_timing = {
            "generation_seconds": 0.0,
            "generation_calls": 0,
            "generated_sequences": 0,
            "generated_tokens": 0,
        }
        n_requested_dummies = int(self.dummy_cfg.n_dummies_per_text or 0)
        if self.dummy_cfg.dummy_mode == "generation" and n_requested_dummies > 0:
            assert base_model is not None, "base_model is required when dummy_mode='generation'."
            print(f"\n[DP-SPD] Privatizing prefixes and generating {dummy_cfg.n_dummies_per_text} continuation dummies per input...")
            base_model.eval()

            for g_idx, text in enumerate(tqdm(texts, desc="Generating Suffixes")):
                enc = self._encode_text(text)
                input_ids = enc['input_ids'][0].to(self.device)
                attn_mask = enc['attention_mask'][0].to(self.device)
                real_len, n_private, prefix_len = self._compute_private_layout(attn_mask)

                with torch.no_grad():
                    clean_emb = self.emb_layer(input_ids)

                # Continuation dummies share the same noised prefix as the real
                # query, then branch into generated suffixes. The prefix is
                # projected only for dummy generation; returned training/eval
                # batches still expose noisy embeddings to the server.
                noisy_prefix_emb = self._privatize_prefix_embeddings(
                    clean_emb[:prefix_len],
                    group_index=g_idx,
                )
                projected_prefix_ids = self._project_embeddings_to_tokens(
                    noisy_prefix_emb,
                    original_ids=input_ids[:prefix_len],
                )
                prefix_ids = projected_prefix_ids.unsqueeze(0).to(self._module_device(base_model))
                prefix_mask = torch.ones_like(prefix_ids)

                k = self.dummy_cfg.n_dummies_per_text
                # CUDA launches are asynchronous.  Synchronize around the
                # actual `generate` call so the recorded time is suitable for
                # comparing continuation-model scales.  Dataset creation may
                # still include additional CPU projection/noise overhead,
                # which callers can measure independently.
                self._synchronize_model_cuda(base_model)
                generation_start = time.perf_counter()
                with torch.no_grad():
                    outputs = base_model.generate(
                        prefix_ids,
                        attention_mask=prefix_mask,
                        max_new_tokens=n_private,
                        min_new_tokens=n_private,
                        num_return_sequences=k,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=1.0,
                        pad_token_id=self.pad_token_id,
                    )
                self._synchronize_model_cuda(base_model)
                generation_seconds = time.perf_counter() - generation_start
                self.generation_timing["generation_seconds"] += generation_seconds
                self.generation_timing["generation_calls"] += 1
                self.generation_timing["generated_sequences"] += int(k)
                self.generation_timing["generated_tokens"] += int(k * n_private)

                final_dummies = []
                for i in range(k):
                    gen_seq = outputs[i].detach().cpu()
                    generated_suffix = gen_seq[prefix_len:prefix_len + n_private]
                    if generated_suffix.numel() < n_private:
                        pad_count = n_private - generated_suffix.numel()
                        fallback = torch.full((pad_count,), self.pad_token_id, dtype=torch.long)
                        generated_suffix = torch.cat([generated_suffix, fallback], dim=0)

                    full_seq = input_ids.clone()
                    full_seq[:prefix_len] = projected_prefix_ids
                    full_seq[prefix_len:real_len] = generated_suffix[:n_private]
                    final_dummies.append(full_seq)

                self.generation_cache.append(
                    {
                        "input_ids": input_ids.cpu(),
                        "attention_mask": attn_mask.cpu(),
                        "real_len": real_len,
                        "n_private": n_private,
                        "prefix_len": prefix_len,
                        "noisy_prefix_embeddings": noisy_prefix_emb.cpu(),
                        "projected_prefix_ids": projected_prefix_ids.cpu(),
                        "dummy_input_ids": final_dummies,
                    }
                )
            print("[DP-SPD] Continuation dummy generation finished.\n")

        n_items_per_text = 1 + n_requested_dummies
        n_texts = len(texts)
        self.mapping = []
        rng = np.random.default_rng(self.seed)

        # mapping entries are (source_text_index, is_real, dummy_index). Shuffling
        # hides the real candidate position from downstream attack/eval code.
        if shuffle_mode == "block":
            group_indices = np.arange(n_texts)
            rng.shuffle(group_indices)
            for g_idx in group_indices:
                items = np.arange(n_items_per_text)
                rng.shuffle(items)
                for item_code in items:
                    is_real = (item_code == 0)
                    dummy_idx = 0 if is_real else (item_code - 1)
                    self.mapping.append((g_idx, is_real, dummy_idx))
        else:
            temp_mapping = []
            for g_idx in range(n_texts):
                for item_code in range(n_items_per_text):
                    is_real = (item_code == 0)
                    dummy_idx = 0 if is_real else (item_code - 1)
                    temp_mapping.append((g_idx, is_real, dummy_idx))
            perm = rng.permutation(len(temp_mapping))
            self.mapping = [temp_mapping[i] for i in perm]
            self.perm = [item[0] for item in self.mapping]

    def _encode_text(self, text_to_encode: Any):
        if isinstance(text_to_encode, (tuple, list)) and len(text_to_encode) == 2:
            return self.tokenizer(
                str(text_to_encode[0]),
                str(text_to_encode[1]),
                truncation=True,
                padding='max_length',
                max_length=self.batch_cfg.max_length,
                return_tensors='pt',
            )
        return self.tokenizer(
            str(text_to_encode),
            truncation=True,
            padding='max_length',
            max_length=self.batch_cfg.max_length,
            return_tensors='pt',
        )

    def _resolve_pad_token_id(self) -> int:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return int(eos_id)
        return 0

    @staticmethod
    def _module_device(module: torch.nn.Module) -> torch.device:
        return next(module.parameters()).device

    @staticmethod
    def _synchronize_model_cuda(module: torch.nn.Module) -> None:
        """Synchronize only CUDA devices used by ``module``.

        ``device_map=auto`` proxy generators may span several GPUs.  A plain
        ``torch.cuda.synchronize()`` only waits on the current default device,
        which can under-report decoding time for a sharded model.
        """
        if not torch.cuda.is_available():
            return
        devices = set()
        device_map = getattr(module, "hf_device_map", None)
        if isinstance(device_map, dict):
            for placement in device_map.values():
                if isinstance(placement, int):
                    devices.add(f"cuda:{placement}")
                elif isinstance(placement, str) and placement.startswith("cuda"):
                    devices.add(placement)
        if not devices:
            try:
                device = PrivacyDataset._module_device(module)
                if device.type == "cuda":
                    devices.add(str(device))
            except (StopIteration, AttributeError):
                pass
        for device_name in sorted(devices):
            torch.cuda.synchronize(torch.device(device_name))

    def _compute_private_layout(self, attn_mask: Tensor) -> Tuple[int, int, int]:
        # DP-SPD treats the prefix as lower-sensitivity context and the suffix
        # as private content. Padding tokens are excluded from this split.
        real_len = int(attn_mask.sum().item())
        if real_len <= 1 or self.sensitive_ratio <= 0:
            return real_len, 0, real_len
        prefix_len = math.floor((1.0 - self.sensitive_ratio) * real_len)
        prefix_len = max(1, min(real_len - 1, prefix_len))
        n_private = real_len - prefix_len
        return real_len, n_private, prefix_len

    def _clip_embeddings(self, embeddings: Tensor) -> Tensor:
        if self.ldp_cfg.clip_norm is None or embeddings.numel() == 0:
            return embeddings
        current_norms = embeddings.norm(p=2, dim=-1, keepdim=True)
        clip_coef = torch.clamp(self.ldp_cfg.clip_norm / (current_norms + 1e-8), max=1.0)
        return embeddings * clip_coef

    def _noise_scale(self, epsilon: float) -> float:
        sens = 2.0 * (self.ldp_cfg.clip_norm if self.ldp_cfg.clip_norm else 1.0)
        return (sens / max(epsilon, 1e-8)) / math.sqrt(self.batch_cfg.hidden_size)

    def _sample_noise(self, shape: torch.Size, scale: float, seed: int, dtype: torch.dtype) -> Tensor:
        rng = np.random.default_rng(seed)
        noise_np = rng.laplace(loc=0.0, scale=scale, size=tuple(shape))
        return torch.from_numpy(noise_np).to(device=self.device, dtype=dtype)

    def _privatize_prefix_embeddings(self, prefix_embeddings: Tensor, group_index: int) -> Tensor:
        if prefix_embeddings.numel() == 0:
            return prefix_embeddings.clone()
        clipped_prefix = self._clip_embeddings(prefix_embeddings.clone())
        scale = self._noise_scale(self.ldp_cfg.epsilon_non_sensitive)
        noise = self._sample_noise(
            clipped_prefix.shape,
            scale=scale,
            seed=self.seed + group_index + 20000,
            dtype=clipped_prefix.dtype,
        )
        return clipped_prefix + noise

    def _project_embeddings_to_tokens(
        self,
        embeddings: Tensor,
        original_ids: Optional[Tensor] = None,
        chunk_size: int = 8,
    ) -> Tensor:
        if embeddings.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        # Nearest-neighbor projection is used only when a text-only surface is
        # required, for example API-only black-box experiments and dummy suffix
        # generation. Direct embedding experiments skip this step.
        weight = self.emb_layer.weight.detach().float().cpu()
        emb = embeddings.detach().float().cpu()
        weight_norm_sq = (weight * weight).sum(dim=1)
        projected_chunks = []
        for start in range(0, emb.size(0), chunk_size):
            chunk = emb[start:start + chunk_size]
            chunk_norm_sq = (chunk * chunk).sum(dim=1, keepdim=True)
            distances = chunk_norm_sq + weight_norm_sq.unsqueeze(0) - 2.0 * chunk @ weight.t()
            blocked_ids = [tok_id for tok_id in self.special_token_ids if 0 <= int(tok_id) < weight.size(0)]
            if blocked_ids:
                distances[:, blocked_ids] = float("inf")
            projected_chunks.append(torch.argmin(distances, dim=1).to(torch.long))
        projected = torch.cat(projected_chunks, dim=0)

        # Preserve special tokens exactly. Replacing BOS/EOS/PAD with nearest
        # neighbors can corrupt chat templates or padded batches.
        if original_ids is not None and self.special_token_ids:
            original_cpu = original_ids.detach().cpu().to(torch.long)
            special_mask = torch.tensor(
                [int(tok.item()) in self.special_token_ids for tok in original_cpu],
                dtype=torch.bool,
            )
            projected[special_mask] = original_cpu[special_mask]
        return projected

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        g_idx, is_real, dummy_idx = self.mapping[idx]

        orig_text = self.texts[g_idx]
        if not is_real and self.dummy_cfg.dummy_mode == "sentence":
            rng_sentence = np.random.default_rng(self.seed + g_idx + 3000 + dummy_idx)
            text_to_encode = self.dummy_cfg.dummy_corpus[rng_sentence.integers(0, len(self.dummy_cfg.dummy_corpus))]
        else:
            text_to_encode = orig_text

        enc = self._encode_text(text_to_encode)
        input_ids = enc['input_ids'][0].to(self.device)
        attn_mask = enc['attention_mask'][0].to(self.device)
        token_type_ids = enc.get('token_type_ids', torch.zeros_like(input_ids))[0].to(self.device) if 'token_type_ids' in enc else torch.zeros_like(input_ids)

        real_len, n_private, prefix_len = self._compute_private_layout(attn_mask)

        # Generated dummy mode caches the real input, generated dummy suffixes,
        # and the shared noised prefix so every group is internally consistent.
        if self.dummy_cfg.dummy_mode == "generation" and self.generation_cache:
            cache = self.generation_cache[g_idx]
            attn_mask = cache["attention_mask"].clone().to(self.device)
            real_len = int(cache["real_len"])
            n_private = int(cache["n_private"])
            prefix_len = int(cache["prefix_len"])
            if is_real:
                input_ids = cache["input_ids"].clone().to(self.device)
            else:
                input_ids = cache["dummy_input_ids"][dummy_idx].clone().to(self.device)

        with torch.no_grad():
            orig_emb = self.emb_layer(input_ids)
        target_emb = orig_emb.clone()

        private_mask = torch.zeros_like(attn_mask, dtype=torch.bool)
        if n_private > 0:
            if self.dummy_cfg.dummy_mode in ("generation", "suffix"):
                # For suffix-style DP-SPD the private region is exactly the tail
                # after the shared prefix.
                private_mask[prefix_len:real_len] = True
            else:
                valid_indices = []
                if self.dummy_cfg.dummy_mode == "token":
                    for i in range(real_len):
                        if input_ids[i].item() in self.id_to_pos:
                            valid_indices.append(i)
                else:
                    valid_indices = list(range(real_len))

                valid_indices = np.array(valid_indices)
                if len(valid_indices) > 0:
                    rng = np.random.default_rng(self.seed + g_idx)
                    chosen_indices = rng.choice(valid_indices, size=min(n_private, len(valid_indices)), replace=False)
                    private_mask[np.sort(chosen_indices)] = True

        if self.dummy_cfg.dummy_mode == "generation" and self.generation_cache:
            target_emb = self._clip_embeddings(target_emb)
            # Reuse the cached noised prefix for all candidates in the group;
            # only suffix content differs between the real item and dummies.
            target_emb[:prefix_len] = self.generation_cache[g_idx]["noisy_prefix_embeddings"].to(
                device=self.device,
                dtype=target_emb.dtype,
            )
        elif self.ldp_cfg.clip_norm is not None:
            target_emb = self._clip_embeddings(target_emb)

        if not (self.dummy_cfg.dummy_mode == "generation" and self.generation_cache):
            # Non-private/context tokens use the public epsilon. Private tokens
            # are masked out here and handled by the private-noise block below.
            scale_non = self._noise_scale(self.ldp_cfg.epsilon_non_sensitive)
            context_noise = self._sample_noise(
                target_emb.shape,
                scale=scale_non,
                seed=self.seed + g_idx + 20000,
                dtype=target_emb.dtype,
            )
            context_mask_float = (~private_mask).unsqueeze(-1).to(target_emb.dtype)
            target_emb = target_emb + context_noise * context_mask_float

        scale_priv = self._noise_scale(self.ldp_cfg.epsilon_private)

        item_code = 0 if is_real else (dummy_idx + 1)
        torch.manual_seed(self.seed + g_idx + 50000 + item_code)

        # Private noise uses a per-candidate seed so dummies do not share the
        # same suffix perturbation as the real query.
        if self.ldp_cfg.mechanism.lower().startswith("lap"):
            lap = torch.distributions.Laplace(loc=0.0, scale=scale_priv)
            priv_noise = lap.sample(target_emb.shape).to(self.device)
        else:
            priv_noise = torch.randn_like(target_emb) * scale_priv

        priv_mask_float = private_mask.unsqueeze(-1).to(target_emb.dtype)
        target_emb = target_emb + priv_noise * priv_mask_float

        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'token_type_ids': token_type_ids,
            'noisy_embeddings': target_emb,
            'original_index': idx,
            'group_index': g_idx,
            'dummy_index': dummy_idx,
            'prefix_len': prefix_len,
            'private_token_count': n_private,
            'is_real': is_real
        }

class BatchShuffleMapper:
    """Mix real requests and dummy requests while keeping a reversible map.

    The class is intentionally independent from ``PrivacyDataset`` so future
    client/server integrations can reuse it for text requests, embedding
    requests, or already-built candidate objects.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.seed = 42 if seed is None else int(seed)

    @staticmethod
    def _normalise_dummy_groups(
        originals: Sequence[Any],
        dummy_groups: Optional[Sequence[Sequence[Any]]],
    ) -> List[List[Any]]:
        if dummy_groups is None:
            return [[] for _ in originals]
        if len(dummy_groups) != len(originals):
            raise ValueError(
                "dummy_groups must have the same length as originals: "
                f"{len(dummy_groups)} != {len(originals)}"
            )
        return [list(group) for group in dummy_groups]

    def shuffle(
        self,
        originals: Sequence[Any],
        dummy_groups: Optional[Sequence[Sequence[Any]]] = None,
        mode: str = "global",
    ) -> Tuple[List[Any], Dict[int, Dict[str, Any]], List[int]]:
        """Return ``mixed_items``, ``index_map``, and real-item positions.

        ``index_map[mixed_index]`` records the source ``original_index``,
        whether the item is real, and the dummy index when applicable. ``mode``
        supports ``global`` shuffling, ``block`` shuffling by original group, or
        ``none``/``sequential`` for deterministic debugging.
        """
        groups = self._normalise_dummy_groups(originals, dummy_groups)
        grouped_entries: List[List[Dict[str, Any]]] = []
        for original_index, original in enumerate(originals):
            entries = [
                {
                    "item": original,
                    "original_index": original_index,
                    "is_real": True,
                    "dummy_index": None,
                    "position_within_group": 0,
                }
            ]
            for dummy_index, dummy in enumerate(groups[original_index]):
                entries.append(
                    {
                        "item": dummy,
                        "original_index": original_index,
                        "is_real": False,
                        "dummy_index": dummy_index,
                        "position_within_group": dummy_index + 1,
                    }
                )
            grouped_entries.append(entries)

        rng = np.random.default_rng(self.seed)
        mode_key = str(mode).lower()
        if mode_key in {"none", "sequential"}:
            ordered_entries = [entry for group in grouped_entries for entry in group]
        elif mode_key == "block":
            group_order = list(range(len(grouped_entries)))
            rng.shuffle(group_order)
            ordered_entries = []
            for group_index in group_order:
                candidate_order = list(range(len(grouped_entries[group_index])))
                rng.shuffle(candidate_order)
                ordered_entries.extend(grouped_entries[group_index][i] for i in candidate_order)
        elif mode_key == "global":
            flattened = [entry for group in grouped_entries for entry in group]
            order = list(range(len(flattened)))
            rng.shuffle(order)
            ordered_entries = [flattened[i] for i in order]
        else:
            raise ValueError("mode must be one of: global, block, none, sequential")

        mixed_items: List[Any] = []
        index_map: Dict[int, Dict[str, Any]] = {}
        real_positions: List[Optional[int]] = [None for _ in originals]
        for mixed_index, entry in enumerate(ordered_entries):
            mixed_items.append(entry["item"])
            metadata = {key: value for key, value in entry.items() if key != "item"}
            index_map[mixed_index] = metadata
            if metadata["is_real"]:
                real_positions[int(metadata["original_index"])] = mixed_index

        if any(position is None for position in real_positions):
            raise RuntimeError("internal shuffle error: at least one real item was dropped")
        return mixed_items, index_map, [int(position) for position in real_positions]


class ClientResponseSelector:
    """Recover real-query responses from a mixed real+dummy response batch."""

    @staticmethod
    def _metadata(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            return {
                "original_index": int(value[0]),
                "is_real": bool(value[1]),
                "dummy_index": value[2] if len(value) > 2 else None,
            }
        raise TypeError(f"Unsupported index_map entry: {value!r}")

    @staticmethod
    def select_responses(
        mixed_responses: Sequence[Any],
        index_map: Dict[int, Any],
    ) -> List[Any]:
        """Select real responses and restore their original request order."""
        if len(mixed_responses) != len(index_map):
            raise ValueError(
                "mixed_responses and index_map must have the same length: "
                f"{len(mixed_responses)} != {len(index_map)}"
            )

        selected: Dict[int, Any] = {}
        for mixed_index, response in enumerate(mixed_responses):
            raw_meta = index_map.get(mixed_index, index_map.get(str(mixed_index)))
            if raw_meta is None:
                raise KeyError(f"index_map is missing mixed index {mixed_index}")
            meta = ClientResponseSelector._metadata(raw_meta)
            if not bool(meta.get("is_real", False)):
                continue
            original_index = int(meta["original_index"])
            if original_index in selected:
                raise ValueError(f"Duplicate real response for original_index={original_index}")
            selected[original_index] = response

        if not selected:
            return []
        expected = set(range(max(selected) + 1))
        missing = sorted(expected.difference(selected))
        if missing:
            raise ValueError(f"Missing real responses for original indices: {missing}")
        return [selected[i] for i in sorted(selected)]
