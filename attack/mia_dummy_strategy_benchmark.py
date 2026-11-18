# -*- coding: utf-8 -*-
import argparse
import csv
import fcntl
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from attack.mia_mink import run_min_k_prob_mia
from dpspd.util.utils import load_local_causal_lm, str2bool


LOCAL_EPS_MAPPING = {
    1.5: {9: 1.50, 19: 2.08, 29: 2.46},
    2.0: {9: 3.03, 19: 3.71, 29: 4.12},
    2.5: {9: 4.48, 19: 5.17, 29: 5.57},
    3.0: {9: 5.77, 19: 6.46, 29: 6.86},
}

STRATEGY_LABELS = {
    "dp_spd_generation": "DP-SPD Prefix-Conditioned Dummy Suffix",
    "random_sampled_suffix": "Random-Sampled Suffix",
    "random_token_suffix": "Random-Token Suffix",
}


def parse_float_csv(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_int_csv(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_csv(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def fmt_float(value: float) -> str:
    return str(value).replace(".", "p")


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MIA ASR and perplexity for three dummy strategies on IMDb "
            "using a grouped (1 real + k dummy) DP-SPD-style evaluation pipeline."
        )
    )
    parser.add_argument(
        "--continuation_model",
        type=str,
        default="models/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--continuation_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--attack_model",
        type=str,
        default="models/Llama-2-7b-chat-hf",
    )
    parser.add_argument(
        "--attack_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--dataset_path", type=str, default="data/imdb")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--min_len", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--num_groups", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--eval_manifest_path", type=str, default=None)
    parser.add_argument("--token_pool_texts", type=int, default=0)

    parser.add_argument("--strategies", type=str, default="dp_spd_generation,random_sampled_suffix,random_token_suffix")
    parser.add_argument("--ratios", type=str, default="0.2,0.5,0.8")
    parser.add_argument("--ks", type=str, default="9,19,29")
    parser.add_argument("--target_eps_list", type=str, default="1.5,2.0,2.5,3.0")
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mia_k_ratio", type=float, default=0.20)
    parser.add_argument("--ppl_batch_size", type=int, default=4)
    parser.add_argument(
        "--reverse_order",
        type=str2bool,
        default=False,
        help="Run strategy/ratio/k/epsilon grids in reverse order for a second parallel worker.",
    )
    parser.add_argument(
        "--reset_outputs",
        type=str2bool,
        default=False,
        help="Overwrite existing CSV outputs before running. Default keeps and skips completed rows.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/mia",
    )
    return parser


def resolve_dataset_split(raw_ds, split_name: str):
    if hasattr(raw_ds, "keys"):
        available = list(raw_ds.keys())
        if split_name not in raw_ds:
            raise ValueError(
                f"Dataset split '{split_name}' was not found in {available}. "
                "Please pass --dataset_split with one of the available splits."
            )
        return raw_ds[split_name]
    return raw_ds


def default_eval_manifest_path(args) -> Path:
    dataset_name = Path(args.dataset_path.rstrip("/")).name or "dataset"
    filename = (
        f"{dataset_name}_{args.dataset_split}_seed{args.seed}"
        f"_n{args.num_groups}_len{args.min_len}_{args.max_len}.json"
    )
    return Path(ROOT) / "fixed_eval_sets" / filename


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def build_eval_manifest(
    dataset_texts: Sequence[str],
    dataset_path: str,
    dataset_split: str,
    min_len: int,
    max_len: int,
    sample_size: int,
    seed: int,
) -> Dict[str, Any]:
    eligible_samples = []
    for idx, text in enumerate(dataset_texts):
        word_count = len(text.split())
        if min_len <= word_count <= max_len:
            eligible_samples.append(
                {
                    "dataset_index": idx,
                    "word_count": word_count,
                    "text": text,
                }
            )

    if len(eligible_samples) < sample_size:
        raise ValueError(
            f"Only {len(eligible_samples)} eligible texts were found, "
            f"but num_groups={sample_size} was requested."
        )

    rng = random.Random(seed)
    selected = rng.sample(eligible_samples, sample_size)
    return {
        "dataset_path": dataset_path,
        "dataset_split": dataset_split,
        "seed": seed,
        "num_groups": sample_size,
        "min_len": min_len,
        "max_len": max_len,
        "eligible_pool_size": len(eligible_samples),
        "samples": selected,
    }


def load_or_create_eval_manifest(args) -> Tuple[Dict[str, Any], Path, List[str]]:
    raw_ds = load_from_disk(args.dataset_path)
    split_ds = resolve_dataset_split(raw_ds, args.dataset_split)
    dataset_texts = split_ds["text"]

    manifest_path = Path(args.eval_manifest_path) if args.eval_manifest_path else default_eval_manifest_path(args)
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = build_eval_manifest(
            dataset_texts=dataset_texts,
            dataset_path=args.dataset_path,
            dataset_split=args.dataset_split,
            min_len=args.min_len,
            max_len=args.max_len,
            sample_size=args.num_groups,
            seed=args.seed,
        )
        atomic_write_json(manifest_path, manifest)

    selected_indices = {item["dataset_index"] for item in manifest["samples"]}
    corpus_texts = [text for idx, text in enumerate(dataset_texts) if idx not in selected_indices]
    return manifest, manifest_path, corpus_texts


def tokenize_text(text: str, tokenizer, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"][0].cpu(), enc["attention_mask"][0].cpu()


def build_corpus_cache(
    corpus_texts: Sequence[str],
    tokenizer,
    max_length: int,
    token_pool_texts: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    rng = random.Random(seed)
    if token_pool_texts > 0 and len(corpus_texts) > token_pool_texts:
        selected_texts = rng.sample(list(corpus_texts), token_pool_texts)
    else:
        selected_texts = list(corpus_texts)

    cached_entries: List[Dict[str, Any]] = []
    token_pool: List[int] = []
    special_ids = set(tokenizer.all_special_ids)

    for text in tqdm(selected_texts, desc="Caching corpus suffix pool"):
        input_ids, attn_mask = tokenize_text(text, tokenizer, max_length)
        valid_len = int(attn_mask.sum().item())
        if valid_len <= 1:
            continue
        valid_ids = input_ids[:valid_len].tolist()
        valid_ids = [tok_id for tok_id in valid_ids if tok_id not in special_ids]
        if not valid_ids:
            continue
        cached_entries.append(
            {
                "input_ids": torch.tensor(valid_ids, dtype=torch.long),
                "valid_len": len(valid_ids),
            }
        )
        token_pool.extend(valid_ids)

    if not cached_entries or not token_pool:
        raise ValueError("Failed to build a usable corpus cache or token pool from the selected dataset.")
    return cached_entries, token_pool


def sample_suffix_from_corpus_entry(
    cached_entry: Dict[str, Any],
    suffix_len: int,
    token_pool: Sequence[int],
    rng: random.Random,
) -> torch.Tensor:
    valid_ids = cached_entry["input_ids"]
    valid_len = int(cached_entry["valid_len"])
    if valid_len >= suffix_len:
        start = valid_len - suffix_len
        return valid_ids[start:start + suffix_len].clone()

    padded = valid_ids.tolist()
    while len(padded) < suffix_len:
        padded.append(token_pool[rng.randrange(len(token_pool))])
    return torch.tensor(padded[:suffix_len], dtype=torch.long)


def compute_private_layout(attn_mask: torch.Tensor, ratio: float) -> Tuple[int, int, int]:
    real_len = int(attn_mask.sum().item())
    if real_len <= 1 or ratio <= 0:
        return real_len, 0, real_len
    prefix_len = math.floor((1.0 - ratio) * real_len)
    prefix_len = max(1, min(real_len - 1, prefix_len))
    n_private = real_len - prefix_len
    return real_len, n_private, prefix_len


def clip_embeddings(embeddings: torch.Tensor, clip_norm: float | None) -> torch.Tensor:
    if clip_norm is None or embeddings.numel() == 0:
        return embeddings
    current_norms = embeddings.norm(p=2, dim=-1, keepdim=True)
    clip_coef = torch.clamp(clip_norm / (current_norms + 1e-8), max=1.0)
    return embeddings * clip_coef


def project_embeddings_to_token_ids(
    embeddings: torch.Tensor,
    emb_layer: torch.nn.Embedding,
    tokenizer,
    original_ids: torch.Tensor | None = None,
    chunk_size: int = 8,
) -> torch.Tensor:
    if embeddings.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    weight = emb_layer.weight.detach().float().cpu()
    emb = embeddings.detach().float().cpu()
    weight_norm_sq = (weight * weight).sum(dim=1)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    blocked_ids = [tok_id for tok_id in special_ids if 0 <= int(tok_id) < weight.size(0)]

    projected_chunks: List[torch.Tensor] = []
    for start in range(0, emb.size(0), chunk_size):
        chunk = emb[start:start + chunk_size]
        chunk_norm_sq = (chunk * chunk).sum(dim=1, keepdim=True)
        distances = chunk_norm_sq + weight_norm_sq.unsqueeze(0) - 2.0 * chunk @ weight.t()
        if blocked_ids:
            distances[:, blocked_ids] = float("inf")
        projected_chunks.append(torch.argmin(distances, dim=1).to(torch.long))
    projected = torch.cat(projected_chunks, dim=0)

    if original_ids is not None and special_ids:
        original_cpu = original_ids.detach().cpu().to(torch.long)
        special_mask = torch.tensor(
            [int(tok.item()) in special_ids for tok in original_cpu],
            dtype=torch.bool,
        )
        projected[special_mask] = original_cpu[special_mask]
    return projected


def privatize_and_project_prefix(
    real_ids: torch.Tensor,
    prefix_len: int,
    emb_layer: torch.nn.Embedding,
    tokenizer,
    hidden_size: int,
    clip_norm: float,
    epsilon_non_sensitive: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        prefix_emb = emb_layer(real_ids[:prefix_len]).detach().cpu()
    clipped_prefix = clip_embeddings(prefix_emb.clone(), clip_norm)
    sens = 2.0 * (clip_norm if clip_norm else 1.0)
    scale = (sens / max(epsilon_non_sensitive, 1e-8)) / math.sqrt(hidden_size)
    rng = np.random.default_rng(seed)
    noise_np = rng.laplace(loc=0.0, scale=scale, size=tuple(clipped_prefix.shape))
    noisy_prefix = clipped_prefix + torch.from_numpy(noise_np).to(dtype=clipped_prefix.dtype)
    projected_prefix_ids = project_embeddings_to_token_ids(
        noisy_prefix,
        emb_layer=emb_layer,
        tokenizer=tokenizer,
        original_ids=real_ids[:prefix_len],
    )
    return projected_prefix_ids, noisy_prefix


def generate_suffix_dummies(
    prefix_text: str,
    suffix_len: int,
    k: int,
    continuation_tokenizer,
    continuation_model,
    attack_tokenizer,
    token_pool: Sequence[int],
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    prefix_enc = continuation_tokenizer(prefix_text, return_tensors="pt")
    prefix_ids = prefix_enc["input_ids"].to(continuation_model.device)
    prefix_mask = prefix_enc["attention_mask"].to(continuation_model.device)

    torch.manual_seed(seed)
    approx_new_tokens = max(suffix_len * 3, suffix_len + 24)
    with torch.no_grad():
        outputs = continuation_model.generate(
            prefix_ids,
            attention_mask=prefix_mask,
            max_new_tokens=approx_new_tokens,
            min_new_tokens=max(suffix_len, 1),
            num_return_sequences=k,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=continuation_tokenizer.pad_token_id,
        )

    suffix_tensors: List[Tuple[torch.Tensor, torch.Tensor]] = []
    prefix_token_count = int(prefix_ids.shape[1])
    for row in outputs:
        generated_suffix_ids = row[prefix_token_count:]
        generated_suffix_text = continuation_tokenizer.decode(
            generated_suffix_ids,
            skip_special_tokens=True,
        )

        attack_suffix_ids = attack_tokenizer(
            generated_suffix_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].cpu()

        suffix_tokens = attack_suffix_ids.tolist()
        if not suffix_tokens:
            suffix_tokens = [token_pool[(seed + len(suffix_tensors)) % len(token_pool)]]
        while len(suffix_tokens) < suffix_len:
            suffix_tokens.append(token_pool[(seed + len(suffix_tensors) + len(suffix_tokens)) % len(token_pool)])

        suffix_tensor = torch.tensor(suffix_tokens[:suffix_len], dtype=torch.long)
        suffix_mask = torch.ones_like(suffix_tensor)
        suffix_tensors.append((suffix_tensor, suffix_mask))

    return suffix_tensors


def build_strategy_group_items(
    strategy: str,
    real_texts: Sequence[str],
    corpus_cache: Sequence[Dict[str, Any]],
    token_pool: Sequence[int],
    continuation_tokenizer,
    continuation_model,
    attack_tokenizer,
    max_length: int,
    ratio: float,
    k: int,
    seed: int,
    projection_embeddings: torch.nn.Module | None = None,
    hidden_size: int | None = None,
    clip_norm: float | None = None,
    epsilon_non_sensitive: float | None = None,
    projection_seed: int | None = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed + int(ratio * 1000) + k * 100)
    items: List[Dict[str, Any]] = []
    projection_emb_layer = None
    if projection_embeddings is not None:
        cloned_weight = projection_embeddings.weight.detach().cpu().clone()
        projection_emb_layer = torch.nn.Embedding.from_pretrained(cloned_weight)

    print(f"\n[Build Groups] strategy={strategy} | ratio={ratio} | k={k}")
    for group_id, text in enumerate(tqdm(real_texts, desc=f"Preparing {strategy} groups"), start=0):
        real_ids, real_mask = tokenize_text(text, attack_tokenizer, max_length)
        real_len, n_private, prefix_len = compute_private_layout(real_mask, ratio)

        group_entries = [
            {
                "input_ids": real_ids.clone(),
                "attention_mask": real_mask.clone(),
                "is_real": True,
                "item_code": 0,
                "group_id": group_id,
                "real_len": real_len,
                "prefix_len": prefix_len,
            }
        ]

        if strategy == "dp_spd_generation":
            projected_prefix_ids = None
            noisy_prefix_embeddings = None
            if (
                projection_emb_layer is not None
                and hidden_size is not None
                and clip_norm is not None
                and epsilon_non_sensitive is not None
            ):
                projected_prefix_ids, noisy_prefix_embeddings = privatize_and_project_prefix(
                    real_ids=real_ids,
                    prefix_len=prefix_len,
                    emb_layer=projection_emb_layer,
                    tokenizer=attack_tokenizer,
                    hidden_size=hidden_size,
                    clip_norm=clip_norm,
                    epsilon_non_sensitive=epsilon_non_sensitive,
                    seed=(projection_seed if projection_seed is not None else seed) + group_id + 20000,
                )
                group_entries[0]["noisy_prefix_embeddings"] = noisy_prefix_embeddings
                prefix_text = attack_tokenizer.decode(projected_prefix_ids, skip_special_tokens=True)
            else:
                prefix_text = attack_tokenizer.decode(real_ids[:prefix_len], skip_special_tokens=True)

            dummy_sequences = generate_suffix_dummies(
                prefix_text=prefix_text,
                suffix_len=n_private,
                k=k,
                continuation_tokenizer=continuation_tokenizer,
                continuation_model=continuation_model,
                attack_tokenizer=attack_tokenizer,
                token_pool=token_pool,
                seed=seed + group_id + int(ratio * 1000) + k * 100,
            )
            for dummy_idx, (dummy_suffix_ids, _) in enumerate(dummy_sequences, start=1):
                dummy_ids = real_ids.clone()
                if projected_prefix_ids is not None:
                    dummy_ids[:prefix_len] = projected_prefix_ids
                dummy_ids[prefix_len:real_len] = dummy_suffix_ids[:n_private]
                entry = {
                    "input_ids": dummy_ids,
                    "attention_mask": real_mask.clone(),
                    "is_real": False,
                    "item_code": dummy_idx,
                    "group_id": group_id,
                    "real_len": real_len,
                    "prefix_len": prefix_len,
                }
                if noisy_prefix_embeddings is not None:
                    entry["noisy_prefix_embeddings"] = noisy_prefix_embeddings
                group_entries.append(entry)
        elif strategy == "random_sampled_suffix":
            for dummy_idx in range(1, k + 1):
                corpus_entry = corpus_cache[rng.randrange(len(corpus_cache))]
                suffix_ids = sample_suffix_from_corpus_entry(corpus_entry, n_private, token_pool, rng)
                dummy_ids = real_ids.clone()
                dummy_ids[prefix_len:real_len] = suffix_ids
                group_entries.append(
                    {
                        "input_ids": dummy_ids,
                        "attention_mask": real_mask.clone(),
                        "is_real": False,
                        "item_code": dummy_idx,
                        "group_id": group_id,
                        "real_len": real_len,
                        "prefix_len": prefix_len,
                    }
                )
        elif strategy == "random_token_suffix":
            for dummy_idx in range(1, k + 1):
                suffix_ids = torch.tensor(
                    [token_pool[rng.randrange(len(token_pool))] for _ in range(n_private)],
                    dtype=torch.long,
                )
                dummy_ids = real_ids.clone()
                dummy_ids[prefix_len:real_len] = suffix_ids
                group_entries.append(
                    {
                        "input_ids": dummy_ids,
                        "attention_mask": real_mask.clone(),
                        "is_real": False,
                        "item_code": dummy_idx,
                        "group_id": group_id,
                        "real_len": real_len,
                        "prefix_len": prefix_len,
                    }
                )
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        shuffle_rng = np.random.default_rng(seed + group_id + 70000)
        order = np.arange(len(group_entries))
        shuffle_rng.shuffle(order)
        for idx in order.tolist():
            items.append(group_entries[idx])

    return items


class GroupedMIASequenceDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Dict[str, Any]],
        pretrained_embeddings: torch.nn.Module,
        hidden_size: int,
        clip_norm: float,
        mechanism: str = "Laplace",
        epsilon_private: float = 1.0,
        epsilon_non_sensitive: float = 1.0,
        seed: int = 42,
    ):
        self.items = list(items)
        self.hidden_size = hidden_size
        self.clip_norm = clip_norm
        self.mechanism = mechanism
        self.epsilon_private = epsilon_private
        self.epsilon_non_sensitive = epsilon_non_sensitive
        self.seed = seed

        cloned_weight = pretrained_embeddings.weight.detach().cpu().clone()
        self.emb_layer = torch.nn.Embedding.from_pretrained(cloned_weight)

    def set_noise_config(self, epsilon_non_sensitive: float, epsilon_private: float, seed: int) -> None:
        self.epsilon_non_sensitive = epsilon_non_sensitive
        self.epsilon_private = epsilon_private
        self.seed = seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        input_ids = item["input_ids"].clone()
        attn_mask = item["attention_mask"].clone()

        with torch.no_grad():
            orig_emb = self.emb_layer(input_ids)
        target_emb = orig_emb.clone()

        real_len = int(item["real_len"])
        prefix_len = int(item["prefix_len"])
        private_mask = torch.zeros_like(attn_mask, dtype=torch.bool)
        private_mask[prefix_len:real_len] = True

        target_emb = clip_embeddings(target_emb, self.clip_norm)

        sens = 2.0 * (self.clip_norm if self.clip_norm else 1.0)
        if "noisy_prefix_embeddings" in item:
            target_emb[:prefix_len] = item["noisy_prefix_embeddings"].to(dtype=target_emb.dtype)
        else:
            raw_scale_non = sens / max(self.epsilon_non_sensitive, 1e-8)
            scale_non = raw_scale_non / math.sqrt(self.hidden_size)

            rng_noise = np.random.default_rng(self.seed + int(item["group_id"]) + 20000)
            noise_np = rng_noise.laplace(loc=0.0, scale=scale_non, size=target_emb.shape)
            context_noise = torch.from_numpy(noise_np).to(dtype=target_emb.dtype)
            context_mask_float = (~private_mask).unsqueeze(-1).to(target_emb.dtype)
            target_emb = target_emb + context_noise * context_mask_float

        raw_scale_priv = sens / max(self.epsilon_private, 1e-8)
        scale_priv = raw_scale_priv / math.sqrt(self.hidden_size)
        torch.manual_seed(self.seed + int(item["group_id"]) + 50000 + int(item["item_code"]))
        if self.mechanism.lower().startswith("lap"):
            lap = torch.distributions.Laplace(loc=0.0, scale=scale_priv)
            priv_noise = lap.sample(target_emb.shape)
        else:
            priv_noise = torch.randn_like(target_emb) * scale_priv

        priv_mask_float = private_mask.unsqueeze(-1).to(target_emb.dtype)
        target_emb = target_emb + priv_noise * priv_mask_float

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "noisy_embeddings": target_emb,
            "is_real": item["is_real"],
            "group_id": item["group_id"],
            "item_code": item["item_code"],
        }


class CleanSequenceDataset(Dataset):
    def __init__(self, items: Sequence[Dict[str, Any]]):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "input_ids": item["input_ids"].clone(),
            "attention_mask": item["attention_mask"].clone(),
            "is_real": item["is_real"],
        }


def compute_average_ppl(
    items: Sequence[Dict[str, Any]],
    model,
    device: str,
    batch_size: int,
) -> Tuple[float, float, float]:
    dataset = CleanSequenceDataset(items)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    real_ppls: List[float] = []
    dummy_ppls: List[float] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing clean-sequence PPL"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            is_real = batch["is_real"]

            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attn_mask[..., 1:].contiguous()

            token_losses = loss_fct(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(input_ids.size(0), -1)
            denom = shift_mask.sum(dim=1).clamp(min=1)
            seq_nll = (token_losses * shift_mask).sum(dim=1) / denom
            seq_ppl = torch.exp(torch.clamp(seq_nll, max=50.0)).detach().cpu().tolist()

            for flag, ppl in zip(is_real.tolist(), seq_ppl):
                if flag:
                    real_ppls.append(float(ppl))
                else:
                    dummy_ppls.append(float(ppl))

    real_ppl = sum(real_ppls) / max(len(real_ppls), 1)
    dummy_ppl = sum(dummy_ppls) / max(len(dummy_ppls), 1)
    ppl_gap = abs(dummy_ppl - real_ppl)
    return real_ppl, dummy_ppl, ppl_gap


def build_output_paths(output_dir: Path) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_eps_path = output_dir / "mia_dummy_strategy_per_eps.csv"
    avg_path = output_dir / "mia_dummy_strategy_avg_over_eps.csv"
    config_path = output_dir / "mia_dummy_strategy_benchmark_config.json"
    return per_eps_path, avg_path, config_path


def eps_key(value: Any) -> str:
    return f"{float(value):.12g}"


def ratio_label(ratio: float) -> str:
    return f"{ratio * 100:.0f}%"


def per_eps_key(strategy_label: str, ratio_text: str, k: int, target_eps: float) -> Tuple[str, str, int, str]:
    return (strategy_label, ratio_text, int(k), eps_key(target_eps))


def row_per_eps_key(row: Dict[str, Any]) -> Tuple[str, str, int, str]:
    return (
        row["Strategy"],
        row["Dummy Ratio"],
        int(row["Dummy (k)"]),
        eps_key(row["Target Eps"]),
    )


def avg_key(strategy_label: str, ratio_text: str, k: int) -> Tuple[str, str, int]:
    return (strategy_label, ratio_text, int(k))


def row_avg_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return (row["Strategy"], row["Dummy Ratio"], int(row["Dummy (k)"]))


def lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def write_csv_header(path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def ensure_csv_header_unlocked(path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        write_csv_header(path, fieldnames)
        return
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header != fieldnames:
        raise ValueError(f"CSV header mismatch in {path}: {header} != {fieldnames}")


def ensure_csv_header(path: Path, fieldnames: List[str]) -> None:
    with open(lock_path(path), "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        ensure_csv_header_unlocked(path, fieldnames)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def read_csv_rows_unlocked(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_csv_rows(path: Path, fieldnames: List[str]) -> List[Dict[str, str]]:
    with open(lock_path(path), "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_SH)
        ensure_csv_header_unlocked(path, fieldnames)
        rows = read_csv_rows_unlocked(path)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
    return rows


def append_row_if_missing(
    path: Path,
    fieldnames: List[str],
    row: Dict[str, Any],
    key_fn,
    expected_key,
) -> bool:
    with open(lock_path(path), "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        ensure_csv_header_unlocked(path, fieldnames)
        existing_keys = {key_fn(existing_row) for existing_row in read_csv_rows_unlocked(path)}
        if expected_key in existing_keys:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            return False
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
    return True


def combo_per_eps_rows(
    path: Path,
    fieldnames: List[str],
    strategy_label: str,
    ratio_text: str,
    k: int,
) -> List[Dict[str, str]]:
    return [
        row
        for row in read_csv_rows(path, fieldnames)
        if row["Strategy"] == strategy_label
        and row["Dummy Ratio"] == ratio_text
        and int(row["Dummy (k)"]) == int(k)
    ]


def build_avg_row_from_per_eps(
    rows: List[Dict[str, str]],
    target_eps_list: Sequence[float],
    manifest_path: Path,
) -> Dict[str, Any] | None:
    rows_by_eps = {eps_key(row["Target Eps"]): row for row in rows}
    selected_rows = []
    for target_eps in target_eps_list:
        row = rows_by_eps.get(eps_key(target_eps))
        if row is None:
            return None
        selected_rows.append(row)
    first = selected_rows[0]
    avg_asr = sum(float(row["MIA ASR (%)"]) for row in selected_rows) / len(selected_rows)
    avg_real_ppl = sum(float(row["Real PPL"]) for row in selected_rows) / len(selected_rows)
    avg_dummy_ppl = sum(float(row["Dummy PPL"]) for row in selected_rows) / len(selected_rows)
    avg_ppl_gap = sum(float(row["PPL Gap"]) for row in selected_rows) / len(selected_rows)
    return {
        "Strategy": first["Strategy"],
        "Dummy Ratio": first["Dummy Ratio"],
        "Dummy (k)": int(first["Dummy (k)"]),
        "Average MIA ASR (%)": f"{avg_asr:.2f}",
        "Average Random Baseline (%)": first["Random Baseline (%)"],
        "Real PPL": f"{avg_real_ppl:.4f}",
        "Dummy PPL": f"{avg_dummy_ppl:.4f}",
        "PPL Gap": f"{avg_ppl_gap:.4f}",
        "Averaged Target Eps": ",".join(str(x) for x in target_eps_list),
        "Eval Manifest": str(manifest_path),
    }


def main():
    args = get_parser().parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"

    strategies = parse_str_csv(args.strategies)
    ratios = parse_float_csv(args.ratios)
    ks = parse_int_csv(args.ks)
    target_eps_list = parse_float_csv(args.target_eps_list)
    run_strategies = list(reversed(strategies)) if args.reverse_order else list(strategies)
    run_ratios = list(reversed(ratios)) if args.reverse_order else list(ratios)
    run_ks = list(reversed(ks)) if args.reverse_order else list(ks)
    run_target_eps_list = list(reversed(target_eps_list)) if args.reverse_order else list(target_eps_list)

    for strategy in strategies:
        if strategy not in STRATEGY_LABELS:
            raise ValueError(f"Unsupported strategy '{strategy}'. Supported: {sorted(STRATEGY_LABELS)}")
    for target_eps in target_eps_list:
        if target_eps not in LOCAL_EPS_MAPPING:
            raise ValueError(f"Unsupported target_eps={target_eps}. Supported: {sorted(LOCAL_EPS_MAPPING)}")
    for k in ks:
        for target_eps in target_eps_list:
            if k not in LOCAL_EPS_MAPPING[target_eps]:
                raise ValueError(
                    f"Missing local-eps mapping for target_eps={target_eps}, k={k}. "
                    f"Available k values: {sorted(LOCAL_EPS_MAPPING[target_eps])}"
                )

    output_dir = Path(args.output_dir)
    per_eps_path, avg_path, config_path = build_output_paths(output_dir)

    print("Loading continuation model (Qwen-side)...")
    continuation_tokenizer, continuation_model, continuation_dtype = load_local_causal_lm(
        model_path=args.continuation_model,
        device=args.device,
        dtype_preference=args.continuation_dtype,
        local_files_only=args.local_files_only,
    )
    print(f"[Continuation Model] {os.path.basename(args.continuation_model.rstrip('/'))} | dtype={continuation_dtype}")

    print("Loading attack/scoring model (server-side)...")
    attack_tokenizer, attack_model, attack_dtype = load_local_causal_lm(
        model_path=args.attack_model,
        device=args.device,
        dtype_preference=args.attack_dtype,
        local_files_only=args.local_files_only,
    )
    print(f"[Attack Model] {os.path.basename(args.attack_model.rstrip('/'))} | dtype={attack_dtype}")

    manifest, manifest_path, corpus_texts = load_or_create_eval_manifest(args)
    real_texts = [item["text"] for item in manifest["samples"]]
    print(
        f"[Data Ready] real_groups={len(real_texts)} | "
        f"split={args.dataset_split} | eval_manifest={manifest_path}"
    )

    print("Building reusable corpus cache for suffix sampling and token sampling...")
    corpus_cache, token_pool = build_corpus_cache(
        corpus_texts=corpus_texts,
        tokenizer=attack_tokenizer,
        max_length=args.max_length,
        token_pool_texts=args.token_pool_texts,
        seed=args.seed,
    )
    print(f"[Corpus Cache] cached_texts={len(corpus_cache)} | token_pool_size={len(token_pool)}")

    atomic_write_json(
        config_path,
        {
            "args": vars(args),
            "strategy_labels": STRATEGY_LABELS,
            "eval_manifest_path": str(manifest_path),
            "continuation_dtype": str(continuation_dtype),
            "attack_dtype": str(attack_dtype),
        },
    )

    per_eps_fields = [
        "Strategy",
        "Dummy Ratio",
        "Dummy (k)",
        "Target Eps",
        "Required Local Eps",
        "MIA ASR (%)",
        "Random Baseline (%)",
        "Real PPL",
        "Dummy PPL",
        "PPL Gap",
        "Eval Manifest",
    ]
    avg_fields = [
        "Strategy",
        "Dummy Ratio",
        "Dummy (k)",
        "Average MIA ASR (%)",
        "Average Random Baseline (%)",
        "Real PPL",
        "Dummy PPL",
        "PPL Gap",
        "Averaged Target Eps",
        "Eval Manifest",
    ]

    if args.reset_outputs:
        write_csv_header(per_eps_path, per_eps_fields)
        write_csv_header(avg_path, avg_fields)
    else:
        ensure_csv_header(per_eps_path, per_eps_fields)
        ensure_csv_header(avg_path, avg_fields)

    print("\n" + "=" * 80)
    print("Starting MIA + PPL dummy-strategy benchmark...")
    print("=" * 80)

    for strategy in run_strategies:
        for ratio in run_ratios:
            for k in run_ks:
                strategy_label = STRATEGY_LABELS[strategy]
                ratio_text = ratio_label(ratio)
                existing_per_eps_keys = {
                    row_per_eps_key(row)
                    for row in read_csv_rows(per_eps_path, per_eps_fields)
                }
                existing_avg_keys = {
                    row_avg_key(row)
                    for row in read_csv_rows(avg_path, avg_fields)
                }
                missing_eps = [
                    target_eps
                    for target_eps in run_target_eps_list
                    if per_eps_key(strategy_label, ratio_text, k, target_eps) not in existing_per_eps_keys
                ]
                this_avg_key = avg_key(strategy_label, ratio_text, k)
                if not missing_eps:
                    if this_avg_key in existing_avg_keys:
                        print(f"[Skip] strategy={strategy} | ratio={ratio:.1f} | k={k} already complete.")
                        continue
                    existing_rows = combo_per_eps_rows(per_eps_path, per_eps_fields, strategy_label, ratio_text, k)
                    avg_row = build_avg_row_from_per_eps(existing_rows, target_eps_list, manifest_path)
                    if avg_row is not None:
                        append_row_if_missing(avg_path, avg_fields, avg_row, row_avg_key, this_avg_key)
                        print(f"[Recovered avg] strategy={strategy} | ratio={ratio:.1f} | k={k}")
                        continue

                print(
                    f"\n>>> Strategy={strategy} | Ratio={ratio:.1f} | k={k} | "
                    + (
                        "building privatized-prefix candidates per eps..."
                        if strategy == "dp_spd_generation"
                        else "building grouped candidates once for all eps..."
                    )
                )

                group_items = None
                noisy_dataset = None
                real_ppl = dummy_ppl = ppl_gap = float("nan")
                if strategy != "dp_spd_generation":
                    group_items = build_strategy_group_items(
                        strategy=strategy,
                        real_texts=real_texts,
                        corpus_cache=corpus_cache,
                        token_pool=token_pool,
                        continuation_tokenizer=continuation_tokenizer,
                        continuation_model=continuation_model,
                        attack_tokenizer=attack_tokenizer,
                        max_length=args.max_length,
                        ratio=ratio,
                        k=k,
                        seed=args.seed,
                    )

                    real_ppl, dummy_ppl, ppl_gap = compute_average_ppl(
                        items=group_items,
                        model=attack_model,
                        device=args.device,
                        batch_size=args.ppl_batch_size,
                    )

                    noisy_dataset = GroupedMIASequenceDataset(
                        items=group_items,
                        pretrained_embeddings=attack_model.get_input_embeddings(),
                        hidden_size=attack_model.config.hidden_size,
                        clip_norm=args.clip_norm,
                        mechanism="Laplace",
                        epsilon_private=1.0,
                        epsilon_non_sensitive=1.0,
                        seed=args.seed,
                    )

                for target_eps in missing_eps:
                    local_eps = LOCAL_EPS_MAPPING[target_eps][k]
                    noise_seed = int(target_eps * 1000 + k * 10 + ratio * 100)
                    if strategy == "dp_spd_generation":
                        group_items = build_strategy_group_items(
                            strategy=strategy,
                            real_texts=real_texts,
                            corpus_cache=corpus_cache,
                            token_pool=token_pool,
                            continuation_tokenizer=continuation_tokenizer,
                            continuation_model=continuation_model,
                            attack_tokenizer=attack_tokenizer,
                            max_length=args.max_length,
                            ratio=ratio,
                            k=k,
                            seed=args.seed,
                            projection_embeddings=attack_model.get_input_embeddings(),
                            hidden_size=attack_model.config.hidden_size,
                            clip_norm=args.clip_norm,
                            epsilon_non_sensitive=target_eps,
                            projection_seed=noise_seed,
                        )
                        real_ppl, dummy_ppl, ppl_gap = compute_average_ppl(
                            items=group_items,
                            model=attack_model,
                            device=args.device,
                            batch_size=args.ppl_batch_size,
                        )
                        noisy_dataset = GroupedMIASequenceDataset(
                            items=group_items,
                            pretrained_embeddings=attack_model.get_input_embeddings(),
                            hidden_size=attack_model.config.hidden_size,
                            clip_norm=args.clip_norm,
                            mechanism="Laplace",
                            epsilon_private=1.0,
                            epsilon_non_sensitive=1.0,
                            seed=args.seed,
                        )

                    noisy_dataset.set_noise_config(
                        epsilon_non_sensitive=target_eps,
                        epsilon_private=local_eps,
                        seed=noise_seed,
                    )
                    dataloader = DataLoader(noisy_dataset, batch_size=k + 1, shuffle=False)

                    print(
                        f"  -> target_eps={target_eps} | local_eps={local_eps} | "
                        f"baseline={100.0 / (k + 1):.2f}%"
                    )
                    asr = run_min_k_prob_mia(dataloader, attack_model, args, k_ratio=args.mia_k_ratio)
                    baseline = 100.0 / (k + 1)

                    row = {
                        "Strategy": strategy_label,
                        "Dummy Ratio": ratio_text,
                        "Dummy (k)": k,
                        "Target Eps": target_eps,
                        "Required Local Eps": local_eps,
                        "MIA ASR (%)": f"{asr * 100:.2f}",
                        "Random Baseline (%)": f"{baseline:.2f}",
                        "Real PPL": f"{real_ppl:.4f}",
                        "Dummy PPL": f"{dummy_ppl:.4f}",
                        "PPL Gap": f"{ppl_gap:.4f}",
                        "Eval Manifest": str(manifest_path),
                    }
                    appended = append_row_if_missing(
                        per_eps_path,
                        per_eps_fields,
                        row,
                        row_per_eps_key,
                        per_eps_key(strategy_label, ratio_text, k, target_eps),
                    )
                    if appended:
                        print(f"     [Saved per-eps] target_eps={target_eps}")
                    else:
                        print(f"     [Skip per-eps] target_eps={target_eps} already exists")

                existing_rows = combo_per_eps_rows(per_eps_path, per_eps_fields, strategy_label, ratio_text, k)
                avg_row = build_avg_row_from_per_eps(existing_rows, target_eps_list, manifest_path)
                if avg_row is not None:
                    append_row_if_missing(avg_path, avg_fields, avg_row, row_avg_key, this_avg_key)

                print(
                    f"[Saved] strategy={strategy} | ratio={ratio:.1f} | k={k} | "
                    f"last_real_ppl={real_ppl:.4f} | last_dummy_ppl={dummy_ppl:.4f} | last_gap={ppl_gap:.4f}"
                )

    print("\n" + "=" * 80)
    print("MIA dummy-strategy benchmark complete.")
    print(f"Per-eps CSV: {per_eps_path}")
    print(f"Average-over-eps CSV: {avg_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
