#!/usr/bin/env python3
"""Qwen3-14B/CNN-DailyMail utility evaluation for six DP baseline adapters.

The DP-SPD Figure-4(b) evaluator uses a 768-token ``Article -> Summary``
context, 128 greedy continuation tokens, clean next-token PPL, and ROUGE-L on
the same continuation.  This runner deliberately reuses that exact data
formatter and metric functions for CAPE, CusText, SanText, DP-Forward and
TextObfuscator.  InferDPT is a two-stage architecture, so its extractor PPL
is emitted separately and explicitly marked as non-comparable to server PPL.

The original papers target classification/MLM or a different model family;
the rows are therefore transparent frozen-Qwen3 *generation adapters*, not
claims of the authors' original CNN/DailyMail results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from datasets import load_from_disk
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.cape.sanitizer import CAPEConfig, normalize_and_perturb_sequence
from baseline.custext.sanitizer import CusTextConfig, CusTextSanitizer
from baseline.dp_forward.noise import DPForwardConfig
from baseline.inferdpt.rantxt import RANTextConfig, RANTextSanitizer
from baseline.qwen3_split import Qwen3DPForwardSplit, Qwen3TextObfuscatorSplit
from baseline.santext.sanitizer import SanTextConfig, SanTextTokenSanitizer, SanTextVocabulary, build_vocabulary
from baseline.textobfuscator.split_llama import TextObfuscatorConfig
from experiments.generation.qwen3_cnndm_eval import (
    build_cnndm_eval_pack,
    build_prompt,
    calculate_ppl as dpspd_calculate_ppl,
    calculate_rouge as dpspd_calculate_rouge,
    load_qwen3_model,
)


METHODS = ("plaintext", "cape", "custext", "dp_forward", "inferdpt", "santext", "textobfuscator")
RESULT_FIELDS = [
    "method", "adapter_status", "dataset", "model", "epsilon", "epsilon_role", "epsilon_semantics",
    "eval_samples", "rouge_reference_samples", "max_prompt_length", "max_ref_tokens", "max_new_tokens", "rouge_context_len",
    "batch_size", "ppl", "rouge_l", "remote_rouge_l", "extractor_conditional_ppl", "ppl_comparability",
    "representation", "paper_component", "omitted_paper_component", "mapping_cache", "active_tokens",
    "realized_replacements", "realized_replacement_fraction", "clean_split_max_abs_logit_delta",
    "clean_split_mean_abs_logit_delta", "details_json", "seed",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_splits(value: str) -> list[str]:
    splits = [part.strip() for part in value.split(",") if part.strip()]
    if not splits:
        raise ValueError("mapping_splits must contain at least one split")
    return splits


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True, help="Local DatasetDict containing CNN/DailyMail train/validation/test.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--epsilon", required=True, type=float)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument(
        "--sample_start",
        type=int,
        default=0,
        help="Zero-based offset within the deterministically prepared valid examples; used for exact multi-GPU sharding.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--max_ref_tokens", type=int, default=160)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--rouge_context_len", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_samples", type=str2bool, default=False)
    parser.add_argument("--run_ppl", type=str2bool, default=True)
    parser.add_argument("--run_rouge", type=str2bool, default=True)
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--prepare_only", type=str2bool, default=False,
                        help="Build method-wide public mappings/prototypes, then exit. Used by the launch script.")
    parser.add_argument("--save_examples_jsonl", default=None)

    # CusText: full model-token mapping, matching the existing Figure-3 token adapter.
    parser.add_argument("--custext_top_k", type=int, default=20)
    parser.add_argument("--custext_metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--custext_mapping_strategy", choices=["release_compatible", "conservative", "aggressive"], default="release_compatible")
    parser.add_argument("--custext_mapping_cache", default=None)
    parser.add_argument("--force_rebuild_mapping", type=str2bool, default=False)

    # SanText: paper defaults correspond to SanText+ Table-1 settings.
    parser.add_argument("--santext_variant", choices=["santext", "santext_plus"], default="santext")
    parser.add_argument("--santext_sensitive_fraction", type=float, default=0.9)
    parser.add_argument("--santext_non_sensitive_sanitize_probability", type=float, default=0.3)
    parser.add_argument("--santext_source_chunk_size", type=int, default=8,
                        help="Unique source-token rows per GPU chunk; 8 is memory-safe for Qwen3-14B.")
    parser.add_argument("--mapping_splits", default="train")
    parser.add_argument("--mapping_max_samples", type=int, default=0)
    parser.add_argument("--mapping_batch_size", type=int, default=64)
    parser.add_argument("--santext_mapping_cache", default=None)

    # InferDPT/RANTEXT.
    parser.add_argument("--inferdpt_candidate_vocab_size", type=int, default=11000)
    parser.add_argument("--inferdpt_source_chunk_size", type=int, default=16)
    parser.add_argument("--inferdpt_mapping_cache", default=None)
    parser.add_argument("--max_extractor_prompt_tokens", type=int, default=768)

    # Split adapters. -1 preserves the original paper relative depth.
    parser.add_argument("--dp_forward_noise_layer", type=int, default=-1,
                        help="0-based boundary; -1 selects the original ~92%% depth automatically.")
    parser.add_argument("--dp_forward_delta", type=float, default=1e-5)
    parser.add_argument("--dp_forward_norm_c", type=float, default=1.0)
    parser.add_argument("--textobfuscator_client_layers", type=int, default=-1,
                        help="-1 selects the original 3/12=25%% client-depth ratio automatically.")
    return parser


def require_paths(args) -> None:
    missing = [value for value in (args.model_path, args.dataset_path) if not Path(value).is_dir()]
    if missing:
        raise FileNotFoundError("Missing local directory/directories: " + ", ".join(missing))
    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if args.batch_size < 1 or args.max_samples < 1 or args.max_new_tokens < 1:
        raise ValueError("max_samples, batch_size and max_new_tokens must be positive")
    if args.rouge_context_len != args.max_prompt_length:
        raise ValueError(
            "For Figure-4(b) comparability, rouge_context_len must equal max_prompt_length "
            f"({args.max_prompt_length}); got {args.rouge_context_len}."
        )


class TokenDataset(Dataset):
    """The exact padded causal-LM table used by the DP-SPD CNN/DM evaluator."""

    def __init__(self, texts: Sequence[str], tokenizer, total_length: int) -> None:
        encoded = tokenizer(
            list(texts), truncation=True, max_length=total_length, padding="max_length", return_tensors="pt"
        )
        self.input_ids = encoded["input_ids"].long().contiguous()
        self.attention_mask = encoded["attention_mask"].long().contiguous()

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "sample_index": torch.tensor(index, dtype=torch.long),
        }


def collate_ids(batch: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in ("input_ids", "attention_mask", "sample_index")}


def collate_embeddings(batch: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in ("input_ids", "attention_mask", "noisy_embeddings")}


class CAPEDataset(Dataset):
    def __init__(self, clean: TokenDataset, embedding_weight, config: CAPEConfig, *, seed: int) -> None:
        self.input_ids = clean.input_ids
        self.attention_mask = clean.attention_mask
        self.embedding_weight = embedding_weight.detach().cpu().contiguous()
        self.config = config
        self.seed = seed

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        input_ids = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        clean = F.embedding(input_ids, self.embedding_weight)
        noised, _ = normalize_and_perturb_sequence(
            clean, attention_mask, self.config, seed=self.seed + 1_000_003 * index
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "noisy_embeddings": noised}


class DiscreteSanitizedDataset(Dataset):
    def __init__(self, clean: TokenDataset, embedding_weight, sanitized_ids: torch.Tensor) -> None:
        self.input_ids = clean.input_ids
        self.attention_mask = clean.attention_mask
        if sanitized_ids.shape != self.input_ids.shape:
            raise ValueError("sanitized_ids must match the clean token table")
        self.sanitized_ids = sanitized_ids.long().cpu().contiguous()
        self.embedding_weight = embedding_weight.detach().cpu().contiguous()

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "noisy_embeddings": F.embedding(self.sanitized_ids[index], self.embedding_weight),
        }


def output_path_for_cache(args, method: str, suffix: str) -> Path:
    root = Path(args.output_csv).resolve().parents[1] / "mappings"
    name = Path(args.model_path).name.replace(" ", "_")
    return root / f"{method}_cnndm_{name}_{suffix}"


def build_custext(args, model, tokenizer) -> tuple[CusTextSanitizer, Path]:
    config = CusTextConfig(
        epsilon=args.epsilon,
        top_k=args.custext_top_k,
        metric=args.custext_metric,
        mapping_strategy=args.custext_mapping_strategy,
        seed=args.seed,
    )
    cache = Path(args.custext_mapping_cache) if args.custext_mapping_cache else output_path_for_cache(
        args, "custext", f"eps{args.epsilon:g}_k{args.custext_top_k}_{args.custext_metric}_{args.custext_mapping_strategy}.json"
    )
    if cache.is_file() and not args.force_rebuild_mapping:
        sanitizer = CusTextSanitizer.load(cache)
        expected = {"model_path": args.model_path, "tokenizer_vocab_size": int(len(tokenizer))}
        if sanitizer.kind != "model_token" or sanitizer.config != config or any(
            sanitizer.metadata.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"CusText cache does not match this experiment: {cache}")
        print(f"[CusText] Reusing mapping {cache}")
        return sanitizer, cache
    sanitizer = CusTextSanitizer.from_model_token_embeddings(
        model.get_input_embeddings().weight,
        tokenizer.all_special_ids,
        config,
        device=str(next(model.parameters()).device),
        metadata={
            "model_path": args.model_path,
            "tokenizer_vocab_size": int(len(tokenizer)),
            "protocol": "qwen3_cnndm_full_sequence_token_adapter_v1",
        },
    )
    sanitizer.save(cache)
    print(f"[CusText] Saved mapping {cache}")
    return sanitizer, cache


def _iter_public_prompt_texts(raw, splits: Sequence[str], limit: int) -> Iterable[str]:
    emitted = 0
    for split in splits:
        if split not in raw:
            raise KeyError(f"Mapping split {split!r} not found; available={list(raw.keys())}")
        dataset = raw[split]
        for item in dataset:
            article = str(item.get("article", "")).strip()
            if article:
                yield build_prompt(article)
                emitted += 1
                if limit > 0 and emitted >= limit:
                    return


def count_public_prompt_tokens(raw, tokenizer, splits: Sequence[str], limit: int, batch_size: int) -> Counter:
    special = set(int(item) for item in tokenizer.all_special_ids)
    counts: Counter = Counter()
    pending: list[str] = []
    for text in tqdm(_iter_public_prompt_texts(raw, splits, limit), desc="Building public CNN/DM vocabulary"):
        pending.append(text)
        if len(pending) < batch_size:
            continue
        encoded = tokenizer(pending, add_special_tokens=True, padding=False, truncation=False)
        for ids in encoded["input_ids"]:
            counts.update(int(token) for token in ids if int(token) not in special)
        pending.clear()
    if pending:
        encoded = tokenizer(pending, add_special_tokens=True, padding=False, truncation=False)
        for ids in encoded["input_ids"]:
            counts.update(int(token) for token in ids if int(token) not in special)
    if not counts:
        raise ValueError("No non-special token was found in the public CNN/DailyMail mapping corpus")
    return counts


def static_mapping_metadata(args, tokenizer, method: str) -> dict[str, object]:
    return {
        "schema": f"qwen3_cnndm_{method}_public_vocab_v1",
        "model_path": args.model_path,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "mapping_splits": parse_splits(args.mapping_splits),
        "mapping_max_samples": int(args.mapping_max_samples),
        "public_text": "CNN/DailyMail_article_summary_prompt",
    }


def build_santext_vocabulary(args, raw, tokenizer) -> tuple[SanTextVocabulary, Path]:
    cache = Path(args.santext_mapping_cache) if args.santext_mapping_cache else output_path_for_cache(
        args, "santext", f"w{args.santext_sensitive_fraction:g}_n{args.mapping_max_samples or 'all'}.json"
    )
    expected = static_mapping_metadata(args, tokenizer, "santext")
    expected["sensitive_fraction"] = float(args.santext_sensitive_fraction)
    if cache.is_file() and not args.force_rebuild_mapping:
        vocabulary = SanTextVocabulary.from_dict(json.loads(cache.read_text(encoding="utf-8")))
        if any(vocabulary.metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"SanText mapping does not match this experiment: {cache}")
        print(f"[SanText] Reusing vocabulary {cache}")
        return vocabulary, cache
    config = SanTextConfig(
        variant=args.santext_variant,
        epsilon=args.epsilon,
        sensitive_fraction=args.santext_sensitive_fraction,
        non_sensitive_sanitize_probability=args.santext_non_sensitive_sanitize_probability,
        source_chunk_size=args.santext_source_chunk_size,
        seed=args.seed,
    )
    counts = count_public_prompt_tokens(
        raw, tokenizer, parse_splits(args.mapping_splits), args.mapping_max_samples, args.mapping_batch_size
    )
    vocabulary = build_vocabulary(counts, config, metadata=expected)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(vocabulary.to_dict(), ensure_ascii=False), encoding="utf-8")
    print(f"[SanText] Saved vocabulary {cache} (|V|={len(vocabulary.source_token_ids)})")
    return vocabulary, cache


def build_inferdpt_vocab(args, raw, tokenizer) -> tuple[list[int], Path]:
    cache = Path(args.inferdpt_mapping_cache) if args.inferdpt_mapping_cache else output_path_for_cache(
        args, "inferdpt", f"top{args.inferdpt_candidate_vocab_size}_n{args.mapping_max_samples or 'all'}.json"
    )
    expected = static_mapping_metadata(args, tokenizer, "inferdpt")
    expected["candidate_vocab_size"] = int(args.inferdpt_candidate_vocab_size)
    if cache.is_file() and not args.force_rebuild_mapping:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if any(payload.get("metadata", {}).get(key) != value for key, value in expected.items()):
            raise ValueError(f"InferDPT mapping does not match this experiment: {cache}")
        return [int(token) for token in payload["candidate_token_ids"]], cache
    counts = count_public_prompt_tokens(
        raw, tokenizer, parse_splits(args.mapping_splits), args.mapping_max_samples, args.mapping_batch_size
    )
    ids = [int(token) for token, _ in counts.most_common(args.inferdpt_candidate_vocab_size)]
    if len(ids) < args.inferdpt_candidate_vocab_size:
        raise ValueError(f"Only {len(ids)} public tokens exist, fewer than requested candidate vocabulary")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"metadata": expected, "candidate_token_ids": ids}), encoding="utf-8")
    print(f"[InferDPT] Saved candidate vocabulary {cache} (|V|={len(ids)})")
    return ids, cache


def left_pad(sequences: Sequence[Sequence[int]], pad_token_id: int, device: torch.device):
    width = max(max(len(sequence), 1) for sequence in sequences)
    ids = torch.full((len(sequences), width), int(pad_token_id), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        if sequence:
            values = torch.tensor(sequence, dtype=torch.long, device=device)
            ids[row, width - len(sequence):] = values
            mask[row, width - len(sequence):] = 1
    return ids, mask


@torch.no_grad()
def generate_batches(model, tokenizer, sequences, batch_size: int, max_new_tokens: int, device: torch.device) -> list[list[int]]:
    outputs: list[list[int]] = []
    for start in tqdm(range(0, len(sequences), batch_size), desc="Generate"):
        rows = sequences[start:start + batch_size]
        ids, mask = left_pad(rows, tokenizer.pad_token_id, device)
        generated = model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        outputs.extend([list(map(int, row[ids.shape[1]:].detach().cpu().tolist())) for row in generated])
    return outputs


def rouge_from_texts(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Match Evaluate ROUGE-L locally, without a Hub metric-script lookup."""
    try:
        from rouge_score import rouge_scorer
    except Exception as exc:
        raise RuntimeError(
            "ROUGE-L requires the local `rouge_score` package. Install it with `pip install rouge-score`."
        ) from exc
    pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not pairs:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(sum(scorer.score(reference, prediction)["rougeL"].fmeasure for prediction, reference in pairs) / len(pairs))


def extraction_prompt(raw_prompt: str, remote_generation: str) -> str:
    return (
        "Your task is to write a concise news summary. Use the Perturbed Generation as primary material, "
        "while retaining factual consistency with the Article. Output only the summary.\n\n"
        f"Article Prompt:\n{raw_prompt}\n\nPerturbed Generation:\n{remote_generation}\n\nSummary:\n"
    )


@torch.no_grad()
def extractor_ppl(model, tokenizer, prompts: Sequence[str], targets: Sequence[Sequence[int]], *, batch_size: int, max_prompt_tokens: int, device: torch.device) -> float:
    total_nll, total_tokens = 0.0, 0
    loss = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
    for start in tqdm(range(0, len(prompts), batch_size), desc="InferDPT extractor PPL"):
        prompt_batch = prompts[start:start + batch_size]
        target_batch = targets[start:start + batch_size]
        contexts = [tokenizer(text, add_special_tokens=False)["input_ids"][-max_prompt_tokens:] for text in prompt_batch]
        ids, attention = left_pad([context + list(target) for context, target in zip(contexts, target_batch)], tokenizer.pad_token_id, device)
        labels = torch.full_like(ids, -100)
        for row, (context, target) in enumerate(zip(contexts, target_batch)):
            left = ids.shape[1] - len(context) - len(target)
            labels[row, left + len(context):left + len(context) + len(target)] = torch.tensor(target, device=device)
        logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        total_nll += float(loss(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)).item())
        total_tokens += int(labels[:, 1:].ne(-100).sum().item())
    return math.exp(total_nll / total_tokens) if total_tokens else float("nan")


@torch.no_grad()
def split_ppl(runner, loader, device: torch.device) -> tuple[float, dict[str, float]]:
    total_nll, total_tokens = 0.0, 0
    for batch in tqdm(loader, desc="PPL"):
        ids, mask, indices = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["sample_index"].to(device)
        logits = runner.forward(ids, mask, indices, perturb=True)
        losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none")
        active = mask[:, 1:].float()
        total_nll += float((losses.view_as(active) * active).sum().item())
        total_tokens += int(active.sum().item())
    mean_nll = total_nll / total_tokens if total_tokens else float("nan")
    return (math.exp(mean_nll) if total_tokens else float("nan")), {"ppl_scored_tokens": total_tokens, "mean_nll": mean_nll}


@torch.no_grad()
def split_rouge(runner, tokenizer, loader, device: torch.device, context_len: int, gen_len: int) -> float:
    predictions, references = [], []
    for batch in tqdm(loader, desc="ROUGE"):
        ids, indices = batch["input_ids"].to(device), batch["sample_index"].to(device)
        current_ids = ids[:, :context_len].clone()
        current_mask = torch.ones_like(current_ids)
        cache, generated = None, []
        for _ in range(gen_len):
            logits, cache = runner.forward_cached(current_ids, current_mask, indices, past_key_values=cache, perturb=True)
            next_token = logits[:, -1].argmax(dim=-1)
            generated.append(next_token)
            current_ids = next_token.unsqueeze(1)
            current_mask = torch.cat([current_mask, torch.ones_like(current_ids)], dim=1)
        predictions.extend(tokenizer.batch_decode(torch.stack(generated, dim=1).cpu(), skip_special_tokens=True))
        references.extend(tokenizer.batch_decode(ids[:, context_len:context_len + gen_len].cpu(), skip_special_tokens=True))
    return rouge_from_texts(predictions, references)


def method_common(args, dataset: TokenDataset) -> dict[str, object]:
    # The original Figure-4(b) evaluator scores labels at fixed token offsets
    # [768, 896).  CNN/DM examples shorter than that have no usable ROUGE
    # reference and are intentionally omitted by the same evaluator.  Record
    # their count so a tiny smoke run cannot be mistaken for a failed metric.
    reference_mask = dataset.attention_mask[
        :, args.rouge_context_len:args.rouge_context_len + args.max_new_tokens
    ]
    return {
        "dataset": "cnn_dailymail",
        "model": args.model_path,
        "epsilon": args.epsilon,
        # epsilon=10000 is a diagnostic large-budget control.  It is not one
        # of Figure-4(b)'s privacy points and must not be plotted as one.
        "epsilon_role": "large_epsilon_control" if args.epsilon >= 10_000.0 else "privacy_budget",
        "eval_samples": len(dataset),
        "rouge_reference_samples": int(reference_mask.any(dim=1).sum().item()),
        "max_prompt_length": args.max_prompt_length,
        "max_ref_tokens": args.max_ref_tokens,
        "max_new_tokens": args.max_new_tokens,
        "rouge_context_len": args.rouge_context_len,
        "batch_size": args.batch_size,
        "active_tokens": int(dataset.attention_mask.sum().item()),
        "seed": args.seed,
    }


@torch.no_grad()
def plaintext_ppl(model, loader, device: torch.device) -> float:
    total_nll, total_tokens = 0.0, 0
    for batch in tqdm(loader, desc="PPL"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        losses = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
            reduction="none",
        ).view_as(mask[:, 1:])
        active = mask[:, 1:].float()
        total_nll += float((losses * active).sum().item())
        total_tokens += int(active.sum().item())
    return math.exp(total_nll / total_tokens) if total_tokens else float("nan")


@torch.no_grad()
def plaintext_rouge(
    model, tokenizer, loader, device: torch.device, context_len: int, gen_len: int
) -> tuple[float, dict[str, int]]:
    predictions, references = [], []
    emitted_lengths: list[int] = []
    for batch in tqdm(loader, desc="ROUGE"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        prompt_ids = ids[:, :context_len]
        prompt_mask = mask[:, :context_len]
        generated = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=gen_len,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[:, prompt_ids.shape[1]:]
        for row in new_tokens:
            eos_positions = row.eq(tokenizer.eos_token_id).nonzero(as_tuple=False)
            emitted_lengths.append(int(eos_positions[0].item() + 1) if eos_positions.numel() else int(row.numel()))
        gen_texts = tokenizer.batch_decode(new_tokens.cpu(), skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(ids[:, context_len:context_len + gen_len].cpu(), skip_special_tokens=True)
        for prediction, reference in zip(gen_texts, ref_texts):
            if reference.strip():
                predictions.append(prediction)
                references.append(reference)
    score = rouge_from_texts(predictions, references) if predictions else 0.0
    stats = {
        "generation_samples": len(emitted_lengths),
        "early_eos_samples": sum(length < gen_len for length in emitted_lengths),
        "min_generated_tokens": min(emitted_lengths, default=0),
        "max_generated_tokens": max(emitted_lengths, default=0),
    }
    return score, stats


def evaluate_plaintext(args, model, tokenizer, clean: TokenDataset) -> dict[str, object]:
    """Evaluate unperturbed input IDs with the exact DP-SPD prompt and offsets."""
    loader = DataLoader(clean, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ids)
    ppl = plaintext_ppl(model, loader, torch.device(args.device)) if args.run_ppl else float("nan")
    loader = DataLoader(clean, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ids)
    rouge, rouge_stats = plaintext_rouge(
        model, tokenizer, loader, torch.device(args.device), args.rouge_context_len, args.max_new_tokens
    ) if args.run_rouge else (float("nan"), {})
    return {
        "method": "Plaintext",
        "adapter_status": "unperturbed_upper_bound",
        "epsilon_role": "plaintext_upper_bound",
        "epsilon_semantics": "not_applicable__reported_alongside_selected_privacy_budget",
        "representation": "clean_Qwen3_input_ids",
        "paper_component": "none",
        "omitted_paper_component": "none",
        "ppl": ppl,
        "rouge_l": rouge,
        "ppl_comparability": "direct_server_clean_label_PPL",
        "details_json": json.dumps(
            {
                "sample_start": args.sample_start,
                "transport": "exact_clean_input_ids",
                **rouge_stats,
            },
            sort_keys=True,
        ),
    }


def evaluate_embedding_method(args, model, tokenizer, clean: TokenDataset, raw, method: str) -> dict[str, object]:
    details: dict[str, object] = {}
    mapping_cache = ""
    if method == "cape":
        config = CAPEConfig(epsilon=args.epsilon, seed=args.seed)
        dataset = CAPEDataset(clean, model.get_input_embeddings().weight, config, seed=args.seed)
        details = {"laplace_scale": 1.0 / args.epsilon, "normalization": "per_active_sequence_min_max_[0,1]"}
        labels = {
            "method": "CAPE-style",
            "adapter_status": "frozen_generation_adapter",
            "epsilon_semantics": "CAPE_Laplace_scale_1_over_epsilon__not_end_to_end_sequence_DP",
            "representation": "full_Qwen3_input_embedding_sequence__not_original_mean_pooled_BERT",
            "paper_component": "min_max_normalization_plus_iid_Laplace_embedding_noise",
            "omitted_paper_component": "adversarial_target_private_attribute_training",
        }
    elif method == "custext":
        sanitizer, cache = build_custext(args, model, tokenizer)
        mapping_cache = str(cache)
        sanitized_ids, counts = sanitizer.sanitize_token_ids(clean.input_ids, clean.attention_mask, rng=np.random.default_rng(args.seed))
        dataset = DiscreteSanitizedDataset(clean, model.get_input_embeddings().weight, sanitized_ids)
        # ``CusTextSanitizer.sanitize_token_ids`` deliberately returns only
        # mapping-domain counters (``mapped``/``unmapped``).  Unlike the
        # SanText adapter, it does not manufacture a replacement statistic:
        # a sampled output may legitimately equal its input.  Measure that
        # statistic here on the evaluated token table, excluding the special
        # IDs that CusText intentionally preserves.
        active = clean.attention_mask.bool()
        special = torch.zeros_like(active)
        for token_id in tokenizer.all_special_ids:
            special |= clean.input_ids.eq(int(token_id))
        eligible_mask = active & ~special
        eligible = int(eligible_mask.sum().item())
        mapped = int(counts.get("mapped", 0))
        unmapped = int(counts.get("unmapped", 0))
        if mapped != eligible:
            raise RuntimeError(
                "CusText full-token adapter did not cover every non-special active token: "
                f"mapped={mapped}, eligible={eligible}, unmapped={unmapped}."
            )
        realized_replacements = int((eligible_mask & sanitized_ids.ne(clean.input_ids)).sum().item())
        details = {
            "top_k": args.custext_top_k,
            "metric": args.custext_metric,
            "mapping_strategy": args.custext_mapping_strategy,
            "eligible_non_special_tokens": eligible,
            "special_active_tokens": int((active & special).sum().item()),
            "realized_replacements": realized_replacements,
            "realized_replacement_fraction": realized_replacements / eligible if eligible else float("nan"),
            **counts,
        }
        labels = {
            "method": "CusText-style",
            "adapter_status": "full_sequence_Qwen_token_adapter",
            "epsilon_semantics": "per_token_exponential_mechanism",
            "representation": "full_Qwen3_BPE_token_replacement__not_original_word_vector_reconstruction",
            "paper_component": "CusText_shared_candidate_group_plus_exponential_mechanism",
            "omitted_paper_component": "original_word_level_Counter_Fitting_text_reconstruction",
            "realized_replacements": realized_replacements,
            "realized_replacement_fraction": realized_replacements / eligible if eligible else float("nan"),
        }
    elif method == "santext":
        vocabulary, cache = build_santext_vocabulary(args, raw, tokenizer)
        mapping_cache = str(cache)
        config = SanTextConfig(
            variant=args.santext_variant, epsilon=args.epsilon, sensitive_fraction=args.santext_sensitive_fraction,
            non_sensitive_sanitize_probability=args.santext_non_sensitive_sanitize_probability,
            source_chunk_size=args.santext_source_chunk_size, seed=args.seed,
        )
        # Qwen tokenizers can expose fewer encodable IDs than the tied input
        # embedding table (reserved rows remain in ``config.vocab_size``).
        # The sanitizer lookup must follow the embedding table, not len(tokenizer).
        sanitizer = SanTextTokenSanitizer(
            vocabulary,
            config,
            vocab_size=int(model.get_input_embeddings().weight.shape[0]),
            special_token_ids=tokenizer.all_special_ids,
        )
        sanitized_ids, counts = sanitizer.sanitize_token_ids(
            clean.input_ids, clean.attention_mask, model.get_input_embeddings().weight.detach(), seed=args.seed
        )
        dataset = DiscreteSanitizedDataset(clean, model.get_input_embeddings().weight, sanitized_ids)
        eligible = int(counts["eligible_non_special_tokens"])
        details = {"variant": args.santext_variant, "mapping_vocab_size": len(vocabulary.source_token_ids), **counts}
        labels = {
            "method": "SanText-style",
            "adapter_status": "full_sequence_Qwen_token_adapter",
            "epsilon_semantics": "per_token_metric_LDP_epsilon",
            "representation": "full_Qwen3_BPE_token_sanitization__not_original_word_level_BERT",
            "paper_component": "Euclidean_exponential_mechanism_over_public_vocabulary",
            "omitted_paper_component": "original_word_tokenization_and_downstream_task_training",
            "realized_replacements": int(counts["realized_replacements"]),
            "realized_replacement_fraction": counts["realized_replacements"] / eligible if eligible else float("nan"),
        }
    else:
        raise ValueError(method)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embeddings)
    ppl = dpspd_calculate_ppl(model, loader, torch.device(args.device)) if args.run_ppl else float("nan")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embeddings)
    rouge = dpspd_calculate_rouge(model, tokenizer, loader, torch.device(args.device), args.rouge_context_len, args.max_new_tokens) if args.run_rouge else {"rougeL": float("nan")}
    return {**labels, "ppl": ppl, "rouge_l": float(rouge.get("rougeL", float("nan"))), "mapping_cache": mapping_cache,
            "details_json": json.dumps(details, sort_keys=True)}


def evaluate_split_method(args, model, tokenizer, clean: TokenDataset, method: str) -> dict[str, object]:
    total_layers = len(model.model.layers)
    if method == "dp_forward":
        noise_layer = args.dp_forward_noise_layer if args.dp_forward_noise_layer >= 0 else round(0.92 * total_layers) - 1
        config = DPForwardConfig(args.epsilon, args.dp_forward_delta, args.dp_forward_norm_c, noise_layer, args.seed)
        runner = Qwen3DPForwardSplit(model, config)
        labels = {
            "method": "DP-Forward-style",
            "adapter_status": "frozen_Qwen3_deep_boundary_adapter",
            "epsilon_semantics": "official_release_analytic_Gaussian_epsilon_delta",
            "representation": f"Qwen3_post_decoder_layer_{noise_layer}_contextual_matrix",
            "paper_component": "sequence_Frobenius_clip_plus_analytic_Gaussian_forward_noise",
            "omitted_paper_component": "joint_noisy_BERT_finetuning_and_classification_head",
        }
        details = {"noise_layer": noise_layer, "client_layers": runner.client_layers, "noise_std": config.noise_std,
                   "delta": args.dp_forward_delta, "norm_c": args.dp_forward_norm_c}
    elif method == "textobfuscator":
        client_layers = args.textobfuscator_client_layers if args.textobfuscator_client_layers >= 0 else round(0.25 * total_layers)
        config = TextObfuscatorConfig(args.epsilon, client_layers, args.seed)
        runner = Qwen3TextObfuscatorSplit(model, config)
        labels = {
            "method": "TextObfuscator-style",
            "adapter_status": "frozen_Qwen3_client_server_adapter",
            "epsilon_semantics": "Laplace_scale_1_over_epsilon_hyperparameter",
            "representation": f"Qwen3_post_decoder_layer_{client_layers - 1}_contextual_state",
            "paper_component": "client_server_split_plus_all_token_boundary_Laplace_noise",
            "omitted_paper_component": "joint_RoBERTa_training_TFIDF_redivision_and_prototype_losses",
        }
        details = {"client_layers": client_layers, "server_layers": total_layers - client_layers, "laplace_scale": 1 / args.epsilon}
    else:
        raise ValueError(method)
    loader = DataLoader(clean, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ids)
    first = next(iter(loader))
    check = runner.assert_clean_equivalence(first["input_ids"][:1].to(args.device), first["attention_mask"][:1].to(args.device))
    ppl, ppl_stats = split_ppl(runner, loader, torch.device(args.device)) if args.run_ppl else (float("nan"), {})
    loader = DataLoader(clean, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ids)
    rouge = split_rouge(runner, tokenizer, loader, torch.device(args.device), args.rouge_context_len, args.max_new_tokens) if args.run_rouge else float("nan")
    details.update(ppl_stats)
    return {**labels, "ppl": ppl, "rouge_l": rouge, "ppl_comparability": "direct_server_clean_label_PPL",
            "clean_split_max_abs_logit_delta": check["clean_split_max_abs_logit_delta"],
            "clean_split_mean_abs_logit_delta": check["clean_split_mean_abs_logit_delta"],
            "details_json": json.dumps(details, sort_keys=True)}


def evaluate_inferdpt(args, model, tokenizer, clean: TokenDataset, raw) -> dict[str, object]:
    candidate_ids, cache = build_inferdpt_vocab(args, raw, tokenizer)
    config = RANTextConfig(args.epsilon, args.inferdpt_source_chunk_size, args.seed)
    sanitizer = RANTextSanitizer(
        candidate_ids,
        config,
        vocab_size=int(model.get_input_embeddings().weight.shape[0]),
    )
    prompt_ids, target_ids, raw_prompts, references = [], [], [], []
    for ids, mask in zip(clean.input_ids, clean.attention_mask):
        active = ids[mask.bool()].tolist()
        prompt = active[:args.rouge_context_len]
        target = active[args.rouge_context_len:args.rouge_context_len + args.max_new_tokens]
        prompt_ids.append(prompt)
        target_ids.append(target)
        raw_prompts.append(tokenizer.decode(prompt, skip_special_tokens=True))
        references.append(tokenizer.decode(target, skip_special_tokens=True))
    width = max(max(len(value), 1) for value in prompt_ids)
    prompt_table = torch.full((len(prompt_ids), width), tokenizer.pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros_like(prompt_table)
    for row, value in enumerate(prompt_ids):
        prompt_table[row, :len(value)] = torch.tensor(value, dtype=torch.long)
        prompt_mask[row, :len(value)] = 1
    perturbed, retained, counts = sanitizer.sanitize_token_ids(
        prompt_table, prompt_mask, model.get_input_embeddings().weight.detach(), seed=args.seed
    )
    private_prompts = [perturbed[row][retained[row]].tolist() for row in range(len(prompt_ids))]
    remote_ids = generate_batches(model, tokenizer, private_prompts, args.batch_size, args.max_new_tokens, torch.device(args.device))
    remote_texts = [tokenizer.decode(item, skip_special_tokens=True) for item in remote_ids]
    prompts = [extraction_prompt(raw_prompt, remote_text) for raw_prompt, remote_text in zip(raw_prompts, remote_texts)]
    extractor_inputs = [tokenizer(item, add_special_tokens=False)["input_ids"][-args.max_extractor_prompt_tokens:] for item in prompts]
    final_ids = generate_batches(model, tokenizer, extractor_inputs, args.batch_size, args.max_new_tokens, torch.device(args.device))
    final_texts = [tokenizer.decode(item, skip_special_tokens=True) for item in final_ids]
    final_rouge = rouge_from_texts(final_texts, references)
    remote_rouge = rouge_from_texts(remote_texts, references)
    conditional_ppl = extractor_ppl(
        model, tokenizer, prompts, target_ids, batch_size=args.batch_size,
        max_prompt_tokens=args.max_extractor_prompt_tokens, device=torch.device(args.device)
    ) if args.run_ppl else float("nan")
    if args.save_examples_jsonl:
        path = Path(args.save_examples_jsonl); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in range(min(8, len(final_texts))):
                handle.write(json.dumps({"raw_prompt": raw_prompts[row], "remote": remote_texts[row], "final": final_texts[row], "reference": references[row]}, ensure_ascii=False) + "\n")
    return {
        "method": "InferDPT-style",
        "adapter_status": "two_stage_Qwen3_local_remote_harness",
        "epsilon_semantics": "RANTEXT_per_token_random_adjacency_epsilon_LDP",
        "representation": "Qwen3_BPE_input_token_RANTEXT__same_local_Qwen3_for_remote_and_extractor",
        "paper_component": "RANTEXT_perturbed_remote_prompt_plus_local_response_extractor",
        "omitted_paper_component": "published_GPT4_remote_model_and_text_embedding_ada_002_space",
        "ppl": float("nan"), "rouge_l": final_rouge, "remote_rouge_l": remote_rouge,
        "extractor_conditional_ppl": conditional_ppl,
        "ppl_comparability": "not_directly_comparable__extractor_retains_raw_prompt_locally",
        "mapping_cache": str(cache), "realized_replacements": int(counts["realized_replacements"]),
        "realized_replacement_fraction": counts["realized_replacement_fraction"],
        "details_json": json.dumps({"candidate_vocab_size": len(candidate_ids), **counts}, sort_keys=True),
    }


def prepare_method(args, raw, tokenizer, model=None) -> None:
    if args.method == "santext":
        vocabulary, cache = build_santext_vocabulary(args, raw, tokenizer)
        print(f"[Prepared] SanText vocabulary: {cache}, |V|={len(vocabulary.source_token_ids)}")
    elif args.method == "inferdpt":
        ids, cache = build_inferdpt_vocab(args, raw, tokenizer)
        print(f"[Prepared] InferDPT vocabulary: {cache}, |V|={len(ids)}")
    else:
        print(f"[Prepared] {args.method} has no epsilon-independent public mapping to build.")


def write_result(path: Path, row: Mapping[str, object], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing result: {path}. Use --overwrite true after checking it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        values = dict(row)
        for key, value in list(values.items()):
            if isinstance(value, float):
                values[key] = f"{value:.8f}" if math.isfinite(value) else "nan"
        writer.writerow({field: values.get(field, "") for field in RESULT_FIELDS})


def main() -> None:
    args = get_parser().parse_args()
    require_paths(args)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This Qwen3-14B benchmark requires a CUDA --device; refusing CPU fallback.")
    tokenizer, model = load_qwen3_model(args.model_path, torch.device(args.device), args.dtype)
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError(f"Expected Qwen3 model, found model_type={getattr(model.config, 'model_type', None)!r}")
    raw = load_from_disk(args.dataset_path)
    if not hasattr(raw, "keys") or "test" not in raw or "train" not in raw:
        raise ValueError("CNN/DailyMail dataset must be a DatasetDict with at least train and test splits.")
    if args.prepare_only:
        prepare_method(args, raw, tokenizer, model)
        return
    pack = build_cnndm_eval_pack(
        args.dataset_path, tokenizer, args.max_samples, args.max_prompt_length, args.max_ref_tokens,
        args.max_prompt_length + args.max_ref_tokens, args.shuffle_samples, args.seed,
        sample_start=args.sample_start,
    )
    clean = TokenDataset(pack.eval_texts, tokenizer, args.max_prompt_length + args.max_ref_tokens)
    if args.method == "plaintext":
        result = evaluate_plaintext(args, model, tokenizer, clean)
    elif args.method in {"cape", "custext", "santext"}:
        result = evaluate_embedding_method(args, model, tokenizer, clean, raw, args.method)
    elif args.method in {"dp_forward", "textobfuscator"}:
        result = evaluate_split_method(args, model, tokenizer, clean, args.method)
    else:
        result = evaluate_inferdpt(args, model, tokenizer, clean, raw)
    row = {**method_common(args, clean), **result}
    path = Path(args.output_csv)
    write_result(path, row, overwrite=args.overwrite)
    print(f"[Result] method={row['method']} eps={args.epsilon:g} PPL={row.get('ppl')} ROUGE-L={row.get('rouge_l')}")
    print(f"[Saved] {path}")


if __name__ == "__main__":
    main()
