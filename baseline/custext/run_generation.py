#!/usr/bin/env python
"""Evaluate original word-level CusText on LLaMA-2 generation.

CusText changes words before they reach the server.  Unlike DP-SPD, it has no
dummy ratio or dummy count; each invocation below produces one horizontal
baseline point for a chosen *per-word* epsilon.  The evaluator keeps the
continuation clean and scores it after a sanitized context, avoiding invalid
position-wise comparisons when replacement words tokenize to a different
number of LLaMA subword IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.custext.sanitizer import CusTextConfig, CusTextSanitizer, load_text_embeddings
from experiments.generation.llama_generation_eval import calculate_ppl, calculate_rouge


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Original word-level CusText on WikiText-2 generation.")
    parser.add_argument("--model_path", type=str, default="models/Llama-2-7b-chat-hf")
    parser.add_argument("--dataset_path", type=str, default="data/wikitext/wikitext-2-v1")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--mapping_split", type=str, default="train")
    parser.add_argument("--embedding_path", type=str, default="data/embeddings/counter-fitted-vectors.txt")
    parser.add_argument("--variant", choices=["custext", "custext_plus"], default="custext")
    parser.add_argument("--epsilon", type=float, required=True, help="CusText's per-word exponential-mechanism budget.")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument(
        "--mapping_strategy",
        choices=["release_compatible", "conservative", "aggressive"],
        default="release_compatible",
    )
    parser.add_argument("--mapping_device", type=str, default=None)
    parser.add_argument("--mapping_cache", type=str, default=None)
    parser.add_argument("--force_rebuild_mapping", type=str2bool, default=False)
    parser.add_argument("--max_mapping_samples", type=int, default=0, help="0 uses every mapping-split text.")
    parser.add_argument("--max_embedding_vectors", type=int, default=0, help="Smoke-test limit only; 0 loads all vectors.")
    parser.add_argument("--numeric_policy", choices=["release_compatible", "keep"], default="release_compatible")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--context_len", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--run_ppl", type=str2bool, default=True)
    parser.add_argument("--run_rouge", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default="outputs/baselines/custext/wikitext_generation.csv")
    parser.add_argument("--save_examples_jsonl", type=str, default=None)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def resolve_local_path(value: str) -> str:
    """Allow calls from either the workspace root or ``dp_spd``."""

    candidates = [Path(value), REPO_ROOT / value, REPO_ROOT.parent / value]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return value


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def resolve_dtype(name: str):
    if name == "auto":
        return None
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_split(dataset_path: str, split: str):
    raw = load_from_disk(dataset_path)
    if hasattr(raw, "keys"):
        if split not in raw:
            raise KeyError(f"Split {split!r} is not present in {dataset_path}")
        return raw[split]
    return raw


def collect_wikitext_lines(dataset, max_samples: int) -> List[str]:
    texts: List[str] = []
    for item in tqdm(dataset, desc="Formatting WikiText-2"):
        text = str(item.get("text", "")).strip()
        if len(text.split()) > 8:
            texts.append(text)
        if max_samples > 0 and len(texts) >= max_samples:
            break
    if not texts:
        raise ValueError("No WikiText-2 lines with more than eight words were found.")
    return texts


def iter_words(texts: Iterable[str]) -> Iterable[str]:
    for text in texts:
        yield from text.split()


def default_mapping_cache(args) -> Path:
    mapping_count = "all" if args.max_mapping_samples <= 0 else str(args.max_mapping_samples)
    name = (
        f"wikitext_word_vectors_{args.metric}_{args.mapping_strategy}_"
        f"eps{args.epsilon:g}_k{args.top_k}_{args.mapping_split}_n{mapping_count}.json"
    )
    return Path("outputs/baselines/custext/mappings") / name


def build_or_load_sanitizer(args, mapping_texts: Sequence[str]) -> Tuple[CusTextSanitizer, Path]:
    config = CusTextConfig(
        epsilon=args.epsilon,
        top_k=args.top_k,
        metric=args.metric,
        mapping_strategy=args.mapping_strategy,
        seed=args.seed,
    )
    cache_path = Path(args.mapping_cache) if args.mapping_cache else default_mapping_cache(args)
    if cache_path.is_file() and not args.force_rebuild_mapping:
        sanitizer = CusTextSanitizer.load(cache_path)
        if sanitizer.kind != "word" or sanitizer.config != config:
            raise ValueError(f"Cached mapping {cache_path} has a different CusText configuration.")
        if args.embedding_path and sanitizer.metadata.get("embedding_path") != str(Path(args.embedding_path).resolve()):
            raise ValueError(f"Cached mapping {cache_path} was built from a different embedding file.")
        print(f"[CusText] Reusing mapping: {cache_path}")
        return sanitizer, cache_path

    vocabulary, embeddings = load_text_embeddings(args.embedding_path, args.max_embedding_vectors)
    sanitizer = CusTextSanitizer.from_word_vectors(
        vocabulary,
        embeddings,
        iter_words(mapping_texts),
        config,
        device=args.mapping_device or args.device,
        metadata={
            "embedding_path": str(Path(args.embedding_path).resolve()),
            "dataset": "wikitext-2-v1",
            "mapping_split": args.mapping_split,
            "mapping_text_count": len(mapping_texts),
            "protocol": "clean_continuation_v1",
        },
    )
    sanitizer.save(cache_path)
    print(f"[CusText] Saved mapping: {cache_path}")
    return sanitizer, cache_path


def split_at_word_boundary(tokenizer, text: str, context_len: int) -> Optional[Tuple[str, str]]:
    """Return a word-aligned context and the remaining text.

    CusText works on whitespace words.  The context is therefore cut only at a
    word boundary while respecting the DP-SPD evaluator's BPE-token budget.
    The target begins with a space so that independently tokenizing it retains
    its continuation semantics for LLaMA's sentencepiece tokenizer.
    """

    words = text.split()
    selected: List[str] = []
    for word in words:
        candidate = " ".join([*selected, word])
        candidate_ids = tokenizer(candidate, add_special_tokens=True)["input_ids"]
        if len(candidate_ids) > context_len and selected:
            break
        selected.append(word)
        if len(candidate_ids) >= context_len:
            break
    if not selected or len(selected) >= len(words):
        return None
    return " ".join(selected), " " + " ".join(words[len(selected):])


def pad_ids(ids: Sequence[int], length: int, pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    clipped = list(ids[:length])
    result = torch.full((length,), int(pad_token_id), dtype=torch.long)
    mask = torch.zeros((length,), dtype=torch.long)
    if clipped:
        result[: len(clipped)] = torch.as_tensor(clipped, dtype=torch.long)
        mask[: len(clipped)] = 1
    return result, mask


class CusTextGenerationDataset(Dataset):
    def __init__(self, samples: Sequence[Mapping[str, torch.Tensor]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        return self.samples[index]


def prepare_eval_dataset(
    texts: Sequence[str],
    tokenizer,
    sanitizer: CusTextSanitizer,
    args,
) -> Tuple[CusTextGenerationDataset, Dict[str, int], List[Dict[str, str]]]:
    """Create batches accepted directly by DP-SPD's PPL/ROUGE functions."""

    rng = np.random.default_rng(args.seed)
    counters: Counter = Counter()
    examples: List[Dict[str, str]] = []
    samples: List[Mapping[str, torch.Tensor]] = []
    pad_token_id = int(tokenizer.pad_token_id)
    for text in tqdm(texts, desc="Preparing CusText generation examples"):
        split = split_at_word_boundary(tokenizer, text, args.context_len)
        if split is None:
            counters["skipped_no_continuation"] += 1
            continue
        clean_context, clean_target = split
        sanitized_context, sample_counts = sanitizer.sanitize_text(
            clean_context,
            rng=rng,
            preserve_stopwords=args.variant == "custext_plus",
            numeric_policy=args.numeric_policy,
        )
        counters.update(sample_counts)

        prompt_ids = tokenizer(
            sanitized_context,
            add_special_tokens=True,
            truncation=True,
            max_length=args.context_len,
        )["input_ids"]
        if len(prompt_ids) < 1:
            counters["skipped_empty_prompt"] += 1
            continue
        target_ids = tokenizer(clean_target, add_special_tokens=False)["input_ids"]
        available_target = args.max_length - len(prompt_ids)
        target_ids = target_ids[:available_target]
        if not target_ids:
            counters["skipped_empty_target"] += 1
            continue

        model_input_ids, attention_mask = pad_ids([*prompt_ids, *target_ids], args.max_length, pad_token_id)
        labels = torch.full((args.max_length,), -100, dtype=torch.long)
        loss_mask = torch.zeros((args.max_length,), dtype=torch.long)
        start = len(prompt_ids)
        labels[start:start + len(target_ids)] = torch.as_tensor(target_ids, dtype=torch.long)
        loss_mask[start:start + len(target_ids)] = 1
        generation_ids, generation_mask = pad_ids(prompt_ids, args.context_len, pad_token_id)
        reference_ids, _ = pad_ids(target_ids, args.max_new_tokens, pad_token_id)
        samples.append(
            {
                "model_input_ids": model_input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "loss_mask": loss_mask,
                "generation_input_ids": generation_ids,
                "generation_attention_mask": generation_mask,
                "reference_ids": reference_ids,
            }
        )
        examples.append(
            {
                "clean_context": clean_context,
                "sanitized_context": sanitized_context,
                "clean_target": clean_target,
            }
        )
    if not samples:
        raise ValueError("No valid generation samples were produced.")
    return CusTextGenerationDataset(samples), dict(counters), examples


RESULT_FIELDS = [
    "method",
    "dataset",
    "model",
    "epsilon",
    "epsilon_semantics",
    "variant",
    "top_k",
    "metric",
    "mapping_strategy",
    "mapping_cache",
    "mapping_split",
    "mapping_texts",
    "eval_samples",
    "context_len",
    "max_length",
    "max_new_tokens",
    "ppl_protocol",
    "rouge_protocol",
    "mapped_words",
    "total_words",
    "mapped_fraction",
    "unmapped_words",
    "stopword_preserved",
    "ppl",
    "rouge_l",
    "seed",
]


def append_row(path: str | Path, row: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def main() -> None:
    args = get_parser().parse_args()
    args.model_path = resolve_local_path(args.model_path)
    args.dataset_path = resolve_local_path(args.dataset_path)
    args.embedding_path = resolve_local_path(args.embedding_path)
    args.mapping_device = args.mapping_device or args.device
    if args.context_len < 2 or args.context_len >= args.max_length:
        raise ValueError("context_len must be at least 2 and smaller than max_length")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if args.dry_run:
        print("[Dry Run]")
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        print("epsilon is per-word CusText epsilon; output is one horizontal baseline with no r or k.")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type != "cuda" and args.device.startswith("cuda"):
        print(f"[Warning] CUDA device {args.device} is unavailable; using CPU.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    mapping_raw = load_split(args.dataset_path, args.mapping_split)
    eval_raw = load_split(args.dataset_path, args.eval_split)
    mapping_texts = collect_wikitext_lines(mapping_raw, args.max_mapping_samples)
    eval_texts = collect_wikitext_lines(eval_raw, args.max_samples)
    sanitizer, mapping_cache = build_or_load_sanitizer(args, mapping_texts)
    dataset, counters, examples = prepare_eval_dataset(eval_texts, tokenizer, sanitizer, args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
        torch_dtype=resolve_dtype(args.dtype),
    ).to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    ppl = calculate_ppl(model, loader, device) if args.run_ppl else float("nan")
    rouge_l = calculate_rouge(model, tokenizer, loader, device, args.context_len, args.max_new_tokens) if args.run_rouge else float("nan")

    if args.save_examples_jsonl:
        examples_path = Path(args.save_examples_jsonl)
        examples_path.parent.mkdir(parents=True, exist_ok=True)
        with examples_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"[CusText] Saved examples: {examples_path}")

    total_words = int(counters.get("total", 0))
    mapped_words = int(counters.get("mapped", 0))
    row = {
        "method": "CusText+" if args.variant == "custext_plus" else "CusText",
        "dataset": "wikitext-2-v1",
        "model": args.model_path,
        "epsilon": args.epsilon,
        "epsilon_semantics": "per_word_exponential_mechanism",
        "variant": args.variant,
        "top_k": args.top_k,
        "metric": args.metric,
        "mapping_strategy": args.mapping_strategy,
        "mapping_cache": str(mapping_cache),
        "mapping_split": args.mapping_split,
        "mapping_texts": len(mapping_texts),
        "eval_samples": len(dataset),
        "context_len": args.context_len,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "ppl_protocol": "clean_continuation_teacher_forcing_after_sanitized_context",
        "rouge_protocol": "clean_continuation_after_sanitized_context",
        "mapped_words": mapped_words,
        "total_words": total_words,
        "mapped_fraction": f"{mapped_words / total_words:.6f}" if total_words else "",
        "unmapped_words": int(counters.get("unmapped", 0)),
        "stopword_preserved": int(counters.get("stopword_preserved", 0)),
        "ppl": f"{ppl:.6f}",
        "rouge_l": f"{rouge_l:.6f}",
        "seed": args.seed,
    }
    append_row(args.output_csv, row)
    print(f"[Result] epsilon={args.epsilon:g} ppl={ppl:.6f} rouge_l={rouge_l:.6f}")
    print(f"[Result] mapped_fraction={row['mapped_fraction']} eval_samples={len(dataset)}")
    print(f"[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()
