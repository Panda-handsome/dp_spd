#!/usr/bin/env python
"""Full-sequence CusText-style evaluation aligned with DP-SPD Figure 3.

Chen et al.'s CusText is a word-level mechanism.  A replacement word can have
a different LLaMA BPE length, so it cannot be paired position-by-position with
the clean labels used by the DP-SPD Figure-3 PPL calculation.  This runner is
therefore an explicitly labelled *token-level CusText-style adaptation*: it
uses CusText's conservative candidate groups and exponential mechanism over
the LLaMA input-embedding vocabulary, and replaces every active non-special
token in the full sequence.  The resulting batches have exactly the same
``input_ids`` (clean targets), ``attention_mask`` and full-length
``noisy_embeddings`` interface as the DP-SPD evaluator.

Do not label its rows as the paper's original word-level CusText result.  The
paper-faithful textual runner remains ``run_generation.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.custext.sanitizer import CusTextConfig, CusTextSanitizer
from baseline.generation_data import (
    DATASET_LABELS,
    DATASET_NAMES,
    dataset_protocol_label,
    default_split,
    format_generation_texts,
    load_dataset_bundle,
    resolve_local_dataset_path,
)
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
    parser = argparse.ArgumentParser(
        description="Full-sequence token-level CusText-style generation evaluation for DP-SPD Figure 3."
    )
    parser.add_argument("--model_path", type=str, default="models/Llama-2-7b-chat-hf")
    parser.add_argument("--dataset_name", choices=DATASET_NAMES, default="wikitext")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--epsilon", type=float, required=True, help="Per-token CusText exponential-mechanism budget.")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument(
        "--mapping_strategy",
        choices=["release_compatible", "conservative", "aggressive"],
        default="release_compatible",
        help="Use release_compatible to mirror the authors' public-code grouping behavior.",
    )
    parser.add_argument("--tail_policy", choices=["merge_last", "allow_singleton"], default="merge_last")
    parser.add_argument("--mapping_device", type=str, default=None)
    parser.add_argument("--mapping_cache", type=str, default=None)
    parser.add_argument("--force_rebuild_mapping", type=str2bool, default=False)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument(
        "--selection_protocol",
        choices=["current_dpspd", "legacy_dp_generate"],
        default="current_dpspd",
        help=(
            "current_dpspd keeps the current Figure-3 word-count filter; "
            "legacy_dp_generate exactly reproduces the old dp_generate/results.py len(text)>50 filter."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--rouge_context_len", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--run_ppl", type=str2bool, default=True)
    parser.add_argument("--run_rouge", type=str2bool, default=True)
    parser.add_argument("--rouge_use_stemmer", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default="outputs/baselines/custext/wikitext_figure3.csv")
    parser.add_argument("--save_examples_jsonl", type=str, default=None)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def resolve_local_path(value: str) -> str:
    """Allow calls from the workspace root or ``dp_spd``."""

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


def collect_wikitext_lines(dataset, max_samples: int, selection_protocol: str) -> List[str]:
    texts: List[str] = []
    for item in tqdm(dataset, desc=f"Formatting WikiText-2 ({selection_protocol})"):
        raw_text = str(item.get("text", ""))
        if selection_protocol == "legacy_dp_generate":
            # Exact old rule from dp_generate/results.py: no stripping and a
            # character-length (not token-length) threshold.
            text = raw_text
            keep = len(text) > 50
        else:
            text = raw_text.strip()
            keep = len(text.split()) > 8
        if keep:
            texts.append(text)
        if max_samples > 0 and len(texts) >= max_samples:
            break
    if not texts:
        raise ValueError("No WikiText-2 lines with more than eight words were found.")
    return texts


def default_mapping_cache(args) -> Path:
    model_name = Path(args.model_path).name.replace(" ", "_")
    filename = (
        f"wikitext_figure3_{model_name}_model_tokens_{args.metric}_{args.mapping_strategy}_"
        f"eps{args.epsilon:g}_k{args.top_k}.json"
    )
    return Path("outputs/baselines/custext/mappings") / filename


def build_or_load_sanitizer(args, model, tokenizer) -> Tuple[CusTextSanitizer, Path]:
    config = CusTextConfig(
        epsilon=args.epsilon,
        top_k=args.top_k,
        metric=args.metric,
        mapping_strategy=args.mapping_strategy,
        tail_policy=args.tail_policy,
        seed=args.seed,
    )
    cache_path = Path(args.mapping_cache) if args.mapping_cache else default_mapping_cache(args)
    if cache_path.is_file() and not args.force_rebuild_mapping:
        sanitizer = CusTextSanitizer.load(cache_path)
        if sanitizer.kind != "model_token" or sanitizer.config != config:
            raise ValueError(
                f"Cached mapping {cache_path} has a different CusText configuration. "
                "Pass --force_rebuild_mapping true or use a different --mapping_cache."
            )
        if sanitizer.metadata.get("model_path") != args.model_path:
            raise ValueError(f"Cached mapping {cache_path} was built for another --model_path.")
        if sanitizer.metadata.get("tokenizer_vocab_size") != int(len(tokenizer)):
            raise ValueError(f"Cached mapping {cache_path} was built for another tokenizer vocabulary.")
        print(f"[CusText-style] Reusing mapping: {cache_path}")
        return sanitizer, cache_path

    sanitizer = CusTextSanitizer.from_model_token_embeddings(
        model.get_input_embeddings().weight,
        tokenizer.all_special_ids,
        config,
        device=args.mapping_device or args.device,
        metadata={
            "model_path": args.model_path,
            "tokenizer_vocab_size": int(len(tokenizer)),
            "protocol": "figure3_full_sequence_token_adapter_v1",
            "representation": "llama_bpe_token_embedding",
        },
    )
    sanitizer.save(cache_path)
    print(f"[CusText-style] Saved mapping: {cache_path}")
    return sanitizer, cache_path


class FullSequenceCusTextDataset(Dataset):
    """Clean labels plus full-length CusText-style replacement embeddings."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        sanitizer: CusTextSanitizer,
        embedding_layer: torch.nn.Module,
        *,
        max_length: int,
        seed: int,
    ) -> None:
        encoded = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"].to(torch.long).contiguous()
        self.attention_mask = encoded["attention_mask"].to(torch.long).contiguous()
        self.sanitized_ids, raw_counts = sanitizer.sanitize_token_ids(
            self.input_ids,
            self.attention_mask,
            rng=np.random.default_rng(seed),
        )
        self.sanitized_ids = self.sanitized_ids.to(torch.long).contiguous()

        active = self.attention_mask.bool()
        special_ids = {int(item) for item in tokenizer.all_special_ids}
        special_mask = torch.zeros_like(active)
        for token_id in special_ids:
            special_mask |= self.input_ids.eq(token_id)
        special_mask &= active
        eligible_mask = active & ~special_mask
        active_token_ids = {int(item) for item in self.input_ids[eligible_mask].unique().tolist()}
        missing_ids = sorted(active_token_ids.difference(sanitizer.distributions))
        if missing_ids:
            raise RuntimeError(
                "The full-sequence adapter left non-special tokens unmapped; "
                f"first missing IDs: {missing_ids[:8]}"
            )

        eligible = int(eligible_mask.sum().item())
        mapped = int(raw_counts.get("mapped", 0))
        if mapped != eligible:
            raise RuntimeError(
                f"Expected every eligible token to be sampled, but mapped={mapped} and eligible={eligible}."
            )
        replacement_mask = eligible_mask & self.sanitized_ids.ne(self.input_ids)
        self.counts: Dict[str, int] = dict(raw_counts)
        self.counts.update(
            {
                "active_tokens": int(active.sum().item()),
                "eligible_non_special_tokens": eligible,
                "special_tokens_preserved": int(special_mask.sum().item()),
                "realized_replacements": int(replacement_mask.sum().item()),
            }
        )

        # Keep the table on CPU; each batch is transferred by calculate_ppl /
        # calculate_rouge and converted to the loaded model's dtype there.
        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            embedding_layer.weight.detach().float().cpu(), freeze=True
        )

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        with torch.no_grad():
            replacement_embeddings = self.embedding_layer(self.sanitized_ids[index])
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "noisy_embeddings": replacement_embeddings,
        }

    def examples(self, tokenizer, max_examples: int = 8) -> List[Dict[str, str]]:
        records: List[Dict[str, str]] = []
        for index in range(min(max_examples, len(self))):
            active = self.attention_mask[index].bool()
            records.append(
                {
                    "clean_text": tokenizer.decode(self.input_ids[index][active], skip_special_tokens=False),
                    "sanitized_text": tokenizer.decode(self.sanitized_ids[index][active], skip_special_tokens=False),
                }
            )
        return records


RESULT_FIELDS = [
    "method",
    "representation",
    "dataset",
    "model",
    "epsilon",
    "epsilon_semantics",
    "top_k",
    "metric",
    "mapping_strategy",
    "mapping_cache",
    "selection_protocol",
    "eval_samples",
    "max_length",
    "rouge_context_len",
    "max_new_tokens",
    "rouge_use_stemmer",
    "ppl_protocol",
    "rouge_protocol",
    "mapped_tokens",
    "eligible_non_special_tokens",
    "mapped_fraction",
    "realized_replacements",
    "realized_replacement_fraction",
    "special_tokens_preserved",
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
    args.dataset_path = resolve_local_dataset_path(args.dataset_path, REPO_ROOT, args.dataset_name)
    args.eval_split = args.eval_split or default_split(args.dataset_name)
    args.mapping_device = args.mapping_device or args.device
    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if args.max_length < 2:
        raise ValueError("max_length must be at least two")
    if not 1 <= args.rouge_context_len < args.max_length:
        raise ValueError("rouge_context_len must be in [1, max_length)")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if args.dry_run:
        print("[Dry Run]")
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        print("This is the full-sequence LLaMA-token CusText-style adapter used for Figure 3.")
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
        torch_dtype=resolve_dtype(args.dtype),
    ).to(device)
    model.eval()

    raw = load_dataset_bundle(args.dataset_path)
    if args.eval_split not in raw:
        raise KeyError(f"Evaluation split {args.eval_split!r} is not present in {args.dataset_path}")
    texts = format_generation_texts(
        args.dataset_name, raw[args.eval_split], args.max_samples, selection_protocol=args.selection_protocol
    )
    sanitizer, mapping_path = build_or_load_sanitizer(args, model, tokenizer)
    dataset = FullSequenceCusTextDataset(
        texts,
        tokenizer,
        sanitizer,
        model.get_input_embeddings(),
        max_length=args.max_length,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    ppl = calculate_ppl(model, loader, device) if args.run_ppl else float("nan")
    rouge_l = (
        calculate_rouge(
            model,
            tokenizer,
            loader,
            device,
            args.rouge_context_len,
            args.max_new_tokens,
            use_stemmer=args.rouge_use_stemmer,
        )
        if args.run_rouge
        else float("nan")
    )

    if args.save_examples_jsonl:
        examples_path = Path(args.save_examples_jsonl)
        examples_path.parent.mkdir(parents=True, exist_ok=True)
        with examples_path.open("w", encoding="utf-8") as handle:
            for example in dataset.examples(tokenizer):
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"[CusText-style] Saved examples: {examples_path}")

    eligible = int(dataset.counts["eligible_non_special_tokens"])
    mapped = int(dataset.counts.get("mapped", 0))
    replacements = int(dataset.counts["realized_replacements"])
    row = {
        "method": "CusText-style",
        "representation": "full_sequence_llama_bpe_token_adapter",
        "dataset": DATASET_LABELS[args.dataset_name],
        "model": args.model_path,
        "epsilon": args.epsilon,
        "epsilon_semantics": "per_token_exponential_mechanism",
        "top_k": args.top_k,
        "metric": args.metric,
        "mapping_strategy": args.mapping_strategy,
        "mapping_cache": str(mapping_path),
        "selection_protocol": dataset_protocol_label(args.dataset_name, args.selection_protocol),
        "eval_samples": len(dataset),
        "max_length": args.max_length,
        "rouge_context_len": args.rouge_context_len,
        "max_new_tokens": args.max_new_tokens,
        "rouge_use_stemmer": args.rouge_use_stemmer,
        "ppl_protocol": "DP-SPD_full_sequence_clean_labels_after_token_replacement_embeddings",
        "rouge_protocol": "DP-SPD_replaced_first_context_embeddings_to_clean_next_tokens",
        "mapped_tokens": mapped,
        "eligible_non_special_tokens": eligible,
        "mapped_fraction": f"{mapped / eligible:.6f}" if eligible else "",
        "realized_replacements": replacements,
        "realized_replacement_fraction": f"{replacements / eligible:.6f}" if eligible else "",
        "special_tokens_preserved": int(dataset.counts["special_tokens_preserved"]),
        "ppl": f"{ppl:.6f}",
        "rouge_l": f"{rouge_l:.6f}",
        "seed": args.seed,
    }
    append_row(args.output_csv, row)
    print(
        f"[Result] epsilon={args.epsilon:g} ppl={ppl:.6f} rouge_l={rouge_l:.6f} "
        f"mapped_fraction={row['mapped_fraction']} replacement_fraction={row['realized_replacement_fraction']}"
    )
    print(f"[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()
