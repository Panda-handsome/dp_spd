#!/usr/bin/env python
"""CAPE-style frozen-LLaMA generation evaluation on WikiText-2.

The original CAPE work is a BERT/Trustpilot classification method: it
mean-pools an entire contextual BERT review embedding, perturbs it with
Laplace(1 / epsilon), and *trains* target and private-attribute classifiers
with an adversarial objective.  It neither has an autoregressive generation
setting nor provides WikiText-2 private labels.

This runner is therefore intentionally a **DP-only LLaMA sequence adapter**:
every active LLaMA input embedding is min--max normalised within its text and
perturbed with the paper's Laplace scale.  Clean labels plus the project's
unchanged DP-SPD PPL/ROUGE functions make its utility measurement mechanically
consistent with Figure 3, but it is not the paper's full CAPE method and its
epsilon must not be presented as the same end-to-end guarantee.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from datasets import load_from_disk
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.cape.sanitizer import CAPEConfig, normalize_and_perturb_sequence
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
    parser = argparse.ArgumentParser(description="CAPE DP-only LLaMA embedding adapter for WikiText-2 generation.")
    parser.add_argument("--model_path", type=str, default="models/Llama-2-7b-chat-hf")
    parser.add_argument("--dataset_name", choices=DATASET_NAMES, default="wikitext")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--require_cuda", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument(
        "--selection_protocol",
        choices=["current_dpspd", "legacy_dp_generate"],
        default="current_dpspd",
        help="Use DP-SPD's current word-count filtering or the old dp_generate length filter.",
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
    parser.add_argument("--output_csv", type=str, default="outputs/baselines/cape/wikitext_figure3.csv")
    parser.add_argument("--save_examples_jsonl", type=str, default=None)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def resolve_local_path(value: str) -> str:
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


def collect_wikitext_lines(dataset, max_samples: int, selection_protocol: str) -> List[str]:
    texts: List[str] = []
    for item in tqdm(dataset, desc=f"Formatting WikiText-2 ({selection_protocol})"):
        raw_text = str(item.get("text", ""))
        if selection_protocol == "legacy_dp_generate":
            text, keep = raw_text, len(raw_text) > 50
        else:
            text = raw_text.strip()
            keep = len(text.split()) > 8
        if keep:
            texts.append(text)
        if max_samples > 0 and len(texts) >= max_samples:
            break
    if not texts:
        raise ValueError("No WikiText-2 lines passed the selected evaluation filter")
    return texts


class CAPESequenceDataset(Dataset):
    """Clean causal labels and deterministic CAPE-noised LLaMA embeddings."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        embedding_weight: torch.Tensor,
        config: CAPEConfig,
        *,
        max_length: int,
        seed: int,
    ) -> None:
        encoded = tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"].to(torch.long).contiguous()
        self.attention_mask = encoded["attention_mask"].to(torch.long).contiguous()
        self.embedding_weight = embedding_weight.detach().float().cpu().contiguous()
        self.config = config
        self.seed = int(seed)

        active_tokens = int(self.attention_mask.sum().item())
        self.stats: Dict[str, float] = {
            "active_tokens": float(active_tokens),
            "privatized_coordinates": float(active_tokens * self.embedding_weight.shape[1]),
            "laplace_scale": 1.0 / config.epsilon,
        }
        ranges: List[float] = []
        with torch.no_grad():
            for token_ids, mask in zip(self.input_ids, self.attention_mask):
                active_embeddings = F.embedding(token_ids[mask.bool()], self.embedding_weight)
                if active_embeddings.numel():
                    ranges.append(float((active_embeddings.max() - active_embeddings.min()).item()))
        self.stats["normalization_range_mean"] = float(np.mean(ranges)) if ranges else 0.0
        self.stats["normalization_range_min"] = float(np.min(ranges)) if ranges else 0.0
        self.stats["normalization_range_max"] = float(np.max(ranges)) if ranges else 0.0

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        token_ids = self.input_ids[index]
        mask = self.attention_mask[index]
        with torch.no_grad():
            clean_embeddings = F.embedding(token_ids, self.embedding_weight)
            noisy_embeddings, _ = normalize_and_perturb_sequence(
                clean_embeddings,
                mask,
                self.config,
                # A large stride prevents overlap between each sample's PRNG stream.
                seed=self.seed + 1_000_003 * int(index),
            )
        return {
            "input_ids": token_ids,
            "attention_mask": mask,
            "noisy_embeddings": noisy_embeddings,
        }

    def examples(self, tokenizer, max_examples: int = 4) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for index in range(min(max_examples, len(self))):
            active = self.attention_mask[index].bool()
            clean_embeddings = F.embedding(self.input_ids[index], self.embedding_weight)
            noisy_embeddings, stats = normalize_and_perturb_sequence(
                clean_embeddings, self.attention_mask[index], self.config,
                seed=self.seed + 1_000_003 * index,
            )
            records.append({
                "clean_text": tokenizer.decode(self.input_ids[index][active], skip_special_tokens=False),
                "active_tokens": int(active.sum().item()),
                "noisy_embedding_mean": float(noisy_embeddings[active].mean().item()),
                "noisy_embedding_std": float(noisy_embeddings[active].std(unbiased=False).item()),
                "normalization_range": stats["normalization_range"],
            })
        return records


RESULT_FIELDS = [
    "method", "paper_component", "omitted_paper_component", "representation", "dataset", "model", "epsilon",
    "epsilon_semantics", "selection_protocol", "eval_samples", "max_length", "rouge_context_len", "max_new_tokens",
    "rouge_use_stemmer", "ppl_protocol", "rouge_protocol", "active_tokens", "privatized_coordinates", "laplace_scale",
    "normalization_range_mean", "normalization_range_min", "normalization_range_max", "laplace_endpoint_guard", "ppl",
    "rouge_l", "seed",
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
    config = CAPEConfig(epsilon=args.epsilon, seed=args.seed)
    config.validate()
    if args.max_length < 2:
        raise ValueError("max_length must be at least two")
    if not 1 <= args.rouge_context_len < args.max_length:
        raise ValueError("rouge_context_len must be in [1, max_length)")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if args.dry_run:
        print("[Dry Run]")
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        print("This is the DP-only CAPE LLaMA sequence adapter, not full adversarial CAPE training.")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type != "cuda" and args.device.startswith("cuda"):
        message = f"CUDA device {args.device} is unavailable."
        if args.require_cuda:
            raise RuntimeError(message + " Refusing CPU fallback for a GPU benchmark.")
        print(f"[Warning] {message} Using CPU because --require_cuda false was supplied.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    raw = load_dataset_bundle(args.dataset_path)
    if not hasattr(raw, "keys") or args.eval_split not in raw:
        raise ValueError(f"Expected a DatasetDict containing split {args.eval_split!r}")
    texts = format_generation_texts(
        args.dataset_name, raw[args.eval_split], args.max_samples, selection_protocol=args.selection_protocol
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=args.local_files_only, torch_dtype=resolve_dtype(args.dtype)
    ).to(device)
    model.eval()
    dataset = CAPESequenceDataset(
        texts, tokenizer, model.get_input_embeddings().weight, config,
        max_length=args.max_length, seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    ppl = calculate_ppl(model, loader, device) if args.run_ppl else float("nan")
    rouge_l = (
        calculate_rouge(
            model, tokenizer, loader, device, args.rouge_context_len, args.max_new_tokens,
            use_stemmer=args.rouge_use_stemmer,
        )
        if args.run_rouge
        else float("nan")
    )
    if args.save_examples_jsonl:
        examples_path = Path(args.save_examples_jsonl)
        examples_path.parent.mkdir(parents=True, exist_ok=True)
        with examples_path.open("w", encoding="utf-8") as handle:
            for record in dataset.examples(tokenizer):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[CAPE-style] Saved examples: {examples_path}")

    row = {
        "method": "CAPE-DP-only-style",
        "paper_component": "min_max_normalization_to_[0,1]_plus_iid_Laplace(1/epsilon)",
        "omitted_paper_component": "private_attribute_adversarial_training__generation_datasets_lack_paper_gender_location_birth_year_labels",
        "representation": "full_sequence_llama_input_embedding_adapter__not_mean_pooled_contextual_BERT",
        "dataset": DATASET_LABELS[args.dataset_name],
        "model": args.model_path,
        "epsilon": args.epsilon,
        "epsilon_semantics": "paper_Laplace_scale_1_over_epsilon__not_an_end_to_end_DP_claim_for_sequence_adapter",
        "selection_protocol": dataset_protocol_label(args.dataset_name, args.selection_protocol),
        "eval_samples": len(dataset),
        "max_length": args.max_length,
        "rouge_context_len": args.rouge_context_len,
        "max_new_tokens": args.max_new_tokens,
        "rouge_use_stemmer": args.rouge_use_stemmer,
        "ppl_protocol": "DP-SPD_full_sequence_clean_labels_after_CAPE_style_embedding_noise",
        "rouge_protocol": "DP-SPD_noised_first_context_embeddings_to_clean_next_tokens",
        "active_tokens": int(dataset.stats["active_tokens"]),
        "privatized_coordinates": int(dataset.stats["privatized_coordinates"]),
        "laplace_scale": f"{dataset.stats['laplace_scale']:.6f}",
        "normalization_range_mean": f"{dataset.stats['normalization_range_mean']:.6f}",
        "normalization_range_min": f"{dataset.stats['normalization_range_min']:.6f}",
        "normalization_range_max": f"{dataset.stats['normalization_range_max']:.6f}",
        "laplace_endpoint_guard": "float32_inverse_cdf_endpoint_only",
        "ppl": f"{ppl:.6f}",
        "rouge_l": f"{rouge_l:.6f}",
        "seed": args.seed,
    }
    append_row(args.output_csv, row)
    print(
        f"[Result] epsilon={args.epsilon:g} ppl={ppl:.6f} rouge_l={rouge_l:.6f} "
        f"laplace_scale={row['laplace_scale']} active_tokens={row['active_tokens']}"
    )
    print(f"[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()
