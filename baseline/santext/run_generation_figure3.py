#!/usr/bin/env python
"""SanText/SanText+ evaluation aligned with the DP-SPD Figure-3 protocol.

Yue et al. (2021) sanitize *word* tokens and evaluate BERT classification and
MLM tasks.  A word replacement can change LLaMA BPE length, which cannot be
paired position-by-position with the clean labels of this project's Figure-3
PPL calculation.  This runner is therefore explicitly a **LLaMA-token
SanText-style adapter**: it applies the paper's Euclidean exponential
mechanism directly over LLaMA input-token embeddings, preserves clean labels,
and calls the unchanged DP-SPD PPL/ROUGE-L evaluators.

Do not label these rows as the original word-level/BERT SanText experiments.
The adaptation is provided only for a fixed-model comparison with DP-SPD.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.santext.sanitizer import (
    SanTextConfig,
    SanTextTokenSanitizer,
    SanTextVocabulary,
    build_vocabulary,
)
from baseline.generation_data import (
    DATASET_LABELS,
    DATASET_NAMES,
    dataset_protocol_label,
    default_split,
    format_generation_record,
    format_generation_texts,
    load_dataset_bundle as load_generation_dataset_bundle,
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
        description="Full-sequence LLaMA-token SanText-style evaluation for DP-SPD Figure 3."
    )
    parser.add_argument("--model_path", type=str, default="models/Llama-2-7b-chat-hf")
    parser.add_argument("--dataset_name", choices=DATASET_NAMES, default="wikitext")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument(
        "--mapping_splits",
        type=str,
        default="train,validation",
        help="Comma-separated public reference splits used to build V; paper/release analogue is train/dev.",
    )
    parser.add_argument(
        "--mapping_max_samples",
        type=int,
        default=0,
        help="Limit reference rows only for a smoke test; zero uses every requested mapping split.",
    )
    parser.add_argument("--mapping_batch_size", type=int, default=128)
    parser.add_argument("--mapping_cache", type=str, default=None)
    parser.add_argument("--force_rebuild_mapping", type=str2bool, default=False)
    parser.add_argument(
        "--build_mapping_only",
        type=str2bool,
        default=False,
        help="Build/reuse the public reference vocabulary then exit; used once before a GPU sweep.",
    )

    parser.add_argument("--variant", choices=["santext", "santext_plus"], default="santext")
    parser.add_argument("--epsilon", type=float, required=True, help="Per-token SanText exponential-mechanism budget.")
    parser.add_argument(
        "--sensitive_fraction",
        type=float,
        default=0.9,
        help="SanText+ w: least-frequent fraction treated as sensitive (paper Table-1 value: 0.9).",
    )
    parser.add_argument(
        "--non_sensitive_sanitize_probability",
        type=float,
        default=0.3,
        help="SanText+ p: probability to sanitize a non-sensitive token (paper Table-1 value: 0.3).",
    )
    parser.add_argument("--source_chunk_size", type=int, default=96)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--require_cuda",
        type=str2bool,
        default=True,
        help="Fail rather than silently running a requested CUDA benchmark on CPU.",
    )
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument(
        "--selection_protocol",
        choices=["current_dpspd", "legacy_dp_generate"],
        default="current_dpspd",
        help="Use current DP-SPD word-count filtering or the exact older dp_generate len(text)>50 rule.",
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
    parser.add_argument("--output_csv", type=str, default="outputs/baselines/santext/wikitext_figure3.csv")
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


def load_dataset_bundle(dataset_path: str):
    raw = load_from_disk(dataset_path)
    if not hasattr(raw, "keys"):
        raise ValueError("SanText mapping requires a DatasetDict with train/validation/test splits.")
    return raw


def collect_wikitext_lines(dataset, max_samples: int, selection_protocol: str) -> List[str]:
    texts: List[str] = []
    for item in tqdm(dataset, desc=f"Formatting WikiText-2 ({selection_protocol})"):
        raw_text = str(item.get("text", ""))
        if selection_protocol == "legacy_dp_generate":
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
        raise ValueError("No WikiText-2 lines passed the selected evaluation filter.")
    return texts


def parse_splits(value: str) -> List[str]:
    splits = [part.strip() for part in value.split(",") if part.strip()]
    if not splits:
        raise ValueError("mapping_splits must contain at least one split")
    return splits


def count_mapping_tokens(
    raw,
    tokenizer,
    *,
    dataset_name: str,
    selection_protocol: str,
    splits: Sequence[str],
    max_samples: int,
    batch_size: int,
) -> Counter:
    """Build V from all public train/validation text, as the release uses train/dev."""

    if batch_size < 1:
        raise ValueError("mapping_batch_size must be positive")
    special_ids = set(int(item) for item in tokenizer.all_special_ids)
    counts: Counter = Counter()
    for split in splits:
        if split not in raw:
            raise KeyError(f"Mapping split {split!r} is not in {list(raw.keys())}")
        dataset = raw[split]
        limit = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
        for start in tqdm(range(0, limit, batch_size), desc=f"Building SanText V ({split})"):
            # datasets.Dataset slicing returns a column dictionary, unlike
            # integer indexing which returns one row dictionary.
            columns = dataset[start : min(start + batch_size, limit)]
            row_count = len(next(iter(columns.values()), []))
            batch = []
            for row in range(row_count):
                item = {key: values[row] for key, values in columns.items()}
                text = format_generation_record(dataset_name, item, selection_protocol=selection_protocol)
                if text is not None:
                    batch.append(text)
            if not batch:
                continue
            encoded = tokenizer(batch, add_special_tokens=True, truncation=False, padding=False)
            for ids in encoded["input_ids"]:
                counts.update(int(token_id) for token_id in ids if int(token_id) not in special_ids)
    if not counts:
        raise ValueError("No non-special BPE tokens were found in mapping splits")
    return counts


def default_mapping_cache(args) -> Path:
    model_name = Path(args.model_path).name.replace(" ", "_")
    split_tag = "-".join(parse_splits(args.mapping_splits))
    dataset_tag = "wikitext" if args.dataset_name == "wikitext" else args.dataset_name
    filename = (
        f"{dataset_tag}_{model_name}_llama_bpe_{split_tag}_w{args.sensitive_fraction:g}_"
        f"n{args.mapping_max_samples or 'all'}_{args.selection_protocol}_v2.json"
    )
    return Path("outputs/baselines/santext/mappings") / filename


def _mapping_metadata(args, tokenizer) -> Dict[str, object]:
    metadata = {
        "schema": "santext_llama_token_vocab_v2",
        "model_path": args.model_path,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "mapping_splits": parse_splits(args.mapping_splits),
        "mapping_max_samples": int(args.mapping_max_samples),
        "sensitive_fraction": float(args.sensitive_fraction),
        "frequency_rule": "Counter.most_common_then_release_words[-int(w*len(words))-1:]",
        "representation": "llama_bpe_token_embedding",
        "selection_protocol": args.selection_protocol,
    }
    # Retain compatibility with the pre-existing WikiText cache, while making
    # a public instruction-corpus cache impossible to confuse with it.
    if args.dataset_name != "wikitext":
        metadata["dataset_name"] = args.dataset_name
    return metadata


def build_or_load_vocabulary(args, raw, tokenizer) -> tuple[SanTextVocabulary, Path]:
    cache_path = Path(args.mapping_cache) if args.mapping_cache else default_mapping_cache(args)
    expected = _mapping_metadata(args, tokenizer)
    if cache_path.is_file() and not args.force_rebuild_mapping:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        vocabulary = SanTextVocabulary.from_dict(payload)
        metadata = dict(vocabulary.metadata)
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            raise ValueError(
                f"Cached SanText vocabulary {cache_path} does not match this run: {mismatches}. "
                "Pass --force_rebuild_mapping true or choose --mapping_cache."
            )
        print(f"[SanText-style] Reusing vocabulary: {cache_path}")
        return vocabulary, cache_path

    counts = count_mapping_tokens(
        raw,
        tokenizer,
        dataset_name=args.dataset_name,
        selection_protocol=args.selection_protocol,
        splits=parse_splits(args.mapping_splits),
        max_samples=args.mapping_max_samples,
        batch_size=args.mapping_batch_size,
    )
    config = SanTextConfig(
        variant=args.variant,
        epsilon=args.epsilon,
        sensitive_fraction=args.sensitive_fraction,
        non_sensitive_sanitize_probability=args.non_sensitive_sanitize_probability,
        source_chunk_size=args.source_chunk_size,
        seed=args.seed,
    )
    vocabulary = build_vocabulary(counts, config, metadata=expected)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(vocabulary.to_dict(), ensure_ascii=False), encoding="utf-8")
    print(
        f"[SanText-style] Saved vocabulary: {cache_path} "
        f"(|V|={len(vocabulary.source_token_ids)}, |Vs|={len(vocabulary.sensitive_token_ids)})"
    )
    return vocabulary, cache_path


class FullSequenceSanTextDataset(Dataset):
    """Clean labels plus full-length, discrete SanText replacement embeddings."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        sanitizer: SanTextTokenSanitizer,
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
        self.sanitized_ids, self.counts = sanitizer.sanitize_token_ids(
            self.input_ids,
            self.attention_mask,
            embedding_layer.weight.detach(),
            seed=seed,
        )
        eligible = int(self.counts["eligible_non_special_tokens"])
        if int(self.counts["mapped"]) != eligible:
            raise RuntimeError("SanText adapter did not process every eligible token")
        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            embedding_layer.weight.detach().float().cpu(), freeze=True
        )

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        with torch.no_grad():
            noisy_embeddings = self.embedding_layer(self.sanitized_ids[index])
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "noisy_embeddings": noisy_embeddings,
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
    "method", "variant", "representation", "dataset", "model", "epsilon", "epsilon_semantics",
    "sensitive_fraction", "non_sensitive_sanitize_probability", "mapping_cache", "mapping_splits",
    "mapping_vocab_size", "sensitive_vocab_size", "selection_protocol", "eval_samples", "max_length",
    "rouge_context_len", "max_new_tokens", "rouge_use_stemmer", "ppl_protocol", "rouge_protocol",
    "mapped_tokens", "eligible_non_special_tokens", "mapped_fraction", "in_vocab_tokens",
    "oov_tokens_uniformly_sampled", "exponential_sampled_tokens", "preserved_non_sensitive_tokens",
    "realized_replacements", "realized_replacement_fraction", "special_tokens_preserved", "ppl", "rouge_l", "seed",
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
    if args.dataset_name != "wikitext" and args.mapping_splits == "train,validation":
        args.mapping_splits = "train"
    config = SanTextConfig(
        variant=args.variant,
        epsilon=args.epsilon,
        sensitive_fraction=args.sensitive_fraction,
        non_sensitive_sanitize_probability=args.non_sensitive_sanitize_probability,
        source_chunk_size=args.source_chunk_size,
        seed=args.seed,
    )
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
        print("This is the full-sequence LLaMA-token SanText-style Figure-3 adapter.")
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
    raw = load_generation_dataset_bundle(args.dataset_path)
    if args.eval_split not in raw:
        raise KeyError(f"Evaluation split {args.eval_split!r} is not in {list(raw.keys())}")
    vocabulary, mapping_path = build_or_load_vocabulary(args, raw, tokenizer)
    if args.build_mapping_only:
        print(
            f"[SanText-style] Vocabulary ready: {mapping_path} "
            f"(|V|={len(vocabulary.source_token_ids)}, |Vs|={len(vocabulary.sensitive_token_ids)})"
        )
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
        torch_dtype=resolve_dtype(args.dtype),
    ).to(device)
    model.eval()
    texts = format_generation_texts(
        args.dataset_name, raw[args.eval_split], args.max_samples, selection_protocol=args.selection_protocol
    )
    sanitizer = SanTextTokenSanitizer(
        vocabulary,
        config,
        vocab_size=int(len(tokenizer)),
        special_token_ids=tokenizer.all_special_ids,
    )
    dataset = FullSequenceSanTextDataset(
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
            for example in dataset.examples(tokenizer):
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"[SanText-style] Saved examples: {examples_path}")

    eligible = int(dataset.counts["eligible_non_special_tokens"])
    replacements = int(dataset.counts["realized_replacements"])
    epsilon_semantics = (
        "per_token_metric_ldp_epsilon"
        if args.variant == "santext"
        else f"per_token_umldp_epsilon__epsilon0=ln(1/p)={np.log(1.0 / args.non_sensitive_sanitize_probability):.6f}"
    )
    row = {
        "method": "SanText-style",
        "variant": args.variant,
        "representation": "full_sequence_llama_bpe_token_adapter",
        "dataset": DATASET_LABELS[args.dataset_name],
        "model": args.model_path,
        "epsilon": args.epsilon,
        "epsilon_semantics": epsilon_semantics,
        "sensitive_fraction": args.sensitive_fraction if args.variant == "santext_plus" else "",
        "non_sensitive_sanitize_probability": args.non_sensitive_sanitize_probability if args.variant == "santext_plus" else "",
        "mapping_cache": str(mapping_path),
        "mapping_splits": args.mapping_splits,
        "mapping_vocab_size": len(vocabulary.source_token_ids),
        "sensitive_vocab_size": len(vocabulary.sensitive_token_ids) if args.variant == "santext_plus" else "",
        "selection_protocol": dataset_protocol_label(args.dataset_name, args.selection_protocol),
        "eval_samples": len(dataset),
        "max_length": args.max_length,
        "rouge_context_len": args.rouge_context_len,
        "max_new_tokens": args.max_new_tokens,
        "rouge_use_stemmer": args.rouge_use_stemmer,
        "ppl_protocol": "DP-SPD_full_sequence_clean_labels_after_discrete_token_sanitization",
        "rouge_protocol": "DP-SPD_sanitized_first_context_embeddings_to_clean_next_tokens",
        "mapped_tokens": int(dataset.counts["mapped"]),
        "eligible_non_special_tokens": eligible,
        "mapped_fraction": f"{int(dataset.counts['mapped']) / eligible:.6f}" if eligible else "",
        "in_vocab_tokens": int(dataset.counts["in_vocab_tokens"]),
        "oov_tokens_uniformly_sampled": int(dataset.counts["oov_tokens_uniformly_sampled"]),
        "exponential_sampled_tokens": int(dataset.counts["exponential_sampled_tokens"]),
        "preserved_non_sensitive_tokens": int(dataset.counts["preserved_non_sensitive_tokens"]),
        "realized_replacements": replacements,
        "realized_replacement_fraction": f"{replacements / eligible:.6f}" if eligible else "",
        "special_tokens_preserved": int(dataset.counts["special_tokens_preserved"]),
        "ppl": f"{ppl:.6f}",
        "rouge_l": f"{rouge_l:.6f}",
        "seed": args.seed,
    }
    append_row(args.output_csv, row)
    print(
        f"[Result] variant={args.variant} epsilon={args.epsilon:g} ppl={ppl:.6f} rouge_l={rouge_l:.6f} "
        f"mapped_fraction={row['mapped_fraction']} replacement_fraction={row['realized_replacement_fraction']}"
    )
    print(f"[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()
