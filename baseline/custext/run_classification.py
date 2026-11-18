#!/usr/bin/env python
"""Run CusText/CusText+ on GLUE SST-2 or QNLI.

Two protocols are intentionally supported:

* ``paper`` sanitizes train and evaluation text, then fine-tunes a fresh BERT
  classifier.  This is the evaluation protocol of Chen et al. (ACL 2023).
* ``inference`` sanitizes evaluation inputs only and runs an existing
  fine-tuned checkpoint.  It is the appropriate, explicitly labelled adapter
  when comparing against DP-SPD's fixed-server inference experiments.

Do not combine rows from those protocols in one utility table: they make
different assumptions about whether the server can retrain on sanitized data.
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
from datasets import load_dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.custext.sanitizer import CusTextConfig, CusTextSanitizer, load_text_embeddings


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
    parser = argparse.ArgumentParser(description="CusText/CusText+ classification baseline")
    parser.add_argument("--task", choices=["sst2", "qnli"], default="sst2")
    parser.add_argument("--protocol", choices=["paper", "inference"], default="paper")
    parser.add_argument("--variant", choices=["custext", "custext_plus"], default="custext")
    parser.add_argument(
        "--representation",
        choices=["word_vectors", "model_tokens"],
        default="word_vectors",
        help="word_vectors is paper-faithful; model_tokens is a complete-vocabulary comparison adapter.",
    )
    parser.add_argument("--embedding_path", type=str, default=None)
    parser.add_argument("--max_embedding_vectors", type=int, default=0, help="Smoke-test limit; 0 loads all vectors.")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--mapping_strategy",
        choices=["conservative", "release_compatible", "aggressive"],
        default="release_compatible",
        help="release_compatible mirrors the authors' public code; conservative enforces disjoint supports.",
    )
    parser.add_argument("--tail_policy", choices=["merge_last", "allow_singleton"], default="merge_last")
    parser.add_argument("--mapping_device", type=str, default="cuda:0")
    parser.add_argument("--mapping_scope", choices=["train", "train_eval"], default="train")
    parser.add_argument("--mapping_cache", type=str, default=None)
    parser.add_argument("--force_rebuild_mapping", type=str2bool, default=False)
    parser.add_argument("--numeric_policy", choices=["release_compatible", "keep"], default="release_compatible")

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=0, help="0 means all selected train examples.")
    parser.add_argument("--max_eval_samples", type=int, default=0, help="0 means all selected evaluation examples.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default="outputs/baselines/custext/results.csv")
    parser.add_argument("--save_sanitized_jsonl", type=str, default=None)
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--model_output_dir", type=str, default="outputs/baselines/custext/models")
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def default_dataset_path(task: str) -> str:
    return {"sst2": "data/glue/sst2", "qnli": "data/glue/qnli"}[task]


def default_model_path(task: str, protocol: str) -> str:
    if protocol == "paper":
        return "models/bert-base-uncased"
    return {"sst2": "models/bert-base-uncased-sst2", "qnli": "models/bert-base-uncased-qnli"}[task]


def resolve_local_default(path: str) -> str:
    """Support running either from the workspace root or from ``dp_spd``."""

    candidates = [Path(path), REPO_ROOT / path, REPO_ROOT.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return path


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def load_task_split(task: str, dataset_path: str, split: str):
    if dataset_path and os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path)
        if hasattr(dataset, "keys"):
            if split not in dataset:
                raise KeyError(f"Split {split!r} is not present in {dataset_path}")
            return dataset[split]
        return dataset
    return load_dataset("glue", task, split=split)


def maybe_limit(dataset, max_samples: int, seed: int):
    if max_samples > 0 and len(dataset) > max_samples:
        return dataset.shuffle(seed=seed).select(range(max_samples))
    return dataset


def records_from_dataset(task: str, dataset) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for item in dataset:
        if task == "sst2":
            records.append({"text": str(item["sentence"]), "label": int(item["label"])})
        else:
            records.append(
                {
                    "text": str(item["question"]),
                    "text_pair": str(item["sentence"]),
                    "label": int(item["label"]),
                }
            )
    return records


def iter_record_words(records: Iterable[Mapping[str, object]]) -> Iterable[str]:
    for record in records:
        yield from str(record["text"]).split()
        if "text_pair" in record:
            yield from str(record["text_pair"]).split()


def add_counters(target: Counter, source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[key] += int(value)


def sanitize_text_records(
    records: Sequence[Mapping[str, object]],
    sanitizer: CusTextSanitizer,
    *,
    custext_plus: bool,
    numeric_policy: str,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    rng = np.random.default_rng(seed)
    totals: Counter = Counter()
    sanitized: List[Dict[str, object]] = []
    for record in tqdm(records, desc="Sanitizing text"):
        updated = dict(record)
        updated["text"], counters = sanitizer.sanitize_text(
            str(record["text"]),
            rng=rng,
            preserve_stopwords=custext_plus,
            numeric_policy=numeric_policy,
        )
        add_counters(totals, counters)
        if "text_pair" in record:
            updated["text_pair"], counters = sanitizer.sanitize_text(
                str(record["text_pair"]),
                rng=rng,
                preserve_stopwords=custext_plus,
                numeric_policy=numeric_policy,
            )
            add_counters(totals, counters)
        sanitized.append(updated)
    return sanitized, dict(totals)


class EncodedClassificationDataset(Dataset):
    def __init__(self, encoded: Mapping[str, torch.Tensor], labels: Sequence[int]) -> None:
        self.encoded = {key: value for key, value in encoded.items() if key in {"input_ids", "attention_mask", "token_type_ids"}}
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        if not self.encoded or len(self.labels) != next(iter(self.encoded.values())).shape[0]:
            raise ValueError("Encoded inputs and labels have inconsistent lengths")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encoded.items()}
        item["labels"] = self.labels[index]
        return item


def encode_records(
    records: Sequence[Mapping[str, object]],
    tokenizer,
    max_length: int,
    *,
    sanitizer: Optional[CusTextSanitizer] = None,
    seed: int = 42,
) -> Tuple[EncodedClassificationDataset, Dict[str, int]]:
    first = [str(record["text"]) for record in records]
    second = [str(record["text_pair"]) for record in records] if records and "text_pair" in records[0] else None
    encoded = tokenizer(
        first,
        second,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    counters: Dict[str, int] = {}
    if sanitizer is not None:
        input_ids, counters = sanitizer.sanitize_token_ids(
            encoded["input_ids"],
            encoded.get("attention_mask"),
            rng=np.random.default_rng(seed),
        )
        encoded["input_ids"] = input_ids
    return EncodedClassificationDataset(encoded, [int(record["label"]) for record in records]), counters


def default_mapping_cache(args) -> Path:
    max_train = "all" if args.max_train_samples <= 0 else str(args.max_train_samples)
    max_eval = "all" if args.max_eval_samples <= 0 else str(args.max_eval_samples)
    filename = (
        f"{args.task}_{args.representation}_{args.metric}_{args.mapping_strategy}_"
        f"eps{args.epsilon:g}_k{args.top_k}_{args.mapping_scope}_train{max_train}_eval{max_eval}.json"
    )
    return Path("outputs/baselines/custext/mappings") / filename


def build_or_load_sanitizer(args, model, tokenizer, mapping_records: Sequence[Mapping[str, object]]) -> Tuple[CusTextSanitizer, Path]:
    config = CusTextConfig(
        epsilon=args.epsilon,
        top_k=args.top_k,
        metric=args.metric,
        mapping_strategy=args.mapping_strategy,
        tail_policy=args.tail_policy,
        seed=args.seed,
    )
    cache_path = Path(args.mapping_cache) if args.mapping_cache else default_mapping_cache(args)
    expected_kind = "word" if args.representation == "word_vectors" else "model_token"
    if cache_path.is_file() and not args.force_rebuild_mapping:
        sanitizer = CusTextSanitizer.load(cache_path)
        if sanitizer.kind != expected_kind or sanitizer.config != config:
            raise ValueError(
                f"Cached mapping {cache_path} was built for a different configuration. "
                "Pass --force_rebuild_mapping true or choose a new --mapping_cache."
            )
        if args.representation == "word_vectors" and args.embedding_path and sanitizer.metadata.get("embedding_path") != str(
            Path(args.embedding_path).resolve()
        ):
            raise ValueError("Cached word mapping was built from a different --embedding_path.")
        if args.representation == "model_tokens" and sanitizer.metadata.get("model_path") != args.model_path:
            raise ValueError("Cached model-token mapping was built for a different --model_path.")
        print(f"[CusText] Reusing mapping: {cache_path}")
        return sanitizer, cache_path

    if args.representation == "word_vectors":
        if not args.embedding_path:
            raise ValueError("--embedding_path is required with --representation word_vectors")
        vocabulary, embeddings = load_text_embeddings(args.embedding_path, args.max_embedding_vectors)
        sanitizer = CusTextSanitizer.from_word_vectors(
            vocabulary,
            embeddings,
            iter_record_words(mapping_records),
            config,
            device=args.mapping_device,
            metadata={
                "embedding_path": str(Path(args.embedding_path).resolve()),
                "mapping_scope": args.mapping_scope,
                "mapping_record_count": len(mapping_records),
            },
        )
    else:
        sanitizer = CusTextSanitizer.from_model_token_embeddings(
            model.get_input_embeddings().weight,
            tokenizer.all_special_ids,
            config,
            device=args.mapping_device,
            metadata={
                "model_path": args.model_path,
                "mapping_scope": "all_non_special_model_tokens",
            },
        )
    sanitizer.save(cache_path)
    print(f"[CusText] Saved mapping: {cache_path}")
    return sanitizer, cache_path


def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def train(model, train_loader: DataLoader, eval_loader: DataLoader, args, device: torch.device) -> float:
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs * len(train_loader),
    )
    best_score = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{args.epochs}"):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
        accuracy = evaluate(model, eval_loader, device)
        print(f"[Epoch {epoch + 1}] mean_loss={running_loss / max(len(train_loader), 1):.6f} eval_accuracy={accuracy:.6f}")
        if accuracy > best_score:
            best_score = accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score


RESULT_FIELDS = [
    "task",
    "protocol",
    "variant",
    "representation",
    "model_path",
    "epsilon",
    "top_k",
    "metric",
    "mapping_strategy",
    "mapping_scope",
    "mapping_cache",
    "train_samples",
    "eval_samples",
    "mapped_tokens",
    "total_active_tokens",
    "mapped_fraction",
    "stopword_preserved",
    "unmapped_tokens",
    "group_size_min",
    "group_size_max",
    "accuracy",
    "seed",
]


def append_result(path: str | Path, row: Mapping[str, object]) -> None:
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
    args.dataset_path = args.dataset_path or resolve_local_default(default_dataset_path(args.task))
    args.model_path = args.model_path or resolve_local_default(default_model_path(args.task, args.protocol))
    if args.dry_run:
        print("[Dry Run]")
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        print("paper + word_vectors requires --embedding_path; model_tokens is an explicitly labelled adaptation.")
        return
    if args.representation == "model_tokens" and args.variant == "custext_plus":
        raise ValueError("CusText+ preserves word-level stopwords and is available only with --representation word_vectors")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    train_data = maybe_limit(load_task_split(args.task, args.dataset_path, args.train_split), args.max_train_samples, args.seed)
    eval_data = maybe_limit(load_task_split(args.task, args.dataset_path, args.eval_split), args.max_eval_samples, args.seed + 1)
    train_records = records_from_dataset(args.task, train_data)
    eval_records = records_from_dataset(args.task, eval_data)
    mapping_records = train_records if args.mapping_scope == "train" else train_records + eval_records

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    model_kwargs = {"local_files_only": args.local_files_only}
    if args.protocol == "paper":
        model_kwargs.update({"num_labels": 2, "ignore_mismatched_sizes": True})
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, **model_kwargs).to(device)
    sanitizer, mapping_path = build_or_load_sanitizer(args, model, tokenizer, mapping_records)
    minimum_group, maximum_group = sanitizer.group_size_range()

    custext_plus = args.variant == "custext_plus"
    if args.representation == "word_vectors":
        eval_records, eval_counts = sanitize_text_records(
            eval_records,
            sanitizer,
            custext_plus=custext_plus,
            numeric_policy=args.numeric_policy,
            seed=args.seed + 1,
        )
        train_counts: Dict[str, int] = {}
        if args.protocol == "paper":
            train_records, train_counts = sanitize_text_records(
                train_records,
                sanitizer,
                custext_plus=custext_plus,
                numeric_policy=args.numeric_policy,
                seed=args.seed,
            )
        eval_dataset, _ = encode_records(eval_records, tokenizer, args.max_length)
        train_dataset = encode_records(train_records, tokenizer, args.max_length)[0] if args.protocol == "paper" else None
        counts = Counter(eval_counts)
        add_counters(counts, train_counts)
    else:
        # Stopword preservation applies only to textual CusText+.  It has no
        # well-defined analogue for subword-ID replacement, so record this in
        # the README and do not silently pretend otherwise.
        eval_dataset, eval_counts = encode_records(
            eval_records,
            tokenizer,
            args.max_length,
            sanitizer=sanitizer,
            seed=args.seed + 1,
        )
        train_dataset = None
        train_counts: Dict[str, int] = {}
        if args.protocol == "paper":
            train_dataset, train_counts = encode_records(
                train_records,
                tokenizer,
                args.max_length,
                sanitizer=sanitizer,
                seed=args.seed,
            )
        counts = Counter(eval_counts)
        add_counters(counts, train_counts)

    if args.save_sanitized_jsonl:
        output_path = Path(args.save_sanitized_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in eval_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[CusText] Saved sanitized evaluation text: {output_path}")

    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    if args.protocol == "paper":
        if train_dataset is None:
            raise RuntimeError("paper protocol requires a sanitized training dataset")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        accuracy = train(model, train_loader, eval_loader, args, device)
    else:
        accuracy = evaluate(model, eval_loader, device)

    if args.save_model and args.protocol == "paper":
        output_dir = Path(args.model_output_dir) / (
            f"{args.task}_{args.variant}_{args.representation}_eps{args.epsilon:g}_k{args.top_k}_seed{args.seed}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[CusText] Saved best checkpoint: {output_dir}")

    total = int(counts.get("total", 0))
    mapped = int(counts.get("mapped", 0))
    row = {
        "task": args.task,
        "protocol": args.protocol,
        "variant": args.variant,
        "representation": args.representation,
        "model_path": args.model_path,
        "epsilon": args.epsilon,
        "top_k": args.top_k,
        "metric": args.metric,
        "mapping_strategy": args.mapping_strategy,
        "mapping_scope": args.mapping_scope,
        "mapping_cache": str(mapping_path),
        "train_samples": len(train_records) if args.protocol == "paper" else 0,
        "eval_samples": len(eval_records),
        "mapped_tokens": mapped,
        "total_active_tokens": total,
        "mapped_fraction": f"{mapped / total:.6f}" if total else "",
        "stopword_preserved": int(counts.get("stopword_preserved", 0)),
        "unmapped_tokens": int(counts.get("unmapped", 0)),
        "group_size_min": minimum_group,
        "group_size_max": maximum_group,
        "accuracy": f"{accuracy:.6f}",
        "seed": args.seed,
    }
    append_result(args.output_csv, row)
    print(f"[Result] accuracy={accuracy:.6f}")
    print(f"[Result] mapped_fraction={row['mapped_fraction']} groups=[{minimum_group}, {maximum_group}]")
    print(f"[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()
