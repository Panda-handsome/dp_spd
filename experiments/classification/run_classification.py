# -*- coding: utf-8 -*-
"""Classification utility evaluation for DP-SPD.

Supported paper settings:
- BERT on GLUE SST-2 and QNLI with a sequence-classification checkpoint.
- LLaMA-2-7B-Chat on IMDb with a lightweight probe trained on hidden features.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpspd.client import BatchConfig, DummyConfig, LDPNoiseConfig, PrivacyDataset


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DP-SPD classification under embedding or projected-token transport.")
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "qnli", "imdb"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--head", type=str, default="sequence", choices=["sequence", "probe"])
    parser.add_argument(
        "--transport_mode",
        type=str,
        default="embedding",
        choices=["embedding", "token"],
        help=(
            "embedding sends DP-SPD's noised vectors directly to the accessible server; "
            "token projects every noised, active non-special vector to its nearest vocabulary token "
            "before fixed-server inference."
        ),
    )
    parser.add_argument(
        "--projection_chunk_tokens",
        type=int,
        default=256,
        help="Number of noised token vectors projected per GPU nearest-neighbour chunk.",
    )
    parser.add_argument(
        "--validate_clean_projection",
        type=str2bool,
        default=True,
        help="Verify exact embedding-to-token recovery and clean input_ids/inputs_embeds parity before evaluation.",
    )
    parser.add_argument("--projection_validation_tokens", type=int, default=64)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sensitive_ratio", type=float, default=0.6)
    parser.add_argument("--epsilon_private", type=float, default=3.07)
    parser.add_argument("--epsilon_non_sensitive", type=float, default=1.8)
    parser.add_argument(
        "--target_epsilon",
        type=float,
        default=None,
        help="Optional reported end-to-end privacy budget label; it does not alter the local noise parameters.",
    )
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default="outputs/classification.csv")
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def default_model_path(task: str, head: str) -> str:
    if task == "imdb" and head == "probe":
        return "models/Llama-2-7b-chat-hf"
    if task == "qnli":
        return "models/bert-base-uncased-qnli"
    return "models/bert-base-uncased-sst2"


def default_dataset_path(task: str) -> str:
    return {
        "imdb": "data/imdb",
        "sst2": "data/glue/sst2",
        "qnli": "data/glue/qnli",
    }[task]


def default_eval_split(task: str) -> str:
    return "test" if task == "imdb" else "validation"


def resolve_dtype(name: str):
    if name == "auto":
        return None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_task_dataset(task: str, dataset_path: str, split: str):
    if dataset_path and os.path.isdir(dataset_path):
        raw = load_from_disk(dataset_path)
        return raw[split] if hasattr(raw, "keys") else raw
    if task == "imdb":
        return load_dataset("imdb", split=split)
    if task == "sst2":
        return load_dataset("glue", "sst2", split=split)
    if task == "qnli":
        return load_dataset("glue", "qnli", split=split)
    raise ValueError(f"Unsupported task: {task}")


def format_examples(task: str, dataset, max_samples: int, seed: int):
    if max_samples > 0 and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
    if task == "imdb":
        return list(dataset["text"]), list(dataset["label"])
    if task == "sst2":
        return list(dataset["sentence"]), list(dataset["label"])
    if task == "qnli":
        return list(zip(dataset["question"], dataset["sentence"])), list(dataset["label"])
    raise ValueError(f"Unsupported task: {task}")


def collate_privacy(batch):
    out = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "noisy_embeddings": torch.stack([item["noisy_embeddings"] for item in batch]),
        "group_index": torch.tensor([int(item["group_index"]) for item in batch], dtype=torch.long),
    }
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.stack([item["token_type_ids"] for item in batch])
    return out


def make_dataset(texts, tokenizer, emb_layer, hidden_size: int, args):
    batch_cfg = BatchConfig(max_length=args.max_length, device=torch.device("cpu"), hidden_size=hidden_size)
    dummy_cfg = DummyConfig(
        m_private_tokens=0,
        sensitive_ratio=args.sensitive_ratio,
        n_dummies_per_text=0,
        dummy_mode="token",
        pos_mapping_path=None,
    )
    ldp_cfg = LDPNoiseConfig(
        mechanism="Laplace",
        epsilon_private=args.epsilon_private,
        epsilon_non_sensitive=args.epsilon_non_sensitive,
        clip_norm=args.clip_norm,
        seed=args.seed,
    )
    return PrivacyDataset(
        texts,
        tokenizer,
        batch_cfg,
        dummy_cfg,
        ldp_cfg,
        seed=args.seed,
        shuffle_mode="global",
        pretrained_embeddings=emb_layer,
    )


def empty_projection_stats() -> Dict[str, int]:
    return {"active_tokens": 0, "changed_tokens": 0}


def merge_projection_stats(total: Dict[str, int], update: Dict[str, int]) -> None:
    total["active_tokens"] += int(update["active_tokens"])
    total["changed_tokens"] += int(update["changed_tokens"])


def project_noisy_embeddings_to_token_ids(
    noisy_embeddings: torch.Tensor,
    original_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    embedding_weight: torch.Tensor,
    special_token_ids: Sequence[int],
    chunk_tokens: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Nearest-token transport for a batch of DP-SPD noised word embeddings.

    The fixed BERT server receives only ``input_ids`` on this path.  CLS, SEP,
    PAD and other tokenizer special tokens are preserved exactly; all active
    lexical positions are independently projected to the closest row of the
    *same server checkpoint's* word-embedding table under squared L2 distance.
    """

    if noisy_embeddings.ndim != 3 or original_ids.shape != noisy_embeddings.shape[:2]:
        raise ValueError("Expected noisy embeddings [batch, length, hidden] and matching input ids.")
    if chunk_tokens < 1:
        raise ValueError("--projection_chunk_tokens must be positive.")

    projected_ids = original_ids.clone()
    active_lexical = attention_mask.to(dtype=torch.bool)
    for special_id in special_token_ids:
        active_lexical &= original_ids.ne(int(special_id))

    active_count = int(active_lexical.sum().item())
    if active_count == 0:
        return projected_ids, {"active_tokens": 0, "changed_tokens": 0}

    # The batched GPU implementation is important for Table 1 scale: doing an
    # independent CPU vocabulary scan for every token would turn token transport
    # itself into the dominant experiment cost.
    vocab = embedding_weight.detach().to(device=noisy_embeddings.device, dtype=torch.float32)
    vocab_norm_sq = (vocab * vocab).sum(dim=1)
    blocked_ids = [token_id for token_id in special_token_ids if 0 <= int(token_id) < vocab.size(0)]
    vectors = noisy_embeddings[active_lexical].to(dtype=torch.float32)
    nearest_chunks: List[torch.Tensor] = []
    for start in range(0, vectors.size(0), chunk_tokens):
        chunk = vectors[start : start + chunk_tokens]
        distances = (
            (chunk * chunk).sum(dim=1, keepdim=True)
            + vocab_norm_sq.unsqueeze(0)
            - 2.0 * chunk @ vocab.t()
        )
        if blocked_ids:
            distances[:, blocked_ids] = float("inf")
        nearest_chunks.append(distances.argmin(dim=1).to(dtype=torch.long))

    projected_ids[active_lexical] = torch.cat(nearest_chunks, dim=0)
    changed = int((projected_ids[active_lexical] != original_ids[active_lexical]).sum().item())
    return projected_ids, {"active_tokens": active_count, "changed_tokens": changed}


def make_model_inputs(batch, model, tokenizer, args, device: torch.device):
    """Build either the white-box embedding inputs or black-box-style token inputs."""

    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    if args.transport_mode == "embedding":
        inputs = {
            "inputs_embeds": batch["noisy_embeddings"].to(device=device, dtype=next(model.parameters()).dtype),
            "attention_mask": attention_mask,
        }
        stats = empty_projection_stats()
    else:
        input_ids, stats = project_noisy_embeddings_to_token_ids(
            batch["noisy_embeddings"].to(device),
            batch["input_ids"].to(device),
            attention_mask,
            model.get_input_embeddings().weight,
            tokenizer.all_special_ids,
            args.projection_chunk_tokens,
        )
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if token_type_ids is not None:
        inputs["token_type_ids"] = token_type_ids
    return inputs, attention_mask, stats


def validate_token_transport(model, tokenizer, texts, args, device: torch.device) -> float:
    """Fail fast if exact projection or BERT's clean input paths disagree."""

    if not args.validate_clean_projection:
        return float("nan")

    emb_weight = model.get_input_embeddings().weight
    special_ids = set(int(token_id) for token_id in tokenizer.all_special_ids)
    candidate_ids = [token_id for token_id in range(emb_weight.size(0)) if token_id not in special_ids]
    candidate_ids = candidate_ids[: max(1, args.projection_validation_tokens)]
    exact_ids = torch.tensor(candidate_ids, device=device, dtype=torch.long).unsqueeze(0)
    exact_embeddings = emb_weight.detach()[exact_ids]
    recovered_ids, stats = project_noisy_embeddings_to_token_ids(
        exact_embeddings,
        exact_ids,
        torch.ones_like(exact_ids),
        emb_weight,
        tokenizer.all_special_ids,
        args.projection_chunk_tokens,
    )
    if not torch.equal(exact_ids, recovered_ids):
        raise RuntimeError("Exact embedding-to-token projection check failed; refusing to run token-level evaluation.")

    sample_texts = list(texts[: min(2, len(texts))])
    if not sample_texts:
        raise ValueError("At least one example is required for clean token-transport validation.")
    if isinstance(sample_texts[0], tuple):
        first, second = zip(*sample_texts)
        encoded = tokenizer(list(first), list(second), truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
    else:
        encoded = tokenizer(sample_texts, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask", "token_type_ids"}}
    with torch.no_grad():
        token_logits = model(**encoded).logits
        embedding_inputs = dict(encoded)
        input_ids = embedding_inputs.pop("input_ids")
        embedding_inputs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
        embedding_logits = model(**embedding_inputs).logits
    max_abs_diff = float((token_logits - embedding_logits).abs().max().item())
    if max_abs_diff > 1e-4:
        raise RuntimeError(
            "Clean input_ids/inputs_embeds classifier parity check failed: "
            f"max |logit difference|={max_abs_diff:.3e}."
        )
    print(
        "[Token transport check] "
        f"exact_recovery={stats['active_tokens']}/{stats['active_tokens']} "
        f"clean_max_logit_diff={max_abs_diff:.3e}"
    )
    return max_abs_diff


def evaluate_sequence_classifier(model, tokenizer, texts, labels, args, device):
    emb_layer = model.get_input_embeddings()
    hidden_size = model.config.hidden_size
    dataset = make_dataset(texts, tokenizer, emb_layer, hidden_size, args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_privacy)

    y_true: List[int] = []
    y_pred: List[int] = []
    projection_stats = empty_projection_stats()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Sequence classification"):
            model_inputs, _, batch_stats = make_model_inputs(batch, model, tokenizer, args, device)
            merge_projection_stats(projection_stats, batch_stats)
            outputs = model(**model_inputs)
            preds = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
            y_pred.extend(preds)
            y_true.extend([int(labels[idx]) for idx in batch["group_index"].tolist()])
    return accuracy_score(y_true, y_pred), projection_stats


def extract_probe_features(model, tokenizer, texts, labels, args, device):
    emb_layer = model.get_input_embeddings()
    hidden_size = model.config.hidden_size
    dataset = make_dataset(texts, tokenizer, emb_layer, hidden_size, args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_privacy)

    features: List[np.ndarray] = []
    targets: List[int] = []
    projection_stats = empty_projection_stats()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Feature extraction"):
            model_inputs, attention_mask, batch_stats = make_model_inputs(batch, model, tokenizer, args, device)
            merge_projection_stats(projection_stats, batch_stats)
            outputs = model(**model_inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=device), lengths]
            features.append(pooled.float().cpu().numpy())
            targets.extend([int(labels[idx]) for idx in batch["group_index"].tolist()])
    return np.concatenate(features, axis=0), np.asarray(targets), projection_stats


def evaluate_probe(model, tokenizer, train_texts, train_labels, eval_texts, eval_labels, args, device):
    x_train, y_train, train_stats = extract_probe_features(model, tokenizer, train_texts, train_labels, args, device)
    x_eval, y_eval, eval_stats = extract_probe_features(model, tokenizer, eval_texts, eval_labels, args, device)
    clf = LogisticRegression(max_iter=2000, n_jobs=1, random_state=args.seed)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_eval)
    return accuracy_score(y_eval, preds), {
        "active_tokens": train_stats["active_tokens"] + eval_stats["active_tokens"],
        "changed_tokens": train_stats["changed_tokens"] + eval_stats["changed_tokens"],
    }


def main() -> None:
    args = get_parser().parse_args()
    args.model_path = args.model_path or default_model_path(args.task, args.head)
    args.dataset_path = args.dataset_path or default_dataset_path(args.task)
    args.eval_split = args.eval_split or default_eval_split(args.task)

    if args.dry_run:
        print("[Dry Run]")
        print(vars(args))
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    eval_data = load_task_dataset(args.task, args.dataset_path, args.eval_split)
    eval_texts, eval_labels = format_examples(args.task, eval_data, args.max_eval_samples, args.seed + 1)
    if args.head == "probe":
        train_data = load_task_dataset(args.task, args.dataset_path, args.train_split)
        train_texts, train_labels = format_examples(args.task, train_data, args.max_train_samples, args.seed)
    else:
        train_texts, train_labels = [], []

    model_kwargs = {"local_files_only": args.local_files_only}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    if args.head == "sequence":
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path, **model_kwargs).to(device)
        clean_max_logit_diff = (
            validate_token_transport(model, tokenizer, eval_texts, args, device)
            if args.transport_mode == "token"
            else float("nan")
        )
        acc, projection_stats = evaluate_sequence_classifier(model, tokenizer, eval_texts, eval_labels, args, device)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs).to(device)
        except Exception:
            model = AutoModel.from_pretrained(args.model_path, **model_kwargs).to(device)
        clean_max_logit_diff = float("nan")
        acc, projection_stats = evaluate_probe(model, tokenizer, train_texts, train_labels, eval_texts, eval_labels, args, device)

    row = {
        "task": args.task,
        "model_path": args.model_path,
        "head": args.head,
        "transport_mode": args.transport_mode,
        "sensitive_ratio": args.sensitive_ratio,
        "target_epsilon": "" if args.target_epsilon is None else args.target_epsilon,
        "epsilon_private": args.epsilon_private,
        "epsilon_non_sensitive": args.epsilon_non_sensitive,
        "clip_norm": args.clip_norm,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "projection_active_tokens": projection_stats["active_tokens"],
        "projection_changed_tokens": projection_stats["changed_tokens"],
        "projection_token_recovery": (
            f"{1.0 - projection_stats['changed_tokens'] / max(projection_stats['active_tokens'], 1):.6f}"
            if args.transport_mode == "token"
            else ""
        ),
        "clean_transport_max_logit_diff": "" if np.isnan(clean_max_logit_diff) else f"{clean_max_logit_diff:.8g}",
        "accuracy": f"{acc:.6f}",
        "train_samples": len(train_texts),
        "eval_samples": len(eval_texts),
    }
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[Result] accuracy={acc:.6f}")
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
