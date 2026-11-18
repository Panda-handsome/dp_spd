#!/usr/bin/env python
"""IMDb accuracy adapters for the six non-DP-SPD baselines.

The legacy code used for this project's Figure 2 freezes LLaMA-2-7B-Chat,
extracts the final active token representation, and fits a two-layer
``4096 -> 2048 -> 2`` ReLU MLP (SGD, 50 epochs) on the complete shuffled IMDb
training split.
This runner preserves that dataset/model/classifier protocol for every
baseline: transform an IMDb example, extract that LLaMA feature, fit the
same legacy probe on privatized train features, and score the complete
privatized IMDb test split.

It is deliberately not a claim that every original method was designed for
this exact task.  ``paper_component`` and ``omitted_paper_component`` in the
CSV make the unavoidable LLaMA adaptations explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.cape.sanitizer import CAPEConfig, normalize_and_perturb_sequence
from baseline.custext.sanitizer import CusTextConfig, CusTextSanitizer
from baseline.dp_forward.noise import DPForwardConfig
from baseline.dp_forward.split_llama import LlamaDPForwardSplit
from baseline.inferdpt.rantxt import RANTextConfig, RANTextSanitizer
from baseline.santext.sanitizer import SanTextConfig, SanTextTokenSanitizer, SanTextVocabulary, build_vocabulary
from baseline.textobfuscator.split_llama import LlamaTextObfuscatorSplit, TextObfuscatorConfig


METHODS = ("clean", "cape", "custext", "dp_forward", "inferdpt", "santext", "textobfuscator")
PROTOCOL = "legacy_dp_classification_frozen_llama_enhanced_mlp_v1"


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description="Six baseline adapters on Figure-2 IMDb/LLaMA probing.")
    out.add_argument("--method", choices=METHODS, required=True)
    out.add_argument("--epsilon", type=float, default=None, help="Required and positive for every privatizing method; omitted for clean.")
    out.add_argument("--model_path", default="models/Llama-2-7b-chat-hf")
    out.add_argument("--dataset_path", default="data/imdb")
    out.add_argument("--train_split", default="train")
    out.add_argument("--eval_split", default="test")
    out.add_argument("--max_train_samples", type=int, default=0, help="0 reproduces legacy drivers: all 25,000 IMDb train rows.")
    out.add_argument("--max_eval_samples", type=int, default=0, help="0 reproduces legacy drivers: all 25,000 IMDb test rows.")
    out.add_argument("--max_length", type=int, default=128)
    out.add_argument("--batch_size", type=int, default=32)
    out.add_argument("--device", default="cuda:0")
    out.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    out.add_argument("--local_files_only", type=str2bool, default=True)
    out.add_argument("--require_cuda", type=str2bool, default=True)
    out.add_argument("--seed", type=int, default=42)
    out.add_argument(
        "--classifier",
        choices=("legacy_enhanced_mlp", "logistic_regression"),
        default="legacy_enhanced_mlp",
        help="legacy_enhanced_mlp exactly follows dp_classification/tools/classifier.py; logistic_regression is retained only for diagnostics.",
    )
    out.add_argument("--classifier_epochs", type=int, default=50)
    out.add_argument("--classifier_batch_size", type=int, default=256)
    out.add_argument("--classifier_learning_rate", type=float, default=1e-3)
    out.add_argument("--classifier_momentum", type=float, default=0.9)
    out.add_argument("--output_csv", default="outputs/baselines/figure2/imdb_classification.csv")
    out.add_argument(
        "--prepare_mapping_only",
        type=str2bool,
        default=False,
        help="Build/reuse SanText or InferDPT's public IMDb-train vocabulary, then exit before feature extraction.",
    )
    out.add_argument("--dry_run", type=str2bool, default=False)

    # CusText: original word vectors are unavailable locally; model-token is
    # the same fully specified LLaMA adapter used by the generation baseline.
    out.add_argument("--custext_top_k", type=int, default=20)
    out.add_argument("--custext_metric", choices=("cosine", "euclidean"), default="cosine")
    out.add_argument(
        "--custext_mapping_strategy",
        choices=("release_compatible", "conservative", "aggressive"),
        default="release_compatible",
    )
    out.add_argument("--custext_mapping_cache", default=None)
    out.add_argument("--force_rebuild_mapping", type=str2bool, default=False)

    # SanText: build its public vocabulary from raw IMDb training text only.
    out.add_argument("--santext_variant", choices=("santext", "santext_plus"), default="santext")
    out.add_argument("--santext_sensitive_fraction", type=float, default=0.9)
    out.add_argument("--santext_non_sensitive_sanitize_probability", type=float, default=0.3)
    out.add_argument("--santext_source_chunk_size", type=int, default=96)
    out.add_argument("--santext_mapping_max_samples", type=int, default=0)
    out.add_argument("--santext_mapping_batch_size", type=int, default=128)
    out.add_argument("--santext_mapping_cache", default=None)

    # InferDPT: 11k is the local LLaMA-BPE counterpart used by the existing
    # generation adapter; it must be reduced only for a smoke test.
    out.add_argument("--inferdpt_candidate_vocab_size", type=int, default=11000)
    out.add_argument("--inferdpt_source_chunk_size", type=int, default=64)
    out.add_argument("--inferdpt_mapping_max_samples", type=int, default=0)
    out.add_argument("--inferdpt_mapping_batch_size", type=int, default=128)
    out.add_argument("--inferdpt_mapping_cache", default=None)

    out.add_argument("--dp_forward_delta", type=float, default=1e-5)
    out.add_argument("--dp_forward_norm_c", type=float, default=1.0)
    out.add_argument("--dp_forward_noise_layer", type=int, default=28)
    out.add_argument("--textobfuscator_client_layers", type=int, default=3)
    out.add_argument("--verify_clean_split", type=str2bool, default=True)
    return out


def local_path(value: str) -> str:
    for candidate in (Path(value), REPO_ROOT / value, REPO_ROOT.parent / value):
        if candidate.exists():
            return str(candidate)
    return value


def make_device(requested: str, require_cuda: bool) -> torch.device:
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            if require_cuda:
                raise RuntimeError("CUDA was required, but torch.cuda.is_available() is false")
            return torch.device("cpu")
        return torch.device(requested)
    return torch.device(requested)


def dataset_examples(path: str, split: str, max_samples: int, seed: int) -> Tuple[List[str], np.ndarray]:
    raw = load_from_disk(path)
    if not hasattr(raw, "keys") or split not in raw:
        raise KeyError(f"IMDb split {split!r} was not found in {path}")
    data = raw[split]
    if max_samples > 0 and len(data) > max_samples:
        data = data.shuffle(seed=seed).select(range(max_samples))
    return [str(text) for text in data["text"]], np.asarray(data["label"], dtype=np.int64)


class LegacyEnhancedClsModel(nn.Module):
    """The exact MLP architecture from legacy ``tools/classifier.py``."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim // 2, num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(values)))


def legacy_enhanced_mlp_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    args,
    device: torch.device,
) -> Tuple[float, Dict[str, float | int | str]]:
    """Run the historic Figure-2 enhanced MLP evaluator.

    The old driver reports the best test accuracy across its 50 epochs.  This
    function deliberately keeps that selection rule for numerical
    comparability with the already reported DP-SPD Figure-2 curve.  It is
    recorded in output metadata rather than presented as a held-out model
    selection protocol.
    """

    if args.classifier_epochs < 1 or args.classifier_batch_size < 1:
        raise ValueError("classifier_epochs and classifier_batch_size must be positive")
    train_x = torch.from_numpy(x_train).to(device=device, dtype=torch.float32)
    train_y = torch.from_numpy(y_train).to(device=device, dtype=torch.long)
    eval_x = torch.from_numpy(x_eval).to(device=device, dtype=torch.float32)
    eval_y = torch.from_numpy(y_eval).to(device=device, dtype=torch.long)
    classes = int(torch.unique(train_y).numel())
    if classes != 2:
        raise ValueError(f"IMDb training subset must contain both labels; observed {classes} class(es)")
    model = LegacyEnhancedClsModel(train_x.shape[1], classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.classifier_learning_rate, momentum=args.classifier_momentum)
    best_accuracy = 0.0
    last_loss = float("nan")
    for _ in range(args.classifier_epochs):
        model.train()
        permutation = torch.randperm(len(train_x), device=device)
        for start in range(0, len(train_x), args.classifier_batch_size):
            indices = permutation[start : start + args.classifier_batch_size]
            logits = model(train_x.index_select(0, indices))
            loss = loss_fn(logits, train_y.index_select(0, indices))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
        model.eval()
        with torch.no_grad():
            correct = 0
            for start in range(0, len(eval_x), args.classifier_batch_size):
                predicted = model(eval_x[start : start + args.classifier_batch_size]).argmax(dim=-1)
                correct += int((predicted == eval_y[start : start + args.classifier_batch_size]).sum().item())
        best_accuracy = max(best_accuracy, correct / len(eval_y))
    return best_accuracy, {
        "name": "legacy_enhanced_mlp", "architecture": f"{train_x.shape[1]}->{train_x.shape[1] // 2}->2_relu",
        "optimizer": "SGD", "learning_rate": args.classifier_learning_rate, "momentum": args.classifier_momentum,
        "epochs": args.classifier_epochs, "batch_size": args.classifier_batch_size,
        "selection": "best_test_accuracy_across_epochs__legacy_figure2_compatibility", "last_loss": last_loss,
    }


def special_ids(tokenizer) -> List[int]:
    return sorted({int(token) for token in tokenizer.all_special_ids if token is not None})


def count_public_tokens(
    texts: Sequence[str], tokenizer, max_samples: int, batch_size: int
) -> Counter:
    if batch_size < 1:
        raise ValueError("mapping batch size must be positive")
    chosen = texts if max_samples <= 0 else texts[:max_samples]
    excluded = set(special_ids(tokenizer))
    counts: Counter = Counter()
    for start in tqdm(range(0, len(chosen), batch_size), desc="Building public IMDb BPE vocabulary"):
        encoded = tokenizer(chosen[start : start + batch_size], add_special_tokens=True, truncation=False, padding=False)
        for ids in encoded["input_ids"]:
            counts.update(int(token) for token in ids if int(token) not in excluded)
    if not counts:
        raise ValueError("No non-special tokens found in selected IMDb training text")
    return counts


def cache_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def custext_cache_path(args) -> Path:
    if args.custext_mapping_cache:
        return Path(args.custext_mapping_cache)
    name = Path(args.model_path).name.replace(" ", "_")
    # A model-token CusText mapping is independent of the downstream corpus.
    # Reuse the already materialized Figure-3 mapping when its mechanism
    # configuration is identical; ``make_custext`` still verifies that config.
    shared = Path("outputs/baselines/custext/mappings") / (
        f"wikitext_figure3_{name}_model_tokens_{args.custext_metric}_"
        f"{args.custext_mapping_strategy}_eps{args.epsilon:g}_k{args.custext_top_k}.json"
    )
    if shared.is_file():
        return shared
    return Path("outputs/baselines/custext/mappings") / (
        f"imdb_probe_{name}_model_tokens_{args.custext_metric}_{args.custext_mapping_strategy}_"
        f"eps{args.epsilon:g}_k{args.custext_top_k}.json"
    )


def make_custext(args, tokenizer, embedding_weight: torch.Tensor) -> Tuple[CusTextSanitizer, Dict[str, Any]]:
    config = CusTextConfig(
        epsilon=args.epsilon,
        top_k=args.custext_top_k,
        metric=args.custext_metric,
        mapping_strategy=args.custext_mapping_strategy,
        seed=args.seed,
    )
    path = custext_cache_path(args)
    if path.is_file() and not args.force_rebuild_mapping:
        sanitizer = CusTextSanitizer.load(path)
        if sanitizer.kind != "model_token" or sanitizer.config != config:
            raise ValueError(f"CusText cache configuration mismatch: {path}")
        return sanitizer, {"mapping_cache": str(path), "mapping_reused": True, **dict(sanitizer.metadata)}
    sanitizer = CusTextSanitizer.from_model_token_embeddings(
        embedding_weight,
        special_ids(tokenizer),
        config,
        device=args.device,
        metadata={
            "protocol": PROTOCOL,
            "dataset": "IMDb",
            "representation": "all_non_special_llama_bpe_input_tokens",
            "model_path": str(Path(args.model_path).resolve()),
        },
    )
    sanitizer.save(path)
    return sanitizer, {"mapping_cache": str(path), "mapping_reused": False, **dict(sanitizer.metadata)}


def santext_cache_path(args) -> Path:
    if args.santext_mapping_cache:
        return Path(args.santext_mapping_cache)
    name = Path(args.model_path).name.replace(" ", "_")
    sample_tag = "all" if args.santext_mapping_max_samples <= 0 else str(args.santext_mapping_max_samples)
    return Path("outputs/baselines/santext/mappings") / (
        f"imdb_train_{name}_{args.santext_variant}_n{sample_tag}_v2.json"
    )


def make_santext(args, tokenizer, train_texts: Sequence[str]) -> Tuple[SanTextTokenSanitizer, Dict[str, Any]]:
    config = SanTextConfig(
        epsilon=args.epsilon,
        variant=args.santext_variant,
        sensitive_fraction=args.santext_sensitive_fraction,
        non_sensitive_sanitize_probability=args.santext_non_sensitive_sanitize_probability,
        source_chunk_size=args.santext_source_chunk_size,
    )
    path = santext_cache_path(args)
    requested = {
        "protocol": PROTOCOL,
        "mapping_split": args.train_split,
        "mapping_max_samples": args.santext_mapping_max_samples,
        "mapping_selection": "raw_train_first_n_or_all_v1",
        "tokenizer_vocab_size": len(tokenizer),
    }
    if path.is_file():
        payload = load_json(path)
        if payload.get("metadata") != requested:
            raise ValueError(f"SanText cache metadata mismatch: {path}; choose another cache or rebuild it")
        vocabulary = SanTextVocabulary.from_dict(payload["vocabulary"])
        reused = True
    else:
        counts = count_public_tokens(train_texts, tokenizer, args.santext_mapping_max_samples, args.santext_mapping_batch_size)
        vocabulary = build_vocabulary(counts, config, metadata=requested)
        cache_json(path, {"metadata": requested, "vocabulary": vocabulary.to_dict()})
        reused = False
    sanitizer = SanTextTokenSanitizer(vocabulary, config, vocab_size=len(tokenizer), special_token_ids=special_ids(tokenizer))
    return sanitizer, {
        "mapping_cache": str(path), "mapping_reused": reused, "source_vocabulary_size": len(vocabulary.source_token_ids),
        "sensitive_output_vocabulary_size": len(vocabulary.sensitive_token_ids), "mapping_total_observed_tokens": vocabulary.total_observed_tokens,
    }


def inferdpt_cache_path(args) -> Path:
    if args.inferdpt_mapping_cache:
        return Path(args.inferdpt_mapping_cache)
    name = Path(args.model_path).name.replace(" ", "_")
    sample_tag = "all" if args.inferdpt_mapping_max_samples <= 0 else str(args.inferdpt_mapping_max_samples)
    return Path("outputs/baselines/inferdpt/mappings") / (
        f"imdb_train_{name}_top{args.inferdpt_candidate_vocab_size}_n{sample_tag}_v2.json"
    )


def make_inferdpt(args, tokenizer, train_texts: Sequence[str]) -> Tuple[RANTextSanitizer, Dict[str, Any]]:
    path = inferdpt_cache_path(args)
    requested = {
        "protocol": PROTOCOL,
        "mapping_split": args.train_split,
        "mapping_max_samples": args.inferdpt_mapping_max_samples,
        "mapping_selection": "raw_train_first_n_or_all_v1",
        "candidate_vocab_size": args.inferdpt_candidate_vocab_size,
        "tokenizer_vocab_size": len(tokenizer),
    }
    if path.is_file():
        payload = load_json(path)
        if payload.get("metadata") != requested:
            raise ValueError(f"InferDPT cache metadata mismatch: {path}; choose another cache or rebuild it")
        candidates = [int(item) for item in payload["candidate_token_ids"]]
        observed = int(payload["total_observed_tokens"])
        reused = True
    else:
        counts = count_public_tokens(train_texts, tokenizer, args.inferdpt_mapping_max_samples, args.inferdpt_mapping_batch_size)
        candidates = [int(token) for token, _ in counts.most_common(args.inferdpt_candidate_vocab_size)]
        if not candidates:
            raise ValueError("InferDPT candidate vocabulary is empty")
        observed = int(sum(counts.values()))
        cache_json(path, {"metadata": requested, "candidate_token_ids": candidates, "total_observed_tokens": observed})
        reused = False
    sanitizer = RANTextSanitizer(
        candidates,
        RANTextConfig(args.epsilon, args.inferdpt_source_chunk_size, args.seed),
        vocab_size=len(tokenizer),
    )
    return sanitizer, {"mapping_cache": str(path), "mapping_reused": reused, "candidate_vocabulary_size": len(candidates), "mapping_total_observed_tokens": observed}


NON_ADDITIVE_STATISTICS = {
    "adjacency_mean_size", "adjacency_min_size", "adjacency_max_size", "coverage_fraction",
    "realized_replacement_fraction", "laplace_denominator_z", "laplace_scale", "normalization_range",
}


def sum_counts(total: Counter, update: Mapping[str, float | int]) -> None:
    """Accumulate token/sample counts, never summing rates or fixed parameters."""

    for key, value in update.items():
        if key not in NON_ADDITIVE_STATISTICS and isinstance(value, (int, float)):
            total[key] += float(value)


def compact_inferdpt(ids: torch.Tensor, retained: torch.Tensor, pad_id: int, eos_id: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Drop InferDPT OOV slots so the server sees its actual advertised prompt."""

    rows = [row[keep].tolist() for row, keep in zip(ids, retained)]
    empty = sum(not row for row in rows)
    rows = [row if row else [eos_id] for row in rows]
    width = max(len(row) for row in rows)
    output = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        output[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, : len(row)] = 1
    return output, mask, empty


class FeatureExtractor:
    def __init__(self, args, model, tokenizer, mapping_texts: Sequence[str], device: torch.device) -> None:
        self.args, self.model, self.tokenizer, self.device = args, model, tokenizer, device
        self.embedding_weight = model.get_input_embeddings().weight
        self.counts: Counter = Counter()
        self.metadata: Dict[str, Any] = {}
        self._cape_ranges: List[float] = []
        self._infer_adjacency_total = 0.0
        self._infer_adjacency_min: float | None = None
        self._infer_adjacency_max = 0.0
        self._infer_laplace_denominator: float | None = None
        self.dp_split = None
        self.to_split = None
        self.sanitizer = None

        if args.method == "custext":
            self.sanitizer, self.metadata = make_custext(args, tokenizer, self.embedding_weight)
        elif args.method == "santext":
            self.sanitizer, self.metadata = make_santext(args, tokenizer, mapping_texts)
        elif args.method == "inferdpt":
            self.sanitizer, self.metadata = make_inferdpt(args, tokenizer, mapping_texts)
        elif args.method == "dp_forward":
            config = DPForwardConfig(args.epsilon, args.dp_forward_delta, args.dp_forward_norm_c, args.dp_forward_noise_layer, args.seed)
            self.dp_split = LlamaDPForwardSplit(model, config)
            self.metadata = {"noise_std": config.noise_std, "noise_layer": args.dp_forward_noise_layer, "client_layers": self.dp_split.client_layer_count}
        elif args.method == "textobfuscator":
            config = TextObfuscatorConfig(args.epsilon, args.textobfuscator_client_layers, args.seed)
            self.to_split = LlamaTextObfuscatorSplit(model, config)
            self.metadata = {"client_layers": args.textobfuscator_client_layers, "server_layers": len(model.model.layers) - args.textobfuscator_client_layers, "laplace_scale": 1.0 / args.epsilon}
        elif args.method == "cape":
            self.metadata = {"laplace_scale": 1.0 / args.epsilon}
        elif args.method == "clean":
            self.metadata = {"privacy": "none"}

    def _model_features(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        result = self.model(
            input_ids=ids.to(self.device), attention_mask=mask.to(self.device), output_hidden_states=True, use_cache=False
        )
        return result.hidden_states[-1]

    def features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
        args = self.args
        if args.method == "clean":
            hidden = self._model_features(input_ids, attention_mask)
        elif args.method == "custext":
            sanitized, stats = self.sanitizer.sanitize_token_ids(
                input_ids, attention_mask, rng=np.random.default_rng(args.seed + int(sample_indices[0]))
            )
            sum_counts(self.counts, stats)
            hidden = self._model_features(sanitized, attention_mask)
        elif args.method == "santext":
            sanitized, stats = self.sanitizer.sanitize_token_ids(
                input_ids, attention_mask, self.embedding_weight, seed=args.seed + int(sample_indices[0])
            )
            sum_counts(self.counts, stats)
            hidden = self._model_features(sanitized, attention_mask)
        elif args.method == "inferdpt":
            sampled, retained, stats = self.sanitizer.sanitize_token_ids(
                input_ids, attention_mask, self.embedding_weight, seed=args.seed + int(sample_indices[0])
            )
            ids, mask, empty = compact_inferdpt(sampled, retained, int(self.tokenizer.pad_token_id), int(self.tokenizer.eos_token_id))
            stats = {**stats, "empty_after_discard": empty}
            sum_counts(self.counts, stats)
            sampled_count = float(stats["sampled_tokens"])
            self._infer_adjacency_total += float(stats["adjacency_mean_size"]) * sampled_count
            value_min = float(stats["adjacency_min_size"])
            self._infer_adjacency_min = value_min if self._infer_adjacency_min is None else min(self._infer_adjacency_min, value_min)
            self._infer_adjacency_max = max(self._infer_adjacency_max, float(stats["adjacency_max_size"]))
            self._infer_laplace_denominator = float(stats["laplace_denominator_z"])
            hidden = self._model_features(ids, mask)
            attention_mask = mask
        elif args.method == "cape":
            clean = self.model.get_input_embeddings()(input_ids.to(self.device)).detach().cpu()
            rows = []
            for position in range(clean.shape[0]):
                noisy, stats = normalize_and_perturb_sequence(
                    clean[position], attention_mask[position], CAPEConfig(args.epsilon, args.seed),
                    seed=args.seed + 1_000_003 * int(sample_indices[position]),
                )
                rows.append(noisy)
                sum_counts(self.counts, stats)
                self._cape_ranges.append(float(stats["normalization_range"]))
            embeds = torch.stack(rows).to(device=self.device, dtype=self.embedding_weight.dtype)
            result = self.model(inputs_embeds=embeds, attention_mask=attention_mask.to(self.device), output_hidden_states=True, use_cache=False)
            hidden = result.hidden_states[-1]
        elif args.method == "dp_forward":
            hidden = self.dp_split.forward_hidden(input_ids.to(self.device), attention_mask.to(self.device), sample_indices.to(self.device))
        elif args.method == "textobfuscator":
            hidden = self.to_split.forward_hidden(input_ids.to(self.device), attention_mask.to(self.device), sample_indices.to(self.device))
        else:  # defensive: argparse constrains this already
            raise ValueError(args.method)
        lengths = attention_mask.to(self.device).sum(dim=1).clamp(min=1).to(torch.long) - 1
        return hidden[torch.arange(hidden.shape[0], device=self.device), lengths].float().cpu()

    def statistics(self) -> Dict[str, float]:
        """Return auditable corpus aggregates with correctly recomputed rates."""

        output = dict(self.counts)
        if self.args.method == "cape":
            output["mean_normalization_range"] = float(np.mean(self._cape_ranges)) if self._cape_ranges else 0.0
            output["laplace_scale"] = 1.0 / self.args.epsilon
        elif self.args.method == "inferdpt":
            active = float(output.get("active_tokens", 0.0))
            sampled = float(output.get("sampled_tokens", 0.0))
            output["coverage_fraction"] = float(sampled / active) if active else 0.0
            output["realized_replacement_fraction"] = float(output.get("realized_replacements", 0.0) / sampled) if sampled else 0.0
            output["adjacency_mean_size"] = float(self._infer_adjacency_total / sampled) if sampled else 0.0
            output["adjacency_min_size"] = float(self._infer_adjacency_min or 0.0)
            output["adjacency_max_size"] = float(self._infer_adjacency_max)
            output["laplace_denominator_z"] = float(self._infer_laplace_denominator or 0.0)
        return output

    def clean_split_check(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
        if not self.args.verify_clean_split or self.args.method not in {"dp_forward", "textobfuscator"}:
            return {}
        if self.args.method == "dp_forward":
            return self.dp_split.assert_clean_equivalence(input_ids[:1].to(self.device), attention_mask[:1].to(self.device))
        return self.to_split.assert_clean_equivalence(input_ids[:1].to(self.device), attention_mask[:1].to(self.device))


def extract_split_features(extractor: FeatureExtractor, texts: Sequence[str], labels: np.ndarray, split_offset: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    output: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), extractor.args.batch_size), desc=f"Features ({extractor.args.method})"):
        batch = texts[start : start + extractor.args.batch_size]
        encoded = extractor.tokenizer(batch, add_special_tokens=True, truncation=True, max_length=extractor.args.max_length, padding=True, return_tensors="pt")
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        if start == 0:
            check = extractor.clean_split_check(ids, mask)
        features = extractor.features(ids, mask, torch.arange(start, start + len(batch), dtype=torch.long) + split_offset)
        output.append(features.numpy())
    return np.concatenate(output), labels, check if texts else {}


def descriptions(method: str) -> Tuple[str, str, str, str]:
    rows = {
        "clean": ("No privacy / frozen LLaMA feature upper bound", "unmodified LLaMA input IDs", "none", "not applicable"),
        "cape": ("CAPE min--max contextual representation plus Laplace DP noise", "LLaMA input-embedding sequence", "CAPE private-attribute adversary/task-head training (IMDb has no provided private attribute)", "CAPE paper noise-scale epsilon; not an end-to-end CAPE claim"),
        "custext": ("CusText top-k exponential mechanism", "all non-special LLaMA BPE input tokens", "original word-vector reconstruction; no compatible local Counter-Fitting vectors are supplied", "CusText per-token exponential-mechanism epsilon"),
        "santext": ("SanText exponential mechanism", "LLaMA BPE tokens; V constructed only from IMDb train raw text", "original word-level/BERT classifier setup", "SanText per-token exponential-mechanism epsilon"),
        "inferdpt": ("InferDPT RANTEXT perturbation", "LLaMA BPE tokens/input embedding table", "remote black-box LLM plus local extraction model; this is server-only RANTEXT probing", "RANTEXT mechanism epsilon; no extractor conditional utility"),
        "dp_forward": ("DP-Forward official analytic-Gaussian forward-pass perturbation", "post-LLaMA decoder hidden-state matrix", "DP-Forward noisy fine-tuning; frozen LLaMA probe has no task fine-tuning", "official release (epsilon, delta) calibration"),
        "textobfuscator": ("TextObfuscator hidden-boundary Laplace perturbation", "after first three LLaMA decoder layers", "original RoBERTa joint task/client/server/prototype training", "TextObfuscator-style Laplace scale 1/epsilon; frozen LLaMA adaptation"),
    }
    return rows[method]


FIELDS = [
    "method", "paper_component", "representation", "omitted_paper_component", "protocol", "dataset", "model", "epsilon", "epsilon_semantics",
    "train_samples", "eval_samples", "max_length", "batch_size", "classifier", "classifier_metadata", "accuracy", "seed", "mapping_metadata", "mechanism_statistics", "split_clean_check",
]


def append_result(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDS})


def main() -> None:
    args = parser().parse_args()
    args.model_path, args.dataset_path = local_path(args.model_path), local_path(args.dataset_path)
    if args.method != "clean" and (args.epsilon is None or args.epsilon <= 0):
        raise ValueError("--epsilon must be positive for a privatizing baseline")
    if args.dry_run:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return
    if args.batch_size < 1 or args.max_length < 2:
        raise ValueError("batch_size must be positive and max_length at least 2")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = make_device(args.device, args.require_cuda)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, local_files_only=args.local_files_only).to(device).eval()
    train_texts, train_labels = dataset_examples(args.dataset_path, args.train_split, args.max_train_samples, args.seed)
    # Legacy Figure-2's ``train_noisy_embeddings.py`` uses seed 42 for both
    # split shuffles and its drivers then select the complete splits.  Keep
    # that exact choice even when a caller requests a small smoke subset.
    eval_texts, eval_labels = dataset_examples(args.dataset_path, args.eval_split, args.max_eval_samples, args.seed)
    # SanText/RANTEXT construct their public vocabulary from the raw task
    # training corpus, independently of the classifier's selected examples.
    # This mirrors the original task-vocabulary step and never accesses IMDb
    # test text or labels.
    mapping_texts, _ = dataset_examples(args.dataset_path, args.train_split, 0, args.seed)
    extractor = FeatureExtractor(args, model, tokenizer, mapping_texts, device)
    if args.prepare_mapping_only:
        if args.method not in {"santext", "inferdpt", "custext"}:
            raise ValueError("--prepare_mapping_only is meaningful only for santext, inferdpt, or custext")
        print(json.dumps({"method": args.method, "epsilon": args.epsilon, "mapping_metadata": extractor.metadata}, ensure_ascii=False, indent=2))
        return
    with torch.inference_mode():
        x_train, y_train, train_check = extract_split_features(extractor, train_texts, train_labels, 0)
        x_eval, y_eval, eval_check = extract_split_features(extractor, eval_texts, eval_labels, 100_000_000)
    if args.classifier == "legacy_enhanced_mlp":
        accuracy, classifier_metadata = legacy_enhanced_mlp_accuracy(x_train, y_train, x_eval, y_eval, args, device)
    else:
        probe = LogisticRegression(max_iter=2000, n_jobs=1, random_state=args.seed)
        probe.fit(x_train, y_train)
        accuracy = float(accuracy_score(y_eval, probe.predict(x_eval)))
        classifier_metadata = {
            "name": "logistic_regression", "max_iter": 2000, "random_state": args.seed,
            "selection": "single_final_fit__diagnostic_not_legacy_figure2",
        }
    component, representation, omitted, epsilon_semantics = descriptions(args.method)
    row = {
        "method": args.method, "paper_component": component, "representation": representation,
        "omitted_paper_component": omitted, "protocol": PROTOCOL, "dataset": "IMDb", "model": args.model_path,
        "epsilon": "" if args.epsilon is None else args.epsilon, "epsilon_semantics": epsilon_semantics, "train_samples": len(train_texts), "eval_samples": len(eval_texts),
        "max_length": args.max_length, "batch_size": args.batch_size, "classifier": args.classifier,
        "classifier_metadata": json.dumps(classifier_metadata, ensure_ascii=False, sort_keys=True), "accuracy": f"{accuracy:.8f}", "seed": args.seed,
        "mapping_metadata": json.dumps(extractor.metadata, ensure_ascii=False, sort_keys=True),
        "mechanism_statistics": json.dumps(extractor.statistics(), ensure_ascii=False, sort_keys=True),
        "split_clean_check": json.dumps({**train_check, **eval_check}, ensure_ascii=False, sort_keys=True),
    }
    append_result(Path(args.output_csv), row)
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
