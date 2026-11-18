#!/usr/bin/env python
"""TextObfuscator-style frozen-LLaMA Figure-3 utility evaluation.

The evaluator preserves the project's WikiText-2/Llama-2-7B PPL and
ROUGE-L definition.  It is not an original TextObfuscator experiment: that
paper jointly trains a split RoBERTa classifier with task-specific prototype
losses, whereas WikiText-2 offers neither that task nor its labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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

from baseline.textobfuscator.split_llama import LlamaTextObfuscatorSplit, TextObfuscatorConfig
from baseline.generation_data import (
    DATASET_LABELS,
    DATASET_NAMES,
    dataset_protocol_label,
    default_split,
    format_generation_record,
    format_generation_texts,
    load_dataset_bundle,
    resolve_local_dataset_path,
)


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
        description="Frozen LLaMA client/server TextObfuscator-style WikiText-2 Figure-3 evaluator."
    )
    parser.add_argument("--model_path", type=str, default="models/Llama-2-7b-chat-hf")
    parser.add_argument("--dataset_name", choices=DATASET_NAMES, default="wikitext")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--client_layers", type=int, default=3, help="Paper's RoBERTa split uses three client layers.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--require_cuda", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--selection_protocol", choices=["current_dpspd", "legacy_dp_generate"], default="current_dpspd")
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--rouge_context_len", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--run_ppl", type=str2bool, default=True)
    parser.add_argument("--run_rouge", type=str2bool, default=True)
    parser.add_argument("--rouge_use_stemmer", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify_clean_split", type=str2bool, default=True)

    parser.add_argument("--mapping_splits", type=str, default="train,validation")
    parser.add_argument("--prototype_count", type=int, default=100, help="Paper sentence-task KMeans count.")
    parser.add_argument("--prototype_max_tokens", type=int, default=4096)
    parser.add_argument("--prototype_kmeans_iters", type=int, default=20)
    parser.add_argument("--prototype_build_batch_size", type=int, default=4)
    parser.add_argument("--prototype_cache", type=str, default=None)
    parser.add_argument("--force_rebuild_prototypes", type=str2bool, default=False)
    parser.add_argument("--build_prototypes_only", type=str2bool, default=False)
    parser.add_argument("--output_csv", type=str, default="outputs/baselines/textobfuscator/wikitext_figure3.csv")
    parser.add_argument("--save_examples_jsonl", type=str, default=None)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def resolve_local_path(value: str) -> str:
    for candidate in (Path(value), REPO_ROOT / value, REPO_ROOT.parent / value):
        if candidate.exists():
            return str(candidate)
    return value


def resolve_device(requested: str) -> torch.device:
    return torch.device(requested) if requested.startswith("cuda") and torch.cuda.is_available() else torch.device("cpu")


def resolve_dtype(name: str):
    if name == "auto":
        return None
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def parse_splits(value: str) -> List[str]:
    splits = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not splits:
        raise ValueError("mapping_splits must contain at least one split")
    return splits


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


class WikiTextDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer, max_length: int) -> None:
        encoded = tokenizer(
            list(texts), add_special_tokens=True, truncation=True, padding="max_length", max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"].to(torch.long).contiguous()
        self.attention_mask = encoded["attention_mask"].to(torch.long).contiguous()

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "sample_index": torch.tensor(index, dtype=torch.long),
        }


def default_prototype_cache(args) -> Path:
    name = Path(args.model_path).name.replace(" ", "_")
    split_tag = "-".join(parse_splits(args.mapping_splits))
    dataset_tag = "wikitext" if args.dataset_name == "wikitext" else args.dataset_name
    return Path("outputs/baselines/textobfuscator/prototypes") / (
        f"{dataset_tag}_{name}_frozen_client{args.client_layers}_{split_tag}_"
        f"k{args.prototype_count}_n{args.prototype_max_tokens}_{args.selection_protocol}_v2.pt"
    )


def prototype_metadata(args, model) -> Dict[str, object]:
    metadata = {
        "schema": "textobfuscator_frozen_llama_prototype_audit_v2",
        "model_path": args.model_path,
        "hidden_size": int(model.config.hidden_size),
        "client_layers": int(args.client_layers),
        "mapping_splits": parse_splits(args.mapping_splits),
        "prototype_count": int(args.prototype_count),
        "prototype_max_tokens": int(args.prototype_max_tokens),
        "cluster_method": "KMeans_on_public_frozen_client_contextual_token_states",
        "paper_difference": "no_task_TFIDF_redivision_and_no_joint_cluster_loss_training",
        "selection_protocol": args.selection_protocol,
    }
    # Preserve the cache schema used by completed WikiText runs, but make
    # instruction-data prototype audits self-identifying.
    if args.dataset_name != "wikitext":
        metadata["dataset_name"] = args.dataset_name
    return metadata


@torch.no_grad()
def collect_public_client_states(raw, tokenizer, runner, args, device: torch.device) -> torch.Tensor:
    collected: List[torch.Tensor] = []
    remaining = int(args.prototype_max_tokens)
    if remaining < args.prototype_count:
        raise ValueError("prototype_max_tokens must be at least prototype_count")
    for split in parse_splits(args.mapping_splits):
        if split not in raw:
            raise KeyError(f"Mapping split {split!r} is not in {list(raw.keys())}")
        dataset = raw[split]
        for start in tqdm(range(0, len(dataset), args.prototype_build_batch_size), desc=f"Client states ({split})"):
            columns = dataset[start : start + args.prototype_build_batch_size]
            row_count = len(next(iter(columns.values()), []))
            texts = []
            for row in range(row_count):
                item = {key: values[row] for key, values in columns.items()}
                text = format_generation_record(args.dataset_name, item, selection_protocol=args.selection_protocol)
                if text is not None:
                    texts.append(text)
            if not texts:
                continue
            encoded = tokenizer(
                texts, add_special_tokens=True, truncation=True, padding=True, max_length=args.max_length,
                return_tensors="pt",
            )
            mask = encoded["attention_mask"].to(device)
            hidden = runner.client_hidden(encoded["input_ids"].to(device), mask)
            active = hidden[mask.bool()]
            if active.numel():
                take = min(remaining, int(active.shape[0]))
                collected.append(active[:take].float().cpu())
                remaining -= take
            if remaining == 0:
                return torch.cat(collected, dim=0)
    if not collected:
        raise ValueError("No active public tokens found for prototype construction")
    result = torch.cat(collected, dim=0)
    if result.shape[0] < args.prototype_count:
        raise ValueError("Not enough public token states for requested prototype_count")
    return result


@torch.no_grad()
def fit_kmeans(vectors_cpu: torch.Tensor, count: int, iterations: int, device: torch.device, seed: int) -> torch.Tensor:
    """Small deterministic KMeans used only to audit the paper component."""

    vectors = vectors_cpu.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    centers = vectors[torch.randperm(vectors.shape[0], device=device, generator=generator)[:count]].clone()
    for _ in tqdm(range(iterations), desc="Frozen-client KMeans"):
        distances = vectors.square().sum(1, keepdim=True) + centers.square().sum(1).unsqueeze(0) - 2 * vectors @ centers.t()
        assignment = distances.argmin(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignment, vectors)
        counts = torch.bincount(assignment, minlength=count).to(centers.dtype).unsqueeze(1)
        centers = torch.where(counts > 0, sums / counts.clamp_min(1.0), centers)
    return centers.cpu()


def build_or_load_prototypes(args, raw, tokenizer, runner, model, device: torch.device) -> Tuple[torch.Tensor, Path, Dict[str, object]]:
    cache_path = Path(args.prototype_cache) if args.prototype_cache else default_prototype_cache(args)
    expected = prototype_metadata(args, model)
    if cache_path.is_file() and not args.force_rebuild_prototypes:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            raise ValueError(f"Cached prototype audit {cache_path} does not match this run: {mismatches}")
        print(f"[TextObfuscator-style] Reusing frozen-client prototype audit: {cache_path}")
        return payload["centers"].float().contiguous(), cache_path, metadata
    vectors = collect_public_client_states(raw, tokenizer, runner, args, device)
    centers = fit_kmeans(vectors, args.prototype_count, args.prototype_kmeans_iters, device, args.seed)
    metadata = dict(expected)
    metadata["collected_public_tokens"] = int(vectors.shape[0])
    metadata["kmeans_iterations"] = int(args.prototype_kmeans_iters)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "centers": centers}, cache_path)
    print(f"[TextObfuscator-style] Saved frozen-client prototype audit: {cache_path}")
    return centers, cache_path, metadata


@torch.no_grad()
def calculate_split_ppl(runner, loader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Project-standard PPL: clean labels after private representation noise."""

    total_nll, total_tokens = 0.0, 0
    runner.model.eval()
    for batch in tqdm(loader, desc="PPL"):
        input_ids, mask, indices = (batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["sample_index"].to(device))
        logits = runner.forward(input_ids, mask, indices, privacy_length=None, perturb=True)
        shift_logits, shift_labels, shift_mask = logits[:, :-1].float(), input_ids[:, 1:], mask[:, 1:].float()
        losses = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), reduction="none")
        losses = losses.view_as(shift_mask)
        total_nll += float((losses * shift_mask).sum().item())
        total_tokens += int(shift_mask.sum().item())
    mean_nll = total_nll / total_tokens if total_tokens else float("nan")
    return (math.exp(mean_nll) if total_tokens else float("nan")), {"ppl_scored_tokens": total_tokens, "mean_nll": mean_nll}


@torch.no_grad()
def calculate_split_rouge(runner, tokenizer, loader, device: torch.device, context_len: int, gen_len: int, *, use_stemmer: bool) -> float:
    """Project-standard ROUGE-L with a split-model greedy decoder.

    ``generate(inputs_embeds=...)`` cannot substitute an intermediate hidden
    state.  An explicit layer-wise KV cache retains one fixed noised 32-token
    private prefix; newly generated public tokens are not re-privatized.
    """

    try:
        import evaluate
        rouge = evaluate.load("rouge", keep_in_memory=True)
    except Exception as exc:
        print(f"[Warning] ROUGE unavailable: {exc}")
        return 0.0
    predictions: List[str] = []
    references: List[str] = []
    runner.model.eval()
    for batch in tqdm(loader, desc="ROUGE"):
        input_ids, indices = batch["input_ids"].to(device), batch["sample_index"].to(device)
        if input_ids.shape[1] <= context_len:
            continue
        current_ids = input_ids[:, :context_len].clone()
        current_mask = torch.ones_like(current_ids, dtype=torch.long)
        cache = None
        generated: List[torch.Tensor] = []
        for _ in range(gen_len):
            logits, cache = runner.forward_cached(
                current_ids, current_mask, indices, past_key_values=cache, perturb=True
            )
            next_token = logits[:, -1].argmax(dim=-1)
            generated.append(next_token)
            # Cached LLaMA expects the *whole* attention mask while it only
            # receives the newly generated final token as input.
            current_ids = next_token.unsqueeze(1)
            current_mask = torch.cat([current_mask, torch.ones_like(next_token).unsqueeze(1)], dim=1)
        pred_texts = tokenizer.batch_decode(torch.stack(generated, 1).cpu(), skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(input_ids[:, context_len : context_len + gen_len].cpu(), skip_special_tokens=True)
        for prediction, reference in zip(pred_texts, ref_texts):
            if reference.strip():
                predictions.append(prediction)
                references.append(reference)
    return float(rouge.compute(predictions=predictions, references=references, use_stemmer=use_stemmer).get("rougeL", 0.0)) if predictions else 0.0


@torch.no_grad()
def prototype_audit(runner, loader, device: torch.device, centers_cpu: torch.Tensor) -> Dict[str, float]:
    """Audit public frozen KMeans; it intentionally does not change logits."""

    centers = centers_cpu.to(device=device, dtype=torch.float32)
    total_distance, total_tokens = 0.0, 0
    occupied = torch.zeros(centers.shape[0], device=device, dtype=torch.bool)
    for batch in tqdm(loader, desc="Prototype audit"):
        input_ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        vectors = runner.client_hidden(input_ids, mask)[mask.bool()].float()
        if not vectors.numel():
            continue
        distances = (vectors.square().sum(1, keepdim=True) + centers.square().sum(1).unsqueeze(0) - 2 * vectors @ centers.t()).clamp_min_(0)
        nearest, assignment = distances.min(1)
        total_distance += float(nearest.sqrt().sum().item())
        total_tokens += int(vectors.shape[0])
        occupied[assignment.unique()] = True
    return {
        "prototype_audit_tokens": total_tokens,
        "prototype_mean_nearest_l2": total_distance / total_tokens if total_tokens else float("nan"),
        "prototype_nonempty_on_eval": int(occupied.sum().item()),
    }


RESULT_FIELDS = [
    "method", "paper_component", "omitted_paper_component", "representation", "dataset", "model", "epsilon",
    "epsilon_semantics", "selection_protocol", "eval_samples", "max_length", "client_layers", "server_layers",
    "rouge_context_len", "max_new_tokens", "rouge_use_stemmer", "ppl_protocol", "rouge_protocol", "active_tokens",
    "privatized_coordinates", "laplace_scale", "laplace_endpoint_guard", "prototype_cache", "prototype_count",
    "prototype_public_tokens", "prototype_mean_nearest_l2", "prototype_nonempty_on_eval", "clean_split_max_abs_logit_delta",
    "clean_split_mean_abs_logit_delta", "ppl_scored_tokens", "mean_nll", "ppl", "rouge_l", "seed",
]


def append_row(path: str | Path, row: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as handle:
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
    if args.max_length < 2 or not 1 <= args.rouge_context_len < args.max_length:
        raise ValueError("max_length must be >=2 and rouge_context_len must be in [1, max_length)")
    if min(args.max_new_tokens, args.batch_size, args.prototype_build_batch_size, args.prototype_kmeans_iters) < 1:
        raise ValueError("batch sizes, generation length and KMeans iterations must be positive")
    if args.dry_run:
        print("[Dry Run]\n" + json.dumps(vars(args), indent=2, sort_keys=True))
        print("Frozen LLaMA TextObfuscator-style adapter, not original jointly trained RoBERTa TextObfuscator.")
        return
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type != "cuda" and args.device.startswith("cuda"):
        if args.require_cuda:
            raise RuntimeError(f"CUDA device {args.device} is unavailable; refusing CPU benchmark fallback.")
        print(f"[Warning] CUDA device {args.device} unavailable; using CPU by explicit opt-in.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    raw = load_dataset_bundle(args.dataset_path)
    if not hasattr(raw, "keys") or args.eval_split not in raw:
        raise ValueError(f"Expected a DatasetDict containing split {args.eval_split!r}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=args.local_files_only, torch_dtype=resolve_dtype(args.dtype)).to(device)
    model.eval()
    runner = LlamaTextObfuscatorSplit(model, TextObfuscatorConfig(args.epsilon, args.client_layers, args.seed))
    centers, cache_path, prototype_meta = build_or_load_prototypes(args, raw, tokenizer, runner, model, device)
    if args.build_prototypes_only:
        print(f"[TextObfuscator-style] Prototype construction complete: {cache_path}")
        return
    texts = format_generation_texts(
        args.dataset_name, raw[args.eval_split], args.max_samples, selection_protocol=args.selection_protocol
    )
    dataset = WikiTextDataset(texts, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    first = next(iter(loader))
    split_check = runner.assert_clean_equivalence(first["input_ids"][:1].to(device), first["attention_mask"][:1].to(device)) if args.verify_clean_split else {"clean_split_max_abs_logit_delta": float("nan"), "clean_split_mean_abs_logit_delta": float("nan")}
    audit = prototype_audit(runner, loader, device, centers)
    ppl, ppl_stats = calculate_split_ppl(runner, loader, device) if args.run_ppl else (float("nan"), {})
    rouge_l = calculate_split_rouge(runner, tokenizer, loader, device, args.rouge_context_len, args.max_new_tokens, use_stemmer=args.rouge_use_stemmer) if args.run_rouge else float("nan")
    active_tokens = int(dataset.attention_mask.sum().item())
    row = {
        "method": "TextObfuscator-style-frozen-LLaMA-split",
        "paper_component": "client_server_split__all_active_boundary_hidden_states_plus_iid_Laplace(1/epsilon)",
        "omitted_paper_component": "joint_client_server_task_training__prototype_close_away_losses__task_TFIDF_redivision",
        "representation": "LLaMA_layer3_contextual_hidden_state__not_original_RoBERTa_layer3",
        "dataset": DATASET_LABELS[args.dataset_name], "model": args.model_path, "epsilon": args.epsilon,
        "epsilon_semantics": "original_code_Laplace_scale_1_over_eps_hyperparameter__not_formal_end_to_end_DP_claim",
        "selection_protocol": dataset_protocol_label(args.dataset_name, args.selection_protocol), "eval_samples": len(dataset), "max_length": args.max_length,
        "client_layers": args.client_layers, "server_layers": len(model.model.layers) - args.client_layers,
        "rouge_context_len": args.rouge_context_len, "max_new_tokens": args.max_new_tokens, "rouge_use_stemmer": args.rouge_use_stemmer,
        "ppl_protocol": "DP-SPD_full_sequence_clean_labels_after_all_active_client_boundary_states_are_noised",
        "rouge_protocol": "DP-SPD_first_32_private_client_boundary_states_to_clean_next_32_tokens_greedy",
        "active_tokens": active_tokens, "privatized_coordinates": active_tokens * runner.hidden_size,
        "laplace_scale": f"{1 / args.epsilon:.6f}", "laplace_endpoint_guard": "float32_inverse_cdf_endpoint_only",
        "prototype_cache": str(cache_path), "prototype_count": int(centers.shape[0]),
        "prototype_public_tokens": prototype_meta.get("collected_public_tokens", "cached"),
        "prototype_mean_nearest_l2": f"{audit['prototype_mean_nearest_l2']:.6f}", "prototype_nonempty_on_eval": audit["prototype_nonempty_on_eval"],
        "clean_split_max_abs_logit_delta": f"{split_check['clean_split_max_abs_logit_delta']:.8f}",
        "clean_split_mean_abs_logit_delta": f"{split_check['clean_split_mean_abs_logit_delta']:.8f}",
        "ppl_scored_tokens": ppl_stats.get("ppl_scored_tokens", 0), "mean_nll": f"{ppl_stats.get('mean_nll', float('nan')):.6f}",
        "ppl": f"{ppl:.6f}", "rouge_l": f"{rouge_l:.6f}", "seed": args.seed,
    }
    append_row(args.output_csv, row)
    if args.save_examples_jsonl:
        examples_path = Path(args.save_examples_jsonl); examples_path.parent.mkdir(parents=True, exist_ok=True)
        with examples_path.open("w", encoding="utf-8") as handle:
            for index in range(min(4, len(dataset))):
                item = dataset[index]; active = item["attention_mask"].bool()
                handle.write(json.dumps({"sample_index": index, "clean_text": tokenizer.decode(item["input_ids"][active], skip_special_tokens=False), "active_tokens": int(active.sum().item()), "boundary": f"after_llama_layer_{args.client_layers}"}, ensure_ascii=False) + "\n")
    print(f"[Result] epsilon={args.epsilon:g} ppl={ppl:.6f} rouge_l={rouge_l:.6f} scale={1 / args.epsilon:.6f} active_tokens={active_tokens}")
    print(f"[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()
