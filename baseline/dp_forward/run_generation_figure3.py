#!/usr/bin/env python
"""DP-Forward-style frozen-LLaMA evaluation on the DP-SPD Figure-3 protocol.

Original DP-Forward fine-tunes BERT classifiers under forward-pass noise.  It
does not report autoregressive language-model PPL/ROUGE.  This runner ports
the public release's inference operator (sequence-Frobenius clipping plus
analytically calibrated i.i.d. Gaussian noise) to a frozen LLaMA deep-layer
boundary, then uses the project's clean-label PPL and 32-to-32 ROUGE-L setup.
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

from baseline.dp_forward.noise import DPForwardConfig
from baseline.dp_forward.split_llama import LlamaDPForwardSplit
from baseline.generation_data import (
    DATASET_LABELS,
    DATASET_NAMES,
    dataset_protocol_label,
    default_split,
    format_generation_texts,
    load_dataset_bundle,
    resolve_local_dataset_path,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    normal = str(value).strip().lower()
    if normal in {"true", "1", "yes", "y"}:
        return True
    if normal in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="DP-Forward-style frozen-LLaMA WikiText-2 Figure-3 evaluator.")
    value.add_argument("--model_path", type=str, default="models/Llama-2-7b-chat-hf")
    value.add_argument("--dataset_name", choices=DATASET_NAMES, default="wikitext")
    value.add_argument("--dataset_path", type=str, default=None)
    value.add_argument("--eval_split", type=str, default=None)
    value.add_argument("--epsilon", type=float, required=True, help="Sequence-level (epsilon, delta)-LDP epsilon.")
    value.add_argument("--delta", type=float, default=1e-5, help="Official release default.")
    value.add_argument("--norm_c", type=float, default=1.0, help="Official release default clipping radius.")
    value.add_argument(
        "--noise_layer", type=int, default=28,
        help="Zero-based LLaMA layer output to perturb; 28 maps BERT release layer 10's ~92%% depth.",
    )
    value.add_argument("--device", type=str, default="cuda:0")
    value.add_argument("--require_cuda", type=str2bool, default=True)
    value.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    value.add_argument("--local_files_only", type=str2bool, default=True)
    value.add_argument("--selection_protocol", choices=["current_dpspd", "legacy_dp_generate"], default="current_dpspd")
    value.add_argument("--max_samples", type=int, default=300)
    value.add_argument("--max_length", type=int, default=256)
    value.add_argument("--batch_size", type=int, default=4)
    value.add_argument("--rouge_context_len", type=int, default=32)
    value.add_argument("--max_new_tokens", type=int, default=32)
    value.add_argument("--run_ppl", type=str2bool, default=True)
    value.add_argument("--run_rouge", type=str2bool, default=True)
    value.add_argument("--rouge_use_stemmer", type=str2bool, default=True)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--verify_clean_split", type=str2bool, default=True)
    value.add_argument("--output_csv", type=str, default="outputs/baselines/dp_forward/wikitext_figure3.csv")
    value.add_argument("--save_examples_jsonl", type=str, default=None)
    value.add_argument("--dry_run", type=str2bool, default=False)
    return value


def resolve_path(value: str) -> str:
    for candidate in (Path(value), REPO_ROOT / value, REPO_ROOT.parent / value):
        if candidate.exists():
            return str(candidate)
    return value


def device_for(requested: str) -> torch.device:
    return torch.device(requested) if requested.startswith("cuda") and torch.cuda.is_available() else torch.device("cpu")


def dtype_for(value: str):
    if value == "auto":
        return None
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[value]


def collect_wikitext_lines(dataset, max_samples: int, protocol: str) -> List[str]:
    texts: List[str] = []
    for item in tqdm(dataset, desc=f"Formatting WikiText-2 ({protocol})"):
        raw = str(item.get("text", ""))
        if protocol == "legacy_dp_generate":
            text, keep = raw, len(raw) > 50
        else:
            text, keep = raw.strip(), len(raw.strip().split()) > 8
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


@torch.no_grad()
def calculate_ppl(runner: LlamaDPForwardSplit, loader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Same numerical PPL definition as DP-SPD's generation evaluator."""

    total_nll, scored_tokens = 0.0, 0
    for batch in tqdm(loader, desc="PPL"):
        ids, mask, index = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["sample_index"].to(device)
        logits = runner.forward(ids, mask, index, perturb=True)
        shift_logits, shift_ids, shift_mask = logits[:, :-1].float(), ids[:, 1:], mask[:, 1:].float()
        losses = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_ids.reshape(-1), reduction="none")
        total_nll += float((losses.view_as(shift_mask) * shift_mask).sum().item())
        scored_tokens += int(shift_mask.sum().item())
    mean_nll = total_nll / scored_tokens if scored_tokens else float("nan")
    return (math.exp(mean_nll) if scored_tokens else float("nan")), {"ppl_scored_tokens": scored_tokens, "mean_nll": mean_nll}


@torch.no_grad()
def calculate_rouge(
    runner: LlamaDPForwardSplit, tokenizer, loader, device: torch.device, context_len: int, gen_len: int, *, use_stemmer: bool
) -> float:
    """Same Evaluate ROUGE-L/stemmer definition as DP-SPD Figure-3.

    A layer-wise KV cache is used because Hugging Face ``generate`` cannot
    replace a deep internal representation.  The Gaussian perturbation is
    sampled once on the private prompt, exactly like normal cached generation
    with a noised prefix; generated tokens are not private user inputs.
    """

    try:
        import evaluate
        rouge = evaluate.load("rouge", keep_in_memory=True)
    except Exception as exc:
        print(f"[Warning] ROUGE unavailable: {exc}")
        return 0.0
    predictions: List[str] = []
    references: List[str] = []
    for batch in tqdm(loader, desc="ROUGE"):
        ids, index = batch["input_ids"].to(device), batch["sample_index"].to(device)
        if ids.shape[1] <= context_len:
            continue
        current_ids = ids[:, :context_len].clone()
        current_mask = torch.ones_like(current_ids, dtype=torch.long)
        cache = None
        generated: List[torch.Tensor] = []
        for _ in range(gen_len):
            logits, cache = runner.forward_cached(current_ids, current_mask, index, past_key_values=cache, perturb=True)
            next_token = logits[:, -1].argmax(dim=-1)
            generated.append(next_token)
            current_ids = next_token.unsqueeze(1)
            current_mask = torch.cat([current_mask, torch.ones_like(next_token).unsqueeze(1)], dim=1)
        predicted = tokenizer.batch_decode(torch.stack(generated, dim=1).cpu(), skip_special_tokens=True)
        reference = tokenizer.batch_decode(ids[:, context_len : context_len + gen_len].cpu(), skip_special_tokens=True)
        for prediction, target in zip(predicted, reference):
            if target.strip():
                predictions.append(prediction)
                references.append(target)
    return float(rouge.compute(predictions=predictions, references=references, use_stemmer=use_stemmer).get("rougeL", 0.0)) if predictions else 0.0


RESULT_FIELDS = [
    "method", "paper_component", "omitted_paper_component", "representation", "dataset", "model", "epsilon", "delta",
    "epsilon_semantics", "selection_protocol", "eval_samples", "max_length", "noise_layer", "client_layers", "server_layers",
    "norm_c", "noise_std", "noise_distribution", "clipping_scope", "rouge_context_len", "max_new_tokens", "rouge_use_stemmer",
    "ppl_protocol", "rouge_protocol", "active_tokens", "privatized_coordinates", "clean_split_max_abs_logit_delta",
    "clean_split_mean_abs_logit_delta", "ppl_scored_tokens", "mean_nll", "ppl", "rouge_l", "seed",
]


def append_row(path: str | Path, row: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in RESULT_FIELDS})


def main() -> None:
    args = parser().parse_args()
    args.model_path = resolve_path(args.model_path)
    args.dataset_path = resolve_local_dataset_path(args.dataset_path, REPO_ROOT, args.dataset_name)
    args.eval_split = args.eval_split or default_split(args.dataset_name)
    if args.max_length < 2 or not 1 <= args.rouge_context_len < args.max_length:
        raise ValueError("max_length must be >=2 and rouge_context_len must be in [1, max_length)")
    if min(args.batch_size, args.max_new_tokens) < 1:
        raise ValueError("batch_size and max_new_tokens must be positive")
    if args.dry_run:
        print("[Dry Run]\n" + json.dumps(vars(args), indent=2, sort_keys=True))
        print("This is a frozen-LLaMA DP-Forward-style adapter, not original noisy BERT fine-tuning.")
        return
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = device_for(args.device)
    if device.type != "cuda" and args.device.startswith("cuda"):
        if args.require_cuda:
            raise RuntimeError(f"CUDA device {args.device} unavailable; refusing CPU benchmark fallback.")
        print(f"[Warning] CUDA device {args.device} unavailable; CPU used by explicit opt-in.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    raw = load_dataset_bundle(args.dataset_path)
    if not hasattr(raw, "keys") or args.eval_split not in raw:
        raise ValueError(f"Expected DatasetDict containing {args.eval_split!r}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=args.local_files_only, torch_dtype=dtype_for(args.dtype)
    ).to(device)
    model.eval()
    config = DPForwardConfig(args.epsilon, args.delta, args.norm_c, args.noise_layer, args.seed)
    runner = LlamaDPForwardSplit(model, config)
    texts = format_generation_texts(
        args.dataset_name, raw[args.eval_split], args.max_samples, selection_protocol=args.selection_protocol
    )
    dataset = WikiTextDataset(texts, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    first = next(iter(loader))
    check = runner.assert_clean_equivalence(first["input_ids"][:1].to(device), first["attention_mask"][:1].to(device)) if args.verify_clean_split else {"clean_split_max_abs_logit_delta": float("nan"), "clean_split_mean_abs_logit_delta": float("nan")}
    ppl, ppl_stats = calculate_ppl(runner, loader, device) if args.run_ppl else (float("nan"), {})
    rouge_l = calculate_rouge(runner, tokenizer, loader, device, args.rouge_context_len, args.max_new_tokens, use_stemmer=args.rouge_use_stemmer) if args.run_rouge else float("nan")
    active_tokens = int(dataset.attention_mask.sum().item())
    row = {
        "method": "DP-Forward-style-frozen-LLaMA-deep-boundary",
        "paper_component": "official_release_sequence_Frobenius_clip_then_analytic_Gaussian_iid_forward_noise",
        "omitted_paper_component": "joint_noisy_BERT_finetuning__classification_task_head__optional_label_randomized_response",
        "representation": f"LLaMA_post_decoder_layer_{args.noise_layer}_contextual_matrix__not_original_BERT_encoder",
        "dataset": DATASET_LABELS[args.dataset_name], "model": args.model_path, "epsilon": args.epsilon, "delta": args.delta,
        "epsilon_semantics": "official_release_local_DP_aGM_epsilon_delta__frozen_adapter_not_original_finetuned_guarantee",
        "selection_protocol": dataset_protocol_label(args.dataset_name, args.selection_protocol), "eval_samples": len(dataset), "max_length": args.max_length,
        "noise_layer": args.noise_layer, "client_layers": runner.client_layer_count,
        "server_layers": len(model.model.layers) - runner.client_layer_count, "norm_c": args.norm_c,
        "noise_std": f"{config.noise_std:.9f}", "noise_distribution": "iid_Normal_0_noise_std__official_release_add_noise",
        "clipping_scope": "per_sequence_active_token_matrix_Frobenius_norm",
        "rouge_context_len": args.rouge_context_len, "max_new_tokens": args.max_new_tokens, "rouge_use_stemmer": args.rouge_use_stemmer,
        "ppl_protocol": "DP-SPD_full_sequence_clean_labels_after_DP-Forward_style_noised_deep_representation",
        "rouge_protocol": "DP-SPD_noised_32_token_deep_representation_to_clean_next_32_tokens_cached_greedy",
        "active_tokens": active_tokens, "privatized_coordinates": active_tokens * runner.hidden_size,
        "clean_split_max_abs_logit_delta": f"{check['clean_split_max_abs_logit_delta']:.8f}",
        "clean_split_mean_abs_logit_delta": f"{check['clean_split_mean_abs_logit_delta']:.8f}",
        "ppl_scored_tokens": ppl_stats.get("ppl_scored_tokens", 0), "mean_nll": f"{ppl_stats.get('mean_nll', float('nan')):.6f}",
        "ppl": f"{ppl:.6f}", "rouge_l": f"{rouge_l:.6f}", "seed": args.seed,
    }
    append_row(args.output_csv, row)
    if args.save_examples_jsonl:
        path = Path(args.save_examples_jsonl); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for index in range(min(4, len(dataset))):
                item, active = dataset[index], dataset[index]["attention_mask"].bool()
                handle.write(json.dumps({"sample_index": index, "clean_text": tokenizer.decode(item["input_ids"][active], skip_special_tokens=False), "active_tokens": int(active.sum()), "noise_boundary": f"post_llama_decoder_layer_{args.noise_layer}"}, ensure_ascii=False) + "\n")
    print(f"[Result] epsilon={args.epsilon:g} delta={args.delta:g} std={config.noise_std:.6f} ppl={ppl:.6f} rouge_l={rouge_l:.6f}")
    print(f"[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()
