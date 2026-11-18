# -*- coding: utf-8 -*-
"""
Grouped EIA benchmark for DP-SPD-style dummy suffix strategies.

This script keeps the same experimental grid as tools/run_mia_dummy_strategy_benchmark.py:
- same dummy strategies
- same dummy ratios
- same k values
- same target epsilon list and local-epsilon mapping
- same WikiText fixed eval manifest logic

It replaces MIA/PPL scoring with an embedding inversion attack (EIA):
noisy embeddings are mapped back to the nearest vocabulary embedding by L2 distance,
then the benchmark reports true private-suffix token recovery plus grouped EIA ASR.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dpspd.util.utils import load_local_causal_lm, str2bool
from attack.mia_dummy_strategy_benchmark import (
    LOCAL_EPS_MAPPING,
    STRATEGY_LABELS,
    GroupedMIASequenceDataset,
    atomic_write_json,
    build_corpus_cache,
    build_strategy_group_items,
    load_or_create_eval_manifest,
    parse_float_csv,
    parse_int_csv,
    parse_str_csv,
)


class GroupedEIASequenceDataset(GroupedMIASequenceDataset):
    """Same noising logic as MIA, with layout metadata exposed for EIA metrics."""

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        item = self.items[idx]
        out["real_len"] = int(item["real_len"])
        out["prefix_len"] = int(item["prefix_len"])
        return out


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark EIA token recovery for the same DP-SPD dummy-strategy grid "
            "used by the MIA benchmark. PPL/ROUGE are not computed."
        )
    )
    parser.add_argument(
        "--continuation_model",
        type=str,
        default="models/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--continuation_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--attack_model",
        type=str,
        default="models/Llama-2-7b-chat-hf",
    )
    parser.add_argument(
        "--attack_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--dataset_path", type=str, default="data/imdb")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--min_len", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--num_groups", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--eval_manifest_path", type=str, default=None)
    parser.add_argument("--token_pool_texts", type=int, default=0)

    # Keep this grid identical to the MIA benchmark defaults.
    parser.add_argument(
        "--strategies",
        type=str,
        default="dp_spd_generation,random_sampled_suffix,random_token_suffix",
    )
    parser.add_argument("--ratios", type=str, default="0.2,0.5,0.8")
    parser.add_argument("--ks", type=str, default="9,19,29")
    parser.add_argument("--target_eps_list", type=str, default="1.5,3.0")
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # EIA-specific runtime knobs.
    parser.add_argument("--eia_batch_size", type=int, default=32)
    parser.add_argument(
        "--inversion_chunk_size",
        type=int,
        default=1024,
        help="Number of flattened tokens to invert per cdist call. Lower this if GPU memory is tight.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eia",
    )
    return parser


def build_output_paths(output_dir: Path) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_eps_path = output_dir / "eia_dummy_strategy_per_eps.csv"
    avg_path = output_dir / "eia_dummy_strategy_avg_over_eps.csv"
    config_path = output_dir / "eia_dummy_strategy_benchmark_config.json"
    return per_eps_path, avg_path, config_path


def batched_embedding_inversion(
    noisy_embs: torch.Tensor,
    vocab_embs: torch.Tensor,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map each noisy embedding vector to the nearest vocabulary embedding by L2 distance.

    noisy_embs: [B, L, D]
    vocab_embs: [V, D]
    returns:
    - [B, L] predicted token ids
    - [B, L] distance to the nearest vocabulary embedding
    """
    B, L, D = noisy_embs.shape
    flat_noisy = noisy_embs.reshape(B * L, D).float()
    vocab_embs = vocab_embs.float()

    pred_chunks: List[torch.Tensor] = []
    dist_chunks: List[torch.Tensor] = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, flat_noisy.size(0), chunk_size):
        end = min(start + chunk_size, flat_noisy.size(0))
        # [chunk, vocab]
        dists = torch.cdist(flat_noisy[start:end], vocab_embs, p=2)
        min_dists, pred_ids = torch.min(dists, dim=-1)
        pred_chunks.append(pred_ids.detach().cpu())
        dist_chunks.append(min_dists.detach().cpu())
        del dists

    pred_ids_flat = torch.cat(pred_chunks, dim=0)
    min_dists_flat = torch.cat(dist_chunks, dim=0)
    return pred_ids_flat.view(B, L), min_dists_flat.view(B, L)


def empty_meter() -> Dict[str, int]:
    return {"correct": 0, "total": 0}


def update_meter(meter: Dict[str, int], correct_mask: torch.Tensor, select_mask: torch.Tensor) -> None:
    selected = select_mask.bool()
    meter["correct"] += int((correct_mask & selected).sum().item())
    meter["total"] += int(selected.sum().item())


def rate(meter: Dict[str, int]) -> float:
    return 100.0 * meter["correct"] / meter["total"] if meter["total"] > 0 else 0.0


def compute_eia_metrics(
    dataloader: DataLoader,
    vocab_embs: torch.Tensor,
    device: str,
    inversion_chunk_size: int,
) -> Dict[str, float]:
    """
    Compute the two EIA metrics used by the grouped dummy benchmark.

    - Real suffix token recovery measures direct embedding leakage on the
      true private suffix tokens.
    - Group EIA ASR chooses one candidate per group using only EIA confidence:
      the row with the lowest mean nearest-vocabulary distance on suffix tokens
      is predicted as real.
    """
    real_suffix_meter = empty_meter()
    group_candidates: Dict[int, List[Dict[str, Any]]] = {}

    for batch in tqdm(dataloader, desc="Running EIA embedding inversion"):
        input_ids_cpu = batch["input_ids"].long()
        attn_mask_cpu = batch["attention_mask"].long()
        noisy_embs = batch["noisy_embeddings"].to(device)

        pred_ids, min_dists = batched_embedding_inversion(
            noisy_embs=noisy_embs,
            vocab_embs=vocab_embs,
            chunk_size=inversion_chunk_size,
        )

        correct = pred_ids.eq(input_ids_cpu)
        B, L = input_ids_cpu.shape
        positions = torch.arange(L).unsqueeze(0).expand(B, L)
        real_len = batch["real_len"].long().unsqueeze(1)
        prefix_len = batch["prefix_len"].long().unsqueeze(1)

        valid_mask = attn_mask_cpu.bool() & (positions < real_len)
        suffix_mask = valid_mask & (positions >= prefix_len) & (positions < real_len)

        is_real = batch["is_real"].bool().unsqueeze(1)
        real_row_mask = is_real.expand(B, L)

        update_meter(
            real_suffix_meter,
            correct_mask=correct,
            select_mask=real_row_mask & suffix_mask,
        )

        suffix_counts = suffix_mask.sum(dim=1).clamp(min=1)
        suffix_scores = (min_dists * suffix_mask.float()).sum(dim=1) / suffix_counts
        for row_idx in range(B):
            group_id = int(batch["group_id"][row_idx].item())
            candidates = group_candidates.setdefault(group_id, [])
            candidates.append(
                {
                    "score": float(suffix_scores[row_idx].item()),
                    "order": len(candidates),
                    "is_real": bool(batch["is_real"][row_idx].item()),
                }
            )

        del noisy_embs, pred_ids, min_dists

    group_total = 0
    group_correct = 0
    for candidates in group_candidates.values():
        if not any(candidate["is_real"] for candidate in candidates):
            continue
        predicted = min(candidates, key=lambda candidate: (candidate["score"], candidate["order"]))
        group_correct += int(predicted["is_real"])
        group_total += 1

    group_asr = 100.0 * group_correct / group_total if group_total > 0 else 0.0
    return {
        "real_suffix": rate(real_suffix_meter),
        "real_suffix_tokens": real_suffix_meter["total"],
        "group_asr": group_asr,
        "group_correct": group_correct,
        "group_total": group_total,
    }


def main():
    args = get_parser().parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"

    strategies = parse_str_csv(args.strategies)
    ratios = parse_float_csv(args.ratios)
    ks = parse_int_csv(args.ks)
    target_eps_list = parse_float_csv(args.target_eps_list)

    for strategy in strategies:
        if strategy not in STRATEGY_LABELS:
            raise ValueError(f"Unsupported strategy '{strategy}'. Supported: {sorted(STRATEGY_LABELS)}")
    for target_eps in target_eps_list:
        if target_eps not in LOCAL_EPS_MAPPING:
            raise ValueError(f"Unsupported target_eps={target_eps}. Supported: {sorted(LOCAL_EPS_MAPPING)}")
    for k in ks:
        for target_eps in target_eps_list:
            if k not in LOCAL_EPS_MAPPING[target_eps]:
                raise ValueError(
                    f"Missing local-eps mapping for target_eps={target_eps}, k={k}. "
                    f"Available k values: {sorted(LOCAL_EPS_MAPPING[target_eps])}"
                )

    output_dir = Path(args.output_dir)
    per_eps_path, avg_path, config_path = build_output_paths(output_dir)

    print("Loading continuation model (Qwen-side, for DP-SPD dummy suffix generation)...")
    continuation_tokenizer, continuation_model, continuation_dtype = load_local_causal_lm(
        model_path=args.continuation_model,
        device=args.device,
        dtype_preference=args.continuation_dtype,
        local_files_only=args.local_files_only,
    )
    print(f"[Continuation Model] {os.path.basename(args.continuation_model.rstrip('/'))} | dtype={continuation_dtype}")

    print("Loading attack model and vocabulary embeddings (server-side white-box EIA)...")
    attack_tokenizer, attack_model, attack_dtype = load_local_causal_lm(
        model_path=args.attack_model,
        device=args.device,
        dtype_preference=args.attack_dtype,
        local_files_only=args.local_files_only,
    )
    print(f"[Attack Model] {os.path.basename(args.attack_model.rstrip('/'))} | dtype={attack_dtype}")
    vocab_embs = attack_model.get_input_embeddings().weight.detach().to(args.device)

    manifest, manifest_path, corpus_texts = load_or_create_eval_manifest(args)
    real_texts = [item["text"] for item in manifest["samples"]]
    print(
        f"[Data Ready] real_groups={len(real_texts)} | "
        f"split={args.dataset_split} | eval_manifest={manifest_path}"
    )

    print("Building reusable WikiText corpus cache for suffix sampling and token sampling...")
    corpus_cache, token_pool = build_corpus_cache(
        corpus_texts=corpus_texts,
        tokenizer=attack_tokenizer,
        max_length=args.max_length,
        token_pool_texts=args.token_pool_texts,
        seed=args.seed,
    )
    print(f"[Corpus Cache] cached_texts={len(corpus_cache)} | token_pool_size={len(token_pool)}")

    atomic_write_json(
        config_path,
        {
            "args": vars(args),
            "strategy_labels": STRATEGY_LABELS,
            "local_eps_mapping": LOCAL_EPS_MAPPING,
            "eval_manifest_path": str(manifest_path),
            "continuation_dtype": str(continuation_dtype),
            "attack_dtype": str(attack_dtype),
            "metrics": {
                "Real Suffix Token Recovery (%)": "Nearest-neighbor token recovery on true private suffix tokens.",
                "Group EIA ASR (%)": (
                    "Grouped 1-real-plus-k-dummy attack. The candidate with the lowest "
                    "mean nearest-vocabulary L2 distance on suffix tokens is predicted as real."
                ),
            },
        },
    )

    per_eps_fields = [
        "Strategy",
        "Dummy Ratio",
        "Dummy (k)",
        "Target Eps",
        "Required Local Eps",
        "Real Suffix Token Recovery (%)",
        "Group EIA ASR (%)",
        "Random Baseline (%)",
        "Real Suffix Tokens",
        "Groups Evaluated",
        "Eval Manifest",
    ]
    avg_fields = [
        "Strategy",
        "Dummy Ratio",
        "Dummy (k)",
        "Average Real Suffix Token Recovery (%)",
        "Average Group EIA ASR (%)",
        "Random Baseline (%)",
        "Averaged Target Eps",
        "Eval Manifest",
    ]

    with open(per_eps_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=per_eps_fields).writeheader()
    with open(avg_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=avg_fields).writeheader()

    print("\n" + "=" * 80)
    print("Starting EIA dummy-strategy benchmark. PPL/ROUGE are intentionally skipped.")
    print("=" * 80)

    metric_keys = ["real_suffix", "group_asr"]

    for strategy in strategies:
        for ratio in ratios:
            for k in ks:
                print(
                    f"\n>>> Strategy={strategy} | Ratio={ratio:.1f} | k={k} | "
                    + (
                        "building privatized-prefix candidates per eps..."
                        if strategy == "dp_spd_generation"
                        else "building grouped candidates once for all eps..."
                    )
                )

                noisy_dataset = None
                if strategy != "dp_spd_generation":
                    group_items = build_strategy_group_items(
                        strategy=strategy,
                        real_texts=real_texts,
                        corpus_cache=corpus_cache,
                        token_pool=token_pool,
                        continuation_tokenizer=continuation_tokenizer,
                        continuation_model=continuation_model,
                        attack_tokenizer=attack_tokenizer,
                        max_length=args.max_length,
                        ratio=ratio,
                        k=k,
                        seed=args.seed,
                    )

                    noisy_dataset = GroupedEIASequenceDataset(
                        items=group_items,
                        pretrained_embeddings=attack_model.get_input_embeddings(),
                        hidden_size=attack_model.config.hidden_size,
                        clip_norm=args.clip_norm,
                        mechanism="Laplace",
                        epsilon_private=1.0,
                        epsilon_non_sensitive=1.0,
                        seed=args.seed,
                    )

                metric_history: Dict[str, List[float]] = {key: [] for key in metric_keys}
                for target_eps in target_eps_list:
                    local_eps = LOCAL_EPS_MAPPING[target_eps][k]
                    noise_seed = int(target_eps * 1000 + k * 10 + ratio * 100)
                    if strategy == "dp_spd_generation":
                        group_items = build_strategy_group_items(
                            strategy=strategy,
                            real_texts=real_texts,
                            corpus_cache=corpus_cache,
                            token_pool=token_pool,
                            continuation_tokenizer=continuation_tokenizer,
                            continuation_model=continuation_model,
                            attack_tokenizer=attack_tokenizer,
                            max_length=args.max_length,
                            ratio=ratio,
                            k=k,
                            seed=args.seed,
                            projection_embeddings=attack_model.get_input_embeddings(),
                            hidden_size=attack_model.config.hidden_size,
                            clip_norm=args.clip_norm,
                            epsilon_non_sensitive=target_eps,
                            projection_seed=noise_seed,
                        )
                        noisy_dataset = GroupedEIASequenceDataset(
                            items=group_items,
                            pretrained_embeddings=attack_model.get_input_embeddings(),
                            hidden_size=attack_model.config.hidden_size,
                            clip_norm=args.clip_norm,
                            mechanism="Laplace",
                            epsilon_private=1.0,
                            epsilon_non_sensitive=1.0,
                            seed=args.seed,
                        )

                    noisy_dataset.set_noise_config(
                        epsilon_non_sensitive=target_eps,
                        epsilon_private=local_eps,
                        seed=noise_seed,
                    )
                    dataloader = DataLoader(noisy_dataset, batch_size=args.eia_batch_size, shuffle=False)

                    print(f"  -> target_eps={target_eps} | local_eps={local_eps} | running EIA...")
                    eia = compute_eia_metrics(
                        dataloader=dataloader,
                        vocab_embs=vocab_embs,
                        device=args.device,
                        inversion_chunk_size=args.inversion_chunk_size,
                    )

                    for key in metric_keys:
                        metric_history[key].append(float(eia[key]))
                    baseline = 100.0 / (k + 1)

                    row = {
                        "Strategy": STRATEGY_LABELS[strategy],
                        "Dummy Ratio": f"{ratio * 100:.0f}%",
                        "Dummy (k)": k,
                        "Target Eps": target_eps,
                        "Required Local Eps": local_eps,
                        "Real Suffix Token Recovery (%)": f"{eia['real_suffix']:.2f}",
                        "Group EIA ASR (%)": f"{eia['group_asr']:.2f}",
                        "Random Baseline (%)": f"{baseline:.2f}",
                        "Real Suffix Tokens": int(eia["real_suffix_tokens"]),
                        "Groups Evaluated": int(eia["group_total"]),
                        "Eval Manifest": str(manifest_path),
                    }
                    with open(per_eps_path, "a", newline="", encoding="utf-8") as f:
                        csv.DictWriter(f, fieldnames=per_eps_fields).writerow(row)

                    print(
                        f"     Real suffix recovery={eia['real_suffix']:.2f}% | "
                        f"Group EIA ASR={eia['group_asr']:.2f}% | "
                        f"baseline={baseline:.2f}%"
                    )

                avg = {
                    key: sum(values) / max(len(values), 1)
                    for key, values in metric_history.items()
                }
                avg_row = {
                    "Strategy": STRATEGY_LABELS[strategy],
                    "Dummy Ratio": f"{ratio * 100:.0f}%",
                    "Dummy (k)": k,
                    "Average Real Suffix Token Recovery (%)": f"{avg['real_suffix']:.2f}",
                    "Average Group EIA ASR (%)": f"{avg['group_asr']:.2f}",
                    "Random Baseline (%)": f"{100.0 / (k + 1):.2f}",
                    "Averaged Target Eps": ",".join(str(x) for x in target_eps_list),
                    "Eval Manifest": str(manifest_path),
                }
                with open(avg_path, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=avg_fields).writerow(avg_row)

                print(
                    f"[Saved] strategy={strategy} | ratio={ratio:.1f} | k={k} | "
                    f"avg_real_suffix_recovery={avg['real_suffix']:.2f}% | "
                    f"avg_group_eia_asr={avg['group_asr']:.2f}%"
                )

    print("\n" + "=" * 80)
    print("EIA dummy-strategy benchmark complete.")
    print(f"Per-eps CSV: {per_eps_path}")
    print(f"Average-over-eps CSV: {avg_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
