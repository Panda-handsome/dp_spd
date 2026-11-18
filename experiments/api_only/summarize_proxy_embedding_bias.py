# -*- coding: utf-8 -*-
"""Summarize proxy embedding model bias runs across models and seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple


def parse_csv_list(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate proxy embedding bias summary CSVs.")
    parser.add_argument("--output_dir", type=str, default="outputs/api_only_projection_ablation")
    parser.add_argument("--prefix_base", type=str, default="gpt4o_mmlu_pro_proxy_embedding_bias")
    parser.add_argument("--model_labels", type=str, default="qwen,llama")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--combined_csv", type=str, default=None)
    parser.add_argument("--aggregate_csv", type=str, default=None)
    return parser


def as_float(value) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def main() -> None:
    args = get_parser().parse_args()
    output_dir = Path(args.output_dir)
    model_labels = parse_csv_list(args.model_labels)
    seeds = parse_csv_list(args.seeds)
    combined_path = Path(args.combined_csv) if args.combined_csv else output_dir / f"{args.prefix_base}_combined.csv"
    aggregate_path = Path(args.aggregate_csv) if args.aggregate_csv else output_dir / f"{args.prefix_base}_aggregate.csv"

    combined_rows: List[Dict[str, str]] = []
    for model_label in model_labels:
        for seed in seeds:
            run_prefix = f"{args.prefix_base}_{model_label}_seed{seed}"
            summary_path = output_dir / f"{run_prefix}_summary.csv"
            if not summary_path.exists():
                print(f"[Missing] {summary_path}")
                continue
            with open(summary_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    combined = dict(row)
                    combined["projection_model_label"] = model_label
                    combined["seed"] = seed
                    combined["run_prefix"] = run_prefix
                    combined_rows.append(combined)

    combined_fields = [
        "projection_model_label",
        "seed",
        "variant",
        "samples",
        "avg_projection_token_accuracy",
        "total_projection_changed_tokens",
        "projection_changed_samples",
        "projection_exact_token_recovery_rate",
        "source_text_exact_match_rate",
        "prompt_exact_match_rate",
        "changed_token_decode_same_rate",
        "changed_token_near_tie_rate",
        "changed_token_original_dist_sq_mean",
        "changed_token_projected_dist_sq_mean",
        "changed_token_distance_margin_sq_mean",
        "reused_response_count",
        "accuracy",
        "rouge_l",
        "avg_prompt_chars",
        "avg_response_chars",
        "total_prompt_tokens",
        "total_completion_tokens",
        "estimated_cost_usd",
        "run_prefix",
    ]
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=combined_fields)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow({field: row.get(field, "") for field in combined_fields})

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in combined_rows:
        grouped.setdefault((row["projection_model_label"], row["variant"]), []).append(row)

    aggregate_fields = [
        "projection_model_label",
        "variant",
        "seeds",
        "total_samples",
        "accuracy_mean",
        "accuracy_std",
        "rouge_l_mean",
        "rouge_l_std",
        "projection_token_accuracy_mean",
        "projection_token_accuracy_std",
        "total_projection_changed_tokens",
        "projection_changed_samples",
        "projection_exact_token_recovery_rate_mean",
        "source_text_exact_match_rate_mean",
        "prompt_exact_match_rate_mean",
        "changed_token_decode_same_rate_mean",
        "changed_token_near_tie_rate_mean",
        "changed_token_original_dist_sq_mean",
        "changed_token_projected_dist_sq_mean",
        "changed_token_distance_margin_sq_mean",
        "reused_response_count_total",
        "estimated_cost_usd_total",
    ]
    with open(aggregate_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fields)
        writer.writeheader()
        for (model_label, variant), rows in sorted(grouped.items()):
            acc = [as_float(row.get("accuracy")) for row in rows if row.get("accuracy") != ""]
            rouge = [as_float(row.get("rouge_l")) for row in rows if row.get("rouge_l") != ""]
            proj = [
                as_float(row.get("avg_projection_token_accuracy"))
                for row in rows
                if row.get("avg_projection_token_accuracy") != ""
            ]
            token_changes = sum(int(float(row.get("total_projection_changed_tokens") or 0)) for row in rows)
            changed_samples = sum(int(float(row.get("projection_changed_samples") or 0)) for row in rows)
            exact_token = [
                as_float(row.get("projection_exact_token_recovery_rate"))
                for row in rows
                if row.get("projection_exact_token_recovery_rate") != ""
            ]
            source_exact = [
                as_float(row.get("source_text_exact_match_rate"))
                for row in rows
                if row.get("source_text_exact_match_rate") != ""
            ]
            prompt_exact = [
                as_float(row.get("prompt_exact_match_rate"))
                for row in rows
                if row.get("prompt_exact_match_rate") != ""
            ]
            decode_same = [
                as_float(row.get("changed_token_decode_same_rate"))
                for row in rows
                if row.get("changed_token_decode_same_rate") != ""
            ]
            near_tie = [
                as_float(row.get("changed_token_near_tie_rate"))
                for row in rows
                if row.get("changed_token_near_tie_rate") != ""
            ]
            orig_dist = [
                as_float(row.get("changed_token_original_dist_sq_mean"))
                for row in rows
                if row.get("changed_token_original_dist_sq_mean") != ""
            ]
            proj_dist = [
                as_float(row.get("changed_token_projected_dist_sq_mean"))
                for row in rows
                if row.get("changed_token_projected_dist_sq_mean") != ""
            ]
            margin_dist = [
                as_float(row.get("changed_token_distance_margin_sq_mean"))
                for row in rows
                if row.get("changed_token_distance_margin_sq_mean") != ""
            ]
            reused_count = sum(int(float(row.get("reused_response_count") or 0)) for row in rows)
            cost = [as_float(row.get("estimated_cost_usd")) for row in rows if row.get("estimated_cost_usd") != ""]
            writer.writerow(
                {
                    "projection_model_label": model_label,
                    "variant": variant,
                    "seeds": len(rows),
                    "total_samples": sum(int(float(row.get("samples") or 0)) for row in rows),
                    "accuracy_mean": f"{mean(acc):.6f}" if acc else "",
                    "accuracy_std": f"{stdev(acc):.6f}" if len(acc) > 1 else "0.000000" if acc else "",
                    "rouge_l_mean": f"{mean(rouge):.6f}" if rouge else "",
                    "rouge_l_std": f"{stdev(rouge):.6f}" if len(rouge) > 1 else "0.000000" if rouge else "",
                    "projection_token_accuracy_mean": f"{mean(proj):.6f}" if proj else "",
                    "projection_token_accuracy_std": f"{stdev(proj):.6f}" if len(proj) > 1 else "0.000000" if proj else "",
                    "total_projection_changed_tokens": token_changes if token_changes else "",
                    "projection_changed_samples": changed_samples if changed_samples else "",
                    "projection_exact_token_recovery_rate_mean": f"{mean(exact_token):.6f}" if exact_token else "",
                    "source_text_exact_match_rate_mean": f"{mean(source_exact):.6f}" if source_exact else "",
                    "prompt_exact_match_rate_mean": f"{mean(prompt_exact):.6f}" if prompt_exact else "",
                    "changed_token_decode_same_rate_mean": f"{mean(decode_same):.6f}" if decode_same else "",
                    "changed_token_near_tie_rate_mean": f"{mean(near_tie):.6f}" if near_tie else "",
                    "changed_token_original_dist_sq_mean": f"{mean(orig_dist):.8f}" if orig_dist else "",
                    "changed_token_projected_dist_sq_mean": f"{mean(proj_dist):.8f}" if proj_dist else "",
                    "changed_token_distance_margin_sq_mean": f"{mean(margin_dist):.8f}" if margin_dist else "",
                    "reused_response_count_total": reused_count if reused_count else "",
                    "estimated_cost_usd_total": f"{sum(cost):.6f}" if cost else "",
                }
            )

    print(f"[Combined] {combined_path}")
    print(f"[Aggregate] {aggregate_path}")


if __name__ == "__main__":
    main()
