# -*- coding: utf-8 -*-
"""Summarize white-box/black-box gap runs across seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple


def parse_csv_list(text: str) -> List[str]:
    return [part.strip() for part in str(text).replace(",", " ").split() if part.strip()]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate white-box/black-box gap summary CSVs.")
    parser.add_argument("--output_dir", type=str, default="outputs/whitebox_blackbox_gap")
    parser.add_argument("--prefix_base", type=str, default="llama2_mmlu_pro_whitebox_blackbox_gap")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--combined_csv", type=str, default=None)
    parser.add_argument("--aggregate_csv", type=str, default=None)
    return parser


def as_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def main() -> None:
    args = get_parser().parse_args()
    output_dir = Path(args.output_dir)
    seeds = parse_csv_list(args.seeds)
    combined_path = Path(args.combined_csv) if args.combined_csv else output_dir / f"{args.prefix_base}_combined.csv"
    aggregate_path = Path(args.aggregate_csv) if args.aggregate_csv else output_dir / f"{args.prefix_base}_aggregate.csv"

    combined_rows: List[Dict[str, str]] = []
    for seed in seeds:
        run_prefix = f"{args.prefix_base}_seed{seed}"
        summary_path = output_dir / f"{run_prefix}_summary.csv"
        if not summary_path.exists():
            print(f"[Missing] {summary_path}")
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                combined = dict(row)
                combined["seed"] = seed
                combined["run_prefix"] = run_prefix
                combined_rows.append(combined)

    combined_fields = [
        "seed",
        "variant",
        "path",
        "samples",
        "accuracy",
        "rouge_l",
        "avg_projection_token_accuracy",
        "total_projected_token_changes",
        "projection_changed_samples",
        "exact_prompt_recovery_rate",
        "changed_subset_samples",
        "changed_subset_accuracy",
        "clean_blackbox_pred_mismatch_count",
        "clean_blackbox_response_mismatch_count",
        "avg_response_chars",
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
        grouped.setdefault((row["variant"], row["path"]), []).append(row)

    aggregate_fields = [
        "variant",
        "path",
        "seeds",
        "total_samples",
        "accuracy_mean",
        "accuracy_std",
        "rouge_l_mean",
        "rouge_l_std",
        "projection_token_accuracy_mean",
        "projection_token_accuracy_std",
        "total_projected_token_changes",
        "projection_changed_samples",
        "exact_prompt_recovery_rate",
        "changed_subset_samples",
        "changed_subset_accuracy",
        "clean_blackbox_pred_mismatch_count",
        "clean_blackbox_response_mismatch_count",
        "avg_response_chars_mean",
    ]
    with open(aggregate_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fields)
        writer.writeheader()
        for (variant, path), rows in sorted(grouped.items()):
            acc = [as_float(row.get("accuracy")) for row in rows if row.get("accuracy") != ""]
            rouge = [as_float(row.get("rouge_l")) for row in rows if row.get("rouge_l") != ""]
            proj = [
                as_float(row.get("avg_projection_token_accuracy"))
                for row in rows
                if row.get("avg_projection_token_accuracy") != ""
            ]
            response_chars = [
                as_float(row.get("avg_response_chars"))
                for row in rows
                if row.get("avg_response_chars") != ""
            ]
            total_samples = sum(int(float(row.get("samples") or 0)) for row in rows)
            has_projection_change_diag = any(
                row.get("total_projected_token_changes") not in (None, "") for row in rows
            )
            token_changes = (
                sum(int(float(row.get("total_projected_token_changes") or 0)) for row in rows)
                if has_projection_change_diag
                else ""
            )
            changed_samples = (
                sum(int(float(row.get("projection_changed_samples") or 0)) for row in rows)
                if has_projection_change_diag
                else ""
            )
            exact_prompt_sample_total = sum(
                int(float(row.get("samples") or 0))
                for row in rows
                if row.get("exact_prompt_recovery_rate") not in (None, "")
            )
            exact_prompt_recovered = sum(
                as_float(row.get("exact_prompt_recovery_rate")) * int(float(row.get("samples") or 0))
                for row in rows
                if row.get("exact_prompt_recovery_rate") not in (None, "")
            )
            changed_subset_samples = (
                sum(int(float(row.get("changed_subset_samples") or 0)) for row in rows)
                if has_projection_change_diag
                else ""
            )
            changed_subset_correct = sum(
                as_float(row.get("changed_subset_accuracy")) * int(float(row.get("changed_subset_samples") or 0))
                for row in rows
                if row.get("changed_subset_accuracy") not in (None, "")
            )
            clean_pred_mismatch = sum(
                int(float(row.get("clean_blackbox_pred_mismatch_count") or 0)) for row in rows
            )
            clean_response_mismatch = sum(
                int(float(row.get("clean_blackbox_response_mismatch_count") or 0)) for row in rows
            )
            writer.writerow(
                {
                    "variant": variant,
                    "path": path,
                    "seeds": len(rows),
                    "total_samples": total_samples,
                    "accuracy_mean": f"{mean(acc):.6f}" if acc else "",
                    "accuracy_std": f"{stdev(acc):.6f}" if len(acc) > 1 else "0.000000" if acc else "",
                    "rouge_l_mean": f"{mean(rouge):.6f}" if rouge else "",
                    "rouge_l_std": f"{stdev(rouge):.6f}" if len(rouge) > 1 else "0.000000" if rouge else "",
                    "projection_token_accuracy_mean": f"{mean(proj):.6f}" if proj else "",
                    "projection_token_accuracy_std": f"{stdev(proj):.6f}" if len(proj) > 1 else "0.000000" if proj else "",
                    "total_projected_token_changes": token_changes,
                    "projection_changed_samples": changed_samples,
                    "exact_prompt_recovery_rate": (
                        f"{exact_prompt_recovered / exact_prompt_sample_total:.6f}"
                        if exact_prompt_sample_total
                        else ""
                    ),
                    "changed_subset_samples": changed_subset_samples,
                    "changed_subset_accuracy": (
                        f"{changed_subset_correct / changed_subset_samples:.6f}"
                        if changed_subset_samples
                        else ""
                    ),
                    "clean_blackbox_pred_mismatch_count": clean_pred_mismatch if clean_pred_mismatch else "",
                    "clean_blackbox_response_mismatch_count": (
                        clean_response_mismatch if clean_response_mismatch else ""
                    ),
                    "avg_response_chars_mean": f"{mean(response_chars):.2f}" if response_chars else "",
                }
            )

    print(f"[Combined] {combined_path}")
    print(f"[Aggregate] {aggregate_path}")


if __name__ == "__main__":
    main()
