#!/usr/bin/env python3
"""Collect one-row Figure 4 baseline result CSVs into a comparison table.

The generation runners write their native metrics, while InferDPT intentionally
reports the extractor-conditional PPL separately.  This utility preserves that
distinction in ``ppl_type`` rather than silently treating the metrics as the
same quantity.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path


METHOD_ORDER = {
    "cape": 0,
    "custext": 1,
    "dp_forward": 2,
    "inferdpt": 3,
    "santext": 4,
    "textobfuscator": 5,
}
DATASET_ORDER = {"alpaca": 0, "dolly": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_glob",
        default="outputs/baselines/figure4/*/*/*.csv",
        help="Glob relative to the dp_spd directory.",
    )
    parser.add_argument(
        "--output_csv",
        default="outputs/baselines/figure4/summary.csv",
        help="Output path relative to the dp_spd directory.",
    )
    return parser.parse_args()


def load_result(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one result row in {path}, got {len(rows)}")

    row = rows[0]
    method = path.parents[1].name
    dataset = path.parents[0].name
    if row.get("ppl"):
        utility_ppl = row["ppl"]
        ppl_type = "server_clean_label_ppl"
    elif row.get("extractor_conditional_ppl"):
        utility_ppl = row["extractor_conditional_ppl"]
        ppl_type = "extractor_conditional_ppl_not_directly_comparable"
    else:
        raise ValueError(f"No PPL field in {path}")

    return {
        "method": method,
        "dataset_key": dataset,
        "dataset": row.get("dataset", dataset),
        "epsilon": row["epsilon"],
        "utility_ppl": utility_ppl,
        "ppl_type": ppl_type,
        "rouge_l": row["rouge_l"],
        "eval_samples": row.get("eval_samples", row.get("samples", "")),
        "source_csv": str(path),
    }


def sort_key(row: dict[str, str]) -> tuple[int, int, float]:
    return (
        DATASET_ORDER.get(row["dataset_key"], 99),
        METHOD_ORDER.get(row["method"], 99),
        float(row["epsilon"]),
    )


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in glob.glob(args.input_glob)]
    if not input_paths:
        raise FileNotFoundError(f"No result CSVs match {args.input_glob!r}")
    rows = sorted((load_result(path) for path in input_paths), key=sort_key)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
