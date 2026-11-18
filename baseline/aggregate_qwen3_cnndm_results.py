#!/usr/bin/env python3
"""Create a compact, audit-preserving Qwen3/CNN-DailyMail baseline table."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path


ORDER = {"plaintext": -1, "cape": 0, "custext": 1, "dp_forward": 2, "inferdpt": 3, "santext": 4, "textobfuscator": 5}
FIELDS = [
    "method", "epsilon", "epsilon_role", "ppl", "rouge_l", "remote_rouge_l", "extractor_conditional_ppl",
    "ppl_comparability", "eval_samples", "rouge_reference_samples", "model", "source_csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    paths = sorted(Path(item) for item in glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No result CSV matches {args.input_glob!r}")
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        if len(parsed) != 1:
            raise ValueError(f"Expected exactly one row in {path}, found {len(parsed)}")
        row = parsed[0]
        rows.append({key: row.get(key, "") for key in FIELDS[:-1]} | {"source_csv": str(path)})
    rows.sort(key=lambda row: (ORDER.get(Path(row["source_csv"]).parent.name, 99), float(row["epsilon"])))
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
