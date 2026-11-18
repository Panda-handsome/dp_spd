#!/usr/bin/env python3
"""Remove only byte-for-byte duplicate result rows from Figure-4 CSVs.

This is intentionally conservative: multiple rows for the same method,
dataset, and epsilon are rejected if any field differs.  It is useful when a
completed retry appends the exact same deterministic evaluation result.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_glob", default="outputs/baselines/figure4/*/*/*.csv")
    args = parser.parse_args()

    changed = 0
    removed = 0
    for raw_path in sorted(glob.glob(args.input_glob)):
        path = Path(raw_path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        if not fields:
            raise ValueError(f"Missing CSV header: {path}")

        unique: list[dict[str, str]] = []
        seen: set[tuple[str, ...]] = set()
        by_config: dict[tuple[str, str, str], tuple[str, ...]] = {}
        for row in rows:
            signature = tuple(row.get(field, "") for field in fields)
            config = (row.get("method", ""), row.get("dataset", ""), row.get("epsilon", ""))
            previous = by_config.setdefault(config, signature)
            if previous != signature:
                raise ValueError(f"Conflicting duplicate configuration in {path}: {config}")
            if signature not in seen:
                unique.append(row)
                seen.add(signature)

        if len(unique) == len(rows):
            continue
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(unique)
        changed += 1
        removed += len(rows) - len(unique)
        print(f"Deduplicated {path}: {len(rows)} -> {len(unique)} rows")
    print(f"Changed {changed} files; removed {removed} duplicate rows")


if __name__ == "__main__":
    main()
