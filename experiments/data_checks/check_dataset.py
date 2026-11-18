# -*- coding: utf-8 -*-
"""Lightweight local dataset sanity checks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_from_disk


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check local dataset splits and eligible text lengths.")
    parser.add_argument("--dataset_path", type=str, default="data/imdb")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--min_len", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write_manifest", type=str, default=None)
    return parser


def resolve_split(raw_ds: Any, split_name: str):
    if hasattr(raw_ds, "keys"):
        if split_name not in raw_ds:
            raise ValueError(f"Split {split_name!r} not found. Available splits: {list(raw_ds.keys())}")
        return raw_ds[split_name]
    return raw_ds


def main() -> None:
    args = get_parser().parse_args()
    raw_ds = load_from_disk(args.dataset_path)
    ds = resolve_split(raw_ds, args.dataset_split)
    if args.text_field not in ds.column_names:
        raise ValueError(f"Column {args.text_field!r} not found. Columns: {ds.column_names}")

    eligible: List[Dict[str, Any]] = []
    lengths: List[int] = []
    for idx, text in enumerate(ds[args.text_field]):
        text = str(text)
        n_words = len(text.split())
        lengths.append(n_words)
        if args.min_len <= n_words <= args.max_len:
            eligible.append({"dataset_index": idx, "word_count": n_words, "text": text})

    print("[Dataset]")
    print(f"  path: {args.dataset_path}")
    print(f"  split: {args.dataset_split}")
    print(f"  rows: {len(ds)}")
    print(f"  eligible[{args.min_len},{args.max_len}]: {len(eligible)}")
    if lengths:
        print(f"  min_words: {min(lengths)}")
        print(f"  max_words: {max(lengths)}")
        print(f"  mean_words: {sum(lengths) / len(lengths):.2f}")

    if args.write_manifest:
        if len(eligible) < args.sample_size:
            raise ValueError(f"Only {len(eligible)} eligible samples; requested {args.sample_size}.")
        selected = random.Random(args.seed).sample(eligible, args.sample_size)
        manifest = {
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "text_field": args.text_field,
            "seed": args.seed,
            "num_groups": args.sample_size,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "eligible_pool_size": len(eligible),
            "samples": selected,
        }
        out_path = Path(args.write_manifest)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
