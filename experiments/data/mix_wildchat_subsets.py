# -*- coding: utf-8 -*-
"""Build a balanced WildChat mix from curated QA and open-ended subsets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mix curated WildChat QA/open-ended subsets.")
    parser.add_argument(
        "--qa_path",
        type=str,
        default="outputs/wildchat_curated/wildchat_answerable_qa.jsonl",
    )
    parser.add_argument(
        "--open_path",
        type=str,
        default="outputs/wildchat_curated/wildchat_open_ended.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/wildchat_curated/wildchat_mixed_qa500_open500_privacy_all.jsonl",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="outputs/wildchat_curated/wildchat_mixed_qa500_open500_privacy_all_manifest.json",
    )
    parser.add_argument("--target_qa", type=int, default=500)
    parser.add_argument("--target_open", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_output", type=str2bool, default=True)
    return parser


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_subset(rows: Sequence[Dict[str, Any]], target: int, rng: random.Random, label: str) -> List[Dict[str, Any]]:
    privacy_rows = [row for row in rows if bool(row.get("privacy_signal"))]
    nonprivacy_rows = [row for row in rows if not bool(row.get("privacy_signal"))]
    if len(privacy_rows) > target:
        raise ValueError(
            f"{label} has {len(privacy_rows)} privacy rows, which exceeds target={target}. "
            "Increase target or decide how to downsample privacy rows."
        )
    needed = target - len(privacy_rows)
    if len(nonprivacy_rows) < needed:
        raise ValueError(f"{label} needs {needed} non-privacy rows, but only {len(nonprivacy_rows)} are available.")
    selected_nonprivacy = rng.sample(nonprivacy_rows, needed)
    selected = list(privacy_rows) + selected_nonprivacy
    rng.shuffle(selected)
    return selected


def add_mix_metadata(rows: Sequence[Dict[str, Any]], source_subset: str) -> List[Dict[str, Any]]:
    tagged: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["mix_source_subset"] = source_subset
        metadata = dict(item.get("metadata") or {})
        metadata["mix_source_subset"] = source_subset
        item["metadata"] = metadata
        tagged.append(item)
    return tagged


def main() -> None:
    args = get_parser().parse_args()
    rng = random.Random(args.seed)

    qa_rows = read_jsonl(Path(args.qa_path))
    open_rows = read_jsonl(Path(args.open_path))
    qa_selected = select_subset(qa_rows, args.target_qa, rng, "answerable_qa")
    open_selected = select_subset(open_rows, args.target_open, rng, "open_ended")

    output_rows = add_mix_metadata(qa_selected, "answerable_qa") + add_mix_metadata(open_selected, "open_ended")
    if args.shuffle_output:
        rng.shuffle(output_rows)

    output_path = Path(args.output_path)
    manifest_path = Path(args.manifest_path)
    write_jsonl(output_path, output_rows)

    manifest = {
        "qa_path": args.qa_path,
        "open_path": args.open_path,
        "output_path": str(output_path),
        "seed": args.seed,
        "target_qa": args.target_qa,
        "target_open": args.target_open,
        "total_count": len(output_rows),
        "qa_count": len(qa_selected),
        "open_count": len(open_selected),
        "qa_privacy_signal_count": sum(bool(row.get("privacy_signal")) for row in qa_selected),
        "open_privacy_signal_count": sum(bool(row.get("privacy_signal")) for row in open_selected),
        "total_privacy_signal_count": sum(bool(row.get("privacy_signal")) for row in output_rows),
        "qa_nonprivacy_count": sum(not bool(row.get("privacy_signal")) for row in qa_selected),
        "open_nonprivacy_count": sum(not bool(row.get("privacy_signal")) for row in open_selected),
        "shuffle_output": args.shuffle_output,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
