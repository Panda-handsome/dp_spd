# -*- coding: utf-8 -*-
"""Relabel curated WildChat JSONL privacy flags using user-side text only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from curate_wildchat import privacy_reasons


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relabel WildChat privacy_signal/privacy_reasons in JSONL files.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser


def user_side_text(row: Dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        parts: List[str] = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                parts.append(str(message.get("content") or ""))
        if parts:
            return " ".join(parts)
    return str(row.get("current_user_text") or row.get("text") or "")


def main() -> None:
    args = get_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    rows: List[Dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            reasons = privacy_reasons(row, user_side_text(row))
            row["privacy_reasons"] = reasons
            row["privacy_signal"] = bool(reasons)
            rows.append(row)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(output_path)
    print(json.dumps({"input": str(input_path), "output": str(output_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
