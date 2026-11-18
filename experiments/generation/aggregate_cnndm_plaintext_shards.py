#!/usr/bin/env python3
"""Merge exact CNN/DailyMail plaintext shards into one compatible result row."""
from __future__ import annotations
import argparse, csv, glob, json, math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--expected_samples", type=int, default=300)
    args = parser.parse_args()
    paths = sorted(Path(item) for item in glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No shard CSV matches {args.input_glob!r}")

    rows, fieldnames = [], None
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            parsed = list(reader)
            fieldnames = fieldnames or list(reader.fieldnames or [])
        if len(parsed) != 1 or parsed[0].get("method") != "Plaintext":
            raise ValueError(f"Expected one Plaintext row in {path}")
        rows.append(parsed[0])

    details = [json.loads(row.get("details_json") or "{}") for row in rows]
    triples = sorted(zip(paths, rows, details), key=lambda item: int(item[2].get("sample_start", -1)))
    paths = [item[0] for item in triples]
    rows = [item[1] for item in triples]
    details = [item[2] for item in triples]

    expected_start = 0
    for path, row, detail in triples:
        sample_start = int(detail.get("sample_start", -1))
        if sample_start != expected_start:
            raise ValueError(f"Non-contiguous or overlapping shard at {path}: expected start {expected_start}, found {sample_start}")
        expected_start += int(row["eval_samples"])
    if expected_start != args.expected_samples:
        raise ValueError(f"Shard coverage ends at {expected_start}, expected {args.expected_samples}")

    invariant_fields = (
        "model", "dataset", "epsilon", "max_prompt_length", "max_ref_tokens",
        "max_new_tokens", "rouge_context_len", "batch_size", "seed",
    )
    for field in invariant_fields:
        values = {row.get(field, "") for row in rows}
        if len(values) != 1:
            raise ValueError(f"Shard mismatch for {field}: {sorted(values)}")

    eval_samples = sum(int(row["eval_samples"]) for row in rows)
    if eval_samples != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} samples, found {eval_samples}")

    scored_tokens = [int(row["active_tokens"]) - int(row["eval_samples"]) for row in rows]
    if any(value <= 0 for value in scored_tokens):
        raise ValueError(f"Invalid scored-token counts: {scored_tokens}")
    total_nll = sum(math.log(float(row["ppl"])) * count for row, count in zip(rows, scored_tokens))
    merged_ppl = math.exp(total_nll / sum(scored_tokens))

    reference_counts = [int(row["rouge_reference_samples"]) for row in rows]
    if sum(reference_counts) <= 0:
        raise ValueError("No shard has a usable ROUGE reference")
    merged_rouge = sum(
        float(row["rouge_l"]) * count for row, count in zip(rows, reference_counts)
    ) / sum(reference_counts)

    merged = dict(rows[0])
    merged["eval_samples"] = str(eval_samples)
    merged["rouge_reference_samples"] = str(sum(reference_counts))
    merged["active_tokens"] = str(sum(int(row["active_tokens"]) for row in rows))
    merged["ppl"] = f"{merged_ppl:.8f}"
    merged["rouge_l"] = f"{merged_rouge:.8f}"
    details_payload = {
        "aggregation": "exact_token_weighted_ppl_and_reference_weighted_rouge_mean",
        "ppl_scored_tokens": sum(scored_tokens),
        "shards": [str(path) for path in paths],
    }
    if all("generation_samples" in detail for detail in details):
        details_payload.update({
            "generation_samples": sum(int(detail["generation_samples"]) for detail in details),
            "early_eos_samples": sum(int(detail["early_eos_samples"]) for detail in details),
            "min_generated_tokens": min(int(detail["min_generated_tokens"]) for detail in details),
            "max_generated_tokens": max(int(detail["max_generated_tokens"]) for detail in details),
        })
    merged["details_json"] = json.dumps(details_payload, sort_keys=True)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: merged.get(field, "") for field in fieldnames or []})
    print(f"[Merged] samples={eval_samples}, references={sum(reference_counts)}, PPL={merged_ppl:.8f}, ROUGE-L={merged_rouge:.8f}")
    print(f"[Saved] {output}")


if __name__ == "__main__":
    main()
