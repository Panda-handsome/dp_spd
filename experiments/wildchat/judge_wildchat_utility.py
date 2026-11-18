# -*- coding: utf-8 -*-
"""GPT-based utility judging for saved WildChat generation outputs.

This script does not generate model responses. It reads plaintext or DP-SPD
WildChat outputs JSONL files, asks a judge model to evaluate them, and writes a
separate judged JSONL plus an aggregate CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from openai import OpenAI


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge saved WildChat utility outputs with GPT.")
    parser.add_argument(
        "--outputs_jsonl",
        type=str,
        required=True,
        help="Comma-separated saved generation outputs JSONL files.",
    )
    parser.add_argument(
        "--input_labels",
        type=str,
        default="",
        help=(
            "Optional comma-separated condition labels matching --outputs_jsonl. "
            "Use labels such as plaintext,k9_eps3,k19_eps3 so k9 and k19 are not merged."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="outputs/wildchat_utility_judge")
    parser.add_argument("--output_prefix", type=str, default="wildchat_utility_judge")
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--timeout_seconds", type=float, default=180.0)
    parser.add_argument("--connect_timeout_seconds", type=float, default=20.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep_seconds", type=float, default=5.0)
    parser.add_argument("--max_completion_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--inter_request_sleep_seconds", type=float, default=0.5)
    parser.add_argument("--max_records", type=int, default=0, help="0 means all records.")
    parser.add_argument(
        "--subsets",
        type=str,
        default="answerable_qa,open_ended",
        help="Comma-separated subsets to judge. Default judges both WildChat subsets.",
    )
    parser.add_argument(
        "--subset_default",
        type=str,
        default="",
        help="Subset to assign when an input row has no subset field, e.g. answerable_qa for QA-only baselines.",
    )
    parser.add_argument(
        "--variant_default",
        type=str,
        default="",
        help="Variant to assign when an input row has no variant field. Empty means use the input label.",
    )
    parser.add_argument(
        "--source_field",
        type=str,
        default="",
        help="Optional source question field for non-standard files, e.g. input_text.",
    )
    parser.add_argument(
        "--response_field",
        type=str,
        default="",
        help="Optional model answer field for non-standard files, e.g. extracted_generation.",
    )
    parser.add_argument(
        "--reference_field",
        type=str,
        default="reference",
        help="Reference answer field. Default: reference.",
    )
    parser.add_argument(
        "--privacy_lookup_jsonl",
        type=str,
        default="",
        help="Optional JSONL file(s) with id/privacy_signal fields to copy privacy labels from.",
    )
    parser.add_argument(
        "--fill_missing_rouge",
        type=str2bool,
        default=True,
        help="Compute rouge_l from response/reference when the input row does not already contain rouge_l.",
    )
    parser.add_argument(
        "--reuse_exact_matches",
        type=str2bool,
        default=True,
        help=(
            "Reuse judge decisions for exact same subset/source/reference/response tuples. "
            "This avoids GPT judge variance when plaintext and DP outputs are identical."
        ),
    )
    parser.add_argument(
        "--cache_jsonl",
        type=str,
        default="",
        help=(
            "Optional append-only exact-match cache JSONL. Useful when judging files in separate runs; "
            "best determinism is still obtained by judging the three files in one combined run."
        ),
    )
    parser.add_argument(
        "--force_rejudge_labels",
        type=str,
        default="",
        help=(
            "Comma-separated input labels for which exact-match cache reuse is disabled. "
            "Completed rows are still skipped when --resume true, so use --resume false for a fresh rerun."
        ),
    )
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def parse_csv_list(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def normalize_for_rouge(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if token_a == token_b else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_for_rouge(prediction).split()
    ref_tokens = normalize_for_rouge(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def strip_code_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def infer_input_label(path_text: str) -> str:
    name = Path(path_text).name
    lowered = name.lower()
    if "plain" in lowered or "original" in lowered:
        return "plaintext"
    if "k9" in lowered:
        return "dpspd_k9_eps3"
    if "k19" in lowered:
        return "dpspd_k19_eps3"
    return Path(path_text).stem


def normalize_labels(paths: Sequence[str], labels_text: str) -> List[str]:
    labels = parse_csv_list(labels_text)
    if not labels:
        return [infer_input_label(path) for path in paths]
    if len(labels) != len(paths):
        raise ValueError(f"--input_labels has {len(labels)} entries but --outputs_jsonl has {len(paths)} files.")
    return labels


def first_present(row: Dict[str, Any], explicit_field: str, fallback_fields: Sequence[str]) -> str:
    fields = [explicit_field] if explicit_field else []
    fields.extend(fallback_fields)
    for field in fields:
        if not field:
            continue
        value = row.get(field)
        if value is not None and str(value) != "":
            return str(value)
    return ""


def load_privacy_lookup(paths_text: str) -> Dict[str, bool]:
    lookup: Dict[str, bool] = {}
    for path_text in parse_csv_list(paths_text):
        path = Path(path_text)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "id" in row and "privacy_signal" in row:
                    lookup[str(row.get("id"))] = bool(row.get("privacy_signal"))
    return lookup


def normalize_input_row(
    row: Dict[str, Any],
    label: str,
    args: argparse.Namespace,
    privacy_lookup: Dict[str, bool],
) -> Dict[str, Any]:
    row = dict(row)
    if not row.get("subset") and args.subset_default:
        row["subset"] = args.subset_default
    if not row.get("variant"):
        row["variant"] = args.variant_default or label
    if not row.get("source_text"):
        row["source_text"] = first_present(row, args.source_field, ["source_text", "input_text", "context_text", "api_prompt"])
    if not row.get("reference"):
        row["reference"] = first_present(row, args.reference_field, ["reference"])
    if not row.get("response"):
        row["response"] = first_present(row, args.response_field, ["response", "extracted_generation", "perturbed_generation"])
    if "privacy_signal" not in row and str(row.get("id", "")) in privacy_lookup:
        row["privacy_signal"] = privacy_lookup[str(row.get("id", ""))]
    if "privacy_signal" not in row:
        row["privacy_signal"] = False
    if args.fill_missing_rouge and "rouge_l" not in row:
        row["rouge_l"] = rouge_l_f1(str(row.get("response") or ""), str(row.get("reference") or ""))
    return row


def load_rows(
    paths: Sequence[str],
    labels: Sequence[str],
    max_records: int,
    args: argparse.Namespace,
    allowed_subsets: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    privacy_lookup = load_privacy_lookup(args.privacy_lookup_jsonl)
    for path_text, label in zip(paths, labels):
        path = Path(path_text)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row = normalize_input_row(row, label, args, privacy_lookup)
                if allowed_subsets and str(row.get("subset", "")) not in allowed_subsets:
                    continue
                row["_source_outputs_jsonl"] = str(path)
                row["_input_label"] = str(label)
                rows.append(row)
                if max_records > 0 and len(rows) >= max_records:
                    return rows
    return rows


def record_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("_source_outputs_jsonl", "")),
        str(row.get("variant", "")),
        str(row.get("subset", "")),
        str(row.get("id", "")),
    )


def load_completed(path: Path) -> set[Tuple[str, str, str, str]]:
    keys: set[Tuple[str, str, str, str]] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            keys.add(record_key(json.loads(line)))
    return keys


def judge_user_query(row: Dict[str, Any]) -> str:
    # Utility is always judged against the original user request, even if the
    # black-box backend received a projected text prompt.
    return str(row.get("source_text") or row.get("input_text") or row.get("context_text") or row.get("projected_source_text") or row.get("api_prompt") or "")


def exact_judge_key(row: Dict[str, Any]) -> str:
    payload = {
        "subset": str(row.get("subset", "")),
        "source_text": judge_user_query(row),
        "reference": str(row.get("reference") or ""),
        "response": str(row.get("response") or ""),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cache(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if path is None or not path.exists():
        return cache
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("exact_judge_key") or "")
            if key:
                cache[key] = row
    return cache


def append_cache(path: Optional[Path], key: str, judged: Dict[str, Any], judge_model: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "exact_judge_key": key,
        "judge_model": judge_model,
        "judge_correct": parse_bool(judged.get("correct")),
        "judge_score": parse_score(judged.get("score")),
        "judge_reason": judged.get("reason"),
        "judge_raw": judged.get("_raw_response_text"),
        "judge_usage": judged.get("_usage", {}),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_client(args) -> OpenAI:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {args.api_key_env} is not set.")
    kwargs = {
        "api_key": api_key,
        "max_retries": 0,
        "timeout": httpx.Timeout(timeout=args.timeout_seconds, connect=args.connect_timeout_seconds),
    }
    if args.openai_base_url:
        kwargs["base_url"] = args.openai_base_url
    return OpenAI(**kwargs)


def build_judge_prompt(row: Dict[str, Any]) -> str:
    subset = str(row.get("subset", ""))
    user_query = judge_user_query(row)
    reference = str(row.get("reference") or "")
    response = str(row.get("response") or row.get("extracted_generation") or "")
    if subset == "answerable_qa":
        return f"""Judge whether the model answer is semantically correct for the user question.

Accept concise paraphrases and equivalent answers. Do not require exact wording.
Mark correct=true if the model answer contains the key facts needed to answer the question, even if it gives extra helpful context.
Mark correct=false if it is wrong, evasive, contradicts the reference on core facts, or misses the main requested answer.
Return valid JSON only:
{{
  "correct": true,
  "score": 1.0,
  "reason": "short reason"
}}

User question:
{user_query}

Reference answer:
{reference}

Model answer:
{response}
"""
    return f"""Judge the model response quality for this open-ended user request.

Score from 1 to 5:
1 = unusable or irrelevant
2 = partially relevant but poor
3 = acceptable
4 = good
5 = excellent

Judge primarily against the user request. Use the reference answer only as optional guidance; do not require exact wording or topic overlap when multiple good responses are possible.
Return valid JSON only:
{{
  "score": 4,
  "relevance": 4,
  "helpfulness": 4,
  "factuality": 4,
  "reason": "short reason"
}}

User request:
{user_query}

Reference answer:
{reference}

Model response:
{response}
"""


def call_judge(client: OpenAI, prompt: str, args) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.judge_model,
                messages=[
                    {"role": "system", "content": "You are a strict but fair evaluator. Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=args.temperature,
                max_completion_tokens=args.max_completion_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            usage = getattr(response, "usage", None)
            parsed = parse_json_response(content)
            parsed["_raw_response_text"] = content
            parsed["_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
            }
            return parsed
        except Exception as exc:
            last_error = exc
            if attempt >= args.max_retries:
                break
            delay = args.retry_sleep_seconds * attempt
            print(f"[Judge Retry] {type(exc).__name__}: {exc}. Sleeping {delay:.1f}s...")
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "correct", "1"}:
            return True
        if lowered in {"false", "no", "incorrect", "0"}:
            return False
    return None


def parse_score(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_summary_rows(rows: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    conditions = sorted({str(row.get("_input_label") or row.get("variant", "")) for row in rows})
    for condition in conditions:
        condition_rows = [row for row in rows if str(row.get("_input_label") or row.get("variant", "")) == condition]
        variants = sorted({str(row.get("variant", "")) for row in condition_rows})
        for variant in variants:
            variant_rows = [row for row in condition_rows if str(row.get("variant", "")) == variant]
            for subset in ["answerable_qa", "open_ended", "all"]:
                subset_rows = variant_rows if subset == "all" else [row for row in variant_rows if row.get("subset") == subset]
                for group in ["all", "privacy_signal", "nonprivacy"]:
                    if group == "privacy_signal":
                        group_rows = [row for row in subset_rows if bool(row.get("privacy_signal"))]
                    elif group == "nonprivacy":
                        group_rows = [row for row in subset_rows if not bool(row.get("privacy_signal"))]
                    else:
                        group_rows = list(subset_rows)
                    if not group_rows:
                        continue
                    qa_values = [
                        bool(row.get("judge_correct"))
                        for row in group_rows
                        if row.get("subset") == "answerable_qa" and row.get("judge_correct") is not None
                    ]
                    qa_total = len([row for row in group_rows if row.get("subset") == "answerable_qa"])
                    open_scores = [
                        float(row.get("judge_score"))
                        for row in group_rows
                        if row.get("subset") == "open_ended" and row.get("judge_score") is not None
                    ]
                    open_total = len([row for row in group_rows if row.get("subset") == "open_ended"])
                    rouge = [float(row.get("rouge_l", 0.0) or 0.0) for row in group_rows]
                    usage = [row.get("judge_usage") or {} for row in group_rows]
                    reused = [row for row in group_rows if bool(row.get("judge_reused_exact_match"))]
                    yield {
                        "condition": condition,
                        "variant": variant,
                        "subset": subset,
                        "group": group,
                        "samples": len(group_rows),
                        "qa_judge_accuracy": (
                            f"{sum(1 for value in qa_values if value) / len(qa_values):.6f}" if qa_values else ""
                        ),
                        "qa_judge_coverage": f"{len(qa_values) / qa_total:.6f}" if qa_total else "",
                        "open_judge_score": f"{sum(open_scores) / len(open_scores):.4f}" if open_scores else "",
                        "open_judge_coverage": f"{len(open_scores) / open_total:.6f}" if open_total else "",
                        "avg_rouge_l": f"{sum(rouge) / len(rouge):.6f}" if rouge else "",
                        "judge_reused_exact_matches": len(reused),
                        "judge_api_calls": len(group_rows) - len(reused),
                        "total_prompt_tokens": sum(int(u.get("prompt_tokens") or 0) for u in usage),
                        "total_completion_tokens": sum(int(u.get("completion_tokens") or 0) for u in usage),
                    }


def summarize(judged_path: Path, summary_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    if judged_path.exists():
        with open(judged_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    fields = [
        "condition",
        "variant",
        "subset",
        "group",
        "samples",
        "qa_judge_accuracy",
        "qa_judge_coverage",
        "open_judge_score",
        "open_judge_coverage",
        "avg_rouge_l",
        "judge_reused_exact_matches",
        "judge_api_calls",
        "total_prompt_tokens",
        "total_completion_tokens",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in iter_summary_rows(rows):
            writer.writerow(row)


def main() -> None:
    args = get_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judged_path = output_dir / f"{args.output_prefix}_judged.jsonl"
    summary_path = output_dir / f"{args.output_prefix}_summary.csv"
    config_path = output_dir / f"{args.output_prefix}_config.json"

    paths = parse_csv_list(args.outputs_jsonl)
    labels = normalize_labels(paths, args.input_labels)
    allowed_subsets = set(parse_csv_list(args.subsets)) if args.subsets else None
    rows = load_rows(paths, labels, args.max_records, args=args, allowed_subsets=allowed_subsets)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "input_files": paths, "input_labels": labels}, f, ensure_ascii=False, indent=2)

    completed = load_completed(judged_path) if args.resume else set()
    if not args.resume and judged_path.exists():
        judged_path.unlink()

    cache_path = Path(args.cache_jsonl) if args.cache_jsonl else None
    exact_cache = load_cache(cache_path)
    if args.reuse_exact_matches and judged_path.exists():
        with open(judged_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                completed_row = json.loads(line)
                key_for_completed = str(completed_row.get("exact_judge_key") or exact_judge_key(completed_row))
                exact_cache[key_for_completed] = completed_row

    force_rejudge_labels = set(parse_csv_list(args.force_rejudge_labels))
    client = None if args.dry_run else build_client(args)
    for row in rows:
        key = record_key(row)
        if key in completed:
            print(f"[Skip] variant={row.get('variant')} subset={row.get('subset')} id={row.get('id')}")
            continue
        exact_key = exact_judge_key(row)
        force_rejudge = str(row.get("_input_label") or "") in force_rejudge_labels
        cached = exact_cache.get(exact_key) if args.reuse_exact_matches and not force_rejudge else None
        if cached is not None and str(cached.get("judge_model", args.judge_model)) != str(args.judge_model):
            cached = None
        if cached is not None:
            out = dict(row)
            out.update(
                {
                    "exact_judge_key": exact_key,
                    "judge_model": cached.get("judge_model", args.judge_model),
                    "judge_correct": parse_bool(cached.get("judge_correct")),
                    "judge_score": parse_score(cached.get("judge_score")),
                    "judge_reason": cached.get("judge_reason"),
                    "judge_raw": cached.get("judge_raw"),
                    "judge_usage": {},
                    "judge_reused_exact_match": True,
                }
            )
            with open(judged_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(
                f"[Reused Judge] condition={row.get('_input_label')} variant={row.get('variant')} "
                f"subset={row.get('subset')} id={row.get('id')}"
            )
            continue
        prompt = build_judge_prompt(row)
        if args.dry_run:
            print(prompt[:4000])
            return
        assert client is not None
        judged = call_judge(client, prompt, args)
        judge_correct = parse_bool(judged.get("correct"))
        judge_score = parse_score(judged.get("score"))
        out = dict(row)
        out.update(
            {
                "exact_judge_key": exact_key,
                "judge_model": args.judge_model,
                "judge_correct": judge_correct,
                "judge_score": judge_score,
                "judge_reason": judged.get("reason"),
                "judge_raw": judged.get("_raw_response_text"),
                "judge_usage": judged.get("_usage", {}),
                "judge_reused_exact_match": False,
            }
        )
        with open(judged_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        if args.reuse_exact_matches:
            exact_cache[exact_key] = out
            append_cache(cache_path, exact_key, judged, args.judge_model)
        print(
            f"[Judged] condition={row.get('_input_label')} variant={row.get('variant')} subset={row.get('subset')} "
            f"id={row.get('id')} correct={judge_correct} score={judge_score}"
        )
        if args.inter_request_sleep_seconds > 0:
            time.sleep(args.inter_request_sleep_seconds)

    summarize(judged_path, summary_path)
    print(f"[Judged] {judged_path}")
    print(f"[Summary] {summary_path}")
    print(f"[Config] {config_path}")


if __name__ == "__main__":
    main()
