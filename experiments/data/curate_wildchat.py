# -*- coding: utf-8 -*-
"""Curate WildChat into answerable QA and open-ended generation sets.

The outputs are JSONL files compatible with the existing custom_generation
loader: each row has ``text`` as the user-side query/context and ``reference``
as the assistant response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from datasets import load_dataset, load_from_disk


USER_ROLES = {"user", "human", "prompter", "client"}
ASSISTANT_ROLES = {"assistant", "gpt", "chatgpt", "bot", "model"}

QA_START_RE = re.compile(
    r"^\s*(what|who|when|where|which|whose|whom|how many|how much|how old|"
    r"how far|how long|how tall|how fast|is|are|was|were|do|does|did|can|"
    r"could|should|would|will|has|have|had|why|how)\b",
    re.IGNORECASE,
)
OPEN_INSTRUCTION_RE = re.compile(
    r"\b(write|draft|compose|create|generate|explain|summari[sz]e|rewrite|"
    r"translate|brainstorm|plan|outline|help me|give me|make me|roleplay|"
    r"review|improve|analy[sz]e|compare|recommend|design)\b",
    re.IGNORECASE,
)
BAD_QA_RE = re.compile(
    r"\b(write|draft|compose|create|generate|story|poem|essay|email|cover letter|"
    r"roleplay\w*|pretend\w*|brainstorm|opinion|recommend|code|program|script|sql|"
    r"translate|summari[sz]e|rewrite|paraphrase|comparison|compare|research|"
    r"retell|re-tell|fanfic|slander|version:|in the style of|"
    r"explain in detail|step by step|can you do|could you do|"
    r"what if|hypothetical|imagine|scenario|how can i|how do i|how should i|"
    r"what should i|what do i do|should i|should we|do next|best way|advice|"
    r"help me|fix|solve|dating|relationship|girl i like|boy i like|"
    r"have you ever wondered)\b",
    re.IGNORECASE,
)
PROMPT_INJECTION_RE = re.compile(
    r"\b(ignore (all )?(previous|prior) instructions|jailbreak|developer mode|"
    r"do anything now|\\bdan\\b|system prompt|act as an unrestricted)\b",
    re.IGNORECASE,
)
REFUSAL_RE = re.compile(
    r"\b(i('| a)m sorry|i cannot|i can('|no)t|as an ai|i don('|o)t have access|"
    r"cannot assist|not able to)\b",
    re.IGNORECASE,
)
CLARIFICATION_RE = re.compile(
    r"\b(please provide|provide the topics|provide more|more information|"
    r"more details|could you clarify|please clarify|please specify|"
    r"need additional context|need more context)\b",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_NUMBER_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
PHONE_CONTEXT_RE = re.compile(r"\b(phone|telephone|mobile|cell|call me|text me|contact me|my number)\b", re.IGNORECASE)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_NUMBER_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
CREDIT_CARD_CONTEXT_RE = re.compile(r"\b(credit card|card number|visa|mastercard|amex|cvv)\b", re.IGNORECASE)
IP_ADDRESS_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IP_CONTEXT_RE = re.compile(r"\b(my ip|ip address|public ip|private ip)\b", re.IGNORECASE)
REDACTION_PLACEHOLDER_RE = re.compile(
    r"\[[A-Z_]*(?:PERSON|EMAIL|PHONE|ADDRESS|LOCATION|DATE|ID|IP|URL|ACCOUNT|SSN)[A-Z_]*\]"
)
STREET_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Z0-9][A-Z0-9 .'-]{2,}\s+"
    r"(?:street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?)\b",
    re.IGNORECASE,
)
ADDRESS_CONTEXT_RE = re.compile(r"\b(my address|home address|shipping address|billing address|where i live)\b", re.IGNORECASE)
SENSITIVE_USER_RE = re.compile(
    r"\b("
    r"passport|social security|ssn|credit card|bank account|routing number|"
    r"password|private key|api key|secret key|confidential|"
    r"diagnosis|prescription|medical record|mental health|therapy|therapist|"
    r"lawyer|attorney|legal case|lawsuit|tax return|salary|income|debt"
    r")\b",
    re.IGNORECASE,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate WildChat into two evaluation subsets.")
    parser.add_argument("--dataset_name", type=str, default="allenai/WildChat")
    parser.add_argument("--dataset_path", type=str, default=None, help="Local load_from_disk dir, parquet, or jsonl.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", type=str2bool, default=True)
    parser.add_argument("--output_dir", type=str, default="outputs/wildchat_curated")
    parser.add_argument("--qa_output", type=str, default="wildchat_answerable_qa.jsonl")
    parser.add_argument("--open_output", type=str, default="wildchat_open_ended.jsonl")
    parser.add_argument("--target_qa", type=int, default=1000)
    parser.add_argument("--target_open", type=int, default=1000)
    parser.add_argument(
        "--candidate_multiplier",
        type=float,
        default=2.0,
        help="Collect this many times the target count before ranking/truncation.",
    )
    parser.add_argument("--max_scan", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--redacted_mode", type=str, default="any", choices=["any", "only", "exclude"])
    parser.add_argument("--include_toxic", type=str2bool, default=False)
    parser.add_argument("--single_turn_only", type=str2bool, default=False)
    parser.add_argument("--first_pair_only", type=str2bool, default=False)
    parser.add_argument("--save_messages", type=str2bool, default=True)
    parser.add_argument("--max_history_messages", type=int, default=12)
    parser.add_argument("--text_includes_context", type=str2bool, default=False)
    parser.add_argument("--min_user_chars", type=int, default=20)
    parser.add_argument("--max_user_chars", type=int, default=1200)
    parser.add_argument("--min_answer_chars", type=int, default=10)
    parser.add_argument("--max_qa_answer_chars", type=int, default=800)
    parser.add_argument("--min_open_answer_chars", type=int, default=120)
    parser.add_argument("--max_open_answer_chars", type=int, default=3000)
    parser.add_argument("--dedupe", type=str2bool, default=True)
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


def normalize_space(text: Any) -> str:
    return " ".join(str(text or "").replace("\r", " ").split()).strip()


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_rows(args) -> Iterable[Dict[str, Any]]:
    if args.dataset_path:
        path = Path(args.dataset_path)
        if path.is_dir():
            return load_from_disk(str(path))[args.split] if hasattr(load_from_disk(str(path)), "keys") else load_from_disk(str(path))
        if path.suffix.lower() in {".jsonl", ".json"}:
            return load_dataset("json", data_files=str(path), split=args.split)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return load_dataset("parquet", data_files=str(path), split=args.split)
        raise ValueError(f"Unsupported dataset_path: {path}")
    return load_dataset(args.dataset_name, split=args.split, streaming=args.streaming)


def parse_conversation(value: Any) -> List[Dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    turns: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = normalize_space(item.get("role") or item.get("from") or item.get("speaker")).lower()
        content = normalize_space(item.get("content") or item.get("value") or item.get("text"))
        if role and content:
            turns.append({"role": role, "content": content})
    return turns


def canonical_role(role: str) -> Optional[str]:
    if role in USER_ROLES:
        return "user"
    if role in ASSISTANT_ROLES:
        return "assistant"
    return None


def context_messages_for_pair(turns: Sequence[Dict[str, str]], pair_index: int, max_history_messages: int) -> List[Dict[str, str]]:
    start = max(0, pair_index + 1 - max_history_messages)
    messages: List[Dict[str, str]] = []
    for turn in turns[start : pair_index + 1]:
        role = canonical_role(turn["role"])
        if role:
            messages.append({"role": role, "content": turn["content"]})
    return messages


def serialize_messages(messages: Sequence[Dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = normalize_space(message.get("content", ""))
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def iter_user_assistant_pairs(row: Dict[str, Any], max_history_messages: int = 12) -> Iterator[Tuple[int, str, str, List[Dict[str, str]]]]:
    turns = parse_conversation(row.get("conversation"))
    for idx in range(len(turns) - 1):
        cur = turns[idx]
        nxt = turns[idx + 1]
        if cur["role"] in USER_ROLES and nxt["role"] in ASSISTANT_ROLES:
            yield idx, cur["content"], nxt["content"], context_messages_for_pair(turns, idx, max_history_messages)


def passes_row_filters(row: Dict[str, Any], args) -> bool:
    if args.language and str(row.get("language", "")).lower() != args.language.lower():
        return False
    if not args.include_toxic and bool(row.get("toxic", False)):
        return False
    redacted = bool(row.get("redacted", False))
    if args.redacted_mode == "only" and not redacted:
        return False
    if args.redacted_mode == "exclude" and redacted:
        return False
    if args.single_turn_only and int(row.get("turn", 0) or 0) != 1:
        return False
    return True


def privacy_reasons(row: Dict[str, Any], user_side_text: str) -> List[str]:
    reasons: List[str] = []
    if bool(row.get("redacted", False)):
        reasons.append("wildchat_redacted")
    if EMAIL_RE.search(user_side_text):
        reasons.append("email")
    if SSN_RE.search(user_side_text):
        reasons.append("ssn")
    if REDACTION_PLACEHOLDER_RE.search(user_side_text):
        reasons.append("redaction_placeholder")
    if PHONE_CONTEXT_RE.search(user_side_text) and PHONE_NUMBER_RE.search(user_side_text):
        reasons.append("phone_number")
    if CREDIT_CARD_CONTEXT_RE.search(user_side_text) and CREDIT_CARD_NUMBER_RE.search(user_side_text):
        reasons.append("credit_card_number")
    if IP_CONTEXT_RE.search(user_side_text) and IP_ADDRESS_RE.search(user_side_text):
        reasons.append("ip_address")
    if ADDRESS_CONTEXT_RE.search(user_side_text) and STREET_ADDRESS_RE.search(user_side_text):
        reasons.append("street_address")
    if SENSITIVE_USER_RE.search(user_side_text):
        reasons.append("sensitive_user_semantics")
    return reasons


def privacy_signal(row: Dict[str, Any], user_side_text: str) -> bool:
    return bool(privacy_reasons(row, user_side_text))


def is_bad_answer(answer_text: str) -> bool:
    if REFUSAL_RE.search(answer_text):
        return True
    if CLARIFICATION_RE.search(answer_text):
        return True
    if "```" in answer_text:
        return True
    return False


def is_answerable_qa(user_text: str, answer_text: str, args) -> bool:
    if not (args.min_user_chars <= len(user_text) <= args.max_user_chars):
        return False
    if not (args.min_answer_chars <= len(answer_text) <= args.max_qa_answer_chars):
        return False
    if is_bad_answer(answer_text):
        return False
    question_like = user_text.endswith("?") or bool(QA_START_RE.search(user_text))
    if not question_like:
        return False
    if BAD_QA_RE.search(user_text):
        return False
    # Prefer compact answers: factual QA, simple advice, or short explanations.
    if len(answer_text.split()) > 90:
        return False
    if answer_text.count("\n") > 5:
        return False
    return True


def is_open_ended(user_text: str, answer_text: str, args) -> bool:
    if not (args.min_user_chars <= len(user_text) <= args.max_user_chars):
        return False
    if not (args.min_open_answer_chars <= len(answer_text) <= args.max_open_answer_chars):
        return False
    if is_bad_answer(answer_text):
        return False
    if PROMPT_INJECTION_RE.search(user_text):
        return False
    if len(answer_text.split()) < 35:
        return False
    return bool(OPEN_INSTRUCTION_RE.search(user_text)) or not is_answerable_qa(user_text, answer_text, args)


def make_record(
    row: Dict[str, Any],
    pair_index: int,
    user_text: str,
    answer_text: str,
    subset: str,
    messages: Sequence[Dict[str, str]],
    args,
) -> Dict[str, Any]:
    user_side_text = " ".join(message["content"] for message in messages if message.get("role") == "user")
    reasons = privacy_reasons(row, user_side_text or user_text)
    context_text = serialize_messages(messages)
    text = context_text if args.text_includes_context else user_text
    return {
        "id": f"{row.get('conversation_id', 'unknown')}:{pair_index}",
        "text": text,
        "current_user_text": user_text,
        "context_text": context_text,
        "messages": list(messages) if args.save_messages else [],
        "reference": answer_text,
        "subset": subset,
        "privacy_signal": bool(reasons),
        "privacy_reasons": reasons,
        "metadata": {
            "source": "WildChat",
            "conversation_id": row.get("conversation_id"),
            "model": row.get("model"),
            "timestamp": str(row.get("timestamp", "")),
            "language": row.get("language"),
            "turn": row.get("turn"),
            "pair_index": pair_index,
            "context_message_count": len(messages),
            "toxic": bool(row.get("toxic", False)),
            "redacted": bool(row.get("redacted", False)),
            "text_hash": short_hash(user_text),
            "context_hash": short_hash(context_text),
            "reference_hash": short_hash(answer_text),
        },
    }


def ranked_insert(records: List[Dict[str, Any]], record: Dict[str, Any], target: int, rng: random.Random) -> None:
    # Keep privacy-bearing examples preferentially, while preserving a bit of randomness.
    score = (10 if record["privacy_signal"] else 0) + min(len(record["reference"]) / 1000.0, 3.0) + rng.random()
    record["_score"] = score
    records.append(record)
    if len(records) > target * 4:
        records.sort(key=lambda item: item["_score"], reverse=True)
        del records[target * 2 :]


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            clean = dict(row)
            clean.pop("_score", None)
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def main() -> None:
    args = get_parser().parse_args()
    rng = random.Random(args.seed)
    rows = load_rows(args)

    qa_records: List[Dict[str, Any]] = []
    open_records: List[Dict[str, Any]] = []
    qa_candidate_limit = max(args.target_qa, math.ceil(args.target_qa * args.candidate_multiplier))
    open_candidate_limit = max(args.target_open, math.ceil(args.target_open * args.candidate_multiplier))
    seen_prompts = set()
    scanned = 0
    candidate_pairs = 0

    for row in rows:
        scanned += 1
        if scanned > args.max_scan:
            break
        if not passes_row_filters(row, args):
            continue
        for pair_index, user_text, answer_text, messages in iter_user_assistant_pairs(row, args.max_history_messages):
            if args.first_pair_only and pair_index != 0:
                continue
            candidate_pairs += 1
            user_text = normalize_space(user_text)
            answer_text = normalize_space(answer_text)
            if args.dedupe:
                key_text = serialize_messages(messages).lower() if args.text_includes_context else user_text.lower()
                key = short_hash(key_text)
                if key in seen_prompts:
                    continue
                seen_prompts.add(key)
            if len(qa_records) < qa_candidate_limit and is_answerable_qa(user_text, answer_text, args):
                ranked_insert(
                    qa_records,
                    make_record(row, pair_index, user_text, answer_text, "answerable_qa", messages, args),
                    args.target_qa,
                    rng,
                )
            elif len(open_records) < open_candidate_limit and is_open_ended(user_text, answer_text, args):
                ranked_insert(
                    open_records,
                    make_record(row, pair_index, user_text, answer_text, "open_ended", messages, args),
                    args.target_open,
                    rng,
                )
        if len(qa_records) >= qa_candidate_limit and len(open_records) >= open_candidate_limit:
            break

    qa_records.sort(key=lambda item: item["_score"], reverse=True)
    open_records.sort(key=lambda item: item["_score"], reverse=True)
    qa_final = qa_records[: args.target_qa]
    open_final = open_records[: args.target_open]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_path = output_dir / args.qa_output
    open_path = output_dir / args.open_output
    write_jsonl(qa_path, qa_final)
    write_jsonl(open_path, open_final)

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "language": args.language,
        "redacted_mode": args.redacted_mode,
        "first_pair_only": args.first_pair_only,
        "single_turn_only": args.single_turn_only,
        "save_messages": args.save_messages,
        "max_history_messages": args.max_history_messages,
        "text_includes_context": args.text_includes_context,
        "privacy_signal_definition": (
            "wildchat row.redacted OR direct PII/sensitive regex on user-side messages only; "
            "reference answers are not scanned."
        ),
        "scanned_conversations": scanned,
        "candidate_pairs": candidate_pairs,
        "answerable_qa_count": len(qa_final),
        "open_ended_count": len(open_final),
        "answerable_qa_privacy_signal_count": sum(1 for row in qa_final if row["privacy_signal"]),
        "open_ended_privacy_signal_count": sum(1 for row in open_final if row["privacy_signal"]),
        "qa_output": str(qa_path),
        "open_output": str(open_path),
    }
    with open(output_dir / "wildchat_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    # Older datasets/pyarrow combinations can crash during streaming iterator
    # teardown after all files are already written. Avoid making batch scripts
    # treat a completed curation run as failed.
    if args.streaming:
        os._exit(0)


if __name__ == "__main__":
    main()
