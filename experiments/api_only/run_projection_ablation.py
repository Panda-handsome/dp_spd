# -*- coding: utf-8 -*-
"""API-only utility and projection ablation for DP-SPD.

This script isolates the token-projection path described in Algorithm 1:
noisy real-query embeddings are projected back to tokens and then evaluated by a
text-only backend. Dummy candidates are intentionally omitted for utility-only
measurement; the DP-SPD variant still uses the same prefix/suffix noise budgets
that a full real+dummy run would use.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import re
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpspd.util.utils import get_model_load_dtype, str2bool


SUPPORTED_VARIANTS = (
    "plain_text",
    "tokenized_decode",
    "clean_embedding_projection",
    "dprivacy_projected",
    "dpspd_projected",
)


PRICE_PER_MTOK = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


@dataclass
class TaskExample:
    sample_id: int
    source_text: str
    prompt: str
    reference: Optional[str]
    label: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class ProjectionResult:
    prompt_text: str
    token_count: int
    projection_token_accuracy: Optional[float]
    projected_ids: Optional[List[int]]
    diagnostics: Dict[str, Any]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate utility loss from the DP-SPD api-only projection path."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="imdb",
        choices=[
            "imdb",
            "sst2",
            "qnli",
            "cnndm",
            "mmlu_pro",
            "gpqa",
            "alpaca_instruction",
            "dolly_instruction",
            "custom_classification",
            "custom_generation",
            "custom_mcq",
        ],
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--input_jsonl", type=str, default=None)
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--question_field", type=str, default="question")
    parser.add_argument("--context_field", type=str, default="sentence")
    parser.add_argument("--choices_field", type=str, default="choices")
    parser.add_argument("--answer_field", type=str, default="answer")
    parser.add_argument("--answer_index_field", type=str, default="answer_index")
    parser.add_argument("--reference_field", type=str, default="reference")
    parser.add_argument("--label_field", type=str, default="label")
    parser.add_argument("--label_names", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--min_words", type=int, default=8)
    parser.add_argument("--max_source_chars", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="Seed used only for dataset subsampling/order. Defaults to --seed for backward compatibility.",
    )

    parser.add_argument("--projection_model", type=str, default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--projection_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--projection_chunk_size", type=int, default=8)
    parser.add_argument("--projection_tie_tolerance", type=float, default=1e-6)
    parser.add_argument("--store_projection_token_diagnostics", type=str2bool, default=False)
    parser.add_argument("--max_projection_token_diagnostics", type=int, default=20)

    parser.add_argument("--variants", type=str, default=",".join(SUPPORTED_VARIANTS))
    parser.add_argument("--dummy_ratio", type=float, default=0.4)
    parser.add_argument("--epsilon_private", type=float, default=2.08)
    parser.add_argument("--epsilon_non_sensitive", type=float, default=1.5)
    parser.add_argument("--baseline_epsilon", type=float, default=None)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--noise_mechanism", type=str, default="Laplace", choices=["Laplace", "Gaussian"])

    parser.add_argument("--backend", type=str, default="dry_run", choices=["dry_run", "openai_chat", "local_hf"])
    parser.add_argument("--api_model", type=str, default="gpt-4o")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--input_price_per_mtok", type=float, default=None)
    parser.add_argument("--output_price_per_mtok", type=float, default=None)
    parser.add_argument("--openai_timeout_seconds", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_completion_tokens", type=int, default=128)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--retry_sleep_seconds", type=float, default=4.0)
    parser.add_argument("--inter_request_sleep_seconds", type=float, default=0.0)
    parser.add_argument("--local_backend_model", type=str, default=None)
    parser.add_argument("--local_backend_dtype", type=str, default="auto")
    parser.add_argument(
        "--reuse_outputs_jsonl",
        type=str,
        default=None,
        help="Comma-separated existing outputs JSONL files. Responses are reused when api_prompt matches exactly.",
    )
    parser.add_argument("--reuse_exact_prompt_outputs", type=str2bool, default=True)

    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--rescore_only", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default="outputs/api_only_projection_ablation")
    parser.add_argument("--output_prefix", type=str, default="api_projection_ablation")
    parser.add_argument("--dry_run_print_samples", type=int, default=2)
    return parser


def parse_csv_list(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def hash_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def validate_variants(variants: Sequence[str]) -> None:
    unsupported = [variant for variant in variants if variant not in SUPPORTED_VARIANTS]
    if unsupported:
        raise ValueError(f"Unsupported variants: {unsupported}. Supported: {SUPPORTED_VARIANTS}")


def validate_local_path(path_text: str, name: str, local_files_only: bool) -> None:
    if not local_files_only:
        return
    looks_like_path = path_text.startswith(".") or path_text.startswith("/") or os.sep in path_text
    if looks_like_path and not Path(path_text).exists():
        raise FileNotFoundError(
            f"{name}={path_text!r} does not exist. Pass an absolute path or set local_files_only=false."
        )


def resolve_token_prices(args) -> Tuple[Optional[float], Optional[float]]:
    if args.input_price_per_mtok is not None or args.output_price_per_mtok is not None:
        return args.input_price_per_mtok, args.output_price_per_mtok
    model_key = str(args.api_model).lower()
    return PRICE_PER_MTOK.get(model_key, (None, None))


def trim_source(text: str, max_chars: int) -> str:
    text = " ".join(str(text).replace("\n", " ").split())
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0]
    return text


def resolve_split(raw_ds: Any, split_name: str):
    if hasattr(raw_ds, "keys"):
        if split_name not in raw_ds:
            raise ValueError(f"Split {split_name!r} not found. Available splits: {list(raw_ds.keys())}")
        return raw_ds[split_name]
    return raw_ds


def default_dataset_path_for_task(task: str) -> Optional[str]:
    rel_paths = {
        "imdb": "data/imdb",
        "sst2": "data/glue/sst2",
        "qnli": "data/glue/qnli",
        "alpaca_instruction": "data/alpaca",
        "dolly_instruction": "data/databricks_dolly",
    }
    rel_path = rel_paths.get(task)
    if not rel_path:
        return None
    candidate = (REPO_ROOT / rel_path).resolve()
    return str(candidate) if candidate.exists() else None


def load_dataset_for_task(args, default_name: str, default_config: Optional[str] = None):
    dataset_path = args.dataset_path or default_dataset_path_for_task(args.task)
    if dataset_path:
        path = Path(dataset_path)
        if path.is_file() and path.suffix == ".jsonl":
            return load_jsonl_path(path)
        if path.is_dir():
            split_jsonl = path / f"{args.dataset_split}.jsonl"
            if split_jsonl.exists():
                return load_jsonl_path(split_jsonl)
            raw_ds = load_from_disk(dataset_path)
            return resolve_split(raw_ds, args.dataset_split)

    dataset_name = args.dataset_name or default_name
    dataset_config = args.dataset_config or default_config
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=args.dataset_split)
    return load_dataset(dataset_name, split=args.dataset_split)


def label_names_for_task(task: str, args) -> List[str]:
    if args.label_names:
        return parse_csv_list(args.label_names)
    if task in ("imdb", "sst2"):
        return ["negative", "positive"]
    if task == "qnli":
        return ["entailment", "not_entailment"]
    if task in ("mmlu_pro", "gpqa", "custom_mcq"):
        return list(string.ascii_uppercase)
    if task == "custom_classification":
        raise ValueError("--label_names is required for custom_classification.")
    return []


def classification_prompt(text: str, labels: Sequence[str], task_name: str) -> str:
    label_text = ", ".join(labels)
    return (
        f"Classify the following {task_name} input. "
        f"Return only one label from this set: {label_text}.\n\n"
        f"Input:\n{text}\n\nLabel:"
    )


def summarization_prompt(article: str) -> str:
    return (
        "Summarize the following article in one concise paragraph.\n\n"
        f"Article:\n{article}\n\nSummary:"
    )


def mcq_prompt(source_text: str) -> str:
    return (
        "Answer the following multiple-choice question. "
        "Return only the single best option letter.\n\n"
        f"{source_text}\n\nAnswer:"
    )


def instruction_prompt(source_text: str) -> str:
    return f"{source_text}\n\nResponse:"


def prompt_parts_from_private_source(
    source_text: str,
    labels: Sequence[str],
    metric_type: str,
    task_name: str,
) -> Tuple[str, str, str]:
    if metric_type == "classification":
        label_text = ", ".join(labels)
        before = (
            f"Classify the following {task_name} input. "
            f"Return only one label from this set: {label_text}.\n\n"
            "Input:\n"
        )
        return before, source_text, "\n\nLabel:"
    if metric_type == "mcq":
        before = (
            "Answer the following multiple-choice question. "
            "Return only the single best option letter.\n\n"
        )
        return before, source_text, "\n\nAnswer:"
    if metric_type == "instruction_generation":
        return "", source_text, "\n\nResponse:"
    return "Summarize the following article in one concise paragraph.\n\nArticle:\n", source_text, "\n\nSummary:"


def prompt_from_private_source(
    source_text: str,
    labels: Sequence[str],
    metric_type: str,
    task_name: str,
) -> str:
    before, source, after = prompt_parts_from_private_source(source_text, labels, metric_type, task_name)
    return before + source + after


def maybe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    if "|||" in text:
        return [part.strip() for part in text.split("|||") if part.strip()]
    if "\n" in text:
        return [part.strip() for part in text.splitlines() if part.strip()]
    return [text]


def first_present(row: Dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def normalize_mcq_answer(answer: Any, choices: Sequence[str]) -> str:
    letters = list(string.ascii_uppercase[: len(choices)])
    if isinstance(answer, (int, np.integer)):
        idx = int(answer)
        if 0 <= idx < len(letters):
            return letters[idx]
        if 1 <= idx <= len(letters):
            return letters[idx - 1]

    text = str(answer).strip()
    text_upper = text.upper()
    if text_upper in letters:
        return text_upper
    match = re.match(r"^\s*([A-Z])[\)\.:]?\s*$", text_upper)
    if match and match.group(1) in letters:
        return match.group(1)
    for idx, choice in enumerate(choices):
        if normalize_label_text(text) == normalize_label_text(choice):
            return letters[idx]
    raise ValueError(f"Cannot map MCQ answer {answer!r} to one of {letters}.")


def format_mcq_source(question: str, choices: Sequence[str]) -> str:
    lines = [f"Question: {trim_source(question, 100000)}", "Choices:"]
    for idx, choice in enumerate(choices):
        lines.append(f"{string.ascii_uppercase[idx]}. {trim_source(choice, 100000)}")
    return "\n".join(lines)


def choices_from_row(row: Dict[str, Any], choices_field: str) -> List[str]:
    direct_choices = maybe_list(row.get(choices_field))
    if len(direct_choices) >= 2:
        return direct_choices

    letter_choices = []
    for letter in string.ascii_uppercase:
        for key in (letter, letter.lower(), f"choice_{letter.lower()}", f"option_{letter.lower()}"):
            if key in row and row[key] is not None:
                letter_choices.append(str(row[key]))
                break
    if len(letter_choices) >= 2:
        return letter_choices

    indexed_choices = []
    idx = 1
    while True:
        found = None
        for key in (f"choice_{idx}", f"option_{idx}", f"Incorrect Answer {idx}"):
            if key in row and row[key] is not None:
                found = str(row[key])
                break
        if found is None:
            break
        indexed_choices.append(found)
        idx += 1
    return indexed_choices


def make_mcq_example(
    sample_id: int,
    question: str,
    choices: Sequence[str],
    answer: Any,
    labels: Sequence[str],
    args,
    metadata: Dict[str, Any],
) -> TaskExample:
    choices = [trim_source(choice, args.max_source_chars) for choice in choices if str(choice).strip()]
    if len(choices) < 2:
        raise ValueError("MCQ examples require at least two choices.")
    if len(choices) > len(labels):
        choices = choices[: len(labels)]
    label = normalize_mcq_answer(answer, choices)
    source = format_mcq_source(question, choices)
    prompt = prompt_from_private_source(source, labels[: len(choices)], "mcq", "multiple-choice reasoning")
    return TaskExample(sample_id, source, prompt, None, label, metadata)


def load_custom_mcq_examples(args, labels: Sequence[str]) -> List[TaskExample]:
    rows = load_jsonl_examples(args)
    rng = random.Random(args.seed)
    if len(rows) > args.max_samples:
        rows = rng.sample(rows, args.max_samples)
    examples = []
    for sample_id, row in enumerate(rows):
        question = str(first_present(row, [args.question_field, "Question", "prompt", args.text_field], ""))
        choices = choices_from_row(row, args.choices_field)
        answer = first_present(row, [args.answer_field, args.answer_index_field, args.label_field, "target"])
        examples.append(
            make_mcq_example(
                sample_id,
                question,
                choices,
                answer,
                labels,
                args,
                {"source": "custom_jsonl", "task_name": "custom_mcq"},
            )
        )
    return examples


def load_mmlu_pro_examples(args, labels: Sequence[str]) -> List[TaskExample]:
    ds = load_dataset_for_task(args, "TIGER-Lab/MMLU-Pro")
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    examples = []
    for idx in indices:
        row = ds[idx]
        question = str(first_present(row, ["question", "Question", args.question_field], ""))
        choices = maybe_list(first_present(row, ["options", "choices", args.choices_field]))
        answer = first_present(row, ["answer", "answer_index", "correct_answer", args.answer_field])
        if not question or not choices or answer is None:
            continue
        try:
            examples.append(
                make_mcq_example(
                    len(examples),
                    question,
                    choices,
                    answer,
                    labels,
                    args,
                    {
                        "dataset_index": idx,
                        "task_name": "MMLU-Pro",
                        "category": str(first_present(row, ["category", "subject", "src"], "")),
                    },
                )
            )
        except ValueError:
            continue
        if len(examples) >= args.max_samples:
            break
    return examples


def load_gpqa_examples(args, labels: Sequence[str]) -> List[TaskExample]:
    ds = load_dataset_for_task(args, "Idavidrein/gpqa", "gpqa_diamond")
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    examples = []
    for idx in indices:
        row = ds[idx]
        question = str(first_present(row, ["Question", "question", args.question_field], ""))
        correct = first_present(row, ["Correct Answer", "correct_answer", "answer", args.answer_field])
        incorrects = [
            first_present(row, [f"Incorrect Answer {i}", f"incorrect_answer_{i}"])
            for i in range(1, 4)
        ]
        if question and correct is not None and all(item is not None for item in incorrects):
            paired = [(str(correct), True)] + [(str(item), False) for item in incorrects]
            rng.shuffle(paired)
            choices = [choice for choice, _ in paired]
            answer = string.ascii_uppercase[[is_correct for _, is_correct in paired].index(True)]
        else:
            choices = choices_from_row(row, args.choices_field)
            answer = first_present(row, ["answer", "answer_index", args.answer_field])
        if not question or len(choices) < 2 or answer is None:
            continue
        try:
            examples.append(
                make_mcq_example(
                    len(examples),
                    question,
                    choices,
                    answer,
                    labels,
                    args,
                    {
                        "dataset_index": idx,
                        "task_name": "GPQA",
                        "category": str(first_present(row, ["High-level domain", "Subdomain", "category"], "")),
                    },
                )
            )
        except ValueError:
            continue
        if len(examples) >= args.max_samples:
            break
    return examples


def load_instruction_dataset_examples(args, dataset_key: str) -> List[TaskExample]:
    default_name = "tatsu-lab/alpaca" if dataset_key == "alpaca" else "databricks/databricks-dolly-15k"
    split_name = args.dataset_split
    if split_name == "test":
        args.dataset_split = "train"
    ds = load_dataset_for_task(args, default_name)
    args.dataset_split = split_name

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    examples = []
    for idx in indices:
        row = ds[idx]
        if dataset_key == "alpaca":
            instruction = trim_source(row.get("instruction", ""), args.max_source_chars)
            context = trim_source(row.get("input", ""), args.max_source_chars)
            reference = trim_source(row.get("output", ""), args.max_source_chars)
            task_name = "Alpaca instruction"
        else:
            instruction = trim_source(row.get("instruction", ""), args.max_source_chars)
            context = trim_source(row.get("context", ""), args.max_source_chars)
            reference = trim_source(row.get("response", ""), args.max_source_chars)
            task_name = "Dolly instruction"
        if not instruction or not reference:
            continue
        source = f"Instruction:\n{instruction}"
        if context:
            source += f"\n\nContext:\n{context}"
        prompt = prompt_from_private_source(source, [], "instruction_generation", task_name)
        examples.append(
            TaskExample(
                len(examples),
                source,
                prompt,
                reference,
                None,
                {"dataset_index": idx, "task_name": task_name, "category": str(row.get("category", ""))},
            )
        )
        if len(examples) >= args.max_samples:
            break
    return examples


def load_jsonl_examples(args) -> List[Dict[str, Any]]:
    if not args.input_jsonl:
        raise ValueError("--input_jsonl is required for custom tasks.")
    return load_jsonl_path(Path(args.input_jsonl))


def load_jsonl_path(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_task_examples(args) -> Tuple[List[TaskExample], List[str], str]:
    rng = random.Random(args.seed)
    examples: List[TaskExample] = []
    labels = label_names_for_task(args.task, args)
    if args.task in ("mmlu_pro", "gpqa", "custom_mcq"):
        metric_type = "mcq"
    elif args.task in ("imdb", "sst2", "qnli", "custom_classification"):
        metric_type = "classification"
    elif args.task in ("alpaca_instruction", "dolly_instruction"):
        metric_type = "instruction_generation"
    else:
        metric_type = "generation"

    if args.task == "mmlu_pro":
        examples = load_mmlu_pro_examples(args, labels)
        if not examples:
            raise ValueError("No usable MMLU-Pro examples were selected.")
        return examples, labels, metric_type

    if args.task == "gpqa":
        examples = load_gpqa_examples(args, labels)
        if not examples:
            raise ValueError("No usable GPQA examples were selected.")
        return examples, labels, metric_type

    if args.task == "alpaca_instruction":
        examples = load_instruction_dataset_examples(args, "alpaca")
        if not examples:
            raise ValueError("No usable Alpaca examples were selected.")
        return examples, labels, metric_type

    if args.task == "dolly_instruction":
        examples = load_instruction_dataset_examples(args, "dolly")
        if not examples:
            raise ValueError("No usable Dolly examples were selected.")
        return examples, labels, metric_type

    if args.task.startswith("custom_"):
        if args.task == "custom_mcq":
            examples = load_custom_mcq_examples(args, labels)
            if not examples:
                raise ValueError("No usable custom MCQ examples were selected.")
            return examples, labels, metric_type
        rows = load_jsonl_examples(args)
        if len(rows) > args.max_samples:
            rows = rng.sample(rows, args.max_samples)
        for sample_id, row in enumerate(rows):
            if args.task == "custom_classification":
                source = trim_source(row[args.text_field], args.max_source_chars)
                label = str(row[args.label_field])
                task_name = "custom"
                prompt = prompt_from_private_source(source, labels, "classification", task_name)
                reference = None
            else:
                source = trim_source(row[args.text_field], args.max_source_chars)
                label = None
                reference = str(row.get(args.reference_field, ""))
                task_name = "generation"
                prompt = prompt_from_private_source(source, labels, "generation", task_name)
            examples.append(
                TaskExample(sample_id, source, prompt, reference, label, {"source": "custom_jsonl", "task_name": task_name})
            )
        return examples, labels, metric_type

    if args.task == "imdb":
        ds = load_dataset_for_task(args, "imdb")
    elif args.task == "sst2":
        ds = load_dataset_for_task(args, "glue", "sst2")
    elif args.task == "qnli":
        ds = load_dataset_for_task(args, "glue", "qnli")
    elif args.task == "cnndm":
        ds = load_dataset_for_task(args, "cnn_dailymail", "3.0.0")
    else:
        raise ValueError(args.task)

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for idx in indices:
        row = ds[idx]
        if args.task == "imdb":
            source = trim_source(row["text"], args.max_source_chars)
            if len(source.split()) < args.min_words:
                continue
            label = labels[int(row["label"])]
            task_name = "movie-review sentiment"
            prompt = prompt_from_private_source(source, labels, metric_type, task_name)
            examples.append(TaskExample(len(examples), source, prompt, None, label, {"dataset_index": idx, "task_name": task_name}))
        elif args.task == "sst2":
            source = trim_source(row["sentence"], args.max_source_chars)
            if len(source.split()) < args.min_words:
                continue
            label = labels[int(row["label"])]
            task_name = "sentence sentiment"
            prompt = prompt_from_private_source(source, labels, metric_type, task_name)
            examples.append(TaskExample(len(examples), source, prompt, None, label, {"dataset_index": idx, "task_name": task_name}))
        elif args.task == "qnli":
            question = trim_source(row["question"], args.max_source_chars // 2)
            sentence = trim_source(row["sentence"], args.max_source_chars)
            source = f"Question: {question}\nSentence: {sentence}"
            label = labels[int(row["label"])]
            task_name = "question-answering entailment"
            prompt = prompt_from_private_source(source, labels, metric_type, task_name)
            examples.append(TaskExample(len(examples), source, prompt, None, label, {"dataset_index": idx, "task_name": task_name}))
        elif args.task == "cnndm":
            article = trim_source(row["article"], args.max_source_chars)
            reference = trim_source(row["highlights"], args.max_source_chars)
            if len(article.split()) < args.min_words or not reference:
                continue
            task_name = "summarization"
            prompt = prompt_from_private_source(article, labels, metric_type, task_name)
            examples.append(TaskExample(len(examples), article, prompt, reference, None, {"dataset_index": idx, "task_name": task_name}))
        if len(examples) >= args.max_samples:
            break

    if not examples:
        raise ValueError("No usable examples were selected.")
    return examples, labels, metric_type


def load_projection_model(args):
    validate_local_path(args.projection_model, "projection_model", args.local_files_only)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    dtype = get_model_load_dtype(
        model_path=args.projection_model,
        device=str(device),
        dtype_preference=args.projection_dtype,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.projection_model, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.projection_model,
            local_files_only=args.local_files_only,
            dtype=dtype,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.projection_model,
            local_files_only=args.local_files_only,
            torch_dtype=dtype,
        ).to(device)
    model.eval()
    return tokenizer, model, device, dtype


def encode_prompt(tokenizer, prompt: str, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"][0].cpu(), enc["attention_mask"][0].cpu()


def decode_valid(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
    valid_len = int(attention_mask.sum().item())
    return tokenizer.decode(input_ids[:valid_len].detach().cpu().tolist(), skip_special_tokens=True).strip()


def clip_embeddings(embeddings: torch.Tensor, clip_norm: Optional[float]) -> torch.Tensor:
    if clip_norm is None or embeddings.numel() == 0:
        return embeddings
    current_norms = embeddings.norm(p=2, dim=-1, keepdim=True)
    clip_coef = torch.clamp(float(clip_norm) / (current_norms + 1e-8), max=1.0)
    return embeddings * clip_coef


def noise_scale(epsilon: float, clip_norm: Optional[float], hidden_size: int) -> float:
    sens = 2.0 * (float(clip_norm) if clip_norm else 1.0)
    return (sens / max(float(epsilon), 1e-8)) / max(hidden_size, 1) ** 0.5


def sample_noise(shape: torch.Size, scale: float, mechanism: str, seed: int, dtype: torch.dtype) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    if mechanism.lower().startswith("lap"):
        noise_np = rng.laplace(loc=0.0, scale=scale, size=tuple(shape))
    else:
        noise_np = rng.normal(loc=0.0, scale=scale, size=tuple(shape))
    return torch.from_numpy(noise_np).to(dtype=dtype)


def compute_private_layout(attn_mask: torch.Tensor, dummy_ratio: float) -> Tuple[int, int, int]:
    real_len = int(attn_mask.sum().item())
    if real_len <= 1 or dummy_ratio <= 0:
        return real_len, 0, real_len
    prefix_len = int(np.floor((1.0 - float(dummy_ratio)) * real_len))
    prefix_len = max(1, min(real_len - 1, prefix_len))
    n_private = real_len - prefix_len
    return real_len, n_private, prefix_len


def project_embeddings_to_tokens(
    embeddings: torch.Tensor,
    emb_layer: torch.nn.Embedding,
    tokenizer,
    original_ids: Optional[torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
    if embeddings.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    weight = emb_layer.weight.detach().float().cpu()
    emb = embeddings.detach().float().cpu()
    weight_norm_sq = (weight * weight).sum(dim=1)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    blocked_ids = [tok_id for tok_id in special_ids if 0 <= int(tok_id) < weight.size(0)]

    projected_chunks = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, emb.size(0), chunk_size):
        chunk = emb[start:start + chunk_size]
        chunk_norm_sq = (chunk * chunk).sum(dim=1, keepdim=True)
        distances = chunk_norm_sq + weight_norm_sq.unsqueeze(0) - 2.0 * chunk @ weight.t()
        if blocked_ids:
            distances[:, blocked_ids] = float("inf")
        projected_chunks.append(torch.argmin(distances, dim=1).to(torch.long))
    projected = torch.cat(projected_chunks, dim=0)

    if original_ids is not None and special_ids:
        original_cpu = original_ids.detach().cpu().to(torch.long)
        special_mask = torch.tensor([int(tok.item()) in special_ids for tok in original_cpu], dtype=torch.bool)
        projected[special_mask] = original_cpu[special_mask]
    return projected


def token_accuracy(projected_ids: torch.Tensor, original_ids: torch.Tensor) -> float:
    n = min(projected_ids.numel(), original_ids.numel())
    if n == 0:
        return 0.0
    return float(projected_ids[:n].eq(original_ids[:n].cpu()).float().mean().item())


def decode_single_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def build_projection_diagnostics(
    embeddings: torch.Tensor,
    emb_layer: torch.nn.Embedding,
    tokenizer,
    original_ids: torch.Tensor,
    projected_ids: torch.Tensor,
    original_source: str,
    projected_source: str,
    original_prompt: str,
    projected_prompt: str,
    args,
) -> Dict[str, Any]:
    n = min(projected_ids.numel(), original_ids.numel())
    original_cpu = original_ids[:n].detach().cpu().to(torch.long)
    projected_cpu = projected_ids[:n].detach().cpu().to(torch.long)
    changed_positions = torch.nonzero(projected_cpu.ne(original_cpu), as_tuple=False).flatten().tolist()

    diagnostics: Dict[str, Any] = {
        "projection_changed_token_count": len(changed_positions),
        "projection_changed_sample": bool(changed_positions),
        "projection_exact_token_recovery": bool(
            not changed_positions and projected_ids.numel() == original_ids.numel()
        ),
        "source_text_exact_match": projected_source == original_source,
        "prompt_exact_match": projected_prompt == original_prompt,
        "changed_token_decode_same_count": 0,
        "changed_token_near_tie_count": 0,
        "changed_token_original_dist_sq_mean": None,
        "changed_token_projected_dist_sq_mean": None,
        "changed_token_distance_margin_sq_mean": None,
    }
    if not changed_positions:
        return diagnostics

    weight = emb_layer.weight.detach().float().cpu()
    emb = embeddings[:n].detach().float().cpu()
    orig_distances: List[float] = []
    proj_distances: List[float] = []
    margins: List[float] = []
    decoded_same = 0
    near_tie = 0
    token_examples = []
    max_examples = max(0, int(args.max_projection_token_diagnostics))

    for pos in changed_positions:
        orig_id = int(original_cpu[pos].item())
        proj_id = int(projected_cpu[pos].item())
        orig_piece = decode_single_token(tokenizer, orig_id)
        proj_piece = decode_single_token(tokenizer, proj_id)
        orig_dist = float((emb[pos] - weight[orig_id]).pow(2).sum().item())
        proj_dist = float((emb[pos] - weight[proj_id]).pow(2).sum().item())
        margin = proj_dist - orig_dist
        is_same_piece = orig_piece == proj_piece
        is_near_tie = abs(margin) <= float(args.projection_tie_tolerance)
        decoded_same += int(is_same_piece)
        near_tie += int(is_near_tie)
        orig_distances.append(orig_dist)
        proj_distances.append(proj_dist)
        margins.append(margin)
        if args.store_projection_token_diagnostics and len(token_examples) < max_examples:
            token_examples.append(
                {
                    "position": int(pos),
                    "original_id": orig_id,
                    "projected_id": proj_id,
                    "original_token": orig_piece,
                    "projected_token": proj_piece,
                    "decode_same": is_same_piece,
                    "original_dist_sq": orig_dist,
                    "projected_dist_sq": proj_dist,
                    "distance_margin_sq": margin,
                    "near_tie": is_near_tie,
                }
            )

    diagnostics.update(
        {
            "changed_token_decode_same_count": decoded_same,
            "changed_token_near_tie_count": near_tie,
            "changed_token_original_dist_sq_mean": mean_or_none(orig_distances),
            "changed_token_projected_dist_sq_mean": mean_or_none(proj_distances),
            "changed_token_distance_margin_sq_mean": mean_or_none(margins),
        }
    )
    if token_examples:
        diagnostics["projection_changed_token_examples"] = token_examples
    return diagnostics


def lookup_embeddings_cpu(input_ids: torch.Tensor, emb_layer: torch.nn.Embedding) -> torch.Tensor:
    # Use the raw embedding table, not emb_layer.forward(). Some models
    # (e.g. Gemma) scale embeddings in forward(), while nearest-token
    # projection is defined against the raw vocabulary embedding matrix.
    weight = emb_layer.weight.detach()
    ids = input_ids.to(weight.device)
    with torch.no_grad():
        return weight.index_select(0, ids.reshape(-1)).reshape(*ids.shape, weight.shape[-1]).cpu()


def build_noisy_embeddings(
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    emb_layer: torch.nn.Embedding,
    args,
    variant: str,
    sample_id: int,
) -> torch.Tensor:
    clean_emb = lookup_embeddings_cpu(input_ids, emb_layer)
    target_emb = clip_embeddings(clean_emb.clone(), args.clip_norm)
    real_len, _, prefix_len = compute_private_layout(attn_mask, args.dummy_ratio)
    hidden_size = target_emb.size(-1)

    if variant == "dprivacy_projected":
        epsilon = args.baseline_epsilon if args.baseline_epsilon is not None else args.epsilon_non_sensitive
        scale = noise_scale(epsilon, args.clip_norm, hidden_size)
        noise = sample_noise(
            target_emb[:real_len].shape,
            scale,
            args.noise_mechanism,
            seed=args.seed + sample_id + 40000,
            dtype=target_emb.dtype,
        )
        target_emb[:real_len] = target_emb[:real_len] + noise
        return target_emb

    if variant == "dpspd_projected":
        private_mask = torch.zeros_like(attn_mask, dtype=torch.bool)
        private_mask[prefix_len:real_len] = True

        scale_non = noise_scale(args.epsilon_non_sensitive, args.clip_norm, hidden_size)
        context_noise = sample_noise(
            target_emb.shape,
            scale_non,
            args.noise_mechanism,
            seed=args.seed + sample_id + 20000,
            dtype=target_emb.dtype,
        )
        context_mask = (~private_mask).unsqueeze(-1).to(target_emb.dtype)
        target_emb = target_emb + context_noise * context_mask

        scale_private = noise_scale(args.epsilon_private, args.clip_norm, hidden_size)
        private_noise = sample_noise(
            target_emb.shape,
            scale_private,
            args.noise_mechanism,
            seed=args.seed + sample_id + 50000,
            dtype=target_emb.dtype,
        )
        private_mask_float = private_mask.unsqueeze(-1).to(target_emb.dtype)
        target_emb = target_emb + private_noise * private_mask_float
        return target_emb

    raise ValueError(f"Unsupported noisy variant: {variant}")


def build_variant_prompt(
    example: TaskExample,
    variant: str,
    tokenizer,
    projection_model,
    labels: Sequence[str],
    metric_type: str,
    args,
) -> ProjectionResult:
    task_name = str(example.metadata.get("task_name", args.task))
    input_ids, attn_mask = encode_prompt(tokenizer, example.source_text, args.max_length)
    valid_len = int(attn_mask.sum().item())
    original_ids = input_ids[:valid_len]

    if variant == "plain_text":
        return ProjectionResult(example.prompt, valid_len, None, None, {})
    if variant == "tokenized_decode":
        decoded_source = decode_valid(tokenizer, input_ids, attn_mask)
        prompt = prompt_from_private_source(decoded_source, labels, metric_type, task_name)
        diagnostics = {
            "projection_changed_token_count": 0,
            "projection_changed_sample": False,
            "projection_exact_token_recovery": True,
            "source_text_exact_match": decoded_source == example.source_text,
            "prompt_exact_match": prompt == example.prompt,
            "changed_token_decode_same_count": 0,
            "changed_token_near_tie_count": 0,
            "changed_token_original_dist_sq_mean": None,
            "changed_token_projected_dist_sq_mean": None,
            "changed_token_distance_margin_sq_mean": None,
        }
        return ProjectionResult(prompt, valid_len, 1.0, original_ids.tolist(), diagnostics)

    emb_layer = projection_model.get_input_embeddings()
    if variant == "clean_embedding_projection":
        embeddings = lookup_embeddings_cpu(input_ids, emb_layer)
    else:
        embeddings = build_noisy_embeddings(input_ids, attn_mask, emb_layer, args, variant, example.sample_id)

    projected_ids = project_embeddings_to_tokens(
        embeddings[:valid_len],
        emb_layer=emb_layer,
        tokenizer=tokenizer,
        original_ids=original_ids,
        chunk_size=args.projection_chunk_size,
    )
    projected_source = tokenizer.decode(projected_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()
    prompt = prompt_from_private_source(projected_source, labels, metric_type, task_name)
    diagnostics = build_projection_diagnostics(
        embeddings[:valid_len],
        emb_layer,
        tokenizer,
        original_ids,
        projected_ids,
        example.source_text,
        projected_source,
        example.prompt,
        prompt,
        args,
    )
    return ProjectionResult(
        prompt,
        valid_len,
        token_accuracy(projected_ids, original_ids),
        projected_ids.tolist(),
        diagnostics,
    )


def output_key(record: Dict[str, Any]) -> Tuple[int, str]:
    return int(record["sample_id"]), str(record["variant"])


def load_existing_outputs(path: Path) -> Dict[Tuple[int, str], Dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[output_key(row)] = row
    return rows


def add_rows_to_prompt_response_cache(
    cache: Dict[str, Dict[str, Any]],
    rows: Iterable[Dict[str, Any]],
    source_name: str,
) -> None:
    for row in rows:
        prompt = row.get("api_prompt")
        if prompt is None or row.get("response") is None:
            continue
        prompt_key = hash_text(str(prompt))
        if prompt_key not in cache:
            cached = dict(row)
            cached["_reuse_source"] = source_name
            cached["_reuse_prompt_hash"] = prompt_key
            cache[prompt_key] = cached


def load_reuse_response_cache(paths_text: Optional[str]) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not paths_text:
        return cache
    for path_text in parse_csv_list(paths_text):
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"--reuse_outputs_jsonl path not found: {path}")
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        add_rows_to_prompt_response_cache(cache, rows, str(path))
    return cache


def reused_record_from_cache(
    base_record: Dict[str, Any],
    cached: Dict[str, Any],
    metric_type: str,
    labels: Sequence[str],
) -> Dict[str, Any]:
    record = dict(base_record)
    record["response"] = cached.get("response", "")
    record["api_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    record["reused_response"] = True
    record["reused_from"] = cached.get("_reuse_source", "")
    record["reused_from_variant"] = cached.get("variant", "")
    record["reused_prompt_hash"] = cached.get("_reuse_prompt_hash", hash_text(str(cached.get("api_prompt", ""))))
    record.update(score_response(record, metric_type, labels))
    return record


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_label_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", " ", str(text).lower()).strip()


def parse_label(response: str, labels: Sequence[str]) -> Optional[str]:
    normalized = normalize_label_text(response)
    for label in labels:
        label_norm = normalize_label_text(label)
        if re.search(rf"(^|\s){re.escape(label_norm)}($|\s)", normalized):
            return label
    return None


def parse_mcq_label(response: str, labels: Sequence[str]) -> Optional[str]:
    valid = {str(label).strip().upper() for label in labels if str(label).strip()}
    valid = {label for label in valid if re.fullmatch(r"[A-Z]", label)}
    if not valid:
        return None

    text = str(response).strip()
    if not text:
        return None

    leading_patterns = [
        r"^\s*([A-Z])(?:\s*[\)\]\.\:,]|\s*$|\s+)",
        r"^\s*\(([A-Z])\)",
        r"^\s*\[([A-Z])\]",
    ]
    for pattern in leading_patterns:
        match = re.search(pattern, text)
        if match:
            label = match.group(1).upper()
            if label in valid:
                return label

    explicit_patterns = [
        r"\b(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*(?:is|=|:|\-)?\s*\(?([A-Z])\)?\b",
        r"\b(?:i\s+)?(?:choose|select|pick)\s*\(?([A-Z])\)?\b",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            label = match.group(1).upper()
            if label in valid:
                return label
    return None


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
    pred_tokens = normalize_label_text(prediction).split()
    ref_tokens = normalize_label_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_response(record: Dict[str, Any], metric_type: str, labels: Sequence[str]) -> Dict[str, Any]:
    response = str(record.get("response", ""))
    if metric_type == "classification":
        pred_label = parse_label(response, labels)
        label = record.get("label")
        correct = float(pred_label == label) if pred_label is not None else 0.0
        return {"pred_label": pred_label, "correct": correct}
    if metric_type == "mcq":
        pred_label = parse_mcq_label(response, labels)
        label = record.get("label")
        correct = float(pred_label == label) if pred_label is not None else 0.0
        return {"pred_label": pred_label, "correct": correct}
    reference = str(record.get("reference") or "")
    return {"rouge_l": rouge_l_f1(response, reference) if reference else None}


class BackendRunner:
    def __init__(self, args):
        self.args = args
        self.client = None
        self.local_tokenizer = None
        self.local_model = None
        self.local_device = None

        if args.backend == "openai_chat":
            self.client = self._build_openai_client()
        elif args.backend == "local_hf":
            model_path = args.local_backend_model or args.projection_model
            validate_local_path(model_path, "local_backend_model", args.local_files_only)
            self.local_device = torch.device(
                args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
            )
            dtype = get_model_load_dtype(
                model_path=model_path,
                device=str(self.local_device),
                dtype_preference=args.local_backend_dtype,
                local_files_only=args.local_files_only,
            )
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_files_only)
            if self.local_tokenizer.pad_token is None:
                self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
            try:
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=args.local_files_only,
                    dtype=dtype,
                ).to(self.local_device)
            except TypeError:
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=args.local_files_only,
                    torch_dtype=dtype,
                ).to(self.local_device)
            self.local_model.eval()

    def _build_openai_client(self):
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai and httpx are required for --backend openai_chat.") from exc

        api_key = os.environ.get(self.args.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable {self.args.api_key_env} is not set. "
                "Use --backend dry_run to only prepare projected prompts."
            )
        kwargs = {
            "api_key": api_key,
            "max_retries": 0,
            "timeout": httpx.Timeout(timeout=self.args.openai_timeout_seconds),
        }
        if self.args.openai_base_url:
            kwargs["base_url"] = self.args.openai_base_url
        return OpenAI(**kwargs)

    def run(self, prompt: str) -> Tuple[str, Dict[str, Optional[int]]]:
        if self.args.backend == "dry_run":
            return "", {}
        if self.args.backend == "local_hf":
            return self._run_local_hf(prompt), {}
        return self._run_openai_chat(prompt)

    def _run_local_hf(self, prompt: str) -> str:
        assert self.local_tokenizer is not None and self.local_model is not None and self.local_device is not None
        enc = self.local_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.args.max_length).to(
            self.local_device
        )
        gen_kwargs = {
            "max_new_tokens": self.args.max_completion_tokens,
            "pad_token_id": self.local_tokenizer.pad_token_id,
            "eos_token_id": self.local_tokenizer.eos_token_id,
        }
        if self.args.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.args.temperature
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            generated = self.local_model.generate(**enc, **gen_kwargs)
        new_tokens = generated[:, enc["input_ids"].shape[1]:]
        return self.local_tokenizer.batch_decode(new_tokens.detach().cpu(), skip_special_tokens=True)[0].strip()

    def _run_openai_chat(self, prompt: str) -> Tuple[str, Dict[str, Optional[int]]]:
        assert self.client is not None
        last_error: Optional[Exception] = None
        for attempt in range(1, self.args.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.args.api_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a careful assistant. Follow the user task and keep the answer concise.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.args.temperature,
                    max_completion_tokens=self.args.max_completion_tokens,
                )
                usage = getattr(response, "usage", None)
                usage_record = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
                    "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
                    "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
                }
                return response.choices[0].message.content or "", usage_record
            except Exception as exc:
                last_error = exc
                if attempt >= self.args.max_retries:
                    break
                delay = self.args.retry_sleep_seconds * attempt
                print(f"[Retry] {type(exc).__name__}: {exc}. Sleeping {delay:.1f}s...")
                time.sleep(delay)
        assert last_error is not None
        raise last_error


def summarize_outputs(
    outputs_path: Path,
    summary_path: Path,
    variants: Sequence[str],
    metric_type: str,
    labels: Sequence[str],
    input_price_per_mtok: Optional[float],
    output_price_per_mtok: Optional[float],
) -> None:
    rows_by_variant: Dict[str, List[Dict[str, Any]]] = {variant: [] for variant in variants}
    if outputs_path.exists():
        with open(outputs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("variant") in rows_by_variant:
                    rows_by_variant[row["variant"]].append(row)

    fields = [
        "variant",
        "samples",
        "avg_projection_token_accuracy",
        "total_projection_changed_tokens",
        "projection_changed_samples",
        "projection_exact_token_recovery_rate",
        "source_text_exact_match_rate",
        "prompt_exact_match_rate",
        "changed_token_decode_same_rate",
        "changed_token_near_tie_rate",
        "changed_token_original_dist_sq_mean",
        "changed_token_projected_dist_sq_mean",
        "changed_token_distance_margin_sq_mean",
        "reused_response_count",
        "accuracy",
        "rouge_l",
        "avg_prompt_chars",
        "avg_response_chars",
        "total_prompt_tokens",
        "total_completion_tokens",
        "estimated_cost_usd",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant in variants:
            rows = rows_by_variant.get(variant, [])
            proj_values = [
                float(row["projection_token_accuracy"])
                for row in rows
                if row.get("projection_token_accuracy") is not None
            ]
            has_projection_diag = any("projection_changed_token_count" in row for row in rows)
            changed_tokens = [
                int(row.get("projection_changed_token_count", 0) or 0)
                for row in rows
                if "projection_changed_token_count" in row
            ]
            changed_samples = [
                bool(row.get("projection_changed_sample"))
                for row in rows
                if "projection_changed_sample" in row
            ]
            exact_token_recovery = [
                bool(row.get("projection_exact_token_recovery"))
                for row in rows
                if "projection_exact_token_recovery" in row
            ]
            source_exact = [
                bool(row.get("source_text_exact_match"))
                for row in rows
                if "source_text_exact_match" in row
            ]
            prompt_exact = [
                bool(row.get("prompt_exact_match"))
                for row in rows
                if "prompt_exact_match" in row
            ]
            decode_same_total = sum(int(row.get("changed_token_decode_same_count", 0) or 0) for row in rows)
            near_tie_total = sum(int(row.get("changed_token_near_tie_count", 0) or 0) for row in rows)
            changed_token_total = sum(changed_tokens)
            original_dist_means = [
                float(row["changed_token_original_dist_sq_mean"])
                for row in rows
                if row.get("changed_token_original_dist_sq_mean") is not None
            ]
            projected_dist_means = [
                float(row["changed_token_projected_dist_sq_mean"])
                for row in rows
                if row.get("changed_token_projected_dist_sq_mean") is not None
            ]
            margin_means = [
                float(row["changed_token_distance_margin_sq_mean"])
                for row in rows
                if row.get("changed_token_distance_margin_sq_mean") is not None
            ]
            reused_response_count = sum(1 for row in rows if row.get("reused_response"))
            rescored_rows = [score_response(row, metric_type, labels) for row in rows]
            correct_values = [
                float(scored["correct"])
                for scored in rescored_rows
                if scored.get("correct") is not None
            ]
            rouge_values = [
                float(scored["rouge_l"])
                for scored in rescored_rows
                if scored.get("rouge_l") is not None
            ]
            prompt_chars = [len(str(row.get("api_prompt", ""))) for row in rows]
            response_chars = [len(str(row.get("response", ""))) for row in rows]
            prompt_tokens = [
                int(row.get("api_usage", {}).get("prompt_tokens") or 0)
                for row in rows
                if row.get("api_usage")
            ]
            completion_tokens = [
                int(row.get("api_usage", {}).get("completion_tokens") or 0)
                for row in rows
                if row.get("api_usage")
            ]
            total_prompt_tokens = sum(prompt_tokens)
            total_completion_tokens = sum(completion_tokens)
            estimated_cost = None
            if input_price_per_mtok is not None and output_price_per_mtok is not None:
                estimated_cost = (
                    total_prompt_tokens * float(input_price_per_mtok)
                    + total_completion_tokens * float(output_price_per_mtok)
                ) / 1_000_000
            writer.writerow(
                {
                    "variant": variant,
                    "samples": len(rows),
                    "avg_projection_token_accuracy": (
                        f"{sum(proj_values) / len(proj_values):.6f}" if proj_values else ""
                    ),
                    "total_projection_changed_tokens": (
                        sum(changed_tokens) if has_projection_diag else ""
                    ),
                    "projection_changed_samples": (
                        sum(1 for value in changed_samples if value) if has_projection_diag else ""
                    ),
                    "projection_exact_token_recovery_rate": (
                        f"{sum(1.0 for value in exact_token_recovery if value) / len(exact_token_recovery):.6f}"
                        if exact_token_recovery
                        else ""
                    ),
                    "source_text_exact_match_rate": (
                        f"{sum(1.0 for value in source_exact if value) / len(source_exact):.6f}"
                        if source_exact
                        else ""
                    ),
                    "prompt_exact_match_rate": (
                        f"{sum(1.0 for value in prompt_exact if value) / len(prompt_exact):.6f}"
                        if prompt_exact
                        else ""
                    ),
                    "changed_token_decode_same_rate": (
                        f"{decode_same_total / changed_token_total:.6f}" if changed_token_total else ""
                    ),
                    "changed_token_near_tie_rate": (
                        f"{near_tie_total / changed_token_total:.6f}" if changed_token_total else ""
                    ),
                    "changed_token_original_dist_sq_mean": (
                        f"{sum(original_dist_means) / len(original_dist_means):.8f}" if original_dist_means else ""
                    ),
                    "changed_token_projected_dist_sq_mean": (
                        f"{sum(projected_dist_means) / len(projected_dist_means):.8f}" if projected_dist_means else ""
                    ),
                    "changed_token_distance_margin_sq_mean": (
                        f"{sum(margin_means) / len(margin_means):.8f}" if margin_means else ""
                    ),
                    "reused_response_count": reused_response_count if reused_response_count else "",
                    "accuracy": (
                        f"{sum(correct_values) / len(correct_values):.6f}"
                        if metric_type in ("classification", "mcq") and correct_values
                        else ""
                    ),
                    "rouge_l": (
                        f"{sum(rouge_values) / len(rouge_values):.6f}"
                        if metric_type in ("generation", "instruction_generation") and rouge_values
                        else ""
                    ),
                    "avg_prompt_chars": f"{sum(prompt_chars) / len(prompt_chars):.2f}" if prompt_chars else "",
                    "avg_response_chars": f"{sum(response_chars) / len(response_chars):.2f}" if response_chars else "",
                    "total_prompt_tokens": total_prompt_tokens if prompt_tokens else "",
                    "total_completion_tokens": total_completion_tokens if completion_tokens else "",
                    "estimated_cost_usd": f"{estimated_cost:.6f}" if estimated_cost is not None else "",
                }
            )


def write_config(path: Path, args, variants: Sequence[str], labels: Sequence[str], metric_type: str) -> None:
    input_price_per_mtok, output_price_per_mtok = resolve_token_prices(args)
    payload = {
        "args": vars(args),
        "variants": list(variants),
        "labels": list(labels),
        "metric_type": metric_type,
        "input_price_per_mtok": input_price_per_mtok,
        "output_price_per_mtok": output_price_per_mtok,
        "note": (
            "Dummy candidates are skipped for utility-only measurement. "
            "The dpspd_projected variant applies the same real-candidate prefix/suffix noise budgets "
            "used by full DP-SPD before projecting to text for an api-only backend."
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_rescore_metadata(config_path: Path, fallback_args, fallback_variants: Sequence[str]):
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        variants = payload.get("variants") or list(fallback_variants)
        labels = payload.get("labels") or label_names_for_task(fallback_args.task, fallback_args)
        metric_type = payload.get("metric_type")
        if metric_type:
            return variants, labels, metric_type

    _, labels, metric_type = load_task_examples(fallback_args)
    return list(fallback_variants), labels, metric_type


def main() -> None:
    args = get_parser().parse_args()
    if args.max_samples < 1:
        raise ValueError("--max_samples must be at least 1.")
    variants = parse_csv_list(args.variants)
    validate_variants(variants)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / f"{args.output_prefix}_prompts.jsonl"
    outputs_path = output_dir / f"{args.output_prefix}_outputs.jsonl"
    summary_path = output_dir / f"{args.output_prefix}_summary.csv"
    config_path = output_dir / f"{args.output_prefix}_config.json"

    if args.rescore_only:
        variants, labels, metric_type = load_rescore_metadata(config_path, args, variants)
        input_price_per_mtok, output_price_per_mtok = resolve_token_prices(args)
        summarize_outputs(
            outputs_path,
            summary_path,
            variants,
            metric_type,
            labels,
            input_price_per_mtok,
            output_price_per_mtok,
        )
        print(f"[Rescored Summary] {summary_path}")
        return

    data_args = copy.copy(args)
    if args.sample_seed is not None:
        data_args.seed = args.sample_seed
    examples, labels, metric_type = load_task_examples(data_args)
    tokenizer, projection_model, projection_device, projection_dtype = load_projection_model(args)
    print(f"[Projection Model] {args.projection_model} | dtype={projection_dtype} | device={projection_device}")
    print(f"[Data] task={args.task} | examples={len(examples)} | metric={metric_type}")

    write_config(config_path, args, variants, labels, metric_type)

    if not args.resume:
        for path in (prompts_path, outputs_path, summary_path):
            if path.exists():
                path.unlink()

    existing_outputs = load_existing_outputs(outputs_path) if args.resume else {}
    existing_prompts = load_existing_outputs(prompts_path) if args.resume else {}
    response_cache = load_reuse_response_cache(args.reuse_outputs_jsonl)
    add_rows_to_prompt_response_cache(response_cache, existing_outputs.values(), str(outputs_path))
    backend = BackendRunner(args)

    printed = 0
    for example in examples:
        for variant in variants:
            projection = build_variant_prompt(example, variant, tokenizer, projection_model, labels, metric_type, args)
            base_record = {
                "sample_id": example.sample_id,
                "variant": variant,
                "task": args.task,
                "metric_type": metric_type,
                "api_prompt": projection.prompt_text,
                "api_prompt_hash": hash_text(projection.prompt_text),
                "reference": example.reference,
                "label": example.label,
                "source_text": example.source_text,
                "projection_token_accuracy": projection.projection_token_accuracy,
                "projection_token_count": projection.token_count,
                "metadata": example.metadata,
            }
            base_record.update(projection.diagnostics)
            key = (example.sample_id, variant)
            if key not in existing_prompts:
                append_jsonl(prompts_path, base_record)
                existing_prompts[key] = base_record

            if args.backend == "dry_run":
                if printed < args.dry_run_print_samples:
                    print("\n" + "=" * 80)
                    print(f"[Dry Run Prompt] sample={example.sample_id} variant={variant}")
                    print(projection.prompt_text[:2000])
                    printed += 1
                continue

            if key in existing_outputs:
                continue

            cached = response_cache.get(base_record["api_prompt_hash"]) if args.reuse_exact_prompt_outputs else None
            if cached is not None:
                record = reused_record_from_cache(base_record, cached, metric_type, labels)
                append_jsonl(outputs_path, record)
                existing_outputs[key] = record
                add_rows_to_prompt_response_cache(response_cache, [record], str(outputs_path))
                print(
                    f"[Reused] sample={example.sample_id} variant={variant} "
                    f"from={record.get('reused_from_variant') or record.get('reused_from')}"
                )
                continue

            response, api_usage = backend.run(projection.prompt_text)
            record = dict(base_record)
            record["response"] = response
            record["api_usage"] = api_usage
            record["reused_response"] = False
            record.update(score_response(record, metric_type, labels))
            append_jsonl(outputs_path, record)
            existing_outputs[key] = record
            add_rows_to_prompt_response_cache(response_cache, [record], str(outputs_path))
            print(f"[Saved] sample={example.sample_id} variant={variant}")

            if args.inter_request_sleep_seconds > 0:
                time.sleep(args.inter_request_sleep_seconds)

    if args.backend == "dry_run":
        print(f"\n[Prepared prompts] {prompts_path}")
        print(f"[Config] {config_path}")
        return

    input_price_per_mtok, output_price_per_mtok = resolve_token_prices(args)
    summarize_outputs(
        outputs_path,
        summary_path,
        variants,
        metric_type,
        labels,
        input_price_per_mtok,
        output_price_per_mtok,
    )
    print(f"[Prompts] {prompts_path}")
    print(f"[Outputs] {outputs_path}")
    print(f"[Summary] {summary_path}")
    print(f"[Config] {config_path}")


if __name__ == "__main__":
    main()
