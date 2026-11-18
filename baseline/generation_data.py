"""Shared Figure-3/Figure-4 text formatting for generation baselines.

The main DP-SPD evaluator serializes Alpaca and Dolly records into an
instruction/response text before tokenization.  Baseline adapters use this
module so their PPL and ROUGE examples are byte-for-byte compatible with that
project protocol, rather than treating an instruction dataset as WikiText's
``text`` column.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from datasets import load_from_disk
from tqdm import tqdm


DATASET_NAMES = ("wikitext", "alpaca", "dolly")
DATASET_LABELS = {
    "wikitext": "wikitext-2-v1",
    "alpaca": "stanford-alpaca",
    "dolly": "databricks-dolly-15k",
}


def default_dataset_path(name: str) -> str:
    return {
        "wikitext": "data/wikitext/wikitext-2-v1",
        "alpaca": "data/alpaca",
        "dolly": "data/databricks_dolly",
    }[name]


def default_split(name: str) -> str:
    return "test" if name == "wikitext" else "train"


def load_dataset_bundle(path: str):
    """Return a local DatasetDict, wrapping a bare Dataset as ``train``."""

    raw = load_from_disk(path)
    if hasattr(raw, "keys"):
        return raw
    return {"train": raw}


def _format_instruction_record(name: str, item) -> str:
    if name == "alpaca":
        instruction = str(item.get("instruction", "")).strip()
        input_text = str(item.get("input", "")).strip()
        output = str(item.get("output", "")).strip()
        return (
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            if input_text
            else f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        )
    if name == "dolly":
        instruction = str(item.get("instruction", "")).strip()
        context = str(item.get("context", "")).strip()
        response = str(item.get("response", "")).strip()
        return (
            f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
            if context
            else f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        )
    raise ValueError(name)


def _format_legacy_dp_generate_instruction_record(name: str, item) -> str:
    """Reproduce the Alpaca/Dolly serializer in ``dp_generate/results*.py``."""

    if name == "alpaca":
        instruction = str(item.get("instruction", ""))
        input_text = str(item.get("input", ""))
        output = str(item.get("output", ""))
        if input_text.strip():
            return (
                "Below is an instruction that describes a task, paired with an input that provides further "
                "context. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            )
        return (
            "Below is an instruction that describes a task. Write a response that appropriately completes "
            "the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        )
    if name == "dolly":
        instruction = str(item.get("instruction", ""))
        context = str(item.get("context", ""))
        response = str(item.get("response", ""))
        if context.strip():
            return f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
        return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    raise ValueError(name)


def format_generation_record(name: str, item, *, selection_protocol: str) -> str | None:
    """Format and filter one record using the project generation protocol."""

    if name == "wikitext":
        raw = str(item.get("text", ""))
        if selection_protocol == "legacy_dp_generate":
            text, keep = raw, len(raw) > 50
        else:
            text = raw.strip()
            keep = len(text.split()) > 8
    elif selection_protocol == "legacy_dp_generate":
        text = _format_legacy_dp_generate_instruction_record(name, item)
        keep = len(text) > 50
    else:
        text = _format_instruction_record(name, item)
        keep = len(text.split()) > 8
    return text if keep else None


def format_generation_texts(name: str, dataset, max_samples: int, *, selection_protocol: str) -> List[str]:
    """Return exactly the Figure-3/4 strings selected by the DP-SPD protocol."""

    if name not in DATASET_NAMES:
        raise ValueError(f"Unsupported dataset {name!r}; choose from {DATASET_NAMES}")
    texts: List[str] = []
    for item in tqdm(dataset, desc=f"Formatting {name} ({selection_protocol})"):
        # The instruction branch is the same template and >8-word filter used
        # by experiments/generation/llama_generation_eval.py for Figure 4.
        text = format_generation_record(name, item, selection_protocol=selection_protocol)
        if text is not None:
            texts.append(text)
        if max_samples > 0 and len(texts) >= max_samples:
            break
    if not texts:
        raise ValueError(f"No eligible {name} texts were found")
    return texts


def dataset_protocol_label(name: str, selection_protocol: str) -> str:
    if selection_protocol == "legacy_dp_generate":
        return selection_protocol
    return "dpspd_instruction_template_v1"


def resolve_local_dataset_path(value: str | None, repo_root: Path, dataset_name: str) -> str:
    candidate_value = value or default_dataset_path(dataset_name)
    for candidate in (Path(candidate_value), repo_root / candidate_value, repo_root.parent / candidate_value):
        if candidate.exists():
            return str(candidate)
    return candidate_value
