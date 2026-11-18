#!/usr/bin/env python
"""InferDPT/RANTEXT generation evaluation on a local LLaMA test harness.

The published InferDPT setup calls remote GPT-4 on an RANTEXT-perturbed
prompt, then asks a lightweight local model to extract a continuation using
the raw prompt and remote perturbed generation.  This runner preserves those
two stages.  For an on-premise, reproducible comparison, one frozen local
LLaMA-2-7B-Chat instance plays both roles sequentially; the remote stage is
only given the perturbed token sequence.

This is not the authors' GPT-4/text-embedding-ada-002 result.  The result CSV
records ``llama_bpe_input_embeddings`` and explicitly labels its conditional
PPL as architecture-specific: InferDPT's extractor retains the raw prompt
locally, whereas DP-SPD's Figure-3 server PPL does not.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.inferdpt.rantxt import RANTextConfig, RANTextSanitizer
from baseline.generation_data import (
    DATASET_LABELS,
    DATASET_NAMES,
    dataset_protocol_label,
    default_split,
    format_generation_record,
    format_generation_texts,
    load_dataset_bundle,
    resolve_local_dataset_path,
)
from experiments.generation.llama_generation_eval import calculate_rouge_from_texts


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RANTEXT + InferDPT extraction generation evaluation.")
    parser.add_argument("--model_path", type=str, default="models/Llama-2-7b-chat-hf")
    parser.add_argument("--extractor_model_path", type=str, default=None)
    parser.add_argument("--dataset_name", choices=DATASET_NAMES, default="wikitext")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--mapping_splits", type=str, default="train,validation")
    parser.add_argument("--mapping_max_samples", type=int, default=0)
    parser.add_argument("--mapping_batch_size", type=int, default=128)
    parser.add_argument("--candidate_vocab_size", type=int, default=11000)
    parser.add_argument("--mapping_cache", type=str, default=None)
    parser.add_argument("--force_rebuild_mapping", type=str2bool, default=False)
    parser.add_argument("--build_mapping_only", type=str2bool, default=False)

    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--source_chunk_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--require_cuda", type=str2bool, default=True)

    parser.add_argument("--selection_protocol", choices=["current_dpspd", "legacy_dp_generate"], default="current_dpspd")
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--prompt_tokens", type=int, default=32)
    parser.add_argument("--remote_max_new_tokens", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_extractor_prompt_tokens", type=int, default=512)
    parser.add_argument("--run_extractor_ppl", type=str2bool, default=True)
    parser.add_argument("--rouge_use_stemmer", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default="outputs/baselines/inferdpt/wikitext_generation.csv")
    parser.add_argument("--save_examples_jsonl", type=str, default=None)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def resolve_local_path(value: str | None) -> str | None:
    if value is None:
        return None
    candidates = [Path(value), REPO_ROOT / value, REPO_ROOT.parent / value]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return value


def resolve_dtype(name: str):
    if name == "auto":
        return None
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def parse_splits(value: str) -> List[str]:
    splits = [part.strip() for part in value.split(",") if part.strip()]
    if not splits:
        raise ValueError("mapping_splits must contain at least one split")
    return splits


def load_dataset_bundle(path: str):
    raw = load_from_disk(path)
    if not hasattr(raw, "keys"):
        raise ValueError("InferDPT mapping requires a DatasetDict with train/validation/test splits")
    return raw


def keep_text(raw_text: str, protocol: str) -> str | None:
    if protocol == "legacy_dp_generate":
        return raw_text if len(raw_text) > 50 else None
    text = raw_text.strip()
    return text if len(text.split()) > 8 else None


def collect_examples(dataset, tokenizer, args) -> List[Dict[str, object]]:
    """Collect the legacy evaluator's first-prompt/next-token continuations.

    ``dp_generate/results.py`` retained every selected row, even when fewer
    than 32 clean continuation tokens remained after its 32-token prefix.
    Preserve those rows here: ROUGE naturally excludes an empty decoded
    reference, and the extractor PPL scores only the available target tokens.
    """

    examples: List[Dict[str, object]] = []
    for item in tqdm(dataset, desc=f"Formatting {args.dataset_name} ({args.selection_protocol})"):
        text = format_generation_record(args.dataset_name, item, selection_protocol=args.selection_protocol)
        if text is None:
            continue
        # PrivacyDataset in the legacy evaluator uses the tokenizer default
        # (add_special_tokens=True), so this is intentionally not False.
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if not ids:
            continue
        prompt = [int(token) for token in ids[: args.prompt_tokens]]
        target = [int(token) for token in ids[args.prompt_tokens : args.prompt_tokens + args.max_new_tokens]]
        examples.append({"prompt_ids": prompt, "target_ids": target})
        if args.max_samples > 0 and len(examples) >= args.max_samples:
            break
    if not examples:
        raise ValueError("No rows passed the selected generation protocol")
    return examples


def count_candidate_tokens(
    raw,
    tokenizer,
    *,
    dataset_name: str,
    selection_protocol: str,
    splits: Sequence[str],
    max_samples: int,
    batch_size: int,
) -> Counter:
    if batch_size < 1:
        raise ValueError("mapping_batch_size must be positive")
    special = set(int(item) for item in tokenizer.all_special_ids)
    counts: Counter = Counter()
    for split in splits:
        if split not in raw:
            raise KeyError(f"Mapping split {split!r} is not present in {list(raw.keys())}")
        dataset = raw[split]
        limit = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
        for start in tqdm(range(0, limit, batch_size), desc=f"Building RANTEXT V ({split})"):
            columns = dataset[start : min(limit, start + batch_size)]
            row_count = len(next(iter(columns.values()), []))
            texts = []
            for row in range(row_count):
                item = {key: values[row] for key, values in columns.items()}
                text = format_generation_record(dataset_name, item, selection_protocol=selection_protocol)
                if text is not None:
                    texts.append(text)
            if not texts:
                continue
            encoded = tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)
            for ids in encoded["input_ids"]:
                counts.update(int(token_id) for token_id in ids if int(token_id) not in special)
    if not counts:
        raise ValueError("No non-special tokens were found in the public mapping corpus")
    return counts


def mapping_metadata(args, tokenizer) -> Dict[str, object]:
    metadata = {
        "schema": "inferdpt_rantext_llama_vocab_v2",
        "model_path": args.model_path,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "mapping_splits": parse_splits(args.mapping_splits),
        "mapping_max_samples": int(args.mapping_max_samples),
        "candidate_vocab_size": int(args.candidate_vocab_size),
        "selection": f"public_{args.dataset_name}_frequency_top_tokens",
        "paper_counterpart": "first_11000_english_cl100k_tokens",
        "representation": "llama_bpe_input_embeddings",
        "selection_protocol": args.selection_protocol,
    }
    # Maintain completed WikiText cache compatibility while making instruction
    # corpus statistics impossible to reuse accidentally as WikiText's V.
    if args.dataset_name != "wikitext":
        metadata["dataset_name"] = args.dataset_name
    return metadata


def default_mapping_cache(args) -> Path:
    model_name = Path(args.model_path).name.replace(" ", "_")
    split_tag = "-".join(parse_splits(args.mapping_splits))
    dataset_tag = "wikitext" if args.dataset_name == "wikitext" else args.dataset_name
    return Path("outputs/baselines/inferdpt/mappings") / (
        f"{dataset_tag}_{model_name}_rantext_{split_tag}_top{args.candidate_vocab_size}_"
        f"n{args.mapping_max_samples or 'all'}_{args.selection_protocol}_v2.json"
    )


def build_or_load_candidate_vocab(args, raw, tokenizer) -> Tuple[List[int], Path]:
    cache_path = Path(args.mapping_cache) if args.mapping_cache else default_mapping_cache(args)
    expected = mapping_metadata(args, tokenizer)
    if cache_path.is_file() and not args.force_rebuild_mapping:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            raise ValueError(f"RANTEXT cache {cache_path} differs from this run: {mismatches}")
        ids = [int(item) for item in payload["candidate_token_ids"]]
        print(f"[InferDPT] Reusing RANTEXT vocabulary: {cache_path} (|V|={len(ids)})")
        return ids, cache_path

    counts = count_candidate_tokens(
        raw,
        tokenizer,
        dataset_name=args.dataset_name,
        selection_protocol=args.selection_protocol,
        splits=parse_splits(args.mapping_splits),
        max_samples=args.mapping_max_samples,
        batch_size=args.mapping_batch_size,
    )
    ids = [int(token_id) for token_id, _ in counts.most_common(args.candidate_vocab_size)]
    if len(ids) < args.candidate_vocab_size:
        raise ValueError(f"Only {len(ids)} public LLaMA tokens exist, fewer than candidate_vocab_size")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"candidate_token_ids": ids, "metadata": expected}, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[InferDPT] Saved RANTEXT vocabulary: {cache_path} (|V|={len(ids)})")
    return ids, cache_path


def left_pad(sequences: Sequence[Sequence[int]], pad_token_id: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Cannot pad an empty sequence batch")
    maximum = max(max(len(sequence), 1) for sequence in sequences)
    ids = torch.full((len(sequences), maximum), int(pad_token_id), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        # An all-discarded prompt conveys no raw token to the remote model.
        # The pad token is used only as a syntactic empty-input placeholder.
        if not sequence:
            continue
        values = torch.tensor(sequence, dtype=torch.long, device=device)
        ids[index, maximum - len(sequence) :] = values
        mask[index, maximum - len(sequence) :] = 1
    return ids, mask


def generate_batches(
    model,
    tokenizer,
    sequences: Sequence[Sequence[int]],
    *,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> List[List[int]]:
    results: List[List[int]] = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="Generate"):
            batch = sequences[start : start + batch_size]
            ids, mask = left_pad(batch, tokenizer.pad_token_id, device)
            generated = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            results.extend([[int(item) for item in row[ids.shape[1] :].detach().cpu().tolist()] for row in generated])
    return results


def extraction_prompt(raw_prefix: str, perturbed_generation: str) -> str:
    """Paper Appendix-A-style instruction for the local extraction module."""

    return (
        "Your task is to extend the Prefix Text. Use the Perturbed Generation as your primary writing "
        "material. Extract coherent and consistent text from the Perturbed Generation and integrate it "
        "into a continuation aligned with the Prefix Text. Provide only the Extended Text.\n\n"
        f"Prefix Text:\n{raw_prefix}\n\nPerturbed Generation:\n{perturbed_generation}\n\nExtended Text:\n"
    )


def conditional_extractor_ppl(
    model,
    tokenizer,
    prompts: Sequence[str],
    target_ids: Sequence[Sequence[int]],
    *,
    batch_size: int,
    max_prompt_tokens: int,
    device: torch.device,
) -> float:
    """Teacher-forced clean-continuation PPL of InferDPT's local extractor.

    This is intentionally distinct from Figure-3 server PPL: the extractor is
    allowed to see the raw prefix by InferDPT design.  Only target continuation
    tokens contribute to the loss.
    """

    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(prompts), batch_size), desc="Extractor PPL"):
            prompt_batch = prompts[start : start + batch_size]
            target_batch = target_ids[start : start + batch_size]
            context_batch = [
                tokenizer(prompt, add_special_tokens=False)["input_ids"][-max_prompt_tokens:]
                for prompt in prompt_batch
            ]
            sequences = [context + list(target) for context, target in zip(context_batch, target_batch)]
            ids, attention = left_pad(sequences, tokenizer.pad_token_id, device)
            labels = torch.full_like(ids, -100)
            for index, (context, target) in enumerate(zip(context_batch, target_batch)):
                left = ids.shape[1] - len(context) - len(target)
                labels[index, left + len(context) : left + len(context) + len(target)] = torch.tensor(
                    target, dtype=torch.long, device=device
                )
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
            nll = loss_fct(logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]), labels[:, 1:].contiguous().view(-1))
            total_nll += float(nll.item())
            total_tokens += int(labels[:, 1:].ne(-100).sum().item())
    return math.exp(total_nll / total_tokens) if total_tokens else float("nan")


RESULT_FIELDS = [
    "method", "perturbation_mechanism", "representation", "remote_model", "extractor_model", "dataset", "epsilon",
    "epsilon_semantics", "candidate_vocab_size", "mapping_cache", "mapping_splits", "selection_protocol", "eval_samples",
    "rouge_reference_samples", "prompt_tokens", "remote_max_new_tokens", "max_new_tokens", "rouge_use_stemmer", "remote_access", "ppl_protocol",
    "ppl_comparability", "active_prompt_tokens", "in_vocabulary_prompt_tokens", "discarded_out_of_vocabulary_prompt_tokens",
    "prompt_coverage_fraction", "realized_replacements", "realized_replacement_fraction", "adjacency_mean_size",
    "adjacency_min_size", "adjacency_max_size", "laplace_denominator_z", "remote_rouge_l", "extractor_conditional_ppl",
    "rouge_l", "seed",
]


def append_row(path: str | Path, row: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def main() -> None:
    args = get_parser().parse_args()
    args.model_path = resolve_local_path(args.model_path)
    args.extractor_model_path = resolve_local_path(args.extractor_model_path) or args.model_path
    args.dataset_path = resolve_local_dataset_path(args.dataset_path, REPO_ROOT, args.dataset_name)
    args.eval_split = args.eval_split or default_split(args.dataset_name)
    if args.dataset_name != "wikitext" and args.mapping_splits == "train,validation":
        args.mapping_splits = "train"
    config = RANTextConfig(epsilon=args.epsilon, source_chunk_size=args.source_chunk_size, seed=args.seed)
    config.validate()
    if args.candidate_vocab_size < 2:
        raise ValueError("candidate_vocab_size must be at least two")
    if args.prompt_tokens < 1 or args.max_new_tokens < 1 or args.remote_max_new_tokens < 1:
        raise ValueError("prompt and generation lengths must be positive")
    if args.dry_run:
        print("[Dry Run]")
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        print("This is the two-stage InferDPT/RANTEXT LLaMA generation adapter.")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type != "cuda" and args.device.startswith("cuda"):
        message = f"CUDA device {args.device} is unavailable."
        if args.require_cuda:
            raise RuntimeError(message + " Refusing CPU fallback for a GPU benchmark.")
        print(f"[Warning] {message} Using CPU because --require_cuda false was supplied.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    raw = load_dataset_bundle(args.dataset_path)
    if args.eval_split not in raw:
        raise KeyError(f"Evaluation split {args.eval_split!r} is not present in {list(raw.keys())}")
    candidate_ids, mapping_path = build_or_load_candidate_vocab(args, raw, tokenizer)
    if args.build_mapping_only:
        print(f"[InferDPT] RANTEXT vocabulary ready: {mapping_path} (|V|={len(candidate_ids)})")
        return
    examples = collect_examples(raw[args.eval_split], tokenizer, args)

    if args.extractor_model_path != args.model_path:
        raise ValueError(
            "This single-GPU runner currently requires --extractor_model_path to equal --model_path. "
            "Use the same local LLaMA for the sequential remote/extractor test harness."
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=args.local_files_only, torch_dtype=resolve_dtype(args.dtype)
    ).to(device)
    model.eval()

    # The legacy padded evaluator retains short rows.  Right-pad only for the
    # sanitizer tensor and use ``clean_mask`` so pads are never transmitted.
    prompt_width = max(max(len(item["prompt_ids"]), 1) for item in examples)
    clean_prompt_ids = torch.full(
        (len(examples), prompt_width), tokenizer.pad_token_id, dtype=torch.long
    )
    clean_mask = torch.zeros_like(clean_prompt_ids)
    for index, item in enumerate(examples):
        width = len(item["prompt_ids"])
        if width:
            clean_prompt_ids[index, :width] = torch.tensor(item["prompt_ids"], dtype=torch.long)
            clean_mask[index, :width] = 1
    sanitizer = RANTextSanitizer(candidate_ids, config, vocab_size=int(len(tokenizer)))
    perturbed_ids, retained_mask, counts = sanitizer.sanitize_token_ids(
        clean_prompt_ids, clean_mask, model.get_input_embeddings().weight.detach(), seed=args.seed
    )
    perturbed_sequences = [
        [int(token) for token in perturbed_ids[index][retained_mask[index]].tolist()]
        for index in range(len(examples))
    ]
    remote_outputs = generate_batches(
        model, tokenizer, perturbed_sequences, batch_size=args.batch_size,
        max_new_tokens=args.remote_max_new_tokens, device=device,
    )

    raw_prefixes = [tokenizer.decode(item["prompt_ids"], skip_special_tokens=True) for item in examples]
    references = [tokenizer.decode(item["target_ids"], skip_special_tokens=True) for item in examples]
    rouge_reference_samples = sum(bool(reference.strip()) for reference in references)
    remote_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in remote_outputs]
    prompts = [extraction_prompt(prefix, generated) for prefix, generated in zip(raw_prefixes, remote_texts)]
    extractor_input_ids = [tokenizer(prompt, add_special_tokens=False)["input_ids"][-args.max_extractor_prompt_tokens:] for prompt in prompts]
    final_outputs = generate_batches(
        model, tokenizer, extractor_input_ids, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens, device=device,
    )
    final_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in final_outputs]
    remote_rouge_l = calculate_rouge_from_texts(remote_texts, references, use_stemmer=args.rouge_use_stemmer)
    final_rouge_l = calculate_rouge_from_texts(final_texts, references, use_stemmer=args.rouge_use_stemmer)
    extractor_ppl = (
        conditional_extractor_ppl(
            model, tokenizer, prompts, [item["target_ids"] for item in examples], batch_size=args.batch_size,
            max_prompt_tokens=args.max_extractor_prompt_tokens, device=device,
        )
        if args.run_extractor_ppl
        else float("nan")
    )

    if args.save_examples_jsonl:
        output_path = Path(args.save_examples_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for index, example in enumerate(examples):
                handle.write(json.dumps({
                    "raw_prefix": raw_prefixes[index],
                    "perturbed_prompt": tokenizer.decode(perturbed_sequences[index], skip_special_tokens=False),
                    "remote_perturbed_generation": remote_texts[index],
                    "extractor_prompt": prompts[index],
                    "final_generation": final_texts[index],
                    "reference_continuation": references[index],
                }, ensure_ascii=False) + "\n")
        print(f"[InferDPT] Saved examples: {output_path}")

    row = {
        "method": "InferDPT-style",
        "perturbation_mechanism": "RANTEXT",
        "representation": "llama_bpe_input_embeddings__local_llama_remote_and_extractor_harness",
        "remote_model": args.model_path,
        "extractor_model": args.extractor_model_path,
        "dataset": DATASET_LABELS[args.dataset_name],
        "epsilon": args.epsilon,
        "epsilon_semantics": "per_token_random_adjacency_epsilon_ldp__conditional_on_Cr",
        "candidate_vocab_size": len(candidate_ids),
        "mapping_cache": str(mapping_path),
        "mapping_splits": args.mapping_splits,
        "selection_protocol": dataset_protocol_label(args.dataset_name, args.selection_protocol),
        "eval_samples": len(examples),
        "rouge_reference_samples": rouge_reference_samples,
        "prompt_tokens": args.prompt_tokens,
        "remote_max_new_tokens": args.remote_max_new_tokens,
        "max_new_tokens": args.max_new_tokens,
        "rouge_use_stemmer": args.rouge_use_stemmer,
        "remote_access": "perturbed_prompt_only; raw_prefix_retained_only_by_local_extractor",
        "ppl_protocol": "teacher_forced_clean_continuation_given_local_raw_prefix_and_remote_perturbed_generation",
        "ppl_comparability": "not_directly_comparable_to_DP-SPD_server_clean-label_PPL",
        "active_prompt_tokens": int(counts["active_tokens"]),
        "in_vocabulary_prompt_tokens": int(counts["in_vocabulary_tokens"]),
        "discarded_out_of_vocabulary_prompt_tokens": int(counts["discarded_out_of_vocabulary_tokens"]),
        "prompt_coverage_fraction": f"{counts['coverage_fraction']:.6f}",
        "realized_replacements": int(counts["realized_replacements"]),
        "realized_replacement_fraction": f"{counts['realized_replacement_fraction']:.6f}",
        "adjacency_mean_size": f"{counts['adjacency_mean_size']:.6f}",
        "adjacency_min_size": int(counts["adjacency_min_size"]),
        "adjacency_max_size": int(counts["adjacency_max_size"]),
        "laplace_denominator_z": f"{counts['laplace_denominator_z']:.6f}",
        "remote_rouge_l": f"{remote_rouge_l:.6f}",
        "extractor_conditional_ppl": f"{extractor_ppl:.6f}",
        "rouge_l": f"{final_rouge_l:.6f}",
        "seed": args.seed,
    }
    append_row(args.output_csv, row)
    print(
        f"[Result] epsilon={args.epsilon:g} remote_rouge_l={remote_rouge_l:.6f} "
        f"extractor_ppl={extractor_ppl:.6f} final_rouge_l={final_rouge_l:.6f} "
        f"coverage={row['prompt_coverage_fraction']}"
    )
    print(f"[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()
