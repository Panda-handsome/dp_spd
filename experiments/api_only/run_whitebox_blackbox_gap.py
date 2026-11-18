# -*- coding: utf-8 -*-
"""Compare local white-box embedding inference with black-box token projection.

This experiment isolates the extra utility cost of an api-only deployment:
the white-box path feeds the same clean/noisy source embeddings directly to a
local causal LM through ``inputs_embeds``; the black-box path first projects
those embeddings back to nearest tokens and then runs ordinary text inference.
Dummy candidates are intentionally omitted, as this is a real-query utility
diagnostic.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpspd.util.utils import get_model_load_dtype, str2bool
from experiments.api_only.run_projection_ablation import (
    append_jsonl,
    build_noisy_embeddings,
    label_names_for_task,
    load_task_examples,
    lookup_embeddings_cpu,
    parse_csv_list,
    project_embeddings_to_tokens,
    prompt_from_private_source,
    prompt_parts_from_private_source,
    score_response,
    token_accuracy,
    validate_local_path,
)


SUPPORTED_GAP_VARIANTS = ("clean", "dprivacy", "dpspd")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure local white-box embedding vs black-box token-projection utility gap."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="custom_mcq",
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
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--min_words", type=int, default=8)
    parser.add_argument("--max_source_chars", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="Seed used only for dataset subsampling/order. Defaults to --seed for backward compatibility.",
    )

    parser.add_argument("--server_model", type=str, default="models/Llama-2-7b-chat-hf")
    parser.add_argument(
        "--server_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--projection_chunk_size", type=int, default=8)
    parser.add_argument("--variants", type=str, default="clean,dpspd")

    parser.add_argument("--dummy_ratio", type=float, default=0.4)
    parser.add_argument("--epsilon_private", type=float, default=2.08)
    parser.add_argument("--epsilon_non_sensitive", type=float, default=1.5)
    parser.add_argument("--baseline_epsilon", type=float, default=None)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--noise_mechanism", type=str, default="Laplace", choices=["Laplace", "Gaussian"])

    parser.add_argument("--max_completion_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    parser.add_argument("--dry_run_print_samples", type=int, default=2)
    parser.add_argument(
        "--store_token_ids",
        type=str2bool,
        default=False,
        help="Store original/projected private-source token ids for spot-checking projection diagnostics.",
    )
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--output_dir", type=str, default="outputs/whitebox_blackbox_gap")
    parser.add_argument("--output_prefix", type=str, default="whitebox_blackbox_gap")
    return parser


def validate_gap_variants(variants: Sequence[str]) -> None:
    unsupported = [variant for variant in variants if variant not in SUPPORTED_GAP_VARIANTS]
    if unsupported:
        raise ValueError(f"Unsupported variants: {unsupported}. Supported: {SUPPORTED_GAP_VARIANTS}")


def output_key(record: Dict[str, Any]) -> Tuple[int, str, str]:
    return int(record["sample_id"]), str(record["variant"]), str(record["path"])


def load_existing_gap_outputs(path: Path) -> Dict[Tuple[int, str, str], Dict[str, Any]]:
    rows = {}
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[output_key(row)] = row
    return rows


def load_server_model(args):
    validate_local_path(args.server_model, "server_model", args.local_files_only)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    dtype = get_model_load_dtype(
        model_path=args.server_model,
        device=str(device),
        dtype_preference=args.server_dtype,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.server_model, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.server_model,
            local_files_only=args.local_files_only,
            dtype=dtype,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.server_model,
            local_files_only=args.local_files_only,
            torch_dtype=dtype,
        ).to(device)
    model.eval()
    clear_generation_length_defaults(model)
    return tokenizer, model, device, dtype


def clear_generation_length_defaults(model) -> None:
    gen_config = getattr(model, "generation_config", None)
    if gen_config is None:
        return
    if hasattr(gen_config, "max_length"):
        gen_config.max_length = None
    if hasattr(gen_config, "min_length"):
        gen_config.min_length = None
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(gen_config, attr):
            setattr(gen_config, attr, None)


def encode_text(tokenizer, text: str, add_special_tokens: bool, max_length: Optional[int] = None) -> torch.Tensor:
    if not text:
        return torch.empty((0,), dtype=torch.long)
    kwargs = {
        "add_special_tokens": add_special_tokens,
        "return_tensors": "pt",
    }
    if max_length is not None:
        kwargs["truncation"] = True
        kwargs["max_length"] = max_length
    encoded = tokenizer(text, **kwargs)
    return encoded["input_ids"][0].detach().cpu().to(torch.long)


def lookup_embeddings_device(input_ids: torch.Tensor, emb_layer: torch.nn.Embedding) -> torch.Tensor:
    if input_ids.numel() == 0:
        return torch.empty((0, emb_layer.embedding_dim), dtype=emb_layer.weight.dtype, device=emb_layer.weight.device)
    with torch.no_grad():
        return emb_layer(input_ids.to(emb_layer.weight.device)).detach()


def tensor_digest(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def encode_prompt_with_source_span(
    tokenizer,
    before_text: str,
    source_text: str,
    after_text: str,
    max_length: int,
) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize the full prompt once and identify tokens overlapping source_text."""
    full_prompt = before_text + source_text + after_text
    source_start = len(before_text)
    source_end = source_start + len(source_text)

    if getattr(tokenizer, "is_fast", False):
        try:
            encoded = tokenizer(
                full_prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"][0].detach().cpu().to(torch.long)
            attention_mask = encoded["attention_mask"][0].detach().cpu().to(torch.long)
            offsets = encoded["offset_mapping"][0].detach().cpu().tolist()
            source_positions = [
                idx
                for idx, ((begin, end), keep) in enumerate(zip(offsets, attention_mask.tolist()))
                if keep and end > begin and end > source_start and begin < source_end
            ]
            if source_positions:
                return (
                    full_prompt,
                    input_ids,
                    attention_mask,
                    torch.tensor(source_positions, dtype=torch.long),
                )
        except (NotImplementedError, TypeError, ValueError):
            pass

    before_ids = encode_text(tokenizer, before_text, add_special_tokens=True)
    source_ids = encode_text(tokenizer, source_text, add_special_tokens=False)
    after_ids = encode_text(tokenizer, after_text, add_special_tokens=False)
    full_ids = torch.cat([before_ids, source_ids, after_ids], dim=0)
    if max_length > 0:
        full_ids = full_ids[:max_length]
    attention_mask = torch.ones_like(full_ids, dtype=torch.long)
    start = min(before_ids.numel(), full_ids.numel())
    end = min(before_ids.numel() + source_ids.numel(), full_ids.numel())
    source_positions = torch.arange(start, end, dtype=torch.long)
    return full_prompt, full_ids, attention_mask, source_positions


def build_prompt_embeddings(
    full_ids: torch.Tensor,
    source_positions: torch.Tensor,
    source_embeddings_cpu: torch.Tensor,
    emb_layer: torch.nn.Embedding,
) -> torch.Tensor:
    full_embeddings = lookup_embeddings_cpu(full_ids, emb_layer)
    if source_positions.numel() > 0:
        full_embeddings[source_positions] = source_embeddings_cpu[: source_positions.numel()].to(full_embeddings.dtype)
    return full_embeddings


def build_source_embeddings(
    source_ids: torch.Tensor,
    emb_layer: torch.nn.Embedding,
    args,
    variant: str,
    sample_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    attn_mask = torch.ones_like(source_ids, dtype=torch.long)
    if variant == "clean":
        source_emb = lookup_embeddings_cpu(source_ids, emb_layer)
    elif variant == "dprivacy":
        source_emb = build_noisy_embeddings(source_ids, attn_mask, emb_layer, args, "dprivacy_projected", sample_id)
    elif variant == "dpspd":
        source_emb = build_noisy_embeddings(source_ids, attn_mask, emb_layer, args, "dpspd_projected", sample_id)
    else:
        raise ValueError(variant)
    return source_emb, attn_mask


def generation_kwargs(args, tokenizer) -> Dict[str, Any]:
    kwargs = {
        "max_new_tokens": args.max_completion_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = args.temperature
    else:
        kwargs["do_sample"] = False
    return kwargs


def run_blackbox_text(model, tokenizer, prompt: str, args, device: torch.device) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
    with torch.no_grad():
        generated = model.generate(**encoded, **generation_kwargs(args, tokenizer))
    new_tokens = generated[:, encoded["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens.detach().cpu(), skip_special_tokens=True)[0].strip()


def run_whitebox_embeddings(
    model,
    tokenizer,
    before_text: str,
    source_embeddings_cpu: torch.Tensor,
    after_text: str,
    args,
) -> str:
    emb_layer = model.get_input_embeddings()
    before_ids = encode_text(tokenizer, before_text, add_special_tokens=True)
    after_ids = encode_text(tokenizer, after_text, add_special_tokens=False)
    before_emb = lookup_embeddings_device(before_ids, emb_layer)
    after_emb = lookup_embeddings_device(after_ids, emb_layer)
    source_emb = source_embeddings_cpu.to(device=emb_layer.weight.device, dtype=emb_layer.weight.dtype)
    full_emb = torch.cat([before_emb, source_emb, after_emb], dim=0).unsqueeze(0)
    attention_mask = torch.ones(full_emb.shape[:2], dtype=torch.long, device=full_emb.device)
    with torch.no_grad():
        generated = model.generate(
            inputs_embeds=full_emb,
            attention_mask=attention_mask,
            **generation_kwargs(args, tokenizer),
        )
    return tokenizer.batch_decode(generated.detach().cpu(), skip_special_tokens=True)[0].strip()


def run_whitebox_prompt_embeddings(
    model,
    tokenizer,
    prompt_ids_cpu: torch.Tensor,
    prompt_embeddings_cpu: torch.Tensor,
    attention_mask_cpu: torch.Tensor,
    args,
) -> str:
    full_emb = prompt_embeddings_cpu.to(
        device=model.get_input_embeddings().weight.device,
        dtype=model.get_input_embeddings().weight.dtype,
    ).unsqueeze(0)
    prompt_ids = prompt_ids_cpu.to(device=full_emb.device, dtype=torch.long).unsqueeze(0)
    attention_mask = attention_mask_cpu.to(device=full_emb.device, dtype=torch.long).unsqueeze(0)
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt_ids,
            inputs_embeds=full_emb,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **generation_kwargs(args, tokenizer),
        )
    new_tokens = generated[:, prompt_ids.shape[1]:]
    return tokenizer.batch_decode(new_tokens.detach().cpu(), skip_special_tokens=True)[0].strip()


def projection_diagnostics(
    projected_ids: torch.Tensor,
    original_ids: torch.Tensor,
    projected_prompt: str,
    original_prompt: str,
) -> Dict[str, Any]:
    n = min(projected_ids.numel(), original_ids.numel())
    if n == 0:
        changed = 0
    else:
        changed = int(projected_ids[:n].ne(original_ids[:n].cpu()).sum().item())
    return {
        "source_token_count": int(n),
        "projected_token_changed_count": changed,
        "projected_sample_changed": bool(changed > 0),
        "projection_exact_recovery": bool(changed == 0 and projected_ids.numel() == original_ids.numel()),
        "exact_prompt_recovery": projected_prompt == original_prompt,
    }


def summarize_gap_outputs(
    outputs_path: Path,
    summary_path: Path,
    variants: Sequence[str],
    metric_type: str,
    labels: Sequence[str],
) -> None:
    rows_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {
        (variant, path): []
        for variant in variants
        for path in ("whitebox_embedding", "blackbox_projection")
    }
    if outputs_path.exists():
        with open(outputs_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row.get("variant"), row.get("path"))
                if key in rows_by_key:
                    rows_by_key[key].append(row)

    fields = [
        "variant",
        "path",
        "samples",
        "accuracy",
        "rouge_l",
        "avg_projection_token_accuracy",
        "total_projected_token_changes",
        "projection_changed_samples",
        "exact_prompt_recovery_rate",
        "changed_subset_samples",
        "changed_subset_accuracy",
        "clean_blackbox_pred_mismatch_count",
        "clean_blackbox_response_mismatch_count",
        "avg_response_chars",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant in variants:
            for path in ("whitebox_embedding", "blackbox_projection"):
                rows = rows_by_key[(variant, path)]
                scored_rows = [score_response(row, metric_type, labels) for row in rows]
                correct_values = [
                    float(scored["correct"])
                    for scored in scored_rows
                    if scored.get("correct") is not None
                ]
                rouge_values = [
                    float(scored["rouge_l"])
                    for scored in scored_rows
                    if scored.get("rouge_l") is not None
                ]
                proj_values = [
                    float(row["projection_token_accuracy"])
                    for row in rows
                    if row.get("projection_token_accuracy") is not None
                ]
                has_projection_change_diag = any("projected_token_changed_count" in row for row in rows)
                token_changes = (
                    [int(row.get("projected_token_changed_count", 0) or 0) for row in rows]
                    if has_projection_change_diag
                    else []
                )
                changed_rows = (
                    [
                        row
                        for row in rows
                        if int(row.get("projected_token_changed_count", 0) or 0) > 0
                    ]
                    if has_projection_change_diag
                    else []
                )
                exact_prompt = [
                    bool(row.get("exact_prompt_recovery"))
                    for row in rows
                    if row.get("exact_prompt_recovery") is not None
                ]
                changed_scored = [
                    score_response(row, metric_type, labels)
                    for row in changed_rows
                ]
                changed_correct = [
                    float(scored["correct"])
                    for scored in changed_scored
                    if scored.get("correct") is not None
                ]
                response_chars = [len(str(row.get("response", ""))) for row in rows]
                clean_blackbox_by_sample: Dict[int, Dict[str, Any]] = {}
                if variant != "clean" and path == "blackbox_projection":
                    clean_rows = rows_by_key.get(("clean", "blackbox_projection"), [])
                    clean_blackbox_by_sample = {
                        int(row["sample_id"]): score_response(row, metric_type, labels)
                        for row in clean_rows
                    }
                pred_mismatch = 0
                response_mismatch = 0
                if clean_blackbox_by_sample:
                    for row, scored in zip(rows, scored_rows):
                        clean_row = clean_blackbox_by_sample.get(int(row["sample_id"]))
                        if not clean_row:
                            continue
                        if scored.get("pred_label") != clean_row.get("pred_label"):
                            pred_mismatch += 1
                        if str(scored.get("response", "")) != str(clean_row.get("response", "")):
                            response_mismatch += 1
                writer.writerow(
                    {
                        "variant": variant,
                        "path": path,
                        "samples": len(rows),
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
                        "avg_projection_token_accuracy": (
                            f"{sum(proj_values) / len(proj_values):.6f}" if proj_values else ""
                        ),
                        "total_projected_token_changes": (
                            sum(token_changes) if has_projection_change_diag else ""
                        ),
                        "projection_changed_samples": (
                            len(changed_rows) if has_projection_change_diag else ""
                        ),
                        "exact_prompt_recovery_rate": (
                            f"{sum(1.0 for value in exact_prompt if value) / len(exact_prompt):.6f}"
                            if exact_prompt
                            else ""
                        ),
                        "changed_subset_samples": (
                            len(changed_rows) if has_projection_change_diag else ""
                        ),
                        "changed_subset_accuracy": (
                            f"{sum(changed_correct) / len(changed_correct):.6f}"
                            if metric_type in ("classification", "mcq") and changed_correct
                            else ""
                        ),
                        "clean_blackbox_pred_mismatch_count": pred_mismatch if clean_blackbox_by_sample else "",
                        "clean_blackbox_response_mismatch_count": response_mismatch if clean_blackbox_by_sample else "",
                        "avg_response_chars": (
                            f"{sum(response_chars) / len(response_chars):.2f}" if response_chars else ""
                        ),
                    }
                )


def write_config(path: Path, args, variants: Sequence[str], labels: Sequence[str], metric_type: str) -> None:
    payload = {
        "args": vars(args),
        "variants": list(variants),
        "labels": list(labels),
        "metric_type": metric_type,
        "note": (
            "White-box path feeds source embeddings directly through inputs_embeds. "
            "Black-box path projects the same source embeddings to nearest tokens before text inference. "
            "No dummy candidates are used."
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = get_parser().parse_args()
    variants = parse_csv_list(args.variants)
    validate_gap_variants(variants)
    labels = label_names_for_task(args.task, args)
    data_args = copy.copy(args)
    if args.sample_seed is not None:
        data_args.seed = args.sample_seed
    examples, labels, metric_type = load_task_examples(data_args)
    tokenizer, model, device, dtype = load_server_model(args)
    emb_layer = model.get_input_embeddings()
    print(f"[Server Model] {args.server_model} | dtype={dtype} | device={device}")
    print(f"[Data] task={args.task} | examples={len(examples)} | metric={metric_type}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = output_dir / f"{args.output_prefix}_outputs.jsonl"
    summary_path = output_dir / f"{args.output_prefix}_summary.csv"
    config_path = output_dir / f"{args.output_prefix}_config.json"
    write_config(config_path, args, variants, labels, metric_type)

    if not args.resume:
        for path in (outputs_path, summary_path):
            if path.exists():
                path.unlink()

    existing = load_existing_gap_outputs(outputs_path) if args.resume else {}
    printed = 0
    for example in examples:
        task_name = str(example.metadata.get("task_name", args.task))
        before_text, _, after_text = prompt_parts_from_private_source(
            example.source_text,
            labels,
            metric_type,
            task_name,
        )
        original_prompt = prompt_from_private_source(example.source_text, labels, metric_type, task_name)
        full_prompt, full_ids, full_attention_mask, source_positions = encode_prompt_with_source_span(
            tokenizer,
            before_text,
            example.source_text,
            after_text,
            args.max_length,
        )
        source_ids = full_ids[source_positions]
        if source_ids.numel() == 0:
            continue
        for variant in variants:
            source_embeddings, _ = build_source_embeddings(source_ids, emb_layer, args, variant, example.sample_id)
            prompt_embeddings = build_prompt_embeddings(full_ids, source_positions, source_embeddings, emb_layer)
            projected_ids = project_embeddings_to_tokens(
                source_embeddings,
                emb_layer=emb_layer,
                tokenizer=tokenizer,
                original_ids=source_ids,
                chunk_size=args.projection_chunk_size,
            )
            projected_source = tokenizer.decode(projected_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()
            projected_prompt = prompt_from_private_source(projected_source, labels, metric_type, task_name)
            proj_acc = token_accuracy(projected_ids, source_ids)
            diagnostics = projection_diagnostics(projected_ids, source_ids, projected_prompt, original_prompt)

            base = {
                "sample_id": example.sample_id,
                "variant": variant,
                "task": args.task,
                "metric_type": metric_type,
                "label": example.label,
                "reference": example.reference,
                "source_text": example.source_text,
                "projected_source": projected_source,
                "projection_token_accuracy": proj_acc,
                "source_embedding_sha256_16": tensor_digest(source_embeddings),
                "prompt_embedding_sha256_16": tensor_digest(prompt_embeddings),
                "metadata": example.metadata,
            }
            base.update(diagnostics)
            if args.store_token_ids:
                base["source_token_ids"] = source_ids.detach().cpu().tolist()
                base["projected_token_ids"] = projected_ids.detach().cpu().tolist()

            if args.dry_run and printed < args.dry_run_print_samples:
                print("\n" + "=" * 80)
                print(f"[Dry Run] sample={example.sample_id} variant={variant}")
                print("[Black-box projected prompt]")
                print(projected_prompt[:2000])
                printed += 1

            if args.dry_run:
                continue

            for path_name in ("whitebox_embedding", "blackbox_projection"):
                key = (example.sample_id, variant, path_name)
                if key in existing:
                    continue
                if path_name == "whitebox_embedding":
                    response = run_whitebox_prompt_embeddings(
                        model,
                        tokenizer,
                        full_ids,
                        prompt_embeddings,
                        full_attention_mask,
                        args,
                    )
                    api_prompt = before_text + "[SOURCE_EMBEDDINGS]" + after_text
                else:
                    response = run_blackbox_text(model, tokenizer, projected_prompt, args, device)
                    api_prompt = projected_prompt
                record = dict(base)
                record["path"] = path_name
                record["api_prompt"] = api_prompt
                record["response"] = response
                record.update(score_response(record, metric_type, labels))
                append_jsonl(outputs_path, record)
                print(f"[Saved] sample={example.sample_id} variant={variant} path={path_name}")

    if args.dry_run:
        print(f"\n[Config] {config_path}")
        return

    summarize_gap_outputs(outputs_path, summary_path, variants, metric_type, labels)
    print(f"[Outputs] {outputs_path}")
    print(f"[Summary] {summary_path}")
    print(f"[Config] {config_path}")


if __name__ == "__main__":
    main()
