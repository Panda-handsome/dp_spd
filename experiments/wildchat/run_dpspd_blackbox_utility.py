# -*- coding: utf-8 -*-
"""Run DP-SPD real-query utility on curated WildChat.

This runner intentionally omits dummy generation for utility-only measurement.
It applies the DP-SPD real-candidate prefix/suffix embedding perturbation and
supports two transport modes:

* token_projection: project noisy embeddings back to nearest tokens and send
  text to a local or OpenAI-compatible chat backend.
* embedding: preserve the earlier client/server behavior by sending noisy
  inputs_embeds directly to a local Hugging Face backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.api_only.run_projection_ablation import (  # noqa: E402
    build_projection_diagnostics,
    clip_embeddings,
    compute_private_layout,
    hash_text,
    lookup_embeddings_cpu,
    noise_scale,
    project_embeddings_to_tokens,
    sample_noise,
    token_accuracy,
    validate_local_path,
)
from experiments.wildchat.run_original_upper_bound import (  # noqa: E402
    ChatClient,
    DEFAULT_QWEN3_32B_PATH,
    build_generation_prompt,
    normalize_chat_messages,
    normalize_space,
    read_jsonl,
    rouge_l_f1,
    select_rows,
    serialize_api_messages,
    str2bool,
    trim_text,
)
from dpspd.util.utils import get_model_load_dtype  # noqa: E402


DEFAULT_PROJECTION_MODEL = "models/Qwen2.5-0.5B-Instruct"
DEFAULT_SYSTEM_PROMPT = "You are a careful assistant. Follow the user task and answer directly."


def parse_csv_list(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WildChat utility for DP-SPD token-projection or direct-embedding transport."
    )
    parser.add_argument("--qa_jsonl", type=str, default="data/wildchat/wildchat_qa500_independent_userprivacy.jsonl")
    parser.add_argument("--open_jsonl", type=str, default="data/wildchat/wildchat_open500_independent_userprivacy.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/wildchat_dpspd_blackbox_utility")
    parser.add_argument("--output_prefix", type=str, default="qwen3_32b_wildchat_dpspd_blackbox_k9")

    parser.add_argument("--backend", type=str, default="local_hf", choices=["local_hf", "openai_chat", "dry_run"])
    parser.add_argument(
        "--transport_mode",
        type=str,
        default="token_projection",
        choices=["token_projection", "embedding"],
        help=(
            "token_projection sends nearest-neighbor projected text; "
            "embedding sends noisy inputs_embeds directly and requires --backend local_hf."
        ),
    )
    parser.add_argument("--api_model", type=str, default="qwen3-32b")
    parser.add_argument("--openai_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--timeout_seconds", type=float, default=300.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep_seconds", type=float, default=5.0)
    parser.add_argument("--inter_request_sleep_seconds", type=float, default=0.0)

    parser.add_argument("--local_model_path", type=str, default=DEFAULT_QWEN3_32B_PATH)
    parser.add_argument("--local_dtype", type=str, default="float16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--local_device_map", type=str, default="auto")
    parser.add_argument("--local_device", type=str, default="cuda:0")
    parser.add_argument("--local_max_memory", type=str, default=None)
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--trust_remote_code", type=str2bool, default=False)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--enable_thinking", type=str2bool, default=False)

    parser.add_argument("--projection_model", type=str, default=DEFAULT_PROJECTION_MODEL)
    parser.add_argument("--projection_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--projection_device", type=str, default="cuda:0")
    parser.add_argument("--projection_max_tokens", type=int, default=4096)
    parser.add_argument("--projection_chunk_size", type=int, default=8)
    parser.add_argument("--projection_tie_tolerance", type=float, default=1e-6)
    parser.add_argument("--store_projection_token_diagnostics", type=str2bool, default=False)
    parser.add_argument("--max_projection_token_diagnostics", type=int, default=20)

    parser.add_argument("--target_epsilon", type=float, default=3.0)
    parser.add_argument("--dummy_k", type=int, default=9)
    parser.add_argument("--dummy_ratio", type=float, default=0.2)
    parser.add_argument("--epsilon_non_sensitive", type=float, default=3.0)
    parser.add_argument("--epsilon_private", type=float, default=7.82)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--noise_mechanism", type=str, default="Laplace", choices=["Laplace", "Gaussian"])

    parser.add_argument("--max_samples_qa", type=int, default=500)
    parser.add_argument("--max_samples_open", type=int, default=500)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_mode", type=str, default="first", choices=["first", "shuffle"])
    parser.add_argument("--seed", type=int, default=42, help="Noise seed. Keep sample_seed fixed across DP seeds.")
    parser.add_argument("--max_source_chars", type=int, default=2400)
    parser.add_argument("--use_messages", type=str2bool, default=True)
    parser.add_argument("--prompt_style", type=str, default="raw", choices=["raw", "instruct"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_completion_tokens_qa", type=int, default=256)
    parser.add_argument("--max_completion_tokens_open", type=int, default=512)

    parser.add_argument("--input_price_per_mtok", type=float, default=None)
    parser.add_argument("--output_price_per_mtok", type=float, default=None)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--dry_run_print_samples", type=int, default=2)
    return parser


def iter_examples(args) -> Iterable[Tuple[str, Dict[str, Any]]]:
    qa_rows = (
        select_rows(read_jsonl(Path(args.qa_jsonl)), args.max_samples_qa, args.sample_mode, args.sample_seed)
        if args.max_samples_qa > 0
        else []
    )
    open_rows = (
        select_rows(read_jsonl(Path(args.open_jsonl)), args.max_samples_open, args.sample_mode, args.sample_seed + 17)
        if args.max_samples_open > 0
        else []
    )
    for row in qa_rows:
        yield "answerable_qa", row
    for row in open_rows:
        yield "open_ended", row


def variant_for_transport(transport_mode: str) -> str:
    if transport_mode == "embedding":
        return "dpspd_embedding_local_hf"
    return "dpspd_projected_blackbox"


def row_transport_mode(row: Dict[str, Any]) -> str:
    # Older result files were produced before transport_mode existed; treat
    # those rows as token_projection so resume does not mix the two modes.
    return str(row.get("transport_mode") or "token_projection")


def load_existing_keys(path: Path, transport_mode: Optional[str] = None) -> set[Tuple[str, str]]:
    keys: set[Tuple[str, str]] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row_transport = row_transport_mode(row)
            if transport_mode is not None and row_transport != transport_mode:
                continue
            keys.add((str(row.get("subset")), str(row.get("id"))))
    return keys


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_projection_dtype(model_path: str, device: str, dtype_preference: str, local_files_only: bool):
    return get_model_load_dtype(
        model_path=model_path,
        device=device,
        dtype_preference=dtype_preference,
        local_files_only=local_files_only,
    )


def load_projection_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    validate_local_path(args.projection_model, "projection_model", args.local_files_only)
    device = torch.device(
        args.projection_device if torch.cuda.is_available() and str(args.projection_device).startswith("cuda") else "cpu"
    )
    dtype = resolve_projection_dtype(
        args.projection_model,
        str(device),
        args.projection_dtype,
        args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.projection_model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_kwargs = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.projection_model,
            dtype=dtype,
            **model_kwargs,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.projection_model,
            torch_dtype=dtype,
            **model_kwargs,
        ).to(device)
    model.eval()
    return tokenizer, model, device, dtype


def encode_text(tokenizer, text: str, max_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return enc["input_ids"][0].detach().cpu(), enc["attention_mask"][0].detach().cpu()


def build_noised_embedding_payload(
    text: str,
    sample_index: int,
    tokenizer,
    emb_layer,
    max_tokens: int,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    # Shared DP-SPD perturbation routine. token_projection mode consumes the
    # returned embeddings for nearest-token projection; embedding mode sends the
    # same tensors directly to a local HF model as inputs_embeds.
    input_ids, attn_mask = encode_text(tokenizer, text, max_tokens)
    valid_len = int(attn_mask.sum().item())
    original_ids = input_ids[:valid_len]
    valid_attn_mask = attn_mask[:valid_len]

    clean_emb = lookup_embeddings_cpu(original_ids, emb_layer)
    target_emb = clip_embeddings(clean_emb.clone(), args.clip_norm)
    real_len, _, prefix_len = compute_private_layout(valid_attn_mask, args.dummy_ratio)
    hidden_size = int(target_emb.size(-1)) if target_emb.numel() else 0

    prefix_scale = noise_scale(args.epsilon_non_sensitive, args.clip_norm, hidden_size)
    suffix_scale = noise_scale(args.epsilon_private, args.clip_norm, hidden_size)

    if prefix_len > 0:
        prefix_noise = sample_noise(
            target_emb[:prefix_len].shape,
            prefix_scale,
            args.noise_mechanism,
            seed=args.seed + sample_index + 20000,
            dtype=target_emb.dtype,
        )
        target_emb[:prefix_len] = target_emb[:prefix_len] + prefix_noise
    if prefix_len < real_len:
        suffix_noise = sample_noise(
            target_emb[prefix_len:real_len].shape,
            suffix_scale,
            args.noise_mechanism,
            seed=args.seed + sample_index + 50000,
            dtype=target_emb.dtype,
        )
        target_emb[prefix_len:real_len] = target_emb[prefix_len:real_len] + suffix_noise

    diagnostics = {
        "target_epsilon": float(args.target_epsilon),
        "dummy_k": int(args.dummy_k),
        "dummy_ratio": float(args.dummy_ratio),
        "epsilon_non_sensitive": float(args.epsilon_non_sensitive),
        "epsilon_private": float(args.epsilon_private),
        "prefix_len": int(prefix_len),
        "suffix_len": int(real_len - prefix_len),
        "embedding_token_count": int(real_len),
        "noised_text_hash": hash_text(text),
    }
    return original_ids, valid_attn_mask[:real_len], target_emb[:real_len], diagnostics


def build_projected_user_text(
    user_text: str,
    sample_index: int,
    tokenizer,
    projection_model,
    original_api_prompt: str,
    args,
) -> Tuple[str, Dict[str, Any]]:
    emb_layer = projection_model.get_input_embeddings()
    # In the black-box text path the client may use a small/proxy embedding
    # model only for noising and projection. The server/backend then receives
    # plain text, never inputs_embeds.
    original_ids, _, target_emb, noise_diagnostics = build_noised_embedding_payload(
        user_text,
        sample_index,
        tokenizer,
        emb_layer,
        args.projection_max_tokens,
        args,
    )
    real_len = int(target_emb.size(0))

    projected_ids = project_embeddings_to_tokens(
        target_emb[:real_len],
        emb_layer=emb_layer,
        tokenizer=tokenizer,
        original_ids=original_ids,
        chunk_size=args.projection_chunk_size,
    )
    projected_text = tokenizer.decode(
        projected_ids.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    diagnostics = build_projection_diagnostics(
        target_emb[:real_len],
        emb_layer,
        tokenizer,
        original_ids,
        projected_ids,
        user_text,
        projected_text,
        original_api_prompt,
        original_api_prompt,
        args,
    )
    diagnostics.update(noise_diagnostics)
    diagnostics.update(
        {
            "projection_token_count": int(real_len),
            "projection_token_accuracy": token_accuracy(projected_ids, original_ids),
            "projected_source_hash": hash_text(projected_text),
        }
    )
    return projected_text, diagnostics


def build_embedding_transport_payload(
    original_prompt: str,
    original_messages: Optional[List[Dict[str, str]]],
    sample_index: int,
    runner: ChatClient,
    args,
) -> Tuple[str, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if runner.local_model is None or runner.local_tokenizer is None:
        raise ValueError("Embedding transport requires --backend local_hf.")

    # Direct embedding transport must noise the exact formatted prompt that the
    # local model would normally tokenize, including system/user chat template
    # tokens. Text-only OpenAI-compatible APIs cannot represent this payload.
    formatted_prompt = runner._format_chat_prompt(
        original_prompt,
        DEFAULT_SYSTEM_PROMPT,
        messages=original_messages,
    )
    emb_layer = runner.local_model.get_input_embeddings()
    _, attention_mask, noisy_embeddings, diagnostics = build_noised_embedding_payload(
        formatted_prompt,
        sample_index,
        runner.local_tokenizer,
        emb_layer,
        args.max_input_tokens,
        args,
    )
    diagnostics.update(
        {
            "embedding_prompt_hash": hash_text(formatted_prompt),
            "embedding_prompt_chars": len(formatted_prompt),
        }
    )
    return (
        formatted_prompt,
        noisy_embeddings.unsqueeze(0),
        attention_mask.unsqueeze(0),
        diagnostics,
    )


def make_projected_messages(
    row: Dict[str, Any],
    subset: str,
    original_user_text: str,
    projected_user_text: str,
    args,
) -> Optional[List[Dict[str, str]]]:
    if not args.use_messages:
        return None
    original_messages = normalize_chat_messages(row, original_user_text, subset, args)
    if not original_messages:
        return None
    messages = [dict(item) for item in original_messages]
    last_user_index = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            last_user_index = idx
            break
    if last_user_index is None:
        messages.append({"role": "user", "content": projected_user_text})
        return messages
    replacement = build_generation_prompt(projected_user_text, subset, args.prompt_style)
    messages[last_user_index]["content"] = replacement
    return messages


def write_config(path: Path, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.transport_mode == "embedding":
        note = (
            "Utility-only DP-SPD run: no dummy candidates are generated. "
            "The formatted chat prompt is embedded by the local backend model, "
            "noised with prefix/suffix budgets, and sent directly as inputs_embeds. "
            "This mode requires backend=local_hf and preserves the earlier client/server behavior."
        )
    else:
        note = (
            "Utility-only DP-SPD run: no dummy candidates are generated. "
            "The real query is embedded by the client projection model, noised with prefix/suffix budgets, "
            "projected back to nearest tokens, and sent as text to the backend model."
        )
    payload = {
        "args": vars(args),
        "variant": variant_for_transport(args.transport_mode),
        "transport_mode": args.transport_mode,
        "note": note,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def summarize(outputs_path: Path, summary_path: Path, args) -> None:
    rows: List[Dict[str, Any]] = []
    if outputs_path.exists():
        with open(outputs_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

    fields = [
        "variant",
        "transport_mode",
        "dummy_k",
        "epsilon_non_sensitive",
        "epsilon_private",
        "subset",
        "group",
        "samples",
        "rouge_l",
        "avg_latency_sec",
        "avg_prompt_chars",
        "avg_projected_prompt_chars",
        "avg_response_chars",
        "avg_projection_token_accuracy",
        "total_projection_changed_tokens",
        "projection_changed_samples",
        "projection_exact_token_recovery_rate",
        "source_text_exact_match_rate",
        "prompt_exact_match_rate",
        "avg_prefix_len",
        "avg_suffix_len",
        "total_prompt_tokens",
        "total_completion_tokens",
        "estimated_cost_usd",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        transport_modes = sorted({row_transport_mode(row) for row in rows})
        if not transport_modes:
            transport_modes = [args.transport_mode]
        for transport_mode in transport_modes:
            mode_rows = [
                row
                for row in rows
                if row_transport_mode(row) == transport_mode
            ]
            for subset in ["answerable_qa", "open_ended", "all"]:
                subset_rows = mode_rows if subset == "all" else [row for row in mode_rows if row.get("subset") == subset]
                for group in ["all", "privacy_signal", "nonprivacy"]:
                    if group == "privacy_signal":
                        group_rows = [row for row in subset_rows if bool(row.get("privacy_signal"))]
                    elif group == "nonprivacy":
                        group_rows = [row for row in subset_rows if not bool(row.get("privacy_signal"))]
                    else:
                        group_rows = list(subset_rows)
                    if not group_rows:
                        continue
                    rouge_values = [float(row.get("rouge_l", 0.0) or 0.0) for row in group_rows]
                    latencies = [float(row.get("latency_sec", 0.0) or 0.0) for row in group_rows]
                    prompt_chars = [len(str(row.get("api_prompt", ""))) for row in group_rows]
                    projected_prompt_chars = [
                        len(str(row.get("projected_api_prompt")))
                        for row in group_rows
                        if row.get("projected_api_prompt") is not None
                    ]
                    response_chars = [len(str(row.get("response", ""))) for row in group_rows]
                    projection_acc = [
                        float(row.get("projection_token_accuracy"))
                        for row in group_rows
                        if row.get("projection_token_accuracy") is not None
                    ]
                    changed_tokens = [
                        int(row.get("projection_changed_token_count") or 0)
                        for row in group_rows
                        if row.get("projection_changed_token_count") is not None
                    ]
                    changed_samples = [
                        bool(row.get("projection_changed_sample"))
                        for row in group_rows
                        if row.get("projection_changed_sample") is not None
                    ]
                    exact_recovery = [
                        bool(row.get("projection_exact_token_recovery"))
                        for row in group_rows
                        if row.get("projection_exact_token_recovery") is not None
                    ]
                    source_exact = [
                        bool(row.get("source_text_exact_match"))
                        for row in group_rows
                        if row.get("source_text_exact_match") is not None
                    ]
                    prompt_exact = [
                        bool(row.get("prompt_exact_match"))
                        for row in group_rows
                        if row.get("prompt_exact_match") is not None
                    ]
                    prefix_lens = [
                        float(row.get("prefix_len", 0) or 0)
                        for row in group_rows
                        if row.get("prefix_len") is not None
                    ]
                    suffix_lens = [
                        float(row.get("suffix_len", 0) or 0)
                        for row in group_rows
                        if row.get("suffix_len") is not None
                    ]
                    prompt_tokens = [
                        int((row.get("api_usage") or {}).get("prompt_tokens") or 0)
                        for row in group_rows
                        if row.get("api_usage")
                    ]
                    completion_tokens = [
                        int((row.get("api_usage") or {}).get("completion_tokens") or 0)
                        for row in group_rows
                        if row.get("api_usage")
                    ]
                    total_prompt_tokens = sum(prompt_tokens)
                    total_completion_tokens = sum(completion_tokens)
                    estimated_cost = None
                    if args.input_price_per_mtok is not None and args.output_price_per_mtok is not None:
                        estimated_cost = (
                            total_prompt_tokens * float(args.input_price_per_mtok)
                            + total_completion_tokens * float(args.output_price_per_mtok)
                        ) / 1_000_000
                    writer.writerow(
                        {
                            "variant": group_rows[0].get("variant") or variant_for_transport(transport_mode),
                            "transport_mode": transport_mode,
                            "dummy_k": args.dummy_k,
                            "epsilon_non_sensitive": f"{float(args.epsilon_non_sensitive):.4f}",
                            "epsilon_private": f"{float(args.epsilon_private):.4f}",
                            "subset": subset,
                            "group": group,
                            "samples": len(group_rows),
                            "rouge_l": f"{_mean(rouge_values):.6f}" if rouge_values else "",
                            "avg_latency_sec": f"{_mean(latencies):.4f}" if latencies else "",
                            "avg_prompt_chars": f"{_mean(prompt_chars):.2f}" if prompt_chars else "",
                            "avg_projected_prompt_chars": (
                                f"{_mean(projected_prompt_chars):.2f}" if projected_prompt_chars else ""
                            ),
                            "avg_response_chars": f"{_mean(response_chars):.2f}" if response_chars else "",
                            "avg_projection_token_accuracy": f"{_mean(projection_acc):.6f}" if projection_acc else "",
                            "total_projection_changed_tokens": sum(changed_tokens) if changed_tokens else "",
                            "projection_changed_samples": (
                                sum(1 for value in changed_samples if value) if changed_samples else ""
                            ),
                            "projection_exact_token_recovery_rate": (
                                f"{sum(1.0 for value in exact_recovery if value) / len(exact_recovery):.6f}"
                                if exact_recovery
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
                            "avg_prefix_len": f"{_mean(prefix_lens):.2f}" if prefix_lens else "",
                            "avg_suffix_len": f"{_mean(suffix_lens):.2f}" if suffix_lens else "",
                            "total_prompt_tokens": total_prompt_tokens if prompt_tokens else "",
                            "total_completion_tokens": total_completion_tokens if completion_tokens else "",
                            "estimated_cost_usd": f"{estimated_cost:.6f}" if estimated_cost is not None else "",
                        }
                    )


def main() -> None:
    args = get_parser().parse_args()
    if args.transport_mode == "embedding" and args.backend != "local_hf":
        raise ValueError("--transport_mode embedding requires --backend local_hf.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / f"{args.output_prefix}_prompts.jsonl"
    outputs_path = output_dir / f"{args.output_prefix}_outputs.jsonl"
    summary_path = output_dir / f"{args.output_prefix}_summary.csv"
    config_path = output_dir / f"{args.output_prefix}_config.json"

    write_config(config_path, args)
    if not args.resume:
        for path in (prompts_path, outputs_path, summary_path):
            if path.exists():
                path.unlink()

    runner = ChatClient(
        backend=args.backend,
        api_model=args.api_model,
        base_url=args.openai_base_url,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        local_model_path=args.local_model_path,
        local_dtype=args.local_dtype,
        local_device_map=args.local_device_map,
        local_device=args.local_device,
        local_max_memory=args.local_max_memory,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        max_input_tokens=args.max_input_tokens,
        enable_thinking=args.enable_thinking,
    )

    tokenizer = None
    projection_model = None
    projection_dtype = None
    if args.transport_mode == "token_projection":
        # Projection mode intentionally decouples the client-side embedding
        # model from the backend model so a text-only service can be evaluated.
        tokenizer, projection_model, projection_device, projection_dtype = load_projection_model(args)
        print(
            f"[Transport] token_projection | projection_model={args.projection_model} "
            f"| dtype={projection_dtype} | device={projection_device}"
        )
    else:
        print(
            f"[Transport] embedding | local_model={args.local_model_path} "
            "| noisy inputs_embeds are sent directly to the local HF backend"
        )

    existing_keys = load_existing_keys(outputs_path, args.transport_mode) if args.resume else set()
    prompt_existing_keys = load_existing_keys(prompts_path, args.transport_mode) if args.resume else set()
    mode = "a" if args.resume else "w"
    printed = 0
    total = 0
    variant = variant_for_transport(args.transport_mode)

    with open(outputs_path, mode, encoding="utf-8") as out_f:
        for sample_index, (subset, row) in enumerate(iter_examples(args)):
            total += 1
            row_id = str(row.get("id"))
            key = (subset, row_id)
            if key in existing_keys:
                print(f"[Skip] subset={subset} id={row_id}")
                continue

            original_user_text = trim_text(row.get("current_user_text") or row.get("text"), args.max_source_chars)
            reference = normalize_space(row.get("reference"))
            original_prompt = build_generation_prompt(original_user_text, subset, args.prompt_style)
            original_messages = normalize_chat_messages(row, original_user_text, subset, args)
            original_api_prompt = serialize_api_messages(original_messages, original_prompt)

            projected_user_text = None
            projected_prompt = original_prompt
            projected_messages = original_messages
            projected_api_prompt = None
            embedding_prompt_hash = None
            embedding_prompt_chars = None
            noisy_inputs_embeds = None
            noisy_attention_mask = None

            if args.transport_mode == "token_projection":
                assert tokenizer is not None and projection_model is not None
                # Noise the user query, project it back to text, then replace
                # only the latest user message while preserving conversation
                # context and roles.
                projected_user_text, diagnostics = build_projected_user_text(
                    original_user_text,
                    sample_index,
                    tokenizer,
                    projection_model,
                    original_api_prompt,
                    args,
                )
                projected_prompt = build_generation_prompt(projected_user_text, subset, args.prompt_style)
                projected_messages = make_projected_messages(
                    row,
                    subset,
                    original_user_text,
                    projected_user_text,
                    args,
                )
                projected_api_prompt = serialize_api_messages(projected_messages, projected_prompt)
                diagnostics["prompt_exact_match"] = projected_api_prompt == original_api_prompt
                active_prompt_hash = hash_text(projected_api_prompt)
            else:
                # Preserve the older DP-SPD deployment shape: the client sends
                # noised embeddings for the full chat prompt to the local model.
                formatted_prompt, noisy_inputs_embeds, noisy_attention_mask, diagnostics = build_embedding_transport_payload(
                    original_prompt,
                    original_messages,
                    sample_index,
                    runner,
                    args,
                )
                embedding_prompt_hash = diagnostics.get("embedding_prompt_hash")
                embedding_prompt_chars = diagnostics.get("embedding_prompt_chars")
                active_prompt_hash = hash_text(formatted_prompt)

            base_record = {
                "id": row_id,
                "sample_index": sample_index,
                "subset": subset,
                "mix_source_subset": row.get("mix_source_subset") or row.get("subset"),
                "privacy_signal": bool(row.get("privacy_signal")),
                "metadata": row.get("metadata", {}),
                "source_text": original_user_text,
                "projected_source_text": projected_user_text,
                "context_text": row.get("context_text"),
                "reference": reference,
                "api_prompt": original_api_prompt,
                "projected_api_prompt": projected_api_prompt,
                "api_prompt_hash": active_prompt_hash,
                "api_messages": projected_messages,
                "original_api_messages": original_messages,
                "backend": args.backend,
                "api_model": args.api_model,
                "local_model_path": args.local_model_path if args.backend == "local_hf" else None,
                "projection_model": args.projection_model if args.transport_mode == "token_projection" else None,
                "projection_dtype": str(projection_dtype) if projection_dtype is not None else None,
                "transport_mode": args.transport_mode,
                "embedding_prompt_hash": embedding_prompt_hash,
                "embedding_prompt_chars": embedding_prompt_chars,
                "variant": variant,
            }
            base_record.update(diagnostics)

            if key not in prompt_existing_keys:
                append_jsonl(prompts_path, base_record)
                prompt_existing_keys.add(key)

            if args.backend == "dry_run":
                if printed < args.dry_run_print_samples:
                    print("\n" + "=" * 80)
                    print(f"[Dry Run] subset={subset} id={row_id}")
                    dry_prompt = projected_api_prompt if projected_api_prompt is not None else original_api_prompt
                    print(dry_prompt[:2000])
                    printed += 1
                continue

            max_tokens = args.max_completion_tokens_qa if subset == "answerable_qa" else args.max_completion_tokens_open
            started = time.perf_counter()
            if args.transport_mode == "embedding":
                assert noisy_inputs_embeds is not None and noisy_attention_mask is not None
                response, usage = runner.generate_from_embeddings(
                    noisy_inputs_embeds,
                    noisy_attention_mask,
                    max_tokens=max_tokens,
                    temperature=args.temperature,
                )
            else:
                response, usage = runner.chat(
                    projected_prompt,
                    max_tokens=max_tokens,
                    temperature=args.temperature,
                    messages=projected_messages,
                )
            latency_sec = time.perf_counter() - started
            rouge_l = rouge_l_f1(response, reference) if reference else None

            record = dict(base_record)
            record.update(
                {
                    "response": response,
                    "rouge_l": rouge_l,
                    "api_usage": usage,
                    "latency_sec": latency_sec,
                    "seed": args.seed,
                }
            )
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            existing_keys.add(key)
            changed_or_tokens = (
                f"changed_tokens={record.get('projection_changed_token_count')}"
                if args.transport_mode == "token_projection"
                else f"embedding_tokens={record.get('embedding_token_count')}"
            )
            print(
                f"[Saved] subset={subset} id={row_id} k={args.dummy_k} "
                f"{changed_or_tokens} "
                f"latency={latency_sec:.2f}s rouge_l={rouge_l}"
            )
            if args.inter_request_sleep_seconds > 0:
                time.sleep(args.inter_request_sleep_seconds)

    if args.backend == "dry_run":
        print(f"[Prepared Prompts] {prompts_path}")
        print(f"[Config] {config_path}")
        return

    summarize(outputs_path, summary_path, args)
    print(f"[Examples seen] {total}")
    print(f"[Prompts] {prompts_path}")
    print(f"[Outputs] {outputs_path}")
    print(f"[Summary] {summary_path}")
    print(f"[Config] {config_path}")


if __name__ == "__main__":
    main()
