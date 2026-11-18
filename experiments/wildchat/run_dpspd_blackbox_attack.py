# -*- coding: utf-8 -*-
"""WildChat attacks for the DP-SPD black-box projected-token path.

The attacker sees the same surface that a black-box LLM service would see in
the API-only DP-SPD deployment: k+1 projected text candidates, exactly one of
which is the real user query. No attack in this runner consumes inputs_embeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
import numpy as np
import torch
import torch.nn as nn
from openai import OpenAI
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpspd.client import BatchConfig, DummyConfig, LDPNoiseConfig, PrivacyDataset  # noqa: E402
from dpspd.util.utils import load_local_causal_lm, str2bool  # noqa: E402
from experiments.wildchat.run_original_upper_bound import (  # noqa: E402
    ChatClient,
    normalize_space,
    read_jsonl,
    select_rows,
    trim_text,
)


DEFAULT_PROJECTION_MODEL = "models/Qwen2.5-0.5B-Instruct"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MIA/GPT attacks on WildChat DP-SPD black-box candidate groups."
    )
    parser.add_argument("--qa_jsonl", type=str, default="data/wildchat/wildchat_qa500_independent_userprivacy.jsonl")
    parser.add_argument("--open_jsonl", type=str, default="data/wildchat/wildchat_open500_independent_userprivacy.jsonl")
    parser.add_argument("--subsets", type=str, default="answerable_qa,open_ended")
    parser.add_argument("--max_groups_qa", type=int, default=100)
    parser.add_argument("--max_groups_open", type=int, default=100)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_mode", type=str, default="first", choices=["first", "shuffle"])
    parser.add_argument("--max_source_chars", type=int, default=2400)

    parser.add_argument("--projection_model", type=str, default=DEFAULT_PROJECTION_MODEL)
    parser.add_argument("--projection_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--projection_device", type=str, default="cuda:0")
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--trust_remote_code", type=str2bool, default=False)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--projection_chunk_size", type=int, default=8)

    parser.add_argument("--target_epsilon", type=float, default=3.0)
    parser.add_argument("--dummy_k", type=int, default=9)
    parser.add_argument("--dummy_ratio", type=float, default=0.2)
    parser.add_argument("--epsilon_non_sensitive", type=float, default=3.0)
    parser.add_argument("--epsilon_private", type=float, default=7.82)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--noise_mechanism", type=str, default="Laplace", choices=["Laplace", "Gaussian"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--attack_modes",
        type=str,
        default="mia",
        help="Comma-separated: mia,gpt_attack",
    )
    parser.add_argument(
        "--mia_attack_model",
        type=str,
        default=None,
        help="Local text scorer for MIA. Defaults to --projection_model, so Qwen3-32B is not loaded.",
    )
    parser.add_argument("--mia_attack_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--mia_device_map", type=str, default="none")
    parser.add_argument("--mia_device", type=str, default="cuda:0")
    parser.add_argument("--mia_max_memory", type=str, default=None)
    parser.add_argument("--mia_max_length", type=int, default=512)
    parser.add_argument("--mia_k_ratio", type=float, default=0.20)

    parser.add_argument("--gpt_model", type=str, default="gpt-4o")
    parser.add_argument("--gpt_temperature", type=float, default=0.0)
    parser.add_argument("--gpt_max_completion_tokens", type=int, default=900)
    parser.add_argument("--gpt_max_retries", type=int, default=5)
    parser.add_argument("--gpt_retry_sleep_seconds", type=float, default=5.0)
    parser.add_argument("--gpt_retry_backoff_multiplier", type=float, default=2.0)
    parser.add_argument("--gpt_retry_jitter_seconds", type=float, default=1.0)
    parser.add_argument("--openai_timeout_seconds", type=float, default=180.0)
    parser.add_argument("--openai_connect_timeout_seconds", type=float, default=20.0)
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--inter_request_sleep_seconds", type=float, default=0.5)
    parser.add_argument("--save_gpt_prompts", type=str2bool, default=True)

    parser.add_argument("--output_dir", type=str, default="outputs/wildchat_dpspd_blackbox_attack")
    parser.add_argument("--output_prefix", type=str, default="qwen3_32b_wildchat_dpspd_blackbox_attack_k9_eps3")
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    parser.add_argument("--dry_run_print_groups", type=int, default=1)
    return parser


def parse_csv_list(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


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


def load_selected_rows(args) -> Dict[str, List[Dict[str, Any]]]:
    selected: Dict[str, List[Dict[str, Any]]] = {}
    requested = set(parse_csv_list(args.subsets))
    if "qa" in requested:
        requested.add("answerable_qa")
    if "open" in requested:
        requested.add("open_ended")
    if "answerable_qa" in requested:
        rows = (
            select_rows(read_jsonl(Path(args.qa_jsonl)), args.max_groups_qa, args.sample_mode, args.sample_seed)
            if args.max_groups_qa > 0
            else []
        )
        selected["answerable_qa"] = rows
    if "open_ended" in requested:
        rows = (
            select_rows(
                read_jsonl(Path(args.open_jsonl)),
                args.max_groups_open,
                args.sample_mode,
                args.sample_seed + 17,
            )
            if args.max_groups_open > 0
            else []
        )
        selected["open_ended"] = rows
    if not selected:
        raise ValueError(f"No valid subsets requested: {args.subsets!r}")
    return selected


def row_user_text(row: Dict[str, Any], args) -> str:
    return trim_text(row.get("current_user_text") or row.get("text"), args.max_source_chars)


def decode_ids(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(
        token_ids.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def first_model_device(model) -> torch.device:
    try:
        return model.device
    except AttributeError:
        pass
    for param in model.parameters():
        if str(param.device) != "meta":
            return param.device
    return torch.device("cpu")


def load_mia_model(args) -> Tuple[Any, Any]:
    if not args.mia_attack_model:
        args.mia_attack_model = args.projection_model
    loader = ChatClient(
        backend="local_hf",
        api_model="mia-scorer",
        base_url="http://127.0.0.1:8000/v1",
        api_key_env=args.api_key_env,
        timeout_seconds=args.openai_timeout_seconds,
        max_retries=1,
        retry_sleep_seconds=1.0,
        local_model_path=args.mia_attack_model,
        local_dtype=args.mia_attack_dtype,
        local_device_map=args.mia_device_map,
        local_device=args.mia_device,
        local_max_memory=args.mia_max_memory,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=None,
        max_input_tokens=args.mia_max_length,
        enable_thinking=False,
    )
    assert loader.local_tokenizer is not None and loader.local_model is not None
    return loader.local_tokenizer, loader.local_model


def build_privacy_dataset(texts: Sequence[str], tokenizer, model, args) -> PrivacyDataset:
    batch_cfg = BatchConfig(
        max_length=args.max_length,
        device=torch.device("cpu"),
        hidden_size=model.config.hidden_size,
    )
    dummy_cfg = DummyConfig(
        m_private_tokens=0,
        sensitive_ratio=args.dummy_ratio,
        n_dummies_per_text=args.dummy_k,
        dummy_mode="generation",
        dummy_corpus=None,
        pos_mapping_path=None,
    )
    ldp_cfg = LDPNoiseConfig(
        mechanism=args.noise_mechanism,
        epsilon_private=args.epsilon_private,
        epsilon_non_sensitive=args.epsilon_non_sensitive,
        clip_norm=args.clip_norm,
        seed=args.seed,
    )
    return PrivacyDataset(
        list(texts),
        tokenizer,
        batch_cfg,
        dummy_cfg,
        ldp_cfg,
        seed=args.seed,
        shuffle_mode="block",
        pretrained_embeddings=model.get_input_embeddings(),
        base_model=model,
    )


def load_projection_model(args):
    device = args.projection_device if torch.cuda.is_available() else "cpu"
    tokenizer, model, load_dtype = load_local_causal_lm(
        model_path=args.projection_model,
        device=device,
        dtype_preference=args.projection_dtype,
        local_files_only=args.local_files_only,
    )
    return tokenizer, model, load_dtype


def project_batch_candidates(batch: Dict[str, Any], dataset: PrivacyDataset, tokenizer, args) -> Dict[str, Any]:
    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"]
    noisy_embs = batch["noisy_embeddings"]
    is_real_flags = batch["is_real"].bool()
    group_indices = batch["group_index"].long()
    dummy_indices = batch["dummy_index"].long()
    prefix_lens = batch["prefix_len"].long()
    private_counts = batch["private_token_count"].long()

    actual_real_positions = torch.nonzero(is_real_flags, as_tuple=False).flatten()
    if actual_real_positions.numel() != 1:
        raise ValueError(f"Expected exactly one real candidate, found {actual_real_positions.numel()}.")
    actual_real_index = int(actual_real_positions[0].item())
    group_index = int(group_indices[0].item())

    candidates: List[Dict[str, Any]] = []
    projection_acc_values: List[float] = []
    changed_tokens_total = 0
    changed_samples = 0
    for idx in range(input_ids.shape[0]):
        valid_len = int(attn_mask[idx].sum().item())
        prefix_len = int(prefix_lens[idx].item())
        projected_ids = dataset._project_embeddings_to_tokens(
            noisy_embs[idx, :valid_len],
            original_ids=input_ids[idx, :valid_len],
            chunk_size=args.projection_chunk_size,
        )
        original_ids = input_ids[idx, :valid_len].detach().cpu()
        clean_text = decode_ids(tokenizer, original_ids)
        projected_text = decode_ids(tokenizer, projected_ids)
        projected_prefix = decode_ids(tokenizer, projected_ids[:prefix_len])
        projected_suffix = decode_ids(tokenizer, projected_ids[prefix_len:valid_len])
        projection_token_accuracy = float(projected_ids.eq(original_ids).float().mean().item()) if valid_len else 0.0
        changed_count = int(projected_ids.ne(original_ids).sum().item())
        projection_acc_values.append(projection_token_accuracy)
        changed_tokens_total += changed_count
        changed_samples += int(changed_count > 0)
        candidates.append(
            {
                "candidate_index": idx,
                "is_real_hidden": bool(is_real_flags[idx].item()),
                "dummy_index_hidden": int(dummy_indices[idx].item()),
                "clean_candidate_text": clean_text,
                "projected_text": projected_text,
                "projected_prefix_text": projected_prefix,
                "projected_suffix_text": projected_suffix,
                "projection_token_accuracy": projection_token_accuracy,
                "projection_changed_token_count": changed_count,
                "token_count": valid_len,
                "prefix_len": prefix_len,
                "suffix_len": int(private_counts[idx].item()),
            }
        )

    shared_prefix_text = candidates[0]["projected_prefix_text"] if candidates else ""
    return {
        "group_index": group_index,
        "actual_real_index": actual_real_index,
        "prefix_len": int(prefix_lens[actual_real_index].item()),
        "suffix_len": int(private_counts[actual_real_index].item()),
        "candidates": candidates,
        "shared_prefix_text": shared_prefix_text,
        "avg_projection_token_accuracy": sum(projection_acc_values) / max(len(projection_acc_values), 1),
        "total_projection_changed_tokens": changed_tokens_total,
        "projection_changed_samples": changed_samples,
    }


def score_text_candidates_mia(
    candidate_texts: Sequence[str],
    tokenizer,
    model,
    args,
) -> Dict[str, Any]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = first_model_device(model)
    enc = tokenizer(
        list(candidate_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.mia_max_length,
    )
    enc = {key: value.to(device) for key, value in enc.items()}
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        outputs = model(**enc)
    logits = outputs.logits
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attn_mask[..., 1:].contiguous()
    token_losses = loss_fct(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    ).view(input_ids.size(0), -1)

    mean_nlls: List[float] = []
    ppls: List[float] = []
    min_k_losses: List[float] = []
    valid_tokens: List[int] = []
    for idx in range(input_ids.size(0)):
        mask = shift_mask[idx].bool()
        losses = token_losses[idx][mask]
        valid_count = int(losses.numel())
        valid_tokens.append(valid_count)
        if valid_count == 0:
            mean_nlls.append(float("inf"))
            ppls.append(float("inf"))
            min_k_losses.append(float("inf"))
            continue
        mean_loss = float(losses.mean().item())
        k_num = max(1, int(valid_count * args.mia_k_ratio))
        hard_losses, _ = torch.topk(losses, k_num, largest=True)
        min_k_loss = float(hard_losses.mean().item())
        mean_nlls.append(mean_loss)
        ppls.append(float(math.exp(min(mean_loss, 50.0))))
        min_k_losses.append(min_k_loss)

    predicted_by_min_k = int(np.argmin(np.array(min_k_losses)))
    predicted_by_ppl = int(np.argmin(np.array(mean_nlls)))
    return {
        "mia_mean_nlls": mean_nlls,
        "mia_ppls": ppls,
        "mia_min_k_losses": min_k_losses,
        "mia_valid_tokens": valid_tokens,
        "mia_predicted_real_index_min_k": predicted_by_min_k,
        "mia_predicted_real_index_ppl": predicted_by_ppl,
    }


def build_gpt_attack_prompt(shared_prefix_text: str, candidates: Sequence[Dict[str, Any]]) -> str:
    blocks = []
    for candidate in candidates:
        idx = candidate["candidate_index"]
        blocks.append(
            f"Candidate {idx} full query:\n{candidate['projected_text']}\n\n"
            f"Candidate {idx} suffix after the shared prefix:\n{candidate['projected_suffix_text']}"
        )
    rendered = "\n\n".join(blocks)
    return f"""You are evaluating a privacy attack on a black-box LLM API defense.

The defense sent one group of projected text candidates to a server. Exactly one
candidate is the real user query; the others are dummy suffixes generated from
the same prefix. The texts may contain small token-projection artifacts.

Your task is to identify which candidate is most likely to be the real user
query, and to reconstruct the most likely original real query.

Return valid JSON only with this schema:
{{
  "predicted_real_index": 0,
  "recovered_query": "most likely original full query",
  "confidence": 0.0,
  "brief_reasoning": "short reason"
}}

Shared projected prefix:
{shared_prefix_text}

Candidate texts:
{rendered}
"""


def build_openai_client(args) -> OpenAI:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Environment variable {args.api_key_env} is not set. "
            "Use --dry_run true to only prepare the first prompt."
        )
    kwargs = {
        "api_key": api_key,
        "max_retries": 0,
        "timeout": httpx.Timeout(timeout=args.openai_timeout_seconds, connect=args.openai_connect_timeout_seconds),
    }
    if args.openai_base_url:
        kwargs["base_url"] = args.openai_base_url
    return OpenAI(**kwargs)


def call_gpt_attack(client: OpenAI, prompt: str, args) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(1, args.gpt_max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.gpt_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful privacy and security evaluator. "
                            "Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=args.gpt_temperature,
                max_completion_tokens=args.gpt_max_completion_tokens,
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
            if attempt >= args.gpt_max_retries:
                break
            delay = args.gpt_retry_sleep_seconds * (args.gpt_retry_backoff_multiplier ** (attempt - 1))
            if args.gpt_retry_jitter_seconds > 0:
                delay += random.uniform(0.0, args.gpt_retry_jitter_seconds)
            print(f"[OpenAI Retry] {type(exc).__name__}: {exc}. Sleeping {delay:.1f}s...")
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def compare_recovered_query(
    recovered_query: str,
    real_query: str,
    tokenizer,
    max_length: int,
    prefix_len: int,
) -> Dict[str, float]:
    real_ids = tokenizer(real_query, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]
    pred_ids = tokenizer(recovered_query, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]
    real_len = int(real_ids.numel())
    pred_list = pred_ids.tolist()[:real_len] + [-1] * max(0, real_len - int(pred_ids.numel()))
    real_list = real_ids.tolist()
    full_matches = sum(int(a == b) for a, b in zip(real_list, pred_list[:real_len]))
    suffix_real = real_list[prefix_len:real_len]
    suffix_pred = pred_list[prefix_len:real_len]
    suffix_matches = sum(int(a == b) for a, b in zip(suffix_real, suffix_pred))
    return {
        "recovered_query_exact_match": float(normalize_text(recovered_query) == normalize_text(real_query)),
        "full_token_acc": full_matches / max(real_len, 1),
        "suffix_token_acc": suffix_matches / max(len(suffix_real), 1),
    }


def load_completed_keys(path: Path) -> set[Tuple[str, str, str]]:
    keys: set[Tuple[str, str, str]] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((str(row.get("attack_mode")), str(row.get("subset")), str(row.get("id"))))
    return keys


def write_config(path: Path, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "note": (
            "Black-box DP-SPD attack. Candidate groups include one real query plus k generated dummies. "
            "All candidates are noised in embedding space and projected back to text before attack."
        ),
        "eia_note": (
            "Embedding inversion attack is not run here because this black-box path exposes projected text, "
            "not embeddings. Use the existing EIA benchmark for the white-box embedding threat model."
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(outputs_path: Path, summary_path: Path, args) -> None:
    rows: List[Dict[str, Any]] = []
    if outputs_path.exists():
        with open(outputs_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    fields = [
        "attack_mode",
        "dummy_k",
        "epsilon_non_sensitive",
        "epsilon_private",
        "subset",
        "group",
        "groups",
        "random_baseline",
        "mia_min_k_asr",
        "mia_ppl_asr",
        "gpt_selection_asr",
        "gpt_recovered_exact_match",
        "gpt_full_token_acc",
        "gpt_suffix_token_acc",
        "avg_projection_token_accuracy",
        "total_projection_changed_tokens",
        "projection_changed_samples",
        "avg_real_ppl",
        "avg_dummy_ppl",
        "avg_ppl_gap",
        "total_gpt_prompt_tokens",
        "total_gpt_completion_tokens",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for attack_mode in ["mia", "gpt_attack"]:
            mode_rows = [row for row in rows if row.get("attack_mode") == attack_mode]
            if not mode_rows:
                continue
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
                    random_baseline = 1.0 / max(args.dummy_k + 1, 1)
                    projection_acc = [float(row.get("avg_projection_token_accuracy", 0.0) or 0.0) for row in group_rows]
                    changed_tokens = [int(row.get("total_projection_changed_tokens", 0) or 0) for row in group_rows]
                    changed_samples = [int(row.get("projection_changed_samples", 0) or 0) for row in group_rows]
                    real_ppl = [float(row.get("real_ppl")) for row in group_rows if row.get("real_ppl") is not None]
                    dummy_ppl = [float(row.get("avg_dummy_ppl")) for row in group_rows if row.get("avg_dummy_ppl") is not None]
                    ppl_gap = [float(row.get("ppl_gap")) for row in group_rows if row.get("ppl_gap") is not None]
                    gpt_usage = [row.get("gpt_usage") or {} for row in group_rows]
                    writer.writerow(
                        {
                            "attack_mode": attack_mode,
                            "dummy_k": args.dummy_k,
                            "epsilon_non_sensitive": f"{args.epsilon_non_sensitive:.4f}",
                            "epsilon_private": f"{args.epsilon_private:.4f}",
                            "subset": subset,
                            "group": group,
                            "groups": len(group_rows),
                            "random_baseline": f"{random_baseline:.6f}",
                            "mia_min_k_asr": (
                                f"{sum(float(row.get('mia_min_k_correct', 0.0)) for row in group_rows) / len(group_rows):.6f}"
                                if attack_mode == "mia"
                                else ""
                            ),
                            "mia_ppl_asr": (
                                f"{sum(float(row.get('mia_ppl_correct', 0.0)) for row in group_rows) / len(group_rows):.6f}"
                                if attack_mode == "mia"
                                else ""
                            ),
                            "gpt_selection_asr": (
                                f"{sum(float(row.get('gpt_selection_correct', 0.0)) for row in group_rows) / len(group_rows):.6f}"
                                if attack_mode == "gpt_attack"
                                else ""
                            ),
                            "gpt_recovered_exact_match": (
                                f"{sum(float(row.get('recovered_query_exact_match', 0.0)) for row in group_rows) / len(group_rows):.6f}"
                                if attack_mode == "gpt_attack"
                                else ""
                            ),
                            "gpt_full_token_acc": (
                                f"{sum(float(row.get('full_token_acc', 0.0)) for row in group_rows) / len(group_rows):.6f}"
                                if attack_mode == "gpt_attack"
                                else ""
                            ),
                            "gpt_suffix_token_acc": (
                                f"{sum(float(row.get('suffix_token_acc', 0.0)) for row in group_rows) / len(group_rows):.6f}"
                                if attack_mode == "gpt_attack"
                                else ""
                            ),
                            "avg_projection_token_accuracy": f"{sum(projection_acc) / len(projection_acc):.6f}",
                            "total_projection_changed_tokens": sum(changed_tokens),
                            "projection_changed_samples": sum(changed_samples),
                            "avg_real_ppl": f"{sum(real_ppl) / len(real_ppl):.4f}" if real_ppl else "",
                            "avg_dummy_ppl": f"{sum(dummy_ppl) / len(dummy_ppl):.4f}" if dummy_ppl else "",
                            "avg_ppl_gap": f"{sum(ppl_gap) / len(ppl_gap):.4f}" if ppl_gap else "",
                            "total_gpt_prompt_tokens": sum(int(u.get("prompt_tokens") or 0) for u in gpt_usage),
                            "total_gpt_completion_tokens": sum(int(u.get("completion_tokens") or 0) for u in gpt_usage),
                        }
                    )


def run() -> None:
    args = get_parser().parse_args()
    if args.dummy_k < 1:
        raise ValueError("--dummy_k must be positive for candidate-identification attacks.")
    attack_modes = parse_csv_list(args.attack_modes)
    for mode in attack_modes:
        if mode not in {"mia", "gpt_attack"}:
            raise ValueError(f"Unsupported attack mode: {mode}")
    if not args.mia_attack_model:
        args.mia_attack_model = args.projection_model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / f"{args.output_prefix}_gpt_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = output_dir / f"{args.output_prefix}_groups.jsonl"
    summary_path = output_dir / f"{args.output_prefix}_summary.csv"
    config_path = output_dir / f"{args.output_prefix}_config.json"
    write_config(config_path, args)
    if not args.resume:
        for path in (outputs_path, summary_path):
            if path.exists():
                path.unlink()

    selected_rows = load_selected_rows(args)

    print("Loading client projection/continuation model for DP-SPD candidate generation...")
    projection_tokenizer, projection_model, projection_dtype = load_projection_model(args)
    print(f"[Projection Model] {args.projection_model} | dtype={projection_dtype}")
    if not args.mia_attack_model:
        args.mia_attack_model = args.projection_model

    mia_tokenizer = mia_model = None
    if "mia" in attack_modes and not args.dry_run:
        if str(args.mia_attack_model) == str(args.projection_model):
            mia_tokenizer, mia_model = projection_tokenizer, projection_model
            print("[MIA Model] Reusing projection/continuation model as text scorer.")
        else:
            print("Loading MIA text scoring model...")
            mia_tokenizer, mia_model = load_mia_model(args)
            print(f"[MIA Model] {args.mia_attack_model}")

    gpt_client = None
    if "gpt_attack" in attack_modes and not args.dry_run:
        gpt_client = build_openai_client(args)

    completed = load_completed_keys(outputs_path) if args.resume else set()
    printed = 0

    for subset, rows in selected_rows.items():
        texts = [row_user_text(row, args) for row in rows]
        print(f"\n[Subset] {subset} | groups={len(texts)} | k={args.dummy_k}")
        dataset = build_privacy_dataset(texts, projection_tokenizer, projection_model, args)
        dataloader = DataLoader(dataset, batch_size=args.dummy_k + 1, shuffle=False)

        for batch_index, batch in enumerate(dataloader, start=0):
            group = project_batch_candidates(batch, dataset, projection_tokenizer, args)
            row = rows[int(group["group_index"])]
            row_id = str(row.get("id"))
            real_query = texts[int(group["group_index"])]
            reference = normalize_space(row.get("reference"))
            candidate_texts = [candidate["projected_text"] for candidate in group["candidates"]]
            real_candidate = group["candidates"][group["actual_real_index"]]
            base_record = {
                "subset": subset,
                "id": row_id,
                "group_index": int(group["group_index"]),
                "batch_index": batch_index,
                "privacy_signal": bool(row.get("privacy_signal")),
                "metadata": row.get("metadata", {}),
                "reference": reference,
                "real_query": real_query,
                "actual_real_index": int(group["actual_real_index"]),
                "dummy_k": args.dummy_k,
                "target_epsilon": args.target_epsilon,
                "epsilon_non_sensitive": args.epsilon_non_sensitive,
                "epsilon_private": args.epsilon_private,
                "dummy_ratio": args.dummy_ratio,
                "prefix_len": group["prefix_len"],
                "suffix_len": group["suffix_len"],
                "shared_prefix_text": group["shared_prefix_text"],
                "candidates": group["candidates"],
                "avg_projection_token_accuracy": group["avg_projection_token_accuracy"],
                "total_projection_changed_tokens": group["total_projection_changed_tokens"],
                "projection_changed_samples": group["projection_changed_samples"],
                "real_projection_token_accuracy": real_candidate["projection_token_accuracy"],
                "real_projection_changed_token_count": real_candidate["projection_changed_token_count"],
            }

            if "mia" in attack_modes:
                key = ("mia", subset, row_id)
                if key not in completed:
                    if args.dry_run:
                        print("\n" + "=" * 80)
                        print(f"[Dry Run MIA] subset={subset} id={row_id}")
                        for candidate in group["candidates"]:
                            print(f"Candidate {candidate['candidate_index']} real={candidate['is_real_hidden']}")
                            print(candidate["projected_text"][:400])
                        printed += 1
                    else:
                        assert mia_tokenizer is not None and mia_model is not None
                        mia_scores = score_text_candidates_mia(candidate_texts, mia_tokenizer, mia_model, args)
                        predicted_min_k = int(mia_scores["mia_predicted_real_index_min_k"])
                        predicted_ppl = int(mia_scores["mia_predicted_real_index_ppl"])
                        real_ppl = float(mia_scores["mia_ppls"][group["actual_real_index"]])
                        dummy_ppls = [
                            float(score)
                            for idx, score in enumerate(mia_scores["mia_ppls"])
                            if idx != group["actual_real_index"]
                        ]
                        record = dict(base_record)
                        record.update(mia_scores)
                        record.update(
                            {
                                "attack_mode": "mia",
                                "mia_min_k_correct": float(predicted_min_k == group["actual_real_index"]),
                                "mia_ppl_correct": float(predicted_ppl == group["actual_real_index"]),
                                "real_ppl": real_ppl,
                                "avg_dummy_ppl": sum(dummy_ppls) / max(len(dummy_ppls), 1),
                                "ppl_gap": abs((sum(dummy_ppls) / max(len(dummy_ppls), 1)) - real_ppl),
                            }
                        )
                        append_jsonl(outputs_path, record)
                        completed.add(key)
                        print(
                            f"[MIA] subset={subset} id={row_id} "
                            f"pred_min_k={predicted_min_k} pred_ppl={predicted_ppl} actual={group['actual_real_index']}"
                        )

            if "gpt_attack" in attack_modes:
                key = ("gpt_attack", subset, row_id)
                if key not in completed:
                    prompt = build_gpt_attack_prompt(group["shared_prefix_text"], group["candidates"])
                    if args.save_gpt_prompts:
                        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{subset}_{row_id}")
                        with open(prompts_dir / f"{safe_id}.txt", "w", encoding="utf-8") as f:
                            f.write(prompt)
                    if args.dry_run:
                        if printed < args.dry_run_print_groups:
                            print("\n" + "=" * 80)
                            print(f"[Dry Run GPT Attack] subset={subset} id={row_id}")
                            print(prompt[:5000])
                            printed += 1
                    else:
                        assert gpt_client is not None
                        result = call_gpt_attack(gpt_client, prompt, args)
                        try:
                            predicted = int(result.get("predicted_real_index", -1))
                        except (TypeError, ValueError):
                            predicted = -1
                        recovered = str(result.get("recovered_query", "")).strip()
                        token_metrics = compare_recovered_query(
                            recovered_query=recovered,
                            real_query=real_query,
                            tokenizer=projection_tokenizer,
                            max_length=args.max_length,
                            prefix_len=group["prefix_len"],
                        )
                        record = dict(base_record)
                        record.update(
                            {
                                "attack_mode": "gpt_attack",
                                "gpt_predicted_real_index": predicted,
                                "gpt_selection_correct": float(predicted == group["actual_real_index"]),
                                "recovered_query": recovered,
                                "gpt_confidence": result.get("confidence"),
                                "gpt_brief_reasoning": result.get("brief_reasoning"),
                                "gpt_raw_response": result.get("_raw_response_text"),
                                "gpt_usage": result.get("_usage", {}),
                            }
                        )
                        record.update(token_metrics)
                        append_jsonl(outputs_path, record)
                        completed.add(key)
                        print(
                            f"[GPT Attack] subset={subset} id={row_id} "
                            f"pred={predicted} actual={group['actual_real_index']} "
                            f"ok={record['gpt_selection_correct']}"
                        )
                        if args.inter_request_sleep_seconds > 0:
                            time.sleep(args.inter_request_sleep_seconds)

            if args.dry_run and printed >= args.dry_run_print_groups:
                summarize(outputs_path, summary_path, args)
                print(f"[Dry Run Config] {config_path}")
                return

    summarize(outputs_path, summary_path, args)
    print(f"[Outputs] {outputs_path}")
    print(f"[Summary] {summary_path}")
    print(f"[Config] {config_path}")


if __name__ == "__main__":
    run()
