# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import openai
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from openai import OpenAI
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dpspd.client import BatchConfig, DummyConfig, LDPNoiseConfig, PrivacyDataset
from dpspd.util.utils import load_local_causal_lm, str2bool


LOCAL_EPS_MAPPING = {
    1.5: {9: 1.50, 19: 2.08, 29: 2.46, 39: 2.74},
    2.0: {9: 3.03, 19: 3.71, 29: 4.12, 39: 4.40},
    2.5: {9: 4.48, 19: 5.17, 29: 5.57, 39: 5.86},
    3.0: {9: 5.77, 19: 6.46, 29: 6.86, 39: 7.15},
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="GPT inference attack adapted to the DP-SPD grouped (real + dummy) setting."
    )

    parser.add_argument("--continuation_model", type=str, default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--continuation_dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--dataset_path", type=str, default="data/imdb")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--min_len", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--num_groups", type=int, default=10,
                        help="number of (real + dummy) groups to evaluate")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--eval_manifest_path", type=str, default=None,
                        help="path to a fixed evaluation-set manifest; created automatically if missing")

    parser.add_argument("--dummy_ratio", type=float, default=0.2,
                        help="private suffix ratio r in DP-SPD")
    parser.add_argument("--n_dummy", type=int, default=9,
                        help="number of dummy suffixes in each group")
    parser.add_argument("--target_eps", type=float, default=1.5,
                        help="privacy budget for the shared prefix")
    parser.add_argument("--local_eps", type=float, default=None,
                        help="privacy budget for the private suffix; if omitted, infer from the built-in table")
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--gpt_temperature", type=float, default=0.0)
    parser.add_argument("--gpt_max_completion_tokens", type=int, default=1200)
    parser.add_argument("--gpt_max_retries", type=int, default=5)
    parser.add_argument("--gpt_retry_sleep_seconds", type=float, default=5.0)
    parser.add_argument("--gpt_retry_backoff_multiplier", type=float, default=2.0)
    parser.add_argument("--gpt_retry_jitter_seconds", type=float, default=1.0)
    parser.add_argument("--openai_timeout_seconds", type=float, default=120.0)
    parser.add_argument("--openai_connect_timeout_seconds", type=float, default=20.0)
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")

    parser.add_argument("--chunk_size", type=int, default=8,
                        help="how many token positions to invert at once during nearest-neighbor textualization")
    parser.add_argument("--inter_request_sleep_seconds", type=float, default=0.5)
    parser.add_argument("--resume", type=str2bool, default=True,
                        help="resume from existing per-group outputs if available")
    parser.add_argument("--dry_run", type=str2bool, default=False,
                        help="only build and print the first GPT prompt without sending API requests")
    parser.add_argument("--save_prompts", type=str2bool, default=True)
    parser.add_argument("--output_dir", type=str, default="outputs/gpt_inference_attack")

    return parser


def filter_and_sample(texts: List[str], min_len=15, max_len=60, sample_size=10):
    filtered = [t for t in texts if min_len <= len(t.split()) <= max_len]
    return random.sample(filtered, min(sample_size, len(filtered)))


def resolve_dataset_split(raw_ds, split_name: str):
    if hasattr(raw_ds, "keys"):
        available = list(raw_ds.keys())
        if split_name not in raw_ds:
            raise ValueError(
                f"Dataset split '{split_name}' was not found in {available}. "
                "Please pass --dataset_split with one of the available splits."
            )
        return raw_ds[split_name]
    return raw_ds


def infer_local_eps(target_eps: float, n_dummy: int) -> float:
    rounded_target = round(target_eps, 2)
    if rounded_target not in LOCAL_EPS_MAPPING or n_dummy not in LOCAL_EPS_MAPPING[rounded_target]:
        raise ValueError(
            f"No built-in local-eps mapping for target_eps={target_eps}, n_dummy={n_dummy}. "
            "Please pass --local_eps explicitly."
        )
    return LOCAL_EPS_MAPPING[rounded_target][n_dummy]


def get_default_eval_manifest_path(args) -> Path:
    dataset_name = Path(args.dataset_path.rstrip("/")).name or "dataset"
    filename = (
        f"{dataset_name}_{args.dataset_split}_seed{args.seed}"
        f"_n{args.num_groups}_len{args.min_len}_{args.max_len}.json"
    )
    return Path(ROOT) / "fixed_eval_sets" / filename


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def build_eval_manifest(
    dataset_texts: List[str],
    dataset_path: str,
    dataset_split: str,
    min_len: int,
    max_len: int,
    sample_size: int,
    seed: int,
) -> Dict[str, Any]:
    eligible_samples = []
    for idx, text in enumerate(dataset_texts):
        word_count = len(text.split())
        if min_len <= word_count <= max_len:
            eligible_samples.append(
                {
                    "dataset_index": idx,
                    "word_count": word_count,
                    "text": text,
                }
            )

    if len(eligible_samples) < sample_size:
        raise ValueError(
            f"Only {len(eligible_samples)} eligible texts were found, "
            f"but num_groups={sample_size} was requested."
        )

    rng = random.Random(seed)
    selected_samples = rng.sample(eligible_samples, sample_size)

    return {
        "dataset_path": dataset_path,
        "dataset_split": dataset_split,
        "seed": seed,
        "num_groups": sample_size,
        "min_len": min_len,
        "max_len": max_len,
        "eligible_pool_size": len(eligible_samples),
        "samples": selected_samples,
    }


def load_or_create_eval_set(args) -> Tuple[List[str], Dict[str, Any], Path]:
    raw_ds = load_from_disk(args.dataset_path)
    split_ds = resolve_dataset_split(raw_ds, args.dataset_split)
    dataset_texts = split_ds["text"]

    manifest_path = Path(args.eval_manifest_path) if args.eval_manifest_path else get_default_eval_manifest_path(args)
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = build_eval_manifest(
            dataset_texts=dataset_texts,
            dataset_path=args.dataset_path,
            dataset_split=args.dataset_split,
            min_len=args.min_len,
            max_len=args.max_len,
            sample_size=args.num_groups,
            seed=args.seed,
        )
        atomic_write_json(manifest_path, manifest)

    if manifest.get("num_groups") != args.num_groups:
        raise ValueError(
            f"Evaluation manifest at {manifest_path} contains num_groups={manifest.get('num_groups')}, "
            f"but the current run requested num_groups={args.num_groups}. "
            "Use a matching manifest path or align --num_groups."
        )
    if manifest.get("dataset_split") != args.dataset_split:
        raise ValueError(
            f"Evaluation manifest at {manifest_path} was built from split='{manifest.get('dataset_split')}', "
            f"but the current run requested split='{args.dataset_split}'."
        )

    real_texts = [item["text"] for item in manifest["samples"]]
    return real_texts, manifest, manifest_path


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
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


def parse_version(version_text: str) -> Tuple[int, ...]:
    parts = []
    for chunk in re.split(r"[.+-]", version_text):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            break
    return tuple(parts) if parts else (0,)


def build_openai_client(args) -> OpenAI:
    openai_ver = parse_version(openai.__version__)
    httpx_ver = parse_version(httpx.__version__)
    if openai_ver < (1, 14, 0) and httpx_ver >= (0, 28, 0):
        raise RuntimeError(
            "Incompatible OpenAI stack detected: "
            f"openai=={openai.__version__} with httpx=={httpx.__version__}. "
            "This combination triggers `Client.__init__() got an unexpected keyword argument 'proxies'`. "
            "Recommended fix: upgrade OpenAI, e.g. `pip install -U openai` "
            "or install a tested version such as `pip install openai==2.30.0`. "
            "Alternative fix: downgrade httpx with `pip install \"httpx<0.28\"`."
        )

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Environment variable {args.api_key_env} is not set. "
            "Set it before running the GPT inference attack, or use --dry_run true."
        )

    client_kwargs = {
        "api_key": api_key,
        "max_retries": 0,
        "timeout": httpx.Timeout(
            timeout=args.openai_timeout_seconds,
            connect=args.openai_connect_timeout_seconds,
        ),
    }
    if args.openai_base_url:
        client_kwargs["base_url"] = args.openai_base_url

    try:
        return OpenAI(**client_kwargs)
    except TypeError as e:
        if "unexpected keyword argument 'proxies'" in str(e):
            raise RuntimeError(
                "Failed to initialize OpenAI client because the installed "
                f"openai=={openai.__version__} is incompatible with httpx=={httpx.__version__}. "
                "Upgrade OpenAI or downgrade httpx."
            ) from e
        raise


def invert_token_chunk(chunk_embs: torch.Tensor, vocab_norm: torch.Tensor) -> torch.Tensor:
    query = F.normalize(chunk_embs.to(dtype=vocab_norm.dtype), p=2, dim=-1)
    sims = torch.matmul(query, vocab_norm.transpose(0, 1))
    return sims.argmax(dim=-1)


def invert_sequence_to_ids(
    noisy_seq_embs: torch.Tensor,
    attn_mask: torch.Tensor,
    vocab_norm: torch.Tensor,
    chunk_size: int,
) -> List[int]:
    valid_len = int(attn_mask.sum().item())
    pred_ids: List[int] = []
    for start in range(0, valid_len, chunk_size):
        end = min(start + chunk_size, valid_len)
        chunk_ids = invert_token_chunk(noisy_seq_embs[start:end], vocab_norm)
        pred_ids.extend(chunk_ids.detach().cpu().tolist())
    return pred_ids


def build_group_attack_prompt(
    shared_prefix_text: str,
    candidate_suffix_texts: List[str],
) -> str:
    candidate_block = []
    for idx, suffix in enumerate(candidate_suffix_texts):
        candidate_block.append(f"Candidate {idx} suffix:\n{suffix}")
    rendered_candidates = "\n\n".join(candidate_block)

    return f"""You are performing a GPT inference attack adapted from InferDPT.

Context:
- A privacy-preserving inference system produced one group of candidate queries.
- Exactly one candidate is the real user query.
- The other candidates are dummy continuations generated from the same shared prefix.
- The text you see below was reconstructed from noised embeddings, so it may contain token errors.

Your tasks:
1. Decide which candidate suffix is most likely to come from the real query.
2. Recover the most likely original full query before noise.
3. Briefly explain which semantic or stylistic cues made that candidate more plausible than the others.

Return valid JSON only in the following schema:
{{
  "predicted_real_index": 0,
  "recovered_query": "full recovered query here",
  "confidence": 0.0,
  "brief_reasoning": "short explanation"
}}

Shared prefix:
{shared_prefix_text}

Candidate suffixes:
{rendered_candidates}
"""


def call_gpt_attack(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    max_completion_tokens: int,
    max_retries: int,
    retry_sleep_seconds: float,
    retry_backoff_multiplier: float,
    retry_jitter_seconds: float,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful security research assistant. "
                            "Follow the user instructions exactly and respond with valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = parse_json_response(content)
            parsed["_raw_response_text"] = content
            return parsed
        except Exception as e:
            last_error = e
            if attempt >= max_retries:
                break
            delay = retry_sleep_seconds * (retry_backoff_multiplier ** (attempt - 1))
            if retry_jitter_seconds > 0:
                delay += random.uniform(0.0, retry_jitter_seconds)
            print(
                f"[OpenAI Retry] attempt {attempt}/{max_retries} failed: "
                f"{type(e).__name__}: {e}. Sleeping {delay:.1f}s before retrying..."
            )
            time.sleep(delay)

    assert last_error is not None
    raise last_error


def load_resume_state(summary_path: Path) -> Tuple[Set[int], Dict[str, float]]:
    completed_group_ids: Set[int] = set()
    aggregates = {
        "selection_hits": 0.0,
        "exact_match_hits": 0.0,
        "full_token_acc_sum": 0.0,
        "private_token_acc_sum": 0.0,
        "num_groups": 0.0,
    }

    if not summary_path.exists():
        return completed_group_ids, aggregates

    with open(summary_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                group_id = int(row["group_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if group_id in completed_group_ids:
                continue

            completed_group_ids.add(group_id)
            aggregates["selection_hits"] += float(row["selection_correct"])
            aggregates["exact_match_hits"] += float(row["recovered_query_exact_match"])
            aggregates["full_token_acc_sum"] += float(row["full_token_acc"])
            aggregates["private_token_acc_sum"] += float(row["private_token_acc"])
            aggregates["num_groups"] += 1.0

    return completed_group_ids, aggregates


def compare_recovered_query(
    recovered_query: str,
    real_input_ids: torch.Tensor,
    prefix_len: int,
    real_len: int,
    tokenizer,
    max_length: int,
) -> Dict[str, float]:
    recovered_ids = tokenizer(
        recovered_query,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"][0].tolist()

    padded = recovered_ids[:real_len] + [-1] * max(0, real_len - len(recovered_ids))
    real_ids = real_input_ids[:real_len].detach().cpu().tolist()

    full_matches = sum(int(a == b) for a, b in zip(real_ids, padded[:real_len]))
    private_real = real_ids[prefix_len:real_len]
    private_pred = padded[prefix_len:real_len]
    private_matches = sum(int(a == b) for a, b in zip(private_real, private_pred))

    return {
        "full_token_acc": full_matches / max(real_len, 1),
        "private_token_acc": private_matches / max(len(private_real), 1),
    }


def main():
    args = get_parser().parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.local_eps is None:
        args.local_eps = infer_local_eps(args.target_eps, args.n_dummy)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    experiment_config_path = output_dir / "experiment_config.json"
    metrics_path = output_dir / "gpt_inference_attack_metrics.json"

    client = None
    if not args.dry_run:
        client = build_openai_client(args)

    print("Loading continuation model and tokenizer for DP-SPD group construction...")
    tokenizer, model, load_dtype = load_local_causal_lm(
        model_path=args.continuation_model,
        device=args.device,
        dtype_preference=args.continuation_dtype,
        local_files_only=args.local_files_only,
    )
    print(f"[Model] {os.path.basename(args.continuation_model.rstrip('/'))} | dtype={load_dtype}")

    print("Preparing vocabulary matrix for nearest-neighbor textualization...")
    vocab_norm = F.normalize(model.get_input_embeddings().weight.detach(), p=2, dim=-1)

    print("Loading evaluation texts...")
    real_texts, eval_manifest, manifest_path = load_or_create_eval_set(args)
    print(
        f"[Data Ready] Groups to evaluate: {len(real_texts)} | "
        f"split={args.dataset_split} | manifest={manifest_path}"
    )
    atomic_write_json(
        experiment_config_path,
        {
            "args": vars(args),
            "resolved_eval_manifest_path": str(manifest_path),
            "eval_manifest_metadata": {
                "dataset_path": eval_manifest["dataset_path"],
                "dataset_split": eval_manifest["dataset_split"],
                "seed": eval_manifest["seed"],
                "num_groups": eval_manifest["num_groups"],
                "min_len": eval_manifest["min_len"],
                "max_len": eval_manifest["max_len"],
                "eligible_pool_size": eval_manifest["eligible_pool_size"],
            },
        },
    )

    batch_cfg = BatchConfig(
        max_length=args.max_length,
        device=torch.device("cpu"),
        hidden_size=model.config.hidden_size,
    )
    dummy_cfg = DummyConfig(
        m_private_tokens=0,
        sensitive_ratio=args.dummy_ratio,
        n_dummies_per_text=args.n_dummy,
        dummy_mode="generation",
        dummy_corpus=None,
        pos_mapping_path=None,
    )
    ldp_cfg = LDPNoiseConfig(
        mechanism="Laplace",
        epsilon_private=args.local_eps,
        epsilon_non_sensitive=args.target_eps,
        clip_norm=args.clip_norm,
        seed=args.seed,
    )

    dataset = PrivacyDataset(
        real_texts,
        tokenizer,
        batch_cfg,
        dummy_cfg,
        ldp_cfg,
        shuffle_mode="block",
        pretrained_embeddings=model.get_input_embeddings(),
        base_model=model,
    )
    dataloader = DataLoader(dataset, batch_size=args.n_dummy + 1, shuffle=False)

    per_group_path = output_dir / "gpt_inference_attack_groups.jsonl"
    summary_path = output_dir / "gpt_inference_attack_summary.csv"
    summary_fields = [
        "group_id",
        "actual_real_index",
        "predicted_real_index",
        "selection_correct",
        "confidence",
        "recovered_query_exact_match",
        "full_token_acc",
        "private_token_acc",
        "real_query",
        "recovered_query",
    ]

    completed_group_ids: Set[int] = set()
    selection_hits = 0.0
    exact_match_hits = 0.0
    full_token_acc_sum = 0.0
    private_token_acc_sum = 0.0
    num_groups = 0.0
    if args.resume and not args.dry_run:
        completed_group_ids, resume_aggregates = load_resume_state(summary_path)
        selection_hits = resume_aggregates["selection_hits"]
        exact_match_hits = resume_aggregates["exact_match_hits"]
        full_token_acc_sum = resume_aggregates["full_token_acc_sum"]
        private_token_acc_sum = resume_aggregates["private_token_acc_sum"]
        num_groups = resume_aggregates["num_groups"]
        if completed_group_ids:
            print(
                f"[Resume] Found {int(num_groups)} completed groups in {summary_path}. "
                "Will continue from the next unfinished group."
            )

    jsonl_mode = "a" if args.resume and per_group_path.exists() and not args.dry_run else "w"
    summary_mode = "a" if args.resume and summary_path.exists() and not args.dry_run else "w"

    with open(per_group_path, jsonl_mode, encoding="utf-8") as jsonl_f, \
         open(summary_path, summary_mode, newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=summary_fields)
        if summary_mode == "w":
            writer.writeheader()

        for group_id, batch in enumerate(tqdm(dataloader, desc="Running GPT inference attack"), start=1):
            if group_id in completed_group_ids and not args.dry_run:
                continue

            input_ids = batch["input_ids"].to(args.device)
            noisy_embs = batch["noisy_embeddings"].to(args.device)
            attn_mask = batch["attention_mask"].to(args.device)
            is_real_flags = batch["is_real"]

            if not is_real_flags.any():
                continue

            actual_real_index = int((is_real_flags == True).nonzero(as_tuple=True)[0].item())
            real_len = int(attn_mask[actual_real_index].sum().item())
            if "prefix_len" in batch:
                prefix_len = int(batch["prefix_len"][actual_real_index].item())
            elif real_len <= 1:
                prefix_len = real_len
            else:
                prefix_len = max(1, min(real_len - 1, int((1.0 - args.dummy_ratio) * real_len)))

            candidate_id_sequences: List[List[int]] = []
            for candidate_idx in range(noisy_embs.size(0)):
                candidate_id_sequences.append(
                    invert_sequence_to_ids(
                        noisy_seq_embs=noisy_embs[candidate_idx],
                        attn_mask=attn_mask[candidate_idx],
                        vocab_norm=vocab_norm,
                        chunk_size=args.chunk_size,
                    )
                )

            shared_prefix_ids = candidate_id_sequences[0][:prefix_len]
            shared_prefix_text = tokenizer.decode(shared_prefix_ids, skip_special_tokens=True)
            candidate_suffix_texts = [
                tokenizer.decode(ids[prefix_len:real_len], skip_special_tokens=True)
                for ids in candidate_id_sequences
            ]
            reconstructed_queries = [
                tokenizer.decode(ids[:real_len], skip_special_tokens=True)
                for ids in candidate_id_sequences
            ]
            prompt = build_group_attack_prompt(shared_prefix_text, candidate_suffix_texts)

            if args.save_prompts:
                with open(prompts_dir / f"group_{group_id:04d}.txt", "w", encoding="utf-8") as f:
                    f.write(prompt)

            if args.dry_run:
                print("\n" + "=" * 80)
                print(f"[Dry Run] Group {group_id}")
                print(prompt)
                print("=" * 80 + "\n")
                break

            result = call_gpt_attack(
                client=client,
                model_name=args.gpt_model,
                prompt=prompt,
                temperature=args.gpt_temperature,
                max_completion_tokens=args.gpt_max_completion_tokens,
                max_retries=args.gpt_max_retries,
                retry_sleep_seconds=args.gpt_retry_sleep_seconds,
                retry_backoff_multiplier=args.gpt_retry_backoff_multiplier,
                retry_jitter_seconds=args.gpt_retry_jitter_seconds,
            )

            predicted_real_index = int(result.get("predicted_real_index", -1))
            recovered_query = str(result.get("recovered_query", "")).strip()
            confidence = float(result.get("confidence", 0.0))

            real_query = tokenizer.decode(
                input_ids[actual_real_index, :real_len].detach().cpu().tolist(),
                skip_special_tokens=True,
            )
            token_metrics = compare_recovered_query(
                recovered_query=recovered_query,
                real_input_ids=input_ids[actual_real_index].detach().cpu(),
                prefix_len=prefix_len,
                real_len=real_len,
                tokenizer=tokenizer,
                max_length=args.max_length,
            )

            selection_correct = int(predicted_real_index == actual_real_index)
            recovered_query_exact_match = int(normalize_text(recovered_query) == normalize_text(real_query))

            selection_hits += selection_correct
            exact_match_hits += recovered_query_exact_match
            full_token_acc_sum += token_metrics["full_token_acc"]
            private_token_acc_sum += token_metrics["private_token_acc"]
            num_groups += 1

            record = {
                "group_id": group_id,
                "actual_real_index": actual_real_index,
                "predicted_real_index": predicted_real_index,
                "selection_correct": selection_correct,
                "confidence": confidence,
                "recovered_query_exact_match": recovered_query_exact_match,
                "full_token_acc": f"{token_metrics['full_token_acc']:.4f}",
                "private_token_acc": f"{token_metrics['private_token_acc']:.4f}",
                "real_query": real_query,
                "recovered_query": recovered_query,
                "shared_prefix_text": shared_prefix_text,
                "candidate_suffix_texts": candidate_suffix_texts,
                "candidate_reconstructed_queries": reconstructed_queries,
                "gpt_response": result,
            }
            jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.writerow({k: record[k] for k in summary_fields})
            jsonl_f.flush()
            csv_f.flush()

            print(
                f"[Group {group_id}] "
                f"pred={predicted_real_index}, actual={actual_real_index}, "
                f"select_ok={selection_correct}, private_token_acc={token_metrics['private_token_acc']:.4f}"
            )

            if args.inter_request_sleep_seconds > 0:
                time.sleep(args.inter_request_sleep_seconds)

    if args.dry_run:
        return

    if num_groups == 0:
        print("No valid groups were evaluated.")
        return

    print("\n" + "=" * 80)
    print("GPT Inference Attack Summary")
    print(f"Selection Accuracy: {selection_hits / num_groups * 100:.2f}%")
    print(f"Recovered Query Exact Match: {exact_match_hits / num_groups * 100:.2f}%")
    print(f"Average Full-Query Token Accuracy: {full_token_acc_sum / num_groups * 100:.2f}%")
    print(f"Average Private-Suffix Token Accuracy: {private_token_acc_sum / num_groups * 100:.2f}%")
    print(f"Per-group JSONL: {per_group_path}")
    print(f"Summary CSV: {summary_path}")
    print("=" * 80)

    atomic_write_json(
        metrics_path,
        {
            "num_groups": num_groups,
            "selection_accuracy": selection_hits / num_groups,
            "recovered_query_exact_match": exact_match_hits / num_groups,
            "average_full_token_accuracy": full_token_acc_sum / num_groups,
            "average_private_token_accuracy": private_token_acc_sum / num_groups,
            "target_eps": args.target_eps,
            "local_eps": args.local_eps,
            "dummy_ratio": args.dummy_ratio,
            "n_dummy": args.n_dummy,
            "gpt_model": args.gpt_model,
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "eval_manifest_path": str(manifest_path),
            "per_group_jsonl": str(per_group_path),
            "summary_csv": str(summary_path),
            "experiment_config": str(experiment_config_path),
        },
    )


if __name__ == "__main__":
    main()
