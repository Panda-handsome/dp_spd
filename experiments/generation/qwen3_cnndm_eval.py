# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpspd.client import BatchConfig, DummyConfig, LDPNoiseConfig, PrivacyDataset


DEFAULT_MODEL_PATH = "models/Qwen3-14B"
DEFAULT_DATASET_PATH = "data/cnn_dailymail"

CALIBRATION_TABLE = {
    1.0: {9: 0.84, 19: 1.24, 29: 1.56, 39: 1.81},
    1.5: {9: 2.62, 19: 3.31, 29: 3.72, 39: 4.01},
    2.0: {9: 4.48, 19: 5.17, 29: 5.57, 39: 5.86},
}


@dataclass
class CnnDmEvalPack:
    eval_texts: List[str]
    references: List[str]


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y", "t"}:
        return True
    if lowered in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_csv_numbers(value: str, cast=float):
    if value is None or str(value).strip() == "":
        return []
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def resolve_dtype(dtype_name: str):
    name = str(dtype_name).replace("torch.", "").lower()
    if name == "auto":
        return torch.bfloat16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_qwen3_model(model_path: str, device: torch.device, dtype_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = resolve_dtype(dtype_name)
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {"": str(device)},
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **kwargs)

    if not getattr(model, "hf_device_map", None):
        model = model.to(device)
    model.eval()
    return tokenizer, model


def choose_split(dataset_raw):
    if hasattr(dataset_raw, "keys"):
        for split in ("test", "validation", "val", "train"):
            if split in dataset_raw:
                return dataset_raw[split], split
    return dataset_raw, "single"


def truncate_text_by_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max(1, max_tokens),
    )["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_prompt(article: str) -> str:
    return (
        "Article:\n"
        f"{article}\n\n"
        "Summarize the article in one concise paragraph.\n"
        "Summary:\n"
    )


def build_cnndm_eval_pack(
    dataset_path: str,
    tokenizer,
    max_samples: int,
    max_prompt_length: int,
    max_ref_tokens: int,
    max_total_length: int,
    shuffle_samples: bool,
    sample_seed: int,
    sample_start: int = 0,
) -> CnnDmEvalPack:
    if sample_start < 0:
        raise ValueError("sample_start must be non-negative")
    if max_samples < 1:
        raise ValueError("max_samples must be positive")

    dataset_raw = load_from_disk(dataset_path)
    dataset, split_name = choose_split(dataset_raw)
    print(f"[Data] Loaded split={split_name}, rows={len(dataset)}")

    if shuffle_samples and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=sample_seed)

    prompt_overhead_ids = tokenizer(
        build_prompt(""),
        add_special_tokens=False,
    )["input_ids"]
    article_budget = max(32, max_prompt_length - len(prompt_overhead_ids) - 8)

    eval_texts: List[str] = []
    references: List[str] = []

    iterator = range(len(dataset))
    for idx in tqdm(iterator, desc="Formatting CNN/DailyMail"):
        item = dataset[idx]
        if "article" not in item or "highlights" not in item:
            raise ValueError(f"Expected CNN/DailyMail columns article/highlights, got {list(item.keys())}")

        article = str(item["article"]).strip()
        reference = str(item["highlights"]).strip()
        if not article or not reference:
            continue

        article = truncate_text_by_tokens(tokenizer, article, article_budget)
        reference = truncate_text_by_tokens(tokenizer, reference, max_ref_tokens)
        prompt = build_prompt(article)
        eval_text = prompt + reference
        full_enc = tokenizer(
            eval_text,
            truncation=True,
            max_length=max_total_length,
            padding="max_length",
            return_tensors="pt",
        )
        full_len = int(full_enc["attention_mask"][0].sum().item())
        if full_len <= 1:
            continue

        eval_texts.append(eval_text)
        references.append(reference)

        if len(eval_texts) >= sample_start + max_samples:
            break

    if len(eval_texts) <= sample_start:
        raise ValueError(
            f"No valid CNN/DailyMail samples were prepared for sample_start={sample_start}."
        )

    eval_texts = eval_texts[sample_start:sample_start + max_samples]
    references = references[sample_start:sample_start + max_samples]

    print(
        f"[Data] Prepared samples={len(eval_texts)}, "
        f"valid_sample_range=[{sample_start}, {sample_start + len(eval_texts)})"
    )
    return CnnDmEvalPack(
        eval_texts=eval_texts,
        references=references,
    )


def collate_privacy_batch(batch):
    keys = [
        "input_ids",
        "attention_mask",
        "noisy_embeddings",
        "group_index",
        "prefix_len",
        "private_token_count",
    ]
    out = {}
    for key in keys:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values)
        else:
            out[key] = torch.tensor(values)
    return out


def mean_embedding_norm(emb_layer, chunk_size: int = 4096) -> float:
    weight = emb_layer.weight.detach()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, weight.size(0), chunk_size):
            chunk = weight[start:start + chunk_size].float()
            total += chunk.norm(dim=1).sum().item()
            count += chunk.size(0)
    return total / max(count, 1)


def decode_generated(generated_ids: torch.Tensor, tokenizer, max_new_tokens: int) -> List[str]:
    if generated_ids.dim() == 1:
        generated_ids = generated_ids.unsqueeze(0)
    if generated_ids.size(1) > max_new_tokens:
        generated_ids = generated_ids[:, -max_new_tokens:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def calculate_ppl(
    model,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    total_nll = 0.0
    total_tokens = 0
    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for batch in tqdm(loader, desc="PPL"):
            labels = batch["input_ids"].to(device)
            noisy_embs = batch["noisy_embeddings"].to(device=device, dtype=model_dtype)
            attn_mask = batch["attention_mask"].to(device)

            outputs = model(
                inputs_embeds=noisy_embs,
                attention_mask=attn_mask,
                use_cache=False,
            )
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = attn_mask[..., 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss = loss.view(shift_labels.size())
            active_loss = loss * shift_mask

            total_nll += active_loss.sum().item()
            total_tokens += shift_mask.sum().item()

    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)


def calculate_rouge(
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    context_len: int = 32,
    gen_len: int = 160,
) -> dict:
    """Compute the standard ROUGE-L mean without hub-backed metric loading.

    Hugging Face Evaluate's ``rouge`` metric is a thin wrapper around
    ``rouge_score.RougeScorer`` followed by an arithmetic mean (its bootstrap
    mid-point).  Calling the local package directly gives the same score while
    avoiding network access and cache locks during parallel GPU sweeps.
    """
    try:
        from rouge_score import rouge_scorer
    except Exception as e:
        raise RuntimeError(
            "ROUGE-L requires the local `rouge_score` package. Install it in the benchmark environment "
            "(e.g. `pip install rouge-score`) before running the evaluation."
        ) from e

    model.eval()
    predictions: List[str] = []
    references: List[str] = []
    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for batch in tqdm(loader, desc="ROUGE"):
            full_noisy = batch["noisy_embeddings"].to(device=device, dtype=model_dtype)
            full_mask = batch["attention_mask"].to(device)
            full_ids = batch["input_ids"].to(device)

            if full_noisy.size(1) < context_len + 1:
                continue

            ctx_noisy = full_noisy[:, :context_len, :]
            ctx_mask = full_mask[:, :context_len]

            generated_ids = model.generate(
                inputs_embeds=ctx_noisy,
                attention_mask=ctx_mask,
                max_new_tokens=gen_len,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            gen_texts = decode_generated(generated_ids.detach().cpu(), tokenizer, gen_len)
            ref_ids = full_ids[:, context_len: context_len + gen_len]
            ref_texts = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

            for pred, ref in zip(gen_texts, ref_texts):
                if len(ref.strip()) > 0:
                    predictions.append(pred)
                    references.append(ref)

    if not predictions:
        return {"rougeL": 0.0}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(scorer.score(reference, prediction)["rougeL"].fmeasure for prediction, reference in zip(predictions, references))
    return {"rougeL": float(rouge_l / len(predictions))}


def build_parser(default_target_eps: float, default_device: str, default_output_csv: str):
    parser = argparse.ArgumentParser(description="DP-SPD no-dummy CNN/DailyMail evaluation for Qwen3-32B.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output_csv", type=str, default=default_output_csv)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--target_eps", type=float, default=default_target_eps)
    parser.add_argument("--ks", type=str, default="9,19,29,39")
    parser.add_argument("--sensitive_ratios", type=str, default="0.6,0.8")
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--max_ref_tokens", type=int, default=160)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--rouge_context_len", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--clip_norm", type=str, default="auto")
    parser.add_argument("--shuffle_samples", type=str2bool, default=False)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--run_rouge", type=str2bool, default=True)
    parser.add_argument("--run_ppl", type=str2bool, default=True)
    parser.add_argument("--append", type=str2bool, default=False)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def validate_paths(args):
    missing = []
    if not os.path.isdir(args.model_path):
        missing.append(f"model_path={args.model_path}")
    if not os.path.isdir(args.dataset_path):
        missing.append(f"dataset_path={args.dataset_path}")
    if missing:
        raise FileNotFoundError("Missing local path(s): " + ", ".join(missing))


def write_header_if_needed(path: Path, append: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write = (not append) or (not path.exists()) or path.stat().st_size == 0
    mode = "a" if append else "w"
    handle = open(path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "model",
            "dataset",
            "target_eps",
            "k",
            "epsilon_pre",
            "epsilon_suf",
            "sensitive_ratio",
            "n_dummy",
            "max_samples",
            "batch_size",
            "max_prompt_length",
            "max_ref_tokens",
            "max_new_tokens",
            "rouge_context_len",
            "clip_norm",
            "rouge_l",
            "ppl",
        ],
    )
    if should_write:
        writer.writeheader()
        handle.flush()
    return handle, writer


def run_for_target(default_target_eps: float, default_device: str, default_output_csv: str):
    parser = build_parser(default_target_eps, default_device, default_output_csv)
    args = parser.parse_args()

    target_eps = round(float(args.target_eps), 2)
    if target_eps not in CALIBRATION_TABLE:
        raise ValueError(f"target_eps={target_eps} not in calibration table: {sorted(CALIBRATION_TABLE)}")

    ks = parse_csv_numbers(args.ks, int)
    sensitive_ratios = parse_csv_numbers(args.sensitive_ratios, float)
    for k in ks:
        if k not in CALIBRATION_TABLE[target_eps]:
            raise ValueError(f"k={k} not available for target_eps={target_eps}")

    print("[Config]")
    print(f"  target_eps / epsilon_pre = {target_eps}")
    print(f"  ks = {ks}")
    print(f"  sensitive_ratios = {sensitive_ratios}")
    print(f"  device = {args.device}")
    print(f"  output_csv = {args.output_csv}")

    if args.dry_run:
        validate_paths(args)
        print("[Dry Run] Paths exist and configuration is valid.")
        return

    validate_paths(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_qwen3_model(args.model_path, device, args.dtype)
    emb_layer = model.get_input_embeddings()

    max_total_length = args.max_prompt_length + args.max_ref_tokens
    pack = build_cnndm_eval_pack(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_prompt_length=args.max_prompt_length,
        max_ref_tokens=args.max_ref_tokens,
        max_total_length=max_total_length,
        shuffle_samples=args.shuffle_samples,
        sample_seed=args.sample_seed,
    )

    if args.clip_norm.strip().lower() == "auto":
        clip_norm = mean_embedding_norm(emb_layer)
    else:
        clip_norm = float(args.clip_norm)
    print(f"[Noise] clip_norm={clip_norm:.6f}")

    output_path = Path(args.output_csv)
    handle, writer = write_header_if_needed(output_path, args.append)

    try:
        for ratio in sensitive_ratios:
            print(f"\n[Ratio] sensitive_ratio={ratio}")
            dummy_cfg = DummyConfig(
                m_private_tokens=0,
                sensitive_ratio=ratio,
                n_dummies_per_text=0,
                dummy_mode="generation",
                dummy_corpus=None,
                pos_mapping_path=None,
            )
            ldp_cfg = LDPNoiseConfig(
                mechanism="Laplace",
                epsilon_private=1.0,
                epsilon_non_sensitive=target_eps,
                clip_norm=clip_norm,
                seed=args.sample_seed,
            )
            batch_cfg = BatchConfig(
                max_length=max_total_length,
                device=torch.device("cpu"),
                hidden_size=model.config.hidden_size,
                preserve_prefix_hash=False,
            )
            privacy_ds = PrivacyDataset(
                pack.eval_texts,
                tokenizer,
                batch_cfg,
                dummy_cfg,
                ldp_cfg,
                seed=args.sample_seed,
                shuffle_mode="global",
                pretrained_embeddings=emb_layer,
                base_model=None,
            )

            for k in ks:
                epsilon_suf = CALIBRATION_TABLE[target_eps][k]
                privacy_ds.ldp_cfg.epsilon_non_sensitive = target_eps
                privacy_ds.ldp_cfg.epsilon_private = epsilon_suf

                loader = DataLoader(
                    privacy_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate_privacy_batch,
                )

                print(f"[Case] eps_pre={target_eps}, k={k}, eps_suf={epsilon_suf}, ratio={ratio}")
                rouge_res = calculate_rouge(
                    model=model,
                    tokenizer=tokenizer,
                    loader=loader,
                    device=device,
                    context_len=args.rouge_context_len,
                    gen_len=args.max_new_tokens,
                ) if args.run_rouge else float("nan")
                rouge_l = rouge_res.get("rougeL", 0.0) if isinstance(rouge_res, dict) else float("nan")

                loader = DataLoader(
                    privacy_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate_privacy_batch,
                )
                ppl = calculate_ppl(
                    model=model,
                    loader=loader,
                    device=device,
                ) if args.run_ppl else float("nan")

                row = {
                    "model": args.model_path,
                    "dataset": args.dataset_path,
                    "target_eps": target_eps,
                    "k": k,
                    "epsilon_pre": target_eps,
                    "epsilon_suf": epsilon_suf,
                    "sensitive_ratio": ratio,
                    "n_dummy": 0,
                    "max_samples": len(pack.eval_texts),
                    "batch_size": args.batch_size,
                    "max_prompt_length": args.max_prompt_length,
                    "max_ref_tokens": args.max_ref_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "rouge_context_len": args.rouge_context_len,
                    "clip_norm": clip_norm,
                    "rouge_l": rouge_l,
                    "ppl": ppl,
                }
                writer.writerow(row)
                handle.flush()
                print(f"[Saved] ROUGE-L={rouge_l:.6f}, PPL={ppl:.6f}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        handle.close()

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    run_for_target(
        default_target_eps=1.5,
        default_device="cuda:0",
        default_output_csv="outputs/qwen3_cnndm.csv",
    )
