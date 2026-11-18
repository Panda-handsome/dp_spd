# -*- coding: utf-8 -*-
"""LLaMA generation utility evaluation for WikiText-2, Alpaca, and Dolly."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpspd.client import BatchConfig, DummyConfig, LDPNoiseConfig, PrivacyDataset
from dpspd.util.utils import str2bool


CALIBRATION_TABLE = {
    1.5: {9: 1.50, 19: 2.08, 29: 2.46, 39: 2.74},
    2.5: {9: 4.48, 19: 5.17, 29: 5.57, 39: 5.86},
    4.0: {9: 8.05, 19: 8.75, 29: 9.15, 39: 9.44},
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DP-SPD generation utility with LLaMA-2-7B-Chat.")
    parser.add_argument("--model_path", type=str, default="models/Llama-2-7b-chat-hf")
    parser.add_argument("--dataset_name", type=str, default="wikitext", choices=["wikitext", "alpaca", "dolly"])
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--target_eps_list", type=str, default="1.5,2.5,4.0")
    parser.add_argument("--ks", type=str, default="9,19,29,39")
    parser.add_argument("--sensitive_ratios", type=str, default="0.2,0.4")
    parser.add_argument("--clip_norm", type=str, default="auto")
    parser.add_argument("--rouge_context_len", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--run_rouge", type=str2bool, default=True)
    parser.add_argument("--run_ppl", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default="outputs/llama_generation.csv")
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def parse_csv(text: str, cast):
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def resolve_dtype(name: str):
    if name == "auto":
        return torch.bfloat16
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def default_dataset_path(name: str) -> str:
    return {
        "wikitext": "data/wikitext/wikitext-2-v1",
        "alpaca": "data/alpaca",
        "dolly": "data/databricks_dolly",
    }[name]


def default_split(name: str) -> str:
    return "test" if name == "wikitext" else "train"


def load_raw_dataset(name: str, path: str, split: str):
    if os.path.isdir(path):
        raw = load_from_disk(path)
        return raw[split] if hasattr(raw, "keys") else raw
    if name == "wikitext":
        return load_dataset("wikitext", "wikitext-2-v1", split=split)
    raise FileNotFoundError(f"Dataset path not found: {path}. Please provide a local {name} dataset.")


def format_texts(name: str, dataset, max_samples: int) -> List[str]:
    texts: List[str] = []
    for item in tqdm(dataset, desc=f"Formatting {name}"):
        if name == "wikitext":
            text = str(item.get("text", "")).strip()
        elif name == "alpaca":
            inst = str(item.get("instruction", "")).strip()
            inp = str(item.get("input", "")).strip()
            out = str(item.get("output", "")).strip()
            if inp:
                text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
        elif name == "dolly":
            inst = str(item.get("instruction", "")).strip()
            ctx = str(item.get("context", "")).strip()
            resp = str(item.get("response", "")).strip()
            if ctx:
                text = f"### Instruction:\n{inst}\n\n### Context:\n{ctx}\n\n### Response:\n{resp}"
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{resp}"
        else:
            raise ValueError(name)
        if len(text.split()) > 8:
            texts.append(text)
        if len(texts) >= max_samples:
            break
    if not texts:
        raise ValueError("No valid texts were found for generation evaluation.")
    return texts


def collate_privacy(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "noisy_embeddings": torch.stack([item["noisy_embeddings"] for item in batch]),
    }


def mean_embedding_norm(emb_layer) -> float:
    return emb_layer.weight.detach().float().norm(dim=1).mean().item()


def build_loader(texts, tokenizer, model, args, epsilon_private: float, epsilon_non_sensitive: float, ratio: float, clip_norm: float):
    batch_cfg = BatchConfig(
        max_length=args.max_length,
        device=torch.device("cpu"),
        hidden_size=model.config.hidden_size,
        preserve_prefix_hash=False,
    )
    dummy_cfg = DummyConfig(
        m_private_tokens=0,
        sensitive_ratio=ratio,
        n_dummies_per_text=0,
        dummy_mode="token",
        pos_mapping_path=None,
    )
    ldp_cfg = LDPNoiseConfig(
        mechanism="Laplace",
        epsilon_private=epsilon_private,
        epsilon_non_sensitive=epsilon_non_sensitive,
        clip_norm=clip_norm,
        seed=args.seed,
    )
    dataset = PrivacyDataset(
        texts,
        tokenizer,
        batch_cfg,
        dummy_cfg,
        ldp_cfg,
        seed=args.seed,
        shuffle_mode="global",
        pretrained_embeddings=model.get_input_embeddings(),
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_privacy)


def calculate_ppl(model, loader, device) -> float:
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    total_nll = 0.0
    total_tokens = 0
    model_dtype = next(model.parameters()).dtype
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="PPL"):
            # The default path is the historical DP-SPD evaluator.  The
            # optional fields make the same calculation usable for discrete
            # text-sanitization baselines whose prompt length can differ from
            # their clean continuation length (e.g. word-level CusText).
            labels = (batch["labels"] if "labels" in batch else batch["input_ids"]).to(device)
            if "model_input_ids" in batch:
                model_input_ids = batch["model_input_ids"].to(device)
                noisy = model.get_input_embeddings()(model_input_ids).to(dtype=model_dtype)
            else:
                noisy = batch["noisy_embeddings"].to(device=device, dtype=model_dtype)
            mask = batch["attention_mask"].to(device)
            loss_mask = batch.get("loss_mask", batch["attention_mask"]).to(device)
            outputs = model(inputs_embeds=noisy, attention_mask=mask, use_cache=False)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = loss_mask[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())
            total_nll += (loss * shift_mask).sum().item()
            total_tokens += shift_mask.sum().item()
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("nan")


def calculate_rouge_from_texts(
    predictions: List[str],
    references: List[str],
    *,
    use_stemmer: bool = True,
) -> float:
    """Use the project-standard ROUGE-L implementation for finished text.

    InferDPT's extraction module produces final text outside the tensor-batch
    generation route used below.  This companion function preserves the same
    Evaluate ROUGE definition and stemmer choice for that architecture.
    """

    try:
        import evaluate
        rouge = evaluate.load("rouge", keep_in_memory=True)
    except Exception as exc:
        print(f"[Warning] ROUGE unavailable: {exc}")
        return 0.0
    pairs = [(prediction, reference) for prediction, reference in zip(predictions, references) if reference.strip()]
    if not pairs:
        return 0.0
    preds, refs = zip(*pairs)
    return float(
        rouge.compute(predictions=list(preds), references=list(refs), use_stemmer=use_stemmer).get("rougeL", 0.0)
    )


def calculate_rouge(
    model,
    tokenizer,
    loader,
    device,
    context_len: int,
    gen_len: int,
    *,
    use_stemmer: bool = True,
) -> float:
    try:
        import evaluate
        # ``keep_in_memory`` prevents parallel GPU sweeps from contending for
        # the Arrow-backed default metric cache.  Metric definition and scores
        # are unchanged (the standard Evaluate ROUGE implementation remains
        # the source of truth).
        rouge = evaluate.load("rouge", keep_in_memory=True)
    except Exception as exc:
        print(f"[Warning] ROUGE unavailable: {exc}")
        return 0.0

    preds: List[str] = []
    refs: List[str] = []
    model_dtype = next(model.parameters()).dtype
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="ROUGE"):
            if "generation_input_ids" in batch:
                prompt_ids = batch["generation_input_ids"].to(device)
                noisy = model.get_input_embeddings()(prompt_ids).to(dtype=model_dtype)
                mask = batch["generation_attention_mask"].to(device)
                reference_ids = batch["reference_ids"].to(device)
            else:
                noisy = batch["noisy_embeddings"].to(device=device, dtype=model_dtype)
                mask = batch["attention_mask"].to(device)
                input_ids = batch["input_ids"].to(device)
                if noisy.size(1) <= context_len:
                    continue
                noisy = noisy[:, :context_len, :]
                mask = mask[:, :context_len]
                reference_ids = input_ids[:, context_len:context_len + gen_len]
            generated = model.generate(
                inputs_embeds=noisy,
                attention_mask=mask,
                max_new_tokens=gen_len,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            pred_texts = tokenizer.batch_decode(generated[:, -gen_len:].detach().cpu(), skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(reference_ids.detach().cpu(), skip_special_tokens=True)
            for pred, ref in zip(pred_texts, ref_texts):
                if ref.strip():
                    preds.append(pred)
                    refs.append(ref)
    if not preds:
        return 0.0
    return float(rouge.compute(predictions=preds, references=refs, use_stemmer=use_stemmer).get("rougeL", 0.0))


def main() -> None:
    args = get_parser().parse_args()
    args.dataset_path = args.dataset_path or default_dataset_path(args.dataset_name)
    args.dataset_split = args.dataset_split or default_split(args.dataset_name)
    target_eps_list = parse_csv(args.target_eps_list, float)
    ks = parse_csv(args.ks, int)
    ratios = parse_csv(args.sensitive_ratios, float)

    if args.dry_run:
        print("[Dry Run]")
        print(vars(args))
        print(f"target_eps_list={target_eps_list}, ks={ks}, ratios={ratios}")
        return

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
        torch_dtype=resolve_dtype(args.dtype),
    ).to(device)
    model.eval()

    raw = load_raw_dataset(args.dataset_name, args.dataset_path, args.dataset_split)
    texts = format_texts(args.dataset_name, raw, args.max_samples)
    clip_norm = mean_embedding_norm(model.get_input_embeddings()) if str(args.clip_norm).lower() == "auto" else float(args.clip_norm)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "model", "target_eps", "k", "epsilon_private", "epsilon_non_sensitive",
        "sensitive_ratio", "clip_norm", "max_samples", "ppl", "rouge_l",
    ]
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for ratio in ratios:
            for target_eps in target_eps_list:
                for k in ks:
                    local_eps = CALIBRATION_TABLE[target_eps][k]
                    loader = build_loader(texts, tokenizer, model, args, local_eps, target_eps, ratio, clip_norm)
                    ppl = calculate_ppl(model, loader, device) if args.run_ppl else float("nan")
                    loader = build_loader(texts, tokenizer, model, args, local_eps, target_eps, ratio, clip_norm)
                    rouge_l = calculate_rouge(model, tokenizer, loader, device, args.rouge_context_len, args.max_new_tokens) if args.run_rouge else float("nan")
                    row = {
                        "dataset": args.dataset_name,
                        "model": args.model_path,
                        "target_eps": target_eps,
                        "k": k,
                        "epsilon_private": local_eps,
                        "epsilon_non_sensitive": target_eps,
                        "sensitive_ratio": ratio,
                        "clip_norm": clip_norm,
                        "max_samples": len(texts),
                        "ppl": ppl,
                        "rouge_l": rouge_l,
                    }
                    writer.writerow(row)
                    f.flush()
                    print(f"[Saved] {row}")


if __name__ == "__main__":
    main()
