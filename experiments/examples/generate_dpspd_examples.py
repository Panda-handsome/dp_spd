# -*- coding: utf-8 -*-
"""Generate qualitative natural-language examples for the DP-SPD pipeline.

The script is intentionally small and descriptive: it reuses PrivacyDataset so
the generated prefix, dummy suffixes, noised candidates, shuffle order, and
de-shuffle index match the implementation used by the attack/speed experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpspd.client import BatchConfig, DummyConfig, LDPNoiseConfig, PrivacyDataset
from dpspd.util.utils import load_local_causal_lm, str2bool


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write concrete DP-SPD real/dummy/shuffle/de-shuffle examples."
    )
    parser.add_argument(
        "--continuation_model",
        type=str,
        default=os.environ.get("DPSPD_CONTINUATION_MODEL", "models/Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument(
        "--continuation_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--dataset_path", type=str, default="data/imdb")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--manual_text", action="append", default=None)
    parser.add_argument("--min_len", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--max_examples", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=96)

    parser.add_argument("--dummy_ratio", type=float, default=0.4)
    parser.add_argument("--n_dummy", type=int, default=4)
    parser.add_argument("--epsilon_private", type=float, default=2.08)
    parser.add_argument("--epsilon_non_sensitive", type=float, default=1.5)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--include_projected_api_text",
        type=str2bool,
        default=True,
        help="Project each noised candidate back to tokens to illustrate the api-only path.",
    )
    parser.add_argument("--projection_chunk_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/examples")
    parser.add_argument("--output_prefix", type=str, default="dpspd_qualitative_examples")
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def resolve_split(raw_ds: Any, split_name: str):
    if hasattr(raw_ds, "keys"):
        if split_name not in raw_ds:
            raise ValueError(f"Split {split_name!r} not found. Available splits: {list(raw_ds.keys())}")
        return raw_ds[split_name]
    return raw_ds


def select_texts(args) -> List[str]:
    if args.manual_text:
        return [text.strip() for text in args.manual_text if text.strip()][: args.max_examples]

    raw_ds = load_from_disk(args.dataset_path)
    ds = resolve_split(raw_ds, args.dataset_split)
    if args.text_field not in ds.column_names:
        raise ValueError(f"Column {args.text_field!r} not found. Columns: {ds.column_names}")

    eligible = []
    for text in ds[args.text_field]:
        text = str(text).strip()
        n_words = len(text.split())
        if args.min_len <= n_words <= args.max_len:
            eligible.append(text)
    if len(eligible) < args.max_examples:
        raise ValueError(f"Only {len(eligible)} eligible texts; requested {args.max_examples}.")
    return random.Random(args.seed).sample(eligible, args.max_examples)


def decode_ids(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(token_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()


def clean_for_markdown(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def validate_local_model_path(args) -> None:
    if not args.local_files_only:
        return
    model_path = Path(args.continuation_model)
    looks_like_path = (
        args.continuation_model.startswith(".")
        or args.continuation_model.startswith("/")
        or os.sep in args.continuation_model
    )
    if looks_like_path and not model_path.exists():
        raise FileNotFoundError(
            f"continuation_model={args.continuation_model!r} does not exist. "
            "Pass --continuation_model /path/to/model or set CONTINUATION_MODEL in the wrapper."
        )


def build_dataset(texts: Sequence[str], tokenizer, model, args) -> PrivacyDataset:
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
        epsilon_private=args.epsilon_private,
        epsilon_non_sensitive=args.epsilon_non_sensitive,
        clip_norm=args.clip_norm,
        seed=args.seed,
    )
    return PrivacyDataset(
        texts,
        tokenizer,
        batch_cfg,
        dummy_cfg,
        ldp_cfg,
        seed=args.seed,
        shuffle_mode="block",
        pretrained_embeddings=model.get_input_embeddings(),
        base_model=model,
    )


def collect_shuffled_positions(dataset: PrivacyDataset) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = {}
    for pos, (group_index, _, _) in enumerate(dataset.mapping):
        grouped.setdefault(int(group_index), []).append(pos)
    return grouped


def project_candidate_text(dataset: PrivacyDataset, item: Dict[str, Any], tokenizer, args) -> str:
    real_len = int(item["attention_mask"].sum().item())
    projected_ids = dataset._project_embeddings_to_tokens(
        item["noisy_embeddings"][:real_len],
        original_ids=item["input_ids"][:real_len],
        chunk_size=args.projection_chunk_size,
    )
    return decode_ids(tokenizer, projected_ids)


def build_examples(dataset: PrivacyDataset, texts: Sequence[str], tokenizer, args) -> List[Dict[str, Any]]:
    grouped_positions = collect_shuffled_positions(dataset)
    examples: List[Dict[str, Any]] = []

    for group_index in range(len(texts)):
        cache = dataset.generation_cache[group_index]
        real_len = int(cache["real_len"])
        prefix_len = int(cache["prefix_len"])
        private_count = int(cache["n_private"])
        original_ids = cache["input_ids"][:real_len]
        projected_prefix_ids = cache["projected_prefix_ids"][:prefix_len]

        clean_prefix = decode_ids(tokenizer, original_ids[:prefix_len])
        privatized_prefix = decode_ids(tokenizer, projected_prefix_ids)
        real_suffix = decode_ids(tokenizer, original_ids[prefix_len:real_len])
        real_query = decode_ids(tokenizer, original_ids)

        dummy_suffixes = []
        for dummy_idx, dummy_ids in enumerate(cache["dummy_input_ids"]):
            suffix_ids = dummy_ids[prefix_len:real_len]
            dummy_suffixes.append(
                {
                    "dummy_index": dummy_idx,
                    "suffix_text": decode_ids(tokenizer, suffix_ids),
                    "full_candidate_text": decode_ids(tokenizer, dummy_ids[:real_len]),
                }
            )

        shuffled_candidates = []
        real_shuffled_index = None
        for shuffled_index, dataset_pos in enumerate(grouped_positions[group_index]):
            item = dataset[dataset_pos]
            label = "real" if bool(item["is_real"]) else f"dummy_{int(item['dummy_index'])}"
            candidate_text = decode_ids(tokenizer, item["input_ids"][:real_len])
            candidate = {
                "shuffled_index": shuffled_index,
                "label_hidden_from_server": label,
                "clean_candidate_text_for_explanation": candidate_text,
                "server_response_placeholder": f"response_for_candidate_{shuffled_index}",
            }
            if args.include_projected_api_text:
                candidate["api_only_projected_noisy_text"] = project_candidate_text(dataset, item, tokenizer, args)
            if bool(item["is_real"]):
                real_shuffled_index = shuffled_index
            shuffled_candidates.append(candidate)

        examples.append(
            {
                "group_index": group_index,
                "original_text": texts[group_index],
                "tokenized_real_query": real_query,
                "dummy_ratio": args.dummy_ratio,
                "n_dummy": args.n_dummy,
                "epsilon_non_sensitive": args.epsilon_non_sensitive,
                "epsilon_private": args.epsilon_private,
                "max_length": args.max_length,
                "real_len": real_len,
                "prefix_len": prefix_len,
                "private_token_count": private_count,
                "clean_prefix": clean_prefix,
                "privatized_projected_prefix": privatized_prefix,
                "real_suffix": real_suffix,
                "dummy_suffixes": dummy_suffixes,
                "shuffled_candidates": shuffled_candidates,
                "client_real_response_index": real_shuffled_index,
                "de_shuffle_demo": (
                    f"The client records index {real_shuffled_index}; after the server returns responses "
                    f"in shuffled order, it selects response_for_candidate_{real_shuffled_index}."
                ),
            }
        )

    return examples


def write_json(path: Path, examples: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"examples": examples}, f, ensure_ascii=False, indent=2)


def render_markdown(examples: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# DP-SPD Qualitative Examples",
        "",
        "Each example shows the real query split, prefix-conditioned dummy suffixes, shuffled candidates, and the client-side de-shuffling index.",
        "",
    ]
    for example_id, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"## Example {example_id}",
                "",
                f"- Dummy ratio `r`: {example['dummy_ratio']}",
                f"- Dummy count `k`: {example['n_dummy']}",
                f"- Prefix/private token counts: {example['prefix_len']} / {example['private_token_count']}",
                f"- Prefix epsilon / suffix epsilon: {example['epsilon_non_sensitive']} / {example['epsilon_private']}",
                "",
                "**Original real query**",
                "",
                clean_for_markdown(example["tokenized_real_query"]),
                "",
                "**Split and privatized prefix**",
                "",
                f"- Clean prefix: {clean_for_markdown(example['clean_prefix'])}",
                f"- Real suffix: {clean_for_markdown(example['real_suffix'])}",
                f"- Projected privatized prefix: {clean_for_markdown(example['privatized_projected_prefix'])}",
                "",
                "**Generated dummy suffixes**",
                "",
            ]
        )
        for dummy in example["dummy_suffixes"]:
            lines.append(f"- Dummy {dummy['dummy_index']}: {clean_for_markdown(dummy['suffix_text'])}")
        lines.extend(
            [
                "",
                "**Shuffled candidate order**",
                "",
                "Labels are shown only for explanation. In the embedding path, the server receives noisy embeddings; in the api-only path, it receives projected noisy text.",
                "",
            ]
        )
        for candidate in example["shuffled_candidates"]:
            lines.append(
                f"- Candidate {candidate['shuffled_index']} "
                f"(hidden label: `{candidate['label_hidden_from_server']}`, clean text for explanation): "
                f"{clean_for_markdown(candidate['clean_candidate_text_for_explanation'])}"
            )
            if "api_only_projected_noisy_text" in candidate:
                lines.append(
                    f"  - API-only projected noisy text: "
                    f"{clean_for_markdown(candidate['api_only_projected_noisy_text'])}"
                )
        lines.extend(
            [
                "",
                "**De-shuffling**",
                "",
                example["de_shuffle_demo"],
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = get_parser().parse_args()
    if args.n_dummy < 1:
        raise ValueError("--n_dummy must be at least 1 for qualitative real/dummy examples.")
    if args.max_examples < 1:
        raise ValueError("--max_examples must be at least 1.")
    args.device = args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"

    texts = select_texts(args)
    if args.dry_run:
        print("[Dry Run]")
        print(vars(args))
        print(f"selected_examples={len(texts)}")
        for idx, text in enumerate(texts, start=1):
            print(f"[{idx}] {clean_for_markdown(text)}")
        return

    validate_local_model_path(args)
    tokenizer, model, load_dtype = load_local_causal_lm(
        model_path=args.continuation_model,
        device=args.device,
        dtype_preference=args.continuation_dtype,
        local_files_only=args.local_files_only,
    )
    print(f"[Model] {os.path.basename(args.continuation_model.rstrip('/'))} | dtype={load_dtype}")
    print(f"[Data] examples={len(texts)}")

    dataset = build_dataset(texts, tokenizer, model, args)
    examples = build_examples(dataset, texts, tokenizer, args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.output_prefix}.json"
    md_path = out_dir / f"{args.output_prefix}.md"
    write_json(json_path, examples)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(examples))

    print(f"[Saved] {json_path}")
    print(f"[Saved] {md_path}")


if __name__ == "__main__":
    main()
