# -*- coding: utf-8 -*-
"""Measure how the DP-SPD continuation/proxy generator scale affects quality.

The experiment deliberately separates three roles that were coupled in older
generation scripts:

* ``transport_model`` provides the fixed client embedding vocabulary, DP noise,
  and nearest-token projection.
* ``proxy_model`` is the *only* model varied across 0.5B/3B/7B/32B.  It
  generates the continuation dummies after a privatized projected prefix.
* ``quality_model`` and ``mia_attack_model`` are fixed scorers.  Keeping them
  fixed makes PPL and MIA curves attributable to the proxy scale rather than a
  stronger evaluator/attacker.

All Qwen2.5 models in the intended sweep share one tokenizer.  A strict vocab
check rejects accidental cross-family configurations because generation dummies
must be valid tokens for the fixed transport projection model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpspd.client import BatchConfig, DummyConfig, LDPNoiseConfig, PrivacyDataset  # noqa: E402
from dpspd.util.utils import get_model_load_dtype, str2bool  # noqa: E402
from experiments.generation.llama_generation_eval import default_dataset_path, default_split, format_texts  # noqa: E402
from experiments.wildchat.run_dpspd_blackbox_attack import (  # noqa: E402
    project_batch_candidates,
    score_text_candidates_mia,
)
from experiments.wildchat.run_original_upper_bound import (  # noqa: E402
    parse_max_memory_spec,
    rouge_l_f1,
)


DEFAULT_TRANSPORT_MODEL = "models/Qwen2.5-0.5B-Instruct"
_ROUGE_L_SCORER = None
_ROUGE_L_SCORER_READY = False


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DP-SPD proxy-generator scale evaluation with fixed transport and fixed MIA scorer."
    )
    # Data and common DP-SPD setting.  These defaults match the generation
    # experiments' WikiText-2 scale, and every proxy run must use exactly the
    # same values.
    parser.add_argument("--dataset_name", choices=["wikitext", "alpaca", "dolly"], default="wikitext")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_mode", choices=["first", "shuffle"], default="first")

    parser.add_argument("--target_epsilon", type=float, default=3.0)
    parser.add_argument("--dummy_k", type=int, default=9)
    parser.add_argument("--dummy_ratio", type=float, default=0.2)
    parser.add_argument("--epsilon_non_sensitive", type=float, default=3.0)
    parser.add_argument("--epsilon_private", type=float, default=7.82)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--noise_mechanism", choices=["Laplace", "Gaussian"], default="Laplace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection_chunk_size", type=int, default=8)

    # Fixed client transport/projection role.
    parser.add_argument("--transport_model", type=str, default=DEFAULT_TRANSPORT_MODEL)
    parser.add_argument("--transport_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--transport_device", type=str, default="cuda:0")

    # The one intervention in this experiment.
    parser.add_argument("--proxy_model", type=str, required=True)
    parser.add_argument("--proxy_label", type=str, required=True, help="Plot label, e.g. 0.5B, 3B, 7B, 32B.")
    parser.add_argument("--proxy_size_b", type=float, default=None, help="Numeric parameter scale for plot ordering.")
    parser.add_argument("--proxy_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--proxy_device", type=str, default="cuda:0")
    parser.add_argument("--proxy_device_map", type=str, default="none", help='"none" or a Transformers device_map such as "auto".')
    parser.add_argument("--proxy_max_memory", type=str, default=None, help='For a sharded 32B proxy, e.g. "3:44GiB,4:44GiB".')

    # The quality and attack scorers must stay fixed across all four runs.
    # An omitted quality model deliberately reuses the attack scorer, avoiding
    # an unnecessary third 0.5B copy on a GPU.
    parser.add_argument("--quality_model", type=str, default=None)
    parser.add_argument("--quality_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--quality_device", type=str, default="cuda:0")
    parser.add_argument("--quality_device_map", type=str, default="none")
    parser.add_argument("--quality_max_memory", type=str, default=None)
    parser.add_argument("--mia_attack_model", type=str, default=None)
    parser.add_argument("--mia_attack_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--mia_device", type=str, default="cuda:0")
    parser.add_argument("--mia_device_map", type=str, default="none")
    parser.add_argument("--mia_max_memory", type=str, default=None)
    parser.add_argument("--mia_max_length", type=int, default=256)
    parser.add_argument("--mia_k_ratio", type=float, default=0.20)

    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--trust_remote_code", type=str2bool, default=False)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--save_candidate_text", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default="outputs/proxy_generator_scale")
    parser.add_argument("--output_prefix", type=str, default=None)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if torch.cuda.is_available() and str(requested).startswith("cuda"):
        return torch.device(requested)
    return torch.device("cpu")


def model_input_device(model, fallback: str) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        if emb is not None and emb.weight.device.type != "meta":
            return emb.weight.device
    except (AttributeError, StopIteration):
        pass
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return resolve_device(fallback)


def load_model(
    *,
    role: str,
    model_path: str,
    dtype_name: str,
    device_name: str,
    device_map: str,
    max_memory: Optional[str],
    args,
) -> Tuple[Any, Any, torch.dtype]:
    """Load one local HF causal LM, including sharded proxy models."""
    device = resolve_device(device_name)
    dtype = get_model_load_dtype(
        model_path=model_path,
        device=str(device),
        dtype_preference=dtype_name,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs: Dict[str, Any] = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if str(device_map).lower() != "none":
        kwargs["device_map"] = device_map
        parsed_max_memory = parse_max_memory_spec(max_memory)
        if parsed_max_memory:
            kwargs["max_memory"] = parsed_max_memory
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
    if str(device_map).lower() == "none":
        model = model.to(device)
    model.eval()
    placement = getattr(model, "hf_device_map", None)
    print(f"[{role}] {model_path} | dtype={dtype} | input_device={model_input_device(model, device_name)}")
    if placement:
        print(f"[{role}] hf_device_map={placement}")
    return tokenizer, model, dtype


def tokenizer_signature(tokenizer) -> Dict[str, Any]:
    sentinel = "DP-SPD tokenizer compatibility: privacy, continuation, 12345."
    return {
        "length": int(len(tokenizer)),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "sentinel_ids": tuple(tokenizer(sentinel, add_special_tokens=True)["input_ids"]),
    }


def first_token_id_mapping_mismatch(transport_tokenizer, proxy_tokenizer) -> Optional[Tuple[int, str, str]]:
    """Return the first different ID->token entry, if any.

    The client sends *transport-tokenizer IDs* directly to the proxy and later
    returns proxy-generated IDs to the transport path.  Bidirectional ID
    semantics, rather than tokenizer template metadata, is therefore the
    safety condition for this experiment.
    """
    transport_len = int(len(transport_tokenizer))
    proxy_len = int(len(proxy_tokenizer))
    if transport_len != proxy_len:
        return (-1, f"vocab_length={transport_len}", f"vocab_length={proxy_len}")
    batch_size = 4096
    for start in range(0, transport_len, batch_size):
        ids = list(range(start, min(start + batch_size, transport_len)))
        transport_tokens = transport_tokenizer.convert_ids_to_tokens(ids)
        proxy_tokens = proxy_tokenizer.convert_ids_to_tokens(ids)
        for token_id, transport_token, proxy_token in zip(ids, transport_tokens, proxy_tokens):
            if transport_token != proxy_token:
                return (int(token_id), str(transport_token), str(proxy_token))
    return None


def require_matching_tokenizers(transport_tokenizer, proxy_tokenizer) -> None:
    """Verify that both models assign identical semantics to every token ID.

    Qwen checkpoints can carry tokenizer metadata/template differences (for
    example a different configured BOS/PAD default) while retaining precisely
    the same vocabulary and ID mapping.  That difference is harmless here:
    PrivacyDataset tokenizes only with the fixed transport tokenizer and passes
    those IDs directly to the proxy.  The former check rejected this valid 7B
    configuration before generation began.
    """
    mismatch = first_token_id_mapping_mismatch(transport_tokenizer, proxy_tokenizer)
    if mismatch is not None:
        token_id, transport_token, proxy_token = mismatch
        raise ValueError(
            "The proxy and transport tokenizers do not assign the same token to every ID; "
            f"first mismatch at ID {token_id}: transport={transport_token!r}, proxy={proxy_token!r}. "
            "This proxy cannot safely consume DP-SPD transport IDs."
        )
    transport_signature = tokenizer_signature(transport_tokenizer)
    proxy_signature = tokenizer_signature(proxy_tokenizer)
    if transport_signature != proxy_signature:
        differing_fields = [
            key for key in transport_signature
            if transport_signature[key] != proxy_signature[key]
        ]
        print(
            "[Tokenizer compatibility] Token-ID mapping is identical; continuing despite "
            f"non-semantic tokenizer metadata differences: {', '.join(differing_fields)}."
        )


def load_texts(args) -> List[str]:
    dataset_path = args.dataset_path or default_dataset_path(args.dataset_name)
    dataset_split = args.dataset_split or default_split(args.dataset_name)
    path = Path(dataset_path)
    if path.is_dir():
        raw = load_from_disk(str(path))
        dataset = raw[dataset_split] if hasattr(raw, "keys") else raw
    elif args.dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-v1", split=dataset_split)
    else:
        raise FileNotFoundError(f"Local dataset path does not exist: {dataset_path}")
    texts = format_texts(args.dataset_name, dataset, max_samples=0 if args.max_samples <= 0 else args.max_samples * 4)
    if args.sample_mode == "shuffle":
        rng = random.Random(args.sample_seed)
        rng.shuffle(texts)
    selected = texts[: args.max_samples] if args.max_samples > 0 else texts
    if not selected:
        raise ValueError("No valid examples are available for this proxy-scale run.")
    return selected


def build_privacy_dataset(texts, tokenizer, transport_model, proxy_model, args) -> PrivacyDataset:
    batch_cfg = BatchConfig(
        max_length=args.max_length,
        device=torch.device("cpu"),
        hidden_size=int(transport_model.config.hidden_size),
        preserve_prefix_hash=True,
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
        pretrained_embeddings=transport_model.get_input_embeddings(),
        base_model=proxy_model,
    )


def conditional_suffix_ppl(
    model,
    full_ids: torch.Tensor,
    prefix_len: int,
    fallback_device: str,
) -> float:
    """PPL of the clean suffix conditioned on DP-projected prefix tokens.

    This is an oracle continuation-quality metric: the prompt is the exact
    privatized prefix seen by the proxy, while loss is restricted to the clean
    held-out continuation.  It isolates LM quality from the dummy sampling
    randomness and is comparable across Qwen2.5's shared vocabulary.
    """
    if int(full_ids.numel()) <= int(prefix_len):
        return float("nan")
    device = model_input_device(model, fallback_device)
    input_ids = full_ids.to(device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    # logit at position t predicts token t+1.  The first private token is at
    # `prefix_len`, hence its loss is at shifted index `prefix_len - 1`.
    first_loss = max(int(prefix_len) - 1, 0)
    if first_loss >= shift_labels.shape[1]:
        return float("nan")
    losses = nn.functional.cross_entropy(
        shift_logits[:, first_loss:, :].reshape(-1, shift_logits.size(-1)),
        shift_labels[:, first_loss:].reshape(-1),
        reduction="mean",
    )
    return float(math.exp(min(float(losses.item()), 50.0)))


def finite(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def summary_stats(values: Iterable[float], prefix: str) -> Dict[str, Any]:
    valid = finite(values)
    if not valid:
        return {f"{prefix}_{key}": "" for key in ("mean", "min", "max", "std")}
    return {
        f"{prefix}_mean": float(np.mean(valid)),
        f"{prefix}_min": float(np.min(valid)),
        f"{prefix}_max": float(np.max(valid)),
        f"{prefix}_std": float(np.std(valid)),
    }


def parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def metric_value(values: Sequence[float]) -> float:
    valid = finite(values)
    return float(np.mean(valid)) if valid else float("nan")


def standard_rouge_l_f1(prediction: str, reference: str) -> float:
    """One-example ROUGE-L with the exact scorer used for corpus ROUGE."""
    global _ROUGE_L_SCORER, _ROUGE_L_SCORER_READY
    if not _ROUGE_L_SCORER_READY:
        try:
            from rouge_score import rouge_scorer

            _ROUGE_L_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        except ImportError:
            _ROUGE_L_SCORER = None
        _ROUGE_L_SCORER_READY = True
    if _ROUGE_L_SCORER is None:
        return rouge_l_f1(prediction, reference)
    return float(_ROUGE_L_SCORER.score(reference, prediction)["rougeL"].fmeasure)


def calculate_project_rouge(predictions: Sequence[str], references: Sequence[str]) -> Tuple[float, str]:
    """Compute the generation evaluator's standard stemmed ROUGE-L offline.

    The historic evaluator calls ``evaluate.load('rouge')``, whose ROUGE-L
    implementation is ``rouge_score`` with ``use_stemmer=True``.  Loading the
    Evaluate metric can download its metric script on first use, which is
    undesirable on isolated experiment servers.  Calling its installed
    underlying scorer directly preserves the same ROUGE-L F1 definition and
    corpus mean without a network dependency.
    """
    pairs = [(str(prediction), str(reference)) for prediction, reference in zip(predictions, references) if str(reference).strip()]
    if not pairs:
        return 0.0, "empty"
    values = [standard_rouge_l_f1(prediction, reference) for prediction, reference in pairs]
    evaluator = "rouge_score_rougeL_stemmer_mean" if _ROUGE_L_SCORER is not None else "local_lcs_f1_fallback"
    return float(np.mean(values)), evaluator


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def model_cache_key(path: str, dtype: str, device: str, device_map: str, max_memory: Optional[str]) -> Tuple[str, ...]:
    return (str(path), str(dtype), str(device), str(device_map), str(max_memory))


def make_prefix(args) -> str:
    if args.output_prefix:
        return args.output_prefix
    raw = f"{args.dataset_name}_qwen25_{args.proxy_label}_eps{args.target_epsilon:g}_k{args.dummy_k}_r{args.dummy_ratio:g}"
    return raw.replace("/", "_").replace(" ", "_")


def run() -> None:
    args = get_parser().parse_args()
    if args.dummy_k < 1:
        raise ValueError("--dummy_k must be at least 1 for MIA candidate-group evaluation.")
    if not 0.0 < args.dummy_ratio < 1.0:
        raise ValueError("--dummy_ratio must be strictly between 0 and 1 for continuation dummies.")
    args.quality_model = args.quality_model or args.mia_attack_model or args.transport_model
    args.mia_attack_model = args.mia_attack_model or args.quality_model
    prefix = make_prefix(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_path = output_dir / f"{prefix}_groups.jsonl"
    summary_path = output_dir / f"{prefix}_summary.csv"
    config_path = output_dir / f"{prefix}_config.json"
    if not args.overwrite and any(path.exists() for path in (groups_path, summary_path, config_path)):
        raise FileExistsError(
            f"Output already exists for {prefix}. Use --overwrite true or choose --output_prefix."
        )
    for path in (groups_path, summary_path, config_path):
        if path.exists():
            path.unlink()

    seed_everything(args.seed)
    if args.dry_run:
        write_json(config_path, {"args": vars(args), "dry_run": True})
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return

    # Keep identical model specifications in one process truly shared.  In the
    # recommended command the transport, quality scorer, and MIA scorer are a
    # single fixed Qwen2.5-0.5B object on the same GPU.
    loaded: Dict[Tuple[str, ...], Tuple[Any, Any, torch.dtype]] = {}

    def cached_load(role: str, path: str, dtype: str, device: str, device_map: str, max_memory: Optional[str]):
        key = model_cache_key(path, dtype, device, device_map, max_memory)
        if key not in loaded:
            loaded[key] = load_model(
                role=role,
                model_path=path,
                dtype_name=dtype,
                device_name=device,
                device_map=device_map,
                max_memory=max_memory,
                args=args,
            )
        else:
            print(f"[{role}] Reusing already-loaded model: {path}")
        return loaded[key]

    transport_tokenizer, transport_model, transport_dtype = cached_load(
        "Transport", args.transport_model, args.transport_dtype, args.transport_device, "none", None
    )
    proxy_tokenizer, proxy_model, proxy_dtype = cached_load(
        "Proxy", args.proxy_model, args.proxy_dtype, args.proxy_device, args.proxy_device_map, args.proxy_max_memory
    )
    require_matching_tokenizers(transport_tokenizer, proxy_tokenizer)
    quality_tokenizer, quality_model, quality_dtype = cached_load(
        "Quality scorer", args.quality_model, args.quality_dtype, args.quality_device,
        args.quality_device_map, args.quality_max_memory,
    )
    mia_tokenizer, mia_model, mia_dtype = cached_load(
        "MIA scorer", args.mia_attack_model, args.mia_attack_dtype, args.mia_device,
        args.mia_device_map, args.mia_max_memory,
    )

    texts = load_texts(args)
    dataset_start = time.perf_counter()
    dataset = build_privacy_dataset(texts, transport_tokenizer, transport_model, proxy_model, args)
    dataset_build_seconds = time.perf_counter() - dataset_start
    timing = dict(dataset.generation_timing)
    print(
        "[Timing] proxy_generate="
        f"{timing['generation_seconds']:.3f}s for {timing['generated_sequences']} dummies "
        f"({timing['generated_tokens']} tokens); dataset_build={dataset_build_seconds:.3f}s"
    )

    config = {
        "args": vars(args),
        "roles": {
            "transport": {"model": args.transport_model, "dtype": str(transport_dtype), "device": str(model_input_device(transport_model, args.transport_device))},
            "proxy": {"model": args.proxy_model, "dtype": str(proxy_dtype), "device": str(model_input_device(proxy_model, args.proxy_device))},
            "quality_scorer": {"model": args.quality_model, "dtype": str(quality_dtype), "device": str(model_input_device(quality_model, args.quality_device))},
            "mia_scorer": {"model": args.mia_attack_model, "dtype": str(mia_dtype), "device": str(model_input_device(mia_model, args.mia_device))},
        },
        "design": {
            "only_proxy_model_varies": True,
            "transport_and_projection_fixed": True,
            "quality_and_mia_scorers_fixed": True,
            "proxy_oracle_ppl": "Clean suffix PPL conditioned on the DP-projected prefix, scored by the tested proxy.",
            "surface_dummy_ppl": "PPL of final DP-projected dummy candidates, scored by a fixed evaluator.",
            "mia": "Fixed scorer selects one real candidate among k+1 final DP-projected candidates; random baseline is 1/(k+1).",
        },
        "dataset_examples": len(texts),
        "client_generation_timing": timing,
        "dataset_build_seconds": dataset_build_seconds,
    }
    write_json(config_path, config)

    dataloader = DataLoader(dataset, batch_size=args.dummy_k + 1, shuffle=False, num_workers=0)
    records: List[Dict[str, Any]] = []
    raw_predictions: List[str] = []
    raw_references: List[str] = []
    surface_predictions: List[str] = []
    surface_references: List[str] = []

    for batch_index, batch in enumerate(dataloader):
        group = project_batch_candidates(batch, dataset, transport_tokenizer, args)
        group_idx = int(group["group_index"])
        cache = dataset.generation_cache[group_idx]
        clean_ids = cache["input_ids"][: int(cache["real_len"])].to(torch.long)
        prefix_len = int(cache["prefix_len"])
        real_len = int(cache["real_len"])
        clean_suffix_ids = clean_ids[prefix_len:real_len]
        clean_suffix_text = transport_tokenizer.decode(
            clean_suffix_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()
        oracle_ids = torch.cat([cache["projected_prefix_ids"], clean_suffix_ids], dim=0)
        proxy_oracle_ppl = conditional_suffix_ppl(proxy_model, oracle_ids, prefix_len, args.proxy_device)

        raw_dummy_texts = [
            transport_tokenizer.decode(
                dummy_ids[prefix_len:real_len].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            for dummy_ids in cache["dummy_input_ids"]
        ]
        raw_dummy_rouges = [standard_rouge_l_f1(dummy_text, clean_suffix_text) for dummy_text in raw_dummy_texts]
        raw_predictions.extend(raw_dummy_texts)
        raw_references.extend([clean_suffix_text] * len(raw_dummy_texts))

        candidates = group["candidates"]
        actual_real_index = int(group["actual_real_index"])
        candidate_texts = [str(candidate["projected_text"]) for candidate in candidates]
        mia_scores = score_text_candidates_mia(candidate_texts, mia_tokenizer, mia_model, args)
        quality_scores = (
            mia_scores
            if quality_model is mia_model and quality_tokenizer is mia_tokenizer
            else score_text_candidates_mia(candidate_texts, quality_tokenizer, quality_model, args)
        )
        dummy_candidates = [candidate for candidate in candidates if not bool(candidate["is_real_hidden"])]
        surface_dummy_texts = [str(candidate["projected_suffix_text"]) for candidate in dummy_candidates]
        surface_dummy_rouges = [standard_rouge_l_f1(dummy_text, clean_suffix_text) for dummy_text in surface_dummy_texts]
        surface_predictions.extend(surface_dummy_texts)
        surface_references.extend([clean_suffix_text] * len(surface_dummy_texts))

        dummy_positions = [idx for idx, candidate in enumerate(candidates) if not bool(candidate["is_real_hidden"])]
        record: Dict[str, Any] = {
            "group_index": group_idx,
            "batch_index": batch_index,
            "prefix_len": prefix_len,
            "suffix_len": int(cache["n_private"]),
            "proxy_oracle_suffix_ppl": proxy_oracle_ppl,
            "raw_dummy_rouge_l": metric_value(raw_dummy_rouges),
            "surface_dummy_rouge_l": metric_value(surface_dummy_rouges),
            "quality_real_ppl": float(quality_scores["mia_ppls"][actual_real_index]),
            "quality_dummy_ppl": metric_value([quality_scores["mia_ppls"][idx] for idx in dummy_positions]),
            "mia_real_ppl": float(mia_scores["mia_ppls"][actual_real_index]),
            "mia_dummy_ppl": metric_value([mia_scores["mia_ppls"][idx] for idx in dummy_positions]),
            "mia_ppl_correct": float(int(mia_scores["mia_predicted_real_index_ppl"]) == actual_real_index),
            "mia_min_k_correct": float(int(mia_scores["mia_predicted_real_index_min_k"]) == actual_real_index),
            "mia_predicted_real_index_ppl": int(mia_scores["mia_predicted_real_index_ppl"]),
            "mia_predicted_real_index_min_k": int(mia_scores["mia_predicted_real_index_min_k"]),
            "actual_real_index_hidden": actual_real_index,
            "avg_projection_token_accuracy": float(group["avg_projection_token_accuracy"]),
            "total_projection_changed_tokens": int(group["total_projection_changed_tokens"]),
        }
        if args.save_candidate_text:
            record.update(
                {
                    "clean_suffix_text": clean_suffix_text,
                    "raw_dummy_suffixes": raw_dummy_texts,
                    "surface_dummy_suffixes": surface_dummy_texts,
                    "candidate_texts": candidate_texts,
                }
            )
        append_jsonl(groups_path, record)
        records.append(record)
        print(
            f"[Group {batch_index + 1}/{len(dataloader)}] proxy-oracle PPL={proxy_oracle_ppl:.3f} "
            f"surface ROUGE-L={record['surface_dummy_rouge_l']:.4f} "
            f"MIA(min-k)={record['mia_min_k_correct']:.0f}"
        )

    # These two means use the same RougeScore ROUGE-L F1 and stemmer choice as
    # the paper's historical Evaluate-based generation evaluator. Per-group
    # values above are retained only to draw the min--max band in the plot.
    raw_corpus_rouge, rouge_evaluator = calculate_project_rouge(raw_predictions, raw_references)
    surface_corpus_rouge, surface_rouge_evaluator = calculate_project_rouge(surface_predictions, surface_references)
    generated_tokens = int(timing["generated_tokens"])
    generation_seconds = float(timing["generation_seconds"])
    summary: Dict[str, Any] = {
        "proxy_label": args.proxy_label,
        "proxy_size_b": args.proxy_size_b if args.proxy_size_b is not None else "",
        "proxy_model": args.proxy_model,
        "proxy_parameter_count": parameter_count(proxy_model),
        "transport_model": args.transport_model,
        "quality_model": args.quality_model,
        "mia_attack_model": args.mia_attack_model,
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path or default_dataset_path(args.dataset_name),
        "dataset_split": args.dataset_split or default_split(args.dataset_name),
        "groups": len(records),
        "target_epsilon": args.target_epsilon,
        "dummy_k": args.dummy_k,
        "dummy_ratio": args.dummy_ratio,
        "epsilon_non_sensitive": args.epsilon_non_sensitive,
        "epsilon_private": args.epsilon_private,
        "client_proxy_generation_seconds": generation_seconds,
        "client_dataset_build_seconds": dataset_build_seconds,
        "generated_dummy_sequences": int(timing["generated_sequences"]),
        "generated_dummy_tokens": generated_tokens,
        "client_proxy_dummy_tokens_per_second": generated_tokens / generation_seconds if generation_seconds > 0 else float("nan"),
        "client_proxy_seconds_per_group": generation_seconds / max(len(records), 1),
        "raw_dummy_rouge_l_corpus": raw_corpus_rouge,
        "surface_dummy_rouge_l_corpus": surface_corpus_rouge,
        "rouge_evaluator": rouge_evaluator,
        "surface_rouge_evaluator": surface_rouge_evaluator,
        "mia_random_baseline": 1.0 / float(args.dummy_k + 1),
    }
    for source_key, result_key in (
        ("proxy_oracle_suffix_ppl", "proxy_oracle_suffix_ppl"),
        ("raw_dummy_rouge_l", "raw_dummy_rouge_l"),
        ("surface_dummy_rouge_l", "surface_dummy_rouge_l"),
        ("quality_real_ppl", "quality_real_ppl"),
        ("quality_dummy_ppl", "quality_dummy_ppl"),
        ("mia_real_ppl", "mia_real_ppl"),
        ("mia_dummy_ppl", "mia_dummy_ppl"),
        ("mia_ppl_correct", "mia_ppl_asr"),
        ("mia_min_k_correct", "mia_min_k_asr"),
        ("avg_projection_token_accuracy", "projection_token_accuracy"),
        ("total_projection_changed_tokens", "projection_changed_tokens"),
    ):
        summary.update(summary_stats([record[source_key] for record in records], result_key))
    write_summary(summary_path, summary)
    config["rouge_evaluator"] = rouge_evaluator
    config["surface_rouge_evaluator"] = surface_rouge_evaluator
    write_json(config_path, config)
    print(f"[Saved] config:  {config_path}")
    print(f"[Saved] groups:  {groups_path}")
    print(f"[Saved] summary: {summary_path}")


if __name__ == "__main__":
    run()
