# -*- coding: utf-8 -*-
"""WildChat DP-SPD grouped-serving speed benchmark for large local Qwen models.

This benchmark measures the efficiency contribution of DP-SPD's shared-prefix
design. It supports two modes:

1. real: construct continuation dummies through the current PrivacyDataset
   pipeline, then compare server-side paths.
2. synthetic: skip client dummy generation and directly construct shared-prefix
   real+dummy embedding groups. This isolates the server-side KV-cache reuse
   benefit and is useful for stable server-side speed measurements.

For both modes, it compares two server-side transformer paths:

1. Plain: run all real+dummy candidates as an ordinary batch.
2. Grouped: compute the shared prefix once, expand the KV cache, and run the
   suffix batch with past_key_values.

The benchmark intentionally skips answer generation/utility judging. It times
transformer forward passes and optionally uses torch.profiler to aggregate CUDA
time for attention/MHA-like kernels and linear/GEMM-like kernels.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpspd.client import BatchConfig, DummyConfig, LDPNoiseConfig, PrivacyDataset  # noqa: E402
from dpspd.util.utils import get_model_load_dtype, str2bool  # noqa: E402
from experiments.wildchat.run_original_upper_bound import parse_max_memory_spec  # noqa: E402


DEFAULT_QWEN3_32B_PATH = "models/Qwen3-32B"
DEFAULT_QA_JSONL = "data/wildchat/wildchat_qa500_independent_userprivacy.jsonl"


def parse_csv_list(text: str, cast):
    return [cast(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_k_float_map(text: str) -> Dict[int, float]:
    mapping: Dict[int, float] = {}
    for item in str(text or "").split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        mapping[int(key.strip())] = float(value.strip())
    return mapping


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WildChat Qwen3 DP-SPD grouped-serving speed benchmark.")
    parser.add_argument("--input_jsonl", type=str, default=DEFAULT_QA_JSONL)
    parser.add_argument("--output_dir", type=str, default="outputs/wildchat_speed_benchmark")
    parser.add_argument("--output_prefix", type=str, default="qwen3_32b_wildchat_dpspd_speed")
    parser.add_argument(
        "--benchmark_mode",
        type=str,
        default="real",
        choices=["real", "synthetic"],
        help=(
            "real uses WildChat + PrivacyDataset dummy generation. synthetic directly constructs "
            "shared-prefix embedding groups and isolates server-side KV-cache reuse."
        ),
    )

    parser.add_argument("--model_path", type=str, default=DEFAULT_QWEN3_32B_PATH)
    parser.add_argument("--dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda:0", help="Input device when device_map is none.")
    parser.add_argument("--device_map", type=str, default="auto", help='Use "none" to load the whole model on --device.')
    parser.add_argument("--max_memory", type=str, default="all:42GiB")
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--trust_remote_code", type=str2bool, default=False)
    parser.add_argument("--attn_implementation", type=str, default=None)

    parser.add_argument("--max_samples", type=int, default=8, help="Number of real WildChat queries/groups.")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_mode", type=str, default="first", choices=["first", "shuffle"])
    parser.add_argument("--min_words", type=int, default=5)
    parser.add_argument("--max_words", type=int, default=260)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_source_chars", type=int, default=2400)
    parser.add_argument(
        "--synthetic_seq_len",
        type=int,
        default=0,
        help="Sequence length for --benchmark_mode synthetic. 0 means use --max_length.",
    )
    parser.add_argument(
        "--synthetic_vocab_start",
        type=int,
        default=100,
        help="Lower token id bound for synthetic token sampling.",
    )
    parser.add_argument(
        "--synthetic_vocab_limit",
        type=int,
        default=32000,
        help="Upper token id bound for synthetic token sampling before clamping to vocab size.",
    )

    parser.add_argument("--ks", type=str, default="9,19,29")
    parser.add_argument("--dummy_ratios", type=str, default="0.2,0.5,0.8")
    parser.add_argument("--epsilon_non_sensitive", type=float, default=3.0)
    parser.add_argument("--epsilon_private_default", type=float, default=8.51)
    parser.add_argument(
        "--epsilon_private_map",
        type=str,
        default="9:7.82,19:8.51",
        help="Optional k-specific suffix epsilon map. k values not listed use --epsilon_private_default.",
    )
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--noise_mechanism", type=str, default="Laplace", choices=["Laplace", "Gaussian"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--repeat_steps", type=int, default=3)
    parser.add_argument("--profile", type=str2bool, default=True)
    parser.add_argument(
        "--module_profile",
        type=str2bool,
        default=True,
        help="Also time self-attention, MLP, and Linear modules with CUDA events for a cleaner layer-level view.",
    )
    parser.add_argument("--profile_top_ops", type=int, default=12)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


def normalize_space(text: Any) -> str:
    return " ".join(str(text or "").replace("\r", " ").split()).strip()


def trim_text(text: Any, max_chars: int) -> str:
    value = normalize_space(text)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + " ..."


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_text(row: Dict[str, Any], max_chars: int) -> str:
    for key in ("current_user_text", "text", "input_text", "source_text", "prompt"):
        value = normalize_space(row.get(key))
        if value:
            return trim_text(value, max_chars)
    messages = row.get("messages")
    if isinstance(messages, list):
        for item in reversed(messages):
            if isinstance(item, dict) and item.get("role") == "user":
                return trim_text(item.get("content"), max_chars)
    return ""


def load_texts(args) -> List[str]:
    rows = read_jsonl(Path(args.input_jsonl))
    texts = []
    for row in rows:
        text = row_text(row, args.max_source_chars)
        n_words = len(text.split())
        if text and args.min_words <= n_words <= args.max_words:
            texts.append(text)
    if not texts:
        raise ValueError(f"No eligible texts found in {args.input_jsonl}")
    if args.sample_mode == "shuffle":
        rng = random.Random(args.sample_seed)
        rng.shuffle(texts)
    selected = texts[: args.max_samples]
    if len(selected) < args.max_samples:
        print(f"[Warning] Only {len(selected)} eligible texts; requested {args.max_samples}.")
    return selected


def resolve_model_input_device(model, fallback: str) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def resolve_model_input_dtype(model, fallback: torch.dtype) -> torch.dtype:
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb.weight.dtype
    except Exception:
        pass
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return fallback


def load_local_causal_lm(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    load_dtype = get_model_load_dtype(
        model_path=args.model_path,
        device=str(device),
        dtype_preference=args.dtype,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: Dict[str, Any] = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if str(args.device_map).lower() != "none":
        model_kwargs["device_map"] = args.device_map
        max_memory = parse_max_memory_spec(args.max_memory)
        if max_memory:
            model_kwargs["max_memory"] = max_memory

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=load_dtype, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=load_dtype, **model_kwargs)
    if str(args.device_map).lower() == "none":
        model = model.to(device)
    model.eval()
    return tokenizer, model, load_dtype


def transformer_core(model):
    return getattr(model, "model", model)


def collate_group(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "noisy_embeddings": torch.stack([item["noisy_embeddings"] for item in batch]),
        "prefix_hashes": torch.tensor([int(item["group_index"]) for item in batch], dtype=torch.long),
        "prefix_lens": torch.tensor([int(item["prefix_len"]) for item in batch], dtype=torch.long),
        "private_token_counts": torch.tensor([int(item["private_token_count"]) for item in batch], dtype=torch.long),
        "is_real": torch.tensor([bool(item["is_real"]) for item in batch], dtype=torch.bool),
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device, input_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if key == "noisy_embeddings":
            moved[key] = value.to(device=device, dtype=input_dtype)
        elif key in {"prefix_hashes", "prefix_lens", "private_token_counts", "is_real"}:
            moved[key] = value.to(device)
        else:
            moved[key] = value.to(device)
    return moved


def build_batches(
    dataset: PrivacyDataset,
    k: int,
    input_device: torch.device,
    input_dtype: torch.dtype,
) -> List[Dict[str, torch.Tensor]]:
    loader = DataLoader(dataset, batch_size=k + 1, shuffle=False, collate_fn=collate_group)
    return [move_batch(batch, input_device, input_dtype) for batch in loader]


def synthetic_prefix_len(seq_len: int, ratio: float) -> int:
    prefix_len = int(round(seq_len * (1.0 - float(ratio))))
    return max(1, min(seq_len - 1, prefix_len))


def build_synthetic_batches(
    num_groups: int,
    k: int,
    ratio: float,
    tokenizer,
    model,
    input_device: torch.device,
    input_dtype: torch.dtype,
    seq_len: int,
    seed: int,
    vocab_start: int,
    vocab_limit: int,
) -> List[Dict[str, torch.Tensor]]:
    """Construct shared-prefix candidate groups without running dummy generation.

    The embedding values are obtained from the model's own token embedding
    matrix, so the server forward path is realistic while avoiding client-side
    continuation-model latency. Candidate 0 is marked as the real query; the
    remaining k candidates act as synthetic dummy suffixes.
    """
    if seq_len < 2:
        raise ValueError("--synthetic_seq_len/--max_length must be at least 2 for synthetic benchmarking.")
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or len(tokenizer))
    low = max(0, min(int(vocab_start), vocab_size - 1))
    high = max(low + 1, min(int(vocab_limit), vocab_size))
    prefix_len = synthetic_prefix_len(seq_len, ratio)
    suffix_len = seq_len - prefix_len
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(k) * 1009 + int(round(float(ratio) * 1000)) * 917)
    embeddings = model.get_input_embeddings()
    batches: List[Dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for group_idx in range(num_groups):
            prefix_ids = torch.randint(low, high, (prefix_len,), generator=generator, dtype=torch.long)
            suffix_ids = torch.randint(low, high, (k + 1, suffix_len), generator=generator, dtype=torch.long)
            input_ids = torch.cat([prefix_ids.unsqueeze(0).expand(k + 1, -1), suffix_ids], dim=1).to(input_device)
            embeds = embeddings(input_ids).detach().to(dtype=input_dtype)
            batch = {
                "attention_mask": torch.ones((k + 1, seq_len), dtype=torch.long, device=input_device),
                "noisy_embeddings": embeds,
                "prefix_hashes": torch.full((k + 1,), group_idx, dtype=torch.long, device=input_device),
                "prefix_lens": torch.full((k + 1,), prefix_len, dtype=torch.long, device=input_device),
                "private_token_counts": torch.full((k + 1,), suffix_len, dtype=torch.long, device=input_device),
                "is_real": torch.tensor([True] + [False] * k, dtype=torch.bool, device=input_device),
            }
            batches.append(batch)
    return batches


def expand_legacy_tuple_cache(past_key_values, batch_size: int):
    expanded_layers = []
    for layer_past in past_key_values:
        if not isinstance(layer_past, (tuple, list)):
            expanded_layers.append(layer_past)
            continue
        expanded_items = []
        for item in layer_past:
            if torch.is_tensor(item) and item.size(0) == 1:
                expanded_items.append(item.expand(batch_size, *item.shape[1:]).contiguous())
            else:
                expanded_items.append(item)
        expanded_layers.append(tuple(expanded_items))
    return tuple(expanded_layers)


def expand_past_key_values(past_key_values, batch_size: int):
    if hasattr(past_key_values, "batch_repeat_interleave"):
        cache = copy.deepcopy(past_key_values)
        result = cache.batch_repeat_interleave(batch_size)
        return result if result is not None else cache
    return expand_legacy_tuple_cache(past_key_values, batch_size)


def plain_forward(core_model, batch: Dict[str, torch.Tensor]):
    return core_model(
        inputs_embeds=batch["noisy_embeddings"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )


def grouped_forward(core_model, batch: Dict[str, torch.Tensor]):
    inputs_embeds = batch["noisy_embeddings"]
    attention_mask = batch["attention_mask"]
    prefix_hashes = batch["prefix_hashes"]
    prefix_lens = batch["prefix_lens"]
    unique_hashes, inverse_indices = torch.unique(prefix_hashes, return_inverse=True)
    outputs = []

    for group_idx, _ in enumerate(unique_hashes):
        group_indices = (inverse_indices == group_idx).nonzero(as_tuple=True)[0]
        if group_indices.numel() == 0:
            continue
        rep_idx = int(group_indices[0].item())
        prefix_len = int(prefix_lens[rep_idx].item())
        prefix_len = max(0, min(prefix_len, inputs_embeds.size(1)))
        if prefix_len <= 0:
            outputs.append(
                core_model(
                    inputs_embeds=inputs_embeds[group_indices],
                    attention_mask=attention_mask[group_indices],
                    use_cache=False,
                )
            )
            continue

        prefix_out = core_model(
            inputs_embeds=inputs_embeds[rep_idx : rep_idx + 1, :prefix_len, :],
            attention_mask=attention_mask[rep_idx : rep_idx + 1, :prefix_len],
            use_cache=True,
        )
        if prefix_len >= inputs_embeds.size(1):
            outputs.append(prefix_out)
            continue

        past = expand_past_key_values(prefix_out.past_key_values, int(group_indices.numel()))
        outputs.append(
            core_model(
                inputs_embeds=inputs_embeds[group_indices, prefix_len:, :],
                attention_mask=attention_mask[group_indices],
                past_key_values=past,
                use_cache=True,
            )
        )
    return outputs


def run_batches(core_model, batches: Sequence[Dict[str, torch.Tensor]], grouped: bool):
    for batch in batches:
        if grouped:
            _ = grouped_forward(core_model, batch)
        else:
            _ = plain_forward(core_model, batch)


def timed_run(core_model, batches: Sequence[Dict[str, torch.Tensor]], repeat_steps: int, warmup_steps: int, device: torch.device, grouped: bool) -> Dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup_steps):
            run_batches(core_model, batches, grouped=grouped)
    sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat_steps):
            run_batches(core_model, batches, grouped=grouped)
    sync(device)
    elapsed = time.perf_counter() - start

    total_candidates = sum(int(batch["attention_mask"].size(0)) for batch in batches) * repeat_steps
    total_queries = len(batches) * repeat_steps
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    return {
        "elapsed_seconds": elapsed,
        "candidates_per_second": total_candidates / max(elapsed, 1e-12),
        "queries_per_second": total_queries / max(elapsed, 1e-12),
        "peak_memory_gb": peak_memory,
    }


def event_time_ms(event, device: torch.device) -> float:
    if device.type == "cuda" and hasattr(event, "self_cuda_time_total"):
        return float(event.self_cuda_time_total) / 1000.0
    return float(getattr(event, "self_cpu_time_total", 0.0)) / 1000.0


def profile_once(core_model, batches: Sequence[Dict[str, torch.Tensor]], device: torch.device, grouped: bool, top_ops: int) -> Dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        with torch.no_grad():
            run_batches(core_model, batches, grouped=grouped)
        sync(device)

    total_ms = 0.0
    attention_ms = 0.0
    linear_ms = 0.0
    top = []
    for event in prof.key_averages():
        ms = event_time_ms(event, device)
        if ms <= 0:
            continue
        key = str(event.key).lower()
        total_ms += ms
        if any(token in key for token in ("attention", "flash", "sdpa", "scaled_dot_product")):
            attention_ms += ms
        if any(token in key for token in ("linear", "mm", "matmul", "addmm", "gemm")):
            linear_ms += ms
        top.append((ms, event.key))
    top.sort(reverse=True)
    top_payload = [{"op": key, "self_time_ms": ms} for ms, key in top[: max(0, top_ops)]]
    return {
        "profile_total_self_time_ms": total_ms,
        "profile_attention_self_time_ms": attention_ms,
        "profile_linear_self_time_ms": linear_ms,
        "profile_other_self_time_ms": max(0.0, total_ms - attention_ms - linear_ms),
        "profile_attention_share": attention_ms / total_ms if total_ms > 0 else 0.0,
        "profile_linear_share": linear_ms / total_ms if total_ms > 0 else 0.0,
        "profile_top_ops": json.dumps(top_payload, ensure_ascii=False),
    }


def first_tensor_device(value: Any) -> Optional[torch.device]:
    if torch.is_tensor(value):
        return value.device
    if isinstance(value, (tuple, list)):
        for item in value:
            device = first_tensor_device(item)
            if device is not None:
                return device
    if isinstance(value, dict):
        for item in value.values():
            device = first_tensor_device(item)
            if device is not None:
                return device
    return None


class ModuleTimer:
    """CUDA-event module timer for coarse attention/MLP/Linear attribution."""

    def __init__(self, model: torch.nn.Module):
        self.handles = []
        self.active: Dict[int, List[Tuple[List[str], Any, Any, Optional[torch.device]]]] = {}
        self.cuda_records: List[Tuple[List[str], torch.cuda.Event, torch.cuda.Event, torch.device]] = []
        self.cpu_totals = {"attention": 0.0, "mlp": 0.0, "linear": 0.0}
        self.used_cuda_devices: set[torch.device] = set()
        self._register(model)

    @staticmethod
    def categories_for(name: str, module: torch.nn.Module) -> List[str]:
        lowered_name = str(name).lower()
        lowered_cls = module.__class__.__name__.lower()
        categories: List[str] = []
        if "self_attn" in lowered_name or "attention" in lowered_cls:
            categories.append("attention")
        if lowered_name.endswith("mlp") or ".mlp" in lowered_name or "mlp" in lowered_cls:
            categories.append("mlp")
        if isinstance(module, torch.nn.Linear):
            categories.append("linear")
        return categories

    def _register(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            categories = self.categories_for(name, module)
            if not categories:
                continue

            def pre_hook(mod, inputs, cats=tuple(categories)):
                device = first_tensor_device(inputs)
                if device is None:
                    try:
                        device = next(mod.parameters()).device
                    except StopIteration:
                        device = None
                if device is not None and device.type == "cuda":
                    with torch.cuda.device(device):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record(torch.cuda.current_stream(device))
                    self.active.setdefault(id(mod), []).append((list(cats), start, end, device))
                    self.used_cuda_devices.add(device)
                else:
                    self.active.setdefault(id(mod), []).append((list(cats), time.perf_counter(), None, None))

            def post_hook(mod, inputs, output):
                stack = self.active.get(id(mod))
                if not stack:
                    return
                cats, start, end, device = stack.pop()
                if device is not None and device.type == "cuda":
                    with torch.cuda.device(device):
                        end.record(torch.cuda.current_stream(device))
                    self.cuda_records.append((cats, start, end, device))
                else:
                    elapsed_ms = (time.perf_counter() - float(start)) * 1000.0
                    for cat in cats:
                        self.cpu_totals[cat] += elapsed_ms

            self.handles.append(module.register_forward_pre_hook(pre_hook))
            self.handles.append(module.register_forward_hook(post_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self) -> Dict[str, float]:
        for device in self.used_cuda_devices:
            torch.cuda.synchronize(device)
        totals = dict(self.cpu_totals)
        for cats, start, end, _device in self.cuda_records:
            elapsed_ms = float(start.elapsed_time(end))
            for cat in cats:
                totals[cat] += elapsed_ms
        return {
            "module_attention_time_ms": totals["attention"],
            "module_mlp_time_ms": totals["mlp"],
            "module_linear_time_ms": totals["linear"],
        }


def module_profile_once(core_model, batches: Sequence[Dict[str, torch.Tensor]], device: torch.device, grouped: bool) -> Dict[str, float]:
    timer = ModuleTimer(core_model)
    try:
        with torch.no_grad():
            run_batches(core_model, batches, grouped=grouped)
        sync(device)
        return timer.summary()
    finally:
        timer.close()


def private_epsilon_for_k(k: int, epsilon_map: Dict[int, float], default: float) -> float:
    return float(epsilon_map.get(int(k), default))


def build_dataset(texts: Sequence[str], tokenizer, model, args, ratio: float, k: int, epsilon_private: float) -> Tuple[PrivacyDataset, float]:
    dummy_cfg = DummyConfig(
        m_private_tokens=0,
        sensitive_ratio=ratio,
        n_dummies_per_text=k,
        dummy_mode="generation",
        dummy_corpus=None,
        pos_mapping_path=None,
    )
    ldp_cfg = LDPNoiseConfig(
        mechanism=args.noise_mechanism,
        epsilon_private=epsilon_private,
        epsilon_non_sensitive=args.epsilon_non_sensitive,
        clip_norm=args.clip_norm,
        seed=args.seed,
    )
    hidden_size = int(getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0)))
    batch_cfg = BatchConfig(max_length=args.max_length, device=torch.device("cpu"), hidden_size=hidden_size)
    started = time.perf_counter()
    dataset = PrivacyDataset(
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
    build_seconds = time.perf_counter() - started
    return dataset, build_seconds


def batch_stats(batches: Sequence[Dict[str, torch.Tensor]], max_length: int) -> Dict[str, float]:
    real_lens = []
    prefix_lens = []
    private_lens = []
    for batch in batches:
        mask = batch["attention_mask"]
        real_lens.extend([int(x) for x in mask.sum(dim=1).detach().cpu().tolist()])
        prefix_lens.extend([int(x) for x in batch["prefix_lens"].detach().cpu().tolist()])
        private_lens.extend([int(x) for x in batch["private_token_counts"].detach().cpu().tolist()])
    avg_real = sum(real_lens) / len(real_lens) if real_lens else 0.0
    avg_prefix = sum(prefix_lens) / len(prefix_lens) if prefix_lens else 0.0
    avg_private = sum(private_lens) / len(private_lens) if private_lens else 0.0
    return {
        "avg_real_len": avg_real,
        "avg_prefix_len": avg_prefix,
        "avg_suffix_len": avg_private,
        "avg_prefix_fraction": avg_prefix / max(avg_real, 1e-12),
        "max_length": float(max_length),
    }


def token_work_bound(num_groups: int, k: int, avg_real_len: float, avg_prefix_len: float) -> Dict[str, float]:
    candidates_per_group = k + 1
    suffix_len = max(0.0, avg_real_len - avg_prefix_len)
    plain_work = num_groups * candidates_per_group * avg_real_len
    grouped_work = num_groups * (avg_prefix_len + candidates_per_group * suffix_len)
    return {
        "plain_token_work": plain_work,
        "grouped_token_work": grouped_work,
        "theoretical_token_work_speedup": plain_work / max(grouped_work, 1e-12),
        "theoretical_saved_prefix_token_fraction": 1.0 - grouped_work / max(plain_work, 1e-12),
    }


def benchmark_case(
    texts,
    tokenizer,
    model,
    core_model,
    args,
    ratio: float,
    k: int,
    input_device: torch.device,
    input_dtype: torch.dtype,
    epsilon_private: float,
) -> Dict[str, Any]:
    synthetic_build_seconds = 0.0
    if args.benchmark_mode == "synthetic":
        started = time.perf_counter()
        batches = build_synthetic_batches(
            num_groups=args.max_samples,
            k=k,
            ratio=ratio,
            tokenizer=tokenizer,
            model=model,
            input_device=input_device,
            input_dtype=input_dtype,
            seq_len=int(args.synthetic_seq_len or args.max_length),
            seed=args.seed,
            vocab_start=args.synthetic_vocab_start,
            vocab_limit=args.synthetic_vocab_limit,
        )
        synthetic_build_seconds = time.perf_counter() - started
        build_seconds = 0.0
    else:
        dataset, build_seconds = build_dataset(texts, tokenizer, model, args, ratio, k, epsilon_private)
        batches = build_batches(dataset, k, input_device, input_dtype)
    stats = batch_stats(batches, int(args.synthetic_seq_len or args.max_length))

    plain = timed_run(core_model, batches, args.repeat_steps, args.warmup_steps, input_device, grouped=False)
    grouped = timed_run(core_model, batches, args.repeat_steps, args.warmup_steps, input_device, grouped=True)

    plain_profile: Dict[str, Any] = {}
    grouped_profile: Dict[str, Any] = {}
    if args.profile:
        plain_profile = {f"plain_{k}": v for k, v in profile_once(core_model, batches, input_device, grouped=False, top_ops=args.profile_top_ops).items()}
        grouped_profile = {f"grouped_{k}": v for k, v in profile_once(core_model, batches, input_device, grouped=True, top_ops=args.profile_top_ops).items()}

    module_plain_profile: Dict[str, Any] = {}
    module_grouped_profile: Dict[str, Any] = {}
    if args.module_profile:
        module_plain_profile = {f"plain_{k}": v for k, v in module_profile_once(core_model, batches, input_device, grouped=False).items()}
        module_grouped_profile = {f"grouped_{k}": v for k, v in module_profile_once(core_model, batches, input_device, grouped=True).items()}

    total_candidates = sum(int(batch["attention_mask"].size(0)) for batch in batches)
    num_groups = len(batches)
    work = token_work_bound(num_groups, k, stats["avg_real_len"], stats["avg_prefix_len"])
    server_speedup = plain["elapsed_seconds"] / max(grouped["elapsed_seconds"], 1e-12)
    end_to_end_speedup = (build_seconds + plain["elapsed_seconds"]) / max(build_seconds + grouped["elapsed_seconds"], 1e-12)

    row: Dict[str, Any] = {
        "benchmark_mode": args.benchmark_mode,
        "client_time_included_in_main_speedup": False,
        "dummy_ratio": ratio,
        "k": k,
        "epsilon_non_sensitive": args.epsilon_non_sensitive,
        "epsilon_private": epsilon_private,
        "num_groups": num_groups,
        "candidates_per_group": k + 1,
        "total_candidates": total_candidates,
        "client_dummy_build_seconds": build_seconds,
        "synthetic_batch_build_seconds": synthetic_build_seconds,
        "dummy_generation_seconds_per_group": build_seconds / max(num_groups, 1),
        "dummy_generation_seconds_per_dummy": build_seconds / max(num_groups * max(k, 1), 1),
        "server_plain_elapsed_seconds": plain["elapsed_seconds"],
        "server_grouped_elapsed_seconds": grouped["elapsed_seconds"],
        "server_only_speedup": server_speedup,
        "server_plain_candidates_per_second": plain["candidates_per_second"],
        "server_grouped_candidates_per_second": grouped["candidates_per_second"],
        "server_plain_queries_per_second": plain["queries_per_second"],
        "server_grouped_queries_per_second": grouped["queries_per_second"],
        "plain_elapsed_seconds": plain["elapsed_seconds"],
        "grouped_elapsed_seconds": grouped["elapsed_seconds"],
        "plain_candidates_per_second": plain["candidates_per_second"],
        "grouped_candidates_per_second": grouped["candidates_per_second"],
        "plain_queries_per_second": plain["queries_per_second"],
        "grouped_queries_per_second": grouped["queries_per_second"],
        "server_speedup": server_speedup,
        "end_to_end_speedup_including_dummy_generation": end_to_end_speedup,
        "plain_peak_memory_gb": plain["peak_memory_gb"],
        "grouped_peak_memory_gb": grouped["peak_memory_gb"],
    }
    row.update(stats)
    row.update(work)
    row.update(plain_profile)
    row.update(grouped_profile)
    row.update(module_plain_profile)
    row.update(module_grouped_profile)
    if args.profile:
        row["profile_attention_speedup"] = row.get("plain_profile_attention_self_time_ms", 0.0) / max(
            row.get("grouped_profile_attention_self_time_ms", 0.0), 1e-12
        )
        row["profile_linear_speedup"] = row.get("plain_profile_linear_self_time_ms", 0.0) / max(
            row.get("grouped_profile_linear_self_time_ms", 0.0), 1e-12
        )
    if args.module_profile:
        row["module_attention_speedup"] = row.get("plain_module_attention_time_ms", 0.0) / max(
            row.get("grouped_module_attention_time_ms", 0.0), 1e-12
        )
        row["module_mlp_speedup"] = row.get("plain_module_mlp_time_ms", 0.0) / max(
            row.get("grouped_module_mlp_time_ms", 0.0), 1e-12
        )
        row["module_linear_speedup"] = row.get("plain_module_linear_time_ms", 0.0) / max(
            row.get("grouped_module_linear_time_ms", 0.0), 1e-12
        )
        row["plain_mha_inclusive_time_ms"] = row.get("plain_module_attention_time_ms", 0.0)
        row["grouped_mha_inclusive_time_ms"] = row.get("grouped_module_attention_time_ms", 0.0)
        row["mha_inclusive_speedup"] = row.get("module_attention_speedup", 0.0)
        row["plain_mlp_inclusive_time_ms"] = row.get("plain_module_mlp_time_ms", 0.0)
        row["grouped_mlp_inclusive_time_ms"] = row.get("grouped_module_mlp_time_ms", 0.0)
        row["mlp_inclusive_speedup"] = row.get("module_mlp_speedup", 0.0)
        row["plain_linear_time_ms"] = row.get("plain_module_linear_time_ms", 0.0)
        row["grouped_linear_time_ms"] = row.get("grouped_module_linear_time_ms", 0.0)
        row["linear_speedup"] = row.get("module_linear_speedup", 0.0)
    return row


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


SERVER_ONLY_FIELDS = [
    "benchmark_mode",
    "dummy_ratio",
    "k",
    "num_groups",
    "candidates_per_group",
    "total_candidates",
    "avg_real_len",
    "avg_prefix_len",
    "avg_suffix_len",
    "avg_prefix_fraction",
    "server_plain_elapsed_seconds",
    "server_grouped_elapsed_seconds",
    "server_only_speedup",
    "server_plain_queries_per_second",
    "server_grouped_queries_per_second",
    "theoretical_token_work_speedup",
    "theoretical_saved_prefix_token_fraction",
    "plain_mha_inclusive_time_ms",
    "grouped_mha_inclusive_time_ms",
    "mha_inclusive_speedup",
    "plain_mlp_inclusive_time_ms",
    "grouped_mlp_inclusive_time_ms",
    "mlp_inclusive_speedup",
    "plain_linear_time_ms",
    "grouped_linear_time_ms",
    "linear_speedup",
    "plain_peak_memory_gb",
    "grouped_peak_memory_gb",
]


def write_server_only_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field for field in SERVER_ONLY_FIELDS if any(field in row for row in rows)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fmt_float(value: Any, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_server_only_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    headers = [
        "ratio",
        "k",
        "prefix%",
        "plain s",
        "grouped s",
        "server x",
        "theory x",
        "MHA x",
        "MLP x",
        "Linear x",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        prefix_pct = 100.0 * float(row.get("avg_prefix_fraction", 0.0) or 0.0)
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt_float(row.get("dummy_ratio"), 1),
                    str(row.get("k", "")),
                    fmt_float(prefix_pct, 1),
                    fmt_float(row.get("server_plain_elapsed_seconds"), 2),
                    fmt_float(row.get("server_grouped_elapsed_seconds"), 2),
                    fmt_float(row.get("server_only_speedup"), 2),
                    fmt_float(row.get("theoretical_token_work_speedup"), 2),
                    fmt_float(row.get("mha_inclusive_speedup"), 2),
                    fmt_float(row.get("mlp_inclusive_speedup"), 2),
                    fmt_float(row.get("linear_speedup"), 2),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = get_parser().parse_args()
    ks = parse_csv_list(args.ks, int)
    ratios = parse_csv_list(args.dummy_ratios, float)
    epsilon_map = parse_k_float_map(args.epsilon_private_map)

    if args.dry_run:
        print("[Dry Run]")
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        print(f"ks={ks} ratios={ratios} epsilon_private_map={epsilon_map}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{args.output_prefix}_config.json"
    summary_path = output_dir / f"{args.output_prefix}_summary.csv"
    server_only_csv_path = output_dir / f"{args.output_prefix}_server_only.csv"
    server_only_md_path = output_dir / f"{args.output_prefix}_server_only.md"

    texts = load_texts(args) if args.benchmark_mode == "real" else []
    tokenizer, model, load_dtype = load_local_causal_lm(args)
    core_model = transformer_core(model).eval()
    input_device = resolve_model_input_device(model, args.device)
    input_dtype = resolve_model_input_dtype(model, load_dtype)
    print(
        f"[Model] {args.model_path} | dtype={load_dtype} | device_map={args.device_map} "
        f"| input_device={input_device} | input_dtype={input_dtype} "
        f"| mode={args.benchmark_mode} | groups={len(texts) if texts else args.max_samples}"
    )
    write_json(
        config_path,
        {
            "args": vars(args),
            "ks": ks,
            "dummy_ratios": ratios,
            "epsilon_private_map": epsilon_map,
            "load_dtype": str(load_dtype),
            "input_device": str(input_device),
            "input_dtype": str(input_dtype),
            "note": (
                "Speed benchmark compares plain transformer forward against grouped prefix-KV-cache reuse. "
                "real mode builds DP-SPD continuation dummies; synthetic mode directly constructs shared-prefix "
                "embedding groups and isolates server-side reuse. Utility/judge metrics are intentionally skipped."
            ),
        },
    )

    rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        for k in ks:
            epsilon_private = private_epsilon_for_k(k, epsilon_map, args.epsilon_private_default)
            print(f"[Case] ratio={ratio} k={k} eps_private={epsilon_private}")
            row = benchmark_case(
                texts,
                tokenizer,
                model,
                core_model,
                args,
                ratio,
                k,
                input_device,
                input_dtype,
                epsilon_private,
            )
            rows.append(row)
            write_csv(summary_path, rows)
            write_server_only_csv(server_only_csv_path, rows)
            write_server_only_markdown(server_only_md_path, rows)
            attention_speedup = row.get("module_attention_speedup", row.get("profile_attention_speedup", 0.0))
            linear_speedup = row.get("module_linear_speedup", row.get("profile_linear_speedup", 0.0))
            print(
                f"[Done] ratio={ratio} k={k} "
                f"plain={row['plain_elapsed_seconds']:.3f}s grouped={row['grouped_elapsed_seconds']:.3f}s "
                f"server_speedup={row['server_speedup']:.2f}x "
                f"attention_speedup={attention_speedup:.2f}x "
                f"linear_speedup={linear_speedup:.2f}x"
            )

    print(f"[Config] {config_path}")
    print(f"[Summary] {summary_path}")
    print(f"[ServerOnlyCSV] {server_only_csv_path}")
    print(f"[ServerOnlyMD] {server_only_md_path}")


if __name__ == "__main__":
    main()
