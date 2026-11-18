# -*- coding: utf-8 -*-
"""Run the non-private original upper-bound on curated WildChat subsets.

This script can either call an OpenAI-compatible chat endpoint or directly load
a local Hugging Face causal LM. It stores raw model outputs plus lightweight
automatic metrics. For QA, semantic accuracy is optionally estimated by an LLM
judge because the curated WildChat QA subset has reference answers rather than
single gold labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_max_memory_spec(spec: Optional[str]) -> Optional[Dict[Any, str]]:
    """Parse a compact max_memory spec for Transformers device_map.

    Examples:
      all:44GiB
      0:44GiB,1:44GiB,cpu:200GiB
    """
    if not spec:
        return None
    text = spec.strip()
    if not text:
        return None
    if text.startswith("all:"):
        value = text.split(":", 1)[1].strip()
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for --local_max_memory all:...") from exc
        return {idx: value for idx in range(torch.cuda.device_count())}
    max_memory: Dict[Any, str] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key.lower() == "cpu":
            max_memory["cpu"] = value
        else:
            max_memory[int(key)] = value
    return max_memory


DEFAULT_QWEN3_32B_PATH = "models/Qwen3-32B"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WildChat original upper-bound runner.")
    parser.add_argument("--qa_jsonl", type=str, default="data/wildchat/wildchat_qa500_independent_userprivacy.jsonl")
    parser.add_argument("--open_jsonl", type=str, default="data/wildchat/wildchat_open500_independent_userprivacy.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/wildchat_original_upper_bound")
    parser.add_argument("--output_prefix", type=str, default="qwen3_32b_plaintext_wildchat")

    parser.add_argument("--backend", type=str, default="local_hf", choices=["local_hf", "openai_chat", "dry_run"])
    parser.add_argument("--api_model", type=str, default="qwen3-32b")
    parser.add_argument("--openai_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--timeout_seconds", type=float, default=300.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep_seconds", type=float, default=5.0)
    parser.add_argument("--inter_request_sleep_seconds", type=float, default=0.0)
    parser.add_argument("--local_model_path", type=str, default=DEFAULT_QWEN3_32B_PATH)
    parser.add_argument("--local_dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--local_device_map", type=str, default="auto", help="Use auto for multi-GPU loading.")
    parser.add_argument("--local_device", type=str, default="cuda:0", help="Used only when --local_device_map none.")
    parser.add_argument(
        "--local_max_memory",
        type=str,
        default=None,
        help='Optional Transformers max_memory, e.g. "all:44GiB" or "0:44GiB,1:44GiB,cpu:200GiB".',
    )
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--trust_remote_code", type=str2bool, default=False)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--enable_thinking", type=str2bool, default=False)

    parser.add_argument("--max_samples_qa", type=int, default=500)
    parser.add_argument("--max_samples_open", type=int, default=500)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_mode", type=str, default="first", choices=["first", "shuffle"])
    parser.add_argument("--max_source_chars", type=int, default=2400)
    parser.add_argument("--use_messages", type=str2bool, default=True)
    parser.add_argument("--prompt_style", type=str, default="raw", choices=["raw", "instruct"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_completion_tokens_qa", type=int, default=256)
    parser.add_argument("--max_completion_tokens_open", type=int, default=512)

    parser.add_argument("--run_qa_judge", type=str2bool, default=False)
    parser.add_argument(
        "--judge_existing_only",
        type=str2bool,
        default=False,
        help="Only add QA judge results to an existing outputs JSONL, without regenerating model responses.",
    )
    parser.add_argument("--judge_backend", type=str, default=None, choices=["local_hf", "openai_chat", "dry_run"])
    parser.add_argument("--judge_api_model", type=str, default=None)
    parser.add_argument("--judge_openai_base_url", type=str, default=None)
    parser.add_argument("--judge_api_key_env", type=str, default=None)
    parser.add_argument("--judge_local_model_path", type=str, default=None)
    parser.add_argument("--judge_max_completion_tokens", type=int, default=160)

    parser.add_argument("--input_price_per_mtok", type=float, default=None)
    parser.add_argument("--output_price_per_mtok", type=float, default=None)
    parser.add_argument("--resume", type=str2bool, default=True)
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


def select_rows(rows: Sequence[Dict[str, Any]], max_samples: int, mode: str, seed: int) -> List[Dict[str, Any]]:
    if max_samples <= 0 or max_samples >= len(rows):
        return list(rows)
    if mode == "first":
        return list(rows[:max_samples])
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[idx] for idx in indices[:max_samples]]


def build_generation_prompt(user_text: str, subset: str, prompt_style: str = "raw") -> str:
    if prompt_style == "raw":
        return user_text
    if subset == "answerable_qa":
        return (
            "Answer the following question. Keep the answer concise and factual. "
            "If the question includes answer choices, give the selected choice first.\n\n"
            f"Question:\n{user_text}\n\nAnswer:"
        )
    return (
        "Respond to the following user request. Be helpful, specific, and concise. "
        "Do not include hidden reasoning or analysis notes.\n\n"
        f"User request:\n{user_text}\n\nResponse:"
    )


def normalize_for_rouge(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if token_a == token_b else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_for_rouge(prediction).split()
    ref_tokens = normalize_for_rouge(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class ChatClient:
    def __init__(
        self,
        backend: str,
        api_model: str,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float,
        max_retries: int,
        retry_sleep_seconds: float,
        local_model_path: Optional[str] = None,
        local_dtype: str = "bfloat16",
        local_device_map: str = "auto",
        local_device: str = "cuda:0",
        local_max_memory: Optional[str] = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
        max_input_tokens: int = 4096,
        enable_thinking: bool = False,
    ) -> None:
        self.backend = backend
        self.api_model = api_model
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.max_input_tokens = max_input_tokens
        self.enable_thinking = enable_thinking
        self.client = None
        self.local_model = None
        self.local_tokenizer = None
        if backend == "openai_chat":
            self.client = self._build_openai_client(base_url, api_key_env, timeout_seconds)
        elif backend == "local_hf":
            if not local_model_path:
                raise ValueError("--local_model_path is required for --backend local_hf.")
            self._load_local_hf_model(
                local_model_path=local_model_path,
                local_dtype=local_dtype,
                local_device_map=local_device_map,
                local_device=local_device,
                local_max_memory=local_max_memory,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

    def _build_openai_client(self, base_url: str, api_key_env: str, timeout_seconds: float):
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai and httpx packages are required for --backend openai_chat.") from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            if "127.0.0.1" in base_url or "localhost" in base_url:
                api_key = "EMPTY"
            else:
                raise ValueError(f"Environment variable {api_key_env} is not set.")
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
            max_retries=0,
        )

    def _resolve_torch_dtype(self, local_dtype: str):
        if local_dtype == "auto":
            return "auto"
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for --backend local_hf.") from exc
        if local_dtype == "bfloat16":
            return torch.bfloat16
        if local_dtype == "float16":
            return torch.float16
        if local_dtype == "float32":
            return torch.float32
        raise ValueError(f"Unsupported local dtype: {local_dtype}")

    def _load_local_hf_model(
        self,
        local_model_path: str,
        local_dtype: str,
        local_device_map: str,
        local_device: str,
        local_max_memory: Optional[str],
        local_files_only: bool,
        trust_remote_code: bool,
        attn_implementation: Optional[str],
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for --backend local_hf.") from exc

        dtype = self._resolve_torch_dtype(local_dtype)
        self.local_tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if self.local_tokenizer.pad_token is None:
            self.local_tokenizer.pad_token = self.local_tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = "auto"
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if local_device_map.lower() != "none":
            model_kwargs["device_map"] = local_device_map
            parsed_max_memory = parse_max_memory_spec(local_max_memory)
            if parsed_max_memory:
                model_kwargs["max_memory"] = parsed_max_memory

        print(
            f"[LocalHF] Loading model={local_model_path} dtype={local_dtype} "
            f"device_map={local_device_map} max_memory={model_kwargs.get('max_memory')} "
            f"local_files_only={local_files_only}"
        )
        self.local_model = AutoModelForCausalLM.from_pretrained(local_model_path, **model_kwargs)
        if local_device_map.lower() == "none":
            self.local_model.to(torch.device(local_device))
        self.local_model.eval()

        device_map = getattr(self.local_model, "hf_device_map", None)
        if device_map:
            print(f"[LocalHF] hf_device_map={device_map}")

    def chat(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str = "You are a careful assistant. Follow the user task and answer directly.",
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, Dict[str, Optional[int]]]:
        if self.backend == "dry_run":
            return "", {}
        if self.backend == "local_hf":
            return self._chat_local_hf(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                messages=messages,
            )
        assert self.client is not None
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                request_messages = (
                    [{"role": "system", "content": system_prompt}] + messages
                    if messages
                    else [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                )
                response = self.client.chat.completions.create(
                    model=self.api_model,
                    messages=request_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                usage = getattr(response, "usage", None)
                usage_record = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
                    "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
                    "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
                }
                return response.choices[0].message.content or "", usage_record
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self.retry_sleep_seconds * attempt
                print(f"[Retry] {type(exc).__name__}: {exc}. Sleeping {delay:.1f}s...")
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    def _first_model_device(self):
        assert self.local_model is not None
        try:
            return self.local_model.device
        except AttributeError:
            pass
        for param in self.local_model.parameters():
            if str(param.device) != "meta":
                return param.device
        raise RuntimeError("Could not infer local model device.")

    def _first_model_dtype(self):
        assert self.local_model is not None
        # Device maps may leave meta tensors on large models; use the first real
        # parameter to choose the dtype for external inputs_embeds.
        for param in self.local_model.parameters():
            if str(param.device) != "meta":
                return param.dtype
        raise RuntimeError("Could not infer local model dtype.")

    def _format_chat_prompt(
        self,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        assert self.local_tokenizer is not None
        if messages:
            chat_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            chat_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        if hasattr(self.local_tokenizer, "apply_chat_template") and self.local_tokenizer.chat_template:
            try:
                return self.local_tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                return self.local_tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return f"{system_prompt}\n\n{prompt}"

    def _chat_local_hf(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, Dict[str, Optional[int]]]:
        assert self.local_model is not None and self.local_tokenizer is not None
        import torch

        formatted_prompt = self._format_chat_prompt(prompt, system_prompt, messages=messages)
        enc = self.local_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        input_len = int(enc["input_ids"].shape[1])
        device = self._first_model_device()
        enc = {key: value.to(device) for key, value in enc.items()}
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.local_tokenizer.pad_token_id,
            "eos_token_id": self.local_tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            generated = self.local_model.generate(**enc, **gen_kwargs)
        new_tokens = generated[:, input_len:]
        text = self.local_tokenizer.batch_decode(new_tokens.detach().cpu(), skip_special_tokens=True)[0].strip()
        usage = {
            "prompt_tokens": input_len,
            "completion_tokens": int(new_tokens.shape[1]),
            "total_tokens": input_len + int(new_tokens.shape[1]),
        }
        return text, usage

    def generate_from_embeddings(
        self,
        inputs_embeds,
        attention_mask,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, Dict[str, Optional[int]]]:
        if self.backend != "local_hf":
            raise ValueError("Embedding transport is only supported with --backend local_hf.")
        assert self.local_model is not None and self.local_tokenizer is not None
        import torch

        # This path mirrors the earlier DP-SPD client/server contract: the
        # server receives already-noised embeddings instead of token ids.
        device = self._first_model_device()
        dtype = self._first_model_dtype()
        inputs_embeds = inputs_embeds.to(device=device, dtype=dtype)
        attention_mask = attention_mask.to(device=device)
        prompt_tokens = int(attention_mask.sum().item())
        gen_kwargs: Dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "pad_token_id": self.local_tokenizer.pad_token_id,
            "eos_token_id": self.local_tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            generated = self.local_model.generate(**gen_kwargs)
        # Transformers may return only generated ids for inputs_embeds, while
        # token-id generation returns prompt+completion. Taking the tail matches
        # the old evaluation scripts and keeps decoding focused on completion.
        new_tokens = generated[:, -max_tokens:] if generated.shape[1] > max_tokens else generated
        text = self.local_tokenizer.batch_decode(new_tokens.detach().cpu(), skip_special_tokens=True)[0].strip()
        completion_tokens = int(new_tokens.shape[1])
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return text, usage


def build_qa_judge_prompt(question: str, reference: str, response: str) -> str:
    return (
        "Judge whether the model answer is semantically correct for the question.\n"
        "Accept paraphrases and equivalent concise answers. For multiple-choice questions, "
        "the answer is correct if it selects the same option or gives the same option meaning. "
        "Return only compact JSON with keys correct and reason.\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Model answer:\n{response}\n\n"
        'JSON: {"correct": true/false, "reason": "..."}'
    )


def parse_judge_correct(text: str) -> Optional[bool]:
    raw = text.strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        data = json.loads(raw)
        value = data.get("correct")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "correct", "1"}:
                return True
            if lowered in {"false", "no", "incorrect", "0"}:
                return False
    except json.JSONDecodeError:
        pass
    lowered = text.lower()
    if re.search(r"\b(correct|yes|true)\b", lowered) and not re.search(r"\b(incorrect|not correct|false)\b", lowered):
        return True
    if re.search(r"\b(incorrect|not correct|false|no)\b", lowered):
        return False
    return None


def load_existing_keys(path: Path) -> set[Tuple[str, str]]:
    keys: set[Tuple[str, str]] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.add((str(row.get("subset")), str(row.get("id"))))
    return keys


def iter_examples(args) -> Iterable[Tuple[str, Dict[str, Any]]]:
    qa_rows = select_rows(read_jsonl(Path(args.qa_jsonl)), args.max_samples_qa, args.sample_mode, args.sample_seed)
    open_rows = select_rows(
        read_jsonl(Path(args.open_jsonl)), args.max_samples_open, args.sample_mode, args.sample_seed + 17
    )
    for row in qa_rows:
        yield "answerable_qa", row
    for row in open_rows:
        yield "open_ended", row


def normalize_chat_messages(
    row: Dict[str, Any],
    fallback_user_text: str,
    subset: str,
    args,
) -> Optional[List[Dict[str, str]]]:
    if not args.use_messages:
        return None
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return None
    messages: List[Dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = trim_text(item.get("content"), args.max_source_chars)
        if content:
            messages.append({"role": role, "content": content})
    if not messages:
        return None
    if messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": fallback_user_text})
    if args.prompt_style == "instruct":
        messages[-1]["content"] = build_generation_prompt(messages[-1]["content"], subset, "instruct")
    return messages


def serialize_api_messages(messages: Optional[Sequence[Dict[str, str]]], fallback_prompt: str) -> str:
    if not messages:
        return fallback_prompt
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def write_config(path: Path, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "note": "Original upper-bound: no DP noise, no projection, no dummies."}, f, indent=2)


def summarize(outputs_path: Path, summary_path: Path, args) -> None:
    rows: List[Dict[str, Any]] = []
    if outputs_path.exists():
        with open(outputs_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

    fields = [
        "subset",
        "group",
        "samples",
        "rouge_l",
        "qa_judge_accuracy",
        "qa_judge_coverage",
        "avg_latency_sec",
        "avg_prompt_chars",
        "avg_response_chars",
        "total_prompt_tokens",
        "total_completion_tokens",
        "estimated_cost_usd",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for subset in ["answerable_qa", "open_ended", "all"]:
            subset_rows = rows if subset == "all" else [row for row in rows if row.get("subset") == subset]
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
                judge_values = [
                    bool(row.get("qa_judge_correct"))
                    for row in group_rows
                    if row.get("qa_judge_correct") is not None
                ]
                latencies = [float(row.get("latency_sec", 0.0) or 0.0) for row in group_rows]
                prompt_chars = [len(str(row.get("api_prompt", ""))) for row in group_rows]
                response_chars = [len(str(row.get("response", ""))) for row in group_rows]
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
                        "subset": subset,
                        "group": group,
                        "samples": len(group_rows),
                        "rouge_l": f"{sum(rouge_values) / len(rouge_values):.6f}",
                        "qa_judge_accuracy": (
                            f"{sum(1 for value in judge_values if value) / len(judge_values):.6f}"
                            if judge_values
                            else ""
                        ),
                        "qa_judge_coverage": f"{len(judge_values) / len(group_rows):.6f}",
                        "avg_latency_sec": f"{sum(latencies) / len(latencies):.4f}" if latencies else "",
                        "avg_prompt_chars": f"{sum(prompt_chars) / len(prompt_chars):.2f}",
                        "avg_response_chars": f"{sum(response_chars) / len(response_chars):.2f}",
                        "total_prompt_tokens": total_prompt_tokens if prompt_tokens else "",
                        "total_completion_tokens": total_completion_tokens if completion_tokens else "",
                        "estimated_cost_usd": f"{estimated_cost:.6f}" if estimated_cost is not None else "",
                    }
                )


def add_judge_to_existing_outputs(
    outputs_path: Path,
    args,
    judge_runner: ChatClient,
) -> None:
    if not outputs_path.exists():
        raise FileNotFoundError(f"Cannot judge existing outputs because {outputs_path} does not exist.")
    rows: List[Dict[str, Any]] = []
    with open(outputs_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    changed = 0
    for row in rows:
        if row.get("subset") != "answerable_qa":
            continue
        if row.get("qa_judge_correct") is not None and row.get("qa_judge_raw"):
            continue
        judge_prompt = build_qa_judge_prompt(
            str(row.get("source_text", "")),
            str(row.get("reference", "")),
            str(row.get("response", "")),
        )
        judge_raw, judge_usage = judge_runner.chat(
            judge_prompt,
            max_tokens=args.judge_max_completion_tokens,
            temperature=0.0,
            system_prompt="You are a strict but fair evaluator. Return only valid JSON.",
        )
        row["qa_judge_raw"] = judge_raw
        row["qa_judge_correct"] = parse_judge_correct(judge_raw)
        row["judge_api_usage"] = judge_usage
        changed += 1
        print(f"[Judged] id={row.get('id')} correct={row.get('qa_judge_correct')}")
        if args.inter_request_sleep_seconds > 0:
            time.sleep(args.inter_request_sleep_seconds)

    tmp_path = outputs_path.with_suffix(outputs_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, outputs_path)
    print(f"[JudgeExisting] updated={changed} path={outputs_path}")


def main() -> None:
    args = get_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = output_dir / f"{args.output_prefix}_outputs.jsonl"
    summary_path = output_dir / f"{args.output_prefix}_summary.csv"
    config_path = output_dir / f"{args.output_prefix}_config.json"

    write_config(config_path, args)
    existing_keys = load_existing_keys(outputs_path) if args.resume else set()
    mode = "a" if args.resume else "w"

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
    judge_runner: Optional[ChatClient] = None
    if args.run_qa_judge:
        judge_backend = args.judge_backend or args.backend
        judge_model_path = args.judge_local_model_path or args.local_model_path
        same_local_model = (
            judge_backend == "local_hf"
            and args.backend == "local_hf"
            and str(judge_model_path) == str(args.local_model_path)
        )
        if same_local_model:
            judge_runner = runner
        else:
            judge_runner = ChatClient(
                backend=judge_backend,
                api_model=args.judge_api_model or args.api_model,
                base_url=args.judge_openai_base_url or args.openai_base_url,
                api_key_env=args.judge_api_key_env or args.api_key_env,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
                local_model_path=judge_model_path,
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

    if args.judge_existing_only:
        if judge_runner is None:
            raise ValueError("--judge_existing_only true requires --run_qa_judge true.")
        add_judge_to_existing_outputs(outputs_path, args, judge_runner)
        summarize(outputs_path, summary_path, args)
        print(f"[Outputs] {outputs_path}")
        print(f"[Summary] {summary_path}")
        print(f"[Config] {config_path}")
        return

    with open(outputs_path, mode, encoding="utf-8") as out_f:
        for subset, row in iter_examples(args):
            row_id = str(row.get("id"))
            if (subset, row_id) in existing_keys:
                print(f"[Skip] subset={subset} id={row_id}")
                continue
            user_text = trim_text(row.get("current_user_text") or row.get("text"), args.max_source_chars)
            reference = normalize_space(row.get("reference"))
            prompt = build_generation_prompt(user_text, subset, args.prompt_style)
            api_messages = normalize_chat_messages(row, user_text, subset, args)
            api_prompt = serialize_api_messages(api_messages, prompt)
            max_tokens = args.max_completion_tokens_qa if subset == "answerable_qa" else args.max_completion_tokens_open

            started = time.perf_counter()
            response, usage = runner.chat(
                prompt,
                max_tokens=max_tokens,
                temperature=args.temperature,
                messages=api_messages,
            )
            latency_sec = time.perf_counter() - started
            rouge_l = rouge_l_f1(response, reference) if reference else None

            judge_raw = None
            judge_correct = None
            judge_usage: Dict[str, Optional[int]] = {}
            if judge_runner is not None and subset == "answerable_qa":
                judge_prompt = build_qa_judge_prompt(user_text, reference, response)
                judge_raw, judge_usage = judge_runner.chat(
                    judge_prompt,
                    max_tokens=args.judge_max_completion_tokens,
                    temperature=0.0,
                    system_prompt="You are a strict but fair evaluator. Return only valid JSON.",
                )
                judge_correct = parse_judge_correct(judge_raw)

            record = {
                "id": row_id,
                "subset": subset,
                "mix_source_subset": row.get("mix_source_subset") or row.get("subset"),
                "privacy_signal": bool(row.get("privacy_signal")),
                "metadata": row.get("metadata", {}),
                "source_text": user_text,
                "context_text": row.get("context_text"),
                "reference": reference,
                "api_prompt": api_prompt,
                "api_messages": api_messages,
                "response": response,
                "rouge_l": rouge_l,
                "qa_judge_correct": judge_correct,
                "qa_judge_raw": judge_raw,
                "api_usage": usage,
                "judge_api_usage": judge_usage,
                "latency_sec": latency_sec,
                "backend": args.backend,
                "api_model": args.api_model,
                "local_model_path": args.local_model_path if args.backend == "local_hf" else None,
                "variant": "original",
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"[Saved] subset={subset} id={row_id} latency={latency_sec:.2f}s rouge_l={rouge_l}")
            if args.inter_request_sleep_seconds > 0:
                time.sleep(args.inter_request_sleep_seconds)

    summarize(outputs_path, summary_path, args)
    print(f"[Outputs] {outputs_path}")
    print(f"[Summary] {summary_path}")
    print(f"[Config] {config_path}")


if __name__ == "__main__":
    main()
