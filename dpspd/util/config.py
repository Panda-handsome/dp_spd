# -*- coding: utf-8 -*-
# Shared metadata for model dimensions, datasets, and server-side KV-cache policy.

emb_size_dict = {
    "gpt2": 768,
    "gpt2-medium": 1024,
    "gpt2-large": 1280,
    "gpt2-xl": 1600,
    "facebook/opt-125m": 768,
    "facebook/opt-350m": 512,
    "facebook/opt-2.7b": 2560,
    "facebook/opt-6.7b": 4096,
    "meta-llama/llama-2-7b-hf": 4096,
    "meta-llama/llama-2-7b-chat-hf": 4096,
    "bert-base-uncased": 768,
    "bert-large-uncased": 1024,
    "t5-small": 512,
    "t5-base": 768,
    "t5-large": 1024,
}

model_name_aliases = {
    "llama-2-7b": "meta-llama/llama-2-7b-hf",
    "llama-2-7b-chat": "meta-llama/llama-2-7b-chat-hf",
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "opt-2.7b": "facebook/opt-2.7b",
    "opt-6.7b": "facebook/opt-6.7b",
}

def normalize_model_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    key = name.strip().lower()
    if key in model_name_aliases:
        return model_name_aliases[key]
    return key

def get_emb_size(model_name: str, default: int = 768) -> int:
    key = normalize_model_name(model_name)
    return emb_size_dict.get(key, default)

dataset_meta = {
    "imdb": {
        "task_type": "classification",
        "dataset": "data/imdb",
        "subset": None,
        "key": ["text"],
        "label": "label",
        "splits": {"train": "train", "test": "test"},
    },
    "ag_news": {
        "task_type": "classification",
        "dataset": "ag_news",
        "subset": None,
        "key": ["text"],
        "label": "label",
        "splits": {"train": "train", "test": "test"},
    },
    "glue_sst2": {
        "task_type": "classification",
        "dataset": "data/glue/sst2",
        "subset": None,
        "key": ["sentence"],
        "label": "label",
        "splits": {"train": "train", "test": "test"},
    },
    "glue_rte": {
        "task_type": "matching",
        "dataset": "glue",
        "subset": "rte",
        "key": ["sentence1", "sentence2"],
        "label": "label",
        "splits": {"train": "train", "test": "validation"},
    },
    "glue_qnli": {
        "task_type": "matching",
        "dataset": "data/glue/qnli",
        "subset": None,
        "key": ["question", "sentence"],
        "label": "label",
        "splits": {"train": "train", "test": "validation"},
    },
    "glue_stsb": {
        "task_type": "regression",
        "dataset": "glue",
        "subset": "stsb",
        "key": ["sentence1", "sentence2"],
        "label": "label",
        "splits": {"train": "train", "test": "validation"},
    },
    "wikitext": {
        "task_type": "generation",
        "dataset": "data/wikitext/wikitext-2-v1",
        "subset": None,
        "key": ["text"],
        "splits": {"train": "train", "test": "test"},
    },
    "alpaca": {
        "task_type": "generation",
        "dataset": "data/alpaca",
        "subset": None,
        "key": ["instruction", "input", "output"],
        "splits": {"train": "train", "test": "train"},
    },
    "databricks_dolly": {
        "task_type": "generation",
        "dataset": "data/databricks_dolly",
        "subset": None,
        "key": ["instruction", "context", "response"],
        "splits": {"train": "train", "test": "train"},
    },
    "cnn_dailymail": {
        "task_type": "generation",
        "dataset": "data/cnn_dailymail",
        "subset": None,
        "key": ["article", "highlights"],
        "splits": {"train": "train", "test": "test"},
    },
    "daily_dialog": {
        "task_type": "generation",
        "dataset": "daily_dialog",
        "subset": None,
        "key": ["dialog"],
        "splits": {"train": "train", "test": "test"},
    },
}

kv_cache_cfg = {
    "enable": True,
    "group_by_prefix_hash": True,
    "prefix_len_tokens": 16,
    "hash_base": 131,
    "use_prefix_length": True,
    "max_cache_steps": 1024,
    "cross_sample_reuse": True,
}
