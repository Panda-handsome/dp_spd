# -*- coding: utf-8 -*-
# privacy_utils.py

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from torch.distributions.gamma import Gamma
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification, AutoTokenizer,
    AutoModel, GPT2Tokenizer, GPT2Model,
    OPTForSequenceClassification,
    AutoModelForCausalLM, BertModel,
    T5Model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpspd.util.config import dataset_meta

# Noise samplers.
def sample_noise_laplace(shape, epsilon: float, device: str = "cpu"):
    if epsilon is None or epsilon <= 0:
        return torch.zeros(shape, device=device)
    scale = 1.0 / max(epsilon, 1e-8)
    dist = torch.distributions.Laplace(loc=0.0, scale=scale)
    return dist.sample(shape).to(device)

def sample_noise_gaussian(shape, std: float, device: str = "cpu"):
    if std is None or std <= 0:
        return torch.zeros(shape, device=device)
    return torch.normal(mean=0.0, std=std, size=shape, device=device)

def sample_noise_chi(shape, eta: float, device: str = "cpu"):
    n_dim = shape[-1]
    alpha = torch.ones(shape) * n_dim
    beta = torch.ones(shape) * max(eta, 1e-8)
    m = Gamma(alpha, beta)
    l_lst = m.sample().to(device)
    v_lst = (torch.rand(shape, device=device) * 2.0 - 1.0)
    return l_lst * v_lst

# Embedding helpers.
def get_embeddings(
    attention_mask,
    model,
    args,
    decoder_start_token=None,
    clean_embeddings=None,
    noisy_embeddings=None
):
    """
    Extract CLS-style vectors from client-provided clean or noisy embeddings.
    The server does not inject additional noise.
    """
    source_embeddings = clean_embeddings if clean_embeddings is not None else noisy_embeddings
    if source_embeddings is None:
        raise ValueError("Either clean_embeddings or noisy_embeddings must be provided.")
    batch_size = source_embeddings.shape[0]

    # T5 decoder start token
    is_t5 = "t5" in args.base_model.lower()
    if is_t5:
        if decoder_start_token is None:
            start_id = getattr(model.config, "decoder_start_token_id", None)
            if start_id is None:
                raise ValueError("decoder_start_token_id is required for T5")
            decoder_start_token = torch.full((batch_size, 1), start_id,
                                             device=attention_mask.device,
                                             dtype=torch.long)
        else:
            decoder_start_token = decoder_start_token.repeat(batch_size, 1)

    def forward_emb(inputs_embeds):
        if is_t5:
            outputs = model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_start_token,
                            output_hidden_states=True)
        else:
            outputs = model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        return get_cls_embedding(outputs, attention_mask, args)

    clean_cls = forward_emb(clean_embeddings) if clean_embeddings is not None else None
    noisy_cls = forward_emb(noisy_embeddings) if noisy_embeddings is not None else None
    return {"clean_cls": clean_cls, "noisy_cls": noisy_cls}

# ----------------------------
# Argparse helpers
# ----------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    if text in ('yes', 'true', 't', 'y', '1'):
        return True
    if text in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def str2type(v):
    if isinstance(v, torch.dtype):
        return v
    if "float32" in v.lower():
        return torch.float32
    elif "float16" in v.lower():
        return torch.float16
    elif "bfloat16" in v.lower():
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype string: {v}")

def resolve_dtype(dtype_name, fallback: torch.dtype = torch.float16):
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    if dtype_name is None:
        return fallback

    name = str(dtype_name).replace("torch.", "").lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if name == "auto":
        return fallback
    raise ValueError(f"Unsupported dtype string: {dtype_name}")

def get_model_load_dtype(model_path: str,
                         device: str = "cpu",
                         dtype_preference: str = "auto",
                         local_files_only: bool = True) -> torch.dtype:
    """
    Choose a stable loading dtype for local causal LMs.
    - CPU defaults to float32 for compatibility.
    - GPU defaults to the model config dtype when available.
    """
    device_str = str(device)
    default_dtype = torch.float32 if device_str == "cpu" else torch.float16

    if dtype_preference != "auto":
        return resolve_dtype(dtype_preference, fallback=default_dtype)

    if device_str == "cpu":
        return torch.float32

    try:
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
        cfg_dict = cfg.to_dict()
        return resolve_dtype(cfg_dict.get("torch_dtype"), fallback=default_dtype)
    except Exception:
        return default_dtype

def load_local_causal_lm(model_path: str,
                         device: str = "cpu",
                         dtype_preference: str = "auto",
                         local_files_only: bool = True):
    """
    Load a local causal LM + tokenizer pair for DP-SPD generation scripts.
    """
    load_dtype = get_model_load_dtype(
        model_path=model_path,
        device=device,
        dtype_preference=dtype_preference,
        local_files_only=local_files_only,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        dtype=load_dtype,
    ).to(device)
    base_model.eval()

    return tokenizer, base_model, load_dtype

# CLS embedding extraction.
def get_cls_embedding(outputs, attention_mask, args):
    if args.base_model in ("bert-base-uncased", "bert-large-uncased"):
        hid_states = outputs.hidden_states
        cls_embs = hid_states[-1][:, 0, :]
    elif "gpt2" in args.base_model:
        hid_states = outputs.last_hidden_state
        sum_attention_mask = attention_mask.sum(dim=1)
        last_pad = torch.where(sum_attention_mask > 0,
                               sum_attention_mask - 1,
                               torch.zeros_like(sum_attention_mask))
        last_pad = last_pad.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hid_states.size(2))
        cls_embs = torch.gather(hid_states, index=last_pad, dim=1).squeeze()
    elif "t5" in args.base_model:
        hid_states = outputs.last_hidden_state
        cls_embs = hid_states[:, 0, :]
    elif any(model in args.base_model.lower() for model in ['opt', 'llama', 'qwen']):
        hid_states = outputs.hidden_states[-1]
        last_pad = attention_mask.sum(dim=1) - 1
        last_pad = last_pad.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, hid_states.shape[-1])
        cls_embs = torch.gather(hid_states, index=last_pad, dim=1).squeeze()
    else:
        raise ValueError(f"Unsupported base_model for CLS extraction: {args.base_model}")
    return cls_embs

# Token embedding extraction.
def get_token_embedding(token_id, model, args, squeeze=False):
    base_model_name = args.base_model.lower()
    with torch.no_grad():
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()(token_id)
        elif base_model_name in ("bert-base-uncased", "bert-large-uncased"):
            embeddings = model.embeddings.word_embeddings(token_id)
        elif 'gpt2' in base_model_name:
            embeddings = model.wte(token_id)
        elif 'opt' in base_model_name:
            try:
                embeddings = model.model.decoder.embed_tokens(token_id)
            except:
                embeddings = model.decoder.embed_tokens(token_id)
        elif 'llama' in base_model_name or 'qwen' in base_model_name:
            try:
                embeddings = model.model.embed_tokens(token_id)
            except:
                embeddings = model.embed_tokens(token_id)
        elif 't5' in base_model_name:
            embeddings = model.encoder.embed_tokens(token_id)
        else:
            raise ValueError(f"Unsupported base_model for token embedding: {args.base_model}")
        if squeeze:
            embeddings = embeddings.squeeze()
    return embeddings

# Closest token search.
def get_closest_token(embedding, tokenizer, model, args):
    vocab = getattr(tokenizer, "get_vocab", None)
    if callable(vocab):
        vocabulary = tokenizer.get_vocab()
    else:
        vocabulary = tokenizer.vocab
    token_ids = [token_id for _, token_id in vocabulary.items()]
    token_ids = torch.tensor(token_ids).to(args.device)
    word_embeddings = get_token_embedding(token_ids, model, args, squeeze=True)
    embedding = embedding.unsqueeze(dim=0).expand(word_embeddings.size())
    distance = torch.norm(embedding - word_embeddings, 2, dim=1)
    closest_idx = distance.argmin()
    closest_token = token_ids[closest_idx]
    return closest_token.item()

# Text-to-text privacy helper.
def text2text_priv(inputs, tokenizer, model, args):
    """
    Add noise to embeddings and replace each token with its nearest neighbor.
    This helper is used by local tools, not by the server.
    """
    input_ids = inputs['input_ids']
    attn_masks = inputs['attention_mask']
    init_embeddings = get_token_embedding(input_ids, model, args)

    shape = init_embeddings.shape
    if args.noise_mechanism == "Gaussian":
        noises = sample_noise_gaussian(shape, args.noise_std, device=args.device)
        noisy_embs = init_embeddings + noises
    elif args.noise_mechanism == "Laplace":
        noises = sample_noise_laplace(shape, args.epsilon, device=args.device)
        noisy_embs = init_embeddings + noises
    elif args.noise_mechanism == "ChiDP":
        noises = sample_noise_chi(shape, args.eta, device=args.device)
        noisy_embs = init_embeddings + noises
    else:
        raise ValueError(f"Unsupported noise mechanism: {args.noise_mechanism}")

    for i in tqdm(range(len(input_ids)), desc="Applying text2text privacy"):
        this_mask = attn_masks[i]
        this_embeds = noisy_embs[i][this_mask == 1]
        for j in range(len(this_embeds)):
            embed = this_embeds[j]
            closest_token = get_closest_token(embed, tokenizer, model, args)
            input_ids[i, j] = closest_token

    return {"input_ids": input_ids, 'attention_mask': attn_masks}


# Pretrained model loading.
def get_pretrained_model(args):
    """
    Load a pretrained model and tokenizer from a local path or model name.
    Supports BERT, GPT-2, OPT, LLaMA, Qwen, and T5 backbones.
    """
    import os

    base_model_name = args.base_model.lower()

    if os.path.isdir(args.base_model):
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model = AutoModel.from_pretrained(args.base_model)
        return tokenizer, base_model

    if base_model_name in ("bert-base-uncased", "bert-large-uncased"):
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model = BertModel.from_pretrained(args.base_model)

    elif 'gpt2' in base_model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = GPT2Model.from_pretrained(args.base_model)

    elif 'opt' in base_model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model = OPTForSequenceClassification.from_pretrained(args.base_model)

    elif 'llama' in base_model_name or 'qwen' in base_model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        if getattr(args, "llama_dir", None) is not None:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.llama_dir, torch_dtype=args.base_precision
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model, torch_dtype=args.base_precision
            )

    elif 't5' in base_model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model = T5Model.from_pretrained(args.base_model)

    else:
        raise ValueError(f"Unsupported base_model: {args.base_model}")

    return tokenizer, base_model

# Token distribution statistics.
def get_token_distribution(dataloader, tokenizer, vocabulary, args):
    """
    Count token frequencies by label.
    """
    token_ids = [token_id for _, token_id in vocabulary.items()]
    max_id = max(token_ids)
    freq_dict = {}
    for batch in tqdm(dataloader, desc="Computing token distribution"):
        input_ids = batch['input_ids']
        attention_masks = batch['attention_mask']
        labels = batch['labels']
        input_ids = input_ids.masked_fill(attention_masks == 0, 0)
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            this_input_ids = input_ids[labels == label]
            this_freq = torch.bincount(torch.flatten(this_input_ids), minlength=max_id)
            if label.item() in freq_dict.keys():
                freq_dict[label.item()] += this_freq
            else:
                freq_dict[label.item()] = this_freq
    return freq_dict


# UI value computation.
def get_ui_top(dataloader, tokenizer, args):
    """
    Compute UI values and return the top tokens for each label.
    """
    vocab = getattr(tokenizer, "get_vocab", None)
    if callable(vocab):
        vocabulary = tokenizer.get_vocab()
    else:
        vocabulary = tokenizer.vocab
    top_k = int(len(vocabulary) * 0.01)

    freq_dict = get_token_distribution(dataloader, tokenizer, vocabulary, args)
    esp = 1e-8
    ui_dict = {}
    for label1 in freq_dict:
        this_freq = freq_dict[label1]
        this_ui = torch.zeros_like(this_freq, dtype=torch.float)
        for label2 in freq_dict:
            if label2 != label1:
                other_freq = freq_dict[label2] + esp
                this_ui += torch.log((this_freq + esp) / other_freq)
        ui_dict[label1] = this_ui

    top_token_dict = {}
    for label in ui_dict:
        output = torch.topk(ui_dict[label], top_k)
        top_tokens = output.indices
        top_tokens = top_tokens[top_tokens != 0]
        top_token_dict[label] = top_tokens
    return top_token_dict


# Text-to-text privacy with UI filtering.
def text2text_priv_ui(inputs, labels, tokenizer, model, top_ui_dict, args):
    """
    Apply nearest-neighbor replacement only to tokens outside the label-specific UI set.
    """
    input_ids = inputs['input_ids']
    attn_masks = inputs['attention_mask']
    init_embeddings = get_token_embedding(input_ids, model, args)

    shape = init_embeddings.shape
    if args.noise_mechanism == "Gaussian":
        noises = sample_noise_gaussian(shape, args.noise_std, device=args.device)
        noisy_embs = init_embeddings + noises
    elif args.noise_mechanism == "Laplace":
        noises = sample_noise_laplace(shape, args.epsilon, device=args.device)
        noisy_embs = init_embeddings + noises
    elif args.noise_mechanism == "ChiDP":
        noises = sample_noise_chi(shape, args.eta, device=args.device)
        noisy_embs = init_embeddings + noises
    else:
        raise ValueError(f"Unsupported noise mechanism: {args.noise_mechanism}")

    for i in tqdm(range(len(input_ids)), desc="Applying UI-based privacy"):
        this_mask = attn_masks[i]
        this_embeds = noisy_embs[i][this_mask == 1]
        this_label = labels[i]
        for j in range(len(this_embeds)):
            if input_ids[i, j] not in top_ui_dict[this_label.item()]:
                embed = this_embeds[j]
                closest_token = get_closest_token(embed, tokenizer, model, args)
                input_ids[i, j] = closest_token

    return {"input_ids": input_ids, 'attention_mask': attn_masks}
