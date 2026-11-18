# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dpspd.util.parameters import get_args
from dpspd.util.utils import load_local_causal_lm
from dpspd.client import PrivacyDataset, BatchConfig, DummyConfig, LDPNoiseConfig

def run_min_k_prob_mia(dataloader, model, args, k_ratio=0.20):
    """
    Run a Min-K% probability membership-inference attack.
    """
    model.eval()
    vocab_embeddings = model.get_input_embeddings().weight.detach().to(args.device)

    total_groups = 0
    success_attacks = 0

    loss_fct = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Min-K% Prob MIA"):
            noisy_embs = batch['noisy_embeddings'].to(device=args.device, dtype=model.dtype)
            attn_mask = batch['attention_mask'].to(args.device)
            is_real_flags = batch['is_real']

            B, L, D = noisy_embs.shape

            if not is_real_flags.any():
                continue

            guessed_ids = torch.zeros((B, L), dtype=torch.long, device=args.device)
            vocab_norm = torch.nn.functional.normalize(vocab_embeddings, p=2, dim=-1)

            for i in range(B):
                for j in range(L):
                    if attn_mask[i, j] == 0:
                        continue

                    emb_norm = torch.nn.functional.normalize(noisy_embs[i, j], p=2, dim=-1)
                    sim = torch.matmul(vocab_norm, emb_norm)
                    guessed_ids[i, j] = torch.argmax(sim)

            outputs = model(inputs_embeds=noisy_embs, attention_mask=attn_mask)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = guessed_ids[..., 1:].contiguous()
            shift_mask = attn_mask[..., 1:].contiguous()

            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_losses = token_losses.view(B, L - 1)

            min_k_scores = []

            for i in range(B):
                valid_len = shift_mask[i].sum().item()
                if valid_len == 0:
                    min_k_scores.append(float('inf'))
                    continue

                valid_losses = token_losses[i, :valid_len]
                k_num = max(1, int(valid_len * k_ratio))
                topk_losses, _ = torch.topk(valid_losses, k_num, largest=True)
                score = topk_losses.mean().item()
                min_k_scores.append(score)

            predicted_real_idx = np.argmin(min_k_scores)
            actual_real_idx = (is_real_flags == True).nonzero(as_tuple=True)[0].item()

            if predicted_real_idx == actual_real_idx:
                success_attacks += 1
            total_groups += 1

    asr = success_attacks / total_groups if total_groups > 0 else 0
    print(f"\n=====================================")
    print(f"Total Groups Evaluated: {total_groups}")
    print(f"Attack Success Rate (ASR): {asr * 100:.2f}%")
    print(f"Random Guess Baseline: {100.0 / B:.2f}%")
    print(f"=====================================\n")
    return asr

def main():
    args = get_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"

    print("Loading DP-SPD continuation model...")
    tokenizer, model, load_dtype = load_local_causal_lm(
        model_path=args.continuation_model,
        device=args.device,
        dtype_preference=args.continuation_dtype,
        local_files_only=args.local_files_only,
    )
    print(f"[Model] {os.path.basename(args.continuation_model.rstrip('/'))} | dtype={load_dtype}")

    print("Loading local real dataset (WikiText)...")
    from datasets import load_from_disk
    import random

    raw_ds = load_from_disk("data/wikitext/wikitext-2-v1")

    if 'train' in raw_ds:
        all_texts = raw_ds['train']['text']
    else:
        all_texts = raw_ds['text']

    random.seed(42)
    filtered_texts = [text for text in all_texts if 50 <= len(text.split()) <= 200]
    sampled_texts = random.sample(filtered_texts, min(1000, len(filtered_texts)))

    num_eval_groups = 50
    real_texts = sampled_texts[:num_eval_groups]
    corpus_texts = sampled_texts[num_eval_groups:]

    print(f"Selected {len(real_texts)} real queries and a dummy corpus of {len(corpus_texts)} texts.")

    batch_cfg = BatchConfig(max_length=64, device=torch.device('cpu'), hidden_size=model.config.hidden_size)

    dummy_cfg = DummyConfig(
        m_private_tokens=args.m_private,
        sensitive_ratio=1.0 if args.dummy_mode == "sentence" else 0.5,
        n_dummies_per_text=args.n_dummy,
        dummy_mode=args.dummy_mode,
        dummy_corpus=corpus_texts,
        pos_mapping_path=None
    )

    current_epsilon = getattr(args, 'epsilon_private', 1.0)

    ldp_cfg = LDPNoiseConfig(
        mechanism="Laplace",
        epsilon_private=current_epsilon,
        epsilon_non_sensitive=args.epsilon_non_sensitive,
        clip_norm=1.0,
        seed=np.random.randint(0, 10000)
    )

    print(f"Initializing PrivacyDataset in {args.dummy_mode} mode...")
    dataset = PrivacyDataset(real_texts, tokenizer, batch_cfg, dummy_cfg, ldp_cfg, shuffle_mode="block", pretrained_embeddings=model.get_input_embeddings())

    dataloader = DataLoader(dataset, batch_size=args.n_dummy + 1, shuffle=False)

    run_min_k_prob_mia(dataloader, model, args, k_ratio=0.20)

if __name__ == "__main__":
    main()
