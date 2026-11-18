# -*- coding: utf-8 -*-
# Shared CLI parameters for experiments and attack scripts.

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpspd.util.utils import str2bool, str2type
from dpspd.util.config import dataset_meta

def get_args():
    parser = argparse.ArgumentParser()

    # DP-SPD and LDP parameters.
    parser.add_argument("--n_dummy", type=int, default=3,
                        help="Number of dummy sequences per real sequence (for efficiency experiment)")
    parser.add_argument("--m_private", type=int, default=1,
                        help="Number of private tokens to protect in each sequence")
    parser.add_argument("--epsilon_non_sensitive", type=float, default=10.0,
                        help="Privacy budget for non-sensitive (prefix) tokens. Must be high but finite.")

    # Training parameters.
    parser.add_argument("--cls_epochs", type=int, default=50,
                        help="number of epochs to train classification model")
    parser.add_argument("--train_epochs", type=int, default=4,
                        help="number of epochs to prompt-tune base model")

    # Paths.
    parser.add_argument("--cls_save_dir", type=str, default="./embeddings_cls",
                        help="Directory to save/load extracted embeddings for classification")
    parser.add_argument("--denoise_data_dir", type=str, default="data",
                        help="Root directory for local datasets")

    # Batch sizes.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for benchmark/extraction (total samples)")
    parser.add_argument("--base_batch_size", type=int, default=8,
                        help="batch size to make prediction with LLM (legacy)")
    parser.add_argument("--ft_batch_size", type=int, default=8,
                        help="batch size to finetune LLM")
    parser.add_argument("--cls_batch_size", type=int, default=256,
                        help="batch size to train classification model")

    # Device and precision.
    parser.add_argument("--device", type=str, default="cuda:1",
                        choices=["cuda", "cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
                        help="cpu or gpu")
    parser.add_argument("--base_precision", type=str2type, default=torch.float32,
                        help="Precision of base model")

    # Models and tasks.
    parser.add_argument("--base_model", type=str, default="gpt2",
                        help="large language model for prediction (name or local path)")
    parser.add_argument("--continuation_model", type=str, default="models/Qwen2.5-0.5B-Instruct",
                        help="client-side continuation model used by DP-SPD generation/verification scripts")
    parser.add_argument("--continuation_dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="precision used when loading the DP-SPD continuation model")
    parser.add_argument("--local_files_only", type=str2bool, default=True,
                        help="whether to load local model/tokenizer files only")
    parser.add_argument("--task", type=str, default="imdb",
                        choices=list(dataset_meta.keys()),
                        help="downstream task")
    parser.add_argument("--downstream_task_train_size", type=int, default=5000,
                        help="size of data to train the downstream task model")
    parser.add_argument("--downstream_task_test_size", type=int, default=1000,
                        help="size of data to test the downstream task model")
    parser.add_argument("--ft_data", type=str, default="glue_sst2",
                        choices=list(dataset_meta.keys()),
                        help="dataset to finetune the base model")
    parser.add_argument("--use_ft_base", type=str2bool, default=False,
                        help="whether to use the finetuned base model or not")

    # Noise mechanism.
    parser.add_argument("--noise_mechanism", type=str, default="Laplace",
                        choices=["Laplace", "Gaussian", "ChiDP"],
                        help="noise mechanism for privacy (client-side only)")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="privacy budget epsilon for private tokens")
    parser.add_argument("--noise_std", type=float, default=1.0,
                        help="standard deviation for Gaussian mechanism")
    parser.add_argument("--eta", type=float, default=100.0,
                        help="parameter for ChiDP mechanism")
    parser.add_argument("--clip_norm", type=float, default=10.0,
                        help="Clipping norm for LDP")

    # Transformer hyperparameters.
    parser.add_argument("--token_length", type=int, default=512,
                        help="length of token sequence")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of heads in the transformer")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="number of layers in the transformer")
    parser.add_argument("--d_ff", type=int, default=1280,
                        help="hidden dimension for the feed-forward network")
    parser.add_argument("--dim_head", type=int, default=256,
                        help="dimension of each attention head")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout rate")

    # Prompt tuning and reconstruction parameters.
    parser.add_argument("--num_virtual_tokens", type=int, default=150,
                        help="number of virtual tokens in prompt tuning")
    parser.add_argument("--lr_peft", type=float, default=6e-5,
                        help="learning rate in prompt tuning")
    parser.add_argument("--n_plain_tok", type=int, default=40,
                        help="number of plain text tokens for reconstruction")
    parser.add_argument("--rec_vocab_size", type=int, default=7630,
                        help="vocabulary size of reconstruction layer")

    # KV-cache configuration.
    parser.add_argument("--use_kv_cache", type=str2bool, default=True,
                        help="whether to enable KV-Cache reuse on server side")
    parser.add_argument("--kv_cache_max_steps", type=int, default=1024,
                        help="maximum steps to keep in KV-Cache")
    parser.add_argument("--kv_cache_prefix_len", type=int, default=16,
                        help="number of tokens used to compute prefix hash")
    parser.add_argument("--kv_cache_cross_sample", type=str2bool, default=True,
                        help="whether to allow cross-sample reuse")

    # Miscellaneous.
    parser.add_argument("--attack_data", type=str, default="women_clothing",
                        choices=["tweets_gender", "women_clothing"],
                        help="dataset for inference attack")
    parser.add_argument("--cal_time", type=str2bool, default=False,
                        help="whether to compute the training time of a model")
    parser.add_argument("--loss", type=str, default="mse",
                        choices=["mse", "mae", "cosEmb", "cross_entropy", "bce"],
                        help="type of training loss")

    parser.add_argument("--dummy_mode", type=str, default="token", choices=["token", "sentence", "generation"])

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(f"Args loaded successfully.")
