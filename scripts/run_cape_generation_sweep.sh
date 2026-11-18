#!/usr/bin/env bash
set -euo pipefail

# CAPE-DP-only-style LLaMA sequence adapter.  This is intentionally not named
# the original CAPE method: CAPE's private-attribute adversarial training
# cannot be run on WikiText-2 because the required labels do not exist.
#
# Smoke test:
#   MAX_SAMPLES=4 bash scripts/run_cape_generation_sweep.sh
# Full same-scale run:
#   MAX_SAMPLES=300 bash scripts/run_cape_generation_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EPSILONS=(1.5 2.0 2.5 3.0)
GPUS=(4 5 6 7)
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-models/Llama-2-7b-chat-hf}"
DATASET_PATH="${DATASET_PATH:-data/wikitext/wikitext-2-v1}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
MAX_LENGTH="${MAX_LENGTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ROUGE_CONTEXT_LEN="${ROUGE_CONTEXT_LEN:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-current_dpspd}"
ROUGE_USE_STEMMER="${ROUGE_USE_STEMMER:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baselines/cape/wikitext_figure3_laplace_guard_v2}"
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_DIR}/logs"
pids=()
for index in "${!EPSILONS[@]}"; do
  epsilon="${EPSILONS[$index]}"
  gpu="${GPUS[$index]}"
  output_csv="${OUTPUT_DIR}/cape_wikitext_eps${epsilon}.csv"
  log_file="${OUTPUT_DIR}/logs/cape_wikitext_eps${epsilon}_gpu${gpu}.log"
  "${PYTHON_BIN}" baseline/cape/run_generation_figure3.py \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --epsilon "${epsilon}" \
    --device "cuda:${gpu}" \
    --max_samples "${MAX_SAMPLES}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --rouge_context_len "${ROUGE_CONTEXT_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --selection_protocol "${SELECTION_PROTOCOL}" \
    --rouge_use_stemmer "${ROUGE_USE_STEMMER}" \
    --seed "${SEED}" \
    --output_csv "${output_csv}" \
    "$@" >"${log_file}" 2>&1 &
  pids+=("$!")
  echo "[Launched] epsilon=${epsilon} gpu=cuda:${gpu} pid=${pids[-1]} log=${log_file}"
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[Failed] epsilon=${EPSILONS[$index]} gpu=cuda:${GPUS[$index]}" >&2
    status=1
  else
    echo "[Completed] epsilon=${EPSILONS[$index]} gpu=cuda:${GPUS[$index]}"
  fi
done
exit "${status}"
