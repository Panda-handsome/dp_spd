#!/usr/bin/env bash
set -euo pipefail

# Four independent full-sequence CusText-style runs, one per requested GPU.
# This runner adapts CusText's candidate-group exponential mechanism to LLaMA
# BPE IDs, so every non-special token in the 256-token sequence is replaced
# and evaluated by exactly the DP-SPD Figure-3 PPL/ROUGE protocol.  It must be
# labelled "CusText-style", not the original word-level CusText paper result.
# CusText has no dummy ratio r or dummy count k; each output is a horizontal
# baseline point over Figure 3's k axis for its corresponding epsilon.
#
# Smoke test:
#   MAX_SAMPLES=8 bash scripts/run_custext_generation_sweep.sh
# Full WikiText-2 run:
#   MAX_SAMPLES=300 bash scripts/run_custext_generation_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EPSILONS=(1.5 2.0 2.5 3.0)
GPUS=(4 5 6 7)
MODEL_PATH="${MODEL_PATH:-models/Llama-2-7b-chat-hf}"
DATASET_PATH="${DATASET_PATH:-data/wikitext/wikitext-2-v1}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CONTEXT_LEN="${CONTEXT_LEN:-32}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-current_dpspd}"
ROUGE_USE_STEMMER="${ROUGE_USE_STEMMER:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baselines/custext/wikitext_figure3}"
MAPPING_STRATEGY="${MAPPING_STRATEGY:-release_compatible}"
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_DIR}/logs"
pids=()

for index in "${!EPSILONS[@]}"; do
  epsilon="${EPSILONS[$index]}"
  gpu="${GPUS[$index]}"
  output_csv="${OUTPUT_DIR}/custext_wikitext_eps${epsilon}.csv"
  log_file="${OUTPUT_DIR}/logs/custext_wikitext_eps${epsilon}_gpu${gpu}.log"
  python baseline/custext/run_generation_figure3.py \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --epsilon "${epsilon}" \
    --device "cuda:${gpu}" \
    --mapping_device "cuda:${gpu}" \
    --max_samples "${MAX_SAMPLES}" \
    --selection_protocol "${SELECTION_PROTOCOL}" \
    --batch_size "${BATCH_SIZE}" \
    --rouge_context_len "${CONTEXT_LEN}" \
    --max_length "${MAX_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --rouge_use_stemmer "${ROUGE_USE_STEMMER}" \
    --mapping_strategy "${MAPPING_STRATEGY}" \
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
