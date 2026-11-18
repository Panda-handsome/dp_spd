#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-${1:-wikitext}}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  shift
fi

DEVICE="${DEVICE:-cuda:0}"

case "${TASK}" in
  wikitext|alpaca|dolly)
    MODEL_PATH="${MODEL_PATH:-models/Llama-2-7b-chat-hf}"
    case "${TASK}" in
      wikitext) DATASET_PATH="${DATASET_PATH:-data/wikitext/wikitext-2-v1}" ;;
      alpaca) DATASET_PATH="${DATASET_PATH:-data/alpaca}" ;;
      dolly) DATASET_PATH="${DATASET_PATH:-data/databricks_dolly}" ;;
    esac
    OUTPUT_CSV="${OUTPUT_CSV:-outputs/generation_${TASK}.csv}"
    python experiments/generation/llama_generation_eval.py \
      --dataset_name "${TASK}" \
      --model_path "${MODEL_PATH}" \
      --dataset_path "${DATASET_PATH}" \
      --device "${DEVICE}" \
      --output_csv "${OUTPUT_CSV}" \
      "$@"
    ;;
  cnndm|qwen3_cnndm)
    MODEL_PATH="${MODEL_PATH:-models/Qwen3-14B}"
    DATASET_PATH="${DATASET_PATH:-data/cnn_dailymail}"
    OUTPUT_CSV="${OUTPUT_CSV:-outputs/generation_qwen3_cnndm.csv}"
    python experiments/generation/qwen3_cnndm_eval.py \
      --model_path "${MODEL_PATH}" \
      --dataset_path "${DATASET_PATH}" \
      --device "${DEVICE}" \
      --output_csv "${OUTPUT_CSV}" \
      "$@"
    ;;
  *)
    echo "[Error] Unknown generation TASK='${TASK}'. Use wikitext, alpaca, dolly, or cnndm." >&2
    exit 2
    ;;
esac
