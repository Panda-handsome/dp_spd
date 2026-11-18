#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ATTACK="${ATTACK:-${1:-mia}}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  shift
fi

DATASET="${DATASET:-imdb}"
DEVICE="${DEVICE:-cuda:0}"
CONTINUATION_MODEL="${CONTINUATION_MODEL:-models/Qwen2.5-0.5B-Instruct}"
ATTACK_MODEL="${ATTACK_MODEL:-models/Llama-2-7b-chat-hf}"

case "${DATASET}" in
  imdb)
    DATASET_PATH="${DATASET_PATH:-data/imdb}"
    DATASET_LABEL="imdb"
    ;;
  wikitext|wikitext2)
    DATASET_PATH="${DATASET_PATH:-data/wikitext/wikitext-2-v1}"
    DATASET_LABEL="wikitext"
    ;;
  custom)
    if [[ -z "${DATASET_PATH:-}" ]]; then
      echo "[Error] DATASET=custom requires DATASET_PATH=/path/to/dataset." >&2
      exit 2
    fi
    DATASET_LABEL="${DATASET_LABEL:-custom}"
    ;;
  *)
    echo "[Error] Unknown DATASET='${DATASET}'. Use imdb, wikitext, or custom." >&2
    exit 2
    ;;
esac

case "${ATTACK}" in
  mia)
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/attack_mia_${DATASET_LABEL}}"
    python attack/mia_dummy_strategy_benchmark.py \
      --device "${DEVICE}" \
      --continuation_model "${CONTINUATION_MODEL}" \
      --attack_model "${ATTACK_MODEL}" \
      --dataset_path "${DATASET_PATH}" \
      --output_dir "${OUTPUT_DIR}" \
      "$@"
    ;;
  eia)
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/attack_eia_${DATASET_LABEL}}"
    python attack/eia_dummy_strategy_benchmark.py \
      --device "${DEVICE}" \
      --continuation_model "${CONTINUATION_MODEL}" \
      --attack_model "${ATTACK_MODEL}" \
      --dataset_path "${DATASET_PATH}" \
      --output_dir "${OUTPUT_DIR}" \
      "$@"
    ;;
  gpt)
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/attack_gpt_${DATASET_LABEL}}"
    python attack/gpt_inference_attack.py \
      --device "${DEVICE}" \
      --continuation_model "${CONTINUATION_MODEL}" \
      --dataset_path "${DATASET_PATH}" \
      --output_dir "${OUTPUT_DIR}" \
      "$@"
    ;;
  *)
    echo "[Error] Unknown ATTACK='${ATTACK}'. Use mia, eia, or gpt." >&2
    exit 2
    ;;
esac
