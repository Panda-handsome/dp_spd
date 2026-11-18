#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-${1:-sst2}}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  shift
fi

DEVICE="${DEVICE:-cuda:0}"

case "${TASK}" in
  sst2)
    MODEL_PATH="${MODEL_PATH:-models/bert-base-uncased-sst2}"
    DATASET_PATH="${DATASET_PATH:-data/glue/sst2}"
    HEAD="${HEAD:-sequence}"
    DTYPE="${DTYPE:-auto}"
    OUTPUT_CSV="${OUTPUT_CSV:-outputs/classification_sst2.csv}"
    ;;
  qnli)
    MODEL_PATH="${MODEL_PATH:-models/bert-base-uncased-qnli}"
    DATASET_PATH="${DATASET_PATH:-data/glue/qnli}"
    HEAD="${HEAD:-sequence}"
    DTYPE="${DTYPE:-auto}"
    OUTPUT_CSV="${OUTPUT_CSV:-outputs/classification_qnli.csv}"
    ;;
  imdb|imdb_llama)
    TASK="imdb"
    MODEL_PATH="${MODEL_PATH:-models/Llama-2-7b-chat-hf}"
    DATASET_PATH="${DATASET_PATH:-data/imdb}"
    HEAD="${HEAD:-probe}"
    DTYPE="${DTYPE:-bfloat16}"
    OUTPUT_CSV="${OUTPUT_CSV:-outputs/classification_imdb_llama.csv}"
    ;;
  *)
    echo "[Error] Unknown classification TASK='${TASK}'. Use sst2, qnli, or imdb." >&2
    exit 2
    ;;
esac

python experiments/classification/run_classification.py \
  --task "${TASK}" \
  --head "${HEAD}" \
  --model_path "${MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --output_csv "${OUTPUT_CSV}" \
  "$@"
