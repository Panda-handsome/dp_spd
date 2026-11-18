#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cuda:0}"
CONTINUATION_MODEL="${CONTINUATION_MODEL:-${DPSPD_CONTINUATION_MODEL:-models/Qwen2.5-0.5B-Instruct}}"
DATASET_PATH="${DATASET_PATH:-data/imdb}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/examples}"

python experiments/examples/generate_dpspd_examples.py \
  --device "${DEVICE}" \
  --continuation_model "${CONTINUATION_MODEL}" \
  --dataset_path "${DATASET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
