#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cuda:0}"
SERVER_MODEL="${SERVER_MODEL:-models/Llama-2-7b-chat-hf}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/whitebox_blackbox_gap}"

DATASET_ARGS=()
if [[ -n "${DATASET_PATH:-}" ]]; then
  DATASET_ARGS+=(--dataset_path "${DATASET_PATH}")
fi

python experiments/api_only/run_whitebox_blackbox_gap.py \
  --device "${DEVICE}" \
  --server_model "${SERVER_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  "${DATASET_ARGS[@]}" \
  "$@"
