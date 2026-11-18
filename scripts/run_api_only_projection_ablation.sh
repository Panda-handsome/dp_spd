#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cuda:0}"
PROJECTION_MODEL="${PROJECTION_MODEL:-models/Qwen2.5-0.5B-Instruct}"
API_MODEL="${API_MODEL:-gpt-4o}"
BACKEND="${BACKEND:-dry_run}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/api_only_projection_ablation}"

if [[ "${BACKEND}" == "openai_chat" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[Error] BACKEND=openai_chat requires OPENAI_API_KEY in the environment." >&2
  echo "Run: export OPENAI_API_KEY=...   or prefix each background command with OPENAI_API_KEY=..." >&2
  exit 2
fi

DATASET_ARGS=()
if [[ -n "${DATASET_PATH:-}" ]]; then
  DATASET_ARGS+=(--dataset_path "${DATASET_PATH}")
fi

EXTRA_ARGS=()
if [[ -n "${SAMPLE_SEED:-}" ]]; then
  EXTRA_ARGS+=(--sample_seed "${SAMPLE_SEED}")
fi
if [[ -n "${REUSE_OUTPUTS_JSONL:-}" ]]; then
  EXTRA_ARGS+=(--reuse_outputs_jsonl "${REUSE_OUTPUTS_JSONL}")
fi
if [[ -n "${STORE_PROJECTION_TOKEN_DIAGNOSTICS:-}" ]]; then
  EXTRA_ARGS+=(--store_projection_token_diagnostics "${STORE_PROJECTION_TOKEN_DIAGNOSTICS}")
fi
if [[ -n "${PROJECTION_TIE_TOLERANCE:-}" ]]; then
  EXTRA_ARGS+=(--projection_tie_tolerance "${PROJECTION_TIE_TOLERANCE}")
fi
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  EXTRA_ARGS+=(--openai_base_url "${OPENAI_BASE_URL}")
fi

python experiments/api_only/run_projection_ablation.py \
  --device "${DEVICE}" \
  --projection_model "${PROJECTION_MODEL}" \
  --backend "${BACKEND}" \
  --api_model "${API_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  "${DATASET_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
