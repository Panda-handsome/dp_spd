#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Original upper-bound on curated WildChat QA/open-ended subsets.
# Default mode directly loads the local Hugging Face model with Transformers.
#
# Smoke test:
#   bash scripts/run_wildchat_original_upper_bound.sh --max_samples_qa 20 --max_samples_open 20 --run_qa_judge false --resume false
#
# Full run:
#   bash scripts/run_wildchat_original_upper_bound.sh --resume false --run_qa_judge false

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

QA_JSONL="${QA_JSONL:-data/wildchat/wildchat_qa500_independent_userprivacy.jsonl}"
OPEN_JSONL="${OPEN_JSONL:-data/wildchat/wildchat_open500_independent_userprivacy.jsonl}"
BACKEND="${BACKEND:-local_hf}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-models/Qwen3-32B}"
LOCAL_DTYPE="${LOCAL_DTYPE:-bfloat16}"
LOCAL_DEVICE_MAP="${LOCAL_DEVICE_MAP:-auto}"
LOCAL_DEVICE="${LOCAL_DEVICE:-cuda:0}"
LOCAL_MAX_MEMORY="${LOCAL_MAX_MEMORY:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
USE_MESSAGES="${USE_MESSAGES:-true}"
PROMPT_STYLE="${PROMPT_STYLE:-raw}"
API_MODEL="${API_MODEL:-qwen3-32b}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wildchat_original_upper_bound}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_32b_plaintext_wildchat_local_hf}"

python experiments/wildchat/run_original_upper_bound.py \
  --qa_jsonl "${QA_JSONL}" \
  --open_jsonl "${OPEN_JSONL}" \
  --backend "${BACKEND}" \
  --local_model_path "${LOCAL_MODEL_PATH}" \
  --local_dtype "${LOCAL_DTYPE}" \
  --local_device_map "${LOCAL_DEVICE_MAP}" \
  --local_device "${LOCAL_DEVICE}" \
  --local_max_memory "${LOCAL_MAX_MEMORY}" \
  --local_files_only "${LOCAL_FILES_ONLY}" \
  --trust_remote_code "${TRUST_REMOTE_CODE}" \
  --enable_thinking "${ENABLE_THINKING}" \
  --max_input_tokens "${MAX_INPUT_TOKENS}" \
  --use_messages "${USE_MESSAGES}" \
  --prompt_style "${PROMPT_STYLE}" \
  --api_model "${API_MODEL}" \
  --openai_base_url "${OPENAI_BASE_URL}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_prefix "${OUTPUT_PREFIX}" \
  "$@"
