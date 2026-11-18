#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# DP-SPD utility-only run on curated WildChat.
# TRANSPORT_MODE=token_projection keeps the current black-box text path:
# noisy embeddings are projected back to nearest tokens and sent as text.
# TRANSPORT_MODE=embedding preserves the earlier client/server path:
# noisy inputs_embeds are sent directly to a local Hugging Face backend.

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

QA_JSONL="${QA_JSONL:-data/wildchat/wildchat_qa500_independent_userprivacy.jsonl}"
OPEN_JSONL="${OPEN_JSONL:-data/wildchat/wildchat_open500_independent_userprivacy.jsonl}"
BACKEND="${BACKEND:-local_hf}"
TRANSPORT_MODE="${TRANSPORT_MODE:-token_projection}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-models/Qwen3-32B}"
LOCAL_DTYPE="${LOCAL_DTYPE:-float16}"
LOCAL_DEVICE_MAP="${LOCAL_DEVICE_MAP:-auto}"
LOCAL_DEVICE="${LOCAL_DEVICE:-cuda:0}"
LOCAL_MAX_MEMORY="${LOCAL_MAX_MEMORY:-all:42GiB}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"

PROJECTION_MODEL="${PROJECTION_MODEL:-models/Qwen2.5-0.5B-Instruct}"
PROJECTION_DTYPE="${PROJECTION_DTYPE:-auto}"
PROJECTION_DEVICE="${PROJECTION_DEVICE:-cuda:0}"
PROJECTION_MAX_TOKENS="${PROJECTION_MAX_TOKENS:-4096}"
PROJECTION_CHUNK_SIZE="${PROJECTION_CHUNK_SIZE:-8}"
STORE_PROJECTION_TOKEN_DIAGNOSTICS="${STORE_PROJECTION_TOKEN_DIAGNOSTICS:-false}"

TARGET_EPSILON="${TARGET_EPSILON:-3.0}"
DUMMY_K="${DUMMY_K:-9}"
DUMMY_RATIO="${DUMMY_RATIO:-0.2}"
EPSILON_NON_SENSITIVE="${EPSILON_NON_SENSITIVE:-3.0}"
EPSILON_PRIVATE="${EPSILON_PRIVATE:-7.82}"
CLIP_NORM="${CLIP_NORM:-1.0}"
NOISE_MECHANISM="${NOISE_MECHANISM:-Laplace}"
SEED="${SEED:-42}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

USE_MESSAGES="${USE_MESSAGES:-true}"
PROMPT_STYLE="${PROMPT_STYLE:-raw}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_COMPLETION_TOKENS_QA="${MAX_COMPLETION_TOKENS_QA:-256}"
MAX_COMPLETION_TOKENS_OPEN="${MAX_COMPLETION_TOKENS_OPEN:-512}"
API_MODEL="${API_MODEL:-qwen3-32b}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wildchat_dpspd_blackbox_utility}"
if [[ -z "${OUTPUT_PREFIX:-}" ]]; then
  if [[ "${TRANSPORT_MODE}" == "embedding" ]]; then
    OUTPUT_PREFIX="qwen3_32b_wildchat_dpspd_embedding_k${DUMMY_K}_eps3_seed${SEED}"
  else
    OUTPUT_PREFIX="qwen3_32b_wildchat_dpspd_blackbox_k${DUMMY_K}_eps3_seed${SEED}"
  fi
fi

python experiments/wildchat/run_dpspd_blackbox_utility.py \
  --qa_jsonl "${QA_JSONL}" \
  --open_jsonl "${OPEN_JSONL}" \
  --backend "${BACKEND}" \
  --transport_mode "${TRANSPORT_MODE}" \
  --api_model "${API_MODEL}" \
  --openai_base_url "${OPENAI_BASE_URL}" \
  --local_model_path "${LOCAL_MODEL_PATH}" \
  --local_dtype "${LOCAL_DTYPE}" \
  --local_device_map "${LOCAL_DEVICE_MAP}" \
  --local_device "${LOCAL_DEVICE}" \
  --local_max_memory "${LOCAL_MAX_MEMORY}" \
  --local_files_only "${LOCAL_FILES_ONLY}" \
  --trust_remote_code "${TRUST_REMOTE_CODE}" \
  --enable_thinking "${ENABLE_THINKING}" \
  --max_input_tokens "${MAX_INPUT_TOKENS}" \
  --projection_model "${PROJECTION_MODEL}" \
  --projection_dtype "${PROJECTION_DTYPE}" \
  --projection_device "${PROJECTION_DEVICE}" \
  --projection_max_tokens "${PROJECTION_MAX_TOKENS}" \
  --projection_chunk_size "${PROJECTION_CHUNK_SIZE}" \
  --store_projection_token_diagnostics "${STORE_PROJECTION_TOKEN_DIAGNOSTICS}" \
  --target_epsilon "${TARGET_EPSILON}" \
  --dummy_k "${DUMMY_K}" \
  --dummy_ratio "${DUMMY_RATIO}" \
  --epsilon_non_sensitive "${EPSILON_NON_SENSITIVE}" \
  --epsilon_private "${EPSILON_PRIVATE}" \
  --clip_norm "${CLIP_NORM}" \
  --noise_mechanism "${NOISE_MECHANISM}" \
  --seed "${SEED}" \
  --sample_seed "${SAMPLE_SEED}" \
  --use_messages "${USE_MESSAGES}" \
  --prompt_style "${PROMPT_STYLE}" \
  --temperature "${TEMPERATURE}" \
  --max_completion_tokens_qa "${MAX_COMPLETION_TOKENS_QA}" \
  --max_completion_tokens_open "${MAX_COMPLETION_TOKENS_OPEN}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_prefix "${OUTPUT_PREFIX}" \
  "$@"
