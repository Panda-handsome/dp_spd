#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# WildChat DP-SPD black-box attacks.
# Candidate groups are fully constructed as 1 real + k generated dummies, but
# the attacker only sees projected text, not embeddings.
#
# Smoke test:
#   BACKEND=dry_run is not used here; pass --dry_run true instead.
#   bash scripts/run_wildchat_dpspd_blackbox_attack.sh --max_groups_qa 1 --max_groups_open 0 --dry_run true

QA_JSONL="${QA_JSONL:-data/wildchat/wildchat_qa500_independent_userprivacy.jsonl}"
OPEN_JSONL="${OPEN_JSONL:-data/wildchat/wildchat_open500_independent_userprivacy.jsonl}"

PROJECTION_MODEL="${PROJECTION_MODEL:-models/Qwen2.5-0.5B-Instruct}"
PROJECTION_DTYPE="${PROJECTION_DTYPE:-auto}"
PROJECTION_DEVICE="${PROJECTION_DEVICE:-cuda:0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
MAX_LENGTH="${MAX_LENGTH:-512}"
PROJECTION_CHUNK_SIZE="${PROJECTION_CHUNK_SIZE:-8}"

TARGET_EPSILON="${TARGET_EPSILON:-3.0}"
DUMMY_K="${DUMMY_K:-9}"
DUMMY_RATIO="${DUMMY_RATIO:-0.2}"
EPSILON_NON_SENSITIVE="${EPSILON_NON_SENSITIVE:-3.0}"
EPSILON_PRIVATE="${EPSILON_PRIVATE:-7.82}"
CLIP_NORM="${CLIP_NORM:-1.0}"
NOISE_MECHANISM="${NOISE_MECHANISM:-Laplace}"
SEED="${SEED:-42}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

ATTACK_MODES="${ATTACK_MODES:-mia}"
MIA_ATTACK_MODEL="${MIA_ATTACK_MODEL:-${PROJECTION_MODEL}}"
MIA_ATTACK_DTYPE="${MIA_ATTACK_DTYPE:-auto}"
MIA_DEVICE_MAP="${MIA_DEVICE_MAP:-none}"
MIA_DEVICE="${MIA_DEVICE:-cuda:0}"
MIA_MAX_MEMORY="${MIA_MAX_MEMORY:-}"
MIA_MAX_LENGTH="${MIA_MAX_LENGTH:-512}"
MIA_K_RATIO="${MIA_K_RATIO:-0.20}"

GPT_MODEL="${GPT_MODEL:-gpt-4o}"
GPT_TEMPERATURE="${GPT_TEMPERATURE:-0}"
GPT_MAX_COMPLETION_TOKENS="${GPT_MAX_COMPLETION_TOKENS:-900}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/wildchat_dpspd_blackbox_attack}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_32b_wildchat_dpspd_blackbox_attack_k${DUMMY_K}_eps3_seed${SEED}}"

python experiments/wildchat/run_dpspd_blackbox_attack.py \
  --qa_jsonl "${QA_JSONL}" \
  --open_jsonl "${OPEN_JSONL}" \
  --projection_model "${PROJECTION_MODEL}" \
  --projection_dtype "${PROJECTION_DTYPE}" \
  --projection_device "${PROJECTION_DEVICE}" \
  --local_files_only "${LOCAL_FILES_ONLY}" \
  --trust_remote_code "${TRUST_REMOTE_CODE}" \
  --max_length "${MAX_LENGTH}" \
  --projection_chunk_size "${PROJECTION_CHUNK_SIZE}" \
  --target_epsilon "${TARGET_EPSILON}" \
  --dummy_k "${DUMMY_K}" \
  --dummy_ratio "${DUMMY_RATIO}" \
  --epsilon_non_sensitive "${EPSILON_NON_SENSITIVE}" \
  --epsilon_private "${EPSILON_PRIVATE}" \
  --clip_norm "${CLIP_NORM}" \
  --noise_mechanism "${NOISE_MECHANISM}" \
  --seed "${SEED}" \
  --sample_seed "${SAMPLE_SEED}" \
  --attack_modes "${ATTACK_MODES}" \
  --mia_attack_model "${MIA_ATTACK_MODEL}" \
  --mia_attack_dtype "${MIA_ATTACK_DTYPE}" \
  --mia_device_map "${MIA_DEVICE_MAP}" \
  --mia_device "${MIA_DEVICE}" \
  --mia_max_memory "${MIA_MAX_MEMORY}" \
  --mia_max_length "${MIA_MAX_LENGTH}" \
  --mia_k_ratio "${MIA_K_RATIO}" \
  --gpt_model "${GPT_MODEL}" \
  --gpt_temperature "${GPT_TEMPERATURE}" \
  --gpt_max_completion_tokens "${GPT_MAX_COMPLETION_TOKENS}" \
  --openai_base_url "${OPENAI_BASE_URL}" \
  --api_key_env "${API_KEY_ENV}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_prefix "${OUTPUT_PREFIX}" \
  "$@"
