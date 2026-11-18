#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# DP-SPD efficiency benchmark for Qwen3-32B.
#
# Default mode is synthetic: directly construct shared-prefix real+dummy
# embedding groups and isolate server-side KV-cache reuse. Set
# BENCHMARK_MODE=real to use curated WildChat prompts plus PrivacyDataset
# continuation dummy generation.
# Correctness/judge metrics are intentionally skipped.

INPUT_JSONL="${INPUT_JSONL:-data/wildchat/wildchat_qa500_independent_userprivacy.jsonl}"
BENCHMARK_MODE="${BENCHMARK_MODE:-synthetic}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-32B}"
DTYPE="${DTYPE:-float16}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_MEMORY="${MAX_MEMORY:-all:42GiB}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"

MAX_SAMPLES="${MAX_SAMPLES:-8}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SAMPLE_MODE="${SAMPLE_MODE:-first}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SYNTHETIC_SEQ_LEN="${SYNTHETIC_SEQ_LEN:-512}"
SYNTHETIC_VOCAB_START="${SYNTHETIC_VOCAB_START:-100}"
SYNTHETIC_VOCAB_LIMIT="${SYNTHETIC_VOCAB_LIMIT:-32000}"
MIN_WORDS="${MIN_WORDS:-5}"
MAX_WORDS="${MAX_WORDS:-260}"

KS="${KS:-9,19,29}"
DUMMY_RATIOS="${DUMMY_RATIOS:-0.2,0.5,0.8}"
EPSILON_NON_SENSITIVE="${EPSILON_NON_SENSITIVE:-3.0}"
EPSILON_PRIVATE_DEFAULT="${EPSILON_PRIVATE_DEFAULT:-8.51}"
EPSILON_PRIVATE_MAP="${EPSILON_PRIVATE_MAP:-9:7.82,19:8.51}"
CLIP_NORM="${CLIP_NORM:-1.0}"
NOISE_MECHANISM="${NOISE_MECHANISM:-Laplace}"
SEED="${SEED:-42}"

WARMUP_STEPS="${WARMUP_STEPS:-1}"
REPEAT_STEPS="${REPEAT_STEPS:-3}"
PROFILE="${PROFILE:-false}"
MODULE_PROFILE="${MODULE_PROFILE:-true}"
PROFILE_TOP_OPS="${PROFILE_TOP_OPS:-12}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/wildchat_speed_benchmark}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_32b_${BENCHMARK_MODE}_dpspd_speed_seed${SEED}}"

ATTN_ARGS=()
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  ATTN_ARGS=(--attn_implementation "${ATTN_IMPLEMENTATION}")
fi

python experiments/speed/run_wildchat_qwen3_speed_benchmark.py \
  --input_jsonl "${INPUT_JSONL}" \
  --benchmark_mode "${BENCHMARK_MODE}" \
  --model_path "${MODEL_PATH}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --device_map "${DEVICE_MAP}" \
  --max_memory "${MAX_MEMORY}" \
  --local_files_only "${LOCAL_FILES_ONLY}" \
  --trust_remote_code "${TRUST_REMOTE_CODE}" \
  "${ATTN_ARGS[@]}" \
  --max_samples "${MAX_SAMPLES}" \
  --sample_seed "${SAMPLE_SEED}" \
  --sample_mode "${SAMPLE_MODE}" \
  --min_words "${MIN_WORDS}" \
  --max_words "${MAX_WORDS}" \
  --max_length "${MAX_LENGTH}" \
  --synthetic_seq_len "${SYNTHETIC_SEQ_LEN}" \
  --synthetic_vocab_start "${SYNTHETIC_VOCAB_START}" \
  --synthetic_vocab_limit "${SYNTHETIC_VOCAB_LIMIT}" \
  --ks "${KS}" \
  --dummy_ratios "${DUMMY_RATIOS}" \
  --epsilon_non_sensitive "${EPSILON_NON_SENSITIVE}" \
  --epsilon_private_default "${EPSILON_PRIVATE_DEFAULT}" \
  --epsilon_private_map "${EPSILON_PRIVATE_MAP}" \
  --clip_norm "${CLIP_NORM}" \
  --noise_mechanism "${NOISE_MECHANISM}" \
  --seed "${SEED}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --repeat_steps "${REPEAT_STEPS}" \
  --profile "${PROFILE}" \
  --module_profile "${MODULE_PROFILE}" \
  --profile_top_ops "${PROFILE_TOP_OPS}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_prefix "${OUTPUT_PREFIX}" \
  "$@"
