#!/usr/bin/env bash
# Sweep the Qwen2.5 continuation/proxy generator while holding all DP-SPD
# transport, scoring, data, and privacy settings fixed.
#
# Expected server layout (override MODEL_ROOT if different):
#   models/Qwen2.5-{0.5B,3B,7B,32B}-Instruct
#
# GPU plan for a four-GPU server (two phases):
#   Phase 1, concurrent: 0 -> 0.5B, 1 -> 3B, 2 -> 7B.
#   Phase 2, after phase 1 releases memory: 0,1 -> sharded 32B;
#   2 -> the fixed 0.5B transport/quality/MIA scorer.
# GPU 3 remains available for other work. Set GPU32_MAX_MEMORY to match the
# two cards used for the 32B proxy.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

MODEL_ROOT="${MODEL_ROOT:-models}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/proxy_generator_scale}"
LOG_DIR="${OUTPUT_DIR}/logs"
DATASET_PATH="${DATASET_PATH:-${REPO_DIR}/data/wikitext/wikitext-2-v1}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DUMMY_K="${DUMMY_K:-9}"
DUMMY_RATIO="${DUMMY_RATIO:-0.2}"
TARGET_EPSILON="${TARGET_EPSILON:-3.0}"
EPSILON_PRIVATE="${EPSILON_PRIVATE:-7.82}"
GPU32_MAX_MEMORY="${GPU32_MAX_MEMORY:-0:44GiB,1:44GiB}"
GPU32_PROXY_DEVICE="${GPU32_PROXY_DEVICE:-cuda:0}"
GPU32_SCORER_DEVICE="${GPU32_SCORER_DEVICE:-cuda:2}"

TRANSPORT_MODEL="${MODEL_ROOT}/Qwen2.5-0.5B-Instruct"
for model_path in \
  "${TRANSPORT_MODEL}" \
  "${MODEL_ROOT}/Qwen2.5-3B-Instruct" \
  "${MODEL_ROOT}/Qwen2.5-7B-Instruct" \
  "${MODEL_ROOT}/Qwen2.5-32B-Instruct"; do
  if [[ ! -d "${model_path}" ]]; then
    echo "Missing model directory: ${model_path}" >&2
    exit 2
  fi
done
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "Missing WikiText-2 dataset directory: ${DATASET_PATH}" >&2
  echo "Set DATASET_PATH=/absolute/path/to/wikitext-2-v1 before launching." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

COMMON=(
  --dataset_name wikitext
  --dataset_path "${DATASET_PATH}"
  --dataset_split test
  --max_samples "${NUM_SAMPLES}"
  --max_length "${MAX_LENGTH}"
  --target_epsilon "${TARGET_EPSILON}"
  --dummy_k "${DUMMY_K}"
  --dummy_ratio "${DUMMY_RATIO}"
  --epsilon_non_sensitive "${TARGET_EPSILON}"
  --epsilon_private "${EPSILON_PRIVATE}"
  --clip_norm 1.0
  --transport_model "${TRANSPORT_MODEL}"
  --transport_dtype bfloat16
  --quality_model "${TRANSPORT_MODEL}"
  --quality_dtype bfloat16
  --mia_attack_model "${TRANSPORT_MODEL}"
  --mia_attack_dtype bfloat16
  --mia_max_length "${MAX_LENGTH}"
  --local_files_only true
  --output_dir "${OUTPUT_DIR}"
  --overwrite true
)

launch () {
  local tag="$1"
  shift
  echo "[Launch] ${tag} -> ${LOG_DIR}/${tag}.log"
  python experiments/proxy_scale/run_proxy_generator_scale.py \
    "${COMMON[@]}" "$@" > "${LOG_DIR}/${tag}.log" 2>&1 &
  PIDS+=("$!")
  TAGS+=("${tag}")
}

PIDS=()
TAGS=()

wait_for_phase () {
  local phase_status=0
  local index
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[${index}]}"; then
      echo "[Failed] ${TAGS[${index}]}; inspect ${LOG_DIR}/${TAGS[${index}]}.log" >&2
      phase_status=1
    else
      echo "[Done] ${TAGS[${index}]}"
    fi
  done
  PIDS=()
  TAGS=()
  return "${phase_status}"
}

# Each of the first three runs is deliberately self-contained.  Its fixed
# scorer is the same 0.5B model as its transport role, so the Python runner
# reuses one in-memory object rather than loading three copies.
launch qwen25_05b \
  --proxy_model "${TRANSPORT_MODEL}" --proxy_label 0.5B --proxy_size_b 0.5 \
  --proxy_dtype bfloat16 --proxy_device cuda:0 --proxy_device_map none \
  --transport_device cuda:0 --quality_device cuda:0 --mia_device cuda:0 \
  --output_prefix "wikitext_qwen25_0.5B_eps${TARGET_EPSILON}_k${DUMMY_K}_r${DUMMY_RATIO}"

launch qwen25_3b \
  --proxy_model "${MODEL_ROOT}/Qwen2.5-3B-Instruct" --proxy_label 3B --proxy_size_b 3 \
  --proxy_dtype bfloat16 --proxy_device cuda:1 --proxy_device_map none \
  --transport_device cuda:1 --quality_device cuda:1 --mia_device cuda:1 \
  --output_prefix "wikitext_qwen25_3B_eps${TARGET_EPSILON}_k${DUMMY_K}_r${DUMMY_RATIO}"

launch qwen25_7b \
  --proxy_model "${MODEL_ROOT}/Qwen2.5-7B-Instruct" --proxy_label 7B --proxy_size_b 7 \
  --proxy_dtype bfloat16 --proxy_device cuda:2 --proxy_device_map none \
  --transport_device cuda:2 --quality_device cuda:2 --mia_device cuda:2 \
  --output_prefix "wikitext_qwen25_7B_eps${TARGET_EPSILON}_k${DUMMY_K}_r${DUMMY_RATIO}"

if ! wait_for_phase; then
  exit 1
fi

# A 32B bfloat16 checkpoint does not fit on a typical 48-GiB card.  Run it
# only after phase 1: shard the proxy across GPUs 0 and 1, then put the fixed
# 0.5B transport/quality/MIA scorer on GPU 2. For 80-GiB cards, e.g. export
# GPU32_MAX_MEMORY='0:72GiB,1:72GiB' before launching this script.
launch qwen25_32b \
  --proxy_model "${MODEL_ROOT}/Qwen2.5-32B-Instruct" --proxy_label 32B --proxy_size_b 32 \
  --proxy_dtype bfloat16 --proxy_device "${GPU32_PROXY_DEVICE}" --proxy_device_map auto --proxy_max_memory "${GPU32_MAX_MEMORY}" \
  --transport_device "${GPU32_SCORER_DEVICE}" --quality_device "${GPU32_SCORER_DEVICE}" --mia_device "${GPU32_SCORER_DEVICE}" \
  --output_prefix "wikitext_qwen25_32B_eps${TARGET_EPSILON}_k${DUMMY_K}_r${DUMMY_RATIO}"

if ! wait_for_phase; then
  exit 1
fi

python experiments/proxy_scale/plot_proxy_generator_scale.py \
  --summary_glob "${OUTPUT_DIR}/*_summary.csv" \
  --output "${OUTPUT_DIR}/qwen25_proxy_generator_scale.png"

echo "[Saved] summaries and per-group metrics: ${OUTPUT_DIR}"
echo "[Saved] scale figure: ${OUTPUT_DIR}/qwen25_proxy_generator_scale.png"
