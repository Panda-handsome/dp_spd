#!/usr/bin/env bash
# Run the 7B and 32B Qwen2.5 proxy scales after 0.5B and 3B.
# Four-GPU layout:
#   GPU 0: 7B proxy + its shared fixed 0.5B transport/quality/MIA scorer
#   GPUs 1,2: 32B proxy with device_map=auto
#   GPU 3: fixed 0.5B transport/quality/MIA scorer for the 32B run

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

MODEL_ROOT="${MODEL_ROOT:-models}"
DATASET_PATH="${DATASET_PATH:-${REPO_DIR}/data/wikitext/wikitext-2-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/proxy_generator_scale}"
LOG_DIR="${OUTPUT_DIR}/logs"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DUMMY_K="${DUMMY_K:-9}"
DUMMY_RATIO="${DUMMY_RATIO:-0.2}"
TARGET_EPSILON="${TARGET_EPSILON:-3.0}"
EPSILON_PRIVATE="${EPSILON_PRIVATE:-7.82}"

GPU7_DEVICE="${GPU7_DEVICE:-cuda:0}"
GPU32_PROXY_DEVICE="${GPU32_PROXY_DEVICE:-cuda:1}"
GPU32_SCORER_DEVICE="${GPU32_SCORER_DEVICE:-cuda:3}"
GPU32_MAX_MEMORY="${GPU32_MAX_MEMORY:-1:44GiB,2:44GiB}"

TRANSPORT_MODEL="${MODEL_ROOT}/Qwen2.5-0.5B-Instruct"
for model_path in \
  "${TRANSPORT_MODEL}" \
  "${MODEL_ROOT}/Qwen2.5-7B-Instruct" \
  "${MODEL_ROOT}/Qwen2.5-32B-Instruct"; do
  if [[ ! -d "${model_path}" ]]; then
    echo "Missing model directory: ${model_path}" >&2
    exit 2
  fi
done
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "Missing WikiText-2 dataset directory: ${DATASET_PATH}" >&2
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
  --noise_mechanism Laplace
  --seed 42
  --projection_chunk_size 8
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

launch qwen25_7b \
  --proxy_model "${MODEL_ROOT}/Qwen2.5-7B-Instruct" --proxy_label 7B --proxy_size_b 7 \
  --proxy_dtype bfloat16 --proxy_device "${GPU7_DEVICE}" --proxy_device_map none \
  --transport_device "${GPU7_DEVICE}" --quality_device "${GPU7_DEVICE}" --mia_device "${GPU7_DEVICE}" \
  --output_prefix "wikitext_qwen25_7B_eps${TARGET_EPSILON}_k${DUMMY_K}_r${DUMMY_RATIO}"

launch qwen25_32b \
  --proxy_model "${MODEL_ROOT}/Qwen2.5-32B-Instruct" --proxy_label 32B --proxy_size_b 32 \
  --proxy_dtype bfloat16 --proxy_device "${GPU32_PROXY_DEVICE}" --proxy_device_map auto --proxy_max_memory "${GPU32_MAX_MEMORY}" \
  --transport_device "${GPU32_SCORER_DEVICE}" --quality_device "${GPU32_SCORER_DEVICE}" --mia_device "${GPU32_SCORER_DEVICE}" \
  --output_prefix "wikitext_qwen25_32B_eps${TARGET_EPSILON}_k${DUMMY_K}_r${DUMMY_RATIO}"

status=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[${index}]}"; then
    echo "[Failed] ${TAGS[${index}]}; inspect ${LOG_DIR}/${TAGS[${index}]}.log" >&2
    status=1
  else
    echo "[Done] ${TAGS[${index}]}"
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

python experiments/proxy_scale/plot_proxy_generator_scale.py \
  --summary_glob "${OUTPUT_DIR}/*_summary.csv" \
  --output "${OUTPUT_DIR}/qwen25_proxy_generator_scale.png"

echo "[Saved] results and figure: ${OUTPUT_DIR}"
