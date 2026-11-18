#!/usr/bin/env bash
set -euo pipefail

# Qwen3-14B / CNN-DailyMail baseline sweep for DP-SPD Figure 4(b).
# Each physical 48-GB GPU runs one complete Qwen3-14B process.  Do not run a
# second process on the same GPU: the model itself occupies roughly 28 GB in
# bfloat16 before activations and method-specific mapping buffers.
#
# Required remote configuration example:
#   MODEL_PATH=models/Qwen3-14B \
#   DATASET_PATH=/path/to/the/same/cnn_dailymail/DatasetDict \
#   CONDA_ENV=dpllm bash scripts/run_qwen3_cnndm_baselines.sh
#
# Optional selective rerun (set CONTROL_EPSILON=none to omit the diagnostic
# large-epsilon control):
#   METHODS='cape,custext' EPSILONS='1.0,1.5,2.0' CONTROL_EPSILON=10000 RESUME=false \
#   CONDA_ENV=dpllm bash scripts/run_qwen3_cnndm_baselines.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-models/Qwen3-14B}"
DATASET_PATH="${DATASET_PATH:-data/cnn_dailymail}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
METHODS="${METHODS:-cape,custext,dp_forward,inferdpt,santext,textobfuscator}"
EPSILONS="${EPSILONS:-1.0,1.5,2.0}"
# This is deliberately a separate control, not an additional Figure-4(b)
# privacy budget.  It measures the implementation's large-epsilon limit for
# every adapter.  CAPE's min--max transformation and DP-Forward's clipping
# remain active at this limit, so it is not labelled a clean-model upper bound.
CONTROL_EPSILON="${CONTROL_EPSILON:-10000}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-1,2}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-42}"
MAPPING_MAX_SAMPLES="${MAPPING_MAX_SAMPLES:-0}"
SHUFFLE_SAMPLES="${SHUFFLE_SAMPLES:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/baselines/qwen3_cnndm}"
RESUME="${RESUME:-true}"
OVERWRITE="${OVERWRITE:-false}"

IFS=',' read -r -a METHOD_LIST <<< "${METHODS}"
IFS=',' read -r -a EPS_LIST <<< "${EPSILONS}"
IFS=',' read -r -a GPU_LIST <<< "${PHYSICAL_GPUS}"
if [[ "${#GPU_LIST[@]}" -ne 2 ]]; then
  echo "PHYSICAL_GPUS must contain exactly two physical GPU indices, e.g. 1,2." >&2
  exit 2
fi
if [[ "${#EPS_LIST[@]}" -ne 3 ]]; then
  echo "This Figure-4(b) launcher requires exactly three privacy budgets: 1.0,1.5,2.0." >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" || ! -d "${DATASET_PATH}" ]]; then
  echo "MODEL_PATH or DATASET_PATH does not exist:" >&2
  echo "  MODEL_PATH=${MODEL_PATH}" >&2
  echo "  DATASET_PATH=${DATASET_PATH}" >&2
  exit 2
fi

if [[ -n "${CONDA_ENV}" ]]; then
  RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  RUNNER=("${PYTHON_BIN}")
fi

COMMON=(
  baseline/run_qwen3_cnndm_baseline.py
  --model_path "${MODEL_PATH}"
  --dataset_path "${DATASET_PATH}"
  --device cuda:0
  --dtype bfloat16
  --max_samples "${MAX_SAMPLES}"
  --max_prompt_length 768
  --max_ref_tokens 160
  --rouge_context_len 768
  --max_new_tokens 128
  --batch_size "${BATCH_SIZE}"
  --seed "${SEED}"
  --shuffle_samples "${SHUFFLE_SAMPLES}"
  --mapping_splits train
  --mapping_max_samples "${MAPPING_MAX_SAMPLES}"
  --overwrite "${OVERWRITE}"
)

validate_method() {
  case "$1" in
    cape|custext|dp_forward|inferdpt|santext|textobfuscator) ;;
    *) echo "Unknown method: $1" >&2; exit 2 ;;
  esac
}

prepare_once() {
  local method="$1"
  case "${method}" in
    santext|inferdpt)
      echo "[Prepare] ${method}: building/reusing its shared public mapping on physical GPU ${GPU_LIST[0]}."
      env CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "${RUNNER[@]}" "${COMMON[@]}" \
        --method "${method}" --epsilon "${EPS_LIST[0]}" --prepare_only true \
        --output_csv "${OUTPUT_ROOT}/${method}/prepare_marker.csv"
      ;;
    *) ;;
  esac
}

launch_case() {
  local method="$1"
  local epsilon="$2"
  local physical_gpu="$3"
  local output="${OUTPUT_ROOT}/${method}/${method}_cnndm_eps${epsilon}.csv"
  local log="${OUTPUT_ROOT}/${method}/logs/${method}_cnndm_eps${epsilon}_gpu${physical_gpu}.log"
  mkdir -p "$(dirname "${output}")" "$(dirname "${log}")"
  if [[ "${RESUME}" == "true" && -s "${output}" ]]; then
    echo "[Skip] existing result: ${output}"
    LAST_PID=""
    return
  fi
  echo "[Launch] method=${method} epsilon=${epsilon} physical_gpu=${physical_gpu}"
  env CUDA_VISIBLE_DEVICES="${physical_gpu}" "${RUNNER[@]}" "${COMMON[@]}" \
    --method "${method}" --epsilon "${epsilon}" --output_csv "${output}" \
    >"${log}" 2>&1 &
  LAST_PID="$!"
}

for method in "${METHOD_LIST[@]}"; do
  validate_method "${method}"
  prepare_once "${method}"

  # Two 48-GB GPUs: epsilon 1.0 and 1.5 run together, then epsilon 2.0 and
  # the large-epsilon diagnostic control run together.  Static mappings are
  # prepared above to prevent cache races.
  launch_case "${method}" "${EPS_LIST[0]}" "${GPU_LIST[0]}"; pid_a="${LAST_PID}"
  launch_case "${method}" "${EPS_LIST[1]}" "${GPU_LIST[1]}"; pid_b="${LAST_PID}"
  if [[ "${pid_a}" =~ ^[0-9]+$ ]] && ! wait "${pid_a}"; then
    echo "[Failed] ${method} epsilon=${EPS_LIST[0]} (see ${OUTPUT_ROOT}/${method}/logs)" >&2
    exit 1
  fi
  if [[ "${pid_b}" =~ ^[0-9]+$ ]] && ! wait "${pid_b}"; then
    echo "[Failed] ${method} epsilon=${EPS_LIST[1]} (see ${OUTPUT_ROOT}/${method}/logs)" >&2
    exit 1
  fi
  launch_case "${method}" "${EPS_LIST[2]}" "${GPU_LIST[0]}"; pid_c="${LAST_PID}"
  pid_d=""
  if [[ -n "${CONTROL_EPSILON}" && "${CONTROL_EPSILON}" != "none" ]]; then
    launch_case "${method}" "${CONTROL_EPSILON}" "${GPU_LIST[1]}"; pid_d="${LAST_PID}"
  fi
  if [[ "${pid_c}" =~ ^[0-9]+$ ]] && ! wait "${pid_c}"; then
    echo "[Failed] ${method} epsilon=${EPS_LIST[2]} (see ${OUTPUT_ROOT}/${method}/logs)" >&2
    exit 1
  fi
  if [[ "${pid_d}" =~ ^[0-9]+$ ]] && ! wait "${pid_d}"; then
    echo "[Failed] ${method} epsilon=${CONTROL_EPSILON} (see ${OUTPUT_ROOT}/${method}/logs)" >&2
    exit 1
  fi
done

"${RUNNER[@]}" baseline/aggregate_qwen3_cnndm_results.py \
  --input_glob "${OUTPUT_ROOT}/*/*_cnndm_eps*.csv" \
  --output_csv "${OUTPUT_ROOT}/summary.csv"
echo "[Done] Per-method CSVs and summary are in ${OUTPUT_ROOT}/"
