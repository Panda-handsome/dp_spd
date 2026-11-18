#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cuda:0}"
API_MODEL="${API_MODEL:-gpt-4o}"
BACKEND="${BACKEND:-openai_chat}"
TASK="${TASK:-mmlu_pro}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-16}"
VARIANTS="${VARIANTS:-clean_embedding_projection,dpspd_projected}"
SEEDS="${SEEDS:-42 43 44 45 46}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SHARED_PLAIN_BASELINE="${SHARED_PLAIN_BASELINE:-false}"
PLAIN_BASELINE_VARIANTS="${PLAIN_BASELINE_VARIANTS:-plain_text}"
PROJECTION_VARIANTS="${PROJECTION_VARIANTS:-${VARIANTS}}"
AGGREGATE_AFTER="${AGGREGATE_AFTER:-true}"
STORE_PROJECTION_TOKEN_DIAGNOSTICS="${STORE_PROJECTION_TOKEN_DIAGNOSTICS:-false}"
PROJECTION_TIE_TOLERANCE="${PROJECTION_TIE_TOLERANCE:-1e-6}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/api_only_projection_ablation}"
OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE:-gpt4o_mmlu_pro_proxy_embedding_bias}"

QWEN_PROJECTION_MODEL="${QWEN_PROJECTION_MODEL:-models/Qwen2.5-0.5B-Instruct}"
GEMMA_PROJECTION_MODEL="${GEMMA_PROJECTION_MODEL:-models/gemma-2-2b-it}"
MODEL_SPECS="${MODEL_SPECS:-qwen=${QWEN_PROJECTION_MODEL},gemma=${GEMMA_PROJECTION_MODEL}}"

DATASET_ARGS=()
if [[ -n "${DATASET_PATH:-}" ]]; then
  DATASET_ARGS+=(--dataset_path "${DATASET_PATH}")
fi

MODEL_LABELS=()
IFS=',' read -ra SPECS <<< "${MODEL_SPECS}"
for spec in "${SPECS[@]}"; do
  label="${spec%%=*}"
  model_path="${spec#*=}"
  if [[ -z "${label}" || -z "${model_path}" || "${label}" == "${model_path}" ]]; then
    echo "[Error] Invalid model spec: ${spec}. Use label=/path/to/model." >&2
    exit 2
  fi
  if [[ ! -d "${model_path}" ]]; then
    echo "[Error] Projection model for '${label}' not found: ${model_path}" >&2
    echo "Set MODEL_SPECS='qwen=/path/to/qwen,gemma=/path/to/gemma' or GEMMA_PROJECTION_MODEL=/path/to/gemma-2-2b-it." >&2
    exit 2
  fi
  MODEL_LABELS+=("${label}")
done

FIRST_MODEL_PATH="${SPECS[0]#*=}"

for seed in ${SEEDS}; do
  reuse_outputs_jsonl="${REUSE_OUTPUTS_JSONL:-}"
  if [[ "${SHARED_PLAIN_BASELINE}" == "true" ]]; then
    plain_prefix="${OUTPUT_PREFIX_BASE}_plain_seed${seed}"
    echo "[Run] shared plain seed=${seed} prefix=${plain_prefix}"
    STORE_PROJECTION_TOKEN_DIAGNOSTICS="${STORE_PROJECTION_TOKEN_DIAGNOSTICS}" \
    PROJECTION_TIE_TOLERANCE="${PROJECTION_TIE_TOLERANCE}" \
    SAMPLE_SEED="${SAMPLE_SEED}" \
    bash scripts/run_api_only_projection_ablation.sh \
      --projection_model "${FIRST_MODEL_PATH}" \
      --device "${DEVICE}" \
      --backend "${BACKEND}" \
      --api_model "${API_MODEL}" \
      --task "${TASK}" \
      --dataset_split "${DATASET_SPLIT}" \
      --max_samples "${MAX_SAMPLES}" \
      --variants "${PLAIN_BASELINE_VARIANTS}" \
      --max_completion_tokens "${MAX_COMPLETION_TOKENS}" \
      --seed "${seed}" \
      --output_dir "${OUTPUT_DIR}" \
      --output_prefix "${plain_prefix}" \
      "${DATASET_ARGS[@]}" \
      "$@"
    reuse_outputs_jsonl="${OUTPUT_DIR}/${plain_prefix}_outputs.jsonl"
  fi

  IFS=',' read -ra SPECS <<< "${MODEL_SPECS}"
  for spec in "${SPECS[@]}"; do
    label="${spec%%=*}"
    model_path="${spec#*=}"
    output_prefix="${OUTPUT_PREFIX_BASE}_${label}_seed${seed}"
    echo "[Run] model=${label} seed=${seed} prefix=${output_prefix}"
    REUSE_OUTPUTS_JSONL="${reuse_outputs_jsonl}" \
    STORE_PROJECTION_TOKEN_DIAGNOSTICS="${STORE_PROJECTION_TOKEN_DIAGNOSTICS}" \
    PROJECTION_TIE_TOLERANCE="${PROJECTION_TIE_TOLERANCE}" \
    SAMPLE_SEED="${SAMPLE_SEED}" \
    bash scripts/run_api_only_projection_ablation.sh \
      --projection_model "${model_path}" \
      --device "${DEVICE}" \
      --backend "${BACKEND}" \
      --api_model "${API_MODEL}" \
      --task "${TASK}" \
      --dataset_split "${DATASET_SPLIT}" \
      --max_samples "${MAX_SAMPLES}" \
      --variants "${PROJECTION_VARIANTS}" \
      --max_completion_tokens "${MAX_COMPLETION_TOKENS}" \
      --seed "${seed}" \
      --output_dir "${OUTPUT_DIR}" \
      --output_prefix "${output_prefix}" \
      "${DATASET_ARGS[@]}" \
      "$@"
  done
done

if [[ "${SHARED_PLAIN_BASELINE}" == "true" ]]; then
  MODEL_LABELS=("plain" "${MODEL_LABELS[@]}")
fi
MODEL_LABELS_CSV="$(IFS=','; echo "${MODEL_LABELS[*]}")"
SEEDS_CSV="$(echo "${SEEDS}" | tr ' ' ',')"
if [[ "${AGGREGATE_AFTER}" == "true" ]]; then
  python experiments/api_only/summarize_proxy_embedding_bias.py \
    --output_dir "${OUTPUT_DIR}" \
    --prefix_base "${OUTPUT_PREFIX_BASE}" \
    --model_labels "${MODEL_LABELS_CSV}" \
    --seeds "${SEEDS_CSV}"
fi
