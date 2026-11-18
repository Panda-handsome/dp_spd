#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cuda:0}"
SERVER_MODEL="${SERVER_MODEL:-models/Llama-2-7b-chat-hf}"
SERVER_DTYPE="${SERVER_DTYPE:-float16}"
TASK="${TASK:-mmlu_pro}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
MAX_LENGTH="${MAX_LENGTH:-512}"
PROJECTION_CHUNK_SIZE="${PROJECTION_CHUNK_SIZE:-8}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-16}"
VARIANTS="${VARIANTS:-clean,dpspd}"
SEEDS="${SEEDS:-42 43 44 45 46}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
CLEAN_ONCE="${CLEAN_ONCE:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/whitebox_blackbox_gap}"
OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE:-llama2_mmlu_pro_whitebox_blackbox_gap}"
AGGREGATE_AFTER="${AGGREGATE_AFTER:-true}"

DATASET_ARGS=()
if [[ -n "${DATASET_PATH:-}" ]]; then
  DATASET_ARGS+=(--dataset_path "${DATASET_PATH}")
fi

without_clean_variants() {
  python -c "parts=[p.strip() for p in '$1'.replace(',', ' ').split() if p.strip() and p.strip() != 'clean']; print(','.join(parts))"
}

first_seed="true"
for seed in ${SEEDS}; do
  output_prefix="${OUTPUT_PREFIX_BASE}_seed${seed}"
  run_variants="${VARIANTS}"
  if [[ "${CLEAN_ONCE}" == "true" && "${first_seed}" != "true" ]]; then
    run_variants="$(without_clean_variants "${VARIANTS}")"
    if [[ -z "${run_variants}" ]]; then
      echo "[Skip] seed=${seed} because CLEAN_ONCE=true and no non-clean variants remain"
      first_seed="false"
      continue
    fi
  fi
  echo "[Run] seed=${seed} prefix=${output_prefix}"
  bash scripts/run_whitebox_blackbox_gap.sh \
    --server_model "${SERVER_MODEL}" \
    --server_dtype "${SERVER_DTYPE}" \
    --device "${DEVICE}" \
    --task "${TASK}" \
    --dataset_split "${DATASET_SPLIT}" \
    --max_samples "${MAX_SAMPLES}" \
    --variants "${run_variants}" \
    --max_length "${MAX_LENGTH}" \
    --projection_chunk_size "${PROJECTION_CHUNK_SIZE}" \
    --max_completion_tokens "${MAX_COMPLETION_TOKENS}" \
    --temperature 0 \
    --seed "${seed}" \
    --sample_seed "${SAMPLE_SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_prefix "${output_prefix}" \
    "${DATASET_ARGS[@]}" \
    "$@"
  first_seed="false"
done

SEEDS_CSV="$(echo "${SEEDS}" | tr ' ' ',')"
if [[ "${AGGREGATE_AFTER}" == "true" ]]; then
  python experiments/api_only/summarize_whitebox_blackbox_gap.py \
    --output_dir "${OUTPUT_DIR}" \
    --prefix_base "${OUTPUT_PREFIX_BASE}" \
    --seeds "${SEEDS_CSV}"
fi
