#!/usr/bin/env bash
set -euo pipefail

# Four independent full-sequence SanText-style runs, one on each requested GPU.
# The default is *SanText* (not SanText+): every active non-special LLaMA BPE
# token is sampled by the paper's Euclidean exponential mechanism.  This is
# the suitable Figure-3 baseline when all tokens must receive sanitization.
#
# This is a LLaMA-token adapter, not the paper's original word-level/BERT
# experiment.  It uses the same clean-label PPL and clean-continuation
# ROUGE-L functions as DP-SPD.  SanText has no dummy ratio r or dummy count k,
# so plot the result as one horizontal baseline per epsilon.
#
# Smoke test (from the repository root after `source .venv/bin/activate`):
#   MAX_SAMPLES=8 MAPPING_MAX_SAMPLES=256 bash scripts/run_santext_generation_sweep.sh
# Full WikiText-2 run:
#   MAX_SAMPLES=300 bash scripts/run_santext_generation_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EPSILONS=(1.5 2.0 2.5 3.0)
GPUS=(4 5 6 7)
MODEL_PATH="${MODEL_PATH:-models/Llama-2-7b-chat-hf}"
DATASET_PATH="${DATASET_PATH:-data/wikitext/wikitext-2-v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VARIANT="${VARIANT:-santext}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CONTEXT_LEN="${CONTEXT_LEN:-32}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-current_dpspd}"
ROUGE_USE_STEMMER="${ROUGE_USE_STEMMER:-true}"
MAPPING_SPLITS="${MAPPING_SPLITS:-train,validation}"
MAPPING_MAX_SAMPLES="${MAPPING_MAX_SAMPLES:-0}"
MAPPING_BATCH_SIZE="${MAPPING_BATCH_SIZE:-128}"
MAPPING_CACHE="${MAPPING_CACHE:-}"
PREBUILD_MAPPING="${PREBUILD_MAPPING:-true}"
SENSITIVE_FRACTION="${SENSITIVE_FRACTION:-0.9}"
NON_SENSITIVE_SANITIZE_PROBABILITY="${NON_SENSITIVE_SANITIZE_PROBABILITY:-0.3}"
SOURCE_CHUNK_SIZE="${SOURCE_CHUNK_SIZE:-96}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baselines/santext/wikitext_figure3_${VARIANT}}"
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_DIR}/logs"
pids=()
mapping_cache_args=()
python_cmd=("${PYTHON_BIN}")
if [[ -n "${MAPPING_CACHE}" ]]; then
  mapping_cache_args=(--mapping_cache "${MAPPING_CACHE}")
fi

# V and Vs do not depend on epsilon.  Construct one shared cache before
# loading four copies of LLaMA, which avoids four redundant train/dev scans
# and simultaneous writes to the same mapping file.
if [[ "${PREBUILD_MAPPING}" == "true" ]]; then
  "${python_cmd[@]}" baseline/santext/run_generation_figure3.py \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --variant "${VARIANT}" \
    --epsilon "${EPSILONS[0]}" \
    --device cpu \
    --mapping_splits "${MAPPING_SPLITS}" \
    --mapping_max_samples "${MAPPING_MAX_SAMPLES}" \
    --mapping_batch_size "${MAPPING_BATCH_SIZE}" \
    --sensitive_fraction "${SENSITIVE_FRACTION}" \
    --non_sensitive_sanitize_probability "${NON_SENSITIVE_SANITIZE_PROBABILITY}" \
    --source_chunk_size "${SOURCE_CHUNK_SIZE}" \
    --seed "${SEED}" \
    "${mapping_cache_args[@]}" \
    --build_mapping_only true
fi

for index in "${!EPSILONS[@]}"; do
  epsilon="${EPSILONS[$index]}"
  gpu="${GPUS[$index]}"
  output_csv="${OUTPUT_DIR}/santext_${VARIANT}_wikitext_eps${epsilon}.csv"
  log_file="${OUTPUT_DIR}/logs/santext_${VARIANT}_wikitext_eps${epsilon}_gpu${gpu}.log"
  "${python_cmd[@]}" baseline/santext/run_generation_figure3.py \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --variant "${VARIANT}" \
    --epsilon "${epsilon}" \
    --device "cuda:${gpu}" \
    --mapping_splits "${MAPPING_SPLITS}" \
    --mapping_max_samples "${MAPPING_MAX_SAMPLES}" \
    --mapping_batch_size "${MAPPING_BATCH_SIZE}" \
    "${mapping_cache_args[@]}" \
    --sensitive_fraction "${SENSITIVE_FRACTION}" \
    --non_sensitive_sanitize_probability "${NON_SENSITIVE_SANITIZE_PROBABILITY}" \
    --source_chunk_size "${SOURCE_CHUNK_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --selection_protocol "${SELECTION_PROTOCOL}" \
    --batch_size "${BATCH_SIZE}" \
    --rouge_context_len "${CONTEXT_LEN}" \
    --max_length "${MAX_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --rouge_use_stemmer "${ROUGE_USE_STEMMER}" \
    --seed "${SEED}" \
    --output_csv "${output_csv}" \
    "$@" >"${log_file}" 2>&1 &
  pids+=("$!")
  echo "[Launched] variant=${VARIANT} epsilon=${epsilon} gpu=cuda:${gpu} pid=${pids[-1]} log=${log_file}"
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[Failed] variant=${VARIANT} epsilon=${EPSILONS[$index]} gpu=cuda:${GPUS[$index]}" >&2
    status=1
  else
    echo "[Completed] variant=${VARIANT} epsilon=${EPSILONS[$index]} gpu=cuda:${GPUS[$index]}"
  fi
done
exit "${status}"
