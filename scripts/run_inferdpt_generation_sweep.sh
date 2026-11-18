#!/usr/bin/env bash
set -euo pipefail

# Four independent InferDPT/RANTEXT generation runs.  The remote and local
# extractor roles are executed sequentially by the same frozen LLaMA-2-7B
# test-harness model on each GPU.  Only the perturbed prompt reaches the
# remote role; the raw prefix remains local for extraction, as in InferDPT.
#
# This is not the original GPT-4 + text-embedding-ada-002 setting.  Its final
# ROUGE-L measures the completed extraction output.  The optional PPL is
# explicitly labelled extractor-conditional PPL and must not be substituted
# for DP-SPD's server clean-label Figure-3 PPL.
#
# Smoke test:
#   MAX_SAMPLES=4 REMOTE_MAX_NEW_TOKENS=8 MAX_NEW_TOKENS=8 \
#   bash scripts/run_inferdpt_generation_sweep.sh
# Full local-LLaMA comparison:
#   MAX_SAMPLES=300 bash scripts/run_inferdpt_generation_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EPSILONS=(1.5 2.0 2.5 3.0)
GPUS=(4 5 6 7)
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-models/Llama-2-7b-chat-hf}"
DATASET_PATH="${DATASET_PATH:-data/wikitext/wikitext-2-v1}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PROMPT_TOKENS="${PROMPT_TOKENS:-32}"
REMOTE_MAX_NEW_TOKENS="${REMOTE_MAX_NEW_TOKENS:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
MAX_EXTRACTOR_PROMPT_TOKENS="${MAX_EXTRACTOR_PROMPT_TOKENS:-512}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-current_dpspd}"
ROUGE_USE_STEMMER="${ROUGE_USE_STEMMER:-true}"
MAPPING_SPLITS="${MAPPING_SPLITS:-train,validation}"
MAPPING_MAX_SAMPLES="${MAPPING_MAX_SAMPLES:-0}"
MAPPING_BATCH_SIZE="${MAPPING_BATCH_SIZE:-128}"
CANDIDATE_VOCAB_SIZE="${CANDIDATE_VOCAB_SIZE:-11000}"
MAPPING_CACHE="${MAPPING_CACHE:-}"
PREBUILD_MAPPING="${PREBUILD_MAPPING:-true}"
SOURCE_CHUNK_SIZE="${SOURCE_CHUNK_SIZE:-64}"
RUN_EXTRACTOR_PPL="${RUN_EXTRACTOR_PPL:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baselines/inferdpt/wikitext_local_llama}"
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_DIR}/logs"
pids=()
mapping_cache_args=()
if [[ -n "${MAPPING_CACHE}" ]]; then
  mapping_cache_args=(--mapping_cache "${MAPPING_CACHE}")
fi

# The public token vocabulary does not depend on epsilon, so build exactly one
# shared cache before launching GPU workers.
if [[ "${PREBUILD_MAPPING}" == "true" ]]; then
  "${PYTHON_BIN}" baseline/inferdpt/run_generation.py \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --epsilon "${EPSILONS[0]}" \
    --device cpu \
    --mapping_splits "${MAPPING_SPLITS}" \
    --mapping_max_samples "${MAPPING_MAX_SAMPLES}" \
    --mapping_batch_size "${MAPPING_BATCH_SIZE}" \
    --candidate_vocab_size "${CANDIDATE_VOCAB_SIZE}" \
    "${mapping_cache_args[@]}" \
    --build_mapping_only true
fi

for index in "${!EPSILONS[@]}"; do
  epsilon="${EPSILONS[$index]}"
  gpu="${GPUS[$index]}"
  output_csv="${OUTPUT_DIR}/inferdpt_wikitext_eps${epsilon}.csv"
  log_file="${OUTPUT_DIR}/logs/inferdpt_wikitext_eps${epsilon}_gpu${gpu}.log"
  "${PYTHON_BIN}" baseline/inferdpt/run_generation.py \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --epsilon "${epsilon}" \
    --device "cuda:${gpu}" \
    --mapping_splits "${MAPPING_SPLITS}" \
    --mapping_max_samples "${MAPPING_MAX_SAMPLES}" \
    --mapping_batch_size "${MAPPING_BATCH_SIZE}" \
    --candidate_vocab_size "${CANDIDATE_VOCAB_SIZE}" \
    "${mapping_cache_args[@]}" \
    --source_chunk_size "${SOURCE_CHUNK_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --prompt_tokens "${PROMPT_TOKENS}" \
    --remote_max_new_tokens "${REMOTE_MAX_NEW_TOKENS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_extractor_prompt_tokens "${MAX_EXTRACTOR_PROMPT_TOKENS}" \
    --selection_protocol "${SELECTION_PROTOCOL}" \
    --rouge_use_stemmer "${ROUGE_USE_STEMMER}" \
    --run_extractor_ppl "${RUN_EXTRACTOR_PPL}" \
    --seed "${SEED}" \
    --output_csv "${output_csv}" \
    "$@" >"${log_file}" 2>&1 &
  pids+=("$!")
  echo "[Launched] epsilon=${epsilon} gpu=cuda:${gpu} pid=${pids[-1]} log=${log_file}"
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[Failed] epsilon=${EPSILONS[$index]} gpu=cuda:${GPUS[$index]}" >&2
    status=1
  else
    echo "[Completed] epsilon=${EPSILONS[$index]} gpu=cuda:${GPUS[$index]}"
  fi
done
exit "${status}"
