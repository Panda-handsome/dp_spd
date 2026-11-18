#!/usr/bin/env bash
set -euo pipefail

# Figure-4 adapter sweep for one baseline and one instruction dataset.
# Example:
#   MAX_SAMPLES=300 bash scripts/run_figure4_baseline_generation.sh santext alpaca
# The paper's Figure 4a budgets are 1.5, 2.5, and 4.0. Set GPU_IDS to
# three comma-separated physical GPU indices (default: 0,1,2).

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {cape|custext|dp_forward|inferdpt|santext|textobfuscator} {alpaca|dolly} [extra runner args...]" >&2
  exit 2
fi

BASELINE="$1"
DATASET="$2"
shift 2
case "${DATASET}" in alpaca|dolly) ;; *) echo "Dataset must be alpaca or dolly" >&2; exit 2 ;; esac
case "${BASELINE}" in cape|custext|dp_forward|inferdpt|santext|textobfuscator) ;; *) echo "Unknown baseline ${BASELINE}" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

case "${DATASET}" in
  alpaca) DATASET_PATH="${DATASET_PATH:-data/alpaca}" ;;
  dolly) DATASET_PATH="${DATASET_PATH:-data/databricks_dolly}" ;;
esac
EPSILONS=(1.5 2.5 4.0)
IFS=',' read -r -a GPUS <<< "${GPU_IDS:-0,1,2}"
if [[ "${#GPUS[@]}" -ne "${#EPSILONS[@]}" ]]; then
  echo "GPU_IDS must contain three comma-separated GPU indices." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-models/Llama-2-7b-chat-hf}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
MAX_LENGTH="${MAX_LENGTH:-256}"
# These default values reproduce dp_generate/results.py and results2.py.
BATCH_SIZE="${BATCH_SIZE:-32}"
ROUGE_CONTEXT_LEN="${ROUGE_CONTEXT_LEN:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-legacy_dp_generate}"
ROUGE_USE_STEMMER="${ROUGE_USE_STEMMER:-false}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baselines/figure4/${BASELINE}/${DATASET}}"
mkdir -p "${OUTPUT_DIR}/logs"

case "${BASELINE}" in
  cape) RUNNER="baseline/cape/run_generation_figure3.py"; EXTRA=() ;;
  custext) RUNNER="baseline/custext/run_generation_figure3.py"; EXTRA=() ;;
  dp_forward) RUNNER="baseline/dp_forward/run_generation_figure3.py"; EXTRA=(--delta 1e-5 --norm_c 1.0 --noise_layer 28) ;;
  santext) RUNNER="baseline/santext/run_generation_figure3.py"; EXTRA=() ;;
  textobfuscator) RUNNER="baseline/textobfuscator/run_generation_figure3.py"; EXTRA=() ;;
  inferdpt) RUNNER="baseline/inferdpt/run_generation.py"; EXTRA=(--prompt_tokens 32 --remote_max_new_tokens "${MAX_NEW_TOKENS}") ;;
esac

# Methods that build a public mapping/prototype must do it once before the
# concurrent epsilon jobs, preventing cache races and giving every epsilon the
# identical public object.
case "${BASELINE}" in
  santext)
    "${PYTHON_BIN}" "${RUNNER}" --model_path "${MODEL_PATH}" --dataset_name "${DATASET}" --dataset_path "${DATASET_PATH}" --epsilon 1.5 --device "cuda:${GPUS[0]}" --selection_protocol "${SELECTION_PROTOCOL}" --build_mapping_only true "${EXTRA[@]}" "$@"
    ;;
  inferdpt)
    "${PYTHON_BIN}" "${RUNNER}" --model_path "${MODEL_PATH}" --dataset_name "${DATASET}" --dataset_path "${DATASET_PATH}" --epsilon 1.5 --device "cuda:${GPUS[0]}" --selection_protocol "${SELECTION_PROTOCOL}" --build_mapping_only true "${EXTRA[@]}" "$@"
    ;;
  textobfuscator)
    "${PYTHON_BIN}" "${RUNNER}" --model_path "${MODEL_PATH}" --dataset_name "${DATASET}" --dataset_path "${DATASET_PATH}" --epsilon 1.5 --device "cuda:${GPUS[0]}" --selection_protocol "${SELECTION_PROTOCOL}" --build_prototypes_only true "${EXTRA[@]}" "$@"
    ;;
esac

pids=()
for index in "${!EPSILONS[@]}"; do
  epsilon="${EPSILONS[$index]}"
  gpu="${GPUS[$index]}"
  output_csv="${OUTPUT_DIR}/${BASELINE}_${DATASET}_eps${epsilon}.csv"
  log_file="${OUTPUT_DIR}/logs/${BASELINE}_${DATASET}_eps${epsilon}_gpu${gpu}.log"
  command=("${PYTHON_BIN}" "${RUNNER}" --model_path "${MODEL_PATH}" --dataset_name "${DATASET}" --dataset_path "${DATASET_PATH}" --epsilon "${epsilon}" --device "cuda:${gpu}" --selection_protocol "${SELECTION_PROTOCOL}" --max_samples "${MAX_SAMPLES}" --batch_size "${BATCH_SIZE}" --rouge_context_len "${ROUGE_CONTEXT_LEN}" --max_new_tokens "${MAX_NEW_TOKENS}" --rouge_use_stemmer "${ROUGE_USE_STEMMER}" --seed "${SEED}" --output_csv "${output_csv}")
  if [[ "${BASELINE}" != "inferdpt" ]]; then
    command+=(--max_length "${MAX_LENGTH}")
  fi
  command+=("${EXTRA[@]}" "$@")
  "${command[@]}" >"${log_file}" 2>&1 &
  pids+=("$!")
  echo "[Launched] baseline=${BASELINE} dataset=${DATASET} epsilon=${epsilon} gpu=cuda:${gpu} pid=${pids[-1]}"
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
