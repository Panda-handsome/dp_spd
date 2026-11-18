#!/usr/bin/env bash
# Token-level DP-SPD inference for the Table-1 SST-2/QNLI setting.
# The client adds DP-SPD noise in the checkpoint word-embedding space, then
# projects all active lexical positions back to nearest vocabulary token ids.
set -euo pipefail

task="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$task" in
  all) tasks=(sst2 qnli) ;;
  sst2|qnli) tasks=("$task") ;;
  *) echo "Usage: $0 [sst2|qnli|all] [extra run_classification.py arguments]" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

device="${DEVICE:-cuda:0}"
target_epsilon="${TARGET_EPSILON:-3.0}"
# These are the legacy embedding-level evaluator's DP-SPD local settings.  They
# are intentionally exposed separately from TARGET_EPSILON because a target
# privacy budget and local prefix/suffix budgets are not interchangeable.
epsilon_private="${EPSILON_PRIVATE:-3.07}"
epsilon_non_sensitive="${EPSILON_NON_SENSITIVE:-1.8}"
sensitive_ratio="${SENSITIVE_RATIO:-0.6}"
clip_norm="${CLIP_NORM:-1.0}"
max_length="${MAX_LENGTH:-128}"
batch_size="${BATCH_SIZE:-64}"
max_eval_samples="${MAX_EVAL_SAMPLES:-0}"
python_bin="${PYTHON_BIN:-python}"

for current_task in "${tasks[@]}"; do
  case "$current_task" in
    sst2)
      model_path="${SST2_MODEL_PATH:-models/bert-base-uncased-sst2}"
      dataset_path="${SST2_DATASET_PATH:-data/glue/sst2}"
      ;;
    qnli)
      model_path="${QNLI_MODEL_PATH:-models/bert-base-uncased-qnli}"
      dataset_path="${QNLI_DATASET_PATH:-data/glue/qnli}"
      ;;
  esac

  output_csv="${OUTPUT_DIR:-outputs/table1_token_level}/dpspd_${current_task}_eps${target_epsilon}.csv"
  "$python_bin" experiments/classification/run_classification.py \
    --task "$current_task" --head sequence --transport_mode token \
    --model_path "$model_path" --dataset_path "$dataset_path" \
    --device "$device" --local_files_only true \
    --target_epsilon "$target_epsilon" \
    --epsilon_private "$epsilon_private" --epsilon_non_sensitive "$epsilon_non_sensitive" \
    --sensitive_ratio "$sensitive_ratio" --clip_norm "$clip_norm" \
    --max_length "$max_length" --max_eval_samples "$max_eval_samples" --batch_size "$batch_size" \
    --output_csv "$output_csv" "$@"
done
