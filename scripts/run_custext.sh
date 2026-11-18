#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   bash scripts/run_custext.sh sst2 --dry_run true
#   bash scripts/run_custext.sh sst2 --protocol inference --representation model_tokens
#   CUSTEXT_EMBEDDING_PATH=/path/to/counter-fitted-vectors.txt \
#     bash scripts/run_custext.sh qnli --variant custext_plus

task="${1:-sst2}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

arguments=(--task "$task")
if [[ -n "${CUSTEXT_EMBEDDING_PATH:-}" ]]; then
  arguments+=(--embedding_path "$CUSTEXT_EMBEDDING_PATH")
fi
if [[ -n "${DEVICE:-}" ]]; then
  arguments+=(--device "$DEVICE")
fi
if [[ -n "${MAPPING_DEVICE:-}" ]]; then
  arguments+=(--mapping_device "$MAPPING_DEVICE")
fi
if [[ -n "${MODEL_PATH:-}" ]]; then
  arguments+=(--model_path "$MODEL_PATH")
fi

python baseline/custext/run_classification.py "${arguments[@]}" "$@"
