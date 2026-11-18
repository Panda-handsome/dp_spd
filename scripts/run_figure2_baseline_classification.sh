#!/usr/bin/env bash
# One Figure-2 baseline point. Run independent invocations for a sweep.
set -euo pipefail

method=${1:?usage: run_figure2_baseline_classification.sh METHOD EPSILON GPU}
epsilon=${2:?usage: run_figure2_baseline_classification.sh METHOD EPSILON GPU}
gpu=${3:?usage: run_figure2_baseline_classification.sh METHOD EPSILON GPU}

case "$method" in
  cape|custext|dp_forward|inferdpt|santext|textobfuscator) ;;
  *) echo "Unknown method: $method" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
mkdir -p outputs/baselines/figure2 logs
# Each process owns its file, so a four-GPU epsilon sweep never races on a
# shared CSV append.  Aggregate these rows only after all workers finish.
CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_BIN:-python}" baseline/run_imdb_classification.py \
  --method "$method" --epsilon "$epsilon" --device cuda:0 \
  --model_path "${MODEL_PATH:-models/Llama-2-7b-chat-hf}" --dataset_path "${DATASET_PATH:-data/imdb}" \
  --max_train_samples 0 --max_eval_samples 0 --max_length 128 --batch_size 32 \
  --output_csv "outputs/baselines/figure2/${method}_eps${epsilon}.csv"
