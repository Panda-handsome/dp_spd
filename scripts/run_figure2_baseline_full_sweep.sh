#!/usr/bin/env bash
# Run the six IMDb Figure-2 baseline adapters over all privacy budgets.
# The shared SanText and InferDPT IMDb-train mapping caches must already be
# prepared; run_imdb_classification.py --prepare_mapping_only true does that.
set -euo pipefail

cd "$(dirname "$0")/.."

# Override from the environment for an intentionally smaller/debug sweep.
methods=${METHODS:-"cape custext dp_forward inferdpt santext textobfuscator"}
epsilons=${EPSILONS:-"1.5 2.0 2.5 3.0 4.0"}
workers=${WORKERS:-8}

if (( workers < 1 || workers > 8 )); then
  echo "WORKERS must be in [1, 8], got $workers" >&2
  exit 2
fi

mkdir -p logs outputs/baselines/figure2
job_file=$(mktemp /tmp/figure2_baseline_jobs.XXXXXX)
trap 'rm -f "$job_file"' EXIT
for method in $methods; do
  for epsilon in $epsilons; do
    printf '%s %s\n' "$method" "$epsilon" >> "$job_file"
  done
done

run_point() {
  local method=$1
  local epsilon=$2
  local gpu=$3
  local tag="${method}_eps${epsilon}"
  echo "[$(date '+%F %T')] GPU ${gpu}: ${tag}"
  bash scripts/run_figure2_baseline_classification.sh "$method" "$epsilon" "$gpu" \
    > "logs/figure2_${tag}_gpu${gpu}.log" 2>&1
  echo "[$(date '+%F %T')] GPU ${gpu}: completed ${tag}"
}
export -f run_point

# GNU xargs assigns one persistent process slot (0--workers-1) to each
# concurrent child.  That slot is exactly the CUDA physical index passed to
# the point launcher; a GPU receives its next job only after the prior one
# terminates.
xargs -r -P "$workers" -n 2 --process-slot-var=FIGURE2_GPU \
  bash -c 'run_point "$1" "$2" "$FIGURE2_GPU"' _ < "$job_file"
