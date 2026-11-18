#!/usr/bin/env bash
set -euo pipefail

# Entry point for the shared-prefix speed benchmark.
# By default this runs the Qwen3-32B synthetic server-side benchmark, which
# isolates the shared-prefix KV-cache reuse effect without spending time on
# real dummy generation. Set BENCHMARK_MODE=real to use WildChat prompts and
# the full PrivacyDataset dummy-generation pipeline.

cd "$(dirname "$0")/.."

BENCHMARK_MODE="${BENCHMARK_MODE:-synthetic}" \
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wildchat_speed_benchmark}" \
OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_32b_${BENCHMARK_MODE}_dpspd_speed_seed${SEED:-42}}" \
bash scripts/run_wildchat_qwen3_32b_speed_benchmark.sh "$@"
