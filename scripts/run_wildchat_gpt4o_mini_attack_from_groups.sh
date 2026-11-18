#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Reuse MIA-generated DP-SPD candidate groups and run GPT-4o-mini attack.
# This does not regenerate dummies and does not load local GPU models.
#
# Example:
#   export OPENAI_API_KEY=...
#   GROUPS_JSONL="outputs/wildchat_dpspd_blackbox_attack/k9_groups.jsonl,outputs/wildchat_dpspd_blackbox_attack/k19_groups.jsonl" \
#   bash scripts/run_wildchat_gpt4o_mini_attack_from_groups.sh --resume false

GROUPS_JSONL="${GROUPS_JSONL:-}"
if [[ -z "${GROUPS_JSONL}" ]]; then
  echo "[Error] Set GROUPS_JSONL to one or more comma-separated *_groups.jsonl files." >&2
  exit 2
fi

GPT_MODEL="${GPT_MODEL:-gpt-4o-mini}"
API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wildchat_dpspd_blackbox_gpt_attack}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-wildchat_gpt4o_mini_attack_from_mia_groups}"

python experiments/wildchat/run_gpt_attack_from_groups.py \
  --groups_jsonl "${GROUPS_JSONL}" \
  --gpt_model "${GPT_MODEL}" \
  --api_key_env "${API_KEY_ENV}" \
  --openai_base_url "${OPENAI_BASE_URL}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_prefix "${OUTPUT_PREFIX}" \
  "$@"
