#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PRESET="${PRESET:-${1:-custom}}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  shift
fi

JUDGE_MODEL="${JUDGE_MODEL:-}"
API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wildchat_utility_judge}"
CACHE_JSONL="${CACHE_JSONL:-${OUTPUT_DIR}/qwen3_32b_wildchat_exact_judge_cache.jsonl}"
PRIVACY_LOOKUP_JSONL="${PRIVACY_LOOKUP_JSONL:-data/wildchat/wildchat_qa500_independent_userprivacy.jsonl}"
EXTRA_ARGS=()

case "${PRESET}" in
  custom)
    if [[ -z "${OUTPUTS_JSONL:-}" ]]; then
      echo "[Error] PRESET=custom requires OUTPUTS_JSONL=path1.jsonl[,path2.jsonl]." >&2
      exit 2
    fi
    INPUT_LABELS="${INPUT_LABELS:-}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX:-wildchat_utility_judge}"
    ;;
  plaintext)
    OUTPUTS_JSONL="${OUTPUTS_JSONL:-outputs/wildchat_original_upper_bound/qwen3_32b_plaintext_wildchat_local_hf_outputs.jsonl}"
    INPUT_LABELS="${INPUT_LABELS:-plaintext}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_32b_wildchat_plaintext_gpt_judge}"
    ;;
  dpspd_k9_eps3)
    OUTPUTS_JSONL="${OUTPUTS_JSONL:-outputs/wildchat_dpspd_blackbox_utility/qwen3_32b_wildchat_dpspd_blackbox_k9_eps3_seed42_outputs.jsonl}"
    INPUT_LABELS="${INPUT_LABELS:-dpspd_k9_eps3}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_32b_wildchat_dpspd_k9_eps3_gpt_judge}"
    ;;
  dpspd_k19_eps3)
    OUTPUTS_JSONL="${OUTPUTS_JSONL:-outputs/wildchat_dpspd_blackbox_utility/qwen3_32b_wildchat_dpspd_blackbox_k19_eps3_seed42_outputs.jsonl}"
    INPUT_LABELS="${INPUT_LABELS:-dpspd_k19_eps3}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_32b_wildchat_dpspd_k19_eps3_gpt_judge}"
    ;;
  threeway)
    OUTPUTS_JSONL="${OUTPUTS_JSONL:-outputs/wildchat_original_upper_bound/qwen3_32b_plaintext_wildchat_local_hf_outputs.jsonl,outputs/wildchat_dpspd_blackbox_utility/qwen3_32b_wildchat_dpspd_blackbox_k9_eps3_seed42_outputs.jsonl,outputs/wildchat_dpspd_blackbox_utility/qwen3_32b_wildchat_dpspd_blackbox_k19_eps3_seed42_outputs.jsonl}"
    INPUT_LABELS="${INPUT_LABELS:-plaintext,dpspd_k9_eps3,dpspd_k19_eps3}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_32b_wildchat_plain_k9_k19_gpt_judge}"
    ;;
  inferdpt)
    OUTPUTS_JSONL="${OUTPUTS_JSONL:-outputs/wildchat_qwen3_32b_eps3_inferdpt}"
    INPUT_LABELS="${INPUT_LABELS:-inferdpt_eps3}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_32b_wildchat_inferdpt_eps3_gpt_judge}"
    EXTRA_ARGS+=(--source_field input_text --response_field extracted_generation --reference_field reference --subset_default answerable_qa --subsets answerable_qa --privacy_lookup_jsonl "${PRIVACY_LOOKUP_JSONL}")
    ;;
  santext_plus)
    OUTPUTS_JSONL="${OUTPUTS_JSONL:-outputs/wildchat_qa500_independent_userprivacy_qwen3_32b_eps3_santext_plus.jsonl}"
    INPUT_LABELS="${INPUT_LABELS:-santext_plus_eps3}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_32b_wildchat_santext_plus_eps3_gpt_judge}"
    EXTRA_ARGS+=(--source_field input_text --response_field extracted_generation --reference_field reference --subset_default answerable_qa --subsets answerable_qa --privacy_lookup_jsonl "${PRIVACY_LOOKUP_JSONL}")
    ;;
  baselines)
    OUTPUTS_JSONL="${OUTPUTS_JSONL:-outputs/wildchat_qwen3_32b_eps3_inferdpt,outputs/wildchat_qa500_independent_userprivacy_qwen3_32b_eps3_santext_plus.jsonl}"
    INPUT_LABELS="${INPUT_LABELS:-inferdpt_eps3,santext_plus_eps3}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_32b_wildchat_baselines_eps3_gpt_judge}"
    EXTRA_ARGS+=(--source_field input_text --response_field extracted_generation --reference_field reference --subset_default answerable_qa --subsets answerable_qa --privacy_lookup_jsonl "${PRIVACY_LOOKUP_JSONL}")
    ;;
  all)
    OUTPUTS_JSONL="${OUTPUTS_JSONL:-outputs/wildchat_original_upper_bound/qwen3_32b_plaintext_wildchat_local_hf_outputs.jsonl,outputs/wildchat_dpspd_blackbox_utility/qwen3_32b_wildchat_dpspd_blackbox_k9_eps3_seed42_outputs.jsonl,outputs/wildchat_dpspd_blackbox_utility/qwen3_32b_wildchat_dpspd_blackbox_k19_eps3_seed42_outputs.jsonl,outputs/wildchat_qwen3_32b_eps3_inferdpt,outputs/wildchat_qa500_independent_userprivacy_qwen3_32b_eps3_santext_plus.jsonl}"
    INPUT_LABELS="${INPUT_LABELS:-plaintext,dpspd_k9_eps3,dpspd_k19_eps3,inferdpt_eps3,santext_plus_eps3}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_32b_wildchat_all_methods_eps3_gpt_judge}"
    EXTRA_ARGS+=(--privacy_lookup_jsonl "${PRIVACY_LOOKUP_JSONL}")
    ;;
  *)
    echo "[Error] Unknown PRESET='${PRESET}'. Use custom, plaintext, dpspd_k9_eps3, dpspd_k19_eps3, threeway, inferdpt, santext_plus, baselines, or all." >&2
    exit 2
    ;;
esac

JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"

python experiments/wildchat/judge_wildchat_utility.py \
  --outputs_jsonl "${OUTPUTS_JSONL}" \
  --input_labels "${INPUT_LABELS}" \
  --judge_model "${JUDGE_MODEL}" \
  --api_key_env "${API_KEY_ENV}" \
  --openai_base_url "${OPENAI_BASE_URL}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_prefix "${OUTPUT_PREFIX}" \
  --cache_jsonl "${CACHE_JSONL}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
