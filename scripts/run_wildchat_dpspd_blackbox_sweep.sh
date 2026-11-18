#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Semicolon-separated specs: label:k:target_epsilon:epsilon_private[:cuda_visible_devices]
RUNS="${RUNS:-k9_eps3:9:3.0:7.82;k19_eps3:19:3.0:8.51}"
PARALLEL="${PARALLEL:-false}"
LOG_DIR="${LOG_DIR:-logs}"
TRANSPORT_MODE="${TRANSPORT_MODE:-token_projection}"
if [[ -z "${OUTPUT_PREFIX_BASE:-}" ]]; then
  if [[ "${TRANSPORT_MODE}" == "embedding" ]]; then
    OUTPUT_PREFIX_BASE="qwen3_32b_wildchat_dpspd_embedding"
  else
    OUTPUT_PREFIX_BASE="qwen3_32b_wildchat_dpspd_blackbox"
  fi
fi

run_one() {
  local label="$1"
  local k="$2"
  local target_eps="$3"
  local eps_private="$4"
  local cuda_devices="${5:-}"
  local prefix="${OUTPUT_PREFIX_BASE}_${label}_seed${SEED:-42}"
  shift 5

  echo "[Run] utility label=${label} transport=${TRANSPORT_MODE} k=${k} target_eps=${target_eps} eps_private=${eps_private}"
  if [[ -n "${cuda_devices}" ]]; then
    CUDA_VISIBLE_DEVICES="${cuda_devices}" \
    TRANSPORT_MODE="${TRANSPORT_MODE}" \
    DUMMY_K="${k}" \
    TARGET_EPSILON="${target_eps}" \
    EPSILON_NON_SENSITIVE="${target_eps}" \
    EPSILON_PRIVATE="${eps_private}" \
    OUTPUT_PREFIX="${prefix}" \
    bash scripts/run_wildchat_dpspd_blackbox_qwen3_32b.sh "$@"
  else
    TRANSPORT_MODE="${TRANSPORT_MODE}" \
    DUMMY_K="${k}" \
    TARGET_EPSILON="${target_eps}" \
    EPSILON_NON_SENSITIVE="${target_eps}" \
    EPSILON_PRIVATE="${eps_private}" \
    OUTPUT_PREFIX="${prefix}" \
    bash scripts/run_wildchat_dpspd_blackbox_qwen3_32b.sh "$@"
  fi
}

mkdir -p "${LOG_DIR}"
pids=()
IFS=';' read -ra SPECS <<< "${RUNS}"
for spec in "${SPECS[@]}"; do
  [[ -z "${spec}" ]] && continue
  IFS=':' read -r label k target_eps eps_private cuda_devices <<< "${spec}"
  if [[ -z "${label}" || -z "${k}" || -z "${target_eps}" || -z "${eps_private}" ]]; then
    echo "[Error] Bad RUNS item '${spec}'. Expected label:k:target_epsilon:epsilon_private[:cuda]." >&2
    exit 2
  fi
  if [[ "${PARALLEL}" == "true" ]]; then
    run_one "${label}" "${k}" "${target_eps}" "${eps_private}" "${cuda_devices:-}" "$@" \
      > "${LOG_DIR}/wildchat_dpspd_blackbox_${label}.log" 2>&1 &
    pids+=("$!")
  else
    run_one "${label}" "${k}" "${target_eps}" "${eps_private}" "${cuda_devices:-}" "$@"
  fi
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
