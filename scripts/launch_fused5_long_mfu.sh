#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"

B="${B:-1}"
H="${H:-16}"
S="${S:-8192}"
D="${D:-128}"
CAUSAL="${CAUSAL:-1}"
FROZEN_DIR="${FUSED5_FROZEN_DIR:?set FUSED5_FROZEN_DIR}"
RUN_ROOT="${FUSED5_MFU_RUN_ROOT:?set FUSED5_MFU_RUN_ROOT}"
LOG="${RUN_ROOT}/launcher.log"

mkdir -p "${RUN_ROOT}"
set +e
GPU_CHIP="${GPU_CHIP:-sb}" \
GPU_ARGS="${GPU_ARGS:-['--SQCIPfLines=7']}" \
SHAOBO_RUN_ROOT="${RUN_ROOT}" \
FUSED5_FULL_RUN_ROOT="${RUN_ROOT}/cases" \
FUSED5_FULL_BUILD_DIR="${FROZEN_DIR}" \
FUSED5_FULL_BIN="${FROZEN_DIR}/fa3_bwd_fused5_full" \
FUSED5_FULL_PERF_ONLY=1 \
FUSED5_FULL_CAPTURE_PERF=0 \
FUSED5_FULL_PMD_TIMEOUT="${FUSED5_FULL_PMD_TIMEOUT:-43200}" \
SKIP_BUILD=1 B="${B}" H="${H}" S="${S}" D="${D}" CAUSAL="${CAUSAL}" \
scripts/run_fused5_full_bwd_correctness.sh >"${LOG}" 2>&1
rc="$?"
set -e
printf '[remote-work exit=%s]\n' "${rc}" >>"${LOG}"
exit "${rc}"
