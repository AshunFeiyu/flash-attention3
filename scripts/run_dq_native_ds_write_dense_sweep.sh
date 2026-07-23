#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

SWEEP_ROOT="${DQ_DENSE_SWEEP_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
SWEEP_DIR="${SWEEP_ROOT}/dq_native_ds_dense_sweep_$(date +%Y%m%d_%H%M%S)"
KV_LEFT_OWNER="${DQ_DENSE_KV_LEFT_OWNER:-0}"
HEAD_DIM="${DQ_DENSE_HEAD_DIM:-32}"
DK_READER="${DQ_DENSE_DK_READER:-3}"
NATIVE_F16_SCORE="${DQ_DENSE_NATIVE_F16_SCORE:-1}"
LTS_LIST="${DQ_DENSE_LTS_LIST:-0 1}"
if [[ "${KV_LEFT_OWNER}" == "1" ]]; then
  NATIVE_F16_SCORE=0
  LTS_LIST="${DQ_DENSE_LTS_LIST:-0}"
fi
mkdir -p "${SWEEP_DIR}"
SUMMARY="${SWEEP_DIR}/summary.tsv"
printf 'owner\tD\tlts\twriter_t\twriter_alt\tdq_reader\tdk_reader\trc\tdq_result\tfinal\trun\n' >"${SUMMARY}"

for lts in ${LTS_LIST}; do
  for writer_t in 0 1; do
    for writer_alt in 0 1; do
      for dq_reader in 0 1 2 3 4; do
        name="owner${KV_LEFT_OWNER}_d${HEAD_DIM}_lts${lts}_t${writer_t}_a${writer_alt}_dq${dq_reader}_dk${DK_READER}"
        log="${SWEEP_DIR}/${name}.log"
        set +e
        DQ_DENSE_NATIVE_F16_SCORE="${NATIVE_F16_SCORE}" \
        DQ_DENSE_NATIVE_F16_LTS="${lts}" \
        DQ_DENSE_WRITER_T="${writer_t}" \
        DQ_DENSE_WRITER_ALT="${writer_alt}" \
        DQ_DENSE_DQ_READER="${dq_reader}" \
        DQ_DENSE_DK_READER="${DK_READER}" \
        DQ_DENSE_HEAD_DIM="${HEAD_DIM}" \
        DQ_DENSE_KV_LEFT_OWNER="${KV_LEFT_OWNER}" \
        DQ_DENSE_REQUIRE_SEMANTIC_PASS=0 \
          scripts/run_dq_native_ds_write_dense_probe.sh >"${log}" 2>&1
        rc=$?
        set -e
        dq_result="$(grep '^dense_native_ds dQ_trans_reader ' "${log}" | tail -1 || true)"
        final="$(grep '^dense_native_ds_final ' "${log}" | tail -1 || true)"
        run="$(grep '^dq_native_ds_dense ' "${log}" | tail -1 | sed -n 's/.* run=//p' || true)"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${KV_LEFT_OWNER}" "${HEAD_DIM}" "${lts}" "${writer_t}" \
          "${writer_alt}" "${dq_reader}" "${DK_READER}" "${rc}" \
          "${dq_result}" "${final}" "${run}" | tee -a "${SUMMARY}"
      done
    done
  done
done

echo "dense sweep summary: ${SUMMARY}"
