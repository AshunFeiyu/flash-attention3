#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

: "${PMD_CONFIG_SEED:?set PMD_CONFIG_SEED to a locked non-empty config.ini}"

SWEEP_ROOT="${DQ_DENSE_SWEEP_ROOT:-${SHAOBO_RUN_ROOT}/layout_probes}"
SWEEP_DIR="${SWEEP_ROOT}/dq_native_ds_dense_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"
SUMMARY="${SWEEP_DIR}/summary.tsv"
printf 'lts\twriter_t\twriter_alt\tdq_reader\tdk_reader\trc\tfinal\trun\n' >"${SUMMARY}"

for lts in 0 1; do
  for writer_t in 0 1; do
    for writer_alt in 0 1; do
      for dq_reader in 0 1 2 3 4; do
        name="lts${lts}_t${writer_t}_a${writer_alt}_dq${dq_reader}_dk3"
        log="${SWEEP_DIR}/${name}.log"
        set +e
        DQ_DENSE_NATIVE_F16_SCORE=1 \
        DQ_DENSE_NATIVE_F16_LTS="${lts}" \
        DQ_DENSE_WRITER_T="${writer_t}" \
        DQ_DENSE_WRITER_ALT="${writer_alt}" \
        DQ_DENSE_DQ_READER="${dq_reader}" \
        DQ_DENSE_DK_READER=3 \
        DQ_DENSE_REQUIRE_SEMANTIC_PASS=0 \
          scripts/run_dq_native_ds_write_dense_probe.sh >"${log}" 2>&1
        rc=$?
        set -e
        final="$(grep '^dense_native_ds_final ' "${log}" | tail -1 || true)"
        run="$(grep '^dq_native_ds_dense ' "${log}" | tail -1 | sed -n 's/.* run=//p' || true)"
        printf '%s\t%s\t%s\t%s\t3\t%s\t%s\t%s\n' \
          "${lts}" "${writer_t}" "${writer_alt}" "${dq_reader}" \
          "${rc}" "${final}" "${run}" | tee -a "${SUMMARY}"
      done
    done
  done
done

echo "dense sweep summary: ${SUMMARY}"
