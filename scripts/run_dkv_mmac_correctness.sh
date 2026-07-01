#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source scripts/env.sh

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  TARGET_GFX="${TARGET_GFX:-946}" BUILD_ASM="${BUILD_ASM:-1}" ./build.sh
fi

python3 scripts/check_dkv_kernel_gate.py
export WAVES="${WAVES:-16}"
export MQ64="${MQ64:-0}"
export OVERLAY="${OVERLAY:-0}"
if [[ "${WAVES}" == "12" && "${MQ64}" == "1" ]]; then
  metadata_regex="fa3_bwd_dkv_mmac12_mq64"
elif [[ "${WAVES}" == "12" && "${OVERLAY}" == "1" ]]; then
  metadata_regex="fa3_bwd_dkv_mmac12_sidecar_overlay"
elif [[ "${WAVES}" == "12" ]]; then
  metadata_regex="fa3_bwd_dkv_mmac12_kernel"
else
  metadata_regex="fa3_bwd_dkv_mmac_kernel"
fi
python3 scripts/check_symbol_metadata_gate.py \
  --asm build/fa3_bwd_wasp_clean.asm \
  --symbol-regex "${metadata_regex}"

case_id="dkv_mmac_correctness_$(date +%Y%m%d_%H%M%S)"
case_dir="${SHAOBO_RUN_ROOT}/${case_id}"
mkdir -p "${case_dir}"

case_script="${case_dir}/run_case.sh"
cat > "${case_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /zys/shaobo/fa3_bwd_wasp_clean
export HSA_TOOLS_LIB="${HSA_TOOLS_LIB:-}"
export B="\${B:-1}"
export H="\${H:-1}"
export S="\${S:-128}"
export D="\${D:-128}"
export CAUSAL="\${CAUSAL:-1}"
export WAVES="\${WAVES:-${WAVES}}"
export MQ64="\${MQ64:-${MQ64}}"
export OVERLAY="\${OVERLAY:-${OVERLAY}}"
if [[ "\${WAVES}" == "12" && "\${MQ64}" == "1" ]]; then
  ./build/fa3_bwd_wasp_clean --dkv-mmac12-mq64-check=1 --B=\${B} --H=\${H} --S=\${S} --D=\${D} --causal=\${CAUSAL}
elif [[ "\${WAVES}" == "12" && "\${OVERLAY}" == "1" ]]; then
  ./build/fa3_bwd_wasp_clean --dkv-mmac12-overlay-check=1 --B=\${B} --H=\${H} --S=\${S} --D=\${D} --causal=\${CAUSAL}
elif [[ "\${WAVES}" == "12" ]]; then
  ./build/fa3_bwd_wasp_clean --dkv-mmac12-check=1 --B=\${B} --H=\${H} --S=\${S} --D=\${D} --causal=\${CAUSAL}
else
  ./build/fa3_bwd_wasp_clean --dkv-mmac-check=1 --B=\${B} --H=\${H} --S=\${S} --D=\${D} --causal=\${CAUSAL}
fi
EOF
chmod +x "${case_script}"

run_py="${PMD_PATH}/scripts/run.py"
if [[ ! -f "${run_py}" ]]; then
  run_py="${PMD_PATH}/core/scripts/run.py"
fi

stdout_log="${case_dir}/pmd_stdout.log"
python3 "${run_py}" -c "${GPU_CHIP}" -m "${case_dir}/m5out" -e "${case_script}" \
  2>&1 | tee "${stdout_log}"

if grep -Eiq 'panic|Program aborted|core dumped|Aborted' "${stdout_log}"; then
  echo "PMD dKV MMAC correctness detected model abort; see ${stdout_log}" >&2
  exit 1
fi

if ! grep -Eq 'fa3_bwd_dkv_mmac(12(_mq64|_overlay)?)?_correctness status=success' "${stdout_log}"; then
  echo "PMD dKV MMAC correctness did not report successful status" >&2
  exit 1
fi

if ! grep -q 'pass=1' "${stdout_log}"; then
  echo "PMD dKV MMAC correctness failed numerical comparison" >&2
  exit 1
fi

echo "dKV MMAC correctness m5out: ${case_dir}/m5out"
