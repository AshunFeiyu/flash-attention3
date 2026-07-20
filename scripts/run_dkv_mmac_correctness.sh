#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${repo_dir}"

source scripts/env.sh

BUILD_DIR="${BUILD_DIR:-build}"
BIN="${BIN:-${BUILD_DIR}/fa3_bwd_wasp_clean}"
ASM="${ASM:-${BUILD_DIR}/fa3_bwd_wasp_clean.asm}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  TARGET_GFX="${TARGET_GFX:-946}" BUILD_ASM="${BUILD_ASM:-1}" \
    BUILD_DIR="${BUILD_DIR}" BIN="${BIN}" ASM="${ASM}" ./build.sh
fi

python3 scripts/check_dkv_kernel_gate.py --asm "${ASM}"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${ASM}" \
  --symbol-regex "fa3_bwd_dkv_kernel"

BIN_ABS="$(realpath "${BIN}")"

case_id="dkv_mmac_correctness_$(date +%Y%m%d_%H%M%S)"
case_dir="${SHAOBO_RUN_ROOT}/${case_id}"
mkdir -p "${case_dir}"
if [[ -n "${PMD_CONFIG_SEED:-}" ]]; then
  mkdir -p "${case_dir}/m5out"
  cp "${PMD_CONFIG_SEED}" "${case_dir}/m5out/config.ini"
fi

case_script="${case_dir}/run_case.sh"
cat > "${case_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${case_dir}
export HSA_TOOLS_LIB="${HSA_TOOLS_LIB:-}"
export B="\${B:-1}"
export H="\${H:-1}"
export S="\${S:-128}"
export D="\${D:-128}"
export CAUSAL="\${CAUSAL:-1}"
echo "PMD_BINARY=${BIN_ABS}"
sha256sum "${BIN_ABS}"
exec "${BIN_ABS}" --B=\${B} --H=\${H} --S=\${S} --D=\${D} --causal=\${CAUSAL}
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

if ! grep -Eq 'fa3_bwd_dkv_correctness status=success' "${stdout_log}"; then
  echo "PMD dKV MMAC correctness did not report successful status" >&2
  exit 1
fi

if ! grep -q 'pass=1' "${stdout_log}"; then
  echo "PMD dKV MMAC correctness failed numerical comparison" >&2
  exit 1
fi

echo "dKV MMAC correctness m5out: ${case_dir}/m5out"
