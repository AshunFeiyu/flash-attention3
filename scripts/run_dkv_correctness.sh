#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source scripts/env.sh

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  TARGET_GFX="${TARGET_GFX:-946}" BUILD_ASM="${BUILD_ASM:-1}" ./build.sh
fi

python3 scripts/check_dkv_kernel_gate.py
python3 scripts/check_symbol_metadata_gate.py \
  --asm build/fa3_bwd_wasp_clean.asm \
  --symbol-regex fa3_bwd_dkv_probe

case_id="dkv_correctness_$(date +%Y%m%d_%H%M%S)"
case_dir="${SHAOBO_RUN_ROOT}/${case_id}"
mkdir -p "${case_dir}"

case_script="${case_dir}/run_case.sh"
cat > "${case_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /zys/shaobo/fa3_bwd_wasp_clean
export B="\${B:-1}"
export H="\${H:-1}"
export S="\${S:-128}"
export D="\${D:-128}"
./build/fa3_bwd_wasp_clean --check=1 --B=\${B} --H=\${H} --S=\${S} --D=\${D}
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
  echo "PMD dKV correctness detected model abort; see ${stdout_log}" >&2
  exit 1
fi

if ! grep -q 'fa3_bwd_dkv_correctness status=success' "${stdout_log}"; then
  echo "PMD dKV correctness did not report successful status" >&2
  exit 1
fi

if ! grep -q 'pass=1' "${stdout_log}"; then
  echo "PMD dKV correctness failed numerical comparison" >&2
  exit 1
fi

echo "dKV correctness m5out: ${case_dir}/m5out"
