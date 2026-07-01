#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source scripts/env.sh

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  TARGET_GFX="${TARGET_GFX:-946}" BUILD_ASM="${BUILD_ASM:-1}" ./build.sh
fi

python3 scripts/check_fwdstyle_scaffold_gate.py
python3 scripts/check_symbol_metadata_gate.py \
  --asm build/fa3_bwd_wasp_fwdstyle_clean.asm \
  --symbol-regex bwd_dkv_stage61_fwdstyle_s2
./scripts/check_repo_clean.sh

case_id="stage61_clean_s2_score_dp_probe_$(date +%Y%m%d_%H%M%S)"
case_dir="${SHAOBO_RUN_ROOT}/${case_id}"
mkdir -p "${case_dir}"

case_script="${case_dir}/run_case.sh"
cat > "${case_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /zys/shaobo/fa3_bwd_wasp_fwdstyle_clean
export B="\${B:-1}"
export H="\${H:-1}"
export S="\${S:-1024}"
export D="\${D:-128}"
./build/fa3_bwd_wasp_fwdstyle_clean --B=\${B} --H=\${H} --S=\${S} --D=\${D}
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
  echo "PMD scaffold smoke detected model abort; see ${stdout_log}" >&2
  exit 1
fi

if ! grep -q 'stage61_fwdstyle_s2 status=success' "${stdout_log}"; then
  echo "PMD S2 smoke did not report successful status" >&2
  exit 1
fi

echo "S2 score/dP probe smoke m5out: ${case_dir}/m5out"
