#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${repo_dir}"

source scripts/env.sh

DQ_BIN="${DQ_BIN:-build/fa3_bwd_dq_clean}"
DQ_ASM="${DQ_ASM:-build/fa3_bwd_dq_clean.asm}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  TARGET_GFX="${TARGET_GFX:-946}" \
  BUILD_ASM="${BUILD_ASM:-1}" \
  SRC=src/dq_kernel.cpp \
  BIN="${DQ_BIN}" \
  ASM="${DQ_ASM}" \
  ./build.sh
fi

python3 scripts/check_dq_kernel_gate.py \
  --source src/dq_kernel.cpp \
  --contract include/dq_contract.h \
  --asm "${DQ_ASM}"

case_id="dq_correctness_$(date +%Y%m%d_%H%M%S)"
case_dir="${SHAOBO_RUN_ROOT}/${case_id}"
mkdir -p "${case_dir}"

case_script="${case_dir}/run_case.sh"
cat > "${case_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${repo_dir}
export HSA_TOOLS_LIB="${HSA_TOOLS_LIB:-}"
export B="\${B:-1}"
export H="\${H:-1}"
export S="\${S:-128}"
export D="\${D:-128}"
export CAUSAL="\${CAUSAL:-1}"
export CANONICAL_DQ="\${CANONICAL_DQ:-0}"
export DQ_DIAG_STORE="\${DQ_DIAG_STORE:-0}"
export DQ_TILES_PER_DISPATCH="\${DQ_TILES_PER_DISPATCH:-0}"
echo "PMD_BINARY=${repo_dir}/${DQ_BIN}"
sha256sum "${repo_dir}/${DQ_BIN}"
exec "${repo_dir}/${DQ_BIN}" --B=\${B} --H=\${H} --S=\${S} --D=\${D} --causal=\${CAUSAL} --canonical=\${CANONICAL_DQ} --diag-store=\${DQ_DIAG_STORE} --tiles-per-dispatch=\${DQ_TILES_PER_DISPATCH}
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
  echo "PMD dQ correctness detected model abort; see ${stdout_log}" >&2
  exit 1
fi

if ! grep -Eq 'fa3_bwd_dq_correctness status=success' "${stdout_log}"; then
  echo "PMD dQ correctness did not report successful status" >&2
  exit 1
fi

if ! grep -q 'pass=1' "${stdout_log}"; then
  echo "PMD dQ correctness failed numerical comparison" >&2
  exit 1
fi

echo "dQ correctness m5out: ${case_dir}/m5out"
