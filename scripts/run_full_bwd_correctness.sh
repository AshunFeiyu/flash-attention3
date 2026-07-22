#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${repo_dir}"
source scripts/env.sh

B="${B:-1}"
H="${H:-1}"
S="${S:-128}"
D="${D:-128}"
CAUSAL="${CAUSAL:-1}"
SOFTMAX_SCALE="${SOFTMAX_SCALE:-0.08838834764831845}"
GOLDEN_ROOT="${SHAOBO_GOLDEN_ROOT:-/zys/shaobo_golden/fa3_bwd_7gemm}"
BIN="${FULL_BWD_BIN:-build/full/fa3_bwd_full_correctness}"

golden_output="$(python3 scripts/generate_full_bwd_golden.py \
  --root "${GOLDEN_ROOT}" --batch "${B}" --heads "${H}" \
  --seqlen "${S}" --dim "${D}" --causal "${CAUSAL}" \
  --softmax-scale "${SOFTMAX_SCALE}")"
printf '%s\n' "${golden_output}"
golden_dir="$(printf '%s\n' "${golden_output}" | \
  awk -F= '/^golden_cache_path=/{print substr($0, index($0, "=") + 1)}')"
[[ -n "${golden_dir}" && -s "${golden_dir}/manifest.json" ]] || {
  echo "golden cache did not produce a valid manifest" >&2
  exit 1
}

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  scripts/build_full_bwd_correctness.sh
fi
python3 scripts/check_dkv_kernel_gate.py --asm build/dkv/fa3_bwd_dkv.asm
python3 scripts/check_symbol_metadata_gate.py \
  --asm build/dkv/fa3_bwd_dkv.asm --symbol-regex fa3_bwd_dkv_kernel
python3 scripts/check_dq_kernel_gate.py --asm build/dq/fa3_bwd_dq.asm
python3 scripts/check_symbol_metadata_gate.py \
  --asm build/dq/fa3_bwd_dq.asm --symbol-regex fa3_bwd_dq_kernel
python3 scripts/check_dot_do_o_kernel_gate.py \
  --asm build/dot/dot_do_o_kernel.asm
python3 scripts/check_symbol_metadata_gate.py \
  --asm build/dot/dot_do_o_kernel.asm --symbol-regex dot_do_o_kernel \
  --max-vgpr-count 32

bin_abs="$(realpath "${BIN}")"
case_id="full_bwd_correctness_$(date +%Y%m%d_%H%M%S)"
case_dir="${SHAOBO_RUN_ROOT}/${case_id}"
mkdir -p "${case_dir}"
if [[ -n "${PMD_CONFIG_SEED:-}" ]]; then
  mkdir -p "${case_dir}/m5out"
  cp "${PMD_CONFIG_SEED}" "${case_dir}/m5out/config.ini"
fi

case_script="${case_dir}/run_case.sh"
cat >"${case_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${case_dir}
export HSA_TOOLS_LIB="${HSA_TOOLS_LIB:-}"
echo "PMD_BINARY=${bin_abs}"
sha256sum "${bin_abs}"
exec "${bin_abs}" --golden-dir=${golden_dir} --B=${B} --H=${H} --S=${S} --D=${D} --causal=${CAUSAL} --softmax-scale=${SOFTMAX_SCALE}
EOF
chmod +x "${case_script}"

run_py="${PMD_PATH}/scripts/run.py"
if [[ ! -f "${run_py}" ]]; then
  run_py="${PMD_PATH}/core/scripts/run.py"
fi
stdout_log="${case_dir}/pmd_stdout.log"
python3 "${run_py}" -c "${GPU_CHIP}" -m "${case_dir}/m5out" \
  -e "${case_script}" 2>&1 | tee "${stdout_log}"

if grep -Eiq 'panic|Program aborted|core dumped|Aborted' "${stdout_log}"; then
  echo "full backward correctness detected model abort: ${stdout_log}" >&2
  exit 1
fi
if ! grep -Eq 'fa3_bwd_full_correctness .*dot_pass=1 dkv_pass=1 dq_pass=1 pass=1' \
    "${stdout_log}"; then
  echo "full backward correctness did not pass all stages" >&2
  exit 1
fi

dispatches="$(find "${case_dir}/m5out" -name stats.txt -type f -size +0c | wc -l | tr -d ' ')"
if [[ "${dispatches}" != "3" ]]; then
  echo "expected 3 kernel dispatch stats, found ${dispatches}" >&2
  exit 1
fi
bank_conflicts="$(find "${case_dir}/m5out" -name stats.txt -type f -size +0c \
  -exec awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' {} + | \
  awk '{ sum += $1 } END { print sum + 0 }')"
if [[ "${bank_conflicts}" != "0" ]]; then
  echo "full backward correctness found LDS bank conflicts: ${bank_conflicts}" >&2
  exit 1
fi

echo "full backward correctness golden: ${golden_dir}"
echo "full backward correctness m5out: ${case_dir}/m5out"
echo "full backward correctness dispatches=${dispatches} ldsBankConflict=${bank_conflicts}"
python3 scripts/parse_full_bwd_dispatches.py --m5out "${case_dir}/m5out" \
  --json-out "${case_dir}/full_bwd_metrics.json"
