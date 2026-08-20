#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"
source scripts/env.sh

B="${B:-1}"
H="${H:-1}"
HKV="${HKV:-${H}}"
S="${S:-128}"
D="${D:-128}"
CAUSAL="${CAUSAL:-1}"
SOFTMAX_SCALE="${SOFTMAX_SCALE:-0.08838834764831845}"
GOLDEN_ROOT="${SHAOBO_GOLDEN_ROOT:-/zys/shaobo_golden/fa3_bwd_fused5}"
BUILD_DIR="${FUSED5_FULL_BUILD_DIR:-build/fused5_full}"
BIN="${FUSED5_FULL_BIN:-${BUILD_DIR}/fa3_bwd_fused5_full_correctness}"
RUN_ROOT="${FUSED5_FULL_RUN_ROOT:-${SHAOBO_RUN_ROOT}/fused5_full}"
TIMEOUT="${FUSED5_FULL_PMD_TIMEOUT:-1800}"
CAPTURE_PERF="${FUSED5_FULL_CAPTURE_PERF:-0}"
PERF_ONLY="${FUSED5_FULL_PERF_ONLY:-0}"
HELPER="${HSA_TOOLS_LIB:-/opt/rocm-6.3.3/lib/xprofiler/libperf_gen_helper.so}"
PERF_DFLAGS="${FUSED5_FULL_GPU_DFLAGS:-['StatLog','SQAbar','SQEbar','MMUCheck','TT','Perf']}"

if [[ "${CAPTURE_PERF}" != 0 && "${CAPTURE_PERF}" != 1 ]]; then
  echo "FUSED5_FULL_CAPTURE_PERF must be 0 or 1" >&2
  exit 2
fi
if [[ "${PERF_ONLY}" != 0 && "${PERF_ONLY}" != 1 ]]; then
  echo "FUSED5_FULL_PERF_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "${CAPTURE_PERF}" == 1 ]]; then
  [[ -r "${HELPER}" ]]
fi

golden_dir=""
if [[ "${PERF_ONLY}" == 0 ]]; then
  golden_output="$(python3 scripts/generate_full_bwd_golden.py \
    --root "${GOLDEN_ROOT}" --batch "${B}" --heads "${H}" \
    --heads-kv "${HKV}" \
    --seqlen "${S}" --dim "${D}" --causal "${CAUSAL}" \
    --softmax-scale "${SOFTMAX_SCALE}")"
  printf '%s\n' "${golden_output}"
  golden_dir="$(printf '%s\n' "${golden_output}" | \
    awk -F= '/^golden_cache_path=/{print substr($0, index($0, "=") + 1)}')"
  [[ -n "${golden_dir}" && -s "${golden_dir}/manifest.json" ]]
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  scripts/build_fused5_full_bwd_correctness.sh
fi

bin_abs="$(realpath "${BIN}")"
case_suffix=""
if [[ "${CAPTURE_PERF}" == 1 ]]; then
  case_suffix="_fullperf"
fi
if [[ "${PERF_ONLY}" == 1 ]]; then
  case_suffix="${case_suffix}_perfonly"
fi
case_dir="${RUN_ROOT}/b${B}_hq${H}_hkv${HKV}_s${S}_d${D}_c${CAUSAL}${case_suffix}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${case_dir}/m5out"
cp "${PMD_CONFIG_SEED}" "${case_dir}/m5out/config.ini"
cp "${BUILD_DIR}/toolchain_fingerprint.txt" "${case_dir}/"

case_script="${case_dir}/run_case.sh"
cat >"${case_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${case_dir}
echo "PMD_BINARY=${bin_abs}"
sha256sum "${bin_abs}"
exec "${bin_abs}" --golden-dir=${golden_dir} --perf-only=${PERF_ONLY} --B=${B} --H=${H} --Hkv=${HKV} --S=${S} --D=${D} --causal=${CAUSAL} --softmax-scale=${SOFTMAX_SCALE}
EOF
chmod +x "${case_script}"

run_py="${PMD_PATH}/scripts/run.py"
if [[ ! -f "${run_py}" ]]; then
  run_py="${PMD_PATH}/core/scripts/run.py"
fi

cd "${case_dir}"
if [[ "${CAPTURE_PERF}" == 1 ]]; then
  export GPU_DFLAGS="${PERF_DFLAGS}"
  export HSA_TOOLS_LIB="${HELPER}"
else
  unset GPU_DFLAGS HSA_TOOLS_LIB
fi
set +e
timeout --kill-after=5 "${TIMEOUT}" python3 "${run_py}" \
  -c "${GPU_CHIP}" -m m5out -e "${case_script}" \
  >pmd_stdout.log 2>&1
pmd_status="$?"
set -e

grep -E 'fa3_bwd_full_(correctness|perf) path=fused5 ' pmd_stdout.log \
  | tee correctness.txt || true
if [[ "${PERF_ONLY}" == 1 ]]; then
  semantic="$(grep -c 'fa3_bwd_full_perf path=fused5 .*pass=1' pmd_stdout.log || true)"
else
  semantic="$(grep -c 'fa3_bwd_full_correctness path=fused5 .*dot_pass=1 dkv_pass=1 dq_pass=1 pass=1' pmd_stdout.log || true)"
fi
panic="$(grep -ciE 'panic:|fatal:|not init or has been freed|Program aborted|core dumped' pmd_stdout.log || true)"
vgpr_warning="$(grep -ciE 'read vgpr.*before writing' pmd_stdout.log || true)"
mapfile -t stats_files < <(find m5out -type f -name stats.txt -size +0c | sort)
mapfile -t perf_files < <(find . -type f -name '*.perf' -size +0c | sort)
bank=0
if [[ "${#stats_files[@]}" -gt 0 ]]; then
  bank="$(awk '/ldsBankConflict/ { sum += $2 } END { print sum + 0 }' \
    "${stats_files[@]}")"
fi

if [[ "${#stats_files[@]}" -eq 3 ]]; then
  python3 "${ROOT}/scripts/parse_full_bwd_dispatches.py" \
    --m5out m5out --labels dot_do_o,fused5,dq_reduce \
    --json-out full_bwd_metrics.json | tee dispatch_summary.txt
elif [[ "${#stats_files[@]}" -eq 4 ]]; then
  python3 "${ROOT}/scripts/parse_full_bwd_dispatches.py" \
    --m5out m5out --labels dot_do_o,fused5,dq_reduce,dkv_reduce \
    --json-out full_bwd_metrics.json | tee dispatch_summary.txt
fi

perf_ok=1
if [[ "${CAPTURE_PERF}" == 1 && "${#perf_files[@]}" -lt 1 ]]; then
  perf_ok=0
fi

if [[ "${pmd_status}" == 0 && "${semantic}" == 1 && "${panic}" == 0 &&
      "${vgpr_warning}" == 0 && "${#stats_files[@]}" -ge 2 &&
      "${bank}" == 0 && "${perf_ok}" == 1 ]]; then
  status=PASS
else
  status=FAIL
fi

printf '%s\n' "${perf_files[@]}" > perf_files.txt
printf 'fused5_full_status=%s pmd=%s semantic=%s panic=%s vgpr_warning=%s dispatches=%s bank=%s capture_perf=%s perf_only=%s perf=%s run=%s\n' \
  "${status}" "${pmd_status}" "${semantic}" "${panic}" \
  "${vgpr_warning}" "${#stats_files[@]}" "${bank}" "${CAPTURE_PERF}" "${PERF_ONLY}" \
  "${#perf_files[@]}" "${case_dir}" \
  | tee result.txt

[[ "${status}" == PASS ]]
