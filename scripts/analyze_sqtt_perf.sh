#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/analyze_sqtt_perf.sh --perf case.perf [--dispatch 1] [--out-dir DIR]
  scripts/analyze_sqtt_perf.sh --perf case.perf --dispatch 1 \
    --time-range START:END \
    --location xcd=0,se=0,cu=6,simd=1,wave=1

Outputs stay outside the repo by default:
  ${SHAOBO_RUN_ROOT}/xcu_outputs/<perf-name>_<timestamp>/

The first form records detail and wavefront/bubble tables.  The second form
also exports pipeline and SIMD CSV for the selected time window and location.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=scripts/env.sh
source "${repo_root}/scripts/env.sh"

perf_file=""
dispatch="1"
out_dir=""
time_range=""
location=""
top_n="50"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --perf)
      perf_file="${2:-}"
      shift 2
      ;;
    --dispatch)
      dispatch="${2:-}"
      shift 2
      ;;
    --out-dir)
      out_dir="${2:-}"
      shift 2
      ;;
    --time-range)
      time_range="${2:-}"
      shift 2
      ;;
    --location)
      location="${2:-}"
      shift 2
      ;;
    --top)
      top_n="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${perf_file}" ]]; then
  echo "--perf is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "${perf_file}" ]]; then
  echo "Perf file not found: ${perf_file}" >&2
  exit 1
fi
if [[ -n "${time_range}" && -z "${location}" ]]; then
  echo "--location is required when --time-range is set" >&2
  exit 2
fi

perf_abs="$(cd "$(dirname "${perf_file}")" && pwd)/$(basename "${perf_file}")"
perf_base="$(basename "${perf_file}" .perf)"
timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${out_dir}" ]]; then
  out_dir="${SHAOBO_RUN_ROOT}/xcu_outputs/${perf_base}_${timestamp}"
fi
mkdir -p "${out_dir}"

xcu_bin="$("${repo_root}/scripts/xcu_preflight.sh" --print-bin)"

run_capture() {
  local output_file="$1"
  shift
  {
    echo "$ ${xcu_bin} $*"
    "${xcu_bin}" "$@"
  } > "${output_file}" 2>&1
}

run_capture "${out_dir}/detail.txt" \
  status -P "${perf_abs}" --sqtt-sections detail --sqtt-dispatches "${dispatch}"

run_capture "${out_dir}/wavefronts_bubbles.txt" \
  status -P "${perf_abs}" --sqtt-sections wavefronts,bubbles \
  --sqtt-dispatches "${dispatch}" --sqtt-top "${top_n}"

{
  echo "# SQTT CLI Manifest"
  echo
  echo "timestamp: $(date -Iseconds)"
  echo "repo: ${repo_root}"
  echo "perf: ${perf_abs}"
  echo "dispatch: ${dispatch}"
  echo "xcu: ${xcu_bin}"
  echo "out_dir: ${out_dir}"
  echo
  echo "## Required first-pass files"
  echo
  echo "- detail.txt"
  echo "- wavefronts_bubbles.txt"
} > "${out_dir}/manifest.md"

if [[ -n "${time_range}" ]]; then
  pipeline_dir="${out_dir}/pipeline"
  simd_dir="${out_dir}/simd"
  mkdir -p "${pipeline_dir}" "${simd_dir}"

  simd_location="$(printf '%s\n' "${location}" | sed -E 's/,?wave=[^,]+//g; s/,,+/,/g; s/,$//')"

  "${xcu_bin}" status -P "${perf_abs}" \
    --sqtt-sections pipeline \
    --sqtt-dispatches "${dispatch}" \
    --sqtt-time-range "${time_range}" \
    --sqtt-location "${location}" \
    -F csv -D "${pipeline_dir}" > "${out_dir}/pipeline_export.txt" 2>&1

  "${xcu_bin}" status -P "${perf_abs}" \
    --sqtt-sections simd \
    --sqtt-dispatches "${dispatch}" \
    --sqtt-time-range "${time_range}" \
    --sqtt-location "${simd_location}" \
    -F csv -D "${simd_dir}" > "${out_dir}/simd_export.txt" 2>&1

  {
    echo
    echo "## Window exports"
    echo
    echo "time_range: ${time_range}"
    echo "pipeline_location: ${location}"
    echo "simd_location: ${simd_location}"
    echo "pipeline_dir: ${pipeline_dir}"
    echo "simd_dir: ${simd_dir}"
  } >> "${out_dir}/manifest.md"
else
  {
    echo
    echo "## Next step"
    echo
    echo "Choose a time window and location from wavefronts_bubbles.txt, then rerun:"
    echo
    echo '```bash'
    echo "scripts/analyze_sqtt_perf.sh --perf '${perf_abs}' --dispatch '${dispatch}' \\"
    echo "  --time-range START:END \\"
    echo "  --location xcd=0,se=0,cu=CU,simd=SIMD,wave=WAVE"
    echo '```'
  } >> "${out_dir}/manifest.md"
fi

echo "SQTT analysis artifacts: ${out_dir}"
