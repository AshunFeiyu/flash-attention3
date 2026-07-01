#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=scripts/env.sh
source "${repo_root}/scripts/env.sh"

print_bin_only=0
if [[ "${1:-}" == "--print-bin" ]]; then
  print_bin_only=1
fi

log_dir="${SHAOBO_RUN_ROOT}/xcu_outputs"
mkdir -p "${log_dir}"
log_file="${log_dir}/xcu_preflight_$(date +%Y%m%d_%H%M%S).txt"

find_deb() {
  if [[ -n "${XCU_DEB}" && -f "${XCU_DEB}" ]]; then
    printf '%s\n' "${XCU_DEB}"
    return 0
  fi

  local candidate
  for candidate in \
    "${SHAOBO_RUN_ROOT}/tools/XCompute-Light-4.6.3-Linux-sqtt-cli.deb" \
    "/Volumes/172.20.68.76/共享/工具/XCompute-Light-4.6.3-Linux-sqtt-cli.deb" \
    "/共享/工具/XCompute-Light-4.6.3-Linux-sqtt-cli.deb" \
    "/mnt/共享/工具/XCompute-Light-4.6.3-Linux-sqtt-cli.deb"
  do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

find_xcu_in_sidecar() {
  find "${XCU_SIDECAR_ROOT}" -type f -name xcu -perm -111 2>/dev/null | head -n 1
}

xcu_bin=""
if command -v xcu >/dev/null 2>&1; then
  xcu_bin="$(command -v xcu)"
else
  xcu_bin="$(find_xcu_in_sidecar || true)"
  if [[ -z "${xcu_bin}" ]]; then
    deb_path="$(find_deb)" || {
      echo "xcu not found and sidecar deb not found. Set XCU_DEB=/path/to/XCompute-Light-*.deb" >&2
      exit 1
    }
    mkdir -p "${XCU_SIDECAR_ROOT}"
    dpkg-deb -x "${deb_path}" "${XCU_SIDECAR_ROOT}"
    xcu_bin="$(find_xcu_in_sidecar || true)"
    if [[ -z "${xcu_bin}" ]]; then
      echo "Sidecar unpacked but no executable named xcu was found under ${XCU_SIDECAR_ROOT}" >&2
      exit 1
    fi
  fi
fi

{
  echo "# XCU Preflight"
  echo "timestamp=$(date -Iseconds)"
  echo "repo_root=${repo_root}"
  echo "SHAOBO_RUN_ROOT=${SHAOBO_RUN_ROOT}"
  echo "XCU_SIDECAR_ROOT=${XCU_SIDECAR_ROOT}"
  echo "xcu_bin=${xcu_bin}"
  echo
  "${xcu_bin}" --help 2>&1 | head -n 40 || true
} > "${log_file}"

if [[ "${print_bin_only}" == "1" ]]; then
  printf '%s\n' "${xcu_bin}"
else
  echo "xcu preflight PASS"
  echo "xcu_bin=${xcu_bin}"
  echo "log=${log_file}"
fi
