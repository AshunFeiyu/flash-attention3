#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != "2" ]]; then
  echo "usage: $0 <hip-executable> <output-dir>" >&2
  exit 2
fi

BIN="$1"
OUT_DIR="$2"
TARGET_GFX="${TARGET_GFX:-946}"

compiler_rocm="${SHAOBO_COMPILER_ROOT:-}"
if [[ ! -x "${compiler_rocm}/llvm/bin/llvm-objcopy" ]]; then
  compiler_rocm="${SHAOBO_COMPILER_ROOT:-}/opt/rocm-6.3.3"
fi
if [[ ! -x "${compiler_rocm}/llvm/bin/llvm-objcopy" ]]; then
  compiler_rocm="${ROCM_PATH:-/opt/rocm-6.3.3}"
fi

LLVM_BIN="${compiler_rocm}/llvm/bin"
OBJCOPY="${LLVM_BIN}/llvm-objcopy"
BUNDLER="${LLVM_BIN}/clang-offload-bundler"
OBJDUMP="${LLVM_BIN}/llvm-objdump"
for tool in "${OBJCOPY}" "${BUNDLER}" "${OBJDUMP}"; do
  [[ -x "${tool}" ]] || {
    echo "missing LLVM tool: ${tool}" >&2
    exit 1
  }
done

mkdir -p "${OUT_DIR}"
fatbin="${OUT_DIR}/device.hipfat"
code_object="${OUT_DIR}/device_gfx${TARGET_GFX}.co"
raw_isa="${OUT_DIR}/device_isa_raw.txt"

"${OBJCOPY}" --dump-section .hip_fatbin="${fatbin}" "${BIN}"
"${BUNDLER}" --unbundle --type=o --input="${fatbin}" \
  --targets="hipv4-amdgcn-amd-amdhsa--gfx${TARGET_GFX}" \
  --output="${code_object}"
"${OBJDUMP}" -d "${code_object}" >"${raw_isa}"

printf 'device_isa_status=PASS code_object=%s raw_isa=%s\n' \
  "${code_object}" "${raw_isa}"
