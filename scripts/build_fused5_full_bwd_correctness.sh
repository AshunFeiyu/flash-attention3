#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"

TARGET_GFX="${TARGET_GFX:-946}"
BUILD_DIR="${FUSED5_FULL_BUILD_DIR:-build/fused5_full}"
mkdir -p "${BUILD_DIR}"

TARGET_GFX="${TARGET_GFX}" BUILD_DIR="${BUILD_DIR}" \
  BIN="${BUILD_DIR}/fused_bwd_kernel.o" \
  ASM="${BUILD_DIR}/fused_bwd_kernel.asm" \
  SRC=src/fused_bwd_kernel.cpp BUILD_ASM=1 COMPILE_ONLY=1 \
  SHAOBO_EXPLICIT_WDRA_INIT=1 ./build.sh

python3 scripts/check_fused_bwd_kernel_gate.py \
  --asm "${BUILD_DIR}/fused_bwd_kernel.asm"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${BUILD_DIR}/fused_bwd_kernel.asm" \
  --symbol-regex fa3_bwd_5gemm_kernel \
  --max-private-segment 0 --max-sgpr-spill 0 --max-vgpr-spill 0

TARGET_GFX="${TARGET_GFX}" BUILD_DIR="${BUILD_DIR}" \
  BIN="${BUILD_DIR}/dot_do_o_kernel.o" \
  ASM="${BUILD_DIR}/dot_do_o_kernel.asm" \
  SRC=src/dot_do_o_kernel.cpp BUILD_ASM=1 COMPILE_ONLY=1 \
  SHAOBO_DISABLE_WDRA_FLAGS=1 SHAOBO_EXPLICIT_WDRA_INIT=0 ./build.sh

TARGET_GFX="${TARGET_GFX}" BUILD_DIR="${BUILD_DIR}" \
  BIN="${BUILD_DIR}/fused_bwd_dq_reduce.o" \
  ASM="${BUILD_DIR}/fused_bwd_dq_reduce.asm" \
  SRC=src/fused_bwd_dq_reduce.cpp BUILD_ASM=1 COMPILE_ONLY=1 \
  SHAOBO_DISABLE_WDRA_FLAGS=1 SHAOBO_EXPLICIT_WDRA_INIT=0 ./build.sh

python3 scripts/check_dot_do_o_kernel_gate.py \
  --asm "${BUILD_DIR}/dot_do_o_kernel.asm"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${BUILD_DIR}/dot_do_o_kernel.asm" \
  --symbol-regex dot_do_o_kernel --max-vgpr-count 32
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${BUILD_DIR}/fused_bwd_dq_reduce.asm" \
  --symbol-regex fa3_bwd_dq_reduce_kernel \
  --max-private-segment 0 --max-sgpr-spill 0 --max-vgpr-spill 0

cvt_count="$(grep -c 'v_cvt_pk_f16_f32' \
  "${BUILD_DIR}/fused_bwd_dq_reduce.asm" || true)"
store_count="$(grep -c 'global_store_dwordx2' \
  "${BUILD_DIR}/fused_bwd_dq_reduce.asm" || true)"
if [[ "${cvt_count}" -lt 2 || "${store_count}" -lt 1 ]]; then
  echo "dQ reduction must pack four FP32 sums into one FP16 vector store" >&2
  exit 1
fi
if grep -Eq '^[[:space:]]+global_store_dwordx4' \
    "${BUILD_DIR}/fused_bwd_dq_reduce.asm"; then
  echo "dQ reduction unexpectedly retains an FP32 vector output store" >&2
  exit 1
fi

TARGET_GFX="${TARGET_GFX}" BUILD_DIR="${BUILD_DIR}" \
  BIN="${BUILD_DIR}/full_bwd_correctness.o" \
  SRC=src/full_bwd_correctness.cpp BUILD_ASM=0 COMPILE_ONLY=1 \
  SHAOBO_DISABLE_WDRA_FLAGS=1 SHAOBO_EXPLICIT_WDRA_INIT=0 \
  EXTRA_CXXFLAGS="-DSHAOBO_FULL_BWD_FUSED5=1" ./build.sh

TARGET_GFX="${TARGET_GFX}" BUILD_DIR="${BUILD_DIR}" \
  BIN="${BUILD_DIR}/fa3_bwd_fused5_full_correctness" BUILD_ASM=0 \
  LINK_OBJECTS="${BUILD_DIR}/fused_bwd_kernel.o ${BUILD_DIR}/dot_do_o_kernel.o ${BUILD_DIR}/fused_bwd_dq_reduce.o ${BUILD_DIR}/full_bwd_correctness.o" \
  ./build.sh

echo "fused5 full backward correctness build: ${ROOT}/${BUILD_DIR}/fa3_bwd_fused5_full_correctness"
