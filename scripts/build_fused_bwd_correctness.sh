#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}"

BUILD_DIR="${FUSED5_BUILD_DIR:-build/fused5_correctness}"
mkdir -p "${BUILD_DIR}"

TARGET_GFX=946 BUILD_ASM=1 COMPILE_ONLY=1 \
SRC=src/fused_bwd_kernel.cpp BUILD_DIR="${BUILD_DIR}" \
BIN="${BUILD_DIR}/fused_bwd_kernel.o" \
ASM="${BUILD_DIR}/fused_bwd_kernel.asm" \
SHAOBO_EXPLICIT_WDRA_INIT=1 ./build.sh

TARGET_GFX=946 BUILD_ASM=0 COMPILE_ONLY=1 \
SRC=tests/fused_bwd_correctness.cpp BUILD_DIR="${BUILD_DIR}" \
BIN="${BUILD_DIR}/fused_bwd_correctness.o" \
SHAOBO_DISABLE_WDRA_FLAGS=1 SHAOBO_EXPLICIT_WDRA_INIT=0 ./build.sh

TARGET_GFX=946 BUILD_ASM=0 \
LINK_OBJECTS="${BUILD_DIR}/fused_bwd_kernel.o ${BUILD_DIR}/fused_bwd_correctness.o" \
BUILD_DIR="${BUILD_DIR}" BIN="${BUILD_DIR}/fused_bwd_correctness" ./build.sh

python3 scripts/check_fused_bwd_kernel_gate.py \
  --asm "${BUILD_DIR}/fused_bwd_kernel.asm"
python3 scripts/check_symbol_metadata_gate.py \
  --asm "${BUILD_DIR}/fused_bwd_kernel.asm" \
  --symbol-regex fa3_bwd_5gemm_kernel \
  --max-private-segment 0 --max-sgpr-spill 0 --max-vgpr-spill 0

echo "fused5 correctness build: ${BUILD_DIR}/fused_bwd_correctness"
