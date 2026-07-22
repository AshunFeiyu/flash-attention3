#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${repo_dir}"

TARGET_GFX="${TARGET_GFX:-946}"

TARGET_GFX="${TARGET_GFX}" BUILD_DIR=build/dkv \
  BIN=build/dkv/fa3_bwd_dkv ASM=build/dkv/fa3_bwd_dkv.asm \
  SRC=src/dkv_kernel.cpp BUILD_ASM=1 ./build.sh
python3 scripts/check_dkv_kernel_gate.py --asm build/dkv/fa3_bwd_dkv.asm
python3 scripts/check_symbol_metadata_gate.py \
  --asm build/dkv/fa3_bwd_dkv.asm --symbol-regex fa3_bwd_dkv_kernel

TARGET_GFX="${TARGET_GFX}" BUILD_DIR=build/dq \
  BIN=build/dq/fa3_bwd_dq ASM=build/dq/fa3_bwd_dq.asm \
  SRC=src/dq_kernel.cpp BUILD_ASM=1 ./build.sh
python3 scripts/check_dq_kernel_gate.py --asm build/dq/fa3_bwd_dq.asm
python3 scripts/check_symbol_metadata_gate.py \
  --asm build/dq/fa3_bwd_dq.asm --symbol-regex fa3_bwd_dq_kernel

full_flags="${EXTRA_CXXFLAGS:-} -DSHAOBO_FA3_NO_STANDALONE"
TARGET_GFX="${TARGET_GFX}" BUILD_DIR=build/full BIN=build/full/dkv_kernel.o \
  SRC=src/dkv_kernel.cpp BUILD_ASM=0 COMPILE_ONLY=1 \
  EXTRA_CXXFLAGS="${full_flags}" ./build.sh
TARGET_GFX="${TARGET_GFX}" BUILD_DIR=build/full BIN=build/full/dq_kernel.o \
  SRC=src/dq_kernel.cpp BUILD_ASM=0 COMPILE_ONLY=1 \
  EXTRA_CXXFLAGS="${full_flags}" ./build.sh
TARGET_GFX="${TARGET_GFX}" BUILD_DIR=build/full BIN=build/full/dot_do_o_kernel.o \
  SRC=src/dot_do_o_kernel.cpp BUILD_ASM=0 COMPILE_ONLY=1 \
  SHAOBO_DISABLE_WDRA_FLAGS=1 EXTRA_CXXFLAGS="${full_flags}" ./build.sh
TARGET_GFX="${TARGET_GFX}" BUILD_DIR=build/full \
  BIN=build/full/full_bwd_correctness.o SRC=src/full_bwd_correctness.cpp \
  BUILD_ASM=0 COMPILE_ONLY=1 SHAOBO_DISABLE_WDRA_FLAGS=1 \
  EXTRA_CXXFLAGS="${full_flags}" ./build.sh
TARGET_GFX="${TARGET_GFX}" BUILD_DIR=build/full \
  BIN=build/full/fa3_bwd_full_correctness BUILD_ASM=0 \
  LINK_OBJECTS="build/full/dkv_kernel.o build/full/dq_kernel.o build/full/dot_do_o_kernel.o build/full/full_bwd_correctness.o" \
  ./build.sh

echo "full backward correctness build: ${repo_dir}/build/full/fa3_bwd_full_correctness"
