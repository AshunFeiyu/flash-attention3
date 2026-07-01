#!/usr/bin/env bash
set -euo pipefail

bad=$(
  find . \
    \( -name 'm5out*' -o -name '*.perf' -o -name '*.log' -o -name '*.stdout' -o -name '*.stderr' \) \
    -not -path './.git/*' \
    -print
)

if [[ -n "${bad}" ]]; then
  echo "Clean repo gate failed: generated artifacts found inside repo" >&2
  echo "${bad}" >&2
  exit 1
fi

echo "Clean repo gate: PASS"

