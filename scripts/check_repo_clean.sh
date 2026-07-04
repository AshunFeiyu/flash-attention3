#!/usr/bin/env bash
set -euo pipefail

bad=$(
  find . \
    \( -name 'm5out*' -o -name '*.perf' -o -name '*.csv' -o -name '*.jsonl' -o -name '*.log' -o -name '*.stdout' -o -name '*.stderr' -o -name 'xcu_outputs' -o -name 'sqtt_csv' -o -name 'xcu_sidecar' \) \
    -not -path './.git/*' \
    -not -path './results/perf_ledger.csv' \
    -print
)

if [[ -n "${bad}" ]]; then
  echo "Clean repo gate failed: generated artifacts found inside repo" >&2
  echo "${bad}" >&2
  exit 1
fi

echo "Clean repo gate: PASS"
