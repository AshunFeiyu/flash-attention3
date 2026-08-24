#!/usr/bin/env python3
"""Reject a performance promotion that loses or bypasses the global best."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "results" / "best_baseline.json"


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-commit", default="HEAD")
    parser.add_argument("--shape", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--pmd", required=True)
    parser.add_argument("--gpu-chip", required=True)
    parser.add_argument("--gpu-args", required=True)
    parser.add_argument("--fused-ticks", type=int, required=True)
    parser.add_argument("--total-ticks", type=int, required=True)
    parser.add_argument("--mmac-active", type=float, required=True)
    parser.add_argument("--bank-conflict", type=int, required=True)
    parser.add_argument("--correctness-pass", action="store_true")
    parser.add_argument("--resource-pass", action="store_true")
    parser.add_argument("--perf", type=Path, required=True)
    parser.add_argument(
        "--mmac-active-tolerance",
        type=float,
        default=0.10,
        help="Allowed absolute percentage-point noise; default 0.10",
    )
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    failures: list[str] = []
    best_commit = baseline["commit"]
    candidate = git("rev-parse", args.candidate_commit)
    if candidate.returncode != 0:
        failures.append(f"candidate commit cannot be resolved: {args.candidate_commit}")
        candidate_commit = args.candidate_commit
    else:
        candidate_commit = candidate.stdout.strip()
        lineage = git("merge-base", "--is-ancestor", best_commit, candidate_commit)
        if lineage.returncode != 0:
            failures.append(
                "candidate does not contain the recorded best commit; rebase or "
                "port the change onto the best baseline before promotion"
            )

    expected = baseline["environment"]
    actual_environment = {
        "compiler": args.compiler,
        "pmd": args.pmd,
        "gpu_chip": args.gpu_chip,
        "gpu_args": args.gpu_args,
    }
    for key, value in actual_environment.items():
        if value != expected[key]:
            failures.append(
                f"environment mismatch {key}: candidate={value!r}, best={expected[key]!r}"
            )
    if args.shape != baseline["case"]["shape"]:
        failures.append(
            f"shape mismatch: candidate={args.shape}, best={baseline['case']['shape']}"
        )
    if not args.correctness_pass:
        failures.append("correctness gate did not pass")
    if not args.resource_pass:
        failures.append("resource/no-spill gate did not pass")
    if args.bank_conflict != 0:
        failures.append(f"LDS bank conflict is {args.bank_conflict}, expected 0")
    if not args.perf.is_file():
        failures.append(f"perf archive does not exist: {args.perf}")

    best_metrics = baseline["metrics"]
    if args.fused_ticks > best_metrics["fused_ticks"]:
        failures.append(
            f"fused ticks regress: {args.fused_ticks} > {best_metrics['fused_ticks']}"
        )
    if args.total_ticks > best_metrics["total_ticks"]:
        failures.append(
            f"total ticks regress: {args.total_ticks} > {best_metrics['total_ticks']}"
        )
    minimum_active = (
        best_metrics["mmac_active_percent"] - args.mmac_active_tolerance
    )
    if args.mmac_active < minimum_active:
        failures.append(
            "MMAC active regresses: "
            f"{args.mmac_active:.6f}% < {minimum_active:.6f}% gate"
        )

    report = {
        "result": "PASS" if not failures else "BLOCK",
        "best_commit": best_commit,
        "candidate_commit": candidate_commit,
        "best": {
            "fused_ticks": best_metrics["fused_ticks"],
            "total_ticks": best_metrics["total_ticks"],
            "mmac_active_percent": best_metrics["mmac_active_percent"],
        },
        "candidate": {
            "fused_ticks": args.fused_ticks,
            "total_ticks": args.total_ticks,
            "mmac_active_percent": args.mmac_active,
        },
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

