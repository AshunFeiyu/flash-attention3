#!/usr/bin/env python3
"""Prove fused5 causal CTA permutations and per-shape admission decisions."""

from __future__ import annotations


CU_COUNT = 48
IDENTITY = 0
SERPENTINE = 1


def remap(physical: int, k_tiles: int, mode: int, width: int) -> int:
    if mode == IDENTITY:
        return physical
    round_index = physical // width
    slot = physical - round_index * width
    base = round_index * width
    round_width = min(width, k_tiles - base)
    return base + (slot if round_index % 2 == 0 else round_width - 1 - slot)


def modeled_max(k_tiles: int, bh_count: int, mode: int, width: int) -> int:
    loads = [0] * CU_COUNT
    remainder = bh_count % CU_COUNT
    q_tiles = 2 * k_tiles
    for physical in range(k_tiles):
        logical = remap(physical, k_tiles, mode, width)
        work = q_tiles - 2 * logical
        start = physical * bh_count % CU_COUNT
        for offset in range(remainder):
            loads[(start + offset) % CU_COUNT] += work
    return max(loads)


def plan(k_tiles: int, bh_count: int, causal: bool) -> tuple[int, int, int, int]:
    baseline = modeled_max(k_tiles, bh_count, IDENTITY, 1)
    best_max, best_mode, best_width = baseline, IDENTITY, 1
    if not causal or bh_count % CU_COUNT == 0:
        return best_mode, best_width, baseline, best_max
    for width in range(1, min(k_tiles, CU_COUNT) + 1):
        candidate = modeled_max(k_tiles, bh_count, SERPENTINE, width)
        if candidate < best_max:
            best_max, best_mode, best_width = candidate, SERPENTINE, width
    return best_mode, best_width, baseline, best_max


def main() -> int:
    for k_tiles in range(1, 129):
        for width in range(1, min(k_tiles, CU_COUNT) + 1):
            order = [remap(p, k_tiles, SERPENTINE, width) for p in range(k_tiles)]
            assert sorted(order) == list(range(k_tiles))

    representative_bh = (1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 47, 48, 49, 64)
    representative_k = (1, 2, 4, 8, 16, 32, 64, 128)
    improved = 0
    unchanged = 0
    for bh_count in representative_bh:
        for k_tiles in representative_k:
            mode, width, baseline, candidate = plan(k_tiles, bh_count, True)
            assert candidate <= baseline
            if candidate < baseline:
                improved += 1
            else:
                unchanged += 1
            if mode == IDENTITY:
                assert width == 1 and candidate == baseline

    mode, width, baseline, candidate = plan(64, 16, True)
    assert (mode, width, baseline, candidate) == (SERPENTINE, 3, 1430, 1390)
    assert plan(64, 16, False)[:2] == (IDENTITY, 1)
    assert plan(64, 48, True)[:2] == (IDENTITY, 1)

    print("fused5 causal CTA order: PASS")
    print(
        "  representative_shapes="
        f"{improved + unchanged} improved={improved} unchanged={unchanged}"
    )
    print(
        "  h16_s8192="
        f"mode=serpentine width={width} max:{baseline}->{candidate} "
        f"gain={(baseline - candidate) / baseline:.6%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
