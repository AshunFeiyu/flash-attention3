#pragma once

#include <cstdint>

namespace shaobo::fa3::bwd::dq {

inline constexpr int kDqPathReferenceCorrectness = 1;
inline constexpr int kDqPathCanonicalDq = 6;

template <int BlockMq, int BlockNk>
struct DqTileD128MqNk {
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockMq = BlockMq;
    static constexpr int kBlockNk = BlockNk;
    static constexpr int kWaveSize = 64;
    static constexpr int kWavesPerCta = 16;
    static constexpr int kThreadsPerCta = kWaveSize * kWavesPerCta;

    static constexpr int kMmacM = 16;
    static constexpr int kMmacN = 16;
    static constexpr int kMmacK = 16;
    static constexpr int kScoreMmacPerKTile =
        (kBlockMq / kMmacM) * (kBlockNk / kMmacN) *
        (kHeadDim / kMmacK);
    static constexpr int kDpMmacPerKTile = kScoreMmacPerKTile;
    static constexpr int kDqMmacPerKTile = kScoreMmacPerKTile;
    static constexpr int kTotalMmacPerKTile =
        kScoreMmacPerKTile + kDpMmacPerKTile + kDqMmacPerKTile;

    static constexpr int kHalfBytes = 2;
    static constexpr int kQDoBytes =
        2 * kBlockMq * kHeadDim * kHalfBytes;
    static constexpr int kKvBytes =
        2 * kBlockNk * kHeadDim * kHalfBytes;
    static constexpr int kSidecarFloats = 3 * kBlockMq;
    static constexpr int kSidecarBytes =
        kSidecarFloats * static_cast<int>(sizeof(float));
    static constexpr int kLdsBudgetBytes = 128 * 1024;
    static constexpr int kPlannedLdsBytes =
        kQDoBytes + kKvBytes + kSidecarBytes;

    static_assert(kBlockMq % 32 == 0,
                  "dQ target consumes Mq in M32 consumer pairs");
    static_assert(kBlockNk % 32 == 0,
                  "dQ K/V stream tile must align to 32-row MLS tiles");
    static_assert(kHeadDim == 128, "first clean dQ path is D128-only");
    static_assert(kPlannedLdsBytes <= kLdsBudgetBytes,
                  "dQ target LDS plan must fit 128KB");
};

using ActiveDqTile = DqTileD128MqNk<64, 128>;

struct DqBarrierLedger {
    static constexpr int kQDoFilled = 0;
    static constexpr int kQDoUsed = 1;
    static constexpr int kKvFilled = 2;
    static constexpr int kKvUsed = 3;
    static constexpr int kAllDone = 4;
};

struct OptimizationTargets {
    static constexpr int kTargetMmacActiveSharePercent = 60;
    static constexpr bool kRequireNoScratch = true;
    static constexpr bool kRequireNoSpill = true;
    static constexpr bool kRequireNoLdsBankConflict = true;
    static constexpr bool kForbidDuplicateScoreDpInsideDq = true;
    static constexpr bool kForbidDqAtomicAdd = true;
};

struct WdraResourceWindows {
    static constexpr int kProducerVgprs = 16;
    static constexpr int kConsumerTargetVgprs = 220;
    static constexpr int kConsumerCeilingVgprs = 248;
};

}  // namespace shaobo::fa3::bwd::dq
