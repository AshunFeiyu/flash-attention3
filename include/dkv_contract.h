#pragma once

#include <cstdint>

namespace shaobo::fa3::bwd::dkv {

inline constexpr int kDkvPathReferenceCorrectness = 1;
inline constexpr int kDkvPathCanonicalDkv = 5;

struct ActiveDkvTile {
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockMq = 128;
    static constexpr int kNkPerConsumerWave = 32;
    static constexpr int kConsumerGroups = 2;
    static constexpr int kWavesPerConsumerGroup = 4;
    static constexpr int kConsumerWaves =
        kConsumerGroups * kWavesPerConsumerGroup;
    static constexpr int kResidentNk =
        kNkPerConsumerWave * kConsumerWaves;
    static constexpr int kWaveSize = 64;
    static constexpr int kWavesPerCta = 16;
    static constexpr int kThreadsPerCta = kWaveSize * kWavesPerCta;

    static constexpr int kMmacM = 16;
    static constexpr int kMmacN = 16;
    static constexpr int kMmacK = 16;

    static constexpr int kScoreMmacPerConsumer =
        (kBlockMq / kMmacM) * (kNkPerConsumerWave / kMmacN) *
        (kHeadDim / kMmacK);
    static constexpr int kDpMmacPerConsumer = kScoreMmacPerConsumer;
    static constexpr int kDvMmacPerConsumer =
        (kNkPerConsumerWave / kMmacM) * (kHeadDim / kMmacN) *
        (kBlockMq / kMmacK);
    static constexpr int kDkMmacPerConsumer = kDvMmacPerConsumer;
    static constexpr int kTotalMmacPerConsumer =
        kScoreMmacPerConsumer + kDpMmacPerConsumer +
        kDvMmacPerConsumer + kDkMmacPerConsumer;

    static constexpr int kHalfBytes = 2;
    static constexpr int kKvBytes =
        2 * kResidentNk * kHeadDim * kHalfBytes;
    static constexpr int kRawBuffers = 1;
    static constexpr int kRawBytes =
        kRawBuffers * 2 * kBlockMq * kHeadDim * kHalfBytes;
    static constexpr int kSidecarRows = kBlockMq;
    static constexpr int kSidecarMaxLog2Base = 0;
    static constexpr int kSidecarInvSumBase =
        kSidecarMaxLog2Base + kSidecarRows;
    static constexpr int kSidecarDeltaBase =
        kSidecarInvSumBase + kSidecarRows;
    static constexpr int kSidecarFloats =
        kSidecarDeltaBase + kSidecarRows;
    static constexpr int kPackedSidecarFields = 3;
    static constexpr int kSidecarBytes =
        kRawBuffers * kSidecarFloats * static_cast<int>(sizeof(float));
    static constexpr int kSourceLayoutBytes = 0;
    static constexpr int kLdsBudgetBytes = 128 * 1024;
    static constexpr bool kOverlayRawOnResidentKv =
        kKvBytes + kRawBytes + kSidecarBytes + kSourceLayoutBytes >
        kLdsBudgetBytes;
    static constexpr int kSteadyRawBytes =
        kRawBytes + kSidecarBytes + kSourceLayoutBytes;
    static constexpr int kPlannedLdsBytes =
        kOverlayRawOnResidentKv
            ? (kSteadyRawBytes > kKvBytes ? kSteadyRawBytes : kKvBytes)
            : kKvBytes + kSteadyRawBytes;

    static_assert(kBlockMq % 32 == 0,
                  "dKV clean tile consumes Mq in pairs of M16 blocks");
    static_assert(kResidentNk == 256, "dKV owner32 lane expects Nk=256");
    static_assert(kKvBytes == kLdsBudgetBytes,
                  "resident K/V startup epoch must occupy exactly 128KB");
    static_assert(kTotalMmacPerConsumer == 4 * kBlockMq,
                  "dKV consumer MMAC count should scale with BlockMq");
    static_assert(kPlannedLdsBytes <= kLdsBudgetBytes,
                  "dKV clean LDS plan must fit 128KB");
};

struct DkvBarrierLedger {
    static constexpr int kResidentFilled = 0;
    static constexpr int kResidentUsed = 1;
    static constexpr int kQ0Filled = 2;
    static constexpr int kQ0Used = 3;
    static constexpr int kDout0Used = 4;
    static constexpr int kQ1Filled = 5;
    static constexpr int kQ1Used = 6;
    static constexpr int kDout1Used = 7;
    static constexpr int kAllDone = 8;
};

struct OptimizationTargets {
    static constexpr int kTargetMmacActiveSharePercent = 60;
    static constexpr bool kRequireNoScratch = true;
    static constexpr bool kRequireNoSpill = true;
    static constexpr bool kRequireNoLdsBankConflict = true;
    static constexpr bool kForbidDuplicateScoreDp = true;
};

struct WdraResourceWindows {
    static constexpr int kProducerKqVgprs = 16;
    static constexpr int kConsumer0Vgprs = 244;
    static constexpr int kConsumer1Vgprs = 244;
    static constexpr int kProducerVdoutVgprs = 8;
    static constexpr int kConsumerTargetVgprs = 200;
    static constexpr int kConsumerCeilingVgprs = 248;
    static_assert(
        kProducerKqVgprs + kConsumer0Vgprs + kConsumer1Vgprs +
                kProducerVdoutVgprs ==
            512,
        "per-SIMD WDRA windows must exactly fit the Shaobo VGPR file");
};

}  // namespace shaobo::fa3::bwd::dkv
