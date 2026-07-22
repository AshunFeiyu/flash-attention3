#pragma once

#include <cstdint>

namespace shaobo::fa3::bwd::dkv {

inline constexpr int kDkvPathReferenceCorrectness = 1;
inline constexpr int kDkvPathCanonicalDkv = 5;

struct ActiveDkvTile {
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockMq = 128;
    static constexpr int kRawPageMq = 64;
    static constexpr int kRawPages = 2;
    static constexpr int kHeadReadyMq = kRawPageMq;
    static constexpr int kTailReadyMq = kRawPageMq;
    static constexpr int kNkPerConsumerWave = 16;
    static constexpr int kConsumerGroups = 2;
    static constexpr int kWavesPerConsumerGroup = 4;
    static constexpr int kConsumerWaves =
        kConsumerGroups * kWavesPerConsumerGroup;
    static constexpr int kLogicalConsumer0Rows = 64;
    static constexpr int kLogicalConsumer1Rows = 32;
    static constexpr int kLogicalConsumer2Rows = 32;
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
    static constexpr int kRawBytes =
        kRawPages * 2 * kRawPageMq * kHeadDim * kHalfBytes;
    static constexpr int kSidecarRows = kRawPageMq;
    static constexpr int kSidecarMaxLog2Base = 0;
    static constexpr int kSidecarInvSumBase =
        kSidecarMaxLog2Base + kSidecarRows;
    static constexpr int kSidecarDeltaBase =
        kSidecarInvSumBase + kSidecarRows;
    static constexpr int kSidecarFloats =
        kSidecarDeltaBase + kSidecarRows;
    static constexpr int kPackedSidecarFields = 3;
    static constexpr int kSidecarBytes =
        kRawPages * kSidecarFloats * static_cast<int>(sizeof(float));
    static constexpr int kSourceLayoutBytes = 0;
    static constexpr int kLdsBudgetBytes = 128 * 1024;
    static constexpr bool kOverlaySidecarOnResidentKv = true;
    static constexpr int kPlannedLdsBytes =
        kKvBytes + kRawBytes + kSourceLayoutBytes;

    static_assert(kBlockMq % kWaveSize == 0,
                  "sidecar publisher requires complete 64-row wave stripes");
    static_assert(kHeadReadyMq % kWaveSize == 0 &&
                      kTailReadyMq % kWaveSize == 0,
                  "head and tail readiness must use complete wave stripes");
    static_assert(kRawPages * kRawPageMq == kBlockMq,
                  "two M64 physical pages must cover one Mq128 tile");
    static_assert(kBlockMq <= 4 * kWaveSize,
                  "four producer waves must cover one sidecar packet");
    static_assert(kLogicalConsumer0Rows + kLogicalConsumer1Rows +
                          kLogicalConsumer2Rows ==
                      kResidentNk,
                  "logical 64/32/32 ownership must cover Nk exactly");
    static_assert(kResidentNk == 128,
                  "owner16 physical 2P2C tile expects Nk=128");
    static_assert(kKvBytes == 64 * 1024,
                  "resident K/V startup epoch must occupy exactly 64KB");
    static_assert(kRawBytes == 64 * 1024,
                  "two physical M64 Q/dO pages must occupy exactly 64KB");
    static_assert(kSidecarBytes <= kKvBytes,
                  "sidecar overlay must fit the released resident region");
    static_assert(kPlannedLdsBytes == kLdsBudgetBytes,
                  "consumer-assisted dKV plan intentionally uses 128KB LDS");
    static_assert(kTotalMmacPerConsumer == 2 * kBlockMq,
                  "dKV consumer MMAC count should scale with BlockMq");
    static_assert(kPlannedLdsBytes <= kLdsBudgetBytes,
                  "dKV clean LDS plan must fit 128KB");
};

struct DkvBarrierLedger {
    static constexpr int kResidentFilled = 0;
    static constexpr int kResidentUsed = 1;
    static constexpr int kRawHeadFilled = 2;
    static constexpr int kRawHeadUsed = 3;
    static constexpr int kRawTailFilled = 4;
    static constexpr int kRawTailUsed = 5;
    static constexpr int kAllDone = 6;
};

struct OptimizationTargets {
    static constexpr int kTargetMmacActiveSharePercent = 60;
    static constexpr bool kRequireNoScratch = true;
    static constexpr bool kRequireNoSpill = true;
    static constexpr bool kRequireNoLdsBankConflict = true;
    static constexpr bool kForbidDuplicateScoreDp = true;
};

struct WdraResourceWindows {
    static constexpr int kProducerVgprs = 32;
    static constexpr int kConsumerVgprs = 192;
    static constexpr int kConsumerTargetVgprs = 192;
    static constexpr int kConsumerCeilingVgprs = 192;
    static_assert(2 * kProducerVgprs + 2 * kConsumerVgprs <= 512,
                  "per-SIMD WDRA window sum must fit 512 VGPRs");
};

}  // namespace shaobo::fa3::bwd::dkv
