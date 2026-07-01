#pragma once

#include <cstdint>

namespace shaobo::fa3::bwd::dkv {

inline constexpr int kDkvPathWaspProbe = 0;
inline constexpr int kDkvPathReferenceCorrectness = 1;
inline constexpr int kDkvPathWaspSoftmaxDsSidecar = 2;

struct DkvTileD128Mq32Nk128 {
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockMq = 32;
    static constexpr int kProbeSeqLen = 1024;
    static constexpr int kQTilesPerCta = kProbeSeqLen / kBlockMq;
    static constexpr int kProbeDiagFloatsPerBlock = 24;
    static constexpr int kProbeScoreDiagBase = 0;
    static constexpr int kProbeProbDiagBase = 8;
    static constexpr int kProbeDsDiagBase = 16;
    static constexpr int kNkPerConsumerWave = 16;
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
    static constexpr int kRawBuffers = 2;
    static constexpr int kRawBytes =
        kRawBuffers * 2 * kBlockMq * kHeadDim * kHalfBytes;
    static constexpr int kDoutTBuffers = 2;
    static constexpr int kDoutTBytes =
        kDoutTBuffers * kBlockMq * kHeadDim * kHalfBytes;
    static constexpr int kSidecarBytes =
        kRawBuffers * 3 * kBlockMq * static_cast<int>(sizeof(float));
    static constexpr int kPlannedLdsBytes =
        kKvBytes + kRawBytes + kDoutTBytes + kSidecarBytes;
    static constexpr int kLdsBudgetBytes = 128 * 1024;

    static_assert(kResidentNk == 128, "dKV clean lane expects Nk=128");
    static_assert(kTotalMmacPerConsumer == 64,
                  "Balanced dKV consumer should issue 64 MMAC per q tile");
    static_assert(kPlannedLdsBytes <= kLdsBudgetBytes,
                  "dKV clean LDS plan must fit 128KB");
};

struct DkvBarrierLedger {
    static constexpr int kResidentFilled = 0;
    static constexpr int kRaw0Filled = 1;
    static constexpr int kRaw0Used = 2;
    static constexpr int kRaw1Filled = 3;
    static constexpr int kRaw1Used = 4;
    static constexpr int kAllDone = 5;
};

enum class WaveRole : int {
    kProducerA = 0,
    kConsumerGroup0 = 1,
    kConsumerGroup1 = 2,
    kProducerB = 3,
};

inline constexpr WaveRole role_for_wave(int wave_id) {
    return wave_id < 4
               ? WaveRole::kProducerA
               : (wave_id < 8 ? WaveRole::kConsumerGroup0
                              : (wave_id < 12 ? WaveRole::kConsumerGroup1
                                               : WaveRole::kProducerB));
}

struct OptimizationTargets {
    static constexpr int kTargetMmacActiveSharePercent = 60;
    static constexpr bool kRequireNoScratch = true;
    static constexpr bool kRequireNoSpill = true;
    static constexpr bool kRequireNoLdsBankConflict = true;
    static constexpr bool kForbidDuplicateScoreDp = true;
};

struct WdraResourceWindows {
    static constexpr int kProducerVgprs = 8;
    static constexpr int kConsumerVgprs = 160;
    static constexpr int kConsumerTargetVgprs = 200;
    static constexpr int kConsumerCeilingVgprs = 248;
};

}  // namespace shaobo::fa3::bwd::dkv
