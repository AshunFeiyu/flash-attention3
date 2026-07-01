#pragma once

#include <cstdint>

namespace shaobo::fa3::bwd::stage61 {

struct DkvTileD128Mq32Nk128 {
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockMq = 32;
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

    static_assert(kResidentNk == 128, "Stage61 clean lane expects Nk=128");
    static_assert(kTotalMmacPerConsumer == 64,
                  "Balanced dKV consumer should issue 64 MMAC per q tile");
    static_assert(kPlannedLdsBytes <= kLdsBudgetBytes,
                  "Stage61 clean LDS plan must fit 128KB");
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

}  // namespace shaobo::fa3::bwd::stage61

