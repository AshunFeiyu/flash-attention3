#pragma once

#include <cstdint>

namespace shaobo::fa3::bwd::dkv {

inline constexpr int kDkvPathWaspProbe = 0;
inline constexpr int kDkvPathReferenceCorrectness = 1;
inline constexpr int kDkvPathWaspSoftmaxDsSidecar = 2;
inline constexpr int kDkvPathWaspFragmentSidecar = 3;
inline constexpr int kDkvPathWaspDkvMmac = 4;
inline constexpr int kDkvPathWaspDkvMmac12Wave = 5;
inline constexpr int kDkvPathWaspDkvMmac12WaveMq64 = 6;
inline constexpr int kDkvPathWaspDkvMmac12WaveSidecarOverlay = 7;
inline constexpr int kDkvPathWaspDkvMmac12WaveScoreDpBrick = 8;
inline constexpr int kDkvPathWaspDkvMmac12WaveMq64Semantic = 9;
inline constexpr int kDkvPathWaspDkvMmac12WaveCausalSkip = 10;
inline constexpr int kDkvPathWaspDkvMmac12WaveMixedScoreBrick = 11;

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
    static constexpr int kSourceLayoutBytes =
        kRawBuffers * 2 * kBlockMq * kHeadDim * kHalfBytes;
    static constexpr int kPlannedLdsBytes =
        kKvBytes + kRawBytes + kSourceLayoutBytes;
    static constexpr int kLdsBudgetBytes = 128 * 1024;

    static_assert(kResidentNk == 128, "dKV clean lane expects Nk=128");
    static_assert(kTotalMmacPerConsumer == 64,
                  "Balanced dKV consumer should issue 64 MMAC per q tile");
    static_assert(kPlannedLdsBytes <= kLdsBudgetBytes,
                  "dKV clean LDS plan must fit 128KB");
};

struct DkvTileD128Mq32Nk128W12 : DkvTileD128Mq32Nk128 {
    static constexpr int kWavesPerCta = 12;
    static constexpr int kThreadsPerCta = kWaveSize * kWavesPerCta;
};

struct DkvTileD128Mq64Nk128W12 {
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockMq = 64;
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
    static constexpr int kWavesPerCta = 12;
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
    static constexpr int kRawBuffers = 1;
    static constexpr int kKvBytes =
        2 * kResidentNk * kHeadDim * kHalfBytes;
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
    static constexpr int kSourceLayoutBytes =
        kRawBuffers * 2 * kBlockMq * kHeadDim * kHalfBytes;
    static constexpr int kPlannedLdsBytes =
        kKvBytes + kRawBytes + kSourceLayoutBytes;
    static constexpr int kLdsBudgetBytes = 128 * 1024;

    static_assert(kResidentNk == 128, "dKV Mq64 lane expects Nk=128");
    static_assert(kTotalMmacPerConsumer == 128,
                  "Mq64 dKV consumer should issue 128 MMAC per q tile");
    static_assert(kPlannedLdsBytes == kLdsBudgetBytes,
                  "Mq64 single-buffer plan intentionally fills 128KB LDS");
};

struct DkvTileD128Mq64SemanticNk128W12 : DkvTileD128Mq64Nk128W12 {
    static constexpr int kSemanticPages = 2;
    static constexpr int kSemanticMatrixBytes =
        kBlockMq * kHeadDim * kHalfBytes;
    static constexpr int kSemanticPageBytes = 2 * kSemanticMatrixBytes;
    static constexpr int kSemanticLdsBytes =
        kKvBytes + kSemanticPages * kSemanticPageBytes;

    static_assert(kSemanticMatrixBytes == 16 * 1024,
                  "Mq64 semantic matrix page should be 16KB");
    static_assert(kSemanticPageBytes == 32 * 1024,
                  "Mq64 semantic page should hold Q/dO or QT/dOT");
    static_assert(kSemanticLdsBytes == kLdsBudgetBytes,
                  "Mq64 semantic-page conveyor intentionally fills 128KB LDS");
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
    static constexpr int kProducer12Vgprs = 16;
    static constexpr int kConsumerVgprs = 160;
    static constexpr int kConsumerMq64Vgprs = 208;
    static constexpr int kConsumerTargetVgprs = 200;
    static constexpr int kConsumerCeilingVgprs = 248;
};

}  // namespace shaobo::fa3::bwd::dkv
