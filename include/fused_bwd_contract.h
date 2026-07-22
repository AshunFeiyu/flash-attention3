#pragma once

#include <cstdint>

namespace shaobo::fa3::bwd::fused_bwd {

struct FusedBwdContract {
    static constexpr int kMq = 64;
    static constexpr int kNk = 128;
    static constexpr int kHeadDim = 128;
    static constexpr int kWaveSize = 64;
    static constexpr int kWavesPerProducer = 4;
    static constexpr int kWavesPerConsumer = 4;
    static constexpr int kConsumerGroups = 2;
    static constexpr int kProducerWaveBegin = 0;
    static constexpr int kConsumer0WaveBegin = 4;
    static constexpr int kConsumer1WaveBegin = 8;
    static constexpr int kWavesPerCta = kWavesPerProducer +
                                        kConsumerGroups * kWavesPerConsumer;
    static constexpr int kThreadsPerCta = kWavesPerCta * kWaveSize;
    static constexpr int kNkPerConsumer = kNk / kConsumerGroups;
    static constexpr int kMqPerConsumerWave = 16;
    static constexpr int kNkPerPanel = 32;
    static constexpr int kPanelsPerConsumer = kNkPerConsumer / kNkPerPanel;
    static constexpr int kConsumerWaves =
        kConsumerGroups * kWavesPerConsumer;
    static constexpr int kPanelCount =
        kConsumerWaves * kPanelsPerConsumer;

    static constexpr int kMmacM = 16;
    static constexpr int kMmacN = 16;
    static constexpr int kMmacK = 16;
    static constexpr int kHalfBytes = 2;
    static constexpr int kMmacPerLogicalGemm =
        (kMq / kMmacM) * (kNk / kMmacN) * (kHeadDim / kMmacK);
    static constexpr int kMmacPerConsumerPartial =
        (kMq / kMmacM) * (kNkPerConsumer / kMmacN) *
        (kHeadDim / kMmacK);

    enum class LogicalGemm : uint8_t {
        kScore,
        kDp,
        kDv,
        kDsToDk,
        kDsToDq,
        kCount,
    };
    static constexpr int kLogicalGemmCount =
        static_cast<int>(LogicalGemm::kCount);
    static constexpr int kGemmCount = 5;
    static constexpr const char* kScore = "score=Q@K^T";
    static constexpr const char* kDp = "dP=dO@V^T";
    static constexpr const char* kDv = "dV=P^T@dO";
    static constexpr const char* kDk = "dK=dS^T@Q";
    static constexpr const char* kDq = "dQ=dS@K";
    static constexpr int kMmacPerTile =
        kLogicalGemmCount * kMmacPerLogicalGemm;
    static constexpr int kMmacPerConsumer =
        kLogicalGemmCount * kMmacPerConsumerPartial;

    static constexpr bool kDqIsFullDPartial = true;
    static constexpr bool kDqUsesFp32AtomicAdd = true;

    static constexpr int kResidentKvBytes =
        2 * kNk * kHeadDim * kHalfBytes;
    static constexpr int kRawQDoBytes =
        2 * kMq * kHeadDim * kHalfBytes;
    static constexpr int kPdsLogicalBytes = kMq * kNk * kHalfBytes;
    // One t=1 m32x16 writer carries 1KB of values in a 2KB physical
    // footprint. Each consumer wave owns one page and reuses it for its two
    // N32 panels, then reuses the page again for P and dS.
    static constexpr int kWriterPageBytes = 64 * 16 * kHalfBytes;
    static constexpr int kActiveWriterPages = kConsumerWaves;
    static constexpr int kPdsPageBytes =
        kActiveWriterPages * kWriterPageBytes;
    static constexpr int kPdsPageCount = 1;
    static constexpr int kSidecarBytes = 3 * kMq * sizeof(float);
    static constexpr bool kSidecarUsesDedicatedLds = true;
    static constexpr int kLdsBudgetBytes = 128 * 1024;
    static constexpr int kPlannedLdsBytes =
        kResidentKvBytes + kRawQDoBytes + kPdsPageCount * kPdsPageBytes +
        kSidecarBytes;

    enum class LifetimeState : uint8_t {
        kResidentKvPublished,
        kResidentKvLatched,
        kRawQDoPublished,
        kRawQDoConsumed,
        kPPagePublished,
        kDsPageConsumed,
        kComplete,
        kCount,
    };

    enum class PdsPhase : uint8_t {
        kP,
        kDs,
        kCount,
    };

    static_assert(kMq == 64 && kNk == 128 && kHeadDim == 128,
                  "fused FA3 BWD contract is fixed to M64/N128/D128");
    static_assert(kWavesPerCta == 12 && kProducerWaveBegin == 0 &&
                      kConsumer0WaveBegin == 4 && kConsumer1WaveBegin == 8,
                  "wave roles must be producer 0-3, consumer0 4-7, consumer1 8-11");
    static_assert(kNk == kConsumerGroups * kNkPerConsumer,
                  "consumers must partition Nk into two N64 owners");
    static_assert(kMq == kWavesPerConsumer * kMqPerConsumerWave,
                  "each consumer wave owns one M16 score/dP row panel");
    static_assert(kNkPerConsumer ==
                      kPanelsPerConsumer * kNkPerPanel,
                  "each consumer covers two N32 native-writer panels");
    static_assert(kMq % kMmacM == 0 && kNk % kMmacN == 0 &&
                      kHeadDim % kMmacK == 0,
                  "tile dimensions must be MMAC aligned");
    static_assert(kLogicalGemmCount == 5,
                  "fused backward must contain exactly five logical GEMMs");
    static_assert(kGemmCount == kLogicalGemmCount,
                  "formula and enum GEMM counts must agree");
    static_assert(kMmacPerLogicalGemm == 256 && kMmacPerTile == 1280,
                  "five-GEMM tile work must be 256 MMAC per GEMM and 1280 total");
    static_assert(kMmacPerConsumer == 5 * kMmacPerConsumerPartial,
                  "each N64 consumer must execute every logical GEMM fragment");
    static_assert(kDqIsFullDPartial && kDqUsesFp32AtomicAdd,
                  "dQ must be a full-D FP32 atomic partial from both consumers");
    static_assert(kResidentKvBytes == 64 * 1024 &&
                      kRawQDoBytes == 32 * 1024 &&
                      kPdsLogicalBytes == 16 * 1024 &&
                      kPdsPageBytes == 16 * 1024,
                  "resident/raw/P-dS LDS regions must retain their fixed sizes");
    static_assert(kPdsPageCount == 1,
                  "P and dS must reuse one shared LDS page by phase");
    static_assert(kPlannedLdsBytes <= kLdsBudgetBytes,
                  "fused FA3 BWD LDS plan must fit within 128KB");
    static_assert(kPlannedLdsBytes == 112 * 1024 + 3 * 64 * 4,
                  "canonical physical LDS plan is 112.75KB");
};

struct FusedBwdBarrierLedger {
    static constexpr int kResidentFilled = 0;
    static constexpr int kRawFilled = 1;
    static constexpr int kRawUsed = 2;
    static constexpr int kCount = 3;

    static_assert(kRawUsed + 1 == kCount,
                  "barrier IDs must be contiguous and explicit");
};

struct FusedBwdWdraResourceWindow {
    static constexpr int kProducerVgprs = 24;
    static constexpr int kConsumerVgprs = 240;
    static constexpr int kPhysicalVgprTarget =
        kProducerVgprs + 2 * kConsumerVgprs;
    static constexpr int kPhysicalVgprGuard = 504;
    static constexpr int kPhysicalVgprBudget = 512;
    static constexpr bool kRequireNoSpill = true;
    static constexpr bool kRequireNoPrivateSegment = true;
    static constexpr bool kRequireNoScratch = true;

    static_assert(kPhysicalVgprTarget <= kPhysicalVgprGuard,
                  "physical WDRA target must stay at or below 504 VGPRs");
    static_assert(kPhysicalVgprTarget <= kPhysicalVgprBudget,
                  "physical WDRA target must fit the 512 VGPR budget");
    static_assert(kPhysicalVgprTarget % 3 == 0 &&
                      (kPhysicalVgprTarget / 3) % 8 == 0,
                  "three-role average must satisfy the compiler VGPR granularity");
    static_assert(kRequireNoSpill, "fused path must be spill-free");
    static_assert(kRequireNoPrivateSegment && kRequireNoScratch,
                  "fused path must not allocate private or scratch storage");
};

using ActiveFusedBwdContract = FusedBwdContract;

}  // namespace shaobo::fa3::bwd::fused_bwd
