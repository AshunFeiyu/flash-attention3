#pragma once

#include <cstdint>

namespace shaobo::fa3::bwd::fused_bwd {

struct FusedBwdContract {
    static constexpr int kMq = 64;
    static constexpr int kNk = 128;
    static constexpr int kHeadDim = 128;
    static constexpr int kWaveSize = 64;
    static constexpr int kWavesPerProducer = 4;
    static constexpr int kConsumerGroups = 2;
    static constexpr int kWavesPerConsumerGroup = 4;
    static constexpr int kProducerWaveBegin = 0;
    static constexpr int kConsumer0WaveBegin = 4;
    static constexpr int kConsumer1WaveBegin = 8;
    static constexpr int kWavesPerCta =
        kWavesPerProducer + kConsumerGroups * kWavesPerConsumerGroup;
    static constexpr int kThreadsPerCta = kWavesPerCta * kWaveSize;
    static constexpr int kMqPerPanel = 16;
    static constexpr int kMqPanels = kMq / kMqPerPanel;
    static constexpr int kNkPerConsumerWave =
        kNk / (kConsumerGroups * kWavesPerConsumerGroup);
    static constexpr int kNkPerConsumerGroup = kNk / kConsumerGroups;
    static constexpr int kHeadDimPerConsumerWave =
        kHeadDim / (kConsumerGroups * kWavesPerConsumerGroup);

    static constexpr int kMmacM = 16;
    static constexpr int kMmacN = 16;
    static constexpr int kMmacK = 16;
    static constexpr int kHalfBytes = 2;
    static constexpr int kMmacPerLogicalGemm =
        (kMq / kMmacM) * (kNk / kMmacN) * (kHeadDim / kMmacK);
    static constexpr int kMmacPerConsumerWaveDkv =
        4 * (kMq / kMmacM) * (kNkPerConsumerWave / kMmacN) *
        (kHeadDim / kMmacK);
    static constexpr int kMmacPerConsumerWaveDq =
        (kMq / kMmacM) * (kNk / kMmacN) *
        (kHeadDimPerConsumerWave / kMmacK);
    static constexpr int kMmacPerConsumerWave =
        kMmacPerConsumerWaveDkv + kMmacPerConsumerWaveDq;

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
    static constexpr bool kDkvHasUniqueOutputOwner = true;
    static constexpr bool kDqHasUniqueD16Owner = true;
    static constexpr bool kDqUsesFp32AtomicAdd = true;

    static constexpr int kResidentKvBytes =
        2 * kNk * kHeadDim * kHalfBytes;
    static constexpr int kRawQDoBytes =
        2 * kMq * kHeadDim * kHalfBytes;
    static constexpr int kPdsLogicalBytes = kMq * kNk * kHalfBytes;
    // One t=1 m32x16 writer carries one real N16 half plus one zero-padded
    // N16 half in a 2KB physical footprint. Eight owners publish two
    // independent N64 groups and fully consume the pages before reuse.
    static constexpr int kWriterPageBytes = 64 * 16 * kHalfBytes;
    static constexpr int kActiveWriterPages =
        kConsumerGroups * kWavesPerConsumerGroup;
    static constexpr int kPdsGenerationBytes =
        kActiveWriterPages * kWriterPageBytes;
    static constexpr int kPdsGenerationCount = 1;
    static constexpr int kPdsPageBytes =
        kPdsGenerationCount * kPdsGenerationBytes;
    static constexpr int kSidecarBytes = 3 * kMq * sizeof(float);
    static constexpr bool kSidecarAliasesPds = false;
    static constexpr int kLdsBudgetBytes = 128 * 1024;
    static constexpr int kPlannedLdsBytes =
        kResidentKvBytes + kRawQDoBytes + kPdsPageBytes + kSidecarBytes;

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
                  "wave roles must be producer 0-3 and symmetric groups 4-7/8-11");
    static_assert(kMqPanels == 4 && kNkPerConsumerWave == 16 &&
                      kNkPerConsumerGroup == 64 &&
                      kHeadDimPerConsumerWave == 16,
                  "each consumer must own N16 dKV and globally unique D16 dQ");
    static_assert(kMq % kMmacM == 0 && kNk % kMmacN == 0 &&
                      kHeadDim % kMmacK == 0,
                  "tile dimensions must be MMAC aligned");
    static_assert(kLogicalGemmCount == 5,
                  "fused backward must contain exactly five logical GEMMs");
    static_assert(kGemmCount == kLogicalGemmCount,
                  "formula and enum GEMM counts must agree");
    static_assert(kMmacPerLogicalGemm == 256 && kMmacPerTile == 1280,
                  "five-GEMM tile work must be 256 MMAC per GEMM and 1280 total");
    static_assert(kConsumerGroups * kWavesPerConsumerGroup *
                          kMmacPerConsumerWave ==
                      kMmacPerTile,
                  "owner work must equal exactly five logical GEMMs");
    static_assert(kDkvHasUniqueOutputOwner && kDqHasUniqueD16Owner &&
                      kDqUsesFp32AtomicAdd,
                  "dKV must store once and dQ must emit one D16 partial");
    static_assert(kResidentKvBytes == 64 * 1024 &&
                      kRawQDoBytes == 32 * 1024 &&
                      kPdsLogicalBytes == 16 * 1024 &&
                      kPdsGenerationBytes == 16 * 1024 &&
                      kPdsPageBytes == 16 * 1024,
                  "resident/raw/P-dS LDS regions must retain their fixed sizes");
    static_assert(kPdsGenerationCount == 1,
                  "symmetric path uses one fully-consumed dS generation");
    static_assert(kPlannedLdsBytes <= kLdsBudgetBytes,
                  "fused FA3 BWD LDS plan must fit within 128KB");
    static_assert(!kSidecarAliasesPds && kSidecarBytes == 768,
                  "sidecar must remain independently readable per panel");
    static_assert(kPlannedLdsBytes == 115456,
                  "symmetric physical LDS plan must use 112.75KB");
};

struct FusedBwdBarrierLedger {
    static constexpr int kResidentFilled0 = 0;
    static constexpr int kResidentFilled1 = 1;
    static constexpr int kRawFilled = 2;
    static constexpr int kRawUsed = 3;
    static constexpr int kDsFilledG0 = 4;
    static constexpr int kDsUsedG0 = 5;
    static constexpr int kDsFilledG1 = 6;
    static constexpr int kDsUsedG1 = 7;
    static constexpr int kCount = 8;

    static_assert(kDsUsedG1 + 1 == kCount,
                  "barrier IDs must be contiguous and explicit");
};

struct FusedBwdWdraResourceWindow {
    static constexpr int kProducerVgprs = 32;
    static constexpr int kConsumer0Vgprs = 176;
    static constexpr int kConsumer1Vgprs = 176;
    static constexpr int kPhysicalVgprTarget =
        kProducerVgprs + kConsumer0Vgprs + kConsumer1Vgprs;
    static constexpr int kPhysicalVgprGuard = 384;
    static constexpr int kPhysicalVgprBudget = 512;
    static constexpr bool kRequireNoSpill = true;
    static constexpr bool kRequireNoPrivateSegment = true;
    static constexpr bool kRequireNoScratch = true;

    static_assert(kPhysicalVgprTarget <= kPhysicalVgprGuard,
                  "physical WDRA target must stay at or below 384 VGPRs");
    static_assert(kPhysicalVgprTarget <= kPhysicalVgprBudget,
                  "physical WDRA target must fit the 512 VGPR budget");
    static_assert(kPhysicalVgprTarget == 384,
                  "symmetric three-role WDRA target must be exactly 384 VGPRs");
    static_assert(kRequireNoSpill, "fused path must be spill-free");
    static_assert(kRequireNoPrivateSegment && kRequireNoScratch,
                  "fused path must not allocate private or scratch storage");
};

using ActiveFusedBwdContract = FusedBwdContract;

}  // namespace shaobo::fa3::bwd::fused_bwd
