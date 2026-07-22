#pragma once

#include <cstdint>

namespace shaobo::fa3::bwd::fused_bwd {

struct FusedBwdContract {
    static constexpr int kMq = 64;
    static constexpr int kNk = 128;
    static constexpr int kHeadDim = 128;
    static constexpr int kWaveSize = 64;
    static constexpr int kWavesPerProducer = 4;
    static constexpr int kWavesPerDkv = 4;
    static constexpr int kWavesPerDq = 4;
    static constexpr int kProducerWaveBegin = 0;
    static constexpr int kDkvWaveBegin = 4;
    static constexpr int kDqWaveBegin = 8;
    static constexpr int kWavesPerCta =
        kWavesPerProducer + kWavesPerDkv + kWavesPerDq;
    static constexpr int kThreadsPerCta = kWavesPerCta * kWaveSize;
    static constexpr int kMqPerPanel = 16;
    static constexpr int kMqPanels = kMq / kMqPerPanel;
    static constexpr int kNkPerDkvWave = kNk / kWavesPerDkv;
    static constexpr int kHeadDimPerDqWave = kHeadDim / kWavesPerDq;

    static constexpr int kMmacM = 16;
    static constexpr int kMmacN = 16;
    static constexpr int kMmacK = 16;
    static constexpr int kHalfBytes = 2;
    static constexpr int kMmacPerLogicalGemm =
        (kMq / kMmacM) * (kNk / kMmacN) * (kHeadDim / kMmacK);
    static constexpr int kMmacPerDkvWave =
        4 * (kMq / kMmacM) * (kNkPerDkvWave / kMmacN) *
        (kHeadDim / kMmacK);
    static constexpr int kMmacPerDqWave =
        (kMq / kMmacM) * (kNk / kMmacN) *
        (kHeadDimPerDqWave / kMmacK);

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
    static constexpr bool kDqOwnsD32Partial = true;
    static constexpr bool kDqUsesFp32AtomicAdd = true;

    static constexpr int kResidentKvBytes =
        2 * kNk * kHeadDim * kHalfBytes;
    static constexpr int kRawQDoBytes =
        2 * kMq * kHeadDim * kHalfBytes;
    static constexpr int kPdsLogicalBytes = kMq * kNk * kHalfBytes;
    // One t=1 m32x16 writer carries 1KB of values in a 2KB physical
    // footprint. Four N32 dKV owners publish one M16xN128 generation; two
    // generations ping-pong while dQ consumes the trans view.
    static constexpr int kWriterPageBytes = 64 * 16 * kHalfBytes;
    static constexpr int kActiveWriterPages = kWavesPerDkv;
    static constexpr int kPdsGenerationBytes =
        kActiveWriterPages * kWriterPageBytes;
    static constexpr int kPdsGenerationCount = 2;
    static constexpr int kPdsPageBytes =
        kPdsGenerationCount * kPdsGenerationBytes;
    static constexpr int kSidecarBytes = 3 * kMq * sizeof(float);
    static constexpr bool kSidecarUsesDedicatedLds = true;
    static constexpr int kLdsBudgetBytes = 128 * 1024;
    static constexpr int kPlannedLdsBytes =
        kResidentKvBytes + kRawQDoBytes + kPdsPageBytes +
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
                      kDkvWaveBegin == 4 && kDqWaveBegin == 8,
                  "wave roles must be producer 0-3, dKV 4-7, dQ 8-11");
    static_assert(kMqPanels == 4 && kNkPerDkvWave == 32 &&
                      kHeadDimPerDqWave == 32,
                  "dKV must own N32 and dQ must own D32");
    static_assert(kMq % kMmacM == 0 && kNk % kMmacN == 0 &&
                      kHeadDim % kMmacK == 0,
                  "tile dimensions must be MMAC aligned");
    static_assert(kLogicalGemmCount == 5,
                  "fused backward must contain exactly five logical GEMMs");
    static_assert(kGemmCount == kLogicalGemmCount,
                  "formula and enum GEMM counts must agree");
    static_assert(kMmacPerLogicalGemm == 256 && kMmacPerTile == 1280,
                  "five-GEMM tile work must be 256 MMAC per GEMM and 1280 total");
    static_assert(kWavesPerDkv * kMmacPerDkvWave +
                          kWavesPerDq * kMmacPerDqWave ==
                      kMmacPerTile,
                  "owner work must equal exactly five logical GEMMs");
    static_assert(kDkvHasUniqueOutputOwner && kDqOwnsD32Partial &&
                      kDqUsesFp32AtomicAdd,
                  "dKV must store once and dQ must emit one D32 partial");
    static_assert(kResidentKvBytes == 64 * 1024 &&
                      kRawQDoBytes == 32 * 1024 &&
                      kPdsLogicalBytes == 16 * 1024 &&
                      kPdsGenerationBytes == 8 * 1024 &&
                      kPdsPageBytes == 16 * 1024,
                  "resident/raw/P-dS LDS regions must retain their fixed sizes");
    static_assert(kPdsGenerationCount == 2,
                  "dS must use two native ping-pong generations");
    static_assert(kPlannedLdsBytes <= kLdsBudgetBytes,
                  "fused FA3 BWD LDS plan must fit within 128KB");
    static_assert(kPlannedLdsBytes == 112 * 1024 + 3 * 64 * 4,
                  "canonical physical LDS plan is 112.75KB");
};

struct FusedBwdBarrierLedger {
    static constexpr int kResidentFilled = 0;
    static constexpr int kRawFilled = 1;
    static constexpr int kRawUsed = 2;
    static constexpr int kDsFilled0 = 3;
    static constexpr int kDsUsed0 = 4;
    static constexpr int kDsFilled1 = 5;
    static constexpr int kDsUsed1 = 6;
    static constexpr int kCount = 7;

    static_assert(kDsUsed1 + 1 == kCount,
                  "barrier IDs must be contiguous and explicit");
};

struct FusedBwdWdraResourceWindow {
    static constexpr int kProducerVgprs = 24;
    static constexpr int kDkvVgprs = 240;
    static constexpr int kDqVgprs = 96;
    static constexpr int kPhysicalVgprTarget =
        kProducerVgprs + kDkvVgprs + kDqVgprs;
    static constexpr int kPhysicalVgprGuard = 384;
    static constexpr int kPhysicalVgprBudget = 512;
    static constexpr bool kRequireNoSpill = true;
    static constexpr bool kRequireNoPrivateSegment = true;
    static constexpr bool kRequireNoScratch = true;

    static_assert(kPhysicalVgprTarget <= kPhysicalVgprGuard,
                  "physical WDRA target must stay at or below 384 VGPRs");
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
