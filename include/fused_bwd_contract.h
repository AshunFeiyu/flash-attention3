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
    static constexpr int kDqWriterWaves = 4;
    static constexpr int kProducerWaveBegin = 0;
    static constexpr int kConsumer0WaveBegin = 4;
    static constexpr int kConsumer1WaveBegin = 8;
    static constexpr int kDqWriterWaveBegin = 12;
    static constexpr int kWavesPerCta =
        kWavesPerProducer + kConsumerGroups * kWavesPerConsumerGroup +
        kDqWriterWaves;
    static constexpr int kThreadsPerCta = kWavesPerCta * kWaveSize;
    static constexpr int kMqPerPanel = 16;
    static constexpr int kMqPanels = kMq / kMqPerPanel;
    static constexpr int kNkPerConsumerWave =
        kNk / (kConsumerGroups * kWavesPerConsumerGroup);
    static constexpr int kNkPerConsumerGroup = kNk / kConsumerGroups;
    static constexpr int kHeadDimPerDqWriter =
        kHeadDim / kDqWriterWaves;

    static constexpr int kMmacM = 16;
    static constexpr int kMmacN = 16;
    static constexpr int kMmacK = 16;
    static constexpr int kHalfBytes = 2;
    static constexpr int kMmacPerLogicalGemm =
        (kMq / kMmacM) * (kNk / kMmacN) * (kHeadDim / kMmacK);
    static constexpr int kMmacPerConsumerWaveDkv =
        4 * (kMq / kMmacM) * (kNkPerConsumerWave / kMmacN) *
        (kHeadDim / kMmacK);
    static constexpr int kMmacPerDqWriterWave =
        (kMq / kMmacM) * (kNk / kMmacN) *
        (kHeadDimPerDqWriter / kMmacK);

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
    static constexpr bool kDqHasUniqueD32Owner = true;
    static constexpr bool kDqUsesFp32AtomicAdd = false;
    static constexpr bool kDqUsesWorkspaceReduction = true;
    static constexpr bool kDqWorkspaceIsFp32 = true;
    static constexpr bool kDqFinalOutputIsFp16 = true;

    static constexpr int kResidentKvBytes =
        2 * kNk * kHeadDim * kHalfBytes;
    static constexpr int kRawQDoBytes =
        2 * kMq * kHeadDim * kHalfBytes;
    static constexpr int kRawQDoPages = 2;
    static constexpr int kRawQDoPhysicalBytes =
        kRawQDoPages * kRawQDoBytes;
    static constexpr int kPdsLogicalBytes = kMq * kNk * kHalfBytes;
    // Raw/resident MLS panels retain their 2KB matrix-block spacing. A focused
    // writer/reader/MMAC probe proves native ds_write_matrix pages can be
    // packed at their 1KB touched footprint without changing either reader
    // fragment view.
    static constexpr int kWriterPageBytes = 64 * 16 * kHalfBytes;
    static constexpr int kWriterStrideBytes = 32 * 16 * kHalfBytes;
    static constexpr int kActiveWriterPages =
        kConsumerGroups * kWavesPerConsumerGroup;
    static constexpr int kPdsGenerationBytes =
        kActiveWriterPages * kWriterStrideBytes;
    static constexpr int kPdsGenerationCount = 1;
    static constexpr int kPdsPageBytes =
        kPdsGenerationCount * kPdsGenerationBytes;
    static constexpr int kBatchPdsBytes = kMqPanels * kPdsGenerationBytes;
    static constexpr int kSidecarBytes = 3 * kMq * sizeof(float);
    static constexpr int kSidecarPages = 2;
    static constexpr bool kSidecarAliasesPds = false;
    static constexpr int kLdsBudgetBytes = 128 * 1024;
    static constexpr int kStartupLdsBytes =
        kResidentKvBytes + kRawQDoPhysicalBytes;
    static constexpr int kSteadyKvReuseBytes =
        kBatchPdsBytes + kPdsPageBytes + kSidecarPages * kSidecarBytes;
    static constexpr int kPlannedLdsBytes = kStartupLdsBytes;

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
    static_assert(kWavesPerCta == 16 && kProducerWaveBegin == 0 &&
                      kConsumer0WaveBegin == 4 &&
                      kConsumer1WaveBegin == 8 &&
                      kDqWriterWaveBegin == 12,
                  "roles must be P0 0-3, dKV 4-11 and dQ writer 12-15");
    static_assert(kMqPanels == 4 && kNkPerConsumerWave == 16 &&
                      kNkPerConsumerGroup == 64 &&
                      kHeadDimPerDqWriter == 32,
                  "dKV waves own N16 and dQ writers own unique D32");
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
                              kMmacPerConsumerWaveDkv +
                          kDqWriterWaves * kMmacPerDqWriterWave ==
                      kMmacPerTile,
                  "owner work must equal exactly five logical GEMMs");
    static_assert(kMmacPerConsumerWaveDkv == 128 &&
                      kMmacPerDqWriterWave == 64,
                  "per-role MMAC ledger must match the 16-wave design");
    static_assert(kDkvHasUniqueOutputOwner && kDqHasUniqueD32Owner &&
                      !kDqUsesFp32AtomicAdd &&
                      kDqUsesWorkspaceReduction && kDqWorkspaceIsFp32 &&
                      kDqFinalOutputIsFp16,
                  "dKV stores once; dQ reduces fp32 partials into fp16 output");
    static_assert(kResidentKvBytes == 64 * 1024 &&
                      kRawQDoBytes == 32 * 1024 &&
                      kRawQDoPhysicalBytes == 64 * 1024 &&
                      kPdsLogicalBytes == 16 * 1024 &&
                      kWriterPageBytes == 2 * 1024 &&
                      kWriterStrideBytes == 1024 &&
                      kPdsGenerationBytes == 8 * 1024 &&
                      kPdsPageBytes == 8 * 1024,
                  "resident/raw/P-dS LDS regions must retain their fixed sizes");
    static_assert(kPdsGenerationCount == 1,
                  "local P/dS conversion uses one private-page generation");
    static_assert(kBatchPdsBytes == kResidentKvBytes / 2,
                  "four packed dS panels must reuse half the resident K/V LDS");
    static_assert(kSteadyKvReuseBytes <= kResidentKvBytes,
                  "P/dS and sidecar pages must fit released resident K/V LDS");
    static_assert(kPlannedLdsBytes <= kLdsBudgetBytes,
                  "fused FA3 BWD LDS plan must fit within 128KB");
    static_assert(!kSidecarAliasesPds && kSidecarBytes == 768,
                  "sidecar must remain independently readable per panel");
    static_assert(kPlannedLdsBytes == 131072,
                  "raw Q/dO double pages plus resident K/V must use 128KB");
};

struct FusedBwdBarrierLedger {
    static constexpr int kResidentFilled = 0;
    // One startup-only token: all non-producer waves latch resident K/V before
    // producer waves reuse the released region for raw packet sidecar data.
    // Startup scratch/sidecar reuse overlays only consumer0's V sub-region.
    // Its four waves are therefore the only arrivals needed for this release.
    static constexpr int kVSidecarReady = 1;
    static constexpr int kRawFilled0 = 2;
    static constexpr int kRawUsed0 = 3;
    static constexpr int kRawFilled1 = 4;
    static constexpr int kRawUsed1 = 5;
    static constexpr int kBatchDsFilled0 = 6;
    static constexpr int kBatchDsFilled1 = 7;
    // Runtime dS ownership is group-local. Each token has four dKV waves and
    // four dQ-writer waves, so one consumer group need not wait for its peer.
    static constexpr int kDqDone0 = 8;
    static constexpr int kDqDone1 = 9;
    static constexpr int kCount = 10;

    static_assert(kDqDone1 + 1 == kCount,
                  "barrier IDs must be contiguous and explicit");
};

struct FusedBwdWdraResourceWindow {
    static constexpr int kProducerVgprs = 16;
    static constexpr int kConsumer0Vgprs = 204;
    static constexpr int kConsumer1Vgprs = 204;
    static constexpr int kDqWriterVgprs = 88;
    static constexpr int kPhysicalVgprTarget =
        kProducerVgprs + kConsumer0Vgprs + kConsumer1Vgprs +
        kDqWriterVgprs;
    static constexpr int kPhysicalVgprGuard = 512;
    static constexpr int kPhysicalVgprBudget = 512;
    static constexpr bool kRequireNoSpill = true;
    static constexpr bool kRequireNoPrivateSegment = true;
    static constexpr bool kRequireNoScratch = true;

    static_assert(kPhysicalVgprTarget <= kPhysicalVgprGuard,
                  "physical WDRA target exceeds the admitted guard");
    static_assert(kPhysicalVgprTarget <= kPhysicalVgprBudget,
                  "physical WDRA target must fit the 512 VGPR budget");
    static_assert(kPhysicalVgprTarget == 512,
                  "double-page roles use the legal full-pool WDRA quantum");
    static_assert(kRequireNoSpill, "fused path must be spill-free");
    static_assert(kRequireNoPrivateSegment && kRequireNoScratch,
                  "fused path must not allocate private or scratch storage");
};

using ActiveFusedBwdContract = FusedBwdContract;

}  // namespace shaobo::fa3::bwd::fused_bwd
