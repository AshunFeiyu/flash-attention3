#pragma once

#include <cstdint>

namespace shaobo::fa3::bwd::dq {

inline constexpr int kDqPathReferenceCorrectness = 1;
inline constexpr int kDqPathCanonicalDq = 6;
inline constexpr int kDqPathNativeDsRingPrototype = 7;

template <int BlockMq, int BlockNk>
struct DqTileD128MqNk {
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockMq = BlockMq;
    static constexpr int kRowsPerConsumerGroup = BlockMq / 2;
    static constexpr int kRowsPerConsumerWave = 16;
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
    static constexpr int kStartupLdsBytes = kQDoBytes + kSidecarBytes;
    static constexpr int kSteadyLdsBytes = 2 * kKvBytes;
    static constexpr int kPlannedLdsBytes =
        kStartupLdsBytes > kSteadyLdsBytes ? kStartupLdsBytes
                                           : kSteadyLdsBytes;

    static_assert(kBlockMq == 128,
                  "dQ 16-wave path maps two consumer groups over M128");
    static_assert(kRowsPerConsumerGroup == 64,
                  "dQ consumer group owns four M16 rows");
    static_assert(kBlockNk % 32 == 0,
                  "dQ K/V stream tile must align to 32-row MLS tiles");
    static_assert(kHeadDim == 128, "first clean dQ path is D128-only");
    static_assert(kPlannedLdsBytes <= kLdsBudgetBytes,
                  "dQ target LDS plan must fit 128KB");
};

using ActiveDqTile = DqTileD128MqNk<128, 128>;

template <int BlockMq, int BlockNk>
struct DqNativeDsRingTileD128 {
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockMq = BlockMq;
    static constexpr int kBlockNk = BlockNk;
    static constexpr int kWaveSize = 64;
    static constexpr int kWavesPerCta = 12;
    static constexpr int kThreadsPerCta = kWaveSize * kWavesPerCta;
    static constexpr int kHalfBytes = 2;
    static constexpr int kDqSlotNk = 32;
    static constexpr int kDsSlots = 2;

    static constexpr int kMmacM = 16;
    static constexpr int kMmacN = 16;
    static constexpr int kMmacK = 16;
    static constexpr int kScoreMmacPerKTile =
        (kBlockMq / kMmacM) * (kBlockNk / kMmacN) *
        (kHeadDim / kMmacK);
    static constexpr int kDpMmacPerKTile = kScoreMmacPerKTile;
    static constexpr int kDqMmacPerKTile = kScoreMmacPerKTile;

    static constexpr int kQDoBytes =
        2 * kBlockMq * kHeadDim * kHalfBytes;
    static constexpr int kKvBytes =
        2 * kBlockNk * kHeadDim * kHalfBytes;
    static constexpr int kDsSlotBytes = kBlockMq * kDqSlotNk * kHalfBytes;
    static constexpr int kDsRingBytes = kDsSlots * kDsSlotBytes;
    static constexpr int kSidecarBytes =
        3 * kBlockMq * static_cast<int>(sizeof(float));
    static constexpr int kStartupLdsBytes = kQDoBytes + kSidecarBytes;
    static constexpr int kSteadyLdsBytes = kKvBytes + kDsRingBytes;
    static constexpr int kPlannedLdsBytes =
        kStartupLdsBytes > kSteadyLdsBytes ? kStartupLdsBytes
                                           : kSteadyLdsBytes;
    static constexpr int kLdsBudgetBytes = 128 * 1024;

    static_assert(kBlockMq == 64,
                  "first native dS ring prototype uses Mq64");
    static_assert(kBlockNk == 128,
                  "first native dS ring prototype uses Nk128");
    static_assert(kBlockNk % kDqSlotNk == 0,
                  "dS ring slots must tile the K dimension");
    static_assert(kPlannedLdsBytes <= kLdsBudgetBytes,
                  "native dS ring LDS plan must fit 128KB");
};

using NativeDsRingDqTile = DqNativeDsRingTileD128<64, 128>;

struct DqBarrierLedger {
    static constexpr int kPage0Filled = 0;
    static constexpr int kPage0Used = 1;
    static constexpr int kPage1Filled = 2;
    static constexpr int kPage1Used = 3;
    static constexpr int kQDoFilled = 4;
    static constexpr int kQDoLatched = 5;
    static constexpr int kAllDone = 6;
};

struct DqNativeDsRingBarrierLedger {
    static constexpr int kQDoFilled = 0;
    static constexpr int kQDoLatched = 1;
    static constexpr int kKvFilled = 2;
    static constexpr int kKvUsed = 3;
    static constexpr int kDsSlot0Filled = 4;
    static constexpr int kDsSlot0Used = 5;
    static constexpr int kDsSlot1Filled = 6;
    static constexpr int kDsSlot1Used = 7;
};

struct NativeDsSlotMap {
    static constexpr int kInvalid = -1;
    static constexpr int kWordsPerFrag = 8;
    static constexpr int kWaveSize = 64;

    static constexpr int slot_krow(int group, int word) {
        return group * 4 + (word & 3);
    }

    static constexpr int dst_raw_source_lane(int group, int q, int word) {
        return 4 + 2 * group + 16 * (q & 3) + ((word >> 1) & 1) +
               8 * (word >> 2);
    }

    static constexpr int dst_source_lane(int group, int q, int word) {
        return dst_source_word(q, word, dst_raw_source_lane(group, q, word)) <
                       kWordsPerFrag
                   ? (dst_raw_source_lane(group, q, word) & (kWaveSize - 1))
                   : kInvalid;
    }

    static constexpr int dst_source_word(int q, int word) {
        return dst_source_word(q, word, 0);
    }

    static constexpr int dst_source_word(int q, int word, int raw_lane) {
        return 2 * (q >> 2) + (word & 1) + 2 * (raw_lane >> 6);
    }

    static constexpr bool is_mapped(int group, int q, int word) {
        return dst_source_lane(group, q, word) != kInvalid;
    }
};

static_assert(NativeDsSlotMap::dst_source_lane(0, 0, 0) == 4,
              "slotmap formula must match focused probe");
static_assert(NativeDsSlotMap::dst_source_lane(0, 1, 4) == 28,
              "slotmap formula must match focused probe");
static_assert(NativeDsSlotMap::dst_source_lane(2, 3, 4) == 0,
              "slotmap formula must include source-lane wrap");
static_assert(NativeDsSlotMap::dst_source_word(3, 4, 64) == 2,
              "slotmap formula must carry lane wrap into source word");
static_assert(NativeDsSlotMap::dst_source_word(0, 7) == 1,
              "slotmap formula must match focused probe");
static_assert(NativeDsSlotMap::dst_source_word(15, 7) == 7,
              "slotmap formula must match focused probe");
static_assert(NativeDsSlotMap::dst_source_lane(3, 15, 7) ==
                  NativeDsSlotMap::kInvalid,
              "slotmap formula must preserve known boundary holes");

struct OptimizationTargets {
    static constexpr int kTargetMmacActiveSharePercent = 40;
    static constexpr bool kRequireNoScratch = true;
    static constexpr bool kRequireNoSpill = true;
    static constexpr bool kRequireNoLdsBankConflict = true;
    static constexpr bool kForbidDuplicateScoreDpInsideDq = true;
    static constexpr bool kForbidDqAtomicAdd = true;
};

struct WdraResourceWindows {
    static constexpr int kProducerVgprs = 16;
    static constexpr int kConsumerTargetVgprs = 216;
    static constexpr int kConsumerCeilingVgprs = 248;
};

}  // namespace shaobo::fa3::bwd::dq
