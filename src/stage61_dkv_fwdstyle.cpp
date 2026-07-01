#include "stage61_dkv_contract.h"

namespace shaobo::fa3::bwd::stage61 {

// Implementation status:
// This file is the clean target for the next port.  Do not copy the historical
// Cxx phase stack here.  Port one cohesive block at a time:
//   1. producer A/B packet publishers
//   2. consumer score+dP MMAC island
//   3. softmax+dS VALU island
//   4. dV+dK MMAC island
//   5. store epilogue
//
// Keep each block small enough to inspect in asm and Source CSV.

static_assert(DkvTileD128Mq32Nk128::kPlannedLdsBytes == 115456,
              "Unexpected clean-lane LDS budget");
static_assert(DkvTileD128Mq32Nk128::kTotalMmacPerConsumer == 64,
              "Unexpected clean-lane MMAC count");

}  // namespace shaobo::fa3::bwd::stage61

