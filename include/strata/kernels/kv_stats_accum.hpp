// include/strata/kernels/kv_stats_accum.hpp - the STRATA_KVSTATS accumulator layout (both TUs + Python).
//
// One struct per (QSA layer, head, tensor K/V, tap pre/post): doubles for exact moments, per-32-block sums,
// a 32-bin absolute log2 histogram (bin b holds 2^(b-8) <= |x| < 2^(b-7) for b = 1..30, bin 0 the tiny
// (|x| < 2^-7, exact zeros included), bin 31 the overflow (|x| >= 2^23)), and a running per-channel absmax.
// Group index g = ((layer*8+head)*2+is_v)*2+tap. Python mirrors this layout byte for byte.
#pragma once

#include <cstdint>

namespace strata::kernels {

struct KvStatsAccum {
    double cnt, sum, sumsq, sum4, mx;
    double bsum[8], bsumsq[8], bmx[8];
    uint64_t hist[32];
    float chmax[256];
};

constexpr int KVSTATS_MAX_QSA = 64;
constexpr int KVSTATS_MAX_HEAD = 8;
constexpr uint32_t KVSTATS_NGROUPS = 64u * 8u * 2u * 2u;   // 2048 groups x 1512 B = ~3.1 MB device

}  // namespace strata::kernels
