// include/strata/kernels/kv_stats.hpp - STRATA_KVSTATS per-token K/V distribution tap (diagnostic, v2).
//
// Env-gated, default off. An accumulator kernel (a graph node, so it replays correctly under CUDA-graph
// capture) folds every appended K/V row into per-(QSA layer, head, tensor, tap) accumulators (kv_stats_accum.hpp:
// count, moments, per-32-block sums, 32-bin log2 histogram, running per-channel absmax). The driver
// (generate.cpp's serial verify loop, next to the LOGPOS block) copies the accumulators out per window and
// resets them: one file section per window {pos0, n, present groups}. The harness aggregates sections into
// kurtosis/crest/block-max/channel statistics in Python (tools/quant_quality.py kvstats_aggregate).
//
// Call twice per append: tap=false BEFORE the FWHT (post-norm-rope K / raw V), tap=true AFTER it (the
// quantizer input). Defined in src/kernels/cuda/kv_q4.cu (CUDA/HIP); SYCL and Metal carry silent stubs -
// the diagnostic never fails an engine and never fires without the env var. Harness runs only: one engine,
// one GEN, then QUIT - a reused server process would mix requests into the same accumulators. Deliberately
// not hooked: the MTP drafter state and the prefill drafter cache (other states than the scored one), the
// k8v4 hybrid branch (mixed rotated/unrotated semantics), and the pipelined prompt path (kvstats forces the
// serial verify loop via STRATA_LOGPOS=os.devnull, the same path every measurement uses).
#pragma once

#include <cstdint>

#include "strata/kernels/kv_stats_accum.hpp"

namespace strata::kernels {

void kv_stats_launch(const float* kcur, const float* vcur, int n_head_kv, int head_dim, int64_t qsa, int n_tok,
                     bool rotated, void* stream);
void kv_stats_consume_window(int64_t pos0, int64_t n_tok);
void kv_stats_alloc();

}  // namespace strata::kernels
