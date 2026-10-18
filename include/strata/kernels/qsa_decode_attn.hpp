// include/strata/kernels/qsa_decode_attn.hpp - plan v0.3 P3/P7: split-K decode attention over the selected cells.
//
// The first attention kernel runs one block per query head (24 blocks on a 48-SM part), reads each key row with one
// thread (uncoalesced) and walks every selected cell serially for the values; at a 4K context (2,051 selected
// cells) that costs ~0.3-0.5 ms per QSA layer.  This one reads the KV POOLS directly through the page table and the
// selection ids (no gather copy), in chunks of CHUNK cells per block, and serves all `n_head / n_head_kv` query
// heads that share a KV head from one read of the chunk:
//
//     grid (ceil(cap / CHUNK), n_head_kv):  s = q.k * scale for 12 heads x CHUNK cells, chunk max m and sum l,
//                                           acc = p . V  ->  partials
//     grid (n_head):                        merge the chunks with the usual log-sum-exp rescale
//
// FP16 pools or INT8 pools (codes + fp16 scale per 64 values, `kv_q8.hpp`).  The grid is sized by the capacity;
// the real count comes from `step[kStepWidth]` as for every other capturable QSA kernel.
#pragma once

#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

struct QsaAttnPools {
    const uint16_t* k_pool = nullptr;   ///< fp16 [page][kv_head][page_size][head_dim], or null when int8
    const uint16_t* v_pool = nullptr;
    const int8_t* k_q = nullptr;        ///< int8 codes, same layout
    const int8_t* v_q = nullptr;
    const uint16_t* k_scale = nullptr;  ///< fp16 [page][kv_head][page_size][head_dim / 64]
    const uint16_t* v_scale = nullptr;
    const uint8_t* k_q4 = nullptr;      ///< q4_0 block_q4_0 [page][kv_head][page_size][head_dim / 32 * 18]
    const uint8_t* v_q4 = nullptr;
    Q4Mode q4_mode = Q4Mode::kUniform;  ///< `--kv q4lm32`/`q4lm32s`: k_q4/v_q4 hold Max-Lloyd levels
    const int32_t* page_table = nullptr;
    /// WP-2a window (`--kv q4lm32w`): exact fp16 side pools, flat [(cell, kv_head, head_dim)] - indexed
    /// directly by cell, no paging. A reader serves cell `c` from here when `win_on && win_cell(c, n)` with
    /// `n` = step[kStepNKv], else from the quantized pools. Null (and win_s/r <= 0) when the window is off.
    const uint16_t* k_win = nullptr;
    const uint16_t* v_win = nullptr;
    int win_s = 0;    ///< recent exact cells (STRATA_KV_WINDOW_S, default 128)
    int win_r = 0;    ///< exact sink cells at the start (STRATA_KV_WINDOW_R, default 8)
    // NOTE: no horizon field - readers take it from step[kStepNKv] (pos + 1, device-fresh and graph-safe;
    // qsa_step_fill derives all four step entries from one position). A host horizon would bake stale on
    // graph reuse.
};

#if defined(__CUDACC__) || defined(__HIPCC__)
/// True when the pools carry a live window: both side pools present and at least one exact cell.
inline __host__ __device__ bool qsa_win_on(const QsaAttnPools& p) {
    return p.k_win != nullptr && p.v_win != nullptr && (p.win_s > 0 || p.win_r > 0);
}

/// Cell `c` reads exact (sink, or recent under horizon `n` = step[kStepNKv]) under a live window.
inline __host__ __device__ bool qsa_win_cell(const QsaAttnPools& p, long long c, long long n) {
    return c < (long long) p.win_r || c >= n - (long long) p.win_s;
}
#else
/// True when the pools carry a live window: both side pools present and at least one exact cell.
inline bool qsa_win_on(const QsaAttnPools& p) {
    return p.k_win != nullptr && p.v_win != nullptr && (p.win_s > 0 || p.win_r > 0);
}

/// Cell `c` reads exact (sink, or recent under horizon `n` = step[kStepNKv]) under a live window.
inline bool qsa_win_cell(const QsaAttnPools& p, long long c, long long n) {
    return c < (long long) p.win_r || c >= n - (long long) p.win_s;
}
#endif

/// Scratch floats for `cap` selected cells: partial accumulators, maxima and sums.
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s);

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream);

/// Plan v0.3 P5: `n_q` queries at once, each with its own selection: q [n_q, n_head, 256], ids [n_q, cap], steps
/// [n_q, kStepCount], attn [n_q, n_head, 256]; scratch is `n_q` times the single-query size.
void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);

}  // namespace strata::kernels
