// src/kernels/cuda/kv_q4.cu - see include/strata/kernels/kv_q4.hpp. Q4_0 KV with Walsh-Hadamard rotation
// (from PR #21 by code-martin; KV-streaming integration and the deterministic group maximum added on merge).
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stats_accum.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q4: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Fast Walsh-Hadamard Transform for N = 256: one warp per row, 8 values per lane in registers.
// Orthonormal (scale 1/sqrt(256) = 1/16), so it is its own inverse.
__global__ void fwht256_kernel(const float* __restrict__ src, float* __restrict__ dst, int64_t n_rows, float scale) {
    constexpr int warp_size = 32;
    constexpr int N = 256;
    constexpr int el_w = N / warp_size;   // 8

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) return;

    const float* row_src = src + r * N;
    float* row_dst = dst + r * N;

    float reg[el_w];
    const int lane = threadIdx.x;

#pragma unroll
    for (int i = 0; i < el_w; ++i) reg[i] = row_src[i * warp_size + lane] * scale;

    // the low 5 index bits live across lanes
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val = reg[j];
            const float val2 = __shfl_xor_sync(0xffffffffu, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
    // the high 3 bits across each lane's registers
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < el_w; ++i) row_dst[i * warp_size + lane] = reg[i];
}

// One 32-value group in one warp (lane = element), ggml's q4_0: d = (the value of largest |x|) / -8,
// q = clamp(trunc(x / d + 8.5), 0, 15). Returns the scale bits; `byte` is lane t's packed byte for t < 16
// (element t in the low nibble, t + 16 in the high one). Ties in |x| resolve to the larger value in EVERY lane,
// so all lanes agree on d (a plain `a > amax` could leave lanes with opposite signs and one block two scales).
__device__ __forceinline__ uint16_t q4_group(float x, int lane, uint8_t& byte) {
    float amax = fabsf(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float a = __shfl_xor_sync(0xffffffffu, amax, o);
        const float v = __shfl_xor_sync(0xffffffffu, mval, o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = __float2int_rz(x * id + 8.5f);
    const uint8_t qc = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
    const uint8_t qhi = __shfl_down_sync(0xffffffffu, qc, 16);
    byte = (uint8_t) (qc | (qhi << 4));
    (void) lane;
    return f16_from_f32(d);
}

// The nearest of the 16 levels for a value already divided by the scale. The 15 boundaries are the midpoints of
// consecutive levels; written as literals so the ladder unrolls into predicated increments (a table read would be
// 15 uniform-index loads for no gain).
__device__ __forceinline__ int q4l_level(float y) {
    int k = 0;
    if (y >= -76.5f) k = 1;
    if (y >= -59.0f) k = 2;
    if (y >= -46.0f) k = 3;
    if (y >= -35.0f) k = 4;
    if (y >= -25.5f) k = 5;
    if (y >= -16.5f) k = 6;
    if (y >= -8.0f) k = 7;
    if (y >= 0.0f) k = 8;
    if (y >= 8.0f) k = 9;
    if (y >= 16.5f) k = 10;
    if (y >= 25.5f) k = 11;
    if (y >= 35.0f) k = 12;
    if (y >= 46.0f) k = 13;
    if (y >= 59.0f) k = 14;
    if (y >= 76.5f) k = 15;
    return k;
}

// The 8-level ladder (kv_q4.hpp's kQ4l8Levels): the midpoints of -69, -43, -24, -8, 8, 24, 43, 69.
__device__ __forceinline__ int q4l_level8(float y) {
    int k = 0;
    if (y >= -56.0f) k = 1;
    if (y >= -33.5f) k = 2;
    if (y >= -16.0f) k = 3;
    if (y >= 0.0f) k = 4;
    if (y >= 16.0f) k = 5;
    if (y >= 33.5f) k = 6;
    if (y >= 56.0f) k = 7;
    return k;
}

// The same group for `--kv q4lm32` / `--kv q4lm32s` (kv_q4.hpp's level table): the nibble is the nearest level and
// the scale is max|x| / 87, or - with FIT - the block's own least-squares scale. max|x| is a plain shuffle maximum
// (unlike q4_0's d there is no sign or tie to agree on, and a maximum is the same in every lane whatever the
// pairing), and the two fit sums are butterflies whose pairing is identical in every lane, so with fp addition
// commuting every lane still reaches the same value. `byte` packs as in q4_group.
// N is 16 (the q4lm32 table) or 8 (the WP-2 probe table); FIT is only ever used with N = 16 today.
template <bool FIT, int N>
__device__ __forceinline__ uint16_t q4l_group(float x, uint8_t& byte) {
    constexpr int8_t lmax = (N == 8) ? kQ4l8Max : kQ4lMax;
    const int8_t* lmap = nullptr;
    if constexpr (N == 8) lmap = kQ4l8Map; else lmap = kQ4lMap;
    float amax = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float d = amax * (1.0f / (float) lmax);
    const float y = d != 0.0f ? x / d : 0.0f;
    int k = (N == 8) ? q4l_level8(y) : q4l_level(y);
    if constexpr (FIT) {
        // One least-squares pass over this block: with the level assignment held, d = sum(l * x) / sum(l * l)
        // minimizes the block's error against these very levels, and the assignment is then redone with that d. One
        // pass is most of what the levels can be given (kv_q4.hpp). An all-zero block keeps d = 0 and k = 8, whose
        // level is 0, so the divide guards are never taken.
        const float l = (float) lmap[(uint8_t) k];
        float num = l * x, den = l * l;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            num += __shfl_xor_sync(0xffffffffu, num, o);
            den += __shfl_xor_sync(0xffffffffu, den, o);
        }
        if (den > 0.0f) {
            d = num / den;
            k = (N == 8) ? q4l_level8(d != 0.0f ? x / d : 0.0f) : q4l_level(d != 0.0f ? x / d : 0.0f);
        }
    }
    const uint8_t kc = (uint8_t) k;
    const uint8_t khi = __shfl_down_sync(0xffffffffu, kc, 16);
    byte = (uint8_t) (kc | (khi << 4));
    return f16_from_f32(d);
}

// The WP-1 treatment arm's V scale (kv_q4.hpp, kLloydVrms): d = rms / 32 over the block's own 32 values, with the
// same 16-level table and the same nearest-level assignment as q4l_group<false, 16> - only the scale rule differs.
// The sum of squares runs through the same xor-butterfly as the max above (and the FIT sums), so every lane
// reaches the same value whatever the pairing and the host reference can replay it line for line.
__device__ __forceinline__ uint16_t q4l_group_vrms(float x, uint8_t& byte) {
    float ss = x * x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    float d = sqrtf(ss * (1.0f / 32.0f)) * (1.0f / 32.0f);
    const float y = d != 0.0f ? x / d : 0.0f;
    int k = q4l_level(y);
    const uint8_t kc = (uint8_t) k;
    const uint8_t khi = __shfl_down_sync(0xffffffffu, kc, 16);
    byte = (uint8_t) (kc | (khi << 4));
    return f16_from_f32(d);
}

// The WP-2a clipping arm's scale (kv_q4.hpp, kLloydClip): d = quantile(|x|, kappa) / 87 over the block's own
// 32 magnitudes, same 16-level table and nearest-level assignment as q4l_group<false, 16> - only the scale rule
// differs, so outliers saturate instead of stretching the grid. The 32 magnitudes sort through a bitonic network
// (one value per lane, uniform shuffles, no divergence), then every lane reads the quantile's own lane - so all
// lanes hold the identical scale with no reduction-order question at all, and the host replays it line for line.
// kappa = 1 sorts index 31 to the maximum, which is exactly q4l_group<false, 16> (the arm's identity check).
__device__ __forceinline__ float q4l_quantile(float ax, float kappa) {
    float a = ax;
    for (int k = 2; k <= 32; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int lane = threadIdx.x;
            const float o = __shfl_xor_sync(0xffffffffu, a, j);
            const bool up = (((lane & k) == 0) != ((lane & j) != 0));
            a = up ? fminf(a, o) : fmaxf(a, o);
        }
    }
    int idx = (int) (kappa * 32.0f);
    if (idx < 0) idx = 0;
    if (idx > 31) idx = 31;
    return __shfl_sync(0xffffffffu, a, idx);
}

namespace {
// Set by qsa_set_kv_clip (host): the WP-2a arm's per-tensor quantiles. __constant__ so the writer reads them
// free and every grid point is an env var, not a new mode.
__constant__ float kQ4lClipKappaK = 0.94f;
__constant__ float kQ4lClipKappaV = 0.94f;
}  // namespace

__device__ __forceinline__ uint16_t q4l_group_clip(float x, uint8_t& byte, bool is_v) {
    const float kappa = is_v ? kQ4lClipKappaV : kQ4lClipKappaK;
    float d = q4l_quantile(fabsf(x), kappa) * (1.0f / (float) kQ4lMax);
    const float y = d != 0.0f ? x / d : 0.0f;
    int k = q4l_level(y);
    const uint8_t kc = (uint8_t) k;
    const uint8_t khi = __shfl_down_sync(0xffffffffu, kc, 16);
    byte = (uint8_t) (kc | (khi << 4));
    return f16_from_f32(d);
}
// `is_v` is the tensor of the pair this block belongs to; the kLloyd8* probe modes, the WP-1 treatment mode
// (kLloydVrms) and the WP-2a clipping mode (kLloydClip) look at it, the rest quantize K and V alike.
template <Q4Mode M>
__device__ __forceinline__ uint16_t q4_group_any(float x, int lane, uint8_t& byte, bool is_v = false) {
    if constexpr (M == Q4Mode::kUniform) {
        return q4_group(x, lane, byte);
    } else if constexpr (M == Q4Mode::kLloydClip) {
        return q4l_group_clip(x, byte, is_v);
    } else if constexpr (M == Q4Mode::kLloydVrms) {
        if (is_v) return q4l_group_vrms(x, byte);
        return q4l_group<false, 16>(x, byte);
    } else if constexpr (M == Q4Mode::kLloyd8) {
        return q4l_group<false, 8>(x, byte);
    } else if constexpr (M == Q4Mode::kLloyd8K) {
        if (is_v) return q4l_group<false, 16>(x, byte);
        return q4l_group<false, 8>(x, byte);
    } else if constexpr (M == Q4Mode::kLloyd8V) {
        if (is_v) return q4l_group<false, 8>(x, byte);
        return q4l_group<false, 16>(x, byte);
    } else {
        constexpr bool fit = (M == Q4Mode::kLloydLs);
        return q4l_group<fit, 16>(x, byte);
    }
}

__device__ __forceinline__ void q4_store(uint8_t* pool, long long row, int b, int lane, uint16_t d, uint8_t byte) {
    block_q4_0* blk = reinterpret_cast<block_q4_0*>(pool + row * (long long) sizeof(block_q4_0) * 8) + b;
    if (lane == 0) blk->d = d;
    if (lane < 16) blk->qs[lane] = byte;
}

// One block = one 32-value group of one KV head of K (plane 0) or V (plane 1) for token step_idx; 32 threads.
template <Q4Mode MODE>
__global__ void kv_append_q4_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                    int step_stride, int planes,
                                    const float* __restrict__ kcur, const float* __restrict__ vcur,
                                    int kv_heads, int head_dim, int page_size, KvHostPools host) {
    const int step_idx = (int) (blockIdx.z / (unsigned) planes);
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u;
    const long long pos = (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos);
    const int h = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const long long step_off = (long long) step_idx * kv_heads * head_dim;
    const float x = (is_v ? vcur : kcur)[step_off + h * head_dim + b * QK4_0 + t];
    uint8_t byte;
    const uint16_t d = q4_group_any<MODE>(x, t, byte, is_v);
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, t, d, byte);
    if (host.k_q4 != nullptr)
        q4_store(is_v ? host.v_q4 : host.k_q4, ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), b, t,
                 d, byte);
}

// The prompt path: grid (T, kv_heads, groups), K then V; also into the staging pool (identity layout) when given.
template <Q4Mode MODE>
__global__ void kv_append_q4_batch_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                          const int32_t* __restrict__ table, int64_t pos0,
                                          const float* __restrict__ K, const float* __restrict__ V,
                                          int kv_heads, int head_dim, int page_size, int is_v_grid, KvHostPools host,
                                          KvHostPools stage) {
    const long long t = blockIdx.x;
    const long long pos = pos0 + t;
    const int h = blockIdx.y, b = blockIdx.z, th = threadIdx.x;
    const bool is_v = is_v_grid != 0;
    const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + b * QK4_0 + th];
    uint8_t byte;
    const uint16_t d = q4_group_any<MODE>(x, th, byte, is_v);
    const long long page = (long long) table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, th, d, byte);
    if (host.k_q4 != nullptr) q4_store(is_v ? host.v_q4 : host.k_q4, row_id, b, th, d, byte);
    if (stage.k_q4 != nullptr) q4_store(is_v ? stage.v_q4 : stage.k_q4, row_id, b, th, d, byte);
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths). NK / NV are the level count of
// the K side and the V side (0 = q4_0's uniform grid), because the probe can put 8 levels on one and 16 on the other.
template <int NK, int NV>
__global__ void kv_gather_q4_kernel(const uint8_t* __restrict__ k_q4, const uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                    uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch,
                                    const uint16_t* __restrict__ k_win, const uint16_t* __restrict__ v_win,
                                    int win_s, int win_r) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int blocks_per_head = head_dim / QK4_0;                        // 8
    const int bytes_per_head = blocks_per_head * sizeof(block_q4_0);    // 144
    const long long total_blocks = n_ids * kv_heads * blocks_per_head;

    const long long blk_idx = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_idx >= total_blocks) return;

    const int t = threadIdx.x;
    const long long id = blk_idx / (kv_heads * blocks_per_head);
    const int rem = (int) (blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);

    const long long dst_offset = ((id * kv_heads + h) * head_dim) + (b * QK4_0 + t);
    // WP-2a window: exact fp16 from the always-resident side pool (flat row, no paging) - also when the
    // quantized block is not resident. The horizon is step[kStepNKv] (device-fresh, graph-safe). Null pools
    // (every other mode) skip this uniformly.
    if (k_win != nullptr && v_win != nullptr && (win_s > 0 || win_r > 0)) {
        const long long nkv = (long long) __ldg(step + kStepNKv);
        if (cell < win_r || cell >= nkv - win_s) {
            const long long wrow = ((long long) cell * kv_heads + h) * head_dim + (b * QK4_0 + t);
            k_scratch[dst_offset] = k_win[wrow];
            v_scratch[dst_offset] = v_win[wrow];
            return;
        }
    }

    const block_q4_0* k_blk = reinterpret_cast<const block_q4_0*>(k_q4 + row * bytes_per_head) + b;
    const block_q4_0* v_blk = reinterpret_cast<const block_q4_0*>(v_q4 + row * bytes_per_head) + b;
    const float kd = f32_from_f16(k_blk->d);
    const float vd = f32_from_f16(v_blk->d);
    const int j = t < 16 ? t : (t - 16);
    const uint8_t k_byte = k_blk->qs[j];
    const uint8_t v_byte = v_blk->qs[j];
    const int kn = (t < 16) ? (k_byte & 0x0F) : (k_byte >> 4);
    const int vn = (t < 16) ? (v_byte & 0x0F) : (v_byte >> 4);
    const int kq = (NK == 0) ? (kn - 8) : (int) ((NK == 8) ? kQ4l8Map[kn] : kQ4lMap[kn]);
    const int vq = (NV == 0) ? (vn - 8) : (int) ((NV == 8) ? kQ4l8Map[vn] : kQ4lMap[vn]);
    k_scratch[dst_offset] = f16_from_f32((float) kq * kd);
    v_scratch[dst_offset] = f16_from_f32((float) vq * vd);
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    const int rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    fwht256_kernel<<<dim3((unsigned) num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        src, dst, n_rows, 1.0f / 16.0f);
    check("fwht256 launch");
}

// The STRATA_KVSTATS accumulator (kv_stats.hpp): one block per (head, tensor), so every global update has a
// single writer and no atomics exist anywhere - bit-identical across runs. A graph node, so it replays
// correctly under CUDA-graph capture (unlike host math). Lanes own fixed channels (i = lane + 32j), the token
// loop is sequential, cross-lane reduction runs through shared memory in fixed order.
__global__ void kv_stats_accum_kernel(const float* __restrict__ kcur, const float* __restrict__ vcur,
                                      KvStatsAccum* __restrict__ acc, int n_tok, int head_dim, int64_t qsa,
                                      int tap, int nh) {
    const int lane = threadIdx.x;
    const int tensor = blockIdx.x / nh;   // 0 K, 1 V
    const int h = blockIdx.x % nh;
    __shared__ double shp[32][28];         // per-lane partials: sum,sq,s4,mx + bs/bsq/bx x8
    __shared__ uint64_t shh[32][32];       // per-lane histogram slices (one writer per slice)
    for (int b = 0; b < 32; ++b) shh[lane][b] = 0;   // the lane's whole slice (not one element!)
    const int64_t per = (int64_t) nh * head_dim;
    const float* base = (tensor == 0 ? kcur : vcur);
    double sum = 0.0, sq = 0.0, s4 = 0.0, mx = 0.0;
    double bs[8] = {0, 0, 0, 0, 0, 0, 0, 0}, bsq[8] = {0, 0, 0, 0, 0, 0, 0, 0},
           bx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float chm[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int t = 0; t < n_tok; ++t) {
        const float* row = base + (int64_t) t * per + (int64_t) h * head_dim;
        for (int i = lane; i < head_dim; i += 32) {
            const double x = (double) row[i];
            sum += x;
            sq += x * x;
            s4 += x * x * x * x;
            const double ax = x < 0.0 ? -x : x;
            if (ax > mx) mx = ax;
            const int b = i >> 5;
            bs[b] += x;
            bsq[b] += x * x;
            if (ax > bx[b]) bx[b] = ax;
            const int j = (i - lane) >> 5;
            if ((float) ax > chm[j]) chm[j] = (float) ax;
            int hb = 0;   // absolute log2 bins: 2^(b-8) <= |x|, bin 0 the tiny/zero, bin 31 the overflow
            if (ax > 0.0) {
                if (ax >= 1e30)
                    hb = 31;
                else {
                    // floor, not truncation: (int) would lift [-1, 0) into the wrong bin (test_kvstats caught it).
                    const double l = log2(ax);
                    hb = l < -8.0 ? 0 : (l >= 24.0 ? 31 : (int) floor(l) + 8);
                }
            }
            shh[lane][hb] += 1;
        }
    }
    shp[lane][0] = sum;
    shp[lane][1] = sq;
    shp[lane][2] = s4;
    shp[lane][3] = mx;
    for (int b = 0; b < 8; ++b) {
        shp[lane][4 + b] = bs[b];
        shp[lane][12 + b] = bsq[b];
        shp[lane][20 + b] = bx[b];
    }
    __syncthreads();
    const uint64_t gi = (uint64_t) (((qsa * 8 + h) * 2 + tensor) * 2 + tap);
    if (gi >= KVSTATS_NGROUPS) return;
    KvStatsAccum* g = acc + gi;
    if (lane == 0) {
        double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
        double tb[8] = {0, 0, 0, 0, 0, 0, 0, 0}, tbq[8] = {0, 0, 0, 0, 0, 0, 0, 0},
               tx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int l = 0; l < 32; ++l) {
            t0 += shp[l][0];
            t1 += shp[l][1];
            t2 += shp[l][2];
            if (shp[l][3] > t3) t3 = shp[l][3];
            for (int b = 0; b < 8; ++b) {
                tb[b] += shp[l][4 + b];
                tbq[b] += shp[l][12 + b];
                if (shp[l][20 + b] > tx[b]) tx[b] = shp[l][20 + b];
            }
        }
        g->cnt += (double) n_tok * head_dim;
        g->sum += t0;
        g->sumsq += t1;
        g->sum4 += t2;
        if (t3 > g->mx) g->mx = t3;
        for (int b = 0; b < 8; ++b) {
            g->bsum[b] += tb[b];
            g->bsumsq[b] += tbq[b];
            if (tx[b] > g->bmx[b]) g->bmx[b] = tx[b];
        }
    }
    if (lane < 32) {
        uint64_t tot = 0;
        for (int t = 0; t < 32; ++t) tot += shh[t][lane];
        g->hist[lane] += tot;
    }
    for (int j = 0; j < 8; ++j) {
        const int c = lane + (j << 5);
        if (c < head_dim && chm[j] > g->chmax[c]) g->chmax[c] = chm[j];
    }
}

// The append kernels are templated on the write side's full Q4Mode; the wrappers pick the variant so that no caller
// has to know which kernel a format needs.
#define STRATA_KV_Q4_APPEND(M)                                                                   \
    kv_append_q4_kernel<M><<<grid, 32, 0, (cudaStream_t) stream>>>(                              \
        k_q4, v_q4, page_table, step, step_stride, planes, kcur, vcur, (int) s.n_head_kv,        \
        (int) s.head_dim, (int) s.page_size, host ? *host : KvHostPools{})

void kv_append_q4_steps(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host, Q4Mode mode) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_q4");
    const int planes = (v_q4 == nullptr || (v_q4 == k_q4 && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0), (unsigned) (n_steps * planes));
    if (mode == Q4Mode::kLloydLs) STRATA_KV_Q4_APPEND(Q4Mode::kLloydLs);
    else if (mode == Q4Mode::kLloyd) STRATA_KV_Q4_APPEND(Q4Mode::kLloyd);
    else if (mode == Q4Mode::kLloydVrms) STRATA_KV_Q4_APPEND(Q4Mode::kLloydVrms);
    else if (mode == Q4Mode::kLloydNoRot) STRATA_KV_Q4_APPEND(Q4Mode::kLloydNoRot);
    else if (mode == Q4Mode::kLloydClip) STRATA_KV_Q4_APPEND(Q4Mode::kLloydClip);
    else if (mode == Q4Mode::kLloydWin) STRATA_KV_Q4_APPEND(Q4Mode::kLloydWin);   // kLloyd's writer (else branch)
    else if (mode == Q4Mode::kLloyd8) STRATA_KV_Q4_APPEND(Q4Mode::kLloyd8);
    else if (mode == Q4Mode::kLloyd8K) STRATA_KV_Q4_APPEND(Q4Mode::kLloyd8K);
    else if (mode == Q4Mode::kLloyd8V) STRATA_KV_Q4_APPEND(Q4Mode::kLloyd8V);
    else STRATA_KV_Q4_APPEND(Q4Mode::kUniform);
    check("kv_append_q4 launch");
}
#undef STRATA_KV_Q4_APPEND

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host, Q4Mode mode) {
    kv_append_q4_steps(k_q4, v_q4, page_table, step, 0, 1, kcur, vcur, s, stream, host, mode);
}

#define STRATA_KV_Q4_BATCH(M)                                                                    \
    kv_append_q4_batch_kernel<M><<<grid, 32, 0, cs>>>(k_q4, v_q4, page_table, pos0, K, V,        \
                                                     (int) s.n_head_kv, (int) s.head_dim,        \
                                                     (int) s.page_size, is_v, h, st)

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage,
                  Q4Mode mode) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0));
    cudaStream_t cs = (cudaStream_t) stream;
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v) {
        if (mode == Q4Mode::kLloydLs) STRATA_KV_Q4_BATCH(Q4Mode::kLloydLs);
        else if (mode == Q4Mode::kLloyd) STRATA_KV_Q4_BATCH(Q4Mode::kLloyd);
        else if (mode == Q4Mode::kLloydVrms) STRATA_KV_Q4_BATCH(Q4Mode::kLloydVrms);
        else if (mode == Q4Mode::kLloydNoRot) STRATA_KV_Q4_BATCH(Q4Mode::kLloydNoRot);
        else if (mode == Q4Mode::kLloydClip) STRATA_KV_Q4_BATCH(Q4Mode::kLloydClip);
        else if (mode == Q4Mode::kLloydWin) STRATA_KV_Q4_BATCH(Q4Mode::kLloydWin);   // kLloyd's writer (else branch)
        else if (mode == Q4Mode::kLloyd8) STRATA_KV_Q4_BATCH(Q4Mode::kLloyd8);
        else if (mode == Q4Mode::kLloyd8K) STRATA_KV_Q4_BATCH(Q4Mode::kLloyd8K);
        else if (mode == Q4Mode::kLloyd8V) STRATA_KV_Q4_BATCH(Q4Mode::kLloyd8V);
        else STRATA_KV_Q4_BATCH(Q4Mode::kUniform);
    }
    check("kv_append_q4 batch launch");
}
#undef STRATA_KV_Q4_BATCH

// WP-2a window side pool (kv_q4.hpp, kLloydWin): the exact fp16 twin of every appended cell, flat
// [(cell, kv_head, head_dim)] per tensor - indexed directly by position, no paging, always resident.
// One kernel for every append shape: positions come from the same step array the q4 steps-kernel reads
// (step[step_idx * step_stride + kStepPos]), or from a contiguous (pos0, T) batch. Called with the same
// (already rotated) K/V the q4 append sees, right next to it, on every append path.
__global__ void kv_win_append_kernel(uint16_t* __restrict__ win, const float* __restrict__ cur,
                                     const int32_t* __restrict__ step, int step_stride, int n_head_kv,
                                     int head_dim, int64_t win_cells) {
    const int64_t step_idx = (int64_t) blockIdx.z;
    const int64_t pos = (int64_t) step[step_idx * step_stride + kStepPos];
    const int h = blockIdx.x, d0 = (blockIdx.y * (int) blockDim.x + (int) threadIdx.x);
    if (d0 >= head_dim) return;
    // K3: the pool is indexed by absolute cell, so a position outside [0, win_cells) would reach into the
    // neighbouring tensor's memory. The position is device-fresh (graph-safe) and a Release build has no
    // device-side loud fail (NDEBUG drops assert, this codebase has no device printf), so the store is dropped.
    // The host-side contract is that the pool is sized with the session's `max_cells` - the same bound the
    // horizon (step[kStepNKv]) runs to; a hit here means that bookkeeping drifted, not a normal state.
    if (pos < 0 || pos >= win_cells) return;
    const int64_t n = (int64_t) n_head_kv * head_dim;
    win[pos * n + h * head_dim + d0] = f16_from_f32(cur[step_idx * n + h * head_dim + d0]);
}

void kv_append_win_steps(uint16_t* k_win, uint16_t* v_win, const int32_t* step, int step_stride, int n_steps,
                         const float* kcur, const float* vcur, int n_head_kv, int head_dim, void* stream,
                         int64_t win_cells) {
    if (n_steps <= 0) return;
    // K2: this used to return silently, leaving the side pool stale while the caller believed it was written.
    // A geometry the grid cannot cover is a programming error, so it fails like need_256 does.
    if (head_dim <= 0 || head_dim % 32 != 0 || win_cells <= 0) {
        std::fprintf(stderr,
                     "strata: kv_append_win_steps: head_dim %d must be a positive multiple of 32 (the window "
                     "writer's grid) and win_cells %lld must be positive - the exact side pool would not be "
                     "written\n", head_dim, (long long) win_cells);
        std::exit(1);
    }
    if (k_win != nullptr)
        kv_win_append_kernel<<<dim3((unsigned) n_head_kv, (unsigned) (head_dim / 32), (unsigned) n_steps), 32, 0,
                                    (cudaStream_t) stream>>>(k_win, kcur, step, step_stride, n_head_kv, head_dim,
                                                             win_cells);
    if (v_win != nullptr && (v_win != k_win || vcur != kcur))
        kv_win_append_kernel<<<dim3((unsigned) n_head_kv, (unsigned) (head_dim / 32), (unsigned) n_steps), 32, 0,
                                    (cudaStream_t) stream>>>(v_win, vcur, step, step_stride, n_head_kv, head_dim,
                                                             win_cells);
    check("kv_append_win launch");
}

// The contiguous batch twin (the prefill path's kv_append_q4 with pos0/T): same flat pool, positions pos0 + t.
__global__ void kv_win_append_batch_kernel(uint16_t* __restrict__ win, const float* __restrict__ cur, int64_t pos0,
                                           int n_head_kv, int head_dim) {
    const int64_t t = (int64_t) blockIdx.z;
    const int h = blockIdx.x, d0 = (blockIdx.y * (int) blockDim.x + (int) threadIdx.x);
    if (d0 >= head_dim) return;
    const int64_t n = (int64_t) n_head_kv * head_dim;
    win[(pos0 + t) * n + h * head_dim + d0] = f16_from_f32(cur[t * n + h * head_dim + d0]);
}

void kv_append_win(uint16_t* k_win, uint16_t* v_win, int64_t pos0, int64_t T, const float* K, const float* V,
                   int n_head_kv, int head_dim, void* stream, int64_t win_cells) {
    if (T <= 0) return;
    // K2: loud instead of the old silent return (a stale side pool the caller believes was written).
    if (head_dim <= 0 || head_dim % 32 != 0) {
        std::fprintf(stderr, "strata: kv_append_win: head_dim %d must be a positive multiple of 32 (the window "
                             "writer's grid) - the exact side pool would not be written\n", head_dim);
        std::exit(1);
    }
    // K3: unlike the steps variant, these positions are host-known, so the pool bound IS checked here.
    if (pos0 < 0 || pos0 + T > win_cells) {
        std::fprintf(stderr, "strata: kv_append_win: cells [%lld, %lld) do not fit the exact side pool of %lld "
                             "cells\n", (long long) pos0, (long long) (pos0 + T), (long long) win_cells);
        std::exit(1);
    }
    if (k_win != nullptr)
        kv_win_append_batch_kernel<<<dim3((unsigned) n_head_kv, (unsigned) (head_dim / 32), (unsigned) T), 32, 0,
                                     (cudaStream_t) stream>>>(k_win, K, pos0, n_head_kv, head_dim);
    if (v_win != nullptr && (v_win != k_win || V != K))
        kv_win_append_batch_kernel<<<dim3((unsigned) n_head_kv, (unsigned) (head_dim / 32), (unsigned) T), 32, 0,
                                     (cudaStream_t) stream>>>(v_win, V, pos0, n_head_kv, head_dim);
    check("kv_append_win batch launch");
}

#define STRATA_KV_Q4_GATHER(NK, NV)                                                                              \
    kv_gather_q4_kernel<NK, NV><<<dim3(num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(        \
        k_q4, v_q4, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,     \
        v_scratch, k_win, v_win, win_s, win_r)

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream, Q4Mode mode, const uint16_t* k_win,
                       const uint16_t* v_win, int win_s, int win_r) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / QK4_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    if (q4_lloyd(mode)) {   // the gather reads every level-table mode alike: the scale is in the block
        if (q4_l8_k(mode) && q4_l8_v(mode)) STRATA_KV_Q4_GATHER(8, 8);
        else if (q4_l8_k(mode)) STRATA_KV_Q4_GATHER(8, 16);
        else if (q4_l8_v(mode)) STRATA_KV_Q4_GATHER(16, 8);
        else STRATA_KV_Q4_GATHER(16, 16);
    } else {
        STRATA_KV_Q4_GATHER(0, 0);
    }
    check("kv_gather_q4 launch");
}
#undef STRATA_KV_Q4_GATHER

namespace {
// File-local diagnostic state (one instance: this TU is compiled once per build).
bool kvst_enabled() {
    static bool on = (std::getenv("STRATA_KVSTATS") != nullptr);
    return on;
}
bool& kvst_broken() {
    static bool b = false;
    return b;
}
KvStatsAccum*& kvst_acc() {
    static KvStatsAccum* p = nullptr;
    return p;
}
}  // namespace

// The WP-2a clipping arm's per-tensor quantiles (kv_q4.hpp): copied to __constant__ so the writer reads them
// free. Called at startup (layer.cpp) and by the parity test; never fails the engine (a bad value keeps 0.94).
void qsa_set_kv_clip(float kappa_k, float kappa_v) {
    if (!(kappa_k > 0.0f) || !(kappa_k <= 1.0f)) kappa_k = 0.94f;
    if (!(kappa_v > 0.0f) || !(kappa_v <= 1.0f)) kappa_v = 0.94f;
    if (cudaMemcpyToSymbol(kQ4lClipKappaK, &kappa_k, sizeof(float)) != cudaSuccess) return;
    if (cudaMemcpyToSymbol(kQ4lClipKappaV, &kappa_v, sizeof(float)) != cudaSuccess) return;
}

// Eager startup allocation (generate.cpp serve init): the first window recordings happen before any
// launcher could allocate, and a graph recorded without tap kernels replays empty for every reuse - so the
// accumulators must exist before the first recording, not after the first consume.
void kv_stats_alloc() {
    static bool done = false;
    if (done) return;
    done = true;
    if (std::getenv("STRATA_KVSTATS") == nullptr) return;
    KvStatsAccum*& acc = kvst_acc();
    if (acc == nullptr && cudaMalloc((void**) &acc, sizeof(KvStatsAccum) * KVSTATS_NGROUPS) == cudaSuccess &&
        acc != nullptr)
        cudaMemset(acc, 0, sizeof(KvStatsAccum) * KVSTATS_NGROUPS);
}

// One branch when the env var is unset. Never fails the engine: no exits, silent disable on any error.
void kv_stats_launch(const float* kcur, const float* vcur, int n_head_kv, int head_dim, int64_t qsa, int n_tok,
                     bool rotated, void* stream) {
    KvStatsAccum*& acc = kvst_acc();
    if (!kvst_enabled() || kvst_broken() || kcur == nullptr || vcur == nullptr || n_head_kv <= 0 ||
        head_dim <= 0 || n_tok <= 0 || qsa < 0 || qsa >= KVSTATS_MAX_QSA || n_head_kv > KVSTATS_MAX_HEAD ||
        head_dim % 32 != 0 || head_dim > 256)
        return;
    if (acc == nullptr) {
        // Allocating inside a capture errors - skip recording windows (a few tokens); replays accumulate.
        // The harness sends a tiny warmup GEN so the main run records with accumulators present.
        const cudaStream_t cs0 = (cudaStream_t) stream;
        if (cs0 != nullptr) {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(cs0, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) return;
        }
        if (cudaMalloc((void**) &acc, sizeof(KvStatsAccum) * KVSTATS_NGROUPS) != cudaSuccess || acc == nullptr) {
            kvst_broken() = true;
            return;
        }
        if (cudaMemset(acc, 0, sizeof(KvStatsAccum) * KVSTATS_NGROUPS) != cudaSuccess) {
            kvst_broken() = true;
            return;
        }
    }
    const int blocks = n_head_kv * 2;
    kv_stats_accum_kernel<<<blocks, 32, 0, (cudaStream_t) stream>>>(kcur, vcur, acc, n_tok, head_dim, qsa,
                                                                     rotated ? 1 : 0, n_head_kv);
    if (cudaGetLastError() != cudaSuccess) kvst_broken() = true;   // silent from here: never break the engine
}

// Driver-side: copy the accumulators out, append one section, reset. Call outside capture, after the window's
// work has landed (generate.cpp's serial loop, next to the LOGPOS block).
void kv_stats_consume_window(int64_t pos0, int64_t n_tok) {
    static std::FILE* f = [] {
        const char* p = std::getenv("STRATA_KVSTATS");
        return p != nullptr ? std::fopen(p, "ab") : nullptr;
    }();
    static std::vector<unsigned char> staging;
    static bool header_done = false;
    if (f == nullptr || !kvst_enabled()) return;
    if (!header_done) {
        header_done = true;
        const char magic[8] = {'K', 'V', 'S', 'T', 'A', 'T', 'S', 2};
        std::fwrite(magic, 1, 8, f);
    }
    // Allocate here too (eager, outside capture): the first recordings happen before any launcher could
    // allocate, and the harness sends a tiny warmup GEN so the main run records with accumulators present.
    KvStatsAccum*& acc = kvst_acc();
    if (acc == nullptr) {
        if (cudaMalloc((void**) &acc, sizeof(KvStatsAccum) * KVSTATS_NGROUPS) != cudaSuccess || acc == nullptr)
            return;
        if (cudaMemset(acc, 0, sizeof(KvStatsAccum) * KVSTATS_NGROUPS) != cudaSuccess) return;
    }
    const size_t bytes = sizeof(KvStatsAccum) * KVSTATS_NGROUPS;
    staging.resize(bytes);
    if (cudaDeviceSynchronize() != cudaSuccess) return;   // diagnostic-only full sync
    if (cudaMemcpy(staging.data(), acc, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    if (cudaMemset(acc, 0, bytes) != cudaSuccess) return;
    const KvStatsAccum* g = reinterpret_cast<const KvStatsAccum*>(staging.data());
    uint32_t ng = 0;
    for (uint32_t i = 0; i < KVSTATS_NGROUPS; ++i)
        if (g[i].cnt > 0.0) ++ng;
    if (ng == 0) return;   // warmup windows leave no trace; coverage counts scored tokens only
    std::fwrite(&pos0, 8, 1, f);
    std::fwrite(&n_tok, 8, 1, f);
    std::fwrite(&ng, 4, 1, f);
    for (uint32_t i = 0; i < KVSTATS_NGROUPS; ++i) {
        if (g[i].cnt == 0.0) continue;
        const uint16_t gi = (uint16_t) i;
        std::fwrite(&gi, 2, 1, f);
        std::fwrite(&g[i], sizeof(KvStatsAccum), 1, f);
    }
}

}  // namespace strata::kernels
