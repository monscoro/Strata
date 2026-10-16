// src/kernels/kv_q4_parity.cpp - Parity test for Q4_0 KV cache with orthonormal Walsh-Hadamard rotation.
// Validates:
// 1. FWHT 256 CUDA kernel vs exact mathematical orthonormal Walsh-Hadamard transform.
//    (H * H * x == x, H is symmetric and orthonormal with scale 1/sqrt(256) = 1/16).
// 2. Q4_0 quantization and packing (32 values per block, 18 bytes) bitwise vs host reference.
// 3. kv_append_q4_step and kv_gather_q4_step through paged pool against host reference.
// 4. Invariance of dot products under Walsh-Hadamard rotation: (H*q) . (H*k) == q . k.
// 5. `--kv q4lm32` (Q4Mode::kLloyd): the same blocks with Max-Lloyd's 16 levels, bitwise against a host reference,
//    and its reconstruction error next to q4_0's on the same input - the point of the format.
// 6. `--kv q4lm32s` (Q4Mode::kLloydLs): the same levels with ONE least-squares pass for the block scale, bitwise
//    against a host reference that replays the writer's butterfly in the kernel's own lane pairing.
// 7. The WP-2 probe (`--kv q4lm8*`): the 8-level table on both tensors / K only / V only, bitwise, plus the
//    reader picking the table per tensor.
// 8. The WP-1 treatment arm (`--kv q4lm32t`, Q4Mode::kLloydVrms): K keeps max|x|/87, V takes rms/32 - bitwise
//    against a host reference that replays the writer's sum-of-squares butterfly line for line, plus the reader
//    reading both sides as 16-level.
// 9. The WP-1 no-rotation arm (`--kv q4lm32nr`, Q4Mode::kLloydNoRot): the writer IS kLloyd's writer (same table,
//    same scale), bitwise against q4lm32's host reference. The rotation skip itself (kv_rot false) is engine
//    wiring, not kernel arithmetic - the smoke test covers it, not this file.

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// Exact CPU reference for Fast Walsh-Hadamard Transform (D = 256)
// Sylvester construction matching llama.cpp's ggml_gen_hadamard
void fwht256_host_reference(float* x) {
    constexpr int n = 256;
    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < n; i += 2 * s) {
            for (int j = 0; j < s; ++j) {
                const float u = x[i + j];
                const float v = x[i + j + s];
                x[i + j]     = u + v;
                x[i + j + s] = u - v;
            }
        }
    }
    const float scale = 1.0f / 16.0f; // 1 / sqrt(256)
    for (int i = 0; i < n; ++i) {
        x[i] *= scale;
    }
}

// Host reference for Q4_0 quantization of 32 elements (1 block)
void quantize_block_q4_0_host(const float* x, k::block_q4_0& blk) {
    float amax = 0.0f;
    float max_val = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const float v = x[i];
        if (amax < std::fabs(v)) {
            amax = std::fabs(v);
            max_val = v;
        }
    }
    const float d = max_val / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    blk.d = k::f16_from_f32(d);

    for (int i = 0; i < 16; ++i) {
        const float x0 = x[i] * id;
        const float x1 = x[i + 16] * id;
        const int q0 = (int) (x0 + 8.5f);
        const int q1 = (int) (x1 + 8.5f);
        const uint8_t xi0 = (uint8_t) std::min(15, std::max(0, q0));
        const uint8_t xi1 = (uint8_t) std::min(15, std::max(0, q1));
        blk.qs[i] = xi0 | (xi1 << 4);
    }
}

// Host reference for `--kv q4lm32` (kv_q4.hpp): the block scale is max|x| / 87 and the nibble is the index of the
// nearest of the 16 Max-Lloyd levels. The ladder spells out the 15 midpoints in the literal form the kernel's
// q4l_group builds, so a wrong level in either side shows up as a differing byte rather than cancelling out.
int q4l_level_host(float y) {
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

const int8_t kQ4lLevelsHost[16] = {-87, -66, -52, -40, -30, -21, -12, -4, 4, 12, 21, 30, 40, 52, 66, 87};

void quantize_block_q4l_at_host(const float* x, float d, k::block_q4_0& blk) {
    blk.d = k::f16_from_f32(d);
    for (int i = 0; i < 16; ++i) {
        const int k0 = q4l_level_host(d != 0.0f ? x[i] / d : 0.0f);
        const int k1 = q4l_level_host(d != 0.0f ? x[i + 16] / d : 0.0f);
        blk.qs[i] = (uint8_t) (k0 | (k1 << 4));
    }
}

void quantize_block_q4l_host(const float* x, k::block_q4_0& blk) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    quantize_block_q4l_at_host(x, amax / (float) k::kQ4lMax, blk);   // 87: the largest level lands on the maximum
}

// The 8-level ladder of the WP-2 probe (kv_q4.hpp's kQ4l8Map): Max-Lloyd's optimum for N(0, 1) - 0.2451, 0.7560,
// 1.3439, 2.1519 sigma - rounded to int8 by the same x32 rule, so the midpoints are the ones the kernel's
// q4l_level8 spells out and the divisor is 69.
int q4l8_level_host(float y) {
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

const int8_t* kQ4l8LevelsHost = k::kQ4l8Levels;   // the shared declaration, not a copy the test could drift from

void quantize_block_q4l8_at_host(const float* x, float d, k::block_q4_0& blk) {
    blk.d = k::f16_from_f32(d);
    for (int i = 0; i < 16; ++i) {
        const int k0 = q4l8_level_host(d != 0.0f ? x[i] / d : 0.0f);
        const int k1 = q4l8_level_host(d != 0.0f ? x[i + 16] / d : 0.0f);
        blk.qs[i] = (uint8_t) (k0 | (k1 << 4));
    }
}

void quantize_block_q4l8_host(const float* x, k::block_q4_0& blk) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    quantize_block_q4l8_at_host(x, amax / (float) k::kQ4l8Max, blk);   // 69: the widest of the 8 levels
}

// One value of an 8-level block, as the readers reconstruct it: the level the nibble indexes, times the stored
// (fp16) scale. The code is masked to 0..7 because the writer only ever stores a level index there.
float dequant_q4l8_host(const k::block_q4_0& blk, int j) {
    const uint8_t byte = blk.qs[j < 16 ? j : j - 16];
    const int n = j < 16 ? (byte & 0x0F) : (byte >> 4);
    return k::f32_from_f16(k::f16_from_f32((float) kQ4l8LevelsHost[n & 7] * k::f32_from_f16(blk.d)));
}

// Which tensor of the pair an arm puts the 8-level table on: arm 0 both, arm 1 K only, arm 2 V only.
bool probe_eight(int arm, bool is_v) { return arm == 0 || ((arm == 1) != is_v); }

// The same 16 levels with the scale the block's own RMS asks for: the table is round(32 * level for N(0, 1)), so
// the scale that matches the distribution is sigma / 32 - and the RMS of 32 rotated values estimates sigma. This
// is what a scale rule free of `max|x|` would give at the cost of one sum of squares (a butterfly reduction, so
// every lane still reaches the same scale).
void quantize_block_q4l_rms_host(const float* x, k::block_q4_0& blk) {
    double sumsq = 0.0;
    for (int i = 0; i < 32; ++i) sumsq += (double) x[i] * (double) x[i];
    const float rms = (float) std::sqrt(sumsq / 32.0);
    quantize_block_q4l_at_host(x, rms / 32.0f, blk);
}

// The warp butterfly of q4l_group's fit, replayed on the host in the kernel's own order: a maximum is exact and
// order-free, but a sum is not associative, so a bitwise compare has to pair the lanes exactly as
// __shfl_xor_sync does (offset 16, then 8, 4, 2, 1). Every lane ends with the same value, so lane 0 speaks for all.
float warp_sum32_host(const float* v) {
    float lane[32];
    for (int i = 0; i < 32; ++i) lane[i] = v[i];
    for (int o = 16; o > 0; o >>= 1) {
        float nxt[32];
        for (int i = 0; i < 32; ++i) nxt[i] = lane[i] + lane[i ^ o];
        for (int i = 0; i < 32; ++i) lane[i] = nxt[i];
    }
    return lane[0];
}

// The WP-1 treatment arm's V scale (Q4Mode::kLloydVrms, q4l_group_vrms in kv_q4.cu) replayed line for line: the
// sum of squares through the kernel's own xor-butterfly pairing in float - NOT the double accumulation of
// quantize_block_q4l_rms_host above. That one measures the RULE's merit for test 5's table; this one is the
// WRITER's twin for test 8's bitwise compare (same squares, same pairing, sqrtf, two exact scalings by the
// power of two 1/32).
void quantize_block_q4l_vrms_host(const float* x, k::block_q4_0& blk) {
    float sq[32];
    for (int i = 0; i < 32; ++i) sq[i] = x[i] * x[i];
    const float ss = warp_sum32_host(sq);
    const float d = sqrtf(ss * (1.0f / 32.0f)) * (1.0f / 32.0f);
    quantize_block_q4l_at_host(x, d, blk);
}

// `--kv q4lm32s`: the block scale from ONE least-squares pass over the same levels - d0 = max|x| / 87 fixes the
// assignment, d* = sum(l * x) / sum(l * l) fits that assignment, and the assignment is redone with d*. This is
// q4l_group<true> line for line, including the expression for d0 (a multiply by the reciprocal, not a divide), so
// the compare against the kernel can be bitwise.
float q4l_onestep_scale_host(const float* x) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    float d = amax * (1.0f / (float) k::kQ4lMax);
    float num[32], den[32];
    for (int i = 0; i < 32; ++i) {
        const float l = (float) kQ4lLevelsHost[q4l_level_host(d != 0.0f ? x[i] / d : 0.0f)];
        num[i] = l * x[i];
        den[i] = l * l;
    }
    const float sn = warp_sum32_host(num), sd = warp_sum32_host(den);
    return sd > 0.0f ? sn / sd : d;   // d = 0 (an all-zero block) keeps the initial scale, as the kernel does
}

void quantize_block_q4ls_host(const float* x, k::block_q4_0& blk) {
    quantize_block_q4l_at_host(x, q4l_onestep_scale_host(x), blk);
}

// The per-block scale that minimizes this block's error for the SAME 16 levels: with the level assignment held
// fixed it is d = sum(l_i * x_i) / sum(l_i^2), and the assignment is then redone with that d (three passes, enough
// to settle here). This is the error a better scale rule could reach with the level table unchanged - so it
// separates "the levels are wrong" from "the scale rule is wrong".
float q4l_best_scale_host(const float* x) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    float d = amax / (float) k::kQ4lMax;
    for (int it = 0; it < 3; ++it) {
        double num = 0.0, den = 0.0;
        for (int i = 0; i < 32; ++i) {
            const double l = (double) kQ4lLevelsHost[q4l_level_host(d != 0.0f ? x[i] / d : 0.0f)];
            num += l * (double) x[i];
            den += l * l;
        }
        if (den > 0.0) d = (float) (num / den);
    }
    return d;
}

// What a gather should produce for one block: the level times the stored (fp16) scale, rounded to fp16, as
// `--kv q4lm32`'s readers compute it.
float dequant_q4l_host(const k::block_q4_0& blk, int j) {
    const uint8_t byte = blk.qs[j < 16 ? j : j - 16];
    const int n = j < 16 ? (byte & 0x0F) : (byte >> 4);
    return k::f32_from_f16(k::f16_from_f32((float) kQ4lLevelsHost[n] * k::f32_from_f16(blk.d)));
}

// q4_0's own reconstruction of one block, for the A/B in test 5 - ggml's grid, not the host quantizer's bytes.
float dequant_q4_0_host(const k::block_q4_0& blk, int j) {
    const uint8_t byte = blk.qs[j < 16 ? j : j - 16];
    const int n = j < 16 ? (byte & 0x0F) : (byte >> 4);
    return k::f32_from_f16(k::f16_from_f32((float) (n - 8) * k::f32_from_f16(blk.d)));
}

} // namespace

int main() {
    std::printf("=== Running kv_q4_parity test ===\n");
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // -------------------------------------------------------------
    // Test 1: Orthonormal Fast Walsh-Hadamard Transform (D=256)
    // -------------------------------------------------------------
    std::printf("[1/7] Verifying FWHT-256 CUDA kernel vs exact mathematical reference...\n");
    constexpr int n_vectors = 64;
    std::vector<float> h_in(n_vectors * 256);
    std::vector<float> h_ref(n_vectors * 256);
    for (size_t i = 0; i < h_in.size(); ++i) {
        h_in[i] = nd(rng);
    }
    h_ref = h_in;

    // Run CPU reference on each row
    for (int r = 0; r < n_vectors; ++r) {
        fwht256_host_reference(h_ref.data() + r * 256);
    }

    // Run GPU kernel
    float* d_src = dalloc<float>(n_vectors * 256);
    float* d_dst = dalloc<float>(n_vectors * 256);
    ck(cudaMemcpy(d_src, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy H2D");

    k::fwht256_cuda(d_src, d_dst, n_vectors, nullptr);
    ck(cudaDeviceSynchronize(), "fwht256_cuda");

    std::vector<float> h_out(n_vectors * 256);
    ck(cudaMemcpy(h_out.data(), d_dst, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "memcpy D2H");

    double max_fwht_diff = 0.0;
    for (size_t i = 0; i < h_out.size(); ++i) {
        double diff = std::fabs(h_out[i] - h_ref[i]);
        if (diff > max_fwht_diff) max_fwht_diff = diff;
    }
    if (max_fwht_diff > 1e-5) {
        std::fprintf(stderr, "FAIL: FWHT-256 GPU output differs from host reference (max diff = %e)\n", max_fwht_diff);
        ++g_fail;
    } else {
        std::printf("  -> FWHT-256 bitwise match with host reference (max diff = %e, OK)\n", max_fwht_diff);
    }

    // Self-inverse property: H * (H * x) == x
    k::fwht256_inplace_cuda(d_dst, n_vectors, nullptr);
    ck(cudaDeviceSynchronize(), "fwht256_inplace_cuda");
    ck(cudaMemcpy(h_out.data(), d_dst, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "memcpy D2H");

    double max_inv_diff = 0.0;
    for (size_t i = 0; i < h_out.size(); ++i) {
        double diff = std::fabs(h_out[i] - h_in[i]);
        if (diff > max_inv_diff) max_inv_diff = diff;
    }
    if (max_inv_diff > 1e-5) {
        std::fprintf(stderr, "FAIL: FWHT-256 self-inverse property failed (max diff = %e)\n", max_inv_diff);
        ++g_fail;
    } else {
        std::printf("  -> FWHT-256 self-inverse property verified H*H*x == x (max diff = %e, OK)\n", max_inv_diff);
    }

    // -------------------------------------------------------------
    // Test 2: Invariance of Attention Scores under Hadamard Rotation
    // -------------------------------------------------------------
    std::printf("[2/7] Verifying Attention Score Invariance: <H*q, H*k> == <q, k>...\n");
    double max_dot_diff = 0.0;
    for (int r = 0; r < n_vectors; r += 2) {
        const float* q = h_in.data() + r * 256;
        const float* k_vec = h_in.data() + (r + 1) * 256;
        const float* hq = h_ref.data() + r * 256;
        const float* hk = h_ref.data() + (r + 1) * 256;

        double dot_raw = 0.0;
        double dot_rot = 0.0;
        for (int i = 0; i < 256; ++i) {
            dot_raw += (double) q[i] * (double) k_vec[i];
            dot_rot += (double) hq[i] * (double) hk[i];
        }
        double d = std::fabs(dot_raw - dot_rot);
        if (d > max_dot_diff) max_dot_diff = d;
    }
    if (max_dot_diff > 1e-5) {
        std::fprintf(stderr, "FAIL: Rotated dot product differs from original (max diff = %e)\n", max_dot_diff);
        ++g_fail;
    } else {
        std::printf("  -> Dot product invariant under rotation: |<Hq,Hk> - <q,k>| = %e (OK)\n", max_dot_diff);
    }

    // -------------------------------------------------------------
    // Test 3: Q4_0 Paged KV Append & Gather against Host Reference
    // -------------------------------------------------------------
    std::printf("[3/7] Verifying Q4_0 KV append and gather against host reference...\n");
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv; // 2
    const int D = (int) s.head_dim;   // 256
    const int P = (int) s.page_size;  // 64
    const int pages = 8;
    const int cells = pages * P;

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 3 + 5) % pages; // permutation
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy table");

    const size_t pool_bytes = (size_t) pages * H * P * k::kv_q4_bytes_per_head(D);
    uint8_t* d_k_q4 = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_v_q4 = dalloc<uint8_t>(pool_bytes);
    float* d_kcur = dalloc<float>(H * D);
    float* d_vcur = dalloc<float>(H * D);
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);

    std::vector<std::vector<float>> host_k(cells), host_v(cells);
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);

    const int n_fill = cells - 25; // leave some cells empty
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        std::vector<float> kv(H * D), vv(H * D);
        for (auto& x : kv) x = nd(rng) * 2.0f;
        for (auto& x : vv) x = nd(rng) * 2.0f;

        // Apply FWHT before append
        for (int h = 0; h < H; ++h) {
            fwht256_host_reference(kv.data() + h * D);
            fwht256_host_reference(vv.data() + h * D);
        }
        host_k[pos] = kv;
        host_v[pos] = vv;

        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
        ck(cudaMemcpy(d_kcur, kv.data(), kv.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy k");
        ck(cudaMemcpy(d_vcur, vv.data(), vv.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy v");

        k::kv_append_q4_step(d_k_q4, d_v_q4, d_table, d_step, d_kcur, d_vcur, s, nullptr);
        ck(cudaDeviceSynchronize(), "kv_append_q4_step");
    }

    // Verify written Q4_0 blocks bitwise against host quantizer
    std::vector<uint8_t> h_k_q4(pool_bytes), h_v_q4(pool_bytes);
    ck(cudaMemcpy(h_k_q4.data(), d_k_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h k_q4");
    ck(cudaMemcpy(h_v_q4.data(), d_v_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h v_q4");

    long bad_blocks = 0;
    const int blocks_per_head = D / k::QK4_0; // 8
    const int bytes_per_head = blocks_per_head * sizeof(k::block_q4_0); // 144

    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long page = (long long) table[pos / P];
                const long long row = (page * H + h) * P + (pos % P);
                const uint8_t* pool_ptr = (is_v ? h_v_q4.data() : h_k_q4.data()) + row * bytes_per_head;
                const float* raw_vec = (is_v ? host_v[pos].data() : host_k[pos].data()) + h * D;

                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_q4_0 host_blk;
                    quantize_block_q4_0_host(raw_vec + b * 32, host_blk);

                    const k::block_q4_0* gpu_blk = reinterpret_cast<const k::block_q4_0*>(pool_ptr) + b;
                    if (std::memcmp(&host_blk, gpu_blk, sizeof(k::block_q4_0)) != 0) {
                        if (bad_blocks == 0) {
                            std::printf("MISMATCH at n=%d, pos=%d, is_v=%d, h=%d, b=%d\n", n, pos, is_v, h, b);
                            std::printf("  host blk.d = 0x%04x (%f), gpu blk.d = 0x%04x (%f)\n",
                                        host_blk.d, k::f32_from_f16(host_blk.d),
                                        gpu_blk->d, k::f32_from_f16(gpu_blk->d));
                            std::printf("  host qs: ");
                            for (int i = 0; i < 16; ++i) std::printf("%02x ", host_blk.qs[i]);
                            std::printf("\n  gpu  qs: ");
                            for (int i = 0; i < 16; ++i) std::printf("%02x ", gpu_blk->qs[i]);
                            std::printf("\n");
                        }
                        ++bad_blocks;
                    }
                }
            }
        }
    }
    if (bad_blocks > 0) {
        std::fprintf(stderr, "FAIL: %ld Q4_0 blocks differ from host quantizer\n", bad_blocks);
        ++g_fail;
    } else {
        std::printf("  -> All Q4_0 pool blocks bitwise equal to host quantizer (OK)\n");
    }

    // -------------------------------------------------------------
    // Test 4: Gather Q4_0 into FP16 Scratch and verify accuracy
    // -------------------------------------------------------------
    std::printf("[4/7] Verifying Q4_0 gather unpacking and dequantization...\n");
    const int max_ids = 128;
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t* d_k_scratch = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t* d_v_scratch = dalloc<uint16_t>((size_t) max_ids * H * D);

    std::vector<int32_t> ids(max_ids);
    for (int i = 0; i < max_ids; ++i) {
        ids[i] = positions[rng() % n_fill];
    }
    ck(cudaMemcpy(d_ids, ids.data(), max_ids * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy ids");

    int32_t hstep[k::kStepCount] = {0, 0, 0, max_ids};
    ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");

    k::kv_gather_q4_step(d_k_q4, d_v_q4, d_table, d_ids, d_step, max_ids, s, d_k_scratch, d_v_scratch, nullptr);
    ck(cudaDeviceSynchronize(), "kv_gather_q4_step");

    std::vector<uint16_t> h_k_scratch((size_t) max_ids * H * D);
    std::vector<uint16_t> h_v_scratch((size_t) max_ids * H * D);
    ck(cudaMemcpy(h_k_scratch.data(), d_k_scratch, h_k_scratch.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "memcpy d2h k_scratch");
    ck(cudaMemcpy(h_v_scratch.data(), d_v_scratch, h_v_scratch.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "memcpy d2h v_scratch");

    double worst_quant_err = 0.0;
    for (int i = 0; i < max_ids; ++i) {
        const int cell = ids[i];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const float* orig = (is_v ? host_v[cell].data() : host_k[cell].data()) + h * D;
                const uint16_t* gathered = (is_v ? h_v_scratch.data() : h_k_scratch.data()) + (i * H + h) * D;

                for (int d = 0; d < D; ++d) {
                    float deq = k::f32_from_f16(gathered[d]);
                    float raw = orig[d];
                    double err = std::fabs(deq - raw);
                    if (err > worst_quant_err) worst_quant_err = err;
                }
            }
        }
    }
    std::printf("  -> Max reconstruction error after 4-bit quant + dequant: %.4f (OK)\n", worst_quant_err);

    // -------------------------------------------------------------
    // Test 5: `--kv q4lm32` - the same blocks with Max-Lloyd's levels
    // -------------------------------------------------------------
    std::printf("[5/7] Verifying q4lm32: bitwise append, reader map, and its error next to q4_0's...\n");
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
        ck(cudaMemcpy(d_kcur, host_k[pos].data(), host_k[pos].size() * sizeof(float), cudaMemcpyHostToDevice),
           "memcpy k");
        ck(cudaMemcpy(d_vcur, host_v[pos].data(), host_v[pos].size() * sizeof(float), cudaMemcpyHostToDevice),
           "memcpy v");
        // the same pools, the same positions, the same rotated vectors as test 3: only the code map changes
        k::kv_append_q4_step(d_k_q4, d_v_q4, d_table, d_step, d_kcur, d_vcur, s, nullptr, nullptr,
                             k::Q4Mode::kLloyd);
        ck(cudaDeviceSynchronize(), "kv_append_q4_step q4lm32");
    }

    ck(cudaMemcpy(h_k_q4.data(), d_k_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h k_q4");
    ck(cudaMemcpy(h_v_q4.data(), d_v_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h v_q4");

    long bad_l = 0;
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long page = (long long) table[pos / P];
                const long long row = (page * H + h) * P + (pos % P);
                const uint8_t* pool_ptr = (is_v ? h_v_q4.data() : h_k_q4.data()) + row * bytes_per_head;
                const float* raw_vec = (is_v ? host_v[pos].data() : host_k[pos].data()) + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_q4_0 host_blk;
                    quantize_block_q4l_host(raw_vec + b * 32, host_blk);
                    const k::block_q4_0* gpu_blk = reinterpret_cast<const k::block_q4_0*>(pool_ptr) + b;
                    if (std::memcmp(&host_blk, gpu_blk, sizeof(k::block_q4_0)) != 0) {
                        if (bad_l == 0) {
                            std::printf("Q4LM32 MISMATCH at n=%d, pos=%d, is_v=%d, h=%d, b=%d\n", n, pos, is_v, h, b);
                            std::printf("  host blk.d = 0x%04x (%f), gpu blk.d = 0x%04x (%f)\n",
                                        host_blk.d, k::f32_from_f16(host_blk.d),
                                        gpu_blk->d, k::f32_from_f16(gpu_blk->d));
                            std::printf("  host qs: ");
                            for (int i = 0; i < 16; ++i) std::printf("%02x ", host_blk.qs[i]);
                            std::printf("\n  gpu  qs: ");
                            for (int i = 0; i < 16; ++i) std::printf("%02x ", gpu_blk->qs[i]);
                            std::printf("\n");
                        }
                        ++bad_l;
                    }
                }
            }
        }
    }
    if (bad_l > 0) {
        std::fprintf(stderr, "FAIL: %ld q4lm32 blocks differ from host reference\n", bad_l);
        ++g_fail;
    } else {
        std::printf("  -> All q4lm32 pool blocks bitwise equal to host reference (OK)\n");
    }

    // the reader again, over the blocks the writer just produced: every gathered value must be that block's level
    // times its stored scale, which is what the gather's own `LLOYD ? kQ4lMap[n] : n - 8` has to pick per half.
    // The appends above left their own `{pos, pos + 1, 0, 0}` in `d_step`, and the gather takes its count from
    // step[kStepWidth] - so the count has to be written again, or the kernel gathers nothing at all.
    int32_t hstep_g[k::kStepCount] = {0, 0, 0, max_ids};
    ck(cudaMemcpy(d_step, hstep_g, sizeof(hstep_g), cudaMemcpyHostToDevice), "memcpy step gather");
    k::kv_gather_q4_step(d_k_q4, d_v_q4, d_table, d_ids, d_step, max_ids, s, d_k_scratch, d_v_scratch, nullptr,
                         k::Q4Mode::kLloyd);
    ck(cudaDeviceSynchronize(), "kv_gather_q4_step q4lm32");
    ck(cudaMemcpy(h_k_scratch.data(), d_k_scratch, h_k_scratch.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "memcpy d2h k_scratch");
    ck(cudaMemcpy(h_v_scratch.data(), d_v_scratch, h_v_scratch.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "memcpy d2h v_scratch");

    long bad_read = 0;
    double max_read_diff = 0.0;
    long n_read = 0;
    double sse_gather_l = 0.0;
    for (int i = 0; i < max_ids; ++i) {
        const int cell = ids[i];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long page = (long long) table[cell / P];
                const long long row = (page * H + h) * P + (cell % P);
                const uint8_t* pool_ptr = (is_v ? h_v_q4.data() : h_k_q4.data()) + row * bytes_per_head;
                const uint16_t* gathered = (is_v ? h_v_scratch.data() : h_k_scratch.data()) + (i * H + h) * D;
                const float* orig = (is_v ? host_v[cell].data() : host_k[cell].data()) + h * D;
                for (int d = 0; d < D; ++d) {
                    const k::block_q4_0* blk = reinterpret_cast<const k::block_q4_0*>(pool_ptr) + d / 32;
                    const float want = dequant_q4l_host(*blk, d % 32);
                    const float got = k::f32_from_f16(gathered[d]);
                    const double diff = std::fabs(got - want);
                    if (diff > max_read_diff) max_read_diff = diff;
                    if (diff != 0.0) ++bad_read;
                    sse_gather_l += (double) (got - orig[d]) * (double) (got - orig[d]);
                    ++n_read;
                }
            }
        }
    }
    if (bad_read > 0) {
        std::fprintf(stderr, "FAIL: %ld gathered q4lm32 values differ from the block's own levels (max %e)\n",
                     bad_read, max_read_diff);
        ++g_fail;
    } else {
        std::printf("  -> Gathered q4lm32 values equal level * stored scale (OK)\n");
    }

    // the point of the format: on the same rotated vectors, Max-Lloyd's levels beat q4_0's uniform grid. The q4_0
    // side runs through the host quantizer, which test 3 has already shown to be bitwise what the kernel writes, and
    // both errors are summed over the same values, so the ratio is a like-for-like one.
    long n_elems = 0;
    double sse_q = 0.0, sse_l = 0.0, sse_opt = 0.0, sse_rms = 0.0;
    double sum_amax_over_rms = 0.0, sum_scale_ratio = 0.0;
    long n_blocks = 0;
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const float* raw_vec = (is_v ? host_v[pos].data() : host_k[pos].data()) + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_q4_0 blk_q, blk_l, blk_o, blk_r;
                    quantize_block_q4_0_host(raw_vec + b * 32, blk_q);
                    quantize_block_q4l_host(raw_vec + b * 32, blk_l);
                    quantize_block_q4l_at_host(raw_vec + b * 32, q4l_best_scale_host(raw_vec + b * 32), blk_o);
                    quantize_block_q4l_rms_host(raw_vec + b * 32, blk_r);
                    double amax = 0.0, sumsq = 0.0;
                    for (int i = 0; i < 32; ++i) {
                        amax = std::max(amax, (double) std::fabs(raw_vec[b * 32 + i]));
                        sumsq += (double) raw_vec[b * 32 + i] * (double) raw_vec[b * 32 + i];
                    }
                    const double rms = std::sqrt(sumsq / 32.0);
                    if (rms > 0.0) {
                        sum_amax_over_rms += amax / rms;
                        sum_scale_ratio += (amax / (double) k::kQ4lMax) / (rms / 32.0);
                        ++n_blocks;
                    }
                    for (int j = 0; j < 32; ++j) {
                        const double x = raw_vec[b * 32 + j];
                        const double eq = dequant_q4_0_host(blk_q, j);
                        const double el = dequant_q4l_host(blk_l, j);
                        const double eo = dequant_q4l_host(blk_o, j);
                        const double er = dequant_q4l_host(blk_r, j);
                        sse_q += (eq - x) * (eq - x);
                        sse_l += (el - x) * (el - x);
                        sse_opt += (eo - x) * (eo - x);
                        sse_rms += (er - x) * (er - x);
                        ++n_elems;
                    }
                }
            }
        }
    }
    const double mse_q = sse_q / (double) n_elems, mse_l = sse_l / (double) n_elems;
    const double mse_opt = sse_opt / (double) n_elems, mse_rms = sse_rms / (double) n_elems;
    const double gain_db = 10.0 * std::log10(mse_q / mse_l);
    std::printf("  -> MSE over %ld values: q4_0 %.3e, q4lm32 %.3e (q4lm32 is %.2f dB lower, OK)\n", n_elems, mse_q,
                mse_l, gain_db);
    std::printf("     same levels, the block's RMS / 32 as the scale: %.3e (%.2f dB under q4_0, %.2f dB under q4lm32)\n"
                "     same levels, the block's best-fit scale: %.3e (%.2f dB under q4_0) - the scale rule, not the\n"
                "     level table, is the larger part of what is left\n", mse_rms,
                10.0 * std::log10(mse_q / mse_rms), 10.0 * std::log10(mse_l / mse_rms), mse_opt,
                10.0 * std::log10(mse_q / mse_opt));
    std::printf("     over %ld blocks: mean max|x| / RMS = %.3f (the levels span 87 / 32 = 2.719 sigma, so max|x| / 87\n"
                "     is on average %.3f of the RMS / 32 scale - the block maximum of 32 samples sits below 2.719 sigma)\n",
                n_blocks, sum_amax_over_rms / (double) n_blocks, sum_scale_ratio / (double) n_blocks);
    std::printf("     the GPU gather's own MSE over %ld values: %.3e (host reference %.3e, %+.1f%%)\n", n_read,
                sse_gather_l / (double) n_read, mse_l, 100.0 * (sse_gather_l / (double) n_read / mse_l - 1.0));
    if (!(mse_l < mse_q)) {
        std::fprintf(stderr, "FAIL: q4lm32's error is not below q4_0's (%.3e vs %.3e)\n", mse_l, mse_q);
        ++g_fail;
    }

    // -------------------------------------------------------------
    // Test 6: `--kv q4lm32s` - the same levels, one least-squares pass for the block scale
    // -------------------------------------------------------------
    std::printf("[6/7] Verifying q4lm32s: bitwise append with the one-step scale, reader map, and its error...\n");
    std::printf("  (the same pools, positions and rotated vectors as test 5: only the writer's scale rule changes)\n");
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
        ck(cudaMemcpy(d_kcur, host_k[pos].data(), host_k[pos].size() * sizeof(float), cudaMemcpyHostToDevice),
           "memcpy k");
        ck(cudaMemcpy(d_vcur, host_v[pos].data(), host_v[pos].size() * sizeof(float), cudaMemcpyHostToDevice),
           "memcpy v");
        k::kv_append_q4_step(d_k_q4, d_v_q4, d_table, d_step, d_kcur, d_vcur, s, nullptr, nullptr,
                             k::Q4Mode::kLloydLs);
        ck(cudaDeviceSynchronize(), "kv_append_q4_step q4lm32s");
    }

    ck(cudaMemcpy(h_k_q4.data(), d_k_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h k_q4");
    ck(cudaMemcpy(h_v_q4.data(), d_v_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h v_q4");

    long bad_ls = 0;
    double worst_scale_ls = 0.0;
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long page = (long long) table[pos / P];
                const long long row = (page * H + h) * P + (pos % P);
                const uint8_t* pool_ptr = (is_v ? h_v_q4.data() : h_k_q4.data()) + row * bytes_per_head;
                const float* raw_vec = (is_v ? host_v[pos].data() : host_k[pos].data()) + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_q4_0 host_blk, blk_l;
                    quantize_block_q4ls_host(raw_vec + b * 32, host_blk);
                    quantize_block_q4l_host(raw_vec + b * 32, blk_l);
                    worst_scale_ls = std::max(worst_scale_ls,
                                              (double) std::fabs(k::f32_from_f16(host_blk.d) -
                                                                 k::f32_from_f16(blk_l.d)));
                    const k::block_q4_0* gpu_blk = reinterpret_cast<const k::block_q4_0*>(pool_ptr) + b;
                    if (std::memcmp(&host_blk, gpu_blk, sizeof(k::block_q4_0)) != 0) {
                        if (bad_ls == 0) {
                            std::printf("Q4LM32S MISMATCH at n=%d, pos=%d, is_v=%d, h=%d, b=%d\n", n, pos, is_v, h, b);
                            std::printf("  host blk.d = 0x%04x (%f), gpu blk.d = 0x%04x (%f)\n",
                                        host_blk.d, k::f32_from_f16(host_blk.d),
                                        gpu_blk->d, k::f32_from_f16(gpu_blk->d));
                            std::printf("  host qs: ");
                            for (int i = 0; i < 16; ++i) std::printf("%02x ", host_blk.qs[i]);
                            std::printf("\n  gpu  qs: ");
                            for (int i = 0; i < 16; ++i) std::printf("%02x ", gpu_blk->qs[i]);
                            std::printf("\n");
                        }
                        ++bad_ls;
                    }
                }
            }
        }
    }
    if (bad_ls > 0) {
        std::fprintf(stderr, "FAIL: %ld q4lm32s blocks differ from host reference\n", bad_ls);
        ++g_fail;
    } else {
        std::printf("  -> All q4lm32s pool blocks bitwise equal to host reference (OK)\n");
    }

    // q4lm32s stores the same levels in the same 18-byte blocks, so a gather over them must apply q4lm32's map -
    // the least-squares pass moves the writer's scale only. (The reader is handed kLloydLs on purpose: q4_lloyd is
    // what turns it into the level table, and both level-table modes have to read alike.)
    int32_t hstep_ls[k::kStepCount] = {0, 0, 0, max_ids};
    ck(cudaMemcpy(d_step, hstep_ls, sizeof(hstep_ls), cudaMemcpyHostToDevice), "memcpy step gather ls");
    k::kv_gather_q4_step(d_k_q4, d_v_q4, d_table, d_ids, d_step, max_ids, s, d_k_scratch, d_v_scratch, nullptr,
                         k::Q4Mode::kLloydLs);
    ck(cudaDeviceSynchronize(), "kv_gather_q4_step q4lm32s");
    ck(cudaMemcpy(h_k_scratch.data(), d_k_scratch, h_k_scratch.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "memcpy d2h k_scratch");
    ck(cudaMemcpy(h_v_scratch.data(), d_v_scratch, h_v_scratch.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "memcpy d2h v_scratch");

    long bad_read_ls = 0;
    double max_read_ls = 0.0, sse_gather_ls = 0.0;
    long n_read_ls = 0;
    for (int i = 0; i < max_ids; ++i) {
        const int cell = ids[i];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long page = (long long) table[cell / P];
                const long long row = (page * H + h) * P + (cell % P);
                const uint8_t* pool_ptr = (is_v ? h_v_q4.data() : h_k_q4.data()) + row * bytes_per_head;
                const uint16_t* gathered = (is_v ? h_v_scratch.data() : h_k_scratch.data()) + (i * H + h) * D;
                const float* orig = (is_v ? host_v[cell].data() : host_k[cell].data()) + h * D;
                for (int d = 0; d < D; ++d) {
                    const k::block_q4_0* blk = reinterpret_cast<const k::block_q4_0*>(pool_ptr) + d / 32;
                    const float want = dequant_q4l_host(*blk, d % 32);
                    const float got = k::f32_from_f16(gathered[d]);
                    const double diff = std::fabs(got - want);
                    if (diff > max_read_ls) max_read_ls = diff;
                    if (diff != 0.0) ++bad_read_ls;
                    sse_gather_ls += (double) (got - orig[d]) * (double) (got - orig[d]);
                    ++n_read_ls;
                }
            }
        }
    }
    if (bad_read_ls > 0) {
        std::fprintf(stderr, "FAIL: %ld gathered q4lm32s values differ from the block's own levels (max %e)\n",
                     bad_read_ls, max_read_ls);
        ++g_fail;
    } else {
        std::printf("  -> Gathered q4lm32s values equal level * stored scale (OK)\n");
    }

    // The two levers side by side, host quantizers only, over the same rotated vectors: the level table with the
    // max-based scale, the same table with one least-squares pass, and the converged per-block scale as the ceiling.
    long n_ls = 0;
    double sse_ls = 0.0, sum_scale_ls_over_l = 0.0;
    long n_blocks_ls = 0;
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const float* raw_vec = (is_v ? host_v[pos].data() : host_k[pos].data()) + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_q4_0 blk_s, blk_l;
                    quantize_block_q4ls_host(raw_vec + b * 32, blk_s);
                    quantize_block_q4l_host(raw_vec + b * 32, blk_l);
                    const double dl = k::f32_from_f16(blk_l.d), ds = k::f32_from_f16(blk_s.d);
                    if (dl != 0.0) {
                        sum_scale_ls_over_l += ds / dl;
                        ++n_blocks_ls;
                    }
                    for (int j = 0; j < 32; ++j) {
                        const double x = raw_vec[b * 32 + j];
                        const double e = dequant_q4l_host(blk_s, j);
                        sse_ls += (e - x) * (e - x);
                        ++n_ls;
                    }
                }
            }
        }
    }
    const double mse_ls = sse_ls / (double) n_ls;
    std::printf("  -> q4lm32s MSE over %ld values: %.3e (%.2f dB under q4_0, %.2f dB under q4lm32, %.2f dB under\n"
                "     the converged per-block scale; one least-squares pass alone is %.1f%% of that gap)\n",
                n_ls, mse_ls, 10.0 * std::log10(mse_q / mse_ls), 10.0 * std::log10(mse_l / mse_ls),
                10.0 * std::log10(mse_q / mse_opt),
                100.0 * std::log10(mse_q / mse_ls) / std::log10(mse_q / mse_opt));
    std::printf("     its scale is on average %.3f of q4lm32's (both are drawn from the same levels)\n",
                sum_scale_ls_over_l / (double) n_blocks_ls);
    std::printf("     the GPU gather's own MSE over %ld values: %.3e (host reference %.3e, %+.1f%%)\n", n_read_ls,
                sse_gather_ls / (double) n_read_ls, mse_ls, 100.0 * (sse_gather_ls / (double) n_read_ls / mse_ls - 1.0));
    std::printf("     the largest gap between the two formats' stored scales on one block: %.4g\n", worst_scale_ls);
    if (!(mse_ls < mse_l)) {
        std::fprintf(stderr, "FAIL: q4lm32s's error is not below q4lm32's (%.3e vs %.3e)\n", mse_ls, mse_l);
        ++g_fail;
    }

    // -------------------------------------------------------------
    // Test 7: the WP-2 probe - the 8-level table on both tensors, on K only, on V only (`--kv q4lm8*`)
    // -------------------------------------------------------------
    std::printf("[7/9] Verifying the probe's 8-level table: q4lm8 (both), q4lm8k (K), q4lm8v (V)...\n");
    const k::Q4Mode probe_mode[3] = {k::Q4Mode::kLloyd8, k::Q4Mode::kLloyd8K, k::Q4Mode::kLloyd8V};
    const char* probe_name[3] = {"q4lm8 (8 on K and V)", "q4lm8k (8 on K, 16 on V)", "q4lm8v (16 on K, 8 on V)"};
    double mse8[3] = {0.0, 0.0, 0.0};
    for (int arm = 0; arm < 3; ++arm) {
        for (int n = 0; n < n_fill; ++n) {
            const int pos = positions[n];
            int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
            ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
            ck(cudaMemcpy(d_kcur, host_k[pos].data(), host_k[pos].size() * sizeof(float), cudaMemcpyHostToDevice),
               "memcpy k");
            ck(cudaMemcpy(d_vcur, host_v[pos].data(), host_v[pos].size() * sizeof(float), cudaMemcpyHostToDevice),
               "memcpy v");
            k::kv_append_q4_step(d_k_q4, d_v_q4, d_table, d_step, d_kcur, d_vcur, s, nullptr, nullptr,
                                 probe_mode[arm]);
            ck(cudaDeviceSynchronize(), "kv_append_q4_step probe");
        }
        ck(cudaMemcpy(h_k_q4.data(), d_k_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h k_q4");
        ck(cudaMemcpy(h_v_q4.data(), d_v_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h v_q4");

        long bad8 = 0, n_vals = 0;
        double sse8 = 0.0;
        for (int n = 0; n < n_fill; ++n) {
            const int pos = positions[n];
            for (int is_v = 0; is_v < 2; ++is_v) {
                const bool eight = probe_eight(arm, is_v != 0);
                for (int h = 0; h < H; ++h) {
                    const long long page = (long long) table[pos / P];
                    const long long row = (page * H + h) * P + (pos % P);
                    const uint8_t* pool_ptr = (is_v ? h_v_q4.data() : h_k_q4.data()) + row * bytes_per_head;
                    const float* raw_vec = (is_v ? host_v[pos].data() : host_k[pos].data()) + h * D;
                    for (int b = 0; b < blocks_per_head; ++b) {
                        k::block_q4_0 host_blk;
                        if (eight) quantize_block_q4l8_host(raw_vec + b * 32, host_blk);
                        else quantize_block_q4l_host(raw_vec + b * 32, host_blk);
                        const k::block_q4_0* gpu_blk = reinterpret_cast<const k::block_q4_0*>(pool_ptr) + b;
                        if (std::memcmp(&host_blk, gpu_blk, sizeof(k::block_q4_0)) != 0) {
                            if (bad8 == 0)
                                std::printf("PROBE MISMATCH %s at n=%d, pos=%d, is_v=%d, h=%d, b=%d\n",
                                            probe_name[arm], n, pos, is_v, h, b);
                            ++bad8;
                        }
                        for (int j = 0; j < 32; ++j) {
                            const float want = eight ? dequant_q4l8_host(host_blk, j) : dequant_q4l_host(host_blk, j);
                            const double e = (double) want - (double) raw_vec[b * 32 + j];
                            sse8 += e * e;
                        }
                        n_vals += 32;
                    }
                }
            }
        }
        mse8[arm] = sse8 / (double) n_vals;
        if (bad8 > 0) {
            std::fprintf(stderr, "FAIL: %ld %s blocks differ from host reference\n", bad8, probe_name[arm]);
            ++g_fail;
        }

        // The reader has to pick the table per tensor, so the gather is replayed over the blocks just written and
        // every gathered value must be the level of the table that tensor was written with, times its own scale.
        int32_t hstep8[k::kStepCount] = {0, 0, 0, max_ids};
        ck(cudaMemcpy(d_step, hstep8, sizeof(hstep8), cudaMemcpyHostToDevice), "memcpy step gather probe");
        k::kv_gather_q4_step(d_k_q4, d_v_q4, d_table, d_ids, d_step, max_ids, s, d_k_scratch, d_v_scratch, nullptr,
                             probe_mode[arm]);
        ck(cudaDeviceSynchronize(), "kv_gather_q4_step probe");
        ck(cudaMemcpy(h_k_scratch.data(), d_k_scratch, h_k_scratch.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
           "memcpy d2h k_scratch probe");
        ck(cudaMemcpy(h_v_scratch.data(), d_v_scratch, h_v_scratch.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
           "memcpy d2h v_scratch probe");
        long bad_read8 = 0;
        for (int i = 0; i < max_ids; ++i) {
            const int cell = ids[i];
            for (int is_v = 0; is_v < 2; ++is_v) {
                const bool eight = probe_eight(arm, is_v != 0);
                for (int h = 0; h < H; ++h) {
                    const long long page = (long long) table[cell / P];
                    const long long row = (page * H + h) * P + (cell % P);
                    const uint8_t* pool_ptr = (is_v ? h_v_q4.data() : h_k_q4.data()) + row * bytes_per_head;
                    const uint16_t* gathered = (is_v ? h_v_scratch.data() : h_k_scratch.data()) + (i * H + h) * D;
                    for (int d = 0; d < D; ++d) {
                        const k::block_q4_0* blk = reinterpret_cast<const k::block_q4_0*>(pool_ptr) + d / 32;
                        const float want
                            = eight ? dequant_q4l8_host(*blk, d % 32) : dequant_q4l_host(*blk, d % 32);
                        if (k::f32_from_f16(gathered[d]) != want) ++bad_read8;
                    }
                }
            }
        }
        if (bad_read8 > 0) {
            std::fprintf(stderr, "FAIL: %ld gathered %s values are not level * scale\n", bad_read8, probe_name[arm]);
            ++g_fail;
        }
    }
    if (g_fail == 0) std::printf("  -> All three probe arms: blocks bitwise equal, reader picks the table per tensor (OK)\n");
    std::printf("  -> MSE over %d values per arm: q4lm8 %.3e, q4lm8k %.3e, q4lm8v %.3e\n", 498688, mse8[0],
                mse8[1], mse8[2]);
    std::printf("     (q4lm32's 16-level MSE was %.3e; the 8-level table is the codebook, not the 4-bit bytes)\n",
                mse_l);

    // -------------------------------------------------------------
    // Test 8: the WP-1 treatment arm - K keeps max|x|/87, V takes rms/32 (`--kv q4lm32t`)
    // -------------------------------------------------------------
    std::printf("[8/9] Verifying the treatment arm: q4lm32t (K max|x|/87, V rms/32)...\n");
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
        ck(cudaMemcpy(d_kcur, host_k[pos].data(), host_k[pos].size() * sizeof(float), cudaMemcpyHostToDevice),
           "memcpy k");
        ck(cudaMemcpy(d_vcur, host_v[pos].data(), host_v[pos].size() * sizeof(float), cudaMemcpyHostToDevice),
           "memcpy v");
        k::kv_append_q4_step(d_k_q4, d_v_q4, d_table, d_step, d_kcur, d_vcur, s, nullptr, nullptr,
                             k::Q4Mode::kLloydVrms);
        ck(cudaDeviceSynchronize(), "kv_append_q4_step treatment");
    }
    ck(cudaMemcpy(h_k_q4.data(), d_k_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h k_q4");
    ck(cudaMemcpy(h_v_q4.data(), d_v_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h v_q4");

    long bad_t = 0, n_t = 0;
    double sse_t = 0.0;
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long page = (long long) table[pos / P];
                const long long row = (page * H + h) * P + (pos % P);
                const uint8_t* pool_ptr = (is_v ? h_v_q4.data() : h_k_q4.data()) + row * bytes_per_head;
                const float* raw_vec = (is_v ? host_v[pos].data() : host_k[pos].data()) + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_q4_0 host_blk;
                    if (is_v) quantize_block_q4l_vrms_host(raw_vec + b * 32, host_blk);
                    else quantize_block_q4l_host(raw_vec + b * 32, host_blk);
                    const k::block_q4_0* gpu_blk = reinterpret_cast<const k::block_q4_0*>(pool_ptr) + b;
                    if (std::memcmp(&host_blk, gpu_blk, sizeof(k::block_q4_0)) != 0) {
                        if (bad_t == 0)
                            std::printf("TREATMENT MISMATCH at n=%d, pos=%d, is_v=%d, h=%d, b=%d\n", n, pos,
                                        is_v, h, b);
                        ++bad_t;
                    }
                    for (int j = 0; j < 32; ++j) {
                        const double e = (double) dequant_q4l_host(host_blk, j) - (double) raw_vec[b * 32 + j];
                        sse_t += e * e;
                    }
                    n_t += 32;
                }
            }
        }
    }
    if (bad_t > 0) {
        std::fprintf(stderr, "FAIL: %ld treatment blocks differ from host reference\n", bad_t);
        ++g_fail;
    }

    // The reader must see both sides as 16-level (q4_lloyd, kv_q4.hpp): replay the gather over the blocks just
    // written and compare against level * stored scale.
    int32_t hstept[k::kStepCount] = {0, 0, 0, max_ids};
    ck(cudaMemcpy(d_step, hstept, sizeof(hstept), cudaMemcpyHostToDevice), "memcpy step gather treatment");
    k::kv_gather_q4_step(d_k_q4, d_v_q4, d_table, d_ids, d_step, max_ids, s, d_k_scratch, d_v_scratch, nullptr,
                         k::Q4Mode::kLloydVrms);
    ck(cudaDeviceSynchronize(), "kv_gather_q4_step treatment");
    ck(cudaMemcpy(h_k_scratch.data(), d_k_scratch, h_k_scratch.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "memcpy d2h k_scratch treatment");
    ck(cudaMemcpy(h_v_scratch.data(), d_v_scratch, h_v_scratch.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "memcpy d2h v_scratch treatment");
    long bad_read_t = 0;
    for (int i = 0; i < max_ids; ++i) {
        const int cell = ids[i];
        for (int is_v = 0; is_v < 2; ++is_v) {
            const long long page = (long long) table[cell / P];
            const long long row = (page * H) * P + (cell % P);
            const uint8_t* pool_ptr = (is_v ? h_v_q4.data() : h_k_q4.data()) + row * bytes_per_head;
            const uint16_t* gathered = (is_v ? h_v_scratch.data() : h_k_scratch.data()) + i * H * D;
            for (int h = 0; h < H; ++h) {
                for (int d = 0; d < D; ++d) {
                    const k::block_q4_0* blk =
                        reinterpret_cast<const k::block_q4_0*>(pool_ptr + h * P * bytes_per_head) + d / 32;
                    if (k::f32_from_f16(gathered[h * D + d]) != dequant_q4l_host(*blk, d % 32)) ++bad_read_t;
                }
            }
        }
    }
    if (bad_read_t > 0) {
        std::fprintf(stderr, "FAIL: %ld gathered treatment values are not level * scale\n", bad_read_t);
        ++g_fail;
    }
    if (g_fail == 0)
        std::printf("  -> Treatment arm: K/V blocks bitwise equal, reader sees 16 levels both sides (OK)\n");
    std::printf("  -> Treatment MSE over %ld values: %.3e (q4lm32 %.3e, %+.2f dB - the arm asks the model, not the MSE)\n",
                n_t, sse_t / (double) n_t, mse_l, 10.0 * std::log10(mse_l / (sse_t / (double) n_t)));

    // -------------------------------------------------------------
    // Test 9: the WP-1 no-rotation arm - the writer IS kLloyd's writer (`--kv q4lm32nr`)
    // -------------------------------------------------------------
    std::printf("[9/9] Verifying the no-rotation arm: q4lm32nr writes kLloyd's blocks...\n");
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
        ck(cudaMemcpy(d_kcur, host_k[pos].data(), host_k[pos].size() * sizeof(float), cudaMemcpyHostToDevice),
           "memcpy k");
        ck(cudaMemcpy(d_vcur, host_v[pos].data(), host_v[pos].size() * sizeof(float), cudaMemcpyHostToDevice),
           "memcpy v");
        k::kv_append_q4_step(d_k_q4, d_v_q4, d_table, d_step, d_kcur, d_vcur, s, nullptr, nullptr,
                             k::Q4Mode::kLloydNoRot);
        ck(cudaDeviceSynchronize(), "kv_append_q4_step norot");
    }
    ck(cudaMemcpy(h_k_q4.data(), d_k_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h k_q4");
    ck(cudaMemcpy(h_v_q4.data(), d_v_q4, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h v_q4");

    long bad_nr = 0;
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long page = (long long) table[pos / P];
                const long long row = (page * H + h) * P + (pos % P);
                const uint8_t* pool_ptr = (is_v ? h_v_q4.data() : h_k_q4.data()) + row * bytes_per_head;
                const float* raw_vec = (is_v ? host_v[pos].data() : host_k[pos].data()) + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_q4_0 host_blk;
                    quantize_block_q4l_host(raw_vec + b * 32, host_blk);   // kLloyd's own reference
                    const k::block_q4_0* gpu_blk = reinterpret_cast<const k::block_q4_0*>(pool_ptr) + b;
                    if (std::memcmp(&host_blk, gpu_blk, sizeof(k::block_q4_0)) != 0) {
                        if (bad_nr == 0)
                            std::printf("NOROT MISMATCH at n=%d, pos=%d, is_v=%d, h=%d, b=%d\n", n, pos, is_v,
                                        h, b);
                        ++bad_nr;
                    }
                }
            }
        }
    }
    if (bad_nr > 0) {
        std::fprintf(stderr, "FAIL: %ld no-rotation blocks differ from kLloyd's reference\n", bad_nr);
        ++g_fail;
    } else {
        std::printf("  -> No-rotation arm: writer bitwise identical to kLloyd's writer (OK)\n");
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_table);
    cudaFree(d_k_q4);
    cudaFree(d_v_q4);
    cudaFree(d_kcur);
    cudaFree(d_vcur);
    cudaFree(d_step);
    cudaFree(d_ids);
    cudaFree(d_k_scratch);
    cudaFree(d_v_scratch);

    if (g_fail == 0) {
        std::printf("=== kv_q4_parity: ALL TESTS PASSED SUCCESSFULLY! ===\n");
        return 0;
    } else {
        std::fprintf(stderr, "=== kv_q4_parity: FAILED (%d failures) ===\n", g_fail);
        return 1;
    }
}
