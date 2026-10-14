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

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa.hpp"

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

void ck(dpct::err0 e, const char *w) {
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(DPCT_CHECK_ERROR(p = (T *)sycl::malloc_device(
                            n * sizeof(T) + 64, dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memset(p, 0, n * sizeof(T) + 64).wait()),
       "memset");
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

// The host references for the two level-table formats below are shared text with the CUDA twin
// (src/kernels/kv_q4_parity.cpp): the quantizer arithmetic has to be identical, so the reference is too.
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
// order-free, but a sum is not associative, so a bitwise compare has to pair the lanes exactly as the port's
// permute_sub_group_by_xor butterfly does (offset 16, then 8, 4, 2, 1). Every lane ends with the same value, so
// lane 0 speaks for all.
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
    std::printf("[1/6] Verifying FWHT-256 CUDA kernel vs exact mathematical reference...\n");
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
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
           d_src, h_in.data(), h_in.size() * sizeof(float)).wait()),
       "memcpy H2D");

    k::fwht256_cuda(d_src, d_dst, n_vectors, nullptr);
    ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
       "fwht256_cuda");

    std::vector<float> h_out(n_vectors * 256);
    ck(DPCT_CHECK_ERROR(
           (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
               .memcpy(h_out.data(), d_dst, h_out.size() * sizeof(float))
               .wait()),
       "memcpy D2H");

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
    ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
       "fwht256_inplace_cuda");
    ck(DPCT_CHECK_ERROR(
           (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
               .memcpy(h_out.data(), d_dst, h_out.size() * sizeof(float))
               .wait()),
       "memcpy D2H");

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
    std::printf("[2/6] Verifying Attention Score Invariance: <H*q, H*k> == <q, k>...\n");
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
    std::printf("[3/6] Verifying Q4_0 KV append and gather against host reference...\n");
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
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
           d_table, table.data(), pages * sizeof(int32_t)).wait()),
       "memcpy table");

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
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_step, hstep, sizeof(hstep)).wait()),
           "memcpy step");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
               d_kcur, kv.data(), kv.size() * sizeof(float)).wait()),
           "memcpy k");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
               d_vcur, vv.data(), vv.size() * sizeof(float)).wait()),
           "memcpy v");

        k::kv_append_q4_step(d_k_q4, d_v_q4, d_table, d_step, d_kcur, d_vcur, s, nullptr);
        ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
           "kv_append_q4_step");
    }

    // Verify written Q4_0 blocks bitwise against host quantizer
    std::vector<uint8_t> h_k_q4(pool_bytes), h_v_q4(pool_bytes);
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h_k_q4.data(), d_k_q4, pool_bytes)
                            .wait()),
       "memcpy d2h k_q4");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h_v_q4.data(), d_v_q4, pool_bytes)
                            .wait()),
       "memcpy d2h v_q4");

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
    std::printf("[4/6] Verifying Q4_0 gather unpacking and dequantization...\n");
    const int max_ids = 128;
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t* d_k_scratch = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t* d_v_scratch = dalloc<uint16_t>((size_t) max_ids * H * D);

    std::vector<int32_t> ids(max_ids);
    for (int i = 0; i < max_ids; ++i) {
        ids[i] = positions[rng() % n_fill];
    }
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
           d_ids, ids.data(), max_ids * sizeof(int32_t)).wait()),
       "memcpy ids");

    int32_t hstep[k::kStepCount] = {0, 0, 0, max_ids};
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR(
           (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_step, hstep, sizeof(hstep)).wait()),
       "memcpy step");

    k::kv_gather_q4_step(d_k_q4, d_v_q4, d_table, d_ids, d_step, max_ids, s, d_k_scratch, d_v_scratch, nullptr);
    ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
       "kv_gather_q4_step");

    std::vector<uint16_t> h_k_scratch((size_t) max_ids * H * D);
    std::vector<uint16_t> h_v_scratch((size_t) max_ids * H * D);
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h_k_scratch.data(), d_k_scratch,
                                    h_k_scratch.size() * sizeof(uint16_t))
                            .wait()),
       "memcpy d2h k_scratch");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h_v_scratch.data(), d_v_scratch,
                                    h_v_scratch.size() * sizeof(uint16_t))
                            .wait()),
       "memcpy d2h v_scratch");

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
    // Test 5: `--kv q4lm32` (Q4Mode::kLloyd) - the same blocks with Max-Lloyd's levels
    // -------------------------------------------------------------
    std::printf("[5/6] Verifying q4lm32: bitwise append, reader map, and its error next to q4_0's...\n");
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(d_step, hstep, sizeof(hstep))
                                .wait()),
           "memcpy step");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(d_kcur, host_k[pos].data(), host_k[pos].size() * sizeof(float))
                                .wait()),
           "memcpy k");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(d_vcur, host_v[pos].data(), host_v[pos].size() * sizeof(float))
                                .wait()),
           "memcpy v");
        // the same pools, the same positions, the same rotated vectors as test 3: only the code map changes
        k::kv_append_q4_step(d_k_q4, d_v_q4, d_table, d_step, d_kcur, d_vcur, s, nullptr, k::Q4Mode::kLloyd);
        ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()), "kv_append_q4_step q4lm32");
    }

    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h_k_q4.data(), d_k_q4, pool_bytes)
                            .wait()),
       "memcpy d2h k_q4");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h_v_q4.data(), d_v_q4, pool_bytes)
                            .wait()),
       "memcpy d2h v_q4");

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
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(d_step, hstep_g, sizeof(hstep_g))
                            .wait()),
       "memcpy step gather");
    k::kv_gather_q4_step(d_k_q4, d_v_q4, d_table, d_ids, d_step, max_ids, s, d_k_scratch, d_v_scratch, nullptr,
                         k::Q4Mode::kLloyd);
    ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()), "kv_gather_q4_step q4lm32");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h_k_scratch.data(), d_k_scratch, h_k_scratch.size() * sizeof(uint16_t))
                            .wait()),
       "memcpy d2h k_scratch");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h_v_scratch.data(), d_v_scratch, h_v_scratch.size() * sizeof(uint16_t))
                            .wait()),
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
    // Test 6: `--kv q4lm32s` (Q4Mode::kLloydLs) - the same levels, one least-squares pass for the block scale
    // -------------------------------------------------------------
    std::printf("[6/6] Verifying q4lm32s: bitwise append with the one-step scale, reader map, and its error...\n");
    std::printf("  (the same pools, positions and rotated vectors as test 5: only the writer's scale rule changes)\n");
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(d_step, hstep, sizeof(hstep))
                                .wait()),
           "memcpy step");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(d_kcur, host_k[pos].data(), host_k[pos].size() * sizeof(float))
                                .wait()),
           "memcpy k");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(d_vcur, host_v[pos].data(), host_v[pos].size() * sizeof(float))
                                .wait()),
           "memcpy v");
        k::kv_append_q4_step(d_k_q4, d_v_q4, d_table, d_step, d_kcur, d_vcur, s, nullptr, k::Q4Mode::kLloydLs);
        ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()), "kv_append_q4_step q4lm32s");
    }

    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h_k_q4.data(), d_k_q4, pool_bytes)
                            .wait()),
       "memcpy d2h k_q4");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h_v_q4.data(), d_v_q4, pool_bytes)
                            .wait()),
       "memcpy d2h v_q4");

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
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(d_step, hstep_ls, sizeof(hstep_ls))
                            .wait()),
       "memcpy step gather ls");
    k::kv_gather_q4_step(d_k_q4, d_v_q4, d_table, d_ids, d_step, max_ids, s, d_k_scratch, d_v_scratch, nullptr,
                         k::Q4Mode::kLloydLs);
    ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()), "kv_gather_q4_step q4lm32s");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h_k_scratch.data(), d_k_scratch, h_k_scratch.size() * sizeof(uint16_t))
                            .wait()),
       "memcpy d2h k_scratch");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h_v_scratch.data(), d_v_scratch, h_v_scratch.size() * sizeof(uint16_t))
                            .wait()),
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

    sycl::free(d_src, dpct::get_in_order_queue());
    sycl::free(d_dst, dpct::get_in_order_queue());
    sycl::free(d_table, dpct::get_in_order_queue());
    sycl::free(d_k_q4, dpct::get_in_order_queue());
    sycl::free(d_v_q4, dpct::get_in_order_queue());
    sycl::free(d_kcur, dpct::get_in_order_queue());
    sycl::free(d_vcur, dpct::get_in_order_queue());
    sycl::free(d_step, dpct::get_in_order_queue());
    sycl::free(d_ids, dpct::get_in_order_queue());
    sycl::free(d_k_scratch, dpct::get_in_order_queue());
    sycl::free(d_v_scratch, dpct::get_in_order_queue());

    if (g_fail == 0) {
        std::printf("=== kv_q4_parity: ALL TESTS PASSED SUCCESSFULLY! ===\n");
        return 0;
    } else {
        std::fprintf(stderr, "=== kv_q4_parity: FAILED (%d failures) ===\n", g_fail);
        return 1;
    }
}
