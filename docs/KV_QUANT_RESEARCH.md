# KV cache quantization: Strata against the field

Status: **research note, not a shipped claim.** Everything under "Strata today" was read from this repository.
Everything under "the field" and "open pull requests" is a literature and PR review, dated **2026-10-14**; the
papers' numbers were measured on other models and other metrics and are **not** directly comparable to ours.
Branch: `KV_QUANT_research`.

The implementation plan derived from this note is [KV_QUANT_WORKPACKAGE.md](KV_QUANT_WORKPACKAGE.md).

The first KV-cache **measurement point** (the fixed yardstick set, same weights, only `--kv` differs) is in
[bench/results/2026-10-14-kv-first-run/](../../bench/results/2026-10-14-kv-first-run/README.md): on Q2_0, int8 is
-0.10% perplexity against fp16 K/V and q4_0 is **+1.12%** (mean KL 0.0364, 2.7x int8's). That is the gap WP-1 has to
close; the older "+8% @1K" document probe used a different, removed tool and a single long document and is not
comparable to this set.

---

## 1. What Strata does today (`--kv`)

| `--kv` | K | V | Bytes/cell | Default | Source |
| --- | --- | --- | ---: | --- | --- |
| `fp16` | fp16 | fp16 | 2048 | | `include/strata/kernels/kv_q4.hpp` |
| `int8` | int8 | int8 | 1056 | **yes** | |
| `q4_0` | rotated 4-bit | rotated 4-bit | 576 | opt-in | PR #21, engine 0.1.8 |
| `k8v4` | int8 | rotated 4-bit | 816 | opt-in | PR #120, engine 0.1.25 |

`q4_0` is a fixed, orthonormal 256-point Walsh-Hadamard rotation of K, V and Q, then ggml `q4_0` (32 values per
block: an fp16 scale and 4-bit codes). The rotation spreads a head's outlier channels over all 256 dimensions;
the query is rotated the same way, so `<Hq, Hk> = <q, k>` and the scores are unchanged. What is lossy is the 4-bit
rounding itself (`include/strata/kernels/kv_q4.hpp:1`).

Measured (`bench/results/2026-09-27-kv-q4/README.md`, Q2_0, 3,000 fixed slots, `--adapt-swaps 0`, fp16 reference):

- int8 is indistinguishable from fp16 (within ~2 standard errors everywhere).
- q4_0 costs real precision: document perplexity **+8% at 1K** and **+12% at 8K**; KL from fp16 is 2.6-4.2x int8's;
  code and chat are near neutral; needles **5/5** (1K-262K).
- Speed/memory at 128K with `--kv-resident 32768`: q4_0 67.4 tok/s (4,190 slots, 0.83 GiB K/V in RAM) vs int8
  62.4 tok/s (4,044 slots, 1.52 GiB).

So the shipped design is: **rotation + uniform `q4_0`, no codebook tuning, no residual correction.**

## 2. The field

**TurboQuant** (`arXiv:2504.19874`, Zandieh et al., ICLR 2026). Randomized WHT (random sign flips) makes the
coordinates follow a **Beta distribution**; a single analytical **Lloyd-Max codebook** fitted to that distribution
is applied per coordinate. An optional second stage adds a **1-bit QJL sketch** of the residual, which makes the
inner-product estimate unbiased. Reported: quality-neutral at **3.5 bits/channel**, marginal at 2.5, within a
factor ~2.7 of the Shannon bound. An AMD reference build reports `turbo4` at **+0.23% perplexity vs q8_0**,
`turbo3` at +1.06%, decode at 89% of f16.

**SpectralQuant (SQ)** (eigenspectral, `pypi.org/project/spectralquant`). Data-adaptive: per-head **eigenbasis**
from calibration tokens (separate `U_K`, `U_V`), then **water-filling** bit allocation (high-variance dimensions
get more bits, the tail a floor), optional QJL on the top-`d_eff`. Reported **5.95x / 6.55x / 6.68x** compression
with FP16-equivalent quality on Mistral 7B / Qwen 2.5 7B.

**SpectrumQuant** (Wang et al., ACL 2026; note the different name). DCT **along the token dimension** for the key
cache, exploiting RoPE-induced spectral sparsity, with a hybrid bit-width scheme and fused Triton kernels;
reported comparable to FP16.

Related families: KIVI (per-channel K, per-token V), KVQuant (per-channel pre-RoPE K, non-uniform codebooks),
GEAR (4-bit + low-rank + sparse residual), NQKV (per-block normal-float), Coupled Quantization.

**An independent ablation** (`arXiv:2607.09683`, AMD, synthetic regimes) compares TQ and SQ directly and is the
useful part for us:

- Heavy-tailed data is **fatal to any eigenbasis method** (TQ d2 <= 0.01, all SQ variants ~0.7) — more bits do not
  recover a misaligned subspace.
- **Water-filling reduces to uniform allocation** in most regimes.
- A single spurious eigenvalue can flip SQ to d2 > 1 (output anti-correlated).
- QJL on K helps only when residual norms are small; the softmax amplifies the extra variance (Jensen).
- Their recommendation: **SQ for prefill (small S, structured), TQ as the safe default for generation.**

## 3. Strata against the field

| | Strata `q4_0` | TurboQuant | SpectralQuant |
| --- | --- | --- | --- |
| Transform | fixed WHT-256 | randomized WHT -> Beta | per-head eigenbasis (calibrated) |
| Codebook | uniform `q4_0` | Beta Lloyd-Max (optimal) | Gaussian Lloyd-Max, var = eigenvalue |
| Bit allocation | uniform | uniform across dims | water-filling (usually ~uniform) |
| Residual | none | 1-bit QJL (optional) | QJL on top-`d_eff` (optional) |
| Calibration | none | none | yes (sensitive) |
| Local maturity | shipped, CUDA/HIP/SYCL, KV streaming | research + vLLM/AMD prototype | PyPI/Torch, HF kernels |
| Quality @4 bit | PPL +8-12% (document) | ~neutral @3.5 bit | FP16-equivalent |

In one line: **Strata `q4_0` is "TurboQuant minimised"** — the same Hadamard intuition, but with a fixed
(non-randomised) rotation, a uniform grid instead of the Beta-optimal codebook, and no QJL. That is exactly why
TurboQuant is near-lossless at 3.5 bits while Strata needs 4 and still pays +8-12% on long documents. In exchange
Strata's path is trivial to kernel and rides the existing `q4_0`/streaming code. Strata's `k8v4` (K int8, V rotated
4-bit) already sits on the **K/V asymmetry** that the TurboQuant reimplementation and the SQ/TQ ablation keep
recommending.

## 4. What the Strata pull requests already propose (this is the answer to the PR question)

Searched the `Niko1221/Strata` PRs on 2026-10-14 (`gh pr list --search "kv in:title"` and related).

**Shipped KV-format work**

- **#21** (MERGED) — `--kv q4_0` with FWHT-256 Hadamard rotation. engine 0.1.8. This is the format we documented.
- **#120** (MERGED) — `--kv k8v4`: int8 K + Hadamard-rotated `q4_0` V, 816 B/cell (issue #116). The asymmetry is
  the point: K enters the scores (errors redirect attention), V only a softmax-weighted average.
- **#293** (released in 0.1.32) — the **same Hadamard rotation applied to int8 K/V** (`STRATA_KV_ROT=1`), opt-in.
  Author's measurement (IQ2_XS, RTX 5090, Ryzen 9 9950X3D, first-token KL): int8 mean KL 0.0022 vs rotated int8
  0.0054 — it helps the NVFP4 pack and hurts IQ2_XS, so it is **off by default** (mean KL 0.0066 vs 0.0051).
- **#711** (merged) — `--kv k8v4` streams with `--kv-resident`.

**Open / in review — an improvement is indeed being proposed**

- **#2318** (OPEN, bsorensen110, created 2026-10-13) — **`--kv q2k`: K at 2 bits with per-channel scales + V at
  4 bits per token (`q4_0`), plus an FP16 sliding window (`--kv-window N`) and attention sinks (`--kv-sink S`).**
  Marked **[EXPERIMENTAL]**, opt-in, default unchanged. 448 B/cell vs q4_0's 576 (-22%). Measured on a real dumped
  cache (RTX 3090): attention error L3 1.911% (q2k) vs 0.888% (q4_0); teacher-forced top-1/KL vs FP16: document
  84.0% / 0.135 vs q4_0's 94.1% / 0.028; recall probe matches q4_0 (4/4, 4/4, 3/3) at ~13% lower decode
  (137.5 vs 159.0 tok/s); 199K-token needles 3/3 at 25/50/90% depth. It cites and re-tests **KIVI, SKVQ, KVSink,
  QuaRot, KVQuant, "More for Keys, Less for Values", AsymKV**. Its own finding at equal bytes: **K2c+V4 beats
  K4+V2c** (the bits belong on V on this rotated cache), and it needs no FP16 window to match q4_0.
- **#2351** (OPEN) — suffix-reuse conversation KV prefix mount with **quant-safe splices across all five formats**
  (fp16, int8, int8+`KV_ROT`, k8v4, q4_0); defines the format list a new format must join.
- **#1095** (OPEN) — `kv_q8` gather decodes 8 int8 per thread (V100 experimental, +11%). Related closed: #1019.
- **#1614** (OPEN) — one shared KV pool for all lanes (`--kv-pool-tokens`). **#1231 / #1232** (OPEN) — `--kv-grow`
  robustness and under `--batch`.
- **#2220** / **#1367** (OPEN) — attention-side, not format: int8-KV decode/attention kernels (sm_75, int8 tensor
  cores).

**Owner policy visible in the threads** (`Niko1221` on #293): a KV change ships only through an **identity gate**
(byte-identical output on four models) plus a **speed A/B**, and the default stays unchanged unless measured
better. That matches the [contributing rules](../AGENTS.md): a new feature is opt-in and the default path stays
byte-identical.

### Reading

- The only **new KV-cache quantization format** currently proposed upstream is **#2318 (`q2k`)** — no TurboQuant
  or SpectralQuant integration is open or merged. `TurboQuant`, `SpectralQuant`, `KIVI`, `QuaRot`, `GEAR` have
  **zero hits** in this repository.
- Everything else open is about *where* the K/V lives (streaming, pools, prefix reuse), not about a better
  quantizer.
- If the goal is to close the quality gap of `q4_0`, the two cheapest levers suggested by the literature are: (a)
  a **Beta/Lloyd-Max codebook** after the same rotation instead of uniform `q4_0`, and (b) a **1-bit QJL residual
  on K** — both data-oblivious and rotation-compatible, i.e. they fit the existing `kv_q4` path.

## 5. Follow-up work on TurboQuant (since April 2025)

### 5.1 The paper itself, clarified

TurboQuant (`arXiv:2504.19874`, ICLR 2026) is a composition of two earlier works: **PolarQuant** (norm + direction,
random preconditioning; `arXiv:2502.02617`, AISTATS 2026) for stage 1, and **QJL** (1-bit sign residual;
`arXiv:2406.03482`, AAAI 2025) for stage 2. Google Research posted the blog on 2026-03-24. Two author
clarifications (Han In-su, per a 2026-04-13 interview): the **6x** memory and **8x** attention numbers come from
different settings and must not be multiplied, and the 8x is H100-at-4-bit attention, not end-to-end. The
"3-bit zero loss" headline is shorthand: quality-neutral at **3.5 bits/channel**, marginal at 2.5.

### 5.2 Independent replications: the QJL residual is dubious for KV caches

Several careful reproductions converge on the same surprising result — **the 1-bit QJL residual adds little or
nothing, and on K it tends to hurt**:

- **quant.cpp** (Apple M1 Pro, Llama 3.2 3B, 1,040-token PPL): a literal TurboQuant port (RHT -> Lloyd-Max ->
  1-bit QJL) was **byte-for-byte identical to MSE-only in ablation — "the QJL residual stage contributed exactly
  nothing to scores."** Dropping QJL and spending the freed bytes on a **2x larger codebook** gave
  `turbo_kv_4b` PPL **14.28 (+5.3%)** vs `uniform_4b` 14.41 (+6.3%) vs llama.cpp's `q4_0` KV **~14.99 (+10.6%)**
  at the same 4-bit budget. They also credit the RHT + MSE-grid pattern to **HIGGS** (`arXiv:2411.17525`).
  *(Caveat added after reading their code, see §6.4: those runs keep V in FP16, so it is not the same budget as a
  `q4_0` K+V cache; and the codebook only wins together with a per-head max-abs scale.)*
- **scos-lab reference implementation**: MSE beats Prod for attention ("the paper recommends Prod for keys; our
  experiments show MSE for both is better"), and outlier-aware mixed precision (a few K channels at 8 bit, the
  rest at 3) recovers most of the loss.
- **AMD ablation** (`arXiv:2607.09683`): QJL on K amplifies variance through the softmax (Jensen) and is worse
  than no QJL on structured regimes; and a companion (`arXiv:2605.08114`) finds that **at 4 bits `KQV` (QJL on V
  only) wins on every measure, while `QKQV` (QJL on both) is consistently worse in KL.**
- **ik_llama.cpp #1509**: implementation note — QJL is not needed for the KV cache; `TurboQuant_mse` (Algorithm 1)
  suffices, and pre-rotated queries let attention dot against the codebook directly.

**Consequence for our own plan:** the earlier suggestion to add a "1-bit QJL residual on K" is **not supported** by
the follow-up literature and is withdrawn. The part that survives is the rotation plus a **codebook matched to the
rotated distribution**.

### 5.3 Improvements on the same core that we could use

| Idea | Source | Fits Strata? |
| --- | --- | --- |
| **Beta/Lloyd-Max codebook instead of a uniform grid** (same 4 bits, better centroids) | TurboQuant/PolarQuant, quant.cpp, vLLM presets | yes — data-oblivious, no layout size change |
| **No stored per-block scale** (norm + direction; random preconditioning removes the 1-2 bits/block of scale/zero-point) | PolarQuant `2502.02617` | partly — q4_0 pays an fp16 scale per 32 values (~0.5 bit/value) |
| **Per-channel K scales** (K outliers are channel-aligned: RMS max/median 5.8x on L3 up to 74.9x on L47) | KIVI, KVQuant, OSCAR, Strata PR #2318 | yes for K |
| **Asymmetric K/V bits** (give K the extra bit) | vLLM `turboquant_k8v4` = its best 4-bit preset (+1.17% PPL at 2.6x) | already have `k8v4` |
| **A full-precision window + attention sinks** (a slice of recent/first cells exact) | KVSink `2508.04257`, SKVQ, KIVI, Strata PR #2318 | yes, format-agnostic |
| **Keep the top-k coordinates high precision, before and after rotation** | TORQUE `2609.36032` | yes, but new code |
| **Random sign flips (SRHT) instead of a plain Hadamard** | TurboQuant; SRFT vs SRHT is "statistically indistinguishable" per `2605.05699` | cheap, theory-aligned |
| **Calibrated / covariance-aware rotation** (beats fixed Hadamard but is model- and calibration-specific) | SpinQuant `2405.16406`, OSCAR `2605.17757`, OptR `2608.02691`, WUSH-KV `2609.38121` | opt-in only — breaks data-obliviousness |

OSCAR reports INT2 staying near BF16 on Qwen3-32B and GLM-4.7 (358B) where a naive rotation "collapses to nearly
zero"; OptR raises Qwen3-8B AIME25 from 17.33% to 66.67% over QuaRot. Both are strong but need calibration data
and per-model work, which cuts against Strata shipping one code path for many packs.

### 5.4 Integration status elsewhere

- **vLLM**: PR #38479 merged (2026-04-15). Named presets with measured PPL: `turboquant_k8v4` 2.6x / **+1.17%**,
  `turboquant_4bit_nc` 3.8x / +2.71%, `turboquant_k3v4_nc` ~3.5x / +10.63%, `turboquant_3bit_nc` 4.9x / +20.59%.
  Note it applies **Lloyd-Max on K and uniform on V**, again the asymmetric split.
- **llama.cpp**: the rotation idea landed as a Hadamard KV-cache rotation (#21038, citing TurboQuant) with WHT
  kernels on CPU (#22631), CUDA (#23615) and Vulkan (#23687). Many downstream forks (e.g. `turbo3` ~4.6x at
  <1.5% PPL on Llama 3.1 8B).
- **AMD**: `turbo4` +0.23% PPL vs q8_0, `turbo3` +1.06%, decode 89% of f16.

### 5.5 What this means for Strata (prioritised)

1. **Replace the uniform `q4_0` grid with a Beta/Lloyd-Max codebook** after the existing rotation. Data-oblivious,
   same 4-bit layout plus a scale, so it slots into `kv_q4.hpp`. The code read (§6.4) says the realistic gain is
   **small** — in quant.cpp the codebook+max-abs config beat uniform min-max by ~0.9% relative on K — so measure it,
   and pair it with a per-head (not per-32) max-abs scale as they did.
2. **Per-channel scales for K** (V keeps the token-wise `q4_0` path).
3. **Keep the K/V asymmetry** (`k8v4`); vLLM's best 4-bit preset is exactly that shape.
4. **Random sign flips** on the Hadamard (SRHT) — cheap, and closer to what the theory assumes.
5. **Do not add QJL.**
6. Later / opt-in: the FP16 window + sinks (also the subject of open PR #2318), and any calibrated rotation.

All numbers above were measured on other models (GPT-2, Llama 3.2, Qwen2.5, Qwen3, MiniMax, GLM) and other metrics
(PPL vs FP32/FP16/q8_0, LongBench, RULER, AIME). Treat them as directional, not as Strata results.

## 6. Code deep-dive #1: `quant.cpp` (`quantumaikr/quant.cpp`)

Read from a shallow clone on 2026-10-14. Files: `src/core/tq_rht.c`, `tq_codebook.c`, `tq_turbo_kv.c`,
`tq_uniform.c`, `tq_value_quant.c`, `src/cache/tq_progressive.c`, `src/engine/tq_adaptive.c`,
`src/engine/tq_transformer.c`, `src/engine/tq_model.c`, `src/server/tq_server.c`,
`bench/results/turboquant_reproduction.md`, `docs/blog/breaking-3bit-barrier.md`, `README.md`,
`spec/tq_format_v1.md`.

### 6.1 What it is

Pure C, zero dependencies, ~72k LOC, ships as a single header (`quant.h`) and a WASM build. It calls itself a
**research platform for KV compression**, not a speed competitor ("Non-goals: GPU speed competition"). Seven KV
types, a tiered/progressive cache and a delta mode. That is our problem, so it is worth reading as a **worked
experiment**, not as a product.

### 6.2 The shipped pipeline, read from the code

`tq_turbo_kv_4b_quantize_ref` (`src/core/tq_turbo_kv.c:480`):

1. per-head **L2 norm** -> fp16 (2 bytes);
2. normalise to a **unit vector**;
3. **RHT**: Rademacher sign flip (`random_sign`, Knuth hash) + Walsh-Hadamard butterfly + `1/sqrt(n)`
   (`tq_rht.c`) — called with the **constant** seed `TKV_DEFAULT_SEED = 0x12345678` (`tq_turbo_kv.c:71`), so the
   sign pattern is fixed and the shipped transform is a **fixed signed Hadamard**, not a per-block randomised one;
4. **max-abs scaling**: `inv_std = CENT_4BIT_MAX / max_abs(x)` — *not* the paper's `inv_std = sqrt(d)`, not the
   true standard deviation (the header comment says `sqrt(d)`; the code does max-abs — doc and code disagree);
5. **16-level Lloyd-Max Gaussian codebook** from hard-coded Max-1960 constants (`tq_codebook.c`);
6. **no residual stage** (Variant F drops QJL);
7. attention rotates the **query once** and dots against the codebook in rotated space; the 16 centroids are
   int8-quantised and looked up with `vqtbl1q_s8` (NEON) / `pshufb` (AVX2) — that round took decode from -45% to
   fp32 parity.

Block = **128 elements = one head vector**; 72 B = 8 B header (norm, an unused residual norm, `inv_std`, padding)
+ 64 B codes = **4.5 bits/value**. Values are a **separate** path: `tq_value_quant.c` is plain **uniform**, and the
engine default is `value_quant_bits = 0`, i.e. **V stays FP16** (`tq_server.c:1524`). The CLI default is
`kv_type = TQ_TYPE_UNIFORM_4B` + V FP16 (`tq_server.c:1524-1525`); `turbo_kv_4b` is the *recommended* type, not
the default. K and V use deliberately different algorithms.

### 6.3 The one table that matters (a 9-round measured loop)

`bench/results/turboquant_reproduction.md`, Llama 3.2 3B, 1,040-token PPL, FP32 = 13.56:

| Round | Change | 4-bit PPL | 3-bit PPL |
| --- | --- | ---: | ---: |
| 0 | literal paper port, `inv_std = sqrt(d)` | 16.03 (+18.2%) | 25.84 (+90.6%) |
| A | per-block `1/std` | 15.87 | 25.07 |
| B | `inv_std = CENT_MAX/max(|x|)` | **15.39** | 84.97 (breaks) |
| C | 99th-percentile winsorising | 17.24 | — |
| D | `K·std` sweep | 15.53 (best K=2) | — |
| E | uniform 8-level min/max | 16.28 | — |
| F | **drop QJL, reinvest 16 B in a finer codebook** | **14.28** | 15.36 |
| ref | `uniform_4b` | 14.41-14.60 | — |

### 6.4 Corrections this force on §5.2 and §5.5

1. **The "same 4-bit budget" framing in §5.2 is wrong.** Every `turbo_kv_*` number in the reproduction is run with
   `-v fp16`, so quant.cpp's 14.28 is **4-bit K + FP16 V**, while our `q4_0` figure is 4-bit K **and** 4-bit V.
   14.28-vs-+12% is **not** a like-for-like gap; the comparable config is `-k turbo_kv_4b -v q4`.
2. **The codebook alone is not the win.** Their own ablation: the literal Lloyd-Max codebook was *worse* than
   uniform per-block min-max (16.03 vs 14.41) because Gaussian centroids clip a heavier-than-Gaussian tail; it only
   became best **after** the max-abs scaling (round B) and after the QJL bytes were reinvested. So §5.5 item 1
   ("a codebook... roughly halves the q4_0 gap") overstates it: the **final** codebook+max-abs config beats
   uniform by ~0.9% relative (14.41 -> 14.28) on K, not by half.
3. **quant.cpp does not randomise the rotation** (constant seed). The SRHT idea in §5.3 is ours; it is not what
   they ship.
4. **They decline to quantise QK-normed keys** rather than rotating them: `is_gemma4 && use_qk_norm` forces
   `use_quant_kv = 0` ("QK-normed keys are too sparse, cosine 0.62", `tq_transformer.c:1970`). Our target
   **Qwen3.8-Flash-Next has per-head `attn_q_norm`/`attn_k_norm`**, i.e. exactly that case — and Strata's
   rotation is the thing that makes it work where quant.cpp bailed. That is a genuine strength of our design.

### 6.5 The biggest validated lever is *not* the codebook

`README.md` (Llama 3.2 3B, **3,970 tokens** — their honest longer eval): FP32 19.41; **4-bit + a 128-token FP32
window (`progressive`) = 19.39 (-0.1%)**; 4-bit flat 20.02 (+3.1%). A tiny full-precision recent window removes
essentially the whole 4-bit penalty, and it is context-length-invariant (at 128K only ~0.1% of cells stay FP32).
It needs no calibration and is model-agnostic; it is the same idea as StreamingLLM / KVSink / SKVQ and as our own
open PR **#2318** (`--kv-window` + `--kv-sink`).

Their own caveats: at short context the same trick carries +0.6% (13.64 vs 13.56 at 957 tokens), so the flat
penalty there was already small; **no window rescues 2-bit** (flat +165%; k512 +4.3% short but +36.7% at 3,970
tokens); and 3-bit survives only **with delta compression** (+1.3% on SmolLM2/WikiText-2), at ~3.5x slower decode
(25 -> 7 tok/s) and with batched prefill disabled.

### 6.6 Other transferable details

- **Delta compression** (an FP32 "I-frame" every 64 tokens, "P-frames" store the quantised delta from the last
  anchor): the only thing that made 3-bit usable. A documented **negative result** at 2 bits — reconstruction
  drift, cosine 0.997 -> 0.885 over 200 steps. Irrelevant at 4 bits (costs speed, buys nothing).
- **Block size 128 (one head vector), not 32**: fewer scales, and a distribution a codebook can be matched to.
- **K/V asymmetry is explicit and measured**: K gets the codebook path; V is uniform and by default FP16, because
  they measured uniform V quantisation to have "terrible cosine (0.62)".
- **Per-layer bit allocation** by post-rotation kurtosis (`tq_adaptive.c`: `kurtosis > 6 -> 3-bit else 1-bit`) and
  an online Lloyd-Max recalibration — unproven, later/opt-in only.
- **Methodology debt they paid**: an early "1-bit is lossless" result was a **fallback bug** — prefill wrote FP32
  into the cache slots before quantisation ran, so the prompt attention was FP32 and masked the noise. Real cosine
  0.634 ≈ 2/pi. Their rule ("never trust a number not traced end-to-end") is worth copying.
- **Denominator discipline**: their "7.1x" is vs **FP32**; vs FP16 it is **3.5x**. Their "llama.cpp q4_0" contrast
  is partly a literature number, and their absolute PPL differs from llama.cpp's because the weight-quant path
  differs (Q8_0 vs Q4-at-load) — only deltas are comparable.
- **Honest attribution**: they renamed away from "TurboQuant" and credit **HIGGS** for RHT + MSE grids; Variant F
  is "inspired by, not identical to" the paper.

### 6.7 First candidate from this repo alone (to be re-checked after deep-dives #2 and #3)

Ranked by validated impact per unit of effort, model-agnostic:

1. **A precise recent window + sinks on top of the existing `q4_0`.** Largest validated effect (removes ~all of a
   +3% flat penalty), no calibration, works for any model; our PR #2318 is the same idea for `q2k`. Effort is in
   layout and block selection, not in new quantiser maths.
2. **A matched (Lloyd-Max) codebook with one max-abs scale per head, L2-normalised**, replacing the per-32
   uniform grid. Data-oblivious, same container, but the expected gain is **small** (~1% relative PPL on their
   benchmark) and it partially depends on moving from a per-32 to a per-head scale. Measure first.
3. **No QJL** (we never had it) and **yes to random signs** (SRHT) — unchanged from §5.5.
4. Later / opt-in: delta compression for a 3-bit mode, per-layer bit allocation.

### 6.8 Follow-up (2026-10-14): our own measurement of §6.7 item 2

Item 2 was the one candidate this repo could build without calibration, and it was flagged "measure first". It is now
built and measured as `--kv q4lm32` / `--kv q4lm32s` (design: `docs/KV_QUANT_WORKPACKAGE.md` §5.1-Nachtrag). Two
things changed against the sketch in item 2, and the measurement forced both:

- **Per-32 block, not per head.** Max-Lloyd's 16 reconstruction points for N(0,1) - the table from quant.cpp /
  TurboQuant's `tq_codebook.c` - multiplied by 32 and rounded are integers, so a scale of `max|x|/87` keeps the
  **existing** 18-byte block: the same layout, block size, block movers, streaming granule and int8 prompt-MMA
  operands as `q4_0`. Only the nibble -> value map changes. A per-head block would have been a new format in every
  backend for a lever this size.
- **The scale is a second, separate lever.** Measured in `src/kernels/kv_q4_parity.cpp` (HIP, gfx1200 / RX 9060 XT;
  the parity test's own drawn input: 487 cells x 2 KV heads x 256 values x K and V = 498,688 rotated values, 15,584
  blocks). MSE, and dB under `q4_0`'s 2.938e-02:

| Configuration (the same 16 levels unless stated) | MSE | vs `q4_0` |
| --- | ---: | ---: |
| `q4_0` uniform grid, ggml scale | 2.938e-02 | 0 dB |
| Lloyd levels, `max|x|/87` | 2.740e-02 | **-0.30 dB** |
| the same levels, `RMS/32` scale | 3.547e-02 | -0.82 dB (**worse**) |
| the same levels, converged per-block best fit | 2.456e-02 | -0.78 dB (ceiling) |
| Lloyd levels + **one** least-squares pass (`q4lm32s`) | 2.551e-02 | **-0.61 dB** |

So the codebook alone is worth 0.30 dB and the block's own scale another 0.31 dB. One least-squares pass reaches
**78.8 %** of the way to a converged fit, and the scale it lands on is only **0.4 %** from `max|x|/87` on average -
what the pass buys is the *fit*, not a wider range. That matters for item 2's suspicion that the gain "partially
depends on moving from a per-32 to a per-head scale": it does not, a per-32 block is enough. What item 2 did not say
is that the scale rule is the *larger* half, and that a fixed rescale of the levels does not reproduce it - with
`RMS/32` the error is **worse** than `q4_0`'s, because the block maximum of 32 samples sits at `max|x|/RMS = 2.363`
sigma on average, below the 2.72 sigma the widest level is drawn for.

Both formats are bitwise equal to a host reference, and the reader is unchanged between them (the scale is in the
block). The end-to-end check against the fp16-KV arm at the same weights - PPL, mean and p99 KL, top-1 - is in
`docs/KV_QUANT_WORKPACKAGE.md` §7.4: in one 8K run (24,573 positions) it is enough to take `q4_0`'s **+0.23 %** PPL
penalty down to **+0.02 %** at the same byte count, with mean KL 8.6 % and p99 KL 12.5 % lower.

## 7. Sources

Repository: `include/strata/kernels/kv_q4.hpp`, `bench/results/2026-09-27-kv-q4/README.md`, `docs/DETAILS.md:86`,
`docs/DETAILS.md:91`, PR #21, #120, #293, #711, #2318, #2351, #1095, #1614.

Literature:

1. TurboQuant — <https://arxiv.org/abs/2504.19874>
2. TQ vs SQ ablation (AMD) — <https://arxiv.org/html/2607.09683v1>
3. Statistical analysis "inspired by TurboQuant" (AMD) — <https://arxiv.org/abs/2605.08114>
4. TurboQuant AMD reference — <https://github.com/andyluo7/turboquant-amd>
5. TurboQuant reference implementation (engineering findings) — <https://github.com/scos-lab/turboquant>
6. SpectralQuant — <https://pypi.org/project/spectralquant>
7. SpectrumQuant (ACL 2026) — <https://aclanthology.org/2026.acl-long.1732>
8. PolarQuant (TurboQuant stage 1) — <https://arxiv.org/abs/2502.02617>
9. QJL (TurboQuant stage 2) — <https://arxiv.org/abs/2406.03482>
10. HIGGS (rotation + MSE-optimal grids, the earlier origin) — <https://arxiv.org/abs/2411.17525>
11. quant.cpp reproduction (QJL contributed nothing; larger codebook wins) — <https://github.com/quantumaikr/quant.cpp>
12. TORQUE — <https://arxiv.org/abs/2609.36032>
13. OSCAR (offline spectral covariance-aware rotation, INT2) — <https://arxiv.org/abs/2605.17757>
14. OptR (output-aware rotation) — <https://arxiv.org/abs/2608.02691>
15. SpinQuant (learned rotations) — <https://arxiv.org/abs/2405.16406>
16. KVSink (attention sinks) — <https://arxiv.org/abs/2508.04257>
17. "When Quantization Is Free" (SRFT vs SRHT, learned-rotation ablation) — <https://arxiv.org/abs/2605.05699>
18. ik_llama.cpp TurboQuant implementation thread (#1509) — <https://github.com/ikawrakow/ik_llama.cpp/issues/1509>
19. vLLM TurboQuant (Lloyd-Max K, uniform V, presets) — <https://docs.vllm.ai/projects/vllm-metal/en/latest/turboquant>
20. llama.cpp Hadamard KV rotation/WHT kernels — PR #21038, #22631, #23615, #23687
21. quant.cpp code deep-dive (read 2026-10-14, shallow clone): `src/core/tq_turbo_kv.c`, `tq_rht.c`,
    `tq_codebook.c`, `tq_value_quant.c`, `src/cache/tq_progressive.c`, `src/engine/tq_transformer.c`,
    `src/server/tq_server.c` — <https://github.com/quantumaikr/quant.cpp>
22. quant.cpp variant history and QJL ablation — `bench/results/turboquant_reproduction.md`; honest 4-bit/window
    numbers — `README.md`; 3-bit/delta and the 2-bit wall — `docs/blog/breaking-3bit-barrier.md`
