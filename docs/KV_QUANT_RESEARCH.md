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

### 6.9 Deep-dive #4: WUSH-KV (ISTA, 2026-09-29) and where the fixed Hadamard ends

The newest KV-specific work in the field is **WUSH-KV** (`arXiv:2609.38121`, Jiale Chen, Vage Egiazarian, Eldar
Kurtić, Torsten Hoefler, Dan Alistarh; ISTA + ETH Zürich + Red Hat AI, submitted 2026-09-29). It shares authors with
GPTQ, AQLM, QuaRot, HIGGS and QuEST, so it is also the group whose line this repository's format sits in.

**What it does.** For **each KV head separately** it builds a `d x d` transform for K and another for V, from the Gram
matrix of the cached tensor and the Hessian of the layer output with respect to that tensor's perturbation (for K the
Hessian is the query Gram, for V the output projection). The V transform folds into `W_V`/`W_O` offline and costs
nothing online; the K transform must stay online, because it acts after the headwise norm and RoPE - Appendix A shows
that folding it requires the transform to commute with RoPE, which leaves only sign flips, and sign flips cannot
redistribute coordinate magnitudes. Calibration is 128 FineWeb-Edu sequences (12 minutes, 39 GiB on one L40S),
reused across bitwidths. The quantizer is **QuEST** (Panferov et al., 2025): per token, `2^b` levels spaced evenly
between `+/- alpha_b` times the group RMS, with `alpha_b` the constant that minimises the MSE of that grid on a
standard Gaussian, and clipping outside.

**The numbers that matter to us.** The fixed-rotation arm and the calibrated arms are published on Qwen3-8B,
WikiText-2, 2048-token sequences processed in 16-token chunks, fp16 cache = 9.72:

| transform (the same QuEST quantizer throughout) | 4-bit | 3-bit | 2-bit |
| --- | ---: | ---: | ---: |
| identity | 11.36 | 15.29 | 38.08 |
| random orthogonal | 9.77 | 10.18 | 14.22 |
| **normalized Hadamard - fixed, data-oblivious rotation** | **9.77** | **10.06** | **258.29** |
| OSCAR (calibrated, separate K/V transforms) | 9.77 | 10.19 | 13.74 |
| WUSH-KV (calibrated, separate K/V transforms) | 9.73 | 9.82 | 10.51 |

At 2 bits the Hadamard arm does not merely lose points, it stops the model: Qwen3-8B on AIME 2025 scores
`0.0 +/- 0.0` with 98 % of runs hitting the 92,160-token budget and on LiveCodeBench v6 `1.0 +/- 0.3`; WUSH-KV scores
46.7 and 35.2 there, the fp16 cache 67.8 and 47.8. Their reading is specific: key-vector spikes in the first attention
module sit in essentially one coordinate, the Hadamard spreads them into near-equal magnitudes, and at 2 bits those
land close to the midpoints between adjacent reconstruction levels. The controlled attention-stage ablation agrees -
at 2 bits the module-output error (geometric mean over Qwen3-8B's 36 modules) is 0.325 for the Hadamard transform
against 0.208 for WUSH-KV. RotateKV (IJCAI 2025) reports the same wall independently: it finds the QuaRot rotation
degrading sharply at 2 bits and repairs it with channel reordering, a pre-RoPE grouped-head rotation and explicit
attention-sink protection.

**Three readings, all load-bearing for our claim.**

1. **Our claim is a 4-bit claim, and 4 bits is where a fixed transform is safe.** On Qwen3-8B the Hadamard arm is
   9.77 against 9.72 fp16, +0.5 %. Our own four-bit formats sit in the same place on a different model with a
   different quantizer (Qwen3.8-Flash-Next, ggml per-32-block levels, 8K, 24,573 positions): `q4_0` +0.23 %,
   `q4lm32s` +0.02 % (§6.8, `docs/KV_QUANT_WORKPACKAGE.md` §7.4). "Same quality, half the bytes" - 576 B/cell
   against int8's 1056 (§1) - is inside the range this literature supports.
2. **It does not extend to 2 or 3 bits.** At 3 bits the fixed Hadamard is already +3.5 % (10.06) while the calibrated
   transforms are +1.0 % (9.82) and +4.8 % (10.19); at 2 bits it fails against +8.1 % (10.51) and +41 % (13.74). A
   data-oblivious Strata format below 4 bits therefore needs a different lever - the K/V asymmetry already shipped as
   `k8v4` (816 B/cell, §1), a full-precision window with sinks, or calibration - not more of the same rotation.
3. **The per-block scalar quantizer is exhausted as a lever.** The parity table in §6.8 gives `q4_0` 2.938e-02,
   Max-Lloyd levels at `max|x|/87` 2.740e-02, one least-squares scale pass 2.551e-02, and the *converged* best fit
   for the same 16 levels 2.456e-02. That leaves 0.17 dB of headroom for any further work on level placement or scale
   rule inside this container, and the end-to-end result is already +0.02 % PPL. What remains is structural (separate
   K and V treatment, windows/sinks) or calibrated (SpinQuant, OSCAR, OptR, WUSH-KV) - which is the split §5.3
   already makes.

### 6.10 The decision test, and what it can actually resolve (measured 2026-10-14)

Every number in §6.8, §6.9 and `docs/KV_QUANT_WORKPACKAGE.md` §7.4 rests on the harness's own readout; this is the
one measurement that says whether that readout is strong enough to carry a claim. The test is RULER FWE/CWE
(`arXiv:2404.06654` §3.3) read **teacher-forced**: 8 segments of 2048 tokens, a 40-word vocabulary of 6-character
coded words drawn with a Zipf exponent of 2.0, ten answer words, and every arm scored against the `fp16` arm of the
same run (Q2_0 pack, gfx1200, weights fixed). FWE and CWE are the retrieval half of a long context - the part a
lossy cache damages first - where a mean KL over all positions is dominated by the easy ones. Sets:
`data/quant_quality/fwe-2k/` and `data/quant_quality/cwe-2k/`.

The readout that was pre-registered - the KL over the answer region - **measures false**, and the failure is larger
on the set it was designed for. `fwe-2k` has 84 answer positions over 8 segments; int8 against `q4_0` there pairs to
`t -0.69` with 6 of 8 segments going the *other* way, on the largest difference in the run. The task accuracy is
worse than weak: `fwe-2k` has all three arms hitting the same 68 of 84, and at `cwe-2k` the 4-bit arm hits **more**
answer words than fp16 (243 against 241 of 266). A task-accuracy readout cannot see the format at all.

What replaces it is the **whole-segment KL, paired over the segments**: every arm reads the identical segment, so
`KL_a(s) - KL_b(s)` cancels the text, which is most of a single arm's spread. On `cwe-2k` (8 segments, 16,351
positions) the int8-to-4-bit boundary is then decisive - every pair 0/8 segments with `|t| >= 4.19`, int8 against
`q4_0` at `t -5.66`, sem 17.7 % of the effect - while `q4_0`, `q4lm32` and `q4lm32s` sit **inside one another's
noise** (`t` between -0.91 and -1.35, 2-5 of 8 segments the other way). `cwe-2k` is the sharper of the two sets for
that same contrast (`t -5.66` against `fwe-2k`'s `t -3.50`, sem 28.7 %). Full tables, per-segment values and the
committed `compare`/`areport`/`paired` JSON: `bench/results/2026-10-14-kv-fwe/`.

**What this forces on the wording.** The 4-bit formats are real at the quantizer (0.30 dB / 0.61 dB MSE, §6.8) and
below this test's resolution at the model. The half-the-bytes claim is therefore *not* "same quality": int8 is
measurably better than any 4-bit arm, at a cost this test can put a number on - `+0.13 %` PPL for `q4_0` at
`cwe-2k`, `+0.23 %` on the 8K whole-model set. And the three 4-bit maps cannot be ranked against each other on this
evidence: the 8K set puts `q4lm32s` first and `cwe-2k` puts `q4_0` first, which is itself the sign that the gap is
at the noise floor. §6.9's point 1 ("inside the range this literature supports") stands as a comparison with the
field; it is not a claim of equality.

### 6.11 Deep-dive #5: the 3-bit K/V allocation question, and the probe that asks it

§6.9 leaves the direction explicit: a fixed Hadamard is safe at 4 bits and fails at 3 (10.06 against the calibrated
9.82 on Qwen3-8B), so below 4 bits the next move is **how bits are allocated between K and V**, not more rotation.
The field disagrees on which side wants them, and the two sources disagree in opposite directions: llama.cpp
**#2318** measures `K2c+V4` *better* than `K4+V2c`, while **`arXiv:2502.15075`** keeps 98.3 % of K4V4 accuracy with
K4V2. Both can be right for their own cache geometry; neither answers for our 256-wide head with an FWHT-256 in
front of it.

So the question is asked before the layout is built. The **probe** puts Max-Lloyd's 8-level ladder for N(0,1)
(`{±69, ±43, ±24, ±8}`, `d = max|x|/69`, rounded by the same x32 rule as `kQ4lMap`) into the **unchanged 18-byte
`block_q4_0`** - same block, same rotation, same readers, so it is a *quality* probe at 576 B/cell and not a byte
layout. Three modes make the K-vs-V question an equal-average-width comparison (3.5 bits): `q4lm8` (8/8), `q4lm8k`
(8 on K, 16 on V), `q4lm8v` (16 on K, 8 on V), with `q4lm32` (16/16) as the reference. `q4lm8k` and `q4lm8v` are
byte-exact equals, so the pair answers #2318 against 2502.15075 on our own model.

The model-free measurement (`kv_q4_parity` test 7, synthetic N(0,1), 498,688 rotated values, gfx1200) already
settles what the probe *cannot* do: on the codebook, `q4lm8k` and `q4lm8v` are 0.1 % apart (7.672e-02 against
7.662e-02 MSE, q4lm32 2.740e-02, q4lm8 1.259e-01). Both tensors supply half the values and use the same table, so
their MSE contribution is symmetric, and only their slightly different drawn statistics separate them. **The
exchange rate between a K error and a V error is a property of the model, not of the quantizer**: a K error enters
the attention scores multiplicatively (a *relative* error after the softmax), a V error enters the output linearly
(a *per-position additive* error). That is exactly why the decision has to come from the model run and not from a
distortion table, and it is the reason the probe was built before the layout.

The 3-Bit-Sonde is throwaway code: it lives only on `KV_QUANT_research`, is marked as such in the header, and is
dropped before the WP-1/WP-2 PR. Its design and model-free numbers are in `docs/KV_QUANT_WORKPACKAGE.md` §5.4/§7.5.

**The model run (Q2_0, cwe-2k, 2026-10-14).** The run reproduces its own anchors exactly (`q4lm32` KL 0.02022,
`int8` 0.01033 as in §6.10), and it points one way. Paired whole-segment KL against `q4lm32`: K 16 -> 8 levels costs
+0.00726 (`t -3.25`, 0/8), V 16 -> 8 costs +0.00597 (`t -2.39`, 1/8) - so cutting K costs about 1.2x as much. The
direct byte-exact comparison `q4lm8k` vs `q4lm8v` is **not resolved** by that readout alone (+0.00129, `t +0.60`,
4/8). The **answer region separates categorically and only along K**: every 16-level-K arm reads 90.98 %, every
8-level-K arm 89.10 %, whatever V does; exact McNemar gives `q4lm32` vs `q4lm8v` (V 16 -> 8) 2:2, `p 1.000`, and
`q4lm8k` vs `q4lm8v` 0:5, `p 0.062`. And PPL is blind to all of it (`q4lm8` reads +0.81 % against `q4lm32`'s
+0.86 % while its KL is 65 % higher). Symmetric 3-bit (`q4lm8`) is decisively too expensive: +0.01306 KL,
`t -5.04`, 0/8, 0.39 pp of top-1.

So the direction is **K wants the bits, not V** - the direction of `arXiv:2502.15075` and of the already-shipped
`k8v4` (PR #120), and the opposite of PR #2318's headline. It is stated as a *direction, not an established
effect*: no single readout crosses the conventional threshold at 8 segments, though all three agree in sign.

**This is Q2_0, and Q2_0 is the damped case.** Q2_0's own weight error is the largest in the line (+10.4 % PPL
against Q8_0, §6.7) and it lays itself over every KV degradation, so these gaps are a *lower bound*. The
K-vs-V decision is therefore repeated on **Swift IQ3_XXS**, which is the measurement basis from here on; the Q2_0
numbers stay as the bound and as proof that the probe's wiring self-diagnoses. Table and full readouts:
`bench/results/2026-10-14-kv-3bit-probe/`.

**The Swift repeat (2026-10-14, same day).** The falsification rule from §6.12/workpackage §9 fired on schedule.
Direct byte-exact contrast q4lm8k-vs-q4lm8v on Swift: **+0.00011, t +0.05, 3/8 segments**; answer-region t +0.74,
4/8; McNemar over all pairs p = 1.000 (largest discordance 3:2) - no K-split anywhere, where Q2_0 read t +0.60
with 0:5 discordant and a categorical accuracy split. The Q2_0 direction does not survive the regime change,
and the WP-2 asymmetric-layout shape cannot rest on it anymore. Two things survive intact: the instrument
(fp16-vs-int8 pairs to t -13.34 on Swift against -4.43 on Q2_0 - segment noise collapses with the weight floor,
sd 0.00107 vs 0.00658 on the same contrast), and every boundary except K-vs-V (int8/4-bit t -4.84,
4-bit/symmetric-3-bit t -5.43 at 0/8, 4-bit/split t -2.56/-3.60; symmetric 3-bit top-1 -0.38 pp). The Swift null
is consistent with FlatQuant Table 11, which is tied at 7 total bits (K4V3 vs K3V4 nearly tied) and separates
only at 6 (K4V2 vs K2V4 decisively): at 7 total bits the split does not matter on either model. Whether it
matters at 6 on ours is an open probe question, not measured here. Treatment asymmetry (scale rules) is a
different mechanism and untouched. PPL blindness, fourth exhibit: q4lm8k reads +0.13 % (best of all 4/3-bit
arms, better than q4lm32's +0.43 %) at 1.52x its KL. Full tables:
`bench/results/2026-10-14-kv-3bit-probe-swift/`.

### 6.12 Critical review (2026-10-14): what the literature says about our plan

Read in full for this review: WUSH-KV (all sections and tables - every number quoted in §6.9 verified against
the source) and KIVI (2402.02750, full text); as abstracts plus their reception inside KIVI: SmoothQuant,
LLM.int8(), GPTQ, AWQ; as practitioner voice: ik_llama.cpp thread #1509 (March-April 2026) in full; now also
OSCAR (2605.17757, all sections, tables) and FlatQuant (2410.09426, §§1-4 plus App. C KV tables 11/12/15
and W3A3KV3 table 13). Not read in full: KVQuant
(abstract + survey state), QuaRot/SpinQuant/RotateKV (survey state).

**What holds.** The paired decision test over MSE (ikawrakow: "RMSE is not a very good predictor"); the QJL
exclusion (quant.cpp: "contributed exactly nothing"; AMD: harmful on K through the softmax; vLLM ships Hadamard
*without* QJL - so stated in the WUSH-KV paper itself); Swift over Q2_0 (ikawrakow's "forgiving model" and
"typical example" warnings cut the same way); probe-before-layout (the anchors reproduced to the digit).

**Objection 1: V-folding is dead on quantized weights.** WUSH-KV (§4.2) and OSCAR (§4) both fold the value
transform into W_V/W_O - "adds no online cost" - which assumes fp16 weights. On Swift IQ3_XXS folding means a
derived pack (re-quantizing dense-transformed projections into 3-bit weights) or an online V transform (which
kills the "free"). The arithmetic is friendly - 12 QSA layers x 2 KV heads x (T_K + T_K^{-T} + T_V) = 72
transforms of 256^2 fp16, ~9 MB download per model - but only without V-folding. Consequence, now adopted: the
Strata shape is **calibrated-K-only with fixed Hadamard on V**, which happens to be the probe's direction.
Nobody has shown calibrated-K-only; that is the innovation gap, not a deficit.

**Objection 2: windows are prerequisite at low bits, and they sit at WP-5.** KIVI in full: fake 2-bit without
the R<=128 fp16 residual window collapses on GSM8K; *with* it, ~2 % drop. OSCAR Table 5: INT2 with
rotation+clip but windows (0,0) scores **0.00 on every task**; the knee is at sink 64 + recent 256. WUSH-KV's
2/3-bit numbers all include sink+recent windows. Our probe has neither - so "symmetric 3-bit is too expensive"
(1.65x KL) is **conditional on windowless**, and every sub-4-bit layout, calibrated or not, likely needs
windows first. Consequence, now adopted: a window arm (the q2k mechanism exists) and the clipped-scale arm
below precede any layout.

**Objection 3: clipping is the cheapest unmeasured axis.** Our scale is `max|x|` - hostage to the largest
outlier, which is exactly the pathology behind WUSH-KV's H-2-bit collapse (first-module key spikes spread onto
level midpoints). OSCAR's clip thresholds (k 0.96, v 0.92) come out of calibration within 0.15 points of the
best grid point (Table 6) - no hand-tuning. FlatQuant's ablation (Table 4, LLaMA-3-8B): transform 1266 -> 8.50
PPL, then per-channel scaling +0.55, then **learnable clipping +0.84**. The older line converges on the same
lesson - LLM.int8() isolates 0.1 % outlier dims in fp16, SmoothQuant migrates difficulty, AWQ protects 1 %
salient channels, GPTQ weights by the Hessian: **tail first, bulk cheap**. Our order (codebook, then transform,
tail later) stands it on its head. Consequence, now adopted: a **clipped-scale arm on the unchanged 18-byte
blocks** (scale from the k-quantile, nothing else moves) - zero bytes, writer-only, potentially as large as the
whole transform slot.

**Objection 4: K-vs-V was 2:2; FlatQuant's Table 11 makes it 4:2.** Our probe + 2502.15075 (K4V2 keeps 98.3 %)
against #2318 (K2c+V4 > K4+V2c) + jagmarques in the ik thread (April 2026: "values need more precision than
keys", K3V8 - dismantled in-thread: K3V8 at 6 bpw is dominated on both axes by Q4_0 at 4.5 bpw, conceded;
Mistral-7B at 1742-token PPL). FlatQuant, read in full since, measures the split systematically on LLaMA-3-8B
KV-only (Table 11, group-128 asymmetric, everything else fp16): at equal 6 bits K4V2 beats K2V4 decisively (PPL
6.60 vs 7.70, QA avg 71.04 vs 68.07); at 7 bits K4V3 vs K3V4 is nearly tied (6.25/73.15 vs 6.35/72.59); K4V3
alone is nearly lossless (QA -0.08). The paper states it outright - "keys are more sensitive to quantization
than values" - as "consistent with previous studies (Hooper et al., 2024; Liu et al., 2024c; Ashkboos et al.,
2024)". Caveats stand: LLaMA-3-8B dense, head-dim 128, KV-only with fp16 rest (the undamped regime), PPL/QA
metrics rather than retrieval - and ikawrakow's "typical example" warning now covers every side including the
new one. Our direction still holds for one model, one pack, 2K context until Swift + power decide. Power math
unchanged (t=2.0 needs ~87 segments, ~11 h; or cwe-4k; or McNemar-primary at 5:0, p=0.062).

**Objection 5: the 4-bit tie is a finding, not a placeholder.** WUSH-KV's own Table 1: calibration buys 0.04 PPL
at 4 bits. Ours: three 4-bit maps inseparable. FlatQuant Table 12 is the third witness, KV-only: K4V4 tied
(7B PPL 5.50 vs 5.51, 13B 4.91 vs 4.91), K3V3 nearly (5.61 vs 5.68), and only K2V2 separates (6.66 vs 9.23) -
the calibrated advantage lives at <=3 bits. Together: **the transform slot is not a 4-bit lever.** WP-1 stays
"half the bytes for measurable +0.13 %", and anything calibrated must prove itself at <=3.5 bits against the
cheaper opponents from objections 2 and 3 - not against fixed Hadamard alone, or window/clipping gains get
attributed to the transform.

**OSCAR deltas that change our plan.** The rotation is R = U.H.P_br: an attention-aware eigenbasis (query
covariance for K, score-weighted for V), *then* Hadamard, then a bit-reversal permutation - even the calibrated
SOTA keeps Hadamard inside the composition. Ablation (Table 4, Qwen3-8B INT2, mean over 5 tasks): full 70.01,
w/o Hadamard 51.74, w/o eigenbasis (QuaRot+P_br) 32.82, no rotation (clip+sink+recent only) 4.23. Both factors
contribute; the steelman for Hadamard is measured, not assumed. Calibration is light: one pass dumping Q,K,V
(8878 tokens x layers), 8k tokens enough and domain-robust (Table 7: MMLU/WikiText/GPQA within noise) - far
below WUSH's 128 x 32K sequences, which reframes our calibration budget. Bit accounting is honest (2.28 BPE
*including* scale/zero metadata and windows at 128K) - our future layout needs the same rule. RULER-NIAH holds
to 128K while QuaRot-INT2 collapses past 16k on 8B. Theory scope as with WUSH: optimality under a frozen-error
surrogate, not covering the deployed quantizer.

**FlatQuant deltas that change our plan.** Per-layer affine P = P1xP2 (Kronecker: memory n/2, compute sqrt(n)/2)
+ per-channel scaling + learnable clipping, block-wise MSE training (~0.9 h, 26 GB for LLaMA-3-8B); per-head P_h,
P_v deliberately *not* decomposed; P_o fused into P_v (QuaRot-style); transform+quant in one Triton kernel,
overhead 0.26x (QuaRot) -> 0.07x. The ablation splits the prize three ways (transform, scaling granularity,
clipping) - our plan now measures all three separately instead of buying them as one transform. "Train one, get
more": one calibration serves several widths - calibration cost amortizes. And Figure 2: the largest errors sit
at pivot tokens (sinks again), where scaling *and* Hadamard are "powerless" - a third vote for windows from an
unexpected corner.

**FlatQuant's KV appendix (C.1/C.2) upgrades two plan points.** Table 11 (objection 4 above) makes K4V3 the
reference shape for the WP-2 layout: QA -0.08, nearly lossless on LLaMA-3-8B KV-only. Table 15 orders the
clipping work: clip *after* the transform (PPL 6.98, QA 71.23) beats clip-before (7.37/68.62) and the QuaRot
fixed threshold (0.9 act / 0.95 KV: 7.25/69.87) - while the fixed threshold already captures ~70 % of the
no-clip-to-learned gap (7.95 to 6.98) at zero calibration cost, which is exactly the WP-2a sequencing
(fixed-kappa first, learned later). The two groups' kappas converge (OSCAR 0.96/0.92, FlatQuant-fixed 0.9/0.95)
- a usable prior, not a tuned constant. Table 13 bounds the ambition below (W3A3KV3: QuaRot 686 PPL dead vs
FlatQuant 10.82 alive; "4-bit remains a better balance", their words). Table 14 (calibration on
WikiText2/C4/Pile: PPL 6.98-7.04) says the transform generalizes because it learns an equivalent map, not
weights - the mild version of our data-obliviousness demand.

**The path (adopted as Plan v2 in `docs/KV_QUANT_WORKPACKAGE.md` §7.6).** Clipped-scale arm (days, writer-only,
zero bytes) first; window arm (the q2k mechanism exists) alongside it; power (overnight segments or cwe-4k, or
McNemar-primary); then the asymmetric real layout (K4/V3 class, honest BPE); calibrated-K-only beside it as the
comparison arm, never folded into quantized weights; SRHT/no-rotation ablation riding along (the Hadamard
skepticism gets measured, both directions). Explicitly not: QJL, symmetric 2-bit, delta mode (stays WP-4),
V-folding into quantized weights.


### 6.13 The decision test against the field (2026-10-14)

How the field measures KV quantization, and where our instrument stands in it. Read in full: RULER (2404.06654,
all sections - the task catalogue we borrow from) and Yuan et al. 2407.01527 ("KV Cache Compression, But What
Must We Give in Return", full text - the benchmark-of-benchmarks); our own `tools/needle_bench.py` (118 lines)
audited line by line.

**What the field measures with.** PPL on WikiText-2/C4 is a coarse screen everywhere, never the decision metric;
every group pairs it with task metrics. KIVI: LM-Eval generation (CoQA/TruthfulQA/GSM8K) + LongBench + NIAH.
WUSH-KV: WikiText-2 PPL + AIME/MATH/GPQA/LiveCodeBench with long traces + RULER NIAH/MRCR. OSCAR: 32K-generation
evals on five reasoning/coding tasks + RULER-NIAH to 128K. Yuan et al.: LongBench (6 categories) + NIAH across
10+ methods. The pattern: **PPL screens, NIAH stresses, long reasoning traces test compounding.** Our paired KL
replaces their PPL with something strictly more sensitive; what we lack is their stress leg.

**Teacher-forced vs free generation - the real divide.** Our harness scores P(x_t | correct prefix): no error
compounding, perfect pairing, KL per position, cheap. The field (nearly) always free-generates: compounding
included, which is exactly the failure mode that kills low-bit caches downstream (WUSH-KV Table 2: H-2bit
generating 92K tokens of garbage until budget exhaustion - teacher-forced PPL shows high loss but not the
behavioral collapse). Our answer-accuracy readout is a hybrid (greedy one-step argmax, still no compounding).
Neither side subsumes the other: teacher-forced KL is the sensitive *instrument* (it found everything so far),
free-generation NIAH is the *validity gate* (does retrieval survive at length?). The division of labor is
deliberate: we rank close variants with the sensitive instrument and validate the winner at length - not the
other way round.

**Saturation policy (the H2O lesson).** Yuan et al. caught H2O at 100 % needle retrieval - an artifact of the
7-digit passkey plus a no-eviction first token; at 64 digits it fell to 35 %, while KIVI-2bit held 91 %. A
saturated test is not a result, it is an instruction to harden. This is our 4-bit answer accuracy (null):
harden, don't conclude. Hardening knobs per RULER §5: more queries (1->8 costs ~15 points), hard distractor
needles (~40 points at 256K), needle types (UUID hardest), lower alpha, longer context. Our cwe-4k/cwe-4k-flat
are built on exactly this logic; the power rule (§8.6 of the workpackage) is its statistical form.

**Prefill deviation, disclosed.** Yuan OB1: every tested KV method quantizes decode only - "keeping the prefill
process uncompressed is crucial"; KIVI/FlexGen/H2O compress nothing during prefill. Our harness routes prefill
through the quantized decode-attn-batch path, so prefill KV is quantized too. Internal comparisons stay valid
(the same-run fp16 arm shares the treatment), but absolute gaps are not comparable to field numbers, and a
shipped format would likely be decode-only. Stated here so it cannot surprise: a decode-only arm is filed work
(needs engine prefill-path support for the format), not present work.

**Discipline.** Pre-register the primary readout per question (whole-segment paired KL for ranking, McNemar
answer accuracy for K-vs-V - the fwe->cwe switch was data-driven and is now locked, workpackage §8.7);
RULER-scale n (500/length) is out of reach, pairing + power rule is the substitute; one model plus ikawrakow's
"typical example" warning plus Yuan's alignment caveat means every bench README carries the scope stamp (model,
pack, context, prefill treatment) and length-scaling (cwe-4k) is the generality proxy. Q2_0->Swift is already
the weight-regime cross-check.

**The added test: hardened needle_bench as validity gate.** `tools/needle_bench.py` exists and fits the missing
leg: free generation through the server API, greedy, thinking off, exact-match grading, repo-text haystack,
depth sweep 10/50/90, 32K/128K/262K, timing recorded. Audited gaps before it can gate WP-2: single shot per
cell (no variance - needs --repeats with fresh words/positions), one needle type (RULER: add UUID-ish codes),
no hard distractors (MK-analogue: distractor needles), single-query only (MV/MQ-analogue: incomplete-set
failures invisible), never run per-KV-arm (needs server restarts per --kv arm - operational cost, noted).
Specified in workpackage §8.8; implemented when the GPU is free. A teacher-forced NIAH inside the harness
(needle logprob + KL at needle depths, paired, no server) stays the fallback if free generation proves too
noisy per GPU-hour.

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
23. WUSH-KV (ISTA; per-head calibrated K/V transforms; the fixed-Hadamard wall at 2 and 3 bits) —
    <https://arxiv.org/abs/2609.38121>
24. RotateKV (IJCAI 2025; outlier-aware pre-RoPE grouped-head rotation, attention-sink protection, 2-bit) —
    <https://www.ijcai.org/proceedings/2025/0690>
25. FlatQuant (ICML 2025; learned Kronecker affine transforms P=P1xP2 + per-channel scaling + learnable
    clipping; read §§1-4 + App. C: ablation LT 1266->8.50 PPL, +PS 0.55, +LCT 0.84; per-head P_h/P_v not
    decomposed; "train one, get more"; pivot-token errors; Tab. 11 K-sensitivity K4V2>>K2V4, K4V3 -0.08 QA;
    Tab. 12 calibrated advantage only <=3 bits; Tab. 15 clip-after-transform, fixed kappa ~70 % of the gain;
    Tab. 13 W3A3KV3 QuaRot dead / FlatQuant alive) —
    <https://arxiv.org/abs/2410.09426>
26. RULER (the long-context benchmark whose FWE/CWE retrieval tasks the decision test borrows) —
    <https://arxiv.org/abs/2404.06654>
27. "K4V2 keeps 98.3 % of K4V4" (the counter-position to PR #2318's K2c+V4 > K4+V2c) —
    <https://arxiv.org/abs/2502.15075>
