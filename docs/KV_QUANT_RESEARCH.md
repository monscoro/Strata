# KV cache quantization: Strata against the field

Status: **research note, not a shipped claim.** Everything under "Strata today" was read from this repository.
Everything under "the field" and "open pull requests" is a literature and PR review, dated **2026-10-14**; the
papers' numbers were measured on other models and other metrics and are **not** directly comparable to ours.
Branch: `KV_QUANT_research`.

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

## 5. Sources

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
