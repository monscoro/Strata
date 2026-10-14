# 2026-10-14: `--kv q4lm32` / `q4lm32s`, the Q4_0 block with Max-Lloyd's levels

`q4_0` carries K/V in 18-byte blocks whose 16 codes sit on a **uniform** grid. The Hadamard rotation that makes 4 bits
usable also makes a rotated coordinate roughly Gaussian, and a uniform grid is not the best 16-level quantizer for
that. This change keeps the block byte for byte and puts **Max-Lloyd's 16 reconstruction points for N(0,1)** in place
of the grid, times 32 and rounded so they stay integers. Two format values, because there are two separable levers:

| format | level set | scale `d` per 32-value block |
| --- | --- | --- |
| `--kv q4lm32` | Max-Lloyd table | `max|x| / 87` |
| `--kv q4lm32s` | the same table | `sum(l*x) / sum(l*l)` - one least-squares pass |

The reader is identical for both (the scale is in the block), and the layout, block size, block movers, streaming
granule, snapshot layout and the int8 prompt-MMA operands are `q4_0`'s. Design and the build decision:
[docs/KV_QUANT_WORKPACKAGE.md](../../../docs/KV_QUANT_WORKPACKAGE.md) §5.1-Nachtrag; the measurement: §7.3 and §7.4.

**Measured on:** Windows 11, one gfx1200 Radeon (RX 9060 XT, 16 GB), engine built from source on branch
`KV_QUANT_research` (HIP, ROCm 7.1), Qwen3.8-Flash-Next **Q2_0** pack, MTP on, `--spec 4`, and the fixed set
`data/quant_quality/v2-8k` (docs, code, multilingual; **8,191 positions each, 24,573 in all**). Reference: `--kv fp16`
in the same run. Teacher-forced through the decode windows (`STRATA_LOGPOS`, top-64). The same three texts were also
scored at **999 positions** each (the 1K arms, from `set-1k.ids.json`).

## Commands

```
# model-free: both formats bitwise against a host reference, plus the MSE table
kv_q4_parity                                     # 6 tests; [5/6] q4lm32, [6/6] q4lm32s

# end to end, five arms, one after the other
tools/quant_quality.py run --config <cfg> --kv q4lm32s --ids data/quant_quality/v2-8k/set.ids.json --out kv-q4lm32s.tsv
tools/quant_quality.py run --config <cfg> --kv q4lm32  --ids data/quant_quality/v2-8k/set.ids.json --out kv-q4lm32.tsv
tools/quant_quality.py run --config <cfg> --kv fp16    --ids data/quant_quality/v2-8k/set.ids.json --out kv-fp16.tsv
tools/quant_quality.py run --config <cfg> --kv int8    --ids data/quant_quality/v2-8k/set.ids.json --out kv-int8.tsv
tools/quant_quality.py run --config <cfg> --kv q4_0    --ids data/quant_quality/v2-8k/set.ids.json --out kv-q4_0.tsv
tools/quant_quality.py compare kv-fp16.tsv kv-int8.tsv kv-q4_0.tsv kv-q4lm32.tsv kv-q4lm32s.tsv --segments --json compare.json

# and the 1K pair: the same three texts, first 1000 tokens
tools/quant_quality.py run --config <cfg> --kv <FMT> --ids data/quant_quality/v2-8k/set-1k.ids.json --out kv1k-<FMT>.tsv
tools/quant_quality.py compare kv1k-fp16.tsv kv1k-int8.tsv kv1k-q4_0.tsv kv1k-q4lm32.tsv kv1k-q4lm32s.tsv \
    --segments --json compare-1k.json
```

`<cfg>` is the Q2_0 config with `exe` pointed at the source build; `--kv` and the derived `--max-context`/`--prefill`
args come from the harness.

## Model-free: is the new quantizer actually better?

`src/kernels/kv_q4_parity.cpp` tests 5 and 6 write 487 drawn cells (2 KV heads x 256 values x K and V = 498,688
rotated values, 15,584 blocks), compare **bitwise** against a host reference, gather them back and measure the error
per block. Both formats are bitwise equal, and the gather returns exactly `level * stored scale`:

| Configuration (the same 16 levels unless stated) | MSE | vs `q4_0` |
| --- | ---: | ---: |
| `q4_0` uniform grid, ggml scale | 2.938e-02 | 0 dB |
| `q4lm32` Lloyd levels, `max|x|/87` | 2.740e-02 | **-0.30 dB** |
| the same levels, `RMS/32` scale | 3.547e-02 | -0.82 dB (**worse**) |
| the same levels, converged per-block best fit | 2.456e-02 | -0.78 dB (ceiling) |
| `q4lm32s` the same levels + **one** least-squares pass | 2.551e-02 | **-0.61 dB** |

The table is worth 0.30 dB, the block's own scale another 0.31 dB; one least-squares pass reaches 78.8% of the way to
a converged fit and lands within 0.4% of `max|x|/87` on average. It is **not** a fixed rescale of the levels: with
`RMS/32` the same levels are worse than `q4_0`, because the block maximum of 32 samples sits at `max|x|/RMS = 2.363`
sigma on average, below the 2.72 sigma the widest level is drawn for.

## End to end: 24,573 positions, the same run's fp16 K/V as the reference

| Arm | Top-1 | mean KL | KL p99 | PPL | vs fp16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp16` (reference) | - | 0 | 0 | 3.31026 | - |
| `int8` (default) | 97.04% | 0.006290 | 0.0675 | 3.30669 | -0.11% |
| `q4_0` | 95.27% | 0.016584 | 0.1833 | 3.31781 | **+0.23%** |
| `q4lm32` | 95.46% | 0.015788 | 0.1648 | 3.31147 | **+0.04%** |
| `q4lm32s` | 95.50% | 0.015162 | 0.1603 | 3.31078 | **+0.02%** |

Per segment: PPL difference against the same run's fp16 arm, mean KL in brackets (the fp16 PPL as an absolute value):

| Segment | fp16 PPL | `int8` | `q4_0` | `q4lm32` | `q4lm32s` |
| --- | ---: | ---: | ---: | ---: | ---: |
| docs | 6.79727 | -0.27% (0.0085) | +0.06% (0.0230) | -0.08% (0.0214) | -0.41% (0.0203) |
| code | 2.05395 | -0.16% (0.0049) | +0.57% (0.0139) | +0.70% (0.0140) | +0.74% (0.0137) |
| multilingual | 2.59814 | +0.11% (0.0054) | +0.06% (0.0129) | -0.51% (0.0119) | -0.28% (0.0115) |

- At the **same byte count** as `q4_0` (576 B per cell), `q4lm32s` takes its PPL penalty from **+0.23% to +0.02%** -
  fp16 level - and cuts mean KL by 8.6% and p99 KL by 12.5%.
- The **table alone** (`q4lm32`) already gets most of it: +0.04% PPL, -4.8% mean KL.
- The gain sits in `docs` (-0.41%) and `multilingual` (-0.28%). On `code` every 4-bit format pays 0.6-0.7% PPL, and
  the new formats do not reduce that (they are marginally worse than `q4_0` there in PPL, better in mean KL).
- Top-1 does not separate the formats at this size: the lead of `q4lm32s` over `q4_0` is as large as the run-to-run
  spread quoted below.

## The 1K pair (the first 1000 tokens of the same three texts)

999 positions each, 2,997 in all, fp16 reference 3.74137:

| Arm | Top-1 | mean KL | KL p99 | PPL | vs fp16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `int8` | 96.96% | 0.007032 | 0.0755 | 3.75142 | +0.27% |
| `q4_0` | 95.33% | 0.016052 | 0.2173 | 3.78984 | **+1.30%** |
| `q4lm32` | 95.06% | 0.014846 | 0.1713 | 3.74192 | **+0.02%** |
| `q4lm32s` | 95.13% | 0.015313 | 0.1603 | 3.74178 | **+0.01%** |

The level table closes the `q4_0` gap at 1K too (+1.30% -> +0.02%), and both new formats land at fp16 level. Two
honest notes:

- `q4lm32` **loses** top-1 to `q4_0` at 1K (95.06% vs 95.33%) while winning mean KL (-7.5%) and p99 (-21%), and the
  order flips again at 8K - top-1 is not a stable criterion at this effect size, mean KL is.
- The **second lever is not consistently signed**: at 1K `q4lm32s` has 3.1% **higher** mean KL than `q4lm32` and
  6.4% lower p99, while at 8K both improve (-4.0% KL, -2.7% p99). Its per-block gain is firm (-0.31 dB); its
  end-to-end effect is small.

## What this does not say

- **Single runs.** The committed prebuilt-engine run
  ([2026-10-14-kv-8k](../2026-10-14-kv-8k/README.md)) reads `int8` mean KL 0.006393 / PPL 3.30664 and `q4_0`
  0.016519 / 3.31827; this run reads 0.006290 / 3.30669 and 0.016584 / 3.31781, top-1 97.04/97.04 and 95.35/95.27.
  So mean KL reproduces within ~1.6%, PPL within 0.03%, top-1 within 0.08 pp - but the two runs are **not
  bit-identical** (the fp16 arm reads 3.31026 here against 3.31128 there, +0.03%). Whether that is the binary
  (source build vs prebuilt) or the engine's own run-to-run spread was **not** isolated, so every difference above is
  quoted **within one run**.
- **No identity gate and no speed A/B yet.** The default path is unchanged by construction (`Q4Mode::kUniform` is the
  default of every new parameter, and the CLI selects a mode only for the two new `--kv` values), and the `int8` and
  `q4_0` arms reproduce their committed counterparts within the spread just quoted - but the byte-identical
  four-model identity gate the KV conversations ask for has **not** been run, and neither has a decode speed A/B.
- **No tensor-core prefill.** Both new formats make the prompt path decline, so a prefill runs through the decode
  kernel (`src/kernels/cuda/qsa_prompt_attn.cu`). A deliberate tradeoff while the formats are opt-in.
- **SYCL is mirrored, not built.** This machine has no oneAPI. `sycl/src/` carries the same changes plus a parity twin
  (`sycl/src/kernels/kv_q4_parity.cpp`, 6 tests), and it is **untested here**. The SYCL write side takes the mode as a
  runtime argument instead of a template parameter (documented in the code as a deliberate deviation).
- **`kQ4lMap` sits in `__constant__`.** That is the worst case for a divergently indexed 16-byte table (2.12x
  measured for IQ4_NL's codebook in `s_gemv.cu`), so a shared-memory copy comes before any speed A/B.
- Three texts, one model, one weight quantization (Q2_0), teacher-forced. A different model, a longer context and
  `needle_bench.py` recall are not covered.

## Files

- `compare.json` - the 8K five-arm comparison (`--segments`).
- `compare-1k.json` - the 1K five-arm comparison (`--segments`).
- The raw per-position TSVs (`kv-*.tsv`, 26 MB each) are not committed; `tools/quant_quality.py run` regenerates them.
