# WP-1 auf Swift: asymmetrisch-4-Bit im Direktvergleich (cwe-2k)

Design: `docs/KV_QUANT_WORKPACKAGE.md` §5.5; Arme gebaut in §7.8. Drei Arme, alle in Sonden-Bytes
(576 B/Zelle), alle gegen `q4_0`/`q4lm32`/`int8`/fp16.

- **Where:** one PC, RX 9060 XT (gfx1200, RDNA4, 16 GB), 63.9 GB RAM, HIP build of 2026-10-16
  (commit `6fc7b2f1`, kvstats parked/env-off).
- **What ran:** Qwen3.8-Flash-Next, **Swift IQ3_XXS** weight pack, weights fixed, only `--kv` changed.
- **Test:** RULER CWE at 2K, 8 segments, ~2,048 tokens each, `data/quant_quality/cwe-2k/`.
  Reference: the `fp16` arm of the same run (PPL 2.999). Segment times stable at ~64 s.
- **Incident:** the first `q4lm32nr` attempt died after s02 (AMD driver crash, engine vanished
  without a log line); the rerun passed clean through the same region - a one-off, no pattern.

## Whole segment (16,351 positions)

| arm | top-1 | mean KL | KL / `q4lm32` | KL p99 | PPL | vs fp16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp16` | - | 0 | - | 0 | 2.999 | - |
| `int8` | 99.35% | 0.00503 | 0.43x | 0.0733 | 3.004 | +0.17% |
| `q4_0` | 99.18% | 0.01096 | 0.93x | 0.1647 | 3.013 | +0.47% |
| `q4lm32` | 99.19% | 0.01183 | 1.00x | 0.1464 | 3.012 | +0.43% |
| `q4lm32t` (K max/V rms) | 99.08% | 0.01107 | 0.94x | 0.1343 | 2.996 | −0.10% |
| `q4lm8v` (K16/V8) | 98.92% | 0.01791 | 1.51x | 0.2793 | 3.024 | +0.83% |
| `q4lm32nr` (no Hadamard) | 99.14% | 0.01158 | 0.98x | 0.1772 | 2.998 | −0.03% |

Anchors reproduce exactly (`int8` 0.00503 t −13.34, `q4lm32` 0.01183) - the instrument is sharp.

## Paired over the 8 segments (the decision readout)

| pair | ΔKL | t | segments up |
| --- | ---: | ---: | ---: |
| `q4lm32` − `q4lm32t` | +0.00076 | +0.65 | 6/8 |
| `q4lm32` − `q4lm32nr` | +0.00025 | +0.21 | 5/8 |
| `q4lm32` − `q4lm8v` | −0.00607 | −3.60 | 0/8 |
| `q4_0` − `q4lm32t` | −0.00010 | −0.08 | 6/8 |
| `q4lm32t` − `q4lm8v` | −0.00684 | −2.97 | 1/8 |

McNemar answer accuracy (266 rows, ~91-92% everywhere): all p ≥ 0.25 - resolves nothing at 8 segments.

## Verdict at cwe-2k power

- **Bit (`q4lm8v`) loses decisively:** +51 % KL vs `q4lm32`, t −3.60, 0/8 segments. K16/V8 at 3.5 avg bits
  is not free on Swift either - the 3-bit hope ends here; no cwe-4k for this arm.
- **Treatment (`q4lm32t`) ties `q4lm32`/`q4_0`** (t +0.65 whole-segment, but −2.52 in the answer region -
  inconsistent signs, i.e. noise). V-RMS costs nothing measurable and buys nothing measurable at 8 segments.
- **No-rotation (`q4lm32nr`) ties `q4lm32`** (t +0.21, 5/8). The Hadamard buys nothing measurable at 4-bit
  Lloyd on Swift - against OSCAR's INT2 ablation (+18 pts H), which is a different regime, not a contradiction.
- **Power:** per the power rule (§8) the two ties go to `cwe-4k` (`fp16`/`q4lm32`/`q4lm32t`/`q4lm32nr`).
