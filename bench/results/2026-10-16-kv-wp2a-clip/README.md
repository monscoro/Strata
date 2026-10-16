# WP-2a Clip auf Swift: Tails sind Signal (cwe-2k)

Design: §5.6.1; Arm gebaut in §7.10 (`--kv q4lm32c`, κ per `STRATA_KV_CLIP_K/_V`).
Anker aus `../2026-10-16-kv-wp1-swift/` wiederverwendet (gleiche Engine, gleicher Satz).

- **Where:** one PC, RX 9060 XT (gfx1200, RDNA4, 16 GB), 63.9 GB RAM, HIP build of 2026-10-16
  (commit `4046f71c`, kvstats parked/env-off).
- **What ran:** Qwen3.8-Flash-Next, **Swift IQ3_XXS** weight pack, weights fixed, only `--kv` + κ changed.
- **Test:** RULER CWE at 2K, 8 segments, ~2,048 tokens each, `data/quant_quality/cwe-2k/`.
  Reference: the `fp16` arm of the WP-1 run (PPL 2.999).

## Whole segment (16,351 positions)

| arm | top-1 | mean KL | KL / `q4lm32` | KL p99 | PPL | vs fp16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q4lm32` | 99.19% | 0.01183 | 1.00x | 0.1464 | 3.012 | +0.43% |
| `clip(1.0,1.0)` | 99.19% | 0.01183 | 1.00x | 0.1464 | 3.012 | +0.43% |
| `clip(0.94,0.94)` | 98.88% | 0.01580 | 1.34x | 0.2283 | 3.043 | +1.47% |
| `clip(0.96,0.92)` OSCAR | 98.75% | 0.02491 | 2.11x | 0.3639 | 3.075 | +2.53% |

## Paired over the 8 segments (vs `q4lm32`, whole-segment)

| pair | ΔKL | t | up |
| --- | ---: | ---: | ---: |
| `q4lm32` − `clip(1,1)` | +0.00000 | +nan (sd 0) | 0/8 |
| `q4lm32` − `clip(0.94,0.94)` | −0.00397 | −2.29 | 1/8 |
| `q4lm32` − `clip(0.96,0.92)` | −0.01307 | −6.66 | 0/8 |
| `clip(0.94,0.94)` − `clip(0.96,0.92)` | −0.00910 | −5.16 | 0/8 |

McNemar: all p = 1.000.

## Verdict

- **Identity triple-proven:** parity-memcmp, smoke tokens, and now model KL bit-equal (0.01183, sd 0.00000).
  The env-parametrization machinery is sound.
- **Clipping loses, decisively:** (0.94,0.94) +34 % KL (t −2.29), OSCAR's (0.96,0.92) +111 % (t −6.66).
  Post-FWHT values are tail-light (max/32 ≈ 2.36σ) - clipping discards real mass instead of taming
  outliers. OSCAR's wisdom is for unrotated K channels, not our geometry.
- **V is the fragile side again:** the asymmetric pair loses MORE than symmetric 0.94 (t −5.16) - the
  harder-clipped V (0.92) costs most. Consistent with §7.5's K-sensitivity (V needs care).
- **Consequence for WP-2:** no clipping of post-rotation values in the layout. Clipping only ever
  makes sense pre-rotation per-channel - a pipeline change, explicitly deferred (§5.5).
- **Consequence for the window arm:** tails are signal the model bills - exact tails *may* buy back
  quality. The window gets built (§5.6.2).
