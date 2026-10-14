# The 3-bit probe on Swift: the K-vs-V direction does not survive, the instrument gets sharper

The WP-2a probe re-run on the measurement basis (`bench/results/2026-10-14-kv-3bit-probe/` is Q2_0).
Design: `docs/KV_QUANT_WORKPACKAGE.md` §5.4; Q2_0 run: §7.5 and the sibling README.

- **Where:** one PC, RX 9060 XT (gfx1200, RDNA4, 16 GB), 63.9 GB RAM, HIP build of that day.
- **What ran:** Qwen3.8-Flash-Next, **Swift IQ3_XXS** weight pack, weights fixed, only the KV cache changed.
- **Basis note:** Swift is the measurement basis (workpackage §8); Q2_0 numbers are the damped lower bound.
  Swift segment times are stable at 62-68 s with no contention artefacts (all six arms uniform).
- **Test:** RULER CWE at 2K, 8 segments, 2,048 tokens each, `data/quant_quality/cwe-2k/`. Reference: the `fp16`
  arm of the same run (PPL 2.999).
- **What the probe is:** Max-Lloyd's 8-level ladder in the unchanged 18-byte `block_q4_0` - a *quality* probe at
  **576 B/cell**, not a byte layout. `q4lm8` (8/8, 3.0 bits), `q4lm8k` (8/16, 3.5), `q4lm8v` (16/8, 3.5),
  `q4lm32` (16/16, 4.0 reference).

## Whole segment (16,351 positions)

| arm | top-1 | mean KL | KL / `q4lm32` | KL p99 | PPL | vs fp16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp16` | - | 0 | - | 0 | 2.999 | - |
| `int8` | 99.35% | 0.00503 | 0.43x | 0.0733 | 3.004 | +0.17% |
| `q4lm32` | 99.19% | 0.01183 | 1.00x | 0.1464 | 3.012 | +0.43% |
| `q4lm8k` (K8/V16) | 99.03% | 0.01802 | 1.52x | 0.2246 | 3.003 | **+0.13%** |
| `q4lm8v` (K16/V8) | 98.92% | 0.01791 | 1.51x | 0.2793 | 3.024 | +0.83% |
| `q4lm8` (K8/V8) | 98.81% | 0.02238 | 1.89x | 0.2926 | 3.015 | +0.53% |

**PPL is blind, fourth exhibit.** `q4lm8k` reads the *best* PPL of all 4/3-bit arms (+0.13%, better than `q4lm32`'s
+0.43%) at 1.52x its KL. Likelihood on an IQ3_XXS weight floor cannot rank KV formats; KL can.

## Paired whole-segment KL (8 segments, "up" = left arm worse)

| A - B | mean | sd | sem | t | segments up |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 - int8 | -0.00503 | 0.00107 | 0.00038 | **-13.34** | 0/8 |
| fp16 - q4lm32 | -0.01183 | 0.00440 | 0.00156 | -7.60 | 0/8 |
| int8 - q4lm32 | -0.00680 | 0.00397 | 0.00140 | -4.84 | 0/8 |
| q4lm32 - q4lm8 | -0.01054 | 0.00550 | 0.00194 | **-5.43** | 0/8 |
| q4lm32 - q4lm8k | -0.00618 | 0.00684 | 0.00242 | -2.56 | 1/8 |
| q4lm32 - q4lm8v | -0.00607 | 0.00478 | 0.00169 | -3.60 | 0/8 |
| q4lm8 - q4lm8k | +0.00436 | 0.00518 | 0.00183 | +2.38 | 7/8 |
| q4lm8 - q4lm8v | +0.00447 | 0.00650 | 0.00230 | +1.94 | 7/8 |
| **q4lm8k - q4lm8v** | **+0.00011** | 0.00576 | 0.00204 | **+0.05** | **3/8** |

Paired answer-region KL for the direct contrast: +0.00223, t +0.74, 4/8 - likewise unresolved.

## The answer region: saturated null on Swift (266 rows)

| arm | answer top-1 | vs fp16 | agrees with fp16 | answer KL |
| --- | ---: | ---: | ---: | ---: |
| `fp16` | 91.73% | - | 100.00% | 0 |
| `int8` | 91.73% | - | 99.25% | 0.00299 |
| `q4lm32` | 92.11% | +0.38 pp | 99.25% | 0.00652 |
| `q4lm8` | 92.11% | +0.38 pp | 97.74% | 0.02100 |
| `q4lm8k` | 91.73% | - | 98.12% | 0.01547 |
| `q4lm8v` | 92.11% | +0.38 pp | 98.50% | 0.01299 |

Exact McNemar (`areport --paired`): **every pair p = 1.000** (largest discordance 3:2). No K-split, no split at
all - every quantized arm scores at or above fp16. The Q2_0 categorical split (every K16 arm 90.98%, every K8
arm 89.10%, 0:5 discordant) does not exist here. The answer-accuracy instrument is regime-dependent, and on the
cleaner weight regime it is saturated - the hardening rationale for the needle gate (workpackage §8.8).

## Findings

1. **The instrument gets much sharper on Swift.** fp16-vs-int8 pairs to t -13.34 (sem 7.6% of the effect)
   against Q2_0's t -4.43 on the same contrast. Segment noise collapses when the weight floor drops: sd 0.00107
   against Q2_0's 0.00658. Swift as the basis is vindicated as an instrument decision, independent of any
   format result.
2. **Every boundary resolves except K-vs-V.** int8/4-bit t -4.84, 4-bit/symmetric-3-bit t -5.43 (0/8),
   4-bit/split-3.5-bit t -2.56/-3.60. Symmetric 3-bit is decisively too expensive on Swift too (top-1 -0.38 pp),
   more crisply than on Q2_0.
3. **The K-vs-V direction does not survive the regime change.** Direct byte-exact contrast +0.00011, t +0.05,
   3/8 segments; answer-region t +0.74, 4/8; McNemar all p = 1.000. Q2_0 read t +0.60 with 0:5 discordant
   (p = 0.062) and a categorical accuracy split. Per the workpackage §9 falsification rule, the WP-2
   asymmetric-layout shape cannot rest on the Q2_0 direction anymore. It is consistent with FlatQuant Table 11,
   which is tied at 7 total bits (K4V3 vs K3V4 nearly tied) and separates only at 6 (K4V2 vs K2V4 decisively):
   at 7 total bits the split does not matter on either model; whether it matters at 6 on ours is an open probe
   question, not measured here. Treatment asymmetry (scale rules, WP-1 arm 1) is untouched by this null - a
   different mechanism, still unmeasured.
4. **Split 3.5-bit costs ~+52% KL on Swift** (vs +30-36% on Q2_0) - less damping, bigger gaps, as predicted.
   The split arms sit strictly between 4-bit and symmetric 3-bit on every readout (whole KL t +2.38/+1.94 for
   split-vs-symmetric with 7/8 sign consistency).

## Files

- `compare.json` - per-segment and aggregate top-1 / KL / p99 / PPL for all six arms.
- `areport.json` - the answer region per arm.
- `paired.json` - the paired whole-segment and answer-region tables.
- The per-position TSVs are not committed; the set is in `data/quant_quality/cwe-2k/`.

## Reproduce

```sh
python tools/quant_quality.py run  --config CFG-swift --kv q4lm8v --ids data/quant_quality/cwe-2k/set.ids.json --out /tmp/sw-q4lm8v.tsv
python tools/quant_quality.py run  --config CFG-swift --kv fp16   --ids data/quant_quality/cwe-2k/set.ids.json --out /tmp/sw-fp16.tsv
python tools/quant_quality.py compare /tmp/sw-fp16.tsv /tmp/sw-q4lm8v.tsv --segments --paired \
       --ids data/quant_quality/cwe-2k/set.ids.json --json compare.json --paired-json paired.json
python tools/quant_quality.py areport /tmp/sw-fp16.tsv /tmp/sw-q4lm8v.tsv \
       --ids data/quant_quality/cwe-2k/set.ids.json --paired --json areport.json
```

Six arms at 62-68 s/segment on gfx1200, uniform across arms (no contention artefacts this run).
