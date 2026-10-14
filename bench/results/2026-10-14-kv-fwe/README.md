# The KV decision test: RULER FWE/CWE, 2K

The instrument that decides whether a KV-cache format change is real, and the first readout of it
(`data/quant_quality/fwe-2k/`, `data/quant_quality/cwe-2k/`).

- **Where:** one PC, RX 9060 XT (gfx1200, RDNA4, 16 GB), 63.9 GB RAM, HIP build of that day.
- **What ran:** Qwen3.8-Flash-Next, the `Q2_0` weight pack, weights fixed, only the KV cache changed.
- **Reference:** the `fp16` arm of the same run, not `Q8_0`.
- **Set:** 8 segments, 2,048 tokens each, RULER FWE (frequent-words extraction) and CWE (common-words
  extraction) with a 40-word vocabulary of 6-character coded words, `alpha` 2.0, seed 42, 10 answer
  words. Teacher-forced: every segment is read once per arm and the arm's logits are scored against the
  reference arm's, so no sampling and no answer-parsing sits between the cache and the number.

## Why this test and not perplexity

A WikiText-style PPL or a mean KL over all positions is dominated by the *easy* positions (function
words, repeated context) and averages over everything the cache touches. RULER FWE/CWE asks the model
to *retrieve* the ten most frequent words out of a long context - the part of a long context that a
lossy KV cache destroys first - so a format that is fine everywhere else and bad at retrieval shows up
here. The two arms are compared on the identical text; see "paired" below.

## Byte table (what "half the bytes" means here)

Per KV cell (all 48 layers of one token, 2 KV heads x 256 head dim, 4 QSA blocks):

| format | bytes/cell | whole model, bytes/token |
| --- | ---: | ---: |
| `fp16` | 2048 | 24,576 |
| `int8` | 1056 | 12,672 |
| `q4_0`, `q4lm32`, `q4lm32s` | 576 | 6,912 |

`int8` is 2 bytes/head-dim of scale overhead over its 1 byte of data; the 4-bit formats carry a fp16
scale per 32 rotated values.

## fwe-2k and cwe-2k, whole segment

fp16 reference: `fwe-2k` PPL 3.014973 over 16,355 positions; `cwe-2k` PPL 3.0669 over 16,351.

| | top-1 | mean KL | KL p99 | PPL | vs fp16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **fwe-2k** `int8` | 99.32% | 0.00518 | 0.0623 | 3.013 | -0.06% |
| **fwe-2k** `q4_0` | 99.16% | 0.01100 | 0.1281 | 3.017 | +0.06% |
| **cwe-2k** `int8` | 99.16% | 0.01033 | 0.1767 | 3.070 | +0.11% |
| **cwe-2k** `q4_0` | 98.94% | 0.01856 | 0.2815 | 3.071 | +0.13% |
| **cwe-2k** `q4lm32` | 98.62% | 0.02022 | 0.3458 | 3.093 | +0.86% |
| **cwe-2k** `q4lm32s` | 98.82% | 0.02036 | 0.3285 | 3.105 | +1.24% |

## The answer region, and why it is not the primary readout

The pre-registered FWE readout was the KL over the positions that produce the answer - the part of the
sequence the task is actually about. It does not work, and the failure is bigger at `fwe-2k` than at
`cwe-2k`:

| set | arm | answer positions | hit | agreement with the text | KL |
| --- | --- | ---: | ---: | ---: | ---: |
| fwe-2k | fp16 | 84 | 68 (80.95%) | 84 | 0 |
| fwe-2k | int8 | 84 | 68 (80.95%) | 84 | 0.5465* |
| fwe-2k | q4_0 | 84 | 68 (80.95%) | 83 | 0.6247* |
| cwe-2k | fp16 | 266 | 241 (90.60%) | 266 | 0 |
| cwe-2k | int8 | 266 | 241 (90.60%) | 266 | 0.7897* |
| cwe-2k | q4_0 | 266 | **243** (91.35%) | 264 | 2.1055* |
| cwe-2k | q4lm32 | 266 | 242 (90.98%) | 262 | 2.4755* |
| cwe-2k | q4lm32s | 266 | **243** (91.35%) | 263 | 2.3879* |

\* the `areport` KL is the sum over positions, so divide by the position count for a mean (int8 fwe-2k
0.0065, q4_0 fwe-2k 0.0074). Divide-to-a-mean and mean-of-per-segment-means are not the same number
when segments have different answer-region lengths; the paired table below uses the per-segment mean,
which is the one that pairs correctly.

Two things to read off it:

- **The hit rate is null and slightly anti-correlated.** At `fwe-2k` all three arms hit the same 68 of
  84; at `cwe-2k` the two 4-bit arms hit **more** than fp16 (243 vs 241). Quantizing the cache does not
  make this task harder at 4 bits, so a task-accuracy readout cannot see the format at all.
- **`fwe-2k`'s answer region is too small to pair.** 84 positions over 8 segments is ~10 per segment;
  the answer-region paired difference between int8 and q4_0 is t -0.69 with 6 of 8 segments "up" - a
  coin flip. `cwe-2k`'s answer region is 266 positions and does resolve it (t -3.09, 0/8 up), but it is
  still the weaker of the two readouts (below).

So the readout is the **whole-segment KL, paired over segments**.

## Paired differences (the readout)

Every arm reads the identical segment, so most of the spread of a single arm is *which text the segment
holds*; the paired difference `KL_a(s) - KL_b(s)` cancels it. `quant_quality.py compare ... --paired`
prints it and writes `paired-*.json` here.

Whole-segment KL, `cwe-2k` (8 segments; "up" = segments where the left arm is worse):

| A - B | mean | sd | sem | t | segments up |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 - int8 | -0.01032 | 0.00658 | 0.00233 | **-4.43** | 0/8 |
| fp16 - q4_0 | -0.01855 | 0.00973 | 0.00344 | **-5.39** | 0/8 |
| fp16 - q4lm32 | -0.02021 | 0.01243 | 0.00439 | **-4.60** | 0/8 |
| fp16 - q4lm32s | -0.02035 | 0.01160 | 0.00410 | **-4.96** | 0/8 |
| int8 - q4_0 | -0.00823 | 0.00412 | 0.00146 | **-5.66** | 0/8 |
| int8 - q4lm32 | -0.00989 | 0.00668 | 0.00236 | **-4.19** | 0/8 |
| int8 - q4lm32s | -0.01003 | 0.00602 | 0.00213 | **-4.71** | 0/8 |
| q4_0 - q4lm32 | -0.00166 | 0.00515 | 0.00182 | -0.91 | 2/8 |
| q4_0 - q4lm32s | -0.00180 | 0.00376 | 0.00133 | -1.35 | 2/8 |
| q4lm32 - q4lm32s | -0.00014 | 0.00275 | 0.00097 | -0.14 | 5/8 |

Whole-segment KL, `fwe-2k`, for the instrument comparison:

| A - B | mean | sd | sem | t | sem / \|mean\| | segments up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 - int8 | -0.00518 | 0.00421 | 0.00149 | -3.48 | 28.8% | 0/8 |
| fp16 - q4_0 | -0.01100 | 0.00609 | 0.00215 | -5.11 | 19.5% | 0/8 |
| int8 - q4_0 | -0.00582 | 0.00471 | 0.00167 | **-3.50** | **28.7%** | 0/8 |

## Findings

1. **`cwe-2k` is the sharper instrument than `fwe-2k` for the same contrast.** On the int8-vs-q4_0
   difference, `cwe-2k` gives t -5.66 (sem 17.7% of the effect) against `fwe-2k`'s t -3.50 (sem 28.7%).
   The second ten-answer-word set, `cwe-2k`, is the one to use; `fwe-2k` stays as the default-path
   reproduction of `fwe`.
2. **The pre-registered answer-region readout measured false at `fwe-2k`** (t -0.69, no sign
   consistency over 8 segments) even on the largest difference in the run. The whole-segment paired KL
   is what replaces it, and it is sign-consistent in every arm pair that matters (0/8).
3. **At 4 bits the format differences are not resolvable.** `q4_0`, `q4lm32` and `q4lm32s` sit within
   t ≤ 1.35 of one another at `cwe-2k`, with 2-5 of 8 segments going the *other* way. The codebook and
   the self-fit scale are real at the quantizer (0.30 dB and 0.61 dB MSE, `kv_q4_parity`), but on this
   model at this width they are below the noise floor of the test. Any claim that one 4-bit map beats
   another has to say which set and how many segments, and this set does not support one.
4. **The int8-to-4-bit boundary is real.** Every int8-vs-4-bit pair is 0/8 with |t| ≥ 4.19. That is the
   honest form of the half-the-bytes claim: the bytes halve and the cost is measurable, `+0.13%` PPL for
   `q4_0` at `cwe-2k` (`+0.23%` on the 8K whole-model set, `bench/results/2026-10-14-kv-q4lm32/`).
5. **The second lever did not replicate.** `q4lm32s` beats `q4lm32` on the 8K set (+0.02% vs +0.04%
   PPL) and loses at `cwe-2k` (+1.24% vs +0.86%). With the difference itself inside the noise, the
   honest statement is that the one-pass least-squares scale is not established either way.

## What this does not say

- It does not test weights: only the KV cache changes between arms.
- It does not test long context: 2K here. `cwe-4k` and `cwe-4k-flat` are built
  (`data/quant_quality/`) for that and cost ~2.3x per segment; they are not run in this result.
- It does not rank formats by speed: `int8` and the 4-bit formats differ in the reader as well as the
  bytes, and no speed A/B is in this README.
- The 4-bit/`int8` gaps are quoted within one run. Two 8K runs of the same build differed by 0.03% in
  fp16 PPL, so cross-run single numbers are not compared; every delta above is same-run.

## Files

- `compare-cwe-2k.json`, `areport-cwe-2k.json`, `paired-cwe-2k.json` - the cwe-2k run
  (fp16, int8, q4_0, q4lm32, q4lm32s).
- `compare-fwe-2k.json`, `areport-fwe-2k.json`, `paired-fwe-2k.json` - the fwe-2k run
  (fp16, int8, q4_0).
- The raw per-position TSVs are not committed; `set.ids.json`, `gold.json` and the segment texts are
  in `data/quant_quality/{fwe,cwe}-2k/`.

## Reproduce

```
python tools/quant_quality.py --help              # ids / fwe / run / compare / areport
python tools/quant_quality.py run  --config CFG --kv int8 --ids data/quant_quality/cwe-2k/set.ids.json --out /tmp/cwe-int8.tsv
python tools/quant_quality.py run  --config CFG --kv fp16 --ids data/quant_quality/cwe-2k/set.ids.json --out /tmp/cwe-fp16.tsv
python tools/quant_quality.py compare /tmp/cwe-fp16.tsv /tmp/cwe-int8.tsv --segments --paired \
       --ids data/quant_quality/cwe-2k/set.ids.json --json compare.json --paired-json paired.json
python tools/quant_quality.py areport /tmp/cwe-fp16.tsv /tmp/cwe-int8.tsv \
       --ids data/quant_quality/cwe-2k/set.ids.json --json areport.json
python tools/quant_gold_check.py data/quant_quality/cwe-2k            # gold == realized
```

Gold validation (every segment, both sets, `gold_is_the_realized_order: True`): `fwe-2k` and `cwe-2k`
rank-10 count 5 against rank-11 4, first tie rank 13 - except `cwe-2k` s06, where rank 10 = rank 11
(4 vs 4), so one of its ten answer slots is ambiguous by construction. `cwe-4k` and `cwe-4k-flat` were
built to move that tie out (rank 14-17 and 16-19).
