# The 3-bit probe: which tensor wants the bits (K, not V)

The WP-2a probe. It asks the one question that decides the shape of a sub-4-bit KV format - **how the bits are split
between K and V** - before any 3-bit byte layout is built. Design: `docs/KV_QUANT_WORKPACKAGE.md` §5.4; build and
model-free numbers: §7.5; the question's origin: `docs/KV_QUANT_RESEARCH.md` §6.11.

- **Where:** one PC, RX 9060 XT (gfx1200, RDNA4, 16 GB), 63.9 GB RAM, HIP build of that day.
- **What ran:** Qwen3.8-Flash-Next, `Q2_0` weight pack, weights fixed, only the KV cache changed.
- **Basis note:** this run is **`Q2_0`**, the *damped* weight regime (its own weight error is the largest in the
  line, +10.4 % PPL against Q8_0). From 2026-10-14 the measurement basis is **Swift IQ3_XXS**; these Q2_0 numbers
  stand as a *lower bound* on the KV gaps and as proof that the probe's wiring self-diagnoses, and the K-vs-V
  decision is repeated on Swift.
- **Test:** RULER CWE at 2K, 8 segments, 2,048 tokens each, `data/quant_quality/cwe-2k/` - the sharper of the two
  decision-test sets (`bench/results/2026-10-14-kv-fwe/`). Reference: the `fp16` arm of the same run.
- **What the probe is:** Max-Lloyd's 8-level ladder for N(0,1) (`{±69, ±43, ±24, ±8}`, `d = max|x|/69`) inside the
  **unchanged** 18-byte `block_q4_0` - same block, same fixed Hadamard rotation, same readers. It is a *quality*
  probe at **576 B/cell**, not a byte layout: no byte is saved here.

| `--kv` | K levels | V levels | bits, avg over K/V |
| --- | ---: | ---: | ---: |
| `q4lm32` | 16 | 16 | 4.0 (reference) |
| `q4lm8` | 8 | 8 | 3.0 |
| `q4lm8k` | 8 | 16 | 3.5 |
| `q4lm8v` | 16 | 8 | 3.5 |

`q4lm8k` and `q4lm8v` are byte-exact equals, so their pair answers PR #2318's `K2c+V4 > K4+V2c` against
`arXiv:2502.15075`'s "K4V2 keeps 98.3 % of K4V4" on our own model.

## The run reproduces its own anchors

`q4lm32` and `int8` were re-run as anchors, and both land on the numbers from the earlier cwe-2k run
(`bench/results/2026-10-14-kv-fwe/`):

| arm | this run | the earlier run |
| --- | ---: | ---: |
| `q4lm32` mean KL | 0.02022 | 0.02022 |
| `int8` mean KL | 0.01033 | 0.01033 |
| `fp16` reference PPL | 3.06695 | 3.0669 |

So the probe's table wiring is confirmed end to end, and cwe-2k reproduces across sessions to 4-5 digits (unlike the
8K whole-model set, which differed by 0.03 % between runs).

## Whole segment (16,351 positions)

| arm | top-1 | mean KL | KL / `q4lm32` | KL p99 | PPL | vs fp16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp16` | - | 0 | - | 0 | 3.06695 | - |
| `int8` | 99.16% | 0.01033 | 0.51x | 0.1767 | 3.0704 | +0.11% |
| `q4lm32` | 98.62% | 0.02022 | 1.00x | 0.3458 | 3.0933 | +0.86% |
| **`q4lm8v`** (K16/V8) | **98.52%** | **0.02619** | **1.30x** | 0.4949 | 3.0782 | **+0.37%** |
| `q4lm8k` (K8/V16) | 98.48% | 0.02748 | 1.36x | 0.4310 | 3.0760 | +0.30% |
| `q4lm8` (K8/V8) | 98.23% | 0.03328 | 1.65x | 0.5333 | 3.0918 | +0.81% |

**PPL cannot see any of this.** `q4lm8` reads a *better* PPL than `q4lm32` (+0.81% against +0.86%) while its mean KL
is 65 % higher, its p99 54 % higher and its top-1 0.39 points lower. On a `Q2_0` weight model the weights' own error
floor dominates a likelihood metric; KL and the retrieval metrics below are what carry the signal. (This is the
strongest instance of the `bench/results/2026-10-14-kv-fwe/` finding that PPL and answer accuracy are blunt
instruments here.)

## Paired whole-segment KL (the primary readout)

`quant_quality.py compare ... --paired`, 8 segments, "up" = segments where the left arm is worse:

| A - B | mean | sd | sem | t | segments up |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q4lm32` - `q4lm8` | -0.01306 | 0.00733 | 0.00259 | **-5.04** | 0/8 |
| `q4lm32` - `q4lm8k` | -0.00726 | 0.00632 | 0.00223 | **-3.25** | 0/8 |
| `q4lm32` - `q4lm8v` | -0.00597 | 0.00706 | 0.00250 | -2.39 | 1/8 |
| `q4lm8k` - `q4lm8v` | +0.00129 | 0.00603 | 0.00213 | +0.60 | 4/8 |
| `int8` - `q4lm8v` | -0.01586 | 0.00763 | 0.00270 | -5.88 | 0/8 |

The third row is the one that decides the K-vs-V question, and by itself it **does not resolve**: t +0.60 with 4 of 8
segments going the other way. Cutting *either* tensor from 16 levels to 8 costs measurably (both rows are 0/8), and
cutting K costs ~1.2x as much in the point estimate (0.00726 against 0.00597) - but the difference between those two
costs is inside the noise at 8 segments.

## The answer region: the split is along K, and it is categorical

The task metric is not null here, and it separates by *K precision alone*. 266 answer rows:

| arm | K levels | answer top-1 | vs fp16 | top-1 agrees with fp16 | answer KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp16` | 16 | 90.60% | - | 100.00% | 0 |
| `int8` | 16 | 90.60% | - | 100.00% | 0.00297 |
| `q4lm32` | 16 | **90.98%** | +0.38 pp | 98.50% | 0.00931 |
| **`q4lm8v`** | **16** | **90.98%** | **+0.38 pp** | **98.50%** | **0.01251** |
| `q4lm8k` | 8 | 89.10% | -1.50 pp | 97.74% | 0.01657 |
| `q4lm8` | 8 | 89.10% | -1.50 pp | 96.24% | 0.02808 |

Every arm with 16-level K scores 90.98%; every arm with 8-level K scores 89.10%, whatever V does. Exact McNemar
(`areport --paired`, same rows):

| A vs B | rows only A right : only B right | p (two-sided) |
| --- | ---: | ---: |
| `q4lm32` vs `q4lm8v` (V 16 -> 8) | 2 : 2 | **1.000** |
| `q4lm32` vs `q4lm8k` (K 16 -> 8) | 0 : 5 | 0.062 |
| `q4lm8k` vs `q4lm8v` | **0 : 5** | 0.062 |

**Cutting V from 16 to 8 levels leaves the task untouched** - 2:2 discordant rows, p 1.000, identical 90.98%, and its
answer-region KL is only 0.01251 against `q4lm32`'s 0.00931. **Cutting K from 16 to 8 costs the task** - all 5
discordant rows go one way, p 0.062 two-sided (0.031 one-sided), and the answer KL triples.

## Findings

1. **The direction is K wants the bits, not V.** At an equal average width (3.5 bits) the K-heavy arm keeps the
   retrieval metrics and the V-heavy arm loses them. All three readouts agree in sign: paired whole-segment KL
   (t +0.60), paired answer-region KL (t +1.53), and answer accuracy (5:0 discordant, p 0.062). No single readout
   crosses the conventional threshold at 8 segments, so this is a **direction, not an established effect** - but it is
   the opposite direction from PR #2318's headline, and it agrees with `arXiv:2502.15075` and with the already-shipped
   `k8v4` (#120, K 8-bit / V 4-bit, §1 of the survey) whose allocation it validates.
2. **3 bits on both tensors at once is too much; 3.5 bits split is close.** `q4lm8` costs +0.01306 KL against
   `q4lm32` (t -5.04, 0/8), top-1 -0.39 pp and the answer region 89.10%. The split arms cost +0.0060 / +0.0073 KL,
   top-1 within 0.14 pp, and `q4lm8v` keeps the answer region at `q4lm32`'s level exactly. So the viable shape is
   **asymmetric**, which is what finding 1 predicts.
3. **The state of the art's 3-bit warning is real here too.** WUSH-KV's fixed-Hadamard arm is 10.06 at 3 bits against
   9.82 calibrated on Qwen3-8B (§6.9); on our model an even symmetric 3-bit budget costs 1.65x the mean KL and 0.39 pp
   of top-1 against 4 bits. A symmetric 3-bit Strata format is not on the table; an asymmetric one is what the data
   points at.
4. **The quantizer cannot answer this, and it was worth checking.** The model-free parity test put `q4lm8k` and
   `q4lm8v` 0.1 % apart (7.672e-02 against 7.662e-02 MSE, §7.5) because both tensors supply half the values and use
   the same table - so any MSE-based design would have been blind to a real task-level asymmetry. The exchange rate
   between a K error and a V error is a property of the model.

## What this does not say

- **No bytes are saved.** Every arm is 576 B/cell (fp16 2048, int8 1056). This is a quality probe; the byte layout is
  WP-2.
- **Only `Q2_0` weights, only 2K context, one model.** The `Q2_0` weight error damps KV penalties (§6.7), so these
  gaps are a lower bound on what a cleaner weight quant would show.
- **No speed A/B.** The probe's readers select a per-tensor table at runtime; that is fine for a quality probe and is
  not a speed result.
- **Not a calibrated-transform result.** This is the fixed Hadamard the format ships with; WUSH-KV's lever is a
  different (opt-in) axis and is not measured here.

## Files

- `compare.json` - per-segment and aggregate top-1 / KL / p99 / PPL for all six arms.
- `areport.json` - the answer region (positions, hits, agreement, KL) per arm.
- `paired.json` - the paired whole-segment and answer-region tables.
- The per-position TSVs are not committed; the set is in `data/quant_quality/cwe-2k/`.

## Reproduce

```
python tools/quant_quality.py run  --config CFG --kv q4lm8v --ids data/quant_quality/cwe-2k/set.ids.json --out /tmp/cwe-q4lm8v.tsv
python tools/quant_quality.py run  --config CFG --kv fp16   --ids data/quant_quality/cwe-2k/set.ids.json --out /tmp/cwe-fp16.tsv
python tools/quant_quality.py compare /tmp/cwe-fp16.tsv /tmp/cwe-q4lm8v.tsv --segments --paired \
       --ids data/quant_quality/cwe-2k/set.ids.json --json compare.json --paired-json paired.json
python tools/quant_quality.py areport /tmp/cwe-fp16.tsv /tmp/cwe-q4lm8v.tsv \
       --ids data/quant_quality/cwe-2k/set.ids.json --paired --json areport.json
```

Six arms at ~60 s/segment on gfx1200 (`q4lm32` 59.7 s, `q4lm8` 58.4 s, `q4lm8k` 57.2 s, `q4lm8v` 55.6 s per segment
on average; the `int8` arm's first four segments ran slow at 99-119 s under concurrent CPU load and its last four at
57 s, which is a timing artefact and not a result).
