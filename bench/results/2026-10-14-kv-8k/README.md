# 2026-10-14: the KV cache at 8K, and what "context length" does to its error

The work package ([docs/KV_QUANT_WORKPACKAGE.md](../../../docs/KV_QUANT_WORKPACKAGE.md)) asks the KV format's error
question at 8K because a 1K set cannot: **does a format's error grow as more context precedes a token?** This run
answers it, and the answer changes how the rest of the work package should be read.

**Measured on:** Windows 11, one gfx1200 Radeon (RX 9060 XT, 16 GB), prebuilt engine 0.1.44.1 (HIP), Qwen3.8-Flash-Next
**Q2_0** pack, MTP on, `--spec 4`, and the fixed set `data/quant_quality/v2-8k` (docs, code, multilingual; **8,191
positions each, 24,573 in all**). Reference: `--kv fp16`. Teacher-forced through the decode windows
(`STRATA_LOGPOS`, top-64). The same three texts were also scored at **999 positions** each (the 1K arms, from
`set-1k.ids.json`).

## Commands

```
tools/quant_quality.py run --config strata-q2_0.json --kv fp16 --ids data/quant_quality/v2-8k/set.ids.json    --out kv-fp16.tsv
tools/quant_quality.py run --config strata-q2_0.json --kv int8 --ids data/quant_quality/v2-8k/set.ids.json    --out kv-int8.tsv
tools/quant_quality.py run --config strata-q2_0.json --kv q4_0 --ids data/quant_quality/v2-8k/set.ids.json    --out kv-q4_0.tsv
tools/quant_quality.py compare kv-fp16.tsv kv-int8.tsv kv-q4_0.tsv --segments --json compare.json

tools/quant_quality.py run --config strata-q2_0.json --kv fp16 --ids data/quant_quality/v2-8k/set-1k.ids.json --out kv1k-fp16.tsv
tools/quant_quality.py run --config strata-q2_0.json --kv int8 --ids data/quant_quality/v2-8k/set-1k.ids.json --out kv1k-int8.tsv
tools/quant_quality.py run --config strata-q2_0.json --kv q4_0 --ids data/quant_quality/v2-8k/set-1k.ids.json --out kv1k-q4_0.tsv
tools/quant_quality.py compare kv1k-fp16.tsv kv1k-int8.tsv kv1k-q4_0.tsv --segments --json compare-1k.json

tools/quant_bins.py bins --ref kv-fp16.tsv --run kv-int8.tsv --run kv-q4_0.tsv --bin 1000 --json bins.json
```

One 8K arm is ~3.5-6 min per segment (8191 positions) on this box; the 1K arms are ~30 s each. `tools/quant_bins.py`
is the one tool this run needed and did not have.

## Result (all 24,573 positions, reference fp16)

| KV | top-1 | mean KL | KL p99 | PPL | vs fp16 |
|---|---:|---:|---:|---:|---:|
| fp16 (reference) | - | - | - | 3.3113 | - |
| int8 | 97.04% | 0.00639 | 0.0692 | 3.3066 | **-0.14%** |
| q4_0 | 95.35% | 0.01652 | 0.1843 | 3.3183 | **+0.21%** |

Per segment:

| Segment | int8 top-1 | int8 KL | int8 PPL | q4_0 top-1 | q4_0 KL | q4_0 PPL |
|---|---:|---:|---:|---:|---:|---:|
| docs | 95.25% | 0.00846 | -0.26% | 92.27% | 0.02292 | +0.15% |
| code | 98.07% | 0.00506 | -0.28% | 97.09% | 0.01380 | +0.49% |
| multilingual | 97.80% | 0.00566 | +0.12% | 96.69% | 0.01284 | -0.00% |

- **int8 is free** (-0.14% PPL over 24,573 positions, its worst segment +0.12%).
- **q4_0 costs 0.21% PPL here** and its mean KL is 2.6x int8's. Its loss is in the tail: KL p99 0.184 against 0.069.

## The question this set exists for

### The tool gives every segment a fresh context

`quant_quality.py run` sends each segment as its **own `GEN` request** and sizes `--max-context` as
`seg_tokens + 512`, i.e. for one segment. So position `p` of a segment is scored with that segment's first `p`
tokens as its context - **position is context length**. That reading is not an assumption; it is measured, with
`tools/quant_bins.py segments` on a probe set (`2026-10-14-kv-ctxprobe`, the engine's `docs`/`code` tokens again):

| probe | result |
|---|---|
| `docs_long` (8192 ids) vs `docs_short` (its first 1200), positions 0-1199 | **1199/1199 identical top-64, 0.000e+00 max logp difference** |
| `code_fresh` vs `code_after_docs` (the same `code` text behind 7000 tokens of `docs`), 1199 positions | **0/1199 identical, 92.33% same argmax, KL 0.118, 8.74 max logp difference** |

The first row is what a fresh context looks like; the second row is what 7000 tokens of real context look like, so
the engine **does** read long context and the first row is not a hidden attention window.

### It does not grow

`bins.json` holds the full table; the mean KL per 1000-position bin:

| Segment | KV | 0-1k | 1-2k | 2-3k | 3-4k | 4-5k | 5-6k | 6-7k | 7-8.2k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| docs | int8 | 0.0063 | 0.0061 | 0.0092 | 0.0095 | 0.0090 | 0.0102 | 0.0088 | 0.0084 |
| docs | q4_0 | 0.0169 | 0.0174 | 0.0261 | 0.0254 | 0.0236 | 0.0276 | 0.0240 | 0.0229 |
| code | int8 | 0.0041 | 0.0018 | 0.0104 | 0.0055 | 0.0061 | 0.0026 | 0.0057 | 0.0048 |
| code | q4_0 | 0.0140 | 0.0067 | 0.0241 | 0.0166 | 0.0165 | 0.0096 | 0.0120 | 0.0112 |
| multilingual | int8 | 0.0107 | 0.0057 | 0.0078 | 0.0065 | 0.0075 | 0.0062 | 0.0008 | 0.0009 |
| multilingual | q4_0 | 0.0172 | 0.0131 | 0.0180 | 0.0168 | 0.0158 | 0.0177 | 0.0023 | 0.0036 |

**No segment rises with position.** `docs` is higher after its first 2,000 positions (0.0169/0.0174, then
0.0213-0.0288) and `multilingual` is far *lower* in its last 2,191 positions (0.0008-0.0055). The spread between
bins inside one segment is the size of the spread between segments in the first bin (0.0140-0.0172, three different
texts). The error follows the text, not the length. int8 and q4_0 rise and fall together, so the two formats are
responding to the same thing.

### The 1K arms are a reproducibility check, not a length comparison

`set-1k.ids.json` is the first 1000 tokens of these texts, but because each segment gets a fresh context, position
`p < 1000` has the **same** context in the 1K and the 8K run. At those positions the two runs are identical:

| positions 0-999, pooled | 1K run (2,997 positions) | 8K run's same positions (3,000) |
|---|---:|---:|
| q4_0 | 95.33% top-1, KL 0.01605 | 95.30% top-1, KL 0.01604 |
| int8 | 96.96% top-1, KL 0.00703 | 96.93% top-1, KL 0.00703 |
| fp16 logp of the text | - | **0.000e+00 difference at every shared position** |

So the 1K run costs ~2 min and its only new content is `compare-1k.json`'s per-segment rows at 999 positions; what
it does *not* do is compare two context lengths. The `v2-8k` README's claim that the 1K/8K pair isolates the context
length is wrong for this tool and is corrected there.

PPL between the two runs is **not** comparable (different position sets): q4_0 is +1.30% at 1K and +0.21% at 8K.
The per-position KL and top-1 are comparable, and those are flat.

## Caveats

- **Teacher-forced.** No error compounding: this is the per-token cost of the format, not what a free-running
  answer loses.
- **Content and position are not separable** in a single sequential pass. The three segments give different texts
  at the same position, and the bins give the same text at different positions; both say content. A length term
  smaller than the content spread would not be visible here.
- **KL p99 is pooled only.** `quant_quality.py compare` computes it on the `all` row; per-segment rows show `nan`.
- **Not comparable to `2026-10-14-kv-first-run`.** v1 is different text (and its 1K `docs` excerpt is not inside
  this 8K head).
- **Single run, no error bars.** The reference is fp16 K/V on Q2_0 weights, so only the KV format moves.
- **The probe used the prebuilt engine**; the bins and the probe table are computed from the committed tools on the
  raw dumps.

## What this means for the work package

The error is a **per-token quantizer** effect with no long-context tail term. A 1K set is therefore as good as an
8K one for the mean, and a further 32K point buys nothing for *this* question; the lever a format has to pull is the
accuracy of one 256-wide K/V vector, which is what the codebook work (WP-1 §5.1,
[`--kv q4lm`](../../../docs/KV_QUANT_WORKPACKAGE.md)) targets. The `q4_0` numbers above are the baseline those
changes are measured against.

## Raw data

Committed: `compare.json`, `compare-1k.json`, `bins.json` (this README's numbers). Not committed: the six per-position
`kv*.tsv` dumps (~26 MB and ~3 MB), which the commands above regenerate in ~45 min and ~5 min.
