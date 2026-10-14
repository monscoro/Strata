# 2026-10-14: KV cache, the first measurement point (`tools/quant_quality.py`)

The baseline the work package ([docs/KV_QUANT_WORKPACKAGE.md](../../../docs/KV_QUANT_WORKPACKAGE.md), WP-0) asks for:
the **same weights, the same text, only `--kv` differs**. Nothing here is a new format - it fixes what a KV format
change is compared against.

**Measured on:** Windows 11, one gfx1100 Radeon (16 GB), prebuilt engine 0.1.44.1 (HIP), Qwen3.8-Flash-Next **Q2_0**
pack (the weights are crunched too, so this isolates the *KV* format, not the weight quant), MTP on, `--spec 4`, and
the fixed set `data/quant_quality/v1` (code, docs, english, multilingual, agent; **999 next-token positions each,
4,995 in all**). Reference: `--kv fp16`. Teacher-forced through the decode windows (`STRATA_LOGPOS`, top-64).

## Commands

```
tools/quant_quality.py run --config strata-q2_0.json --kv fp16 --out kv-fp16.tsv
tools/quant_quality.py run --config strata-q2_0.json --kv int8 --out kv-int8.tsv
tools/quant_quality.py run --config strata-q2_0.json --kv q4_0 --out kv-q4_0.tsv
tools/quant_quality.py compare kv-fp16.tsv kv-int8.tsv kv-q4_0.tsv --segments
```

One arm is ~2.5 min (five segments, 26-30 s each) on this box. `--kv FMT` and the Windows `PATH` for the engine's
HIP DLLs are the two tool changes this run needed (commits `f275f3cf`, `94d31866`).

## Result (all 4,995 positions, reference fp16)

| KV | top-1 | mean KL | KL p99 | PPL | vs fp16 |
|---|---:|---:|---:|---:|---:|
| fp16 (reference) | - | - | - | 4.474 | - |
| int8 | 96.48% | 0.0136 | 0.206 | 4.469 | **-0.10%** |
| q4_0 | 94.15% | 0.0364 | 0.575 | 4.524 | **+1.12%** |

Per segment (mean KL, and PPL vs fp16):

| Segment | int8 KL | int8 PPL | q4_0 KL | q4_0 PPL |
|---|---:|---:|---:|---:|
| code | 0.0041 | -0.01% | 0.0140 | +2.12% |
| docs | 0.0130 | -0.36% | 0.0346 | +0.09% |
| english | 0.0073 | -0.46% | 0.0149 | +0.11% |
| multilingual | 0.0044 | -0.45% | 0.0091 | +0.18% |
| agent | 0.0390 | +0.80% | 0.1092 | **+3.15%** |
| **all** | 0.0136 | -0.10% | 0.0364 | +1.12% |

- **int8 is free within this run's spread** (PPL -0.10% overall, its worst segment +0.80%). No error bars: the
  tool gives one number per format, so the per-segment spread is the only handle. A second run would settle it.
- **q4_0 costs real precision, and its KL is 2.7x int8's** (2.0x-3.4x per segment). The loss is in the tail: KL p99
  is 0.575 against 0.206 for int8.
- **The agent segment is where it hurts** (KL 0.109, PPL +3.15%) - long-range exact copying. Code follows
  (+2.12% PPL, but a modest KL). Four of the five segments are near-neutral.
- **Not comparable to `2026-09-27-kv-q4`.** That older probe used a different tool (since removed), a single long
  *document* (1,024 tokens), and reported the document at **+8% PPL @1K**; here the `docs` segment is +0.09% with a
  similar KL (0.036 vs 0.039). Different text, different tokenization, and there free-running vs teacher-forced are
  not separated. **This 1.1% is the number to beat on this set; the 8% is not.**

## Caveats

- **1K only.** The fixed set is 999 positions per segment. 8K/32K (what the work package lists) needs a
  re-tokenized set (`ids --seg-tokens`) or the older long-document probe.
- **Teacher-forced.** No error compounding, so this understates what a greedy free-running answer loses.
- **Single run, no error bars.** The reference is fp16 K/V on Q2_0 weights; `--kv-resident` is dropped by the tool
  (quality only, no streaming).

## Raw data

`compare.json` (the numbers above) is committed. The three per-position `kv-*.tsv` files are ~5 MB each and stay
local; the commands above regenerate them in ~8 min. Not committed: the `*.tsv.seg.json` line ranges and `.stderr.txt`.
