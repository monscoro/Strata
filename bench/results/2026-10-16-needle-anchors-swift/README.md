# Needle anchor matrix (Phase 0, item 0): fp16 + int8 on Swift at 32K

Gate: the needle validity gate from `docs/KV_QUANT_WORKPACKAGE.md` §8 — every later KV arm must
retrieve planted needles at long context before its KL numbers count.

- **Where:** one PC, RX 9060 XT (gfx1200, RDNA4, 16 GB), 63.9 GB RAM, HIP build of 2026-10-16
  (commit `7d6d6268`, kvstats parked/env-off — default path byte-identical).
- **What ran:** Qwen3.8-Flash-Next, **Swift IQ3_XXS** weight pack, weights fixed, only `--kv` changed.
- **Tool:** hardened `tools/needle_bench.py` (`--lengths 32k --depths 10,50,90 --repeats 3
  --needle-types code,uuid --seed 7`): 18 needles per arm, one query each, no distractors
  (baseline anchors — distractors/multi-query come with the WP-1 comparison).
- **Server:** one server per `--kv` arm, explicit-PID kill; `--max-context 131072 --kv-resident 32768`.

## Result

| arm | found / total | prompt tokens | s/prompt |
| --- | ---: | ---: | ---: |
| `fp16` | **18 / 18** | ~32,800 | ~48–65 |
| `int8` | **18 / 18** | ~32,780 | ~60 |

Both anchors green at 32K: fp16 finished 11:17, int8 finished 11:36 (local time). The gate is open —
WP-1 arms will be graded against these two files.
