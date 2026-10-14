# 2026-10-14: what a segment's context is (the 8K run's probe)

`tools/quant_quality.py run` sends each segment of a set as its **own `GEN` request**, so the question that the whole
8K measurement rests on is what context a segment is scored with. This probe answers it with one fp16 engine run, and
separates two explanations that the 8K runs alone cannot:

- *fresh*: a segment's context is only its own earlier tokens, so position `p` is scored with `p` tokens of context
  (and then `position = context length`, which is what makes `bins.json` a curve against the context length);
- *append*: a segment keeps whatever the process has already read, so its context is longer than the segment.

**Measured on:** the same box and engine as [the 8K run](../2026-10-14-kv-8k/README.md) (prebuilt 0.1.44.1 HIP,
Qwen3.8-Flash-Next Q2_0, MTP on, `--spec 4`), `--kv fp16`, `--max-context 8704 --short-read 8256`.

## The set

`set.ids.json` (committed) is four segments taken from the 8K set's own tokens
(`data/quant_quality/v2-8k/set.ids.json`), so the text is the same text:

| segment | ids | what it is |
|---|---:|---|
| `code_fresh` | 1200 | `code[0:1200]` |
| `code_after_docs` | 8200 | `docs[0:7000]` + `code[0:1200]` - the same `code` tokens, behind 7000 tokens of docs |
| `docs_long` | 8192 | `docs[0:8192]` |
| `docs_short` | 1200 | `docs[0:1200]` - the same text as `docs_long`'s first 1200 |

```
tools/quant_quality.py run --config strata-q2_0.json --kv fp16 \
    --ids bench/results/2026-10-14-kv-ctxprobe/set.ids.json --out kv-ctxprobe.tsv
tools/quant_bins.py segments --run kv-ctxprobe.tsv \
    --pair docs_long:0=docs_short:0 --pair code_fresh:0=code_after_docs:7000 --n 1200 --json compare.json
```

`docs_short` runs **last**, so if a segment kept the process's earlier tokens its context would be at least 17,590
tokens long; if it starts fresh, its context is `docs[0:1200]`, exactly like `docs_long`'s.

## Result

| probe | positions | same top-64 | same argmax | KL | max &Delta; logp(text) |
|---|---:|---:|---:|---:|---:|
| `docs_long:0` = `docs_short:0` | 1199 | **1199/1199** | 100.00% | 0.000000 | **0.000e+00** |
| `code_fresh:0` = `code_after_docs:7000` | 1199 | 0/1199 | 92.33% | 0.118211 | 8.742e+00 |

- A segment starts **fresh**: the same 1200 tokens are scored bit-for-bit identically whether they are the whole
  segment or its first 1200 (and regardless of three segments read before).
- 7000 tokens of real preceding context are **not** ignored: the same `code` tokens get a different distribution
  (only 92.33% of argmaxes agree, KL 0.118 against the zero-context version).

So `bins.json`'s x-axis is the context length, and the engine does read context at that length.

## Raw data

Committed: `set.ids.json`, `compare.json` (the table above). Not committed: `kv-ctxprobe.tsv` (~19 MB), which the
command above regenerates in ~10 min.
