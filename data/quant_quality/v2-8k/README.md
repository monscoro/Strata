# quant_quality `v2-8k`: the yardstick at 8192 tokens

The v1 set (`data/quant_quality/v1/`) carries 1,066-2,049 tokens per segment, so it cannot answer the question that
matters most for a KV format: **does its error grow with the context length?** This set is the same idea at 8,192
tokens per segment, built only from material inside this repository - nothing invented, nothing downloaded.

| Segment | Source (in this repository) | Rule |
| --- | --- | --- |
| docs | `docs/DETAILS.md` | first 8192 tokens |
| code | `tools/gguf_reader.py` + `tools/quant_quality.py` + `tools/quant_formats.py` + `tools/quant_bound.py` + `tools/strata_tokenizer.py`, joined with a blank line | first 8192 tokens |
| multilingual | `README.de.md` + `README.fr.md` + `README.es.md`, joined with a blank line | first 8192 tokens |

Each `*.txt` is the source tokenized with the pack tokenizer, cut to its first 8192 ids and decoded back, so
`--seg-tokens 8192` consumes the whole file.

**`english` and `agent` are absent on purpose.** The v1 segments of both are hand-written (a short story, an agent
transcript) and no 8,192-token source for either exists in the repository; writing one to score it would make the set
unreviewable. The `--segments` argument of `tools/quant_quality.py ids` is what lets a set carry only the segments it
has.

## Two lengths, one text

`set.ids.json` is 8192 tokens per segment. `set-1k.ids.json` is the **first 1000 tokens of the same three texts**.
Scoring both means the 1K positions are a prefix of the 8K positions, so what separates the two runs is the context
length and not a different excerpt.

```
.venv/Scripts/python tools/quant_quality.py ids --tokenizer <pack>/tokenizer --set data/quant_quality/v2-8k \
    --seg-tokens 8192 --segments docs,code,multilingual --out data/quant_quality/v2-8k/set.ids.json
.venv/Scripts/python tools/quant_quality.py run --config strata-q2_0.json --kv q4_0 \
    --ids data/quant_quality/v2-8k/set.ids.json --out kv-q4_0.tsv
```

Both ids files are committed, so a box without the tokenizer scores the same positions (the same rule as v1).

Result: [bench/results/2026-10-14-kv-8k/](../../../bench/results/2026-10-14-kv-8k/README.md).

**Not comparable to v1.** These are different excerpts of the same sources - v1's `docs` excerpt starts at token
53,159 of `DETAILS.md`, which is not inside this 8K head. Compare 1K against 8K **within this set**, where the text
is nested.
