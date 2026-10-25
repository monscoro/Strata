# 2026-10-25: S0 — is the yardstick stable under a second KL support?

The ladder's first stage (`docs/paper/Ansatz_Improvements.md` §7, S0). The question is not "does the window win" — that
was measured on engine 0.1.44.1 — but whether the *ordering* the ladder is built on survives a second definition of the
metric. Answer: it does, to the sign, at both supports. The kill (a ranking or sign flip) does not fire.

Second result, unplanned but worth having: engine **0.1.47** reproduces the 0.1.44.1 anchors to the printed digit, on a
fresh corpus import and a fresh config.

## Machine and build

- GPU: AMD Radeon RX 9060 XT (gfx1200, 16 GB). Windows 11. HIP via `C:\Program Files\AMD\ROCm\7.1\bin`.
- Engine: `engine\strata.exe`, `engine/BUILD.json` says version **0.1.47**, backend hip, arch gfx1200, built locally.
  SHA-256 `40942104945F9919268AEC6664DD6E599AB8E20EBD3F7CC84C5B442927007B7B`.
- Weights: Swift 1.5 IQ3_XXS (pack `swift-iq3_xxs`, native GGUF `…IQ3_XXS-00001-of-00002.gguf`), `--spec 4`,
  `--spec-min-p 0.5`, `--mtp`, `--expert-profile data/expert-profile.bin`, `--expert-cache 1`, `--prefill 512`.
- The config used is `strata-swift-iq3_xxs.json`, which is **gitignored** (`/strata-*.json`) and not in the tree. It
  carries no window env: every arm passes its own, so all four arms are the same config with one flag changed.

The harness is `tools/quant_quality.py` (commit `d77748a5`), which is the `KV_QUANT_research` version plus `--kv`,
`--topk` and `--kl-topk`. Fixed by the harness for every arm: `--expert-cache 1 --adapt-every 100000 --pcie-frac 0
--suffix-draft 0 --prompt-cache 0`, `--max-context seg+512`, `--short-read seg+64`, and the env
`STRATA_ROUTE_TAIL_SKIP=0 STRATA_PREFILL_CPU_SHARE=0 STRATA_PCIE_FRAC_DEFAULT=old`.

## Corpus

`data/quant_quality/cwe-4k` (commit `040f9f94`), byte-identical to `KV_QUANT_research`: 8 segments × 4,096 tokens of
code/agent text. `set.ids.json` fixes 32,754 positions; **32,746 were scored** (each segment's tail loses one to four
positions to the scoring window). The paper's cwe-4k table also prints 32,746, so the same positions are being compared.

## Arms and commands

One engine at a time, sequentially, each arm a fresh process:

```powershell
python tools\quant_quality.py run --config strata-swift-iq3_xxs.json --ids data\quant_quality\cwe-4k\set.ids.json `
    --kv <FMT> --topk 256 --out <raw>\<arm>.tsv [--env K=V ...]
```

| Arm | `--kv` | Window env |
|---|---|---|
| `fp16` | `fp16` | — (this is the KL reference) |
| `int8` | `int8` | — |
| `win256` | `q4lm8v` | `STRATA_KV_WINDOW=1`, `STRATA_KV_WINDOW_S=256`, `STRATA_KV_WINDOW_R=64` |
| `plain` | `q4lm8v` | — |

`--topk 256` writes the engine's 256 most likely tokens per position (the engine's cap), so the same runs can be scored
at either support without a second pass. Runtime 19–21 min per arm; segment times are in each `.log`.

## Raw data

Not in the repo (131 MB per arm): `C:\Users\krone\git\Strata-data\bench\2026-10-25-s0\` — four `*.tsv`, their
`*.seg.json` (line ranges, kv, topk, engine args) and `*.stderr.txt`.

| File | SHA-256 |
|---|---|
| `fp16.tsv` | `D47076607103759F6F7334242916956AD17FE9E753BBA052EBEBA741A49D4EB2` |
| `int8.tsv` | `4FB65E55F60355BA3332519E25C1EEC5F8C2B744FB1CA5F3087B85603A863BD9` |
| `win256.tsv` | `E0974D84D9C71FF6611EB2BDD97687342ECB2B6AAF6AC1D28AF41CB8FDCFC32A` |
| `plain.tsv` | `8AE39E2FF714171736F3888DFDED7D6A91E60B30D0D84C699A2CAE1670B2E99A` |

## Results, KL over the reference's 64 most likely tokens (the published definition)

| Arm | top-1 | KL | KL p99 | PPL |
|---|---:|---:|---:|---:|
| `int8` | 99.05 % | **0.01688** | 0.1888 | 2.860 |
| `win256` | 98.84 % | **0.02090** | 0.2811 | 2.858 |
| `plain` | 98.40 % | **0.03182** | 0.5785 | 2.888 |

fp16 reference PPL 2.857. Paired whole segments (`ref − run`, negative = the reference is better):

| Pair | mean | t | segments up |
|---|---:|---:|---:|
| `int8 − win256` | **−0.00402** | −3.55 | 1/8 |
| `int8 − plain` | −0.01495 | −4.58 | 0/8 |
| `win256 − plain` | **−0.01093** | −4.10 | 0/8 |

Answer region: `int8 − win256` −0.00159 (t −0.51, 2/8) — not separable; `int8 − plain` −0.01653 (t −4.72, 0/8);
`win256 − plain` −0.01494 (t −4.30, 0/8).

## Results, KL over the reference's 256 most likely tokens

| Arm | top-1 | KL | KL p99 | PPL |
|---|---:|---:|---:|---:|
| `int8` | 99.05 % | **0.01774** | 0.1980 | 2.860 |
| `win256` | 98.84 % | **0.02223** | 0.2955 | 2.858 |
| `plain` | 98.40 % | **0.03387** | 0.6143 | 2.888 |

Paired whole segments:

| Pair | mean | t | segments up |
|---|---:|---:|---:|
| `int8 − win256` | **−0.00449** | −3.49 | 1/8 |
| `int8 − plain` | −0.01613 | −4.31 | 0/8 |
| `win256 − plain` | **−0.01164** | −3.94 | 0/8 |

Answer region: `int8 − win256` −0.00159 (t −0.51, 2/8) — **identical to the 64-support value to every printed digit**;
`int8 − plain` −0.01655 (t −4.73); `win256 − plain` −0.01496 (t −4.31).

## What the wider support does

- **Level:** every arm rises — `int8` +5.1 %, `win256` +6.4 %, `plain` +6.4 %. This is the floor substitution, not a new
  disagreement: a token outside the run's 256 gets the run's 256th log-probability, and a wider support admits more such
  tokens. The better the arm, the smaller the rise (`int8`'s tail is closest to fp16's).
- **Ordering:** unchanged, all three arms, both supports.
- **Signs:** unchanged, all six pairs, both supports; the 1/8 segment (`s06`: 0.00688 against int8's 0.00822 at 64;
  0.00730 against 0.00876 at 256) is the same segment.
- **Magnitudes:** the paired differences move by ≤12 % (`int8 − win256` −0.00402 → −0.00449); every t keeps its sign
  and its magnitude to within 0.3.
- **Gap closure** of the window against plain: 73.1 % at 64 (`0.01093/0.01495`), 72.2 % at 256 (`0.01164/0.01613`).
- **Answer region:** the two supports are indistinguishable. The answer region's mass is inside the top 64 tokens, so a
  wider support adds nothing there — that is a property of the region, not of the arms.

## The anchors reproduce on a different build

The paper's cwe-4k table was measured on engine 0.1.44.1 with `--prefill 512`; this block runs 0.1.47. Every printed
number matches:

| Value | Paper (0.1.44.1) | This block (0.1.47) |
|---|---|---|
| `int8` KL | 0.01688 | 0.01688 |
| `win256` KL | 0.02090 | 0.02090 |
| `plain` KL | 0.03182 | 0.03182 |
| `int8 − win256`, t, signs | −0.00402, −3.55, 1/8 | −0.00402, −3.55, 1/8 |
| answer `int8 − win256`, t, signs | −0.00159, −0.51, 2/8 | −0.00159, −0.51, 2/8 |

So the anchors are not an artefact of the 0.1.44.1 binary, and the two builds can be compared directly. It does **not**
show the builds are identical in general — only that the KV read path this measurement touches behaves the same.

## Verdict

S0 does not kill. The ladder continues with K1/K2 defined on the top-64 KL, and the top-256 column stands beside it as
the support check.

## Limits

- One model (Swift 1.5 IQ3_XXS), one corpus (code/agent text), 4K. Nothing here says anything about 128K.
- The support test is **top-256, not the full vocabulary**. It bounds how much the truncation can hide; it does not
  remove the floor substitution. The answer region shows the truncation is irrelevant there; the whole-segment level
  shifts by ~6 %.
- `--topk 256` is the engine's cap. A wider support is not available without a code change.
- Four arms, one run each, no repeats. The paired per-segment test is what carries the claim; a single-arm mean over
  positions would not (the segments differ by 20× in KL).
- The config file is gitignored and untracked; the args above are its content. A reader reproducing this needs their
  own config with the same pack, GGUF and expert profile.

## Files here

- `compare-64.json`, `compare-256.json` — per-segment and `all` for every arm (top-1, KL, p99, PPL).
- `paired-64.json`, `paired-256.json` — the paired whole-segment and answer-region tables.
