# Laufzeit-A/B der KV-Formate auf Swift (cwe-unabhängig, 16K-Kontext)

Design: `docs/KV_QUANT_WORKPACKAGE.md` §11.4.3 („Serving-Reife ist ungemessen: kein Speed-A/B") und §11.5.3.
Gemessen mit demselben HIP-Build vom 2026-10-17 wie WP-2a/WP-2b (§7.12/§7.14, uncommitted). Der Lauf liegt
**vor** den Review-Nachbesserungen §7.18 (K1–K4, R6); die Nachbesserung ist datenneutral gezeigt (Parität 12/12,
`q4lm8v` und `8vw-s128-r8` byte-gleich zu ihren 2026-10-17-TSVs), die Zahlen gelten also auch für den korrigierten
Code. R1/R2/R4/R5 hat der Review **bewußt nicht** angefaßt: §7.18 verweist für R1/R4/R5 auf genau diesen Lauf
(kein meßbarer Decode-Effekt), R2/R3 sind der Proben-Überbau selbst.

- **Where:** one PC, RX 9060 XT (gfx1200, RDNA4, 16 GB), 63.9 GB RAM, Windows.
- **What ran:** Qwen3.8-Flash-Next, **Swift IQ3_XXS**, Gewichte/Expert-Cache-Policy fix, `--max-context 16384`,
  nur `--kv` (+ `STRATA_KV_WINDOW*`) je Arm geändert. 8 Arme × 3 Runden, Rundenordnung alternierend
  (`tools/ab_engine.py`), pro Arm ein frischer Server.
- **Abgelesen:** die Engine-eigene Zahl `predicted_per_second` (Decode) und `prompt_per_second` (Prefill) der
  letzten Anfrage; der **long**-Prompt hat 6.345 Token, die drei kurzen 27–34. Die kurzen Prompt-Zahlen sind
  Overhead-dominiert und hier nicht aussagekräftig.
- **Kein OSCAR-Binary im Repo** (OSCAR ist ein Paper) — verglichen werden nur unsere eigenen Arme.
- **Kostenhinweis (R1/R2):** die Fenster-Arme laufen im **Proben-Überbau**: `kv_win_append_kernel`
  (`kv_q4.cu:541`) schreibt für *jede* Zelle einen fp16-Zwilling in einen flachen Seitenpool, plus eine
  zusätzliche Launch pro Append und Layer. Die Decode-Zahl der Fenster-Arme ist deshalb eine **Obergrenze**,
  nicht die Auslieferungszahl (die Auslieferung hält nur die S+R exakten Zellen, nicht den ganzen Kontext).

## Decode, long-Prompt (Median über 3 Runden, 6.345 Token)

| Arm | decode tok/s | Spanne (3 Runden) | vs. fp16 | prompt tok/s (6.345) |
| --- | ---: | ---: | ---: | ---: |
| `int8` | **44,9** | 42,9–46,2 | **+5,6 %** | 444 |
| `fp16` | 42,5 | 42,2–43,4 | +0,0 % | 450 |
| `q4lm32c` | 42,0 | 38,7–43,1 | −1,2 % | 448 |
| `q4lm32` | 41,7 | 40,2–43,0 | −1,9 % | 441 |
| `q4lm8v`+w 512/8 | 41,6 | 39,2–45,0 | −2,1 % | 437 |
| `q4lm32w` 512/8 | 41,4 | 38,2–43,7 | −2,6 % | 435 |
| `q4lm8v`+w 256/64 | 40,3 | 37,1–40,4 | −5,2 % | 429 |
| `q4lm8v` | 39,7 | 38,1–44,7 | −6,6 % | 436 |

`n` (generierte Token) schwankt arms- und rundenweise (107–256) — der Decode-Wert ist pro Token, das streut
deshalb weniger als die Antwortlänge.

## Was das sagt

- **Das Format ist im Decode praktisch frei.** Die ganze Spanne ist 39,7–44,9 tok/s (±6 %); `int8` ist der
  schnellste Arm, `q4lm8v` der langsamste. Die Entscheidung zwischen den KV-Formaten fällt also über
  Qualität (§7.9/§7.13/§7.15/§7.16) und Speicher, **nicht** über Decode-Durchsatz auf dieser Karte.
- **Der Fenster-Überbau ist im Decode nicht sichtbar.** `q4lm32w`(512/8) 41,4 liegt innerhalb der Spanne von
  `q4lm32` 41,7 (38,2–43,7 gegen 40,2–43,0); `q4lm8v`+w(512/8) 41,6 liegt *über* `q4lm8v` 39,7 (dessen Spanne
  aber die breiteste ist). Bei einem Token pro Schritt ist der Zwillings-Append winzig gegen die Attention
  über 6,3K Token. **R1/R2 sind damit im Decode widerlegt** — sie kosten Speicher (s. u.), nicht Decode-Zeit.
- **Prefill zeigt einen Hauch davon.** Die Fenster-Arme liegen bei 435/437/429 gegen 441/436 ihrer Zwillinge
  (≤3 %, innerhalb der Spanne); ein 6.345-Token-Prefill schreibt 6.345 × 24 KiB ≈ 152 MiB Zwilling zusätzlich.
  Eine echte (nicht-probende) Auslieferung würde weniger schreiben.

## Speicher: was das Fenster wirklich kostet (R3)

Der Seitenpool der **Probe** war **flach, pro QSA-Layer, immer resident** und wurde von `max_cells` bemessen
(`qsa_win_bytes(s, max_cells)` bis `d98f6472`): `max_cells × n_head_kv × head_dim × 4 B` = **2.048 B je Zelle** je
Layer.
Der Kopf ist bei diesem Modell `n_head_kv = 2`, `head_dim = 256` (GGUF), und **12 Layer** tragen einen
QSA-Zustand (`qsa_interval = 4` von 48, `geometry.hpp:26/66`) → **24 KiB je Kontextzelle** über alle Layer.

Gemessen (gleicher Build, gleiche Expert-Cache-Vorgabe; Differenz von `strata serve: N MiB of VRAM free with
everything loaded` zwischen Fenster-Arm und seinem Zwilling):

| `--max-context` | Formel (12 Layer) | gemessen | |
| ---: | ---: | ---: | --- |
| 4.096 | 96 MiB | −53 MiB | im Rauschen (der Engine justiert den „festen" Cache auf 1318 Slots nach) |
| 16.384 | 384 MiB | 416 MiB | ✓ (+8 %, im Rauschen) |
| 65.536 | 1.536 MiB | **1.536 MiB** | ✓ exakt |
| 65.536 + `--kv-resident 20480` | 1.536 MiB | **1.536 MiB** | ✓ exakt — **trotz Streaming** |

**R3 ist damit entschieden:** mit `--kv-resident 20480` schrumpft das quantisierte K/V auf 0,42 GiB pinned RAM
(„KV streaming: 20480 of 65536 cells per QSA layer in VRAM"), der Fenster-Pool bleibt aber bei **1.536 MiB**.
Er skaliert mit `max_context`, nicht mit dem Resident-Set — das Fenster hebelt das Streaming aus. Hochgerechnet:
16K ≈ 384 MiB, 64K ≈ 1,5 GiB, 128K ≈ 3 GiB, unabhängig von `--kv-resident`.

**Aber:** das ist die Form der **Probe**, nicht die der Auslieferung. Der flache Pool hält fp16 für *jede* Zelle,
weil der Leser direkt über die Zellnummer indiziert. Die Auslieferung ist ein **Ring** (seit §7.19 des Plans so
gebaut): `R` zellen-adressierte Sink-Zeilen plus `S + B` Ringzeilen (`R + cell % (S + B)`), jede Ringzeile mit der
Zelle, die sie hält, als Tag. Das `B` ist der größte Append-Batch (`STRATA_KV_WINDOW_BATCH`; ohne Angabe folgt er
dem Prompt-Chunk des Laufs, mindestens 512 = `--prefill 512`), weil
**jede Query eines Prefill-Batches ihren eigenen Horizont hat** — ein Batch von `T` Zellen
braucht `S + T` Ringzeilen gleichzeitig. Bei 256/64 und `B = 512`: `(64 + 256 + 512) × 2 KiB × 12 Layer =
**19,5 MiB**`, **kontextunabhängig** (512/128: 27 MiB; ein Token-für-Token-Lauf mit `B = 1` wäre 7,7 MiB, die Vorgabe
ist aber der Prompt-Chunk mit mindestens 512). Das ist ein Rechenweg aus der exakten 2 KiB/Zelle/Layer der
64K-Messung; die Ringgröße selbst ist in §7.19 gemessen. Die Streaming-Unverträglichkeit gilt für die Probe, der Ring
hat sie nicht.

**Korrektur einer Review-Zahl.** Der Code-Review hatte „2K ≈ 10 MB, 16K ≈ 64 MB, 128K ≈ 512 MB" und 48 Layer
genannt. Beides war falsch: es sind **12** QSA-Layer (nicht 48) und **24 KiB pro Kontextzelle** (nicht ~4 KiB) →
2K ≈ 48 MiB, 16K ≈ 384 MiB, 128K ≈ 3 GiB. Die 64K-Messung (1.536 MiB, exakt 12 × 128 MiB) ist die belastbare.

Rohdaten: `kvab.jsonl` (96 Zeilen: 8 Arme × 3 Runden × 4 Anfragen). Auswertung: `analyze.py` neben dem Lauf
(`…\opencode\kv-ab\analyze.py`, liest `kvab.jsonl`, druckt `--markdown`).
