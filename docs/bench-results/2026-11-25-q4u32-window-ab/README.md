# 128K-A/B: reines `q4u32` gegen `q4u32w` (vorab festgelegt, 2026-11-25)

## Frage

Bringt das exakte Fenster bei 128K Kontext **messbar bessere Qualität** als reines `q4u32`, wenn Build,
Modell, Prompt und Lesepfad gleich sind? Das entscheidet, ob wir den KV-Parameter `q4u32w` als
`--kv q4u32` + `STRATA_KV_WINDOW=1` etablieren oder das Fenster eine Opt-in-Option bleibt.

## Entscheidungsregel (vor dem Lauf fixiert)

`d = KL(q4u32w) - KL(q4u32)`, gepaart über dieselben 131.060 Positionen. Negativ heißt: das Fenster ist besser.

**Fenster etablieren** nur, wenn alle drei Bedingungen gelten:

1. `mean(d) < 0`,
2. die obere 95-%-Grenze von `d` aus dem t-Intervall über Positionen ist `< 0`,
3. die obere 95-%-Grenze aus dem Block-Bootstrap (32 Blöcke à 4.096 Positionen, 10.000 Resamples, Seed 20261125) ist `< 0`.

Sonst gilt: **kein nachgewiesener Qualitätsgewinn**. Der KV-Parameter bleibt reines `q4u32`; das Fenster
bleibt eine Opt-in-Wahl. Ein reiner Gleichstand wird nicht als Grund für einen neuen Parameter gewertet.

Top-1 gegen dieselbe fp16-Referenz und ein exakter McNemar-Test werden als Nebenbefunde berichtet,
nicht als zusätzliche Tor.

## Arme

| Arm | `--kv` | `STRATA_KV_WINDOW` | Bedeutung |
|---|---|---|---|
| `fp16` | `fp16` | nicht gesetzt | Referenz, kein KV-Codec |
| `q4u32` | `q4u32` | nicht gesetzt | reines 4,5-Bit-Grid |
| `q4u32w` | `q4u32` | `1` | gleiches Grid + 256 recent / 64 sink fp16-Zellen |

Alles andere ist in allen drei Armen identisch:

- Engine: `terminal/engine/strata.exe`, `0.1.63-q4u32`, SHA `e1be2bc764570709dfb77e075066b474b8099654c09edabef33e0bf5d2fb4532`
- Modell/Pack: `swift-iq3_xxs`, wie im 128K-Lauf vom 13.11.
- Prompt: `terminal/data/quant_quality/cwe-128k/t1.ids.json`, 131.061 Token, eine Segmentierung.
- Pfad: `quant_quality.py run`, Teacher-Forcing über die Decode-Fenster, `--topk 64`, `--prefill 6144`,
  CPU-Experten (ein Slot), keine MTP-Drafts, kein Prompt-Cache, `--max-context 131072`.
- Geteilte Env: `STRATA_KV_WINDOW_S=256`, `STRATA_KV_WINDOW_R=64`, `STRATA_KV_WINDOW_BATCH=6144`,
  `STRATA_VERIFY_ASYNC_LAUNCH=1`, dazu die festen `STRATA_ROUTE_TAIL_SKIP=0`,
  `STRATA_PREFILL_CPU_SHARE=0`, `STRATA_PCIE_FRAC_DEFAULT=old` aus `quant_quality.py`.
- 30 s Pause zwischen den Engine-Starts, ein Job auf der GPU.

Der einzige Unterschied zwischen den beiden q4-Armen ist `STRATA_KV_WINDOW=1`. Es wird kein
`--kv-resident` gesetzt; die Fenstervariante läuft voll resident.

## Dateien

- `config.json`: Engine, Modelle, ROCm-Bibliotheken.
- `run-ab.py`: startet die drei Arme, schreibt TSVs und Compare-JSON.
- `eval-ab.py`: rechnet die oben fixierte Regel mechanisch aus.
- TSVs, `.seg.json`, `.stderr.txt` und `manifest.json`: `C:\Users\krone\git\Strata-data\bench\2026-11-25-q4u32-window-ab\`
  (groß, nicht im Git).

## Ergebnis

**Das Fenster ist etabliert** (25.11.2026): alle drei vorab fixierten Bedingungen sind erfüllt. Damit ist die
Default-Änderung entschieden und umgesetzt — ohne `--kv` füllt die Engine auf gfx12 in einem `q4u32`-Build
`STRATA_KV_WINDOW=1` (S/R 256/64) selbst, und Setup schreibt dieselbe Env in die Konfiguration.

| Arm | mean KL | Top-1 gegen fp16 | McNemar gegen `q4u32` |
|---|---|---|---|
| `fp16` | — (Perplexität 2.751) | — | — |
| `q4u32` | 0.078700 | 98.173 % | — |
| `q4u32w` | **0.069493** | **98.308 %** | 1.063 nur-Fenster gegen 885 nur-Grid, p = 6,0e-5 (Nebenbefund) |

`d = KL(q4u32w) − KL(q4u32)`, gepaart über 131.060 Positionen:

- `mean(d) = −0.0092077` — Bedingung 1 erfüllt,
- t-Intervall 95 % `[−0.012395, −0.006020]`, `t = −5,66` — obere Grenze < 0, Bedingung 2 erfüllt,
- Block-Bootstrap (32 × 4.096, 10.000 Resamples, Seed 20261125) `[−0.013175, −0.005396]` — obere Grenze < 0,
  Bedingung 3 erfüllt.

131.060 Positionen geprüft; die Engine-SHA des Laufs ist die des Gates (`e1be2bc7…`), und `q4u32w` reproduziert
den veröffentlichten Wert 0.06949.

### Laufzeit — aus den eigenen stderr-Zeilen der drei Arme

Derselbe Prompt, 131.061 Token, `--prefill 6144` (die Zeile `strata serve: prompt … read in … ms`):

| Arm | Prompt 131.061 Token | tok/s | 1 generiertes Token | freier VRAM |
|---|---|---|---|---|
| `fp16` | 4.755.277 ms | 27,6 | 52 ms | 3.060 MiB |
| `q4u32` | 4.994.921 ms | 26,2 | 56 ms | 6.838 MiB |
| `q4u32w` | 4.949.004 ms | 26,5 | 52 ms | 6.686 MiB |

Das Fenster ist damit **0,9 % schneller** als reines `q4u32` (26,5 gegen 26,2 tok/s) — innerhalb der
Sitzungsstreuung dieser Maschine (±4 %, A.17): **kein messbarer Kostenfaktor**. Sein Preis ist VRAM:
**152 MiB** (6.838 → 6.686 MiB frei), passend zum Seitenpool-Modell (12,6 MiB × 12 QSA-Layer + Tags bei
B = 6.144).

### Grenzen dieses Laufs

Je Arm **ein** Lauf; die „Decode"-Zahl ist ein einziges Token und keine Decode-Messung. Die Läufe fuhren
CPU-Experten (ein Slot, Trefferquote 0 %) — der frei gewordene VRAM wurde hier also nicht zu Expert-Slots, und
der Laufzeitvergleich ist ein reiner Prompt-Vergleich. Der Windows-Commit meldete 84,9 / 80,5 / 70,4 GiB übrig
(fp16 / `q4u32` / `q4u32w`); die rund 10 GiB Differenz zwischen den beiden q4-Armen ist in diesem Lauf **nicht
erklärt** und wird dem Fenster nicht zugeschrieben. Die Qualität ist für diesen Prompt gemessen, nicht für andere
Dokumente; die Nadelgüte steht in `docs/bench-results/2026-10-24-needle-window/`.

### Folge

Default: Engine-Auto-Pfad (`engine/src/program/generate.cpp`) und Setup (`terminal/strata_control/setup_config.py`,
`terminal/setup.py`) schreiben `q4u32` + `STRATA_KV_WINDOW=1` (S/R 256/64); ein explizites `--kv q4u32` läuft
weiter das reine Grid, `STRATA_KV_WINDOW=0` schaltet das Fenster ab. Nach dem Rebuild (SHA-256
`0a3b2992…`, 45.682.688 Byte) wurden die Gates erneut gefahren: `kv_stream_parity`/`kv_win_parity` 5/5 grün,
`--dump-logits` in allen fünf Gate-Läufen byte-identisch (`75cb396e…`); der Auto-Pfad selbst ist in
`2026-11-25-q4u32w-streaming/` nachgewiesen (`default_check.py`: ohne `--kv` die Zeile `q4u32 with the
exact-cell window (q4u32w)`, 19 MiB Ring bei B = 256 weniger frei gegen `--kv q4u32`). Setup-Suite:
`test_setup_choices` (82), `test_kv_default` (11), `test_setup_golden`, `test_setup_risk` grün.
