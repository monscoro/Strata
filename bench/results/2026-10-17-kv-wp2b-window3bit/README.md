# WP-2b Fenster auf Swift bei ≤3 Bit: das Fenster trägt auch dort, rettet 3 Bit aber nicht (cwe-2k)

Design: `docs/KV_QUANT_WORKPACKAGE.md` §5.6.2; Arm in §7.14 (`STRATA_KV_WINDOW=1` als **zur Level-Tabelle
orthogonale** Überlagerung auf `q4lm8*`, Gitter per `STRATA_KV_WINDOW_S`/`_R`). Gemessen mit demselben Build wie
WP-2a (§7.12), der `q4lm8v`- und `int8`/`fp16`-Anker reproduzieren die WP-1-Läufe **bit-genau** (SHA256-gleich).

- **Where:** one PC, RX 9060 XT (gfx1200, RDNA4, 16 GB), 63.9 GB RAM, HIP build of 2026-10-17 (§7.12/§7.14;
  uncommitted).
- **What ran:** Qwen3.8-Flash-Next, **Swift IQ3_XXS**, weights fixed, only `--kv` + `STRATA_KV_WINDOW*` changed.
- **Test:** RULER CWE at 2K, 8 segments, ~2,048 tokens each, `data/quant_quality/cwe-2k/` (16.351 Positionen).
  Referenz: der `fp16`-Arm desselben Laufs (PPL 2,999). ~9 min/Arm.
- **Nullpunkt:** `q4lm8v` mit `STRATA_KV_WINDOW=1`, S=0/R=0 ist `q4lm8v` — die beiden TSV sind **byte-gleich**
  (SHA256 `83D13E7A…`), nicht nur statistisch gleich. Ebenso `fp16`/`int8`/`q4lm8v` gegen die WP-1-Anker.

## Whole segment (16.351 Positionen)

| arm | S / R | top-1 | mean KL | KL / `q4lm8v` | KL p99 | PPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fp16` | – | – | 0 | – | 0 | 2,999 |
| `q4lm8v` | – | 98,92% | 0,01791 | 1,00x | 0,2793 | 3,024 |
| `q4lm8v`+w | 0 / 0 | 98,92% | 0,01791 | 1,00x | 0,2793 | 3,024 |
| `q4lm8v`+w | 128 / 8 | 99,39% | 0,00869 | **0,49x** | 0,0929 | 3,001 |
| `q4lm8v`+w | 512 / 8 | 99,36% | 0,00828 | **0,46x** | 0,0926 | 2,996 |
| `q4lm8v`+w | 1024 / 8 | 99,35% | **0,00741** | **0,41x** | 0,0924 | 2,999 |
| `int8` (Anker) | – | 99,35% | 0,00503 | 0,28x | 0,0733 | 3,004 |

PPL fällt unter den fp16-Wert (2,996 < 2,999) — derselbe Dämpfungs-Effekt wie beim Clip-Arm (§7.11) und in WP-1
(`q4lm32t` 2,996). **PPL ist hier kein Qualitätsleser; KL ist es.**

## Paired über die 8 Segmente (die Entscheidungsauslese)

| pair | ΔKL | t | segments up |
| --- | ---: | ---: | ---: |
| `q4lm8v` − win 128/8 | +0,00922 | +5,05 | 8/8 |
| `q4lm8v` − win 512/8 | +0,00963 | +7,67 | 8/8 |
| `q4lm8v` − win 1024/8 | +0,01050 | +4,07 | 8/8 |
| win 128/8 − win 512/8 | +0,00041 | +0,38 | 7/8 |
| win 512/8 − win 1024/8 | +0,00167 | +2,80 | 8/8 |
| `int8` − win 1024/8 | −0,00238 | −2,27 | 1/8 |

Antwortbereich (266 Zeilen): `q4lm8v`−512/8 +0,00578 (t +1,61, 7/8), `q4lm8v`−1024/8 +0,00746 (t +1,93, 7/8),
`int8`−512/8 −0,00415 (t −3,31, 1/8) — Vorzeichen wie im Ganzen.

## Was die Bytes kosten

Die **Probe** hält einen fp16-Zwilling des **ganzen** Kontexts (`qsa_win_bytes` über `max_cells`), der Leser bedient
daraus nur das Fenster: alle Arme haben dieselbe Fußspur, die Unterschiede sind reine **Lesepolitik**. Die
**Auslieferung** hält nur die (S+R) exakten Zellen fp16, der Rest bleibt q4 → Basis `q4lm8v` **3,5 Pegel-Bits**,
gespeichert **≈4,0 Bit/Wert** (Pegel + 0,5 Skalen-Overhead, wie der 18-B-Block von `q4_0`), Zusatz
`(16 − 4,0)·(S+R)/L` mit L ≈ 2044:

| arm | Bit/Wert (Auslieferung) | mean KL |
| --- | ---: | ---: |
| `q4lm8v` | ≈4,0 | 0,01791 |
| win 128/8 | ~4,8 | 0,00869 |
| win 512/8 | ~7,1 | 0,00828 |
| win 1024/8 | ~10,1 | 0,00741 |

**Korrektur (§7.19, 2026-10-18).** Diese Spalte ist das Modell `Basis + (16 − Basis)·(S+R)/L` für 2K und **zu
optimistisch**: die exakte Zelle wird **zusätzlich** zum quantisierten Wert gehalten (16 Bit, nicht 16 − Basis), und
die feste Spanne der Auslieferung ist seit dem Ring `S + B` (jede Query eines Chunks hat ihren eigenen Horizont). Mit
`Basis + 16·(S+B)/L` sind es für 256/64 ≈10,5 Bit bei 2K, 4,8 bei 16K, 4,2 bei 64K. Die **KL-Werte und die
Rangfolge** bleiben unberührt.
| `int8` (aus WP-1) | 8,25 | 0,00503 |

## Der Vergleich mit dem 4-Bit-Arm (der eigentliche Befund)

Die `fp16`-Anker **aller** Läufe (WP-1, WP-2a, WP-2b) sind byte-gleich (SHA256 `36C9CE2D…`), also direkt
zusammenlegbar. Dieselbe Byte-Skala (Auslieferungs-Modell, L ≈ 2044 bei 2K), WP-2a und WP-2b zusammen
(4-Bit-Basis `q4lm32` = 4,5 Bit/Wert gespeichert):

| Bit/Wert | 3-Bit-Basis (`q4lm8v`) | 4-Bit-Basis (`q4lm32`) | `int8` |
| ---: | ---: | ---: | ---: |
| ~4,5 | – | 0,01183 (kein Fenster) | – |
| ~4,8 | **0,00869** (128/8) | – | – |
| ~5,3 | – | 0,00815 (128/8) | – |
| ~7,1 | 0,00828 (512/8) | – | – |
| ~7,4 | – | **0,00609** (512/8) | – |
| ~10 | 0,00741 (1024/8) | 0,00672 (1024/8) | – |
| 8,25 | – | – | 0,00503 |

Gepaart über dieselben 8 Segmente (`ΔKL` = linker Arm minus rechter):

| Paar | ΔKL | t | Segmente | Bits links/rechts |
| --- | ---: | ---: | ---: | --- |
| `q4lm32` − `8vw-s128-r8` | +0,00314 | +3,06 | 6/8 | 4,5 / 4,8 |
| `win-s128-r8` − `8vw-s128-r8` | −0,00054 | −0,80 | 4/8 | 5,3 / 4,8 |
| `win-s128-r8` − `8vw-s512-r8` | −0,00012 | −0,13 | 4/8 | 5,3 / 7,1 |
| `int8` − `win-s512-r8` | −0,00106 | −1,40 | 1/8 | 8,25 / 7,4 |

## Verdict

- **Das Fenster trägt bei ≤3 Bit.** `q4lm8v` 0,01791 → 128/8 **0,00869** (−51%), 512/8 0,00828, 1024/8 0,00741
  (−59%); alle 8/8 Segmente in dieselbe Richtung. Die §7.13-Frage, ob das Fenster bei ≤3 Bit trägt, ist damit
  beantwortet: **ja** — es ist keine 4-Bit-Eigenheit, sondern eine allgemeine Mechanik der Lesepolitik.
- **Das Knie liegt bei ≤3 Bit höher.** Bei 4 Bit war 512/8 das Knie (1024/8 flach, t −0,79). Hier ist 512/8 vs
  1024/8 noch signifikant (+0,00167, t +2,80, 8/8): die schlechtere Mitte will ein größeres exaktes Fenster. Genau
  das kostet die Bytes, die den Vorteil auffressen.
- **Cross-Over bei ≈4,8–5,3 Bit — „3 Bit tot" gilt dort nicht.** Im Low-Byte-Bereich ist `8vw-s128-r8` (≈4,8 Bit)
  gegen `win-s128-r8` (≈5,3 Bit) bei statistisch gleicher KL **~0,5 Bit/Wert (~9 %) billiger** (−0,00054, t −0,80).
  Gegen den fensterlosen 4-Bit-Arm ist es **kein** Byte-Sieg (4,8 vs 4,5 Bit), nur bessere KL (+0,00314, t +3,06,
  6/8). *Oberhalb* dreht es: `8vw-s512-r8` (≈7,1 Bit) ist gegen `win-s128-r8` (≈5,3 Bit) Gleichstand bei **1,8 Bit
  mehr** (−0,00012, t −0,13). Die 3-Bit-Basis lebt also **mit** Fenster nur als schmaler Low-Byte-Punkt und
  skaliert nicht; ohne Fenster gilt §7.9 (tot).
- **Der beste Byte-Punkt ist die 4-Bit-Basis mit Fenster.** `q4lm32w`(512/8, ≈7,4 Bit) ist von `int8` (8,25 Bit) mit
  **~10 % weniger Bytes** statistisch nicht zu trennen (−0,00106, t −1,40, 1/8); `int8` bleibt die absolute Decke
  (0,00503), aber nicht mehr pro Byte unerreichbar.
- **Grenze.** Die Bit-Zahlen sind ein **Modell** (Pegel + Block-Skalen-Overhead + `(16 − Basis)·(S+R)/L`, L ≈ 2044
  bei 2K), nicht gemessen; bei längerem Kontext sinkt der Fenster-Anteil, das Cross-Over kann sich verschieben.
  Qualität ist nur bei 2K gemessen.

Rohdaten: `compare.json`, `paired.json` (TSVs 17 MB/Arm, nicht eingecheckt); `compare-pooled.json`/`paired-pooled.json`
sind dieselbe Auswertung **inklusive** der vier Knie-Arme aus §7.16 (13 Arme). Begriffe (KL, ΔKL, t, Segment, Fenster-
und Quantisierungsnotation, Byte-Modell): `docs/KV_QUANT_WORKPACKAGE.md` §12 Glossar.

**Nachtrag (§7.16, `bench/results/2026-10-18-kv-wp2b-oscar-knee/`):** das hier gefundene Cross-Over (≈4,8–5,3 Bit)
gilt nur für das `R = 8`-Gitter. Mit dem OSCAR-Knie (`R = 64`, S 256) und dem Sink-Paar 512/128 — auf beiden Basen
nachgemessen — sind **alle vier** Arme von `int8` nicht zu trennen (ns); die 3-Bit-Basis erreicht `int8`-Niveau bei
**≈5,9 statt 8,25 Bit** und verliert pro Byte in keiner der fünf gemessenen Formen meßbar (`|t| ≤ 1,50`). Der Sink
(`R`) ist ein eigener Hebel (R 8 → 128 bei festem S = 512: +0,0020…0,0026, 7–8 von 8 Segmenten).
