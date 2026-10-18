# WP-2b-Knie (OSCAR-Punkt) auf Swift: beide Basen erreichen int8-Niveau, die 3-Bit-Basis am billigsten (cwe-2k)

Design: `docs/KV_QUANT_WORKPACKAGE.md` §5.6.2, Arm §7.14 (`STRATA_KV_WINDOW=1` als **zur Level-Tabelle
orthogonale** Überlagerung), Bericht §7.16. Gemessen mit demselben Build wie WP-2a/WP-2b (§7.12/§7.14); die Anker
`fp16`/`int8`/`q4lm8v`/`q4lm32` reproduzieren die früheren Läufe **byte-genau** (SHA256-gleich, s. u.).

Vier neue Arme: das **OSCAR-Knie** `S / R = 256 / 64` (`R = 64` Sink-Zellen ist der in der OSCAR-Arbeit gute
Punkt) und das reine **Sink-Paar** `512 / 128`, jeweils auf **beiden** Basen (`q4lm8v` ≤3 Bit und `q4lm32w` 4 Bit).

- **Where:** one PC, RX 9060 XT (gfx1200, RDNA4, 16 GB), 63.9 GB RAM, HIP build of 2026-10-17 (§7.12; uncommitted).
- **What ran:** Qwen3.8-Flash-Next, **Swift IQ3_XXS**, weights fixed, nur `--kv` + `STRATA_KV_WINDOW_S/_R` geändert.
- **Test:** RULER CWE bei 2K, 8 Segmente à ~2.045 Token, `data/quant_quality/cwe-2k/` (16.351 Positionen).
  Referenz: der `fp16`-Arm desselben Testsets (PPL 2,999). ~9,5 min/Arm.
- **Anker:** `fp16` `36C9CE2D…`, `int8` `3CB6A156…`, `q4lm8v` `83D13E7A…`, `q4lm32` `A3F4CE4A…` — alle wie in
  WP-1/WP-2a/§7.15, d. h. die Läufe sind direkt zusammenlegbar. Die Knie-Arme haben keinen Identitäts-Zwilling;
  dass das Fenster greift, zeigen die KL-Abstände (`8vw-s256-r64` 0,00702 ≪ `q4lm8v` 0,01791).

## Whole segment (16.351 Positionen)

| arm | S / R | top-1 | mean KL | KL p99 | PPL | Bit/Wert (Auslieferung, Modell) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fp16` | – | – | 0 | 0 | 2,999 | 16 |
| `q4lm8v` | – | 98,92 % | 0,01791 | 0,2793 | 3,024 | ≈4,0 |
| `q4lm32` | – | 99,19 % | 0,01183 | 0,1464 | 3,012 | 4,5 |
| `int8` (Anker) | – | 99,35 % | 0,00503 | 0,0733 | 3,004 | 8,25 |
| `8vw-s128-r8` | 128 / 8 | 99,39 % | 0,00869 | 0,0929 | 3,001 | ≈4,8 |
| `8vw-s256-r64` | 256 / 64 | 99,33 % | **0,00702** | 0,0835 | 3,003 | **≈5,9** |
| `8vw-s512-r8` | 512 / 8 | 99,36 % | 0,00828 | 0,0926 | 2,996 | ≈7,1 |
| `8vw-s512-r128` | 512 / 128 | 99,39 % | **0,00623** | 0,0743 | 2,995 | ≈7,8 |
| `8vw-s1024-r8` | 1024 / 8 | 99,35 % | 0,00741 | 0,0924 | 2,999 | ≈10,1 |
| `win-s128-r8` | 128 / 8 | 99,41 % | 0,00815 | 0,0856 | 2,994 | ≈5,3 |
| `32w-s256-r64` | 256 / 64 | 99,37 % | 0,00782 | 0,0752 | 3,002 | ≈6,3 |
| `win-s512-r8` | 512 / 8 | 99,37 % | 0,00609 | 0,0743 | 2,992 | ≈7,4 |
| `32w-s512-r128` | 512 / 128 | 99,43 % | **0,00569** | 0,0729 | 2,994 | ≈8,1 |
| `win-s1024-r8` | 1024 / 8 | 99,36 % | 0,00672 | 0,0766 | 2,994 | ≈10,3 |

**Korrektur der Bit-Spalte (§7.19, 2026-10-18).** Die Spalte „Bit/Wert (Auslieferung, Modell)" rechnet
`Basis + (16 − Basis)·(S+R)/L` mit `L ≈ 2044` (2K) — sie ist damit **zweifach zu optimistisch**: der Pool hält die
exakte Zelle **zusätzlich** zum quantisierten Wert (volle 16 Bit, nicht die Differenz), und die feste Spanne der
Auslieferung ist seit dem Ring `S + B`, nicht `S + R` (§7.19 des Plans: `R` Sink- plus `S + B` Ringzeilen mit
Zell-Tag, 19,5 MiB konstant bei 256/64). Richtig ist `Basis + 16·(S+B)/L`: für 256/64 auf `q4lm8v`
sind das **≈10,5 Bit bei 2K** (nicht 5,9), **4,8 bei 16K** und **4,2 bei 64K** (dort unter `int8`s 8,25). Die
**Qualitätszahlen und die Rangfolge der Arme bleiben** — die Byte-Absolutwerte dieser Tabelle gelten nur für 2K und
nur als Modell. Die Qualitätsmessung selbst lief auf der **Probe** (flacher Zwilling, 24 KiB/Zelle), deren Bytes für
alle Arme gleich waren; sie sagt nichts über die Auslieferungsbytes.

PPL fällt mehrfach unter den fp16-Wert (2,992 < 2,999) — derselbe Dämpfungs-Effekt wie in §7.11/§7.15. **KL ist der
Leser, PPL nicht.**

## Die Knie-Arme gegen `int8` (die Kernaussage)

Gepaart über dieselben 8 Segmente (`ΔKL` = linker Arm minus rechter), `int8` hat 8,25 Bit/Wert:

| Arm | Bit/Wert | mean KL | `int8` − Arm | t | Segmente up |
| --- | ---: | ---: | ---: | ---: | ---: |
| `8vw-s256-r64` | ≈5,9 | 0,00702 | −0,00199 | −1,92 | 2/8 |
| `32w-s256-r64` | ≈6,3 | 0,00782 | −0,00279 | −1,84 | 3/8 |
| `8vw-s512-r128` | ≈7,8 | 0,00623 | −0,00120 | −1,22 | 3/8 |
| `32w-s512-r128` | ≈8,1 | 0,00569 | −0,00066 | −0,88 | 4/8 |

**Kein gemessener Unterschied** zu `int8` in allen vier Fällen (`|t| ≤ 1,92`). Wohlgemerkt: „nicht unterscheidbar"
heißt nicht „gleich" — das 95 %-Intervall von `int8`−`8vw-s256-r64` reicht bis ≈ +0,0044 (≈ 88 % von `int8`s
eigenem Abstand zu fp16, 0,00503). Der billigste `int8`-nahe Punkt ist die **3-Bit-Basis bei ≈5,9 Bit** (≈29 %
unter `int8`), der teuerste mit ≈8,1 Bit noch ≈2 % darunter.

Antwortbereich (266 Zeilen) mit denselben Vorzeichen, aber kleiner: `int8`−`8vw-s256-r64` −0,00145 (t −1,07, 2/8),
`int8`−`32w-s512-r128` −0,00085 (t −0,56, 3/8).

## Gleiche Form, beide Basen (der Format-Vergleich)

| Form (S/R) | 3-Bit KL (Bit/Wert) | 4-Bit KL (Bit/Wert) | ΔKL (3 − 4) | t | Segmente up |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 8 | 0,00869 (≈4,8) | 0,00815 (≈5,3) | +0,00054 | +0,80 | 4/8 |
| 256 / 64 | 0,00702 (≈5,9) | 0,00782 (≈6,3) | −0,00080 | −1,26 | 3/8 |
| 512 / 8 | 0,00828 (≈7,1) | 0,00609 (≈7,4) | +0,00218 | +1,50 | 7/8 |
| 512 / 128 | 0,00623 (≈7,8) | 0,00569 (≈8,1) | +0,00054 | +0,75 | 6/8 |
| 1024 / 8 | 0,00741 (≈10,1) | 0,00672 (≈10,3) | +0,00069 | +0,54 | 6/8 |

In **allen fünf** Formen ist die Differenz nicht signifikant (`|t| ≤ 1,50`); nominell liegt die 4-Bit-Basis in 4 von
5 Formen vorn, in einer (`256/64`) die 3-Bit-Basis. Die 3-Bit-Basis kostet in **jeder** Form **0,24–0,47 Bit/Wert
weniger**. Pro Byte ist sie damit in dieser Messung **nirgends messbar schlechter**.

## Der Sink (R) ist eine eigene Achse

Bei festem `S = 512` und fester Basis verbessert mehr Sink (R 8 → 128) die KL, richtungskonstant über die Segmente:

| Paar | ΔKL | t | Segmente up | Antwortbereich |
| --- | ---: | ---: | ---: | --- |
| `8vw-s512-r8` − `8vw-s512-r128` | +0,00204 | +1,88 | 8/8 | +0,00221 (t +2,04, 7/8) |
| `win-s512-r8` − `32w-s512-r128` | +0,00258 | +1,50 | 7/8 | +0,00330 (t +1,83, 6/8) |

`R = 8` ist also zu wenig für dieses Testset; `R = 64` (`8vw-s256-r64`, 320 Zellen) liegt bei **weniger** Zellen
unter `8vw-s512-r8` (520 Zellen, `R = 8`): +0,00126 (t +1,61, 6/8), nicht signifikant. Knie innerhalb einer Familie:

| Paar (3-Bit) | ΔKL | t | Segmente up |
| --- | ---: | ---: | ---: |
| `8vw-s128-r8` − `8vw-s256-r64` | +0,00167 | +2,97 | 8/8 |
| `8vw-s256-r64` − `8vw-s512-r128` | +0,00078 | +1,98 | 7/8 |

## Was die Bytes kosten

Die **Probe** hält einen fp16-Zwilling des **ganzen** Kontexts (`qsa_win_bytes` über `max_cells`); der Leser bedient
daraus nur das Fenster — alle Arme haben dieselbe Fußspur, die Unterschiede sind reine **Lesepolitik**. Die
**Auslieferung** hält nur die (S+R) exakten Zellen fp16, der Rest bleibt quantisiert →
`Basis + (16 − Basis)·(S+R)/L`, L ≈ 2044 bei 2K (`Basis` = gespeicherte Bits: `q4lm8v` ≈4,0, `q4lm32` 4,5,
`int8` 8,25). Das ist eine **Rechnung, keine Messung**.

## Verdict

- **Beide Basen erreichen mit Knie-Fenster `int8`-Niveau** (alle vier `|t| ≤ 1,92`, nicht unterscheidbar), die
  3-Bit-Basis zuerst: `q4lm8v`+w(256/64) liegt bei ≈5,9 Bit, `int8` bei 8,25 — **≈29 % weniger Bytes**, ohne
  gemessenen Qualitätsverlust.
- **Das Cross-Over aus §7.15 (≈4,8–5,3 Bit) reproduziert sich nicht.** Die Knie-Punkte liegen 0,001–0,002 unter
  der r8-Kurve ihrer eigenen Familie; bei gleicher Form ist die 3-Bit-Basis in allen fünf Formen nicht schlechter
  und immer billiger. Pro Byte verliert sie in dieser Messung nirgends messbar, und der Vorsprung reicht bis ≈10 Bit.
  Ein scharfes Cross-Over läßt sich aus 8 Segmenten **nicht mehr** angeben: die Format-Differenz (≤0,0022, immer ns)
  ist von derselben Größe wie der S/R-Formeffekt darüber.
- **`R` ist ein eigener Hebel.** Bei festem `S = 512` bringt `R = 128` gegen `R = 8` 0,0020–0,0026 KL zurück,
  richtungskonstant (7–8 von 8 Segmenten). `R = 8` war in §7.13/§7.15 zu knapp gewählt.
- **Grenze.** Die Bit-Zahlen sind ein **Modell** (Pegel + Skalen-Overhead + `(16 − Basis)·(S+R)/L`, L ≈ 2044 bei
  2K); bei längerem Kontext sinkt der Fenster-Anteil, und die Qualität ist nur bei 2K gemessen. Das S/R-Raster ist
  grob (5 Formen, 8 Segmente); „nicht unterscheidbar" ist ein Ausbleiben eines Nachweises, keine Gleichheit.

Rohdaten: `compare.json`, `paired.json` (TSVs 17 MB/Arm, nicht eingecheckt). Begriffe (KL, ΔKL, t, Segment, Fenster-
und Quantisierungsnotation, Byte-Modell): `docs/KV_QUANT_WORKPACKAGE.md` §12 Glossar.
