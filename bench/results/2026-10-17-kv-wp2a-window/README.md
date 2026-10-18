# WP-2a Fenster auf Swift: das Fenster trägt (cwe-2k)

Design: `docs/KV_QUANT_WORKPACKAGE.md` §5.6.2; Arm gebaut in §7.12 (`--kv q4lm32w`, Gitter per
`STRATA_KV_WINDOW_S`/`_R`). Gemessen mit demselben Build, der den gestagten `q4_mode`-Fix aus §7.12 enthält;
die Anker `fp16` und `q4lm32` reproduzieren die WP-1-Läufe **bit-genau** (SHA256-gleich, s. unten).

- **Where:** one PC, RX 9060 XT (gfx1200, RDNA4, 16 GB), 63.9 GB RAM, HIP build of 2026-10-17 (§7.12; uncommitted).
- **What ran:** Qwen3.8-Flash-Next, **Swift IQ3_XXS** weight pack, weights fixed, only `--kv` + the window grid changed.
- **Test:** RULER CWE at 2K, 8 segments, ~2,048 tokens each, `data/quant_quality/cwe-2k/` (16,351 positions).
  Reference: the `fp16` arm of the same run (PPL 2.999). Segment times ~9 min/arm (engine load + 8 segments).
- **Nullpunkt:** `q4lm32w` mit S=0/R=0 ist `q4lm32` (der Leser benutzt den exakten Pool nie) — und die beiden
  Läufe sind **byte-gleich** (TSV SHA256-gleich), nicht nur statistisch gleich.

## Whole segment (16,351 positions)

| arm | S / R | top-1 | mean KL | KL / `q4lm32` | KL p99 | PPL | vs fp16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp16` | – | – | 0 | – | 0 | 2.999 | – |
| `q4lm32` | – | 99.19% | 0.01183 | 1.00x | 0.1464 | 3.012 | +0.43% |
| `q4lm32w` | 0 / 0 | 99.19% | 0.01183 | 1.00x | 0.1464 | 3.012 | +0.43% |
| `q4lm32w` | 128 / 8 | 99.41% | 0.00815 | 0.69x | 0.0856 | 2.994 | −0.17% |
| `q4lm32w` | 512 / 8 | 99.37% | **0.00609** | **0.51x** | 0.0743 | 2.992 | −0.23% |
| `q4lm32w` | 1024 / 8 | 99.36% | 0.00672 | 0.57x | 0.0766 | 2.994 | −0.17% |

Die PPL fällt unter den fp16-Wert (2.999) — der 4-Bit-Fehler wirkt hier wie Rauschen/Dämpfung, wie schon beim
Clip-Arm (§7.11) und WP-1 (`q4lm32t` 2.996). **PPL ist hier kein Qualitätsleser; KL ist es.**

## Paired over the 8 segments (the decision readout)

| pair | ΔKL | t | segments up |
| --- | ---: | ---: | ---: |
| `q4lm32` − `win 128/8` | +0.00368 | +7.16 | 8/8 |
| `q4lm32` − `win 512/8` | +0.00574 | +5.62 | 8/8 |
| `q4lm32` − `win 1024/8` | +0.00511 | +6.31 | 8/8 |
| `win 512/8` − `win 1024/8` | −0.00063 | −0.79 | 5/8 |
| `win 128/8` − `win 512/8` | +0.00206 | +2.48 | 7/8 |

Answer region (266 rows): `q4lm32`−`win 512/8` +0.00239 (t +3.90, 8/8), `q4lm32`−`win 1024/8` +0.00346
(t +3.20, 7/8), `q4lm32`−`win 128/8` +0.00149 (t +1.41, 6/8, schwach). Vorzeichen wie im Ganzen.

## Was die Bytes kosten

Die **Probe** hält einen fp16-Zwilling des **ganzen** Kontexts (`qsa_win_bytes` über `max_cells`; `kv_append_win`
schreibt jede angehängte Zelle), der Leser bedient daraus nur das Fenster. Damit ist die Fußspur *aller* Arme
gleich (~20,5 Bit/Wert = 4,5 q4 + 16 fp16-Zwilling) und die Qualitätsunterschiede sind reine **Lesepolitik** —
richtig für die Frage, aber **nicht** die Auslieferungs-Kosten.

Ein ausgeliefertes Fenster speichert nur die (S+R) exakten Zellen fp16 (Ring + Sinks), der Rest bleibt q4:
**4,5 + 11,5·(S+R)/L Bit/Wert**, mit L ≈ 2044 (Schnitt der gescorten Positionen):

| arm | Bit/Wert (Auslieferung) | mean KL |
| --- | ---: | ---: |
| `q4lm32` | 4,5 | 0.01183 |
| `win 128/8` | ~5,3 | 0.00815 |
| `win 512/8` | ~7,4 | 0.00609 |
| `win 1024/8` | ~10,3 | 0.00672 |
| `int8` (aus WP-1) | 8,25 | 0.00503 |

## Verdict

- **Das Fenster trägt, eindeutig.** Schon der Default (128/8) senkt die KL um 31 % (t +7.16), 512/8 halbiert sie
  (0,51×, t +5.62); **alle 8 Segmente** in dieselbe Richtung, im ganzen Text und im Antwortbereich.
- **Knie bei S ≈ 512.** 1024 bringt nichts mehr (512−1024: t −0.79, 5/8 = Gleichstand). Der Beitrag ist bis 512
  monoton, danach flach.
- **Gegen int8 ist `q4lm32w` dicht dran — bei weniger Bytes.** `int8` (8,25 Bit) hat KL 0,00503, `q4lm32w` (512/8,
  ~7,4 Bit, **10 % weniger Bits**) hat 0,00609 (**+21 % KL**), der Tail ist praktisch gleich (KL p99 0,0743 vs
  0,0733; PPL 2,992 vs 3,004). `int8` bleibt pro Byte ein wenig effizienter — bei *gleichen* Bytes läge das
  Fenster über int8, weil der Beitrag ab S ≈ 512 flach ist —, aber es ist kein „die 4-Bit schlägt int8"-Märchen
  *und* kein „das Fenster ist nur ein ≤3-Bit-Hebel". Bei 4 Bit ist es ein echter **Rate-Knopf**:
  4,5 → 5,3 (128/8) → 7,4 (512/8) Bit bei int8-naher Qualität. (`q4lm32w` 1024/8 ist bei 10,3 Bit sogar *schlechter*
  als 512/8 — mehr Bytes, mehr KL, also dominiert.)
- **Das ≤3-Bit-Zuhause bleibt die offene Frage.** Dort ist die Mitte billig genug, daß 16 Bit auf einem kleinen
  Fenster relativ wenig kosten — genau die Bedingung, unter die §8.5/§11.4 das Urteil „3 Bit tot" gestellt haben.
  Dafür braucht es `q4lm8v` (3,5 Bit) ± Fenster. — **Erledigt in WP-2b** (§7.15,
  `bench/results/2026-10-17-kv-wp2b-window3bit/`): das Fenster trägt dort (128/8 −51 %, 8/8) und macht die
  3-Bit-Basis im **Low-Byte-Bereich** (≈4,8–5,3 Bit) knapp konkurrenzfähig; ab ≈5 Bit dominiert die 4-Bit-Basis pro Byte
  (Cross-Over). — **Nachtrag §7.16** (`bench/results/2026-10-18-kv-wp2b-oscar-knee/`): mit dem OSCAR-Knie
  (`R = 64`, S 256) erreichen **beide** Basen `int8`-Niveau, die 3-Bit-Basis bei ≈5,9 statt 8,25 Bit, und die
  3-Bit-Basis verliert pro Byte in keiner der fünf gemessenen Formen meßbar — das Cross-Over oben gilt nur für das
  `R = 8`-Gitter.

Rohdaten: `compare.json`, `paired.json` (TSVs 17 MB/Arm, nicht eingecheckt).
