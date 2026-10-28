# G1: drei Merkmale gegen vier Kalibrierungspaare

**Status: vorregistriert (2026-10-28), Läufe sind Analyse vorhandener Daten, kein GPU-Job.**
Katalogzeile: `docs/paper/Ansatz_Improvements.md`, Arbeitsplan G1
(`docs/research/BEYOND_KL_QUANT_FIDELITY.md` §3 und §4).

Dieses Dokument legt **vor** der Auswertung fest, was gerechnet wird, wie gepaart wird und wann ein
Merkmal fällt. Die Daten liegen seit dem 2026-10-25/27/28 in `Strata-data/bench/` und wurden für
andere Fragen schon ausgewertet; neu ist nur, dass dieselben Positionen gegen die Kalibrierungspaare
gestellt werden.

## Frage

Tragen ΔNLL, Margin-Flip und Region etwas, das Mean KL nicht schon sagt? Und wenn nicht: reichen
Mean KL und P99, wie der Arbeitsplan es als mögliches Ergebnis vorsieht?

## Daten

Zwanzig Läufe, fünfzehn Konfigurationen, 32.746 Positionen je Arm, `cwe-4k`, teacher-forced,
`--topk 256`. Anker ist `fp16` (`2026-10-27-s3/fp16.tsv`). Die Zuordnung Name → Datei steht in
`bench/fixtures/beyond-kl-calibration.json`, weil die Rohdaten außerhalb des Repos liegen.

| Verzeichnis | Arme |
|---|---|
| `2026-10-25-s0` | `int8`, `plain`, `win256`, `fp16` |
| `2026-10-27-s3` | `fp16`, `int8`, `q4lm32`, `q4lm32s`, `q4lm32sw`, `q4lm32w`, `q4lm8`, `q4lm8k`, `q4lm8v` |
| `2026-10-28-ring` | `s320r0`, `s192r128`, `s64r256` |
| `2026-10-25-tag-quality` | `int8`, `tag`, `win256`, `fp16` |

Für die Paare wird durchweg der **S3-Satz** benutzt (gleicher Build, gleiche Positionen). `int8`
existiert dreimal; die drei Dateien sind gleich groß, aber nur die S3-Datei ist mit der Leiter
zusammen entstanden.

## Definitionen (je Position q, Referenz `fp16`)

| Merkmal | Definition | Einheit |
|---|---|---|
| `kl64` / `kl256` | `KL(fp16‖Arm)` über die top-64 / top-256 der Referenz, Floor-Substitution wie in `kl_between` | nats |
| `dnll` | `logp_fp16(y) − logp_Arm(y)`, `y` = das Text-Token. Positiv = Arm schlechter | nats |
| `flip` | 1, wenn `argmax(Arm) ≠ argmax(fp16)` | Rate |
| `flip_m1` | 1, wenn `flip` **und** der Referenz-Margin `m = lp₁ − lp₂` ≥ 1 nat | Rate |
| `floor` | 1, wenn die von der Referenz auf die Substitution umgelegte Masse `Σ p_ref(K_ref \\ K_arm)` ≥ 0,01 ist | Rate |
| `cvar99` | Mittel der schlechtesten 1 % der `kl256`-Werte eines Segments | nats |
| `p99` | 99. Perzentil der `kl256`-Werte eines Segments | nats |
| `region_excess` | Mittel `kl256` **im verdrängten** Bereich `[R, n−S)` minus Mittel **im exakten** (Sinks `q < R` plus letzte `S`) | nats |

**Zur Region, offen gesagt:** das Label ist eine **Positionsfunktion**, keine Eigenschaft des Arms.
`R = 64`, `S = 256` sind die Geometrie des Referenz-Fensterarms; `n` ist die Segmentlänge. Damit ist
„außerhalb" bei festem `n` gleichbedeutend mit „weiter hinten", und `region_excess` kann sich als
bloß umskalierte Position erweisen. Genau das prüft die Redundanzmatrix. Ein Arm ohne Fenster
bekommt dasselbe Label — dann ist es ein reiner Distanz-Split, und das steht im Ergebnis.

**Eine Definition wurde nach dem ersten Probelauf korrigiert, und zwar bevor ein Merkmal bewertet
war.** Die erste Fassung von `floor` fragte, ob die top-256-Menge der Referenz in der des Arms
enthalten ist. Das ist auf den ersten zwei geprüften Armen an **100 %** der Positionen wahr: zwei
Listen zu je 256 aus 248.320 Token stimmen praktisch nie als Menge überein. Eine Konstante kann
nicht diskriminieren, also misst die Definition jetzt die **Masse**, die auf die Substitution
umgelegt wird, und zählt Positionen ab 1 %. Das ist eine Korrektur der Definition, kein Ergebnis —
kein Merkmal war zu diesem Zeitpunkt gegen ein Paar gerechnet.

## Aggregation und Statistik

- **Einheit ist das Segment** (n = 8), nie die Position. Je Segment wird der Merkmalswert gemittelt
  (bei Raten: der Anteil), dann werden die acht Segmentwerte gepaart.
- Gepaarte Differenz `d_s = F(A)_s − F(B)_s`; `t` mit df = 7; 95-%-Intervall mit `t_{0.975,7} =
  2,365`; einseitige Grenze bei echter Parität `1,895 · sd/√8`.
- Ein Merkmal ist **aufgelöst** für ein Paar, wenn `|Δ|` die einseitige Grenze überschreitet.

## Die Prüfregel (unverändert aus dem Arbeitsplan §3)

Ein Merkmal wird nur aufgenommen, wenn es **alle vier** besteht:

| Paar | Wahrheit | Prüfung |
|---|---|---|
| `q4lm32w` vs `int8` | Parität (+0,00002 top-64 / +0,00034 top-256; `int8` führt 4/8 bzw. 5/8) | 95-%-Intervall enthält die 0 |
| `s320r0` vs `q4lm32w` | Verlust (+0,00394 [+0,00097, +0,00691] top-256, Arm besser in 0/8) | Δ > 0 **und** Intervall schließt die 0 aus |
| `q4lm8v` vs `q4lm8k` | kein Unterschied (−0,00030, Intervall um 0) | 95-%-Intervall enthält die 0 |
| Leiter `q4lm8` > `q4lm8k` ≈ `q4lm8v` > `q4lm32` > `q4lm32w` ≈ `int8` | Ordnung (top-64: 0,04127 / 0,03212 / 0,03182 / 0,02512 / 0,01689 / 0,01688) | jede der fünf Stufen richtig: `>` wenn Δ über der Grenze, `≈` wenn darunter |

Danach die **Redundanzprüfung**: `|Spearman ρ| ≥ 0,9` gegen Mean KL über die fünfzehn
Konfigurationen → nicht aufnehmen, es sei denn der Zusatzbeitrag ist in der Partialbetrachtung
nachweisbar.

## Was ein Nullresultat heißt

Fällt jedes Merkmal, lautet das Ergebnis: **Mean KL und P99 reichen für die Fragen, die dieser Korpus
beantworten kann**, und das Dashboard entfällt. Das ist ein gültiges Ergebnis (Auftrag §13.10), kein
Fehlschlag — und es ist nach Lage der Daten das wahrscheinlichere, weil die Auflösung 0,0020 nats
beträgt und die Merkmale auf denselben Positionen mit derselben Streuung leben.

## Was diese Stufe nicht kann

- **Kein funktionales Ground Truth.** Alle vier Paare sind KL-Aussagen. Ein Merkmal, das sie besteht,
  ist damit kalibriert, nicht validiert. Das ist G3/G4.
- **Keine Auflösung unter 0,0020.** Ein Merkmal kann die Grenze nicht verschieben, nur die Zuordnung
  verbessern.
- **Trunkierung.** Alle Merkmale außer `dnll` leben auf der top-256-Stütze und erben M1/CB5. Der
  Vollvokabular-Anker ist G2, nicht G1.
- **Acht Segmente.** Jedes „besteht" heißt: in dieser Stichprobe, mit df = 7.
