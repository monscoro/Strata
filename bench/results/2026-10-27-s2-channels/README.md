# S2 – trägt die Kanalachse vor der Rotation?

**Status: gelaufen, Kill feuert.** Ein Probe-Lauf auf cwe-4k, 2026-10-27, AMD Radeon RX 9060 XT
(gfx1200, 16 GB), Windows 11, ROCm 7.1, Engine 0.1.47 Debug-Build (SHA-256 `F9058D56…860F20`).
32.746 Zeilen je QSA-Layer, 12 Layer, 3.072 Kanäle (8 weniger als die 32.754 Prompt-Token, eines je Segment). Katalogzeile:
`docs/paper/Ansatz_Improvements.md`, Leiter S2.

## Frage

`q4lm8v` rotiert K (und V) vor dem Schreiben mit einem Hadamard über `head_dim` = 256 und quantisiert
danach in 32er-Blöcken. Die Frage der Leiter: sitzt ein Tail, den die 4 Bit nicht tragen, auf der
**Kanalachse vor** dieser Rotation? Wenn ja, wäre A1 (per-channel-Skala für K vor dem FWHT, KIVIs
Achse) ein billiger Hebel. Wenn die Verteilung vor der Rotation nicht schwerer ist als danach, ist
die Achse in dieser Pipeline nicht da und A1 wird nicht gebaut.

## Aufbau

| Punkt | Wert |
|---|---|
| Engine | 0.1.47 + S2-Probe (Debug-Build, `build-hip-port\strata.exe`, SHA-256 `F9058D56…860F20`) |
| Modell | swift-1.5-iq3_xxs, `--kv q4lm8v`, `--prefill 512` |
| Passage | cwe-4k, alle 8 Segmente = 32.754 Token, `--max-context 4608` |
| Pfad | **batched Prefill** (`--short-read 0`), `STRATA_KV_S2_STATS=<Datei>`, `--allow-no-positions` |
| Auswertung | `tools/kv_s2_channels.py` |
| Kosten | ein Lauf, 3,5 min GPU-Zeit, kein Kernel geändert |

Die Probe (`include/strata/core/s2_probe.hpp`, Aufrufe in `src/prefill/prefill.cpp` und
`src/core/verify.cpp`) kopiert den K-Puffer an zwei Stellen: direkt vor dem `fwht256_inplace_cuda`
(nach RMS-Norm und RoPE) und direkt danach, jeweils `T·2` Zeilen à 256 Kanäle. Sie akkumuliert die
ersten vier Momente je Kanal (Mittel, Varianz, Schiefe, Exzess-Kurtosis) und `max|x|` und schreibt
alle 500 Token die kumulierten Momente. Ohne `STRATA_KV_S2_STATS` gibt es keinen D2H, keine Allokation
und keine Änderung am Append; der Default-Pfad ist unberührt.

Der Verify-Pfad ist mitgeprobt, aber **graph-capture-geschützt** (`cudaStreamIsCapturing`): ein D2H
innerhalb der Aufnahme bricht sie ab. Alle Zahlen hier kommen deshalb aus dem batched Prefill, nicht
aus den Fenstern. `--short-read` stand vorher über der Promptlänge und schickte jede Passage durch die
Fenster; der Lauf braucht `0`.

### Ein Fehler, der zuerst falsch gemessen hat

Der erste Rauchtest (511 Token) meldete Crest-Faktoren von 13–17 **vor** der Rotation gegen 3,1
danach. Das war kein Messwert, sondern ein Wettlauf: die `cudaMemcpy` lief auf dem Default-Stream und
las den Puffer, bevor die Berechnung auf `m.cs` fertig war. Die Kopie ist jetzt stream-geordnet
(`cudaStreamSynchronize` **vor** der Kopie), und die Probe prüft sich selbst: ein orthonormaler
Hadamard (Skala 1/16) erhält die Energie jeder Zeile. `sum E[x²]` über die 256 Kanäle ist auf allen
12 Layern **1,000000** (pre/post), Zeile 0 inbegriffen (`STRATA_KV_S2_DEBUG=1` druckt den Beleg).
Ohne diese Invariante wären die Zahlen unten nicht glaubwürdig.

Die Probe ist deterministisch: derselbe Lauf mit dem Build nach dem Aufräumen der
Hilfsfunktion ergab eine **byte-identische** Datei (SHA-256 `4994306E…038F7`), und ein
512-Token-Segment vor und nach dem Aufräumen ebenfalls (`15D22009…7BCEDC`).

## Entscheidungsregel (vor dem Lauf festgelegt)

> Vor der Rotation kein schwererer Tail als danach → A1 nicht bauen; KIVIs Achse ist in dieser
> Pipeline nicht da.

„Schwerer Tail" heißt hier: Exzess-Kurtosis und Crest-Faktor `max|x| / sd` je Kanal über die Token,
gemittelt über die Kanäle einer Gruppe, plus die Vorzeichenprobe über alle 3.072 Kanäle. Der
Hadamard mischt die Kanäle, die Gruppen (64 rotiert, 192 unrotiert) sind deshalb nur für die
**Pre**-Seite eine Dimensionsaussage.

## Ergebnis

| | pre (vor FWHT) | post (nach FWHT) |
|---|---|---|
| Exzess-Kurtosis, Mittel über alle Kanäle | **−0,18** | −0,14 |
| Crest `max|x| / sd`, Mittel | **3,72** | 3,93 |
| Kanäle mit schwererem Tail **pre** als post | **1.614 von 3.072 (52,5 %)** | — |

| Layer | Token | Kurtosis pre (rot / unrot) | Kurtosis post (rot / unrot) | Crest pre (rot / unrot) | Crest post (rot / unrot) | var-Spreizung | E pre/post |
|---|---|---|---|---|---|---|---|
| 3 | 32.746 | −0,53 / +0,01 | −0,07 / −0,12 | 3,15 / 3,97 | 4,08 / 3,99 | 71× | 1,000000 |
| 7 | 32.746 | −0,34 / +0,25 | −0,09 / −0,11 | 3,23 / 4,32 | 4,02 / 4,02 | 25× | 1,000000 |
| 11 | 32.746 | −0,41 / −0,14 | −0,11 / −0,12 | 3,33 / 3,94 | 4,04 / 3,99 | 20× | 1,000000 |
| 15 | 32.746 | −0,40 / −0,06 | −0,16 / −0,15 | 3,38 / 3,94 | 3,97 / 3,93 | 16× | 1,000000 |
| 19 | 32.746 | −0,15 / −0,16 | −0,10 / −0,16 | 3,60 / 3,82 | 3,90 / 3,87 | 23× | 1,000000 |
| 23 | 32.746 | −0,38 / −0,10 | −0,15 / −0,14 | 3,41 / 3,95 | 3,94 / 3,97 | 44× | 1,000000 |
| 27 | 32.746 | −0,26 / −0,13 | −0,15 / −0,14 | 3,53 / 4,01 | 4,01 / 3,96 | 39× | 1,000000 |
| 31 | 32.746 | −0,21 / −0,07 | −0,17 / −0,17 | 3,65 / 3,90 | 3,87 / 3,81 | 74× | 1,000000 |
| 35 | 32.746 | +0,04 / −0,14 | −0,14 / −0,18 | 3,86 / 3,91 | 3,88 / 3,82 | 152× | 1,000000 |
| 39 | 32.746 | −0,38 / −0,11 | −0,17 / −0,18 | 3,37 / 3,91 | 3,85 / 3,86 | 107× | 1,000000 |
| 43 | 32.746 | −0,13 / −0,06 | −0,10 / −0,12 | 3,76 / 4,01 | 3,92 / 3,91 | 7.267× | 1,000000 |
| 47 | 32.746 | −0,48 / +0,09 | −0,17 / −0,16 | 3,12 / 4,18 | 3,87 / 3,90 | 722× | 1,000000 |

### Der Kill feuert

Die Kanalachse **vor** der Rotation ist nicht schwerer als danach: die mittlere Exzess-Kurtosis ist
pre negativ (leicht unter der Normalverteilung), der mittlere Crest ist pre **kleiner** als post, und
die Vorzeichenprobe über alle Kanäle ist ein Münzwurf (52,5 %). Damit ist A1 (per-channel K vor dem
FWHT) **nicht gebaut** – die Regel war vor dem Lauf festgelegt, und ein Gleichstand zählt als Kill.

**Die eine Ausnahme, und warum sie kein Hebel ist.** 70 der 3.072 Kanäle (2,3 %) haben pre eine
Exzess-Kurtosis über +1, davon 49 in den 64 rotierten Dimensionen; der schlimmste liegt bei +6,78
(Layer 11). Die Rotation flacht **alle** davon ab (post ≤ +0,46) – der Hadamard tut also genau das,
wofür er da ist. 2,3 % der Kanäle sind kein Ausgabeort für Bits einer 5,0-Komposition.

**Die Breiten-Spreizung ist der Norm-Gewichtvektor, keine Datenachse.** Die Varianz je Kanal
spreizt pre um 16× bis 7.267×. Der Extremfall ist kein Dynamikbereich, sondern tote Kanäle: Layer 43
hat Kanäle mit `var_pre` = 0,001 (Median 1,70, Maximum 7,00), das sind Gewichte ≈ 0 der K-Norm. Nach
der Rotation liegen alle Kanäle zwischen 1,2 und 2,8 – die Rotation ebnet die Achse ein. Eine
per-channel-Skala vor dem FWHT wäre damit exakt das bekannte Norm-Gewicht, also eine feste
Transformation, die die Pipeline schon anwendet.

## Was der Lauf nicht sagt

- **Nichts über 32er-Blöcke innerhalb eines Token.** Gemessen sind Momente je Kanal über die Token;
  das Problem des 4-Bit-Blockquantisierers ist der Block innerhalb einer Zeile. Der Lauf sagt, dass
  die Kanalachse keine Tail-Achse ist – nicht, dass es keinen Tail gibt.
- **Nichts über 128K.** Die Passage ist 4K. Ob Sinks bei 128K die Kanalstatistik verschieben, ist
  offen; für A1 ist die Frage mit diesem Lauf beantwortet.
- **Keine Qualitätszahl.** Der Arm schreibt keine Positionen (`--short-read 0` nimmt den batched
  Pfad, `STRATA_LOGPOS` schreibt nur Dekodier-Positionen); KL kommt aus den normalen Armen.

## Dateien

- Rohdaten (nicht im Repo): `Strata-data/bench/2026-10-27-s2-channels.txt` (377 kB),
  `…-channels.json`, `…-s2-cwe4k.tsv.stderr.txt`
- Auswertung: `tools/kv_s2_channels.py --tsv <Datei> --json <Datei>`
- Probe: `include/strata/core/s2_probe.hpp`, `src/prefill/prefill.cpp`, `src/core/verify.cpp`
- Runner: `tools/quant_quality.py run --short-read 0 --allow-no-positions`
