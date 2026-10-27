# S3(a) – 16/16 gegen 8/8 unter gleichen Bits

**Status: vorregistriert, Läufe laufen.** Katalogzeile: `docs/paper/Ansatz_Improvements.md`, A2 und Leiter S3(a).

## Frage

`q4lm8v` (der heutige Punkt) schreibt K mit der 16-Level-Tabelle und V mit der 8-Level-Tabelle, im
selben 4-Bit-Block und bei **denselben Bytes** (576 B je Zelle und Layer, 4,50 bit/cell). Der Split
ist eine Byte-/Codebook-Wahl, keine gemessene Sensitivität — die *Bit*-Variante („K will Bits") ist
auf Swift bereits falsifiziert (t +0,05, McNemar p = 1,000). A2 fragt die andere Richtung: **zählt
die Level-Zahl bei 4 Bit überhaupt?**

Zwei Arme, gleiche Bytes, gleicher Kernel, nur die Tabelle anders:

| Arm | `--kv` | Tabelle K | Tabelle V | bit/cell |
|---|---|---|---|---|
| 16/16 | `q4lm32` | Max-Lloyd 16 Level | Max-Lloyd 16 Level | 4,50 |
| 8/8 | `q4lm8` | Max-Lloyd 8 Level | Max-Lloyd 8 Level | 4,50 |
| Referenz | `int8` | 8 Bit + Skala | 8 Bit + Skala | 8,25 |

Die Richtung wird **nicht** vorhergesagt. KIVI sagt sie voraus; das darf die Lesung nicht vorwegnehmen.

## Aufbau

| Punkt | Wert |
|---|---|
| Engine | 0.1.47 Debug-Build `build-hip-port\strata.exe`, SHA-256 `F9058D56…860F20` (S2-Probe inaktiv, Env ungesetzt) |
| Modell | swift-1.5-iq3_xxs, `--prefill 512`, `--prompt-cache 0` |
| Passage | cwe-4k, alle 8 Segmente, 32.746 Positionen je Arm, teacher-forced |
| Logs | `STRATA_LOGPOS_TOPK=256` je Position |
| Fenster | **keins** (Bulk allein; `STRATA_KV_WINDOW` ungesetzt) |
| Auswertung | `tools/quant_quality.py compare --paired --kl-topk 256` |
| Kosten | 3 Läufe à ~19 min GPU |

Alle drei Arme laufen nacheinander auf demselben Build und derselben Maschine, damit die Paarung
nicht über Builds hinweg geht. Kein Fenster, damit die Frage nicht mit der Ringgröße vermischt wird.

## Entscheidungsregeln (vor dem Lauf festgelegt)

1. **A2, die eigentliche Frage.** Gepaarte Differenz `KL(8/8) − KL(16/16)` über alle Positionen, mit
   Vorzeichen und 95-%-Grenze, plus die Segmentbilanz. Ist die Grenze um 0, ist die Level-Zahl bei
   4 Bit **kein Hebel** — dann ist der Split in `q4lm8v` eine Byte-Wahl ohne Qualitätsargument, und
   A5 (andere 16-Level-Tabelle, gleiche Bytes) hat keine Grundlage, auf der sie noch etwas holen
   könnte.
2. **K1 je Arm** (Leiter, Latte 1): gepaarte Mean-KL (Arm − `int8`) ≤ 0, obere 95-%-Grenze
   < **+0,001 nats**, `int8` führt in höchstens 4 von 8 Segmenten.
3. **S3-Kill** (Leiter): erreicht **kein** Arm K1 bei ≤ 6,0 bit, ist der Bulk nicht der Hebel — kein
   128K, zurück zu S1 (Overlay-Ort) oder Abbruch. Beide Arme liegen mit 4,50 bit weit unter 6,0.
4. Ein **Gleichstand zwischen 16/16 und 8/8 zählt als Kill für A2**, nicht für den ganzen Bulk: der
   Kill in (3) hängt an K1 gegen `int8`, nicht an der Differenz der beiden Arme.

Der Anker für die Größenordnung: `int8` liegt auf cwe-4k bei KL 0,01688 (top-64) und das heutige
Fenster bei 0,02090; ein Unterschied zwischen 16/16 und 8/8 von weniger als ~0,001 nats wäre kleiner
als der Abstand, den das Fenster heute kostet.

## Ergebnis

Drei Arme, 32.746 Positionen je Arm, derselbe Build, dieselben Einstellungen. KL ist
KL(`int8` ‖ Arm) über `int8`s Top-k, wie in K1 definiert; „vs `int8`" ist der gepaarte
Mittelwert über die Positionen, nicht über die Segmente.

| Arm | bit/cell | KL top-64 | KL top-256 | vs `int8` (top-256) | t | 95-%-Schranke | `int8` führt |
|---|---|---|---|---|---|---|---|
| `int8` (Referenz) | 8,25 | — | — | 0 | — | — | — |
| 16/16 (`q4lm32`) | 4,50 | 0,02558 | **0,02734** | +0,02734 | 14,72 | 0,03040 | 8/8 |
| 8/8 (`q4lm8`) | 4,50 | 0,03891 | **0,04185** | +0,04185 | 20,20 | 0,04526 | 8/8 |

Top-1-Treffer 98,61 % (16/16) gegen 98,12 % (8/8).

### A2: die Level-Zahl ist ein Hebel, kein Gleichstand

Gepaart je Position, gleiche Bytes, nur die Tabelle anders:

| Differenz (top-256) | Mittel | 95-%-Intervall | t | Segmente |
|---|---|---|---|---|
| 16/16 − 8/8 | **−0,01451** | [−0,01842, −0,01060] | −7,28 | 8/8 für 16/16 |

Im Antwortbereich ist der Abstand größer als im Ganzsegment: 16/16 − 8/8 = −0,02094
(Segment-Mittel, `--paired`). Zum Vergleich: der Preis des heutigen Fensters liegt bei
+0,0045 (top-256) beziehungsweise +0,0040 (top-64) gegen `int8` — die Level-Tabelle ist
also **etwa 3,2- bis 3,6-mal so viel wert wie das Fenster heute kostet**, bei null
zusätzlichen Bytes.

Damit ist die Regel „Gleichstand zählt als Kill für A2" **nicht** eingetreten. Die
8-Level-Tabelle ist keine Byte-Variante auf Augenhöhe: sie ist messbar schlechter. Der
Level-Split in `q4lm8v` (16 auf K, 8 auf V) sitzt damit auf dem billigen Ende eines
Hebels, der 0,0145 nats trägt — und die V-Seite ist genau die Hälfte, die heute 8 Level
bekommt. Ob dieser Schluss auf V zutrifft, sagt dieser Lauf **nicht**: dafür fehlt der
Arm 16/8 (das heutige `q4lm8v` ohne Fenster) auf demselben Build.

### K1 und der S3-Kill

| Arm | Mean-KL vs `int8` | 95-%-Schranke | `int8` führt | K1 |
|---|---|---|---|---|
| 16/16 | +0,02734 | 0,03040 | 8/8 | **nein** |
| 8/8 | +0,04185 | 0,04526 | 8/8 | **nein** |

Beide Arme verfehlen K1 deutlich (Schwelle: obere Schranke < +0,001 nats). Nach der
vorregistrierten Regel feuert der **S3-Kill**: kein Arm erreicht K1 bei ≤ 6,0 bit, also
kein 128K.

**Diese Regel ist an dieser Stelle nicht informativ, und das ist ein Befund über den
Plan, nicht über den Bulk.** Ein nackter 4,50-bit-Arm ohne exakten Ring kann `int8`
(8,25 bit) nicht einholen; der heutige Produktpunkt schafft es mit Ring und 5,29 bit
ebenfalls nicht (0,02090 gegen 0,01688, top-64). Der Kill feuert damit auf einer
Tatsache, die vor dem Lauf feststand, und er unterscheidet nicht zwischen „der Bulk ist
der falsche Hebel" und „der Ring ist zu klein". Was der Lauf **wirklich** sagt, steht
in A2: die Tabelle ist ein Hebel bei gleichen Bytes.

### Nachtrag: die 2x2 und die Skala (vorregistriert 2026-10-27, vor den Läufen)

A2 hat die Endpunkte gemessen, aber der Punkt selbst ist 16/8. Drei weitere Arme, alle
bei denselben 576 B/cell und demselben Build, schließen die 2x2-Tabelle und die zweite
Hälfte der Codebook-Frage:

| Arm | `--kv` | K | V | Skala |
|---|---|---|---|---|
| 16/8 (heutiger Punkt, ohne Fenster) | `q4lm8v` | 16 Level | 8 Level | `max\|x\|/87` |
| 8/16 | `q4lm8k` | 8 Level | 16 Level | `max\|x\|/87` |
| 16/16 + LS | `q4lm32s` | 16 Level | 16 Level | Least Squares je Block |

**Fragen.** (1) Trägt der 0,0145-Vorsprung von 16/16 die K-Seite, die V-Seite oder
beide? Das entscheidet, welche Tabelle im Produktpunkt getauscht wird. (2) Ist die
Least-Squares-Skala bei gleichen Bytes besser als `max|x|/87`? Der Blockfehler ist es
bereits (2,551e-02 gegen 2,740e-02), die KL auf cwe-4k ist ungemessen.

**Regel.** Ein Arm ersetzt den Bulk des Produktpunkts nur, wenn er bei **gleichen
Bytes** besser ist als 16/8 mit einem 95-%-Intervall, das die 0 ausschließt, und wenn
K5 (Prefill nicht mehr als 10 % unter `int8`) haelt. Die Richtung wird nicht
vorhergesagt: KIVI sagt "K ist empfindlicher" voraus, A2s Vorgeschichte hat "K will
Bits" auf Swift bereits falsifiziert.

### Die 2x2 und die Skala: Ergebnis

Alle fünf Arme bei 576 B/cell, ein Build, 32.746 Positionen, `max|x|/87` als Skala außer
beim letzten:

| Arm | K | V | Skala | KL top-64 | KL top-256 |
|---|---|---|---|---|---|
| 16/16 + LS (`q4lm32s`) | 16 | 16 | Least Squares | **0,02303** | **0,02457** |
| 16/16 (`q4lm32`) | 16 | 16 | `max\|x\|/87` | 0,02558 | 0,02734 |
| 8/16 (`q4lm8k`) | 8 | 16 | `max\|x\|/87` | 0,03051 | 0,03240 |
| 16/8 (`q4lm8v`, heutiger Bulk) | 16 | 8 | `max\|x\|/87` | 0,03081 | 0,03281 |
| 8/8 (`q4lm8`) | 8 | 8 | `max\|x\|/87` | 0,03891 | 0,04185 |

Effekte, gepaart je Position, top-256:

| Effekt | Mittel | 95-%-Intervall | t | Urteil |
|---|---|---|---|---|
| K 16 -> 8 (16/16 - 8/16) | -0,00506 | [-0,00902, -0,00110] | -2,50 | 16 Level besser |
| V 16 -> 8 (16/16 - 16/8) | -0,00547 | [-0,00922, -0,00171] | -2,86 | 16 Level besser |
| beide (16/16 - 8/8) | -0,01451 | [-0,01842, -0,01060] | -7,28 | 16 Level besser |
| 16/8 - 8/16 | +0,00041 | [-0,00329, +0,00412] | +0,22 | **keine Richtung** |
| LS - max\|x\|/87 (16/16) | -0,00277 | [-0,00608, +0,00053] | +1,64 | nicht gesichert |

Drei Befunde:

1. **Der Split ist qualitativ beliebig.** 16/8 und 8/16 liegen 0,0004 nats auseinander,
   das Intervall enthält die 0, und die Segmentbilanz ist 4/8 gegen 4/8. K und V sind
   **gleich** empfindlich (0,0051 gegen 0,0055); „K braucht die breitere Tabelle" ist in
   dieser Pipeline nicht messbar. `q4lm8v` gibt der V-Seite also 8 Level, ohne dass
   irgendetwas das rechtfertigt — und zahlt dafür 0,0055 nats gegen 16/16.
2. **Die Interaktion ist überadditiv:** 0,0051 + 0,0055 = 0,0106, gemessen 0,0145. Wer
   beide Seiten auf 8 Level setzt, verliert mehr als die Summe der Einzelverluste.
3. **Die Least-Squares-Skala ist ein Vielleicht.** Der Mittelwert ist 0,0028 besser, das
   Intervall enthält aber die 0 (t = 1,64), der Median der Differenz ist 0,000, und nur
   54,4 % der Positionen werden besser. Der Median-KL fällt von 0,00158 auf 0,00145
   (−8 %). Ohne Bytes zu kosten ist sie ein Kandidat, aber kein gesicherter Gewinn.

### Die Lese-Seite ist bereits gemessen: Parität, kein Hebel

Aus dem 2026-10-25-Lauf (`bench/results/2026-10-25-tag-quality/`, gleicher Satz, gleiche
Positionen, 16/8-Bulk, 5,29 bit), hier als gepaarte Differenz nachgerechnet:

| Arm | KL vs fp16 top-256 | gegen `int8` |
|---|---|---|
| `int8` | 0,01774 | — |
| Fenster 256/64 | 0,02223 | +0,00449 [-0,00745, -0,00153] |
| Tag-Read (gleiche Bytes) | 0,02107 | +0,00332 [-0,00620, -0,00045] |
| Tag-Read - Fenster | — | +0,00117 [-0,00162, +0,00395], t +0,82 |

Der Tag-Read ist **nicht** signifikant besser als das Fenster (t = 0,82, 3/8 Segmente).
Der Read ist damit kein Hebel; der Bulk ist einer.

### Was das für das Ziel bedeutet

**Das 5,0-bit/cell-Ziel ist am 2026-10-27 zurückgestellt** (Entscheidung des Nutzers):
ohne deutlich bessere Rotationsansätze und Fehlerkorrekturen ist es mit diesem Codec
nicht erreichbar. Die Leiter S0-S5 wird nicht weiter auf 5,0 hin verfolgt; der Branch
maximiert zuerst **Qualität bei gleichen Bytes**, danach erst Bytes.

Die gemessene Rangfolge der Hebel bei gleichen Bytes:

1. **Bulk-Tabelle: 0,0055 bis 0,0082 nats** (16/8 -> 16/16 -> 16/16+LS). Der heutige
   Punkt verschenkt davon 0,0055 ohne Gegenleistung.
2. **Read (Tag statt Fenster): 0,0012 nats, nicht signifikant.** Kein Hebel.
3. Alles andere (Rotation, per-channel, Level-Tabelle jenseits 16): von S2 und A5
   her ohne gemessene Grundlage.

Der nächste Arm ist damit der Produktpunkt mit dem besseren Bulk: **16/16 (+LS) + Fenster
256/64**, gleiche Bytes wie heute (5,29 bit/cell, 1.016 MiB bei 128K). Die Projektion aus
den Zahlen oben — Bulk −0,0055, Fenster kostet +0,0045 — landet bei oder unter `int8`;
das ist eine **Projektion, keine Messung**, und der Ring hält die letzten 256 Zellen
exakt, wo der Bulk nichts beiträgt. Deshalb wird sie gemessen, nicht behauptet.

## Dateien

- Rohdaten (nicht im Repo): `Strata-data/bench/2026-10-27-s3/{int8,q4lm32,q4lm8}.tsv`
  und `.seg.json`, `compare-{64,256}.json`, `paired-{64,256}.json`, `k1-{64,256}.json`
- K1-Test je Position: `tools/kv_k1.py --ref int8.tsv --run … --kl-topk 256`
- Segment-/Antwortbereich-Tabellen: `tools/quant_quality.py compare … --paired --ids …`
