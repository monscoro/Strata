# S3(a) – Level-Tabelle, Skala und Fenster bei gleichen Bytes

**Status: gemessen (2026-10-27/28), neun Arme auf einem Build.** Katalogzeile:
`docs/paper/Ansatz_Improvements.md`, A2 und Leiter S3(a).

## Frage

`q4lm8v` (der heutige Bulk) schreibt K mit der 16-Level-Tabelle und V mit der 8-Level-Tabelle, im
selben 4-Bit-Block und bei **denselben Bytes** (576 B je Zelle und Layer, 4,50 bit/cell). Der Split
ist eine Byte-/Codebook-Wahl, keine gemessene Sensitivität — die *Bit*-Variante („K will Bits") ist
auf Swift bereits falsifiziert (t +0,05, McNemar p = 1,000). A2 fragt die andere Richtung: **zählt
die Level-Zahl bei 4 Bit überhaupt?**

| Arm | `--kv` | Tabelle K | Tabelle V | Skala | bit/cell |
|---|---|---|---|---|---|
| 16/16 | `q4lm32` | Max-Lloyd 16 | Max-Lloyd 16 | `max\|x\|/87` | 4,50 |
| 16/16 + LS | `q4lm32s` | Max-Lloyd 16 | Max-Lloyd 16 | Least Squares je Block | 4,50 |
| 8/16 | `q4lm8k` | Max-Lloyd 8 | Max-Lloyd 16 | `max\|x\|/87` | 4,50 |
| 16/8 | `q4lm8v` | Max-Lloyd 16 | Max-Lloyd 8 | `max\|x\|/87` | 4,50 |
| 8/8 | `q4lm8` | Max-Lloyd 8 | Max-Lloyd 8 | `max\|x\|/87` | 4,50 |
| 16/16 + Fenster | `q4lm32` + `STRATA_KV_WINDOW=1` (256/64) | 16 | 16 | `max\|x\|/87` | 5,29 |
| 16/16 + LS + Fenster | `q4lm32s` + dito | 16 | 16 | Least Squares | 5,29 |
| Referenz | `fp16` | — | — | — | 16,50 |
| Referenz | `int8` | 8 Bit | 8 Bit | 8 Bit + Skala | 8,25 |

Die Richtung wurde **nicht** vorhergesagt. KIVI sagt sie voraus; das darf die Lesung nicht vorwegnehmen.

## Aufbau

| Punkt | Wert |
|---|---|
| Engine | 0.1.47 Debug-Build `build-hip-port\strata.exe`, SHA-256 `F9058D56…860F20`, ROCm 7.1 |
| Modell | swift-1.5-iq3_xxs (`Strata-data\models\swift-IQ3_XXS`), `--prefill 512`, `--prompt-cache 0` |
| Passage | cwe-4k, alle 8 Segmente, 32.746 Positionen je Arm, teacher-forced |
| Logs | `STRATA_LOGPOS_TOPK=256` je Position |
| Fenster | nur bei den beiden `…w`-Armen; `STRATA_KV_WINDOW_S=256 _R=64` |
| Auswertung | `tools/kv_k1.py` (gepaart je Position), `tools/quant_quality.py compare --paired` |
| Config | `bench/configs/cwe4k-swift-iq3_xxs.json` (festgeschrieben, `env: {}`) |

Alle Arme laufen nacheinander auf demselben Build und derselben Maschine, damit die Paarung nicht
über Builds hinweg geht.

**Warum die Config festgeschrieben ist.** Die App-Config `strata-swift-iq3_xxs.json` wurde am
2026-10-27 auf ein anderes Modell (flash-next abliterated `IQ3_XXS-Q2_0`) umgestellt, mit
`STRATA_KV_WINDOW=1` in ihrem `env`-Block. Beides hätte die Messung stillschweigend verändert: ein
fremdes Modell, dessen KL mit den acht Armen nicht vergleichbar ist, und ein Fenster, das in
`seg.json` nicht auftaucht, weil dort nur die `--env`-Flags vom Kommandozeilenaufruf stehen. Der
erste fp16-Versuch lief deshalb gegen den falschen Modellpfad und lieferte keine Positionen; die
Messung liegt jetzt auf einer eigenen, eingefrorenen Config.

## Entscheidungsregeln (vor dem Lauf festgelegt)

1. **A2, die eigentliche Frage.** Gepaarte Differenz `KL(8/8) − KL(16/16)` über alle Positionen, mit
   Vorzeichen und 95-%-Grenze, plus die Segmentbilanz. Ist die Grenze um 0, ist die Level-Zahl bei
   4 Bit **kein Hebel**.
2. **K1 je Arm** (Leiter, Latte 1): gepaarte Mean-KL (Arm − `int8`) ≤ 0, obere 95-%-Grenze
   < **+0,001 nats**, `int8` führt in höchstens 4 von 8 Segmenten.
3. **S3-Kill** (Leiter): erreicht **kein** Arm K1 bei ≤ 6,0 bit, ist der Bulk nicht der Hebel.
4. Ein **Gleichstand zwischen 16/16 und 8/8 zählt als Kill für A2**, nicht für den ganzen Bulk.

**Der Anker wurde nach dem ersten Ergebnis gewechselt.** Die Läufe gegen `int8` beantworten A2
korrekt (beide Arme gegen dieselbe Referenz, die Differenz ist gepaart), aber sie messen
`KL(int8‖Arm)`, nicht `KL(fp16‖Arm)`. Das ist eine andere Größe: sie bestraft Fehler, die `int8`
selbst schon macht, nicht. Der dokumentierte Anker (`int8` = 0,01688 auf cwe-4k) und die
Fensterzahlen von 2026-10-25 sind in `KL(fp16‖Arm)` gerechnet, also wurde ein `fp16`-Arm auf
denselben Build gelegt und **alle** Arme gegen ihn nachgerechnet. Die vorregistrierten Regeln sind
unverändert; nur die Einheit ist die richtige geworden. Der Wechsel ist hier offen notiert, weil er
nach Sicht der ersten Zahlen passierte.

## Ergebnis

### Der Anker: neun Arme in `KL(fp16‖Arm)`

| Arm | bit/cell | KL top-64 | KL top-256 |
|---|---|---|---|
| `int8` | 8,25 | **0,01688** | 0,01774 |
| `q4lm32` + Fenster 256/64 | 5,29 | **0,01689** | 0,01809 |
| `q4lm32s` + Fenster 256/64 | 5,29 | 0,01881 | 0,02017 |
| `q4lm8v` + Fenster 256/64 (heutiger Produktpunkt, 2026-10-25) | 5,29 | 0,02090 | 0,02223 |
| `q4lm32s` (16/16 + LS) | 4,50 | 0,02420 | 0,02634 |
| `q4lm32` (16/16) | 4,50 | 0,02512 | 0,02667 |
| `q4lm8v` (16/8, heutiger Bulk) | 4,50 | 0,03182 | 0,03387 |
| `q4lm8k` (8/16) | 4,50 | 0,03212 | 0,03417 |
| `q4lm8` (8/8) | 4,50 | 0,04127 | 0,04414 |

**Der Build reproduziert den Altbestand auf fünf Stellen.** `int8` = 0,01688 (top-64) und
`q4lm8v` = 0,03182 sind genau die Zahlen aus dem 2026-10-25-Lauf, gemessen auf einem anderen Build
und einer anderen Config. Deshalb ist der Vergleich mit dem dortigen Fensterarm (0,02090) zulässig,
obwohl er nicht heute gelaufen ist.

### A2: die Level-Zahl ist ein Hebel, kein Gleichstand

Gepaart je Position, gleiche Bytes, nur die Tabelle anders:

| Differenz | Mittel | 95-%-Intervall | t | Segmente |
|---|---|---|---|---|
| 16/16 − 8/8, top-256 | **−0,01748** | [−0,02133, −0,01363] | −8,89 | 16/16 in 8/8 |
| 16/16 − 8/8, top-64 | −0,01615 | [−0,01972, −0,01258] | −8,86 | 16/16 in 8/8 |

(Gegen den `int8`-Anker waren es −0,01451; die Differenz hängt von der Referenz ab, und `fp16` ist
die strengere. Die Aussage ist in beiden Ankern dieselbe.)

Damit ist die Regel „Gleichstand zählt als Kill für A2" **nicht** eingetreten. Die 8-Level-Tabelle
ist keine Byte-Variante auf Augenhöhe: sie kostet 0,0175 nats, das ist viermal der Preis des
heutigen Fensters (0,0045). Top-1-Treffer 98,61 % (16/16) gegen 98,12 % (8/8).

### Die 2×2: K und V sind gleich empfindlich

| Effekt (top-256) | Mittel | 95-%-Intervall | t | Urteil |
|---|---|---|---|---|
| K 16 → 8 (16/16 − 8/16) | −0,00751 | [−0,01121, −0,00381] | −3,97 | 16 Level besser |
| V 16 → 8 (16/16 − 16/8) | −0,00721 | [−0,01066, −0,00376] | −4,10 | 16 Level besser |
| beide (16/16 − 8/8) | −0,01748 | [−0,02133, −0,01363] | −8,89 | 16 Level besser |
| 16/8 − 8/16 | −0,00030 | [−0,00357, +0,00297] | −0,18 | **keine Richtung** |
| LS − `max\|x\|/87` (16/16) | +0,00033 | [−0,00274, +0,00340] | +0,21 | kein Hebel |
| Fenster (16/16 − 16/16+Fenster) | +0,00858 | [+0,00544, +0,01171] | +5,36 | Fenster hilft |

Drei Befunde:

1. **Der Split ist qualitativ beliebig.** 16/8 und 8/16 liegen 0,0003 nats auseinander, das
   Intervall enthält die 0, die Segmentbilanz ist 4/8 gegen 4/8. K und V sind **gleich** empfindlich
   (0,0075 gegen 0,0072); „K braucht die breitere Tabelle" ist in dieser Pipeline nicht messbar.
   `q4lm8v` gibt der V-Seite also 8 Level, ohne dass etwas das rechtfertigt — und zahlt dafür
   0,0072 nats gegen 16/16.
2. **Die Interaktion ist überadditiv:** 0,0075 + 0,0072 = 0,0147, gemessen 0,0175. Wer beide Seiten
   auf 8 Level setzt, verliert ein Fünftel mehr als die Summe der Einzelverluste.
3. **Die Least-Squares-Skala ist kein Hebel.** 0,0003 nats, Intervall um 0, t = 0,21.

**Korrektur zu einer früheren Aussage in diesem Report.** Die LS-Skala sah gegen den
`int8`-Anker 0,0028 besser aus (t = 1,64, Median-KL 0,00158 → 0,00145). Gegen `fp16` bleiben davon
0,0003 (top-256) beziehungsweise 0,0009 (top-64) übrig, beide nicht gesichert. Der scheinbare
Gewinn war zum größten Teil ein Artefakt der Referenz: `KL(int8‖Arm)` enthält auch die Fehler, die
`int8` selbst macht, und die LS-Skala verhält sich dazu anders als zu `fp16`. Die LS-Skala ist damit
**kein Kandidat** mehr; die Blockfehler-Zahl (2,551e-02 gegen 2,740e-02) sagt über die KL nichts.

### Das Fenster: der Punkt erreicht `int8`

Der Arm `q4lm32` + Fenster 256/64 (5,29 bit/cell) gegen `int8` (8,25 bit/cell), gepaart je Position:

| Vergleich | top-64 | top-256 |
|---|---|---|
| `q4lm32w` − `int8` | +0,00002, [−0,00240, +0,00237], t +0,01 | +0,00034, [−0,00290, +0,00221], t +0,26 |
| `q4lm32sw` − `int8` | +0,00194, [−0,00457, +0,00070], t +1,44 | +0,00243, [−0,00527, +0,00041], t +1,68 |

`int8` führt in 4 von 8 Segmenten (top-256) beziehungsweise 5 von 8 (top-64) — ein Münzwurf.
**Der 5,29-bit-Punkt ist von `int8` nicht zu unterscheiden**, bei 36 % weniger Bytes pro Zelle.

Und gegen den **heutigen Produktpunkt** (dasselbe Fenster, aber 16/8-Bulk), gepaart je Position über
die beiden Messverzeichnisse:

| `q4lm32w` − `q4lm8v`+Fenster | Mittel | 95-%-Intervall | t | Segmente |
|---|---|---|---|---|
| top-64 | **−0,00400** | [−0,00663, −0,00138] | −2,99 | besser in 7/8 |
| top-256 | **−0,00414** | [−0,00702, −0,00127] | −2,82 | besser in 7/8 |

**0,004 nats besser bei identischen Bytes** (5,29 bit/cell, 1.016 MiB bei 128K). Das ist der
Austausch der V-Tabelle von 8 auf 16 Level, sonst nichts.

**Die LS-Skala kippt mit dem Fenster, aber nicht gesichert.** Im Bulk ist sie neutral (+0,0003), mit
Fenster liegt `q4lm32sw` um 0,0021 hinter `q4lm32w` (top-256: [−0,00467, +0,00050], t −1,58;
top-64: [−0,00429, +0,00045], t −1,59), `q4lm32w` besser in 6/8 Segmenten. Gegen den
`int8`-Anker war dieselbe Differenz −0,0032 mit einem Intervall, das die 0 ausschloss (t −2,27);
gegen `fp16` schließt es sie ein. Es bleibt also bei „kein Hebel", nicht bei „schädlich".

### K1 und die Auflösung der Regel

| Arm | Arm − `int8` (top-256) | obere 95-%-Grenze | `int8` führt | K1 |
|---|---|---|---|---|
| `q4lm32w` | +0,00034 | 0,00249 | 4/8 | **nein** |
| `q4lm32sw` | +0,00243 | 0,00481 | 5/8 | **nein** |
| `q4lm32s` | +0,00859 | 0,01126 | 8/8 | **nein** |
| `q4lm32` | +0,00892 | 0,01175 | 8/8 | **nein** |

Nach der vorregistrierten Regel ist K1 **nicht** erreicht: die obere Grenze muss unter +0,001
liegen, und der beste Arm liegt bei 0,0025.

**Das ist eine Auflösungsgrenze, keine Aussage über den Arm.** Die gepaarte Differenz hat eine
Standardabweichung von 0,22–0,24 nats; bei n = 32.746 Positionen ist die einseitige 95-%-Grenze bei
echter Parität schon

    1,645 · 0,2204 / √32746 = 0,0020

Für eine Grenze unter 0,001 bräuchte es **n ≈ 131.000 Positionen** — viermal cwe-4k, also genau den
128K-Satz, den es in keinem der beiden Bäume gibt (`cwe-2k`, `cwe-4k`, `cwe-4k-flat`, `fwe-2k` sind
vorhanden, 128K nicht). K1 in dieser Form ist mit dem vorhandenen Korpus für **keinen** Arm
erreichbar, auch nicht für einen, der exakt auf `int8` liegt. Die Regel gehört deshalb umgeschrieben
(zwei einseitige Tests auf Äquivalenz mit einer Grenze, die über der Auflösung liegt, oder der
128K-Satz), nicht der Arm verworfen.

### Die Lese-Seite ist bereits gemessen: Parität, kein Hebel

Aus dem 2026-10-25-Lauf (`bench/results/2026-10-25-tag-quality/`, gleicher Satz, gleiche Positionen,
16/8-Bulk, 5,29 bit), hier als gepaarte Differenz nachgerechnet:

| Arm | KL vs fp16 top-256 | gegen `int8` |
|---|---|---|
| `int8` | 0,01774 | — |
| Fenster 256/64 | 0,02223 | +0,00449 [−0,00745, −0,00153] |
| Tag-Read (gleiche Bytes) | 0,02107 | +0,00332 [−0,00620, −0,00045] |
| Tag-Read − Fenster | — | +0,00117 [−0,00162, +0,00395], t +0,82 |

Der Tag-Read ist **nicht** signifikant besser als das Fenster (t = 0,82, 3/8 Segmente). Der Read ist
damit kein Hebel; der Bulk ist einer.

### Laufzeit (keine K5-Messung)

Prefill-Rate der Qualitätsläufe, Median über 8 Segmente: `q4lm32w` 25,7 tok/s, `int8` 23,4 tok/s,
`q4lm32` 23,5 tok/s. Die Streuung ist groß (Minimum 6,8 tok/s bei `int8`), weil der teacher-forced
Prefill SSD-gebunden ist. Das ist **kein** K5-Lauf; K5 braucht einen eigenen, sauberen Durchgang.

### Was das für das Ziel bedeutet

**Das 5,0-bit/cell-Ziel ist am 2026-10-27 zurückgestellt** (Entscheidung des Nutzers): ohne deutlich
bessere Rotationsansätze und Fehlerkorrekturen ist es mit diesem Codec nicht erreichbar. Der Branch
maximiert zuerst **Qualität bei gleichen Bytes**, danach erst Bytes.

Der gemessene Stand bei gleichen Bytes:

| Hebel | Wirkung | Status |
|---|---|---|
| V-Tabelle 8 → 16 Level (16/8 → 16/16), gleiche Bytes | −0,0040 nats am 5,29-bit-Punkt | **gesichert**, 7/8 Segmente |
| K-Tabelle 8 → 16 Level | −0,0075 nats (Bulk) | gesichert, aber im Produktpunkt schon 16 |
| Level-Tabelle 16 → 8 auf beiden Seiten | +0,0175 nats | gesichert, schlechter |
| Least-Squares-Skala | ±0,0003 bis −0,0021 | **kein Hebel** |
| Tag-Read statt Fenster | +0,0012 nats | nicht signifikant |
| Kanalachse (S2) | — | tot |

Der Produktkandidat ist damit **`--kv q4lm32` + `STRATA_KV_WINDOW=1` (S=256, R=64)**: 5,29 bit/cell,
`KL(fp16‖Arm)` 0,01689 (top-64) gegen `int8`s 0,01688, und 0,0040 besser als der heutige Punkt bei
identischen Bytes. `int8` bleibt der Qualitäts-Default, das Fenster bleibt opt-in — der Kandidat
ändert nur, welche Tabelle der Bulk schreibt.

**Was dieser Lauf nicht sagt.** Er sagt nichts über den Ring selbst: 256/64 war gesetzt, nicht
variiert. Nach dem Tausch der Tabelle ist die naheliegende nächste Frage, ob dieselben Bytes besser
im Ring (S) oder in der Tabelle liegen — das ist ein neuer Arm, kein Ergebnis dieses Laufs.

## Dateien

- Rohdaten (nicht im Repo): `Strata-data/bench/2026-10-27-s3/`
  `{fp16,int8,q4lm32,q4lm32s,q4lm8v,q4lm8k,q4lm8,q4lm32w,q4lm32sw}.tsv` und `.seg.json`
- Anker-Auswertung: `k1-fp16-{64,256}.json` (`tools/kv_k1.py --ref fp16.tsv …`)
- Gegen den alten Produktpunkt: gepaart über `2026-10-27-s3` und `2026-10-25-tag-quality`
- K1 je Position: `tools/kv_k1.py --ref <ref>.tsv --run … --kl-topk 256`
- Segment-/Antwortbereich-Tabellen: `tools/quant_quality.py compare … --paired --ids …`
- Mess-Config: `bench/configs/cwe4k-swift-iq3_xxs.json`
