# Ring-Geometrie bei gleichen Bytes – Sinks gegen recent

**Status: gemessen (2026-10-28), drei Arme; die Geometrie 256/64 bleibt stehen.** Katalogzeile:
`docs/paper/Ansatz_Improvements.md`, Leiter S3, Anschluss an
`bench/results/2026-10-27-s3-level-split/`.

## Frage

S3(a) hat die Codebook-Seite abgeschlossen: 16 Level auf K **und** V, die Least-Squares-Skala ist
kein Hebel, der Split ist beliebig. Damit liegt der Bulk bei 4,50 bit/value und ist mit diesem Codec
erschöpft. Was übrig bleibt, ist der Ring: `R` Sinks plus `S + B` recent-Zellen, die als exakte
fp16-Zellen gelesen werden.

Bisher war die Geometrie gesetzt, nie variiert: 256/64, B aus dem Prefill-Chunk. Diese Frage
verschiebt **nur die Aufteilung** derselben Ringzeilen — wie viele der 320 Zeilen über `B` hinaus
Sinks sind (immer exakt, immer die ersten Token der Passage) und wie viele die letzten `S` Zellen.
Die Bytes bleiben dabei praktisch gleich; es ist ein reiner Geometrie-Tausch, kein Byte-Tausch.

## Arme

Alle Arme: `--kv q4lm32`, `STRATA_KV_WINDOW=1`, `--prefill 512` → **B = 512**, 4.608 Zellen,
cwe-4k, dieselben Positionen. Nur `STRATA_KV_WINDOW_S` und `_R` unterscheiden sich.

| Arm | S | R | B | Ringzeilen | Bytes | bit/value |
|---|---|---|---|---|---|---|
| Referenz `q4lm32w` (gemessen 2026-10-27) | 256 | 64 | 512 | 832 | 52.334.592 | 7,3941 |
| `s320r0` | 320 | 0 | 512 | 832 | 52.337.664 | 7,3945 |
| `s192r128` | 192 | 128 | 512 | 832 | 52.331.520 | 7,3937 |
| `s64r256` | 64 | 256 | 512 | 832 | 52.325.376 | 7,3928 |

**Die Bytes sind nicht ganz identisch, und zwar aus einem Grund, den man kennen muss:** die
Zelltags skalieren mit `S + B`, nicht mit `R` (`qsa_win_bytes`: `(R+S+B)·2.048` für die zwei
fp16-Pools, aber nur `(S+B)·4` für die Tags). Die Spanne über alle vier Arme ist **12.288 B =
0,024 %**. Das ist zwei Größenordnungen unter der Auflösung der KL-Messung und wird hier offen
genannt, statt „identische Bytes" zu behaupten.

## Erwartung, und was ein Nullresultat hieße

Die Literatur (StreamingLLM und Nachfolger) sagt, dass einige exakte Anfangs-Token die
Aufmerksamkeitsstatistik stabilisieren, also `R > 0` hilft. Diese Pipeline hat das **nicht** gemessen:
S1 hat gezeigt, dass das Fenster gelesen wird, nicht dass die Sinks es tragen. Die Richtung wird
deshalb nicht als Ergebnis vorweggenommen; die Messung entscheidet.

Ein Nullresultat über alle drei Arme hieße: bei 4.608 Zellen und diesem Codec ist die S/R-Aufteilung
kein Hebel, und die Ringfrage verschiebt sich von der Geometrie auf die **Zeilenzahl** (Bytes) —
das wäre dann der nächste, andere Arm.

## Entscheidungsregeln (vor dem Lauf festgelegt)

1. **Primär:** gepaarte Differenz `KL(fp16‖Arm) − KL(fp16‖Referenz)` je Position, gegen den
   `fp16`-Arm aus `2026-10-27-s3` (gleicher Build, gleicher Satz, gleiche Positionen), in **beiden**
   KL-Stützen (top-64 und top-256).
2. **Ein Arm gewinnt nur**, wenn das 95-%-Intervall der gepaarten Differenz die 0 in **beiden**
   Stützen ausschließt **und** der Arm in mindestens **6 von 8** Segmenten besser ist.
3. **Gleichstand** (Intervall enthält 0, oder < 6/8 Segmente) heißt: die Referenzgeometrie 256/64
   bleibt stehen. Es wird nichts „knapp Besseres" übernommen.
4. **Zwei Sieger:** der kleinere Mittelwert bei top-256 gewinnt; liegen sie unter 0,001 nats
   auseinander, bleibt die Referenz stehen (unter der Auflösungsgrenze wird nicht entschieden).
5. **Auflösungsgrenze, vorab notiert:** bei n = 32.746 Positionen und einer gepaarten Streuung von
   0,22 nats ist die einseitige 95-%-Grenze bei echter Parität 0,0020. Effekte unter ~0,002 nats
   kann dieser Lauf **nicht** zertifizieren, auch wenn sie real sind. Das ist keine Aussage über den
   Arm, sondern über den Korpus (siehe S3(a), K1).

## Aufbau

| Punkt | Wert |
|---|---|
| Engine | 0.1.47 Debug-Build `build-hip-port\strata.exe`, SHA-256 `F9058D56…860F20`, ROCm 7.1 |
| Modell | swift-1.5-iq3_xxs (`Strata-data\models\swift-IQ3_XXS`), `--prefill 512`, `--prompt-cache 0` |
| Passage | cwe-4k, alle 8 Segmente, 32.746 Positionen je Arm, teacher-forced |
| Bulk | `--kv q4lm32` (16/16, `max\|x\|/87`) — in allen vier Armen gleich |
| Fenster | `STRATA_KV_WINDOW=1`, `S`/`R` je Arm, `B` aus dem Chunk (512), kein `STRATA_KV_WINDOW_BATCH` |
| Referenz | `q4lm32w.tsv` aus `2026-10-27-s3` (S=256, R=64) — gleicher Build, keine Neumessung nötig |
| Anker | `fp16.tsv` aus `2026-10-27-s3` |
| Config | `bench/configs/cwe4k-swift-iq3_xxs.json` (festgeschrieben, `env: {}`) |

`--prefill` bleibt in allen Armen 512. Ein anderer Chunk würde `B` und damit die Bytes **und** die
Prefill-Numerik mitverschieben; dann wäre es kein Geometrie-Tausch mehr.

## Läufe

```sh
tools/quant_quality.py run --config bench/configs/cwe4k-swift-iq3_xxs.json \
  --ids data/quant_quality/cwe-4k/set.ids.json --kv q4lm32 --topk 256 \
  --env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=<S> --env STRATA_KV_WINDOW_R=<R> \
  --out Strata-data/bench/2026-10-28-ring/<arm>.tsv
```

Drei Arme, nacheinander, ein GPU-Job zur Zeit. Erwartete Dauer ~20 min je Arm.

## Ergebnis

Alle vier Arme, `KL(fp16‖Arm)` je Position, 32.746 Positionen:

| Arm | S/R | top-64 | top-256 | gegen 256/64 (top-64) | gegen 256/64 (top-256) |
|---|---|---|---|---|---|
| Referenz | 256/64 | **0,01689** | **0,01809** | — | — |
| `s192r128` | 192/128 | 0,01757 | 0,01871 | +0,00068 [−0,00202, +0,00337] | +0,00063 [−0,00231, +0,00356] |
| `s64r256` | 64/256 | 0,01882 | 0,01995 | +0,00193 [−0,00057, +0,00443] | +0,00186 [−0,00088, +0,00460] |
| `s320r0` | 320/0 | 0,02053 | 0,02203 | **+0,00364 [+0,00091, +0,00637]** | **+0,00394 [+0,00097, +0,00691]** |

Vorzeichen gedreht gegenüber `kv_k1.py`, das `Referenz − Arm` ausgibt: positiv heißt hier „der Arm
ist schlechter".

### Der Spruch der vorregistrierten Regel

**Kein Arm gewinnt.** `s320r0` ist in beiden Stützen signifikant schlechter und in 8/8 Segmenten;
`s192r128` und `s64r256` liegen innerhalb des Intervalls. Nach Regel 3 bleibt die Geometrie
**256/64** stehen. Es wird nichts „knapp Besseres" übernommen.

### Was die drei Arme zusammen sagen: die Sinks tragen, mehr Sinks tragen nicht

Die drei Punkte liegen auf einer asymmetrischen Kurve um 64 Sinks:

| R | S | Verlust gegen 256/64 (top-256) | Arm besser in | Urteil |
|---|---|---|---|---|
| 0 | 320 | **+0,00394** [+0,00097, +0,00691] | 0/8 | **Verlust gesichert** |
| 64 | 256 | 0 (Referenz) | — | — |
| 128 | 192 | +0,00063 [−0,00231, +0,00356] | 5/8 | nicht entscheidbar |
| 256 | 64 | +0,00186 [−0,00088, +0,00460] | 3/8 | nicht entscheidbar |

„Arm besser in" ist die Segmentbilanz des jeweiligen Arms gegen die Referenz, nicht die des
Verlusts: bei `s192r128` ist der Mittelwert schlechter, aber in 5 von 8 Segmenten besser — ein
einzelnes Segment mit großem Ausschlag trägt den Mittelwert. Genau deshalb verlangt Regel 2 beides.

Zwei Dinge sind daran bemerkenswert:

1. **Der Sink-Term ist real.** Ihn ganz zu streichen kostet 0,0039 nats — **etwa so viel wie der
   ganze Gewinn der 16/16-Tabelle am Produktpunkt** (0,0040) und fast das Doppelte der
   Auflösungsgrenze von 0,0020. Das ist die erste gesicherte Aussage dieser Pipeline über die Sinks:
   nicht „die Literatur sagt", sondern gemessen.
2. **Mehr Sinks helfen nicht.** 128 Sinks sind von 64 nicht zu trennen, 256 Sinks sind nominell
   schlechter. Der Grund steht im Byte-Modell: `R + S + B` ist fest, also nimmt jedes Sink eine
   recent-Zelle weg. Bei 4.608 Zellen ist der Ring der knappe Teil, nicht der Sink-Vorrat. Der
   Optimumsbereich ist breit (64 bis 128), nicht spitz.

**Die Richtung ist damit geklärt, die Größe nicht.** Der Unterschied zwischen 64 und 128 Sinks
(0,0006) liegt unter der Auflösung von 0,0020; dieser Lauf kann nicht sagen, ob 64 besser ist als
128 oder umgekehrt. Er kann nur sagen, dass 0 schlechter ist und dass 256 keinen Gewinn bringt.

### Wo der Sink-Verlust sitzt

Dieselben Arme, `KL(fp16‖Arm)` je 512-Positionen-Bucket (Position im Segment, alle 8 Segmente
zusammen):

| Bucket | 256/64 (Ref) | 320/0 | 192/128 | 64/256 |
|---|---|---|---|---|
| 0–511 | 0,00348 | 0,00370 | 0,00357 | 0,00348 |
| 512–1023 | 0,00413 | 0,00357 | 0,00338 | 0,00328 |
| 1024–1535 | 0,00335 | 0,00339 | 0,00280 | 0,00268 |
| 1536–2047 | 0,01281 | 0,01136 | 0,00871 | 0,00899 |
| 2048–2559 | 0,01950 | 0,03154 | 0,02401 | 0,02698 |
| 2560–3071 | 0,02539 | 0,02666 | 0,02871 | 0,02632 |
| 3072–3583 | 0,03959 | 0,04121 | 0,04211 | 0,04723 |
| 3584–4095 | 0,03653 | 0,05497 | 0,03650 | 0,04074 |

Gepaart gegen die Referenz, negativ heißt besser:

| Bucket | 320/0 | 192/128 | 64/256 |
|---|---|---|---|
| 0–511 | +0,00022 [+0,00009, +0,00036] 8/8 | +0,00009 [−0,00015, +0,00032] 3/8 | −0,00000 [−0,00006, +0,00006] 4/8 |
| 512–1023 | −0,00055 [−0,00190, +0,00079] 5/8 | −0,00075 [−0,00350, +0,00201] 1/8 | −0,00085 [−0,00198, +0,00028] 3/8 |
| 1024–1535 | +0,00003 [−0,00097, +0,00104] 4/8 | −0,00055 [−0,00162, +0,00052] 2/8 | −0,00068 [−0,00188, +0,00053] 2/8 |
| 1536–2047 | −0,00144 [−0,00435, +0,00146] 5/8 | −0,00410 [−0,01006, +0,00186] 3/8 | −0,00382 [−0,01182, +0,00417] 3/8 |
| 2048–2559 | +0,01203 [−0,00029, +0,02436] 7/8 | +0,00451 [−0,00842, +0,01743] 5/8 | +0,00748 [−0,00126, +0,01622] 6/8 |
| 2560–3071 | +0,00127 [−0,00583, +0,00836] 5/8 | +0,00332 [−0,00486, +0,01150] 7/8 | +0,00093 [−0,00610, +0,00795] 4/8 |
| 3072–3583 | +0,00162 [−0,00765, +0,01088] 5/8 | +0,00251 [−0,01056, +0,01559] 4/8 | +0,00764 [−0,00564, +0,02091] 6/8 |
| 3584–4095 | **+0,01844 [+0,00212, +0,03476] 6/8** | −0,00003 [−0,01017, +0,01011] 4/8 | +0,00421 [−0,00636, +0,01478] 4/8 |

Die letzte Spalte ist „Arm schlechter in N/8 Segmenten".

**Der Sink-Verlust ist kein konstanter Offset, er wächst mit der Distanz.** In den ersten 1.536
Positionen ist `s320r0` neutral bis leicht besser; ab 2.048 trägt es den ganzen Verlust, und die
beiden letzten Buckets allein ergeben (0,01203 + 0,00127 + 0,00162 + 0,01844) / 8 = **0,00417** von
den 0,00394 Gesamtverlust — also praktisch alles. Das ist genau die Form, die die Sink-Hypothese
vorhersagt: solange alle Zellen im Ring liegen, braucht niemand Sinks; sobald Zellen verdrängt sind,
sind die Sinks der einzige exakte Anker am Anfang der Passage.

Zwei Einschränkungen, damit die Tabelle nicht mehr sagt, als sie kann:

- **Die Buckets sind laut.** Einzelne Positionen mit hartem Kontext tragen die Mittelwerte (der
  Sprung 1536–2047 auf 0,013 und 2048–2559 auf 0,020 ist Inhalt, nicht Geometrie — er steht in allen
  vier Armen). Von den 24 Bucket-Vergleichen ist **einer** gesichert (320/0, 3584–4095); der
  Gesamtwert 0,00394 ist die belastbare Zahl, das Profil ist die Erklärung dazu.
- **Der kleine 0–511-Effekt bei `s320r0` (+0,00022, 8/8, Intervall schließt 0) ist nicht erklärt.**
  Bei Positionen unter 512 liegt alles im Ring, dort sollte die Geometrie nichts ändern. Ich führe
  ihn auf die andere Lesereihenfolge des Grids zurück (andere Anzahl exakter Zellen je Abfrage),
  aber das ist **Vermutung, nicht gemessen** — er ist klein genug, um am Gesamtbild nichts zu ändern.

### Laufzeit

Die drei Läufe brauchten 30, 27 und 17 Minuten (8 Segmente), gegen 18 Minuten für die Referenz am
2026-10-27. Die Streuung ist SSD-gebunden (dieselben Bytes, dieselbe Arbeit); `s64r256` lief in der
schnellsten Phase des Tages. Das ist **keine** K5-Messung — die Geometrie ändert die gelesene
Zeilenzahl, nicht die geschriebene Byte-Menge, und ein sauberer Laufzeitvergleich bräuchte einen
eigenen Durchgang.

## Was dieser Lauf nicht sagt: Decode

**Alle vier Arme sind teacher-forced Prefill mit Chunk 512.** Der Ring hält laut `layer.cpp` die
Zeilen `[R Sinks][S + B recent]`, und die gelesene Spanne ist `[pos0 − S, pos0 + B)` — bei einem
Prefill-Chunk deckt `B` den ganzen Chunk ab, also liest jede Abfrage `S + B` exakte Zellen. **Beim
Decode ist die Spanne ein Token**, `[pos0 − S, pos0 + 1)`: dann werden `R + S` exakte Zellen gelesen
und die `B`-Zeilen liegen zwar im Speicher, aber außerhalb des Lesefensters.

Daraus folgen zwei Dinge, die diese Messung **nicht** beantwortet:

1. **Die `S`-Achse ist im Prefill fast wirkungslos**, weil `B` die recent-Seite dominiert. Das ist
   der Grund, warum die drei Arme so nah beieinander liegen: sie unterscheiden sich im Prefill
   vor allem in `R`, nicht in `S`. Über die Decode-Qualität bei `S = 64` gegen `S = 320` sagt der
   Lauf nichts, und beim Decode ist genau das der Unterschied zwischen 128 und 384 exakten Zellen.
2. **`B` ist Decode-Speicher, der nicht gelesen wird.** Bei 128K mit `B = 6.144` sind das
   6.144 · 2.048 · 12 = 151 MB, die nur während des Prefills eine Rolle spielen. Das ist Katalogzeile
   B4 (kleineres `B`, dafür langsamerer Prefill) — und der Grund, warum die Byte-Frage nach diesem
   Lauf interessanter ist als die Geometrie-Frage.

Wer die Decode-Seite messen will, braucht einen anderen Lauf: `quant_quality.py prompt` bzw. ein
Generate-Lauf mit echten Antworten, nicht die teacher-forced KL über einen 4K-Text.

## Was das für den Kandidaten heißt

**Die Qualitätsseite bei gleichen Bytes ist damit geschlossen.** Drei Hebel waren offen, alle drei
sind gemessen:

| Hebel bei gleichen Bytes | Ergebnis | Status |
|---|---|---|
| Bulk-Tabelle 8/8 → 16/16 | −0,0175 nats | **gesichert**, 8/8 (S3(a)) |
| V-Seite 8 → 16 Level am Produktpunkt | −0,0040 nats | **gesichert**, 7/8 (S3(a)) |
| Sink-Aufteilung 64 → 0 | +0,0039 nats (Verlust) | **gesichert**, 8/8 (dieser Lauf) |
| Sink-Aufteilung 64 → 128 / 256 | ±0,0006 bis +0,0019 | unter der Auflösung |
| Least-Squares-Skala | ±0,0003 | kein Hebel |
| Tag-Read statt Fenster | +0,0012 | nicht signifikant |
| Kanalachse (S2) | — | tot |

Der Kandidat bleibt **`--kv q4lm32` + `STRATA_KV_WINDOW=1` (S=256, R=64)**: `KL(fp16‖Arm)` 0,01689
(top-64) gegen `int8`s 0,01688, bei 7,39 bit/value am 4K-Messpunkt (10,4 % unter `int8`) und
5,29 bit/cell am 128K-Produktpunkt (36 % unter `int8`, nicht hier gemessen). Die Geometrie ist an
ihrem Plateau; an ihr ist bei gleichen Bytes nichts mehr zu holen.

**Was als Nächstes übrig bleibt, ist die Byte-Seite, nicht die Qualität.** Bei 4.608 Zellen ist der
Ring der teure Teil (832 Zeilen = 20,5 MB gegen 31,9 MB Bulk). Ob er kleiner sein darf, ist eine
andere Frage als diese: sie verschiebt Zeilen, nicht deren Aufteilung. Dass B = 512 bei 128K
**billiger** ist als der Produktpunkt (4,60 statt 5,29 bit/value) und nur Prefill-Zeit kostet, ist
Katalogzeile B4 und ebenfalls ungemessen.

## Dateien

- Rohdaten (nicht im Repo): `Strata-data/bench/2026-10-28-ring/`
  `{s320r0,s192r128,s64r256}.tsv`, `.seg.json`, `.log`, `distance-profile.txt`
- Anker-Auswertung: `k1-fp16-{64,256}.json` (`tools/kv_k1.py --ref fp16.tsv --run …`)
- Referenz- und Ankerarm: `Strata-data/bench/2026-10-27-s3/{fp16,q4lm32w}.tsv` (gleicher Build)
- Distanzprofil: 512-Positionen-Buckets, gepaart je Position, top-256
- Mess-Config: `bench/configs/cwe4k-swift-iq3_xxs.json`


