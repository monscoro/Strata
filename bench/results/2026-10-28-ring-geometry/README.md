# Ring-Geometrie bei gleichen Bytes – Sinks gegen recent

**Status: vorregistriert (2026-10-28), Läufe noch nicht gestartet.** Katalogzeile:
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

_(folgt)_
