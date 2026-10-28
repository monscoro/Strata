# Decode-Sonde: was der Ring beim Decode wirklich liest

**Status: gemessen (2026-10-28), sechs Arme; die Geometrie ist beim Decode nicht entscheidbar.**
Katalogzeile:
`docs/paper/Ansatz_Improvements.md`, Leiter S3; Anschluss an
`bench/results/2026-10-28-ring-geometry/`.

## Warum diese Messung

Der Ring-Lauf hat eine Lücke aufgedeckt, die in `src/core/layer.cpp` steht: der Ring hält
`[R Sinks][S + B recent]`, und gelesen wird die Spanne `[pos0 − S, pos0 + B)`. Bei einem
Prefill-Chunk deckt `B` den Chunk ab, also liest jede Abfrage `S + B` exakte Zellen. **Beim Decode
ist die Spanne ein Token** — dann liest der Arm `R + S` Zellen, und die `B`-Zeilen liegen im
Speicher, aber außerhalb des Lesefensters.

Damit ist die `S`-Achse im Prefill fast wirkungslos (deshalb lagen 320/0, 192/128 und 64/256 so nah
beieinander), und der Produktpunkt ist bisher **nur auf teacher-forced Prefill** belegt. Diese Sonde
misst einen Decode-Schritt nach echtem Prefill.

## Instrument

`tools/quant_quality.py prompt` schickt je Segment die ersten 256 / 512 / 768 / 1000 Token als
Prompt (kein Prompt-Cache, batched Prompt-Pfad) und lässt das Modell **ein** Token generieren. Der
Engine schreibt die volle Logit-Verteilung dieses Schritts (`STRATA_DUMP_FIRST_LOGITS`, float32,
ganzes Vokabular). Das ist die Verteilung, die der Decode-Pfad berechnet — also der Pfad, der den
Ring mit `R + S` liest.

`tools/quant_quality.py pcompare` rechnet daraus `KL(fp16‖Arm)` über das ganze Vokabular und die
Top-1-Übereinstimmung. Für die gepaarte Auswertung je Prompt wird zusätzlich ein kleines Skript
über dieselben `first.*`-Dateien laufen (gleiche Formel wie `pcompare`, aber je Prompt gepaart).

## Arme

| Arm | `--kv` | Fenster | Decode-Fenster (R + S) | Zweck |
|---|---|---|---|---|
| `fp16` | `fp16` | aus | voller Kontext | Referenz |
| `int8` | `int8` | aus | voller Kontext | Qualitäts-Default, die Latte |
| `w256r64` | `q4lm32` | 256/64 | 320 Zellen | der Kandidat |
| `w64r64` | `q4lm32` | 64/64 | 128 Zellen | die `S`-Achse beim Decode |
| `w256r0` | `q4lm32` | 256/0 | 256 Zellen | die Sink-Achse beim Decode |
| `w8v` | `q4lm8v` | 256/64 | 320 Zellen | der heutige Produktpunkt |

`B` ist in allen Fensterarmen gleich (Auto-Chunk des Prompts), also unterscheiden sich die Arme nur
in `R` und `S`.

## Fragen und Entscheidungsregeln (vor dem Lauf festgelegt)

1. **Q1 — steht der Kandidat beim Decode auf `int8`?** Gepaart je Prompt:
   `KL(fp16‖w256r64) − KL(fp16‖int8)`. Urteil „Parität" nur, wenn das 95-%-Intervall die 0
   einschließt **oder** der Mittelwert ≤ 0 ist **und** die Top-1-Übereinstimmung mit `fp16` höchstens
   einen Prompt unter der von `int8` liegt. Sonst ist der Kandidat beim Decode schlechter als `int8`.
2. **Q2 — kostet die `S`-Achse beim Decode?** Gepaart `w64r64 − w256r64`. Urteil „`S` ist beim
   Decode frei": Intervall enthält 0 **oder** der Unterschied ist kleiner als die unten ausgerechnete
   Auflösung. Urteil „`S` trägt": Intervall schließt 0 aus **und** mindestens 6 der 8 Segmente
   stimmen zu.
3. **Q3 — tragen die Sinks beim Decode?** Gepaart `w256r0 − w256r64`, gleiche Regel wie Q2.
4. **Kein Arm wird auf Grund dieser Sonde zum Produktpunkt erklärt.** Sie ist eine Probe mit 32
   Prompts; ein Wechsel bräuchte einen Lauf mit echten Generationen.

## Auflösung, vorab notiert

Die 32 Prompts sind **8 Segmente × 4 Längen**, und die Längen sind ineinander geschachtelt
(256 ⊂ 512 ⊂ 768 ⊂ 1000). Die effektive Stichprobe ist also näher bei 8 als bei 32. Die Auflösung
wird nach dem Lauf aus der beobachteten gepaarten Streuung ausgerechnet und im Ergebnis genannt —
vorher ist jede Zahl dafür geraten. Erwartung: die Sonde kann große Effekte (≥ 0,01 nats) trennen
und kleine (≤ 0,002) nicht.

## Aufbau

| Punkt | Wert |
|---|---|
| Engine | 0.1.47 Debug-Build `build-hip-port\strata.exe`, SHA-256 `F9058D56…860F20`, ROCm 7.1 |
| Modell | swift-1.5-iq3_xxs, `bench/configs/cwe4k-swift-iq3_xxs.json` |
| Prompts | cwe-4k, 8 Segmente × 4 Längen (256/512/768/1000) = 32 Anfragen je Arm |
| Pfad | batched Prompt-Pfad (`--prefill auto`, kein `--short-read`), ein Decode-Schritt je Prompt |
| Metrik | `KL(fp16‖Arm)` über das ganze Vokabular, Top-1-Übereinstimmung, gepaart je Prompt |

## Läufe

```sh
tools/quant_quality.py prompt --config bench/configs/cwe4k-swift-iq3_xxs.json \
  --ids data/quant_quality/cwe-4k/set.ids.json --kv <fmt> [--env K=V ...] \
  --out Strata-data/bench/2026-10-28-decode/<arm>
```

Sechs Arme, nacheinander, ein GPU-Job zur Zeit.

## Ergebnis

`KL(fp16‖Arm)` über das **volle Vokabular**, je Prompt, 32 Prompts, 248.320 Token:

| Arm | `--kv` | Fenster | mean KL | median | max | Top-1 = fp16 |
|---|---|---|---|---|---|---|
| `int8` | `int8` | aus | **0,00055** | 0,00014 | 0,00357 | **32/32** |
| `w8v` | `q4lm8v` | 256/64 | 0,00152 | 0,00029 | 0,01772 | 31/32 |
| `w256r64` | `q4lm32` | 256/64 | 0,00225 | 0,00029 | 0,02284 | 31/32 |
| `w256r0` | `q4lm32` | 256/0 | 0,00326 | 0,00029 | 0,05395 | 31/32 |
| `w64r64` | `q4lm32` | 64/64 | 0,00400 | 0,00055 | 0,07449 | 31/32 |

Gepaart, **je Segment** (die 32 Prompts sind 8 Segmente × 4 geschachtelte Längen, also keine 32
unabhängigen Einheiten; `t` mit df = 7, `t_crit` 2,365):

| Vergleich | Δ (Arm − Basis) | 95 % | t | Segmente schlechter | einseitige Grenze bei Parität |
|---|---|---|---|---|---|
| `w256r64` − `int8` | +0,00170 | [−0,00024, +0,00365] | +2,07 | 6/8 | 0,0013 |
| `w64r64` − `w256r64` | +0,00175 | [−0,00216, +0,00566] | +1,06 | 6/8 | 0,0027 |
| `w256r0` − `w256r64` | +0,00100 | [−0,00137, +0,00338] | +1,00 | 6/8 | 0,0016 |
| `w8v` − `w256r64` | −0,00074 | [−0,00216, +0,00069] | −1,22 | 3/8 | 0,0010 |

Vorzeichen: positiv heißt „der Arm ist schlechter". Der Punktwert ist mit der per-Prompt-Rechnung
(n = 32) identisch — jedes Segment hat vier Prompts —, nur die Streuung unterscheidet sich: die
segmentweise ist kleiner, weil die vier Längen eines Segments denselben Text teilen.

### Q1 — steht der Kandidat beim Decode auf `int8`?

**Nach der vorregistrierten Regel: ja, Parität.** Das Intervall enthält die 0, und die
Top-1-Übereinstimmung liegt mit 31/32 genau einen Prompt unter `int8`s 32/32 — beide Klauseln der
Regel sind erfüllt.

**Aber die Parität ist knapp, und das gehört dazu:** der Punktwert +0,0017 liegt über der
einseitigen Grenze von 0,0013, `t` ist 2,07, und der Kandidat ist in 6 von 8 Segmenten schlechter.
Die untere Intervallgrenze ist −0,0002, also praktisch null. Das ist **nicht** „Parität bestätigt",
sondern „nicht unterscheidbar, mit Neigung zu schlechter". Wer den Kandidaten beim Decode als
gleichwertig verkauft, überzieht diese Sonde.

### Q2 — kostet die `S`-Achse beim Decode?

**Nicht entscheidbar.** +0,00175 liegt unter der Auflösung von 0,0027, das Intervall enthält die 0.
Die Richtung (6/8, `S = 64` schlechter) ist die erwartete — beim Decode liest der Arm `R + S`, also
128 statt 320 exakte Zellen —, aber diese Sonde kann sie nicht zertifizieren.

### Q3 — tragen die Sinks beim Decode?

**Nicht entscheidbar.** +0,00100 liegt unter der Auflösung von 0,0016. Die Richtung (6/8, ohne
Sinks schlechter) ist ebenfalls die erwartete und stimmt mit dem Prefill-Befund überein
(`R = 0` kostet dort +0,00394, 8/8), aber hier ist sie nicht gesichert.

### Ein Nebenbefund, der gegen die Prefill-Ordnung läuft

`w8v` (16/8-Bulk) ist beim Decode **nominell besser** als `w256r64` (16/16-Bulk): −0,00074, in 5 von
8 Segmenten besser, nicht signifikant. Im Prefill ist die Ordnung umgekehrt und gesichert
(`q4lm32w` 0,01689 gegen `q4lm8v` + Fenster 0,02090, −0,0040 nats). Die Sonde **reproduziert die
Prefill-Rangfolge also nicht**. Das heißt nicht, dass 16/8 besser ist — dafür ist der Effekt zu
klein und die Stichprobe zu schmal —, aber es heißt, dass diese Sonde den 16/16-Punkt weder stützen
noch kippen kann.

### Das Niveau: die Sonde liegt bei den frühen Buckets, nicht beim Mittel

`w256r64` liegt hier bei 0,00225. Der teacher-forced-Lauf misst für denselben Arm 0,01689 über den
ganzen Text — aber dessen **frühe** Buckets liegen bei 0,00348 (0–511) und 0,00413 (512–1023), und
die 0,01689 sind von den späten Buckets getragen (0,0195 bis 0,0396 ab Position 2.048, siehe
`2026-10-28-ring-geometry`). Die Sonde misst also da, wo der Ring noch nichts verdrängt hat. Das ist
kein Widerspruch, sondern die Bestätigung, dass der teacher-forced-Mittelwert eine
Spätpositions-Zahl ist.

### Auflösung, aus der beobachteten Streuung

Die vorregistrierte Erwartung war „große Effekte (≥ 0,01) trennbar, kleine (≤ 0,002) nicht". Gemessen:
**0,0013 bis 0,0027** je nach Paar — etwas besser als befürchtet, weil die Segmentstreuung kleiner
ist als die Prompt-Streuung. Effekte der Größe 0,001 bleiben trotzdem außerhalb.

### Was die Sonde nicht sagt

Die Prompts sind höchstens **1.000 Token** lang (`PROMPT_LENS` des bestehenden `prompt`-Pfads), und
der Ring hält `R + S + B` Zeilen — bei 1.000 Token liegt der größte Teil des Kontexts noch im Ring,
bei 256 Token alles. Die Sonde misst damit den **Decode-Pfad**, nicht den Ring unter Verdrängung.
Genau deshalb sind Q2 und Q3 strukturell schwer auflösbar: der Unterschied zwischen `S = 64` und
`S = 256` wird erst sichtbar, wenn Zellen tatsächlich aus dem Ring fallen, also bei Kontextlängen
über ~832 Token. Ein Lauf mit 4K-Prompts und einem Decode-Schritt wäre der nächste Arm — und ein
anderer, weil er die Ring-Verdrängung mitbringt.

