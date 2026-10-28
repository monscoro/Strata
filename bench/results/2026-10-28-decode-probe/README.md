# Decode-Sonde: was der Ring beim Decode wirklich liest

**Status: vorregistriert (2026-10-28), Läufe noch nicht gestartet.** Katalogzeile:
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

_(folgt)_
