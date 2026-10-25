# E4 — ist der Restfehler dünn oder dicht?

Katalogzeile: `docs/paper/Ansatz_Improvements.md`, E4 („GEAR-Diagnose, kein GEAR-Kernel"). Offline,
aus den S0-Logs. Keine GPU, kein neuer Lauf.

Die Logs sind die von `bench/results/2026-10-25-s0-kl-support/`: Engine 0.1.47, SHA-256
`40942104…007B7B`, AMD Radeon RX 9060 XT (gfx1200, 16 GB), cwe-4k, 32.746 Positionen, KL gegen
fp16 auf dem 64er-Support des Harness, dieselbe Floor-Substitution. Die Mittelwerte sind die
gedruckten: `int8` 0,01688, Fenster 0,02090, plain 0,03182.

`tools/kv_error_shape.py` zerlegt jede Position in die Terme der Referenz-Ränge 1–8 (Kopf) und
9–64 (Tail). Die Summe ist die KL des Harness.

## Form

| Arm | KL | davon Kopf | Tail | Positionen mit Kopf-Substitution |
|---|---|---|---|---|
| `int8` | 0,01688 | 0,01484 (87,9 %) | 0,00203 | 0,7 % (221) |
| Fenster 256/64 | 0,02090 | 0,01888 (90,4 %) | 0,00202 | 1,0 % (336) |
| plain | 0,03182 | 0,02809 (88,3 %) | 0,00373 | 1,6 % (524) |

Eine Kopf-Substitution ist ein Token aus den acht wahrscheinlichsten der Referenz, den der Arm
auf seinen Floor geschoben hat. Der Katalog warnte, die 17-Nat-Positionen seien Floor-Ereignisse
und damit noch kein Beleg für dünne Ausreißer. Die Zerlegung:

- Der Fehler sitzt im Kopf, nicht im Tail. 88–90 % der KL sind Ränge 1–8.
- Die Floor-Ereignisse im Kopf sind selten und tragen etwa die Hälfte. Bei `int8` haben 32.525
  Positionen keine Kopf-Substitution und im Mittel 0,00753 KL (244,9 nat, 44 % der Summe);
  die 221 Positionen mit Substitution tragen die übrigen 56 %.
- Auch ohne diese Ereignisse ist der Rest dünn. Die schlimmsten 1 % der Positionen tragen
  85,5 % (`int8`), 81,6 % (Fenster) und 70,9 % (plain) der KL; die schlimmsten 1 % der
  Kopf-Terme tragen 86,9 / 82,7 / 72,5 %. Die untere Hälfte der Positionen trägt 0,3–1,0 %.

Tail-Anteile über 100 % in der Konzentrationszeile des Skripts sind negative Terme (der Arm
legt auf einen Rang mehr Masse als die Referenz), keine Rechenfehler. Der Tail ist 12 % der KL
und trägt die Frage nicht.

## Der Gewinn des Fensters

plain − Fenster = +357,8 nat über 32.746 Positionen, Mittel +0,01093. Das ist die gepaarte
Differenz aus S0. Das Fenster ist auf 83,0 % der Positionen besser und auf 17,0 % schlechter.
Die schlimmsten 0,1 % der Positionen tragen 50 % des Nettogewinns, die schlimmsten 1 % tragen
133 % — mehr als der Nettogewinn, weil der Rest dagegen steht. Der Gewinn ist eine dünne Menge
schlechter Positionen, bezahlt mit kleinen Verlusten anderswo.

## Lesart

Der Rest ist dünn, und er ist ein Kopf-Phänomen, nicht ein Floor-Artefakt auf Rang 60. Die
Katalogregel „Kernel nur, wenn die Verteilung dünn ist" lässt einen GEAR-artigen Kernel zu.
Sie startet ihn nicht. S1 (`2026-10-25-s1-selstats`) zeigt, dass 70 % der gelesenen Zellen
bereits eine fp16-Kopie haben, die der Horizont-Check nicht liest. Ein neuer Restfehler-Kernel
ist nicht der nächste Schritt; der vorhandene Ring ist es. GEAR bleibt offen und ungebaut.

## Reproduktion

```powershell
python tools\kv_error_shape.py
```

Die Rohlogs liegen außerhalb des Repos unter `C:\Users\krone\git\Strata-data\bench\2026-10-25-s0\`.
