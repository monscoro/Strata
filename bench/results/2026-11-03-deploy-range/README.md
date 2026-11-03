# Einsatzbereich 32K und 64K: trägt die Tag-Lesung dort, wo das Produkt läuft?

**Status: vorregistriert (2026-11-03), vor dem ersten Arm-Lauf.** Der erste Arm-Lauf darf erst nach
dem Schreiben dieses Dokuments starten. Anschluss an `bench/results/2026-11-03-tag-arm/` (dort 4K
als Diagnose-Punkt und 128K als erster Punkt im Einsatzbereich; 4K ist **kein** Produktpunkt).
Katalogzeilen: `docs/paper/Ansatz_Improvements.md`, E1/S1.

## Warum

Der Einsatzpunkt der Maschine ist ein Kontextfenster von 32K bis 256K, nicht 4K. Gemessen ist in
diesem Bereich bisher nur 128K, und dort ist der Effekt der Tag-Lesung null: `tag − win` =
−0,00001 nats, Block-95 % [−0,00066; +0,00069], bei 48,55 % exakt gelesener Auswahl statt 8,55 %.
Die Abdeckung, die der Tag zusätzlich exakt liest, **fällt mit dem Horizont** (E1/S1 über die
S-Leiter: 100 → 94,6 → 80,0 → 48,6 %). Wenn der Hebel irgendwo im Einsatzbereich etwas bewirkt,
dann bei der kürzesten Länge des Bereichs. Genau das prüft dieser Lauf: **32K und 64K**, mit
demselben Protokoll wie der 128K-Punkt, damit die drei Längen vergleichbar sind.

Die Frage ist ausdrücklich **nicht** „ist 4K besser als 128K", sondern: trägt der Tag bei 32K
und/oder 64K, und wie groß ist dort die Lücke zu `int8`?

## Aufbau

Derselbe teacher-forced Aufbau wie `T3` in `2026-11-03-tag-arm/`: ein Segment, ein `GEN 1`,
per-Position-Logits aus den Verify-Fenstern am Prompt-Ende, `--short-read 6144`, `--topk 256`,
`KL(fp16‖Arm)` über `fp16`s top-256, gepaart je Position, Moving-Block-Bootstrap.

| Punkt | Wert |
|---|---|
| Passagen | `p32k` = die ersten **4** Segmente der E1-Passage (32.756 Token), `p64k` = die ersten **8** (65.511 Token); dieselbe cwe-8k-Passage wie beim 128K-Punkt, verschachtelte Präfixe |
| Warum Präfixe | dieselben Texteinheiten wie 128K, nur kürzer; der Tail jedes Punktes ist das Ende eines anderen Segments, also anderer Text — die Längen sind drei Punkte, kein geschachtelter Satz von Teilmengen desselben Tails |
| Schnitt | `pin = seg_tokens − 6144` (32K: 26.612, 64K: 59.367) plus `--prompt-cache 3`: alles davor batched, die letzten 6.144 Token durch die Fenster |
| Fenster | `STRATA_KV_WINDOW=1 S=256 R=64`, kein `BATCH` (der Ring folgt dem Chunk: 6.144 Zellen, `S+B` = 6.400 Zeilen) |
| Prefill | `--prefill auto` |
| Positionen | 6.144 je Arm und Länge (der Dokument-Tail), `--allow-no-positions` |
| Arme | `fp16`, `int8`, `win` (`q4lm32`, 256/64), `tag` (`q4lm32`, 256/64 + `STRATA_KV_WINDOW_TAG=1`) |

Ein GPU-Job zur Zeit, 30 s zwischen zwei Engine-Starts, Engine-SHA
`CD5700EC0793F83B…3D0394CA` vor dem Lauf geprüft.

## Entscheidungsregeln (vor dem Lauf festgelegt)

Je Länge getrennt, ein Segment, also **keine** Segment-Clusterung; Moving-Block-Bootstrap über
Blöcke von 256 aufeinanderfolgenden Positionen (24 Blöcke, 2.000 Resamples, Seed 20261103).
Δ = KL(fp16‖Arm) − KL(fp16‖Basis), negativ = der Arm ist besser.

- **`D1` — trägt der Tag bei dieser Länge?** Gepaart `tag − win`.
  **„Tag trägt"** nur wenn das Bootstrap-95-%-Intervall die 0 ausschließt, der Mittelwert negativ
  ist **und** |Mittelwert| > 0,0013 nats — dieselbe vor dem 128K-Lauf fixierte Schwelle wie in
  `T3b`, unverändert übernommen. Sonst **„trägt nicht"**, mit Zahl und Richtung.
- **`D2` — wie groß ist die Lücke zu `int8`?** Gepaart `tag − int8` und `win − int8`.
  **„geschlossen"** wenn der Mittelwert von `tag − int8` ≤ 0 ist **und** die einseitige
  95-%-Bootstrap-Obergrenze unter **+0,001 nats** liegt (die K1-Grenze der Leiter). Sonst
  „nicht geschlossen", mit beiden Zahlen.
- **`D3` — ist der Arm informativ?** `KL(fp16‖win)` > 0 mit Bootstrap-95 % ohne 0. Andernfalls hat
  der Ring den Prompt gehalten und `D1` ist für diese Länge **ungültig**.
- **`D4` — Verlauf über die Längen.** 32K, 64K und 128K werden **einzeln** berichtet; keine
  gepoolte Zahl über Längen (verschiedene Positionen, verschiedene Texte — das wäre
  Pseudo-Replikation). Eine Aussage „der Effekt fällt mit der Länge" wird nur gemacht, wenn die
  Vorzeichen der drei Einzelmittel übereinstimmen; sonst bleibt es bei drei Einzelbefunden.

### Güte-Gatter (vor der Auswertung, je Arm)

- `DONE` ohne „nonfinite"; kein „do not fit"; kein `ERR`; Engine-SHA unverändert.
- TSV mit **6.144** Positionen; innerhalb einer Länge in allen vier Armen **dieselbe**
  Positionsmenge; `win`/`tag` mit der Ring-Zeile (6.144 Zellen), `tag` zusätzlich mit der
  `STRATA_KV_WINDOW_TAG=1`-Zeile.
- `seg.json` mit `pin` und `prompt_cache` = 3.

### Abweichungen und Grenzen, offen genannt

1. **Wiederholter Korpus, andere Tails.** 32K endet in Segment s03, 64K in s07, 128K in s15 (dort
   zweimal gelesen). Die drei Punkte sind damit nicht derselbe Text; für die gepaarte Differenz je
   Punkt ist das unkritisch, für einen Längenvergleich ist es eine Grenze und wird so berichtet.
2. **Kein 256K.** Der obere Rand des Bereichs ist mit diesem Lauf nicht abgedeckt; bei 262.144
   Zellen passt `fp16` voraussichtlich nicht in die 16 GB, und der Ring hält dort 2,4 % des
   Kontexts (gegen 4,9 % bei 128K). Eigener Lauf, eigene Entscheidung.
3. **`--short-read 6144`** ist ein Fenster, kein Voll-Prompt-Durchlauf; die 6.144 Positionen sind
   der Dokument-Tail.
4. **Kein Timing, kein Byte-A/B** — der Arm kostet statisch 0 Bytes.

### Kosten

Acht Läufe (2 Längen × 4 Arme), je ~32K: 245 s Prefill + 6.144 Fenster-Schritte, ~64K: 490 s +
6.144 Schritte; nach der 128K-Erfahrung (1.193 s) zusammen etwa 75 Minuten GPU, kein Engine-Bau.

## Reproduktion

```powershell
# ids: die ersten 4 bzw. 8 Segmente von 2026-11-01-e1-128k/cwe8k-x2.ids.json, ein Segment
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids C:\Users\krone\git\Strata-data\bench\2026-11-03-deploy-range\p32k.ids.json `
    --kv <fp16|int8|q4lm32> --prefill auto --short-read 6144 --topk 256 --allow-no-positions `
    --pin 26612 --prompt-cache 3 `
    [--env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 --env STRATA_KV_WINDOW_R=64 `
     [--env STRATA_KV_WINDOW_TAG=1]] `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-03-deploy-range\32k\<arm>.tsv
#   p64k genauso, --pin 59367, Ausgabe ...\64k\<arm>.tsv

python tools\kv_k1.py --ref ...\32k\fp16.tsv --run ...\32k\win.tsv --run ...\32k\tag.tsv `
    --run ...\32k\int8.tsv --kl-topk 256 --block 256 --json ...\32k\k1.json
```

Rohdaten liegen außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-11-03-deploy-range\`; im Repo steht dieses Dokument und
nach dem Lauf der Ergebnisabschnitt darunter.

---

# Ergebnis

**Status: gemessen (2026-11-03).** Alle Güte-Gatter bestanden bis auf eine Zahl: die TSVs haben
**6.143** Positionen statt der registrierten 6.144 (siehe Abweichung 1). Innerhalb jeder Länge ist
die Positionsmenge in allen vier Armen identisch (32K: 26.612…32.754, 64K: 59.367…65.509), `win`
und `tag` tragen die Ring-Zeile, nur `tag` die `STRATA_KV_WINDOW_TAG=1`-Zeile, kein „nonfinite",
kein „do not fit", kein `ERR`, Engine-SHA unverändert.

## Die drei Punkte im Einsatzbereich

`KL(fp16‖Arm)` über `fp16`s top-256, je Position, 6.143 Positionen pro Arm und Länge:

| Länge | `win` | `tag` | `int8` |
|---|---|---|---|
| 32K | 0,12187 [+0,09754; +0,14623] | 0,11179 [+0,08896; +0,13486] | 0,10263 [+0,07767; +0,12876] |
| 64K | 0,04634 [+0,03533; +0,05820] | 0,03680 [+0,02742; +0,04717] | 0,04012 [+0,02921; +0,05261] |
| 128K (aus `2026-11-03-tag-arm/`) | 0,00211 [+0,00047; +0,00486] | 0,00212 [+0,00055; +0,00458] | 0,00085 [+0,00012; +0,00222] |

Die Mittelwerte sind **nicht** über die Längen vergleichbar: sie werden an jedem Punkt von einer
Handvoll Positionen getragen, an denen die Verteilung kippt (nächster Abschnitt). Die gepaarten
Differenzen sind davon weitgehend frei, weil beide Arme dieselben Positionen sehen.

Gepaart je Position (Δ = Arm − Basis, negativ = Arm besser), Moving-Block-Bootstrap 256/2.000/Seed
20261103:

| Länge | Vergleich | Δ | Block-95 % | einseitige Grenze |
|---|---|---|---|---|
| 32K | `tag − win` | **−0,01008** | [−0,03335; +0,01238] | +0,02905 |
| 32K | `tag − int8` | +0,00916 | [−0,01535; +0,03274] | +0,02795 |
| 32K | `win − int8` | +0,01924 | [+0,00252; +0,03680] | +0,03361 |
| 64K | `tag − win` | **−0,00953** | [−0,01877; +0,00049] | +0,01727 |
| 64K | `tag − int8` | −0,00332 | [−0,01528; +0,00815] | +0,00638 |
| 64K | `win − int8` | +0,00622 | [−0,00422; +0,01617] | +0,01495 |

**`D1` — trägt der Tag bei 32K oder 64K?** **Nein, an keinem der beiden Punkte** — nach der
registrierten Regel. Die Mittelwerte zeigen beide zum Tag (−0,01008 bzw. −0,00953, das Siebenfache
der Schwelle 0,0013), aber beide Intervalle schließen die 0 ein: 32K deutlich
[−0,03335; +0,01238], 64K knapp [−0,01877; +0,00049]. Klausel 1 fällt zweimal; die Regel hat für
„Richtung klar, Intervall zu breit" keine Kategorie, und sie wird nicht nachträglich geändert.

**`D2` — ist die Lücke zu `int8` geschlossen?** **Nein.** 32K: `tag − int8` = +0,00916 (der Tag ist
schlechter), einseitige Grenze +0,02795. 64K: −0,00332 (der Tag ist besser), aber die einseitige
Grenze +0,00638 liegt über der K1-Grenze +0,001. Die Klausel „Mittelwert ≤ 0" hält also nur bei
64K, die Klausel „Obergrenze < +0,001" an keinem der beiden Punkte. Dem Punktwert nach ist der Tag
an beiden Längen besser als das Fenster; auf `int8`-Niveau kommt er an keinem der drei Punkte.

**`D3` — ist der Arm informativ?** Ja, an beiden Punkten: `KL(fp16‖win)` = 0,12187
[+0,09754; +0,14623] bzw. 0,04634 [+0,03533; +0,05820], ohne 0.

**`D4` — Verlauf über die Längen.** Die drei Punkte werden einzeln berichtet; eine Aussage „der
Effekt fällt mit der Länge" wird **nicht** gemacht (die Punkte haben verschiedene Tails, und 128K
liest denselben Text ein zweites Mal — siehe unten).

## Was die Mittelwerte tragen — und was nicht

Die Verteilung je Arm (alle 6.143 Positionen):

| Länge | Arm | Median | p90 | p99 | max | Positionen > 1 nat | > 10 nats |
|---|---|---|---|---|---|---|---|
| 32K | `int8` | 0,00089 | 0,0100 | 2,805 | 17,5 | 126 | 13 |
| 32K | `win` | 0,00235 | 0,0228 | 3,723 | 13,6 | 147 | 12 |
| 32K | `tag` | 0,00112 | 0,0116 | 3,704 | 17,7 | 128 | 14 |
| 64K | `int8` | 0,00091 | 0,0091 | 0,770 | 10,5 | 56 | 1 |
| 64K | `win` | 0,00146 | 0,0127 | 1,085 | 11,8 | 64 | 1 |
| 64K | `tag` | 0,00120 | 0,0131 | 0,747 | 17,2 | 50 | 2 |
| 128K | `int8` | 0,00000 | 0,0000 | 0,002 | 2,5 | 1 | 0 |
| 128K | `win` | 0,00002 | 0,0002 | 0,003 | 3,2 | 3 | 0 |
| 128K | `tag` | 0,00000 | 0,0001 | 0,003 | 2,9 | 4 | 0 |

Bei 32K und 64K tragen **50 bis 147 Positionen** (0,8–2,4 %) den Mittelwert; der Median liegt
anderthalb bis zwei Größenordnungen darunter. Das sind die in E4 beschriebenen 17-Nat-Positionen unter der
trunkierten KL, und sie sitzen über den ganzen Tail verteilt (Median-Abstand 32 Positionen, die
erste 23 nach der Schnittkante) — kein Naht-Artefakt.

**Beschreibend, nicht registriert:** ohne die Positionen, an denen ein Arm über 1 nat liegt
(32K: 220, 64K: 102, 128K: 6 entfernt), ergibt sich gepaart

| Länge | `tag − win` (getrimmt) | `tag − int8` (getrimmt) | `win − int8` (getrimmt) |
|---|---|---|---|
| 32K | **−0,00313 [−0,00471; −0,00144]** | +0,00213 [+0,00107; +0,00341] | +0,00526 [+0,00393; +0,00652] |
| 64K | −0,00034 [−0,00144; +0,00068] | +0,00181 [+0,00085; +0,00285] | +0,00215 [+0,00123; +0,00318] |
| 128K | −0,00010 [−0,00053; +0,00034] | +0,00063 [+0,00020; +0,00117] | +0,00073 [+0,00030; +0,00124] |

Danach ist der Tag **bei 32K** besser als das Fenster (0,0031 nats, Intervall ohne 0, an 4.021 von
6.143 Positionen besser), bei 64K und 128K nicht mehr nachweisbar. Der Vorzeichenbefund der
registrierten Mittel (−0,0101 / −0,0095) ist damit überwiegend Floor-Rauschen: bei 64K verschwindet
er ganz, bei 32K bleibt knapp ein Drittel übrig. Die Trimmung ist **post hoc** und entscheidet nichts —
sie steht hier, weil der registrierte Mittelwert an diesen Längen von 2 % der Positionen regiert
wird und das für die nächste Vorregistrierung eine Rolle spielt.

## Der 128K-Punkt ist eine zweite Lesung derselben Passage

Die 128K-Passage ist cwe-8k **zweimal**. Damit ist der 128K-Tail (124.877…131.020) Token für Token
derselbe Text wie der 64K-Tail (59.367…65.509), nur um 65.511 verschoben — geprüft: 6.143 von
6.143 Token identisch. Der 128K-Punkt misst also nicht nur eine andere Länge, sondern eine
**zweite Lesung**: der Text stand 65.511 Token früher schon einmal im Kontext. Dass die KLs dort
ein bis zwei Größenordnungen kleiner sind (`int8` im Mittel 0,00085 gegen 0,04012 bei 64K, `win`
0,00211 gegen 0,04634), ist mit diesem Unterschied konfundiert.

Welcher der beiden Effekte (Länge oder Wiederholung) den Unterschied macht, lässt sich mit diesen
Daten **nicht** trennen. Der saubere Test ist ein 128K-Tail aus frischem Text (16 verschiedene
Segmente statt acht zweimal) — eigener Lauf.

## Abweichungen vom registrierten Plan

1. **6.143 statt 6.144 Positionen.** Registriert war `pin = seg_tokens − 6144`; weil die letzte
   geschriebene Position `n − 2` ist, ergibt das `n − 1 − pin = 6.143`. Beim 128K-Punkt lag der Pin
   eine Stelle tiefer (124.877 = n − 6.145) und ergab 6.144. Ein Rechenfehler in der
   Vorregistrierung, kein Datenfehler: alle vier Arme einer Länge teilen dieselben 6.143
   Positionen, und keine Regel hängt an der Zahl.
2. **Floor-Positionen waren nicht vorhergesehen.** Die Vorregistrierung hat den Mittelwert als
   Statistik fixiert, ohne die 17-Nat-Ereignisse zu berücksichtigen; bei 32K/64K tragen sie 2 % der
   Positionen. Die registrierte Regel wird trotzdem angewandt; die getrimmte Zahl steht daneben und
   ist als post hoc markiert.
3. **Verschiedene Tails je Länge** (s03, s07, s15) — wie in der Vorregistrierung genannt.
4. **`--short-read 6144`** ist ein Fenster, kein Voll-Prompt-Durchlauf.
5. **Kein 256K** (siehe oben), **kein Timing, kein Byte-A/B**.

## Was das heißt

- **Registriert:** der Tag trägt bei 32K und 64K **nicht** (beide Intervalle enthalten 0), und die
  Lücke zu `int8` ist an keinem der beiden Punkte geschlossen.
- **Beschreibend:** bei 32K ist der Tag besser als das Fenster (0,0031 nats getrimmt, Intervall ohne
  0, 65 % der Positionen), bei 64K und 128K nicht nachweisbar. Er bleibt an allen drei Punkten
  schlechter als `int8` (0,0006–0,0021 getrimmt).
- **Der 128K-Nullpunkt ist mit Wiederholung konfundiert.** Er ist damit kein Beleg dafür, dass die
  Tag-Lesung bei 128K nichts wert ist, sondern nur dafür, dass sie auf einem bereits gelesenen Text
  nichts ändert.
- **Für das Produkt:** im Einsatzbereich ist der Tag eine kostenlose, kleine Verbesserung am
  unteren Rand (32K) und kein Hebel darüber. `int8`-Parität erreicht er dort nicht; ein
  5,0-bit/Zelle-Ziel trägt er nicht, und `B1` bleibt geschlossen.
- **Nächster Schritt:** 128K mit frischem Tail (16 verschiedene Segmente), um Länge und
  Wiederholung zu trennen — und, wenn das trägt, dieselbe Frage bei 256K mit `int8`/`win`/`tag`.

## Reproduktion

```powershell
# ids: erste 4 bzw. 8 Segmente von 2026-11-01-e1-128k/cwe8k-x2.ids.json, je ein Segment
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids ...\2026-11-03-deploy-range\p32k.ids.json `
    --kv <fp16|int8|q4lm32> --prefill auto --short-read 6144 --topk 256 --allow-no-positions `
    --pin 26612 --prompt-cache 3 `
    [--env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 --env STRATA_KV_WINDOW_R=64 `
     [--env STRATA_KV_WINDOW_TAG=1]] `
    --out ...\2026-11-03-deploy-range\32k\<arm>.tsv
#   p64k genauso mit --pin 59367 und ...\64k\<arm>.tsv

python tools\kv_k1.py --ref ...\32k\fp16.tsv --run ...\32k\win.tsv --run ...\32k\tag.tsv `
    --run ...\32k\int8.tsv --kl-topk 256 --block 256 --json ...\32k\k1.json
#   64k genauso
```
