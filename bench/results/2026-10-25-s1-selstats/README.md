# S1 — wo landet die Indexer-Selektion?

**Status: gelaufen.** Zwei 32K-Prefills, 2026-10-25, AMD Radeon RX 9060 XT (gfx1200, 16 GB),
Windows 11, ROCm 7.1, Engine 0.1.47 Debug-Build. Der 2.048-Token-Rauchtest hat vorher einen
Fehler im Rand gezeigt: der unvollständige Tail-Block trägt +1e9 als Zwangs-Bias
(`qsa_select.hpp`), kein Score. Der Rand wird nur über die vollständigen Blöcke genommen.
Katalogzeile: `docs/paper/Ansatz_Improvements.md`, E1 („Deckt das Overlay die gelesenen Zellen?").

## Frage

Das Fenster (`q4lm8v` + `STRATA_KV_WINDOW=1`, S=256, R=64) liest zwei Zellmengen exakt: die ersten
R Zellen (Sinks) und die letzten S Zellen (recent). Alles andere liest quantisiert. Der Indexer wählt
pro Query höchstens 2.051 Zellen aus. **Ob diese ausgewählten Zellen in R oder S fallen, ist
ungemessen.** Genau das entscheidet, ob „mehr S/R" (Kataloghebel B1) der richtige Ausgabeort für Bits
ist, oder ob das Overlay Zellen schützt, die niemand liest.

## Aufbau

| Punkt | Wert |
|---|---|
| Engine | 0.1.47 + S1-Zähler (Debug-Build, `build-hip-port\strata.exe`, SHA-256 `61A40CBF…308FB9`) |
| Modell | swift-1.5-iq3_xxs, `--kv q4lm8v`, `--prefill auto` |
| Fenster | `STRATA_KV_WINDOW=1`, `S=256`, `R=64` |
| Passage | cwe-4k, alle 8 Segmente in Reihenfolge konkateniert = **32.754 Token**, kein Trenner |
| Pfad | batched Prefill (`--short-read` entfernt), ein `GEN 1`-Request, `QUIT` |
| Sampling | jede 32. Query je Selektions-Batch → 1.024 Stichproben je QSA-Layer |
| zusätzlich | `STRATA_SEL_OVERLAP` (vorhandener Haken, Schicht 0, 16er-Kacheln) für die Persistenz |

Der Zähler (`src/prefill/prefill.cpp`, hinter `STRATA_KV_SELSTATS`) hängt direkt hinter
`qsa_block_topk` und schreibt eine Zeile je Stichprobe:

```
layer pos n_kv width blocks in_r in_s in_b outside d2 d4 d8 d16 d32 older min_sel max_unsel s_min s_max
```

`d2`…`d32`/`older` zerlegen die nicht-exakten Treffer nach Abstand hinter dem Horizont, in Vielfachen
von S: `d2` ist die nächsten 256 Zellen (ein doppelt so großes S würde sie fangen), `older` liegt
mehr als 32 S zurück. Sie partitionieren `in_b + outside`.

Die vier Fächer folgen **dem Prädikat des Kernels** (`qsa_win_slot`,
`include/strata/kernels/qsa_decode_attn.hpp:73`):

- `in_r` — `c < R`: Sink-Zeile, wird exakt gelesen.
- `in_s` — `c >= n_kv - S`: recent-Zeile, wird exakt gelesen (Tag-Prüfung besteht im Prefill, weil die
  Zelle zu den letzten S angehängten gehört).
- `in_b` — `n_kv - S - B <= c < n_kv - S`: liegt im Ring, **hat ein bezahltes fp16-Overlay, wird aber
  quantisiert gelesen** (`qsa_win_slot` gibt -1 zurück).
- `outside` — außerhalb des Rings: kein Overlay, quantisiert.

`min_sel`/`max_unsel` sind die Blockscores an der Selektionsgrenze, **nur über die vollständigen
Blöcke**. Block `n_bid` trägt +1e9, sobald der Tail unvollständig ist (`qsa_select.hpp`), und wird
dadurch immer gewählt — das ist kein Score und zählt nicht zum Rand. `s_min`/`s_max` ebenso, sonst
dominiert der Bias die Spreizung. Ein Sentinel (`min_sel` = 1e30 oder `max_unsel` = −1e30) bleibt,
wenn eine Seite leer ist (die Identitäts-Selektion unter 2.051 Zellen); die Auswertung lässt diese
Zeilen aus. Die Selektion läuft über **ganze Blöcke**, deshalb ist `min_sel - max_unsel` exakt der
Rand der vollständigen Blöcke: positiv = sie trennt sauber, ≤ 0 = ein Gleichstand, den die
Blockreihenfolge entscheidet.

## Entscheidungsregeln (vor dem Lauf festgelegt)

| Befund | Lesart | nächster Schritt |
|---|---|---|
| Anteil `in_r + in_s` **≥ 50 %** der gewählten Zellen | Das Overlay sitzt auf dem gelesenen Satz | **B1** (mehr S/R) ist der richtige Hebel, Byte-Modell für R+S ≤ 4.096 |
| Anteil **≤ 10 %** | Das Overlay schützt Zellen, die niemand liest | positionsfestes Overlay ist der falsche Ort; Score-getriebener Pool prüfen |
| dazwischen | unentschieden | der Rand entscheidet mit (nächste Zeile) |
| `in_b` **deutlich > 0** | Der Indexer liest Zellen, deren fp16-Kopie bezahlt und nie exakt gelesen wird | Beleg für **B4** (Ringgröße), *nicht* für B1 |
| Rand `p10(margin/(s_max-s_min))` **≥ 5 %**, Inversionen ≤ 1 % | Selektion ist entscheidbar | ein Score-getriebener Pool wäre stabil |
| Inversionen **> 10 %** oder `p50 ≈ 0` | Grenze ist beliebig | Score-Pool würde flackern; **B1** (mehr Zellen) ist die robustere Antwort |

Bindend ist der Kill der Leiter: **unter ½ der Treffer in R+S läuft B1 nicht** (Katalog S1). Diese
Schwelle ist an diesem Punkt **arithmetisch unerreichbar**: sobald die Selektion wählt, ist die
Breite 2.051 (`qsa_selection_width`), R+S ist 320, die Decke liegt bei **15,6 %**. Ein Lauf kann
die 50 % also nicht erfüllen, und sie als Kill zu lesen würde B1 vor jeder Messung töten. Gelesen
wird stattdessen, was die Schwelle fragen wollte:

- **Recall** — welcher Anteil von R und von S wird überhaupt gewählt? Niedriger Recall heißt: das
  Overlay schützt Zellen, die niemand liest.
- **Wo der Rest sitzt** — `d2` (direkt hinter S, ein größeres Fenster finge ihn) gegen `older`
  (verstreut, mehr S hilft nicht) und `in_b` (bezahlt, quantisiert gelesen) gegen `outside`.

Die 10-%-Zeile bleibt die Lesart für „nahe 0" aus E1. Gelesen wird das nur für `n_kv` > 2.051 —
darunter ist die Selektion die Identität, und die Anteile sind Geometrie, keine Auswahl.

## Konfundierungen (ausdrücklich)

- **32K statt 128K.** Die Anteile sind längensensitiv: `in_s` ist eine feste Zellenzahl (256) über
  einem wachsenden `n_kv`, also fällt der `in_s`-Anteil mit der Länge, sofern die Selektion breit
  streut. 32K zeigt die Größenordnung, nicht den 128K-Wert.
- **Acht Dokumente aneinandergehängt, ohne Trenner.** Die Nahtstellen sind Dokumentgrenzen, die das
  Modell nie gesehen hat. Für Auswahlstatistik unkritisch, für Qualitätsurteile nicht — dieser Lauf
  fällt kein Qualitätsurteil.
- **Annahme, nicht geprüft:** die Prefill-Selektion einer Query entspricht der Decode-Selektion an
  derselben Position. Begründung: die Scores kommen aus `qsa_block_scores` über die gepoolten
  Indexer-Keys, die nur von den Token-Keys abhängen, nicht von der Attention. Der Prefill geht durch
  einen anderen Attention-Kernel (`qsa_prompt_attn`), was die Scores nicht berührt. Ein eigener
  Prefill-gegen-Decode-Vergleich ist **nicht** Teil dieses Laufs.
- **Debug-Build.** Zwei Readbacks und ein Sync je Stichprobe. Kein Timing-Lauf; die Laufzeit sagt
  nichts über den Produktionspfad.
- **`B = 6.144`** stammt aus `--prefill auto` (Chunk-Obergrenze in `generate.cpp`), nicht aus einer
  Qualitätsentscheidung. Siehe `in_b`.
- **`STRATA_SEL_OVERLAP` mischt die Identität mit.** Der Haken summiert alle 16er-Kacheln eines Chunks,
  auch die unter 2.051 Zellen, wo die Selektion die Identität ist (Union/Summe ≈ 6,25 %, der
  Rauchtest maß 6,3 %). Nur die Zeilen von Chunks, die ganz über 2.051 liegen, sagen etwas über die
  Persistenz.
- Der `int8`-Pfad ist unberührt: der Zähler läuft nur, wenn `STRATA_KV_SELSTATS` gesetzt ist. Ein
  Byte-Identitätsnachweis für `int8` mit diesem Build steht noch aus (Gate vor einem Merge, nicht vor
  diesem Lauf).

## Ergebnis

Gelesen für `n_kv` > 16.384, wo alle Abstandsfächer existieren: 6.144 Stichproben, 12 QSA-Layer,
12.601.344 gewählte Zellen. Breite der Selektion 2.051. Decke für R+S = 320/2.051 = **15,6 %**.

| | Anteil der gewählten Zellen | Recall |
|---|---|---|
| R (64 Sinks, exakt gelesen) | 0,9 % | **28,9 %** (Schicht 0: 75,6 %; Schichten 4–8: 13–20 %) |
| S (256 recent, exakt gelesen) | 10,8 % | **86,2 %** |
| B (Ring, bezahlt, quantisiert gelesen) | 59,2 % | — |
| außerhalb | 29,1 % | — |
| exakt zusammen | **11,7 %** | 75 % der Decke |

Der Recall von S ist über die Horizonte stabil (89,0 % bei 2–8K, 88,5 % bei 8–16K, 86,2 % darüber).
Der von R fällt (64,5 % → 42,1 % → 28,9 %).

Die nicht-exakten Treffer sind näher am Horizont dichter als eine Gleichverteilung über `[R, n−S)`,
aber kein schmaler Streifen. Bei `n_kv` > 16.384, Vielfaches gegen Gleichverteilung: `d2` 8,9×,
`d4` 7,1×, `d8` 4,6×, `d16` 1,7×, `d32` 0,68×, `older` 0,41×. Ein größeres S klassifiziert Zellen
um, die schon gewählt sind — es ändert die Selektion nicht:

| Horizont | exakt | Zellen/Query |
|---|---|---|
| S = 256 (heute) | 11,7 % | 239 |
| S = 512 | 20,3 % | 416 |
| S = 1.024 | 34,1 % | 699 |
| S = 2.048 | 52,1 % | 1.069 |
| S = 4.096 | 65,6 % | 1.345 |
| S = 8.192 | 76,2 % | 1.562 |

### Das Overlay ist da, es wird nicht gelesen

Zweiter Lauf, gleiche Passage, dieselben Bedingungen, zusätzlich ein Readback von `win_tag`
(SHA-256 `7E766F88…733D6F`). Der Tag ist die Prüfung, die `qsa_win_slot` selbst macht
(`win_tag[c % (S+B)] == c`, `kv_q4.cu:411`). Bei `n_kv` > 16.384:

| Fach | Tag nennt die Zelle noch |
|---|---|
| S | 100 % (1.356.532 / 1.356.532) |
| B | **98,9 %** (7.374.178 / 7.457.951) |
| außerhalb | 0 % (0 / 3.673.095) |

Ein Gather, der jede getaggte Zeile exakt läse statt nur die letzten S, würde **70,2 %** der
Selektion exakt lesen (die heutigen 11,7 % plus 58,5 Punkte aus B). Dieselben Bytes: der Ring ist
schon resident. Die 29 % außerhalb haben keine Kopie. Der Horizont-Check in `qsa_win_slot`
(`c < n − S` → −1, vor dem Tag) ist der Grund, nicht ein fehlendes Overlay.

### Rand und Persistenz

Kein Gleichstand, der die Reihenfolge entscheidet: 1 Inversion in 6.144 Queries (0,016 %). Der
Abstand ist trotzdem dünn. Median des Rands 0,016 bei einer Spreizung von 117,8; der Median des
Rands geteilt durch die Spreizung derselben Query ist **0,01 %** (p10 unter 0,005 %). Die
vorregistrierte 5-%-Regel für einen Score-Pool scheitert um zwei Größenordnungen.

Dass der Rand dünn ist, zeigt auch der Vergleich zweier Läufe auf denselben Positionen (4.096-Token-
Check gegen das Präfix des 32K-Laufs, `n_kv` > 2.051): 142 von 756 Queries haben eine andere
R- oder S-Zahl, die Aggregate liegen bei 13,61 % gegen 13,61 % exakt. Die Anteile sind stabil,
die Grenze nicht.

`STRATA_SEL_OVERLAP`, nur Schicht 0, Chunks ab 6.144 (ganz über der Identitätsgrenze): eine
16er-Kachel liest 12,7–17,8 % der Summe der Einzelbreiten, gegen 6,25 % bei identischer Auswahl.
Das sind etwa 2,5 Query-Breiten Union. Die Auswahl wandert, aber langsam. Der Haken sagt nichts
über die anderen elf Schichten.

### Entscheidung

- **B1 läuft nicht.** Nicht wegen des 50-%-Satzes — der ist an diesem Punkt unerreichbar —, sondern
  weil ein doppelt so großes S 9 Punkte kauft und das Lesen des vorhandenen Tags 58 Punkte, bei
  null zusätzlichen Bytes. Mehr R trifft eine Schicht (Schicht 0 liest 76 % der Sinks, fünf
  Schichten unter 20 %) und ist der falsche Ausgabeort.
- **E3 in der Größe des heutigen Fensters läuft auch nicht.** 320 Zellen auf die Selektion gelegt
  decken 15,6 %, das Fenster deckt 11,7 %. Vier Punkte, ein neuer Pool.
- **Der Hebel ist der Horizont-Check.** Opt-in, Default unverändert, hinter einem Flag: `qsa_win_slot`
  gibt die Ringzeile zurück, wenn der Tag trifft, nicht nur wenn `c ≥ n − S`. Das ist kein
  Qualitätsanspruch. Ob die 70 % exakt die KL-Lücke zu `int8` schließen, ist ungemessen; das ist
  der nächste Arm, auf cwe-4k, gepaart, gegen das heutige Fenster und gegen `int8`.
- Der Rest (29 %, kein Tag) ist danach die Frage für einen auswahlfolgenden Pool, nicht davor.
- Ein Score-getriebener Pool hängt an einem Rand von 0,014 % der Spreizung. Nicht bauen.

### Was das nicht ist

- **32K, nicht 128K.** Der Nah-Bias wird mit der Länge steiler (`d2` gegen Gleichverteilung: 1,7×
  bei 2–8K, 4,7× bei 8–16K, 8,9× darüber). Ob die 70 % bei 128K halten, ist ungemessen; die
  Richtung des `outside`-Anteils ist offen.
- **Prefill, nicht Decode.** Dieselbe Prädikat-Funktion, nicht derselbe Lauf. Der Decode-Pfad ist
  nicht gezählt.
- **Kein Qualitätsurteil.** Der Lauf schreibt keine Logits.
- **Kein Timing.** Der Anteilslauf las 32.754 Token in 253,9 s (129 tok/s) inklusive der
  Debug-Syncs. Das sagt nichts über K5.
- Der `int8`-SHA dieses Debug-Builds ist nicht geprüft. Der Zähler läuft nur mit gesetzter Variable.

## Reproduktion

```powershell
# 1. Bauen. ROCm-clang zielt auf die MSVC-ABI, also zuerst vcvars64 laden (docs/AMD_HIP.md), sonst
#    bricht der Build mit "'cstdint' file not found" ab; danach ROCm und Ninja auf den Pfad.
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && set" |
    ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
$env:PATH = "C:\Program Files\AMD\ROCm\7.1\bin;" + $env:PATH
& "C:\Users\krone\git\kv-quant-strata\.venv\Scripts\ninja.exe" -C build-hip-port strata.exe

# 2. Rauchtest: 2.048 Token, prüft nur, dass die TSV entsteht und plausibel ist
python tools\kv_selstats.py run --out C:\Users\krone\git\Strata-data\bench\2026-10-25-s1-smoke `
    --exe build-hip-port\strata.exe --tokens 2048 --overlap

# 3. Der Lauf
python tools\kv_selstats.py run --out C:\Users\krone\git\Strata-data\bench\2026-10-25-s1-selstats `
    --exe build-hip-port\strata.exe --overlap

# 4. Auswertung
python tools\kv_selstats.py report C:\Users\krone\git\Strata-data\bench\2026-10-25-s1-selstats\selstats.tsv

# 5. Tag-Lauf (win_tag-Readback; überschreibt den Anteilslauf nicht)
python tools\kv_selstats.py run --out C:\Users\krone\git\Strata-data\bench\2026-10-25-s1-taghit `
    --exe build-hip-port\strata.exe
python tools\kv_selstats.py report C:\Users\krone\git\Strata-data\bench\2026-10-25-s1-taghit\selstats.tsv
```

Rohdaten (TSV, `stderr.txt`, `index.json`) liegen außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-10-25-s1-selstats\` und `…-s1-taghit\`; hier im Repo
stehen nur die Aggregate dieses Berichts. Anteilslauf SHA-256 `61A40CBF…308FB9`, Tag-Lauf
`7E766F88…733D6F`.
