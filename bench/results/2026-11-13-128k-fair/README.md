# Die 128K-Frage gegen die faire Referenz (Vorregistrierung, 2026-11-13)

Diese Datei legt fest, **was gemessen wird und wann es zählt**, bevor der Lauf startet. Sie führt die
Frage des 4K-Laufs (`bench/results/2026-11-13-int8-references/`) auf dem langen Text: **ist der
q4-Codec der fairen `int8`-Referenz auf 128K nicht unterlegen?** Und sie prüft die Vorhersage, dass
die **Mittelkalibrierung auf 128K deutlich gewinnt** — die teilweise schon gemessen ist (siehe
„Der Vortest").

## Warum das der richtige nächste Schritt ist

Auf 4K war der Abstand `q4lm32w` zu `int8` praktisch null (+0,00002 nats, die alte Parität). Auf 128K
ist er **300× größer**: `int8 − q4lm32w` = **−0,00741** nats (t −4,80, 95-%-CI [−0,01044, −0,00439]),
gemessen am 11.11. auf demselben Build und demselben Text. Der 4K-Lauf hat gezeigt, dass die alte
Parität gegen eine **unbehandelte** Referenz gelesen war; auf 128K ist der Abstand so groß, dass die
Frage dort **entschieden** werden kann und nicht an der Auflösung hängt.

## Der Vortest: was schon gemessen ist (ohne GPU)

Die 128K-Arme aus `d1` (11.11.) liegen alle auf demselben Build, demselben Text (`t1`, 131.060
Positionen) und denselben `t1.ids.json`. Aus `d1/compare-all6.json` und `d1/paired-all6.json`:

| Arm | mittlere KL gegen fp16 | Top-1 | p99 |
|---|---:|---:|---:|
| `fp16` (Referenz) | 0 | 100 % | 0 |
| `int8` (R1, unbehandelt) | 0,06435 | 98,45 % | 1,6615 |
| `q4lm32w` (Produkt) | 0,07176 | 98,30 % | 2,0583 |
| `q4lm32w`+C | 0,06861 | 98,38 % | 1,9240 |
| `q4lm8v-win` (ausgeliefert, 16/8) | 0,07524 | 98,16 % | 2,1567 |
| `q4lm8v-win`+C | 0,07479 | 98,18 % | 2,3328 |

Die gepaarten Differenzen (per Position, **1 Segment**, deshalb kein Segment-Intervall):

| Differenz | Mittel | t | 95-%-CI |
|---|---:|---:|---|
| `int8 − q4lm32w` | −0,00741 | −4,80 | [−0,01044, −0,00439] |
| `int8 − q4lm32w+C` | −0,00427 | −2,91 | [−0,00714, −0,00139] |
| **`q4lm32w − q4lm32w+C`** | **+0,00314** | **2,05** | **[+0,00014, +0,00615]** |
| `q4lm8v-win − q4lm8v-win+C` | −0,00045 | −0,28 | [−0,00362, +0,00271] |
| `q4lm32w − q4lm8v-win` | −0,00348 | −2,23 | [−0,00655, −0,00042] |

**Das ist die Vorhersage, halb bestätigt und halb offen:**

1. **Die Kalibrierung hilft auf 128K — auf dem 16/16-Arm.** `q4lm32w` wird durch C um **0,00314 nats**
   besser (0,07176 → 0,06861, **−4,4 %**), das ist 1,24× die 128K-Auflösung (0,00254), und das
   Intervall schließt die 0 knapp aus. Auf 4K war dasselbe Paar **+0,00079** in die *falsche*
   Richtung (0,01768 gegen 0,01689). Das Vorzeichen dreht also mit der Länge. **Bestätigt — aber
   nicht „deutlich": die untere Grenze ist +0,00014, der Befund ist ein Einzelsegment-Befund mit
   t = 2,05.**
2. **Auf dem ausgelieferten 16/8-Arm hilft sie nicht:** `q4lm8v-win − center` = −0,00045 (t −0,28),
   Intervall über die 0. Die Vorhersage gilt für 16/16, **nicht** für die ausgelieferte 16/8-Form.
3. **Die Kalibrierung auf `int8` ist auf 128K offen.** R2a/R2b gibt es nur auf 4K. Und der Punkt, der
   leicht verloren geht: die Kalibrierung hilft **beiden Seiten**. Dass der Produktarm durch C
   gewinnt, heißt **nicht**, dass der Abstand zur Referenz kleiner wird.

## Die Arme

Neun Arme, davon **drei neu**. Die sechs vorhandenen werden **nicht** neu gefahren: derselbe Build
(0.1.59.1), derselbe Text, dieselben `set.ids.json` — und die Paarung ist **belegt**, nicht
angenommen: `d2/t1-int8.tsv` (13.11., 12:03) ist **byte-identisch** zu `d1/t1-int8.tsv` (11.11.,
10:43), SHA-256 `f18551dfd9ef…` auf beiden. Es braucht deshalb **keinen Reproduktionsarm**.

| # | Arm | Stand |
|---|---|---|
| 1 | `fp16` (Referenz der absoluten KL) | vorhanden (`d1`) |
| 2 | **R1** `int8` (heutiger Produktpfad) | vorhanden (`d1` = `d2`) |
| 3 | `q4lm32w` (Produkt, 16/16 + Fenster) | vorhanden (`d1`) |
| 4 | `q4lm32w` + C | vorhanden (`d1`) |
| 5 | `q4lm8v-win` (ausgeliefert, 16/8 + Fenster) | vorhanden (`d1`) |
| 6 | `q4lm8v-win` + C | vorhanden (`d1`) |
| 7 | **R2a** `int8` + Rotation (`STRATA_KV_ROT=1`) | **neu** |
| 8 | **R2b** `int8` + Rotation + Kalibrierung (`STRATA_KV_CENTER`) | **neu** |
| 9 | `q4u32` (uniformes Gitter, LS-Schritt, + Fenster) | **neu** |

Geometrie: `cwe-128k`/`t1`, 131.063 Token, Produktionsfenster bei 128K (**S 256, R 64, Chunk 6.144**),
`--prefill 6144`, `STRATA_VERIFY_ASYNC_LAUNCH=1` für alle Arme; das Fenster (`STRATA_KV_WINDOW=1`)
tragen nur die q4-Arme. Die `int8`-Arme tragen es nicht — sie sind die Referenz, nicht ein q4-Modus.

## Die Endpunkte, und die Grenze, die **vorher** feststeht

Alle Differenzen sind **gepaart über Positionen** (dieselben `t1.ids.json`), KL gegen den `fp16`-Arm.
Die **Nichtunterlegenheitsgrenze** ist die Auflösung des Textes, wie sie das Referenzpaar selbst
ausweist: `parity_bound` = 0,0025380 (`int8 − q4lm32w`); die übrigen Paare liegen bei 0,00241–0,00266.
Sie wird hier auf **0,00254** festgelegt und **nicht** nach dem Ergebnis verschoben.

| | Frage | Endpunkt | Gate |
|---|---|---|---|
| **G1** | ist der **Produktarm** der fairen Referenz nicht unterlegen? | `q4lm32w+C − R2b` | obere 95-%-Grenze **< +0,00254**, Punkt **≤ +0,00127** |
| **G2** | **gewinnt die Mittelkalibrierung auf `int8` bei 128K?** | `R2b − R2a` | Punkt **< 0** und 95-%-Intervall **schließt 0 aus** |
| **G3** | ist der **Fast-Path-Kandidat** der fairen Referenz nicht unterlegen? | `q4u32 − R2b` | wie G1 |
| **P1** | rechnet der Auswerter richtig? | `q4lm32w − q4lm32w+C` | **+0,00314, t 2,05** (der Vortest, aus `d1`) |

**P1 wurde vor dem Lauf getrocknet:** der Auswerter lief mit den sechs vorhandenen Armen (die drei
neuen durch Platzhalter belegt) und reproduzierte den Vortest **exakt**: +0,0031433 (t 2,0518). Die
G1/G2/G3-Zahlen des Trockenlaufs sind bedeutungslos; geprüft wurde die Rechenkette (gepaarte
Differenz, t-Intervall, Block-Bootstrap, Tail, McNemar). Dabei fielen zwei Fehler auf, die jetzt
behoben sind: die **Richtung** von P1 (der Vortest-Wert gehört zu `q4lm32w − q4lm32w+C`) und ein
**Overflow** im exakten McNemar (`2^nn` bei 131.060 Positionen; jetzt im Log-Raum).

**G2 ist die Vorhersage des Nutzers, falsifizierbar formuliert:** „die Mittelkalibrierung gewinnt auf
128K deutlich" gilt als **bestätigt**, wenn `R2b − R2a` negativ ist und das Intervall die 0 ausschließt.
Ist der Punkt ≥ 0 oder enthält das Intervall die 0, ist sie für `int8` **nicht bestätigt** — unabhängig
davon, was der 16/16-Arm zeigt. (Der Vortest deutet hin: 0,07176 → 0,06861. Aber das ist ein anderer
Arm.)

**Nebenbefunde, keine Endpunkte:** das 1-%-Tail (über die KL von R2a definiert, also armunabhängig),
Top-1 mit exaktem McNemar, p99, und `q4lm8v-win` gegen `q4lm32w` (der 16/8-Arm ist auf 128K
**schlechter**, −0,00348 im Vortest — das ist eine Aussage über die ausgelieferte Form, die bisher
niemand auf 128K geprüft hat).

## Die Statistik bei **einem** Segment

`t1` ist **ein** Segment (`first_line 0`, `end_line 131060`). Die Segment-t-Verteilung über 8 Segmente
des 4K-Laufs gibt es hier **nicht**. Deshalb sind **zwei** Intervalle vorab festgelegt, und das Gate
benutzt die **breitere obere Grenze** von beiden:

* **per Position** (t-Verteilung über 131.060 Positionen) — wie in `d1/paired-all6.json`;
* **Block-Bootstrap** über **32 feste Blöcke à 4.096 Positionen** (der letzte Block hat 4.052),
  10.000 Resamples, Seed 20261113, Perzentil-Intervall. Das ist die einzige Form, die die
  Autokorrelation der KL-Serie grob auffängt; ohne sie wäre das per-Position-Intervall zu eng.

## Die Erwartung, vorab und falsifizierbar

Aus dem Vortest: `int8 − q4lm32w+C` = −0,00427. Wenn R2b gegenüber `int8` um 0,002–0,004 besser wird
(die 4K-Größe von R1 → R2b war −0,00212), liegt `q4lm32w+C − R2b` bei **+0,006 bis +0,008**. **G1
scheitert damit nach der Vorhersage** — die Nichtunterlegenheit ist auf 128K nicht zu erwarten, und der
Lauf **quantifiziert** den Abstand, statt ihn zu entscheiden. G3 (derselbe Abstand für `q4u32`) ebenso.
**Das ist die ehrliche Erwartung; ein anderes Ergebnis wäre eine Überraschung und würde geprüft, nicht
gefeiert.**

## Die Fallen, die vorher benannt sind

* **R2 kann kaputt sein.** `int8` + Rotation ist auf 128K nie gelaufen. Ist die KL von R2a **mehr als
  3×** die von R1, gilt der Pfad als **defekt**, nicht als Ergebnis; dann sind G1/G3 **nicht
  entscheidbar** und das wird so berichtet.
* **Die Kalibrierungstabelle ist für `kLloydWin` gebaut** (`center-table.bin` aus dem S2-Probe) und
  wird auf `int8` und `q4u32` nur angewandt, wo der Arm sie trägt. Ob sie dort hilft, ist G2 bzw.
  ein Nebenbefund — keine Voraussetzung.
* **Ein Lauf, keine Parallele.** Drei Arme à ~88 min (die `d1`-Zeitstempel: 09:24 → 16:43 für fünf
  Arme) sind ~4,4 h; 30 s zwischen Engine-Starts. Kein Build während des Laufs.
* **`t1` allein.** `t2` wird **nicht** gefahren. Der Vortest ist auf `t1`; eine Aussage über beide
  128K-Texte braucht `t2` und ist eine eigene Stufe.

## Was diese Stufe nicht behauptet

* Nichts über **Laufzeit**. (Der `q4u32`-Leser ist der billigste — `(n−8)*d` ohne Tabelle,
  `qsa_decode_attn.cu:130` — aber die Prompt-Zeiten der 4K-Arme sind Einzelmessungen und hier kein
  Endpunkt.)
* Nichts über **`t2`** und nichts über andere Modelle.
* Nichts über den **Writer-Preis** der Kalibrierung.
* Nichts über die **Ursache** des 128K-Tail-Verhaltens (offene Frage aus dem 4K-Lauf: `int8` hat
  wenige katastrophale Positionen, q4 mehr kleine Fehler).

## Rohdaten

`C:\Users\krone\git\Strata-data\bench\2026-11-13-128k-fair\arms\` (neu) und
`C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\d1\` (die sechs vorhandenen Arme).

---

## Das Ergebnis (nach dem Lauf)

Drei Arme, ein Job, 30 s zwischen den Starts, kein Build. Alle drei auf demselben Build
(0.1.59.1), `cwe-128k`/`t1`, **131.060 Positionen** je Arm:

| Arm | Zeit | SHA-256 |
|---|---:|---|
| R2a `int8` + Rotation | 4.884,7 s | `1f0257996d03…` |
| R2b `int8` + Rotation + Kalibrierung | 4.684,8 s | `407c882f6a5e…` |
| `q4u32` + Fenster (S 256 / R 64 / Chunk 6.144) | 5.043,6 s | `debdee3a8cd3…` |

Die **vorab definierte Defektprüfung** (R2a > 3× R1 ⇒ Pfad defekt) ist **nicht** eingetreten:
R2a/R1 = **0,970**. Der `int8`+ROT-Pfad ist auf 128K benutzbar, G1/G3 sind entscheidbar.

| Arm | mittlere KL gegen fp16 | Top-1 | p99 |
|---|---:|---:|---:|
| `fp16` (Referenz) | 0 | 100 % | 0 |
| R1 `int8` | 0,06435 | 98,45 % | 1,6615 |
| **R2a** `int8`+ROT | **0,06238** | 98,49 % | 1,5133 |
| **R2b** `int8`+ROT+C | **0,06419** | 98,41 % | 1,7455 |
| `q4lm32w` (Produkt) | 0,07176 | 98,30 % | 2,0583 |
| `q4lm32w`+C | 0,06861 | 98,38 % | 1,9240 |
| `q4lm8v-win` (ausgeliefert, 16/8) | 0,07524 | 98,16 % | 2,1567 |
| `q4lm8v-win`+C | 0,07479 | 98,18 % | 2,3328 |
| **`q4u32`** | **0,06949** | 98,31 % | 1,9837 |

### Die Gates, wie sie vor dem Lauf festlagen

| Gate | Paar | Punkt | obere Grenze | Grenze | Urteil |
|---|---|---:|---:|---:|---|
| **G1** | `q4lm32w+C − R2b` | **+0,00442** (t 2,90) | **+0,00741** (Bootstrap +0,00696) | +0,00254 | **NICHT BESTANDEN** |
| **G2** | `R2b − R2a` | **+0,00181** (t 1,29) | +0,00456 (Bootstrap +0,00326) | < 0 | **NICHT BESTANDEN** |
| **G3** | `q4u32 − R2b` | **+0,00530** (t 3,44) | **+0,00832** (Bootstrap +0,00821) | +0,00254 | **NICHT BESTANDEN** |
| **P1** | `q4lm32w − q4lm32w+C` | +0,00314 (t 2,05) | — | +0,00314 | **BESTANDEN** |

**P1 hat den Vortest exakt reproduziert** (+0,0031433, t 2,0518) — der Auswerter rechnet richtig,
und die sechs Altarme sind mit den drei neuen gepaart vergleichbar.

### G1: der Produktarm ist der fairen Referenz auf 128K unterlegen

`q4lm32w+C − R2b` = **+0,00442** nats, 95-%-Intervall [+0,00144, +0,00741] je Position,
Block-Bootstrap [+0,00196, +0,00696]. Die vorab fixierte Grenze war **+0,00254**; die obere Grenze
liegt bei **2,9×** dieser Grenze.

**Und die Referenz ist dabei noch die *milde* Wahl.** R2b ist mit **0,06419** schlechter als R2a
(0,06238) — die Kalibrierung hat die `int8`-Referenz auf 128K *verschlechtert* (G2). Gegen die
**bessere** faire Referenz R2a wird der Abstand größer, nicht kleiner:

| Paar | Punkt | t |
|---|---:|---:|
| `q4lm32w+C − R2b` (vorab fixiert) | +0,00442 | 2,90 |
| `q4lm32w+C − R2a` (bessere Referenz) | **+0,00623** | 4,23 |
| `q4u32 − R2b` (vorab fixiert) | +0,00530 | 3,44 |
| `q4u32 − R2a` (bessere Referenz) | **+0,00711** | 4,83 |

Das Gate ist also nicht an einer zu strengen Referenz gescheitert, sondern trotz einer zu
nachsichtigen.

### G2: die Kalibrierung gewinnt auf `int8` nicht — sie verliert

`R2b − R2a` = **+0,00181** (t 1,29). Das per-Position-Intervall [−0,00095, +0,00456] enthält die 0,
der **Block-Bootstrap [+0,00034, +0,00326] schließt sie aus** — und zwar auf der *positiven* Seite.
Die Annahme hinter der Vorabschätzung (die Kalibrierung verbessert die Referenz wie auf 4K) trägt auf
128K damit **nicht**; das Vorzeichen dreht zwischen den Längen (4K −0,00103, t 0,96; 128K +0,00181).

Die Kalibrierung ist **codec-abhängig**. Wirkung = kalibriert − unkalibriert (negativ ist besser),
auf 128K:

| Arm | Wirkung der Kalibrierung auf 128K |
|---|---|
| `q4lm32w` (16/16) | **−0,00314** (besser, t −2,05) |
| `q4lm8v-win` (16/8) | −0,00045 (t −0,28, nicht signifikant) |
| `int8` | **+0,00181** (schlechter, Bootstrap schließt 0 aus) |

Zwei der drei Arme sind signifikant, und sie zeigen in **entgegengesetzte** Richtungen.

### G3: auch der Fast-Path-Kandidat ist unterlegen — aber er ist der beste q4-Arm

`q4u32 − R2b` = **+0,00530** (t 3,44), obere Grenze +0,00832 → **nicht bestanden**.

Der Nebenbefund ist trotzdem der praktisch interessanteste dieser Stufe: **`q4u32` (0,06949) liegt
auf dem Niveau des kalibrierten Produktarms** —

| Paar | Punkt | 95-%-Intervall | t |
|---|---:|---|---:|
| `q4lm32w+C − q4u32` | **−0,00088** | [−0,00389, +0,00214] | −0,57 |
| `q4lm32w − q4u32` | **+0,00226** | [−0,00076, +0,00529] | 1,47 |
| `q4lm8v-win − q4u32` | **+0,00575** | [+0,00267, +0,00882] | 3,66 |

`q4u32` ist von `q4lm32w+C` **nicht zu unterscheiden** (−0,00088, Intervall um 0) und gegen
`q4lm32w` um 0,00226 besser — dieser Betrag liegt allerdings **unter** der Auflösung des Textes
(0,00254) und ist als Unterschied zwischen zwei Armen nicht belastbar (das gepaarte Intervall
enthält die 0).

Was daran zählt: `q4u32` braucht **keine Kalibrierungstabelle** (die `center-table.bin` dieses
Modells ist 48 KB groß) und kommt im Decode mit einem einheitlichen Schritt ohne Tabellenzugriff aus
(statische Analyse: `qsa_decode_attn.cu:130`; **nicht** gemessen ist, ob das schneller ist). Auf beiden
Längen liegt er auf dem Niveau des kalibrierten 16/16-Arms: 128K −0,00088 (n. s.), 4K +0,00263
(t 1,94, Intervall enthält 0) gegen +0,00293 (t 2,42) für `q4lm32w+C` — beide gegen die faire
Referenz. Das ist eine **relative** Aussage innerhalb der q4-Familie, **keine** Nichtunterlegenheit
gegen `int8` — die ist mit G1/G3 gemessen und gescheitert.

### Der Abstand schrumpft relativ, wächst absolut

| Text | faire Referenz | `q4lm32w+C` | absolut | relativ |
|---|---:|---:|---:|---:|
| cwe-4k | 0,01476 | 0,01769 | +0,00293 | **+19,8 %** |
| cwe-128k | 0,06419 (R2b) | 0,06861 | +0,00442 | **+6,9 %** |

Der q4-Codec skaliert mit der Länge **relativ besser** als `int8` (19,8 % → 6,9 %), **absolut
schlechter** (+0,00293 → +0,00442). Für die Nichtunterlegenheitsfrage zählt absolut in nats — dort
scheitert das Gate. Das Bild „q4 holt relativ auf" ist die gute Nachricht dieser Stufe, aber es ist
keine Parität.

### Meine Vorabschätzung war zu pessimistisch — und warum

Vorab notiert: **+0,006 bis +0,008** („G1 scheitert nach der Vorhersage"). Gemessen: **+0,00442**.
Die Richtung war richtig, die Schätzung lag um **26–45 %** zu hoch (je nach Ende des vorab notierten
Bandes). Der Grund ist benennbar: die Schätzung setzte `int8 − R2b ≈ int8 − R2a` und unterstellte
damit, dass die Kalibrierung die **Referenz** verbessert (4K-Größe R1 → R2b: −0,00212). Sie
verbessert sie auf 128K nicht, sie verschlechtert sie (+0,00181) — die Referenz bleibt praktisch bei
R1 (`R2b − R1` = −0,00015, t −0,11). Der Referenzabstand ist damit um genau diesen Betrag kleiner
als geschätzt.

### Zwei Befunde, die die 4K-Stufe **nicht** reproduziert

1. **Der Tail-Vorteil von q4 ist auf 128K weg.** Auf 4K holte `q4lm32w` auf dem schlimmsten 1 % von
   `int8`+ROT **−0,329** nats (t −3,04). Auf 128K, auf demselben Kriterium (die 1.311 Positionen mit
   der größten KL von R2a): R1 3,408 · R2b 3,332 · **`q4u32` 3,381** · `q4lm32w+C` 3,390 ·
   `q4lm32w` **3,601**. Kein q4-Arm gewinnt den Tail; `q4lm32w` ist dort gegen **beide** `int8`-Arme
   schlechter (R1 +0,19, R2b +0,27). Die Unterschiede der übrigen Arme liegen bei ≤0,2 nats auf einer
   Basis von 3,4.
2. **Top-1:** `q4lm32w+C` gegen R2b ist mit **894 | 934 (p = 0,36)** nicht unterscheidbar,
   `q4u32` gegen R2b verliert dagegen deutlich (**864 | 1000, p = 0,0018**). Auf 4K war es
   umgekehrt (`q4lm32w` besser, McNemar p = 0,0025). Top-1 ist kein Endpunkt dieser Stufe, aber die
   Aussage „q4 hat die bessere Fehlerform" ist längenabhängig und damit nicht als Eigenschaft des
   Codecs zu führen.

### Was diese Stufe entscheidet, und was nicht

**Entschieden:** der Produktcodec (`q4lm32w`, 16/16 + Fenster) ist der **fairen** `int8`-Referenz auf
128K **nicht** nicht-unterlegen (+0,00442, obere Grenze +0,00741 gegen 0,00254). Die
Mittelkalibrierung wirkt auf 128K **codec-abhängig** (−0,00314 auf 16/16, −0,00045 auf 16/8,
+0,00181 auf `int8`) und hilft der Referenz nicht. Der Fast-Path-Kandidat `q4u32` ist der beste
q4-Arm auf 128K (Punktschätzung), aber ebenfalls unterlegen, und er erreicht das Niveau des
kalibrierten Produktarms ohne Tabelle.

**Nicht entschieden:** ob eine andere Geometrie (mehr Ringzeilen, anderes S/R) den Abstand schließt —
das wäre ein eigener Arm. Nichts über `t2` (nicht gefahren, wie vorab festgelegt). Nichts über
Laufzeit (`q4u32` liest ohne Tabelle — statische Analyse, `qsa_decode_attn.cu:130` —, aber die
Prompt-Zeiten sind Einzelmessungen und hier kein Endpunkt). Nichts über andere Modelle. Nichts über
die **Ursache** des Tail-Verhaltens.

## Rohdaten des Laufs

`C:\Users\krone\git\Strata-data\bench\2026-11-13-128k-fair\` — `arms/` (drei TSVs + `seg.json`),
`compare-fp16.json` (absolute Zahlen der acht Nicht-Referenz-Arme), `paired-fp16.json` (die Paare der
sechs Altarme; die Paare mit `q4u32` stehen in `run.log`), `gates.json` (die Gates mechanisch),
`run.log`. Reproduzieren:

```sh
python bench/results/2026-11-13-128k-fair/run-128k-fair.py
```
