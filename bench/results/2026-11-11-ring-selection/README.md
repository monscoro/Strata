# D2 — das Auswahlverhalten des Indexers: trägt ein auswahlfolgender Ring die 128K-Qualität?

**Status: vorregistriert (2026-11-11).** Der Lauf startet, **nachdem** die kalibrierten 128K-Arme
fertig sind (D1: `q4lm32w`+Centering und `q4lm8v`+Fenster+Centering). Dieses Dokument steht vor
dem Lauf; die Gatter stehen fest. Es ist **kein Codec-Experiment**, sondern ein **Zähl-Experiment**:
ein Instrument im Engine, ein 128K-Lauf, eine Auswertung auf dem Host. **Kein Kernel wird gebaut,
bevor die Zahlen vorliegen.**

## 0. Ergebnisoffenheit — was dieses Experiment **nicht** ist

**Die Vorannahme ist „unbekannt", nicht „L5 ist tot" und nicht „L5 trägt".** Die Kampagne hat bis
hierher **nur eine** Eigenschaft der Auswahl gemessen: die **Überdeckung** des positionalen Rings
(8,55 % der Auswahlen fallen in R+S, E1). Das **zeitliche Verhalten** der Auswahl — ihre
Stabilität, ihr Wachstum, ihr Verhältnis zu den schweren Positionen — ist **unmessbar** bisher, und
es gibt **keinen** Grund, es in die eine oder die andere Richtung zu vermuten.

**Daher die Regeln dieses Abschnitts, vor dem Lauf festgelegt:**

1. **Die Messung ist eine Messung.** Die sechs Größen (D2-a … D2-f) werden erhoben, **was auch immer
   sie ergeben**. Sie werden nicht gegen eine Erwartung geprüft, sondern als Zahlen berichtet — mit
   der Verteilung, nicht mit einem Vergleich.
2. **Die Schwellen (G9, G10) sind Entscheidungsregeln, keine Prognosen.** Sie legen fest, **was ich
   tue**, wenn die Zahl in die eine oder die andere Richtung fällt — nicht, was ich erwarte. Eine
   Regel, die man vorher kennt, ist kein Test, sondern eine Vereinbarung.
3. **Drei Ausgänge, nicht zwei.** Der dritte (marginal) wird ausdrücklich benannt, weil er die
   interessanteste sein kann: eine Vereinigung, die *knapp* unter der Kapazität liegt, wäre der
   Hinweis, dass die **5,0-Marke** zu klein für den auswahlfolgenden Pool ist, während der Hebel
   **bei einem größeren Ringbudget** trägt — das wäre ein Ergebnis über die *Zielmarke*, nicht über
   den Hebel.
4. **Ein Fehlschlag der Gatter schließt L5 nicht für alle Zeit**, sondern für **dieses Budget**
   (4.096 Zellen = 0,50 bit/cell bei 128K). Die Regel trägt das Budget im Text, nicht den Hebel
   allgemein.
5. **Die Brücke zum 128K-Befund ist eine Frage, kein Filter.** Ob die schweren Positionen *gelesen*
   werden (D2-f), wird **gemessen**, nicht vorausgesetzt. Lägen sie nicht unter den Auswahlen, wäre
   das eine Aussage über den Indexer (er liest sie nicht) — und **kein** Argument gegen die
   Kalibrierung oder gegen den Codec.

## 1. Die Frage

Der 128K-Befund (Protokoll 2026-11-11 12:40): der Produktarm (16/16 + Fenster) liegt bei 128K
**+0,00741 nats hinter `int8`** (t = +4,80); K2 und K3 fallen. Der Fehler hängt an **~2.400
Positionen**, die **über den ganzen Text verteilt** sind, und **nicht** in den letzten 256 Zellen
liegen — der **positionale** Ring schützt sie nicht. Die Kampagne hat gemessen, dass der positionale
Ring nur **8,55 %** der Indexer-Auswahlen trifft (E1): **91,45 % der gelesenen Zellen bleiben
4-Bit.**

**L5 (der auswahlfolgende Pool) ist die einzige Ring-Variante, die diese Lücke schließen könnte.**
Sie ist im Wortlaut **nicht** umsetzbar, und der Grund ist ein Zeitproblem, kein Speicherproblem:
der exakte fp16-Zwilling entsteht **beim Schreiben** (`kv_append_win_steps`, `kv_q4.cu:479`), und
seine Zeile ist **positional** (`win_row_of(pos, S, R, B)`); die Auswahl des Indexers
(`qsa_block_topk`, `verify.cpp`) kennt man **erst danach**, wenn die Zelle schon quantisiert im
Bulk liegt. Ein „auswahlfolgender Pool" müsste beim Schreiben **vorhersagen**, welche Zellen später
gelesen werden.

**D2 misst die Größe, die dieser Vorhersage zugrunde liegt: Wie verhält sich die Auswahl des
Indexers über die Zeit?** Daraus folgt, ob eine **bounded** Pool-Größe (4.096 Zellen = 0,50 bit/cell
bei 128K) die Auswahl überhaupt „einholen" kann — und ob die Zellen, an denen der Fehler hängt,
überhaupt gelesen werden.

## 2. Was gemessen wird

Der Instrument-Punkt ist die Auswahl selbst: `qsa_block_topk` schreibt die gewählten **Block-Ids**
pro Query in `sel_` (`verify.cpp`, der Verify-Pfad — derselbe Pfad, den die 128K-Läufe von D1
nehmen). Der Dump (compile-time-Schalter, default OFF, wie `STRATA_Q4_DUMP`/`QSA_LAYER_TRACE`)
schreibt **für jeden Token und jeden der 12 QSA-Layer** die gewählten Block-Ids mit den
zugehörigen Positionen: ~131.060 × 2.048 × 12 Einträge ≈ 13 MB.

**Kein Ausschnitt, keine Stichprobe, keine Auswahl der Abschnitte** — der Dump nimmt **jeden** Token
des Textes. (Ein früherer Entwurf dieses Dokuments sah zwei Textabschnitte vor; die Wahl war
unbeabsichtigt **nicht neutral**: die beiden Abschnitte lagen genau dort, wo die schweren Positionen
am seltensten sind. Die Stichprobe ist deshalb entfallen — die volle Auswahl kostet 13 MB und ist
damit billiger als die Bedenken.)

| # | Größe | Warum |
|---|---|---|
| **D2-a** | Größe der Auswahl je Query (in Zellen) | die Grundlast des Pools |
| **D2-b** | Überdeckung des positionalen Rings (R+S) durch die Auswahl | E1 sagt 8,55 %; nachgemessen auf dem Merge-Build und je Layer |
| **D2-c** | **Wachstum der Vereinigung** \|∪ der Auswahlen der letzten N Queries\| gegen N | die Kapazitätsfrage: der Pool hat 4.096 Zellen |
| **D2-d** | **Churn**: der Anteil *neuer* Selektionen je Query (und bei den Lags 1, 8, 64) | wie stabil ist die Auswahl; bestimmt auch, wie weit ein prädiktiver Pool „hinterherlaufen" müsste |
| **D2-e** | Zeitprofil: wo liegen die ausgewählten Positionen (alt/jung)? | der Indexer liest 91,45 % alte Zellen — wie alt? |
| **D2-f** | **Überdeckung der tail-Positionen**: liegen die ~2.400 schweren Positionen unter den Auswahlen? | die Brücke zum 128K-Befund — **gemessen, nicht vorausgesetzt** |

## 3. Entscheidungsregeln (vor dem Lauf festgelegt; keine Prognosen)

- **G9 (Kapazität):** das Wachstum der Vereinigung (D2-c) gegen die Pool-Kapazität (4.096 Zellen).
  **L5 öffnet sich**, wenn die Vereinigung der Auswahlen der letzten 4.096 Queries **wesentlich
  kleiner** ist als 4.096 (Faktor ≥ 4 Reserve). **Marginal** (Faktor 1–4): der Hebel trägt, aber
  nicht **dieses** Budget — die 5,0-Marke ist dann zu klein, der Hebel nicht (das ist eine Aussage
  über die Zielmarke). **Geschlossen** (Faktor < 1, die Vereinigung übersteigt die Kapazität): kein
  bounded Pool kann die Auswahl einholen — arithmetisch geschlossen, ohne Kernel.
- **G10 (die Brücke):** der Anteil der tail-Positionen an den Auswahlen (D2-f). **Die Brücke trägt**
  bei ≥ 50 %. Sie **trägt nicht** darunter: dann gibt es keinen Ring, der diese Zellen schützt, und
  der 128K-Befund ist **kein** Ring-Problem. (Das wäre eine Aussage über den Indexer, nicht über
  den Codec.)
- **G11 (der Vertrag):** unabhängig von G9 und G10 ist zu klären, ob der Ring „exact" noch eine
  **Garantie** ist oder eine **Vermutung** (der Fenster-Vertrag: „ein stilles Fehllesen ist schlimmer
  als eine laute Verweigerung", A1–A5). Ein auswahlfolgender Pool ist per Konstruktion eine
  Vermutung — das muss in Setup und Paper **sichtbar** heißen, sonst wäre es der A3-Fehler. **Das ist
  keine Messung, sondern eine Festlegung**, und sie gilt unabhängig vom Ausgang von G9/G10.

**Was in keinem Ausgang gilt:** dass die Kalibrierung, der Level-Split oder der Stock-Vergleich
anders zu bewerten wären. Die Codec-Achse ist gemessen (Level-Split signifikant +0,00400 nats; der
Per-Kanal-Offset ein **Ersatz** dafür und kein Produktpfad; der Stock-Cache +0,00843 hinter
`int8`).

## 4. Kosten und Ablauf

| Schritt | Kosten |
|---|---|
| Instrument (compile-time-Schalter, default OFF) + inkrementeller HIP-Build | ~1 h (Build **nach** den laufenden Armen, weil `build-hip-port/strata.exe` im Gebrauch ist) |
| ein 128K-Lauf (`int8`, keine Codec-Änderung) | ~80 min GPU |
| Auswertung auf dem Host (Vereinigung, Churn, Profil, tail-Überdeckung) | Minuten, CPU |
| **Summe** | **~3 h**, davon **ein** GPU-Lauf |

## 5. Was D2 **nicht** ist

- Kein Codec-Experiment.
- Kein Beweis, dass ein auswahlfolgender Ring die 128K-Qualität hebt: D2 liefert die
  **Voraussetzungen** (Kapazität und Brücke). Der Bau käme erst, wenn G9 und G10 beide halten — und
  dann als ein eigener Schritt mit eigenem Vertrag (G11).
- Keine Antwort auf die 5,0-Marke selbst: die ist eine Byte-Frage und wird im Ring-Tier entschieden,
  nicht im Ring-Inhalt.
