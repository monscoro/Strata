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

## 6. Nachtrag 2026-11-13: das Instrument steht — und die Größenangabe war falsch

**Das Format (nachgelesen im Code, nicht geschätzt).** Der Dump ist kein Strom aus Block-Ids, sondern je
Fenster und QSA-Layer ein Datensatz: vier int64 (`pos0`, `layer`, `nq`, `cap`) und danach `nq * cap`
int32. `cap` ist die Zeilenbreite (`qsa_selection_width`, hier **2.051** = 2.048 Top-K-Zellen + 3
Block-Ids), und gültig sind nur die ersten `min(pos0 + Zeile + 1, cap)` Ids einer Zeile — unterhalb der
Kapazität ist die Auswahl die Identität.

**Die „13 MB" waren um Faktor 1000 falsch.** 131.060 Token × 2.051 Zellen × 12 Layer × 4 B ≈ **12,9 GB**
(plus 16 B Kopf je Datensatz, ~25 MB). Die Registrierung oben rechnete mit 2.048 Einträgen und in MB;
beides ist hier korrigiert. Auf C: lagen 369 GB frei, der Lauf ist unkritisch — aber die Zahl im Text
war falsch und bleibt als Korrektur stehen.

**Warum die Kopie auf der Karte sitzt.** Das Fenster ist ein aufgezeichneter Graph; ein Host-Copy darin
bricht die Aufzeichnung, und ein Host-Copy danach sähe nur die **letzte** Layer-Zeile, weil `sel_` ein
von allen Layern wiederverwendeter Puffer ist. Die Ids gehen deshalb je Layer als Graph-Knoten
(`cudaMemcpyAsync`, D2D) in einen Staging-Puffer `[layer][row][cap]`; der Host liest ihn einmal je
Fenster in `window_logprobs()`.

**Die Zähler, und warum sie in `run()` stehen.** Der Staging-Aufruf ist Host-Code: er läuft einmal beim
Aufzeichnen und bei einem Replay **nie** — er kann also keine Fenster zählen. Gezählt wird, was zählt:
`run()` (ein Start = ein Fenster, dessen Kopien laufen) und der Drain. Ein Mismatch wird im Destruktor
gemeldet; ein Fenster ohne Logprobs wird als „gestartet, nie gedrained" gemeldet, nicht still
verschluckt.

**Empirisch geprüft (Smoke-Test: 700 Token, `--prefill 512`, gfx1200):** 36 Staging-Aufrufe, davon 0
vom Zeilen-/Layer-Guard abgewiesen; **176 Fenster gestartet, 175 gedrained** (das letzte Fenster hat
keine Logprobs — genau die erwartete Differenz von 1); 68,9 MB Dump; Auswertung: `skipped 0`. Die
Auswertung wurde gegen dieses Fenster plausibilisiert: im Identitäts-Regime ist die mittlere Breite
350,0 = (1+…+700)/700, und der Churn bei Lag 1 ist 0,29 % ≈ 1 neue Zelle von ~350 — die eine neue Zelle
je Query ist die, die mit jeder Query dazukommt.

**Was hier noch nicht steht:** die Zahlen D2-a … D2-f des 128K-Laufs und die Ausgänge von G9/G10. Sie
folgen als eigener Nachtrag, wenn der Lauf durch ist — und sie werden berichtet, wie sie fallen.

## 7. Nachtrag 2026-11-13, 12:20: die Zahlen — **G9 schließt den Ring**, und G10 ist formal erfüllt und sagt nichts

### Der Lauf

Ein 128K-Lauf, `int8`, mit Dump: 131.060 Positionen, 4.807,8 s GPU, Dump 12.915.176.640 Bytes.
Die `int8`-TSV dieses Laufs ist **byte-identisch** zu der aus D1 (139.789.422 Bytes, alle 131.060
Zeilen gleich) — das Instrument hat kein Logprob verändert. Datensätze: 393.180 = 32.765 gedrainte
Fenster × 12 Layer, `skipped 0`; der Destruktor meldet 32.766 gestartete und 32.765 gedrainte Fenster
(das letzte hat keine Logprobs), 24 Staging-Aufrufe, 0 vom Guard abgewiesen.

### Die sechs Größen (gfx1200, cwe-128k, Text 1)

| # | gemessen | Zahl |
|---|---|---|
| **D2-a** | Breite der Auswahl je Query | **2035,0** Zellen im Mittel; **98,44 %** der Queries liegen am Deckel (2.051); Identität (p+1 ≤ 2.051) nur 1,56 % |
| **D2-b** | Überdeckung des positionalen Rings (R=64 + S=256) | **8,22 %** aller ausgewählten Ids (Sink 0,95 %, Fenster 7,27 %), je Layer 5,9–13,2 %. Der Ring ist 0,24 % der Zellen → **33,7×** angereichert. E1s 8,55 % ist damit bestätigt |
| **D2-c** | Vereinigung \|∪ der letzten N Queries\| | N=1: 2.051 · N=2: 2.932 · **N=4: 4.396** · N=8: 6.554 · N=64: 20.837 · **N=4096: 58.758** (max 117.641 = 90 % des Kontexts); 31 Anker je Layer |
| **D2-d** | Churn (Anteil *neuer* Zellen) | Lag 1: **44,96 %** (je Layer 34,1–56,4) · Lag 8: 68,31 % · Lag 64: 74,73 % |
| **D2-e** | Alter der gelesenen Zellen | Median **8.191** Zellen; 7,3 % jünger als 256; **53,4 %** älter als 4.096; Gipfel bei 16k–32k |
| **D2-f** | tail-Positionen (1 % = 1.311 Positionen) | ausgewählt 100 %, von einer späteren Query gelesen 100 %, von der letzten 12,5 %, eigene Query 99,2 %, **Anteil an allen Auswahlen 1,22 %** (2 %-Set: 2,62 %) |

### Die Entscheidungen, gegen die vorregistrierten Regeln

- **G9 (Kapazität): geschlossen.** Die Regel verlangte eine Vereinigung der letzten 4.096 Queries, die
  *wesentlich* kleiner als 4.096 Zellen ist (Faktor ≥ 4 Reserve). Gemessen: **58.758** Zellen,
  Reserve **0,07×**. Die Grenze ist nicht knapp verfehlt — sie ist bereits bei **vier** Queries
  überschritten (4.396 > 4.096). Ein Pool von 4.096 Zellen kann die Auswahl nicht einholen; das ist
  Arithmetik, kein Kernel.
- **G10 (die Brücke): formal erfüllt, und wertlos.** 100 % ≥ 50 % — aber die Metrik *kann* bei dieser
  Breite nicht scheitern: selbst bei **gleichverteilter** Zufallsauswahl derselben Breite (2.051 von
  p+1 Zellen) liegt die ungünstigste tail-Position (p = 118.347, 12.712 Queries danach) je Layer bei
  100 %, über 12 Layer als ODER bei 100 %. Ich habe die 50-%-Schwelle für eine *kleine* Auswahl
  geschrieben; bei 2.048 von 131.072 Zellen misst „irgendeine spätere Query hat sie gelesen" nichts.
  **Ich zähle G10 deshalb nicht als Beleg**, weder für noch gegen den Hebel.
- **Was D2-f stattdessen sagt:** die Anreicherung. Die schweren Positionen bekommen **1,22×** (1 %-Set)
  bzw. **1,31×** (2 %-Set) ihres Anteils — der Indexer bevorzugt sie kaum. Gegenprobe: der positionale
  Ring bekommt 33,7×. Der Indexer hat eine starke Präferenz für den Ring und fast keine für die
  tail-Positionen.
- **G11 (der Vertrag): unverändert.** Ein auswahlfolgender Pool wäre eine Vermutung, keine Garantie —
  und ist mit G9 ohnehin gegenstandslos.

### Was das heißt

**Der Ring-Hebel ist bei dieser Zielmarke tot, und zwar aus einem Grund, den man vorher nicht sehen
konnte.** Die Annahme war: der Indexer wählt eine kleine, stabile Menge, die ein 4.096-Zellen-Pool
halten kann. Gemessen ist das Gegenteil: er wählt bei 128K **immer sein volles Budget** (2.048 Zellen
aus bis zu 131.072, 1,6 % des Kontexts), **45 % davon sind je Query neu**, und die Vereinigung
überschreitet die Poolgröße innerhalb von vier Queries.

Die Auswahl ist dabei **strukturierter als Zufall, aber der Vorsprung schrumpft mit dem Horizont**:
je Query ist der Churn 44,96 % gegen 96,87 % beim Zufall (ein Faktor 2,2), über 4.096 Queries deckt
die Vereinigung 58.758 Zellen — bei im Mittel ~65.500 verfügbaren Zellen am Anker also **90 %** —,
während der Zufall an jedem Anker den ganzen Kontext abgedeckt hätte. Das ist ein echter, messbarer
Unterschied, aber keiner, der einen Pool rettet: 4.096 Zellen sind bei N=4 erreicht, bei N=64 sind es
20.837. Für einen *kleinen* Pool ist die Auswahl nicht stabil genug, für einen *großen* ist sie kein
Gewinn gegenüber dem, was der Cache ohnehin hält.

**Der 128K-Befund bleibt unberührt.** `q4lm32w` liegt bei 128K weiter +0,00741 nats hinter `int8`;
D2 sagt nichts über den Codec, sondern nur, dass dieser *eine* Weg, ihn zu schließen, nicht existiert.
Und die tail-Zahlen sagen zusätzlich: selbst ein perfekter Folger würde die schweren Zellen nicht
bevorzugt schützen, denn der Indexer tut es auch nicht.

**Was ich an mir selbst korrigiere:** die 50-%-Schwelle von G10 war eine Zahl ohne Baseline. Sie hätte
bei 2.048-Zellen-Auswahlen nie scheitern können. Der Fehler ist nicht, dass ich sie gemessen habe,
sondern dass ich sie vorher nicht gegen den Zufall geprüft habe; `d2-baselines.py` rechnet die
Baselines jetzt nach, und sie stehen im Protokoll neben den Messwerten.

### Was als Nächstes **nicht** kommt

Kein auswahlfolgender Ring, kein Pool-Umbau, kein Kernel dafür. Die Ring-Achse ist mit G9 geschlossen
— für dieses Budget, und mit 0,07× Reserve auch nicht „knapp". Offen bleibt allein die Codec-Achse
(Level-Split, Per-Kanal-Offset, Kalibrierung), und die ist gemessen, nicht vermutet.
