# D2 — das Auswahlverhalten des Indexers: trägt ein auswahlfolgender Ring die 128K-Qualität?

**Status: vorregistriert (2026-11-11).** Der Lauf startet, **nachdem** die kalibrierten 128K-Arme
fertig sind (D1: `q4lm32w`+Centering und `q4lm8v`+Fenster+Centering). Dieses Dokument steht vor
dem Lauf; die Gatter stehen fest. Es ist **kein Codec-Experiment**, sondern ein **Zähl-Experiment**:
ein Instrument im Engine, ein 128K-Lauf, eine Auswertung auf dem Host. **Kein Kernel wird gebaut,
bevor die Zahlen vorliegen.**

## 1. Die Frage

Der 128K-Befund (Protokoll 2026-11-11 12:40): der Produktarm (16/16 + Fenster) liegt bei 128K
**+0,00741 nats hinter `int8`** (t = +4,80); K2 und K3 fallen. Der Fehler hängt an **~2.400
Positionen**, die **über den ganzen Text verteilt** sind (Schwerpunkt Blöcke 3–16) und **nicht** in
den letzten 256 Zellen liegen — der **positionale** Ring schützt sie nicht. Die Kampagne hat
gemessen, dass der positionale Ring nur **8,55 %** der Indexer-Auswahlen trifft (E1): **91,45 % der
gelesenen Zellen bleiben 4-Bit.**

**L5 (der auswahlfolgende Pool) ist die einzige Ring-Variante, die diese Lücke schließen könnte.**
Sie ist aber **im Wortlaut nicht umsetzbar**, und der Grund ist ein Zeitproblem, kein
Speicherproblem: der exakte fp16-Zwilling entsteht **beim Schreiben** (`kv_append_win_steps`,
`kv_q4.cu:479`), und seine Zeile ist **positional** (`win_row_of(pos, S, R, B)`); die Auswahl des
Indexers (`qsa_block_topk`, `verify.cpp`) kennt man **erst danach**, wenn die Zelle schon
quantisiert im Bulk liegt. Ein „auswahlfolgender Pool" müsste beim Schreiben **vorhersagen**, welche
Zellen später gelesen werden.

**D2 beantwortet die Frage, die dieser Vorhersage zugrunde liegt: Wie verhält sich die Auswahl
des Indexers über die Zeit?** Aus ihr folgt, ob eine **bounded** Pool-Größe (4.096 Zellen = 0,50
bit/cell bei 128K) die Auswahl überhaupt „einholen" kann.

## 2. Was gemessen wird

Der Instrument-Punkt ist die Auswahl selbst: `qsa_block_topk` schreibt die gewählten **Block-Ids**
pro Query in `sel_` (`verify.cpp`, der Verify-Pfad — derselbe Pfad, den die 128K-Läufe von D1
nehmen). Der Dump (compile-time-Schalter, default OFF, wie `STRATA_Q4_DUMP`/`QSA_LAYER_TRACE`)
schreibt für **jeden Token** zweier Textabschnitte (0–8.191 und 65.535–131.071, zusammen 16.384
Tokens) die gewählten Block-Ids mit ihren Positionen.

| # | Größe | Warum |
|---|---|---|
| **D2-a** | **Größe der Auswahl** je Query (in Zellen) | die Kappe (2.048) ist bekannt; die Zahl ist die Grundlast des Pools |
| **D2-b** | **Überdeckung** des positionalen Rings (R+S) durch die Auswahl | die 8,55 % aus E1, auf dem Merge-Build nachgemessen und **je Layer** |
| **D2-c** | **Wachstum der Vereinigung**: \|∪ der Auswahlen der letzten N Queries\| gegen N | **die entscheidende Zahl**: die Pool-Kapazität ist 4.096 Zellen |
| **D2-d** | **Churn**: der Anteil *neuer* Selektionen je Query | wie stabil ist die Auswahl überhaupt |
| **D2-e** | **Zeitprofil**: wo liegen die ausgewählten Positionen (alt/jung)? | der Indexer liest 91,45 % alte Zellen — wie alt? |
| **D2-f** | **Überdeckung der tail-Positionen**: liegen die ~2.400 schweren Positionen (der Zähler des häufigsten Codeworts) unter den Auswahlen? | **die Brücke zum 128K-Befund**: nur wenn die schweren Positionen *gelesen* werden, kann ein auswahlfolgender Pool sie überhaupt schützen |

## 3. Gatter (vor dem Lauf festgelegt)

- **G9 (die Kapazität):** das Wachstum der Vereinigung (D2-c) muss **langsamer** sein als die
  Pool-Kapazität (4.096 Zellen): die Vereinigung der Auswahlen der letzten 4.096 Queries muss
  **wesentlich kleiner** sein als 4.096 (Faktor ≥ 4 Reserve). Wächst sie schneller, ist L5 als
  Qualitätshebel **geschlossen** — arithmetisch, ohne Kernel.
- **G10 (die Brücke):** die tail-Positionen (die ~2.400 schweren) müssen **überwiegend unter den
  Auswahlen** liegen (Anteil ≥ 50 %). Liegen sie nicht dort, gibt es **keinen** Ring, der sie
  schützt, und die Frage ist geschlossen.
- **G11 (der Vertrag):** unabhängig von G9/G10 ist zu klären, ob der Ring „exact" noch eine
  **Garantie** ist oder eine **Vermutung** (der Fenster-Vertrag: „ein stilles Fehllesen ist schlimmer
  als eine laute Verweigerung", A1–A5). Ein auswahlfolgender Pool ist per Konstruktion eine
  Vermutung — das muss in Setup und Paper **sichtbar** heißen, sonst wäre es der A3-Fehler.

**Kill:** Fällt G9 oder G10, ist L5 geschlossen und die 5,0-Marke bleibt eine reine
**Speicher**frage (der int8-Ring: 7.943 Zellen bei 0,50 bit/cell, ohne Exaktheitszusage). Das ist
auch dann ein Ergebnis, und es ist ein **billiges**.

## 4. Kosten und Ablauf

| Schritt | Kosten |
|---|---|
| Instrument (compile-time-Schalter, default OFF) + inkrementeller HIP-Build | ~1 h (Build **nach** den laufenden Armen, weil `build-hip-port/strata.exe` im Gebrauch ist) |
| ein 128K-Lauf (`int8`, keine Codec-Änderung) | ~80 min GPU |
| Auswertung auf dem Host (Vereinigung, Churn, Profil, tail-Überdeckung) | Minuten, CPU |
| **Summe** | **~3 h**, davon **ein** GPU-Lauf |

## 5. Was D2 **nicht** ist

- Kein Codec-Experiment: die Codec-Achse ist gemessen (Level-Split signifikant +0,00400 nats; der
  Per-Kanal-Offset ein **Ersatz** dafür und kein Produktpfad; der Stock-Cache +0,00843 hinter
  `int8`).
- Kein Beweis, dass ein auswahlfolgender Ring die 128K-Qualität hebt: D2 liefert die
  **Voraussetzungen** (Kapazität und Brücke). Der Bau käme erst, wenn G9 und G10 beide halten.
- Keine Antwort auf die 5,0-Marke selbst: die ist eine Byte-Frage und wird im Ring-Tier
  entschieden, nicht im Ring-Inhalt.
