# Zusatz zum D2-Plan: hypothesengenerierende Auswertungen (2026-11-13)

**Der registrierte D2-Plan (`README.md`) bleibt unverändert.** Diese Datei beschreibt **drei
zusätzliche Auswertungen**, die **ausdrücklich hypothesengenerierend** sind und **nicht** Teil der
Entscheidung G2/G3 (und nicht Teil der G9/G10) sind. Sie laufen, **falls die Daten ohnehin
vorliegen** (also im selben 128K-Lauf, ohne einen zusätzlichen Lauf).

**Der Anlass.** Wir wissen jetzt zwei Dinge, die vorher getrennt waren: (a) die **Brücke Block-MSE →
End-to-End-NLL ist linear und für alle drei untersuchten Eingriffe dieselbe** (0,00116 bis 0,00167
nats/dB ✓) — die Blockmessung sagt die End-to-End-Wirkung voraus ✓; und (b) **der gemessene
Offset-Effekt kehrt zwischen der Segment- und der Positionsauswertung um** (+0,00193 gegen −0,00138
nats ✓✓). (a) und (b) zusammen sind unvereinbar, wenn (b) eine Eigenschaft der Daten wäre — (b) ist
also **zuerst eine Frage an die Statistik**, und **erst danach** eine Frage an die Daten ✓.

## Teil 1: Die Vorzeichenumkehr auflösen (unabhängig von D2 ✓)

**Die formale Feststellung, die den Fall aufklärt:** bei **gleich großen Segmenten** (8 Segmente zu
4.095 Positionen ✓) ist das **Segmentmittel der gepaarten Differenzen** per Definition **identisch**
mit dem **Positionsmittel** ✓✓: das Segmentmittel ist (1/8)·Σ_seg (1/4095)·Σ_{i∈seg} δ_i ✓ und das
Positionsmittel ist (1/32746)·Σ_i δ_i ✓ — dieselbe Zahl ✓. **Die gemessene Umkehr ist damit KEINE
Eigenschaft des Arms, sondern ein Werkzeugbefund** ✓✓ (die zwei Ausgaben des Vergleichswerkzeugs
rechnen nicht dasselbe ✓).

**Die drei Schritte der Auflösung (alle auf dem Host, ohne neuen Lauf ✓):**

1. **Was rechnet die Segmenttabelle?** Die „Ganzsegment-KL" des Werkzeugs muss gegen die
   Positionsmittel **derselben** Datei geprüft werden: das Positionsmittel über alle 32.746
   Positionen einer Armpaarung ✓ gegen das Mittel der acht Segmentmittel ✓ — **bei gleicher
   Segmentgröße müssen beide identisch sein** ✓ (und wenn sie es nicht sind: die Definition
   dokumentieren und die **richtige** Größe benennen ✓✓).
2. **Die effektiven Stichproben.** Für jede Armpaarung: die Verteilung der per-Position-Differenzen
   δ_i ✓, ihre **effektive Stichprobenzahl** `n_eff = (Σδ_i)² / Σ δ_i²` ✓✓ (die Zahl der Positionen,
   die den Unterschied wirklich tragen ✓✓ — gegen die 32.746 nominalen ✓) und die **Beiträge der
   einzelnen Positionen** ✓✓ (welche Positionen tragen das Vorzeichen ✓: die tail-Positionen ✓, die
   Zählerpositionen ✓ — die aus dem 128K-Befund bekannten Positionen ✓).
3. **Die Gewichte.** Segment- gegen Positionsgewichtung ✓✓: welche Gewichte die beiden Ausgaben
   implizit verwenden ✓ (die Positionen mit großem |δ_i| ✓ werden im Positionsmittel *gleich*
   gewichtet ✓, im Segmentmittel über ihre Segmente *verdichtet* ✓) — und **welche der beiden der
   Kriterium der Leiter (K1) meint** ✓ (das Kriterium nennt das gepaarte **Segment**mittel ✓ → und
   wenn die Rechnung dahinter ein anderes Mittel ist, ist das zu benennen ✓✓).

**Ergebnisform:** eine Zahlentabelle mit (nominale n, n_eff, das Mittel beider Statistiken, die
Top-Beiträge je Position) für **alle** 21 Armpaarungen des Shear-Arms ✓.

### Der Befund vom 2026-11-13 (Teil 1 ist vorläufig abgeschlossen)

**Die Gewichtung erklärt die Umkehr nicht** (die Segmentgrößen streuen um 0,2 % ✓). **Die beiden
Statistiken verwenden nicht dieselbe Positionsmenge** (die Segmenttabelle 8 × 4095 = 32.760, die
Positionstabelle 32.746 — **14 Positionen** ✓✓), **weil die Segmentzuordnung des Vergleichswerkzeugs
einer festen Zeilenzahl folgt statt den Ids-Grenzen** ✓✓ (die Ids-Segmente sind 4088 … 4096 groß ✓✓).
**⇒ Die Umkehr ist eine Frage von ~14 Positionen** ✓✓✓ (die tail-Beiträge sind von der Größenordnung
10 ✓✓). **⇒ Die Reihenfolge: erst das Werkzeug korrigieren (die Segmentzuordnung den Ids folgen
lassen ✓✓), dann entscheiden** ✓✓✓ — und danach sind Segment- und Positionsmittel bei praktisch
gleichen Segmentgrößen **identisch** ✓✓ und die Frage „welche Statistik gilt" eine Formalität ✓✓ (die
benannt werden muss ✓✓).

**Eine Warnung an mich selbst (im Protokoll festgehalten ✓✓):** die erste eigene Nachrechnung war
**nicht** die KL ✗✗ (die TSV trägt die Top-k-Listen ✓✓, nicht die per-Position-KL ✓✓) — die daraus
gewonnenen Zahlen sind **ungültig** ✓✓ und werden nicht zitiert ✓✓. **Die erste eigene Zahl wird
gegen das Werkzeug geprüft, bevor sie irgendwo steht** ✓✓✓.

## Teil 2: Die Indexer-Stabilität, positionsabhängig

**Die Frage:** fallen **hohe Auswahlvolatilität** und **besonders empfindliche Modellpositionen**
zusammen? ✓✓ (und wenn ja: ist die Volatilität ein *Ursache*- oder ein *Symptom*-Phänomen ✓).

**Dafür braucht es drei Größen je Position (alle aus dem 128K-Lauf, ohne Zusatzlauf ✓):**
- **Die Auswahlwechsel:** wie oft wechselt die Position in der Indexer-Auswahl (über die Abfragen
  hinweg ✓) — die **Wechselrate** je Position ✓✓ (das ist die Volatilität ✓).
- **Die Top-K-Score-Abstände:** wie eindeutig ist die Auswahl (der Abstand zwischen dem letzten
  ausgewählten und dem ersten nicht ausgewählten Kandidaten ✓✓) — je Position ✓ (das ist die
  *Sicherheit* der Auswahl ✓).
- **Das Zugriffsalter:** wie alt ist der letzte Zugriff auf die Position ✓✓ (das ist die *Aktualität*
  ✓).

**Die Kreuzung mit den Qualitätsdaten:** die tail-Positionen des 128K-Befunds (die ~2.400 Positionen,
die den Fehler tragen ✓) gegen diese drei Größen ✓✓ — und die Frage, ob die Positionen mit dem
größten δ_i (also dem größten Codec-Unterschied ✓) zugleich die volatilsten sind ✓✓.

## Teil 3: Die Positions-NLL mit den Indexer-Mustern verbinden

**Nicht die mittlere NLL, sondern die Verknüpfung:** welche Art von Attention-Zugriff bestimmt den
Unterschied **zwischen den Codecs** und **zwischen den Fensterauswahlen** ✓✓✓. Konkret:
- je Position: δ_i (der Codec-Unterschied ✓), der Indexer-Zugriff (ja/nein ✓), die Auswahlwechselrate
  ✓, das Zugriffsalter ✓, den Score-Abstand ✓;
- und daraus: **welche Zugriffsarten den Unterschied tragen** ✓ (die gelesenen? die frischen? die
  eindeutigen? die volatilen? ✓✓✓).

## Was daraus folgen könnte (die Hypothese, die zu prüfen wäre ✓)

**Wenn** die tail-Positionen (die den Fehler tragen ✓) gerade die **flüchtig gelesenen** sind ✓✓ —
d. h. der Indexer liest sie, aber nur kurzzeitig ✓ —, dann ist der **positionale** Ring (S = 256
jüngste + R = 64 Sinks ✓) **am falschen Ort** ✓✓✓: er deckt die *jüngsten* Zellen ab ✓, nicht die
*flüchtigen* ✓. Die daraus folgende **Fenster-Variante** wäre: **das Fenster folgt der Auswahl** ✓✓
(das ist genau das auswahlfolgende Fenster ✓✓ — das Ziel von L5 ✓✓, dessen **Voraussetzung** diese
Messung ist ✓✓) und die daraus folgende **Präzisionsverteilung**: mehr Bits für die Positionen, die
der Indexer *tatsächlich* liest ✓✓ (statt gleichmäßig ✓).

**Beides ist hypothesengenerierend** ✓✓: die Messung sagt, **ob** die Muster so aussehen ✓, nicht
schon, was zu tun ist ✓. Die Entscheidung über eine Fenster-Variante bleibt bei L5 ✓ und die über die
Präzisionsverteilung bei einer eigenen Stufe ✓✓.

## Der Status, klar benannt

**Diese drei Auswertungen sind hypothesengenerierend** ✓✓ und **kein** Teil der G2/G3-Entscheidung
(der Shear-Arm ist entschieden ✓✓) und **kein** Teil der G9/G10 (der D2-Plan ✓✓). Sie kosten **keinen
zusätzlichen Lauf** ✓ (alle Daten fallen aus dem 128K-Lauf an ✓) — der Aufwand ist die **Auswertung
auf dem Host** ✓ (und die Antwort auf Teil 1: **welche Größe rechnet die Segmenttabelle?** ✓✓ — das ist
die erste Frage, und sie ist eine Frage an das Werkzeug ✓, nicht an die Daten ✓).
