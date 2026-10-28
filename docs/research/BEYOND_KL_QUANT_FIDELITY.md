# Fidelity-Bewertung für KV-Quantisierung: der überarbeitete Arbeitsplan

**Status: Arbeitsplan (2026-10-28). Keine eigene Messung; jede Zahl ist zitiert.** Dieses Dokument
ersetzt den Stufenplan F0–F6 aus `BEYOND_KL_QUANT_FIDELITY_PLAN.md` als **Arbeitsplan**. Der Plan
bleibt als Inventar und Herkunftsnachweis stehen (M1–M12, CB1–CB7, Rohdatenlage, Build-Cluster), der
Auftrag bleibt wörtlich. Warum gekürzt wird, steht in
`BEYOND_KL_QUANT_FIDELITY_ASSESSMENT.md`.

**Was dieses Dokument nicht ist:** der F6-Bericht des Auftrags. Die Kapitel A–I (Literature Review,
Validierung, Empfehlung) sind nicht geschrieben, weil G1–G4 nicht gelaufen sind. Ein Bericht ohne
diese Läufe wäre eine Behauptung.

## 0. Was sich gegenüber dem Plan ändert

| Plan | überarbeitet | Grund |
|---|---|---|
| F0–F6, sechs Stufen | G0–G4, fünf Stufen | F0 und F1 bleiben; F2 wird auf **eine** Task reduziert; F4 (QSA) entfällt bis zur Instrumentierung; F5/F6 erst nach G4 |
| zehn Benchmark-Suiten (§7.1) | eine, die es schon gibt (FWE) | Sättigung und Kosten; ohne trennendes Ground Truth ist keine Metrikforschung möglich |
| QBF als Score (§6) | Dashboard mit Regeln | acht Segmente und fünfzehn Arme tragen kein gelerntes Modell; §13.2 verbietet willkürliche Gewichte |
| QSA-Selektion als Dimension (§4) | gestrichen bis zur Instrumentierung | Indexer-Keys kommen aus `x`, nicht aus dem Cache (statische Analyse); C/D-Arme nicht ausführbar |
| Vollvokabular pro Position (§8.1) | geschichtete Stichprobe | ~32 GB je Arm gegen ~1 GB (Rechnung) |
| Prädiktion als Erfolgskriterium | Zuordnung und Kalibrierung zuerst, Prädiktion als G4 | bei n = 8 ist eine Rangkorrelation eine Richtung, kein Nachweis |

## 1. Der gemessene Ist-Zustand

Die zwölf Befunde M1–M12 stehen mit Fundstellen in `BEYOND_KL_QUANT_FIDELITY_PLAN.md` §1.4. Für den
Maßstab zählen fünf:

1. **Mittelwertverdünnung ist real.** Die schlimmsten 1 % der Positionen tragen 85,5 / 81,6 / 70,9 %
   der KL (`int8` / Fenster / plain), die Ränge 1–8 tragen 88–90 %.
2. **Der Tail ist ein Kopf-Phänomen, kein Floor-Phänomen.** 0,7–1,6 % Kopf-Substitutionen tragen
   ~56 % der `int8`-Summe.
3. **PPL invertiert.** 2,4052 / 2,4079 / 2,4029 gegen fp16 2,4149, bei durchweg positiver KL.
4. **Die KL ist ankerabhängig.** Dieselbe LS-Skala: +0,0028 gegen `int8`, +0,0003 gegen `fp16`.
5. **Die Auflösung ist 0,0020 nats.** Gepaarte Streuung 0,22 bei 32.746 Positionen; unter 0,001 zu
   kommen bräuchte ~131.000 Positionen, und der 128K-Satz existiert nicht.

Dazu **M13** aus der Decode-Sonde (2026-10-28): volles Vokabular, 32 Prompts, `int8` 0,00055, der
Fensterkandidat 0,00225. Die Parität ist **nicht bestätigt, nur nicht ausgeschlossen** (+0,0017,
Intervall berührt die 0, 6/8 Segmente schlechter). `S`-Achse und Sinks liegen unter der Auflösung
(0,0027 bzw. 0,0016). Ein Nebenbefund: der 16/8-Bulk ist beim Decode nominell besser als 16/16 —
die Sonde reproduziert die Prefill-Rangfolge nicht und kann den 16/16-Punkt weder stützen noch
kippen.

## 2. Drei Sätze, die den Maßstab tragen

1. **Anker und Auflösung werden immer mitgeliefert.** Jede KL-Zahl nennt ihre Referenz (`fp16` oder
   `int8`) und die einseitige Grenze bei echter Parität. Eine Zahl ohne beides ist nicht
   vergleichbar — das ist der gemessene Fehler M11, nicht eine Stilfrage.
2. **Drei Merkmale kommen dazu, mehr nicht:** ΔNLL auf dem Ziel-Token, Flip bei Margin ≥ 1 nat,
   Region (Sink / Horizontkante / recent / Ring / outside). Dazu CVaR99 und eine getrennte
   Floor-Ereignisrate als ehrliche Tail-Zusammenfassung.
3. **Es gibt genau ein Ground Truth:** das FWE-Ereignis im vorhandenen Korpus. Ohne es ist jede
   Merkmalsfrage ungeankert.

Mean KL und P99 bleiben die Baselines. Sie werden nicht ersetzt, solange kein Merkmal die Prüfregel
in §3 besteht.

## 3. Die Prüfregel (vor G1 festgelegt)

Ein Merkmal wird nur aufgenommen, wenn es **alle vier** Kalibrierungspaare besteht und danach die
Redundanzprüfung:

| Paar | Wahrheit | Prüfung |
|---|---|---|
| `q4lm32w` vs `int8` | Parität (+0,00002 / +0,00034; `int8` führt 4/8 bzw. 5/8) | das Merkmal darf keinen Unterschied melden |
| `s320r0` vs 256/64 | Verlust (+0,00394 [+0,00097, +0,00691], der Arm besser in 0/8) | das Merkmal muss ihn zeigen, richtige Richtung |
| 16/8 vs 8/16 | kein Unterschied (−0,00030, Intervall um 0) | das Merkmal darf keinen erfinden |
| Leiter 8/8 0,04127 > 8/16 0,03212 ≈ 16/8 0,03182 > 16/16 0,02512 > 16/16+Fenster 0,01689 ≈ `int8` 0,01688 | Ordnung | das Merkmal muss sie erhalten |

Danach: |Spearman ρ| < 0,9 gegen Mean KL, oder ein nachweisbarer Zusatzbeitrag in der
Partialbetrachtung. Statistik: gepaart, **Cluster = Segment** (n = 8, t mit df = 7), niemals
Positionen. Ein Merkmal, das nur auf Positionsebene signifikant ist, ist nicht aufgenommen.

Das ist kein Validierungsnachweis — es fehlt das funktionale Ground Truth —, aber es filtert
Rauschen und umskalierte KL heraus, bevor GPU-Zeit investiert wird.

## 4. Der Arbeitsplan G0–G4

| Stufe | Frage | Arbeit | Kill | Kosten |
|---|---|---|---|---|
| **G0** | Sind alle Zahlen vergleichbar? | `compare` gibt Anker und einseitige Grenze mit aus; jede Ergebnis-README nennt beides; die vier Kalibrierungspaare werden als Fixture festgeschrieben | — | Stunden, kein GPU |
| **G1** | Trägt eines der drei Merkmale? | Rescoring der vorhandenen 20 Läufe (S0 4, S3 9, Ring 3, Tag 4): ΔNLL, Margin-Flip, Region, CVaR99, Floor-Rate; Prüfregel §3; Redundanzmatrix | kein Merkmal besteht §3 → **Kurzbericht: Mean KL + P99 reichen**, Dashboard entfällt | Stunden, kein GPU; braucht die externen TSVs |
| **G2** | Ist die trunkierte KL überhaupt die KL? | Engine-Dump des vollen Vokabulars an ~1.000 geschichteten Positionen je Arm (Top-KL + zufällig + Flips); Rangfolge gegen top-64/256 prüfen; `int8`-SHA-Gate unberührt, eigener vorregistrierter Build | Rangfolge dreht → teacher-forced-top-64 ist nicht reparierbar, alle künftigen Aussagen nur relativ | ein Build + ein Lauf je Arm |
| **G3** | Unterscheiden sich die Arme funktional? | FWE: Prompt = erste `answer_at` Token, greedy, Gold-Wörter zählen; 8 Segmente × Arme; Sättigungsregel vorab | Sättigung nach einer Verschärfungsrunde → G4 verschoben, nicht ergebnislos gestartet | Minuten GPU je Arm |
| **G4** | Sagt ein Merkmal den Ereignisverlust voraus? | Merkmalswerte aus G1 gegen Ereignisraten aus G3; Rangfolge und Fehlalarme; Konfidenz auf Segmentebene | kein Merkmal schlägt Mean KL → Empfehlung: Baselines bleiben, Dashboard ohne Prädiktionsanspruch | offline |

**Reihenfolge ist bindend:** G1 vor G2 vor G3 vor G4. G1 ist gratis und kann sofort laufen; G2 ist
die Voraussetzung dafür, dass G1 nicht durch die Linse M1/CB5 gemessen wird; G3 entscheidet, ob G4
überhaupt einen Gegenstand hat.

**Nicht in diesem Plan:** QBF-Score, QSA-Selektionsmetriken, Zehn-Suiten-Matrix, Vollvokabular pro
Position, K8V4-NL, gelernte Aggregation. Jedes davon ist eine eigene, später vorregistrierte Arbeit.

## 5. Was aus dem Auftrag wörtlich gilt

Diese Teile werden nicht gekürzt, weil sie richtig sind und weil wir sie schon so machen:

- **§13 Forschungsprinzipien**, vollständig — besonders Nr. 1 (kein Zusatznutzen, keine Metrik),
  Nr. 3 (Entwicklung und Validierung getrennt), Nr. 6 (Verteilungs- und Funktionsqualität nicht
  gleichsetzen) und Nr. 10 (negative Ergebnisse berichten).
- **§7.2 zwei Evaluationsarten, strikt getrennt.** Teacher-forced liefert die Merkmale,
  Free-Running das Ground Truth. Sie werden nicht gleichgesetzt.
- **§7.3 Resampling-Einheit.** Aufgabe, Prompt oder Konversation — niemals Position. Bei uns ist die
  Einheit das Segment (n = 8), und die vier geschachtelten Prompt-Längen der Decode-Sonde sind
  **keine** 32 unabhängigen Einheiten.
- **§7.3 Erfolgskriterium.** Ein Kandidat gewinnt nicht durch größere Unterschiede, sondern durch
  bessere Vorhersage bei vertretbarem Aufwand.
- **§14 Fall E, unnötiger Fehlalarm.** Bei uns ist das kein Gedankenspiel: der Kandidat ist von
  `int8` nicht zu trennen, und jede Metrik, die dort einen Unterschied meldet, produziert einen
  Fehlalarm. Die Prüfregel §3 ist genau dafür da.

## 6. Was der Maßstab nicht kann

Die Auflösung. 0,0020 nats bei 32.746 Positionen ist eine Eigenschaft des Korpus, nicht der Metrik.
Eine bessere Metrik auf denselben Positionen hat dieselbe Grenze für „ist der Arm schlechter"; sie
kann sagen **wo** und **wie**, nicht **ob ein 0,001-Effekt real ist**.

Daraus folgt die Umdeutung, die dieser Plan vornimmt: der Wert liegt in **Zuordnung** (welche Region,
welche Distanz, welcher Übergang) und **Kalibrierung** (sagt das Merkmal Parität, wenn die Aufgabe
Parität sagt), nicht in Empfindlichkeit. Wer auflösbare 0,001-Effekte erwartet, braucht einen anderen
Korpus, nicht eine andere Metrik.

## 7. Offene Messungen

1. **G1-Rescoring** der 20 Läufe; Voraussetzung: `Strata-data\bench\{2026-10-25-s0,2026-10-27-s3,
   2026-10-28-ring,2026-10-25-tag-quality}` liegen vor (SHAs in den READMEs).
2. **G2-Engine-Dump** an geschichteten Positionen — neuer Build, vorregistriert, `int8`-Pfad
   byte-identisch.
3. **G3-FWE-Ereignis** — braucht nur ein Skript über der bestehenden Engine, keinen neuen Build.
4. **Decode bei langem Kontext** — die Sonde vom 2026-10-28 misst bei ≤ 1.000 Token, wo der Ring
   nicht verdrängt. Ein Lauf mit 4K-Prompts und einem Decode-Schritt wäre der nächste Arm; er
   gehört zur Qualitätsleiter, nicht zu diesem Plan, weil er die Ring-Frage misst und nicht die
   Metrik-Frage.

## 8. Grenzen

- **Macht:** ein Modell, eine Karte, acht Segmente, eine Task-Familie. Aussagen sind Richtungen mit
  Konfidenzband. Overfitting auf cwe ist die Hauptgefahr; die Prüfregel §3 und die Trennung
  G1/G4 sind der Schutz, kein Beweis.
- **Sättigung:** die Nadel ist 18/18. Ob FWE trennt, ist offen; deshalb die Sättigungsregel vor G3.
- **Rohdaten liegen außerhalb des Repos.** Nur die dokumentierten SHAs schützen die
  Reproduzierbarkeit.
- **Die Engine-Deckel bei 256** begrenzen jedes Merkmal, das die Tail-Masse braucht; die
  Bandbreite `r·ln(V−k)` wird mitberichtet, nicht verschwiegen.
- **Kein Kapitel A–I.** Der Auftrag verlangt einen Research-Report; dieses Dokument ist der
  Arbeitsplan dorthin, nicht der Report.

## 9. Verhältnis zu den anderen Dokumenten

| Dokument | Rolle |
|---|---|
| `BEYOND_KL_QUANT_FIDELITY_ASSIGNMENT.md` | der Auftrag, wörtlich und unverändert |
| `BEYOND_KL_QUANT_FIDELITY_PLAN.md` | Inventar und Herkunftsnachweis (M1–M12, CB1–CB7, Rohdaten, Cluster) |
| `BEYOND_KL_QUANT_FIDELITY_ASSESSMENT.md` | warum der Plan nicht eins zu eins der Arbeitsplan ist |
| `BEYOND_KL_QUANT_FIDELITY.md` | dieses Dokument: was wir tun, in welcher Reihenfolge, mit welchen Kills |

Die Qualitätsleiter S0–S5 in `docs/paper/Ansatz_Improvements.md` läuft unverändert weiter. G1–G4
blockieren sie nicht: G1 und G4 sind offline, G2 und G3 sind kurze, eigene Läufe.
