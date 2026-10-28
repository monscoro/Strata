# Research Assignment: Beyond KL Divergence – A Predictive Quantization Fidelity Framework for Strata

> Eingereichter Arbeitsauftrag (wörtlich, unverändert). Zugehöriger Forschungsansatz:
> `docs/research/BEYOND_KL_QUANT_FIDELITY_PLAN.md`.

## 0. Rolle und Forschungsauftrag

Du agierst als Principal ML Research Scientist mit Spezialisierung auf:

- Informationstheorie und statistische Divergenzmaße
- Quantisierung großer Sprachmodelle
- Long-Context- und Sparse-Attention-Architekturen
- Numerical Stability und Error Propagation
- GPU-Inferenzsysteme
- Experimentelles Benchmarkdesign und statistische Validierung

**Projekt:** [Strata](https://github.com/Niko1221/Strata)
**Modell:** Qwen3.8-Flash-Next
**Forschungskontext:** Asymmetrische KV-Cache-Quantisierung mit K8V4-NL, Windowing und QSA-Indexer.

### Primäres Forschungsziel

Entwickle ein neuartiges, empirisch überprüfbares Bewertungsverfahren, das die Qualität von KV-Cache-Quantisierungen zuverlässiger beurteilt und zukünftige Qualitätsprobleme besser vorhersagen kann als die bisher verwendeten Metriken:

1. Mean KL Divergence
2. KL P99
3. Top-1-Agreement
4. Perplexity

Die genannten Metriken bleiben Referenzen und dürfen nicht durch unbelegte Behauptungen abgewertet werden.

Gesucht wird ein Verfahren, das ihre Schwächen systematisch untersucht, zusätzliche Informationen aus dem Modellzustand nutzt und nachweisbar besser mit relevanten End-to-End-Qualitätsverlusten korreliert.

**Wichtige Anforderung:** Entwickle keine komplizierte neue Metrik allein um ihrer Neuartigkeit willen. Wenn eine geeignete Kombination bestehender Metriken besser funktioniert, ist diese zu bevorzugen.

---

# 1. Problemdefinition

## 1.1 Grenzen der Mean KL Divergence

Die übliche KL-Divergenz lautet:

\[
D_{KL}(P\parallel Q)=\sum_i P_i\log\frac{P_i}{Q_i}
\]

Dabei bezeichnet:

- P die Tokenwahrscheinlichkeiten des Referenzmodells
- Q die Wahrscheinlichkeiten der quantisierten Variante

Die mittlere KL-Divergenz über T Tokenpositionen:

\[
\overline{D}_{KL}=\frac{1}{T}\sum_{t=1}^{T}D_{KL}(P_t\parallel Q_t)
\]

liefert einen aggregierten Wert über alle betrachteten Positionen.

Untersuche insbesondere folgende mögliche Schwächen:

**A. Mittelwertverdünnung**

Seltene, kritische Abweichungen können von vielen unkritischen Positionen überdeckt werden.

**B. Fehlende Entscheidungsrelevanz**

Die KL-Divergenz berücksichtigt nicht direkt, ob die Quantisierung eine für die spätere Generierung wesentliche Tokenentscheidung verändert.

**C. Fehlende zeitliche Struktur**

Einzelne hohe KL-Werte können isoliert auftreten oder sich über zusammenhängende Tokenbereiche konzentrieren. Der Mittelwert unterscheidet diese Fälle nicht.

**D. Fehlende Aufgabenrelevanz**

Abweichungen bei Syntax-, Struktur-, Kontrollfluss- oder Retrieval-Tokens können für eine konkrete Aufgabe größere Auswirkungen besitzen als vergleichbare Abweichungen bei anderen Tokens.

**E. Fehlende Fehlerursachenzuordnung**

Die KL-Divergenz zeigt nicht, ob ein Fehler durch Key-Quantisierung, Value-Quantisierung, Windowing, QSA-Selektion oder numerische Kernel-Effekte verursacht wurde.

Überprüfe jede dieser Hypothesen anhand konkreter Gegenbeispiele und Experimente.

## 1.2 Grenzen von KL P99

KL P99 erfasst das 99. Perzentil der positionsweisen KL-Divergenz.

Das ist für die Analyse seltenerer hoher Abweichungen hilfreich, löst jedoch nicht sämtliche Probleme.

Untersuche:

- Unterschied zwischen isolierten Extremwerten und systematischen Fehlerclustern
- Aussagekraft bei verschiedenen Stichprobengrößen
- Stabilität des P99-Wertes über verschiedene Testsets
- Informationsverlust oberhalb des 99. Perzentils
- Einfluss unterschiedlicher Kontext- und Aufgabentypen
- Empfindlichkeit gegenüber stark unterschiedlichen Referenzentropien
- Zusammenhang mit tatsächlichen Fehlern während autoregressiver Generierung

Beachte, dass P99 bei kleinen Testsets statistisch instabil sein kann.

Das Ziel ist nicht bloß, P99 durch P99.9 zu ersetzen.

**Forschungsfrage:** Welche zusätzlichen Informationen werden benötigt, um die Bedeutung einer Verteilungsabweichung für das Modellverhalten zuverlässiger einzuschätzen?

---

# 2. Zentrale Forschungshypothese

Entwickle und überprüfe folgende Arbeitshypothese:

> Die Schädlichkeit eines Quantisierungsfehlers hängt nicht allein von der Größe der Abweichung in der Tokenwahrscheinlichkeitsverteilung ab, sondern auch von der Entscheidungssensitivität der betroffenen Position, ihrer strukturellen Bedeutung, der zeitlichen Fehlerkorrelation und der Veränderung relevanter Kontextinformationen.

Insbesondere bei Sparse-Attention-Modellen könnte die Änderung einer QSA-Blockselektion eine andere Qualität von Fehler verursachen als eine vergleichbar große numerische Abweichung innerhalb eines unverändert ausgewählten KV-Blocks.

Untersuche diese Hypothese kritisch.

Suche nach Gegenbeispielen.

Falls sich die Hypothese nicht bestätigt, dokumentiere dies ausdrücklich.

---

# 3. Kandidaten für ein erweitertes Bewertungsverfahren

Untersuche die folgenden Metrikfamilien. Keine einzelne gilt vorab als überlegen.

## 3.1 Decision-Boundary Sensitivity

Die KL-Divergenz bewertet die gesamte Wahrscheinlichkeitsverteilung.

Eine zusätzliche Analyse soll untersuchen, ob eine Quantisierung entscheidungsrelevante Änderungen verursacht.

Betrachte dazu den Abstand zwischen den beiden höchsten Referenz-Logits:

\[
m_t=z_{(1),t}^{ref}-z_{(2),t}^{ref}
\]

Prüfe:

- Wie häufig verändert die Quantisierung das Top-1-Token?
- Wie groß war der ursprüngliche Logit-Abstand?
- Führt der Fehler zum Austausch zweier nahezu gleichwertiger Tokens?
- Oder verdrängt ein zuvor deutlich unwahrscheinlicheres Token die Referenzentscheidung?
- Wie verändern sich Top-K-Rankings?
- Welche Änderungen betreffen EOS, Struktur- oder Kontrolltokens?

Entwickle geeignete Sensitivitätsmerkmale.

**Wichtig:** Kleine Referenz-Margins dürfen nicht automatisch als besonders schwerwiegende Fehler bewertet werden.

Eine Entscheidung nahe einer Gleichstandsgrenze kann leicht wechseln, ohne dass daraus ein relevanter Qualitätsverlust entsteht.

Unterscheide deshalb zwischen Entscheidungsinstabilität und funktionaler Schädlichkeit.

## 3.2 Entropieabhängige Quantisierungsabweichung

Untersuche die Entropie der Referenzverteilung:

\[
H(P_t)=-\sum_i P_{t,i}\log P_{t,i}
\]

Vergleiche Quantisierungsfehler bei:

- niedriger Referenzentropie
- mittlerer Referenzentropie
- hoher Referenzentropie

Hypothese:

Eine vergleichbare Verteilungsabweichung kann abhängig von der Vorhersagesicherheit des Referenzmodells unterschiedliche Konsequenzen besitzen.

Untersuche, ob sich daraus eine sinnvolle Konditionierung oder Normalisierung ableiten lässt.

Achte darauf, nicht einfach jede hochkonfidente Referenzentscheidung als korrekt zu behandeln.

## 3.3 Target-Token-Log-Likelihood

Ergänze die Verteilungsvergleiche um die Veränderung der Wahrscheinlichkeit des tatsächlich vorgegebenen Folgetokens.

Definiere:

\[
\Delta NLL_t
=
-\log Q_t(y_t)+\log P_t(y_t)
\]

Dabei ist \(y_t\) das Ground-Truth-Token.

Dieser Wert kann positiv oder negativ sein.

Positive Werte bedeuten, dass das quantisierte Modell dem Ground-Truth-Token eine geringere Wahrscheinlichkeit gibt als die Referenz.

Untersuche:

- Mean Delta NLL
- positive und negative Abweichungen getrennt
- kumulierte Delta NLL über Sequenzen
- Fehler bei besonders seltenen oder aufgabenkritischen Tokens

Prüfe, ob diese Größen zusätzliche prädiktive Informationen gegenüber KL liefern.

## 3.4 Tail-Risk und Conditional Value at Risk

Untersuche Tail-Risk-Metriken als Ergänzung zu P99.

Insbesondere:

\[
CVaR_\alpha(L)
\]

als durchschnittliche Verlusthöhe im schlechtesten Anteil \(1-\alpha\) der Beobachtungen.

Vergleiche:

- Mean KL
- P95 KL
- P99 KL
- P99.9 KL
- CVaR95
- CVaR99
- Maximum KL

Wähle die Tail-Definition und ihre Schätzung explizit.

Prüfe Robustheit, Konfidenzintervalle und Stichprobengröße.

Untersuche insbesondere, ob CVaR kritische Fehler zuverlässiger erkennt als P99.

Berücksichtige, dass auch CVaR zunächst nur eine statistische Zusammenfassung ist und keine funktionale Relevanz beweist.

## 3.5 Temporale Fehlerstruktur

Quantisierungsfehler treten nicht notwendigerweise unabhängig voneinander auf.

Untersuche:

- aufeinanderfolgende Positionen mit erhöhtem KL
- Burst-Länge
- Fehlerautokorrelation
- Fehlercluster an bestimmten Kontextpositionen
- Fehlerakkumulation beim Wachstum des KV-Caches
- Abhängigkeiten von Fensterübergängen
- Abhängigkeiten von Page-Grenzen

Entwickle Metriken, die zwischen isolierten Ausreißern und längeren Fehlerphasen unterscheiden.

Prüfe, ob solche Muster eine spätere Qualitätsverschlechterung zuverlässiger ankündigen.

## 3.6 Alternative Verteilungsmetriken

Vergleiche außerdem:

- Jensen-Shannon-Divergenz
- Total Variation Distance
- Hellinger-Distanz
- symmetrische KL-Varianten
- Ranking-Korrelation
- Top-K-Mass-Shift
- erwartete Veränderung der Token-Loss-Werte

Prüfe, welche dieser Metriken komplementäre Informationen liefern.

Vermeide eine unnötig große Menge stark korrelierter Features.

---

# 4. QSA-aware Fidelity als eigenständige Forschungsdimension

Qwen3.8-Flash-Next verwendet einen QSA-Indexer zur Auswahl relevanter Kontextblöcke.

Hier muss zwischen zwei Fehlerarten unterschieden werden:

**Quantization Distortion:** Die ausgewählten K/V-Werte werden ungenau rekonstruiert.

**Selection Distortion:** Aufgrund einer veränderten Indexer-Repräsentation werden andere Kontextblöcke ausgewählt.

Entwickle ergänzende Metriken zur QSA-Auswahlqualität.

## 4.1 Selection Overlap

Vergleiche die ausgewählten Blöcke:

\[
S_{ref},\quad S_{quant}
\]

Untersuche:

- Jaccard Similarity
- Precision und Recall der ausgewählten Blöcke
- Anzahl veränderter Selektionen
- Positionsverteilung der Abweichungen

## 4.2 Selection Margin

Analysiere die Score-Abstände an der Selektionsgrenze.

Unterscheide:

- instabile Selektionen bei fast identischen Scores
- starke Verschiebungen eindeutig relevanter Blöcke

Erstelle eine Sensitivitätsanalyse.

## 4.3 Attention-Mass Loss

Untersuche, ob der Verlust ausgewählter Kontextblöcke anhand ihrer tatsächlichen Attention-Relevanz besser bewertet werden kann.

Eine mögliche Kandidatenmetrik ist:

\[
L_{mass}=
\sum_{b\in S_{ref}\setminus S_{quant}} A_{ref}(b)
\]

Hier beschreibt \(A_{ref}(b)\) die aggregierte Attention-Masse eines Referenzblocks.

Prüfe sorgfältig:

- Wie wird die Attention-Masse definiert?
- Erfolgt die Aggregation über Tokens, Heads oder Layer?
- Lassen sich Referenz- und Kandidatenpfad sinnvoll vergleichen?
- Ist der Wert aussagekräftiger als reiner Selection Overlap?
- Welche zusätzlichen Messkosten entstehen?
- Kann Attention-Masse tatsächlich funktionale Relevanz vorhersagen?

Ein niedriger Overlap muss nicht automatisch problematisch sein, wenn alternative Blöcke ähnliche Informationen enthalten.

Umgekehrt könnte der Verlust eines einzelnen entscheidenden Blocks schwerwiegend sein.

## 4.4 QSA Counterfactual Analysis

Entwickle ein experimentelles Verfahren, um Ursachen voneinander zu trennen.

Vergleiche mindestens:

| Versuch | Indexer | K/V-Cache |
|---|---|---|
| A | FP32-Referenz | FP16 |
| B | FP32-Referenz | K8V4-NL |
| C | Experimentell quantisiert | FP16 |
| D | Experimentell quantisiert | K8V4-NL |

Ergänze gegebenenfalls gezielte Interventionen, bei denen Selektionen aus dem Referenzlauf in einen quantisierten Lauf übernommen werden.

Prüfe, ob sich dadurch Abweichungen kausal bestimmten Komponenten zuordnen lassen.

Dokumentiere nichtadditive Wechselwirkungen zwischen Indexer- und K/V-Fehlern.

---

# 5. Besonderheiten von K8V4-NL mit Windowing

Untersuche den bereits existierenden Research-Branch.

Analysiere den tatsächlichen Aufbau von:

- asymmetrischer K/V-Quantisierung
- nichtuniformem Value-Codec
- Fensterstrategie
- Präzisionsübergängen
- Cache-Layouts
- Append- und Decode-Pfaden

Verifiziere den Stand gegen den aktuellen Branch, statt die Architektur aus dieser Aufgabenbeschreibung abzuleiten.

### Kritische Untersuchung

Führe eine positionsabhängige Fehleranalyse durch.

Unterteile die KV-Historie mindestens in:

1. aktuelles höherpräzises Fenster
2. Fenstergrenze
3. kürzlich quantisierte ältere Tokens
4. weit zurückliegende Tokens
5. besonders häufig durch QSA selektierte Blöcke

Untersuche, ob Qualitätsabweichungen an bestimmten Übergängen gehäuft auftreten.

Prüfe, ob das Modell nach einer Migration zwischen Präzisionsstufen andere Tokens vorhersagt.

Untersuche, ob ein Window-Migration einen zeitlich konzentrierten Fehler erzeugt, der von Mean KL und P99 nur unzureichend erfasst wird.

Verwechsle Präzisionswechsel nicht mit GPU-Residency-Wechseln.

Beide Ereignistypen müssen separat protokolliert werden.

---

# 6. Entwicklung einer neuen Fidelity-Bewertung

Entwickle aus den bisherigen Erkenntnissen ein quantitatives Modell, das hier vorläufig als:

**Quantization Behavioral Fidelity (QBF)**

bezeichnet wird.

Dieser Name ist ein interner Arbeitsbegriff, keine etablierte wissenschaftliche Kennzahl.

QBF soll nicht zwangsläufig eine einzige Zahl sein.

Prüfe vielmehr, ob ein mehrdimensionales Bewertungssystem sinnvoller ist.

Mögliche unabhängige Dimensionen:

| Dimension | Fragestellung |
|---|---|
| Distribution Fidelity | Wie stark verändern sich die Tokenwahrscheinlichkeiten? |
| Decision Fidelity | Wie stark verändern sich relevante Tokenentscheidungen? |
| Tail Risk | Welche besonders großen und seltenen Fehler treten auf? |
| Temporal Stability | Wie verteilen sich Fehler über Sequenzen? |
| QSA Selection Fidelity | Werden die relevanten Kontextbereiche weiterhin ausgewählt? |
| Functional Fidelity | Wie stark verändern sich tatsächliche Aufgabenresultate? |

### Forschungsauftrag A: Feature-Auswahl

Ermittle empirisch, welche Merkmale zusätzlichen Informationsgehalt besitzen.

Prüfe Korrelationen und Redundanzen.

Wähle möglichst wenige, aussagekräftige Features.

### Forschungsauftrag B: Aggregation

Vergleiche mindestens:

1. Mehrdimensionales Dashboard ohne Gesamtscore
2. Regelbasierte Aggregation mit klar definierten Qualitätsgrenzen
3. Gewichtete Score-Funktion
4. Datengetrieben kalibriertes Vorhersagemodell
5. Pareto-basierte Bewertung von Qualität, Speicher und Geschwindigkeit

Ein datengetriebenes Modell muss auf getrennten Trainings- und Validierungsdatensätzen entwickelt werden.

Vermeide Overfitting auf einzelne Modelle oder Aufgaben.

### Forschungsauftrag C: Risikoprognose

Untersuche, ob QBF eine konkrete Qualitätsverschlechterung vorhersagen kann.

Beispielsweise:

- Verlust einer Needle-in-a-Haystack-Information
- fehlerhafte Codegenerierung
- strukturell ungültige Ausgabe
- falsche Tool-Argumente
- vorzeitiger EOS-Abschluss
- Abbruch einer längeren Reasoning-Kette
- zunehmende Wiederholungen

Die Definition eines Fehlers muss vor der Auswertung festgelegt werden.

Ein Score ist nur dann als Verbesserung zu betrachten, wenn seine Vorhersageleistung gegenüber etablierten Baselines tatsächlich besser ist.

---

# 7. Validierung gegen reale Modellqualität

Der wichtigste Teil dieses Forschungsauftrags ist die externe Validierung.

**Eine neue Metrik darf nicht deshalb als überlegen gelten, weil sie stärkere Unterschiede zwischen Quantisierungen anzeigt.**

Sie muss bessere Aussagen über tatsächliche Qualitätsverluste ermöglichen.

## 7.1 Benchmark-Szenarien

Verwende repräsentative Aufgaben aus:

- Code Completion
- Codegenerierung mit Tests
- Long-Document-Q&A
- Needle-in-a-Haystack
- Multi-Hop-Retrieval
- strukturierter JSON-Ausgabe
- Tool Calling
- Agentenworkflows
- längeren Chats
- mehrsprachigen Texten

Berücksichtige Kontextlängen, die kurze und lange KV-Historien einschließen.

Verwende möglichst auch adversariale Fälle, bei denen kleine numerische Veränderungen große funktionale Auswirkungen haben können.

## 7.2 Zwei getrennte Evaluationsarten

**Teacher-Forced Evaluation**

Beide Modellvarianten erhalten identische Tokenfolgen.

Ziel: Kontrollierter Vergleich numerischer und probabilistischer Abweichungen.

**Free-Running Evaluation**

Die Modelle erzeugen eigene Fortsetzungen.

Ziel: Tatsächliche Auswirkungen von Quantisierungsfehlern auf Modellverhalten und Aufgabenerfüllung messen.

Diese Ergebnisse dürfen nicht gleichgesetzt werden.

Teacher-Forced-Metriken sollen möglichst die Qualität der Free-Running-Ergebnisse vorhersagen.

## 7.3 Prädiktive Überlegenheit messen

Vergleiche Mean KL, KL P99 und alle neuen Kandidaten anhand von:

- Spearman-Rangkorrelation mit gemessenen Qualitätsverlusten
- Fähigkeit zur Erkennung kritischer Regressionen
- Precision-Recall bei seltenen Fehlern
- False-Positive-Rate
- False-Negative-Rate
- Ranking-Stabilität verschiedener Quantisierungsvarianten
- Konfidenzintervallen
- Generalisierung über Aufgaben und Kontextlängen

Nutze für statistische Konfidenzintervalle geeignete Resampling-Einheiten.

Da benachbarte Tokenpositionen korreliert sind, dürfen sie nicht unkritisch als unabhängige Stichproben behandelt werden.

Verwende beispielsweise Bootstrapping auf Aufgaben-, Prompt- oder Konversationsebene.

Vergleiche Kandidaten auf denselben Testfällen und nutze gepaarte statistische Verfahren, soweit angemessen.

### Erfolgskriterium

Ein Kandidat gilt nur dann als nachgewiesene Verbesserung, wenn er auf zuvor nicht verwendeten Testfällen:

- kritische Qualitätsregressionen zuverlässiger erkennt,
- die Rangfolge von Quantisierungsvarianten mindestens ebenso gut oder besser vorhersagt,
- keine unvertretbare Zahl zusätzlicher Fehlalarme verursacht,
- und einen vertretbaren Messaufwand besitzt.

Lege vor den Experimenten konkrete Zielgrößen und Entscheidungsregeln fest.

Berichte die Unsicherheit der gemessenen Verbesserungen.

---

# 8. Computational Efficiency

Eine umfassende Fidelity-Metrik ist nur dann praktisch nutzbar, wenn ihre Berechnungskosten vertretbar bleiben.

Entwickle deshalb zwei Betriebsarten.

## 8.1 Research Mode

Vollständige Analyse mit:

- Full-Vocabulary-Logits
- positionsweisen Divergenzen
- QSA-Selektionen
- Attention-Ausgaben
- Window-Events
- zeitlichen Fehlerprofilen
- Counterfactual Tests

Diese Variante dient der Entwicklung und Validierung.

## 8.2 Fast Evaluation Mode

Entwickle eine vereinfachte Variante mit wenigen besonders aussagekräftigen Metriken.

Untersuche beispielsweise:

- Top-K-Logits mit expliziter Behandlung der restlichen Wahrscheinlichkeitsmasse
- Referenzentropie
- Entscheidungsmargins
- ausgewählte Tail-Risk-Statistiken
- QSA-Selektionsdiagnostik

Wichtig: Eine Berechnung auf Top-K-Logits mit erneuter Normalisierung ist nicht automatisch mit Full-Vocabulary-KL vergleichbar.

Quantifiziere den Approximationsfehler.

Ermittle:

- Messzeit
- Speicherbedarf
- zusätzliches GPU-Profiling
- notwendige Synchronisationen
- Größe der gespeicherten Ergebnisdateien
- Reproduzierbarkeit auf verschiedenen Backends

Ziel ist ein Verfahren, das später automatisiert in Stratas Quantisierungsbenchmarks eingesetzt werden kann.

---

# 9. Implementierung als unabhängiges Research-Werkzeug

Entwickle nach der konzeptionellen Analyse einen funktionierenden Prototyp.

Bevorzugte Sprache: Python für Analyse, Referenzberechnungen und Visualisierung.

GPU-spezifische Modifikationen nur dort vornehmen, wo Daten anders nicht effizient erhoben werden können.

### Anforderungen

- Klar definierte Eingabeformate
- Konsistente Referenz-Kennzeichnung
- Speicherung positionsweiser Kennzahlen
- Reproduzierbare Auswertungen
- Deterministische Testfälle
- Unterstützung unterschiedlicher Kontextlängen
- Vergleich mehrerer Quantisierungsvarianten
- Export in JSON oder CSV
- Automatisierte Testabdeckung

Trenne Datenextraktion, Metrikberechnung, Aggregation und Ergebnisvisualisierung.

Falls bereits kompatible Benchmarkwerkzeuge in Strata vorhanden sind, verwende oder erweitere sie.

Erzeuge keinen zweiten, inkompatiblen Benchmark-Stack ohne sachlichen Grund.

### Testanforderungen

Implementiere Unit-Tests für:

- identische Wahrscheinlichkeitsverteilungen
- nahezu identische Top-1-Tokens
- veränderte Top-1-Entscheidungen
- extrem seltene Token
- numerisch sehr kleine Wahrscheinlichkeiten
- isolierte Ausreißer
- korrelierte Fehlerbursts
- fehlerhafte QSA-Selektionen
- Window-Grenzen
- unterschiedliche Sequenzlängen

Verwende numerisch stabile Berechnungen, insbesondere `log_softmax` für Logit-basierte Divergenzen.

Prüfe Sonderfälle und verhindere NaN-/Infinity-Verfälschungen.

---

# 10. Experimentelles Vergleichsdesign

Untersuche nach Möglichkeit folgende KV-Konfigurationen:

| Konfiguration | Zweck |
|---|---|
| FP16 | Referenz |
| INT8 | Konservativer quantisierter Baseline |
| Q4_0 | Bestehender 4-Bit-Baseline |
| K8V4 | Bestehender asymmetrischer Baseline |
| K8V4-NL | Neuer Quantisierungsansatz |
| K8V4-NL mit Fenster | Windowing-Einfluss |
| QSA-aware Varianten | Zusätzliche Indexer-/Windowing-Experimente |

Verifiziere, welche Varianten tatsächlich implementiert sind.

Erfinde keine Messergebnisse für nicht vorhandene Implementierungen.

Für jede Variante sollten die folgenden Daten vergleichbar vorliegen:

- Mean KL
- KL P99
- Top-1-Agreement
- Delta NLL
- zusätzliche Kandidatenmetriken
- QSA-Selektionsmetriken, soweit relevant
- tatsächlich gemessene Aufgabenqualität
- Speicherverbrauch
- Decode-/Prefill-Geschwindigkeit

### Zentrale Forschungsfrage

Welche Bewertungsmethode würde eine problematische Quantisierung am frühesten und zuverlässigsten als riskant identifizieren?

Untersuche auch den umgekehrten Fall:

Welche Metriken bewerten eine funktional unproblematische Quantisierung unnötig schlecht?

Beide Fehlerarten sind relevant.

---

# 11. Gewünschte wissenschaftliche Ergebnisse

Erstelle einen Research-Report mit folgenden Kapiteln.

### A. Literature Review

Überprüfe geeignete wissenschaftliche Ansätze zu:

- Model Distillation und Teacher-Student Divergence
- Quantization-Aware Evaluation
- Distribution Shift
- Decision-Margin Analysis
- Risk-Sensitive Evaluation
- Tail-Risk Metrics
- Sparse-Attention Stability
- Information-Theoretic Model Comparison

Unterscheide etablierte Verfahren von eigenen Hypothesen.

### B. Failure Analysis of Existing Metrics

Zeige anhand kontrollierter Experimente, welche Fehler Mean KL und KL P99 tatsächlich übersehen oder falsch gewichten.

### C. Candidate Fidelity Metrics

Entwickle mathematisch präzise Definitionen aller neuen Kandidaten.

### D. QSA-Specific Analysis

Dokumentiere die Auswirkungen von QSA-Selektion, Windowing und KV-Quantisierung.

### E. Predictive Validation

Belege, welche Metriken reale Qualitätsverluste besser prognostizieren.

### F. Computational Cost

Vergleiche Berechnungsaufwand und Messkosten.

### G. Recommended Fidelity Framework

Empfiehle eine konkrete, möglichst einfache Lösung.

### H. Limitations

Dokumentiere Fälle, in denen die vorgeschlagene Bewertung versagt oder noch nicht ausreichend validiert ist.

### I. Integration Plan

Beschreibe die Integration in Stratas Research- und Benchmark-Infrastruktur.

---

# 12. Gewünschte Deliverables

Lege die Ergebnisse nach Möglichkeit in folgenden Dateien ab:

```text
docs/research/
    BEYOND_KL_QUANT_FIDELITY.md
    FIDELITY_METRIC_DESIGN.md
    VALIDATION_RESULTS.md

tools/
    quant_fidelity.py
    quant_fidelity_compare.py

tests/
    test_quant_fidelity.py
```

Erstelle ausführbaren Code nur für tatsächlich validierte Teile.

Wenn der vollständige Strata-Branch, GPUs oder Benchmarks nicht verfügbar sind, liefere zunächst:

1. Eine mathematisch fundierte Spezifikation
2. Einen unabhängigen Python-Referenzprototyp
3. Synthetische Gegenbeispiele
4. Ein reproduzierbares Experimentdesign
5. Eine präzise Liste der noch benötigten Messungen

Kennzeichne synthetische Ergebnisse eindeutig.

### Erwünschte Ergebnisübersicht

| Metrik | Erkennt kritische Fehler | Robust gegenüber Ausreißern | QSA-aware | Berechnungskosten | Empirisch validiert |
|---|---|---|---|---|---|
| Mean KL | Zu prüfen | Zu prüfen | Nein | Niedrig | Baseline |
| KL P99 | Zu prüfen | Zu prüfen | Nein | Niedrig | Baseline |
| CVaR | Zu prüfen | Zu prüfen | Nein | Niedrig | Offen |
| Decision-Margin-Metrik | Zu prüfen | Zu prüfen | Nein | Mittel | Offen |
| Temporal-Risk-Metrik | Zu prüfen | Zu prüfen | Nein | Mittel | Offen |
| QSA-Selection-Metrik | Zu prüfen | Zu prüfen | Ja | Mittel bis hoch | Offen |
| QBF-Kandidat | Zu prüfen | Zu prüfen | Optional | Zu messen | Offen |

Keine Spalte darf ohne experimentelle Belege als Erfolg bewertet werden.

---

# 13. Forschungsprinzipien

Halte dich an folgende Regeln:

1. **Keine Metrik ohne messbaren Zusatznutzen.** Neue Größen müssen sich gegen etablierte Baselines beweisen.
2. **Keine unbegründete Gewichtung.** Gewichte eines kombinierten Scores dürfen nicht willkürlich gewählt und anschließend als wissenschaftliche Erkenntnis dargestellt werden.
3. **Keine Datenleckage.** Entwicklung und Validierung müssen getrennt erfolgen.
4. **Keine Mittelwertfixierung.** Seltene, systematische und aufgabenkritische Fehler getrennt analysieren.
5. **Keine unkritische Top-1-Bewertung.** Entscheidungsmargen und tatsächliche Konsequenzen berücksichtigen.
6. **Keine Gleichsetzung von Verteilungs- und Funktionsqualität.** Beide separat messen.
7. **Keine stillschweigenden Architekturannahmen.** Branch und Implementierungen vor Aussagen zum aktuellen Zustand überprüfen.
8. **Keine Veränderung des bestehenden K8V4-NL-Ansatzes während der isolierten Metrikforschung.**
9. **Reproduzierbarkeit sicherstellen.** Testsets, Seeds, Referenzen und Auswertungsdefinitionen dokumentieren.
10. **Negative Ergebnisse ausdrücklich berichten.** Falls Mean KL oder KL P99 in bestimmten Szenarien bereits optimal oder ausreichend sind, darf das Forschungsergebnis genau so lauten.

---

# 14. Forschungsziel und Entscheidungskriterium

Das Projekt ist erfolgreich, wenn ein Verfahren entwickelt und validiert wird, das folgende Frage besser beantwortet als Mean KL und KL P99 allein:

> Wie wahrscheinlich ist es, dass eine bestimmte KV-Cache-Quantisierung bei realen Modellaufgaben eine relevante Qualitätsverschlechterung verursacht, und welche gemessenen Abweichungen erklären dieses Risiko?

Das Verfahren soll insbesondere unterscheiden können zwischen:

**Fall A: Numerische Abweichung ohne praktische Qualitätsrelevanz**

Eine Quantisierung verändert Wahrscheinlichkeiten, aber nicht wesentlich die Funktion des Modells.

**Fall B: Seltene kritische Abweichung**

Eine Quantisierung erzeugt wenige, aber relevante Fehler, beispielsweise bei einem kritischen Kontextblock oder einer entscheidenden Tokenposition.

**Fall C: Kumulativer Qualitätsverlust**

Viele kleine Abweichungen führen über lange Sequenzen zu systematischen Veränderungen.

**Fall D: QSA-Selektionsfehler**

Die Quantisierung verändert die Kontextauswahl und verursacht dadurch einen funktionalen Fehler, der durch reine K/V-Rekonstruktionsmetriken nicht erklärt wird.

**Fall E: Unnötiger Fehlalarm**

Eine Metrik zeigt starke numerische Unterschiede an, obwohl die tatsächliche Aufgabenqualität stabil bleibt.

Bewerte, ob das neue Verfahren diese Fälle zuverlässiger unterscheidet als die bestehenden Metriken.

---

# 15. Arbeitsbeginn

Beginne mit einer kritischen Analyse der bisherigen Metriken.

Erstelle zuerst:

1. Eine Taxonomie möglicher Quantisierungsfehler
2. Mindestens fünf mathematisch oder synthetisch nachvollziehbare Gegenbeispiele, bei denen Mean KL oder P99 keine ausreichende Aussage über funktionale Qualität liefern
3. Eine Liste der vielversprechendsten zusätzlichen Messgrößen
4. Ein experimentelles Design zur Prüfung ihrer Vorhersagequalität
5. Einen minimalen Vorschlag für ein neues Fidelity-Framework

Verwende anschließend den Strata-Research-Branch zur empirischen Überprüfung.

**Die zentrale Zielsetzung lautet nicht „KL durch eine andere Zahl ersetzen“, sondern die Entwicklung eines robusten, prädiktiven und effizient messbaren Qualitätsmodells für KV-Cache-Quantisierung.**

Bevorzuge am Ende die einfachste Lösung, deren Verbesserung gegenüber Mean KL und KL P99 tatsächlich nachgewiesen werden kann.
