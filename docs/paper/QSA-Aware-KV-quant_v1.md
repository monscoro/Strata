Arbeitsprompt: Strata – QSA-aware K8V4-NL, Windowing & KV Page Layout Research
Rolle und Ziel
Du arbeitest als Senior GPU Inference Engineer und ML Systems Researcher am Projekt:

https://github.com/Niko1221/Strata

Im aktuellen Research-Branch existiert bereits ein experimenteller Ansatz für eine effiziente asymmetrische KV-Cache-Quantisierung auf Basis von K8V4-NL mit Fensterstrategie.

Dieser Ansatz soll nicht ersetzt oder von Grund auf neu entworfen werden.

Ziel ist es, die bisher unzureichend berücksichtigte Integration mit folgenden Komponenten systematisch zu erforschen:
1. Qwen3.8-Flash-Next QSA-Indexer
2. QSA-Top-K-Selektion und deren Genauigkeitsanforderungen
3. Logisches und physisches KV-Page-Layout
4. Windowing und unterschiedliche Präzisionsstufen
5. KV-Streaming zwischen GPU und Host-RAM
6. Plattformübergreifende GPU-Ausführung
Übergeordnetes Ziel:

Entwickle eine datenbasierte Architekturstrategie, mit der K8V4-NL einschließlich Fensterverwaltung zu einem effizienten, QSA-aware KV-Cache-System weiterentwickelt werden kann.

Optimiere nicht isoliert die Kompressionsrate. Untersuche den Zusammenhang zwischen Speicherbedarf, Speicherzugriffen, QSA-Selektionsqualität, Decode-Latenz und GPU-Portabilität.
1. Bestehenden Research-Branch vollständig verstehen
Analysiere zunächst den ausgecheckten Branch und dokumentiere den tatsächlichen Implementierungsstand.

Ermittle:
- Branch-Name, Commit und Abweichungen zum Upstream
- Vorhandene Implementierung von K8V4-NL
- Quantisierungsverfahren und verwendete Codebooks
- Funktion und Größe des Fensters
- Quantisierung innerhalb und außerhalb des Fensters
- Vorhandene Append-, Gather- und Decode-Pfade
- Integration mit KV-Streaming
- Bestehende Benchmarks und Qualitätstests
- Fehlende Integrationen und mögliche Inkonsistenzen
Behandle die vorhandene Implementierung als Forschungsgrundlage.

Wichtig: Keine bestehenden Änderungen verwerfen, überschreiben oder stillschweigend durch andere Ansätze ersetzen.

Prüfe insbesondere die relevanten Strata-Komponenten:
include/strata/kernels/qsa.hpp
include/strata/kernels/kv_q8.hpp
include/strata/kernels/kv_q4.hpp
include/strata/kernels/kv_stream.hpp
include/strata/kernels/qsa_decode_attn.hpp

src/kernels/cuda/qsa.cu
src/kernels/cuda/qsa_select.cu
src/kernels/cuda/qsa_decode_attn.cu
src/kernels/cuda/kv_q8.cu
src/kernels/cuda/kv_q4.cu
src/kernels/cuda/kv_stream.cu

src/core/layer.cpp
src/core/conversation_snapshot.cpp

Berücksichtige außerdem die relevanten Implementierungen unter sycl/ und metal/.

Suche nach weiteren Dateien, die der Research-Branch ergänzt oder verändert hat.

Trenne in der Dokumentation eindeutig zwischen vorhandener Funktionalität, unvollständiger Integration und neuen Forschungshypothesen.
2. QSA-Indexer als eigenständiges Optimierungsproblem untersuchen
Qwen3.8-Flash-Next verwendet eine Sparse-Attention-Architektur, bei der der QSA-Indexer relevante KV-Bereiche auswählt.

Die bisherige Strata-Implementierung verwendet nach aktuellem Referenzstand:
QSA-Layer:       12
KV-Heads:         2
Head-Dimension: 256

Indexer-Dim:    128
Indexer-Block:    4 Tokens
Selection:      bis zu 2048 Tokens
Indexer-Cache:  FP32

Verifiziere diese Werte gegen den aktuellen Branch.
Forschungsfrage A: Welche Informationen müssen exakt bleiben?
Untersuche, welche numerischen Änderungen im Indexer zulässig wären, ohne die Auswahl relevanter KV-Blöcke wesentlich zu verändern.

Insbesondere:
- Sensitivität der Indexer-Scores gegenüber Quantisierungsfehlern
- Verteilung der Score-Abstände an der Top-K-Grenze
- Anzahl und Bedeutung veränderter Blockselektionen
- Abhängigkeit von Kontextlänge und Kontextinhalt
- Auswirkungen auf Long-Context-Retrieval
- Verhalten bei unvollständigen Indexer-Blöcken
- Stabilität über mehrere Decode-Schritte
Unterscheide ausdrücklich zwischen:

Indexer-Rekonstruktionsfehler: Wie stark ändern sich gespeicherte Vektoren und berechnete Scores?

Selektionsfehler: Welche anderen KV-Blöcke werden dadurch ausgewählt?

End-to-End-Fehler: Wie verändert die Selektion die tatsächliche Modellantwort?

Ein niedriger Rekonstruktionsfehler ist allein kein Nachweis für eine stabile QSA-Auswahl.
Forschungsfrage B: Muss der Indexer überhaupt quantisiert werden?
Vergleiche mindestens drei Strategien:

Variante A – FP32-Indexer als Referenz

Der Indexer bleibt unverändert. Ausschließlich der eigentliche K/V-Cache wird optimiert.

Variante B – Komprimierter Indexer

Untersuche FP16, INT8 mit gruppenweiser Skalierung und gegebenenfalls nichtuniforme Quantisierung.

Variante C – Zweistufige Indexer-Selektion

Nutze eine komprimierte Repräsentation für die Kandidatensuche und eine höherpräzise Repräsentation für die endgültige Auswahl.

Prüfe dabei insbesondere, wie viele zusätzliche Kandidaten erforderlich wären, um relevante Blöcke nicht vorzeitig auszuschließen.

Bewerte auch mögliche konservative Fehlergrenzen für Scores.

Eine bloße größere Kandidatenmenge garantiert keine korrekte Auswahl.

Wichtig: FP32 bleibt zunächst der verbindliche Referenzpfad. Die Indexer-Quantisierung darf nicht mit den Verbesserungen des K8V4-NL-Codecs vermischt werden.
3. QSA-aware Windowing untersuchen
Der vorhandene K8V4-NL-Prototyp besitzt bereits eine Fensterstrategie.

Untersuche zunächst, welche Funktion dieses Fenster tatsächlich erfüllt.

Klärungsfragen:
- Welche Tokens liegen im Fenster?
- Welche Präzision besitzen sie?
- Wie wird ein Token beim Verlassen des Fensters behandelt?
- Erfolgt eine erneute Quantisierung?
- Bleiben Keys oder Values in höherer Präzision?
- Wie werden Übergänge und Cache-Reuse behandelt?
Forschungsfrage: Kann das Fenster mit QSA verbunden werden?
Untersuche, ob die aktuelle Fensterstrategie die tatsächlichen Zugriffsmuster des QSA-Indexers optimal berücksichtigt.

Vergleiche:

A. Rein zeitbasiertes Fenster

Die letzten N Tokens erhalten eine höhere Präzision.

B. QSA-aware Fenster

Berücksichtige zusätzlich, welche KV-Blöcke regelmäßig von QSA selektiert werden.

C. Hybridstrategie

Eine feste Recent-Token-Region wird durch einen begrenzten Pool häufig selektierter älterer Blöcke ergänzt.

Dabei sind zwei Konzepte strikt zu unterscheiden:
1. Höhere Quantisierungspräzision
2. GPU-Residency
Ein häufig selektierter Block muss nicht zwangsläufig höherpräzise gespeichert werden, um von einer bevorzugten GPU-Residency zu profitieren.

Untersuche daher getrennt:
- Precision-aware Windowing
- Residency-aware Windowing
- Kombination beider Mechanismen
Prüfe, ob QSA-Zugriffsmuster für die Fensterverwaltung tatsächlich prädiktiv sind.

Berücksichtige die zusätzlichen Kosten einer Migration zwischen Quantisierungsformaten.

Ziel: Herausfinden, ob ein QSA-aware Fenster gegenüber dem bestehenden Verfahren einen messbaren Vorteil bringt.

Nicht automatisch eine komplexere Fensterstrategie implementieren.
4. Das KV-Page-Layout grundlegend analysieren
Strata verwendet nach bisherigem Stand eine logische Page-Größe von vier Tokens.

Diese entspricht der QSA-Indexer-Blockgröße.

Prüfe zunächst, ob diese Kopplung im aktuellen Branch unverändert besteht.
4.1 Speicheranalyse
Erstelle eine exakte Speicherbilanz für:
- INT8
- Q4_0
- K8V4
- K8V4-NL
- K8V4-NL mit Fenster
- mögliche weitere untersuchte Varianten
Berechne jeweils:
Bytes pro K-Head
Bytes pro V-Head

Bytes pro Token
Bytes pro logischer Page

Skalierungsmetadaten
Codebook-Metadaten

Indexer-Speicher
Page-Table-Speicher
Residency-Metadaten

Alignment
Padding
Interne Fragmentierung

Host-RAM
GPU-VRAM
Transfer- und Staging-Buffer

Unterscheide dabei zwischen:
1. Nutzdaten
2. Quantisierungsmetadaten
3. Verwaltungsmetadaten
4. Alignment und Padding
5. Physisch alloziertem Speicher
Berücksichtige bei den Rechnungen die tatsächlichen Strides, die der aktuelle Code verwendet.
4.2 Konkrete Layout-Hypothese überprüfen
Für das bisherige K8V4-Format ergibt sich bei vier Tokens pro Page:
K8V4 KV-Daten:       3264 Bytes
FP32 Indexer-Block:   512 Bytes
--------------------------------
Zusammen:            3776 Bytes

Diese Rechnung gilt pro QSA-Layer und unter der Annahme eines vollständigen Indexer-Blocks mit 128 FP32-Werten.

Die Daten würden theoretisch in eine 4096-Byte-Speicherseite passen.

Allerdings wären dann 320 Bytes ungenutzt.

Über zwölf QSA-Layer und 262.144 Tokens entspräche das bei einer tatsächlich auf 4096 Bytes aufgerundeten Allokation ungefähr 240 MiB zusätzlichem Speicherverbrauch.

Untersuche ausdrücklich, ob Strata überhaupt eine solche physische Rundung durchführt.

Vermeide es, logische KV-Pages mit Betriebssystem-Speicherseiten oder GPU-Allokationsgranularitäten gleichzusetzen.
4.3 Layout-Alternativen vergleichen
Untersuche mindestens:

Layout A – bestehende getrennte Speicherbereiche

Indexer, K-Daten, V-Daten und Quantisierungsmetadaten bleiben getrennt.

Layout B – kompakte Page-Struktur

Alle benötigten Daten einer KV-Page werden möglichst kompakt zusammengefasst.

Layout C – Hybrid-Layout

Der Indexer bleibt als zusammenhängendes Array gespeichert, während K/V-Daten und deren Metadaten Page-lokal organisiert werden.

Layout D – Page-übergreifendes Packing

Mehrere logische Pages werden ohne unnötiges Padding in größeren Speicherblöcken angeordnet.

Ermittle für jede Variante:
- tatsächliche Bytes pro Page
- erreichbare Alignment-Grenzen
- Speicherzugriffsmuster
- Coalescing-Effizienz
- Anzahl der Speichertransaktionen
- Pointer- und Page-Table-Zugriffe
- Dequantisierungsaufwand
- Streaming-Transferkosten
- Auswirkungen auf die QSA-Indexer-Performance
Besondere Aufmerksamkeit für Layout C:

Der QSA-Indexer durchsucht wesentlich mehr Kontextinformationen als der eigentliche Attention-Kernel.

Daher könnte die physische Zusammenlegung von Indexer- und KV-Daten trotz kompakter Page-Struktur nachteilig sein.

Ein kontinuierlich gespeicherter Indexer könnte bessere Speicherzugriffsmuster besitzen als ein Page-interleaved Indexer.

Diese Hypothese muss experimentell überprüft werden.
5. Logische Page-Größe von physischem Transferlayout entkoppeln
Untersuche, ob sich eine zusätzliche Layout-Ebene lohnt.

Vorgeschlagenes Modell:
QSA Indexer
     |
     v
Logische Pages
4 Tokens pro Block
     |
     v
Page Mapping
     |
     v
Physische Speicherblöcke
     |
     +-----------------+
     |                 |
     v                 v
GPU-residente      Host-residente
KV-Daten           KV-Daten

Die logische Page-Größe bleibt zunächst bei vier Tokens.

Physische Speicherblöcke und Transfers dürfen jedoch größere Einheiten verwenden, sofern dadurch keine unnötigen KV-Daten übertragen werden.

Prüfe:
- Gruppierung benachbarter Page-Misses
- Vektorisierte Blocktransfers
- Alignment auf 64, 128 und 256 Bytes
- Page-Packing ohne feste 4-KiB-Slots
- Zugriff auf gepackte 3-/4-/8-Bit-Daten
- gemeinsame Verarbeitung benachbarter Selektionen
- Einfluss auf Cachelines und GPU-Memory-Coalescing
Vermeide größere logische Pages, wenn dadurch bei verstreuter QSA-Selektion unnötig viele Tokens geladen werden.
Wichtige Forschungsfrage
Kann ein verbessertes physisches Layout den effektiven Speicherverkehr reduzieren, ohne die logische QSA-Granularität zu verändern?

Untersuche dafür reale QSA-Auswahlmuster und nicht ausschließlich zufällig verteilte synthetische Zugriffe.
6. K8V4-NL und QSA-Indexer gemeinsam betrachten
Untersuche jetzt den Zusammenhang zwischen dem bestehenden K8V4-NL-Codec und der QSA-Selektion.

Insbesondere:

Key-Quantisierung: Wie wirken sich Fehler in quantisierten Keys auf die Attention-Scores aus?

Value-Quantisierung: Welche Qualitätsverluste entstehen durch die nichtuniform quantisierten Values?

Indexer-Quantisierung: Welche zusätzlichen Fehler entstehen durch veränderte Blockselektionen?

Die Effekte dürfen nicht einfach addiert werden.

Es ist möglich, dass kleine Fehler in mehreren Komponenten gemeinsam einen deutlich größeren Qualitätsverlust verursachen.

Führe deshalb kontrollierte Ablationstests durch:
Test	K/V-Format	Indexer	Fenster
Referenz	INT8	FP32	Standard
A	K8V4	FP32	Standard
B	K8V4-NL	FP32	Bestehend
C	K8V4-NL	FP32	QSA-aware
D	K8V4-NL	Komprimiert	Bestehend
E	K8V4-NL	Komprimiert	QSA-aware


Die letzten Varianten sind Forschungsoptionen und dürfen nur aktiviert werden, wenn die jeweiligen Komponenten unabhängig validiert wurden.

Bewerte anschließend mögliche Wechselwirkungen.
7. Portabilität und GPU-Kernel berücksichtigen
Strata verfolgt einen plattformübergreifenden Ansatz.

Bevorzuge deshalb hardwareunabhängige, gepackte Integer-Speicherformate.

Verlasse dich nicht auf:
- native FP8-Instruktionen
- NVFP4-Hardware
- NVIDIA-spezifische Quantisierungsbibliotheken
- spezielle Tensor-Core-Datentypen
Backend-spezifische Optimierungen sind ausdrücklich erlaubt.

Unterscheide:

Portables Speicherformat: Identische Bitrepräsentation und klar definierte Quantisierungsregeln.

Optimierter Backend-Kernel: Unterschiedliche Implementierungen für CUDA, HIP, SYCL oder Metal.

Untersuche insbesondere, ob zusätzliche Codebook-Lookups im K8V4-NL-Pfad den geringeren Speicherverkehr teilweise kompensieren oder sogar übertreffen.

Erstelle eine Support-Matrix für die tatsächlich vorhandenen Backends.

Verlange keine vollständige Implementierung auf allen GPUs im ersten Forschungsdurchlauf.

Eine CPU-Referenzimplementierung mit nachvollziehbaren Quantisierungs- und Dekodierungsregeln soll jedoch als Grundlage für spätere Portierungen dienen.
8. Benchmarks und Qualitätsmetriken
Führe Messungen auf identischen Eingabedaten und mit identischen Modelleinstellungen durch.
Speicher
Erfasse:
- Tatsächlich allozierte GPU-Bytes
- Tatsächlich allozierte Host-Bytes
- KV-Nutzdaten
- Indexer-Daten
- Page-Tables und Residency-Metadaten
- Quantisierungs-Overhead
- Padding und Alignment
- Temporäre Buffer
- Effektiven Datenverkehr je Attention-Schritt
Performance
Messe getrennt:
KV Append
KV Quantization
Indexer Scoring
QSA Top-K Selection
Page Resolution
KV Streaming
Dequantization
Attention Compute

Zusätzlich:
- Prefill tokens/s
- Decode tokens/s
- Time to First Token
- Latenz pro Attention-Layer
- PCIe-Transfervolumen
- GPU-Speicherbandbreite
- Cache-Hit-Rate
- Window-Migration-Kosten
Untersuche mindestens kurze, mittlere und lange Kontexte bis zur nativen Modellkontextlänge.
Qualität
Verwende:
- Perplexity
- Logit-KL-Divergenz
- Top-1-Agreement
- Attention-Output-Fehler
- Indexer-Score-Abweichung
- Top-K-Overlap
- Selection-Flip-Rate
- Needle-in-a-Haystack
- Long-Document-Reasoning
- Code- und Agentenaufgaben
Bei der QSA-Selektion sollen zusätzlich die Score-Abstände an der Selektionsgrenze dokumentiert werden.

Untersuche, ob veränderte Selektionen hauptsächlich nahezu gleich bewertete Blöcke betreffen oder tatsächlich hochrelevante Informationen verdrängen.

Vergleiche Indexer-Varianten bei identischen Eingangswerten und identischem K/V-Cache, um Selektionsfehler isolieren zu können.
Testfälle für Cache-Konsistenz
Berücksichtige außerdem:
- Page-Grenzen
- Fensterübergänge
- unvollständige Indexer-Blöcke
- Prompt-Cache-Reuse
- Conversation Snapshots
- Fortsetzung gespeicherter Konversationen
- parallele Sessions
- KV-Streaming
- speculative Decoding
- unterschiedliche Kontextlängen
Stelle sicher, dass quantisierte Cache-Daten nicht durch unbemerkte Layout- oder Formatwechsel falsch interpretiert werden.
9. Forschungsphasen
Arbeite schrittweise.
Phase 1 – Code-Audit
Verstehe den bestehenden K8V4-NL-Branch.

Dokumentiere:
- Was funktioniert bereits?
- Wie arbeitet das Fenster?
- Welche QSA-Integrationen fehlen?
- Welche impliziten Layoutannahmen bestehen?
- Wo entstehen zusätzliche Kopien oder unnötige Speicherzugriffe?
Noch keine größeren Architekturänderungen durchführen.
Phase 2 – Speicher- und Layoutmodell
Erstelle ein rechnerisches Modell für:
- Speicherbedarf
- Alignment
- Fragmentierung
- Indexer-Overhead
- Cache-Transfers
- unterschiedliche Layoutvarianten
Vergleiche dieses Modell mit tatsächlichen Allokationen und Messungen.
Phase 3 – QSA-Analyse
Untersuche die realen Selektionsmuster des Indexers.

Analysiere:
- Lokalität
- Wiederverwendung
- Fensterbezug
- Blockverteilung
- Selektionsstabilität
Leite daraus ab, ob QSA-aware Windowing oder ein anderes Page-Layout sinnvoll erscheint.
Phase 4 – Isolierte Prototypen
Implementiere nur die aussichtsreichsten Varianten.

Bevorzuge möglichst kleine, isolierte Experimente.

Alle experimentellen Änderungen sollen über klar erkennbare Feature-Flags aktiviert werden können.
Phase 5 – Benchmark und Entscheidung
Vergleiche die Prototypen mit dem vorhandenen K8V4-NL und den bestehenden Strata-KV-Formaten.

Bewerte jeden Ansatz anhand von:
1. Speicherersparnis
2. Qualität
3. Geschwindigkeit
4. Portabilität
5. Integrationsaufwand
6. Wartbarkeit
Verwerfe Ansätze, die nur theoretische Speichergewinne liefern, praktisch aber die Laufzeit deutlich verschlechtern oder die QSA-Auswahl instabil machen.
10. Erwartete Ergebnisse
Erstelle zum Abschluss einen technischen Research-Report:

docs/research/K8V4_NL_QSA_LAYOUT_RESEARCH.md

Der Report soll enthalten:
A. Current State
Was ist im Research-Branch bereits umgesetzt?

Welche Komponenten sind vollständig, teilweise oder noch gar nicht integriert?
B. Architecture Findings
Wie hängen K8V4-NL, Windowing, QSA-Indexer und Page-Layout tatsächlich zusammen?

Welche Architekturannahmen müssen verändert werden?
C. Memory Model
Erstelle eine nachvollziehbare Speicherbilanz für die untersuchten KV-Formate.

Führe Nutzdaten, Indexer, Metadaten, Page-Tables und Padding getrennt auf.
D. Layout Alternatives
Dokumentiere mindestens drei sinnvolle Layoutvarianten mit ihren Vor- und Nachteilen.

Gib an, welche Variante aus Performance- und Portabilitätssicht bevorzugt wird.
E. QSA-aware Windowing
Bewerte, ob ein QSA-aware Fenster gegenüber dem bestehenden Fenster einen messbaren Vorteil verspricht.

Trenne höhere Präzision von GPU-Residency.
F. Indexer Compression Feasibility
Bewerte, ob und unter welchen Bedingungen der FP32-Indexer komprimiert werden könnte.

Dokumentiere Risiken und notwendige Validierungen.
G. Experimental Results
Liefere Benchmarks, Profiling-Daten und Qualitätsergebnisse.

Kennzeichne eindeutig, was gemessen wurde und was lediglich eine Hypothese ist.
H. Recommended Next Steps
Erstelle eine priorisierte Liste konkreter Entwicklungsmaßnahmen.

Für jede Maßnahme:
- Erwarteter Nutzen
- Technisches Risiko
- Abhängigkeiten
- Geschätzter Implementierungsaufwand
- Notwendige Tests
- Klare Akzeptanz- und Abbruchkriterien
11. Wichtige Forschungsprinzipien
Beachte während der gesamten Untersuchung folgende Regeln:

1. Bestehende Arbeit respektieren.
K8V4-NL und das implementierte Fenster bilden die Ausgangsbasis. Keine Neuimplementierung ohne begründeten technischen Vorteil.

2. Qualität und Selektion getrennt betrachten.
Eine gute K/V-Rekonstruktion garantiert keine korrekte QSA-Auswahl.

3. Indexer nicht vorschnell quantisieren.
FP32 bleibt zunächst Referenz. Quantisierte Varianten müssen ihre Selektionsstabilität nachweisen.

4. Logische Pages und physisches Layout trennen.
Vier Tokens pro logischer Page sind zunächst beizubehalten. Größere physische Transferblöcke dürfen untersucht werden.

5. Kein künstliches Padding einführen.
Speicheralignment muss anhand realer Zugriffsvorteile bewertet werden.

6. Keine unnötigen Datenkopien.
Prüfe bevorzugt direkte quantisierte Attention-Zugriffe, statt große temporäre FP16-Buffer anzulegen.

7. Plattformübergreifend denken.
Das Speicherformat darf keine exklusiven GPU-Instruktionen voraussetzen.

8. Keine unbelegten Optimierungsversprechen.
Kompressionsrate, GPU-Traffic und End-to-End-Performance sind getrennt zu messen.

9. Kontrollierte Experimente durchführen.
Ändere nicht gleichzeitig Quantisierung, Indexer, Windowing und Layout, ohne isolierte Vergleichsmessungen.

10. Forschung vor Integration.
Zunächst Hypothesen, Speicheranalyse, isolierte Experimente und belastbare Benchmarks. Erst danach Produktionsintegration.
12. Abschließende Forschungsfrage
Beantworte auf Grundlage der Ergebnisse:
Wie lässt sich der bestehende asymmetrische K8V4-NL-KV-Cache mit Fensterstrategie so an die QSA-Indexer- und Speicherarchitektur von Qwen3.8-Flash-Next anpassen, dass die effektive GPU-Speichernutzung und Speicherzugriffseffizienz maximiert werden, ohne relevante Kontextselektionen zu verfälschen oder die plattformübergreifende Ausführung von Strata einzuschränken?

Identifiziere dabei ausdrücklich, welcher der folgenden Ansätze den größten belegbaren Nutzen bietet:
- Verbesserung des vorhandenen K8V4-NL-Codecs
- QSA-aware Windowing
- Optimiertes physisches Page-Layout
- Bessere KV-Streaming-Transfers
- Indexer-Kompression
- Kombination ausgewählter Maßnahmen
Beginne mit dem Code-Audit und der Speicheranalyse. Leite daraus die nächsten Experimente ab, statt eine bevorzugte Architektur vorwegzunehmen.
Meine Ergänzung: Eine Hypothese würde ich besonders stark verfolgen
Neben dem Prompt würde ich den Research-Agenten gezielt auf einen Aspekt aufmerksam machen:

Ein QSA-aware KV-Cache muss nicht nur die bereits selektierten Daten effizient lesen. Er sollte möglichst vermeiden, große Datenmengen überhaupt in die GPU übertragen zu müssen.

Strata besitzt dafür schon fast alle notwendigen Bausteine:
- einen Indexer, der relevante KV-Blöcke auswählt,
- vier Tokens pro logischer Page,
- quantisierte KV-Daten,
- ein Residency-System,
- und dein neues asymmetrisches K8V4-NL-Fenster.
Die interessante Frage ist daher, ob sich daraus zwei logisch unabhängige Ebenen entwickeln lassen:
Ebene	Optimierungsziel
Precision Layer	Welche Informationen benötigen wie viel Präzision?
Residency Layer	Welche Informationen müssen sich gerade auf der GPU befinden?


Mein Forschungsfavorit wäre eine QSA-aware Precision/Residency-Architektur, in der beide Entscheidungen auf gemeinsamen Zugriffsstatistiken beruhen, aber getrennt optimiert werden.

Dadurch könnte beispielsweise ein älterer, häufig selektierter Block niedrigpräzise im GPU-Speicher verbleiben, während ein neuerer, selten benötigter Block höherpräzise im Host-RAM liegt. Ob das sinnvoll ist, hängt natürlich von den tatsächlich gemessenen QSA-Zugriffsmustern ab.

Das würde auch eine Schwäche vieler allgemeiner KV-Quantisierungsmethoden umgehen: Sie optimieren primär die Kompressionsrate, berücksichtigen aber nicht ausreichend, dass QSA nur einen kleinen Teil des gespeicherten KV-Caches für jeden Decode-Schritt tatsächlich benötigt.

Ich würde deshalb im ersten Durchlauf bewusst keine neue PolarQuant-, TurboQuant- oder SpectralQuant-Implementierung verlangen. Der größere Erkenntnisgewinn dürfte zunächst darin liegen, K8V4-NL und seine Fensterstrategie auf die bestehende QSA-Speicherhierarchie abzustimmen. Erst wenn diese Architektur verstanden und profiliert ist, lässt sich zuverlässig entscheiden, welcher zusätzliche Codec sinnvoll ist.