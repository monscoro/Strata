# Ansatz-Verbesserungen: KV-Cache-Quantisierung & Exact-Cell-Fenster

Stand: 2026-10-24, Literaturabgleich nachgezogen · Autorenkontext: GLM-5.3 (opencode), auf Basis des Papers
`docs/paper/KV-Window-Hardening.tex` (nach der Grok-Revisionsrunde: 9 Seiten),
des GLM-Selbst-Reviews `PEER_REVIEW.md` (dasselbe Modell — Selbst-Review-Caveat
gilt; §8 dokumentiert die Blindstellen) und des unabhängigen Zweit-Reviews
`PEER_REVIEW_GROK.md` (grok-4.7 via opencode, Desk-Review ohne Messungen; alle
8 Major-Befunde wurden gegen die Messquellen verifiziert und in den Artikel
implementiert).

**Tabellenzählung im Paper (Stand PDF 2026-10-24, nach Grok-Revision):**
Tab. 1 Survey, Tab. 2 Footprint, Tab. 3 Decisions (A-Serie), Tab. 4 Runtime 128K,
Tab. 5 cwe-2k, Tab. 6 cwe-4k, Tab. 7 cwe-128k (Tail, erstes der zwei Texte),
Tab. 8 Regionen 128K (erster Text).
Am 2026-10-24 wurden alle Zahlen der Tabellen 4–8 Zeile für Zeile gegen
`docs/KV_QUANT_WORKPACKAGE.md` (Branch `KV_QUANT_research`) geprüft und die
Captions von Tab. 7/8 präzisiert; danach wurden die Revisionsplan-Items 1–10
des Grok-Reviews in den Artikel umgesetzt (u. a. KL-Definition top-64-trunkiert,
Hybrid-Modellbenennung, zwei 128K-Texte, ≈28K-Break-even, eingezogener
Crossover, untraceable Gates-Zählungen entfernt).

## 0. Regeln für dieses Dokument

- **Keine erfundenen Messwerte.** Jede Zahl hier ist entweder messberichtet
  (`bench/results/…`, `docs/DETAILS.md`, `docs/KV_QUANT_WORKPACKAGE.md`)
  oder explizit als Literaturangabe mit Quelle bezeichnet.
- Punkte, die nur aus Code-/Text-Lektüre folgen, sind als **statische
  Analyse** markiert und müssen vor jedem Claim verifiziert werden.
- Neue Features bleiben **opt-in**; der `int8`-Default-Pfad bleibt
  **byte-identisch** (SHA-Gate; Paper §5, A-Serie, und der Abschnitt
  „Correctness gates").
- Jede Prüfung, die sich zu einem offiziellen Claim hardnet, braucht einen
  Messbericht in `bench/results/<datum>-…` mit genau derselben Methodik wie
  das Paper (fp16-Referenz, gepaarte Statistik, vorzeichenbewusste Lesung).

## 1. Status quo (gemessene Basis)

| Punkt | Ergebnis | Quelle |
|---|---|---|
| Production-Point 256/64, B=6.144 | 5.29 bit/cell, 1.016 MiB @128K | `docs/DETAILS.md` |
| `int8`-Default | 8.25 bit/cell, 1.584 MiB | `docs/DETAILS.md` |
| Gap-Closure | 73 % (4K); über die zwei 128K-Texte 34–58 % der Tail-Mean-KL gegenüber plain | Paper Tab. 6, §7.3 |
| 2K-Lage | gegen `int8` nicht trennscharf (t=−1,92, int8 vorn; Fenster nur in 2/8 Segmenten besser) | Paper Tab. 5 |
| 4K-Lage | `int8` signifikant besser (t=−3,55; 7/8 Segmente für int8) im Mittel; Antwortregion nicht trennscharf | Paper Tab. 6 |
| 128K-Lage | Auf beiden Texten: `int8` gewinnt Tail-p99 (0.01655 vs 0.02977 / 0.0165 vs 0.0210), letzte 1.024 Positionen und Ganzsegment; Tail-Mean favorisiert je einen Arm pro Text (textübergreifend nicht konsistent); Ganzsegment-p99 liegt bei allen Armen hoch (3.66–5.55, isolierte Hartpositionen — unter der top-64-KL-Definition u. U. Floor-Ereignisse) | Paper Tab. 7/8, §7.3 |
| 256/64 vs 512/128 | 4K qualitativ identisch (t=+0,02); 128K nie qualitativ gemessen; 512/128 kostet +7 MiB (Tab. 4), decode −0.85 t/s; ihr Prefill-Vorteil reproduzierte sich nicht → 256/64 | Paper Tab. 4, Tab. 6, §8.2 |
| Level-Layout | K 16 uniforme Lloyd-Levels, V 8 Levels, beide im selben 4-Bit-Block (nicht „4-bit K / 8-bit V") | Paper §2/§3, WP §12.2 |
| Hadamard an/aus | `q4lm32nr` gegen `q4lm32`: Gleichstand (cwe-2k t +0,21; cwe-4k t −1,31 / −0,17, nirgends signifikant). Das ist der 16-Level-Basis-Arm, **nicht** das Fenster, nicht `q4lm8v`, nicht 128K | WP §7.9 |
| Schreiber-Platz | Post-Norm/Post-RoPE, dann FWHT (**statische Analyse**, WP §5.1). Pro-Kanal-Skalen im KIVI/KVQuant-Sinn können nur **vor** FWHT leben. `n_rot` = 64 von 256: FWHT mischt rotierte mit unrotierten Dims; das ist ungeprüft | WP §5.1 |
| Clip nach Rotation | Post-FWHT-Clip schadet auf cwe-2k: κ 0,94/0,94 +34 % KL (t −2,29), OSCAR-Paar 0,96/0,92 +111 % (t −6,66). Die OSCAR-Clip-Zahlen gelten für unrotierte Kanäle, nicht für diese Geometrie | WP §7.11 |
| `q4_0`-Anker | Anderes Protokoll, nicht cwe: Document-PPL etwa +8 % / +12 % gegen fp16 (Q2_0, 1K/8K); Swift-`quant_quality` z. B. KL 0,0364 auf dem festen Set (WP §7.1). **Kein** cwe-Paar von `q4_0` gegen das Fenster | `bench/results/2026-09-27-kv-q4`, WP §7.1 |
| Laufzeit | Prefill gleich, Decode −1.0 t/s (Streubreite; decode = Serving-Rate unter `--spec 4`-Drafts); 568 MiB frei, 416 Slots mehr | Paper Tab. 4 |
| Korrektheits-Gates | SHA-Byteidentität gegen den Vorrelease-Stand (dump-logits `A014F504…BA052`, int8 + fp16); Parität 12/12 (Synthetik-Suite `kv_q4_parity`, nicht A4); Nadel 18/18 im WP (plain, 256/64, 512/128, je 6/6, gesättigt); **2026-10-24 Reproduktion** int8 gegen 256/64, Engine 0.1.47, gleiche sechs Nadeln, 6/6 und 6/6, Antworten = gepflanzte Wörter (`bench/results/2026-10-24-needle-window/`); Window+Streaming-Verweigerung vor Load, Exit 2 | Paper „Correctness gates", `PEER_REVIEW_GROK.md` M7 |
| Break-even | ≈27.600 Zellen (B=6.144); darunter ist das Fenster teurer als der 4-bit-Basis-Effekt; bei 32K noch billiger als `int8` (7,67 vs 8,25 Bit) — Setup bietet ab 32K an | WP §7.20, Paper §3.2 |
| Hang | 1× auf *plain* `q4lm8v` @128K-Prefill, 6/6 Re-Runs ohne Reproduktion | Paper §9 (Threats) |
| Offen | A4-Paritätstest; CUDA/SYCL-Builds; 128K-Knie-Dichte; Task-Breite; per-channel-Anker **vor** der Rotation (A1); Indexer-Auswahl↔Overlay (C4) plus die Frage, welcher Anteil der ausgewählten Zellen überhaupt im Overlay liegt (E1) | Paper §8.2, Tab. 3, Abschnitt 6 |

## 2. Katalog

Priorität: **P1** = nächster sinnvoller Zug (hoher Informationsgewinn pro
Aufwand), **P2** = wichtig, danach, **P3** = Warteschlange.
Aufwand: **S** ≤ 1 Tag, **M** = mehrere Tage, **L** = Woche(n).

### A. Quantisierungslayout

| ID | Titel | Motivation (aus Status quo / Review) | Vorschlag | Prüfmessung | Aufwand | Prio |
|---|---|---|---|---|---|---|
| A1 | **Per-channel-K vor der Rotation** | KIVI/KVQuant quantisieren K entlang der Kanalachse; das Paper nennt genau diesen Arm als den fehlenden fairen Konkurrenten. **Korrektur gegen WP §5.1 (statisch):** nach FWHT gibt es keine Kanäle mehr. Ein „per-channel auf derselben Rotation"-Arm misst nicht, was KIVI misst. Der faire Arm skaliert K **vor** FWHT (oder ohne FWHT; die an/aus-Ablation auf `q4lm32` war bei 2K/4K ein Gleichstand, trägt diese Frage also nicht). Weiterhin nur auf diesem Indexer-Cache (12 von 48, top-2048), nicht auf einem Dense-Cache | Pipeline-Änderung, opt-in, V bleibt wie heute. Nicht bauen, bevor E1 sagt, dass der Restfehler in gelesenen Zellen liegt und nicht nur in ungelesenen | cwe-4k zuerst, dann 128K nur bei einem 4K-Unterschied; Gate: byte-identischer `int8`-Pfad | **M** | **P1** |
| A2 | Level-Symmetrie-Check (16/16, 8/8) | Der Level-Split (K 16, V 8, selber 4-Bit-Block) ist eine Byte-/Codebook-Wahl, keine gemessene Sensitivität. Die *Bit*-Variante („K will Bits") ist auf Swift falsifiziert (t +0,05, McNemar p=1,000). KIVI sagt die Richtung voraus; das darf die Lesung nicht vorwegnehmen. Der top-k-Artikel (Token gegen Bits) ist **nicht** die Begründung für 16/8 — das war eine Fehllektüre im ersten Review | Zwei Arme unter gleichen Bits: 16/16 und 8/8. Richtung nicht vorhersagen | Teacher-forced KL, gepaart, Vorzeichen mitberichten | **S** | **P1** |
| A3 | Pre-RoPE-K | KVQuants Pre-RoPE-Gewinn gilt für ihre per-channel Keys. Bei uns ist die Platzierung schon gelesen, nicht offen: der Schreiber sieht Post-Norm/Post-RoPE-K (WP §5.1, statisch), und FWHT mischt danach Kanäle, von denen nur 64/256 rotiert sind. Ein Pre-RoPE-Arm testet ein anderes Fehlermodell, solange A1 nicht gezeigt hat, dass die Kanalachse **vor** der Mischung überhaupt trägt | Nicht bauen, bevor A1 einen Unterschied zeigt. Wenn doch: eigener opt-in Pfad, `int8` unberührt | cwe-4k gegen den Post-RoPE-Schreiber, nur nach A1 | **M** | **P3** |
| A4 | Layer-wise-Präzisionspaare | KVTuner berichtet auf **ihren** Modellen nahezu verlustfreie 3,25–4,0 bit. Das ist kein Zielwert für unsere KL. Unser Format ist global; gemischt werden können nur die 12 Cache-Layer, nicht die 36 ohne diesen Cache. PyramidKV (mehr Budget in unteren Schichten, ihre 12-%-LongBench-Zahl) ist eine andere Achse: Token-Budget pro Layer, nicht Bits pro Layer. Output-KL schreibt den Fehler keiner Schicht zu, also fehlt das Signal für beide | Erst ein per-Layer-Fehlersignal (Instrumentierung, nicht die vorhandenen Logs). Danach höchstens 2 Stufen auf den 12 QSA-Layern, opt-in | cwe-4k; Gate: `int8` byte-identisch | **L** | **P3** |
| A5 | Non-uniform Level für K | KVQuant: sensitivitätsgewichtete Datentypen, ihre Wikitext/C4-PPL, nicht unsere KL. **Nicht noch einmal Clip:** Post-FWHT-Clip ist auf cwe-2k schlechter (WP §7.11). Offen ist nur eine andere 16-Level-Tabelle bei gleicher Blockgröße, nach der Rotation | Eine alternative Codebook-Tabelle, Leser unverändert, opt-in | cwe-4k gegen das uniforme Lloyd-Layout; 128K nur bei einem 4K-Unterschied | **S** | **P2** |

### B. Fenstergeometrie & Kosten

| ID | Titel | Motivation | Vorschlag | Prüfmessung | Aufwand | Prio |
|---|---|---|---|---|---|---|
| B1 | Knie-Sweep für S/R bei 128K | 256/64 ist der billigste **getestete** 128K-Arm mit sauberer Qualitätslesung, nicht das gemessene Optimum. StreamingLLMs „wenige Sinks" ist kein Auftrag, R auf 4 zu senken: R=8 war auf dem alten Gitter zu knapp (WP §7.16). R unter 64 erst nach E1, und nur wenn die ersten Zellen überhaupt gelesen werden | S∈{128, 192, 256, 384, 512} × R∈{64, 128} bei 128K. Kein R=4-Arm in dieser Runde | cwe-128k-KL plus Laufzeit (Tab.-4-Methodik) | **M** | **P1** |
| B2 | ≤3-Bit-Overlay bei 128K | `strata-wp`: die frühen Runden mit dünneren Level-Tabellen zeigten auf dem *R=8*-Gitter einen Byte-/Qualitäts-Crossover bei ≈4,8–5,3 bit/cell — **dieser Crossover wurde später als R=8-Artefakt eingezogen** (bei R=64 waren beide Basen über fünf Formen nicht trennscharf, \|t\| ≤ 1,50) und das alte Byte-Modell war überholt (Paper §8.4, Grok M6). **Was bleibt:** das Fenster trägt kürzere Level-Tabellen bei R=64 ohne messbaren Nachteil; ob ein ≤3-Bit-Base bei 128K konkurrenzfähig ist, ist ungemessen | Arm mit 3-bit-Overlay hinter dem 4-bit-Basisbetrieb bei 128K; Byte-Seite **zwingend mit dem korrigierten Modell** ($R+S+B$ zusätzlich) rechnen, bevor gebaut wird | cwe-128k + Laufzeit-A/B; Gate: ≤ heutige KL oder Memory-Bonus | **M** | **P3** |
| B3 | Streaming-kompatibler Sink-Pool | Paper §4: R-Zellen bleiben nahe Position 0; ein getrimmter side pool könnte mit einem gestreamten Cache kombinieren, ohne das Silent-Failure-Risiko zu vergrößern | Prototyp: *Sink-only*-Fenster (R ohne S-Ring) unter Streaming; A2-Lockerung als separater Flag mit lauter Doku | Nadel-Tests 32K/128K + Parität; zwingend: Verhalten „laut ablehnen" statt „leise quantisiert" | **M** | **P3** |
| B4 | `B`-Term-Kosten | Paper §3 (Byte fraction): 0.75 der 0.79 hinzugefügten bit/cell am 128K-Punkt sind der Ringanteil über B=6.144; B=512 ist billiger (4.60 bit, 884 MiB), aber Prefill ≈5× langsamer (105 t/s, Paper Tab. 4) → ein offline/batch-Punkt. Das Byte-Modell für eine kleinere Ring-Größe mit gleicher Latenz ist die Lücke (**statische Analyse**) | Option: die fp16-Kopie nur für den aktuellen Chunk resident halten und nach der Attention wieder freigeben, sodass mehrere kleinere Chunks vorprefillt werden; vorher das Byte-Modell neu rechnen | Byte-Modell neu rechnen, dann Same-Day A/B gegen Paper Tab. 4 | **M** | **P3** |

### C. Systemintegration & Korrektheit

| ID | Titel | Motivation | Vorschlag | Prüfung | Aufwand | Prio |
|---|---|---|---|---|---|---|
| C1 | **A4: Volle Zell→Zeilen-Parität** | Offener Review-Gate: ein form-spezifisches Fehllesen bliebe stumm | Dump/Re-Read-Test über alle Shapes (Ring-Wrap, veraltete Tags, Sinks, Batch-Grenzen, Leer) | Parität: bitweise, jede Shape | **S** | **P1** |
| C2 | CUDA/SYCL Build-Coverage | Bisher nur geplante Bauten, keine Messung: SYCL bietet `q4lm8v` ohnehin nicht an, aber der Code soll trotzdem bauen | `nvcc` und `dpcpp` kompilieren + Parität-Suiten | beide Backends grün; `q4lm8v` bleibt SYCL-off | **M** | **P1** |
| C3 | Hang-Untersuchung (plain 4-bit, 128K-Prefill) | Paper §9 (Threats): 1× nicht reproduzierbarer Hang auf dem *plain*-Arm; 6/6 Re-Runs ohne Reproduktion | Wiederholte 128K-Prefill-Zyklen auf dem plain-Arm mit Logging; nichts übernehmen, solange der Fund nicht reproduziert | Fund als Bug-Report; keine Aussage im Paper | **S** | **P2** |
| C4 | **Indexer-Auswahl ↔ Overlay verifizieren** | Paper §3, statische Analyse (Grok-Q3): das Overlay zwingt Sinks/Recent nicht in die top-2048-Selektion. Ohne diese Lesung ist E1 nicht interpretierbar | Lesen von `include/strata/kernels/qsa*.hpp` und `src/core/layer.cpp` auf dem Branch, der den Fenster-Code trägt (`KV_QUANT_research`; dieser Branch hat den Kernel nicht): (i) scoret der Indexer vor oder nach dem Overlay, (ii) gibt es einen Zwangspfad für Sinks/Recent | **Zunächst statisch.** Kein Kernel, bevor die Lesung steht | **S (statisch)** | **P1** |

### D. Evaluations-Decke

| ID | Titel | Motivation | Vorschlag | Prüfmessung | Aufwand | Prio |
|---|---|---|---|---|---|---|
| D1 | **Zweiter RULER-Task** | Review M2: Nur cwe gemessen | z. B. `niah_multikey` oder `variable_tracking` bei 4K, im Harness | gleiche Segment-/Scoring-Struktur wie heute; Gate: keine qualitative Abweichung von der Heut-Lesung | **S** | **P1** |
| D2 | **Gepaarte Mittelregion bei 128K** | Review Q4: Tail-Scoring (Paper Tab. 7) übersieht innenliegende Degradation. **Update (Grok):** ein *Ein-Arm*-Innenwert existiert bereits — der WP-Bericht listet für den Window-Arm KL 0,054 auf den ersten 4.096 Positionen und 0,164 auf einem mittleren 4K-Block (Positionen 65.536–69.631); die anderen Arme sind **nicht** auf diesem Block gescoret | Der one-arm-Schnitt steht; fehlen: `int8`-, plain- und fp16-Score auf demselben Block (65.536–69.631), dann gepaarte Differenz | Familien-Logs der WP-Läufe; **welche raw logs dafür noch existieren, ist zu prüfen — die `bench/results`-Ordner des Tree sind leer**, sonst Re-Run mit `--prefill 512` | **S (falls Logs)** | **P1** |
| D3 | Multi-Turn-A/B | Review Q3: Das Fenster ist für „Chat, der wächst" gebaut, aber nie als solches gemessen (Paper §8.3) | 8–16 lange Prompts; danach Follow-up-Runden, jede +512 Anhänge-Tokens; Fenster gegen `int8` | Prefill/Decode wie Paper Tab. 4 + KL der Antwortregion | **M** | **P2** |
| D4 | p99-Bootstrap-CI | Review Minor 5: p99 ohne Konfidenzband (Paper §6, Metrics-Notiz) | Bootstrap (10k Resamples) über die vorhandenen Per-Position-Logs | CI-Intervalle in den Tabellen; **offline, keine GPU** | **S** | **P1** |
| D5 | LongBench-Subset | D1 nachrangig, breiteres Taskfeld | Zwei LongBench-Subsets bei 4K/8K; konkrete Auswahl erst nach D1 | gegen `int8` gemessen, gleiche Methodik | **M** | **P3** |
| D6 | **KL-Ebene rescoren (Reverse + Vokabular)** | Review M4 (GLM) + Grok M2: Zwei Skalen der Frage sind offen. (a) Die KL der Tabellen ist **top-64-trunkiert, renormalisiert und mit Floor-Substitution** gerechnet (`quant_quality.py`, `kl_between`) — Masse außerhalb der Referenz-Top-64 tritt nur über den 64.-Logprob-Floor ein; die 17-Nat-Positionen in Tab. 8 tragen genau diese Floor-Signatur (Paper §6.3). (b) Die Gegenrichtung $D(\mathrm{arm}\Vert\mathrm{ref})$ wurde nie gerechnet | **Offline-Doppelbewertung auf vorhandenen per-Position-Logs:** (a) Full-Vocabulary-KL statt top-64-trunkiert (neue Scoring-Passage im Harness), (b) gepaarte Differenz der Richtungen | beide Sichten auf denselben Logs; Bericht: sind Rangfolge und/oder 2K/4K/128K-Einordnung stabil unter Definitionwechsel? **Nur Offline, keine GPU** — aber nur, wo raw logs noch existieren (siehe D2-Vorbehalt) | **S** | **P1** |

### E. Was der Status quo gegen die Paper-Referenzen noch nicht beantwortet

Diese Zeilen sind der Literaturabgleich (Abschnitt 6), nicht neue Messungen.
Jede Literaturzahl darin ist die des zitierten Papiers auf dessen Modell, nicht unsere.

| ID | Titel | Motivation | Vorschlag | Prüfmessung | Aufwand | Prio |
|---|---|---|---|---|---|---|
| E1 | **Deckt das Overlay die gelesenen Zellen?** | Der teure Restfehler liegt außerhalb des Fensters: `int8` gewinnt p99, letzte 1.024 und das Ganzsegment, das Fenster schließt nur 34–58 % der Tail-Mean-KL. KIVI/StreamingLLM begründen fp16 auf *recent* und *sinks* nur, wenn die Attention diese Stellen liest. Der top-k-Artikel sagt dasselbe von der anderen Seite: Token-Budget und Bits gehören zusammen. Unser Indexer liest höchstens 2.048 Zellen pro Query; das Overlay ändert die Präzision, nicht die Auswahl (Paper §3, statisch). Ob die ausgewählten Zellen in R+S fallen, ist ungemessen | Eine instrumentierte 128K-Prefill-Passage, nach C4: Anteil der Indexer-Treffer in R, in S, in keinem von beiden, pro QSA-Layer. Kein neuer Kernel. Liegt der Anteil nahe 0, ist B1 (mehr S/R) der falsche Ausgabeort und ein positionsfestes Overlay schützt Zellen, die nicht gelesen werden | Ein Lauf, Zähler, kein KL-Urteil. Bericht: Anteile, nicht eine Qualitätszahl | **S** | **P1** |
| E2 | **`q4_0` auf dem cwe-Harness** | llama.cpp ist der Produktionsanker im Paper (q8_0 / q4_0). Unser `q4_0` ist auf dem Document-Probe und auf `quant_quality` gemessen, nicht auf cwe gegen das Fenster. Das Format existiert; es braucht keinen neuen Kernel | `--kv q4_0` (und, wenn der Harness es schon kann, `k8v4`) auf cwe-4k neben `int8`, plain `q4lm8v` und dem Fenster. Die quant_quality-PPL nicht in diese Tabelle schreiben | Dieselbe gepaarte KL wie Tab. 6. 128K nur, wenn 4K das Fenster nicht schon einordnet | **S** | **P2** |
| E3 | Heavy-Hitter-Overlay, keine Eviction | H2O sammelt Attention als Heavy Hitters und **wirft** Zellen weg. Wegwerfen bleibt Nicht-Ziel (A-Serie: eine still fehlende Zelle ist ein Fehler). Die nicht-evictive Lesung — fp16 auf gesammelte Treffer, 4 bit auf den Rest, alles adressierbar — ist nur dann ein Kandidat, wenn E1 zeigt, dass R+S die gelesenen Zellen verfehlt | Nicht bauen vor E1. Wenn E1 daneben liegt: opt-in Overlay-Politik, eigener Härtungs-Punkt, laute Verweigerung wenn die versprochene exakte Zelle nicht im Pool ist | cwe-4k gegen 256/64 bei gleichem fp16-Budget, nicht bei gleichem R+S | **L** | **P3** |
| E4 | **GEAR-Diagnose, kein GEAR-Kernel** | GEAR korrigiert den 4-bit-Rest mit Low-Rank plus dünnen Ausreißern (ihre Messung). Unser Rest nach dem Fenster ist genau die Fläche, die das Overlay nicht deckt. Die 17-Nat-Positionen sind unter der trunkierten KL Floor-Ereignisse, also noch kein Beleg für dünne Ausreißer | Offline, wo per-Position-Logs existieren: ist der Restfehler auf wenige Positionen konzentriert oder über den Bulk verteilt? Kernel nur, wenn die Verteilung dünn ist. Sonst ist mehr Fenster (B1) oder eine andere Basis (A1/A5) die passendere Antwort | Histogramm der vorhandenen per-Position-KL; keine GPU | **S** | **P2** |

## 3. Vorschlag zur Arbeitspaket-Gliederung („WP-3")

Die obenstehenden P1-Punkte ergeben ein sinnvolles nächstes Arbeitspaket
in der Reihenfolge, in der sie voneinander abhängen:

1. **WP-3a — Wohin das fp16-Geld geht (ändert die weiteren Arme):**
   C4 (Indexer↔Overlay, statisch, auf `KV_QUANT_research`) → E1 (Anteil der
   ausgewählten Zellen in R+S, ein Zählerlauf).
   *Ergebnis:* Entweder das positionsfeste Overlay trifft die gelesenen Zellen,
   dann bleiben B1 und die KIVI-Restbegründung stehen — oder es trifft sie nicht,
   dann ist mehr S/R der falsche Ausgabeort und A1/E3 werden anders geschnitten.
2. **WP-3b — Evaluations-Absicherung (offline, wo Logs existieren):**
   D6 (KL-Ebene rescoren) → E4 (Restfehler dicht oder dünn, dieselbe Log-Lage) →
   D4 (p99-CI) → D2 (gepaarte Mittelregion; **Vorbehalt: raw logs prüfen, Tree-Ordner sind leer**) →
   D1 (zweiter Task, GPU S).
   *Ergebnis:* Robustere Berichtzahlen + die Frage, ob GEAR überhaupt die richtige Form für den Rest ist.
3. **WP-3c — Community-Anker:**
   A1 (per-channel **vor** FWHT, nicht danach) + A2 (Level 16/16 vs 8/8, Richtung nicht vorwegnehmen) +
   E2 (`q4_0` auf cwe-4k, der llama.cpp-Anker auf demselben Harness).
   *Ergebnis:* Der Punkt liegt neben KIVI-Achse und Produktionsformat, nicht nur neben `int8`.
4. **WP-3d — Geometrie-Feinschliff:** B1 (Knie-Sweep 128K, R nicht unter 64,
   und nur wenn E1 die Sinks/Recent als gelesen ausweist).
   *Ergebnis:* konfirmierter oder korrigierter Production-Point.
5. **WP-3e — Engineering:** C1 (A4-Parität) → C2 (CUDA/SYCL).
   *Ergebnis:* Die zwei offenen Review-Gates des Papers schließen.
6. **WP-3f (optional, GPU-Schleife):** D3 (Multi-Turn), B2 (≤3-Bit-Overlay
   bei 128K, **vorher Byte-Seite korrekt rechnen**; der alte 4,8–5,3-Crossover
   ist als R=8-Artefakt eingezogen), A5 (andere Level-Tabelle) — nur wenn ein
   neues Speicher-Ziel oder ein 4K-Unterschied es verlangt. A3, A4, E3 bleiben
   hinter ihren Gates.

**Gates für jede Stufe:**

- Standard-Parität + Byteidentitäts-SHA auf `int8` unverändert.
- Messberichts-Zeilen in `bench/results/<datum>-<arm>/` mit Anzahl
  gemessener Positionen, Rounds, Layout.
- Update in `docs/DETAILS.md` und hier: Status-Wechsel P1 → erledigt.

## 4. Bewusste Nicht-Ziele

- **Eviction** (SnapKV/H2O/PyramidKV-artig) bleibt draußen: das Fenster ist
  quantisierungsseitig konzipiert und hält jede Zelle adressierbar (die
  A-Serie behandelt eine still nicht mehr verfügbare Zelle als Fehler).
  H2O als Overlay-Politik (E3) ist davon getrennt und hinter E1 gesperrt.
- **FP8**: Auf Ziel-Hardware (Consumer-AMD über HIP) hat kein FP8-
  Attention-Kernel — siehe Paper §1. vLLMs „1–2 Punkte" sind ihre Suite,
  ihr Head-Dim 256, nicht ein KL-Ziel für uns.
- **Clip nach der Rotation** nicht wieder öffnen: cwe-2k hat es schlechter
  gemessen (WP §7.11). **Hadamard aus** nicht als Qualitätsarm wieder öffnen:
  der Gleichstand gilt für `q4lm32` bei 2K/4K (WP §7.9), und ein Speed-Claim
  über drei gesparte FWHTs ist auf dem Fensterpfad nicht gemessen.
- **R=4** nicht aus StreamingLLM ableiten. R=8 war auf dem alten Gitter zu knapp.
- **Keine Default-Änderung**: `int8` bleibt Standard, bis eine der obigen
  Messungen eine signifikante Überlegenheit belegt **und** alle
  Korrektheits-Gates (SHA-Byteidentität, Parität, Nadel, Setup-Suite) grün
  sind.

## 5. Quellen

Paper-Messbasis: `docs/DETAILS.md` (§ „KV cache quantization"; u. a. der
zweite 128K-Text und der SHA-Gate-Wortlaut),
`docs/KV_QUANT_WORKPACKAGE.md` (Branch `KV_QUANT_research`; §5.1 Schreiber-Platz,
§7.9 Hadamard-Gleichstand, §7.11 Clip-Urteil, §7.12–§7.21 Fenster-Historie,
K-vs-V-Falsifikation und Crossover-Einzug),
`bench/results/2026-09-27-kv-q4/`,
`bench/results/2026-10-24-needle-window/` (Nadel-Paar int8 gegen 256/64, Engine 0.1.47).
**Vorbehalt (verifiziert 2026-10-24):** die Window-Laufordner
`bench/results/2026-10-14-kv-q4lm32/`, `2026-10-17-kv-wp2a-window/`,
`2026-10-17-kv-wp2b-window3bit/`, `2026-10-18-kv-runtime-ab/` und
`2026-10-18-kv-wp2b-oscar-knee/` liegen in diesem Tree **leer** (0 Dateien);
der Workpackage-Bericht auf dem Research-Branch ist die Referenz, keine
per-Position-Raw-Logs (deshalb der D2/D6/E4-Logs-Vorbehalt).
Reviews: GLM-5.3-Selbst-Review (`PEER_REVIEW.md`, Blindstellen in dessen §8)
und `PEER_REVIEW_GROK.md`. Literatur: die Bibliographie von
`KV-Window-Hardening.tex`, nicht die Draft-Formulierungen in `PEER_REVIEW.md` §6
(dort steht u. a. noch „4-bit K / 8-bit V" und der GEAR-Titel „Serving").
**Aufwand und Prioritäten sind Schätzungen (statische Analyse), keine Messwerte.**

## 6. Literaturabgleich (Paper-Bibliographie gegen den Status quo)

Jede Zeile benutzt nur, was das Paper der Arbeit zuschreibt, plus die
WP-Befunde oben. Literaturzahlen sind nicht unsere.

| Referenz | Was davon bei uns schon entschieden ist | Was daraus noch prüfbar ist |
|---|---|---|
| StreamingLLM (Sinks) | Das Overlay behält die ersten R Zellen. R=64 ist das kurze-Kontext-Knie, das den Fenster-Arm an `int8` heranbringt; R=8 war zu knapp | Nicht „weniger Sinks". Erst E1: werden die ersten Zellen vom Indexer gelesen? |
| KIVI (per-channel K, per-token V, fp16-Recent) | Wir haben das Recent-Residual (S), nicht die Kanalachse. Der Bit-Kontrast „K sensitiver als V" ist falsifiziert | A1, und zwar **vor** FWHT. Das Recent-Residual trägt nur, wenn E1 die letzten S Zellen in der Auswahl findet |
| KVQuant (per-channel, pre-RoPE, non-uniform; <0,1 PPL auf ihren Texten) | Platzierung ist post-RoPE (statisch). Clip nach FWHT schadet. Ihre PPL ist kein KL-Ziel | A5 (andere Tabelle, nicht Clip). A3 nur hinter A1 |
| GEAR (4 bit + Low-Rank + dünne Ausreißer) | Unser Restfehler ist die Fläche außerhalb des Fensters. Die 17-Nat-Stellen sind Floor-Ereignisse, kein Ausreißerbeleg | E4, offline. Kein Kernel vorher |
| QuaRot, RotateKV | FWHT an/aus auf `q4lm32`, 2K/4K: Gleichstand. Die Rotation trägt dort keine messbare KL | Nicht wiederholen. Speed (drei FWHTs) auf dem Fensterpfad ist ungemessen und bei Decode-in-Streubreite kein P1 |
| KVTuner (3,25–4,0 bit, ihre Modelle) | Globales Format. Nur 12 Layer haben diesen Cache | A4, hinter einem per-Layer-Signal, das die Output-KL nicht hergibt |
| PyramidKV (mehr Budget unten; 12 % Cache ≈ voll auf LongBench, ihre Zahl) | Dieselbe Lücke: kein per-Layer-Fehler | Nicht als Sweep. Hängt an demselben Signal wie A4 |
| SnapKV, H2O | Eviction ist Nicht-Ziel | E3 nur als Overlay, nur nach E1 |
| top-k (Token gegen Bits) | Nicht die Begründung für 16/8. Die Bit-Richtung ist falsifiziert | E1: das Token-Budget ist bei uns die Indexer-Auswahl, die Bits sind das Overlay |
| llama.cpp q4_0 / q8_0 | `q4_0` ist auf anderen Protokollen gemessen, nicht auf cwe gegen das Fenster | E2 |
| vLLM FP8 | Kein FP8-Attention-Pfad auf der Zielkarte. Ihre 1–2 Punkte sind suite-spezifisch | Nicht verfolgen |
