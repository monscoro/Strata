# Ansatz-Verbesserungen: KV-Cache-Quantisierung & Exact-Cell-Fenster

Stand: 2026-10-28 (S3(a) gemessen, s. §7), Literaturabgleich nachgezogen · Autorenkontext: GLM-5.3 (opencode), auf Basis des Papers
`docs/paper/KV-Window-Hardening.tex` (10 Seiten, 8 Tabellen; Stand 2026-10-25),
des GLM-Selbst-Reviews (dasselbe Modell — Selbst-Review-Caveat gilt; §8
dokumentiert die Blindstellen) und des unabhängigen Zweit-Reviews (grok-4.7 via
opencode, Desk-Review ohne Messungen; alle 8 Major-Befunde wurden gegen die
Messquellen verifiziert und in den Artikel implementiert). Beide Reviews liegen
**außerhalb des Repos** und sind hier nur als Provenienz genannt, nicht als
Quelle für Zahlen.

**Tabellenzählung im Paper (Stand PDF 2026-10-25):**
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
- **Die Reviews sind Werkzeug, nicht Auftrag.** Die Textpunkte beider Reviews
  sind abgeschlossen und werden nicht erneut aufgerollt. Die Messpunkte (GLM
  Nr. 11/12, Grok Q2/Q3) sind in die Leiter S0–S5 eingegangen, **soweit sie dem
  5,0-Bit-Ziel dienen**; sie sind damit Leiterstufen, keine Review-Pflichten.
  Was dem Ziel nicht dient, wird nicht gemessen: Multi-Turn (D3), der
  S/R-Knie-Sweep und ein Vokabular-Vollrescore bleiben offen. Die *Regeln*
  dieses Abschnitts gelten weiter — nicht weil ein Review sie verlangt, sondern
  weil der neue Claim („`int8`-Qualität unter 5,0 bit") an genau denselben
  Schwächen gemessen würde (trunkierte KL, fehlender Build, fehlende Rohlogs).
- **Bilanzregel Indexer:** Der FP32-Indexer (128 Dimensionen je 4-Zellen-Block,
  rechnerisch 192 MiB bei 128K bzw. 1,0 bit je KV-Wert) ist **Teil der
  Modellarchitektur**, nicht des KV-Codecs. Er wird deshalb **nicht** in bit/cell
  eingerechnet und nicht mit dem KV-Speicher gemittelt; eine Indexer-Kompression
  ist eine eigene, spätere Frage mit eigenem Selektions-Nachweis (Score-Abstand an
  der Grenze, Flip-Rate), kein Hebel dieses Arbeitspakets.
- **Sichtbare Versionierung:** jedes Paper trägt auf Seite 1 eine Versionszeile
  (Version, Build-Zeit mit Zeitzone, Datum der Erstfassung) und dieselbe Angabe
  in den PDF-Metadaten; die `.tex`-Datei führt die Versionshistorie als
  Kommentarblock, eine Zeile pro Inhaltsrevision. Die Version zählt
  Inhaltsrevisionen; die Zeit ist der Build und wird beim Kompilieren aus der
  Uhr gesetzt (`\buildstamp`), damit sie nicht von der Datei abweicht.

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
| Korrektheits-Gates | SHA-Byteidentität gegen den Vorrelease-Stand (dump-logits `A014F504…BA052`, int8 + fp16); Parität 12/12 (Synthetik-Suite `kv_q4_parity`, nicht A4); Nadel 18/18 im WP (plain, 256/64, 512/128, je 6/6, gesättigt); **2026-10-24 Reproduktion** int8 gegen 256/64, Engine 0.1.47, gleiche sechs Nadeln, 6/6 und 6/6, Antworten = gepflanzte Wörter (`bench/results/2026-10-24-needle-window/`); Window+Streaming-Verweigerung vor Load, Exit 2 | Paper §7.5 „Correctness gates", `bench/results/2026-10-24-needle-window/` |
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
| C4 | **Indexer-Auswahl ↔ Overlay verifizieren** | Paper §3, statische Analyse (Grok-Q3): das Overlay zwingt Sinks/Recent nicht in die top-2048-Selektion. Ohne diese Lesung ist E1 nicht interpretierbar | Lesen von `include/strata/kernels/qsa*.hpp` und `src/core/layer.cpp` **auf diesem Branch** (`kv_q4_quant_with_int8_quality`, Fenster-Code vorhanden; die Lesung wurde bisher auf `KV_QUANT_research` begonnen): (i) scoret der Indexer vor oder nach dem Overlay, (ii) gibt es einen Zwangspfad für Sinks/Recent | **Zunächst statisch.** Kein Kernel, bevor die Lesung steht | **S (statisch)** | **P1** |

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

Die P1-Punkte zerfallen in zwei Spuren. **Spur 1** ist die Produktentscheidung
und steht vollständig als Leiter S0–S5 in §7; sie wird hier nicht wiederholt.
**Spur 2** rundet den Punkt ab, blockiert die Leiter nicht und läuft parallel,
weil sie offline oder ohne GPU läuft. Diese Gliederung sortiert Spur 2 nach
Informationsgewinn pro Aufwand.

1. **WP-3a — Wohin das fp16-Geld geht** ist S1 der Leiter (C4 statisch →
   E1-Zähler). Ohne diese Lesung ist jeder weitere Arm ein Ratespiel; deshalb
   steht sie in §7, nicht hier.
2. **WP-3b — Evaluations-Absicherung (offline, wo Logs existieren):**
   D6 (KL-Ebene rescoren) → E4 (Restfehler dicht oder dünn, dieselbe Log-Lage) →
   D4 (p99-CI) → D2 (gepaarte Mittelregion; **Vorbehalt: raw logs prüfen, Tree-Ordner sind leer**).
   *Ergebnis:* robustere Berichtszahlen. D6 und E4 liefern zugleich die
   Eingangsgrößen für S0 und S1.
3. **WP-3c — Engineering:** C1 (A4-Parität) → C2 (CUDA/SYCL).
   *Ergebnis:* Die zwei offenen Review-Gates des Papers schließen. Für Latte 1
   (Default-Anspruch) blockierend, für Latte 2 (opt-in) nicht.
4. **WP-3d — Anker und Breite, nur auf Zuruf:** E2 (`q4_0` auf cwe-4k, der
   llama.cpp-Anker auf demselben Harness), D1 (zweiter RULER-Task; als S5 nur
   nach einem Latte-Gewinn), D3 (Multi-Turn), B2 (≤3-Bit-Overlay bei 128K,
   **vorher Byte-Seite korrekt rechnen**; der alte 4,8–5,3-Crossover ist als
   R=8-Artefakt eingezogen). Keiner davon ist eine Stufe: sie ordnen ein oder
   begrenzen die Aussage.
5. **WP-3e — Geometrie, gesperrt bis S1:** B1 (Knie-Sweep 128K, R nicht unter
   64) läuft nur, wenn S1 zeigt, dass die Sinks/Recent gelesen werden. A3, A4,
   E3 bleiben hinter ihren Gates.

**Gates für jede Messung:**

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
- **Keine Default-Änderung ohne Latte 1**: `int8` bleibt Standard, bis ein Arm
  auf **keiner** robusten Statistik hinter `int8` liegt (§7, Latte 1: 4K gepaart,
  128K Ganzsegment + p99 + letzte 1.024 auf beiden Texten) **bei ≤ 5,0 bit/cell**
  **und** alle Korrektheits-Gates grün sind (SHA-Byteidentität, Parität 12/12,
  Nadel, Setup-Suite, plus C1 und C2). Ein Arm, der nur Latte 2 erreicht (Parität
  innerhalb der Margen bei ≤ 6,0 bit/cell), wird als opt-in Modus ausgeliefert und
  ändert den Default nicht.

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
Reviews: GLM-5.3-Selbst-Review (Blindstellen in dessen §8) und der
grok-4.7-Desk-Review. Beide Dateien liegen **außerhalb dieses Repos**; zitiert
wird hier nur, was in Paper oder Messordnern belegt ist. Literatur: die
Bibliographien von `KV-Window-Hardening.tex` und
`KV-int8-quality-below-5bit-quantization.tex`, nicht die
Draft-Formulierungen der Reviews (dort steht u. a. noch „4-bit K / 8-bit V" und
der GEAR-Titel „Serving"). Die **Margen in §7 (Latte 1) stehen nur hier**, nicht
im Paper: das Paper nennt Latte A (Parität bei ≤ 6,0 bit/cell, opt-in) und
Latte B (Default-Wechsel unter 8,25 bit/cell); §7 verschärft beide auf
**5,0 bit/cell mit den K-Margen**. Diese Zahlen sind eine Arbeitsfassung und
werden mit dem ersten Lauf eingefroren.
**Aufwand und Prioritäten sind Schätzungen (statische Analyse), keine Messwerte.**

## 6. Literaturabgleich (Paper-Bibliographie gegen den Status quo)

Jede Zeile benutzt nur, was das Paper der Arbeit zuschreibt, plus die
WP-Befunde oben. Literaturzahlen sind nicht unsere. Der Survey im Paper
(Stand nach dem 2026-10-25-Ausbau) zitiert zusätzlich die nächsten
Mechanismen: SKVQ, IntactKV, RotateKV, PM-KVQ, SAW-INT4, SpinQuant,
Palu, ShadowKV, Quest, DuoAttention, RazorAttention, NSA, QServe,
Scissorhands, ChunkKV, und die Gegenrichtung „keys want the bits"
(`spectralkv`). Die Leiter in Abschnitt 7 ändert das nicht.

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

## 7. Arbeitsplan: besser als `int8` bei 5 bit/cell

Die Leiter steht seit dem 2026-10-25 in zwei Papers: `KV-Window-Hardening.tex`
(Abschnitt „Avenues for Improvement", zwei Latten A/B) und
`KV-int8-quality-below-5bit-quantization.tex` (Status und Arbeitsplan, das
5,0-Ziel mit Kriterien K1–K6 und derselben Leiter S0–S5). Dieses Kapitel ist
die Arbeitsfassung und **verschärft** das Ziel: **5,0 bit/cell** — weniger Speicher
als das heutige Fenster (5,29) bei besserer Qualität als `int8` (8,25). Das ist
die einzige Fassung, die den Branch-Namen trägt: `int8`-Qualität, nicht „etwas
weniger schlecht als `int8`".

### Warum die alte Latte zu schwach war

Die alte Fassung verlangte Überlegenheit und ließ 8,25 bit/cell zu. Ein Arm, der
`int8` bei 8,2 bit/cell schlägt, spart 0,6 % Speicher — ein Sieg ohne Produkt.
Die Byte-Grenze ist das, was aus einem Qualitätsergebnis ein Produkt macht.

Umgekehrt fehlte der Rückfall. Ein Arm, der `int8` nicht schlägt, aber bei
6,0 bit/cell innerhalb einer Marge bleibt, ist dem heutigen Fenster (5,29 mit
gemessenem Rückstand) vorzuziehen und als opt-in auslieferbar. Diesen Rückfall
hält das Paper als Latte A fest; er bleibt hier als Latte 2 stehen, aber er ist
nicht das Ziel.

Und 5,0 ist keine Rundung von 5,29: es sind 6 % weniger Speicher als das
heutige Fenster und 39 % weniger als `int8` (Rechenweg unten). Wer 5,0 erreicht,
hat das Fenster nicht verbessert, sondern ersetzt.

### Das Ziel (Latte 1)

Ein Arm gewinnt nur, wenn **alle** Punkte gelten. „Besser als `int8` in allen
Bereichen" heißt: `int8` liegt auf keiner der robusten Statistiken vorn. Ein
Gleichstand auf einer Statistik ist kein Verlust (die Rate ist 39 % niedriger),
ein Rückstand ist einer.

| # | Punkt | Schwelle | Anker (heute) |
|---|---|---|---|
| K1 | cwe-4k, gepaarte Mean-KL (Arm − `int8`) | Differenz ≤ 0, obere 95-%-Grenze < **+0,001 nats**, `int8` führt in höchstens 4 von 8 Segmenten | +0,00402, t = −3,55, `int8` führt in 7/8 |
| K2 | cwe-128k, **beide** Texte, Ganzsegment-KL | ≤ `int8` auf beiden Texten | +0,02597 (Text 1), +0,01089 (Text 2) |
| K3 | cwe-128k, **beide** Texte, p99 und letzte 1.024 | ≤ `int8` auf beiden Texten und beiden Statistiken | p99 +0,01322 / +0,00450; letzte 1.024: 0,00444 (Fenster) gegen 0,00265 (`int8`) |
| K4 | effektive Rate bei 128K | ≤ **5,0 bit/cell** (Byte-Modell, vor dem Lauf aufgeschrieben) | 39 % unter `int8`, 6 % unter dem Fenster |
| K5 | Laufzeit | Prefill höchstens **10 % unter `int8`** (526 t/s), Decode in der Streubreite | Setzung; B=512 zeigt −80 %, das Risiko ist real |
| K6 | Gates | `int8`-SHA unverändert, Parität 12/12, Nadel 6/6 + 6/6, Setup-Suite, plus C1/C2 | wie heute |

Die 10 % in K5 sind eine Setzung, keine Messung: das heutige Fenster liegt bei
528 gegen 526 t/s (+0,4 %), also praktisch gleich; ein Ring, der ein Drittel
kleiner ist, kann Prefill kosten, und dieser Preis muss begrenzt sein.

**Latte 2 (Rückfall, opt-in).** Wenn Latte 1 nicht erreichbar ist, bleibt der
Rückfall aus dem Paper: Parität innerhalb der halben heutigen Abstände bei
≤ 6,0 bit/cell — 4K-Mean ≤ +0,002, Ganzsegment ≤ +0,013 auf beiden Texten, p99
≤ +0,007 auf beiden Texten, Rate ≤ 6,0. K5 (Laufzeit) und die Gates aus K6 ohne
C1/C2 gelten auch hier. Ein Arm, der das erfüllt, ersetzt das heutige Fenster
als opt-in Punkt und lässt den Default bei `int8`. Latte 2 ist kein Ziel, sondern
die Untergrenze, unter der der Branch nichts ausliefert.

### Das Byte-Budget bei 5,0 (Rechnung, keine Messung)

Das Byte-Modell des Papers (§3.2): bits/cell ≈ 4,50 + 16(R+S+B)/L, mit L =
131.072 bei 128K. Die 4,50 sind die 4-bit-Basis (576 B je Zelle und Layer); der
zweite Term ist der residente Ring, der die Zellen exakt hält. Daraus:

- Bei 5,0 bleiben für den Ring **0,50 bit/cell** → **R+S+B ≤ 4.096 Zellen**.
  Mit 256/64 ist das B ≤ 3.776 gegen heute 6.144; der Ring schrumpft damit von
  6.464 auf 4.096 Zellen (**−37 %**), wenn er fp16 bleibt.
- Der Ring muss aber nicht fp16 sein. `int8` ist das Qualitätsziel, nicht fp16:
  ein Ring aus int8-Zellen (8,25 bit inkl. Skala) erlaubt **7.943 Zellen** —
  23 % mehr als heute — bei 5,0. Ein 12-bit-Ring erlaubt 5.461 (−15 %).
- Die Basis kann nicht unter 4,50: 1.024 Werte je Zelle und Layer ergeben bei
  4,5 bit je Wert 576 Byte (32 Blöcke à 18 Byte), und weniger Level passen nicht
  in dieselben 4 bit.

Damit hat das Ziel zwei Stellschrauben, und beide sind nötig: **weniger oder
billigere exakte Zellen** (Byte-Seite) und **bessere Bulk-Kodierung** (Qualität
für den Rest). Keine allein reicht: ein kleinerer Ring verliert die
Tail-Statistiken, die `int8` heute gewinnt; ein besserer Bulk ohne Byte-Seite
bleibt über 5,0.

Nebenbei: der Break-even gegen `int8` sinkt mit dem kleineren Ring von ≈27.600
auf ≈17.500 Zellen. Setup bietet das Fenster ab 32K an; daran ändert sich
nichts, aber der Punkt wird auch bei 24K schon billiger als `int8`.

### Was das Produkt davon hat (Rechenweg, linear aus Tab. 4)

Die zwei gemessenen Punkte: `int8` 8,25 bit → 1.584 MiB, Fenster 5,29 →
1.016 MiB bei 128K (B=6.144).

- 5,0 bit/cell → ≈ **960 MiB**, also **≈ 624 MiB (39 %) weniger** als `int8`
  und **56 MiB (6 %) weniger** als das heutige Fenster.
- Expert-Cache: 416 Slots je 568 MiB (Tab. 4) → **≈ 457 Slots mehr** als `int8`.
- Der Gewinn ist Speicher und Slots, kein Speed: Prefill muss in der 10-%-Grenze
  bleiben (K5), Decode liegt heute in der Streubreite.

### Schon tot für dieses Ziel

Nicht noch einmal messen: Clip nach FWHT (WP §7.11), Hadamard an/aus als
Qualitätsarm (WP §7.9), K-vs-V-Bit-Story (t +0,05), 3 bit ohne Fenster
(+51 % KL, t −3,60), R=8, Nadel (gesättigt, keine Trennung), PPL (dreht die
Rangfolge um). cwe-2k „erreicht `int8`-Niveau" (WP §7.16) ist kein 4K- und kein
128K-Gewinn. **Neu hier:** ein größerer fp16-Ring als Weg zu `int8` — bei 4K
Gleichstand mit 256/64 (t +0,02), bei 128K nie qualitativ gemessen, und
arithmetisch bei 5,0 auf 4.096 Zellen begrenzt (oben).

### Leiter S0–S5

Jede Stufe hat eine Frage und einen Kill. Die nächste Stufe startet nur, wenn
die vorherige nicht getötet hat. Das Risiko ist nicht „der Arm könnte
scheitern" — Scheitern ist der gewünschte Ausgang —, sondern „Engine-Default
und `int8`-SHA haben sich bewegt". Jede Stufe ist opt-in.

| Stufe | Frage | Prüfung | Kill | Kosten |
|---|---|---|---|---|
| **S0** | Ist die Latte selbst stabil? | Ein vorhandenes cwe-4k-Paar (`int8`, Fenster 256/64, plain) zusätzlich zur heutigen KL unter einer zweiten Definition rescoren (top-256 oder volles Vokabular), **auf denselben Positionen**, Vorzeichen mitberichten. Das Histogramm derselben Positionen ist E4, kein eigener Lauf | Rangfolge oder Vorzeichen drehen → kein Kernel, bis 128K unter derselben Definition neu gelesen ist; K1/K2 sind dann undefiniert | **Nicht scorer-only:** die per-Position-Logs (top-64) liegen in diesem Tree und auf `KV_QUANT_research` nicht vor (dort nur `compare.json`/`paired.json`, Segment-Aggregate). Ein 4K-Lauf je Arm mit `STRATA_LOGPOS_TOPK=256` (3 Arme), Auswertung offline |
| **S1** | Sitzt das fp16-Geld auf Zellen, die der Indexer liest? | C4 (statisch, dieser Branch), dann ein 128K-Prefill-Zähler: Anteil der Indexer-Auswahl in R, in S, in keinem von beiden, je QSA-Layer. Dazu E4: ist der 4K-Restfehler dicht oder dünn? | < ½ der Treffer in R+S → das positionsfeste Overlay ist der falsche Ausgabeort; **B1 läuft nicht**, und E3 (auswahlfolgendes Overlay) wird der Kandidat — unter derselben Regel „eine versprochene exakte Zelle, die den Pool verlassen hat, ist ein Fehler". Dichter Rest → kein GEAR-Kernel | Ein Zählerlauf, kein neues Format |
| **S2** | Trägt die Kanalachse **vor** der Rotation? | Zweite Momente der K-Kanäle vor und nach FWHT, getrennt für die 64 rotierten und die 192 unrotierten Dimensionen, 12 QSA-Layer. Keine Quantisierung ändern | Vor der Rotation kein schwererer Tail als danach → A1 nicht bauen; KIVIs Achse ist in dieser Pipeline nicht da | Debug-Statistik hinter einem Env-Flag, `int8` unberührt |
| **S3** | Reicht das billige Bulk-Werkzeug? | cwe-4k, gepaart gegen `int8`, in dieser Reihenfolge: (a) 16/16 gegen 8/8 im vorhandenen Format (A2), (b) alternative 16-Level-Tabelle (A5), (c) nur wenn S2 überlebt: per-channel K vor FWHT (A1). Jeder Arm berichtet K1, K4 und bit/cell | Kein Arm erreicht K1 bei ≤ 6,0 bit (der weicheren Latte 2, also erst recht nicht bei 5,0) → der Bulk ist nicht der Hebel; **kein 128K**, zurück zu S1 (Overlay-Ort) oder Abbruch. Ein Gleichstand zählt als Kill | (a)/(b) kein neuer Kernel; (c) opt-in Pipeline-Änderung mit SHA-Gate |
| **S4** | Welche 5,0-Komposition hält? | Nur der S3-Sieger als Bulk, drei Byte-Varianten bei 128K, beide Texte, alle K-Statistiken, Byte-Modell vorher: (a) fp16-Ring, R+S+B=4.096 (B=3.776); (b) int8-Ring, R+S+B=7.943 (B=7.623); (c) 12-bit-Ring, R+S+B=5.461 (B=5.141) | Keine Variante erfüllt K1–K5 → Latte 1 ist nicht erreichbar; dann Latte 2 prüfen (Parität bei 6,0) oder negatives Ergebnis. Prefill unter 90 % von `int8` → Variante tot, auch wenn die Qualität passt | (a) nur Konfiguration; (b)/(c) neuer Ring-Tier im Schreiber/Leser, eigener Härtungspunkt |
| **S5** | Ist der Gewinn nicht cwe-spezifisch? | Nur nach einem K1–K6-Gewinn: zweiter RULER-Task bei 4K (D1), dieselben Kriterien | Verliert dort → kein Auslieferungs-Claim; der Arm bleibt Forschungsarm | Ein Task, nicht LongBench |

**Stand 2026-10-25, gemessen** (`bench/results/2026-10-25-s0-kl-support/`,
`…-s1-selstats/`, `…-e4-error-shape/`). S0: Rangfolge und Vorzeichen halten unter
top-64 und top-256, der Kill feuert nicht. E4: der Rest ist dünn (schlimmstes 1 %
trägt 71–86 % der KL) und ein Kopf-Phänomen (88–90 % sind Ränge 1–8); GEAR ist
nicht getötet und nicht gestartet. S1 lief bei 32K, nicht 128K. Die 50-%-Schwelle
ist an diesem Punkt arithmetisch unerreichbar (Breite 2.051, R+S = 320, Decke
15,6 %). Gemessen bei `n_kv` > 16.384: 11,7 % der Treffer werden exakt gelesen,
Recall von S 86 %, von R 29 %. 98,9 % der Treffer im bezahlten Ring haben noch
ihren Tag; ein Tag-Read würde 70,2 % der Selektion exakt lesen, bei denselben
Bytes. **B1 läuft nicht, E3 in Fenstergröße auch nicht** — beide sind von diesem
Read dominiert. Der nächste Arm ist der Read selbst, opt-in, Default unverändert,
Qualität auf cwe-4k gegen das heutige Fenster und gegen `int8`. Kein
Qualitätsanspruch aus dem Zähler. Der Score-Rand (0,01 % der Spreizung) trägt
keinen Score-Pool.

S2 lief am 2026-10-27 auf cwe-4k (32.746 Token je QSA-Layer, 12 Layer, batched Prefill,
`bench/results/2026-10-27-s2-channels/`). Die Kanalachse **vor** der Rotation ist nicht schwerer als
danach: Exzess-Kurtosis im Mittel −0,18 pre gegen −0,14 post, Crest 3,72 gegen 3,93, und die
Vorzeichenprobe ist ein Münzwurf (52,5 % der 3.072 Kanäle). **Der Kill feuert, A1 wird nicht
gebaut.** Die Ausnahme sind 70 Kanäle (2,3 %, 49 davon in den 64 rotierten Dimensionen) mit
Kurtosis > +1 vor der Rotation; der Hadamard flacht alle davon ab (post ≤ +0,46). Die
Breiten-Spreizung von 16× bis 7.267× ist der Norm-Gewichtvektor (tote Kanäle mit
`var` = 0,001), also eine bekannte feste Transformation, keine Datenachse. S3 startet damit bei (a)
16/16 gegen 8/8 und (b) alternative 16-Level-Tabelle; (c) entfällt.

S3(a) lief am 2026-10-27/28 (`bench/results/2026-10-27-s3-level-split/`): neun Arme auf einem
Build, 32.746 Positionen je Arm, gepaart je Position. Nachgerechnet gegen einen `fp16`-Arm auf
demselben Build — die Einheit, in der auch der `int8`-Anker 0,01688 und die Fensterzahlen von
2026-10-25 stehen; `KL(int8‖Arm)` ist eine andere Größe und bestraft Fehler mit, die `int8` selbst
macht. **`int8` 0,01688 (top-64), `q4lm32`+Fenster 0,01689, `q4lm32s`+Fenster 0,01881,
`q4lm8v`+Fenster 0,02090, `q4lm32s` 0,02420, `q4lm32` 0,02512, `q4lm8v` 0,03182, `q4lm8k` 0,03212,
`q4lm8` 0,04127.** Der Build reproduziert den Altbestand auf fünf Stellen (`int8` 0,01688, `q4lm8v`
0,03182), deshalb ist der Vergleich mit dem Fensterarm von 2026-10-25 zulässig.

Drei Befunde. (1) Der Split trägt nicht: 16/8 gegen 8/16 ist −0,00030 [−0,00357, +0,00297], also
**keine Richtung** — K und V sind gleich empfindlich (0,0075 gegen 0,0072), und `q4lm8v` verschenkt
0,0072 nats an die V-Seite ohne Gegenleistung. (2) Die Least-Squares-Skala ist **kein Hebel**:
+0,00033 gegen `max|x|/87` (t 0,21). Die frühere Zahl 0,0028 war ein Artefakt des `int8`-Ankers.
(3) Der 5,29-bit-Punkt mit 16/16-Bulk ist von `int8` nicht zu unterscheiden (+0,00002 top-64,
+0,00034 top-256, `int8` führt in 4/8 Segmenten) und **0,0040 nats besser als der heutige
Produktpunkt** bei identischen Bytes ([−0,00663, −0,00138], t −2,99, besser in 7/8 Segmenten). Der
Read bleibt kein Hebel: Tag-Read gegen Fenster +0,00117 [−0,00162, +0,00395], t = 0,82.

K1 als Regel ist bei diesem Korpus nicht erreichbar: die gepaarte Streuung (0,22 nats) ergibt bei
32.746 Positionen eine einseitige Grenze von 0,0020 selbst bei echter Parität; die Schwelle
< 0,001 bräuchte rund 131.000 Positionen, also den 128K-Satz, den es nicht gibt. Der beste Arm
liegt bei 0,0025 — eine Auflösungsgrenze, kein Urteil über den Arm.

**Zielwechsel 2026-10-27.** Das 5,0-bit/cell-Ziel ist **zurückgestellt**: ohne deutlich
bessere Rotationsansätze und Fehlerkorrekturen ist es mit diesem Codec nicht
erreichbar. Der Branch maximiert ab jetzt **Qualität bei gleichen Bytes** und erst
danach Bytes. Die Latte 1 (K1-K6) bleibt als Definition stehen, ist aber keine
Bedingung mehr für den nächsten Schritt; S4 und S5 (die 128K-Kompositionen auf 5,0)
laufen nicht, solange das Ziel zurückgestellt ist. Der gemessene Hebel bei gleichen
Bytes ist die Bulk-Tabelle (0,0072 bis 0,0175 nats; der Produktpunkt holt davon 0,0040),
nicht der Read (0,0012, nicht signifikant) und nicht die Kanalachse (S2, tot).

S0 brauchte einen 4K-Lauf je Arm (die Logs liegen jetzt vor), S1 einen
Zählerlauf (gelaufen, 32K); S3 kostet 4K-GPU-Zeit für höchstens drei
Arme; S4 ist der einzige 128K-Block (drei Varianten × zwei Texte). Wenn S3 keinen
Sieger hat, endet der Branch mit einem dokumentierten Negativ-Ergebnis — das ist
ein valides Ergebnis und billiger als ein 128K-Sweep.

### Nicht auf dieser Leiter

E2 (`q4_0` auf cwe) ist ein Vergleich, keine Hypothese, die gewinnen kann — es
ordnet den Punkt ein, es bewegt ihn nicht. C1/C2 sind Review-Gates: für Latte 1
(Default-Anspruch) blockierend, für Latte 2 (opt-in) nicht. D2/D3/D4/D6
begrenzen die Aussage, nicht den Punkt. B1 läuft nur, wenn S1 überlebt; B2/B3,
A3, A4, E3 und ein GEAR-Kernel hängen an ihren Gates (E3 an S1, GEAR an der
Restfehler-Form aus S1/E4). A4 mischt höchstens `int8` mit etwas Schlechterem
und kann `int8` damit nicht überholen.

## 8. Was an Qwen3.8-Flash-Next überhaupt ansetzt

„Modelagnostisch" heißt hier vier Dinge gleichzeitig: kein Neutraining, kein Kalibrier-Set, die Indexer-Auswahl bleibt dieselbe, der `int8`-Pfad bleibt byte-identisch. Eine Methode darf im Survey stehen und trotzdem an dieser Architektur scheitern. Die Zahlen unten sind die kompilierte Geometrie (`include/strata/core/geometry.hpp`, `qsa.hpp`), keine Messung.

| Tatsache | Folge |
|---|---|
| 48 Layer, davon 12 QSA (3, 7, …, 47) und 36 GDN | Jede KV-Methode gilt nur für die 12 Cache-Layer. GDN hat einen festen SSM-Zustand, keinen Token-Cache. „Untere Layer brauchen mehr Bits" ist an dichten Attention-Stapeln gemessen und überträgt sich nicht als Konstante |
| Indexer: 4 Köpfe × 128, Block 4, Budget 2.048 (+3 für den unvollständigen Block). Schlüssel sind **FP32 per Vertrag**, weil eine andere Auswahl eine andere Attention ist | Quantisieren der Indexer-Keys oder Ersetzen der Auswahl ist kein Cache-Format. Es ändert das Modell |
| 2 KV-Köpfe, 24 Query-Köpfe, head_dim 256 | Methoden, die Köpfe in „Retrieval" und „Streaming" teilen, haben fast keine Freiheitsgrade. Ein Kopf weg ist die halbe Cache |
| `n_rot` = 64 von 256, und 64 von 128 auf dem Indexer-Key | Pre-RoPE-Rezepte, die den ganzen Head vor RoPE quantisieren oder rotieren, beschreiben eine andere Maschine. 192 Dimensionen sind nicht rotiert |
| Seite 4 je Indexer-Block, Streaming verweigert das Fenster (A2) | Ein Offload-Cache und ein residenter Exact-Pool sind derselbe Konflikt wie heute |

### Setzt an, ohne das Modell anzufassen

Das sind die einzigen Survey-Ideen, die als opt-in auf dem QSA-K/V-Cache baubar sind.

- **Recent- plus Sink-Overlay auf einem 4-bit-Bulk.** Das ist der gemeinsame Kern von KIVI, SKVQ, IntactKV, StreamingLLM und RotateKVs Sink-Schutz, und es ist gebaut. Die kalibrierten Zutaten derselben Papiere (Kanal-Umordnung, gelernte Rotation, Pivot-Tokens als Parameter) sind nicht agnostisch und bleiben draußen.
- **Per-channel vor FWHT, getrennt nach rotiert und nicht rotiert.** Das ist die Form von KVQuant, die hier noch Sinn hat. Nicht deren Pre-RoPE-Rezept, und nicht nach der Rotation: nach FWHT gibt es keine Kanäle mehr (WP §5.1, statisch). Die S2-Sonde muss die Momente für Dimensionen 0–63 und 64–255 getrennt berichten. Das ist die Hypothese, die keines der Papiere so stellt.
- **16/16 gegen 8/8 im vorhandenen Format.** Agnostisch, Kernel steht. Die Bit-Richtung „Keys wollen mehr Bits" (`spectralkv`) ist auf Swift nicht getragen. Die Level-Frage ist S3(a) der Leiter, nicht ein neues Papier.
- **Präzision folgt der Auswahl, nicht der Position.** Der top-k-Artikel sagt Token-Budget und Bits zusammen. Hier ist das Token-Budget der Indexer, und der darf nicht ersetzt werden. Der agnostische Schnitt ist: jede Zelle bleibt gespeichert, fp16 nur für Zellen, die der Indexer tatsächlich geliefert hat. Die Auswahl wechselt pro Query, also ist der Pool eine Vereinigung oder die letzte Auswahl, kein stabiles 2.048er-Fenster. Eine versprochene exakte Zelle, die nicht mehr im Pool liegt, ist der Fehler, den die A-Serie verbietet. Erst nach E1. Wenn R+S schon die gelesenen Zellen deckt, ist diese Methode tot, bevor sie gebaut wird.
- **Alternatives 4-bit-Format ohne Kalibrierung** (token-weises INT4 plus kleine Hadamard, wie SAW-INT4). Agnostisch, aber dieselbe Familie wie Lloyd plus FWHT. Die an/aus-Ablation war bei 2K/4K ein Gleichstand. Kein Grund, das vor S3 zu bauen.

### Sieht agnostisch aus, passt hier nicht

- **DuoAttention, RazorAttention.** Zwei KV-Köpfe. Ein Streaming-Kopf wäre die halbe Cache. Die Papiere rechnen mit vielen KV-Köpfen.
- **Quest, NSA, jede Eviction (H2O, SnapKV, PyramidKV, Scissorhands, ChunkKV).** Ersetzt oder löscht, was der Indexer später auswählen kann. Eine ausgewählte, nicht mehr vorhandene Zelle ist eine stille falsche Attention. NSA ist obendrein trainiert.
- **ShadowKV.** Die dünne Rekonstruktion doppelt den Indexer. Der Offload kollidiert mit dem residenten Pool (A2).
- **Palu, SpinQuant, IntactKV-als-Parameter, KVQuant-Kalibrierung, PM-KVQ-Kalibrierung, die 2026er Low-Rank-Adaption.** Projektionen umschreiben, Rotationen lernen oder an einem Set kalibrieren. Das ist ein anderes Modell, kein Cache-Flag.
- **Pre-RoPE auf dem ganzen Head** (KVQuant, RotateKVs Grouped-Head-Rotation vor RoPE). `n_rot` ist 64, nicht 256. Ein Port muss die unrotierten 192 Dimensionen einzeln behandeln oder er misst ein anderes Fehlermodell.
- **FP8, QServes Serving-Stack.** Kein FP8-Attention-Kernel auf der Zielkarte. Das KV4-Format haben wir der Idee nach schon.
- **Layer-Pyramide als Übernahme der publizierten Rangfolge.** 12 QSA-Layer, dazwischen GDN. Eine eigene 12er-Suche wäre eine neue Messung, kein Port, und ohne per-Layer-Fehlersignal unbegründet.

### Was das an der Leiter ändert

S1 (E1) ist nicht optional. Wenn Sinks und Recent nicht in der Auswahl liegen, ist der Survey-Kern, den wir schon gebaut haben, auf diesem Modell ein No-op, auch wenn er modelagnostisch korrekt ist. S2 berichtet die zwei RoPE-Scheiben getrennt. S3(c), falls sie lebt, ist per-channel vor FWHT auf diesen zwei Scheiben, nicht ein Pre-RoPE-Port. DuoAttention, NSA, Palu, Eviction und FP8 kommen nicht auf die Leiter.
