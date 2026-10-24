# Ansatz-Verbesserungen: KV-Cache-Quantisierung & Exact-Cell-Fenster

Stand: 2026-10-24 · **Status: Arbeitsdokument (uncommitted).** Autorenkontext:
GLM-5.3 (opencode), auf Basis des Papers `docs/paper/KV-Window-Hardening.tex`
(8 Seiten) und des Peer-Reviews `PEER_REVIEW.md` (Review durch dasselbe Modell —
Selbst-Review-Caveat gilt).

**Tabellenzählung im Paper (Stand PDF 2026-10-24):** Tab. 1 Survey,
Tab. 2 Footprint, Tab. 3 Decisions (A-Serie), Tab. 4 Runtime 128K,
Tab. 5 cwe-2k, Tab. 6 cwe-4k, Tab. 7 cwe-128k (Tail), Tab. 8 Regionen 128K.
Am 2026-10-24 wurden alle Zahlen der Tabellen 4–8 Zeile für Zeile gegen
`docs/KV_QUANT_WORKPACKAGE.md` (Branch `KV_QUANT_research`) geprüft und die
Captions von Tab. 7/8 präzisiert.

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
| Gap-Closure | 73 % (4K), 58 % (128K-Tail) | Paper Tab. 6, Tab. 7 |
| 2K-Lage | gegen `int8` nicht trennscharf (t=−1,92) | Paper Tab. 5 |
| 4K-Lage | `int8` signifikant besser (t=−3,55) im Mittel; Antwortregion nicht trennscharf | Paper Tab. 6 |
| 128K-Lage | `int8` gewinnt Tail-p99 (0.0166 vs 0.0298), letzte 1.024 Positionen und Ganzsegment; Ganzsegment-p99 liegt bei allen Armen hoch (3.66–5.55, isolierte Hartpositionen) | Paper Tab. 7/8 |
| 256/64 vs 512/128 | 2K/4K/128K qualitativ ununterscheidbar; 512/128 kostet +8–10 MiB, decode −0.85 t/s; ihr Prefill-Vorteil reproduzierte sich nicht → 256/64 | Paper Tab. 4, 5–7 |
| Level-Layout | K 16 uniforme Lloyd-Levels (4 bit), V 8 Levels | Paper §4 |
| Laufzeit | Prefill gleich, Decode −1.0 t/s (Streubreite); 568 MiB frei, 416 Slots mehr | Paper Tab. 4 |
| Korrektheits-Gates | SHA 8/8; Parität 12/12; Nadel 18/18 (gesättigt); Setup 59/59 | Paper „Correctness gates" |
| Hang | 1× auf *plain* `q4lm8v` @128K-Prefill, 6/6 Re-Runs ohne Reproduktion | Paper §9 (Threats) |
| Offen | A4-Paritätstest; CUDA/SYCL-Builds; Knie-Dichte; Task-Breite; Community-Anker-Arm | Paper §8.2, Tab. 3 |

## 2. Katalog

Priorität: **P1** = nächster sinnvoller Zug (hoher Informationsgewinn pro
Aufwand), **P2** = wichtig, danach, **P3** = Warteschlange.
Aufwand: **S** ≤ 1 Tag, **M** = mehrere Tage, **L** = Woche(n).

### A. Quantisierungslayout

| ID | Titel | Motivation (aus Status quo / Review) | Vorschlag | Prüfmessung | Aufwand | Prio |
|---|---|---|---|---|---|---|
| A1 | **KIVI-artiger Per-channel-K-Arm** | Review M3: Der 5.29-bit-Punkt hängt ohne externen Anker in der Luft; KIVI/KVQuant zeigen, dass K entlang der Kanalachse robuster zu quantisieren ist | Arm, der K entlang der Kanalachse quantisiert (V bleibt wie heute), auf derselben Hadamard-Rotation | cwe-2k/4k/128k mit demselben Harness; Akzeptanz: Per-channel ≈ oder besser als Per-cell bei gleicher bit/cell | **M** | **P1** |
| A2 | Level-Symmetrie-Check (16/16, 8/8) | Review Q1: Es ist ungeprüft, ob der heutige asymmetrische Split — K 16 Levels, V 8 Levels (Paper §4) — wirklich die richtige Verteilung ist. KIVI sagt die Richtung voraus (K braucht mehr Levels), aber es fehlt der symmetrische Gegenarm | Zwei Arme: (i) 16/16 (auch V auf 16 Levels; kostet den V-Anteil mehr Bytes/cell), (ii) 8/8 (auch K auf 8 Levels; spart Bytes) | Teacher-forced KL wie oben; erwartet: 16/16 ≈ heute, 8/8 messbar schlechter — würde die heutige Verteilung aus Strata-Daten belegen statt aus der Theorie abzuleiten | **S** | **P1** |
| A3 | Pre-RoPE-K-Quantisierung | KVQuant zeigt signifikanten Gewinn von Quantisierung vor RoPE; Strata quantisiert vor dem Hadamard-Spread, aber das Verhältnis zur RoPE-Reihenfolge ist unausprobiert (**statische Analyse**) | Kernel-Layout prüfen und Prototyp-Arm bauen, der K vor RoPE quantisiert | cwe-4k-KL gegen das heutige Layout; Schwellwert: ≤ heutige KL | **M** | **P2** |
| A4 | Layer-wise-Präzisionspaare | KVTuner: „nahezu verlustfrei" 3.25–4.0 bit mit Layer-Paaren; Strata-Format ist global pro Cache (Paper §4) | Prototyp: 2–3 kritische Layer auf höherer Präzision, Rest auf heute; Konfigdatei + Setup-Erklärung | cwe-Suite; Gate: byte-identischer `int8`-Pfad bleibt unberührt | **L** | **P3** |
| A5 | Non-uniform Level für K | KVQuant: Sensitivitäts-gewichtete Datentypen schlagen uniforme Lloyd-Level | alternatives Level-Tab-Format (Schema + 16-Level-Tabelle) | cwe-4k/128k gegen das uniforme Lloyd-Layout | **S** | **P2** |

### B. Fenstergeometrie & Kosten

| ID | Titel | Motivation | Vorschlag | Prüfmessung | Aufwand | Prio |
|---|---|---|---|---|---|---|
| B1 | Knie-Sweep für S/R bei 128K | Paper §8.2: 256/64 ist der *billigste getestete* Arm mit `int8`-unkritischer Tail-Qualität, nicht das gemessene Optimum; bei 2K lag der Kniepunkt auf S≈512 (dort dokumentierter Knie-Sweep: `strata-wp`) | S∈{128, 192, 256, 384, 512} × R∈{64, 128} bei 128K | cwe-128k-KL plus Laufzeit (Paper Tab. 4-Methodik); Ziel: kleinsten Arm konfirmieren oder einen besseren finden | **M** | **P1** |
| B2 | ≤3-Bit-Overlay bei 128K | `strata-wp`: dokumentierter Cross-over ≈ 4,8–5,3 bit auf dem Swift-2K-Kontextgitter (Paper §8.4); 128K-Validierung fehlt | Arm mit 3-bit-Overlay hinter dem 4-bit-Basisbetrieb bei 128K | cwe-128k + Laufzeit-A/B; Gate: ≤ heutige KL oder Memory-Bonus | **M** | **P2** |
| B3 | Streaming-kompatibler Sink-Pool | Paper §4: R-Zellen bleiben nahe Position 0; ein getrimmter side pool könnte mit einem gestreamten Cache kombinieren, ohne das Silent-Failure-Risiko zu vergrößern | Prototyp: *Sink-only*-Fenster (R ohne S-Ring) unter Streaming; A2-Lockerung als separater Flag mit lauter Doku | Nadel-Tests 32K/128K + Parität; zwingend: Verhalten „laut ablehnen" statt „leise quantisiert" | **M** | **P3** |
| B4 | `B`-Term-Kosten | Paper §3 (Byte fraction): 0.75 der 0.79 hinzugefügten bit/cell am 128K-Punkt sind der Ringanteil über B=6.144; B=512 ist billiger (4.60 bit, 884 MiB), aber Prefill ≈5× langsamer (105 t/s, Paper Tab. 4) → ein offline/batch-Punkt. Das Byte-Modell für eine kleinere Ring-Größe mit gleicher Latenz ist die Lücke (**statische Analyse**) | Option: die fp16-Kopie nur für den aktuellen Chunk resident halten und nach der Attention wieder freigeben, sodass mehrere kleinere Chunks vorprefillt werden; vorher das Byte-Modell neu rechnen | Byte-Modell neu rechnen, dann Same-Day A/B gegen Paper Tab. 4 | **M** | **P3** |

### C. Systemintegration & Korrektheit

| ID | Titel | Motivation | Vorschlag | Prüfung | Aufwand | Prio |
|---|---|---|---|---|---|---|
| C1 | **A4: Volle Zell→Zeilen-Parität** | Offener Review-Gate: ein form-spezifisches Fehllesen bliebe stumm | Dump/Re-Read-Test über alle Shapes (Ring-Wrap, veraltete Tags, Sinks, Batch-Grenzen, Leer) | Parität: bitweise, jede Shape | **S** | **P1** |
| C2 | CUDA/SYCL Build-Coverage | Bisher nur geplante Bauten, keine Messung: SYCL bietet `q4lm8v` ohnehin nicht an, aber der Code soll trotzdem bauen | `nvcc` und `dpcpp` kompilieren + Parität-Suiten | beide Backends grün; `q4lm8v` bleibt SYCL-off | **M** | **P1** |
| C3 | Hang-Untersuchung (plain 4-bit, 128K-Prefill) | Paper §9 (Threats): 1× nicht reproduzierbarer Hang auf dem *plain*-Arm; 6/6 Re-Runs ohne Reproduktion | Wiederholte 128K-Prefill-Zyklen auf dem plain-Arm mit Logging; nichts übernehmen, solange der Fund nicht reproduziert | Fund als Bug-Report; keine Aussage im Paper | **S** | **P2** |

### D. Evaluations-Decke

| ID | Titel | Motivation | Vorschlag | Prüfmessung | Aufwand | Prio |
|---|---|---|---|---|---|---|
| D1 | **Zweiter RULER-Task** | Review M2: Nur cwe gemessen | z. B. `niah_multikey` oder `variable_tracking` bei 4K, im Harness | gleiche Segment-/Scoring-Struktur wie heute; Gate: keine qualitative Abweichung von der Heut-Lesung | **S** | **P1** |
| D2 | Mittlere Region bei 128K | Review Q4: Tail-Scoring (Paper Tab. 7) übersieht innenliegende Degradation; der Report listet Hartpositionen auch im Inneren (bei 3.033: KL 17.3, bei 18.995: 15.8) | Score-Region um Position 64K im cwe-128k-Scoring | Dasselbe per-Position-Log; **nur Offline-Schnitt** | **S** | **P1** |
| D3 | Multi-Turn-A/B | Review Q3: Das Fenster ist für „Chat, der wächst" gebaut, aber nie als solches gemessen (Paper §8.3) | 8–16 lange Prompts; danach Follow-up-Runden, jede +512 Anhänge-Tokens; Fenster gegen `int8` | Prefill/Decode wie Paper Tab. 4 + KL der Antwortregion | **M** | **P2** |
| D4 | p99-Bootstrap-CI | Review Minor 5: p99 ohne Konfidenzband (Paper §6, Metrics-Notiz) | Bootstrap (10k Resamples) über die vorhandenen Per-Position-Logs | CI-Intervalle in den Tabellen; **offline, keine GPU** | **S** | **P1** |
| D5 | LongBench-Subset | D1 nachrangig, breiteres Taskfeld | Zwei LongBench-Subsets bei 4K/8K; konkrete Auswahl erst nach D1 | gegen `int8` gemessen, gleiche Methodik | **M** | **P3** |
| D6 | Reverse-KL-Sicht | Review M4: KL-Richtung könnte das Ranking ändern; im Paper ausdrücklich offen gelassen (Metrics) | Offline-Doppelbewertung D(arm‖ref) auf vorhandenen Logs | gepaarte Differenz zwischen den Richtungen berichten | **S** | **P2** |

## 3. Vorschlag zur Arbeitspaket-Gliederung („WP-3")

Die obenstehenden P1-Punkte ergeben ein sinnvolles nächstes Arbeitspaket
in der Reihenfolge, in der sie voneinander abhängen:

1. **WP-3a — Evaluations-Absicherung (keine GPU nötig für 2 der 3):**
   D4 (p99-CI) → D2 (Mittelkontext) → D1 (zweiter Task, GPU S).
   *Ergebnis:* Robustere Berichtszahlen + erste Task-Breite.
2. **WP-3b — Community-Anker:**
   A1 (per-channel-K-Arm) + A2 (Level-Symmetrie).
   *Ergebnis:* Der Punkt landet erstmals auf einer externen Vergleichs-
   kurve; die K-/V-Level-Verteilung wird an Strata-Daten belegt oder
   bestätigt.
3. **WP-3c — Geometrie-Feinschliff:** B1 (Knie-Sweep 128K).
   *Ergebnis:* konfirmierter oder korrigierter Production-Point.
4. **WP-3d — Engineering:** C1 (A4-Parität) → C2 (CUDA/SYCL).
   *Ergebnis:* Die zwei offenen Review-Gates des Papers schließen.
5. **WP-3e (optional, GPU-Schleife):** D3 (Multi-Turn), B2 (≤3-Bit-Overlay
   bei 128K) — nur wenn ein neues Speicher-Ziel (z. B. kleinere GPUs) es
   verlangt.

**Gates für jede Stufe:**

- Standard-Parität + Byteidentitäts-SHA auf `int8` unverändert.
- Messberichts-Zeilen in `bench/results/<datum>-<arm>/` mit Anzahl
  gemessener Positionen, Rounds, Layout.
- Update in `docs/DETAILS.md` und hier: Status-Wechsel P1 → erledigt.

## 4. Bewusste Nicht-Ziele

- **Eviction** (SnapKV/H2O/PyramidKV-artig) bleibt draußen: das Fenster ist
  quantisierungsseitig konzipiert und hält jede Zelle adressierbar (die
  A-Serie behandelt eine still nicht mehr verfügbare Zelle als Fehler).
- **FP8**: Auf Ziel-Hardware (Consumer-AMD über HIP) hat kein FP8-
  Attention-Kernel — siehe Paper §1; vLLM-FP8 ist ein Datacenter-Pfad.
- **Keine Default-Änderung**: `int8` bleibt Standard, bis eine der obigen
  Messungen eine signifikante Überlegenheit belegt **und** alle
  Korrektheits-Gates (SHA-Byteidentität, Parität, Nadel, Setup-Suite) grün
  sind.

## 5. Quellen

Paper-Messbasis: `docs/DETAILS.md` (§ „KV cache quantization"),
`docs/KV_QUANT_WORKPACKAGE.md` (Branch `KV_QUANT_research`, die 128K-Tabelle
und -Beschreibungen in §7.20–§7.21), `bench/results/2026-09-27-kv-q4/`,
`bench/results/2026-10-17-kv-wp2a-window/`, `bench/results/2026-10-18-kv-runtime-ab/`,
`bench/results/2026-10-17-kv-wp2b-window3bit/`, `bench/results/2026-10-18-kv-wp2b-oscar-knee/`.
Literaturverweise: `PEER_REVIEW.md` §6 (alle arXiv-IDs verifiziert).
**Aufwand und Prioritäten sind Schätzungen (statische Analyse), keine Messwerte.**
