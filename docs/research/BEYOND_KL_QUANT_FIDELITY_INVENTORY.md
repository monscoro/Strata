# Inventar: was über die Metrikfrage gemessen und geprüft ist

**Status: Inventar (2026-10-28). Keine eigene Messung; jede Zahl ist zitiert oder als statische
Analyse markiert.** Dieses Dokument ist die Belegbasis der Metrikfrage: wie heute gemessen wird, was
die Messungen gezeigt haben (M1–M13), welche Gegenbeispiele tragen, welche Merkmale in Frage kommen
und welche Prämissen des Auftrags der Code nicht hergibt.

Es ist **kein Arbeitsplan**. Der Arbeitsplan ist `BEYOND_KL_QUANT_FIDELITY.md` (Stufen G0–G4); die
Begründung, warum er kürzer ist als der ursprüngliche Plan, steht in
`BEYOND_KL_QUANT_FIDELITY_ASSESSMENT.md`. Der Auftrag selbst steht wörtlich in
`BEYOND_KL_QUANT_FIDELITY_ASSIGNMENT.md`.

**Herkunft.** Dieses Inventar ist der Sachteil des früheren `…_PLAN.md`, der am 2026-10-28 geteilt
wurde: der Arbeitsteil (F0–F6, Experimentdesign, QBF, Deliverables) ist in den Arbeitsplan
übergegangen und dort ersetzt; der Sachteil steht hier unverändert. Die **Abschnittsnummern sind die
des alten Plans** (1–5, 8, 9) und bleiben stehen, damit die Querverweise (§1.5, §9) gültig bleiben.
Wo im Text noch von „F0“, „F1“ usw. die Rede ist, sind das die Stufen des alten Plans; die Zuordnung
zum heutigen Plan:

| alter Plan | heutiger Plan |
|---|---|
| F0 Katalog, F1 Offline-Merkmale | G1 |
| F2 Ground Truth | G3 |
| F3 Prädiktion | G4 |
| F4 QSA-Dimension | gestrichen bis zur Instrumentierung |
| F5 Fast Mode, F6 Bericht | offen, erst nach G4 |

Der Vollvokabular-Anker (alter Plan: Engine-Arbeit, später) ist im heutigen Plan **G2** und damit
Voraussetzung statt Nacharbeit.

## 1. Ist-Zustand: womit heute gemessen wird (statische Analyse)

### 1.1 Der Harness

`tools/quant_quality.py` ist die Yardstick-Instanz (teacher-forced, `GEN 1 <ids>`, fester Text):

| Größe | Definition | Fundstelle |
|---|---|---|
| KL | `KL(Ref ‖ Arm)` über den **top-64** Tokens der Referenz, beide Seiten renormiert, fehlende Tokens bekommen den 64. Logprob der Arm als Floor | `kl_between`, `tools/quant_quality.py:596-605`; `TOPK=64` Zeile 66 |
| Mean KL | Mittel über alle Positionen, gepoolt über Segmente | `compare`, `:608-642` |
| KL p99 | 99. Perzentil der **gepoolten** positionsweisen KL (nicht pro Segment) | `:640` |
| Top-1-Agreement | Anteil Positionen mit gleichem Argmax wie die Referenz (keine Genauigkeit) | `:621` |
| Perplexity | `exp(mean NLL)` des Referenztextes, teacher-forced | `:631,641` |
| Volle Vokabular-KL | nur `pcompare`, **eine Position pro Prompt** (nach Prompt-Ende) | `:549-574`, `STRATA_DUMP_FIRST_LOGITS` |

Engine-seitig schreibt `Verifier::window_logprobs` (`src/core/verify.cpp:3190-3258`): Position,
Target, `logp(target)` und `logp(argmax)` **voll-vokabular-normiert** (Log-Sum-Exp über alle
248.320 Tokens, `include/strata/core/geometry.hpp:27`), Trefferflag, Extra-Logprob, dazu die
Top-k-Paare `id:logprob`. Die Paare fallen nur an, wenn `STRATA_LOGPOS_TOPK` gesetzt ist (Default
aus, `verify.cpp:3240-3253`); `quant_quality.py:247` setzt ihn je Lauf, und S0/S3 liefen mit
`--topk 256` (S0-Bericht Z. 37/47). Der Deckel für k ist hart auf **256** gesezt
(`verify.cpp:3244`).

**Wichtige Folge (statisch):** Weil die geschriebenen Logprobs bereits über das volle Vokabular
normiert sind, ist — wo die Top-k-Paare vorliegen, also in allen quant_quality-Läufen — die
**Tail-Masse** `r = 1 − Σ_{i≤k} exp(logp_i)` offline rekonstruierbar, auch bei k=256. Die Entropie
ist damit offline **bandbegrenzt** berechenbar, ohne Engine-Änderung: mit
`H_k = −Σ_{i≤k} p_i·ln p_i` und der Tail-Masse gilt `H ∈ [H_k , H_k + r·ln(V−k)]`
(untere Schranke: der Tail steckt konzentriert in einem Token; obere: er ist gleichverteilt
über die `V−k` übrigen); der **Mass-Shift** `Δr` zwischen zwei Armen ist aus derselben Größe
sogar exakt. Volle Vokabular-KL dagegen nicht: dafür fehlen die
Arm-Tail-Werte; der Top-k-Deckel von 256 ist eine spätere, getaktete Engine-Arbeit.

### 1.2 Weitere Werkzeuge

- `tools/kv_k1.py` — K1-Gate je Position (gepaarte Mean-KL gegen `int8`, 95-%-Grenze).
- `tools/kv_error_shape.py` — **Ansatz E4** (GEAR-Diagnose, bench-Run `2026-10-25-e4-error-shape`):
  Kopf (Ränge 1–8) gegen Tail-Zerlegung, Konzentration. **Pfad
  hardcodiert** auf `Strata-data\bench\2026-10-25-s0` (Portabilitätslücke).
- `tools/kv_selstats.py` + `STRATA_KV_SELSTATS` — S1: wo die Indexer-Selektion landet (nur ein
  Debug-Lauf, keine Paarung zweier Arme).
- `tools/needle_bench.py` — Free-Running-Nadeln; **gesättigt** (18/18 im WP; 6/6+6/6 am
  2026-10-24, `bench/results/2026-10-24-needle-window`).
- `tools/ab_engine.py`, `bench/bench_vs_llama.py quality`, `tools/qwen35/parity.py` —
  Free-Running-Greedy-Vergleiche; nur `bench/results/2026-09-27-esp/esp_serve.py` `compare`
  kreuzt Free-Running-Greedy-Tokens mit dem teacher-forced-Argmax-Dump (Muster, einmalig).
- Ein zusammenhängendes Harness, das **beide** Evaluationsarten auf denselben Armen liefert,
  existiert nicht (statische Analyse).

### 1.3 Rohdatenlage (entscheidend für den Kostenplan)

| Datensatz | Status |
|---|---|
| S0-Läufe 2026-10-25 (fp16, int8, win256, plain; **top-256**, je 32.746 Positionen, SHA dokumentiert) | vorhanden **außerhalb** des Repo: `C:\Users\krone\git\Strata-data\bench\2026-10-25-s0\` |
| S3-Läufe 2026-10-27/28 (**neun** Arme auf einem Build: fp16, int8, 16/16, 16/16+LS, 16/8, 8/16,
  8/8, 16/16+Fenster, 16/16+LS+Fenster; top-256) | vorhanden, extern: `Strata-data\bench\2026-10-27-s3\`
  (neun TSVs à ~131 MB, 2026-10-28 gezählt) |
| Tag-Qualitäts-Läufe 2026-10-25 (fp16, int8, win256, tag; eigenes Build, top-256) | vorhanden,
  extern: `Strata-data\bench\2026-10-25-tag-quality\`; Bericht `bench/results/2026-10-25-tag-quality/`
  („gesammelt, nicht ausgewertet"; das A/B Fenster↔Tag ist inzwischen im S3-Bericht nachgerechnet,
  kein Hebel) |
| Ring-Geometrie 2026-10-28 (`q4lm32`+Fenster, S/R variiert, vorregistriert) | vorhanden, **gemessen**
  (2026-10-28, drei Arme, derselbe Build wie S3); Bericht `bench/results/2026-10-28-ring-geometry/`,
  Rohdaten extern `Strata-data\bench\2026-10-28-ring\` |
| Decode-Sonde 2026-10-28 (fp16, int8, `q4lm32`+Fenster 256/64, 64/64, 256/0, `q4lm8v`+Fenster; volles Vokabular, 32 Prompts, ein Decode-Schritt) | vorhanden, extern `Strata-data\bench\2026-10-28-decode\`; Bericht `bench/results/2026-10-28-decode-probe/`. Anderer Pfad und andere Metrik — zählt **nicht** in die Teacher-forced-Cluster |
| Ältere WP-Logs (2K, 128K, Fenster-Runden) | **nicht im Tree** (D2/D6-Vorbehalt in `Ansatz_Improvements.md`) |
| Freie Vokabular-Logs pro Position | existieren nicht (Deckel 256) |
| Per-Position-Logs quant_quality v1 | Gewichts-Set, nicht cwe |

**Konsequenz:** Stufe F1 (Metrik-Entwicklung) braucht **keine GPU** — sie arbeitet auf den
vorhandenen Arm-Läufen: S0 (4), S3 (9), Ring-Geometrie (3, derselbe Build wie S3), Tag-Qualität (4)
= **20 Läufe, fünfzehn verschiedene Arm-Konfigurationen** (die Decode-Sonde steht daneben und wird nicht mitgezählt: anderer Pfad, andere Metrik). Vergleiche paarweise nur **innerhalb**
eines Build-Clusters; S0↔S3 ist über die Ankerläufe `int8` 0,01688 und `q4lm8v` 0,03182
gebrückt (S3-Bericht: auf fünf Stellen reproduziert), S3↔Tag nicht (anderes Build, `6F78D941…`).
Das ist die ökonomische Grundlage des ganzen Plans.

### 1.4 Bereits gemessene Schwächen der Baselines (Auftragsteil B ist teilweise erledigt)

Diese Befunde stehen bereits in der Messgeschichte und werden in F0 nur systematisch katalogisiert,
nicht neu erfunden:

| # | Befund | Quelle |
|---|---|---|
| M1 | Die getrackte KL ist top-64-trunkiert mit Floor-Substitution, **nicht** die KL der vollen Verteilung; Positionen mit 17 nats sind Floor-Ereignisse, keine belegten Vollverteilungs-Events | `docs/paper/PEER_REVIEW_GROK.md` M2; `docs/paper/Ansatz_Improvements.md` D6 |
| M2 | Mittelwertverdünnung ist real: die schlimmsten 1 % der Positionen tragen 85,5 / 81,6 / 70,9 % der KL (int8/Fenster/plain), 88–90 % sitzen in den Rängen 1–8; Positionen mit Kopf-Substitution sind rar (0,7 % bei int8, 1,0 % Fenster, 1,6 % plain) und tragen bei int8 trotzdem ~56 % der Summe | `bench/results/2026-10-25-e4-error-shape/README.md` |
| M3 | Ganzsegment-p99 liegt bei allen Armen hoch (3,66 / 4,76 / 5,55) und ist dominiert von Floor-Ereignissen — es trennt nicht das, was der Mittelwert trennt | `Ansatz_Improvements.md` §1 (Paper Tab. 8) |
| M4 | Perplexity **dreht die Rangfolge um** und lag im 128K-Tail unter jeder Quantisierung **unterhalb der fp16-Referenz**, obwohl alle KL positiv sind | `Ansatz_Improvements.md` §4/E2; `PEER_REVIEW_GROK.md` §3 |
| M5 | Tail-Mean kehrt zwischen den beiden 128K-Texten das Vorzeichen um — es trägt kein Ziel | `KV-int8-quality-below-5bit-quantization.tex` §2.2, §4 |
| M6 | Antwortregion und Ganzsegment trennen unterschiedlich: `int8 − Fenster` dort t=−0,51 (nicht trennscharf), Ganzsegment t=−3,55 | `bench/results/2026-10-25-s0-kl-support/README.md` |
| M7 | **Positivbefund:** Rangfolge und Vorzeichen halten zwischen top-64 und top-256; Niveaus steigen +5,1/+6,4/+6,4 % (Floor-Effekt), t-Vorzeichen stabil | S0-Bericht, „What the wider support does" |
| M8 | Kandidatenlage aus S1: nur 11,7 % der Indexer-Treffer werden exakt gelesen (Recall S 86,2 %, R 28,9 %); Randmedian 0,01 % der Spreizung (Selektionsgrenze dünn, Aggregate stabil) | `bench/results/2026-10-25-s1-selstats/README.md` |
| M9 | Die Level-Zahl bei 4 Bit ist ein Hebel: 16/16 gegen 8/8, gepaart je Position, −0,01748 nats (95-% [−0,02133, −0,01363], t −8,89, 16/16 in 8/8 Segmenten; gegen den `int8`-Anker −0,01451); die nackten 4,50-bit-Arme verfehlen K1 offensichtlich | `bench/results/2026-10-27-s3-level-split/README.md` |
| M10 | K1 ist mit dem vorhandenen Korpus für **keinen** Arm erreichbar: gepaarte Streuung 0,22–0,24 nats ⇒ einseitige 95-%-Grenze bei echter Parität 0,0020 > 0,001; eine Grenze unter 0,001 bräuchte n ≈ 131.000 Positionen (128K-Satz, existiert nicht) | S3-Bericht, „K1 und die Auflösung der Regel" |
| M11 | Die KL ist **ankerabhängig**: gegen den `int8`-Anker sah die LS-Skala +0,0028 „besser" aus (t 1,64); gegen `fp16` bleiben +0,0003 (top-256) — der scheinbare Gewinn war ein Referenzartefakt. Jede spätere Vergleichbarkeit muss den Anker mitliefern | S3-Bericht, „Korrektur zu einer früheren Aussage" |
| M12 | Der **Sink-Term ist real, die S/R-Geometrie ist am Plateau**: `R` = 0 kostet +0,00394 nats (95-% [+0,00097, +0,00691], 8/8 Segmente) — fast so viel wie der ganze 16/16-Gewinn (0,0040); 128/256 Sinks sind von 64 nicht zu trennen (unter M10-Auflösung). Der Verlust sitzt ab Position 2.048, wächst mit der Distanz und ist kein konstanter Offset — Beleg für Regionsauflösung (§9) | `bench/results/2026-10-28-ring-geometry/README.md` |
| M13 | Die **Decode-Sonde reproduziert die Prefill-Rangfolge nicht**: volles Vokabular, 32 Prompts, `int8` 0,00055, Fensterkandidat 0,00225; Parität **nicht bestätigt, nur nicht ausgeschlossen** (+0,0017, Intervall berührt die 0, 6/8 Segmente schlechter). `S`-Achse (+0,00175 < 0,0027) und Sinks (+0,00100 < 0,0016) liegen unter der Auflösung. 16/8 ist nominell besser als 16/16 (−0,00074, n. s.). Prompts ≤ 1.000 Token, also Decode-Pfad **ohne** Ring-Verdrängung | `bench/results/2026-10-28-decode-probe/README.md` |

Damit ist der Auftragsteil B („Failure Analysis of Existing Metrics") zu einem erheblichen Teil
**aus vorhandenen Messungen** beantwortbar; F0 schreibt ihn nur zusammen und prüft die noch
fehlenden Behauptungen synthetisch.

### 1.5 Prämissen des Auftrags gegen den Code prüfen (§4, §5, §10)

Der Auftrag nennt Varianten und Mechanismen, die im Baum so nicht existieren. Das steht vor jeder
Messung fest (statische Analyse, verifiziert 2026-10-27, Branch `kv_q4_quant_with_int8_quality`):

1. **„K8V4-NL" existiert in diesem Tree nicht.** Kein Format-Flag, kein `--kv`-Wert, kein Kernel.
   Erwähnt wird der Name nur in `docs/paper/QSA-Aware-KV-quant_v1.md`; `docs/KV_QUANT_WORKPACKAGE.md`
   liegt laut Zitaten nur auf Branch `KV_QUANT_research`. Vor Auftrag §5 ist der Research-Branch zu
   laden und zu diffen; bis dahin gilt: hiesige Formate sind `fp16`, `int8` (8,25 bit/cell),
   `q4_0` (4,50), `k8v4` (K int8, V rotiertes Q4_0, 816 B — statisch 6,38 bit/cell: 816·8/1024,
   dieselbe Buchung wie int8 1056 → 8,25; die oft genannten „6,0 bit/cell" sind die **Bar-A-Decke**
   des Vorgänger-Papers, kein k8v4-Messwert: `KV-Window-Hardening.tex` Bar A), `q4lm*`-Lloyd-Familie
   und `q4lm8v`+Fenster (4,50 Basis + Ring = 5,29 bei 128K). Quelle: `include/strata/kernels/kv_stream.hpp:44-65`,
   `kv_q4.hpp:81-103`, `setup.py:6865` (816 je Token, daneben `docs/DETAILS.md:2312` mit „816 B/cell"),
   `KV-int8-quality-below-5bit-quantization.tex` (8,25 / 5,29).
2. **Eine „Window-Migration" gibt es nicht.** Beide Kopien (quantisiert und exakt) werden beim
   Append geschrieben; das Fenster verlässt eine Zelle nur dadurch, dass der Leser `qsa_win_slot`
   −1 liefert (Tag/Horizont) — es findet **keine erneute Quantisierung und keine Datenwanderung
   statt** (`include/strata/kernels/qsa_decode_attn.hpp:66-107`, `src/core/verify.cpp:1685-1750`,
   `src/kernels/cuda/kv_q4.cu:267-280,381-510`). Der „Präzisionswechsel" ist also ein
   **Lesewechsel**, messbar als regionsbedingte KL an der Horizontkante. GPU-Residency ist
   separat: `kv_stream.cu:93-156` (CLOCK-Verdrängung) — Auftrag §5, letzter Absatz, ist damit
   wörtlich zu trennen. Fenster + Streaming wird außerdem abgelehnt (`generate.cpp:3202-3213`).
3. **Selection Distortion ist im heutigen Pfad nicht strukturell null, aber der Weg ist indirekt.**
   Die Indexer-Keys werden aus dem Hidden State `x` projiziert (`src/core/layer.cpp:1351-1358`,
   beide Zweige `project_bf16[_pair](x, … w_idxk → idx_raw)`, Append `:1470-1474`), **nicht** aus
   dem ggf. quantisierten K-Cache. Für die **erste** QSA-Schicht (Modell-Layer 3, davor nur GDN-Schichten ohne
   Token-KV) ist die Selektion KV-quantisierungsunabhängig. Ab der zweiten QSA-Schicht kann
   KV-Quantisierung über den Attention-Ausgang vorheriger Schichten indirekt in `x` und damit in
   die Selektion wirken. **Hypothese, statisch; zu verifizieren (F4a).** Eine Indexer-Quantisierung
   selbst existiert nicht (FP32 per Vertrag, `include/strata/kernels/qsa.hpp:79-87`) — die
   Gegenläufe C/D des Auftrags §4.4 sind daher in der Form nicht ausführbar; brauchbare Ärmel sind
   A/B (FP32-Indexer × fp16/quantisiertes KV) plus ggf. `--native-qsa-indexer` als
   Implementierungsvariante (`native_qsa_indexer.cu`, `generate.cpp:858`).
4. **Der Tree hat laufende, uncommittete Änderungen** (`git status` 2026-10-27: u. a.
   `bench/results/2026-10-27-s3-level-split/README.md`, `docs/MODELS.md`, `src/core/layer.cpp`,
   `src/core/mtp.cpp`, `src/core/session.cpp`, `src/kernels/cuda/iq_kernels.cu`,
   `src/program/generate.cpp`). Update 2026-10-28 (Review-Durchgang, `git status --porcelain`):
   die S3-README, `session.cpp` und `generate.cpp` sind seither committet; modifiziert bleiben
   `docs/MODELS.md`, `src/core/layer.cpp`, `src/core/mtp.cpp`, `src/kernels/cuda/iq_kernels.cu`;
   Branch unverändert `kv_q4_quant_with_int8_quality`. Vor jeder späteren Messaussage der Stand
   neu aufnehmen.

## 2. Taxonomie der Quantisierungsfehler (Auftrag §15, Punkt 1)

| Klasse | Beschreibung | Deckung heute | Blindstelle |
|---|---|---|---|
| T1 Verteilungsfehler | kontinuierlicher Drift der Token-Wahrscheinlichkeiten | Mean KL, JS/TV (trunkiert) | keine Entscheidungs-, keine Zeitinformation |
| T2 Entscheidungsfehler | Argmax-/Top-k-Wechsel, gewichtet mit Referenz-Margin | Top-1 (unbewertet), Margin fehlt | Flips bei Gleichstand überbewertet, Flips mit großem Margin unterbewertet |
| T3 Tail-Ereignisse | einzelne katastrophale Positionen, inkl. Floor-Artefakte | P99, Max, CVaR (offen) | P99 blind über dem Quantil; Floor dominiert |
| T4 Temporale Fehler | Bursts, Drift, Regionsübergänge (Fensterhorizont, Page) | **keine** | alle Perzentil-/Mittelgrößen sind permutationsinvariant (§3, CB6) |
| T5 Funktionale Fehler | Task-Ergebnis verschlechtert sich (Free-Running) | Needle (gesättigt), keine breite Suite | **kein Ground-Truth-Kanal für „Qualität"** |
| T6 Selektionsfehler (QSA) | andere Blöcke ausgewählt | keine (S1 nur aggregiert, ein Arm) | Paarung Ref/Arm fehlt; Pfad erst F4a prüfen |
| T7 Ursache | K vs V vs Fenster vs Kernel | Counterfactual-Läufe (§4.4) nicht vorhanden | nur A/B-Arme möglich |

Auftrag §1.1 A–E kartiert auf: A→T1/T3 (CB1), B→T2 (CB3), C→T4 (CB6), D→T5 (F2-Suite),
E→T6/T7 (F4).

## 3. Zentrale Hypothese kritisch (Auftrag §2)

> Die Schädlichkeit hängt von Entscheidungssensitivität, struktureller Bedeutung, temporaler
> Korrelation und Kontextveränderung ab — nicht nur von der Abweichungsgröße.

Kritische Prüfung, noch vor Messungen:

- **Für T4 ist die Hypothese halb entschieden:** Mean KL, P99 und CVaR sind
  permutationsinvariant. Zwei Fehlerreihen mit gleicher Wertemenge, einmal gemischt, einmal in
  Bursts, liefern **identische** Werte für alle diese Metriken. Ob Burst die Free-Running-Qualität
  stärker schädigt, ist die einzige offene Hälfte und wird in F2 getestet. Bis dahin gilt: die
  Aussage „zeitliche Struktur ist für heutige Metriken unsichtbar" ist ein **Beweis, keine Messung**.
- **Für T2 gibt es eine echte Ambivalenz**, die der Auftrag selbst benennt: kleiner Margin = oft
  funktional harmlos. Umgekehrt ist ein Flip mit **großem** Margin ein starkes Ereignis. Die
  sinnvolle Größe ist deshalb bedingt: `Flip ∧ Margin groß` statt roher Flip-Rate.
- **Für T6 schränkt die Architektur die Hypothese ein** (§1.5.3): Der direkte Weg „KV-Fehler →
  Selektion" gibt es nicht; der indirekte Weg beginnt erst mit der zweiten QSA-Schicht. Die
  Hypothese ist für dieses Modell schwächer formulierbar als allgemein angenommen; ein
  Gegenbeispiel im Sinne „Selektion unverändert, Funktionsverlust vorhanden" ist bereits durch die
  reine K/V-Strecke möglich.
- **Mögliche Gegenhypothese:** Falls sich zeigt, dass Free-Running-Qualität allein von der
  kumulierten ΔNLL (T1) erklärt wird und Margin/Burst/Regionen keinen messbaren Zusatzbeitrag
  liefern, ist die Hypothese **abzulehnen** — das wäre das ehrliche Ergebnis (Auftrag §13, Nr. 10).

## 4. Gegenbeispiele (Auftrag §15, Punkt 2) — Konstruktionen, die jetzt schon feststehen

Alle sieben sind **synthetisch oder aus vorhandenen Messungen**; keiner erfordert einen Lauf.

| # | Gegenbeispiel | Warum die Baseline versagt | Status |
|---|---|---|---|
| CB1 | **Mittelwertverdünnung.** Eine Position mit 17 nats in 32.746 → +0,00052 nats auf das Mittel, unterhalb der K1-Schwelle +0,001 | Mean-Alarm bleibt unter der Ziel-Schwelle, obwohl eine Position katastrophal ist | synthetisch (Arithmetik); Real-Anker: M2 (schlimmste 1 % tragen 71–86 %) |
| CB2 | **P99-Blindheit.** Arm A: 2 % der Positionen bei 5 nats; Arm B: 1,98 % bei 5 nats und 0,02 % (~7 von 32.746) bei 500 nats. Beide p99 = 5 (robust gegen Perzentil-Konvention, weil > 1 % auf dem Wert liegt); Mittel von B doppelt so groß (0,100 gegen 0,199); die katastrophalen Einzelpunkte liegen oberhalb des 99,98-Perzentils | P99 erfasst nichts oberhalb seines Quantils; Max/CVaR trennen, Mittel trennt nur die Masse, nicht das Extrem | synthetisch; Real-Anker: M3 (Ganzsegment-p99 dominiert von Floor-Ereignissen) |
| CB3 | **Top-1 ohne Relevanz, Qualität ohne Top-1-Wechsel.** (i) Flip zwischen zwei Synonymen bei Margin 0,001 nat → Top-1-Agreement fällt, Funktion gleich. (ii) Kein Flip, aber `P(y_t)` halbiert sich (große ΔNLL) → Top-1 identisch, Ziel-Token-Risiko steigt | Top-1 zählt Wechsel ohne Gewichtung und übersieht Ziel-Token-Verluste ohne Wechsel | synthetisch; offline ab `S0/S3`-TSV prüfbar (beide Größen sind dort vorhanden) |
| CB4 | **PPL-Inversion.** 128K-Tail: PPL 2,4052 / 2,4079 / 2,4029 gegen fp16 2,4149 — jede Quantisierung „besser" als die Referenz, während alle KL positiv sind | PPL misst Textnähe, nicht Ref-Differenz; als Zielgröße invertiert sie die Rangfolge | **gemessen** (M4) |
| CB5 | **Definitions-Artefakt im Tail.** 17-nat-Positionen = top-64-Floor-Substitutionen; Wechsel auf top-256 hebt jedes Niveau um +5,1 bis +6,4 % (S0: +5,1/+6,4/+6,4 %), Rangfolge bleibt | P99/Max teilweise Messung der Definition, nicht der Verteilung; Vergleichbarkeit mit Literatur-VollKL gegebenenfalls trügerisch | **gemessen** (M1, M7) |
| CB6 | **Ordnungsblindheit.** Gleiche Wertemenge der positionsweisen KL, gemischt gegen geclustert: Mean, P99, CVaR, Top-1 **identisch**, Free-Running-Verhalten hypothetisch verschieden | zeigt konstruktiv, dass keine permutationsinvariante Metrik T4 sehen kann | synthetisch (Beweis); ob Clustering schädlicher ist: F2 |
| CB7 | **Selektion gegen Rekonstruktion.** Ein Selektionswechsel, der einen einzigartigen Faktblock verliert, gegen eine gleich große numerische Abweichung ohne Selektionswechsel: K/V-Rekonstruktionsmetriken können gleich sein, die Funktion nicht | T6 liegt prinzipiell außerhalb jeder K/V-Metrik | synthetisch; **Vorbedingung** (Selektion kann sich überhaupt ändern?) = F4a |

Nicht behauptet wird: dass eines dieser Beispiele **in den aktuellen Messungen tatsächlich so
auftritt** — das ist die Aufgabe von F1/F2. CB5 warnt umgekehrt davor, jede scheinbare
Verbesserung im Tail für real zu halten.

## 5. Liste der vielversprechendsten Messgrößen (Auftrag §15, Punkt 3)

Priorität = Informationsgewinn pro Aufwand. „Offline" heißt: berechenbar aus den vorhandenen
S0/S3/Tag-TSVs **ohne GPU und ohne Engine-Änderung**.

| P | Merkmal | Definition (Kurzform) | Offline? | Erwarteter Zusatznutzen | Redundanzrisiko |
|---|---|---|---|---|---|
| 1 | ΔNLL | `log P(y_t) − log Q(y_t)` aus den Target-Spalten; getrennt für Vorzeichen, kumuliert, Antwortregion | ja | Ziel-Token-Risiko, unabhängig vom Argmax (gegen CB3) | hoch zu Mean KL — erst Korrelationsmatrix in F1 |
| 1 | Entscheidungs-Margin | `m_t = lp₁ − lp₂` (=`log(p₁/p₂)`, invariant gegen Logit-Shift) aus den Top-k-Spalten; Ereignis `Flip ∧ m_t ≥ 1 nat` | ja | trennt harmlose Flips von entscheidenden (gegen CB3) | mittel zu Top-1 |
| 1 | CVaR95/99, P95, P99.9, Max, „Floor-/Kopf-Substitutionsrate" | Mittel der schlimmsten 1−α-Positionen; Rate der Kopf-Substitutionen (Ansatz E4 / Error-Shape-Logik) | ja | robustere Tail-Sicht als P99 (gegen CB2), benennt Floor-Artefakte (CB5) | ggf. hoch untereinander; F1 wählt eine |
| 1 | Temporale Kennzahlen | längster Run über Schwellwert, Bursts/1k Positionen, Autokorrelation 1., kumulierte KL-Kurve je Segment | ja | einziger Zugang zu T4 (CB6); permutationsinv. Baselines können hier nicht konkurrieren | niedrig |
| 2 | Regionsbedingung | KL je Region: Sink (c<R), Fensterkante (erste ~S Positionen hinter dem Horizont), Recent, Ring (bezahlt, quantisiert gelesen), Outside, Page-Grenze (c mod 4) | ja (Positionsindizes + S/R/B aus `.seg.json`) | Ursachenzuordnung T7 ohne Counterfactual; testet Auftrag §5 direkt | nur Fenster-Arme betroffen |
| 2 | Entropie-Band | `H(P)` bandbegrenzt aus Top-256 + Tail-Masse `r`; KL/Flip je H-Bin | ja, **mit Bandbreite** `r·ln(V−k)` (Herleitung §1.1) | Konditionierung der KL (Auftrag §3.2) ohne neue Logs | — |
| 2 | Top-k-Overlap/Ranking | Jaccard/Kendall der Top-64/256-Listen, Mass-Shift | ja | komplementär zu KL bei ähnlichem Mean | mittel |
| 3 | Reverse-KL, JS, TV, Hellinger (trunkiert) | auf demselben Support wie `kl_between`, plus Massen-Ausreißer `r` getrennt | ja, mit Vorbehalt | testet Definitions-Abhängigkeit (D6-Teil) | mehrfach untereinander — Doppelzählung vermeiden |
| 3 | Vollvokabular-KL, Reverse über volles Vokabular | Engine-Schreibung erweitern | **nein** — Engine-Änderung, Top-k-Deckel, ~32 GB/Arm Rohdump (8,1·10⁹ Werte ×4 B, Arithmetik) → nur Stichprobenpositionen | beendet die M1-Debatte endgültig | — |
| 4 | QSA-Selektionsmetriken | Jaccard/Recall der Selektionen Ref gegen Arm, Score-Rand, Attention-Mass-Verlust | **nein** — Instrumentierung nötig (S1 liefert nur Aggregate, ein Arm) | Auftrag §4 | — |
| — | **Free-Running-Events** | Ground Truth (kein Metrik-Kandidat): siehe `BEYOND_KL_QUANT_FIDELITY.md` §4 (G3) | nein (Suite zu bauen) | **Voraussetzung für jede Prädiktionsaussage** | — |

Der Screening-Regel in F1: ein Merkmal fällt, wenn es über alle Läufe eines Build-Clusters (S0: 4,
S3: 9, Tag: 4) mit einer Baseline |ρ| ≥ 0,9 korreliert **und** in der Restbetrachtung (partial)
nichts hinzufügt (Setzung, `BEYOND_KL_QUANT_FIDELITY.md` §3); `int8` läuft in allen drei Clustern und ist der natürliche
Paaranker, cross-cluster nur wo gebrückt (§1.3).

## 8. QSA-spezifische Dimension (Auftrag §4) — ehrlich abgestuft

1. **F4a — Pfadprüfung (statisch, zuerst):** Bestätigen, dass `idx_raw` aus `x` kommt und nicht
   aus dem Cache (bereits statisch belegt, `layer.cpp:1351-1358/1470-1474`); prüfen, ob es einen
   Zwangspfad für Sinks/Recent in die Selektion gibt (C4 der offenen Liste). Ergebnis: welche
   QSA-Schichten bei welchem Arm überhaupt eine Selektionänderung haben **können**.
2. **F4b — Messbarkeit:** Selektionen beider Arme auf denselben Positionen aufzeichnen
   (Instrumentierung ähnlich `STRATA_KV_SELSTATS`, aber paarweise und identitätserhaltend, nicht
   nur Aggregate). Erst danach Overlap/Margin/Mass (§4.1–4.3) definieren. Cost: ein
   Debug-Build + Runden; int8-SHA-Gate gilt weiterhin nur für den unaufälligen Default-Pfad.
3. **Counterfactual (§4.4) in realistischer Form:** A = FP32-Indexer × fp16-KV (S0-fp16),
   B = FP32-Indexer × quantisiertes KV (S0-Arme), C/D entfallen, solange es keine
   Indexer-Quantisierung gibt; stattdessen optional `--native-qsa-indexer` als
   Implementierungsarm. Selektions-Override aus dem Referenzlauf braucht zusätzliche
   Instrumentierung (Aufnahme des Ref-Laufs, Wiedergabe im Arm-Lauf) — Postponed auf F4c.
4. **Attention-Mass (§4.3):** erst nach F4b; Aggregationsfrage (Tokens/Heads/Layer) vor der
   Metrik; Achtung: niedriger Overlap ist nicht automatisch schlecht (Auftrag nennt es selbst) —
   die Entscheidung kommt erst aus E1/E2.

Bis F4 steht: **Fall D des Auftrags (§14) ist für die heurige Architektur eingeschränkt
möglich**; alles darüber hinaus ist Instrumentierung, kein Postulat.

## 9. Positionsabhängige Fehleranalyse ohne neue Läufe (Auftrag §5)

Sofort offline möglich, sobald die S0/S3/Tag-TSVs vorliegen:

- Regionszerlegung für den Fenster-Arm: Sink (c<64), Horizontkante (erste ~S=256 Positionen
  hinter `n_kv−S`), Recent, Ring (im Selstats-Debuglauf die Spalte `in_b`: bezahlt, quantisiert
  gelesen), Outside — plus Page-Grenze (c mod 4) und Fensterübergänge. S/R/B je Lauf stehen als
  Env-Liste in der `.seg.json` (`quant_quality.py:300-301`); S0-Arm: `STRATA_KV_WINDOW_S=256`,
  `STRATA_KV_WINDOW_R=64` (S0-Bericht Z. 44), Defaults 128/8 (`src/core/layer.cpp:270-277`),
  Ring = `S+B` (`include/strata/kernels/qsa_decode_attn.hpp:74-84`). Hypothese: der gemessene Restfehler sitzt **unmittelbar außerhalb**
  des Overlays (das ist die Lesart der „letzten 1.024"-Statistik, `KV-int8-quality-below-5bit…` §4);
  die Regionsaufteilung prüft das direkt statt indirekt.
- Präzisions- gegen Residency-Ereignisse trennen: Präzisionsregion = obige Labels; Residency =
  `kv_stream`-Verdrängungen — im Qualitätslauf ohnehin inaktiv (`--kv-resident` mit Fenster
  verboten), also im Report als „nicht zutreffend" benennen, nicht vermischen.
- K8V4-NL-Teil des Auftrags: erst nach dem Branch-Diff (§1.5.1). Keine Architektur aus dem
  Auftragstext ableiten.
