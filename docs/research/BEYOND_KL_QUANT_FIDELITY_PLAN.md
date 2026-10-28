# Forschungsansatz: Predictive Quantization Fidelity für KV-Cache-Quantisierung („Beyond KL")

Stand: 2026-10-27; Review-Durchgänge 2026-10-27/28 (diese Fassung bringt nur Klarstellungen und
das Inventar auf den aktuellen Stand — S3 ist inzwischen ein Neun-Arm-Lauf geworden; **keine
neuen Messungen, kein Code** durch diesen Plan) · Begleitdokument zu
`docs/research/BEYOND_KL_QUANT_FIDELITY_ASSIGNMENT.md`
(dort der wörtliche Arbeitsauftrag). **Diese Fassung enthält keine neuen Messungen und keinen
geänderten Code.** Jede Zahl stammt aus einem zitierten Bericht oder aus `bench/results/`; alles,
was aus Code-Lektüre folgt, ist als **statische Analyse** markiert. Kein Engine-Build, keine Kompilierung.

## 0. Zweck und Abgrenzung

Das übergeordnete Arbeitspaket zielte auf `int8`-Qualität unter 5,0 bit/cell (Ziel K1–K6 in
`docs/paper/Ansatz_Improvements.md` §7 und `docs/paper/KV-int8-quality-below-5bit-quantization.tex`).
**Update 2026-10-27/28:** das 5,0-bit-Ziel ist laut S3-Bericht („Was das für das Ziel bedeutet")
**zurückgestellt** — ohne neue Rotations-/Korrekturansätze nicht erreichbar; der Branch maximiert
zuerst Qualität bei gleichen Bytes. Das ändert an diesem Plan nichts: die Metrikfrage („sagen die
heutigen Statistiken etwas über reale Aufgaben voraus?") hängt am Default `int8` und an den
4,50-bit-Armen, nicht am Zurückstellungsdatum. Die Entscheidungskriterien K1–K3 sind heute
teacher-forced-KL-Statistiken (Mean KL, p99, Ganzsegment) — wobei K1 selbst nach M10 (§1.4) mit
dem vorhandenen Korpus nicht auflösbar ist.
Dieser Ansatz prüft, ob diese Statistiken vorhersagen, was auf realen Aufgaben passiert — und
welche kleinen Zusatzgrößen die Vorhersage verbessern, falls nicht.

Bindende Regeln aus dem Auftrag (§13) und aus `Ansatz_Improvements.md` §0:

1. Kein Ersatz von Mean KL/P99 ohne gemessenen Zusatznutzen. Eine Kombination bestehender
   Metriken ist dem Formelneuland vorzuziehen.
2. K1–K6 der laufenden Leiter werden **nicht** mitten in der Messung umdefiniert (Pre-Registration
   gilt). Neue Metriken können höchstens ergänzend berichten und erst für künftig neu
   vorregistrierte Vergleiche verbindlich werden.
3. Der `int8`-Default bleibt byte-identisch (SHA-Gate); K8V4/Fenster-Code wird in dieser
   Metrikforschung nicht verändert (Auftrag §13, Nr. 8).
4. Negative Ergebnisse sind Ergebnisse: darf Mean KL + P99 für die Frage „wird es schlechter?"
   bereits ausreichen, ist das die Empfehlung.

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
| Ältere WP-Logs (2K, 128K, Fenster-Runden) | **nicht im Tree** (D2/D6-Vorbehalt in `Ansatz_Improvements.md`) |
| Freie Vokabular-Logs pro Position | existieren nicht (Deckel 256) |
| Per-Position-Logs quant_quality v1 | Gewichts-Set, nicht cwe |

**Konsequenz:** Stufe F1 (Metrik-Entwicklung) braucht **keine GPU** — sie arbeitet auf den
vorhandenen Arm-Läufen: S0 (4), S3 (9), Ring-Geometrie (3, derselbe Build wie S3), Tag-Qualität (4)
= **20 Läufe, fünfzehn verschiedene Arm-Konfigurationen**. Vergleiche paarweise nur **innerhalb**
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
| — | **Free-Running-Events** | Ground Truth (kein Metrik-Kandidat): siehe §7.3 | nein (Suite zu bauen) | **Voraussetzung für jede Prädiktionsaussage** | — |

Der Screening-Regel in F1: ein Merkmal fällt, wenn es über alle Läufe eines Build-Clusters (S0: 4,
S3: 9, Tag: 4) mit einer Baseline |ρ| ≥ 0,9 korreliert **und** in der Restbetrachtung (partial)
nichts hinzufügt (Setzung, §7.5 E5); `int8` läuft in allen drei Clustern und ist der natürliche
Paaranker, cross-cluster nur wo gebrückt (§1.3).

## 6. Was F1 offline liefern kann — ein konkretes Beispiel

Bereits jetzt ohne einen einzigen Lauf berechenbar, wenn die externen TSVs vorliegen:

1. Pro Position und Arm: ΔNLL, Margin, Flip-Flag, CVaR-Beiträge, Run-Struktur, Regionslabel.
2. Differenzielle Tabelle je Paar (gepaart wie K1): Kandidat(Arm − `int8`) für jede Kennzahl —
   **Anker vor F1 festlegen und mitliefern** (S3 zeigt: Ankerwahl `int8`↔`fp16` ändert die
   Aussage, M11); S3 hat beide Ankerläufe, S0 nur `fp16`.
3. Redundanzmatrix (Spearman) über die Kennzahlen und gegen die Baselines.
4. Stabilitätsprüfung top-64 gegen top-256 (Wiederholung von S0 für **alle** Kandidaten, nicht
   nur KL) — Kennzahlen, die unter Supportwechsel die Rangfolge drehen, fallen (CB5-Regel).
5. Vorzeichen-/Bootstrap-CIs auf **Segment**ebene (n=8; nie auf Positionenebene — Korrelation
   benachbarter Positionen, Auftrag §7.3).

## 7. Experimentdesign (Auftrag §6, §7, §10)

### 7.1 Arme und deren Verfügbarkeit

| Arm | `--kv` | Teacher-forced-Logs | Free-Running |
|---|---|---|---|
| fp16 (Referenz) | `fp16` | da (S0, S3, Tag) | nötig (Referenzlauf) |
| int8 (Default) | `int8` | da (S0, S3, Tag) | nötig |
| Fenster 256/64 auf 16/8-Bulk | `q4lm8v`+Window | da (S0, Tag) | nötig |
| Fenster 256/64 auf 16/16-Bulk | `q4lm32`+Window | da (S3; **von `int8` nicht zu unterscheiden**, 7,39 bit/value am 4K-Punkt: S3-Bericht, „Das Fenster") | nötig |
| plain 4-bit (16/8) | `q4lm8v` | da (S0, S3) | nötig |
| 16/16, 8/8 | `q4lm32`, `q4lm8` | da (S3) | nötig |
| 16/8, 8/16, 16/16+LS, 16/16+LS+Fenster | `q4lm8v`, `q4lm8k`, `q4lm32s`, `q4lm32s`+Window | **da (S3, 2026-10-27/28 gemessen)** — S3-Bericht: Tabellen-Arme, Anker-Wechsel auf `fp16` dort offengelegt | optional, danach |
| Ring-Geometrie (320/0, 192/128, 64/256) | `q4lm32`+Window, nur S/R variiert | **da (S3-Build, 2026-10-28 gemessen)**: kein Arm gewinnt die vorregistrierte Regel, 256/64 bleibt; Sink-Term +0,00394 nats gesichert (M12); nur Prefill (B = 512), Decode ungemessen | — |
| `q4_0`, `k8v4` auf cwe | existierende Formate | **nein** (Ansatz E2: nie auf cwe gemessen) | nur wenn als Arm benötigt |
| K8V4-NL | existiert hier nicht | vor Auftrag §5: Research-Branch prüfen | — |

Jeder zusätzliche Arm muss **beide** Seiten bekommen (gleiche Konfiguration, gleicher Build,
dokumentierte SHA — Vorbild S0-Bericht).

### 7.2 Zwei Evaluationsarten, strikt getrennt (Auftrag §7.2)

- **Teacher-forced:** liefert alle Kandidatenmerkmale (F1). Existiert für 20 Läufe (15 Arm-Konfigurationen; §1.3).
- **Free-running:** liefert das Ground Truth (F2). Greedy/temperature 0, feste Prompts, dokumentierte
  Engine-SHA. Ergebnis ist **kein** KL, sondern Event-Raten je Task und Arm.
- Die Brücke ist die Prädiktionsfrage (F3): Vorhersage = Kandidatenwert aus dem teacher-forced-Lauf
  gegen den gemessenen Event-Verlust desselben Arms auf demselben Task.

### 7.3 Aufgaben und Fehler-Events (müssen **vor** der Auswertung feststehen)

Vorrang nach automatisierbarer Bewertung und vorhandener Infrastruktur:

| P | Task | Event-Definition (Vorschlag, wird vor F2 eingefroren) | Vorhandene Infrastruktur |
|---|---|---|---|
| 1 | Needle-in-a-Haystack | „gepflanztes Wort nicht reproduziert" | `needle_bench.py`; **gesättigt** → erst härtere Variante (Multi-Key, Distraktoren, mehr Tiefen) |
| 1 | Strukturierte JSON-Ausgabe | Parse-Fehler oder Schema-Verletzung | keine → kleine Suite nötig (nur Skript) |
| 1 | Code mit Tests | Unit-Tests bestanden/nicht (HumanEval-Subset) | Community-Skript vorhanden (`bench/results/2026-10-09-…/scripts/strata_eval_subset.py`) |
| 2 | Tool Calling | Argumentfehler gegen feste Schema-Fixtures | keine → kleine Suite nötig |
| 2 | Längen-/EOS-/Repetition-Ereignisse | vorzeitiger Abbruch, Wiederholungsrate über Fenster | automatisch aus Transkripten ableitbar |
| 2 | Long-Document-Q&A / Multi-Hop | Antwortgold falsch/richtig | RULER-cwe hat `gold.json` (`data/quant_quality/cwe-4k/gold.json`, vom Harness geschrieben: `quant_quality.py:187`) — Antworten liegen da, auch **free-running** einsetzbar (Prompt ohne Answer-Append, Gold per Index) |
| 3 | Agenten-/Chat-Läufe | definierte Pflichtaktion ausgeführt/nicht | aufwändig; erst wenn P1–P2 trennen |

**Sättigungsregel:** zeigt die Pilot-Suite in keiner Arm×Task-Zelle eine Differenz, ist sie erst
zu verschärfen (mehr Tiefen/Distraktoren), bevor F3 startet — sonst ist „Metrik X sagt nichts
voraus" nicht von „es passierte nichts" unterscheidbar (Analogie: Nadel 18/18).

### 7.4 Statistik

- **Resampling-Einheit** = Task bzw. Prompt/Konversation, niemals Position (Auftrag §7.3).
  Zweite Ebene: Segment (n=8) für teacher-forced-Paare.
- **Paarung:** gleiche Testfälle für alle Kandidaten und Baselines; paarweise Tests wie K1.
- **Macht:** fünfzehn Arm-Konfigurationen in drei Build-Clustern ⇒ Spearman-ρ mit breitem CI.
  Erwartung vorab: Aussagen sind Richtungsaussagen mit Konfidenzband, keine Punktschätzungen.
  Leave-one-task-out als Stabilitätsprobe. **Auflösungsgrenze beachten (M10):** bei 32.746
  Positionen und gepaarter Streuung 0,22–0,24 nats liegt die einseitige 95-%-Grenze bei echter
  Parität bei 0,0020 nats — Effekte darunter sind mit cwe-4k grundsätzlich nicht auflösbar, das
  ist eine Eigenschaft des Satzes, nicht der Metrik.
- **Entwicklung/Validierung trennen (Auftrag §13, Nr. 3):**
  - Entwicklungssatz: cwe-4k-Logs (S0/S3/Tag; Paarung nur innerhalb eines Clusters, §1.3) + P1-Tasks.
  - Validierungssatz: **nie** in F1 angesprochen — P2-Tasks (JSON/Tool-Events) und ein zweiter
    RULER-Task (D1), erst in F3 geöffnet.
  - Alle Schwellen und E5/§7.5-Regeln werden vor F3 timestamped in die Ergebnis-README geschrieben
    (Vorbild: „Entscheidungsregeln (vor dem Lauf festgelegt)" in den S1/S3-READMEs).

### 7.5 Erfolgskriterien E1–E6 (Vorschlag, konkret; **Setzungen**, vor F3 einzufrieren)

**Namenskonvention (Disambiguierung):** Die Bezeichnungen E1–E6 unten sind die
Erfolgskriterien **dieses Plans**. Sie sind **nicht** identisch mit den Katalogeinträgen
E1–E4 in `docs/paper/Ansatz_Improvements.md` §2.E (E1 = Indexer-Auswahl↔Overlay,
E2 = q4_0 auf cwe, E3 = Heavy-Hitter-Overlay, E4 = GEAR-Diagnose) und **nicht**
identisch mit dem Bench-Run `2026-10-25-e4-error-shape` (= Ansatz E4). Wo der Plan
Ansatz-Katalogeinträge meint, steht explizit „Ansatz E2/E4" (in §1.2, §5, §7.1, §11 F0, §16).
**F0** bezeichnet die Katalogstufe des Plans (§11), nicht einen Ansatz-Eintrag.
| ID | Regel |
|---|---|
| E1 Prädiktion | Spearman-ρ(Kandidat, Free-Running-Verlust) übersteigt die beste Baseline (min(Mean KL, P99)) im Validierungssatz um **≥ 0,15** und das Vorzeichen hält in ≥ 75 % der Leave-one-task-out-Wiederholungen |
| E2 Frühwarnung | Bei vordefinierten kritischen Events (P1-Tasks) Sensitivität ≥ 0,80 und Spezifität ≥ 0,80 auf Arm×Task-Zellen; mehr als **1 zusätzlicher Fehlalarm pro 10 Zellen** gegenüber der Baseline ist zu viel |
| E3 Rangfolge | Kendall-τ der Arm-Rangfolge gegen die gemessene Qualitätsrangfolge ≥ τ der Baselines, auf beiden Testfällen |
| E4 Kosten | Fast-Mode (§12) ohne zweiten Engine-Lauf und ohne neue Rohdaten; Score-Zeit ≤ 2× der heutigen `compare`-Zeit |
| E5 Redundanz | kein aufgenommenes Merkmal mit \|ρ\| ≥ 0,9 zu einem bereits aufgenommenen |
| E6 Ehrlichkeit | Jede Kennzahl behält ihre Unsicherheit (Bootstrap-CI); „kein Nachweis" ist zulässiges Ergebnis |

Kein Kandidat „gewinnt" durch stärkere Unterschiede allein (Auftrag §7, fettgedruckter Satz) —
erst E1/E2/E3 zusammen zählen.

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

## 10. QBF — Vorschlag zur Fidelity-Bewertung (Auftrag §6)

**QBF bleibt ein Dashboard, keine Zahl**, solange E1–E3 nicht zugunsten einer Aggregation
entscheiden. Vier Dimensionen zuerst (QSA optional bis F4):

| Dimension | Zusammensetzung aus §5 | Satz |
|---|---|---|
| Distribution | Mean KL (top-256), ΔNLL summiert | Baseline behalten |
| Decision | Flip bei Margin ≥ 1 nat, Top-k-Overlap | ergänzend, nicht ersetzend |
| Tail | CVaR99 + Floor-Event-Rate (getrennt ausgewiesen) | P99 bleibt publizierte Baseline |
| Temporal | längster Run, Bursts/1k, Drift je Segment | neu, weil konstruktiv unsichtbar sonst |
| Functional | Free-Running-Event-Raten | Ground Truth, keine Metrik |
| QSA (optional) | Overlap/Margin nach F4b | Gate F4 |

Aggregationsstufen (Reihenfolge wie Auftrag §6, Forschungsauftrag B): (1) Dashboard, (2) **regelbasierte** Schwellen
aus den K1-Prinzipien, (3) gewichteter Score nur mit gelernten statt gesetzten Gewichten,
(4) datengetriebenes Vorhersagemodell nur mit Train/Val-Trennung und nur wenn Stufe 3 scheitert,
(5) Pareto (Qualität, Bits/cell, t/s) als Berichtsansicht für die 5-Bit-Frage. Keine Stufe darf
übersprungen werden, nur weil sie „einfacher" klingt — die einfachste wirksame Stufe ist das Ziel.

## 11. Stufenplan F0–F6 mit Kills und Kosten

Benennung bewusst **F***, um nicht mit der bestehenden Leiter S0–S5 zu kollidieren. Die
bestehende Leiter läuft unverändert weiter; F-Track ist offline-first wie WP-3b und blockiert sie
nicht.

| Stufe | Frage | Arbeit | Kill | Kosten |
|---|---|---|---|---|
| **F0** | Was ist belegt, was behauptet? | Taxonomie (§2), Katalog der M1–M12, CB1–CB7 als Determinist-Fixtures spezifizieren, D6/Ansatz E4/D4 aus dem Katalog einordnen (keine Doppelzählung), Branch-Diff `KV_QUANT_research` (§1.5.1), Split Entwicklung/Validierung schreiben | — | Stunden, kein GPU |
| **F1** | Liefert irgendein Offline-Merkmal mehr als die Baselines — **konsistent**? | Rescoring der S0/S3/Tag-TSVs: §6-Liste, Redundanzmatrix E5, Support-Stabilität top-64/256, Bootstrap-CIs (D4) auf Segmentebene, Regionsaufteilung (§9) | kein Merkmal überlebt E5/Support-Drehung → Kurzbericht: Baselines reichen offline | Stunden, kein GPU; braucht `Strata-data`-Logs + Portierung von `kv_error_shape.py` (Pfad) |
| **F2** | Gibt es ein trennendes Ground Truth? | Free-Running-Suite bauen (§7.3 P1), Sättigungs-Check, Event-Definitionen timestampen, Pilot auf 2 Armen | Sättigung nach einer Verschärfungsrunde → F3 wird verschoben, nicht ergebnislos gestartet | Skripte + wenige GPU-Stunden (Arme × Tasks, greedy) |
| **F3** | Vorhersagen die Kandidaten besser? | je Arm: Kandidatenwerte (F1) gegen Event-Verluste (F2), E1–E6, Validierungssatz öffnen | E1 und E2 verfehlen → **Empfehlung: Mean KL + P99 bleiben**, QBF nur als Dashboard ohne Prädiktionsanspruch | Free-Running je Arm ~kurz; Analyse offline |
| **F4** | Trägt die QSA-Dimension? | F4a statisch → F4b Instrumentierung → ggf. Overlap/Mass | kein messbarer Selektionsunterschied oder kein Zusatznutzen über E1 → Fall D als „im aktuellen Architekturpfad nicht relevant" dokumentieren | Debug-Build + Runden; nur wenn F3 zeigt, dass T5 überhaupt reagiert |
| **F5** | Fast Mode + Integration | §12; Approximationsfehler quantifizieren; Anbindung an `quant_quality.py` (erweitern, keinen zweiten Stack) | Kosten > E4 → nur Research-Mode behalten | Skriptarbeit, Tests |
| **F6** | Bericht | `docs/research/BEYOND_KL_QUANT_FIDELITY.md`, `FIDELITY_METRIC_DESIGN.md`, `VALIDATION_RESULTS.md` | — | Schreiben |

Engine-Änderungen (Vollvokabular-Stichprobe, Selektions-Capture, `STRATA_LOGPOS_TOPK` > 256) sind
**eigene, später vorregistrierte Arbeiten** mit Build, Parität und int8-SHA-Gate — sie sind nicht
Voraussetzung für F0–F3.

## 12. Fast Mode (Auftrag §8)

- **Umfang (Vorschlag):** Mean KL(top-256), CVaR99, ΔNLL(positiv getrennt), Flip bei Margin ≥ 1 nat,
  Floor-Event-Rate, längster Run. Alles aus einem bestehenden TSV, ohne zweiten Engine-Lauf.
- **Approximationsfehler:** KL(top-256) gegen KL(top-64) ist für KL gemessen (S0: +5,1 bis +6,4 % Niveau,
  Rangfolge/Vorzeichen stabil); für jedes neue Merkmal derselbe Support-Test als Pflicht.
  Vollvokabular bleibt die spätere Referenz.
- **Dateigröße:** heutiger TSV ≈ 131 MB/Arm (S0); Vollvokabular-Dump wären ≈ 32 GB/Arm (Arithmetik:
  32.746 × 248.320 × 4 B) — daher Research-Mode nur mit **Stichprobenpositionen** (z. B. alle
  Flip-Positionen plus jede n-te), nie pauschal voll.
- **Messzeit:** Scoring heute im Sekunden- bis Minutenbereich (offline); zu messen in F5, Ziel E4.

## 13. Deliverables und Reihenfolge (Auftrag §12)

Diese Runde geliefert:

- `docs/research/BEYOND_KL_QUANT_FIDELITY_ASSIGNMENT.md` — der Auftrag, wörtlich.
- `docs/research/BEYOND_KL_QUANT_FIDELITY_PLAN.md` — dieses Dokument.

Später, und nur wenn validiert (Auftrag: „ausführbaren Code nur für tatsächlich validierte Teile"):

- `docs/research/BEYOND_KL_QUANT_FIDELITY.md` (F6-Bericht, Kapitel A–I des Auftrags),
  `FIDELITY_METRIC_DESIGN.md` (nach F1), `VALIDATION_RESULTS.md` (nach F3).
- `tools/quant_fidelity.py`, `tools/quant_fidelity_compare.py` als **Erweiterung** der
  `quant_quality`-Familie (Gleiche Eingabeformate, TSV/JSON-Export, keine zweite Pipeline).
- `tests/test_quant_fidelity.py` mit den zehn Testklassen des Auftrags §9 (identische
  Verteilungen, nahezu identische und veränderte Top-1-Entscheidungen, seltene Token, kleine
  Wahrscheinlichkeiten, isolierte Ausreißer, korrelierte Bursts, QSA-Selektionsfehler,
  Window-Grenzen, Sequenzlängen; dazu als eigene Sätze `log_softmax`-Pfad und NaN/Inf-Stabilität).

## 14. Offene Messungen und benötigte Zugänge (Auftrag §12, Punkt 5 der Fallback-Liste: „eine präzise Liste der noch benötigten Messungen")

1. Free-Running-Event-Raten für die entscheidenden Arme (fp16, int8, 16/16+Fenster = Produktkandidat,
   plain 16/8, 16/16, 8/8; F2) —
   die eigentliche Lücke.
2. Rohdaten-Zugang: `Strata-data\bench\2026-10-25-s0\`, `…\2026-10-27-s3\` und
   `…\2026-10-25-tag-quality\` müssen bei der
   Ausführung vorliegen (SHAs im README); sonst Re-Run je Arm ~19–21 min.
3. Vollvokabular-KL (Engine-Deckel 256) — später, getaktet, mit SHA-Gate.
4. Selektionsaufzeichnung paarweise (F4b) — Instrumentierung, kein Postulat.
5. 128K-per-Position-Logs: nicht im Tree; ein 128K-Rescore ist ein eigener Lauf.
6. Härtere Nadel-/JSON-/Tool-Fixtures (Sättigungsvermeidung).
7. Zweite Task-Familie (D1) für den Validierungssatz.
8. K8V4-NL-Ist-Stand: Branch `KV_QUANT_research` laden und diffen, bevor §5 des Auftrags
   irgendetwas behauptet.

## 15. Grenzen und Risiken

- **Macht:** 8 Segmente, fünfzehn Arm-Konfigurationen (drei Build-Cluster), ein Modell, eine Karte,
  eine Task-Familie im
  Entwicklungssatz. Aussagen sind Richtungen mit CI; Overfitting auf cwe ist die Hauptgefahr —
  dagegen steht die harte Trennung des Validierungssatzes (§7.4).
- **Sättigung:** Needle bereits 18/18. Ohne trennendes Ground Truth ist keine Metrikforschung
  möglich; das ist der kritische Pfad von F2.
- **Bestehende Metriken sind nicht zu unterschätzen:** S0 zeigt Definitions-Stabilität, und die
  Baselines haben ihre Rollen bereits abgedeckt (PPL als nicht-Entscheidungsgröße, Tail-Mean als
  unbrauchbar, P99 als Tail-Indikator). Der realistischste Ausgang ist: **eine kleine Erweiterung
  (ΔNLL + Margin-Flip + CVaR + Run) um die Baselines herum, kein Ersatz.**
- **Rohlogs können verschwinden** (externes Verzeichnis); nur dokumentierte SHAs schützen die
  Reproduzierbarkeit.
- **Bestehende Metrik-Code-Lücken:** kein Test deckt `kl_between`/`compare`/p99/PPL ab (statisch);
  `kv_error_shape.py` hat hartcodierten Pfad. Beides erst bei F5 anfassen (Code-Änderung dann mit
  Tests).
- **K8V4-NL-Nennung im Auftrag** ist in diesem Tree unbelegt; Werbung dafür wäre genau die
  Architekturannahme, die §13, Nr. 7 verbietet.

## 16. Was diese Runde nicht getan hat

Kein Code wurde verändert, nichts kompiliert, keine Messung gestartet und keine Zahl erfunden.
Alle Zahlen sind zitiert (S0, S1, Ansatz E4 = bench-Run `2026-10-25-e4-error-shape`, S3,
Ring-Geometrie, Tag-Qualität, Paper/Reviews); alles aus Code-Folgerungen ist oben als
statische Analyse markiert und muss vor jedem Claim gegen den dann aktuellen Branch neu verifiziert
werden. Der Review-Durchgang 2026-10-28 hat nur Bestand gepflegt: Inventar (§1.3/§7.1) auf den
Neun-Arm-S3-Stand und die Ring-Läufe gebracht, M10/M11 aus dem S3-Bericht ergänzt, CB2/CB5
präzisiert, §5-Verweis korrigiert — keine dieser Änderungen ist eine neue Messung. Ein zweiter
Durchgang desselben Tages hat dasselbe mit dem inzwischen fertigen Ring-Lauf getan: Inventar und
Zählungen (§1.3, §7.1, §7.2, §7.4, §15) von „in Arbeit (1/3)" auf „gemessen" gezogen, M12 aus dem
Ring-Bericht ergänzt, F0 auf M1–M12 erweitert — wieder nur gepflegt, nichts davon ist eine eigene
Messung dieser Runde.
