# Arbeitspaket: die KV-Cache-Quantisierung verbessern (Stand 2026-10-14)

Status: **WP-0 steht, WP-1a ist gebaut und gemessen** (`--kv q4lm32` / `--kv q4lm32s`: §5.1-Nachtrag für das
Design, §7.3 für die Parity-Zahlen, §7.4 für den 1K-A/B). Der Plan-Teil §1–§6 stammt aus dem Survey
[KV_QUANT_RESEARCH.md](KV_QUANT_RESEARCH.md); wo die eigenen Messungen ihn korrigieren, steht das dabei. §4/§5 sind
**Erwartungen aus fremden Messungen** und so gekennzeichnet; alle Zahlen in §7 sind **Strata-Messungen** mit dem
Hinweis, worauf sie gemessen wurden. Branch `KV_QUANT_research`, noch kein PR.

## 1. Ziel

Die Qualitätslücke des opt-in `--kv q4_0` verkleinern — mit einem **datenblinden, modell-agnostischen** Verfahren,
das mit **Qwen3.8-Flash-Next** (per-Head `q_norm`/`k_norm`, `head_dim` 256) funktioniert, ohne Kalibrierung und
ohne den Default-Pfad (`int8`) in irgendeiner Weise zu berühren.

## 2. Nicht-Ziele

- Kein TurboQuant-/SpectralQuant-Import: kein Kalibrierungs-Subsystem, keine Eigenbasis, keine Lern-Rotationen.
- Kein QJL-Residuum (in zwei unabhängigen Nachbauten ohne Effekt, auf K sogar schädlich — §5.2 des Surveys).
- Keine Änderung an Default, Byte-Zahl oder Byte-Gleichheit der bestehenden `int8`/`q4_0`/`k8v4`-Pfade.
- Keine neue Abhängigkeit.

## 3. Ausgangslage (gemessen, alles aus diesem Repo)

| Größe | Wert | Quelle |
| --- | --- | --- |
| **int8 gegen fp16** (neues fixes Set, 5×999) | **−0,10 % PPL**, KL 0,0136 | `bench/results/2026-10-14-kv-first-run/README.md` |
| **q4_0 gegen fp16** (neues fixes Set, 5×999) | **+1,12 % PPL**, KL 0,0364 (**2,7× int8**), Top-1 94,15 % | ebd. |
| int8 gegen fp16 (alter Doc-Probe) | ununterscheidbar (±2 Standardfehler) | `bench/results/2026-09-27-kv-q4/README.md` |
| q4_0 Document-PPL (alter Doc-Probe) | **+8 % @1K, +12 % @8K** | ebd. |
| q4_0 KL gegen fp16 (alt) | 0,039 (1K) / 0,297 (8K); 2,6-4,2× int8 | ebd. |
| q4_0 Top-1 (alt) | 91,3 % (Doc 1K), 88,0 % (8K) | ebd. |
| q4_0 Needles | 5/5 (1K-262K) | ebd. |
| q4_0 Decode @128K | 67,4 t/s, 0,83 GiB K/V (`--kv-resident 32768`) | ebd. |
| Layout | 18 B je 32 Werte, 144 B/Head, **4,5 bit/Wert** | `include/strata/kernels/kv_q4.hpp` |

Die beiden neuen Zeilen sind der **erste Meßpunkt auf dem festen Set** (§7.1) und der Maßstab für WP-1. Die alten
Doc-Probe-Zahlen („+8 % @1K") stammen von einem anderen, inzwischen entfernten Werkzeug auf einem einzelnen langen
Dokument und sind mit dem neuen Set **nicht** vergleichbar — der `docs`-Abschnitt zeigt hier bei ähnlicher KL nur
+0,09 % PPL.

Der Kern der Lücke: wir rotieren bereits (festes, orthonormales H) und legen dann das **uniforme** ggml-`q4_0`-Gitter
darüber, mit **einer Skala je 32 Werte**. Nach der Rotation sind die Werte aber näherungsweise gaußverteilt — ein
uniformer Gitter-Quantisierer mit 16 Stufen ist dafür nicht der passende (und der per-32-Skalierer re-normalisiert
jeden Block noch einmal, was den Verteilungsbezug zerstört).

## 4. Die Kandidaten (Wirkung gegen Aufwand) — Stand Plan v2 (2026-10-14, nach Review §6.12 des Surveys)

| # | Kandidat | Belegte Wirkung | Aufwand | Datenblind? | Stand |
| --- | --- | --- | --- | --- | --- |
| **A** | **Asymmetrisch-4-Bit im Direktvergleich** (§5.5): K/V verschiedene Skalenregel bei gleichen Bytes + Bit-Asymmetrie K16/V8 + No-Rotation-Ablation | Richtung K-will-Bits (Sonde §7.5); KIVI: K pro-Kanal / V pro-Token; KVQuant <0,1 PPL bei 3 Bit | S–M | ja | **WP-1 neu, läuft zuerst** |
| **B** | **Clipped-Scale**: Skala aus κ-Quantil statt `max\|x\|`, sonst identische 18-B-Blöcke | OSCAR-Tab. 6: κ aus Kalibrierung 0,15 am Grid-Besten; FlatQuant-LCT +0,84 PPL | S (nur Schreiber) | Schwellen fix = ja / kalibriert = opt-in | **WP-2a, vor jedem Layout** |
| **C** | **Fenster/Sink**: jüngste/erste Zellen exakt | KIVI: Fenster trägt 2 Bit (ohne → GSM8K-Kollaps); OSCAR-Tab. 5: (0,0) = 0,00 überall, Knie (64,256) | M (q2k-Mechanismus existiert) | ja | **WP-2a, vor jedem Layout** |
| **D** | **Echtes asymmetrisches Layout** (K4/V3-Klasse, echte Bytes, ehrliche BPE-Bilanz) mit vorbereitetem, unbestücktem Slot für kalibriert-nur-K | A–C entscheiden die Form | L | ja (+ opt-in Slot) | WP-2, nach A–C |
| **E** | **Kalibriert-nur-K als Vergleichsarm** (WUSH geschlossen / OSCAR Kovarianz; V fest; **kein V-Folding**) | WUSH 2-Bit 10,51 vs H 258,29; OSCAR 2,28 BPE ≈ BF16 auf 32B; bei 4 Bit 0,04 (kein Hebel) | M–L + Kalibrierung (~8k Tokens nach OSCAR-Tab. 7) | nein (opt-in) | WP-3, **Vergleich, nicht Hauptlinie** |
| **F** | **No-Rotation-Ablation + SRHT** | OSCAR-Ablation: ohne H 51,74 vs voll 70,01 (H trägt dort!); nur U 32,82 | S | ja | mitlaufend (Skepsis wird gemessen) |

**Empfehlung (Plan v2):** **A zuerst im Direktvergleich auf Swift** — einziger Kandidat mit eigener Sonden-Evidenz
und ohne Kalibrierungs-/Download-/Byte-Kosten. **B+C vor jedem Layout** — Null-Byte-Arme, die entscheiden, was das
Layout überhaupt tragen muss (ohne sie attribuiert man Fenster-/Clipping-Gewinne der Transformation). **D
danach**, **E daneben als Vergleichsarm** (WUSH bleibt Vergleich, wird nicht Hauptlinie), **F mitlaufend**.
Explizit nicht: QJL, symmetrisch-2-Bit, V-Folding in quantisierte Gewichte (§10).

## 5. Design

### 5.1 `--kv q4lm` (Kandidat A)

Neues, opt-in Format mit eigener Format-ID (`kKvQ4lm`). Die Rotation bleibt `fwht256` (orthonormal, selbstinvers)
für K, V und Q — damit bleiben die Scores exakt `<Hq, Hk> = <q, k>`, und der Ausgang wird mit demselben H
zurückgedreht.

Block = **ein Head-Vektor (256 Werte)** statt 8 × 32:

| Feld | Bytes | Inhalt |
| --- | --- | --- |
| `s` | 2 | fp16-Skala `s = ‖x‖ · amax(H·x̂) / C16`, `C16` = äußerster Codebook-Wert (≈ 2,7326) |
| `qs` | 128 | 256 4-bit-Indizes, LSB-first |
| **Σ** | **130** | **4,06 bit/Wert** (statt 144 B / 4,5 bit) |

Quantisieren: `norm = ‖x‖`, `x̂ = x/norm`, `y = H·x̂`, `amax = max|y|`, `s = norm·amax/C16`,
`idx[j] = argmin_k |y[j]·(C16/amax) − cb[k]|`.
Dequantisieren: `value = cb[idx] · s` (die Norm ist in `s` eingefaltet, ein einziger Skalenwert je Head).

Codebook: 16 feste Konstanten (Max-Lloyd für N(0,1), dieselben, die quant.cpp benutzt), als `constexpr`/`__constant__`
Tabelle. Der Aufmerksamkeitspfad dequantisiert genau wie heute bei `q4_0` (`KV_MODE 4`); nur die Abbildung
Nibble→Wert wird `(q − 8)·d` → `cb[q]·s`.

Byte-Bilanz je Zelle (2 KV-Heads, K+V): `q4lm` **520 B** gegen `q4_0` 576 B (−9,7 %) und int8 1.056 B.

Zwei Kontroll-Varianten, um Ursachen zu trennen (nur intern, kein Extra-Format):
- `q4lu`: wie `q4lm`, aber **uniformes** 16-Level-Gitter — isoliert den Skalenwechsel 32→Head;
- `q4lm32`: per-32-Block wie heute, nur Gitter→Codebook — isoliert den Codebook-Effekt.

**Nachtrag (14.10.2026): gebaut wird die Kontroll-Variante, und sie ist das Produkt-Format.** Der erste Meßpunkt
(§7.1/§7.2) hat die Richtung entschieden: der Fehler ist ein Per-Token-Quantisierer-Effekt **ohne**
Langkontext-Term, es zählt also allein, wie gut ein Vektor aus 256 Werten abgelegt wird. Damit ist Kandidat A
richtig — aber die Ausführung aus §5.1 (ein 130-B-Block je **Head**, 4,06 bit/Wert) kostet in **jedem** Backend ein
neues Layout: neue Blockgröße, neuer Block-Mover (`kv_stream`), neues Streaming-Granulat, neue Snapshot-Regeln, neue
Prompt-MMA-Operanden. Die per-32-Variante dagegen behält die 18-B-Blöcke, die Pools, das Granulat und die
Ganzzahl-Operanden des Prompt-MMA und ändert **nur die Nibble→Wert-Abbildung**. Deshalb wird `q4lm32` nicht als
interner Kontrollarm gebaut, sondern als das Format; `q4lm` (ein Head je Block) wird **nicht** gebaut.

Der Trick, der das ermöglicht: die 16 Level werden als `round(level · 32)` ganzzahlig gespeichert —
`{−87,−66,−52,−40,−30,−21,−12,−4,4,12,21,30,40,52,66,87}`, d. i. Max-Lloyds 16 Rekonstruktionspunkte für N(0,1)
(dieselbe Tabelle wie quant.cpp, aus TurboQuants `tq_codebook.c`) × 32, gerundet — und die Skala als
`d = max|x| / 87`. Die Nibble bleiben Nibble, der Block bleibt 18 B, 8 Blöcke je Head.

Der Survey (§6.4 Punkt 2: „das Codebook allein ist nicht der Gewinn, die Skala gehört dazu") wird von der eigenen
Parity-Messung (§7.3) bestätigt und **getrennt** — deshalb gibt es **zwei** Formatwerte:

| `--kv` | Skala je 32er-Block | Nibble→Wert | Bytes/Block |
| --- | --- | --- | ---: |
| `q4_0` (Bestand) | `d = mval / −8` (ggml, uniform) | `(n − 8) · d` | 18 |
| `q4lm32` | `d = max|x| / 87` | `kQ4lMap[n] · d` | 18 |
| `q4lm32s` | `d* = Σ(l·x) / Σ(l²)`, **ein** Schritt nach `max|x|/87` | `kQ4lMap[n] · d` | 18 |

`q4lm32s` ist ebenfalls datenblind: der eine Least-Squares-Schritt benutzt nur den Block selbst (zwei zusätzliche
Butterfly-Reduktionen über die 32 Werte), keine Kalibrierung, kein Lernschritt, und die zweite Zuweisung läuft mit
demselben Level-Ladder. Beide Level-Tabellen-Formate **lesen** gleich (`q4_lloyd(mode)` in `kv_q4.hpp`), weil die
Skala im Block steht; nur der Schreiber unterscheidet sich.

Öffentliche API: `enum class Q4Mode : int { kUniform = 0, kLloyd = 1, kLloydLs = 2 }` in
`include/strata/kernels/kv_q4.hpp`; alle vier öffentlichen `kv_q4`-Funktionen nehmen `Q4Mode mode =
Q4Mode::kUniform` als letzten Parameter (Default = unverändert). `qsa_kv_format` liefert die neuen Format-IDs
`kKvQ4lm` (4) und `kKvQ4lmS` (5), damit Streaming/Snapshot „dieselben Blöcke, andere Tabelle" mitführen. Der
Prompt-Pfad **lehnt beide Formate ab** (`qsa_prompt_attn_batch` gibt `false` zurück), sodass Prefill über den
Decode-Kernel läuft: kein Tensor-Core-Prefill für die opt-in-Formate, dokumentierter Tausch. `kQ4lMap` liegt auf
CUDA/HIP als `__constant__`; die 16 Bytes divergent zu lesen ist der Worst Case für `__constant__` (für IQ4_NLs
Codebook mit 2,12× gemessen, `s_gemv.cu`) — eine Kopie in Shared Memory ist der nächste Schritt, **vor** jedem
Speed-A/B.

### 5.2 `--kv-window N` / `--kv-sink S` (Kandidat B)

Auf `q4_0` und `k8v4`: die jüngsten `N` Zellen (und `S` Sink-Zellen am Anfang) liegen **fp16** vor, der Rest
4-bit. Kontext-invariant: `N = 128` bleibt bei 8K, 32K und 128K gleich; der Zuschlag ist `128/Zellen` (bei 32K
0,4 %). Beleg: quant.cpp, 3.970 Tokens — 4-bit flach +3,1 %, 4-bit + 128-Token-Fenster −0,1 %, FP32-Referenz.

Umsetzung mit dem, was schon da ist: `kv_stream` hat Host-Kopie, Block-Residency, `page_table` und die Selektion.
Die Auswahl bildet jüngste/erste Zellen auf einen kleinen fp16-Pool ab; alles andere läuft unverändert. Genau der
Ansatz von **#2318** (dort `--kv-window`/`--kv-sink` für `q2k`) — vor Baubeginn dort abstimmen.

### 5.3 SRHT (Kandidat C)

`STRATA_KV_SRHT=1`: eine feste ±1-Vorzeichen-Permutation (aus festem Seed) vor und nach `fwht256`, eigenes Gate,
Default aus. Beachte: quant.cpp fährt die Rotation mit **konstantem** Seed, ist also selbst kein echtes SRHT
(§6.4 des Surveys).

### 5.4 Die 3-Bit-Sonde (WP-2a-Sonde) — erst messen, dann bauen

**Die Frage.** Der Survey (§6.9) hat den Transform-Hebel bei 4 Bit erledigt: WUSH-KVs kalibrierte Transformation
bringt dort 9,77 → 9,73 gegen fp16 9,72, während sie bei 3 Bit 10,06 → 9,82 und bei 2 Bit 258,29 → 10,51 holt.
Der Sprung muss also aus der **Bit-Zuteilung** kommen — und die Literatur widerspricht sich genau darin: llama.cpp
**#2318** mißt K2c+V4 **besser** als K4+V2c, arXiv **2502.15075** hat K4V2 bei 98,3 % der K4V4-Genauigkeit. Beide
können für ihre Cache-Geometrie recht haben; **welche für unseren 256er-Head mit FWHT-256 gilt, sagt keine von
beiden.**

**Die Sonde.** Statt sofort ein 3-Bit-Layout zu bauen, wird die 3-Bit-**Codebook**-Frage in den **unveränderten
4-Bit-Bytes** gestellt: derselbe 18-B-Block, dieselbe Rotation, dieselben Leser, nur die 16-Level-Tabelle wird
durch Max-Lloyds 8-Level-Tabelle für N(0,1) ersetzt — `{±69, ±43, ±24, ±8}`, `d = max|x|/69`, abgerundet nach
derselben ×32-Regel wie `kQ4lMap`. Die Zahl 576 B/Zelle bleibt damit **gleich**: die Sonde mißt **Qualität**, nicht
Bytes. Das Byte-Layout kommt erst in WP-2.

Drei Modi, damit die K-vs-V-Frage bei **gleicher mittlerer Breite** gestellt wird:

| `--kv` | K-Pegel | V-Pegel | Bits, Ø über K/V | was sie beantwortet |
| --- | ---: | ---: | ---: | --- |
| `q4lm32` | 16 | 16 | 4,0 | Referenz (schon gemessen) |
| `q4lm8` | 8 | 8 | 3,0 | wie weit trägt 3 Bit überhaupt |
| `q4lm8k` | 8 | 16 | 3,5 | K-seitig gespart |
| `q4lm8v` | 16 | 8 | 3,5 | V-seitig gespart |

**Warum das die richtige Frage-Reihenfolge ist.** `q4lm8` sagt, ob 3 Bit auf diesem Modell überhaupt lebensfähig
sind; `q4lm8k` gegen `q4lm8v` ist **byte-exakt gleich** und beantwortet #2318-vs-2502.15075 direkt an unserem
Modell, ohne dass ein Byte-Layout-Entwurf vorher festgelegt werden müßte. Erst wenn Zuteilung (§5.5), Schwanz
(§5.6, Clipped-Scale) und Fenster (§5.6) stehen, bekommt das Layout in WP-2 seine Form — mit vorbereitetem, aber
unbestücktem Slot für kalibriert-nur-K (WP-3 als Vergleich, nicht Hauptlinie).

**Die Sonde ist Wegwerf-Code.** Sie ist als solcher gekennzeichnet, lebt nur auf `KV_QUANT_research` und geht
**nicht** in den WP-1/WP-2-PR.

### 5.5 Asymmetrisch-4-Bit im Direktvergleich (WP-1 neu)

Drei Arme, alle auf **Swift**, cwe-2k (+ Power nach §8), alle gegen `q4_0`/`q4lm32`/`int8`/fp16, alle in
Sonden-Bytes (576 B/Zelle) — wegwerfbar bis auf den Gewinner, der in WP-2 echte Bytes bekommt:

1. **Behandlung.** K `max|x|`/32 wie bisher, V RMS/32 (Norm-ähnlich). Gleiche Bytes, nur die Skalenregel je
   Tensor — der Arm prüft, ob V das wirklich billiger sieht als der MSE (Parity: RMS-Skala −0,82 dB gegen
   `max|x|`, §7.3). KIVI/KVQuant-Behandlung (K pro-Kanal-Skalen vor Rotation) wäre der nächste Schritt, nicht
   dieser: erst klären, ob die Skalenregel-Asymmetrie allein trägt.
2. **Bit.** K16/V8 (`q4lm8v`-Sondenstand als Referenzarm, Sonden-Bytes) — die Sonden-Richtung als wiederholter
   Arm auf Swift, nicht als Behauptung aus Q2_0.
3. **Ablation: ohne Rotation.** `q4lm32` ohne FWHT (Schreiber + Leser, Tabelle/Skala gleich, Parität gegen
   Host-Referenz wie üblich). Die Hadamard-Skepsis wird gemessen, nicht gemeint — in beide Richtungen:
   OSCAR-Ablation (INT2, deren Pipeline) sagt H trägt ~18 Punkte, das ist das Gegengewicht, kein Urteil über
   unsere Geometrie.

### 5.6 Clipped-Scale + Fenster vor dem Layout (WP-2a neu)

Zwei Arme, beide **ohne Byte-Änderung**, beide vor jedem Layout-Bau — sie entscheiden, was das Layout überhaupt
tragen muss:

1. **Clipped-Scale.** Skala aus dem κ-Quantil statt `max|x|` (Start κ≈0,92–0,96 — OSCAR-Tab. 6 und
   FlatQuant-fix 0,9/0,95 konvergieren dorthin; dann Grid auf Swift), Tabelle/Blöcke/Leser unverändert. Das
   Clipping greift **nach** der Rotation (der Schreiber sieht post-FWHT-Werte), wie FlatQuant-Tab. 15 verlangt
   (nachher 6,98 vs vorher 7,37; fix fängt ~70 % des Gewinns ohne jede Kalibrierung). Null Bytes, nur Schreiber
   + Parity + cwe-2k-Arm. Schwellen fix = datenblind; aus Kalibrierung = opt-in.
2. **Fenster.** Der q2k-Mechanismus (`--kv-window`/`--kv-sink`, vgl. #2318) auf die Sonden-Arme erweitert:
   jüngste S + erste R Zellen exakt, Rest quantisiert. Misst den Fensterbeitrag bei 3–4 Bit direkt auf unserem
   Modell (OSCAR-Tab. 5: (0,0) ist tot; wo unser Knie liegt, wird gemessen, nicht übernommen).

## 6. Berührte Dateien und Backends (Landkarte am Beispiel `q4_0`)

- Kern/API: `include/strata/kernels/kv_q4.hpp`, `include/strata/kernels/kv_stream.hpp` (Enum),
  `include/strata/core/layer.hpp` (`qsa_kv_format`)
- **CUDA und HIP** (dieselbe `.cu`): `src/kernels/cuda/kv_q4.cu`, `kv_stream.cu`, `qsa_prompt_attn.cu`
  (`KV_MODE 4`), `qsa_decode_attn.cu`
- **SYCL**: `sycl/src/kernels/cuda/kv_q4.dp.cpp`, `kv_stream.dp.cpp`, `qsa_prompt_attn.dp.cpp`,
  `qsa_decode_attn.dp.cpp`
- Dispatch/Layout/Streaming/State: `src/core/layer.cpp` / `.hpp`, `src/core/conversation_snapshot.cpp`,
  `src/core/mtp.cpp`
- CLI/UI: `src/program/generate.cpp` (`--kv`-Liste und Usage), ggf. `serve/web/app.js`
- Tests: `src/kernels/kv_q4_parity.cpp`, `kv_stream_parity.cpp`, `kv_hybrid_parity.cpp`, `kv_steps_parity.cpp`
  (jeweils mit SYCL-Zwilling)
- **Metal**: nur Stub (`metal/src/engine/stubs.cpp`) — keine Arbeit

Regel aus [AGENTS.md](../AGENTS.md): **jede** von einer Datei berührte Backend-Art (CUDA, HIP, SYCL) bauen, bevor
Review.

**WP-2a-Sonde (Wegwerf, §5.4/§7.5)** berührt dieselbe Landkarte, aber **nur CUDA/HIP**: die SYCL-Schreibseite
bleibt absichtlich unangefaßt, und die SYCL-`--kv`-Prüfung lehnt die drei neuen Flags mit Exit 2 ab — sie kann
also nicht stillschweigend falsch gelesen werden. Damit ist der Zweck der Regel (kein Backend unbemerkt brechen)
erfüllt, ohne einen hier nicht baubaren SYCL-Build zu behaupten.

## 7. Arbeitspakete

| WP | Inhalt | Deliverable | Aufwand |
| --- | --- | --- | --- |
| **WP-0** | KV-Qualitäts-Harness: **steht** — `tools/quant_quality.py --kv FMT` (§7.1). Baselines fp16/int8/q4_0/k8v4 auf **Swift** fixieren (Top-1, KL, PPL + Power nach §8 + Speed-A/B) | `bench/results/<datum>-kv-baseline/` | S |
| **WP-1a** (erledigt) | `--kv q4lm32` / `--kv q4lm32s` — Format steht, Messberichte §7.3/§7.4 + bench | — | — |
| **WP-2a-Sonde** (erledigt) | 3-Bit-Sonde Q2_0 — Richtung K-will-Bits, §7.5 + bench | — | — |
| **WP-1** (neu) | **Asymmetrisch-4-Bit im Direktvergleich** (§5.5, §7.6): Behandlung K/V + Bit K16/V8 + No-Rotation-Ablation, Swift, Power nach §8 | Messbericht + Gewinner-Arm | M |
| **WP-2a** (neu) | **Null-Byte-Arme vor dem Layout** (§5.6, §7.6): Clipped-Scale + Fenster auf Sonden-Armen | Messbericht (was das Layout tragen muss) | S–M |
| **WP-2** | **Echtes asymmetrisches Layout** (K4/V3-Klasse, echte Bytes, ehrliche BPE-Bilanz nach OSCAR) mit vorbereitetem, unbestücktem Slot für kalibriert-nur-K. **Bau erst nach WP-1 + WP-2a** | Format + Tests + Messbericht | L |
| **WP-3** | **Kalibriert-nur-K als Vergleichsarm** (WUSH geschlossen / OSCAR Kovarianz; V fest; kein V-Folding; Kalibrierung ~8k Tokens). **Vergleich, nicht Hauptlinie** | Vergleichsarm + Messbericht | M–L |
| **WP-4** | `STRATA_KV_SRHT=1` | Toggle + Messbericht | S |
| **WP-5** (später) | Delta-Modus für 3-bit; Per-Layer-Bit-Zuteilung nach Kurtosis/Sensitivität | — | L |

Reihenfolge: **WP-0 → WP-1 → WP-2a → WP-2 → WP-3 → (WP-4 ‖ WP-5).** Jedes WP ist **ein** PR (AGENTS.md: eine
Änderung je PR). WP-2a-Sonde ist die Ausnahme: sie ist ausdrücklich Sonden-Code und geht in **keinen** PR.

### 7.1 WP-0: der Harness steht, der erste Meßpunkt liegt vor

`tools/quant_quality.py` pinnte `--kv` fix auf `fp16` und warf `--kv`/`--kv-resident` aus der Config — damit war **kein**
KV-Format meßbar. Seit Commit `f275f3cf` ist `--kv FMT` ein Lauf-Argument (Default `fp16`, also unverändertes
Verhalten), und die Engine-Umgebung benutzt unter Windows `PATH` statt `LD_LIBRARY_PATH` — wie `serve/server.py`s
`child_env` (`94d31866`). Beides sind nur Werkzeug-Änderungen; der Default-Pfad der Engine bleibt unberührt.

Erster Meßpunkt (Gewichte fest, nur die K/V-Ablage variiert):

    python tools/quant_quality.py run --config strata-q2_0.json --kv fp16 --out kv-fp16.tsv
    python tools/quant_quality.py run --config strata-q2_0.json --kv int8 --out kv-int8.tsv
    python tools/quant_quality.py run --config strata-q2_0.json --kv q4_0 --out kv-q4_0.tsv
    python tools/quant_quality.py compare kv-fp16.tsv kv-int8.tsv kv-q4_0.tsv --segments

Referenz ist der **fp16-KV-Arm bei gleicher Config**, **nicht** Q8_0: sonst mischt der Vergleich Gewichts- und
KV-Fehler. `run` hat keine Segment-Auswahl (jeder Arm = 5×999 Positionen teacher-forced, ~2,5 min), das ist der Preis
für den festen, committeten Tokensatz.

**Ergebnis** (Q2_0, gfx1100, 0.1.44.1 HIP; 4.995 Positionen; `bench/results/2026-10-14-kv-first-run/README.md`):

| KV | Top-1 | mittlere KL | PPL | gegen fp16 |
| --- | ---: | ---: | ---: | ---: |
| int8 | 96,48 % | 0,0136 | 4,469 | **−0,10 %** |
| q4_0 | 94,15 % | 0,0364 | 4,524 | **+1,12 %** |

Das ist die Zahl, die WP-1 schlagen muß (q4_0-Lücke, hier +1,12 % PPL / KL 0,0364 = 2,7× int8). Achtung: die
September-Zahl „+8 % @1K“ (`2026-09-27-kv-q4`) stammt von einem anderen, seitentfernten Werkzeug auf einem einzelnen
langen Dokument und ist mit diesem Set **nicht** vergleichbar — deshalb jetzt zwei Zahlen statt einer.

Zweiter, **modellfreier** Meßpunkt: die KV-Parity-Tests (`kv_q4_parity`, `kv_q8_parity`, `kv_hybrid_parity`,
`kv_stream_parity`, `kv_steps_parity`) — GPU, synthetisch, in ctest. `kv_hybrid_parity` boundet die quantisierte
Attention **schon gegen die fp32-true** Attention; dort gehört die entsprechende Schranke für `q4lm`/`q4lu` hin.
Sie sind das Korrektheits-Gate, nicht der Qualitätsmaßstab.

### 7.2 WP-0, zweiter Meßpunkt: 8K und die Kontextlängen-Frage

Das 1K-Set kann die Frage nicht beantworten, die für ein KV-Format am wichtigsten ist: **wächst der Fehler mit der
Kontextlänge?** Dafür steht jetzt `data/quant_quality/v2-8k` — docs/code/multilingual, **8.191 Positionen je
Segment (24.573)**, nur Material aus diesem Repo (`ad4deb11`) — samt 1K-Armen (`set-1k.ids.json`, die ersten 1.000
Tokens derselben Texte).

    python tools/quant_quality.py run --config strata-q2_0.json --kv fp16 --ids data/quant_quality/v2-8k/set.ids.json --out kv-fp16.tsv
    python tools/quant_quality.py run --config strata-q2_0.json --kv int8 --ids data/quant_quality/v2-8k/set.ids.json --out kv-int8.tsv
    python tools/quant_quality.py run --config strata-q2_0.json --kv q4_0 --ids data/quant_quality/v2-8k/set.ids.json --out kv-q4_0.tsv
    python tools/quant_quality.py compare kv-fp16.tsv kv-int8.tsv kv-q4_0.tsv --segments --json compare.json
    python tools/quant_bins.py bins --ref kv-fp16.tsv --run kv-int8.tsv --run kv-q4_0.tsv --bin 1000 --json bins.json

**Ergebnis** (Q2_0, gfx1200 RX 9060 XT, 0.1.44.1 HIP; 24.573 Positionen;
`bench/results/2026-10-14-kv-8k/README.md`):

| KV | Top-1 | mittlere KL | KL p99 | PPL | gegen fp16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| int8 | 97,04 % | 0,00639 | 0,0692 | 3,3066 | **−0,14 %** |
| q4_0 | 95,35 % | 0,01652 | 0,1843 | 3,3183 | **+0,21 %** |

Je Segment: int8 −0,26 % / −0,28 % / +0,12 % (docs/code/multilingual), q4_0 +0,15 % / +0,49 % / −0,00 %. Das ist
dieselbe Ordnung wie der erste Meßpunkt, aber auf **5× mehr Positionen**: int8 ist gratis, q4_0 kostet ~0,2 % PPL und
verliert im Ausläufer (KL p99 0,184 gegen 0,069).

Zur Kontextlänge zwei Messungen:

1. `quant_quality.py run` schickt jeden Abschnitt als **eigenen `GEN`-Befehl** und bemisst `--max-context` auf
   `seg_tokens+512` — Position ist also Kontextlänge. Belegt mit dem neuen `tools/quant_bins.py segments` auf der
   Sonde `2026-10-14-kv-ctxprobe`: dieselben 1.200 Tokens sind als eigener Abschnitt und als erste 1.200 Tokens eines
   8.192er Abschnitts **bit-identisch** (1199/1199, 0,000e+00); dieselben `code`-Tokens hinter 7.000 Tokens `docs`
   sind es **nicht** (KL 0,118, 92,33 % gleicher argmax) — die Engine liest den langen Kontext wirklich, es ist kein
   verstecktes Aufmerksamkeitsfenster.
2. Die mittlere KL je 1.000er-Positionstopf steigt in **keinem** Abschnitt: `docs` 0,0169/0,0174, dann 0,0213–0,0288;
   `multilingual` ist in den letzten 2.191 Positionen nahe null (0,0008–0,0055); `code` ohne Trend. Die Streuung
   zwischen den Töpfen eines Abschnitts ist so groß wie die zwischen den drei Abschnitten im selben Topf
   (0,0140–0,0172) — der Fehler folgt dem Text, nicht der Länge.

**Folge für den Plan:** Der Fehler ist ein **Per-Token-Quantisierer-Effekt ohne Langkontext-Term**. Ein 1K-Set ist für
den Mittelwert so gut wie ein 8K-Set, und ein 32K-Punkt wird für den Mittelwert nicht gebraucht (§8.2 angepasst); für
WP-2/`--kv-window` bleibt er nötig, weil dort die *jüngsten* Zellen zählen. Der Hebel ist die Genauigkeit eines
256-Werte-Vektors — genau WP-1/§5.1; die `q4_0`-Zahlen oben sind dessen Baseline.

**Korrektur:** das `v2-8k`-README behauptete, das 1K/8K-Paar trenne die Kontextlänge; für dieses Werkzeug tut es das
nicht (`data/quant_quality/v2-8k/README.md` korrigiert).

### 7.3 WP-1a: `--kv q4lm32` / `--kv q4lm32s` — gebaut und modellfrei gemessen

Design und Bauentscheidung: §5.1-Nachtrag. Kurz: **dieselben 18-B-Blöcke wie `q4_0`**, nur die Nibble→Wert-Abbildung
ändert sich (`q4lm32`: `kQ4lMap[n]·d` mit `d = max|x|/87`), und `q4lm32s` zusätzlich die Skalenregel des Schreibers
(**ein** Least-Squares-Schritt, `d* = Σ(l·x)/Σ(l²)`, danach dasselbe Level-Ladder). Weder Layout noch Byte-Zahl noch
Leser ändern sich zwischen den beiden Level-Tabellen-Formaten.

**Der modellfreie Meßpunkt** (`src/kernels/kv_q4_parity.cpp`, HIP auf gfx1200 / RX 9060 XT). Der Test wiederholt den
Schreiber über 487 gezogene Zellen (2 KV-Heads × 256 Werte × K und V = **498.688 rotierte Werte**, 15.584 Blöcke) und
vergleicht **bitweise** gegen eine Host-Referenz; beide neuen Formate sind bitweise gleich, und der Gather liefert
exakt `Level · gespeicherte Skala`. Auf genau denselben Werten, dieselben Blöcke, je Block die beste der möglichen
Skalen:

| Konfiguration (dieselben 16 Stufen) | MSE | gegen `q4_0` |
| --- | ---: | ---: |
| `q4_0` (uniformes Gitter, ggml-Skala) | 2,938e−02 | 0 dB |
| `q4lm32` (Lloyd-Level, `max|x|/87`) | 2,740e−02 | **−0,30 dB** |
| dieselben Level, Skala `RMS/32` | 3,547e−02 | −0,82 dB (**schlechter**) |
| dieselben Level, konvergierte Best-Fit-Skala je Block | 2,456e−02 | −0,78 dB (Obergrenze) |
| `q4lm32s` (Level + **ein** LS-Schritt) | 2,551e−02 | **−0,61 dB** |

Lesart: die **Level-Tabelle allein bringt 0,30 dB**, die **Skalenregel noch einmal 0,31 dB** — der Survey (§6.4
Punkt 2) hatte recht, dass beides zusammengehört. Ein Least-Squares-Schritt erreicht **78,8 %** des Wegs zur
konvergierten Skala; die Skala selbst liegt danach im Mittel nur **0,4 %** neben `max|x|/87` — der Schritt kauft
also die **Anpassung**, nicht einen größeren Wertebereich. Beides ist **kein** fixes Umskalieren der Level: mit
`RMS/32` als Skala wird der Fehler **schlechter** als `q4_0`, weil der Block-Maximalwert aus 32 Stichproben im
Mittel bei `max|x|/RMS = 2,363` liegt, also unter den 2,72 σ, für die das breiteste Level gezeichnet ist.

Verhältnis zur Literatur (§6.4 Punkt 2 / §6.7 Punkt 2 des Surveys): dort war die Erwartung „Codebook allein ~1 %
relativ, teils abhängig vom Wechsel 32→Head". Bei uns kommt die Größenordnung schon aus dem **per-32**-Format,
sobald Tabelle **und** Skalenschritt zusammenkommen — der Skalenwechsel 32→Head (der `q4lm`-Teil von §5.1) wird
deshalb **nicht** gebaut.

**Backends.** CUDA und HIP teilen dieselbe `.cu`; HIP ist gebaut und getestet. SYCL ist gespiegelt
(`kv_q4.dp.cpp`, `qsa_decode_attn.dp.cpp`, `kv_stream.dp.cpp`, `qsa_prompt_attn.dp.cpp` und die Host-Kopien unter
`sycl/src/`), inklusive Parity-Zwilling — auf dieser Maschine **nicht baubar** (kein oneAPI), also **ungetestet**;
die SYCL-Schreibseite nimmt den Modus als Laufzeit-Argument statt als Template-Parameter (im Code als bewusste
Abweichung begründet, weil eine falsche Template-Instanziierung hier nicht auffallen könnte). Metal hat nur einen
KV-Stub: keine Arbeit.

### 7.4 WP-1a: der A/B bei 8K und 1K

**Aufbau.** `tools/quant_quality.py run` mit derselben Config wie der Meßpunkt in §7.2, je Lauf nur `--kv` gewechselt
(`fp16`, `int8`, `q4_0`, `q4lm32`, `q4lm32s`), auf `data/quant_quality/v2-8k/set.ids.json` — dieselben drei 8K-Texte,
**8.191 Positionen je Abschnitt, 24.573 gesamt**. Referenz ist der **fp16-KV-Arm desselben Laufs** (Gewichte fix,
gleiche Config), nicht `int8` und nicht der Vorbau. Gemessen mit dem Quellbau (`build-hip-win\strata.exe`, HIP auf
gfx1200 / RX 9060 XT), nicht mit dem Vorbau.

| Arm | Top-1 | mittl. KL | KL p99 | PPL | PPL gegen fp16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp16` (Referenz) | — | 0 | 0 | 3,31026 | — |
| `int8` (Default) | 97,04 % | 0,006290 | 0,0675 | 3,30669 | −0,11 % |
| `q4_0` | 95,27 % | 0,016584 | 0,1833 | 3,31781 | **+0,23 %** |
| `q4lm32` | 95,46 % | 0,015788 | 0,1648 | 3,31147 | **+0,04 %** |
| `q4lm32s` | 95,50 % | 0,015162 | 0,1603 | 3,31078 | **+0,02 %** |

Je Abschnitt: PPL gegen den fp16-Arm, mittlere KL in Klammern (fp16-PPL als Absolutwert):

| Abschnitt | fp16-PPL | `int8` | `q4_0` | `q4lm32` | `q4lm32s` |
| --- | ---: | ---: | ---: | ---: | ---: |
| docs | 6,79727 | −0,27 % (0,0085) | +0,06 % (0,0230) | −0,08 % (0,0214) | −0,41 % (0,0203) |
| code | 2,05395 | −0,16 % (0,0049) | +0,57 % (0,0139) | +0,70 % (0,0140) | +0,74 % (0,0137) |
| multilingual | 2,59814 | +0,11 % (0,0054) | +0,06 % (0,0129) | −0,51 % (0,0119) | −0,28 % (0,0115) |

**Was 8K zeigt.**

- `q4lm32s` **schließt die `q4_0`-Lücke ganz**: +0,23 % → **+0,02 %** PPL, also praktisch fp16-Niveau — bei
  **unverändertem** Layout und **gleicher** Byte-Zahl wie `q4_0` (576 B/Zelle, kein Byte gespart).
- Mittlere KL −8,6 % (0,016584 → 0,015162), KL p99 −12,5 % (0,1833 → 0,1603).
- `q4lm32` (nur Tabelle, `max|x|/87`) holt den größten Teil: PPL +0,04 %, KL −4,8 %.
- Der zweite Hebel (Skala) legt noch einmal **−4,0 %** KL darauf (PPL −0,02 %). Das ist deutlich weniger, als die
  **−0,31 dB MSE** aus §7.3 erwarten lassen: der Per-Block-Fehlergewinn überträgt sich **nicht 1:1** auf KL/PPL.
- Top-1 trennt die Formate in dieser Größe **nicht**: der Vorsprung von `q4lm32s` vor `q4_0` (95,50 % gegen 95,27 %)
  ist so groß wie die Reproduktionslücke zwischen zwei Läufen (unten: 0,08 Prozentpunkte für `q4_0`).
- Der Gewinn sitzt in `docs` (−0,41 %) und `multilingual` (−0,28 %). Auf `code` zahlt **jedes** q4-Format 0,6–0,7 %
  PPL, und die neuen Formate senken das **nicht** — auf `code` sind sie in PPL minimal schlechter als `q4_0`, in KL
  besser. Kein Anspruch auf Gleichheit über die Abschnitte.

**1K-Paar.** Dieselben drei Texte, je **999 Positionen** (`set-1k.ids.json`), fp16-Referenz 3,74137. Der 1K-Punkt ist
ein Reproduzierbarkeits-Check, **kein** Kontextlängen-Vergleich (§7.2):

| Arm | Top-1 | mittl. KL | KL p99 | PPL | gegen fp16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `int8` | 96,96 % | 0,007032 | 0,0755 | 3,75142 | +0,27 % |
| `q4_0` | 95,33 % | 0,016052 | 0,2173 | 3,78984 | **+1,30 %** |
| `q4lm32` | 95,06 % | 0,014846 | 0,1713 | 3,74192 | **+0,02 %** |
| `q4lm32s` | 95,13 % | 0,015313 | 0,1603 | 3,74178 | **+0,01 %** |

Auch auf 1K schließt die Level-Tabelle die `q4_0`-Lücke (+1,30 % → +0,02 %), und beide neuen Formate liegen
praktisch auf fp16-Niveau. Zwei ehrliche Nuancen dazu:

- Auf 1K **verliert** `q4lm32` Top-1 gegen `q4_0` (95,06 % gegen 95,33 %), **gewinnt** aber KL (−7,5 %) und p99
  (−21 %). Auf 8K dreht sich die Top-1-Reihenfolge wieder um. Bei diesem Effekt ist Top-1 kein stabiles Kriterium,
  die KL ist es.
- Der **zweite Hebel ist bei 1K nicht gleichgerichtet**: `q4lm32s` hat dort eine um **3,1 % höhere** mittlere KL als
  `q4lm32` (0,015313 gegen 0,014846) und eine um 6,4 % **niedrigere** p99, bei praktisch gleicher PPL (+0,01 % gegen
  +0,02 %). Auf 8K ist die Reihenfolge umgekehrt (KL −4,0 %, p99 −2,7 %). Der Per-Block-Gewinn von **−0,31 dB**
  (§7.3) ist damit gesichert, sein End-zu-End-Effekt aber **klein und nicht vorzeichenstabil** — anders als der der
  Tabelle, die in beiden Sätzen dasselbe Vorzeichen hat.

**Reproduzierbarkeit.** Der committete Vorbau-Lauf (`bench/results/2026-10-14-kv-8k`, prebuilt 0.1.44.1) liest für
`int8` KL 0,006393 / PPL 3,30664 und für `q4_0` KL 0,016519 / PPL 3,31827; dieser Lauf 0,006290 / 3,30669 und
0,016584 / 3,31781, Top-1 97,04/97,04 und 95,35/95,27. Die KL reproduziert auf ~1,6 %, die PPL auf 0,03 %, Top-1 auf
0,08 Prozentpunkte — **aber nicht bit-identisch**: der fp16-Arm liest hier 3,31026 gegen 3,31128 im Vorbau-Lauf
(+0,03 %). Ob das der Binärbau (Quellbau gegen Vorbau) oder die Run-zu-Run-Streuung der Engine ist, ist **nicht**
isoliert. Deshalb stehen **alle** Deltas oben **innerhalb eines Laufs**. Der volle Satz Zahlen liegt in
`bench/results/2026-10-14-kv-q4lm32/` (`compare.json`, `compare-1k.json`).

**Offen vor einem PR:** das Identitäts-Gate (§8.1) und der Speed-A/B (§8.3) — siehe §9.

### 7.5 WP-2a: die 3-Bit-Sonde — gebaut und modellfrei gemessen

Design: §5.4. Gebaut sind drei neue `Q4Mode`-Werte (`kLloyd8`, `kLloyd8K`, `kLloyd8V`), die 8-Pegel-Tabelle in
`kv_q4.hpp`, ein Template-Parameter `N` am Gruppenschreiber (`q4l_group<FIT, N>`, N ∈ {8, 16}), eine
**Laufzeit-Tabellenwahl je Tensor** in den fusionierten Lesern (`q4l_map_k(v)` aus `pools.q4_mode`) und die drei
neuen `--kv`-Strings. Die 16-Pegel-Pfade sind dabei **unverändert**: die Paritätssuite läuft 7/7 mit exakt den
Zahlen aus §7.3 (0,30 dB / 0,61 dB / 78,8 %).

**Der modellfreie Meßpunkt** (`kv_q4_parity.cpp`, Test 7; synthetisch N(0,1), dieselben 498.688 rotierten Werte,
gfx1200). Neben der bitweisen Gleichheit aller drei Arme gegen die Host-Referenz und dem Nachweis, daß der Leser
**je Tensor** die richtige Tabelle zieht:

| Arm | K/V-Pegel | MSE | gegen `q4lm32` (2,740e−02) |
| --- | --- | ---: | ---: |
| `q4lm32` | 16/16 | 2,740e−02 | 0 dB |
| `q4lm8k` | 8/16 | 7,672e−02 | −4,47 dB |
| `q4lm8v` | 16/8 | 7,662e−02 | −4,46 dB |
| `q4lm8` | 8/8 | 1,259e−01 | −6,62 dB |

Zwei Befunde daraus:

- Der 3-Bit-Sprung kostet am Codebook **6,6 dB** MSE — mehr als die 4,09-fache (≈ 6,1 dB) Max-Lloyd-Theorie für
  3-vs-4-Bit, weil die `max|x|`-Skala bei weniger Pegeln relativ teurer wird. `q4lm8` wird also **nicht** gratis
  sein; die Frage ist, ob das Modell es billiger sieht als der MSE.
- **Am Quantisierer sind K8 und V8 ununterscheidbar** (7,672e−02 gegen 7,662e−02, 0,1 %). Beide Tensoren stellen
  die Hälfte der Werte und benutzen dieselbe Tabelle, also ist ihr MSE-Beitrag symmetrisch, und die Differenz
  kann nur aus den leicht verschiedenen Statistiken der beiden gezogenen Tensoren kommen. **Die Entscheidung
  #2318-vs-2502.15075 kann der MSE also nicht treffen** — K geht multiplikativ in die Attention-Scores ein (nach
  Softmax ein *relativer* Fehler), V linear in den Ausgabewert (ein *additiver* Fehler je Position). Nur das
  Modell kennt den Wechselkurs. Genau dafür läuft der Modelllauf.

**Engine-Smoke-Test** (`--tokens 9707,1879,264,1229,13`, `--max-new 6 --greedy`, Q2_0/HIP): alle fünf Arme
(`int8`, `q4lm32`, `q4lm8`, `q4lm8k`, `q4lm8v`) laufen mit Exit 0 durch, und die vier q4-Arme liefern **vier
verschiedene** Ausgaben — mit dem erwarteten Muster: `q4lm8` und `q4lm8k` teilen sich das 8-Pegel-K (gleiche
Scores, gleiche erste Tokens), `q4lm8v` teilt sich das 16-Pegel-K mit `q4lm32`.

**Der Modelllauf** (Entscheidungstest §6.10 des Surveys) ist gelaufen: `data/quant_quality/cwe-2k`, 8 Segmente,
fp16-Referenz, Anker `int8`/`q4lm32` und die drei Sonden-Arme, Q2_0-Gewichte. Er ist selbst-diagnostizierend, und
er bestätigt sich: `q4lm32` reproduziert **bit-genau** die KL 0,02022 (PPL 3,09327), `int8` die 0,01033, die
fp16-Referenz die 3,06695 — die Tabellenverdrahtung stimmt also, und cwe-2k reproduziert über Sitzungen auf 4–5
Stellen (anders als die 8K-Ganzmodell-Menge, die zwischen zwei Läufen um 0,03 % abwich).

Ganzsegmentige KL (16.351 Positionen) und gepaart über die 8 Segmente:

| Arm | K/V-Pegel | mean KL | gegen `q4lm32` | top-1 | PPL vs fp16 | gepaart t (vs `q4lm32`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `int8` | 16/16 (2 B) | 0,01033 | 0,51× | 99,16 % | +0,11 % | −4,19 |
| `q4lm32` | 16/16 | 0,02022 | 1,00× | 98,62 % | +0,86 % | 0 |
| **`q4lm8v`** | **16/8** | **0,02619** | **1,30×** | 98,52 % | **+0,37 %** | **−2,39** |
| `q4lm8k` | 8/16 | 0,02748 | 1,36× | 98,48 % | +0,30 % | −3,25 |
| `q4lm8` | 8/8 | 0,03328 | 1,65× | 98,23 % | +0,81 % | −5,04 |

Die Kernzeile ist `q4lm8k` gegen `q4lm8v` (byte-exakt gleich, gleiche mittlere Breite 3,5 Bit):
**+0,00129 KL, t +0,60, 4/8 Segmente gegenläufig** — der direkte K-vs-V-Vergleich ist mit *diesem* Kriterium allein
**nicht aufgelöst**. Die Richtung zeigt aber in allen drei Auslesungen gleichsinnig, und die Antwortregion trennt
**kategorial entlang K**: alle 16-Pegel-K-Arme lesen 90,98 %, alle 8-Pegel-K-Arme 89,10 % — egal was V tut. Exakter
McNemar (`areport --paired`, dieselben Zeilen): `q4lm32` gegen `q4lm8v` (V 16→8) **2:2, p 1,000** (die Aufgabe
bleibt unberührt); `q4lm8k` gegen `q4lm8v` **0:5, p 0,062** (zweiseitig; einseitig 0,031). Und PPL sieht das
**gar nicht**: `q4lm8` liest +0,81 % gegen `q4lm32` +0,86 % — *besser* —, während seine KL 65 % höher liegt.

**Befund:** K will die Bits, nicht V — die Richtung von arXiv 2502.15075 und des schon ausgelieferten `k8v4`
(#120), und **gegen** die Schlagzeile von #2318. Das ist eine *Richtung, kein etablierter Effekt*: kein Einzel-
kriterium reißt bei 8 Segmenten die konventionelle Schwelle. Vollständige Tabellen, die vierfache Übereinstimmung
der drei Auslesungen und die Reproduktion: `bench/results/2026-10-14-kv-3bit-probe/`.

> **Meßbasis ab 2026-10-14: Swift IQ3_XXS, nicht Q2_0.** Alles oben Gemessene ist Q2_0, und Q2_0 ist der
> **gedämpfteste** Gewichts-Regime (+10,4 % PPL gegen Q8_0, §6.7): sein Gewichtsfehler legt sich auf jede
> KV-Verschlechterung. Der Vergleich mit Q2_0 behält seinen Wert als *untere Schranke* und als Beleg, daß die
> Sonde selbst-diagnostizierend funktioniert — die **Entscheidung K-vs-V wird aber auf Swift IQ3_XXS wiederholt**,
> weil nur dort der KV-Fehler sichtbar ist. Alles Ab hier Neue wird auf Swift gemessen; Q2_0-Zahlen werden als
> solche gekennzeichnet und nicht mehr als Grundlage benutzt.

**Backends.** CUDA und HIP (dieselbe `.cu`), HIP gebaut und getestet. SYCL ist **bewußt nicht angefaßt**: der
SYCL-Port hat einen eigenen Schreiber (`kQ4lMapDev`) und bräuchte eine eigene Dispatch-Verzweigung, und seine
`--kv`-Prüfung lehnt die neuen Flags mit Exit 2 ab — kein Backend kann sie also stillschweigend fehldeuten. Die
additiven Enum-Werte in `kv_q4.hpp` stehen hinter dem `__CUDACC__`/`__HIPCC__`-Guard und ändern SYCL nicht.
Metal hat nur einen KV-Stub.

### 7.6 Plan v2 (2026-10-14, nach Review §6.12 des Surveys)

Der Umbau in einem Absatz: Die alte Reihenfolge (symmetrisch-4-Bit → Sonde → 3-Bit-Layout mit Transform-Slot,
Fenster zurückgestellt, WUSH als Hauptlinie) ist ersetzt durch **asymmetrisch-4-Bit im Direktvergleich (WP-1)
→ Null-Byte-Arme Clipped-Scale + Fenster (WP-2a) → asymmetrisches Layout (WP-2) → kalibriert-nur-K als
Vergleichsarm (WP-3)**. Begründung vollständig in §6.12; die Kurzform:

- Die Sonde hat die Richtung (K-will-Bits), aber kein Urteil — WP-1 wiederholt sie auf Swift **mit** Power
  (§8) und legt Behandlung, Bit und Rotation in drei getrennten Armen auf den Tisch (§5.5).
- Clipping und Fenster sind in der Literatur die tragenden Achsen bei niedrigen Bits (OSCAR-Tab. 5/6, KIVI,
  FlatQuant-Ablation) und kosten **null Bytes** — sie werden gemessen, bevor ein Layout ihre Gewinne der
  Transformation zuschreiben kann (§5.6).
- WUSH bleibt Vergleich, weil V-Folding auf quantisierten Gewichten tot ist und 4 Bit keinen
  Transform-Hebel haben (§6.12 Einwände 1+5); kalibriert wird nur K, V bleibt fest.
- Die Hadamard-Skepsis bekommt ihren Arm (No-Rotation-Ablation in WP-1) statt einer Meinung — mit OSCARs
  Gegengewicht im Rücken (ohne H −18 Punkte bei INT2, deren Pipeline).

Was aus Plan v1 ausdrücklich überlebt: Harness + gepaarte Auslesung + McNemar (§7.1/§7.2, `compare`/`areport
--paired`), die Sonden-Verdrahtung als bewiesene Parität (§7.3/§7.5), Q2_0 als untere Schranke, QJL-Verbot,
Identitäts-Gate + Speed-A/B vor jedem PR (§8.1/§8.3), Sonden-Code geht in keinen PR (§6).

Die „K4/V3-Klasse" in WP-2 ist keine runde Zahl: FlatQuant-Tab. 11 misst K4V3 auf LLaMA-3-8B KV-only bei −0,08
QA (nahezu verlustfrei) — die Referenzform fürs Layout, mit Undamped-Caveat (dort fp16-Rest, Head-dim 128,
dicht, PPL/QA statt Retrieval).

## 8. Messplan und Gate

**Gewichts-Modell (ab 2026-10-14):** die Meßbasis ist **Swift IQ3_XXS**, nicht Q2_0. Begründung in §7.5: Q2_0 ist
das gedämpfteste Regime, sein Gewichtsfehler maskiert den KV-Fehler. Q2_0-Messungen bleiben als untere Schranke
zulässig und werden als solche gekennzeichnet; jede **Entscheidung** (K-vs-V-Zuteilung, 3-vs-4-Bit, Transform-Slot)
fällt auf Swift. Das gilt für alle Punkte unten.

Für jedes WP:

1. **Identitäts-Gate.** Mit dem neuen Format **aus** muss die Ausgabe byte-identisch sein (Default `int8` und die
   bestehenden `q4_0`/`k8v4` unverändert). Ein neues opt-in-Format ist per Konstruktion default-neutral — das
   **zeigen**, nicht behaupten.
2. **Qualität.** `tools/quant_quality.py run` / `compare` — **gegen den fp16-KV-Arm bei gleicher Config** (für
   Gewichts-Quants wäre Q8_0 die Referenz; für KV wäre das eine Vermischung beider Fehler) — Top-1, mean/p99 KL,
   PPL auf dem festen Set `data/quant_quality/v1` (code, docs, english, multilingual, agent), je Format und Kontext
   (1K/8K/32K); `tools/needle_bench.py` für Recall (32K/128K/262K, Tiefen 10/50/90 %). **Nachtrag §7.2:** für den
   Mittelwert ändert die Kontextlänge 1K→8K nichts, der 32K-Punkt wird dafür nicht gebraucht (für WP-2 und den
   Recall schon); das zweite feste Set ist `data/quant_quality/v2-8k`, v1 bleibt nur für den Vergleich mit dem
   ersten Meßpunkt.
3. **Speed und Speicher.** Decode-A/B mit `--kv-resident`, wie in `bench/results/2026-09-27-kv-q4`.
4. **Akzeptanz.** Ein WP wird nur vorgeschlagen, wenn es die q4_0-Lücke bei **gleicher oder kleinerer** Byte-Zahl
   messbar schließt und die Speed-A/B nicht verliert. Sonst bleibt es opt-in/experimentell oder wird verworfen.
5. **BPE-Bilanz (ab Plan v2).** Für jedes Layout mit echten Bytes: ehrliche Bits-pro-Element-Rechnung nach OSCAR
   (Payload + Skalen-/Nullpunkt-Metadaten + Fenster bei **genannter** Kontextlänge, z. B. 128K). Sonden-Bytes
   zählen nicht — wer Bytes behauptet, zeigt die Bilanz.
6. **Power (ab Plan v2).** Ein K-vs-V-Effekt der Sonden-Größe (t≈0,6 bei n=8) braucht ~87 Segmente für t=2,0
   (≈11 h für 6 Arme auf Swift) — oder cwe-4k (Effekt verstärken statt Rauschen mitteln) oder McNemar-primär
   (5:0 steht bei p=0,062; wenige Segmente mehr entscheiden ihn). Vor einem Layout-Urteil eine der drei
   einlösen; „Richtung" ohne eingelöste Power ist kein Bauauftrag.

## 9. Risiken und offene Fragen

- **Transfer der Literatur — beantwortet (14.10.2026).** quant.cpp ist Llama (`head_dim` 128, **kein** QK-Norm); wir
  sind `head_dim` 256 **mit** QK-Norm. Auf unserem Material (§7.3) ist der Gewinn kleiner als die ~1 % relativ aus
  der Literatur, aber real: **−0,30 dB** MSE aus der Tabelle, **−0,61 dB** mit dem Skalenschritt. Eine
  `q4lu`-Kontrolle (Skalenwechsel 32→Head **ohne** Tabelle) wurde nicht gebaut, weil die Bauentscheidung auf das
  per-32-Format gefallen ist (§5.1-Nachtrag) — der Vergleich Tabelle-gegen-Skala ist stattdessen in §7.3 direkt
  gemessen.
- **max-abs je Block ist eine Heuristik — teilweise beantwortet.** `max|x|/87` je 32er-Block ist **besser** als
  `RMS/32` je Block (2,740e−02 gegen 3,547e−02, §7.3), und ein Least-Squares-Schritt verbessert es weiter
  (2,551e−02). Offen: ob die *Beta*-Verteilung der rotierten Koordinaten eine noch bessere Regel hergibt als
  `max|x|`.
- **QK-Norm und die Skalen-Redundanz.** Wird K per Head normiert (Qwen3.8-Flash-Next tut das), ist `‖k‖` fast
  konstant — die Erwartung war, dass ein Skalenwert je Head dann redundant ist. Unsere Messung spricht dagegen, dass
  die Skala *überhaupt* redundant ist: sie ist der **größere** der beiden Hebel (je 32er-Block). Offen bleibt, ob
  `‖k‖` konstant genug ist, um die Skala ganz einzusparen und die Bits einer 5-bit-Stufe zu geben.
- **B überschneidet sich mit #2318** (offen, `q2k`). Vorher im PR-Thread fragen, ob dort gemessen/beigetragen wird.
- **Streaming und STATE_HASH — erledigt.** `kKvQ4lm` (4) / `kKvQ4lmS` (5) sind in `KvFormat` (`kv_stream.hpp`), in
  `kv_fmt_q4`, in `qsa_kv_format` (also im Snapshot-Hash, `src/core/conversation_snapshot.cpp:53`) und in den
  Formatlisten von Suffix-Reuse; ein Block-Mover, der nur Bytes kopiert, behandelt sie wie `kKvQ4`
  (`docs/DETAILS.md:1008`, PR #2351).
- **Gate-Modelle.** Die genaue Vierer-Liste des Identitäts-Gates ist hier noch offen — vor WP-0 festzurren.
- **3-Bit-Sonde (WP-2a) — Q2_0-Richtung liegt vor, Swift-Entscheidung offen.** Der Q2_0-Lauf (§7.5) zeigt die
  Richtung **K will die Bits** (Antwortregion kategorial entlang K, McNemar 0:5 gegen den V-schweren Arm) und daß
  symmetrisch 3 Bit zu teuer ist (`q4lm8` 1,65× KL, t −5,04). Kein Einzelkriterium reißt bei 8 Segmenten die
  Schwelle, und Q2_0 ist das gedämpfteste Gewichts-Regime — deshalb wird der Lauf auf **Swift IQ3_XXS**
  wiederholt, wo der KV-Fehler sichtbar wird. Die Sonde verbraucht **keine** Bytes (576 B/Zelle wie `q4_0`), also
  ist selbst ein gutes Ergebnis **kein** Halbierungs-Beleg, sondern nur die Lizenz, das echte Layout in WP-2 zu
  bauen. Offen bleibt, ob 8 Pegel genug sind oder ob der Transform-Slot die 3-Bit-Stufe erst trägt.
- **Linien mit Falsifikationspflicht (Plan v2).** Hauptlinie: asymmetrisch-4-Bit (WP-1) → Null-Byte-Arme
  (WP-2a) → asymmetrisches Layout (WP-2). WUSH kalibriert-nur-K bleibt **Vergleichsarm** (WP-3) mit denselben
  Kriterien: gleiche Bits, Swift, gepaart + McNemar, Kostenbudget (Kalibrierzeit, Download ~9 MB, Online-K) —
  sonst ungebaut bzw. opt-in. Die Hadamard-Skepsis bekommt ihren Arm (No-Rotation-Ablation in WP-1) statt einer
  Meinung; OSCARs Gegengewicht (ohne H −18 Punkte bei INT2, deren Pipeline) steht daneben, als Messung fremder
  Geometrie, nicht als Urteil über unsere.
- **Kein V-Folding in quantisierte Gewichte.** WUSH (§4.2) wie OSCAR (§4) falten T_V in W_V/W_O — auf Swift
  IQ3_XXS wäre das ein abgeleitetes Pack (dicht-transformierte Projektionen in 3-Bit-Gewichte requantisiert)
  oder eine Online-V-Transformation (dann ist „gratis" weg). Deshalb: nur K wird kalibriert, V bleibt fest.
  (§10)
- **(0,0)-Fenster sind tot.** OSCAR-Tab. 5: INT2 mit Rotation+Clip, aber ohne Sink/Recent = 0,00 überall; Knie
  bei (64,256). Jedes Sub-4-Bit-Layout ohne eigenen Fenster-Arm ist blind — WP-2a misst ihn zuerst.
- **Kalibrierung ist leicht, aber eigene Arbeit.** OSCAR: ein Pass, ~8k Tokens, domänenrobust (Tab. 7) — weit
  unter WUSHs 128×32K; FlatQuant: ~0,9 h für 8B (gelernt, teurer als geschlossen-form). Für unseren Hybrid
  (48 Layer, QSA+PLE+HC, Head-dim 256) heißt das: geschlossene Kovarianz-Form (OSCAR-Stil) zuerst, Lern-Verfahren
  nur bei Bedarf; eine Kalibrierung soll mehrere Breiten bedienen („train one, get more").
- **Identitäts-Gate und Speed-A/B für WP-1a stehen noch aus** (§8.1/§8.3). Der A/B in §7.4 ist ein
  *Qualitäts*-Vergleich gegen den fp16-KV-Arm; „byte-identisch mit dem neuen Format aus" und der Decode-Durchsatz
  sind noch nicht gemessen. Vor einem PR beides nachholen, dann `kQ4lMap` zuerst in Shared Memory.

## 10. Was wir bewusst nicht machen

- Kein QJL-Residuum.
- Kein V-Folding in quantisierte Gewichte (T_V bleibt fest oder online; kalibriert wird nur K, WP-3).
- Kein Kalibrierungs-/Eigenbasis-Default (SpinQuant/OSCAR/OptR) — wenn überhaupt, nur opt-in in einem späteren WP.
- Kein 2-bit; 3-bit für jetzt **ohne** Delta (WP-1 misst zuerst die K/V-Zuteilung; ein Delta-Modus bleibt WP-5).
