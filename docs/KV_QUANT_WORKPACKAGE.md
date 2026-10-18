# Arbeitspaket: die KV-Cache-Quantisierung verbessern (Stand 2026-10-18)

Status: **WP-0 steht; WP-1a, WP-1 und WP-2a-Clip sind gebaut und gemessen** (`q4lm32`/`q4lm32s`: §5.1-Nachtrag,
§7.3, §7.4; WP-1-Arme `q4lm32t`/`q4lm32nr`/`q4lm8v`: §7.8, §7.9; Clip `q4lm32c`: §7.10, §7.11). Der
**WP-2a-Fensterarm ist gebaut und paritäts-/engine-validiert** (§7.12: Parität 11/11 auf gfx1200, Engine-Smoke
unter Streaming), **auf Swift cwe-2k gemessen** (§7.13: das Fenster trägt, Knie S ≈ 512) **und für WP-2b auf die
≤3-Bit-Arme geöffnet** (§7.14: orthogonale Überlagerung, Parität 12/12; §7.15: das Fenster trägt auch bei ≤3 Bit —
mit dem dortigen `R = 8`-Gitter ein Cross-Over bei ≈4,8–5,3 Bit). **§7.16 schließt die Knie-Arme ab** (`R = 64` mit
S 256 sowie das Sink-Paar 512/128, auf beiden Basen): alle vier sind von `int8` nicht zu trennen, die 3-Bit-Basis
erreicht das zuerst bei ≈5,9 statt 8,25 Bit — das §7.15-Cross-Over gilt nur für das zu knappe `R = 8`. **§7.17 füllt
die Serving-Lücke** (`§11.4.3`: 8 Arme × 3 Runden, Swift, 16K): das Format ist im Decode frei (±6 %), der
Fenster-Überbau im Decode unsichtbar, und der Fenster-Pool kostet 24 KiB je Kontextzelle (64K ≈ 1,5 GiB),
unabhängig von `--kv-resident`. Die **Review-Nachbesserungen** am Fenster-Code (K1–K4, R6) sind erledigt und als
datenneutral gezeigt (§7.18: Parität 12/12, zwei Arme byte-gleich zu ihren 2026-10-17-TSVs). **Arbeitspunkt ab
hier: Fenster `S/R = 256/64` auf `q4lm8v`** (§11.5.1). Der
Plan-Teil §1–§6 stammt aus dem Survey
[KV_QUANT_RESEARCH.md](KV_QUANT_RESEARCH.md); wo die eigenen Messungen ihn korrigieren, steht das dabei. §4/§5 sind
**Erwartungen aus fremden Messungen** und so gekennzeichnet; alle Zahlen in §7 sind **Strata-Messungen** mit dem
Hinweis, worauf sie gemessen wurden. Branch `KV_QUANT_research`, noch kein PR. §11 (2026-10-16) ist die Einordnung
gegen das Feld und eine Bewertung des Plans (Analyse, kein Meßpunkt); §11.5 nennt die offenen nächsten Schritte,
§12 ist das **Glossar** der Begriffe (KL/ΔKL/t/Segment, Fenster- und Quantisierungsnotation, Bytes).
**Offen für einen PR:** Identitäts-Gate + Speed-A/B auch für WP-1a/§7.4 (§8.1/§8.3, §9).

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
zurückgedreht. Platzierung festgehalten: Der Schreiber sieht **Post-Norm/Post-RoPE-K** (`mtp.cpp:963`
`native_qsa_rms_norm_rope` vor `kv_append_q4_steps` in Z. 979; ebenso `verify.cpp:1243` vor Z. 1312) — also
WUSH-Regime, nicht KVQuant-pre-RoPE. Das ist kohärent mit FWHT (die mischt Kanäle ohnehin, also überträgt sich
KVQuants Pre-RoPE-Kanalargument nicht 1:1), war aber nie eine *Entscheidung* aus K-Sensitivität: Pro-Kanal-Skalen
für K im KIVI/KVQuant-Sinn können nur **vor** FWHT leben (Pipeline-Änderung, vgl. §5.5), alles nach der Rotation
muss über Skalenregel/Clipping/Fenster laufen. (Nebenbei: `n_rot` ist 64 von 256 Dims — FWHT mischt rotierte mit
unrotierten Dims; ungeprüft, derzeit außer Scope.)

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

**Diagnostik-Lücke (noch kein Arm).** Die Literatur teilt feiner zu als Tensor: KVQuant pro-Layer
sensitivitätsgewichtet, RateQuant pro-Head-Bits, WUSH pro-Head-Transformationen. Wir haben nie gemessen, welche
Layer/Heads wehtun — die Parity-Suite kennt nur synthetisches N(0,1). Billig zu schließen: Pro-Layer-K/V-Statistik
(Kurtosis, max|x|, Outlier-Masse) aus einem Kalibrierungs-Forward-Pass, ohne jede Quantisierung; informiert
Skalenregel (WP-1), κ-Wahl (WP-2a) und spätere Bit-Zuteilung (WP-5). Noch nicht gebaut, nur vorgemerkt.

### 5.6 Clipped-Scale + Fenster vor dem Layout (WP-2a neu)

Zwei Arme, beide **ohne Byte-Änderung**, beide vor jedem Layout-Bau — sie entscheiden, was das Layout überhaupt
tragen muss:

1. **Clipped-Scale.** Skala aus dem κ-Quantil statt `max|x|` (Start κ≈0,92–0,96 — OSCAR-Tab. 6 und
   FlatQuant-fix 0,9/0,95 konvergieren dorthin; dann Grid auf Swift, **mit getrenntem κ_K ≠ κ_V**: OSCAR
   clippt K weniger (0,96) als V (0,92) — K behält mehr Dynamikbereich, wie es seiner Empfindlichkeit
   entspricht), Tabelle/Blöcke/Leser unverändert. Das
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
| **WP-1** (erledigt, §7.9) | **Asymmetrisch-4-Bit im Direktvergleich** (§5.5, §7.6): Behandlung K/V + Bit K16/V8 + No-Rotation-Ablation, Swift, Power nach §8. Ergebnis: Bit tot, Behandlung/No-Rotation Gleichstand | Messbericht + Gewinner-Arm | M |
| **WP-2a** (Clipped erledigt §7.11, Fenster erledigt §7.13) | **Null-Byte-Arme vor dem Layout** (§5.6, §7.6): Clipped-Scale + Fenster auf Sonden-Armen. Clip-Verdikt: Tails sind Signal, kein Post-Rotations-Clipping. Fenster: Parität 11/11, Engine-Smoke unter Streaming grün; cwe-2k fertig — das Fenster trägt, Knie S ≈ 512 | Messbericht (was das Layout tragen muss) | S–M |
| **WP-2b** (erledigt §7.15) | **Fenster auf die ≤3-Bit-Arme** (§7.14): orthogonal zur Level-Tabelle (`STRATA_KV_WINDOW=1` auf jedem q4-Arm), Parität 12/12. Ergebnis: Fenster trägt auch bei ≤3 Bit (128/8 −51 %, 8/8); mit dem `R = 8`-Gitter ein **Cross-Over ≈4,8–5,3 Bit** | Messbericht | S |
| **WP-2b-Knie** (erledigt §7.16) | **OSCAR-Knie `R = 64` (S 256) und Sink-Paar 512/128** auf beiden Basen. Ergebnis: alle vier von `int8` nicht zu trennen (ns), die 3-Bit-Basis am billigsten (**≈5,9 vs. 8,25 Bit**); das §7.15-Cross-Over gilt nur für das zu knappe `R = 8` — pro Byte verliert die 3-Bit-Basis nirgends meßbar | Messbericht | S |
| **WP-2-Laufzeit** (erledigt §7.17) | **Speed-A/B der KV-Arme** (§11.4.3): 8 Arme × 3 Runden, Swift, 16K-Kontext. Ergebnis: Decode ist **formatfrei** (±6 %, `int8` führt), der Fenster-Überbau unsichtbar; der Fenster-Pool kostet **24 KiB je Kontextzelle** (64K ≈ 1,5 GiB) und ignoriert `--kv-resident` | Messbericht | S |
| **WP-2-Review** (erledigt §7.18) | **Nachbesserungen aus dem Code-Review**: K1 Fenster nur auf q4-**Lloyd** (laut statt pfadabhängig), K2/K3 laute Fehler + Pool-Schranke (`win_cells`), K4 Snapshot nur bei vorhandenem Seitenpool, R6 Drafter reserviert keine Fenster-Pools. Als **datenneutral** gezeigt (Parität 12/12, zwei Arme byte-gleich zu ihren TSVs); R1/R4/R5 bewußt offen (Perf, §7.17 zeigt keinen meßbaren Effekt), B1 Metal-Stubs sind **älter als diese Arbeit** | Code + Doku | S |
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

**Nachtrag Swift (2026-10-14, selber Tag).** Die Falsifikation ist eingelöst, wie §9 festhält: Q2_0-Richtung tot
(t +0,05, McNemar überall p=1,000), Instrument intakt (t −13,34), WP-2-Form ab sofort ohne Sonden-Begründung.
WP-1 läuft unverändert (Behandlung + Bit-Referenz + Ablation) — nur die *Deutung* des Bit-Arms hat sich
geändert: Er ist Referenz, keine Richtung.

### 7.7 Nächste Schritte (konkret, Stand 2026-10-14 abends, GPU frei)

Jede Stufe produziert genau eine Entscheidung. CPU-Arbeit (Bauen) läuft parallel zu GPU-Arbeit (Messen).

**Phase 0 — bereit (Tage, GPU frei).**
0. Nadel-Anker: fp16 + int8, 32K, Tiefen 10/50/90, `--repeats 3`, Typen code+uuid → JSON einchecken. *Entscheidung:*
   Gate kalibriert (wo steht fp16/int8 bei Länge?). Aufwand S + GPU (Größenordnung ~1 h, wird gemessen).
1. WP-1-Arme bauen (CUDA/HIP + Parity grün): V-RMS-Skalenregel, No-Rotation-Modus (q4lm8v-Referenz existiert).
   *Entscheidung:* Arme messbereit. Aufwand M.
2. Pro-Layer-Diagnostik: K/V-Kurtosis/max/Outlier-Masse aus Kalibrierungs-Forward (ohne Quantisierung).
   *Entscheidung:* Vorwissen für Skalenregel (WP-1) und κ-Wahl (WP-2a). Aufwand S.

**Phase 1 — WP-1 Swift (darauf).**
3. cwe-2k Swift (fp16/int8/q4_0/q4lm32 + Behandlung + Bit-Referenz + No-Rotation, +evtl. q4lm8) mit Power
   (Segmente hoch oder cwe-4k parallel). *Entscheidung:* WP-1-Urteil — trägt Behandlung? Bit-Split bei 7 Bit?
   Rotation? Aufwand M (GPU über Nacht).
4. cwe-4k Swift für die Gewinner. *Entscheidung:* Längen-Generalisierung. Aufwand S–M.

**Phase 2 — WP-2a (danach).**
5. Clipped-Scale-Arm (κ-Grid, κ_K≠κ_V, fix zuerst) → Swift cwe-2k. *Entscheidung:* Schwanz-Beitrag separiert.
   Aufwand S.
6. Fenster-Arm (q2k-Mechanismus auf Sonden-Arme) → Swift cwe-2k. *Entscheidung:* Fenster-Beitrag separiert
   (ohne ihn keine Transform-Attribution). Aufwand M.
7. Nadel-Gate 32K für WP-1/WP-2a-Gewinner (Server-Neustarts je KV-Arm). *Entscheidung:* Validity bestanden?
   Aufwand S + Betrieb.

**Phase 3 — Layout + Vergleich.**
8. WP-2 echtes asymmetrisches Layout (K4/V3-Klasse, ehrliche BPE, Slot vorbereitet). Aufwand L.
9. WP-3 kalibriert-nur-K (Kalibrierung ~8k Tokens). *Entscheidung:* Layout vs kalibriert, mit BPE. Aufwand M–L.
10. Nadel-Gate 32K/128K + cwe-4k + Power-Top-up falls nötig.

**Phase 4 — PR-Reife.**
11. Identitäts-Gate (4 Modelle) + Speed-A/B (erst `kQ4lMap` in Shared Memory). *Entscheidung:* PR-reif. Aufwand M.
12. Kuratierter PR nach AGENTS.md (eine Änderung je PR; Sonden-Code fällt raus). Aufwand S.

**Explizit zurückgestellt:** Zweit-Gewichtsmodell (Entscheidung nach WP-1 Swift: trennt IQ3_XXS scharf, nur
Bestätigung nötig); ROT-Arm; 16K/32K-Framing (c); SRHT/WP-4; Delta/WP-5; Decode-only-Arm (braucht
Engine-Prefill-Pfad); Teacher-forced-NIAH-Fallback.

### 7.8 WP-1: beide Arme gebaut, Parität 9/9 grün (2026-10-16, Phase-0-Punkte 0+1 erledigt)

Gebaut sind die zwei neuen `Q4Mode`-Werte aus §5.5 (`kLloydVrms` 6 = `--kv q4lm32t`, `kLloydNoRot` 7 =
`--kv q4lm32nr`), die RMS-Skalenregel im Schreiber (`q4l_group_vrms`: Summe der Quadrate durch denselben
Xor-Butterfly, `sqrtf`, zweimal exakt 1/32), der No-Rotation-Pfad (ein Einzeiler in `qsa_state_init`: `kv_rot`
bleibt falsch — trägt automatisch Decode, Verify/Prompt und MTP, weil alle drei nur `st.kv_rot` lesen), zwei
neue Format-Ints (`kKvQ4lmT` 9, `kKvQ4lmNR` 10 — Byte-Mover behandeln sie wie alle q4-Formate), CLI + Harness.
Der Leser ist unverändert: beide Arme lesen 16-Pegel wie `q4lm32` (`q4_lloyd`, `pools.q4_mode`). SYCL-Zwilling
und Metal-Stub sind absichtlich unangefaßt (wie die Sonde in §7.5: dort bauen/verifizieren wir nicht; neue Modi
fallen dort in die bestehenden Zweige).

**Parität** (`kv_q4_parity`, gfx1200): 9/9 grün. Tests 1–7 reproduzieren exakt die Zahlen aus §7.3/§7.5
(q4lm32 2,740e−02, q4lm32s 2,551e−02, Sonde 7,672/7,662e−02, RMS-Regel 3,547e−02 = −0,82 dB) — der Default-Pfad
ist unberührt. Neu: Test 8 schreibt `q4lm32t` bit-identisch zur Host-Referenz (K `max|x|`, V exakter
Float-Butterfly-Replay) und liest beide Seiten 16-Pegel; gemischte Treatment-MSE 3,147e−02 (−0,60 dB gegen
`q4lm32` — der Arm fragt das Modell, nicht den MSE). Test 9 schreibt `q4lm32nr` bit-identisch zu `q4lm32`s
Referenz (Schreiber = kLloyd-Schreiber).

**Engine-Smoke** (Q2_0/HIP, `--tokens 9707,1879,264,1229,13 --max-new 6 --greedy`): alle drei Arme Exit 0 mit
drei verschiedenen Ausgaben (`q4lm32t` 198 19747 5587…, `q4lm32nr` 198 19747 10885…, `q4lm32` 198 257 621…).

**Nadel-Anker** (Phase-0-Punkt 0): `bench/results/2026-10-16-needle-anchors-swift/` — fp16 und int8 je 18/18
bei 32K. **Offen:** Phase 1 (cwe-2k Swift mit Power: fp16/int8/q4_0/q4lm32 + Behandlung + `q4lm8v` + No-Rotation),
dann cwe-4k für die Gewinner; Nadel-Gate 32K für die neuen Arme vor ihrem KL-Urteil.

### 7.9 WP-1-Urteil: Bit tot, zwei Gleichstände (2026-10-16, cwe-2k + cwe-4k auf Swift)

`bench/results/2026-10-16-kv-wp1-swift/` (2K) und `...-4k/` (4K, ohne den entschiedenen Bit-Arm). Anker
reproduzieren exakt. McNemar überall p ≥ 0,25 (16 Segmente ändern daran prinzipiell nichts: 98–99 %
Antwort-Übereinstimmung).

- **Bit (`q4lm8v`, K16/V8): tot.** 2K +51 % KL gegen `q4lm32` (t −3,60, 0/8). Die 3-Bit-Hoffnung endet hier; die
  Fenster-Variante (§7.15) hilft deutlich (128/8 −51 %, 8/8) und macht die 3-Bit-Basis im Low-Byte-Bereich
  konkurrenzfähig — oberhalb ≈5 Bit dominiert die 4-Bit-Basis pro Byte. **Nachtrag §7.16:** mit dem OSCAR-Knie
  (`R = 64`) ist die 3-Bit-Basis über den ganzen gemessenen Byte-Bereich nicht meßbar schlechter und erreicht
  `int8`-Niveau bei ≈5,9 Bit — das „oberhalb" gilt nur für das `R = 8`-Gitter.
- **Behandlung (`q4lm32t`): toter Gleichstand auf beiden Längen** (beide Male t +0,21 ganz). Kein Mandat,
  kein Veto für WP-2.
- **No-Rotation (`q4lm32nr`): Gleichstand** (2K t +0,21, 4K t −1,31/−0,17, nirgends signifikant). Hadamard
  trägt bei 4-Bit-Lloyd nichts Messbares; Produktionsnutzen (3 FWHTs gespart) ist WP-2-/WP-4-Sache.
- Zwei stille Engine-Tode auf dem NoRot-Arm (2 von 4 Läufen, beide früh, kein Log, zwei saubere 8/8
  dazwischen): intermittente Treiber-Instabilität, kein verfolgtes Muster.
- Nadel-Gate für die WP-1-Arme entfällt begründet: ohne Gewinner gibt es kein KL-Urteil, das ein Gate
  bräuchte; die Anker (fp16/int8 18/18) bleiben für WP-2a Vergleichsmaßstab.

### 7.10 WP-2a Clip gebaut, Parität 10/10, Messung offen (2026-10-16)

Ein Modus (`kLloydClip` 8 = `--kv q4lm32c`), keine Enum-Explosion: κ_K/κ_V sind Device-Konstanten
(`qsa_set_kv_clip`, Default 0,94, Grid per `STRATA_KV_CLIP_K/_V` — der Harness reicht sie per `--env`
durch, `.seg.json` protokolliert sie). Skala = κ-Quantil von |x| durch Bitonic-Netz (eine Lane pro Wert,
alle lesen dieselbe Lane — kein Reduktions-Ulp-Risiko), Tabelle/Leser unverändert, eigener Format-Int
(`kKvQ4lmC`).

**Parität 10/10 grün**, inkl. κ=1.0-Identität (MSE 2,740e−02 = `q4lm32` exakt) und Engine-Smoke (κ=1.0
token-identisch zu `q4lm32`, Default-κ verschieden). SYCL/Metal unangefaßt (Präzedenz §7.5).

**MSE-Vorbefund (gfx1200, synthetisch N(0,1) + 498.688 rotierte Werte, Python-Replay bestätigt):**
κ=0,94 → 5,346e−02 (2× schlechter als `q4lm32`), κ=0,92 → 9,139e−02 (3,3×). Post-FWHT-Werte sind
schwanzarm (max/32 ≈ 2,36σ); Clippen wirft echte Masse weg, statt Ausreißer zu zähmen — OSCRARs
0,96/0,92 gelten für unrotierte K-Kanäle, nicht für unsere Geometrie. Das Modell entscheidet (3 Arme
laufen: (1,0/1,0)-Identität, (0,94/0,94), OSCAR-Paar (0,96/0,92)); Anker aus WP-1 werden wiederverwendet.

### 7.11 WP-2a-Clip-Urteil: Tails sind Signal (2026-10-16, Swift cwe-2k)

`bench/results/2026-10-16-kv-wp2a-clip/`. Identität dreifach bewiesen (Parity-memcmp, Smoke-Tokens,
Modell-KL exakt 0,01183, sd 0). (0,94/0,94) +34 % KL (t −2,29), OSCAR-Paar (0,96/0,92) +111 % (t −6,66,
0/8). Asymmetrisch verliert MEHR als symmetrisch (t −5,16) — die härter geclippte V-Seite kostet am
meisten (V ist die fragile Seite, konsistent mit §7.5). **Folge für WP-2:** kein Clipping post-rotation
im Layout. **Folge für den Fenster-Arm:** wird gebaut (§5.6.2) — Tails sind Signal, exakte Tails könnten
Qualität zurückkaufen.

### 7.12 WP-2a-Fenster gebaut, Parität 11/11, gestagter Prefill-Fund behoben (2026-10-17)

Ein Modus (`kLloydWin` 9 = `--kv q4lm32w`), kein neues Schreibformat: der **Schreiber ist kLloyds Schreiber**
(bitweise), der **Leser** zieht die ersten R (Sink) und die jüngsten S Zellen exakt fp16 aus einem flachen
Seitenpool `[(cell, kv_head, head_dim)]`, alle anderen als 16 Pegel. S/R per `STRATA_KV_WINDOW_S/_R` (Default
128/8; 0/0 ist `q4lm32` — der Identitäts-Nullpunkt). Der Horizont ist `step[kStepNKv]` (pos + 1, device-frisch und
graph-sicher). Eigener Format-Int `kKvQ4lmW`; **Snapshots werden verweigert** (der exakte Pool liegt außerhalb des
Bildes — Refusal, kein stilles Ignorieren). Der Pool ist immer resident, per Zelle indiziert, außerhalb von
Paging und Streaming. Engine-Verdrahtung in `layer.cpp` (Pools, Nullen, Horizonte, Elastik-Abweisung), `mtp.cpp`
(Drafter ohne Fenster), `verify.cpp`, `prefill.cpp` (Decode- **und** Draft-Pfad), CLI/Harness.

**Parität** (`kv_q4_parity`, gfx1200, `build-hip-win\kv_q4_parity.exe`): **11/11 grün.** Test 11 schreibt
bit-identisch zur `kLloyd`-Host-Referenz, prüft den Seitenpool als exakten fp16-Zwilling jedes angehängten
Tupels, und läßt den Gather sink-only (R deckt alles → alles exakt) und recent-only (S = 1 → nur die letzte Zelle
exakt, der Rest 16 Pegel) gegen die Host-Referenz laufen. Die Tests 1–10 reproduzieren unverändert.

**Fund (behoben): gestagte Prefill-Pools trugen `q4_mode` nicht.** `pools_of()` (der gestreamte Prefill-Pfad,
`src/prefill/prefill.cpp`) setzt die Byte-Pools und die Page-Tabelle, aber — anders als `qsa_attn_pools(st)` im
nicht-gestagten/Decode-Pfad — **nicht** `pools.q4_mode`. Unter KV-Streaming (`st.kv_mode == 1`) wählte der
Decode-Kernel deshalb `kv_mode = q4_lloyd(kUniform) ? 4 : 2` = **2**, also das uniforme `q4_0`-Gitter statt der
Level-Tabelle; und weil `WinRows` nur bei `KV_MODE == 4` lebt, war der Fenster-Seitenpool dort **komplett
inaktiv** (die exakten Zellen wurden quantisiert gelesen). Auf CUDA hätte zusätzlich der Tensor-Core-Prompt-Kernel
das Level-Table-Format angenommen, statt es abzulehnen — der Hilfetext „the tensor-core prompt kernel refuses this
mode" gilt erst mit dem Fix. Betroffen war **auch das reine `q4lm32`** (§7.3/WP-1a), nicht nur das Fenster. Die
bisherigen Harness-Zahlen bleiben gültig, weil `tools/quant_quality.py` `--kv-resident` verwirft: die Läufe haben
`kv_resident == 0` → `kv_plan` gibt Modus 0 (voll resident) → `staged == false`, der gestagte Pfad lief nie.
**Fix:** für gestagte Pools `pools.q4_mode = st.kv_q4_mode` überlagern (die Fensterüberlagerung k_win/v_win/win_s/r
war schon da, aber ohne den Modus wirkungslos). Default-neutral: für fp16/int8/q4_0 ist `st.kv_q4_mode == kUniform`.

**Beleg (gfx1200, Q2_0, `--max-context 32768 --kv-resident 20480` = Streaming, `--spec 4`, greedy, 20
Prompt-Tokens, 8 neue).** Denselben Quellbau zweimal gebaut, nur die eine Zeile an/aus:

| Arm (gestreamt) | Fix **aus** | Fix **an** |
| --- | --- | --- |
| `q4lm32` | `27 91 316 6018 91 29 248046 198` | `27 91 316 4747 91 29 8678 248046` |
| `q4lm32w` S=0 R=0 | = `q4lm32` | = `q4lm32` |
| `q4lm32w` S=128 R=8 | `… 91 29 361 13` | `… 91 29 717 25` |

Nur die eine Zeile ändert das gestreamte `q4lm32`-Ergebnis ⇒ der gestagte Prompt las ohne Fix wirklich uniform.
Der Identitäts-Nullpunkt (S=R=0 == `q4lm32`) hält in beiden Builds, das Fenster (128/8) weicht in beiden ab ⇒ Arm
und Nullpunkt stimmen. **Der vollständige cwe-2k-Beitrag des Fensters ist damit noch nicht gemessen** (§11.5).

**SYCL-Zwilling.** `sycl/src/prefill/prefill.cpp` hat dieselbe Lücke in `pools_of`; dort sind `q4lm32`/`q4lm32s`
gespiegelt, der Bug greift also auch dort unter Streaming. Absichtlich unangefaßt (Präzedenz §7.5: kein oneAPI
hier, ungetestet) — offener Punkt.

### 7.13 WP-2a-Fenster auf Swift (cwe-2k): das Fenster trägt (2026-10-17)

Nur das Gitter wechselt (`STRATA_KV_WINDOW_S`/`_R`), sonst alles fix: Swift IQ3_XXS, der §7.12-Build, der
Harness (2K, 8 Segmente, 16.351 Positionen). Referenz fp16 (PPL 2.999). Bericht:
`bench/results/2026-10-17-kv-wp2a-window/`.

| Arm | S / R | top-1 | mean KL | KL / `q4lm32` | KL p99 | PPL | vs fp16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q4lm32` | – | 99,19 % | 0,01183 | 1,00x | 0,1464 | 3,012 | +0,43 % |
| `q4lm32w` | 0 / 0 | 99,19 % | 0,01183 | 1,00x | 0,1464 | 3,012 | +0,43 % |
| `q4lm32w` | 128 / 8 | 99,41 % | 0,00815 | 0,69x | 0,0856 | 2,994 | −0,17 % |
| `q4lm32w` | 512 / 8 | 99,37 % | **0,00609** | **0,51x** | 0,0743 | 2,992 | −0,23 % |
| `q4lm32w` | 1024 / 8 | 99,36 % | 0,00672 | 0,57x | 0,0766 | 2,994 | −0,17 % |

Der Nullpunkt ist **byte-gleich**, nicht nur gleich: die TSV von (0/0) und `q4lm32` haben dieselbe SHA256. Gepaart
über die 8 Segmente: `q4lm32`→128/8 **+0,00368** (t +7,16, 8/8), →512/8 **+0,00574** (t +5,62, 8/8), →1024/8
**+0,00511** (t +6,31, 8/8); 512/8 vs 1024/8 −0,00063 (t −0,79, 5/8) = **Knie bei S ≈ 512**. Der Antwortbereich
zeigt dieselben Vorzeichen (`q4lm32`−512/8 t +3,90, 8/8).

**Lesart.** Das Fenster trägt eindeutig, alle 8 Segmente in dieselbe Richtung; schon der Default (128/8) nimmt
31 % der KL, 512/8 die Hälfte. Über 512 hinaus passiert nichts mehr. PPL fällt unter fp16 (2,992 < 2,999) — wie
beim Clip-Arm (§7.11) und `q4lm32t`: der 4-Bit-Fehler dämpft, PPL ist hier kein Qualitätsleser, KL ist es.

**Bytes.** Die **Probe** hält einen fp16-Zwilling des **ganzen** Kontexts (`qsa_win_bytes` über `max_cells`,
`kv_append_win` schreibt jede Zelle), der Leser bedient daraus nur das Fenster: alle Arme haben dieselbe Fußspur
(~20,5 Bit/Wert), die Unterschiede sind reine Lesepolitik. Die **Auslieferung** würde nur die S+R exakten Zellen
fp16 halten (Ring + Sinks) → **4,5 + 11,5·(S+R)/L Bit/Wert** (L ≈ 2044): 128/8 ≈ 5,3, 512/8 ≈ 7,4, 1024/8 ≈ 10,3.

**Folge für WP-2.** Bei 4 Bit ist `q4lm32w` **int8-nah, bei weniger Bytes**: der beste Punkt (512/8) liegt bei
≈7,4 Bit (10 % unter `int8`s 8,25 Bit) und KL 0,00609 gegen `int8`s 0,00503 (+21 %), der Tail praktisch gleich
(KL p99 0,0743 vs 0,0733, PPL 2,992 vs 3,004). `int8` bleibt pro Byte ein wenig effizienter (bei gleichen Bytes
läge das Fenster höher — der Beitrag ist ab S ≈ 512 flach; 1024/8 ist bei 10,3 Bit sogar schlechter als 512/8,
also dominiert), aber `q4lm32w` ist ein echter **Rate-Knopf** (4,5 → 5,3 → 7,4 Bit) mit int8-naher Qualität, kein
bloßer ≤3-Bit-Hebel. Sein größtes Potenzial liegt trotzdem bei ≤3 Bit, wo die Mitte billig genug ist, daß 16 Bit
auf einem kleinen Fenster relativ wenig kosten — genau die Bedingung, unter die §8.5/§11.4 das Urteil „3 Bit tot"
gestellt haben. **Diese Erwartung hat §7.15 geprüft:** das Fenster trägt dort tatsächlich (128/8 −51 %, 8/8) und
macht die 3-Bit-Basis im Low-Byte-Bereich konkurrenzfähig (Cross-Over ≈4,8–5,3 Bit); oberhalb dominiert die
4-Bit-Basis pro Byte. **§7.16 prüft es mit Sink-Tiefe (`R = 64`) nach** und findet die 3-Bit-Basis über den ganzen
Byte-Bereich nicht meßbar schlechter, beide Basen auf `int8`-Niveau — das `R = 8`-Cross-Over war ein Artefakt der
zu knappen Sink-Zahl.

### 7.14 WP-2b: das Fenster auf die ≤3-Bit-Arme geöffnet (2026-10-17)

Die §7.13-Frage — trägt das Fenster auch bei ≤3 Bit, dem eigentlichen Zuhause des Urteils „3 Bit tot"
(§8.5/§11.4) — braucht dieselbe Überlagerung auf `q4lm8`/`q4lm8k`/`q4lm8v`. Der Kernel konnte das längst: der
Decode-Leser liest jede `q4_lloyd`-Mode gefenstert (`q4l_map_v(p.q4_mode)`, `qsa_decode_attn.cu`), der Gather
ebenso (`kv_gather_q4_kernel<NK,NV>`, Sink/jüngste Zellen modusunabhängig). Nur die Host-Verdrahtung war auf
`kLloydWin` festgenagelt. **Kein neuer Modus nötig** — das Fenster ist jetzt **orthogonal zur Level-Tabelle**:

- `qsa_win_wanted()` (`src/core/layer.cpp`) verlangt `g_kv_win && g_kv_q4 && S+R > 0` statt `mode == kLloydWin`.
- `g_kv_win` ist an, wenn der Modus `kLloydWin` ist (Back-compat `q4lm32w`) **oder** `STRATA_KV_WINDOW=1` gesetzt
  ist; dann gilt das S/R-Gitter für **jeden** q4-Arm (`--kv q4lm8v --env STRATA_KV_WINDOW=1 …`). S+R=0 ist wieder
  der Identitäts-Nullpunkt.
- `qsa_kv_format` gibt `kKvQ4lmW` zurück, sobald ein Seitenpool existiert (`st.k_win != nullptr`), unabhängig von
  der Tabelle — der Byte-Mover sieht ohnehin nur q4-Bytes, der Leser trägt seine Tabelle in den Pools
  (`q4_mode`), nicht im Format.
- Die Snapshot-Verweigerung (`conversation_snapshot.cpp`) prüft jetzt den Pool-Zeiger statt nur den Modus (die
  Identity `kLloydWin` mit S+R=0 ohne Pool bleibt wie zuvor verweigert).

**Parität** (`kv_q4_parity`, gfx1200, `build-hip-win\kv_q4_parity.exe`): **12/12 grün.** Neuer Test 12
wiederholt Test 11 mit `kLloyd8V` (16-Pegel-K, 8-Pegel-V): der V-Schreiber ist bitweise die `q4l8`-Host-Referenz,
K die `kLloyd`-Referenz, der Seitenpool ist der exakte fp16-Zwilling, und der Gather liefert sink-only alles
exakt / recent-only gemischt exakt/8-Pegel (K 16, V 8). Tests 1–11 reproduzieren unverändert.

**Default-Neutralität.** Ohne `STRATA_KV_WINDOW=1` und ohne `kLloydWin` ist `g_kv_win == false`: `qsa_win_wanted()`
falsch, kein Pool, `qsa_kv_format` unverändert, Snapshot-Pfad unverändert. `q4lm32w` verhält sich bitidentisch zu
vorher (S+R=0 weiterhin `q4lm32`). Die cwe-2k-Messung des 3-Bit-Fensters folgt in §7.15
(`bench/results/2026-10-17-kv-wp2b-window3bit/`).

### 7.15 WP-2b-Fenster auf Swift bei ≤3 Bit (cwe-2k): das Fenster trägt, 3 Bit nur als Low-Byte-Punkt (2026-10-17)

Nur `STRATA_KV_WINDOW=1` plus das S/R-Gitter wechseln, sonst alles fix: Swift IQ3_XXS, der §7.14-Build, der Harness
(2K, 8 Segmente, 16.351 Positionen). Referenz fp16 (PPL 2,999). Bericht:
`bench/results/2026-10-17-kv-wp2b-window3bit/`.

| Arm | S / R | top-1 | mean KL | KL / `q4lm8v` | KL p99 | PPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `q4lm8v` | – | 98,92 % | 0,01791 | 1,00x | 0,2793 | 3,024 |
| `q4lm8v`+w | 0 / 0 | 98,92 % | 0,01791 | 1,00x | 0,2793 | 3,024 |
| `q4lm8v`+w | 128 / 8 | 99,39 % | 0,00869 | **0,49x** | 0,0929 | 3,001 |
| `q4lm8v`+w | 512 / 8 | 99,36 % | 0,00828 | **0,46x** | 0,0926 | 2,996 |
| `q4lm8v`+w | 1024 / 8 | 99,35 % | **0,00741** | **0,41x** | 0,0924 | 2,999 |
| `int8` | – | 99,35 % | 0,00503 | 0,28x | 0,0733 | 3,004 |

Der Nullpunkt ist byte-gleich (TSV-SHA256 von 0/0 und `q4lm8v` identisch); ebenso reproduzieren `fp16`, `int8` und
`q4lm8v` die WP-1-Anker bitgenau. Gepaart über die 8 Segmente: `q4lm8v`→128/8 **+0,00922** (t +5,05, 8/8), →512/8
**+0,00963** (t +7,67, 8/8), →1024/8 **+0,01050** (t +4,07, 8/8); 128/8 vs 512/8 +0,00041 (t +0,38, 7/8) =
Gleichstand, 512/8 vs 1024/8 +0,00167 (t +2,80, 8/8) = noch ein Beitrag. Der Antwortbereich zeigt dieselben
Vorzeichen (`q4lm8v`−512/8 t +1,61, 7/8).

**Lesart.** Das Fenster trägt auch bei ≤3 Bit, alle 8 Segmente in dieselbe Richtung; 128/8 nimmt die Hälfte der KL,
1024/8 59 %. Die §7.13-Frage, ob das Fenster bei ≤3 Bit überhaupt trägt, ist damit beantwortet: **ja** — es ist keine
4-Bit-Eigenheit, sondern eine allgemeine Lesepolitik. Aber die schlechtere Mitte will ein größeres exaktes Fenster:
das Knie liegt höher als bei 4 Bit (dort war 512/8 das Knie, §7.13; hier ist 512/8 vs 1024/8 noch signifikant), und
genau diese Bytes fehlen.

**Bytes.** Basis `q4lm8v` ist **3,5 Pegel-Bits** (K 16-Pegel = 4, V 8-Pegel = 3) bzw. **≈4,0 Bit/Wert gespeichert**
(Pegel + 0,5 Bit/Wert Skalen-Overhead, wie der 18-B-Block von `q4_0`/`q4lm32`); die Auslieferung hält nur die (S+R)
exakten Zellen fp16 → **4,0 + 12,0·(S+R)/L Bit/Wert** (L ≈ 2044): 128/8 ≈ 4,8, 512/8 ≈ 7,1, 1024/8 ≈ 10,1. Zum
Vergleich die 4-Bit-Basis (`q4lm32w`): 4,5 + 11,5·(S+R)/L → 128/8 ≈ 5,3, 512/8 ≈ 7,4, 1024/8 ≈ 10,3.

**Folge: im Low-Byte-Bereich ist die 3-Bit-Basis mit Fenster knapp konkurrenzfähig — „3 Bit tot" gilt dort nicht.**
Gepoolt mit dem 4-Bit-Lauf (die `fp16`-Anker beider Läufe sind byte-gleich, also direkt zusammenlegbar; gepaart über
dieselben 8 Segmente):

| Paar (Auslieferungs-Bits) | ΔKL | t | Segmente | Lesart |
| --- | ---: | ---: | ---: | --- |
| `q4lm32` (4,5) − `q4lm8v`+w 128/8 (≈4,8) | +0,00314 | +3,06 | 6/8 | 3-Bit hat **0,3 Bit mehr**, klar bessere KL |
| `q4lm32w` 128/8 (≈5,3) − `q4lm8v`+w 128/8 (≈4,8) | −0,00054 | −0,80 | 4/8 | Gleichstand, 4-Bit hat **~0,5 Bit mehr** |
| `q4lm32w` 128/8 (≈5,3) − `q4lm8v`+w 512/8 (≈7,1) | −0,00012 | −0,13 | 4/8 | Gleichstand, 4-Bit hat **1,8 Bit weniger** |
| `int8` (8,25) − `q4lm32w` 512/8 (≈7,4) | −0,00106 | −1,40 | 1/8 | Gleichstand, 4-Bit hat **~0,85 Bit weniger** |

- **Low-Byte-Bereich (≈4,8–5,3 Bit): schmaler Vorteil für die 3-Bit-Basis.** `q4lm8v`+w(128/8, ≈4,8 Bit) ist gegen
  `q4lm32w`(128/8, ≈5,3 Bit) bei statistisch gleicher KL ~0,5 Bit/Wert (~9 %) billiger. Gegen den fensterlosen
  4-Bit-Arm ist es **kein** Byte-Sieg (4,8 vs 4,5 Bit), nur bessere Qualität.
- **Ab ≈5 Bit dreht es: die 4-Bit-Basis dominiert pro Byte.** Größere Fenster helfen der 3-Bit-Basis, kosten aber mehr
  als sie bringen (512/8 gegen 4-Bit 128/8: 1,8 Bit mehr, kein KL-Unterschied). `q4lm32w`(512/8) erreicht mit ≈7,4 Bit
  `int8`-Niveau (8,25 Bit) — der beste Byte-Punkt des Ganzen; `int8` bleibt die absolute Decke.
- **Grenze.** Die Bit-Zahlen sind ein **Modell** (Pegel + Block-Skalen-Overhead + `(16 − Basis)·(S+R)/L`, L ≈ 2044 bei
  2K), nicht gemessen; bei längerem Kontext sinkt der Fenster-Anteil, das Cross-Over kann sich verschieben. Die
  Qualitätsmessung ist 2K; ob das Fenster bei 128K noch so viel zurückholt, ist **nicht** gemessen.
- **„3 Bit tot" also genauer:** ohne Fenster gilt es (§7.9); mit Fenster ist die 3-Bit-Basis im hier gemessenen
  `R = 8`-Gitter (≈4,8–10 Bit) pro Byte knapp konkurrenzfähig, das Cross-Over liegt bei ≈4,8–5,3 Bit. **Beides gilt
  nur für `R = 8`:** §7.16 mißt das OSCAR-Knie (`R = 64`) nach und findet die 3-Bit-Basis pro Byte nirgends meßbar
  schlechter — beide Basen erreichen dort `int8`-Niveau.

### 7.16 WP-2b-Knie auf Swift (cwe-2k): beide Basen erreichen int8-Niveau, die 3-Bit-Basis am billigsten (2026-10-18)

Der §7.15-Abschluß ließ zwei Arme aus dem Beschluß offen: das **OSCAR-Knie** (`R = 64` Sink bei `S = 256`) und ein
reines Sink-Paar. Beide sind auf **beiden** Basen nachgemessen — vier Arme, ~9,5 min/Arm, derselbe Build und Harness
wie §7.13/§7.15. Die Anker `fp16` `36C9CE2D…`, `int8` `3CB6A156…`, `q4lm8v` `83D13E7A…`, `q4lm32` `A3F4CE4A…`
sind byte-gleich, die gepoolte Auswertung umfaßt also 13 Arme. Bericht:
`bench/results/2026-10-18-kv-wp2b-oscar-knee/`.

| Arm | S / R | Bit (Modell) | top-1 | mean KL | KL p99 | PPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `q4lm8v` | – | ≈4,0 | 98,92 % | 0,01791 | 0,2793 | 3,024 |
| `q4lm32` | – | 4,5 | 99,19 % | 0,01183 | 0,1464 | 3,012 |
| `int8` | – | 8,25 | 99,35 % | 0,00503 | 0,0733 | 3,004 |
| `q4lm8v`+w | 256 / 64 | ≈5,9 | 99,33 % | **0,00702** | 0,0835 | 3,003 |
| `q4lm32w` | 256 / 64 | ≈6,3 | 99,37 % | 0,00782 | 0,0752 | 3,002 |
| `q4lm8v`+w | 512 / 128 | ≈7,8 | 99,39 % | **0,00623** | 0,0743 | 2,995 |
| `q4lm32w` | 512 / 128 | ≈8,1 | 99,43 % | **0,00569** | 0,0729 | 2,994 |

**Kernaussage: alle vier Knie-Arme sind von `int8` (8,25 Bit) statistisch nicht zu trennen.** Gepaart über die 8
Segmente: `int8`−`8vw-s256-r64` −0,00199 (t −1,92, 2/8), `int8`−`32w-s256-r64` −0,00279 (t −1,84, 3/8),
`int8`−`8vw-s512-r128` −0,00120 (t −1,22, 3/8), `int8`−`32w-s512-r128` −0,00066 (t −0,88, 4/8). Die 3-Bit-Basis
erreicht das zuerst: `q4lm8v`+w(256/64) liegt bei ≈5,9 Bit gegen `int8`s 8,25 (≈29 % weniger), ohne meßbaren
Verlust. „Nicht unterscheidbar" heißt nicht „gleich": das 95 %-Intervall von `int8`−`8vw-s256-r64` reicht bis
≈ +0,0044.

**Das Cross-Over aus §7.15 (≈4,8–5,3 Bit) reproduziert sich nicht.** Bei gleicher Form ist die 3-Bit-Basis in
**allen fünf** Formen nicht schlechter (`|t| ≤ 1,50`) und kostet **0,24–0,47 Bit weniger**; nominell liegt die
4-Bit-Basis in 4 von 5 Formen vorn, in einer (`256/64`) die 3-Bit-Basis:

| Form (S/R) | 3-Bit KL (Bit) | 4-Bit KL (Bit) | ΔKL (3 − 4) | t | Segmente up |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 8 | 0,00869 (≈4,8) | 0,00815 (≈5,3) | +0,00054 | +0,80 | 4/8 |
| 256 / 64 | 0,00702 (≈5,9) | 0,00782 (≈6,3) | −0,00080 | −1,26 | 3/8 |
| 512 / 8 | 0,00828 (≈7,1) | 0,00609 (≈7,4) | +0,00218 | +1,50 | 7/8 |
| 512 / 128 | 0,00623 (≈7,8) | 0,00569 (≈8,1) | +0,00054 | +0,75 | 6/8 |
| 1024 / 8 | 0,00741 (≈10,1) | 0,00672 (≈10,3) | +0,00069 | +0,54 | 6/8 |

Pro Byte verliert die 3-Bit-Basis damit in dieser Messung **nirgends meßbar** — der §7.15-Vorsprung war kein
Low-Byte-Phänomen, er reicht bis ≈10 Bit. Ein scharfes Cross-Over läßt sich aus 8 Segmenten nicht mehr angeben: die
Format-Differenz (≤0,0022, immer ns) ist von derselben Größe wie der S/R-Formeffekt.

**Der Sink (`R`) ist eine eigene Achse.** Bei festem `S = 512` und fester Basis verbessert mehr Sink die KL, und die
Richtung ist über die Segmente konstant: `8vw-s512-r8` − `8vw-s512-r128` +0,00204 (t +1,88, **8/8**),
`win-s512-r8` − `32w-s512-r128` +0,00258 (t +1,50, **7/8**). `R = 8` (das §7.13/§7.15-Gitter) war also zu knapp:
`8vw-s256-r64` (320 Zellen, `R = 64`) liegt unter `8vw-s512-r8` (520 Zellen, `R = 8`; +0,00126, t +1,61, 6/8).
Innerhalb der 3-Bit-Familie: 128/8 → 256/64 +0,00167 (t +2,97, **8/8**), 256/64 → 512/128 +0,00078 (t +1,98, 7/8).

**Folge für §11.4.4/§11.5.1.** Die Aussage „3 Bit lebt nur als schmaler Low-Byte-Punkt" ist mit dem Knie-Gitter zu
eng: mit `R = 64` ist die 3-Bit-Basis über den ganzen gemessenen Byte-Bereich (≈4,8–10 Bit) pro Byte nicht meßbar
schlechter als die 4-Bit-Basis, und beide erreichen `int8`-Niveau — die 3-Bit-Basis bei ≈5,9 statt 8,25 Bit. Der
beste **absolute** KL-Punkt bleibt die 4-Bit-Basis (`q4lm32w` 512/128, 0,00569 bei ≈8,1 Bit); die 3-Bit-Basis ist
der beste **Byte**-Punkt. Grenzen: Qualität nur bei 2K, Bit-Zahlen nur Modell, S/R-Raster grob (fünf Formen,
8 Segmente).

### 7.17 Laufzeit-A/B der KV-Formate (Swift, 16K-Kontext, 2026-10-18)

§11.4.3 nannte als offene Serving-Lücke „kein Speed-A/B". Der ist jetzt gefahren: 8 Arme × 3 Runden, pro Arm ein
frischer Server, Rundenordnung alternierend (`tools/ab_engine.py`), nur `--kv` (+ `STRATA_KV_WINDOW*`) geändert,
derselbe Build wie §7.12–§7.16. Abgelesen wird die Engine-eigene Zahl `predicted_per_second`; der lange Prompt hat
6.345 Token (die drei kurzen sind Overhead-dominiert). Bericht: `bench/results/2026-10-18-kv-runtime-ab/`.

| Arm | decode tok/s (Median) | Spanne | vs. fp16 | prompt tok/s (6.345) |
| --- | ---: | ---: | ---: | ---: |
| `int8` | 44,9 | 42,9–46,2 | +5,6 % | 444 |
| `fp16` | 42,5 | 42,2–43,4 | ±0 | 450 |
| `q4lm32c` | 42,0 | 38,7–43,1 | −1,2 % | 448 |
| `q4lm32` | 41,7 | 40,2–43,0 | −1,9 % | 441 |
| `q4lm8v`+w 512/8 | 41,6 | 39,2–45,0 | −2,1 % | 437 |
| `q4lm32w` 512/8 | 41,4 | 38,2–43,7 | −2,6 % | 435 |
| `q4lm8v`+w 256/64 | 40,3 | 37,1–40,4 | −5,2 % | 429 |
| `q4lm8v` | 39,7 | 38,1–44,7 | −6,6 % | 436 |

**Zwei Lesarten.** (1) **Das Format ist im Decode praktisch frei**: die ganze Spanne ist ±6 %, `int8` führt,
`q4lm8v` schließt; die Formatwahl fällt über Qualität und Speicher, nicht über Decode-Durchsatz auf dieser Karte.
(2) **Der Fenster-Überbau ist im Decode nicht sichtbar**: `q4lm32w`(512/8) 41,4 liegt innerhalb der Spanne von
`q4lm32` 41,7, und `q4lm8v`+w(512/8) 41,6 liegt über `q4lm8v` 39,7. Damit sind **R1/R2 im Decode widerlegt** —
bei einem Token pro Schritt ist der Zwillings-Append winzig gegen die Attention über 6,3K Token. Im Prefill
(6.345 Token) liegen die Fenster-Arme ≤3 % unter ihren Zwillingen (435/437/429 gegen 441/436), in Richtung des
zusätzlichen Zwillings-Schreibens, aber innerhalb der Spanne.

**Was das Fenster an Speicher kostet (R3, gemessen).** Der flache Seitenpool wird von `max_cells` bemessen
(`qsa_win_bytes`, `layer.cpp:196`): `max_cells × n_head_kv × head_dim × 4 B` = **2.048 B je Zelle** je Layer;
bei diesem Modell sind `n_head_kv = 2`, `head_dim = 256` und **12** Layer tragen einen QSA-Zustand
(`qsa_interval = 4` von 48) → **24 KiB je Kontextzelle** über alle Layer. Gemessen als Differenz von
`N MiB of VRAM free with everything loaded` zwischen Fenster-Arm und Zwilling bei fester Expert-Cache-Vorgabe:
16K → 416 MiB (Formel 384 MiB), **64K → 1.536 MiB (Formel exakt)**. Und mit `--kv-resident 20480` bei
`--max-context 65536` schrumpft das quantisierte K/V auf 0,42 GiB pinned RAM, der Pool bleibt aber bei
**1.536 MiB** — der Pool skaliert mit `max_context`, nicht mit dem Resident-Set, **hebelt also das Streaming
aus** (16K ≈ 384 MiB, 64K ≈ 1,5 GiB, 128K ≈ 3 GiB). **Korrektur:** der Review hatte „2K ≈ 10 MB, 16K ≈ 64 MB,
128K ≈ 512 MB" und 48 Layer genannt; richtig sind 12 QSA-Layer und 24 KiB/Zelle (2K ≈ 48 MiB).
**Aber das ist die Form der *Probe*, nicht die der Auslieferung:** der flache Pool hält fp16 für **jede** Zelle,
weil er direkt über die Zellnummer indiziert wird. Die Auslieferungsform ist ein **Ring von `S + R` Zeilen**
(Sink-Bereich `cell < R`, Ringbereich `R + cell % S`) — bei 256/64 also `320 × 2 KiB × 12 Layer = **7,5 MiB**`,
**kontextunabhängig** (512/128: 15 MiB; Rechenweg aus der exakten 2 KiB/Zelle/Layer der 64K-Messung, keine eigene
Messung). Damit fällt auch der Grund für die laute Verweigerung „Fenster + elastic K/V" weg (der flache Pool paßt
nicht in die per-Slot-Pools). Bis dieser Ring gebaut ist, gilt die Zahl oben **nur für die Probe**.

### 7.18 Nachbesserungen aus dem Code-Review (K1–K4, R6)

Die Review-Befunde am Fenster-Code (nach den Prüfläufen, wie vereinbart) sind behoben; Parität bleibt 12/12, und
die Nachbesserung ist **datenneutral gezeigt**: `q4lm8v` und `8vw-s128-r8` auf cwe-2k ergeben mit dem neuen Build
dieselben TSV-SHA256 wie am 2026-10-17 (`83D13E7A…` / `D07C1492…`).

- **K1 (Fenster war pfadabhängig).** `qsa_win_wanted()` verlangt jetzt einen **q4-Lloyd**-Zustand
  (`q4_lloyd(g_kv_q4_mode)`), und `qsa_set_kv_q4_mode` **verweigert `STRATA_KV_WINDOW=1` auf `--kv q4_0` laut**
  (stderr + Fenster aus): dort hätte der q4-Gather das Fenster gelesen, der fusionierte Decode
  (`KV_MODE 2`) es still ignoriert. Die gemessenen Arme sind alle Lloyd und damit unberührt.
- **K2 (stiller Abbruch).** `kv_append_win`/`kv_append_win_steps` kehren bei `head_dim % 32 != 0` nicht mehr
  still zurück, sondern scheitern laut wie `need_256` — ein stiller Return hätte den Seitenpool **veraltet**
  stehen lassen, während der Aufrufer ihn für geschrieben hielt.
- **K3 (keine Schranke).** Beide Funktionen bekommen die Pool-Kapazität (`win_cells`). Die Batch-Variante prüft
  `pos0 + T ≤ win_cells` **host-seitig laut** (die Prefill-Positionen sind host-bekannt); die Steps-Variante
  schützt den Store im Kernel (die Position ist device-frisch) — dort gibt es in einem Release-Build (NDEBUG)
  keinen lauten Device-Abbruch, deshalb wird der Store verworfen statt Speicher zu überschreiben.
- **K4 (Snapshot-Verweigerung zu grob).** Die Verweigerung hängt jetzt **nur am Seitenpool**. `kLloydWin` mit
  S + R = 0 legt keinen Pool an, ist ein gewöhnlicher `kLloyd`-Cache (Writer und Tabelle identisch) und wird
  als `kKvQ4lm` beschriftet — also **snapshot-fähig**; `--kv q4lm32w` ist damit mit Konversations-Cache nutzbar.
- **R6 (Drafter-Reservierung).** Der MTP-Drafter führt seine State-Init jetzt mit **abgeschalteter Überlagerung**
  aus (`qsa_set_kv_win`), statt zwei Pools zu schnitzen und die Zeiger danach wegzuwerfen; die Arena-Größe und
  `vram_` zählen sie damit nicht mehr mit.

**Bewußt nicht geändert.** R1 (Append falten), R4 (`swin[c]`-Hoist) und R5 (pre75-Schleife) sind
Hot-Loop-/Perf-Eingriffe: §7.17 zeigt für den Fenster-Überbau **keinen meßbaren Decode-Unterschied**, also würde
eine solche Änderung eine eigene Messung brauchen (AGENTS.md: Default nur ändern, wo schneller gemessen). R2/R3
sind der Proben-Überbau selbst und in §7.17 als Speicherzahl dokumentiert.

**B1 (Metal) ist älter als diese Arbeit.** `metal/src/engine/stubs.cpp` kennt kein `Q4Mode`, obwohl
`kv_append_q4_step`/`kv_gather_q4_step` es seit `8f91507f` (WP-1a) im Parameter haben — der Metal-Link ist also
**schon am HEAD der Basis** offen, nicht erst durch das Fenster (das drei Symbole ergänzt:
`kv_append_win`, `kv_append_win_steps`, die neue `kv_gather_q4_step`-Signatur). Die Datei ist generiert
(`metal/tools/gen_stubs.py` aus einem macOS-Link-Log); ohne macOS/ld64 hier nicht regenerierbar, und
Hand-Einträge wären geraten.
**B2 (SYCL) ist ein voller Fork**, der nicht mitwächst: `sycl/src/core/{layer,mtp,verify,prefill,conversation_snapshot}.cpp`
und `sycl/src/kernels/cuda/kv_q4.dp.cpp` sind eigene Kopien. Die Kopie **inkludiert den gemeinsamen Kopf**
`include/strata/kernels/kv_q4.hpp`, definiert `kv_gather_q4_step` aber noch mit 11 Parametern, während der Kopf seit
WP-2a 15 deklariert (Fenster) — die SYCL-Kopien müssen also **nachgezogen** werden, sobald ein oneAPI-Build
verfügbar ist; hier ist keiner (gfx1200/HIP). Der HIP/CUDA-Build ist davon nicht betroffen (`add_subdirectory(sycl)`
greift nur mit oneAPI), und die Forks rufen `kv_append_win*` nicht auf, also ändert die neue Signatur dort nichts.

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
7. **Aussage-Disziplin (ab 2026-10-14, Survey §6.13).** Primär-Auslesung je Frage vorab festlegen und dann
   festhalten: Ganzsegment-gepaarte KL für Format-Rangfolge, McNemar-Antwortgenauigkeit für K-vs-V (der
   fwe→cwe-Wechsel war datengetrieben und ist hiermit verriegelt). Prefill-Abweichung offenlegen: Unser Harness
   quantisiert Prefill-KV mit (Feldstandard ist decode-only, Yuan OB1) — interne Vergleiche bleiben gültig
   (fp16-Arm teilt die Behandlung), absolute Lücken sind nicht mit Feldzahlen vergleichbar; ein Decode-only-Arm
   ist vorgemerkte Arbeit (braucht Engine-Prefill-Pfad fürs Format), keine vorhandene. Jeder Bench-README trägt
   den Scope-Stempel (Modell, Pack, Kontext, Prefill-Behandlung); Längenskalierung (cwe-4k) ist der
   Generalitäts-Proxy; Q2_0→Swift der Gewichtsregime-Gegencheck. Gesättigter Test = Härtevorschrift, kein
   Ergebnis (H2O-Lehre: 100 % → 35 % nach Härtung).
8. **Nadel-Gate (vorgemerkt, für WP-2).** `tools/needle_bench.py` existiert und passt aufs fehlende Bein: freie
   Generierung durch die Server-API, greedy, Thinking aus, Exaktmatch-Grading, Repo-Heuhaufen, Tiefen 10/50/90,
   32K/128K/262K, Timing dabei. Lücken vor dem Gate-Einsatz: ein Schuss pro Zelle (keine Varianz → `--repeats`
   mit frischen Wörtern/Positionen), ein Nadeltyp (RULER: UUID-ähnliche Codes dazu), keine harten Distraktoren
   (MK-Analog: Distraktor-Nadeln), nur Einzelabfrage (MV/MQ-Analog: unvollständige Mengen unsichtbar), nie pro
   KV-Arm gelaufen (braucht Server-Neustarts je `--kv`-Arm — Betriebskosten, vermerkt). Spezifiziert hiermit,
   gebaut bei freier GPU. Fallback: Teacher-forced-NIAH im Harness (Nadel-Logprob + KL an Nadel-Tiefen, gepaart,
   ohne Server), falls freie Generierung pro GPU-Stunde zu verrauscht ist.

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
- **3-Bit-Sonde (WP-2a) — Swift erledigt.** Der Q2_0-Lauf (§7.5) zeigte die Richtung **K will die Bits** (Antwortregion
  kategorial entlang K, McNemar 0:5 gegen den V-schweren Arm) und daß symmetrisch 3 Bit zu teuer ist (`q4lm8` 1,65×
  KL, t −5,04). Der Swift-Lauf (§7.9) hat den Bit-Arm entschieden (`q4lm8v` +51 % KL, t −3,60, 0/8), §7.15 die
  Fenster-Variante (das Fenster trägt, rettet 3 Bit aber nicht). Die Sonde verbraucht **keine** Bytes (576 B/Zelle
  wie `q4_0`), also war selbst ein gutes Ergebnis **kein** Halbierungs-Beleg, sondern nur die Lizenz, das echte
  Layout in WP-2 zu bauen. Die Frage, ob 8 Pegel genug sind, ist damit beantwortet: nur mit Fenster, und nur im
  Low-Byte-Bereich (≈4,8–5,3 Bit); oberhalb dominiert die 4-Bit-Basis pro Byte. **§7.16 relativiert das:** mit dem
  OSCAR-Knie (`R = 64`) erreicht die 3-Bit-Basis `int8`-Niveau bei ≈5,9 Bit und verliert pro Byte in keiner der fünf
  gemessenen Formen meßbar; die „oberhalb"-Aussage gilt nur für das `R = 8`-Gitter.
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
- **Falsifikation eingelöst (Swift, 2026-10-14).** Die Q2_0-Richtung (K-will-Bits, t +0,60, McNemar 0:5) hat den
  Regimewechsel nicht überlebt: direkter Kontrast auf Swift +0,00011, t +0,05, 3/8; McNemar überall p=1,000
  (`bench/results/2026-10-14-kv-3bit-probe-swift/`). Folge: Die WP-2-Form (K4/V3-Klasse) ruht nicht mehr auf der
  Sonden-Richtung, sondern nur noch auf FlatQuant-Tab. 11 (fremde Evidenz, dort bei 7 Bit ebenfalls Gleichstand)
  — plus der offenen 6-Bit-Split-Sondenfrage (dort trennt die Literatur: K4V2 vs K2V4). Unberührt: Behandlung
  (Skalenregeln, WP-1 Arm 1 — anderer Mechanismus), Clipped-Scale, Fenster, kalibriert-nur-K als Vergleich.
  Intakt auch das Instrument (fp16-int8 t −13,34 auf Swift gegen −4,43 auf Q2_0). So arbeitet die Regel: Richtung
  verworfen, Plan angepasst, Messung behalten.
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

## 11. Addendum (2026-10-16): Einordnung gegen das Feld und Bewertung des Plans

Status: **Analyse, kein Meßpunkt.** Was hier unter „eigene Messungen" steht, ist aus §7 dieses Dokuments und dort
belegt. Was unter „das Feld" steht, ist eine Primärrecherche vom 2026-10-16 (arXiv/ACL, 2024–2026); diese Zahlen sind
auf anderen Modellen, Kopfgeometrien und Metriken gemessen und **nicht** direkt mit unseren vergleichbar. Das
Addendum ergänzt §6.12/§6.13 des Surveys um Arbeiten, die nach dem 14.10. erschienen sind, und bewertet den Plan v2
(§7.6) gegen den Stand vom 2026-10-16. Es ändert **keinen** Beschluß.

### 11.1 Die Achsen, die der Plan mißt

| Achse | Flag | Idee | Wo gemessen |
| --- | --- | --- | --- |
| Codebook | `--kv q4lm32` | uniformes ggml-Gitter → Max-Lloyd-16-Pegel für N(0,1), 18-B-Blöcke unverändert | §7.3, §7.4 |
| Skalenregel | `--kv q4lm32s` | ein Least-Squares-Schritt je Block statt max-Betrag/87 | §7.3, §7.4 |
| K/V-Asymmetrie | `q4lm8k` / `q4lm8v` | Bits entlang K gegen V | §7.5, §7.9 |
| Behandlung | `q4lm32t` | V mit RMS-Skala statt max, sonst gleiche Bytes | §7.9 |
| Clipping | `q4lm32c` | κ-Quantil-Skala **nach** der Rotation | §7.11 |
| Fenster/Sink | `--kv q4lm32w` / `STRATA_KV_WINDOW=1` (`STRATA_KV_WINDOW_S/_R`) | jüngste/erste Zellen exakt, Rest quantisiert | §7.12 (Parität 12/12); §7.13 (4 Bit), §7.15 (≤3 Bit), §7.16 (Knie) |
| Rotation | `q4lm32nr` | No-Rotation-Ablation | §7.9 |
| Laufzeit | `tools/ab_engine.py` | Decode/Prompt tok/s je Arm, Runden alternierend | §7.17 |
| Vergleich | WP-3 | kalibriert-nur-K, kein V-Folding | vorgemerkt |

### 11.2 Was gemessen ist (Kurzfassung)

| Frage | Ergebnis | Beleg |
| --- | --- | --- |
| Schließt das Codebook die `q4_0`-Lücke? | **Ja, bei gleichen Bytes**: +0,23 % → +0,02 % PPL (8K), +1,30 % → +0,01 % (1K) | §7.4 |
| Trägt 3 Bit? | **Ohne Fenster nein** (Swift: `q4lm8v` +51 % KL, t −3,60, 0/8). **Mit Fenster ja — wenn das Fenster Sink hat**: bei `R = 8` senkt es 51–59 % (8/8, §7.15), bei `R = 64` erreicht die 3-Bit-Basis `int8`-Niveau (0,00702 gegen 0,00503, t −1,92, ns) bei **≈5,9 statt 8,25 Bit**, und pro Byte verliert sie in fünf Formen nirgends meßbar (`\|t\| ≤ 1,50`, §7.16) | §7.9, §7.15, §7.16, §11.4.4 |
| K oder V die Bits? | Q2_0: Richtung K (0:5, p 0,062); Swift: **widerlegt** (t +0,05, p 1,000) | §7.5, §9 |
| Trägt die Skalenregel-Asymmetrie (V-RMS)? | **Toter Gleichstand** (t +0,21 auf beiden Längen) | §7.9 |
| Trägt Hadamard bei 4-Bit-Lloyd? | **Gleichstand** (2K +0,21, 4K −1,31/−0,17) | §7.9 |
| Trägt Clipping nach der Rotation? | **Nein**: (0,94;0,94) +34 %, OSCAR-Paar +111 % | §7.11 |
| Trägt Fenster/Sink (exakte jüngste/erste Zellen)? | **Ja, eindeutig**: 4-Bit 128/8 −31 % KL (`q4lm32` 0,01183→0,00815), 512/8 −49 %, je 8/8 Segmente; ≤3-Bit 128/8 −51 % (8/8). Mit **`R = 64` (OSCAR-Knie 256/64) erreichen beide Basen `int8`-Niveau** (ns), die 3-Bit-Basis als billigster Punkt: 0,00702 bei ≈5,9 gegen 8,25 Bit (−29 %). **Sink (`R`) ist ein eigener Hebel**: R 8→128 bei festem S = 512 bringt +0,0020…0,0026 KL zurück (8/8 bzw. 7/8) | §7.13, §7.15, §7.16 |

Die positiv entschiedenen Achsen sind das **Codebook** (schließt die `q4_0`-Lücke **bei unveränderten Bytes**;
`q4_0` bleibt 4,5 bit/Wert) und — seit §7.13/§7.15/§7.16 — das **Fenster** (eindeutig bei 4 Bit und bei ≤3 Bit;
mit Knie-Fenster erreichen **beide** Basen `int8`-Niveau, die 3-Bit-Basis am billigsten). Alle weiterführenden
Achsen stehen bislang bei Negativbefund oder Gleichstand; das Layout-Produkt (WP-2) ist damit noch nicht begründet.

### 11.3 Das Feld, seit dem Survey (2026-10-14) — neu gelesen

Vier Arbeiten, die im Survey noch nicht stehen:

- **SAW-INT4** (arXiv:2604.19157, 2026-04). Untersucht genau, was unter Serving-Randbedingungen (paged layout,
  fused attention) noch trägt: **token-weises INT4 + blockdiagonale Hadamard-Rotation** ist der beste
  Genauigkeit/Effizienz-Punkt; Vektor- und Hessian-Quantisierung bringen „nur marginale Zusatzgewinne, sobald
  Serving-Kompatibilität zählt". Fused rotation+quant ohne meßbaren End-zu-End-Overhead. **Deckt sich mit unserem
  4-Bit-Befund** (Codebook/Hadamard schlägt Kalibrierung) und **stützt WP-3 als Vergleichsarm**, nicht als
  Hauptlinie.
- **UltraQuant** (arXiv:2606.20474, 2026-06). 4-Bit-KV für agentische Kontexte; Entwurfsentscheidungen:
  **asymmetrische K/V-Behandlung, Walsh-Hadamard, QJL-Entfernung, Block-Skalen-Varianten**; FP8-KV als starker
  Deployment-Maßstab (~2×, nahezu verlustfrei); warnt, daß ein **Codebook Lookup-/Irregular-Access-Kosten**
  einführt. **Fast deckungsgleich mit unserem §9/§10** — bestätigt den `kQ4lMap`-Vorbehalt und reiht int8/FP8 als
  Default-Maßstab ein.
- **KVarN** (arXiv:2606.03458, 2026-06). **Kalibrierungsfreie** KV-Quantisierung: Hadamard plus duale
  Varianz-Normalisierung über **beide** Achsen (Token und Kanal), **2 Bit**, vLLM-Implementierung. Berührt §10
  („kein 2-Bit") und §5.5 (V-Behandlung) — auf unserer 256er-Kopfgeometrie mit QK-Norm **ungeprüft**.
- **Kitty** (arXiv:2511.18643, 2025-11). 2-Bit mit **per-Kanal-Precision-Boost** (Top-~12 % der K-Kanäle auf 4 Bit)
  und festem Dreiregionen-Layout (S Sink FP16, 2-Bit-Body, R Recent FP16). Kanal-Mischprecision ist ein WP-5-naher
  Hebel, den wir nicht messen; das Fenster ist dort wieder unverzichtbar.

Unverändert als Kontext: **KIVI** (2402.02750, K per-Kanal / V per-Token / fp16-Fenster R≤128 — Grund der
Fenster-Priorisierung), **FlatQuant** (2410.09426, Tab. 11: K4V3 near-lossless, QA −0,08 — die **einzige** Evidenz,
auf der die WP-2-Form jetzt noch ruht, §9), **OSCAR** (2605.17757, Clip-Schwellen 0,96/0,92 für unrotierte
K-Kanäle) und QuaRot/KVQuant als Rotations-/Per-Kanal-Kontext.

### 11.4 Bewertung

**Stärken.**

1. **Die Methodik ist überdurchschnittlich.** Gepaarte KL statt PPL als Entscheidungsmaß, exakter McNemar,
   Power-Regel (§8.6), selbst-diagnostizierende Anker (KL reproduziert auf 4–5 Stellen), ehrliche BPE-Bilanz,
   opt-in/default-neutral per Konstruktion. Der Beleg ist §9: die eigene Q2_0-Richtung wurde auf Swift
   **falsifiziert** und der Plan angepaßt, nicht die Messung.
2. **Die Achsen-Priorisierung deckt sich mit dem Feld.** Tail (Fenster) und Clipping vor dem Layout, Kalibrierung
   nur als Vergleich — SAW-INT4 stützt genau diese Reihenfolge.
3. **Ein auslieferbares Ergebnis.** `q4lm32s` schließt die `q4_0`-Lücke bei gleichen Bytes (§7.4); datenblind,
   default-neutral, ohne Kalibrierung.

**Schwächen und Risiken.**

1. **Kein Byte gespart.** `q4_0` bleibt 4,5 bit/Wert; die Gewinne sind verlustfrei, aber nicht komprimierend. Das
   Produkt entsteht erst in WP-2 (echtes K4/V3-Layout). Bis dahin ist die Kette eine Qualitäts-Übung.
2. **Der Layout-Träger ist nicht eingelöst.** WP-2 soll auf der K-vs-V-Zuteilung ruhen, die unser eigenes Swift-Set
   gerade verworfen hat (§9); sie ruht jetzt auf fremder Evidenz (FlatQuant Tab. 11). Ohne cwe-4k-Power oder
   McNemar-primär baut WP-2 auf einem nicht gezeigten Effekt.
3. **Serving-Reife ist halb gemessen.** §7.17 füllt die Speed-Lücke: 8 Arme × 3 Runden auf Swift, 16K-Kontext —
   das Format ist im Decode **frei** (Spanne ±6 %, `int8` führt), und der Fenster-Überbau ist im Decode nicht
   sichtbar. Ungemessen bleibt die Auslieferungsform: der A/B lief im **Proben-Überbau** (fp16-Zwilling jeder
   Zelle, R1/R2), und `kQ4lMap` liegt noch in `__constant__` (Worst Case, §7.10). Kein Tensor-Core-Prefill
   (Prefill läuft über den Decode-Kernel), Prefill-KV wird mitquantisiert (Feldstandard ist decode-only), und
   der Fenster-Pool ist teuer und streamt nicht (§7.17: 24 KiB je Kontextzelle, 64K ≈ 1,5 GiB, unabhängig von
   `--kv-resident`). SAW-INT4/UltraQuant gewinnen genau hier.
4. **„3 Bit tot" ist eine Frage des Byte-Budgets — und der Sink-Tiefe.** Die Sonde hatte kein Fenster/Sink (§5.4);
   §7.13 zeigt bei 4 Bit, daß das Fenster 31 % (128/8) bzw. die Hälfte (512/8) der KL zurückholt (je 8/8, Knie
   S ≈ 512). §7.15 mißt denselben Arm bei ≤3 Bit: das Fenster trägt genauso (128/8 −51 %, 8/8). §7.16 löst die
   beiden dort offenen Arme ein (OSCAR-Knie `R = 64`, Sink-Paar 512/128, auf beiden Basen): **mit Fenster sind beide
   Basen von `int8` nicht zu trennen**, und die 3-Bit-Basis ist der billigste `int8`-nahe Punkt — `q4lm8v`+w
   (256/64) liegt bei 0,00702 und ≈5,9 Bit gegen `int8`s 8,25 (t −1,92, ns). Pro Byte verliert die 3-Bit-Basis in
   keiner der fünf gemessenen Formen meßbar (`|t| ≤ 1,50`); nominell führt die 4-Bit-Basis in 4 von 5 Formen, aber
   die Differenz (≤0,0022) ist ns und von derselben Größe wie der S/R-Formeffekt. Das §7.15-Cross-Over (≈4,8–5,3 Bit)
   ist damit ein Artefakt des zu knappen `R = 8`. „Tot" gilt also ohne Fenster, nicht als Byte-Aussage. `int8`
   bleibt die absolute Qualitätsdecke.
5. **Der Clipping-Negativbefund gilt nur post-Rotation.** Das Feld clippt pre-Rotation/per-Kanal (OSCAR,
   FlatQuant); dieser Arm ist bewußt nicht gebaut (§5.5) — „Clipping ist tot" ist entsprechend eng zu lesen.

### 11.5 Empfehlungen (ohne Planänderung)

1. **≤3 Bit × Fenster — erledigt (§7.15), entschieden (§7.16).** Mit `R = 64` erreichen beide Basen `int8`-Niveau,
   die 3-Bit-Basis bei ≈5,9 statt 8,25 Bit; die verbleibende Format-Differenz ist ns. Damit entscheidet nicht mehr
   das Format, sondern die **Fensterform**: `R` ist ein eigener Hebel (R 8 → 128 bei festem S = 512 bringt
   +0,0020…0,0026 KL zurück, richtungskonstant über 7–8 von 8 Segmenten). **Arbeitspunkt ab hier:
   Fenster `S/R = 256/64` auf der 3-Bit-Basis `q4lm8v`** (≈5,9 Bit, KL 0,00702, von `int8` nicht zu trennen) —
   das ist der billigste Punkt, der `int8`-Niveau erreicht; 128/8 (≈4,8 Bit) bleibt meßbar darüber. Wichtiger als
   beides wäre ein **längerer Kontext** (der Fenster-Anteil sinkt mit L, das Byte-Bild kann wandern — bei
   2K gemessen).
2. **K-vs-V-Power vor WP-2 einlösen** — cwe-4k oder McNemar-primär (§8.6) —, sonst nagelt WP-2 eine Zuteilung
   fest, die unser Setting nicht zeigt.
3. **Speed-A/B gelaufen (§7.17), `kQ4lMap`→Shared-Memory offen.** Die KV-Arm-Liste (fp16/int8/q4lm32/q4lm32w/
   q4lm32c/q4lm8v ± Fenster) ist auf Swift bei 16K gefahren: das Format ist im Decode **frei** (±6 %, `int8`
   führt), der Fenster-Überbau unsichtbar. Offen bleibt die **Auslieferungsform** (der Lauf war der
   Proben-Überbau, s. §7.17) und `kQ4lMap`→Shared-Memory (§7.10) — Serving ist das Wettbewerbsfeld.
4. **`q4lm32s` als eigenen opt-in-Default-Kandidaten prüfen** (Identitäts-Gate + Speed-A/B, §8.1/§8.3) — der
   billigste echte Nutzen, unabhängig von WP-2/3.
5. **Decode-only-Arm erwägen** (Prefill-Pfad für das Format), weil das der auslieferbare Zustand wäre und unsere
   absoluten Lücken sonst nicht mit Feldzahlen vergleichbar sind (§8.7).
6. **KVarN/Kitty als Vergleich lesen** (kalibrierungsfreies 2-Bit, Kanal-Mischprecision), bevor „kein 2-Bit" in §10
   endgültig verriegelt ist.

Das Addendum ändert keinen Beschluß in §7.6/§8/§10; es hält fest, wo der Plan gegen den Stand vom 2026-10-16 steht.

## 12. Glossar

Die Begriffe so, wie dieses Dokument sie gebraucht — mit Zahlenbeispielen aus den Läufen oben. **Fremdnamen
(Papiere) sind als solche gekennzeichnet**; alle Zahlen ohne Fremdnamen sind Strata-Messungen mit dem genannten
Modell/Set.

### 12.1 Messgrößen und Statistik

- **Position:** eine vorhergesagte Token-Stelle im Testset, an der beide Arme verglichen werden. cwe-2k hat 16.351.
- **KL (mean KL):** mittlere Kullback-Leibler-Divergenz zwischen der fp16-Referenzverteilung und der Verteilung des
  quantisierten Armes, über alle Positionen (in nat; 0 = identisch). **Kleiner ist besser.** Der
  **Entscheidungsleser** für Format-Rangfolgen (§8.7).
- **KL p99:** das 99-%-Perzentil der Positions-KL — der **Tail**. Ein guter Mittelwert kann einen fetten Fehler-Tail
  verbergen; p99 sagt, wie schlimm die schlimmsten Positionen sind.
- **ΔKL (delta-KL):** die KL-**Differenz** zweier Arme, gepaart berechnet. Vorzeichen: **linker Arm minus rechter**
  — positiv = linker Arm hat höhere KL (schlechter).
- **gepaart (paired):** beide Arme durchlaufen dieselben Segmente/Positionen; verglichen wird die Differenz **je
  Segment**, dann über die Segmente gemittelt. Der Text kürzt sich heraus → viel kleinere Streuung als beim Vergleich
  zweier unabhängiger Mittelwerte.
- **sd / sem:** sd = Standardabweichung der 8 Segment-Differenzen; sem = sd/√n = ihr Standardfehler (n = Segmente).
- **t:** die t-Statistik = ΔKL / sem. Grob: |t| ≳ 2,4 ≈ p < 0,05 zweiseitig bei 8 Segmenten (df = 7), |t| ≳ 3,5 ≈
  p < 0,01. t **allein** genügt nicht — gelesen wird t **und** „x/8 Segmente".
- **Segment(e) (n):** das Testset zerfällt in **8 zusammenhängende Textblöcke** (~2.048 Tokens). „gepaart über die 8
  Segmente" = diese 8 sind das n der Statistik.
- **„x/8 Segmente up":** in wie vielen der 8 Segmente die Differenz das erwartete Vorzeichen hat. 8/8 = alle gleich
  gerichtet (robust); 4/8 = Münzwurf (nicht belastbar, egal wie groß t).
- **top-1:** Anteil der Positionen, an denen der quantisierte Arm dieselbe Top-1-Token wählt wie fp16 — grobes
  Zustimmungs-Maß.
- **PPL (Perplexität):** exp(mittlere negative Log-Likelihood). **Vorsicht:** der KV-Fehler **dämpft**; PPL fällt oft
  *unter* fp16 (2,992 < 2,999), ohne besser zu sein. Deshalb ist **KL** der Entscheidungsleser, PPL nur Kontext.
- **McNemar:** gepaarter Test auf binäre Antwortrichtigkeit (wie oft A richtig / B falsch, und umgekehrt) — primär
  für die K-gegen-V-Frage (§8.7).
- **Power:** die Fähigkeit, einen Effekt gegebener Größe zu zeigen. Ein Effekt der Sondengröße (t ≈ 0,6 bei n = 8)
  braucht ~87 Segmente für t = 2,0; sonst ist „Richtung ohne Power" kein Urteil (§8.6).

### 12.2 Formate und Quantisierungsnotation

Das `--kv`-Argument (auch die Formatspalte in den TSVs):

| Name | Was | Pegel K/V | Pegel-Bits Ø | gespeichert |
| --- | --- | ---: | ---: | ---: |
| `fp16` | Referenz, kein Verlust | – | 16 | 16 |
| `int8` | 8-Bit-Ganzzahl + Skala (Default-Maßstab) | – | 8 | ≈8,25 |
| `q4_0` | ggml-Block, 32 Werte je 18-B-Block | – | 4 | 4,5 |
| `q4lm32` | 4-Bit-**Max-Lloyd**-16-Pegel-Codebook, gleiche 18-B-Blöcke | 16/16 | 4,0 | 4,5 |
| `q4lm32s` | wie `q4lm32`, Skala per **einem Least-Squares-Schritt** statt `max\|x\|/87` | 16/16 | 4,0 | 4,5 |
| `q4lm8` | 8-Pegel-Codebook, beide Seiten | 8/8 | 3,0 | 3,5 |
| `q4lm8k` | 8-Pegel K / 16-Pegel V | 8/16 | 3,5 | 4,0 |
| `q4lm8v` | 16-Pegel K / 8-Pegel V („V-seitig gespart") | 16/8 | 3,5 | 4,0 |
| `q4lm32t` | 4-Bit, V mit **RMS-Skala** statt max (Behandlung) | 16/16 | 4,0 | 4,5 |
| `q4lm32nr` | 4-Bit, **No-Rotation** (Hadamard aus) | 16/16 | 4,0 | 4,5 |
| `q4lm32c` | 4-Bit, **Clipping**-Skala aus κ-Quantil | 16/16 | 4,0 | 4,5 |
| `q4lm32w` / `STRATA_KV_WINDOW=1` | 4-Bit **+ Fenster** (§12.3) | 16/16 | 4,0 | 4,5 |
| `k8v4` | Altformat K 8-Bit / V 4-Bit | – | – | – |

- **Pegel-Bits vs. gespeicherte Bits:** Pegel-Bits = log2(Pegelzahl) je Wert; gespeichert kommt der **Metadaten-Overhead**
  dazu (beim 18-B-Block 0,5 Bit/Wert: 16 B Payload + 2 B Skala für 32 Werte). Für die Byte-Bilanz zählt nur die
  **gespeicherte** Zahl — so ist §7.13 gebaut, und so ist die Bit/Wert-Spalte hier gebildet.
- **Rotation / Block:** der KV-Vektor wird vor dem Quantisieren mit einer **FWHT** (schnelle Walsh-Hadamard-Transformation)
  rotiert; quantisiert wird blockweise (18 B). `q4lm32nr` schaltet die Rotation ab.
- **Codebook:** die festen Stützwerte, auf die gerundet wird; `q4lm32` nutzt Max-Lloyds 16-Pegel-Tabelle für N(0,1),
  `q4lm8*` die 8-Pegel-Tabelle (`{±69, ±43, ±24, ±8}`). Kein Training, datenblind.

### 12.3 Fenster (window / sink)

- **Fenster / Sink:** die **jüngsten** Zellen (recent) und die **ersten** Zellen (sink) werden **exakt (fp16)**
  gehalten, der Rest bleibt quantisiert. Motiv: Attention konzentriert sich stark auf die ersten Token und die
  jüngsten.
- **Notation `S / R`** (z. B. `128 / 8`): **S = Zahl der exakten jüngsten Zellen**, **R = Zahl der exakten ersten
  (Sink-)Zellen**. **`0 / 0` = Fenster aus** (Identitäts-Nullpunkt). Herkunft der Buchstaben: `win_s` = recent,
  `win_r` = sink (`qsa_decode_attn.hpp`). **Achtung Notation:** §5.6 schrieb noch `N` (jüngste) und `S` (Sink) —
  dasselbe, andere Buchstaben.
- **Arm-Namen in den TSVs** (z. B. `8vw-s128-r8`): `8v` = `q4lm8v`, `w` = mit Fenster, `s128` = S, `r8` = R.
  `win-s512-r8` = Fenster auf dem 4-Bit-Arm, S = 512 / R = 8; `32w-s256-r64` = `q4lm32w`, S = 256 / R = 64.
- **Knie (knee):** das S, ab dem ein größeres Fenster nichts mehr bringt. Im `R = 8`-Gitter liegt es bei beiden
  Basen bei S ≈ 512; mit `R = 64` wird es zwischen 256/64 und 512/128 flach (§7.16).
- **OSCAR-Knie:** der in der OSCAR-Arbeit gute Punkt R = 64 (Sink) / S = 256 (recent) — hier als Sink-Punkt
  nachgemessen (§7.16: 0,00702 auf `q4lm8v`, 0,00782 auf `q4lm32w`, beide `int8`-nah).
- **Seitenpool / `max_cells`:** die **Probe** hält einen fp16-Zwilling des **ganzen** Kontexts (`qsa_win_bytes` über
  `max_cells`); der Leser bedient daraus nur das Fenster. Alle Arme haben so dieselbe Fußspur — die Messung isoliert
  die **Lesepolitik**, nicht die Auslieferungsbytes. Größe (§7.17, gemessen): `max_cells × n_head_kv × head_dim × 4 B`
  je Layer = 2.048 B/Zelle, bei **12** QSA-Layern dieses Modells (`qsa_interval = 4`) also **24 KiB je Kontextzelle**
  über alle Layer (16K ≈ 384 MiB, 64K ≈ 1,5 GiB). Der Pool folgt `max_context`, **nicht** dem Resident-Set —
  `--kv-resident` verkleinert ihn nicht. **Auslieferung wäre ein Ring** von `S + R` Zeilen (7,5 MiB bei 256/64,
  kontextunabhängig, §7.17) — die Streaming-Unverträglichkeit oben ist die der Probe.

### 12.4 Bytes / BPE

- **Bit/Wert (Auslieferung, Modell):** `Basis + (16 − Basis)·(S+R)/L`, wobei `Basis` die **gespeicherten** Bits des
  Formats und **L** die mittlere Zahl gescorter Zellen je Sequenz ist (cwe-2k: L ≈ 2044 = 16.351 / 8). Das ist eine
  **Rechnung, keine Messung** — die Probe-Bytes sind für alle Arme gleich.
- **BPE (bits per element):** „ehrliche BPE-Bilanz" = Payload + Skalen-/Nullpunkt-Metadaten + Fensterbytes bei
  **genannter** Kontextlänge (§8.5).
- **Cross-Over:** der Byte-Bereich, in dem zwei Basen (hier ≤3 Bit vs. 4 Bit) die Führung **pro Byte** tauschen.
  §7.15 fand ≈4,8–5,3 Bit **im `R = 8`-Gitter**; §7.16 findet mit `R = 64` **keine** meßbare Führung mehr — die
  Format-Differenz (≤0,0022, immer ns) ist kleiner als der S/R-Formeffekt, ein scharfes Cross-Over läßt sich mit
  8 Segmenten nicht mehr angeben.

### 12.5 Läufe, Harness, Namen

- **cwe-2k / cwe-4k:** das feste Testset (RULER „complex word extraction"), 8 Segmente à ~2K (bzw. 4K) Tokens;
  cwe-2k = 16.351 Positionen.
- **Harness (`tools/quant_quality.py`):** `run` fährt **einen** Arm (Engine headless, `--kv FMT`, `--env K=V`);
  `compare` bildet die KL-/Paired-Tabellen. Die Läufe **ersetzen** `--kv`/`--kv-resident` der Config durch das Flag.
- **Identitäts-Nullpunkt / Default-Neutralität:** der Arm mit „aus" (z. B. Fenster S = R = 0) muss **byte-gleich**
  zum Basisarm sein (SHA256 der TSV), und die Anker (`fp16`/`int8`/…) müssen frühere Läufe reproduzieren. Ein neues
  opt-in-Verhalten ist per Konstruktion default-neutral — das **zeigen**, nicht behaupten.
- **Scoring (fester Text) / Prefill:** der Harness gibt dem Modell den Referenztext vor und liest die Logprobs
  (keine freie Generierung); **Prefill-KV wird mitquantisiert** (Feldstandard ist decode-only) — interne Vergleiche
  bleiben gültig, absolute Lücken sind **nicht** mit Feldzahlen vergleichbar (§8.7).
- **Parität (`kv_q4_parity`):** bit-genaue Kernel-Tests gegen Host-Referenzen (12/12 grün).
- **Speed-A/B (`tools/ab_engine.py`):** ein frischer Server je Arm, Runden alternierend; jede Runde wärmt mit einer
  kurzen Antwort, dann dieselben Anfragen. Gelesen wird die Engine-eigene Zahl `predicted_per_second` (Decode) und
  `prompt_per_second` (Prefill) — **nicht** PPL. Das Überbaute (fp16-Zwilling, Extra-Launch) macht die Zahl der
  Fenster-Arme zu einer **Obergrenze**, nicht zur Auslieferungszahl (§7.17).

### 12.6 Fremdnamen (Papiere — nicht unsere Messungen)

- **KIVI** (2402.02750): K pro Kanal, V pro Token, fp16-Fenster (recent R ≤ 128) — Grund der Fenster-Priorisierung.
- **OSCAR** (2605.17757): Clip-Schwellen 0,96/0,92 für **unrotierte** K-Kanäle; Tab. 5 Knie (64, 256).
- **FlatQuant** (2410.09426): Tab. 11 K4V3 „near-lossless" — die einzige fremde Evidenz unter WP-2.
- **SAW-INT4** (2604.19157), **UltraQuant** (2606.20474): 4-Bit-KV unter Serving-Bedingungen (paged/fused).
- **KVarN** (2606.03458), **Kitty** (2511.18643): kalibrierungsfreies 2-Bit bzw. Kanal-Mischprecision.
