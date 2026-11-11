# Der Fast-Path für nichtlineare (Level-Tabellen-) KV-Codecs: Ansätze, Literatur, RDNA4-Sachlage

**Stand: 2026-11-11.** Recherche, während die 128K-Läufe (D1) liefen. Fragen:
(1) Wie lässt sich ein **nichtlinearer** KV-Codec (Lloyd-Level-Tabelle) über einen schnellen
Attention-Pfad führen? (2) Welche Ansätze gibt es prinzipiell? (3) Welche sind für Strata **elegant**
und passend? (4) Was ist auf **RDNA4** anders — und was haben die Community-Benchmarks dazu gesagt?

**Kurzfassung:** Es gibt zwei tragfähige Familien — **(I) vor dem MMA dequantisieren** (Codebook/LUT:
TurboQuant, FLUTE, PolarQuant, Beyond) und **(II) den Codec affine machen** (Nullpunkt-Familie: KIVI,
OffQ, FP4/MXFP4). Familie (I) ist die, mit der die Literatur heute läuft, und sie ist **portabel über
die Backends**; Familie (II) ist die, die auf Strata **elegant** ist, weil sie den vorhandenen Kernel
unverändert lässt. Und die RDNA4-Sachlage: **unser 2,25×-Preis ist ein Prompt-Preis, und der
Matrix-Core-Pfad liefert auf unserer Karte keinen Prompt-Gewinn** (gemessen) — die Matrix-Core-Achse
ist damit **nicht** der Fast-Path, sondern ein **Decode**-Hebel; auf den iGPUs ist derselbe Pfad ein
**2,4×-Prefill**-Hebel.

## 1. Was unser Preis really ist (der Ausgangspunkt)

| Arm | Prompt-Lesedauer (32K-Nadel, freie Maschine, gleicher Build) |
|---|---:|
| `int8` | 51,7 s |
| `q4_0` (uniformes Gitter, **affin**) | 50,0 s |
| `q4lm8v` (Lloyd-Tabelle, **nicht-affin**) | 116,4 s (**2,25×**) |
| `q4lm8v` + Kalibrierung | 118,5 s (+1,8 %) |

Der Grund steht im Quellcode (`src/kernels/cuda/qsa_prompt_attn.cu:2518-2524`): Mode 4 (die batched
Prompt-Attention) liest `code - 8` und wendet die Skala **danach** an — das funktioniert nur für ein
**affines** Gitter. Die Lloyd-Tabelle ist nicht affin (`x = level(code)·d`), also wird sie abgelehnt, und
zurück bleibt der Decode-Kern als Fallback (32 Queries pro Schritt). **Der Eigenkommentar des Kernels
benennt die Schuld:** *„Mode 4 could carry the level table the way it carries the int8 codes; **that
is owed if a level-table format ships**, and not needed to measure one."*

**Also: der zu lösende Teil ist die Affinität, nicht die Matrix-Cores.** Ob der Fast-Path danach
zusätzlich auf die Matrix-Cores geht, ist eine **andere, kartenspezifische** Frage (Abschnitt 4).

## 2. Familie (I): vor dem MMA dequantisieren (die Codebook-Familie)

**Der Referenzfall ist die AMD-Produktivierung von TurboQuant** (Chakrabarti et al., ROCm Blog,
2026-06-11; dieselbe Gruppe wie das arXiv:2607.09683-Paper aus unserer Recherche). TurboQuant ist
**genau unsere Nichtlinearität**: Rotation + **fixed codebook** (Beta-Lloyd-Max) + per-Token-Skala.

**Wie sie es schnell machen (das ist die übertragbare Lehre):**

| Optimierung | Wirkung (AMD, MI355X, Triton → GCN-ISA → FlyDSL) |
|---|---|
| GQA-Grouping + ein parametrisierter Kernel für Prefill und Decode | 2,49–3,30× |
| **SoA-KV-Layout** (coalesced 128-Bit-HBM-Reads) + **Pair-LUT** (FLUTE: zwei Dequant-Werte pro Byte-Load + Gather) + BF16-Dot | +0,2–0,3× |
| **native MFMA-Dispatch** (GCN-ISA-Instruktionen statt `tl.dot`) | +0,05–0,1× |
| 4-Wellen-paralleles Q·K + **V-Dequant in LDS** (alle GQA-Query-Heads teilen einen globalen V-Load) | +0,24–0,36× |
| **BF16-Q-Eingabe** direkt in die MFMA + Matrix-Core-PV | **+1,6–4,4×** |
| breitere MFMA, In-Register-Softmax, transponiertes V-LDS | +3,1–4,6× |

**Ergebnis: 32 % → 65 % (Triton) → 89 % (HIP) → 95 % (FlyDSL) der BF16-Durchsatz**; FlyDSL
überholt BF16 um +15,8 %, FP8-KV führt mit +37,5 %. Der **Decay-Schritt** (Dequantisieren in
LDS/Shared Memory, dann die **BF16**-Werte in die MFMA) ist also genau **unsere A8(a)-Variante
„Dequantisieren vor dem MMA"** — nur mit der LUT-Optimierung von FLUTE und dem GQA-Tiling.

**Die inhaltlichen Lehren der AMD-Studie, die direkt in unsere Leiter gehen:**
- **„Apply rotation and LUT-based quantization to K, and use standard uniform quantization for V."**
  (Keys sind quantisierungsempfindlicher als Values.) Bei uns ist es genau andersherum gemessen: der
  **Level-Split** (16/8 → 16/16) liegt auf der **V**-Seite (1,10 dB Block-MSE), und die
  **Kalibrierung** wirkt auf V stärker (−0,36 dB) als auf K (−0,17 dB). Das ist kein Widerspruch,
  sondern zwei verschiedene Knöpfe: die *Präzision* (Level-Anzahl) und die *Darstellung*
  (Rotation+LUT). Die AMD-Aussage betrifft die Darstellung.
- **„Use Walsh-Hadamard rotation instead of random rotation … both more kernel-friendly and more
  accurate."** ✓ deckt sich mit unserem A7-Befund.
- **„Omit QJL im Default-4-Bit-Pfad."** ✓ wir haben kein QJL.
- **„Skip boundary layers"** (die erste/letzte Attention-Layer sind am empfindlichsten) — **aber nur
  bei Voll-Attention-Modellen**; bei Hybrid-Modellen (unser Fall: 12 von 48 Layern tragen den Cache)
  ist der Effekt schwach, sagen die AMD-Leute explizit. → **für uns kein Hebel.**
- **„The benefit is strongest when KV-cache capacity is the bottleneck. In compute-bound settings,
  TQ can underperform BF16 due to quantization and dequantization overheads."** ← **die Warnung, die
  für unseren Prefill gilt**: unser 2,25× ist ein **Compute**-Fall (Prefill), nicht ein
  Kapazitätsfall. Die LUT-Dequantisierung kostet dort Rechenzeit, die sie in einem
  speicherbegrenzten Decode nicht kostet. **Ein Fast-Path, der den Prefill nur „so weit" bringt wie
  `q4_0`, ist das Richtige; einer, der die Dequant-Kosten in den Prefill legt, ist es nicht.**

**Weitere Mitglieder der Familie:** **FLUTE** (Guo et al., *Fast Matrix Multiplications for Lookup
Table-Quantized LLMs*, EMNLP-Findings 2024) — die LUT-Optimierung selbst; **PolarQuant** (Han et
al., Teil von TurboQuant) — Polarform mit 1-D-Codebooks (immer noch eine LUT, aber eine 1-D-Tabelle);
**Beyond** (EMNLP 2026) — **gelernte** nicht-uniforme Level: die *schwierigste* Variante für den MMA
(nur über den LUT-Weg).

## 3. Familie (II): den Codec **affin** machen (die Nullpunkt-Familie)

**Das Prinzip:** ein **verschobenes uniformes Gitter** mit einem (per Kanal, per Block, per Token)
**Nullpunkt**: `x ≈ s·(code − z)` — das ist **affin** ✓ und reitet damit Mode 4 mit demselben Kernel
wie heute (`Σ qᵢ·(code−z)`, die Skala danach). Das ist genau die Idee, die die Literatur als
**Zero-Point**/asymmetrische Quantisierung führt (OffQ: „can be absorbed into the zero-point"; KIVI:
asymmetrische K-Quantisierung per Kanal) — **und** unsere eigene Kalibrierung ist strukturell
dasselbe Objekt (ein per-Kanal-Offset, exakt und softmax-invariant auf der K-Seite, exakt
zurückaddiert auf der V-Seite).

**Der Preis:** die Lloyd-Tabelle ist **nicht-uniform** (für N(0,1) gerechnet, dichter um null). Ein
verschobenes uniformes Gitter hat diese Dichte nicht → es ist grober. **Diese Differenz ist im
Block-Harness messbar** (`tools/kv_codec_mse.py`, **Host, Minuten**): die Variante „uniform +
per-Block-Nullpunkt" gegen „lloyd16 + max", jeweils mit und ohne die per-Kanal-Kalibrierung.

**Und das ist die elegante Antwort für Strata**, weil sie **zwei Fliegen mit einem Klapfen schlägt:**
- **affin** → Mode 4 reitet ohne Codeänderung am Kernel → **die 2,25× verschwinden**;
- **und die per-Kanal-Kalibrierung bleibt kompatibel**, weil sie eine *exakte* Korrektur ist (K:
  softmax-invariant; V: exakt zurückaddiert) — sie ist **kein** Bestandteil des Codes, sondern eine
  Verschiebung vor der Rotation.

**Die Variantenliste aus A8 shrinks damit auf zwei:**
- **(c) verschobenes uniformes Gitter mit per-Block-Nullpunkt** — **die elegante** (affin, kein LUT,
  kein Kernel-Umbau; der Preis ist die Uniformität, im Harness messbar);
- **(a) Dequantisieren vor dem MMA (LUT)** — die **literaturübliche** (TurboQuant/FLUTE), wenn die
  Lloyd-Tabelle bleiben soll (und wenn der Prefill nicht compute-gebunden ist);
- **(b) stückweise affine Zerlegung** — die aufwendigste, und ohne Beleg, dass sie sich lohnt.

## 4. RDNA4: die Matrix-Core-Achse ist **nicht** unser Fast-Path — und die Community misst sie uneinheitlich

### 4a. Unsere eigene Messung (der Kern)

`bench/results/2026-11-08-wmma-int8/` (`int8`, 4k/16k-Prompts, Commit `c7be7d4b7`),
`STRATA_HIP_WMMA=1` gegen aus:

| | Prompt 4k | Prompt 16k | Decode (short / 4k / 16k) | Sanity |
|---|---:|---:|---|---|
| WMMA **aus** | 479,4 tok/s | 565,4 tok/s | 47,3 / 42,4 / 42,2 | 12/12 ✓ |
| WMMA **an** | 478,0 tok/s | 565,5 tok/s | 45,7 / 45,3 / 42,1 | 12/12 ✓ |

**Die Lesart: der WMMA-Pfad ist bei uns ein Decode-Hebel, kein Prompt-Hebel.** Die Prompt-Raten sind
unverändert; der einzige Unterschied liegt im Decode — im 4k-Fall +6,9 %, im 16k-Fall ±0, im
Short-Fall −3,4 %, mit **überlappenden Spannen** (also kein gesicherter Durchbruch; „durchgängig" ist
er bisher nicht belegt).

**Die entscheidende Konsequenz für den Plan: unsere 2,25× sind ein *Prompt*-Preis — der WMMA-Pfad
kann sie auf dieser Karte nicht bezahlen.** Die Stage, die ich ursprünglich vorgeschlagen hatte (ein
32K-Nadel mit/ohne WMMA, um den Prompt-Gewinn zu suchen) ist damit **hinfällig**: an der Stelle, wo
unser Preis liegt, liefert der Schalter nachweislich nichts. **Der Fast-Path ist die Struktur
(Abschnitt 1), nicht die Matrix-Cores.**

### 4b. Die Community-Benchmarks für unsere Kartenfamilie (RDNA4)

| Bericht | Karte | WMMA | Was er zeigt |
|---|---|---|---|
| `bench/results/2026-10-10-community-rx-9060-xt/` (**unser**) | RX 9060 XT, gfx1200, **Windows**, Ryzen 7 9800X3D | **an** (`STRATA_HIP_WMMA=1`) | Prompt-Rate skaliert mit dem Prefill-Chunk: 346,3 (2.048) → 593,3 (4.096) → **879,4 tok/s** (8.192); Decode flach 52,7–57,9. **Kein WMMA-A/B im Bericht** — der Schalter gehört zur Baseline. |
| `bench/results/2026-10-03-community-r9700-windows/` | R9700, gfx1201, Windows | (der Bericht verweist auf den AMD-Bericht #499 / RX 9070 XT, Prompts bis 31.059 Token) | der Windows-Fall von RDNA4 ist community-validiert |
| `bench/results/2026-10-08-community-r9700-linux/` | R9700, gfx1201, **Linux** | **aus** (`STRATA_HIP_WMMA=0`), dafür `STRATA_PREFILL_MMQ=1` (der **Prompt-GEMM**-Pfad, nicht die Attention) | **die Community-Praxis auf RDNA4-Desktop dreht den Matrix-Core-Attention-Pfad nicht an** |
| PR **#5535** | RX 7900 XTX, gfx1100 (RDNA3) | int8-WMMA-Tier | −32,1 % **Kernel**-Zeit, aber **kein End-to-end-Gewinn** (0,995, CI [0,994; 1,004]) → dieselbe Aussage wie bei uns; und: „RDNA4 ships the same instruction family and already defaults to the f16 WMMA prompt path — the layout re-proof on an RDNA4 card is one probe run" |
| PR **#5725** | gfx1152 iGPU | WMMA-Pfad freigeschaltet | **32K-Prefill 79-80 → 194-201 tok/s (2,4×)**, 4K 84-86 → 97-101 (1,17×), 131K 179,9. Das Rezept: **zwei Hälften, beide nötig** — compile-time der Arch-String in die WMMA-/dp4a-Device-Guards (9 Dateien), runtime in die **zwei** Host-Gates (`gfx_arch_is_gfx11_wmma()`, `igpu_prompt_defaults_arch()`) |

**Das Muster:** Auf den **iGPUs** ist der Matrix-Core-Prompt-Pfad ein **2,4×-Prefill**-Hebel bei 32K
(bei 4K nur 1,17×). Auf den **Desktop-RDNA4-Karten** (gfx1200/1201) liefert er **keinen** Prompt-Gewinn
(bei uns gemessen; auf RDNA3 derselbe Befund) und wird von der Community **uneinheitlich** benutzt
(Windows-Bericht: an, Linux-Bericht: aus). „Wo der Hebel liegt", hängt an der Karte — und
**auf unserer** Karte liegt er (wenn überhaupt) im **Decode**, nicht im Prompt.

### 4c. Ein Nebenbefund aus unserem eigenen Community-Bericht, der größer ist als der WMMA-Schalter

Der 9060-XT-Bericht misst den **Prefill-Chunk** als Konfigurationshebel: **2.048 → 8.192 verdoppelt
die Prompt-Rate** (346,3 → 879,4 tok/s, **2,5×**), 16.384 bringt nichts mehr. Das ist auf unserer Karte
**größer als jeder Schalter**, und es ist eine *Konfiguration*, kein Kernel. (Für den 2,25×-Vergleich
zwischen `q4_0` und dem Lloyd-Arm ist der Chunk irrelevant — beide Arme liefen mit derselben
Geometrie; aber als * absoluter* Prompt-Hebel gehört er in dieselbe Liste.)

## 5. Was ich daraus für den Arbeitsplan vorschlage

1. **S6 wird umgestellt und verkleinert:** der 2,25×-Preis ist ein **Struktur**-Preis (Mode 4 vs.
   Decode-Kern), **kein** Matrix-Core-Preis. Die Matrix-Core-Achse ist (a) kartenspezifisch, (b) auf
   unserer Karte ein **Decode**-Hebel ohne gesicherten Durchbruch und (c) für den Prompt-Preis
   irrelevant. Sie gehört **nicht** in die Fast-Path-Stufe, sondern als eigener, kleiner Punkt
   („Decode-Hebel des WMMA-Pfads: durchgängig belegen oder streichen").
2. **Der Test, der die Architektur entscheidet (Host, Minuten):** im Block-Harness die Variante
   **„uniform + per-Block-Nullpunkt"** (affin!) gegen **„lloyd16 + max"**, jeweils mit und ohne die
   per-Kanal-Kalibrierung. **Fällt der Uniformitäts-Preis klein aus, ist Variante (c) der Produktcodec
   — und der Fast-Path kostet keinen Kernel.** Fällt er groß aus, bleibt die Lloyd-Tabelle, und der
   Fast-Path wird der LUT-Weg (TurboQuant/FLUTE) — mit dem AMD-Vorbehalt: im **compute-gebundenen
   Prefill** zahlt die Dequantisierung.
3. **Die Reihenfolge:** (i) der Harness-Test des Nullpunkt-Gitters (Minuten, Host), (ii) die
   Level-Tabelle in Mode 4 (falls (i) nicht trägt), (iii) der LUT-Weg (falls (i) zu teuer ist).

**Quellen:** AMD ROCm Blog „Productionizing TurboQuant on AMD GPUs for KV-Cache-Bound LLM Inference"
(2026-06-11) · Guo et al., *Fast Matrix Multiplications for Lookup Table-Quantized LLMs*, EMNLP
Findings 2024 (FLUTE) · Zandieh et al., *TurboQuant*, ICLR 2026 (arXiv:2504.19874) · Han et al.,
PolarQuant (in TurboQuant) · *Beyond: Better-than-Full-Precision KV Caches via Learnable
Non-Uniform Quantization*, EMNLP 2026 · D'Alberto et al., arXiv:2607.09683 (unsere Rolle) · Jokar,
*Transforms for LLM Quantization*, arXiv:2608.25188 (der Zahlformat- Strang: FP4/MXFP4/NVFP4 sind
uniform → **affin** → der Fast-Path kommt dort gratis; die „Great Inversion" sagt, dass das uniforme
FP4-Gitter das Flatten weniger belohnt → **Codec- und Rotationswahl hängen aneinander**) · OffQ
arXiv:2606.07116 (Nullpunkt) · KIVI (asymmetrisch per Kanal) · die Community-Berichte
`bench/results/2026-10-10-community-rx-9060-xt/`, `.-2026-10-03-community-r9700-windows/`,
`.-2026-10-08-community-r9700-linux/` · Upstream-PRs **#5535** und **#5725** · unsere eigene
WMMA-A/B-Messung `bench/results/2026-11-08-wmma-int8/`.
