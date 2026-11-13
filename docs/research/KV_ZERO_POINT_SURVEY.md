# Nullpunkt, Offset und Fit: die einschlägige Literatur (Stand 2026-11-12)

**Zweck:** die Belege zu `q4u32zc` — dem uniformen Gitter mit **per-Block-Nullpunkt** (dem Shear) plus
der per-Kanal-Kalibrierung. Der Abschnitt 1 ist die **Numerik** (was die Theorie ueber den optimalen
Quantizer mit freiem Nullpunkt sagt), der Abschnitt 2 die **KV-Cache-Literatur**, der Abschnitt 3 die
**Runtime** (was der Nullpunkt im Attention-Kern kostet), der Abschnitt 4 die **GPU-Primitive**, der
Abschnitt 5 die **Luecken** (was niemand gemessen hat — und was unser Beitrag waere).

Belege aus zwei Recherche-Runden (2026-11-11) und aus der eigenen Recherche-Notiz
(`KV_FASTPATH_LITERATURE.md`). Jede Zahl traegt ihren Kontext; Unsicherheiten sind als solche
markiert.

## 1. Die Numerik

### 1a. Der optimale Uniform-Quantizer (der Kern unseres Fits)

| Arbeit | was sie sagt | für uns |
|---|---|---|
| **J. Max, „Quantizing for Minimum Distortion", IRE Trans. Inf. Theory IT-6(1):7–12, 1960** (DOI 10.1109/TIT.1960.1057548) | der Fall **gleichabständiger Rekonstruktionswerte** wird **eindimensional** (nur die Intervalllänge r), Gl.(8) numerisch gelöst | **das ist unsere Regel**: ein uniformes Gitter, dessen **Lage** (der Nullpunkt) und **Weite** (der Schritt) gesucht werden. **Achtung:** der oft zitierte Titel „An optimal class of companding laws" mit diesem Inhalt **existiert so nicht** — die Rechnung der „Iteration `s = sqrt(12·MSE)` plus zirkuläres Mittel" ließ sich in **keiner** publizierten Quelle in dieser Form verifizieren |
| **Bennett (1948)**, **Panter-Dite (1951)** | `D ≈ s²/12` für einen Uniform-Quantizer, also `s ≈ √(12·D)` | die Schrittweite aus dem Fehler — die Formel, die in der Praxis (TensorRT u. a.) als *eine* Iteration benutzt wird |
| **Lloyd-Max-Iteration** (Max 1960; Übersicht: **R. M. Gray, D. L. Neuhoff, „Quantization algorithms", IEEE Trans. Inf. Theory 44(6):673–688, 1998**) | die klassische Iteration zwischen Zuordnung und optimaler Rekonstruktion | die Form unseres Fits, nur mit **zwei** Parametern (Lage und Weite) statt einem |
| **Bucklew & Gallagher (1982)** | die **Initialisierungsempfindlichkeit** von Max' Verfahren | **unser wichtigster Befund passt hierher**: der Start des Fits entscheidet (bei uns: das Gitter muss den Block **decken**; ein Gitter ±max kostet **1,3 dB**, die LS-Korrektur findet den Weg nicht zurück) |
| **Idelbayev et al., CVPR 2021** | der **global optimale symmetrische** Skalenpunkt in O(NK log K) | eine exakte Alternative zur Iteration — für unsere Blockgröße (32) uninteressant, für große Gruppen interessant |
| **PiSO, arXiv:2606.10890** | löst die Skala **exakt** (stückweise quadratisch, geschlossen) und zeigt, dass eine Gittersuche auf `[0, s_absmax]` suboptimale Lösungen übersieht | dieselbe Warnung gilt für unseren Startwert |
| **QUASAR, arXiv:2608.13966** | der **geschlossene** LSQ-Fit für (Skala, Nullpunkt) bei festen Codes; +1,4 % Step-Zeit | bestätigt: bei **festen** Codes ist der Fit **geschlossen** — die teure Stelle ist das **Neucodieren** (also unsere Pässe) |

### 1b. Der Nullpunkt: wann er überhaupt etwas bringt

| Arbeit | was sie sagt | für uns |
|---|---|---|
| **Fleischer (1964)**, **Trushkin (1983)** | bei **symmetrischer** Dichte ist der optimale Quantizer **symmetrisch** (Offset 0) — eindeutig nur bei Log-Konkavitaet | der Nullpunkt hilft nur bei **schiefen** Daten |
| **Kabal & McGill (1984)** | bei **symmetrischer, aber nicht log-konkaver** Dichte (z. B. Gamma) ist der optimale Quantizer **asymmetrisch** (Offset ≠ 0) | **genau unser Fall**: die rotierten K/V-Zeilen sind symmetrisch in der Norm, aber **schief** in der Verteilung — der fitted Nullpunkt folgt der Schiefe (r = +0,89 mit dem Rest, der Blockmittelwert trägt 0,03 Schritte bei) |
| **KVQuant (Hooper et al., NeurIPS 2024, arXiv:2401.18079)** | asymmetrisch (Scale + Zero-Point), bewusst **fp16**-Zero-Point, weil ein Integer-ZP bei manchen Kanälen ausserhalb des Bereichs läge | die **Praxisentscheidung** gegen einen schmalen ZP — bei uns ist der ZP eng (0,92 Schritte im Median), ein 5-Bit-FP wäre ausreichend |
| **OffQ, arXiv:2606.07116** | der Nullpunkt „kann in den Nullpunkt absorbiert werden"; **per Gruppe** (128) **nach** der Rotation, 16-Bit Scale + 16-Bit ZP → **4,25 bit** | **die gleiche Formel wie unsere Rechnung** (0,5 bit/Zelle bei Gruppe 32) |
| **KIVI (Liu et al., ICML 2024, arXiv:2402.02750)** | asymmetrisch mit `z = min`, `s = (max−min)/(2^b−1)`; K **per Kanal** (G=32 Tokens), V **per Token** | die **einfachste** asymmetrische Regel — und der Beweis, dass sie in der Literatur Standard ist |
| **„Regularize or Localize", arXiv:2607.17019** | Zero-Point-Gitter bewegen die NLL **kaum** (+0,74 gegen +0,765 symmetrisch bei 3 bit), kosten aber **2×16 bit je 32 Werte ≈ 1 bit/Wert** | **eine Gegenstimme, keine pauschale** (Korrektur 2026-11-13): der Abgleich zeigt auch **Gruppen-Ablationen mit deutlichem Nutzen asymmetrischer Quantisierung**; umgekehrt warnt **InnerQ** davor, Asymmetrie ueberall als automatisch besser anzusehen. Entschieden wird es ueber Encoder, Gruppierung und Modellqualitaet. Die Arbeit misst **per-Channel/per-Token**, nicht per Block — unser Fall ist von ihr nicht gemessen |

### 1c. Der Block-Fit und seine Kosten (die GPU-Seite der Numerik)

| Quelle | Aussage |
|---|---|
| **ROCm Optimization Guide**, Kapitel „DPP and data-share permutation builtins" (Spalte RDNAG: **RDNA4: Yes**) | DPP ist auf gfx12 verfuegbar (`mov_dpp`, `ds_swizzle`, `ds_permute`); `__shfl` wird auf RDNA zu `ds_bpermute_b32` (LDS-Pfad) |
| **ROCm Guide**, „Wavefront-level builtins" (Beispiel `dpp_warp_reduce`) | die Muster-Variante mit `ds_swizzle`-Broadcast (Maske 0x1e0) |
| **vLLM ROCm**, `csrc/rocm/skinny_gemms_int4.cu` | `REDUCE_SUM_DPP_WAVE32`: 4 DPP-`row_shr` (8/4/2/1, nur 16-Lane-Zeilen) + **ein** `__shfl_xor(16)` fuer den zeilenuebergreifenden Schritt = 5 Ops statt 5 Shuffles |
| **RDNA3 ISA §7.7** | `v_add_f32_dpp`: DPP kann mit der Arithmetik **verschmolzen** werden → die 5 Reduktionsschritte kosten dann ~5 VALU-Ops ohne separates Shuffle-Add. **Fuer gfx12 nicht verifiziert** |
| **RDNA Notes** (HIP-Faustregeln) | Shuffles nur ueber ≤32 Lanes; DPP bevorzugen; `v_perm` fuer Permutationen |
| **rocMLIR PR #2374** | portiert DPP-`gpu.subgroup_reduce` (getestet auf gfx1201) |
| **Einschraenkungen** | `row_bcast`/`wave_ror` sind **CDNA-only**, auf gfx12 nicht verfuegbar; `__builtin_amdgcn_wave_reduce_add_u32` nur fuer Integer |

**Was daraus fuer unseren Writer folgt:** der Fit kostet **3 Reduktionen je Pass** (die Summe von x
ist invariant und wird einmal genommen) — das ist die **theoriegestuetzte Minimalform** ✓. Auf gfx12
koennte DPP die 5 Shuffle-Stufen einer Reduktion ersetzen; **gemessen ist das nicht** (offen).

### 1d. Die Zahlformat- und Transform-Achse (aus unserer frueheren Recherche)

- **Jokar, „Transforms for LLM Quantization", arXiv:2608.25188** — der Zahlformat-Strang: FP4/MXFP4/NVFP4 sind **uniform** → **affin** → ein Fast-Path käme dort gratis; die „Great Inversion" sagt, dass das uniforme FP4-Gitter das Flatten weniger belohnt → **Codec- und Rotationswahl hängen aneinander**.
- **TurboQuant (Zandieh et al., ICLR 2026, arXiv:2504.19874)** und die **AMD-Produktivierung** (ROCm Blog, 2026-06-11): Rotation + **fixed codebook** (Beta-Lloyd-Max) + per-Token-Skala; „Apply rotation and LUT-based quantization to K, and use standard uniform quantization for V"; „Use Walsh-Hadamard rotation instead of random rotation"; die **Decay**-Stufe (Dequantisieren in LDS, dann BF16 in die MFMA) ist unsere A8(a)-Variante; **32 % → 95 % des BF16-Durchsatzes**; die **Warnung**: im compute-gebundenen Prefill zahlt die Dequantisierung.
- **FLUTE (Guo et al., EMNLP Findings 2024)** — die LUT-Optimierung selbst (zwei Dequant-Werte pro Byte-Load + Gather).
- **PolarQuant** (in TurboQuant) — Polarform mit 1-D-Codebooks; **Beyond (EMNLP 2026)** — **gelernte** nicht-uniforme Level (nur ueber den LUT-Weg).
- **Companding-Literatur:** Max (1960) und die Ingenieursform **μ-Law/A-Law**; **Perić et al., „Designing of the piecewise linear optimal compander", KaunasTech/ITC 2013**.

## 2. Die KV-Cache-Literatur (Granularitaet und Byte-Preis)

| Arbeit | Granularitaet | asymmetrisch? | Byte-Overhead (berichtet) |
|---|---|---|---|
| **KIVI** (ICML 2024) | K per Kanal (G=32 Tokens), V per Token | ja (z=min) | **keine bit/cell-Angabe** |
| **KVQuant** (NeurIPS 2024) | K per Kanal **vor RoPE**, V per Token | ja, fp16-ZP | **keine** bit/cell-Angabe; 3-Bit: 3,82–3,88 PPL gewonnen |
| **QServe** (MLSys 2025) | KV4 **per Head** asymmetrisch (fp16 Scale+ZP pro Head pro Page); Gewichte per **Gruppe g=128** mit UINT4-Group-ZP | ja | Group-ZP: die Subtraktion **nicht mehr in die Epilogue fusierbar** |
| **ZipCache** (NeurIPS 2024) | **groupwise** (G=32) | nein | **3,20×** Kompression (groupwise) gegen 3,99× (tokenwise) und 4,00× (channelweise) — das ist das **ehrlichste** Byte-Bild in der Literatur |
| **QuaRot** (arXiv:2404.00456) | KV asymmetrisch, **Gruppe 128** | ja | — |
| **OffQ** (arXiv:2606.07116) | **per Gruppe** (128) **nach** der Rotation | ja | **4,25 bit** (16-Bit Scale + 16-Bit ZP je 128) |
| **TriAxialKV** (arXiv:2605.17170) | asymmetrisch **G=32** mit (Scale, ZP) je Gruppe; INT2-Keys per Kanal | ja | — |
| **InnerQ** (arXiv:2602.23200) | **per-Group-ZP**, dynamische Sym/Asym-Maske, ZP-Matrix dicht gespeichert, das **Sign-Bit des fp16-Scale** als Maske recycelt | ja | **0,5 bit/Element** bei G=32 (fp16-ZP) — **exakt unser Overhead** |
| **AMXFP4** (Findings ACL 2025, arXiv:2411.09909) | asymmetrische FP8-Shared-Scales, **Gruppe 32** | ja | ~10 % MAC-Overhead gegenueber MXFP4 |
| **SAW-INT4** (arXiv:2604.19157) | asymmetrisch INT4 pro Token/Head | ja | fused Rotation-INT4-Kernel 1,33–1,58×, aber **0,28–0,10 % end-to-end** |
| **UltraQuant** (arXiv:2606.20474, AMD CDNA4) | MXFP4 + UE8M0-Skalen, 4,25 bit | **nein** (sie verzichten auf den ZP) | 4,25 bit |
| **GyRot** (arXiv:2607.27694) | INT8-SF/ZP | ja | — |
| **PrismQuant** (arXiv:2609.32429), **QAQ** (ICCV-W 2025) | nicht-uniform + Outlier | nein | — |
| **QuIK** (EMNLP 2024), **RoLoRA** (Findings EMNLP 2024) | Rotation + Mischpraezision | **keine** per-Group-Affine-Offsets | — |

**Formate mit einem Nullpunkt (der Byte-Preis in der Praxis):**
- **llama.cpp `q4_1`**: fp16-Scale + fp16-Min je 32 Werte → **5,0 statt 4,5 bit** ✓ (**unser Wert**)
- **AWQ-Format**: 4-Bit-ZP (8 Nibbles je int32) → 4 + 16/g + 4/g → **4,63 bit** bei g=32, 4,31 bei g=64, 4,156 bei g=128
- **vLLM-ROCm W4A16**: 4-Bit-ZP → **0,125 bit/Element** bei g=32 → **unsere 6-Bit-Variante (0,19 bit) liegt dazwischen**

## 3. Die Runtime: was der Nullpunkt im Attention-Kern kostet

| Quelle | Messung |
|---|---|
| **QServe** (MLSys 2025) | 4-Bit-Dequant kostet **20–90 %** Overhead auf heutigen GPUs; KV4-Attention: **1,7× Speedup auf L40S, aber 1,2× Slowdown auf A100**; bei **per-Group-ZP** ist die Subtraktion **nicht in die Epilogue fusierbar** → **eine Extra-Op pro Gewicht** |
| **llama.cpp auf AMD RDNA4 (RX 9070)** (Feldmessung, `shawnq-msft/rx9070-qwen-rocm`) | `q4_1`/`q5_1` (mit fp16-Min) sind **30–40 % langsamer** als `q4_0`/`q8_0`; **Issue llama.cpp#27796**: `q4_0`-KV −21 % gegenueber f16 im Decode, `q8_0` −4 % (R9700) |
| **vLLM ROCm** (`rdna_hybrid_w4a16`, gfx11/gfx12) | asymmetrisches uint4 mit 4-Bit-Group-ZP: die ZP-Subtraktion erfolgt **pro Zeile des Tiles** (ein fp16 pro Zeile, nicht pro Wert); der **gepackte Schnell-Dequant-Pfad wird bei has_zp deaktiviert** und die Asymmetrie zahlt erst ab M≥256 ein |
| **vLLM PR #40633** (geschlossen) | 4-Bit-ZP **steganographisch in den Mantissa-Bits der fp32-Scale** + die Korrektur **auf der Score-Seite** via `sum(Q)·zp` — **kein Add pro Wert** |
| **InnerQ** (Jetson Xavier NX) | ZP: „negligible effect on latency"; die Sparse-Masken-Ablation: Value-Cache-Latenz 1685 µs (99 % sparse) → 1815 µs (1 % sparse) bei 32K, also **~+7,7 %**, wenn das ZP wirklich angewandt wird |
| **UltraQuant** (AMD **CDNA4**) | auf CDNA4 faltet das scaled-MFMA die **Skala** in den Akkumulator — **ein Nullpunkt tut das nicht**; UltraQuant verzichtet deshalb auf den ZP. **Die Aussage ist auf CDNA4 gemessen und gilt nicht fuer RDNA4** (siehe Korrektur, Abschnitt 3) |
| **GyRot** | auf GPUs muss Group-Scaling + Zero-Point **nach** dem GEMM auf CUDA-Cores in FP erfolgen; der AMX-INT8-Pfad (INT8 Scale/ZP) ist deutlich billiger |

**Der Abgleich mit unserer Analyse:** unsere Korrektur ist **eine Add je Wert** im explizit
dequantisierenden Leser (der Decode-Kern liest die Bloecke zellenweise) — das entspricht QServes
„eine Extra-Op pro Gewicht" ✓; die **K-S.Seite** kann dagegen **ohne Add pro Wert** auskommen, wenn
die Korrektur auf der Score-Seite als `sum(Q)·zp` gefuehrt wird (vLLM PR #40633 ✓) — **das ist genau
die Form, die wir fuer die K-Seite analysiert hatten** (die Query-Gruppensummen, je Kopf einmal ✓✓).
Der **Writer**-Preis ist der andere Teil (bei uns 6 Iterationen mit je 3 Reduktionen gegen 15
Shuffle-Schritte heute); unsere Rechnung sind 2,3–6,3× der Writer-Kosten → **das ist das Warnsignal
fuer Gate G6** ✓.

**Korrektur (2026-11-13): die 30–40 % sind kein Nullpunkt-Preis.** Die Zahl stammt aus einer
Feldmessung von llama.cpp auf einer RX 9070, in der `q4_1`/`q5_1` (4/5 Bit mit fp16-Min) gegen
`q4_0`/`q8_0` **als ganze Kernelpfade** verglichen werden — unterschiedliche Dequant-Pfade,
Gruppengroessen und Fallbacks. Daraus laesst sich **kein isolierter, unvermeidbarer Preis des
Nullpunkts** ableiten, und die fruehere Formulierung „die Literatur nennt fuer den Offset 30–40 % der
KV-Zeit" war zu stark. Was belegt ist: ein per-Group-ZP verhindert heute den gepackten
Schnell-Dequant-Pfad (vLLM ROCm), und die Subtraktion ist bei per-Group-ZP **nicht** in die Epilogue
fuisierbar (QServe). Was **nicht** belegt ist: dass ein Affin-Werden 30–40 % kostet. Ebenso falsch
waere die Gegenbehauptung „affin = kostenlos": die K-Seite ist ueber die Query-Gruppensumme
organisierbar, die V-Seite ueber einen zusaetzlichen Gruppenakkumulator im vorhandenen Durchlauf —
ob das praktisch guenstig ist, entscheiden Metadatenzugriffe, Registerdruck und die konkrete
Kernelintegration, nicht die Algebra.

## 4. Die GPU-Primitive (RDNA4)

- **DPP** (`mov_dpp`, `ds_swizzle`, `ds_permute`) ist auf gfx12 verfuegbar ✓ (ROCm Optimization Guide,
  RDNAG-Spalte „RDNA4: Yes"); `__shfl` laeuft dort ueber den LDS-Pfad (`ds_bpermute_b32`) ✓
- **das belegte Muster**: vLLM ROCm `REDUCE_SUM_DPP_WAVE32` = 4 DPP-`row_shr` + 1 `__shfl_xor(16)` ✓
- **die Verschmelzung mit der Arithmetik** (`v_add_f32_dpp`) ist fuer gfx12 **nicht verifiziert** ✓
- `row_bcast`/`wave_ror`: **CDNA-only** ✓; `__builtin_amdgcn_wave_reduce_add_u32`: nur Integer ✓

## 5. Die Luecken (was in der bisherigen Recherche nicht gefunden wurde)

**Zur Formulierung (2026-11-13):** „nicht gefunden" heisst, dass der Quellenabgleich diese Messung
nicht enthaelt — nicht, dass es sie nicht gibt. Die Luecken sind Suchstand, kein Neuheitsanspruch.

1. **In der bisherigen Recherche nicht gefunden:** eine dB-Angabe fuer den Gewinn eines
   per-Block-Offsets ueber einem Uniform-Gitter mit fp16-Skala — die Arbeiten messen PPL/Task-Scores
   (InnerQ, TriAxialKV). Unsere 1,16 dB Block-MSE (und die Kombination mit der per-Kanal-Kalibrierung:
   3,18 dB) sind damit **im Quellenabgleich ohne Gegenstueck**.
2. **In der bisherigen Recherche nicht gefunden:** eine publizierte Korrelation „Offset ≈ Formfehler
   des Blocks (r = 0,89 mit dem Rest) gegen Mittelwert". **Und die Korrelation ist kein Mechanismus:**
   r = 0,89 sagt, dass zwei Groessen zusammen schwanken, nicht, dass die eine die andere verursacht.
   Die am 2026-11-13 gemessene Zerlegung (bench/results/2026-11-13-kv-offset-tail/) zeigt, dass der
   reine Restfehler-Offset nur einen kleinen Teil des Shear-Gewinns traegt — der Shear ist ueberwiegend
   eine **Neuplatzierung des Gitters** (Codes + Schritt), nicht die DC-Korrektur.
3. **Keine oeffentliche Instruktionszaehlung** „ein Extra-Add pro Wert" gegen „ein Extra-FMA je Gruppe"
   im Decode-Attention-Kern (nur Latenz-Deltas).
4. **Keine begutachtete RDNA4-Arbeit** zu 4-Bit-KV mit per-Group-ZP (die AMD-Daten stammen aus Code
   und Repos, nicht aus Papern).
5. **Keine publizierte Praezisionskosten** eines 8-Bit/5-Bit-Fixed-Point-Offsets **in Schritten der
   Gruppenskala** — nur Formate (AWQ 4-Bit-ZP, `Q4_K` 6-Bit-Min, INT8-ZP bei GyRot) ohne dB-Angabe.

## 5b. Die beiden Papiere, **in unserem Zusammenhang** (2026-11-13)

### UltraQuant (AMD, arXiv:2606.20474v3) — die Hardware-Aussage von **CDNA4** (nicht RDNA4)

**Das Format:** MXFP4 — FP4-E2M1-Codes (4 Bit) + **UE8M0**-Skalen (8 Bit) je **32 Kanäle** →
17 Bytes je Gruppe = **4,25 bit/Element** ✓✓ (unsere Blockgröße ist dieselbe ✓✓). Die Dequantisierung
**faltet in die Matrix-Cores**: `v_mfma_scale_f32_16x16x128_f8f6f4` liest FP4-Keys, FP8-Queries und die
UE8M0-Skalen in **einer** Instruktion ✓✓ — die Keys werden nie in BF16 materialisiert ✓.

**Die Skalenregel („constant-optimized scaling", §5.3):** die Skala ist **eine Zweierpotenz**
(`E = round(log2(c·m))`, `m` = der Absmax der Gruppe) und **`c` ist EINE globale Konstante**
(**c = 0,156**, modellunabhängig ✓✓ — über MiniMax **und** Qwen hinweg ohne Nachkalibrierung ✓✓,
einmal offline auf rotierten Keys gelöst ✓).

**Drei Befunde, die direkt in unseren Plan gehen:**

1. **Ihre Ablation B.1 ist unsere Messung von der anderen Seite:** „switching the adaptation
   statistic from a single per-token ℓ2 norm to a **per-block absmax** (groups of 32) is the
   load-bearing change: it is what recovers accuracy, **independent of the codebook**" — und: „with
   the per-block scale in place **the codebook itself barely matters** — replacing the calibrated
   Lloyd magnitudes with a **uniform grid stays within single-seed noise**" ✓✓✓.
   **Das ist genau unser Arm-Befund:** die Lloyd-Form kostet gegen das uniforme Gitter nur
   **+0,00050 nats (n.s.)** ✓✓ — **bei 2,25× Prompt-Preis** ✓✓✓. **Die Formatentscheidung ist die
   Skalenregel, nicht das Codebook** ✓✓.
2. **Ihr Argument gegen per-Token-Faktoren ist unser Korrekturterm:** „**a per-token factor cannot be
   folded into the attention score (it does not pass through the softmax), so it would add a stored
   scalar and runtime multiply with no quality gain**" ✓✓. **Unser per-Block-Offset ist genau ein
   solcher Faktor** ✓✓ → **auf der K-Seite ist er nicht faltbar** ✗ → **der Shear ist der falsche Hebel
   für den Fast-Path** ✓✓ (und die Messung sagt es unabhängig: der Offset bringt 0,00193 nats, die
   Lloyd-Form kostet 0,00050 — **der Offset ist damit der teurere Hebel, und er ist derjenige, der
   die Hardware-Abschaltung verhindert** ✓✓✓).
3. **Die Skalenform ist eine konkrete Verbesserung unseres Formats:** die Skala als **Exponent**
   (UE8M0, 8 Bit) statt fp16 → der Bulk bei **4,25 bit/Element** (statt unserer 4,5 ✓) **und ohne
   Artefakt** ✓✓ (die Konstante c = 0,156 ist global ✓✓) — gegen unsere Kalibrierung (12 KB je
   Modell + Basisdisziplin ✓).
   **Nachtrag 2026-11-13, auf unserem Gitter gemessen** (`bench/results/2026-11-13-ultraquant-scale/`,
   1.571.808 Blöcke aus `cells-cwe4k.bin`): die **Byte**-Form ist eine Verbesserung (18 → 17 B,
   −5,6 %), die **Numerik** nicht. Der Snap kostet **2,43 dB** (`2^round`), im besten Fall
   **1,81 dB** (der Writer wählt den besseren von `2^floor`/`2^ceil`) — ~0,0020 nats, also eine volle
   128K-Paritätsauflösung. Unser **fp16**-`d` kostet gegen die ungerundete Rechnung **0,000 dB**:
   das zweite Byte ist numerisch gratis, das gesparte wäre es nicht. Und `c` ist bei uns durch eine
   **kostenlose** LS-Passade ersetzt und dadurch besser: `c = 0,156` liegt **2,29 dB** hinter dem
   LS-Fit, `c = 0,125` (unser Gitter, Spitze 8) **0,98 dB**. Die Konstante ist gitterabhängig und als
   Startwert richtig, als Ersatz für den Fit nicht.

**Und ihr Performance-Befund ist unsere 2,25×-Zahl, von der anderen Seite:** ihr TurboQuant-Kern
(Software-Codebook) kostet **2,45×** die FP8-Kernelzeit ✓✓✓ — **unser 2,25× ist dieselbe
Software-Dequant-Strafe** ✓✓ (die Strafzahl, die UltraQuant mit FP4+UE8M0 beseitigt ✓✓). Und ihre
Kernaussage: „**4-bit KV buys capacity, not kernel speed**" ✓✓ — das ist **unser Fenster-Argument**
✓✓ (das Fenster kauft Kapazität ✓ = die Genauigkeit der Ring-Zellen ✓).

### InnerQ (arXiv:2602.23200) — die zweite Bestätigung, und zwei Tricks zum Stehlen

**Die Innendimension-Gruppierung** (die Gruppen laufen entlang der **inneren** Dimension, G = 32
über die head_dim ✓✓) — **das ist unsere Blockgröße und unsere Blockrichtung** ✓✓✓: „the values of
the same group in each channel of K have the same scale factor — this enables compute units
processing the same channel group to load the scale factor **once** and reuse it" ✓✓. **Unsere
Geometrie ist die hardware-aware** ✓ (und der Abstand zum Nachbarblock wächst mit der Länge ✓).

**Die per-Kanal-Normalisierung des K, „computed once during the prefill stage and **folded into the
model parameters**" ✓✓✓** — **das ist unsere Kalibrierung, aber ohne Artefakt-Tabelle** ✓✓: statt
einer 12-KB-Tabelle je Modell wird die Kanalnorm in `W_Q`/`W_K` **gefaltet** ✓ (der Rechenaufwand
verschwindet ✓✓). **Das ist eine billigere Form desselben Hebels** ✓✓ — und die Frage, ob sie bei
uns trägt, ist eine eigene Messung ✓ (die Form ändert die Basisdisziplin ✓✓).

**Der Nullpunkt kostet 0,5 bit/Element bei G = 32** ✓✓ (**exakt unser Wert** ✓✓✓) — und **er kostet
Latenz**: „InnerQHybrid is slightly slower due to its higher computational demand and **the need to
load additional zero-points for some groups**" ✓✓ (bei 99 % sparsity der Maske ✓) — **passend zu
QServe: die Subtraktion ist nicht in die Epilogue fusierbar** ✓✓.

**Zwei Tricks, die wir stehlen können:**
- **Das Sign-Bit der fp16-Skala als Modus-Maske** ✓✓ („since scale factors are strictly positive, we
  repurpose their sign bit to encode ℳ" ✓✓) — **ein Bit pro Gruppe gratis** ✓✓ (bei uns: die Maske
  „symmetrisch/asymmetrisch" je Gruppe ✓✓ — **genau die Entscheidung, die der Shear je Gruppe trifft** ✓✓
  → **der Shear als *Maske* statt als *Konstante*** ✓✓✓ = **0,19 bit/Zelle statt 0,5** ✓✓ und
  **kein Extra-Skalar** ✓✓ — **das ist die ökonomische Form des Shears** ✓✓✓).
- **Die Maske dicht speichern statt sie zu streuen** ✓✓ („we store the zero-points as a dense matrix
  to avoid the latency overhead associated with sparse representations" ✓✓) — die Latenz schlägt die
  Sparsamkeit ✓✓ (auch für uns eine Lehre ✓).

**Und ein Befund für unser Fenster:** „the benefit of the high-precision sink window **diminishes as
the sequence length increases**" ✓✓ (und KIVI→KIVISink: +1,7 Punkte ✓✓) → **unsere Ring-Geometrie
(S = 256 recent + R = 64 sinks ✓) sollte die Aufteilung an die Länge anpassen** ✓✓ — bei 128K weniger
Sinks ✓✓ (das ist eine Fenster-Frage ✓✓, unabhängig vom Indexer ✓).

**Und ihre Beobachtung, die unserer widerspricht (und die unser Arm entscheidet):** „our experiments
reveal that **hybrid quantization rarely uses asymmetric quantization for groups**" ✓✓ — bei ihnen
(3 Bit, pro-Token-Gruppen ✓) ist die asymmetrische Form selten die bessere ✓; **bei uns** ist der
per-Block-Offset **+0,00193 nats wert** ✓✓ (der 4K-Arm ✓). Der Unterschied: **unsere Gruppen sind
über die head_dim gebildet (Innendimension ✓✓) und unsere Daten sind nach der Rotation schief**
(der Offset folgt der Schiefe ✓✓) — theirs sind pro Token ✓ (dort ist die Verteilung symmetrischer
nach der Normalisierung ✓✓). **Der Widerspruch ist damit erklärt und die Verallgemeinerung unserer
Aussage fällt weg** ✓✓: *der Offset hilft, wo die Gruppe schief ist* ✓✓.

## 6. Was daraus fuer unseren Plan folgt

- **Der Nullpunkt ist kein exotischer Eingriff, sondern in der Literatur Standard** (OffQ, TriAxialKV,
  InnerQ, QuaRot, AMXFP4, KIVI, KVQuant, QServe) — **mit per-Group-Granularitaet** ✓. Wir sind also
  nicht auf eine unbetretente Kante gestossen, sondern auf eine **gemessene, aber im Detail
  unbehandelte** (die dB-Wirkung eines per-Block-Offsets ueber einem Uniform-Gitter).
- **Der Byte-Preis ist bestaetigt** (llama.cpp `q4_1` = 5,0 bit bei Gruppe 32 ✓✓) und **der 6-Bit-Nullpunkt
  in Schritten ist billiger als der fp16** (0,19 gegen 0,5 bit; vLLM-ROCm nutzt 4 Bit = 0,125 ✓).
- **Der Runtime-Preis ist NICHT als isolierter Nullpunkt-Preis gemessen** (Korrektur 2026-11-13: die
  30–40 % der RDNA4-Feldmessung vergleichen ganze Kernelpfade, siehe Abschnitt 3) — **und die Korrektur
  auf der Score-Seite (`sum(Q)·zp`) ist die publizierte Form, die ohne Add pro Wert auskommt** ✓✓
  (vLLM PR #40633 ✓). Beides zusammen heisst: der Reader-Preis ist **offen**, nicht entschieden.
- **Die Hardware-Aussage gilt fuer CDNA4, nicht fuer RDNA4**: UltraQuant (arXiv:2606.20474) ist auf
  **CDNA4** gemessen; dort faltet das scaled-MFMA die **Skala** in den Akkumulator, **nicht** den
  Nullpunkt. Fuer **RDNA4 (gfx12)** ist damit **nichts** ueber scaled-MFMA belegt — die Aussage darf
  nicht als Beleg fuer unseren Pfad dienen. Offen und zu pruefen: ob gfx12 die `*_f8f6f4`-Skalenform
  ueberhaupt hat. Ausserdem: **FP4 ist nicht uniform** (E2M1 hat einen nichtlinearen Abstand) und
  deshalb **nicht automatisch affin**; native Formatunterstuetzung kann einen schnellen Pfad
  ermoeglichen, das ist aber eine andere Begruendung als Affinitaet.
- **Die Gegenstimme bleibt** (arXiv:2607.17019: der Nullpunkt bewegt die NLL kaum) — sie misst aber
  per Channel/Token, nicht per Block; **unser Arm entscheidet genau diese Frage empirisch** ✓.
