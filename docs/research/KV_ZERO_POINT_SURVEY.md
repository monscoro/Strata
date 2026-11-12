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
| **„Regularize or Localize", arXiv:2607.17019** | Zero-Point-Gitter bewegen die NLL **kaum** (+0,74 gegen +0,765 symmetrisch bei 3 bit), kosten aber **2×16 bit je 32 Werte ≈ 1 bit/Wert** | **die Gegenstimme zu unserem Befund** — sie messen aber **per-Channel/per-Token**, nicht per-Block; unser Fall ist von ihnen nicht gemessen |

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
| **UltraQuant** (AMD CDNA4) | RDNA4 hat natives **scaled-MFMA** (die **Skala** faltet in den Akkumulator) — **ein Nullpunkt tut das nicht**; UltraQuant verzichtet deshalb auf den ZP. **Das ist die praezise Hardware-Aussage zu unserem Fast-Path** |
| **GyRot** | auf GPUs muss Group-Scaling + Zero-Point **nach** dem GEMM auf CUDA-Cores in FP erfolgen; der AMX-INT8-Pfad (INT8 Scale/ZP) ist deutlich billiger |

**Der Abgleich mit unserer Analyse:** unsere Korrektur ist **eine Add je Wert** im explizit
dequantisierenden Leser (der Decode-Kern liest die Bloecke zellenweise) — das entspricht QServes
„eine Extra-Op pro Gewicht" ✓; die **K-S.Seite** kann dagegen **ohne Add pro Wert** auskommen, wenn
die Korrektur auf der Score-Seite als `sum(Q)·zp` gefuehrt wird (vLLM PR #40633 ✓) — **das ist genau
die Form, die wir fuer die K-Seite analysiert hatten** (die Query-Gruppensummen, je Kopf einmal ✓✓).
Der **Writer**-Preis ist der andere Teil (bei uns 6 Iterationen mit je 3 Reduktionen gegen 15
Shuffle-Schritte heute); die Literatur nennt fuer den Offset **30–40 %** der KV-Zeit auf RDNA4 ✓
(unsere Rechnung: 2,3–6,3× der Writer-Kosten → **das ist das Warnsignal fuer Gate G6** ✓).

## 4. Die GPU-Primitive (RDNA4)

- **DPP** (`mov_dpp`, `ds_swizzle`, `ds_permute`) ist auf gfx12 verfuegbar ✓ (ROCm Optimization Guide,
  RDNAG-Spalte „RDNA4: Yes"); `__shfl` laeuft dort ueber den LDS-Pfad (`ds_bpermute_b32`) ✓
- **das belegte Muster**: vLLM ROCm `REDUCE_SUM_DPP_WAVE32` = 4 DPP-`row_shr` + 1 `__shfl_xor(16)` ✓
- **die Verschmelzung mit der Arithmetik** (`v_add_f32_dpp`) ist fuer gfx12 **nicht verifiziert** ✓
- `row_bcast`/`wave_ror`: **CDNA-only** ✓; `__builtin_amdgcn_wave_reduce_add_u32`: nur Integer ✓

## 5. Die Luecken (was niemand gemessen hat)

1. **Keine Veroeffentlichung misst den dB-Gewinn eines per-Block-Offsets ueber einem Uniform-Gitter
   mit fp16-Skala** — die Arbeiten messen PPL/Task-Scores (InnerQ, TriAxialKV). **Unsere 1,16 dB
   Block-MSE (und die Additivitaet mit der per-Kanal-Kalibrierung: 3,18 dB) sind unveroeffentlicht.**
2. **Keine publizierte Korrelation** „Offset ≈ Formfehler des Blocks (r = 0,89 mit dem Rest) gegen
   Mittelwert".
3. **Keine oeffentliche Instruktionszaehlung** „ein Extra-Add pro Wert" gegen „ein Extra-FMA je Gruppe"
   im Decode-Attention-Kern (nur Latenz-Deltas).
4. **Keine begutachtete RDNA4-Arbeit** zu 4-Bit-KV mit per-Group-ZP (die AMD-Daten stammen aus Code
   und Repos, nicht aus Papern).
5. **Keine publizierte Praezisionskosten** eines 8-Bit/5-Bit-Fixed-Point-Offsets **in Schritten der
   Gruppenskala** — nur Formate (AWQ 4-Bit-ZP, `Q4_K` 6-Bit-Min, INT8-ZP bei GyRot) ohne dB-Angabe.

## 6. Was daraus fuer unseren Plan folgt

- **Der Nullpunkt ist kein exotischer Eingriff, sondern in der Literatur Standard** (OffQ, TriAxialKV,
  InnerQ, QuaRot, AMXFP4, KIVI, KVQuant, QServe) — **mit per-Group-Granularitaet** ✓. Wir sind also
  nicht auf eine unbetretente Kante gestossen, sondern auf eine **gemessene, aber im Detail
  unbehandelte** (die dB-Wirkung eines per-Block-Offsets ueber einem Uniform-Gitter).
- **Der Byte-Preis ist bestaetigt** (llama.cpp `q4_1` = 5,0 bit bei Gruppe 32 ✓✓) und **der 6-Bit-Nullpunkt
  in Schritten ist billiger als der fp16** (0,19 gegen 0,5 bit; vLLM-ROCm nutzt 4 Bit = 0,125 ✓).
- **Der Runtime-Preis ist gemessen** (30–40 % auf RDNA4 fuer den Offset ✓) — **und die Korrektur auf der
  Score-Seite (`sum(Q)·zp`) ist die publizierte Form, die ohne Add pro Wert auskommt** ✓✓ (vLLM PR
  #40633 ✓).
- **Die Hardware-Aussage ist eindeutig**: das scaled-MFMA faltet die **Skala** in den Akkumulator,
  **nicht** den Nullpunkt ✓ — das ist die Grenze des Fast-Path-Achse und muss im Paper stehen.
- **Die Gegenstimme bleibt** (arXiv:2607.17019: der Nullpunkt bewegt die NLL kaum) — sie misst aber
  per Channel/Token, nicht per Block; **unser Arm entscheidet genau diese Frage empirisch** ✓.
