# Der Matrix-Pfad für `q4u32` auf RDNA4: was die Hardware kann, was die Praxis tut, was uns trennt

Stand 2026-11-14. Gehört zu `bench/results/2026-11-14-q4u32-fastpath/README.md` (Serie #1, Teil B).
Die eigene Messung ist als **Messung** markiert, alles andere als **Quelle** oder als **Rechnung**.
Eine Rechnung ist keine Messung: sie sagt, was zu erwarten wäre, nicht was gilt.

## 1. Die Frage

`q4u32` (= `kUniformLs`: 18-Byte-Block, fp16-Skala `d` + 16 Byte Codes, Skala je 32 Werte,
`code - 8` ist der Wert in Einheiten von `d`) läuft heute über den **Decode**-Kern, weil der
Matrix-Core-Prompt-Kern den 4-Bit-Pool nicht lesen konnte. A1 hat den Preis gemessen: **263,7 tok/s
gegen 634,1 tok/s für int8** (32K-Nadel, 3 Tiefen, Median). B2 hat den Kern gebaut, der den Pool liest:
**97,3 → 19,0 ms/Chunk, 5,12×** (Messung, Paritäts-Harness). Er hängt aber im echten Lauf.

Damit sind es drei verschiedene Fragen, die man nicht vermischen darf:

1. **Läuft er?** (Hänger — offen, blockiert alles)
2. **Was kostet er danach?** (A2 — erst nach 1 messbar)
3. **Wie schnell könnte er sein?** (Matrix-Rate — die Frage dieses Dokuments)

## 2. Was gfx12 an Instruktionen anbietet

RDNA4 hat WMMA in wave32. Die Rate kommt aus der Zyklustabelle von AMDs eigener Referenz
(**Quelle**: ROCm Optimization Guide, „RDNA4 dense WMMA builtins"):

| Builtin | Ops je Instruktion | Zyklen | MACs/Zyklus je CU | gegen fp32-VALU |
|---|---:|---:|---:|---:|
| `wmma_f32_16x16x16_f16` (unser Kern) | 8192 | 16 | 256 | 2× |
| `wmma_i32_16x16x16_iu8` | 8192 | 8 | 512 | 4× |
| `wmma_i32_16x16x16_iu4` | 8192 | 8 | 512 | 4× |
| `wmma_i32_16x16x32_iu4` | 16384 | 8 | 1024 | 8× |
| `wmma_f32_16x16x16_fp8_fp8` | 8192 | 8 | 512 | 4× |

Die „MACs/Zyklus"-Spalte ist meine Umrechnung (Ops/2 = MACs, ÷ Zyklen), nicht AMDs Spalte.
**INT8 ist auf RDNA4 doppelt so schnell wie FP16** — das ist der einzige echte 2×-Schritt, den die
Karte für Q·K hergibt. INT4 verdoppelt noch einmal, zwingt aber **beide** Operanden in 4 Bit.

Drei Nebenbedingungen, die den Schritt teurer machen, als die Tabelle aussieht:

- **Beide Operanden müssen Integer sein.** `iu8` hat `a_neg`/`b_neg` getrennt (u8×s8 geht), aber kein
  fp16×int8. Für Q·K heißt das: **Q muss quantisiert werden.**
- **Die Fragmente sind kleiner.** Unser fp16-Fragment liest 8 Halfs = 128 Bit je Lane; `iu8` liest
  8 Bytes = **64 Bit** (ROCm-Doku; GPUOpen, „WMMA guide part 2"). AMD empfiehlt dort ausdrücklich,
  zwei WMMA zu einem K=32-Schritt zu fusionieren, damit wieder 128-Bit-Ladungen entstehen. Unser K
  ist heute 16 (`PA_KL`) — für int8 wäre der Doppel-K-Schritt also Pflicht, nicht Kür.
- **Das Lane-Mapping ist ein anderes als bei NVIDIA.** gfx12 ist *lane-group-split*: die beiden
  Lanes-Gruppen teilen sich die K-Achse (Gruppe 0: K {0–3, 8–11}, Gruppe 1: K {4–7, 12–15} bei fp16),
  **ohne Replikation**. Die K-Verschränkung ist für `iu8`/`iu4` **nicht dokumentiert** (ROCm-Issue
  #6025 nennt genau das; für `iu4` ist auch die Nibble-Reihenfolge offen). Wer den int8-Pfad baut,
  muss dieses Mapping zuerst mit einer Identitäts-Sonde empirisch festnageln — sonst sind die Tiles
  still falsch, und der Fehler taucht erst viele Kernel-Schritte später als NaN auf.

**rocWMMA hilft nicht weiter:** int8 ist auf gfx1200/1201 unterstützt, aber nur als 16×16×**16**
(BlockK=16, keine 32+). Es ist eine Fragment-Bibliothek über denselben Instruktionen — sie erspart
das Mapping nicht, sie versteckt es.

## 3. Der Preis, der bleibt: Descaling

Das ist der Punkt, an dem die 2× verschwinden — und er ist bei uns strukturell schlechter als in
einem GEMM.

Unser Block ist 32 Werte breit, also **8 Skalen je 256-dim-Head**. Ein Q·K-Ergebnis (eine Query ×
eine Zelle) summiert über 256 Dimensionen, also über **8 Blöcke mit 8 verschiedenen `d`**. Es gibt
zwei Wege, die Skala anzubringen, und sie kosten um Größenordnungen verschieden viel:

| | Skala sitzt auf | Arbeit je 16-Zeilen-Tile |
|---|---|---|
| **fp16 (heute)** | dem **Operanden** K, vor dem MMA: `code * d` → fp16 | 512 Werte × ~2 Ops = 1024 Ops, **einmal**, für alle 16 Query-Zeilen |
| **int8 (Vorschlag)** | dem **Akkumulator**, nach dem MMA: `int32 * (d · q_scale)` | 16 Zeilen × 16 Zellen × 8 Blöcke = 2048 Ops, **je Ausgabe**, nicht je Eingabe |

Die int8-Rechnung tauscht also 1 Op je Eingabewert (durch 16 Zeilen geteilt) gegen 1 Op je Ausgabe
je Block. **Rechnung** (keine Messung, mit den Raten aus §2 und 128 fp32-Ops/Zyklus/CU): fp16-Pfad
16 Zyklen MMA + 8 Zyklen Dequant = 24; int8-Pfad 8 Zyklen MMA + ~4–6 Zyklen Descaling/Konvertierung
= 12–14. Das wären **1,7–2,0×** — und genau hier widerspricht die Fremdmessung.

**Quelle** (zolotukhin.ai, 12.07.2026, RX 9070 XT, Q4_K-GEMM): mit int8-WMMA halbiert sich der Matmul,
aber „ein fp32-Descale je 32 Werte kostet dieselben Zyklen" → **1,20× statt 2×**; mit ganzzahligen
Sub-Skalen, weitem K und gepacktem Entpacken **1,61×**. Und: WMMA und Vektor-Instruktionen teilen sich
auf RDNA4 **einen** Issue-Port, sie können sich also nicht gegenseitig verdecken.

Die ehrliche Lesart: **die 2× kommen nie ganz an, und ob wir bei 1,2× oder 1,7× landen, ist eine
Messung an unserem Kern, keine Ableitung.** Was die Rechnung sicher sagt: der Descaling-Aufwand hängt
daran, ob die Skala auf dem Operanden oder dem Akkumulator sitzt — und bei einem 16-Zeilen-Tile ist
das ein Faktor 16 zugunsten des heutigen fp16-Wegs.

## 4. Was die Praxis mit 4-Bit-KV auf Matrix-Cores macht

Alle drei Lager lösen **nicht** das Descaling-Problem, sondern das Materialisierungs-Problem: sie
vermeiden die fp16-Kopie des Caches. Das ist bei uns schon so (wir dequantisieren im Register/smem),
also ist der Teil bereits erledigt — die Zahlen zeigen aber, wo die Gewinne herkommen.

| Ansatz | Was er tut | Zahl (aus der Quelle) |
|---|---|---|
| llama.cpp **PR #31802** | fusioniert q8_0/q4_0-Dequant **in** die `fattn-mma-f16`-Kerne, statt vorher fp16-Scratch zu schreiben | Decode +17,6 % (32k) … +79 % (171k); ~0,5–1 GiB/GPU Scratch gespart; Prompt innerhalb +2 % |
| **torad-labs/llama.cpp** `60feea0` | liest q4_0 nativ im Tensor-Core-FA: q4-Blöcke → smem → dort fp16 („same values, same math"), **Q als int8 mit einer Skala je Spalte**, Q·K als s8×u8-MMA, V weiter fp16 | Perplexität 4,5065 → 4,5074 (**+0,02 %**); ab 4.096 Kontext schneller als der Vektor-Kern; **braucht `--kv-mean-center`** für brauchbaren 4-Bit-K-Cache |
| **SageAttention** PR #368 (gfx12) | nativer RDNA4-Backend: **int8-WMMA für QK**, fp8-WMMA für PV | Diffusion ~**42 %** schneller; ISA enthält `v_wmma_i32_16x16x16_iu8` und `v_wmma_f32_16x16x16_fp8_fp8` |
| llama.cpp `0bf1eef` | **verwirft** dp4a auf RDNA4 (q8_1-Aktivierungsquantisierung + Durchsatz) | 101 t/s statt 135 t/s auf RX 9070 XT → RDNA4 geht auf den Skalar-Half-Kern zurück |

Drei Schlüsse daraus:

1. **Der int8-QK-Pfad ist auf RDNA4 der Stand der Technik** (SageAttention baut ihn genau so, und der
   llama.cpp-Fork zeigt, dass er mit 4-Bit-KV funktioniert). Der Qualitätspreis der Q-Quantisierung
   ist dort **+0,02 % Perplexität** — klein, aber es ist ein Preis, den wir heute nicht zahlen.
2. **`--kv-mean-center` taucht als Voraussetzung auf.** Ein fremder, unabhängiger Fork kommt aus der
   Praxis zu derselben Bedingung, die unsere Kalibrierung begründet: ohne Mean-Center ist 4-Bit-K auf
   langem Kontext nicht brauchbar. Das ist eine Stütze der These, kein Beweis.
3. **Nicht jede Integer-Instruktion ist ein Gewinn.** `sudot4` (`v_dot4_i32_iu8`) existiert auf RDNA4
   (1/clk/SIMD, gemischt signed/unsigned; `sdot4`/`udot4` sind CDNA-only) — und llama.cpp hat den
   dp4a-Weg auf genau dieser Karte **zurückgebaut**. Ein Dot-Produkt-Pfad ist keine Abkürzung.

## 5. Was das für „voll einsatzbereit und optimiert" heißt

Die Reihenfolge ergibt sich aus den Zahlen, nicht aus der Eleganz:

1. **Der Hänger ist die ganze Lücke.** Solange er offen ist, ist der Fast Path nicht Default-fähig,
   und keine Rate hilft. (B3, das Fenster, ist die zweite Funktionslücke — der Produktarm darf den
   Pfad nicht verlieren.)
2. **A2 entscheidet, ob überhaupt noch etwas fehlt.** B2 hat den Kern 5,12× schneller gemacht. Ob der
   Arm damit in die Nähe von int8 kommt, ist **ungemessen** — jede Optimierung davor wäre Raten.
3. **int8-WMMA ist der zweite Schritt, nicht der erste.** Obergrenze ~2×, realistisch 1,2–1,7×
   (Fremdmessung), Preis: Q in int8 (+0,02 % PPL beim Nachbarn, bei uns ungemessen), ein neues
   K-Mapping (undokumentiert für `iu8`), Doppel-K gegen die 64-Bit-Ladungen. Das ist ein eigenes
   Vorhaben mit eigener Messung — kein Anhängsel an B2.
4. **Was wir *nicht* brauchen:** eine LUT/Codebook-Stufe (die Level-Tabelle ist in A1 als
   Preistreiber ausgeschlossen: `q4lm32` 268,4 ≈ `q4u32` 263,7 tok/s) und Scratch-Vermeidung
   (machen wir schon).

**Was diese Notiz nicht sagt:** dass int8-WMMA sich für uns lohnt. Sie sagt, was es kostet, es
herauszufinden (eine Q-Quantisierung, ein Mapping-Probe, eine A/B-Messung), und dass der heutige
5,12×-Gewinn aus der **Kernel-Struktur** kommt, nicht aus der Matrix-Rate.

## 6. Quellen

- AMD, ROCm Optimization Guide: „RDNA4 dense WMMA builtins" (Raten, Signatur, `a_neg`/`b_neg`,
  Lane-Group-Split) — rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/…/rdna4-dense-wmma-builtins.html
- AMD, GPUOpen: „WMMA guide for AMD RDNA 4 GPUs, part 2" (K-major, 64-Bit-Ladungen für int8/fp8,
  Doppel-K-Empfehlung) — gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/
- ROCm Issue #6025: „Documentation Gap: WMMA Output Lane Mapping for gfx12" (Spalten-Verteilung,
  unbestätigte `iu4`-Nibble-Reihenfolge, unbestätigte K-Verschränkung für Integer)
- rocWMMA: Data-type support (i8 nur 16×16×16 auf gfx1200/1201), CHANGELOG 2.2.0
- ggml-org/llama.cpp PR #31802: fused q8_0/q4_0-Dequant in den MMA-FA-Kernen
- torad-labs/llama.cpp `60feea0`: q4_0 nativ + int8-Q im Tensor-Core-FA, `--kv-mean-center`
- thu-ml/SageAttention PR #368: nativer gfx12-Backend, int8-QK + fp8-PV
- ggml-org/llama.cpp `0bf1eef`: dp4a auf RDNA4 verworfen (101 vs 135 t/s)
- zolotukhin.ai, 12.07.2026: „int8 WMMA doubles RDNA4's matrix rate, and Q4_K's block scales take half
  of it back" (1,20× / 1,61×, Issue-Port-Teilen)
- AMD, „Integer dot product builtins" (Verfügbarkeit von `sudot4`/`sdot4` je Architektur)
- Eigene Messungen: A1 und B2, `bench/results/2026-11-14-q4u32-fastpath/README.md`
