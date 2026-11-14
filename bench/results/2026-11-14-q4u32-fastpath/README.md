# q4u32 als int8-Ersatz: der Fast-Path und die Laufzeitmessung (Vorregistrierung, 2026-11-14)

Diese Datei legt fest, **was gebaut, was gemessen und was behauptet wird**, bevor der erste Bau
startet. Sie deckt zwei Dinge ab, die bisher getrennt liefen:

* **Serie #1 — die offene Laufzeitkontrolle.** Für `q4u32` gibt es bis heute **keine** Laufzeitmessung.
  Die 2,25× aus Tabelle 2 des Papiers gehören dem Level-Tabellen-Arm, nicht `q4u32`.
* **Der Fast-Path.** `q4u32` ist affine und byte-identisch zum `q4_0`-Layout, wird vom Prompt-Pfad
  aber abgelehnt (`src/kernels/cuda/qsa_prompt_attn.cu:2526`). Auf HIP nimmt heute **kein** q4-Codec den
  Matrix-Core-Pfad.

**Branch:** `feature_q4u32_as_an_int8_replacement_for_all`, Basis `ac33b0c4b` (von
`daves_kv_q4_quant_with_int8_quality`, Paper 0.24). Der Branch ist angelegt und gepusht.

**Die zwei Entscheidungen vom 2026-11-14:**

1. **B3 ist entschieden: das Fenster kommt auf den Fast-Path.** Der Produktarm (`q4u32` + Fenster) darf
   den Fast-Path nicht verlieren. B3 ist damit im Umfang; die Wahl des Weges (drei Varianten in B3)
   fällt mit B2.
2. **`STRATA_HIP_WMMA=1` wird in Serie #1 gesetzt** (G-R6). Nach der eigenen A/B-Messung vom 08.11. ist
   der Schalter auf **dieser** Karte allerdings ein Decode-Hebel und **kein** Prompt-Hebel (0.3) — er
   wird gesetzt, damit jeder Arm sagt, welchen Pfad er lief, aber er ist nicht das, worauf wir warten.
   Für die **Qualitätsketten** bleibt er aus (er ändert die Ausgabebits).

**Reihenfolge:** Teil A misst den **Zustand heute** (Baseline, Teil B ist noch nicht gebaut). Teil B
baut den Fast-Path. Teil A2 misst denselben Harness **nach** dem Bau. Der Vergleich A1 gegen A2 ist
die Messung der Engine-Änderung. Teil C ist die Optimierung und beginnt erst, wenn A2 vorliegt.

---

## Teil 0 — Ausgangslage

### 0.1 Gemessen (heute, diese Maschine, ein Test, keine Vorregistrierung nötig)

`build-hip-port\hip_prompt_attn_wmma.exe` — der HIP-Build von `src/kernels/qsa_prompt_attn_parity.cpp`,
künstliche Pools, keine Modelllast:

| Kontext / Queries | FP32-Prompt-Kern | Matrix-Core-Kern | Faktor |
|---|---:|---:|---:|
| 32.768 / 2.048 | 94,593 ms | 8,901 ms | **10,63×** |
| 1.500 / 1.500 | 32,353 ms | 3,388 ms | 9,55× |
| 2.100 / 256 | 14,663 ms | 1,683 ms | 8,71× |

je Chunk. Parität gegen FP64: 3,17e-06 (Matrix-Core) gegen 2,51e-06 (FP32); **0 Fehler**. Das ist der
Preis, den ein q4-Codec zahlt, wenn er den Matrix-Core-Pfad nicht bekommt — aber Achtung: dieser
Vergleich ist **Prompt-Kern gegen Prompt-Kern**, nicht der Decode-Fallback aus Tabelle 2. Die beiden
Zahlen sind nicht dasselbe und werden nicht vermischt.

### 0.2 Gelesen (statische Analyse, keine Messung)

* `q4u32` = `Q4Mode::kUniformLs`: der Block ist `block_q4_0` (18 Byte je 32 Werte: fp16-Skala + 16 Byte
  Nibble-Codes), die Rekonstruktion `v = (code−8)·s` — affine, **keine Tabelle, keine Konstante**
  (`include/strata/kernels/kv_q4.hpp:28-34`, `src/kernels/cuda/qsa_decode_attn.cu:105-117`).
* Der CUDA-Prompt-Pfad liest genau dieses Layout bereits: `KV_MODE 4` = q4_0, „mode 1 with a scale per
  32 values — each q4_0 block's codes enter as exact int8 (code−8) and its fp16 scale is applied in
  FP32" (`qsa_prompt_attn.cu:93-95`).
* Der Dispatch lehnt `kUniformLs` ab (`qsa_prompt_attn.cu:2526`). Die Begründung im Kommentar — die
  Level-Tabellen brauchen einen anderen Leser, der Shear trägt eine Konstante je Block — trifft den
  Uniform-Grid **nicht**. Der Header sagt das Gegenteil: *„kUniformLs is affine, so it rides the
  prompt fast path"* (`kv_q4.hpp:112-113`).
* Auf HIP gilt der ganze `KV_MODE`-Zweig nicht (`qsa_prompt_attn.cu:2505-2514`): dort wird **nur** der
  int8-Weg (Matrix-Core) angeboten, jeder q4-Modus fällt in den Decode-Kernel zurück.
* Auf Windows ist der Matrix-Core-Pfad **aus** (Opt-in): „the prompt attention on matrix cores is off by
  default on Windows (opt-in until tested on Windows); STRATA_HIP_WMMA=1 turns it on (gfx1200)" — so
  steht es in unseren eigenen 128K-Logs vom 13.11.

### 0.3 Der WMMA-Schalter: unsere eigene Messung (2026-11-08) und was daraus folgt

Der Schalter ist eine **Laufzeitvariable**, kein Build-Flag, und er ist **opt-in** (auf Windows aus, auf
gfx12-Linux an). Er ist unbedenklich einzusetzen — mit der Einschränkung, die die Dokumentation nennt:
er ändert die Ausgabebits (nicht bitweise; der Greedy-Text weicht ab etwa Token 50 ab). Für die
**Qualitätsketten** wird er deshalb **nicht** gesetzt (Regel vom 08.11.: die Kette soll mit P1
vergleichbar bleiben), für **Laufzeitläufe** schon.

Unsere eigene A/B-Messung vom 08.11. (Suite `standard-v1`, Swift IQ3_XXS, `--kv int8`, Kontext 32.768,
ROCm 7.2, Engine 0.1.47, RX 9060 XT gfx1200; Rohdaten unter
`C:\Users\krone\git\Strata-data\bench\2026-11-08-wmma-int8\`), Prompt-Raten in tok/s:

| Prompt | WMMA aus | WMMA an |
|---|---:|---:|
| 4k (r1/r2/r3) | 613,08 / 479,40 / 478,94 | 478,73 / 476,73 / 478,01 |
| 16k (r1/r2/r3) | 608,53 / 564,33 / 565,36 | 565,45 / 566,11 / 565,43 |

**Die Prompt-Raten sind gleich** (4k: 479,40 gegen 478,01; 16k: 565,36 gegen 565,45). Die erste
Wiederholung jedes Arms ist ein Ausreißer (613 bzw. 608) und trägt keinen Schaltereffekt; sie ist die
Quelle des scheinbaren „×1,12" im Plan vom 08.11. Im Decode: +6,9 % (4k), ≈0 (16k), −3,4 % (short) —
mit **überlappenden Spannen**.

**Die Lesart: der WMMA-Schalter ist auf dieser Karte ein Decode-Hebel, kein Prompt-Hebel.** Unser Preis
(die 2,25×) ist aber ein **Prompt**-Preis. Das ist genau die Aussage in
`docs/research/KV_FASTPATH_LITERATURE.md`, Abschnitt 4a: *„Der Fast-Path ist die Struktur (Abschnitt 1),
nicht die Matrix-Cores."*

**Folge für diesen Plan:** der Schalter wird in Serie #1 gesetzt (G-R6, damit jeder Arm sagt, welchen
Pfad er lief), aber er ist **nicht** das, worauf wir warten. Erwartet wird die Wirkung der **Struktur**
(B1/B2/B3), und die wird an **zwei** Stellen gemessen (G-R7).

---

## Teil A — Serie #1: die Laufzeitmessung

### A.1 Die Frage

**Wie schnell liest `q4u32` den Prompt, und läuft er dabei auf dem Fast-Path?** Heute ist beides
unbekannt: der 128K-Lauf hat nur *eine* Prompt-Rate je Arm geliefert (26,0 gegen 28,0 tok/s), und die
trennt die Arme nicht.

### A.2 Was gemessen wird

* **Metrik 1 (das, was der Benutzer sieht):** die Prompt-Read-Rate aus dem Engine-Log —
  `strata serve: prompt N tokens = 0 reused + N read in X ms (Y tok/s)`. Das ist die Prompt-Lesedauer
  ohne Decode-Anteil (der Decode ist bei 32K neun bis zehn Token und vernachlässigbar).
* **Metrik 2 (das, was die Arbeit betrifft):** die **Kernel-Zeit der Prompt-Attention** selbst. Auf
  dieser 16-GB-Karte kommt ein großer Teil der Prompt-Zeit vom Nachladen der Experten (Plan vom
  08.11.), deshalb kann die End-to-End-Rate flach bleiben, während die Attention-Kernel-Zeit fällt.
  Die Kernel-Zeit kommt aus dem Paritätstest (`hip_prompt_attn_wmma.exe`, bzw. `qsa_prompt_attn_parity`),
  der sie je Chunk selbst timt.
* **Harness:** `tools/needle_bench.py --lengths 32k --depths 10,50,90` gegen einen laufenden Server.
  Der Heuhaufen ist der Text dieses Repositories in fester Reihenfolge, die Codewörter kommen aus
  `random.Random(7)` — **jeder Arm sieht dieselben Prompts**.
* **Protokoll:** ein Arm, eine Engine, frischer Server, 30 s zwischen den Engine-Starts, Maschine
  sonst idle. Drei Tiefen je Arm; berichtet werden Median und Spanne.
* **Nicht die Metrik:** die `seconds` aus `needles.json` sind die einmalige Wandzeit einer Anfrage. Sie
  sind ein Recall-Hinweis, **kein Durchsatz** (so steht es auch im Protokoll vom 24.10.).

### A.3 Die Arme

Alle am selben Tag, ein Build, eine Engine nach der anderen:

| Arm | `--kv` | Fenster | Zweck |
|---|---|---|---|
| `int8` | `int8` | aus | der Anker: der Fast-Path in Reinform |
| `q4_0` | `q4_0` | aus (das Fenster wird laut abgelehnt) | der Stock-Gitter-Vergleich |
| `q4lm8v` | `q4lm8v` | aus | **Kontrollarm**: der Level-Tabellen-Arm, dessen 2,25× der Harness sehen muss |
| `q4u32` | `q4u32` | aus | der Uniform-Grid **ohne** Fenster (Bulk, byte-gleich zum q4_0-Layout) |
| `q4u32`+Fenster | `q4u32` | S 256 / R 64 | die Produktkonfiguration; **vor B3 ohne Fast-Path** |
| `q4lm32w` | `q4lm32w` | S 256 / R 64 | der Produktarm des Papiers (16/16 + Fenster) |

Der Fenster-Arm trägt `STRATA_KV_WINDOW=1`, `STRATA_KV_WINDOW_S=256`, `STRATA_KV_WINDOW_R=64`;
der Fenster-Arm muss die Logzeile „ring sized" zeigen, die anderen nicht (das prüft der Treiber).

### A.4 Die Gatter (vorab fixiert)

* **G-R1 (Struktur — der Pfad muss sichtbar sein).** Für jede Behauptung „der Arm lief auf dem
  Fast-Path" braucht es eine **Logzeile**, die das belegt. **Die gibt es heute nicht.** Deshalb wird
  sie Teil von B1/B2: der Prompt-Pfad meldet sich, wenn er einen q4-Uniform-Arm bedient. **Ohne diese
  Zeile ist jede Zeitangabe in diesem Plan strukturell nicht zuzuordnen** — sie wird dann als
  „Zeit des Fallbacks" berichtet und nicht als Fast-Path-Zeit.
* **G-R2 (Leistung — das eigentliche Ziel).** `q4u32` ohne Fenster ist am selben Tag **nicht langsamer**
  als `int8` (Median der drei Tiefen). Die Byte-Hypothese sagt mehr: der 4-Bit-Bulk liest die Hälfte
  der KV-Bytes, also *schneller*. Verfehlt der Arm G-R2, ist die Behauptung „q4u32 ist ein int8-Ersatz
  auch in der Laufzeit" **widerlegt** und wird so berichtet.
* **G-R3 (Kontrolle — der Harness muss sensibel sein).** `q4lm8v` muss am selben Tag **deutlich
  langsamer** sein als `int8` (sein Prompt läuft in den Decode-Kernel). Ist er das nicht, sieht der
  Harness den Unterschied nicht und **alle Zeiten in diesem Plan sind wertlos** — dann wird die Serie
  abgebrochen und der Fehler gesucht, nicht die Arme gerechtfertigt.
* **G-R4 (Streuung).** Drei Tiefen je Arm. Ein Unterschied unter 3 % wird **kein** Unterschied
  genannt. Jede Zahl bekommt Median und Spanne.
* **G-R5 (Fenster).** Der Fenster-Arm wird **getrennt** berichtet: vor B3 läuft er per Definition über
  den Fallback. Die „ring sized"-Zeile muss da sein, sonst ist der Arm kein Fenster-Arm.
* **G-R6 (Matrix-Core-Pfad).** Auf Windows ist der Matrix-Core-Pfad Opt-in. Jeder Arm setzt
  `STRATA_HIP_WMMA=1`, und die Logzeile („the prompt attention on matrix cores on (gfx1200)") wird mit
  protokolliert. **Ohne diese Zeile gilt der Lauf als FP32-Lauf** und wird nicht mit dem
  Matrix-Core-Pfad verglichen.
* **G-R7 (zwei Zahlen, oder keine).** Jeder Arm bekommt **beide** Metriken: die End-to-End-Prompt-Rate
  **und** die Kernel-Zeit der Prompt-Attention. Bleibt die End-to-End-Rate flach, während die
  Kernel-Zeit fällt, wird **genau das** berichtet — nicht „kein Unterschied". Wird nur eine der beiden
  gemessen, gilt die Arm-Aussage als unvollständig. (Grund: 0.3 — auf dieser Karte ist der WMMA-Schalter
  kein Prompt-Hebel; der Fast-Path ist die Struktur, und die zeigt sich in der Kernel-Zeit.)
* **G-R8 (die Karte muss frei sein).** Der VRAM-Budget ist die Fallgrube dieser Serie: die Engine
  bekommt von Windows nur so viel VRAM, wie kein fremder Prozess belegt. Am 14.11. hat **Unsloth
  Studio** der Karte rund 7,5 GB genommen — die Engine sah 8.766 von 16.304 MiB statt 15.419, der
  Expert-Cache bekam **0 Slots**, und der Prompt lief über die SSD statt über die VRAM: 36,6 tok/s
  statt rund 557 (A.9). **Gate:** jeder Arm protokolliert die Budget-Zeile und die Slot-Zahl; unter
  14.000 MiB Budget oder unter 1.000 Expert-Slots ist der Arm **ungültig** (nicht „langsam") und wird
  wiederholt. Der Treiber prüft das von sich aus und endet mit einem Fehlercode.

### A.5 Die Befehle

Der Treiber ist eine Abwandlung von `bench/results/2026-11-10-merge-reeval/run-b2.py` (derselbe
Harness, dieselben 30 s, dieselbe Ein-GPU-Job-Sperre) und liegt als
`bench/results/2026-11-14-q4u32-fastpath/run-runtime.py` bei:

```
python -m serve.server --engine strata --config bench/configs/needle-...-<arm>.json --port 7603 --no-api-key
python tools/needle_bench.py --url http://127.0.0.1:7603 --lengths 32k --depths 10,50,90 --out needles-<arm>.json
```

Der Treiber druckt vor jedem Arm die SHA-256 des Engines, prüft die Ein-GPU-Job-Sperre
(`tasklist`), wartet, bis der Engine weg ist, und liest danach die Prompt-Rate aus dem Log.

### A.6 Die Fallen

1. **Der Matrix-Core-Pfad ist auf Windows aus.** Ohne `STRATA_HIP_WMMA=1` messen wir den FP32-Kern.
   Das ist der wichtigste Fang: die 128K-Läufe vom 13.11. liefen **ohne** Matrix-Core.
2. **Heute läuft der Prompt jedes q4-Arms über den Decode-Kernel.** Vor B2 ist die `q4u32`-Zeit also
   die Fallback-Zeit. Sie wird gemessen und so benannt — sie ist die Zahl, die ein Benutzer heute sieht.
3. **Die Fenster-Armen sind nicht mit den Bulk-Armen vergleichbar.** Der Overlay kostet einen
   Side-Pool und einen Zweig je Zelle; er wird separat berichtet.
4. **VRAM.** Der 4-Bit-Bulk braucht die Hälfte, das Fenster einen Side-Pool daneben. Die VRAM-Budget-
   Zeile wird mitprotokolliert (sie steht im Log).
5. **Ein Build, ein Tag.** Ein Arm, der auf einem anderen Build läuft, wird verworfen, nicht
   nachgerechnet.
6. **Die Karte muss frei sein** (G-R8). Ein fremder Prozess auf der GPU macht den Prompt zum
   SSD-Lauf; die Rate misst dann die SSD. Das ist am 14.11. passiert (A.9).

### A.7 Die Rohdaten

Nicht im Repo: `C:\Users\krone\git\Strata-data\bench\2026-11-14-q4u32-fastpath\` — je Arm das
`needles-<arm>.json`, das `needle-<arm>.engine.log` (mit der SHA-256-Zeile), die Server-Ausgabe und die
SHA-256 der Rohdateien. Im Repo bleibt **dieses README** plus der Treiber.

### A.8 Die beiden Läufe

* **A1 — Baseline, vor dem Build.** Alle sechs Arme, ein Build (0.1.59.1, unverändert). Ergebnis: die
  heutigen Zeiten, mit dem Kontrollarm G-R3.
* **A2 — nach Teil B.** Dieselben sechs Arme, der neue Build, derselbe Tag, wenn möglich. Der
  Vergleich A1/A2 ist die Messung der Engine-Änderung; die Zahlen von A1 und A2 stehen **nebeneinander**,
  nie einzeln.

### A.9 Der verworfene erste Lauf (14.11.) — und die Referenz, die er brauchte

Der erste A1-Versuch am 14.11. ist **verworfen**, und der Grund steht im Engine-Log, nicht im
Rauschen:

| Zeile im Log | Referenzlauf (b2, 10./11.11.) | der erste Versuch (14.11.) |
|---|---|---|
| `Windows budgets … of this card's 16.304 MiB` | 15.419 MiB | **8.766 MiB** |
| `expert cache … slots` | 2.664 Slots (int8), 3.239 (Fenster-Arm) | **0 Slots** |
| Prompt-Read-Rate (32K, drei Tiefen) | int8 **557,2 / 578,1 / 361,7** tok/s; Fenster-Arm (q4lm8v) **264,2 / 260,0 / 265,0** tok/s | int8 **≈ 37 tok/s** (11.520 Tokens in 315 s) |

Die 7.538 MiB Differenz sind „the desktop's and other programs' share" (so der Kommentar in
`src/core/device.cu:209`) — ein fremder Prozess auf der Karte, zur Stunde der Messung **Unsloth
Studio**. Folge: 0 Expert-Slots, der Output-Kopf wanderte in den geheften Hostspeicher, und der Prompt
las die Experten über die SSD statt über die VRAM. Gemessen wurde damit die SSD, nicht der KV-Pfad.

**Zwei Dinge folgen daraus.** Erstens: die Referenzzahlen von Tabelle 2 des Papiers sind der
**b2-Lauf vom 10./11.11.** auf dieser Maschine mit diesem Harness — int8 rund 557–578 tok/s, der
Fenster-Arm des Level-Tabellen-Codecs rund 260–265 tok/s, also ein Faktor **2,11×**. Das ist der Wert,
gegen den A1 und A2 geprüft werden, und er ist **heute nicht** zu wiederholen, solange ein fremder
Prozess die Karte belegt. Zweitens: G-R8 existiert erst wegen dieses Laufs, und der Treiber prüft es
jetzt von sich aus (Budget ≥ 14.000 MiB, ≥ 1.000 Expert-Slots; sonst Arm ungültig).

### A.10 A1 — die Ergebnisse (14.11.2026, Build 0.1.59.1, SHA `fdd3823e…`)

Ein Arm, eine Engine, 30 s dazwischen. Die Metrik ist die Prompt-Read-Rate aus dem Engine-Log
(`prompt 30782 tokens = 0 reused + 30782 read in … ms (… tok/s)`), 32K-Nadel, Tiefen 10/50/90 %, in
allen Armen 3 von 3 Nadeln gefunden. VRAM-Budget und Expert-Cache-Slots stehen je Arm daneben (G-R8).

| Arm | 10 % | 50 % | 90 % | Median | Fenster | VRAM-Budget | Expert-Slots | Recall |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `int8` (Anker) | — | — | — | **ungültig** | nein | 8.766 MiB | 0 | — |
| `q4_0` (ausgeliefert) | 606,1 | 609,5 | 607,5 | **607,5** | nein | 15.419 | 3.015 | 3/3 |
| `q4u32` (uniform, ohne Fenster) | 263,1 | 264,4 | 263,7 | **263,7** | nein | 15.419 | 3.015 | 3/3 |
| `q4lm8v` (Level-Tabelle 16/8) | 256,9 | 258,2 | 257,9 | **257,9** | nein | 15.419 | 3.019 | 3/3 |
| `q4u32` + Fenster | 256,1 | 270,2 | 260,3 | **260,3** | ja | 15.419 | 2.947 | 3/3 |
| `q4lm32w` (Produktarm) | 266,0 | 256,6 | 256,6 | **256,6** | ja | 15.419 | 2.947 | 3/3 |

**Der `int8`-Anker ist ungültig** (G-R8: Budget 8.766 MiB, 0 Expert-Slots) und wird wiederholt — der
Wächter hat den Arm vor den Nadeln abgebrochen, das ist richtig so, aber der Anker fehlt (A.11).

**Der Befund, der die bisherige Erklärung kippt.** Der ausgelieferte Stock-Gitter-Arm `q4_0` liest mit
**607,5 tok/s**, jeder andere 4-Bit-Arm mit **256,6 bis 263,7 tok/s** — ein Faktor **2,32**. Der
Uniform-Grid `q4u32` **ohne** Level-Tabelle liegt bei **263,7 tok/s**, also praktisch auf dem Niveau des
Level-Tabellen-Arms (257,9) und **2,31× unter dem Stock-Arm**. Die Messung reproduziert Tabelle 2 des
Papiers (dort: `q4_0` 616 gegen `q4lm8v` 265 tok/s).

Damit steht die Erklärung aus §5 des Papiers infrage: dort wird die 2,25× **allein** der Level-Tabelle
zugeschrieben („the prompt fast path declines the Lloyd modes structurally … so the decode kernel runs
instead"). **Gemessen ist: den Preis zahlt jeder Modus, der nicht der ausgelieferte `kUniform` ist** —
auch der tabellenfreie Uniform-Grid. Zwischen `kUniform` (kv_mode 2) und `kUniformLs` (kv_mode 6) gibt
es im Code genau **einen** Unterschied: die fensterfähigen Instantiationen (KV_MODE ≥ 4) tragen den
Overlay-Zweig, der ausgelieferte nicht (statische Analyse,
`src/kernels/cuda/qsa_decode_attn.cu:213`). Das Fenster selbst kostet **nichts messbares**: `q4u32` mit
Fenster (260,3) liegt auf dem Niveau von `q4u32` ohne Fenster (263,7) — der Unterschied liegt im
Streubereich. Der Preis sitzt also an der **Instantiation**, nicht am Codec und nicht am Fenster.

**Das gibt Teil B ein messbares Ziel:** wenn der `q4u32`-Arm seinen Prompt auf dem Matrix-Core-Pfad
bedient (B2), statt in der Decode-Instantiation mit dem Overlay-Zweig, muss er auf das Niveau des
Stock-Arms (≈ 600 tok/s) kommen. Das ist die Hypothese, die A2 prüft.

---

## Teil B — Die Engine-Arbeit

Zwei Backends, zwei Wege. Beide sind „statische Analyse", bis der Paritätstest sie bestätigt.

### B1 — CUDA: der Uniform-Grid in den Prompt-Pfad

* **Änderung:** eine Zeile in `qsa_prompt_attn.cu:2526`. Der Uniform-Grid wird zugelassen, **solange
  kein Overlay hängt** — der Prompt-Pfad liest keinen Window, ein Overlay-Arm dürfte dort nicht
  laufen (sonst kämen die „exakten" Zellen aus dem Bulk: falsches Ergebnis, kein Fehler).
* **Gate:** `qsa_prompt_attn_parity` (der CUDA-Build) mit einem **neuen Fall**: ein Pool im
  Uniform-Layout muss **bitgleich** dasselbe liefern wie ein q4_0-Pool mit denselben Codes und
  Skalen — der Leser ist derselbe, also muss die Ausgabe identisch sein.
* **Risiko:** keins, solange der Overlay-Arm abgelehnt bleibt. Der `q4_0`-Leser hat keine
  Level-Tabelle; das ist der ganze Unterschied.

### B2 — HIP: der Matrix-Core-Kern liest den 4-Bit-Pool

Das ist die eigentliche Arbeit, und sie ist auf unserer Maschine die entscheidende, weil dort heute
**kein** q4-Codec den Matrix-Core-Pfad bekommt.

* **Struktur heute** (`qsa_prompt_attn.cu:1887-1966`, `:1966-2226`): Welle `w` besitzt die Dims
  `[64w, 64w+64)` — **genau eine** int8-Skalengruppe (64 Dims). Sie liest die V-Codes (4 × `uint4` je
  Zeile), die V-Skala ihrer Gruppe, die K-Fragmente und die K-Skala ihrer Gruppe. Die int8-Codes gehen
  **exakt** als FP16 in den WMMA (`i8_to_h`), die Skala wird in FP32 angewandt.
* **Der 4-Bit-Grid ändert drei Dinge:**
  1. **Die Datenquelle:** statt 1 Byte je Wert 18 Byte je 32 Werte (16 Nibble-Codes + fp16-Skala); die
     Nibbles müssen zu den exakten int8-Codes `(code−8)` expandiert werden (|code−8| ≤ 8, in FP16
     exakt — dieselbe Arithmetik wie im int8-Pfad).
  2. **Die Gruppen:** 8 Gruppen à 32 Dims statt 4 à 64. Eine Welle trägt dann **zwei** Gruppen; die
     Teilsummen pro Gruppe (`part[4][R][WCH+1]`), die V-Skalen (`vs[4][WCH]`) und das V-Staging
     (`v[4][WCH][VS]`) wachsen auf acht Gruppen.
  3. **Die Summationsreihenfolge:** die Skala wird jetzt zweimal je Welle angewandt statt einmal, die
     Reihenfolge der Gruppensummen ändert sich. Das Ergebnis bleibt FP32-nah, aber **nicht bitgleich**
     zum int8-Pfad (das ist im Paritätstest die 1e-4-Gate, nicht die Bitgleichheit).
* **Gate:** `hip_prompt_attn_wmma.exe` mit einem **neuen 4-Bit-Fall**: der Fehler gegen FP64 nicht
  größer als das Vielfache des int8-Pfads, und die Ausgabe gegen den FP32-Pfad auf 1e-4 der
  Ausgabeskala. Zusätzlich **timt** der Test den 4-Bit-Kern gegen den int8-Kern: das ist die erste
  **gemessene** Zahl zur Frage „ist der Uniform-Grid auf dem Matrix-Core-Pfad schneller als int8?".
* **Reihenfolge inside B2:** erst der Leser (Codes holen und expandieren), dann die Gruppen, dann die
  Skalen; nach jedem Schritt der Paritätstest.

### B3 — Das Fenster auf dem Fast-Path (entschieden: es kommt drauf)

* **Die Entscheidung (2026-11-14):** das Fenster **muss** auf dem Fast-Path laufen. Der Produktarm
  (`q4u32` + Fenster, S 256/R 64) ist die Konfiguration, die im Paper steht; ein Produktpfad, der den
  Fast-Path verliert, wäre die halbe Sache. B3 ist damit **im Umfang**, nicht mehr offen.
* **Der Zustand:** der Prompt-Kernel kennt keinen Overlay (kein `k_win`/`v_win`); nur der Decode-Kernel
  liest ihn (`WinRows<KV_MODE>`, `WIN = KV_MODE >= 4`, `qsa_decode_attn.cu:213`).
* **Die Schwierigkeit:** der shared memory des Prompt-Kernels ist typisiert — `KElem`/`VElem` sind
  entweder `int8_t` oder `__half`, je **Zelle** (`qsa_prompt_attn.cu:96-109`). Ein Overlay-Arm braucht
  aber *beides*: die Overlay-Zellen exakt (fp16), die übrigen als 4-Bit-Codes (int8). Das ist kein
  Template-Parameter, das ist ein neues Layout.
* **Die Wege, und welcher welcher ist:**
  1. **Gemischter shared memory**: zwei Puffer je K und V (einer int8, einer fp16), die Zellenverteilung
     im smem. Der WMMA liest die int8-Zellen; die fp16-Zellen brauchen einen eigenen Weg (FMA über
     256 Dims je Overlay-Zelle). Bei 320 Overlay-Zellen (S 256 + R 64) gegen 131.060 Zellen ist das
     ein Bruchteil der Arbeit — aber es ist Arbeit **im** WMMA-Chunk.
  2. **Overlay-Zellen aus dem WMMA-Chunk herausnehmen**: die fp16-Zellen in einem eigenen, kleinen
     FP32-Pass rechnen (Score und p.v) und die Ergebnisse in dieselben Puffer schreiben. Das fasst die
     bestehende Struktur am wenigsten an, ist aber ein zweiter Pass über dieselben Zellen.
  3. **Materialisieren**: die exakten Zellen vorab in einen eigenen Pool schreiben, den der
     Prompt-Kern als zweite Quelle liest. Ein Pool, zwei Zellstrukturen — der Kernel bekommt eine
     Zellentscheidung, aber keinen gemischten smem.
* **Die Wahl des Weges fällt mit B2**, weil B2 die Zellstruktur des Prompt-Pfads für 4 Bit überhaupt erst
  erweitert; B3 baut darauf auf.
* **Gate:** ein Parity-Fall mit Overlay-Zellen gegen den Decode-Pfad — der Decode-Pfad ist die
  Referenz, weil er das Fenster heute schon liest.

### B4 — Die Reihenfolge und die Kosten

| Schritt | Inhalt | Gate | Aufwand |
|---|---|---|---|
| B1 | CUDA-Dispatch, eine Zeile | Parität (bitgleich) | klein |
| B2 | HIP-Matrix-Core liest 4 Bit | Parität + Timing | **groß** |
| B3 | Fenster auf dem Prompt-Pfad (entschieden: es kommt drauf) | Parität mit Overlay | mittel bis groß |
| B4' | **Logzeile** „der Prompt-Pfad bedient diesen Arm" | G-R1 | klein |

**B4' ist Voraussetzung für A2**, nicht für A1: ohne die Zeile ist die Zuordnung „Zeit ↔ Pfad"
nicht belegbar.

---

## Teil C — Die Optimierung (beginnt erst nach A2)

Erst messen, dann optimieren. Die Kandidaten, alle aus dem Code gelesen (statische Analyse), mit der
jeweiligen Erwartung:

1. **Bandbreite.** Der 4-Bit-Bulk liest die Hälfte der Bytes; der Prompt wird bandbreitenbegrenzt.
   Das ist der Grund, warum G-R2 „nicht langsamer" verlangt und nicht „langsamer".
2. **Skalengruppen.** 8 Gruppen à 32 Dims bedeuten doppelt so viele fp32-Skalen-Multiplikationen und
   Skalen-Reads je Zelle wie 4 à 64. Ansätze: die Skala in bf16 anwenden, oder je Zelle einmal in ein
   Register falten.
3. **Shared Memory.** Die Expansion 4 → 8 Bit verdoppelt den Schreibtraffic in den shared memory
   (32 B je 32 Dims). Alternative: die Nibbles gepackt lassen und erst vor dem WMMA entpacken → halbe
   Fläche, mehr Entpack-Instruktionen.
4. **Layout.** Der Zugriff auf die 16 Codebytes je Block ist nach `kv_q4.hpp:73` genau das Muster, für
   das `__constant__` am schlechtesten ist — der Global-Zugriff im Prompt-Pfad ist der Ort, an dem das
   zählt.

**Regel:** keine Optimierung geht ohne eine **gemessene** Verbesserung auf dieser Maschine (A1 gegen
A2, gleicher Harness) und ohne das Paritätsgate. Der Default-Pfad bleibt byte-identisch, solange
keine Messung das Gegenteil sagt.

---

## Teil D — Was *nicht* in diesem Plan ist

Die restlichen offenen Serien aus dem Recap bleiben offen und werden hier **nicht** angefasst:

* **#2 Attribution** (Schrittregel gegen Fenster auf dem uniformen Gitter): die Arme „q4_0 + Fenster"
  und „uniform LS ohne Fenster" fehlen weiterhin.
* **#3 128K-Matrix**: `q4u32+C`, `q4u32z`, `q4u32z+C` fehlen bei 128K.
* **#4 6-Bit-Nullpunkt**, **#5 End-to-end-Gate des Shears** (das ist bereits gescheitert:
  `q4u32z − q4u32` = +0,00138, Ganztext, t 1,64), **#6 seitenweise Geometrie**, **#7 Gruppengröße 32**.

Ein Hinweis für später: **#6 wird billiger**, wenn B2 steht — eine eigene Geometrie je Seite braucht
eigene Leser, und die gibt es dann an zwei Stellen.

---

## Teil E — Die Regeln (unverändert, aber hier bindend)

* Ein GPU-Job zur Zeit. 30 s zwischen zwei Engine-Starts. **Kein Build, während ein Lauf die Binärdatei
  benutzt.**
* Vorregistrierung vor dem Lauf; die Gatter stehen oben, bevor die erste Zahl entsteht.
* Rohdaten außerhalb des Repos; im Repo nur das README, der Treiber und die Kernels.
* Nur selbst erzeugte Messungen; Literaturzahlen nur mit Quelle und als die Zahlen des zitierten
  Papiers.
* Keine Verallgemeinerung ohne Messung. Was aus dem Code gelesen ist, steht als **statische Analyse**
  im Text, nicht als Ergebnis.
* Commits mit Area-Präfix: `[Engine]` für die Kernelarbeit, `[Benchmark]` für die Läufe, `[Docs]` für
  Paper und READMEs.
* Der Default-Pfad bleibt byte-identisch, solange keine Messung ihn rechtfertigt.
