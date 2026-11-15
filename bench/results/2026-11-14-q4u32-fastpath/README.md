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
  bekommt von Windows nur so viel VRAM, wie kein fremder Prozess belegt. Am 14.11. um 13:13 hat **Unsloth
  Studio** der Karte rund 7,5 GB genommen — die Engine sah 8.766 von 16.304 MiB statt 15.419, der
  Expert-Cache bekam **0 Slots**, und der Prompt lief über die SSD statt über die VRAM: 36,6 tok/s
  statt rund 557 (A.9). **Gate:** jeder Arm protokolliert die Budget-Zeile und die Slot-Zahl; unter
  14.000 MiB Budget oder unter 1.000 Expert-Slots ist der Arm **ungültig** (nicht „langsam") und wird
  wiederholt.
  **Ein Fehler, den das Gate selbst gemacht hat (und der behoben ist):** das Engine-Log wird
  **angehängt** — jeder Arm eines Laufs landet in derselben Datei, mit einer „engine started"-Marke pro
  Lauf. Der Wächter las die *erste* Budgetzeile und damit die Zeile des **verworfenen** Laufs vom 13:13
  (8.766 MiB): der `int8`-Arm sah sich dadurch kaputt an, obwohl sein eigener Lauf 15.419 MiB und 2.611
  Slots hatte. Der Wächter liest jetzt den Text **nach der letzten** „engine started"-Marke — und der
  Treiber startet vorher einen kurzen **Probe-Start**, der das Budget liest, damit der erste echte Arm
  nicht auf einem Zustand landet, der dem Lauf davor gehört.

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
   SSD-Lauf; die Rate misst dann die SSD. Das ist am 14.11. um 13:13 passiert (A.9).
7. **Das Engine-Log wird angehängt.** Jeder Arm eines Laufs landet in derselben Datei. Wer daraus
   liest, muss den Text nach der letzten „engine started"-Marke nehmen — sonst liest man den Lauf
   davor (genau das hat der G-R8-Wächter getroffen).

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
| `q4lm32` (Level-Tabelle 16/16) | 257,2 | 268,5 | 268,4 | **268,4** | nein | 15.419 | 3.015 | 3/3 |
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

**Der Mechanik-Arm `q4lm32` entscheidet, wo der Preis sitzt** (der Lloyd-Arm mit der vollen
16-Level-Tabelle, ohne Fenster): **268,4 tok/s** — also **auf dem Niveau des tabellenfreien
Uniform-Grids** (263,7) und **2,27× unter dem Stock-Arm** (607,5). Damit ist die Level-Tabelle als
Ursache **ausgeschlossen**: ein Arm mit Tabelle und ein Arm ohne Tabelle kosten dasselbe. Übrig bleibt
der Unterschied zwischen den Instantiationen — `kUniform` (kv_mode 2, ohne Overlay-Zweig) gegen die
fensterfähigen (KV_MODE ≥ 4, mit Overlay-Zweig). Das ist genau der Zweig, den B2 umgeht: der
`q4u32`-Arm bedient seinen Prompt dann auf dem Matrix-Core-Pfad statt in der Decode-Instantiation.

### A.11 A2 — nach dem Bau (B2), und was der Build bringt

Der neue Build (SHA `497df90d…`) mit dem 4-Bit-Pfad im Matrix-Core-Prompt-Kern. Die Arme, die der
Build nicht ändert, laufen wie gemessen:

| Arm | A1 (Build `fdd3823e…`) | A2 (Build `497df90d…`) | Differenz |
|---|---:|---:|---|
| `int8` | 634,1 | 644,0 | +1,6 % (im Streubereich) |
| `q4_0` | 607,5 | 623,7 | +2,7 % (im Streubereich) |
| `q4u32` | 263,7 | **ungültig** | der Engine hängt (siehe unten) |

**Der `int8`-Pfad ist bitgleich geblieben:** die drei Prüfsummen des Paritätstests
(`84c178024d79e375`, `dc3b41638fab8f77`, `d791e4bfb2fd064d`) sind vor und nach dem Build gleich, und
der Paritätstest meldet **0 Fehler**. Das ist das Gate G-R7 für den Default-Pfad ✓.

**Der 4-Bit-Pfad besteht sein Paritätsgate** (der neue Fall `fmt 3` im Paritätstest, dieselben Blöcke,
der Modus „uniformes Gitter mit gefitteter Stufe"):

```
PASS q4u32 ctx 32768, 2048 queries: vs FP64 old 4e-06 new 9e-06 (output scale 7.61);
     new vs old 1.46e-05 (1.8e-06 of scale); 97.312 -> 19.005 ms per chunk (5.12x)
```

**5,12×** gegenüber dem Decode-Kern im selben Harness — die Wirkung des B2 auf die *Kernel-Zeit* ist
damit gemessen ✓.

**Im Engine hängt der Arm.** Das Symptom, wörtlich aus dem Engine-Log:

```
strata serve: no progress for 60 s during a request (reading the prompt (verify windows), from token 30776)
              - stopping the engine so the server starts it again
verify: an earlier window never finished on the GPU (#267); restart the engine
```

Der Server antwortet danach mit HTTP 503, der Needle-Harness findet 0 von 3 Nadeln. Der Aufruf hängt
**im Kernel**: die GPU war nach dem Abbruch noch 124 s busy („released the verify window's GPU waits:
the GPU finished in 124 ms"). **Die Reproduktion ist billig**: 4k-Nadel gegen den `q4u32`-Arm ≈ 70 s
pro Anfrage (der 32k-Lauf braucht 391 s und drei Neustarts).

**Was ausgeschlossen ist (jedes mit einer Messung):**

| Hypothese | Test | Ergebnis |
|---|---|---|
| der Build hat allgemein etwas kaputtgemacht | die int8-Prüfsummen | **unverändert** ✓ |
| der Hänger ist nicht der 4-Bit-Pfad | `q4u32` mit `STRATA_HIP_WMMA=0` (Decode-Pfad) | **läuft**, 3/3 Nadeln ✓ |
| der ungated Prompt-Pfad ist auf HIP kaputt | `int8` mit `STRATA_QSA_FUSE_GATE=0` | **läuft**, 3/3 ✓ |
| die Rotation ist es | `int8` mit `STRATA_KV_ROT=1` | **läuft**, 3/3 ✓ |
| der Kernel ist falsch | der Paritätstest mit dem q4-Pool (32.768, 512, 64 Kontext) | **besteht alle drei** ✓ |

Übrig bleibt der Unterschied zwischen dem **synthetischen** Pool des Paritätsharness (zufällige Codes,
Skalen in [0,01; 1] mit Zufallszeichen) und dem **echten** Pool des Engines (die LS-gefitten Stufen, die
echten Codes, die rotierte Query). Der Kernel ist also mit synthetischen Daten richtig und hängt mit den
echten — die Ursache ist **nicht isoliert**.

**Deshalb ist der Pfad opt-in** (`STRATA_KV_Q4_PROMPT=1`), und der Default-Pfad bleibt byte-identisch
zum letzten Release: der `q4u32`-Arm läuft ohne die Variable über den Decode-Kern (13 s je 4k-Prompt,
3/3 Nadeln ✓). Das ist die Zwischenstand-Regel des Repos („a new feature is opt-in, and the default
path stays byte-identical to the last release") ✓.

**Nebenbefund mit eigenem Wert:** der Decode-Kern kannte die Modi 5 (`q4u32z`) und 6 (`q4u32`) im
Warp-Scoring-Pfad **nicht** — sie fielen auf den fp16-Kern zurück und lasen Pools, die ein 4-Bit-Zustand
nicht hat (`k_pool`/`v_pool` sind null): der Start schlug mit „unspecified launch failure" fehl. Nur die
Lane-Cell-Instantiation (`STRATA_ATTN_LANECELL=1`, auf gfx12 per Default an) kannte sie. Das ist ein
vorbestehender Bug, unabhängig von B2, und jetzt behoben ✓ — er fällt nur auf Karten oder
Konfigurationen auf, die den Lane-Cell-Pfad nicht nehmen.

### A.12 — Der Hänger braucht **keine** Länge (15.11.2026)

Drei Läufe, alle mit `STRATA_KV_Q4_PROMPT=1` und `needle-fastpath-q4u32.json`:

1. **4k-Nadel: hängt auch.** Bei **3.321 von 3.328** Token, 77 s, alle drei Tiefen HTTP 503. Die Länge
   ist damit **nicht** die Bedingung, und die Reproduktion kostet 77 s statt 391 s.
2. **4k mit `QA_Q4TRACE=1`: der Bulk läuft, danach nichts mehr.** Die zwölf letzten
   `[q4u32] launch`-Zeilen der Session sagen alle „cells 1 chunks 1" — das ist die **erste** Query
   eines Prompt-Chunks (Position 0, eine Zelle), also der Bulk-Prompt (zwölf Aufrufe = die
   QSA-Schichten). Danach kommt **keine** Zeile. Der Trace druckt **vor** dem Launch: der Host hat den
   nächsten `launch_wmma` nie erreicht. Dazwischen liegt nur das 6-Token-Verify-Fenster
   (`strata verify: captured the 6-token window … upload no error, sync no error`, 117,8 ms) und die
   Stall-Meldung „reading the prompt (verify windows), from token 3321".
3. **Der isolierte Kern mit derselben Form läuft.** 3.328 Zellen, 6 Queries, mit Gate:
   `PASS … 0,502 → 0,488 ms per chunk (1,03×)`. Dazu die beiden kleinen Fälle unter dem Gate
   (64/64: 0,116 → 0,055, 2,10×; 512/512: 4,737 → 1,171, 4,04×), **FAILURES: 0**.

**Werkzeugfehler, der hätte täuschen können:** ohne `STRATA_KV_Q4_PROMPT=1` startet der Harness für
fmt 3 den **Decode**-Kern („the q4 arm runs the prefill in the decode kernel … opt-in") und druckt
trotzdem `PASS`. Seit dem Gate ist ein grüner q4-Fall also **kein** Beleg für den Matrix-Kern. Die drei
Zahlen oben sind **mit** Gate gefahren; die B2-Zahlen (32.768/512/64) stammen von **vor** dem Gate und
bleiben gültig — ein erneuter Lauf ohne Gate wäre aber ein falsches Grün. (Für fmt 1 setzt der Harness
`STRATA_HIP_WMMA` selbst; für fmt 3 fehlt das Gegenstück.)

**Was damit ausgeschlossen ist:** die Länge, die Grid-Größe, der Kernel auf synthetischen Daten und der
Bulk-Prompt selbst. Was bleibt, ist der **Übergang** vom Bulk-Prompt zum Verify-Fenster — die Stelle,
an der echter Pool, echte Selektions-IDs und Fenster-Pools zum ersten Mal zusammentreffen.

**Die nächsten drei Schritte** (billig nach teuer):
`STRATA_QSA_SELECT_DUMP=1` (kompiliert, nicht an) schreibt die echten Selektionen des hängenden Laufs —
damit lässt sich die Form im Harness nachspielen; ein Lauf mit `--prefill 512` zeigt, ob es an *jedem*
Chunk-Ende hängt oder nur am letzten; der Host-Stack liegt in `strata-stall-*.dmp` (auf dieser Maschine
ist kein `cdb`, also nicht lesbar).

### A.13 — Der Trace sagt jetzt die Form, und der Verdacht verschiebt sich (15.11.2026)

B6 hat gezeigt, dass der Trace `steps[0]` las (die Selektion der *ersten* Query) und damit „cells 1
chunks 1" für jeden Aufruf druckte — Bulk und Fenster gleich. Der Trace nennt jetzt `n_q`, `cap`, die
Route und die Zellen der ersten **und** letzten Query. Derselbe 4K-Lauf (Gate an, `QA_Q4TRACE=1`,
`STRATA_KV_Q4_PROMPT=1`):

```
[q4u32] launch: n_q 3321 cap 2051 ungated, cells 1..2051 chunks 1..65     (×12, eine je QSA-Schicht)
strata verify: captured the 6-token window (upload no error, sync no error) in 118.4 ms
strata serve: no progress for 60 s during a request (reading the prompt (verify windows), from token 3321)
```

Zwei Dinge stehen damit fest:

1. **Der Bulk-Prompt ist der Matrix-Pfad, vollständig** — 3.321 Queries, `cap` 2.051, 12 Aufrufe (eine je
   QSA-Schicht), keine Fehlermeldung. Die 12 Zeilen aus A.12 waren also wirklich der Bulk.
2. **Das Fenster ruft den 4-Bit-Kern überhaupt nicht.** Nach dem Capture kommt **keine** Trace-Zeile —
   und das ist jetzt eindeutig, weil jede q4-Instantiierung (auch die gated, B6) eine druckt. Der Grund
   steht im Code (statische Analyse): das Verify-Fenster geht über `qsa_decode_attn_batch` / `_rows` /
   `_slots` / `_split` (`verify.cpp` 2227–2330), nie über den Prompt-Pfad. **Die Aussage aus A.12, der
   Host stehe „vor dem nächsten `launch_wmma`", ist damit widerlegt** — es gibt keinen solchen Aufruf.
   Der Hänger liegt im **Decode-Kern des Fensters**, nachdem der Bulk über den Matrix-Pfad lief.

Damit ist die entscheidende Kontrolle die, die wir schon haben: `q4u32` mit `STRATA_HIP_WMMA=0` (Bulk
**und** Fenster über den Decode-Kern) läuft 3/3. Der Unterschied ist also nicht der Decode-Kern und
nicht das Fenster allein, sondern **der Matrix-Pfad davor**. Zwei Möglichkeiten bleiben, beide billig:

* der 4-Bit-Kern **schreibt** irgendwohin, wo das Fenster später liest (Out-of-Bounds an `cap`/`ids`);
* oder der letzte Bulk-Launch **endet nie**, und das „sync no error" des Captures gilt einer anderen
  Wartestelle.

Der Test, der das trennt, ist `HIP_LAUNCH_BLOCKING=1`: dann ist jeder Launch synchron, und die letzte
Zeile vor dem Stillstand ist der Launch, der nicht zurückkommt.

**Nebenbefund — und eine Korrektur an mir selbst:** ich hatte notiert, `cap` sei im Motor 2.051 und im
Harness die volle Kontextlänge. Das ist **falsch**: der Harness nimmt `qsa_selection_width(ctx)` =
`min(ctx, idx_top_k + idx_block − 1)` = `min(ctx, 2051)` (`include/strata/kernels/qsa.hpp` 131–140),
also genau dieselbe 2.051, und die letzten 1.277 Queries eines 3.328er Laufs sättigen sie. Was der
Harness **nicht** prüft, ist ein Aufruf, in dem die Queries **verschiedene** Weiten haben (Position 0
mit `n = 1` bis Position 3.320 mit `n = 2.051` in einem Grid) — der Motor macht genau das (A.13, elf
Bulk-Aufrufe `n_q 3321`) — und den gated-Zweig (B6).

### A.14 — Der Hänger ist ein Doorbell-Deadlock im Fenster-Launch, und der Fast Path läuft (15.11.2026)

Vier Läufe, 4K und 32K gegen `needle-fastpath-q4u32.json`, alle mit `STRATA_KV_Q4_PROMPT=1`.

**1. `HIP_LAUNCH_BLOCKING=1` änderte nichts** (76 s statt 72 s, gleicher Stall) — der Schalter greift in
diesem HIP-Build also **nicht**; der Lauf ist damit **kein** Beleg gegen einen asynchronen Rest des
Bulks. Als Befund gestrichen.

**2. `STRATA_VERIFY_DEBUG=1` nennt den Ort** (Laufzeitvariable, kein Build):

```
verify dbg: captured; staging
verify dbg: staged; launching
   ... 60 s nichts ...
verify dbg: launched                 (erst nach dem Abbruch)
verify dbg: layer 0 rang … layer 15 served      (16 Zeilen in 35 ms)
```

Der Host steht **im Launch-Aufruf** des Fensters — nicht in einer Warteschleife danach, nicht in der
Attention.

**3. `STRATA_LAUNCH_PROBE=1` nennt den Grund:**

```
strata tdr-probe: the 6-token window graph has 1876 nodes, 1876 of them kernels
strata tdr-probe: window 0 T=6 pos=3321: launching
strata tdr-probe: window 0 T=6: the launch call has not returned after 50000 ms;
                  GPU rang 1, flags served 0 A 0 B 0       (bei 50 / 250 / … / 30000 ms unverändert)
strata tdr-probe: window 0 T=6: launch call returned in 62964.00 ms;
                  the GPU had rung 4 layer(s) and the host served 4294967295 at the return
```

Der Launch blockiert **in der Submission**, während die GPU auf die **Host-Doorbell** wartet (`A 0 B 0` =
Plan und Kopien sind noch nicht veröffentlicht). Der aufrufende Thread kann sie nicht bedienen, weil er
im Launch steht — ein Deadlock. Genau dafür gibt es `STRATA_VERIFY_ASYNC_LAUNCH=1` (der Graph startet auf
einem Helfer-Thread, der Aufrufer bedient die Doorbells; `verify.cpp` 3551: „a blocking instrumented
launch cannot also serve the graph's CPU doorbells"). Dass es normalerweise gutgeht, ist ein
**Wettlauf**: der Launch kehrt zurück, *bevor* die erste Doorbell fällig wird. Der 4-Bit-Pfad verliert
ihn.

**4. Mit dem Helfer-Thread läuft es — und zwar richtig.** Dieselbe Nadel, derselbe Build
(`1a63531b8` + Review), derselbe Schalter, nur der KV-Modus verschieden, drei Tiefen je Arm:

| Tiefe | `q4u32` (Matrix-Pfad) | `int8` | Verhältnis |
|---|---|---|---|
| 10 % (~30.782 Token) | **785,1 tok/s** | 626,6 tok/s | 1,25× |
| 50 % | 617,3 tok/s | 626,7 tok/s | 0,985× |
| 90 % | 619,4 tok/s | 620,8 tok/s | 0,998× |
| Nadeln | 3 von 3 | 3 von 3 | |

Alles Prompt-Lesen (30.782 / 30.783 Token); Generierung 27,5–27,6 gegen 35,0–45,3 tok/s (der
4-Bit-Pfad ist ein **Prompt**-Pfad, er ändert die Decode-Kerne nicht — die Differenz dort ist Rauschen
der Draft-Annahme: 6/6, 5/8, 6/11 gegen 6/6, 6/6, 6/9).

**Also: `q4u32` erreicht int8.** Zwei der drei Läufe liegen innerhalb von 0,4 % von int8, einer (der
erste der Serie) ist **25 % schneller** — und das ist **nicht** erklärt; ein Wiederholungslauf gehört
dazu, bevor daraus ein Satz wird. Gegen A.1 (263,7 tok/s, Build `fdd3823e…`, ohne den Pfad) sind es
**2,35×**.

Der Gewinn, der bleibt, ist der **Speicher**: `q4u32` braucht je Kopf 2 × (128/32) × 18 = 144 Byte
gegen int8's 2 × (128 + 2) = 264 Byte — **54,5 %** (aus den Layouts gerechnet: `kv_q4_bytes_per_head` in
`kv_q4.hpp` 127–130, `kv_q8_bytes_per_cell` in `kv_q8.hpp` 25–27; nicht gemessen). Gleiche Prompt-Rate,
gut halb so viel KV.

**Was das für den Default heißt:** der Hänger ist **nicht** der 4-Bit-Kern (das Fenster ruft ihn nie,
A.13) und kein Rechenfehler — die Nadeln werden mit dem Fast Path an allen drei Tiefen gefunden, die
Attention liest den Kontext also richtig. Es ist ein **vorbestehender Wettlauf im Fenster-Launch**, den
der schnellere Pfad verliert. Zwei Wege, und der erste ist der richtige:

1. den Launch reparieren (den Helfer-Thread zum Default machen, wo Doorbells bedient werden müssen, oder
   die Doorbell vor dem Submit veröffentlichen);
2. `q4u32` als Default erst danach — ein Umweg über eine Diagnose-Variable ist kein Default.

Die 619,4 tok/s gelten **mit** `STRATA_VERIFY_ASYNC_LAUNCH=1`; ohne sie hängt der Lauf bei 3.321 von
3.328 Token. Das gehört in jede Tabelle, die diese Zahl zitiert. Auch die 4K-Nadel lief so (3 von 3,
8 s je Anfrage).

### A.15 — Der Fix: der Launch gehört aus dem bedienenden Thread (15.11.2026)

Die Ursache aus A.14 stand schon im Header, nur nicht als Voreinstellung. `host_graph_launch.hpp` 11–14:
*„Instrumented cudaGraphLaunch may block until device work completes. A graph which waits for CPU
doorbells must therefore launch outside its service thread."* Genau das war der Default nicht: der
Helfer-Thread lief nur mit `STRATA_VERIFY_ASYNC_LAUNCH=1`.

**Die Änderung** (`verify.cpp`, eine Voreinstellung, kein neuer Pfad): der Helfer-Thread ist auf dem
**Windows-HIP-Build** der Default — dort ist der blockierende Submit gemessen —, auf allen anderen Builds
bleibt der Inline-Launch der Default (unverändert, CUDA also byte-gleich). `=0` nimmt den Inline-Launch
als A/B-Arm, `=1` den Helfer-Thread auch anderswo. Die Bedingung ist nicht „q4": **jedes** Fenster-Graph
wartet auf die Host-Doorbells, also galt der Wettlauf für jeden KV-Modus.

**Die Messung.** 32K-Nadel, `needle-fastpath-*.json`, dieser Build, **keine** Variable gesetzt (nur
`STRATA_KV_Q4_PROMPT=1` für den noch opt-in q4-Pfad), drei Tiefen je Arm, Prompt-Rate in tok/s:

| Arm | 10 % | 50 % | 90 % | Mittel | Nadeln |
|---|---|---|---|---|---|
| `q4u32` (Matrix-Pfad) | 640,4 | 638,2 | 649,9 | **642,8** | 3/3 |
| `int8` | 644,4 | 651,8 | 650,9 | **649,0** | 3/3 |

**`q4u32` liegt bei 99,1 % von int8.** Der Abstand ist kleiner als die Streuung: dieselbe int8-Konfiguration
las in A.14 (mit dem Schalter, gleicher Build-Vorgänger) 626,6 / 626,7 / 620,8 — **4 % langsamer** als
heute, ohne dass sich an int8 etwas geändert hätte. Zwei Sitzungen, die identisch sein sollten, streuen
also ±4 %; die 1 % zwischen den Armen sind darin nicht auflösbar. (Der 785,1-Ausreißer aus A.14 bleibt
damit auch unerklärt — er ist als Streuung zu lesen, nicht als Vorsprung.)

**4K** mit demselben Fix: 3 von 3 Nadeln, 8 s je Anfrage, ohne Schalter.

**Was der Fix *nicht* ist:** er macht `q4u32` nicht zum Default — der Pfad bleibt hinter
`STRATA_KV_Q4_PROMPT=1`. Er nimmt nur den Hänger weg, und zwar für alle KV-Modi. Die Zahlen oben sind
der Beleg, dass der Default-Pfad darunter nicht leidet (int8 3/3, 649,0 tok/s).

**Offen bleibt:** die Streuung von ±4 % zwischen Sitzungen (zwei Wiederholungen desselben Arms in einer
Sitzung wären billig, um sie zu beziffern), und die Wiederholung der Arme `q4_0` und `q4u32win`, die in
dieser Serie noch fehlen.

### A.16 — Das Gate für B3 steht, und es ist rot (15.11.2026)

**Vorregistriert:** `hip_prompt_attn_wmma.exe` bekommt einen Fensterfall (S 8/R 128, Ring), Referenz ist
der **Decode-Kern** (er liest das Fenster schon) **und** FP64 mit derselben `qsa_win_slot`-Regel. Der Fall
muss **rot** sein, solange der Prompt-Pfad den Seitenpool nicht liest, und darf **nicht** grün werden,
indem er heimlich über den Decode-Kern läuft. Erst danach wird der Pass gebaut.

**Der Harness (`qsa_prompt_attn_parity.cpp`)** hat drei Fenstermodi, alle auf `fmt 3` (`q4u32`):

| Modus | Ring | Was er prüft |
|---|---|---|
| 1 | S + n_q (der ganze Batch passt hinein) | jeder Recent-Cell ist ein Tag-Treffer; Sinks immer |
| 2 | nur S (Tags veraltet) | Tag-Fehltreffer lesen **quantisiert** — beide Quellen in einem Aufruf, wie im Motor mitten im Chunk |
| 3 | nur S, `win_all = 1` | nur der Tag entscheidet (die Horizont-Prüfung fällt weg) |

Die Sinks (Zellen 0..7) werden in jede Selektion gesetzt, damit der sink-Pfad (kein Tag nötig) auch im
sparse Fall läuft. Die Zellzahlen druckt der Harness selbst, damit ein Fensterfall nicht grün werden kann,
ohne eine einzige exakte Zelle zu lesen.

**Der Lauf** (`hip_prompt_attn_wmma.exe 32768 1024 3`, gfx1200, HIP-Build):

```
window1: 139264 of 2100224 cells exact (6.6 %; 8192 sinks, 6.2 % recent), 1960960 read quantized
FAIL q4u32+win ctx 32768, 1024 queries: the prompt path refused the pools (no matrix-core route);
     the decode kernel, the reference, reads this window at 4.72e-06 vs FP64 (output scale 7.42)
window2: 10304 of 503735 cells exact (2.0 %; 2048 sinks, 1.6 % recent), 493431 read quantized
FAIL q4u32+win ctx 2100, 256 queries: ... 3.6e-06 vs FP64 (output scale 7.62)
window3: 10304 of 351360 cells exact (2.9 %; 2048 sinks, 2.3 % recent), 341056 read quantized
FAIL q4u32+win ctx 1500, 256 queries: ... 7.15e-06 vs FP64 (output scale 7.87)
FAILURES: 3
```

**Was daran zählt:**

* **Rot, aus dem richtigen Grund:** `qsa_prompt_attn_batch` lehnt ab (`return false`, 2694–2696), und der
  Harness druckt dafür `FAIL` statt durch den Decode-Kern zu laufen und `PASS` zu sagen (die Falle aus
  B6). Der Abbruch zählt als Fehler (`FAILURES: 3`, Exit 3) — die frühere Fassung hätte hier `exit(2)`
  gemacht und die anderen Fälle verschluckt.
* **Die Referenz ist belegt, nicht angenommen:** der Decode-Kern trifft die FP64-Fensterreferenz auf
  **3,6e-06 … 7,2e-06** bei einer Ausgabenskala von ~7,5 — **~1e-6 relativ**, dieselbe Größenordnung wie
  ohne Fenster. Damit stimmen Zeilenlayout (`[(row, kv_head, head_dim)]`), Ringzeilen, Tags und die
  `qsa_win_slot`-Regel im Harness. Ohne diese Zahl wäre jede spätere Abweichung zwischen Kern und
  Referenz zweideutig.
* **Die Abdeckung ist beziffert:** **6,6 %** exakte Zellen im vollen Fenster (128 recent / 2.051 + 8
  Sinks), **2,0 %** bzw. **2,9 %** in den Tag-Fehltreffer-Fällen. Das ist die Zellrate, die der
  zweite WMMA-Satz zusätzlich rechnen muss — die *Zeit* ist damit noch nicht gemessen.

**Zwei Fehler in meinem eigenen Harness, beide vor dem ersten Befund:** `pl.win_tag` zeigt auf
**Device**-Speicher; die Host-Referenz hat ihn dereferenziert und ist mit einer Zugriffsverletzung
abgestürzt (exit `0xC0000005`). Und `pl.win_b` war auf `n_q` gesetzt, während der Ring auf `ctx`
geklemmt wurde — `qsa_win_slot` rechnet den Ring aber aus `win_s + win_b`, also las es hinter dem
Tag-Array. Beides behoben (`ph.win_tag = wtag.data()`, `win_s + win_b == wring` durch Konstruktion).

### A.17 — Der Fenster-Pass steht, das Gate ist grün (15.11.2026)

**Der Pass, wie er gebaut ist** (`src/kernels/cuda/qsa_prompt_attn.cu`, nur die 4-Bit-Instantiierung
`Q4 = true`):

| Baustein | Wo | Was |
|---|---|---|
| `win_row(p, cell, n, nkv, kvh)` | 1966–1976 | `qsa_win_slot` (die Regel des Decode-Kerns) × `nkv + kvh`; `-1` ohne Fenster |
| `h8_win(base)` | 1959–1965 | `PA_KL` Halves aus dem Seitenpool als B-Fragment (16 B auf gfx12, 32 B auf gfx11) |
| `win_of(ci)` | 2099–2108 | die Fensterzeile der Zelle **dieser Lane** im Chunk `ci`, unabhängig vom Page-Table |
| `Chunk::wrow` / `wr[2]` | 2103–2110 | die Zeile der Zelle bzw. der Zelle der beiden 16er-Kacheln (per `__shfl`) |
| V-Staging `row >= 0 && wrow < 0` | 2132, 2151 | eine Fensterzelle legt **nichts** quantisiert ab (V-Codes 0, `vs` 0) |
| `ksc = 1.0f` bei `wr[nt] >= 0` | 2161–2165 | die exakten Halves sind K selbst, es gibt keinen Blockschritt |
| K-Operand | 2238–2246 | `h8_win(...)` statt `i8_to_h(...)`, sobald die Zelle im Fenster liegt |
| `S.valid = row >= 0 \|\| wrow >= 0` | 2226 | die Maske des Decode-Kerns (`srow < 0 && swin < 0`) |
| zweiter V-Satz `tmpw[4]` | 2308–2356 | `A = p` (ohne `vdown`), `B = V_exakt`, warp-uniform übersprungen ohne Fensterzelle |
| Freigabe | 2659–2662, 2694, 2751 | `q4u_wmma_ok` verlangt keinen leeren Pool mehr; die **int8**-Routen fordern ihn weiter |

**Das Gate** (`hip_prompt_attn_wmma.exe 32768 1024 3`, gfx1200), vorher rot (A.16), jetzt:

```
window1: 139264 of 2100224 cells exact (6.6 %; 8192 sinks, 6.2 % recent), 1960960 read quantized
PASS q4u32+win ctx 32768, 1024: vs FP64 old 4.72e-06 new 5.66e-06 (scale 7.42); new vs old 1.29e-05 (1.6e-06 of scale)
window2: 10304 of 503735 cells exact (2.0 %; 2048 sinks, 1.6 % recent), 493431 read quantized
PASS q4u32+win ctx 2100, 256:    vs FP64 old 3.6e-06 new 8.25e-06 (scale 7.62); new vs old 1.37e-05 (1.7e-06 of scale)
window3: 10304 of 351360 cells exact (2.9 %; 2048 sinks, 2.3 % recent), 341056 read quantized
PASS q4u32+win ctx 1500, 256:    vs FP64 old 7.15e-06 new 7.46e-06 (scale 7.87); new vs old 1.72e-05 (2.2e-06 of scale)
FAILURES: 0
```

* **Die Fensterzellen werden wirklich gelesen.** Der Harness legt in den Seitenpool Werte, die mit den
  Codes nichts zu tun haben. Ein Kern, der das Fenster ignoriert, müsste gegen FP64 um Größenordnungen
  danebenliegen; er liegt bei **5,7e-06 … 8,3e-06** auf einer Skala von ~7,5 — so gut wie der
  Decode-Kern selbst (4,7e-06 … 7,2e-06). Das ist der Beweis, den ein Abbruch-Gate nicht liefern konnte.
* **Kein Rückfall:** die Nicht-Fenster-Fälle sind unverändert grün und **bitgleich** zu vorher
  (`470428b775efed56` int8, `0f694f06948998d5` q4u32, `82ef1dc3fa5dd4e7`/`6c01e0a92922cca5` kurz).

**Ein gemessener Fehler auf dem Weg, und er war nicht klein.** Der zweite V-Satz stand zuerst
**unbewacht** in der int8-Instantiierung: ein `tmpw[4]` = 32 VGPR, das nur `if (anywin)` schreibt, mit
`anywin` als Laufzeitwert. Damit fiel der int8-Prompt-Kern von **8,07 auf 17,4 ms je Chunk** (32K,
1.024 Queries) — Faktor 2,16, und die Ausgabe blieb bitgleich. Mit `if constexpr (Q4)` um die ganze
Fenster-Mechanik (und `Q4 ? tmpw : 0` in der Akkumulation) ist er wieder bei **8,26 ms**. Das ist die
Lehre: ein 32-Register-Akkumulator, der nie benutzt wird, ist nicht gratis — die Occupancy entscheidet
hier über den Faktor 2, nicht die Rechenarbeit.

**Was das Gate *nicht* sagt:**

* **Die Kosten des zweiten Satzes waren in diesem Lauf nicht gemessen.** Im selben Lauf: `q4u32`
  11,33 ms/Chunk, `q4u32+win` 12,34 ms/Chunk (32K, 1.024 Queries) — **aber das sind zwei verschiedene
  Selektionen** (der Fensterfall hat Sinks und einen anderen Horizont), also kein kontrollierter A/B.
  Der kontrollierte A/B steht in A.18: **+6,2 … 13,4 %** der Attention-Kernelzeit.
* **gfx11 ist ungetestet** (`PA_KL = 16`, `PA_KOFF(half) = 0`, `PA_ROW = 2i + half`): `h8_win` spiegelt
  die Adressierung des int8-Pfads, aber auf einer gfx12-Karte ist das statische Analyse.
* **Die Engine ist gelaufen** — A.18: Route-Zeile in allen drei q4-Läufen, 16/16 Nadeln, der Fensterarm
  auf dem Matrix-Pfad. Die Rate dort ist allerdings nicht auflösbar (Zweimoduseffekt).
* **CUDA bleibt zurück** (`qsa_prompt_attn.cu:2793`, `if (pools.k_win != nullptr …) return false;`): die
  Tensor-Core-Route (mode 4) liest kein Fenster, ein Overlay-Arm nimmt dort weiter den Decode-Kern. Auf
  dieser Maschine nicht baubar — als offener Punkt notiert, nicht als erledigt.

### A.18 — vorregistriert: das Fenster in der Engine (15.11.2026, vor dem ersten Lauf)

**Frage:** was kostet das Fenster auf dem Matrix-Pfad in der Engine, und bleibt die Nadelgüte?

**Arme** (je ein Engine-Start, 30 s dazwischen, dieselbe Sitzung, dieser Build):

| Arm | Config | Variablen |
|---|---|---|
| `int8` | `needle-fastpath-int8.json` | `STRATA_HIP_WMMA=1` |
| `q4u32` | `needle-fastpath-q4u32.json` | + `STRATA_KV_Q4_PROMPT=1` |
| `q4u32win` | `needle-fastpath-q4u32win.json` | + `STRATA_KV_Q4_PROMPT=1` (S 256, R 64) |

4K **und** 32K in einem Lauf, Nadeln bei 10/50/90 %.

**Kriterien, vorher festgelegt:**

* **Der Lauf zählt nur mit der Route im Log**: `the matrix-core prompt path serves the 4-bit uniform
  grid` muss für jeden q4-Arm stehen. Steht dort die Rückfall-Zeile („runs the prefill in the decode
  kernel"), ist der Lauf kein Beleg für B3 — er wird wiederholt, nicht umgedeutet.
* **Erwartung 1:** `q4u32win` ≥ 90 % der Prompt-Rate von `q4u32` (Fenster kostet ≤ 10 %).
* **Erwartung 2:** `q4u32win` ≥ `q4u32` in der Nadelgüte — die exakten Zellen sind genauer als die
  quantisierten. Der Unterschied kann unter der Auflösung der Nadel liegen; dann ist er **nicht** belegt.
* **Erwartung 3:** 3/3 Nadeln bei 4K und 32K, kein Hänger (der Launch-Fix aus A.15 ist Default).
* **Streuung:** ±4 % zwischen Sitzungen. Der Fensterarm läuft **zweimal** (zwei Starts); nur ein
  Unterschied > 5 % gilt als Befund. Bei < 90 % wird zuerst gemessen, *welcher* Anteil (Chunks mit
  Fensterzellen), nicht umgebaut.

### A.18 — Das Ergebnis: die Route stimmt, die Prompt-Rate ist auf dieser Maschine nicht auflösbar (15.11.2026)

**Die Route ist belegt.** Alle drei q4-Logs tragen die Zeile

```
strata: prompt attention: the matrix-core prompt path serves the 4-bit uniform grid (q4u32), scale per 32 values
```

und in keinem steht die Rückfall-Zeile. Der Fensterarm schreibt zusätzlich
`KV window ring sized for the prompt chunk (6144 cells)`. **B3 ist damit im Motor angekommen** — vorher lief
dieser Arm über den Decode-Kern.

**Die Nadeln: 16 von 16.** 4K 3/3 und 32K 3/3 in jedem der vier Läufe, kein Hänger. Erwartung 3 ist erfüllt.

**Die Prompt-Raten** (wie in A.15: Prompt-Tokens ÷ Wanduhr der Anfrage; der Decode von 9–10 Token steckt
mit drin, ~1 s), in tok/s:

| Arm | 4K 10/50/90 % | 4K Ø | 32K 10/50/90 % | 32K Ø | Nadeln |
|---|---|---|---|---|---|
| `int8` | 382,4 / 386,9 / 405,9 | **391,7** | 657,8 / 656,3 / 656,3 | **656,8** | 6/6 |
| `q4u32` | 496,6 / 528,1 / 512,0 | **512,2** | 874,5 / 882,0 / 879,5 | **878,7** | 6/6 |
| `q4u32win` (1. Start) | 382,4 / 405,7 / 416,0 | **401,4** | 656,4 / 659,1 / 657,7 | **657,7** | 6/6 |
| `q4u32win` (2. Start) | 378,1 / 405,7 / 416,0 | **399,9** | 663,4 / **872,0 / 869,5** | **801,6** | 6/6 |

**Der Befund ist nicht das Fenster, sondern ein Zweimoduseffekt — und er ist nicht dem Arm zuzuordnen.**
Die Fortschrittszeilen des Motors (`reading the prompt: 6.144 of 30.781 tokens, N s so far`) zeigen
**zwei diskrete Niveaus** für den 6.144-Token-Chunk: **~9 s (langsam)** und **~6,5 s (schnell)**, Faktor
**1,38**. Prompt-Lesezeit je 32K-Anfrage, direkt aus dem Log:

| Arm | 10 % | 50 % | 90 % | Niveau |
|---|---|---|---|---|
| `int8` | 46 s | 46 s | 46 s | langsam |
| `q4u32` | 34 s | 34 s | 34 s | schnell |
| `q4u32win` 1 | 46 s | 46 s | 46 s | langsam |
| `q4u32win` 2 | **46 s** | **35 s** | **35 s** | langsam → **schnell** |

* **Das Niveau ist keine Eigenschaft des Arms.** Der zweite Fensterlauf hat **dieselbe Config** wie der
  erste (gleiche JSON, gleiche Variablen) und liefert beide Niveaus — die erste 32K-Anfrage in der
  langsamen Stufe, die beiden folgenden in der schnellen, **innerhalb eines Prozesses**, ohne Neustart.
  Damit ist „Arm X ist langsam" als Erklärung ausgeschlossen.
* **Erwartung 1 (Fenster ≥ 90 % von `q4u32`) ist nicht auswertbar.** 657,7 gegen 878,7 (75 %) und 801,6
  gegen 878,7 (91 %) sind zwei verschiedene Niveaus, nicht zwei Arme. Das Kriterium wird **nicht**
  umgedeutet; es ist mit dieser Maschine in dieser Form nicht messbar.
* **Was innerhalb eines Niveaus gilt:** im langsamen Niveau liegen `int8` (656,8) und `q4u32win`
  (657,7) **0,1 %** auseinander — nicht unterscheidbar. Im schnellen Niveau liegen `q4u32` (878,7) und
  die beiden schnellen Fensteranfragen (872,0 / 869,5) **0,7–1 %** auseinander — ebenfalls nicht
  unterscheidbar. In beiden Niveaus ist also kein Fenster-Aufpreis sichtbar, aber das ist eine
  **Nicht-Ablehnung**, kein Beleg.
* **Erwartung 2 (Nadelgüte) ist nicht belegt:** 16 von 16 Nadeln in allen Armen. Die Nadel hat für diesen
  Unterschied keine Auflösung; die exakten Zellen sind damit **nicht** als besser belegt.
* **Die Ursache des Niveaus ist nicht belegt.** Kandidaten, alle unbewiesen: WDDM-Seitenwechsel (jeder
  Lauf endet unter seiner eigenen Reserve: „only 1793–1854 MiB free once the slots are written (reserve
  2560 MiB)" steht in **allen vier** Logs), Expert-Paging über PCIe, Takt/Power. Ich habe sie **nicht**
  gemessen und nenne deshalb keinen.

**Was A.15 damit nachträglich bedeutet — und warum es keine neue Zahl ist.** A.15 maß `q4u32` 642,8 gegen
`int8` 649,0 tok/s, also **47,4–48,2 s** bzw. **47,2–47,8 s** je 32K-Anfrage — beide im **langsamen
Niveau** (heute 46,8 s). Die Aussage „99,1 %" war also ein Vergleich *innerhalb* eines Niveaus und bleibt
gültig. Sie sagt aber **nicht**, dass `q4u32` im Prompt gleich schnell ist wie int8 — im schnellen Niveau
steht `q4u32` bei 878,7, und für `int8` gibt es in diesem Niveau **keine** Messung. Der Unterschied (1,34×)
ist damit **offen**, nicht gefunden.

**Die kontrollierte Zahl kommt aus dem Harness, nicht aus dem Motor.** `qsa_prompt_attn_parity.cpp` hat
einen A/B bekommen: derselbe Aufruf, dieselben Selektionen, dieselben quantisierten Pools, nur der
Seitenpool abgehängt (`k_win = v_win = win_tag = nullptr`, `win_* = 0`, also `qsa_win_on` falsch). Zwei
vollständige Läufe (`hip_prompt_attn_wmma.exe 32768 512 3`, gfx1200):

| Fall | exakte Zellen | mit Fenster | ohne | Aufpreis (Lauf 1 / Lauf 2) |
|---|---|---|---|---|
| 1 (S 4096, Ring = S + n_q) | 6,6 % | 6,06 / 6,15 ms | 5,48 / 5,43 ms | **+11,8 % / +13,4 %** |
| 2 (Ring = S, Tags veraltet) | 2,0 % | 2,98 / 3,19 ms | 2,79 / 2,91 ms | **+6,6 % / +9,7 %** |
| 3 (`win_all = 1`) | 2,9 % | 2,40 / 2,38 ms | 2,13 / 2,12 ms | **+13,0 % / +12,3 %** |

Das Fenster kostet also **6,2 % bis 13,4 % der Attention-Kernelzeit** (drei Läufe: der dritte Kontrolllauf
derselben Bauart gab +12,7 % / +6,2 % / +13,4 %, `FAILURES: 0`, Exit 0), und zwar **chunk-granular, nicht
zellproportional**: bei 2,0 % exakten Zellen kostet es 6,2–9,7 %, weil in jeder Selektion Sinks liegen und
damit *jeder* Chunk Fensterzellen hat (der zweite V-Satz fällt je Chunk an, nicht je Zelle).

**Warum der Motor das nie zeigen kann (Hochrechnung, keine Messung am Motor).** Der Harness mißt für
`q4u32` bei 32K **11,2 ms** je Aufruf mit 1.024 Queries. Der Motor liest 6.144 Token je Chunk, also grob
das Sechsfache ≈ **67 ms** — rund **1 %** der 6,5 s, die ein Chunk im schnellen Niveau kostet. 13 % davon
sind **0,1 %** der Prompt-Zeit. Ein Motor-A/B auf das Fenster wäre damit selbst ohne den Zweimoduseffekt
nicht messbar; die belastbare Zahl ist die aus dem Harness.

**Decode, nur zur Vollständigkeit (mit Vorbehalt):** `int8` 38,0 / 36,6 / 33,3 tok/s, `q4u32`
31,2 / 30,6 / 28,2, `q4u32win` 31,4 / 30,7 / 28,2 bzw. 32,3 / 31,2 / 28,3 (32K, 10/50/90 %). Der
Unterschied ist stabil über die Niveaus hinweg — aber **kein sauberer Decode-Vergleich**: die
Spekulationsannahme unterscheidet sich (nur der `q4u32`-Lauf protokolliert `suffix drafts: 1 windows, 2 of
5 drafts accepted`), und die Tokenzahl je Anfrage ist 9–10. Als Hinweis notiert, nicht als Befund.

**Was daraus für A.19 folgt:** ein Motor-A/B „Fenster gegen kein Fenster" ist auf dieser Maschine nicht
auflösbar (0,1 % Effekt gegen 38 % Niveau-Streuung) und wird **nicht** gefahren. Die offene Frage ist
stattdessen die des Zweigs: **ist `q4u32` im Prompt schneller als `int8`** — im schnellen Niveau 878,7
gegen eine fehlende `int8`-Messung? Das braucht einen alternierenden Lauf **mit Niveau-Klassifikation je
Anfrage** (Chunk-Zeit aus dem Log), nicht mehr Starts desselben Arms.


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

> **Stand:** dieser Abschnitt ist die **Planung** (15.11., vor dem Bau). Gebaut und gemessen ist er in
> `A.16` (Gate rot) und `A.17` (Pass, Gate grün). Die Zeilennummern hier sind der Stand der Planung.

* **Die Entscheidung (2026-11-14):** das Fenster **muss** auf dem Fast-Path laufen. Der Produktarm
  (`q4u32` + Fenster, S 256/R 64) ist die Konfiguration, die im Paper steht; ein Produktpfad, der den
  Fast-Path verliert, wäre die halbe Sache. B3 ist damit **im Umfang**, nicht mehr offen.
* **Der Zustand vor B3:** der Prompt-Kernel kannte keinen Overlay (kein `k_win`/`v_win`); nur der
  Decode-Kernel liest ihn (`WinRows<KV_MODE>`, `WIN = KV_MODE >= 4`, `qsa_decode_attn.cu:213`).
* **Die Schwierigkeit, richtig gestellt:** der shared memory des Prompt-Kernels ist typisiert — `KElem`/
  `VElem` sind entweder `int8_t` oder `__half` (`qsa_prompt_attn.cu:96-109`). Der WMMA-Kern von v2 rechnet
  aber **fp16 × fp16** (`wmma_f16`), die int8-Codes sind nur Zulieferung. Damit ist die Frage nicht „int8
  oder fp16", sondern **„woher kommt der Operand je Zelle"** — und das ist für K und V verschieden.
* **Der Kern rechnet fp16, nicht int8 — und das entscheidet den Weg (statische Analyse, 15.11.2026,
  korrigiert):** `wmma_f16` ist
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c)` mit `wh8 = _Float16[8]`
  (`qsa_prompt_attn.cu` 1917–1918, 1941–1950): **fp16 × fp16 → fp32**. Die int8-Codes sind nur
  Zulieferung — `i8_to_h()` (1952) gießt sie exakt nach fp16, und die Blockskala wird **danach in FP32**
  auf die Gruppensumme angewandt (`s * cd.ksc[nt][g]`, 2197). Eine fp16-Zeile aus dem Seitenpool ist für
  diesen Kern also der **natürliche** Operand, kein Fremdkörper.
  *(Zwischenstand, den ich hier zuerst falsch notiert hatte: „der Kern rechnet int8 × int8, eine
  fp16-Zeile ist nicht als Code mit einer Skala je 32 Werte darstellbar, also ist gemischter smem
  ausgeschlossen". Der erste Satz ist falsch, der Schluss damit auch. Ein Korrektur-Pass mit
  `q·(k_fp16 − k_quant)` ist **nicht nötig**.)*
* **Der Weg, der daraus folgt: dieselbe WMMA, exakt gefüttert.** K und V sind verschieden:
  * **K: kostenlos.** Der K-Operand wird je Lane in Registern gebaut (`cd.kx[nt][kk]`, 2136–2159) und
    erst im WMMA-Schritt konvertiert (2190). Für eine Fensterzelle lädt man statt der 8 Codes **8
    Halves** aus dem Seitenpool (16 B, ausgerichtet wie `load8_win`) und setzt `ksc = 1.0`. Es gibt
    keinen Konflikt: die Skala wirkt je Zelle in FP32, nicht auf die Kachel.
  * **V: ein zweiter WMMA-Satz, kein Korrektur-Pass.** V wird als **Bytes** gestagt
    (`S.v[4][32][72]`, gelesen als Codes in 2269–2270); 64 Dims als Halves wären 128 B, das Polster
    gibt es nicht (und die V-Zeile auf 128 B zu verbreitern kostet 7 KiB LDS je Block — bei ~24 KiB
    heute der Sprung von 5 auf 3 residente Blöcke). Also: Fensterzellen in der quantisierten Passung
    auf **0** setzen (beim Stagen, nicht im heißen Loop) und einen **zweiten** Satz von 4 Kacheln × 2
    k-Schritten über genau diese Zellen rechnen, mit `B = V_exakt` aus dem Seitenpool und `A = p`
    (ohne `vdown`, in einen eigenen `tmpw`-Akkumulator, der ohne `vdown` in `acc` geht). Der Zweig ist
    warp-uniform und läuft nur in Chunks, die überhaupt Fensterzellen enthalten — bei S 128/R 8 sind
    das ~7 % der Zellen, also die letzten Chunks; die übrigen ~93 % zahlen **nichts**.
  * **Maske:** `S.valid[c] = row >= 0 || wrow >= 0` — genau die Regel des Decode-Kerns
    (`srow[c] < 0 && swin[c] < 0`, `qsa_decode_attn.cu` 293).
  * **Kosten:** die Schätzung, die hier stand (~7 % zusätzliche V-WMMA-Arbeit), ist in `A.18` gemessen;
    das K-Fenster kostet nichts.
* **Was das für den Arm bedeutet (Stand vor dem Bau):** `q4u32` **mit** Fenster war im Prompt-Pfad
  `false` (`q4u_wmma_ok` verlangte `k_win == nullptr`) und lief über den Decode-Kern; das ist der Arm,
  den A.1 mit 260,3 tok/s gemessen hat. B3 heißt: dieser Arm soll auf den Matrix-Pfad — mit den zwei
  Quellen oben, ohne Quantisierung der Fensterzellen.
* **Gate:** ein Parity-Fall mit Overlay-Zellen gegen den Decode-Pfad — der Decode-Pfad ist die
  Referenz, weil er das Fenster heute schon liest. Der Fall muss in `hip_prompt_attn_wmma.exe` neu
  gebaut werden (das Fenster ist dort bisher **nirgends** getestet, B6/Befund 6). **Der Fall muss
  „rot" auch wirklich rot sein:** mit einem Fenster lehnt `qsa_prompt_attn_batch` heute ab
  (`return false`, 2694–2696), und der Harness wertet das als Abbruch — nicht als grünes `PASS` über
  den Decode-Kern (die Falle aus B6).

### B4 — Die Reihenfolge und die Kosten

| Schritt | Inhalt | Gate | Aufwand |
|---|---|---|---|
| B1 | CUDA-Dispatch, eine Zeile | Parität (bitgleich) | klein |
| B2 | HIP-Matrix-Core liest 4 Bit | Parität + Timing | **groß** |
| B3 | Fenster auf dem Prompt-Pfad (entschieden: es kommt drauf): K exakt als fp16-Operand, V als zweiter WMMA-Satz | Parität mit Overlay | **mittel** (kein Korrektur-Pass, s. o.) |
| B4' | **Logzeile** „der Prompt-Pfad bedient diesen Arm" | G-R1 | klein |

**B4' ist Voraussetzung für A2**, nicht für A1: ohne die Zeile ist die Zuordnung „Zeit ↔ Pfad"
nicht belegbar.

### B5 — Was uns von „voll einsatzbereit und optimiert" trennt (Stand 15.11.2026)

Die Frage aus B lautet nicht „welche Optimierung fehlt", sondern „welche Lücke blockiert welche
andere". Es sind vier, und die Reihenfolge ist nicht frei wählbar:

| # | Lücke | Zustand | Blockiert | Wodurch entschieden |
|---|---|---|---|---|
| 1 | **Der Hänger** mit `STRATA_KV_Q4_PROMPT=1` | **erledigt** (`A.14`/`A.15`): Doorbell-Deadlock im Fenster-Launch, Helfer-Thread ist Default | nichts mehr | A.14 (Sonde), A.15 (Fix + 4K/32K 3/3) |
| 2 | **Das Fenster** auf dem Fast-Path (B3) | **gebaut, Gate grün** (A.17); im Motor belegt (A.18: Route-Zeile, 16/16 Nadeln), Kosten **+6,2…13,4 %** der Attention-Kernelzeit (Harness-A/B) | nichts mehr | A.16 (rot), A.17 (grün), A.18 (Route + Kosten) |
| 3 | **Die Rate** des 4-Bit-Kerns | **teilweise gemessen**: A.15 642,8 gegen int8 649,0 tok/s (99,1 %) — **beide im langsamen Niveau**; im schnellen Niveau steht `q4u32` bei 878,7 ohne `int8`-Gegenstück (A.18) | die Aussage „so schnell wie int8" | A2, A.18 |
| 4 | **Der Rest von Teil C** | offen | Feinschliff | erst nach 2 |

**1. Der Hänger ist erledigt — er war nicht der Kern.** Die Prüfung am Code hatte ihn nicht gefunden,
und das war richtig so: die Chunk-Schleife ist blockuniform, `n` hängt nur an `qi`, die Adressierung des
4-Bit-Pools ist Zeile für Zeile die des Decode-Kerns (`base + row*144 + b*18`), die Gruppen- und
Skalenzuordnung stimmt für `GPW = 2`. `A.14` hat ihn dann mit der Sonde lokalisiert: der **Launch** des
6-Token-Verify-Fensters blockiert, während die GPU auf eine **Host-Doorbell** wartet. Der schnellere
Prompt-Pfad verliert diesen Wettlauf, ein langsamerer gewinnt ihn — deshalb sah es nach „q4 hängt" aus,
obwohl es jeden KV-Modus treffen konnte. `A.15` macht den Helfer-Thread zum Default (Windows-HIP-Build),
seither 4K und 32K 3/3 ohne Schalter. Die Reihenfolge, die hier stand (Selektions-Dump, `--prefill 512`,
echter Pool), ist damit **hinfällig** — sie hätte einen Deadlock im Launch gesucht, wo er nicht war.

**Nicht durch Lesen gefunden, sondern durch Messen:** die Längen-Hypothese (`A.12`), der Ort (`A.14`),
der Fix (`A.15`). Das ist die Lehre für B3: der Fensterfall im Harness muss **rot** sein können, sonst
belegt er nichts.

**Gemessen und nicht mehr offen:** dass es an der Länge liegt (`A.12`), dass ein grüner q4-Fall im
Harness ohne `STRATA_KV_Q4_PROMPT=1` den Matrix-Kern überhaupt testet (tut er nicht — er testet den
Decode-Kern und druckt `PASS`), **was der Hänger ist** (`A.14`: ein Doorbell-Deadlock im **Launch** des
Verify-Fensters, den der schnellere Pfad im Wettlauf verliert) und dass er **weg** ist (`A.15`: der
Helfer-Thread ist auf dem Windows-HIP-Build Default, 4K und 32K laufen 3/3 ohne Schalter). Die Rate ist
damit **keine** Lücke mehr: `q4u32` 642,8 gegen `int8` 649,0 tok/s (99,1 %, innerhalb der Streuung von
±4 %).

**2. Das Fenster** ist keine Optimierung, sondern Funktionsumfang (B3, entschieden). Solange es fehlt,
ist `q4u32` + Fenster — die Konfiguration, die im Paper steht — **kein** Fast-Path-Arm.

**3. Die Rate war der *letzte* Hebel, nicht der erste — und sie ist gemessen.** B2 hat den Kern 5,12×
schneller gemacht (97,3 → 19,0 ms/Chunk, gemessen); A.14 zeigt, dass der Arm damit **auf int8-Niveau**
liegt (Prompt: 619,4 gegen 626,7 tok/s, derselbe Build, dieselbe Nadel). Der naheliegende nächste
Schritt — **int8-WMMA** (`wmma_i32_16x16x16_iu8`, laut AMD-Tabelle 2× fp16) —
ist geprüft und **zurückgestellt** (`docs/research/KV_Q4_MATRIX_PATH_RDNA4.md`): die 2× kommen nicht
ganz an (Fremdmessung 1,20× bzw. 1,61×, weil das Descaling je 32-Werte-Block die Hälfte zurücknimmt),
er erzwingt **Q in int8** (ein Qualitätspreis, den wir heute nicht zahlen: beim Nachbarn +0,02 %
Perplexität) und ein für `iu8` **undokumentiertes** K-Mapping. Das ist ein eigenes Vorhaben mit eigener
Messung, kein Anhängsel an B2.

**4. Was nicht mehr zur Debatte steht** (weil A1 es gemessen hat): die Level-Tabelle ist **kein**
Preistreiber (`q4lm32` 268,4 ≈ `q4u32` 263,7 tok/s) — eine LUT-Stufe wäre die falsche Antwort; und
eine fp16-Kopie des Caches zu vermeiden ist erledigt (wir dequantisieren im Register/smem, der Grund,
aus dem llama.cpp PR #31802 überhaupt 17–79 % holt).

**Die Reihenfolge, die daraus folgt:**

1. 4K-Nadel mit `STRATA_KV_Q4_PROMPT=1`: **erledigt** (A.12) — es hängt auch bei 4k; die Länge ist
   nicht die Bedingung, der Übergang zum Verify-Fenster ist es.
2. Den Trace so erweitern, dass er die **Form des Aufrufs** nennt (`n_q`, Route) statt der Selektion
   der ersten Query — sonst ist „kein Trace nach dem Bulk" nicht von „der Tail nimmt einen anderen
   Zweig" zu unterscheiden (B6, Befund 4). **Erledigt, und das Ergebnis verschiebt den Verdacht**
   (A.13): der Tail nimmt gar keinen q4-Kern, das Fenster läuft im Decode-Kern.
3. Den **Launch** des Fensters reparieren (der Helfer-Thread als Default, wo Doorbells bedient werden
   müssen) — **erledigt** (A.15): Default auf dem Windows-HIP-Build, gemessen 3/3 bei 4K und 32K, ohne
   Schalter. A2 ist damit gelaufen: `int8` 649,0 / `q4u32` 642,8 tok/s (99,1 %). `q4_0` und
   `q4u32win` fehlen noch in dieser Serie.
4. B3 (Fenster) auf dem dann laufenden Kern — erst damit ist der Produktarm ein Fast-Path-Arm.
5. Teil C — und **nur** wenn A2 zeigt, dass die Rate noch die Lücke ist, int8-WMMA als eigenes
   Vorhaben. Nach A.14 ist sie es **nicht**; Teil C ist damit Feinschliff, keine Voraussetzung.

### B6 — Review der Änderungen (statische Analyse, keine Messung)

Der beauftragte Review-Lauf ist am Nutzungslimit gestorben; diese Runde ist die Prüfung von Hand. Sie
liest die drei geänderten Dateien (`qsa_prompt_attn.cu`, `qsa_decode_attn.cu`,
`qsa_prompt_attn_parity.cpp`) gegen den Stand `ac33b0c4b`. Alles hier ist **Code-Lesen**, nicht
gemessen — das steht bei jedem Befund dabei. Die Reihenfolge ist die, in der es weh tut.

| # | Befund | Wirkung | Art |
|---|---|---|---|
| 1 | `g_wmma_failed` ist zwischen int8 und q4 **geteilt** | ein q4-Launchfehler schaltet den int8-Matrixpfad für den Rest des Prozesses ab | Fehler |
| 2 | `qsa_prompt_attn_batch_gated` hat **keinen** q4-Zweig | Gate an ⇒ q4 verliert die Gate-Fusion, und die Logzeile sagt es nicht | Lücke |
| 3 | `pa_said` ist **once-per-process** | eine Zeile je Lauf kann die Route je Schicht nicht belegen — genau die Frage beim Hänger | Werkzeug |
| 4 | `QA_Q4TRACE` liest `steps` **ohne Query-Offset** | die Zeile nennt die Selektion der *ersten* Query, nicht die Form des Aufrufs; sie erscheint außerdem nur im q4-Zweig | Werkzeug |
| 5 | zwei leere `if constexpr (Q4) {}`-Blöcke (`qsa_prompt_attn.cu` 2018–2021) | toter Code, der wie ein fehlender Schritt aussieht | Junk |
| 6 | der Harness prüft weder den gated-Zweig (`Q4, GATE=true`) noch den Fensterfall (`k_win != nullptr`) | zwei Konfigurationen, in denen der Kern *nicht* läuft, sind ungetestet | Abdeckung |

**1. Ein Fehlerflag für zwei Routen** (`qsa_prompt_attn.cu`: `g_wmma_failed` 2360, gesetzt in
`launch_wmma` 2490, gelesen in `hip_wmma_usable` 2396 und 2559). `launch_wmma` ist für int8 **und**
q4 dieselbe Funktion; jeder Fehler setzt das globale Flag, und danach lehnt `hip_wmma_usable` **beide**
ab. Für ein opt-in-Experiment heißt das: **ein Fehler im Experiment verändert den Default** (int8
fällt still auf den FP32-Kern zurück, eine Zeile stderr, danach nichts). Der Fehlerfall des
Experiments darf den geprüften Pfad nicht mitnehmen — getrennte Flags, oder das Flag nur im
int8-Zweig setzen.

**2. Der gated-Zweig kennt q4 nicht** (`qsa_prompt_attn_batch_gated`, 2596–2609): seine Bedingung
fordert `pools.k_q4 == nullptr && pools.v_q4 == nullptr` und gibt für einen 4-Bit-Zustand deshalb
immer `false`. Der Aufrufer (`prefill.cpp` 4824) versucht zuerst den gated-Zweig und fällt dann auf
`qsa_prompt_attn_batch` — mit Gate **an** läuft q4 also über den ungated Kern, während die Zeile
„the matrix-core prompt path serves the 4-bit uniform grid" meldet. Die Fusion fehlt, nicht die
Funktion. Fix: die Bedingung um `q4u_wmma_ok` erweitern und `launch_wmma<8, true>` **mit** `qf/o16`
aufrufen (die Instantiierung `Q4, GATE=true` existiert bereits, sie wird nur nie gewählt).

**3. `pa_said` ist einmal je Prozess** (2589–2594): `static bool said` — die erste Route gewinnt, alle
weiteren Aufrufe schweigen. Genau beim Hänger ist die Frage aber „welche Route nahm **diese** Schicht",
und ein Motor, der Bulk-Prompt und Verify-Fenster in einem Prozess fährt, kann mit dieser Zeile nicht
beantwortet werden. Vorschlag: einmal je (Route × `n_q`-Klasse) oder ein Zähler je Route am Ende.

**4. Der Trace nennt nicht die Form des Aufrufs** (2455–2466): `cudaMemcpy(h.data(), steps, …)` liest
`steps[0]`, nicht `steps + q0 * kStepCount` — die Zeile zeigt also die Selektion der ersten Query
eines Aufrufs, für Bulk-Prompt **und** Verify-Fenster gleichermaßen (`cells 1 chunks 1`). „Nach dem
Bulk kein Trace mehr" belegt damit nur dann „der Host stand vor `launch_wmma<8,true>`", wenn der Tail
diesen Zweig überhaupt nehmen würde — und der Trace erscheint **nicht**, wenn der Aufruf im
Decode-Kern oder im gated-Zweig landet (oder wenn `q4u_wmma_ok` wegen `k_win != nullptr` `false`
sagt). Die Deutung von A.12 hängt an dieser Annahme; sie ist nicht belegt. **Der nächste Schritt ist
deshalb kein weiterer Motor-Lauf, sondern eine Zeile im Aufrufer**: `n_q`, `cap` und welche Funktion
geantwortet hat.

**5. Junk** (2018–2021): zwei leere `if constexpr (Q4) { }` hintereinander, direkt vor `const int t =
threadIdx.x`. Sie tun nichts und riechen nach einem entfernten Schritt. Weg damit.

**6. Was der Harness nicht prüft** (`qsa_prompt_attn_parity.cpp`): kein `gated`, kein `k_win`, kein
`STRATA_KV_Q4_PROMPT` (die Falle aus A.12). Er prüft genau die Instantiierung `Q4, GATE=false` ohne
Fenster — also die, die in der Engine am wenigsten Kritik hat. Der gated-Zweig ist heute **nirgends**
getestet, und der Fensterfall ist der, den B3 bauen soll: beides braucht einen Fall im Harness, bevor
es Default werden kann.

**Was das Review *nicht* gefunden hat** (und was damit als geprüft gilt): int8 bleibt byte-identisch
(`NGRP = 4` ist der Default, damit sind `SmemWT`-Layout, Ausdrücke und Reihenfolge die alten; die
Prüfsummen sind unverändert — das ist gemessen, nicht gelesen); die Chunk-Schleife ist blockuniform
(`n` hängt nur an `qi`, kein Barrieren-Deadlock); die 4-Bit-Adressierung ist Zeile für Zeile die des
Decode-Kerns (`base + row*144 + b*18`), und `q4u_block_codes`/`q4u_lane_codes` lesen nie über die 16
Byte eines Blocks hinaus; die Gruppen- und Skalenzuordnung stimmt für `GPW = 2`. **Kein** Befund in
der Arithmetik — der Hänger ist damit weiter offen, aber die Liste der Verdächtigen ist kürzer und die
Werkzeuge, die ihn finden sollen, sind jetzt selbst geprüft.

**Was von diesem Review schon gebaut ist** (Befunde 1, 2, 5 und die Werkzeuge 3 und 4): `g_wmma_failed`
und `g_q4_wmma_failed` sind getrennt, `qsa_prompt_attn_batch_gated` hat seinen q4-Zweig (mit demselben
Schalter, über `q4_prompt_on()`), die zwei leeren Blöcke sind weg, `pa_said` spricht einmal je *Route*,
und der Trace nennt `n_q`/`cap`/Route/Zellen der ersten und letzten Query. Der Harness setzt für fmt 3
jetzt `STRATA_KV_Q4_PROMPT=1` selbst. Gebaut (`strata`, `hip_prompt_attn_wmma`), Harness danach:
`FAILURES: 0`, **Prüfsummen unverändert** (`c1dc89ec2629c0b8`, `82ef1dc3fa5dd4e7`, `6c01e0a92922cca5`)
— die int8- und die q4-Ausgabe sind bitgleich zu vorher. B6, Befund 4, hat im selben Zug den
Verdacht verschoben (A.13).

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
