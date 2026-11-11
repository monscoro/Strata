# Der gescherte Codec: der End-to-End-Arm (vorregistriert)

**Stand: 2026-11-11, 19:40.** Vorregistrierung, **vor** dem Bau und **vor** dem Lauf. Die Frage, die
der Blockmaß nicht beantworten kann: **kommt die Blockverbesserung im Modell an?**

## Warum eine eigene Stufe (und nicht S6)

S6 ist der **Fast-Path** (der Prompt-Schnellpfad lehnt die nicht-affine Level-Tabelle ab → 2,25× Prompt-
Lesedauer). Der Nullpunkt ist kein Fast-Path-Thema, sondern ein **Codec**: er ändert, *welche* Zahl ein
Code darstellt, und er braucht ein eigenes Byte-Format. Er kommt deshalb als **A9** (der Hebel) und
**S7** (die Stufe) in die Leiter, nicht in S6.

**Die gute Nachricht für die Messung: die Messung braucht keinen Fast-Path.** Die Qualitätsarme der
Kampagne laufen über den **Verify-Pfad** (`verify.cpp`), nicht über Mode 4. Der Codec wird also im
**Writer** und im **Leser/Verify-Pfad** gebaut; die batched Prompt Attention (Mode 4) bleibt unberührt.
Ein späterer, eigener Schritt kann Mode 4 die Konstante beibringen (K-Seite: die Query-Gruppensummen,
je Kopf einmal; V-Seite: eine gewichtete Summe je Gruppe).

## Die Frage, in drei Teilfragen

1. **Ist der gescherte Codec end-to-end besser als die Lloyd-Tabelle?** (Paar gegen `q4lm32w`.)
2. **Hilft der Nullpunkt auch neben der Kalibrierung?** (Paar gegen `q4lm32w` + Centering.) Der
   Blockmaß sagt: ja, +1,18 dB zusätzlich. End-to-end ist das offen — und es ist die Frage, die
   entscheidet, ob der Nullpunkt ein **Ersatz** für die Kalibrierung ist oder eine **Ergänzung**.
3. **Trägt er die Bytes?** (Die Byte-Rechnung, kein Messwert: 4,66 oder 5,0 bit/Zelle im Bulk.)

## Der Codec

**`q4u32`** — 16 **gleichmäßige** Stufen (4 Bit, ein 4-Bit-Codebook), die Skala **per LS** je
32-Wert-Block (wie die FIT-Variante der Lloyd-Tabelle, aber mit gleichmäßigen Stufen), **optional** ein
**Nullpunkt je Block**.

**Der Schalter `STRATA_KV_ZERO=1`** schaltet den Nullpunkt **ein**. Das ist der Kern des Designs:
**die beiden Arme unterscheiden sich nur im Nullpunkt** — die Wirkung des Nullpunkts ist damit
end-to-end **isoliert** und nicht mit dem Skalen-Fit oder dem Fenster vermischt. Ohne den Schalter
wären „uniform mit Nullpunkt" und „uniform ohne Nullpunkt" zwei verschiedene Codecs, und ihre
Differenz enthielte auch die Skalenregel.

**Die Rekonstruktion:** `v = s·(code − 8) + c` mit `c = o + 8·s` (dem Nullpunkt in Wert-Einheiten).
Gespeichert wird `c` (fp16) neben der Skala → **+2 Bytes je 32 Werte** (der Nullpunkt muss nicht
selbst gespeichert werden; der Leser addiert die Konstante).

**Die Reihenfolge im Writer** (zwei Durchläufe, wie die FIT-Variante):
1. Codes zuweisen (`code = clip(trunc(x/s + 8,5), 0, 15)`), `s` per LS fitten (`s = Σc·x / Σc²`).
2. **Nullpunkt:** `o = mean(x − s·c)` (der Blockmittelwert des Restes), Codes neu zuweisen
   (`code = clip(rint((x − o)/s), 0, 15)`), `s` noch einmal per LS fitten, `c = o + 8·s`.
3. Prüfen: `c` muss in fp16 genau darstellbar sein (der Leser rechnet in fp32; die Parität
   entscheidet).

**Die Bytes:** 4 Bit Code + 16 Bit Skala + 16 Bit Konstante je 32 Werte = **5,0 bit/Zelle** (mit einem
8-Bit-Nullpunkt wären es 4,66). **Ein 8-Bit-Nullpunkt ist die Variante, die die 5,0-Bit-Ziel tragen
könnte** — er ist die *zweite* Byte-Variante des Arms, falls das fp16-Format das Ziel verfehlt.

## Die Arme (cwe-4k, 32.746 Positionen, dieselben Ids wie C3/C4, Build 0.1.59.1)

| # | Arm | `--kv` | Schalter | Fenster | Zweck |
|---|---|---|---|---|---|
| 0 | `int8` | `int8` | — | — | der K1-Anker (in **einem** Lauf mitgemessen: die Merge-Gegenprobe hat gezeigt, dass er exakt reproduzierbar ist) |
| 1 | `q4lm32w` | `q4lm32w` | — | ja | der **Produktarm** (Lloyd 16/16) |
| 2 | `q4lm32w`+C | `q4lm32w` | `STRATA_KV_CENTER` | ja | der kalibrierte Produktarm |
| 3 | `q4u32` | `q4u32` | — | ja | das affine Gitter **ohne** Nullpunkt (der kernelfreie Weg) |
| 4 | `q4u32`+Z | `q4u32` | `STRATA_KV_ZERO=1` | ja | **der Shear** |
| 5 | `q4u32`+Z+C | `q4u32` | `STRATA_KV_ZERO=1`, `STRATA_KV_CENTER` | ja | die **Additivität** (Nullpunkt neben der Kalibrierung) |

**Alle sechs Arme in einem Lauf** (ein `quant_quality.py compare` gegen die fp16-Referenz), damit die
gepaarten Differenzen vergleichbar sind — genau der Fehler, der in Abweichung 10 passiert ist.

## Die Gatter (vor dem Lauf festgelegt)

- **G1 (Wirksamkeit, das Wichtigste):** die **Block-MSE der Engine-Implementierung** muss die
  Harness-Vorhersage treffen. Die Vorhersage steht in `tools/kv_affine_price.py` (auf **denselben**
  Zellen): `uniform+LS` 1,738780e-03 und `uniform+LS+z` 1,291868e-03 (unkalibriert, voller Dump).
  **Tolleranz: 5 %.** Verfehlt die Engine diesen Wert, ist die Implementierung falsch und der Arm
  zählt nicht — die **Paritätsprüfung** (`kv_q4_parity`) muss zusätzlich die Rekonstruktion
  zellenweise gegen den Writer prüfen.
- **G2 (der Shear schlägt die Lloyd-Tabelle):** die gepaarte per-Position-Differenz
  (Arm 4 − Arm 1, gegen fp16) muss **negativ** sein und das 95-%-Intervall muss 0 **nicht** enthalten.
  *Kill:* ist das Intervall nicht negativ, ist der Shear im Modell nichts → A9 ist eine
  Blockeigenschaft ohne Modellwirkung und wird **nicht** weiterverfolgt.
- **G3 (Additivität):** die gepaarte Differenz (Arm 4 − Arm 2) muss **negativ** sein (der Nullpunkt
  muss auch neben der Kalibrierung noch etwas holen). *Kill:* ist sie ≥ 0, ist der Nullpunkt ein
  **Ersatz** für die Kalibrierung — dann wäre die Kalibrierung (die im Modell liegt und im Cache
  nichts kostet) die bessere Wahl, und der Nullpunkt fällt.
- **G4 (kein Schaden):** der **p99** und die **Top-1-Übereinstimmung** (exakter McNemar gegen fp16)
  von Arm 4 dürfen nicht schlechter sein als von Arm 1.
- **G5 (die Bytes):** die Byte-Rechnung muss im Repo stehen (bit/celle des Bulks mit und ohne
  Nullpunkt, plus der Ring) — gegen das 5,0-Bit-Ziel.

## Die Statistik

- **gepaart per Position** gegen die **fp16-Referenz** (die Differenz der KLs, nicht die KL gegen
  `int8` — der Mix, der in Abweichung 10 passiert ist), mit dem 95-%-Intervall und der
  Paritätsauflösung des Korpus (0,00200 bei cwe-4k);
- **Top-1** mit dem exakten McNemar (`tools/kv_tail_top1.py`), **Tail** (die 1 % der Positionen, in
  denen `int8` am schlechtesten ist) und der p99;
- **Block-MSE** aus dem Engine-Dump gegen die Harness-Vorhersage (G1).

## Was **nicht** Teil dieses Arms ist

- **Mode 4** (der Fast-Path): die Batch-Attention bleibt unberührt. Ein eigener, späterer Schritt.
- **128K**: erst wenn 4K eine Entscheidung trägt (das ist die Stufenregel der Leiter, S4).
- **Der zweite 128K-Text** (zurückgestellt).
- **Die Byte-Variante mit 8-Bit-Nullpunkt** (4,66 bit/celle): sie wird nur gebaut, wenn das fp16-
  Format (5,0) das Ziel verfehlt.

## Die Kosten

- **Bau:** ein Codec im Writer (`src/kernels/cuda/kv_q4.cu`), der Leser/Verify-Pfad
  (`v = s·(code−8) + c`), ein Format-Byte im Snapshot (+64), ein CLI-Schalter `--kv q4u32` und der
  Schalter `STRATA_KV_ZERO=1`, ein Paritätstest. **`int8` byte-identisch.**
- **Messung:** sechs 4K-Arme (≈ 12 min je Arm → ≈ 1,5 h GPU) + die Auswertung auf dem Host.
- **Reihenfolge:** erst G1 (der Paritätstest und die Block-MSE), dann die Arme. Ein Arm, der G1
  nicht besteht, wird **nicht** gefahren.

## Die Reproduktion

```powershell
# der Arm (alle sechs in einem Lauf: der Treiber unten)
$env:RUN_SHEAR_ONLY=""; & ".venv\Scripts\python.exe" bench\results\2026-11-11-shear-codec\run-shear.py

# die Auswertung
python tools\quant_quality.py compare <fp16.tsv> <int8.tsv> <q4lm32w.tsv> <q4lm32w-center.tsv> `
    <q4u32.tsv> <q4u32-zero.tsv> <q4u32-zero-center.tsv> --paired --ids data\quant_quality\cwe-4k\set.ids.json
python tools\kv_tail_top1.py --ref <fp16.tsv> --int8 <int8.tsv> --a <q4lm32w.tsv> --b <q4u32-zero.tsv> `
    --ids data\quant_quality\cwe-4k\set.ids.json
```

Die Rohdaten (TSVs, stderr, `seg.json`) liegen außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-11-11-shear-codec\`.


## Die Macht des Arms (die wichtigste Entwurfsfrage)

**Vorher geschätzt, was der Arm zeigen kann.** Die Brücke von der Block-MSE zu den nats ist an
**drei bereits gemessenen Paaren** ablesbar (alle cwe-4k, alle aus dieser Kampagne):

| Paar (End-to-End) | Block-Differenz | nats-Differenz | Brücke |
|---|---:|---:|---:|
| `q4_0` gegen `q4lm8v` (plain) | +1,512 dB | +0,00652 | **0,00431 nats/dB** |
| `q4lm8v`+Fenster gegen `q4lm32w` | +1,834 dB | +0,00401 | **0,00219 nats/dB** |
| `q4lm8v`+Fenster+C gegen `q4lm32w`+C | +1,111 dB | +0,00103 | **0,00093 nats/dB** |

**Die Brücke ist keine Konstante: sie streut über den Faktor 4,7.** Für den Shear (+0,86 dB Block)
erwartet sie **+0,0008 bis +0,0037 nats**.

**Die Auflösung des Korpus** (die 95-%-Halbbreite der gepaarten Differenz): 4K per Position 0,0094,
4K als Segmentstatistik (8 Segmente) 0,0026, 128K per Position 0,00254. **Die erwartete Wirkung liegt
an oder unter der Auflösung** — mit zwei Folgen:

1. **Die Paarung ist die ganze Macht.** Die genannten Auflösungen stammen aus Paaren, die sich
   *überall* unterscheiden (`int8` gegen einen 4-Bit-Arm: die gepaarte Streuung 0,56 sd je Position).
   **Der Shear gegen die Lloyd-Tabelle ist ein viel engeres Paar** (gleiches Bytebudget, gleiche
   Klasse, nur die Level-Form und der Nullpunkt) → die gepaarte Streuung ist kleiner und die
   Auflösung besser. **Deshalb wird die Arm-Reihenfolge streng gepaart** (derselbe Text, dieselben
   Ids, nur der Codec unterscheidet sich) und **der Arm berichtet die gepaarte Streuung und die
   Paritätsauflösung mit** (`paired.json`), damit der Leser die Macht des Ergebnisses sieht.
2. **Der 4K-Lauf ist der Bildschirm, der 128K-Lauf ist der Beweis.** Der 4K-Lauf entscheidet die
   *Wirkrichtung* (die vier Ebenen: Block-MSE, Tail, Top-1, Parität) und trägt die Aussage „der Shear
   schadet nicht"; der 128K-Lauf (131.060 Positionen, Paritätsauflösung 0,00254) ist der, der die
   **Größe** der Wirkung zeigen kann — er kostet aber je Arm 78–89 Minuten. **Die Reihenfolge ist
   festgelegt: erst 4K (alle sechs Arme), dann 128K (nur `int8`, `q4lm32w`, `q4u32`+Z und — falls
   G3 offen bleibt — `q4lm32w`+C und `q4u32`+Z+C).**

**Ein Null-Ergebnis ist nur aussagekräftig, wenn das Intervall eng genug ist.** Deshalb gilt G2 streng:
der Shear muss **besser** sein, und das Intervall muss 0 **nicht** enthalten — ein „im Rauschen"
ist kein Bestehen.

## Der Bau, Datei für Datei

**Der Codec `q4u32`**: 16 gleichmäßige Stufen, die Skala **per LS** je 32-Wert-Block, optional ein
**Nullpunkt je Block**. Die Rekonstruktion ist `v = s·(code − 8) + c` mit `c = o + 8·s`.

**Der Kern des Designs: der Schalter `STRATA_KV_ZERO=1`** — die beiden uniformen Arme unterscheiden
sich **nur** im Nullpunkt. Damit ist seine end-to-End-Wirkung isoliert statt mit dem Skalen-Fit
vermischt.

| # | Datei | Die Arbeit |
|---|---|---|
| 1 | `include/strata/kernels/kv_q4.hpp` | eine neue Basisform `kUniform` **mit** Nullpunkt (als **Overlay** neben dem Fenster, nicht als eigener Blocktyp: der Block bleibt 18 Bytes, damit **kein** ausgeliefertes Format sich ändert); die Deklaration des **Seitenpools** für die Konstanten (eine fp16 je Block, adressiert wie die Blöcke: `row·8 + b`) |
| 2 | `src/kernels/cuda/kv_q4.cu` | der Writer: `q4u_group` — die Codes zuweisen, `s` per LS fitten (derselbe Schmetterlings-Butterfly wie die FIT-Variante), dann der **Nullpunkt**: `o = (Σx − s·Σcode)/32`, die Codes neu zuweisen, `s` noch einmal fitten, `c = o + 8·s`; `c` in den Seitenpool schreiben. Und der **Leser** (`kv_gather_q4_kernel`): `+ c` |
| 3 | `src/kernels/cuda/qsa_decode_attn.cu` | `load8_q4`: **eine Addition je Wert** (`out[k] = level[n]·d + c`) — der Decode-Kern dequantisiert explizit, es ist **kein** MMA-Umbau nötig. **Das ist der Grund, warum die Messung ohne Fast-Path geht** |
| 4 | `src/core/layer.cpp` | die **Allokation** des Seitenpools (dieselbe Blockzahl wie der Bulk-Pool, 2 Bytes je Block) — nach dem Muster des Fenster-Seitenpools; der **Formatbyte** im Snapshot (+64: ein zentrierter Zustand darf nicht in einen unzentrierten zurückgespielt werden) |
| 5 | `src/program/generate.cpp` | die CLI: `--kv q4u32` und der Schalter `STRATA_KV_ZERO=1` (mit der Hilfezeile) |
| 6 | `src/core/verify.cpp` | **nichts** — der Verify-Pfad ruft den Writer und liest über die Attention-Kerne; die Korrektur sitzt im Decode-Kern |
| 7 | `tests/` | der **Paritätstest**: die Rekonstruktion des neuen Codecs zellenweise gegen den Writer (der Test `kv_q4_parity` bekommt den neuen Fall) — **das ist das Wirksamkeitsgatter G1 in Codeform** |

**Die Reihenfolge:** (1) der Writer **und** der Paritätstest, (2) G1 (die Block-MSE der
Engine-Implementierung gegen die Harness-Vorhersage), (3) der Decode-Kern, (4) die CLI, (5) der
4K-Lauf, (6) der 128K-Lauf.

**Was ausdrücklich nicht gebaut wird:** der Fast-Path (Mode 4 mit der Konstante — die K-Seite braucht
die Query-Gruppensummen, die V-Seite eine gewichtete Summe je Gruppe) und die 8-Bit-Variante des
Nullpunkts (4,66 bit/celle). Beides erst, wenn der Arm eine Wirkung zeigt.

**Die Risiken, und woran sie auffallen:** ein Fehler in der Rekonstruktion fällt sofort auf (G1, der
Paritätstest); ein Fehler in der *Zuordnung* (der Nullpunkt gehört zur falschen Seite) fällt über den
Vergleich K gegen V auf; ein Fehler in der Byte-Rechnung fällt über die Rechnung im Repo auf. **Der
Arm, der ohne Wirkung bleibt, wird nicht weiterverfolgt** — das ist der Kill.


## Die Regel, die in die Engine kommt (und ihr Preis)

**Vorab auf dem Host geprüft, bevor CUDA geschrieben wird** (`tools/kv_codec_mse.py` und die
Engine-Regel nebeneinander auf denselben Blöcken). Das Modell ist `x ≈ o + s·code` mit `code ∈ [0,15]`
und `s > 0` — das Gitter ist also `[o, o + 15s]`. **Die Konstante für den Leser** (die Kernel-Konvention
ist `value = s·(code − 8) + c`) ist **`c = o + 8·s`**; die Konstante wird gespeichert, nicht der Nullpunkt.

**Die Regel (zwei Bausteine, beide über Schmetterlings-Summen machbar):**
1. **Start:** `s = (max − min)/15`, `o = min` — das Gitter deckt den Block ab. (Der Start mit `±amax`
   ist **falsch** und kostet 1,3 dB: er lässt die obere Gitterhälfte leer, und die LS-Korrektur
   findet das nicht mehr zurück. Das ist der teuerste Einzelfehler in diesem Schritt.)
2. **Durchlauf** (je Durchlauf vier Reduktionen: `Σcode`, `Σx`, `Σcode²`, `Σcode·x`): Codes zuweisen
   (`code = clip(rint((x − o)/s), 0, 15)`), dann die **zentrierte LS-Lösung** für `(o, s)`:
   `s = [Σ(c−c̄)(x−x̄)] / [Σ(c−c̄)²]`, `o = x̄ − s·c̄`.

**Die Iteration kostet — und sie ist der Grund, warum der Arm Aufwand hat:**

| Durchläufe | Block-MSE (K, 4096 Blöcke) | gegen die Lloyd-Tabelle | Abstand zum Harness |
|---:|---:|---:|---:|
| 1 | 9,438023e-03 | +0,814 dB | +0,306 dB |
| 2 | 9,076392e-03 | +0,984 dB | +0,136 dB |
| 3 | 8,911394e-03 | +1,064 dB | +0,057 dB |
| 4 | 8,841831e-03 | +1,098 dB | +0,023 dB |
| **6** | **8,795968e-03** | **+1,120 dB** | **0,000 dB** |

**Die Engine-Regel und das Harness sind damit dieselbe Rechnung** (6 Durchläufe, 0,000 dB Abstand) —
das Wirksamkeitsgatter **G1** ist also nicht „innerhalb 5 %", sondern **exakt**: die Engine muss
**1,291868e-03** (unkalibriert, voller Dump, stride 8) liefern, und jede Abweichung ist ein Fehler im
Writer.

**Der Preis im Writer:** der Nullpunkt kostet je Block **6 Iterationen mit je vier Schmetterlings-
Reduktionen** (≈ 80–120 Shuffle-Schritte) gegen **15** heute (das Maximum plus die LS-Skala). Der
**Writer** wird also deutlich teurer; der **Leser** nicht (eine Addition je Wert). Das ist eine neue
Gate-Frage: **G6 (der Writer-Preis)** — die Prompt-Lesedauer des Shear-Arms gegen den Produktarm.
Der Vorschlag steht bei **≤ 1,25×**; die Zahl wird gemessen, nicht geschätzt.

**Und die Konstante, die der Leser addiert:** Median **0,32** (K) bzw. **0,14** (V) in Roheinheiten,
also **0,92** bzw. **1,07 Schritte** — sie ist klein genug, um in fp16 zu stehen, und der Leser
rechnet sie in fp32.
