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
| 2 | `q4lm32w`c | `q4lm32w` | `STRATA_KV_CENTER` | ja | der kalibrierte Produktarm |
| 3 | **`q4u32`** | `q4u32` | — | ja | das affine Gitter **ohne** Nullpunkt (der kernelfreie Weg) |
| 4 | **`q4u32c`** | `q4u32` | `STRATA_KV_CENTER` | ja | dasselbe **mit** der Kalibrierung (die andere Hälfte der Additivität) |
| 5 | **`q4u32z`** | `q4u32z` | — | ja | **der Shear** (das affine Gitter **mit** dem per-Block-Nullpunkt) |
| 6 | **`q4u32zc`** | `q4u32z` | `STRATA_KV_CENTER` | ja | die **Additivität** (der Nullpunkt neben der Kalibrierung) |

**Die Namen (2026-11-11, vom Nutzer festgelegt):** `q4u32` = das uniforme Gitter mit der per Block
gefitten Stufe, `z` = der **per-Block-Nullpunkt** (der Shear), `c` = die **Kalibrierung**
(`STRATA_KV_CENTER`). **CLI-Namen sind `q4u32` und `q4u32z`** — das `c` ist der Schalter, kein eigener
Name, weil die Kalibrierung ihre Tabelle (`STRATA_KV_CENTER=<datei>`) braucht und ein Name sie nicht
tragen kann. Die Arm-Matrix ist damit **2 × 2**: {ohne, mit} Nullpunkt × {ohne, mit} Kalibrierung,
plus die zwei Anker (`int8`, `q4lm32w`).

### Der Zielpunkt ist `q4u32zc` — die anderen drei Modi sind die **Ablation**

**Der avisierte Zielpunkt des Arms ist `q4u32zc`**: der Shear **und** die per-Kanal-Kalibrierung
zusammen. Der Grund ist die gemessene Additivität: die Kalibrierung holt 2,00 dB Blockfehler (ein
**Vektor** je Layer/Kopf/Kanal), der per-Block-Nullpunkt 1,16 dB (eine **Form** je Block), **beide
zusammen 3,18 dB** — sie korrigieren verschiedene Anteile, und wer nur einen davon nimmt, verschenkt
den anderen. Die drei anderen Modi (`q4u32`, `q4u32c`, `q4u32z`) sind die **Ablation**, die zeigt,
welcher Teil wie viel trägt; sie sind kein Ziel, sondern die Zerlegung des Ziels.

**Und die Byte-Folge, die daraus folgt (sie ist der Grund, warum das Ziel ein Qualitäts-, kein
Ratenpunkt ist):** die Konstante des Nullpunkts kostet **einen Wert je 32-Wert-Block** — als fp16
**0,5 bit/Zelle** (der Bulk: 4 Bits Codes + 0,5 Skala + 0,5 Konstante = **5,0 bit/Zelle**), als 6-Bit
Konstante in Schritten (die Konstante liegt im Median 0,92 Schritten vom Gitterzentrum, die
Auflösung 1/16 Schritt ist weit unter dem Quantisierungsfehler) **0,19 bit/Zelle** (der Bulk
**4,69 bit/Zelle**). Mit dem Ring des Fensters (0,79 bit/Zelle) sind das **5,79** (fp16) bzw. **5,48**
(6 Bit) bit/Zelle gegen die **5,29** des ausgelieferten Fensters. **Der Zielpunkt des Papiers, 5,0
bit/celle, ist damit genau der Bulk des Shears ohne den Ring** — mit dem Ring ist er nicht zu
erreichen. **Das heisst: der Shear ist ein Qualitätspunkt, und der Ratenpunkt bleibt eine andere
Frage.** Diese Rechnung gehoert ins Paper, sobald der Arm eine Wirkung zeigt.

**Die sechs Arme werden in EINEM Lauf gefahren** (ein `quant_quality.py compare` gegen die fp16-Referenz),
damit die gepaarten Differenzen vergleichbar sind — genau der Fehler, der in Abweichung 10 passiert ist.

### Was am Code noch fehlt (Stand 2026-11-11, 20:30)

**Gebaut und grün** (`build-hip-port`, HIP 7.2): der Writer des Shears (`q4u_group`, der Fit mit dem
Offset), der Writer des Zwillings ohne Offset (`q4u_ls_group`), der 20-Byte-Block mit der Konstante,
beide Leser (der fused Decode und der q4-Gather addieren die Konstante), der Modus `kUniformLs` und
der Modus `kShear`, die Blockgröße als Eigenschaft des Modus, und die CLI `--kv q4u32`.

**Noch offen:** (1) der **`kv_mode` des Shears** — der Leser, der die Konstante addiert, braucht eine
eigene Instanziierung im fused Decode, und der **Window-Overlay** wird im fused Decode bisher nur von
den Lloyd-Instanziierungen gelesen (kv_mode 4) — beide neuen Modi brauchen ihn, sonst sind die Arme
**nicht mit `q4lm32w` vergleichbar** (das Fenster ist der Unterschied zwischen dem Produktarm und dem
Bulk); (2) damit zusammen der CLI-Namen `q4u32z` und die Pools-Größe für den 20-Byte-Block im
Runtime-Pfad (VRAM und Host-Pool).

**Bis dahin gilt:** `--kv q4u32` ist erreichbar und **korrekt** (der Writer fitet die Stufe, der Leser
ist q4_0's, das Layout ist identisch) — aber **ohne Fenster** (der fused Decode liest das Overlay nur
in den Lloyd-Instanziierungen), also noch **nicht arm-vergleichbar**.

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

## Der Preis des Fits: das Knie (gemessen auf dem Host, K-Seite, 4096 Bloecke)

| Regel | Reduktionen | Shuffles | Block-MSE | gegen die Lloyd-Tabelle |
|---|---:|---:|---:|---:|
| 6 LS-Durchlaeufe (die Referenz) | 19 | 95 | 8,795968e-03 | **+1,120 dB** |
| 4 LS-Durchlaeufe | 13 | 65 | 8,841831e-03 | +1,098 dB |
| 2 LS-Durchlaeufe | 7 | 35 | 9,076392e-03 | +0,984 dB |
| 1 LS-Durchlauf | 4 | 20 | 9,438023e-03 | +0,814 dB |
| **nur die Spannweite** (das Gitter [min, min+15s], keine Verfeinerung) | **2** | **10** | 1,031334e-02 | **+0,429 dB** |

**Der Preis ist abstellbar, und das Knie ist bei 2 bis 4 Durchlaeufen.** Die **Spannweitenregel
allein** (2 Reduktionen: min und max -- **weniger Shuffles als der heutige Writer** mit 15) holt
**+0,43 dB** gegen die Lloyd-Tabelle, und **zwei LS-Durchlaeufe** (35 Shuffles) holen **+0,98 dB** --
88 % des Gewinns bei 37 % der Kosten. Der Offset ist in beiden Faellen derselbe (die linke
Gitterkante, c = o + 8s) -- **die Konstante wird in jedem Fall gespeichert**, die Frage ist nur, wie
genau (o, s) geschaetzt werden.

**Die klassische Alternative (Max 1960, die Iteration s = sqrt(12*MSE) mit dem Offset aus dem
kreisfoermigen Mittel der Residuen) ist in meiner Fassung DIVERGIERT** (1,884852e-02, also 3 dB
schlechter als die Lloyd-Tabelle) -- sie ist damit **nicht** die gesuchte Entlastung. Die
LS-Iteration mit der richtigen Initialisierung (das Gitter deckt den Block) ist es.

## Die Entscheidung, die ein **Stellrad** ist: der Durchlauf-Zaehler

**Der Fit kostet, und der Preis ist abstellbar.** Die Entscheidung ist deshalb **keine Festlegung,
sondern ein benanntes Stellrad mit Protokoll** — der Zaehler steht an **einer** Stelle im Code
(`kShearPasses`, ein benannter Wert in `kv_q4.hpp`), und jede Stellung ist **vor** dem GPU-Lauf
gegen das Harness gemessen.

| Stellung | Reduktionen | Shuffles je Block | Block-MSE (K) | gegen die Lloyd-Tabelle | der Preis gegen die Referenz |
|---|---:|---:|---:|---:|---|
| 6 Durchlaeufe (**die Referenz**, definiert den Gate-Wert G1) | 19 | 95 | 8,795968e-03 | +1,120 dB | — |
| 4 Durchlaeufe | 13 | 65 | 8,841831e-03 | +1,098 dB | 0,022 dB |
| **2 Durchlaeufe (die Voreinstellung)** | **7** | **35** | **9,076392e-03** | **+0,984 dB** | **0,136 dB** |
| 1 Durchlauf | 4 | 20 | 9,438023e-03 | +0,814 dB | 0,306 dB |
| 0 Durchlaeufe (nur die Spannweite) | 2 | 10 | 1,031334e-02 | +0,429 dB | 0,691 dB |

**Die Voreinstellung ist 2 Durchlaeufe**, aus zwei Gruenden:
1. **der Preis gegen die Referenz liegt unter der Aufloesung des Korpus** — 0,136 dB Block sind rund
   0,0006 nats End-to-End, und die kleinste messbare gepaarte Differenz auf 128K ist 0,00254 nats.
   **Der Verzicht auf vier Durchlaeufe ist also end-to-end nicht unterscheidbar** — er kostet nichts,
   was der Arm zeigen koennte;
2. **der Writer-Preis** (Gate G6): 35 Shuffles je Block gegen 15 heute sind rund 2,3x, bei 95 (sechs
   Durchlaeufe) waeren es 6,3x. Das Gate (Vorschlag: die Prompt-Lesedauer des Shear-Arms hoechstens
   1,25x des Produktarms) ist bei zwei Durchlaeufen mit Abstand zu erreichen und bei sechs vermutlich
   nicht.

**Die Regel, nach der nachgestellt wird** (sie steht hier, damit sie nicht spaeter ad hoc geaendert
wird):
- **Nach oben**, wenn das Gate G6 (der Writer-Preis) mit Reserve passiert **und** der End-to-End-Arm
  eine Wirkung zeigt, die an der Aufloesungsgrenze liegt: der Zaehler geht auf 4 (0,022 dB Block,
  65 Shuffles) — mehr als 4 bringt messbar nichts (0,022 dB sind 0,0001 nats).
- **Nach unten**, wenn das Gate G6 (der Writer-Preis) nicht passiert: auf 1 oder 0. Der Preis ist
  bekannt (0,306 dB bzw. 0,691 dB Block, rund 0,0014 bzw. 0,0030 nats End-to-End) und **vor** dem
  Arm zu nennen, damit die Aenderung nicht hinterher kommt.
- **Nie** ohne neue Harness-Messung: der Wert des Zaehlers ist **die** Zahl, die G1 prueft.

**Und was das ueber die Regelung des Zaehlers hinaus bedeutet:** der Nullpunkt ist **kein
Produktformat**, sondern eine **Messgroesse**, bis der Arm eine Wirkung zeigt. Seine Form
(Gitter [o, o+15s], die Konstante c = o + 8s) ist stabil; **nur die Genauigkeit der Schaetzung ist
verhandelbar** — und das ist der Unterschied zwischen einem Produktpfad und einem Messarm.
