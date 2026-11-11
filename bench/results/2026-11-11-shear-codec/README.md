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
