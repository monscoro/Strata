# Plan nach Daves zweiter Antwort - P0b

Stand 2026-11-07. Dieser Plan ist **vor** dem P0b-Lauf geschrieben. Er ersetzt den
P0/P1/P2-Plan aus `email_to_dave_reply.txt` nicht, sondern schaerft ihn nach P0a.

## Was feststeht

- Der Tag-Arm ist geschlossen (2026-11-04): kein Punkt mit einem Intervall ohne 0
  **und** |Mittel| > 0,0013. Kein Default, keine weiteren Laeufe.
- **16/16 ist der Produktkandidat.** Dave hat die pauschale Abwertung von L1
  zurueckgenommen; `turboquant_4bit_nc` ist 16/16, nicht 16/8.
- Der `int8`-Pfad bleibt byte-identisch. Kein Default aendert sich.
- Kein 5,0-bit/Zelle-Ziel; B1 bleibt geschlossen.

## P0a - gemessen (offline, 2026-11-07)

Daten: `2026-11-03-tag-arm/128k`, 6.144 Positionen (124.877...131.020),
`KL(fp16||Arm)` ueber fp16s top-256. `win` und `tag` sind `q4lm32` = 16/16;
`int8` ist die Latte.

| Arm | mean | median | p99 | max | mean ohne top 1 % |
|---|---:|---:|---:|---:|---:|
| `int8` | 0,000847 | 0,000002 | 0,001623 | 2,493 | 0,000021 |
| `win` (16/16) | 0,002108 | 0,000023 | 0,003362 | 3,211 | 0,000075 |
| `tag` (16/16) | 0,002115 | 0,000005 | 0,002849 | 2,881 | 0,000033 |

Gepaart `win - int8`: mean **+0,001261**, aber mean ohne top 1 % **-0,000079**;
**91,5 %** der Positionen sind besser. 18 Positionen mit Delta > 0,1 nats, 36 mit
Delta > 0,01. Die fuenf groessten: 130972 (+0,99), 127887 (+0,93), 130973 (+0,84),
130952 (+0,72), 130942 (+0,71) - zusammen **54 % des gesamten Abstands**. Letzte
1.024: +0,003790, ohne top 1 % **-0,000472**.

`tag - win`: mean +0,000007 (null), aber 5.407 Positionen besser und 26 schlechter;
letzte 1.024 -0,000516. Der Tag repariert z. B. 127887 und erzeugt 126234 - der
Tail wandert, der Mittelwert bleibt.

**Lesart.** Der 16/16-Abstand zu `int8` ist kein uniformer Shift, sondern ein Tail
aus wenigen Positionen. Auf 99 % der Positionen ist 16/16 nicht schlechter. Die
Normkorrektur ist eine radiale, per-Vektor-Korrektur; sie trifft unseren Tail nur,
wenn genau diese Ausreisser einen Normfehler haben.

## P0b - die Frage

Ist der 16/16-Fehler an den Tail-Positionen **radial** (Norm) oder **tangential**
(Richtung)? Nur wenn der radiale Anteil gross genug ist, kann eine per-Vektor-
Normkorrektur den Tail bewegen. Die 0,8 % PPL aus der Literatur sind eine
Mittel-Zahl und beantworten das nicht.

## P0b - der Capture

Ein Probelauf auf dem batched Prompt-Pfad (`--short-read 0`,
`--allow-no-positions`), `--kv q4lm32`, dieselbe Passage wie T3
(`p128k.ids.json`). Der Dump liegt **nach** der Hadamard-Rotation und **vor** der
Quantisierung; er enthaelt K und V fuer alle 12 QSA-Layer, 2 KV-Koepfe,
`head_dim` 256, Positionen 124.877...131.021.

Schalter (opt-in, Default aus, kein Byte auf dem schnellen Pfad):

| Variable | Bedeutung |
|---|---|
| `STRATA_KV_Q4_DUMP` | Zieldatei; ohne sie wird nichts kopiert |
| `STRATA_KV_Q4_DUMP_FROM` | erste Position (inklusive), Standard 0 |
| `STRATA_KV_Q4_DUMP_TO` | letzte Position (exklusive), Standard unbegrenzt |

Format: pro Record `magic`, `layer`, `p0`, `T`, `NKV`, `HD`, dann K
(`T*NKV*HD` floats), dann V. Der Dump laeuft nur auf dem Prompt-Pfad, nicht in
einem Capture-Graphen.

## P0b - die Zerlegung

Fuer jeden Vektor `x` (Position, Layer, Kopf, K oder V), `d = 256`:

```
x_dq    = quantize(x)                     # Lloyd-Tabelle, Skala s
n       = ||x||,  n_dq = ||x_dq||
cos     = <x, x_dq> / (n * n_dq)
radial  = (n - n_dq)^2
tangent = 2 * n * n_dq * (1 - cos)
total   = ||x - x_dq||^2 = radial + tangent
```

Die Orakel-Normkorrektur ist `x_corr = x_dq * (n / n_dq)`; dann ist
`||x - x_corr||^2 = tangent`. Der radiale Anteil ist `r = radial / total`.

Vier Arme auf denselben Vektoren:

1. **max/87** - der heutige 18-Byte-Block (`kLloyd`), `s = max|x| / 87`.
2. **RMS/32** - Daves feste Tabelle, `s = RMS(x) / 32`.
3. **max/87 + Norm** - Orakel-Korrektur auf Arm 1 (minimaler 18-Byte-Build).
4. **RMS/32 + Norm** - Orakel-Korrektur auf Arm 2 (16-Byte-Format).

Zusaetzlich **int8** auf denselben Vektoren (8-bit uniform, eine Skala pro 64
Werte) als Codec-Latte. Ausgegeben wird die per-Wert-MSE (`total / 256`) fuer
alle Arme, der radiale Anteil `r`, und dieselben Zahlen nur fuer die Top-Gap-
Positionen aus P0a.

## P0b - Entscheidungsregel (vor dem Lauf)

- **Tot** wenn `r < 0,05` im Mittel **und** an den Top-Gap-Positionen: die
  Korrektur kauft < 0,22 dB, nicht die 12 MiB und die Kernel-Aenderung.
- **Marginal** wenn `0,05 <= r < 0,20`: 0,22...0,97 dB; erst P2 entscheidet, kein
  Build.
- **Ueberlebt** wenn `r >= 0,20`: >= 0,97 dB; der minimale 18-Byte-Build wird
  gebaut und in P2 (4K, 8 Segmente) gemessen.
- **Tail-Klausel:** ein hohes `r` im Mittel bei `r < 0,05` an den Top-Gap-
  Positionen ist **kein** Tail-Gewinn. Dann wird nicht fuer den Tail gebaut; die
  Korrektur bleibt ein Mittelwert-Hebel.
- **16-Byte-Format** nur wenn `MSE(RMS/32 + Norm) < MSE(max/87)`. Sonst bleibt der
  18-Byte-Block.
- **Minimaler Build** nur wenn `MSE(max/87 + Norm) <= 0,95 * MSE(max/87)`.
- **P2** (4K teacher-forced, 8 Segmente, Arme 16/16 / 16/16+Norm / int8) nur nach
  einem "ueberlebt". P0b entscheidet den Codec, P2 den Produkt-KL.

## Nachtrag nach Daves dritter Antwort (vor dem Lauf)

Drei Punkte aus `daves_thanks_reply.md` aendern den Lauf, keinen Schwellenwert:

1. **Die gesuchte Zahl existiert nicht.** Es gibt keine veroeffentlichte p99-,
   letzte-1.024- oder Worst-Position-Zahl fuer `turboquant_4bit_nc`, nur PPL. P0b
   produziert sie; der Plan bleibt.
2. **Der Qwen-Worst-Case-Kollaps.** Villines' Per-Head-Minimum-Cosine-Sonde
   berichtet fuer uniformes TurboQuant einen minimalen Cosinus nahe null oder
   negativ auf Qwen - bei nominalem Mittelwert. Deshalb misst das Werkzeug
   zusaetzlich den **minimalen Cosinus pro (Layer, Kopf, K/V)**, einmal ueber
   alle 6.144 Positionen (die Sonde) und einmal an den Top-Gap-Positionen. Das
   ist ein Befund, keine neue Entscheidungsregel: der Cosinus sagt, ob die
   Richtung kollabiert, `r` sagt, ob eine Normkorrektur das beheben koennte.
3. **Die Schwelle gilt an den Top-Gap-Positionen.** `r` wird primaer an den 18
   Positionen mit dem groessten P0a-Abstand gelesen, nicht als Mittel ueber
   6.144. Die Werte bleiben 5 % / 20 %; die Tail-Klausel von oben wird damit zur
   Hauptregel statt zur Nachbemerkung. Das Mittel wird weiter berichtet.

**KVarN-Per-Token-Skalierung ist schon im Capture.** Einen Vektor vor der
Quantisierung auf RMS 1 zu normieren, mit der festen Tabelle und `d = 1/32` zu
quantisieren und die gespeicherte Skala danach zurueckzumultiplizieren, ergibt
`code * RMS/32` - genau den `RMS/32`-Arm. Der mechanistisch andere Hebel ist
damit ohne weiteren Arm gemessen; er unterscheidet sich von der Normkorrektur
darin, dass die Skala **vor** der Quantisierung bekannt ist, nicht danach
geschaetzt wird.

Vor dem 128K-Lauf steht ein 32K-Smoke (`p32k.ids.json`, Dump 26.756...32.755):
er prueft Format, Chunking und den Analysepfad, bevor der grosse Lauf startet.

## P0b - Ergebnis (gemessen 2026-11-07)

**Capture.** `p128k.ids.json` (131.022 Token), batched Prefill (`--short-read 0`),
`--kv q4lm32`, Dump nach der Hadamard-Rotation und vor `kv_append_q4`. 12 QSA-
Layer (3...47), 2 KV-Koepfe, K und V getrennt, Positionen 124.877...131.020
(6.144), 75.497.472 Werte, 302 MB, Laufzeit 981,7 s. Engine 0.1.47 aus
`build-hip-port` (ROCm 7.1, gfx1200, RX 9060 XT). Der 32K-Smoke (5.999
Positionen, 26.756...32.754) lief mit denselben Zahlen und diente nur der
Formatpruefung.

**Alle Werte, 6.144 Positionen:**

| Arm | bit/Wert | MSE | radialer Anteil | Gewinn gegen max/87 |
|---|---:|---:|---:|---:|
| `max/87` (18 B) | 4,50 | 8,4041e-03 | **0,6 %** | 0,00 dB |
| `RMS/32` per Vektor (16 B) | 4,12 | 1,1505e-02 | 1,2 % | −1,36 dB |
| `max/87` + Norm | 4,62 | 8,3394e-03 | 0,0 % | +0,03 dB |
| `RMS/32` + Norm | 4,12 | 1,1429e-02 | 0,0 % | −1,34 dB |
| `int8` | 8,25 | 4,2964e-05 | 0,4 % | +22,91 dB |
| `RMS/32` per 32er-Block | 4,00 | 1,0919e-02 | 0,9 % | −1,14 dB |

K/V getrennt: K 2,472e-03 / 0,6 %, V 3,296e-04 / 0,7 % (max/87); `RMS/32`
K 3,382e-03 / 1,2 %, V 4,526e-04 / 1,3 %.

**Die 18 Top-Gap-Positionen aus P0a** (130972, 127887, 130973, 130952, 130942,
130968, 126191, 125074, 128793, 130954, 128883, 124891, 127727, 128709, 128516,
125064, 128583, 130951): `max/87` MSE 8,2036e-03, radial **0,6 %**;
`max/87`+Norm 8,1403e-03 (+0,03 dB); `int8` 4,1521e-05 (+22,96 dB). Der
KV-Fehler an diesen Positionen ist also **nicht groesser** als im Mittel.

**Norm und Cosinus.** Mittlere Vektor-Norm 15,871 im Ganzen, 15,702 an den
Top-Gap-Positionen (1,0x); Pearson r(Norm, P0a-Gap) = −0,041. Minimaler Cosinus
pro (Layer, Kopf, K/V) fuer `max/87`: **0,9949** ueber alle 48 Einheiten, keine
unter 0,5, keine negativ; an den Top-Gap-Positionen min 0,9955, p1 0,9959,
Median 0,9966. Fuer `RMS/32`: min 0,9790, an den Top-Gap-Positionen 0,9898.
Ein Spot-Check am Dump: Position 130972 hat MSE 8,18e-03, min Cosinus 0,9959,
mittlere Norm 15,88; Position 130000 7,57e-03, 125500 8,44e-03.

**Lesart.** Der 16/16-Fehler ist **tangential**: die Normkorrektur kauft
0,03 dB, die Schwelle verlangt 0,77 dB (0,95x MSE). Daves Vermutung trifft zu.
Der Tail ist kein Codec-Fehler: an den 18 Positionen, die 54 % des P0a-Abstands
tragen, sind Norm, MSE und Cosinus unauffaellig. Die Logits sind dort
empfindlich, nicht die K/V schlechter. Villines' Worst-Case-Kollaps (min
Cosinus nahe null) reproduziert auf diesem Modell mit dieser Tabelle **nicht**;
das ist ein Befund ueber unseren Arm, keine Widerlegung seiner Messung an
`turboquant_4bit_nc`. `RMS/32` ist zudem schlechter als `max/87` (−1,36 dB) und
hat den groesseren radialen Anteil - das 16-Byte-Format traegt nicht.

**Entscheidung (nach der vorab festgelegten Regel).** `r` = 0,6 % an den
Top-Gap-Positionen < 5 % → **tot**. Kein minimaler 18-Byte-Build, kein
16-Byte-Format, kein P2. Kein 5,0-bit-Budget, kein Default, B1 bleibt
geschlossen.

**Grenze.** Die 18 Positionen stammen aus dem gefensterten P0a-Arm (`win`).
Der reine `q4lm32`-Arm schreibt auf dem batched Pfad keine Logits, sein eigener
Tail ist also nicht gemessen. Die Aussage gilt fuer den KV-Fehler an diesen
Positionen, nicht fuer die Behauptung, der reine Arm habe dort keinen
Logit-Tail. Genau das ist die verbleibende P1-Kontrolle.

## Was P1 jetzt ist

`p99` und `letzte 1.024` liegen bereits in den 6.144 Tail-Positionen; sie brauchen
keinen neuen GPU-Lauf. P0b hat den **Codec** entschieden (Normkorrektur tot,
16-Byte-Format schlechter), aber nicht den Tail des **reinen** Arms: die 18
Positionen stammen aus dem gefensterten `win`-Lauf. Der reine 16/16-Lauf **ohne
Fenster** (`--kv q4lm32`, `--short-read 6144`, kein `STRATA_KV_WINDOW`) ist
damit die naechste und einzige fehlende Kontrolle: er sagt, ob der P0a-Tail ein
Codec-Effekt oder ein Effekt des Lesepfads (exaktes Fenster plus q4-Bulk) ist.
Er aendert keinen Default und oeffnet kein 5,0-bit-Budget.

## Reproduktion (gelaufen 2026-11-07)

```powershell
$out = 'C:\Users\krone\git\Strata-data\bench\2026-11-07-p0b'
# 32K-Smoke (Format und Chunking)
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids C:\Users\krone\git\Strata-data\bench\2026-11-03-deploy-range\p32k.ids.json `
    --kv q4lm32 --short-read 0 --prefill auto --topk 256 --allow-no-positions `
    --env STRATA_KV_Q4_DUMP=$out\kv32k.bin `
    --env STRATA_KV_Q4_DUMP_FROM=26756 --env STRATA_KV_Q4_DUMP_TO=32756 `
    --out $out\probe32k.tsv
# 128K-Capture
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\p128k.ids.json `
    --kv q4lm32 --short-read 0 --prefill auto --topk 256 --allow-no-positions `
    --env STRATA_KV_Q4_DUMP=$out\kv.bin `
    --env STRATA_KV_Q4_DUMP_FROM=124877 --env STRATA_KV_Q4_DUMP_TO=131021 `
    --out $out\probe.tsv
# Zerlegung; die Top-Gap-Positionen kommen aus P0a (win gegen int8)
python tools\kv_p0b_radial.py --dump $out\kv.bin --top-n 18 `
    --arm-tsv C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\128k\win.tsv `
    --ref-tsv C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\128k\fp16.tsv `
    --base-tsv C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\128k\int8.tsv `
    --json $out\p0b.json
```

Der Dump (`kv.bin`) und die Zahlen (`p0b.json`) liegen in
`Strata-data\bench\2026-11-07-p0b`; der 32K-Smoke dort als `kv32k.bin` und
`p0b32k.json`. Die Analyse braucht keine GPU.
