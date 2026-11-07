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

## Was P1 jetzt ist

`p99` und `letzte 1.024` liegen bereits in den 6.144 Tail-Positionen; sie brauchen
keinen neuen GPU-Lauf. Das **Ganzsegment** ist der falsche Zielwert: es verwassert
den Tail. Ein reiner 16/16-Lauf **ohne Fenster** bleibt die einzige fehlende
Kontrolle, falls P0b Codec und Lesepfad trennen muss; er ist nicht P0b.

## Reproduktion (geplant)

```powershell
$out = 'C:\Users\krone\git\Strata-data\bench\2026-11-07-p0b'
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\p128k.ids.json `
    --kv q4lm32 --short-read 0 --prefill auto --topk 256 --allow-no-positions `
    --env STRATA_KV_Q4_DUMP=$out\kv.bin `
    --env STRATA_KV_Q4_DUMP_FROM=124877 --env STRATA_KV_Q4_DUMP_TO=131021 `
    --out $out\probe.tsv
python tools\kv_p0b_radial.py --dump $out\kv.bin --json $out\p0b.json
```

Der Lauf braucht die GPU; er startet erst nach dem ausdruecklichen OK.
