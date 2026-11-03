# Wiederholungs-Kontrolle bei 64K: ist der 128K-Nullpunkt ein Längen- oder ein Wiederholungsartefakt?

**Status: vorregistriert (2026-11-03), vor dem ersten Lauf.** Der erste Arm-Lauf darf erst nach dem
Schreiben dieses Dokuments starten. Anschluss an `bench/results/2026-11-03-deploy-range/`.

## Warum

Der 128K-Punkt ist mit Wiederholung konfundiert: die Passage ist cwe-8k **zweimal**, und der
128K-Tail (124.877…131.020) ist Token für Token derselbe Text wie der 64K-Tail (59.367…65.509),
nur um 65.511 verschoben — geprüft, 6.143 von 6.143 Token identisch. Die KLs sind dort 20- bis
50-mal kleiner als bei 64K (`int8` im Mittel 0,00085 gegen 0,04012). Ob die Länge oder die zweite
Lesung das verursacht, lässt sich mit den vorhandenen Punkten nicht trennen.

Ein 128K-Tail aus **frischem** Text ist mit den vorhandenen Korpora nicht baubar: cwe-8k hat
8 Segmente = 65.511 Token, cwe-4k ist anderer Text (32.754 Token), `v2-8k` 24.576, `v1` 5.000 —
131.072 gingen nur durch Mischen verschiedener Textsorten, und dann wäre der Tail nicht mehr mit
32K/64K/128K vergleichbar. Deshalb wird die Länge **festgehalten** und nur die Wiederholung
variiert.

## Aufbau

Zwei 65,5K-Kontexte aus denselben acht cwe-8k-Segmenten, gleicher Tail-Text, gleiche 6.144
Tail-Token, gleiches Protokoll wie der 64K-Punkt (`--short-read 6144`, `--topk 256`,
`--prefill auto`, `--prompt-cache 3`, `pin = n − 6144`, `--allow-no-positions`):

| Punkt | Reihenfolge | Token | `pin` | Tail | Vorgeschichte des Tails |
|---|---|---|---|---|---|
| `F` (frisch) | s04 s05 s06 s07 **s00 s01 s02 s03** | 65.511 | 59.367 | Ende s03 | s00–s03 **nirgends vorher** im Kontext |
| `R` (wiederholt) | s00 s01 s02 s03 **s00 s01 s02 s03** | 65.512 | 59.368 | Ende s03 | s00–s03 **32.756 Token früher** schon einmal gelesen |

In beiden Kontexten ist der unmittelbare Präfix des Tails derselbe Text (s00, s01, s02, Anfang
s03) — in `F` frisch, in `R` eine Wiederholung. Der Unterschied zwischen den Punkten ist damit
die Wiederholung, nicht der Tail-Text. Die 6.144 Tail-Token sind in beiden Punkten identisch, also
lassen sich die per-Position-KLs **über den Tail-Offset** paaren (Offset 0…6.143), obwohl die
absoluten Positionen um eins differieren.

| Punkt | Arme |
|---|---|
| `F`, `R` | `fp16`, `int8`, `win` (`q4lm32`, 256/64), `tag` (`q4lm32`, 256/64 + `STRATA_KV_WINDOW_TAG=1`) |

Ein GPU-Job zur Zeit, 30 s zwischen zwei Engine-Starts, Engine-SHA `CD5700EC0793F83B…3D0394CA`
vor dem Lauf geprüft.

## Entscheidungsregeln (vor dem Lauf festgelegt)

Δ = KL(fp16‖Arm) − KL(fp16‖Basis), negativ = der Arm ist besser. Gepaart über den Tail-Offset,
Moving-Block-Bootstrap über Blöcke von 256 aufeinanderfolgenden Offsets (24 Blöcke, 2.000
Resamples, Seed 20261103).

- **`P1` — senkt Wiederholung den Fehler?** Verglichen wird `KL(fp16‖int8)` in `R` gegen `F`.
  **„Wiederholung ist der Treiber"** nur wenn **sowohl** der Median **als auch** der Mittelwert in
  `R` höchstens **die Hälfte** von `F` sind. Sonst „Wiederholung senkt den Fehler nicht messbar".
  (Der 128K/64K-Vergleich lag bei 47× im Mittel und bei Median 0,00000 gegen 0,00091; die
  Halbierungs-Schwelle ist bewusst konservativ.)
- **`P2` — trägt der Tag auch auf wiederholtem Text?** Gepaart `tag − win` in `R` (und in `F` als
  Kontrolle). **„Tag trägt"** nur wenn das Bootstrap-95-%-Intervall die 0 ausschließt, der
  Mittelwert negativ ist **und** |Mittelwert| > 0,0013 nats — dieselbe Schwelle wie `D1`/`T3b`,
  unverändert.
- **`P3` — ist die Lücke zu `int8` geschlossen?** Gepaart `tag − int8` in `R` und `F`.
  **„geschlossen"** wenn der Mittelwert ≤ 0 ist **und** die einseitige 95-%-Obergrenze unter
  +0,001 nats liegt (die K1-Grenze).
- **`P4` — kein Quervergleich.** `F`, `R`, 32K, 64K und 128K werden einzeln berichtet; keine
  gepoolte Zahl und keine Aussage „der Effekt fällt mit der Länge".

### Güte-Gatter (vor der Auswertung, je Arm)

- `DONE` ohne „nonfinite"; kein „do not fit"; kein `ERR`; Engine-SHA unverändert.
- TSV mit **6.143** Positionen; innerhalb eines Punktes in allen vier Armen **dieselbe**
  Positionsmenge; `win`/`tag` mit der Ring-Zeile, `tag` zusätzlich mit der
  `STRATA_KV_WINDOW_TAG=1`-Zeile; `seg.json` mit `pin` und `prompt_cache` = 3.
- Die 6.144 Tail-Token von `F` und `R` sind identisch (vor der Auswertung geprüft).

### Grenzen, offen genannt

1. **Die erste Hälfte unterscheidet sich** (`s04–s07` gegen `s00–s03`). „Wiederholung" ist damit
   nicht perfekt isoliert: in `R` stand 32,8K früher derselbe Text, in `F` anderer. Der
   unmittelbare Präfix des Tails ist in beiden derselbe — das ist der Teil, den die Attention
   überwiegend liest, aber ein Rest-Konfund bleibt.
2. **Abstand 32,8K, nicht 65,5K.** Ein positives `P1` zeigt, dass Wiederholung überhaupt wirkt;
   es beweist nicht, dass der 128K-Abfall (Abstand 65,5K) daher kommt. Ein negatives `P1` schließt
   Wiederholung bei 65,5K Abstand **nicht** aus.
3. **Kein Timing, kein Byte-A/B** — der Tag kostet statisch 0 Bytes.

### Kosten

Acht Läufe (2 Punkte × 4 Arme), je ~65,5K: ~490 s Prefill + 6.144 Fenster-Schritte, nach der
64K-Erfahrung (629–667 s) zusammen etwa 90 Minuten GPU, kein Engine-Bau.

## Reproduktion

```powershell
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids C:\Users\krone\git\Strata-data\bench\2026-11-03-repeat-control\p64k-fresh.ids.json `
    --kv <fp16|int8|q4lm32> --prefill auto --short-read 6144 --topk 256 --allow-no-positions `
    --pin 59367 --prompt-cache 3 `
    [--env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 --env STRATA_KV_WINDOW_R=64 `
     [--env STRATA_KV_WINDOW_TAG=1]] `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-03-repeat-control\fresh\<arm>.tsv
#   p64k-rep genauso, --pin 59368, Ausgabe ...\rep\<arm>.tsv
```

Rohdaten liegen außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-11-03-repeat-control\`; im Repo steht dieses Dokument und
nach dem Lauf der Ergebnisabschnitt darunter.
