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

---

# Ergebnis

**Status: gemessen (2026-11-03/04).** Alle Güte-Gatter bestanden: je Punkt und Arm **6.143**
Positionen, innerhalb eines Punktes dieselbe Positionsmenge in allen vier Armen, `win`/`tag` mit
der Ring-Zeile, nur `tag` mit der `STRATA_KV_WINDOW_TAG=1`-Zeile, kein „nonfinite", kein
„do not fit", kein `ERR`, Engine-SHA unverändert. Die 6.144 Tail-Token von `F` und `R` sind
identisch (6.144/6.144 geprüft), die Positionen liegen bei `F` 59.367…65.509 und bei `R`
59.368…65.510.

## Die zwei Punkte

`KL(fp16‖Arm)` über `fp16`s top-256, je Position, 6.143 Positionen pro Arm und Punkt:

| Punkt | Arm | Mittel | Median | Positionen > 1 nat | max |
|---|---|---|---|---|---|
| `F` (frisch) | `int8` | 0,02669 | 0,000739 | 37 | 11,13 |
| `F` (frisch) | `win` | 0,02810 | 0,001323 | 36 | 13,25 |
| `F` (frisch) | `tag` | 0,02066 | 0,000963 | 32 | 7,92 |
| `R` (wiederholt) | `int8` | 0,00352 | 0,000055 | 2 | 7,74 |
| `R` (wiederholt) | `win` | 0,00483 | 0,000089 | 4 | 9,10 |
| `R` (wiederholt) | `tag` | 0,00659 | 0,000302 | 4 | 8,15 |

## `P1` — senkt Wiederholung den Fehler? **Ja, deutlich.**

`KL(fp16‖int8)`: Mittel **0,02669 → 0,00352** (Verhältnis 0,132), Median **0,000739 →
0,000055** (Verhältnis 0,074). Beide Klauseln der Regel (Mittel **und** Median höchstens die
Hälfte) halten klar. Dasselbe Bild bei den anderen Armen: `win` 0,172 im Mittel / 0,067 im
Median, `tag` 0,319 / 0,314. Gepaart über den Tail-Offset (dieselben 6.144 Token, nur in einem
anderen Kontext gelesen) ist die Differenz `R − F` negativ mit Intervall ohne 0: `int8`
−0,02317 [−0,03136; −0,01486], `win` −0,02327 [−0,03387; −0,01379], `tag` −0,01407
[−0,02095; −0,00736].

**Damit ist Wiederholung als Treiber bestätigt — bei einem Abstand von 32.756 Token.** Der
128K-Punkt liest denselben Text bei einem Abstand von 65.511 Token und liegt 47-mal unter dem
64K-Punkt; der 64K-Punkt mit 32,8K Abstand liegt 7,6-mal darunter. Beide Beobachtungen zeigen in
dieselbe Richtung; sie beweisen nicht, dass der 128K-Abfall **nur** von der Wiederholung kommt
(siehe Grenzen).

## `P2` — trägt der Tag auch auf wiederholtem Text? **Nein — und auf wiederholtem Text schadet er.**

Gepaart `tag − win`, negativ = Tag besser:

| Punkt | `tag − win` | Block-95 % | einseitige Grenze |
|---|---|---|---|
| `F` (frisch) | **−0,00743** | [−0,01728; +0,00093] | −0,00014 |
| `R` (wiederholt) | **+0,00176** | [+0,00024; +0,00359] | +0,00331 |

In `F` zeigt der Punktwert zum Tag, das Intervall schließt die 0 aber ein (wie bei 32K/64K/128K):
die registrierte Regel „Tag trägt" ist **nicht** erfüllt. In `R` ist das Vorzeichen umgedreht und
das Intervall schließt die 0 aus: der Tag ist dort **signifikant schlechter** als das Fenster
(+0,00176 nats, 4.749 von 6.143 Positionen schlechter). Der Effekt der Tag-Lesung wechselt mit
der Wiederholung das Vorzeichen.

## `P3` — ist die Lücke zu `int8` geschlossen? **In `F` nach der Regel ja, in `R` nein.**

| Punkt | `tag − int8` | Block-95 % | einseitige Grenze | Regel |
|---|---|---|---|---|
| `F` (frisch) | **−0,00603** | [−0,01204; −0,00033] | −0,00134 | **geschlossen** (Mittel ≤ 0, Obergrenze < +0,001) |
| `R` (wiederholt) | **+0,00307** | [+0,00177; +0,00471] | +0,00439 | nicht geschlossen |

Im frischen 64K-Kontext liegt der Tag **unter** `int8` (0,02066 gegen 0,02669), mit Intervall ohne
0. Das ist der erste Punkt der Reihe, an dem der gepaarte Unterschied zu `int8` das Intervall ohne
0 hat (am 64K-Punkt aus `2026-11-03-deploy-range/` war der Punktwert ebenfalls zugunsten des Tags,
das Intervall schloss die 0 aber ein). Der Befund
ist allerdings **floor-getragen** (nächster Abschnitt): ohne die Positionen über 1 nat bleibt
`tag − int8` = +0,00096 [−0,00003; +0,00196], also Gleichstand statt Vorsprung. In `R` ist der Tag
auch nach der Regel schlechter als `int8`.

## Beschreibend, nicht registriert

Ohne die Offsets, an denen ein Arm im jeweiligen Kontext über 1 nat liegt (`F`: 65 entfernt,
`R`: 7), ergibt sich gepaart:

| Punkt | `tag − win` (getrimmt) | `tag − int8` (getrimmt) |
|---|---|---|
| `F` | **−0,00130 [−0,00244; −0,00009]** | +0,00096 [−0,00003; +0,00196] |
| `R` | **+0,00175 [+0,00079; +0,00298]** | +0,00240 [+0,00156; +0,00362] |

Getrimmt ist der Tag auf frischem Text besser als das Fenster (0,0013 nats, Intervall ohne 0,
knapp an der 0,0013-Schwelle) und auf wiederholtem Text schlechter (+0,0018, Intervall ohne 0).
Gegen `int8` bleibt er an beiden Punkten auf oder hinter `int8`-Niveau. Die Trimmung ist
**post hoc** und entscheidet nichts; sie steht hier, weil der registrierte Mittelwert an beiden
Punkten von 0,1–1,1 % der Positionen (65 bzw. 7 Offsets) regiert wird.

## Was das heißt

- **Registriert:** Wiederholung senkt den Fehler deutlich (`P1`: 7,6-mal im Mittel, 13-mal im
  Median bei 32,8K Abstand). Der Tag trägt auf keinem der beiden Punkte (`P2`); auf wiederholtem
  Text ist er signifikant schlechter als das Fenster. Die `int8`-Lücke ist nur im frischen
  Kontext nach der Regel geschlossen (`P3`), und dort nur, solange die Floor-Positionen im
  Mittel bleiben.
- **Der 128K-Nullpunkt ist damit zum großen Teil erklärt:** die zweite Lesung derselben Passage
  senkt die KL um fast eine Größenordnung, und sie dreht das Vorzeichen des Tag-Effekts. Der
  128K-Punkt ist kein Beleg dafür, dass der Tag bei 128K nichts kann — er ist ein Beleg dafür,
  dass er auf bereits gelesenem Text nichts kann.
- **Für das Produkt:** auf frischem Text ist der Tag eine kleine, kostenlose Verbesserung
  (getrimmt 0,0013 nats), auf wiederholtem Text ein kleiner Schaden (+0,0018). Das ist kein
  Auslieferungs-Default und kein 5,0-bit/Zelle-Hebel; `B1` bleibt geschlossen.
- **Offen:** 256K (`2026-11-03-256k/`, eigener Lauf) — dort ist der Tail die vierte Lesung
  desselben Textes, also maximal wiederholt.

## Abweichungen vom registrierten Plan

1. **6.143 statt 6.144 Positionen** — dieselbe Pin-Arithmetik wie bei 32K/64K (`pin = n − 6144`
   ergibt `n − 1 − pin = 6.143`); alle vier Arme eines Punktes teilen dieselben Positionen, keine
   Regel hängt an der Zahl.
2. **Der erste Lauf wurde durch einen Server-Neustart abgebrochen** (2026-11-03, `fresh fp16`,
   0-Byte-TSV) und am 2026-11-04 vollständig neu gefahren; die TSV ist die des Neustarts.
3. **`P1` ist ein Vergleich zweier Kontexte, kein gepaarter Eingriff** — die erste Hälfte
   unterscheidet sich (`s04–s07` gegen `s00–s03`), siehe Grenzen.
4. **Kein Timing, kein Byte-A/B.**

## Grenzen, offen genannt

1. **Die erste Hälfte unterscheidet sich.** In `R` stand 32,8K früher derselbe Text, in `F`
   anderer. Der unmittelbare Präfix des Tails ist in beiden derselbe (`s00`, `s01`, `s02`, Anfang
   `s03`), aber ein Rest-Konfund bleibt: „Wiederholung" ist hier „derselbe Text 32,8K früher plus
   anderer Text davor".
2. **Abstand 32,8K, nicht 65,5K.** `P1` zeigt, dass Wiederholung überhaupt wirkt; es beweist
   nicht, dass der 128K-Abfall (Abstand 65,5K) allein daher kommt.
3. **Ein Segment je Punkt**, keine Segment-Clusterung; die 6.143 Positionen sind geschachtelt.
4. **`P3`-„geschlossen" in `F` ist floor-getragen** — die getrimmte Zahl steht daneben und ist als
   post hoc markiert.

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
