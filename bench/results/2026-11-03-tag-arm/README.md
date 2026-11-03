# T1/T2/T3 — der Tag-Arm: Lesepolitik am 4K-Decode-Punkt und teacher-forced am 128K-Punkt

**Status: gemessen (2026-11-03), Ergebnis unten. Vorregistriert vor dem ersten Arm-Lauf.** Der
erste Arm-Lauf (auch der Reuse-Gate-Lauf `D0`) durfte erst nach dem Schreiben dieses Dokuments
starten. Katalogzeile:
`docs/paper/Ansatz_Improvements.md`, E1/S1-Fortsetzung (der Qualitätsbeweis, den
`bench/results/2026-11-01-e1-128k/` ausdrücklich offen lässt). Anschluss an
`bench/results/2026-10-27-s3-level-split/` (die vorhandene teacher-forced 4K-Messung) und
`bench/results/2026-11-01-decode-4k/` (Protokoll und Auflösung des Decode-Punkts).

## Warum

E1 hat den Hebel benannt und die Zahl dazu geliefert: am 128K-Punkt macht die Tag-Lesung
(`STRATA_KV_WINDOW_TAG=1`) **48,55 %** der Indexer-Auswahl exakt, statt 8,55 % — für **null
zusätzliche Bytes**. Die 32K-Hälfte (S1) deckte 70,2 %. Was fehlt, ist die Qualitätsseite: schlägt
sich die größere exakte Lesemenge in der Verteilung nieder, und reicht sie an `int8` heran? Die
eine vorhandene Messung (s3, teacher-forced 4K) sagt „nicht signifikant"; sie lief aber am
`q4lm8v`-Bulk (7,39 bit/value) und mittelt über alle Prefill-Positionen. Dieser Lauf misst am
Produktpunkt (`q4lm32`, 16/16) und an zwei Orten, an denen der Effekt nicht verwässert: dem
Decode-Punkt bei 4K (die Verteilung ist eingefroren) und dem 128K-Punkt (dort ist die Lücke
zwischen Tag und Fenster am größten).

## Was der Arm technisch ist (statische Analyse)

Ein reiner **Lesepolitik**-Arm: derselbe Ring, dieselben Bytes, dieselbe Indexer-Auswahl. Nur die
Frage „liest diese Zelle ihre residente fp16-Kopie oder den quantisierten Bulk?" wird anders
beantwortet.

| | |
|---|---|
| Schalter | `STRATA_KV_WINDOW_TAG=1` → `g_kv_win_all` (`src/core/layer.cpp:215, 281–290`), weitergereicht als `pools.win_all` (`layer.cpp:1166`) |
| Prädikat | `qsa_win_slot` (`include/strata/kernels/qsa_decode_attn.hpp:74–82`): `win_all = 0` liest exakt nur `c < R` oder `c ≥ n − S` (publizierter Horizont, unverändert). `win_all = 1` lässt den Horizont-Check fallen; es entscheidet der Zelltag `win_tag[c % (S+B)] == c`. Sinks `c < R` bleiben in beiden Fällen exakt |
| Kosten | 0 zusätzliche Bytes: die Tags (4 B je recent-Zeile) und die fp16-Zeilen sind schon resident. Kein neuer Kernel, kein neues Format, Ringgröße und Auswahl unberührt |
| Default | unverändert (Flag aus); der `int8`-Pfad ist von beidem nicht berührt |
| Korrektheit | `kv_win_parity` (C1) prüft beide TAG-Stellungen: 31.462 Zell-Checks je Horizontphase, 0 Abweichungen |

Der Arm ist also nicht „mehr Kontext" und nicht „mehr Genauigkeit im Bulk" — er ist die Frage, ob
das *schon bezahlte* fp16-Material dort gelesen wird, wo der Tag es noch benennt.

## Was schon gemessen ist

| Messung | Ergebnis |
|---|---|
| Abdeckung, 128K (E1, `n_kv` > 16.384) | Tag-Lesung **48,55 %** [48,54; 48,56] der Auswahl exakt, gegen 8,55 % heute; je Horizont 100,0 → 94,6 → 80,0 → **48,6 %** |
| Abdeckung, 32K (S1) | Tag-Lesung 70,2 %, exakt 11,7 % |
| Qualität teacher-forced 4K (s3, Daten aus `2026-10-25-tag-quality`) | `q4lm8v`, 256/64/512, cwe-4k, dieselben Positionen: KL vs fp16 **0,02107 (Tag)** gegen **0,02223 (Fenster)** — der Tag liegt 0,00116 nats niedriger; gepaartes 95-%-Intervall [−0,00162; +0,00395], `t` 0,82, 3/8 Segmente. **Nicht signifikant** |
| Auflösung des 4K-Decode-Punkts (decode-4k) | zwei Fenster-Arme am selben Punkt: `w256r64B512 − w256r64` = +0,00005 [−0,00188; +0,00197], einseitige Grenze bei Parität **0,0013**; die Spanne über die dortigen Fenster-Paare ist 0,0008–0,0027 |
| Decode-Punkt 4K, Tag | **nicht gemessen** — die Lücke, die T1/T2 schließen |
| 128K-Qualität | **nicht gemessen** — die Lücke, die T3 schließt |

Das s3-Dokument druckt die gepaarte Differenz mit umgekehrtem Vorzeichen („Tag-Read − Fenster
+0,00117", obwohl 0,02107 − 0,02223 = −0,00116 ist). Hier gilt durchgehend:
**Δ = KL(fp16‖Arm) − KL(fp16‖Basis), negativ = der Arm ist besser.**

## Aufbau

### T1/T2 — 4K-Decode-Punkt

Derselbe Aufbau wie `2026-11-01-decode-4k` (dort schon gemessen und dokumentiert): `prompt`-Pfad,
8 Segmente × 6 geschachtelte Längen (256/512/768/1000/2048/4056) = 48 Requests je Arm, ein
Decode-Schritt, `STRATA_DUMP_FIRST_LOGITS` (float32, ganzes Vokabular),
`KL(fp16‖Arm)` über das volle Vokabular, gepaart je Prompt, geclustert je Segment (df = 7,
`t_crit` 2,365).

| Arm | `--kv` | Fenster-Umgebung | `--prefill` | Herkunft |
|---|---|---|---|---|
| `fp16` | `fp16` | — | auto | **Reuse** `2026-11-01-decode4k/fp16` (Referenz) |
| `int8` | `int8` | — | auto | **Reuse** `2026-11-01-decode4k/int8` (die Latte) |
| `winB512` | `q4lm32` | `WINDOW=1 S=256 R=64 BATCH=512` | **512** | **Reuse** `2026-11-01-decode4k/w256r64B512` |
| `tagB512` | `q4lm32` | dasselbe **+ `WINDOW_TAG=1`** | **512** | neu |
| `tagAuto` | `q4lm32` | `WINDOW=1 S=256 R=64` **+ `WINDOW_TAG=1`** | auto | neu |

Zwei Designpunkte, ohne die der Arm nichts aussagt:

1. **`B = 512`, nicht `auto`, für T1/T2.** Mit `--prefill auto` hält der Ring bei 4K den ganzen
   Prompt (6.400 Zeilen ≥ 4.056 Token), jeder Tag trifft → der Tag liest *alles* exakt, der Arm
   ist identisch mit `fp16`. Erst die Verdrängung macht den Vergleich informativ. Genau diesen
   Fall (`tagAuto`) fahren wir als **Mechanik-Nullprobe** mit: er muss `fp16` bit für bit
   reproduzieren.
2. **`q4lm32`, nicht `q4lm8v`.** Der Produktpunkt ist 16/16; s3 lief am 16/8-Bulk. Der
   Tag-Mechanismus hängt nicht am Codec, die Baseline-KL schon.

Reuse-Gate `D0`: die drei Reuse-Arme stammen vom 2026-11-01. Der Build ist derselbe (SHA
`CD5700EC0793F83B…3D0394CA`, vor dem Lauf erneut geprüft), Config, `--ids`, `--lens` und die
`env`-Blöcke sind in ihren `index.json` festgehalten. Damit der Tagesunterschied nicht
stillschweigend in den gepaarten Vergleich wandert: ein **fp16-Kontrolllauf heute** über die
Länge 4056 (8 Requests) muss die `first.*`-Dumps von 2026-11-01 **byte-identisch** reproduzieren.
Gelingt das nicht, werden alle vier 4K-Arme heute neu gefahren (Kosten verdoppeln sich, das
Ergebnis bleibt dasselbe Protokoll).

### T3 — 128K, teacher-forced

Ein Segment, ein `GEN 1`, per-Position-Logits aus dem Verify-Fenster am Prompt-Ende.

| Punkt | Wert |
|---|---|
| Passage | cwe-8k, 8 Segmente, dann dieselben 8 noch einmal — **131.022 Token in *einem* Segment** (`p128k`). Die Ids sind die Konkatenation von `2026-11-01-e1-128k/cwe8k-x2.ids.json`; nur so entsteht ein 128K-Horizont (`seg_tokens` 131.022 → `max-context` 131.072) |
| Warum ein Segment | `run` schickt ein `GEN 1` je Segment. Zwei 65k-Segmente ergäben `max-context` 66.023 und damit einen 65k-Horizont, nicht 128K |
| KV | `q4lm32` für die drei Fenster-Arme, `fp16` und `int8` als Referenz/Latte |
| Fenster | `STRATA_KV_WINDOW=1 S=256 R=64`, **kein** `BATCH`: der Ring folgt dem Chunk |
| Prefill | `--prefill auto` → Chunk 6.144 (Engine-Deckel `kKvWinBatchMax`), Ringzeilen 256 + 6.144 = **6.400** (erwartete stderr-Zeile „ring sized for the prompt chunk (6144 cells)") |
| Logits | `--short-read 6144`: die letzten 6.144 Token laufen durch die Verify-Fenster und schreiben per-Position-Logits (Positionen 124.878…131.021, ein Dokument-Tail). `--topk 256` |
| Werkzeug | `quant_quality.py run` (neu: `--prefill N|auto`, Standard 512 = alte Läufe byte-identisch), `--allow-no-positions` (ein Segment mit weniger als der Hälfte der Positionen ist hier Absicht, nicht Fehler) |
| Metrik | `KL(fp16‖Arm)` je Position über fp16s top-256, gepaart je Position |

| Arm | `--kv` | Fenster | Zweck |
|---|---|---|---|
| `fp16` | `fp16` | — | Referenz der KL |
| `int8` | `int8` | — | die Latte |
| `win` | `q4lm32` | 256/64 | Fenster am 128K-Punkt (heutige Lesepolitik) |
| `tag` | `q4lm32` | 256/64 + `WINDOW_TAG=1` | der Arm |

## Fragen und Entscheidungsregeln (vor dem Lauf festgelegt)

### `T1` — trägt der Tag am 4K-Decode-Punkt?

Gepaart `tagB512 − winB512`, je Prompt, geclustert je Segment (df = 7, `t_crit` 2,365).

**„Tag trägt"** nur wenn alle vier Klauseln halten:

1. das 95-%-Intervall über die 8 Segmentwerte schließt die 0 aus,
2. der Mittelwert ist **negativ** (der Tag ist besser),
3. in **≥ 6/8** Segmenten ist der Segmentmittelwert negativ,
4. `|Mittelwert|` ≥ die einseitige Grenze bei Parität dieses Vergleichs (1,645 · SEM über die 8
   Segmentwerte, im Bericht ausgewiesen) — der Effekt liegt über der Auflösung des eigenen Paares.

Sonst: **„unter der Auflösung"**, wenn das Intervall die 0 einschließt *und* `|Mittelwert|` unter
dieser Grenze liegt; **„gegen den Tag"**, wenn das Intervall die 0 ausschließt und der Mittelwert
positiv ist. Dazu Top-1-Übereinstimmung mit `fp16` für beide Arme.

### `T2` — schließt der Tag die Lücke zu `int8`?

Gepaart `tagB512 − int8`, dieselbe Clusterung.

**„Lücke geschlossen"** nur wenn alle drei Klauseln halten:

1. das 95-%-Intervall schließt die 0 ein (kein nachweisbarer Abstand),
2. `|Mittelwert|` ≤ die einseitige Grenze bei Parität dieses Vergleichs,
3. Top-1-Übereinstimmung mit `fp16` liegt **höchstens einen Request** unter der von `int8`
   (48/48 → ≥ 47/48).

Sonst **„nicht geschlossen"**, mit Zahl und Richtung. Die Zahl, die zählt, ist der Mittelwert
gegen den von `int8` (0,00184 am 4K-Punkt).

### `T3` — 128K: trägt der Tag dort, und wie groß ist die Lücke zu `int8`?

Ein Segment, also **keine** Segment-Clusterung. Stattdessen eine **Moving-Block-Bootstrap** über
die 6.144 Positionen (Blöcke von 256 aufeinanderfolgenden Positionen = 24 Blöcke, 2.000
Resamples, fester Seed 20261103, Blöcke innerhalb des Segments gezogen): Positionen in einem
Dokument sind nicht unabhängig, ein Positions-Bootstrap wäre zu schmal. Zusätzlich als
Robustheit (nicht entscheidend) Blockweiten 512 und 1.024.

- **`T3a` — ist der Arm informativ?** Mittelwert `KL(fp16‖win) − KL(fp16‖fp16) = KL(fp16‖win)` >
  0 mit Bootstrap-95-%-Intervall ohne 0. Andernfalls hat der Ring den Prompt gehalten (kein
  Verdrängen), `T3` ist **ungültig** und wird als solche berichtet.
- **`T3b` — trägt der Tag am 128K-Punkt?** Gepaart `tag − win` je Position. **„Tag trägt"** nur
  wenn das Bootstrap-95-%-Intervall die 0 ausschließt, der Mittelwert negativ ist **und**
  `|Mittelwert| > 0,0013` nats (die am 4K-Decode-Punkt gemessene Auflösung eines Fenster-Paares,
  decode-4k — eine vor diesem Lauf feststehende Zahl, keine aus diesen Daten gewonnene).
- **`T3c` — wie groß ist die Lücke zu `int8`?** Gepaart `tag − int8` und `win − int8`.
  **„geschlossen"** wenn der Mittelwert von `tag − int8` ≤ 0 ist **und** die einseitige
  95-%-Bootstrap-Obergrenze unter **+0,001 nats** liegt (die K1-Grenze der Leiter, unverändert
  übernommen). Sonst „nicht geschlossen", mit beiden Zahlen.
- **`T3d` — Mechanik-Nullprobe.** `tagAuto` (4K, `--prefill auto`) muss `fp16` **bit für bit**
  reproduzieren: bei 4.056 Token hält der 6.400-Zeilen-Ring den ganzen Prompt, jeder Tag trifft,
  der Tag liest alles exakt. Weicht ein einziges Byte ab, ist der Schalter mehr als eine
  Lesepolitik und `T1`/`T3` sind ungültig.

### Güte-Gatter (vor der Auswertung, je Arm)

- Engine-SHA vor und nach dem Lauf `CD5700EC0793F83B…3D0394CA`.
- `DONE` ohne „nonfinite"; kein „do not fit"; kein `ERR`.
- 4K: `index.json` mit genau der registrierten `args`/`env`-Umgebung, 48 `first.*`-Dateien.
- 128K: TSV mit **6.144** Positionen, in allen vier Armen **dieselbe** Positionsmenge; `win`/`tag`
  mit der Ring-Zeile (6.144 Zellen), `tag` zusätzlich mit der `STRATA_KV_WINDOW_TAG=1`-Zeile.
- Ein GPU-Job zur Zeit, 30 s zwischen zwei Engine-Starts.

### Abweichungen und Grenzen, offen genannt

1. **Wiederholter Korpus.** Es gibt kein 128K-Qualitätsset; die Passage ist cwe-8k zweimal. Das
   Tail-Fenster (die letzten 6.144 Token) liegt vollständig im **zweiten** Durchlauf von s15 — der
   Text war also schon einmal im Kontext. Für die *gepaarte* Differenz (Tag gegen Fenster auf
   denselben Positionen) ist das unkritisch; als absolutes 128K-Qualitätsurteil ist es keins.
2. **Reuse der drei 4K-Arme** (Gate `D0` oben). Wenn `D0` hält, ist der Tagesunterschied
   nachweislich null.
3. **Prefill- vs. Decode-Selektion.** Der Tag-Lesepfad ist derselbe; die Auswahl stammt am 4K-Punkt
   aus dem Prompt-Pfad, am 128K-Punkt aus den Verify-Fenstern. Die 32K-Annahme „Prefill-Selektion
   ≈ Decode-Selektion" (S1) wird übernommen, nicht neu bemessen.
4. **`--short-read 6144`** ist ein Fenster, kein Voll-Prompt-Durchlauf: 124.878 Token laufen
   batched (keine Logits). Die 6.144 Positionen sind der Dokument-Tail, nicht der ganze Kontext.
5. **Kein Timing, kein Byte-A/B.** Der Arm kostet statisch 0 Bytes; ein Laufzeit-A/B ist nicht Teil
   dieses Laufs.
6. **`T3` entscheidet kein 5,0-bit/Zelle-Ziel** und öffnet `B1` nicht (per S1/E1-Lesung
   geschlossen).

### Kosten

Fünf 4K-Läufe (davon zwei neu à 48 Requests, einer à 8 Requests für `D0`) und vier 128K-Läufe
(je ~125k batched + 6.144 Fenster-Schritte). Ein GPU-Job zur Zeit; nach der decode-4k-Erfahrung
(7 Arme an einem Tag) einige Stunden, kein Engine-Bau.

## Reproduktion

```powershell
# D0: Reuse-Gate (fp16, nur die Länge 4056) — muss 2026-11-01-decode4k/fp16 byte-identisch treffen
python tools\quant_quality.py prompt --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids data\quant_quality\cwe-4k\set.ids.json --lens 4056 --kv fp16 `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\4k\d0-fp16

# T1/T2: der neue 4K-Arm (winB512/int8/fp16 kommen aus 2026-11-01-decode4k)
python tools\quant_quality.py prompt --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids data\quant_quality\cwe-4k\set.ids.json --lens 256,512,768,1000,2048,4056 --kv q4lm32 `
    --prefill 512 --env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 `
    --env STRATA_KV_WINDOW_R=64 --env STRATA_KV_WINDOW_BATCH=512 --env STRATA_KV_WINDOW_TAG=1 `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\4k\tagB512

# T3d: Mechanik-Nullprobe (auto, Ring hält den Prompt) — muss fp16 byte-identisch treffen
python tools\quant_quality.py prompt --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids data\quant_quality\cwe-4k\set.ids.json --lens 256,512,768,1000,2048,4056 --kv q4lm32 `
    --env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 --env STRATA_KV_WINDOW_R=64 `
    --env STRATA_KV_WINDOW_TAG=1 `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\4k\tagAuto

# T3: 128K teacher-forced, vier Arme
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\p128k.ids.json `
    --kv q4lm32 --prefill auto --short-read 6144 --topk 256 --allow-no-positions `
    --env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 --env STRATA_KV_WINDOW_R=64 `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\128k\win.tsv
#   tag zusätzlich mit --env STRATA_KV_WINDOW_TAG=1; fp16/int8 ohne Fenster-Umgebung

# Auswertung
python tools\kv_decode.py --ref ...\4k\fp16 --run ...\4k\int8 --run ...\4k\w256r64B512 `
    --run ...\4k\tagB512 --run ...\4k\tagAuto --base int8 --base w256r64B512
python tools\kv_k1.py --ref ...\128k\fp16.tsv --run ...\128k\win.tsv --run ...\128k\tag.tsv `
    --run ...\128k\int8.tsv --kl-topk 256 --block 256 --json ...\128k\k1.json
```

Rohdaten (TSVs, `first.*`, stderr, `index.json`) liegen außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\`; im Repo steht dieses Dokument und nach
dem Lauf der Ergebnisabschnitt darunter.

---

# Ergebnis

**Status: gemessen (2026-11-03).** Alle Güte-Gatter bestanden: Engine-SHA
`CD5700EC0793F83B…3D0394CA` vor jedem Lauf unverändert; jeder Arm `DONE` ohne „nonfinite", kein
„do not fit", kein `ERR`; die vier 4K-Arme mit 48 `first.*`-Dateien und der registrierten
`env`-Umgebung; die vier 128K-Arme mit **6.144** Positionen, derselben Positionsmenge
(124.877…131.020) in allen vieren, der Ring-Zeile („ring sized for the prompt chunk (6144 cells)", also
`B` = 6.144 und `S+B` = 6.400 Ringzeilen) in `win` und `tag` und der `STRATA_KV_WINDOW_TAG=1`-Zeile
nur in `tag`. Ein GPU-Job zur Zeit, 30 s zwischen zwei Starts.

Reuse-Gate `D0` hielt: der fp16-Kontrolllauf über die Länge 4056 reproduzierte die acht
`first.*`-Dumps vom 2026-11-01 **byte-identisch** (8/8, je 993.280 B).

## Der 4K-Decode-Punkt (T1, T2)

`KL(fp16‖Arm)` über das **volle Vokabular**, 48 Requests je Arm (8 Segmente × 6 Längen):

| Arm | `--kv` | Fenster | `--prefill` | mean KL | median | max | Top-1 = fp16 |
|---|---|---|---|---|---|---|---|
| `int8` | `int8` | aus | auto | **0,00184** | 0,00014 | 0,04381 | **48/48** |
| `winB512` | `q4lm32` | 256/64, `B=512` | 512 | 0,00434 | 0,00018 | 0,09059 | 47/48 |
| `tagB512` | `q4lm32` | dasselbe + `TAG=1` | 512 | **0,00235** | 0,00009 | 0,06422 | **48/48** |
| `tagAuto` | `q4lm32` | 256/64 + `TAG=1` | auto | 0,00134 | 0,00018 | 0,02467 | **48/48** |

Die Längenstruktur (Mittel über die 8 Segmente je Länge) zeigt, wo der Tag liest:

| Arm | 256 | 512 | 768 | 1000 | 2048 | 4056 |
|---|---|---|---|---|---|---|
| `int8` | 0,000234 | 0,000808 | 0,000927 | 0,000240 | 0,000546 | 0,008314 |
| `winB512` | 0,000302 | 0,003592 | 0,007416 | 0,000237 | 0,001481 | 0,012985 |
| `tagB512` | 0,000302 | 0,003056 | **0,000607** | 0,000271 | **0,000427** | **0,009449** |
| `tagAuto` | 0,000302 | 0,003056 | 0,000576 | 0,000307 | 0,000257 | **0,003548** |

Bei 256 sind `winB512` und `tagB512` **byte-identisch** (8/8): dort lesen beide jede Zelle exakt,
der Schalter hat nichts zu tun. Die 0,000302, die beide dort gegen `fp16` haben, sind der
Ring-gegen-Pool-Pfad — beiden Armen gemeinsam, also kein Verzerrer des gepaarten Vergleichs. Ab 768
trennt der Tag die beiden Arme deutlich (0,000607 gegen 0,007416), bei 4056 halbiert `tagAuto` den
Fehler des Fensters (0,003548 gegen 0,012985).

Gepaart, **je Segment** (df = 7, `t_crit` 2,365; positiv heißt „der Arm ist schlechter"; Grenze =
1,645 · SEM über die 8 Segmentwerte):

| Vergleich | Δ (Arm − Basis) | 95 % | `t` | Segmente schlechter | Grenze bei Parität |
|---|---|---|---|---|---|
| `tagB512 − winB512` | **−0,00198** | [−0,00458; +0,00061] | −1,81 | **1/8** | 0,00180 |
| `tagB512 − int8` | **+0,00051** | [−0,00068; +0,00170] | +1,01 | 4/8 | 0,00083 |
| `tagAuto − winB512` | −0,00299 | [−0,00682; +0,00084] | −1,85 | 0/8 | 0,00266 |
| `tagAuto − int8` | −0,00050 | [−0,00143; +0,00042] | −1,29 | 3/8 | 0,00064 |

**`T1` — der Tag trägt nicht, die Richtung zeigt zu ihm.** Drei der vier registrierten Klauseln
halten: der Mittelwert ist negativ (−0,00198), in 7 von 8 Segmenten ist der Tag besser, und
der Betrag des Mittelwerts liegt mit 0,00198 über der Paar-Grenze 0,00180. Klausel 1 fällt: das
95-%-Intervall [−0,00458; +0,00061] schließt die 0 **nicht** aus. Die registrierte Regel hat für
diesen Fall („Intervall enthält 0, Effekt über der Grenze") keine Kategorie, und sie wird nicht
nachträglich ergänzt. Berichtet wird also der Buchstabe der Regel — **nicht getragen** — mit der
Richtung: 4 von 6 Längen und 7 von 8 Segmenten zeigen zum Tag (256 ist ein Gleichstand, 1000 steht
mit +0,000033 minimal dagegen), und die Punktzahl 0,00198 sind 79 % des Abstands, den `winB512` zu
`int8` hat (0,00249). Der Tag nimmt dem Fensterarm also fast seinen ganzen Rückstand zu `int8` ab;
mit 8 Segmenten lässt sich aber nur zeigen, dass er es nicht schlechter macht.

**`T2` — die Lücke zu `int8` ist geschlossen.** Alle drei Klauseln halten: das Intervall
[−0,00068; +0,00170] schließt die 0 ein, |Mittel| 0,00051 ≤ 0,00083, Top-1 48/48 wie `int8`
(48/48). Der Mittelwert ist positiv, also **Parität, kein Gewinn** — und er liegt innerhalb der
Auflösung dieses Paares.

Zwei Einschränkungen, die zur Zahl gehören: der 512-Punkt wird von **einem** Prompt getragen
(s02 bei 512: 0,0166 im Tag-Arm, 0,0178 im Fenster-Arm, 0,0005 bei `int8`), deshalb sind die
Segmentmittel bei 512 kein Längeneffekt; und die Segment-Clusterung hat df = 7, die Längen innerhalb
eines Segments sind geschachtelt, nicht unabhängig.

**Einordnung: 4K ist ein Diagnose-Punkt, kein Produktpunkt.** Er ist billig (48 Requests in
Minuten), die Verteilung steht still, und ein Effekt, der hier nicht auftritt, tritt nirgends auf —
deshalb stand er in der Vorregistrierung. Der Einsatzpunkt der Maschine liegt bei 32K bis 256K
Kontext; dort entscheidet sich, ob der Tag etwas wert ist, und dort ist bisher nur 128K gemessen
(ungültig wäre die Verallgemeinerung von 4K auf 32K+). Die `T1`/`T2`-Zahlen bleiben, was sie sind:
ein Mechanik- und Paritätstest am kürzesten Punkt, nicht der Produktnachweis.

## Der 128K-Punkt (T3)

Ein Segment (`p128k`, 131.022 Token), ein `GEN 1`, per-Position-Logits aus den Verify-Fenstern,
6.144 Positionen je Arm, `KL(fp16‖Arm)` über `fp16`s top-256:

| Arm | mean KL | sd | Block-95 % (256) | einseitige 95-%-Grenze |
|---|---|---|---|---|
| `win` | **0,00211** | 0,0513 | [+0,00047; +0,00486] | +0,00458 |
| `tag` | **0,00212** | 0,0516 | [+0,00055; +0,00458] | +0,00413 |
| `int8` | 0,00085 | 0,0333 | [+0,00012; +0,00222] | +0,00218 |

Gepaart je Position (Δ = Arm − Basis, negativ = Arm besser), Moving-Block-Bootstrap über 24 Blöcke
à 256 Positionen, 2.000 Resamples, Seed 20261103:

| Vergleich | Δ | Block-95 % | einseitige Grenze |
|---|---|---|---|
| `tag − win` | **−0,00001** | [−0,00066; +0,00069] | +0,00053 |
| `win − int8` | +0,00126 | [+0,00033; +0,00265] | +0,00244 |
| `tag − int8` | **+0,00127** | [+0,00040; +0,00234] | +0,00214 |

**`T3a` — der Arm ist informativ.** `KL(fp16‖win)` = 0,00211 mit Block-95 % [+0,00047; +0,00486]
ohne 0: der Ring verdrängt, `win` ist nicht heimlich `fp16`. Die 128K-Frage ist damit gültig.

**`T3b` — der Tag trägt am 128K-Punkt nicht.** Δ = −0,00001, das 95-%-Intervall
[−0,00066; +0,00069] enthält die 0, und |Δ| liegt zwei Größenordnungen unter der vorab fixierten
Schwelle 0,0013 nats. Beide Klauseln fallen. Robust über die Blockweiten: 512 → −0,00001
[−0,00063; +0,00066]; 1.024 → −0,00001 [−0,00049; +0,00045]. Der Arm liest also nachweislich
anders (die Dumps sind nicht identisch) und ändert die Verteilung trotzdem nicht.

**`T3c` — die Lücke zu `int8` ist am 128K-Punkt nicht geschlossen.** `tag − int8` = +0,00127
(positiv: der Tag ist schlechter), einseitige 95-%-Grenze +0,00214, über der K1-Grenze +0,001.
`win − int8` = +0,00126 — der Tag verschiebt den Abstand um 0,00001. Praktisch nichts.

Zur Einordnung, **beschreibend und nicht entscheidungsrelevant**: der Tag ist an 5.407 der 6.144
Positionen besser (Median −1,3·10⁻⁵ nats) und an 737 schlechter, dort aber mit größerem Betrag
(p95 +1,9·10⁻⁵); das Mittel ist deshalb ≈ 0. Die exakte Lesung hilft an der Mehrheit der
Positionen ein wenig und schadet an einer Minderheit stärker.

## `T3d` — die registrierte Mechanik-Nullprobe ist gescheitert

Wie registriert muss `tagAuto` `fp16` **bit für bit** reproduzieren. Es tut es nicht: 0 von 48
Dumps identisch (mean KL 0,00134, Top-1 48/48).

Der Grund ist die Prämisse, nicht der Mechanismus: ein `q4lm32`-Arm kann am 4K-Punkt `fp16` nie
byte-gleich treffen, weil schon die KV des Prompts quantisiert ist und der Ring ein anderer
Kernel-Aufruf ist. Sichtbar wird dieser Rest bei 256 Token, wo **beide** Fenster-Arme jede Zelle
exakt lesen: `winB512` und `tagB512` sind dort byte-identisch (8/8), und beide liegen 0,000302 von
`fp16` entfernt. Dieser Betrag gehört zum Ring-Pfad, nicht zum Schalter — er ist beiden Armen
gemeinsam und verzerrt den gepaarten Vergleich nicht.

Damit ist die registrierte Klausel **gescheitert und nicht erfüllbar**. Nach ihrem Buchstaben
wären `T1` und `T3` ungültig; diese Lesung wird hier nicht angewandt, weil ihre Voraussetzung (ein
`q4lm32`-Lauf gleich `fp16`) konstruktiv unmöglich ist und jeder `q4lm32`-Arm daran scheitern
müsste. Der Mechanismus, den die Probe prüfen sollte, ist durch die 256-Token-Identität belegt:
wo die Lesemengen gleich sind, sind die Ausgaben gleich. Das ist **schwächer** als die registrierte
Probe — sie zeigt, dass der Schalter bei gleicher Lesemenge nichts anderes tut, nicht dass er bei
jeder Länge nur die Lesemenge ändert. Es ist eine berichtete Abweichung, keine stillschweigende
Regeländerung.

## Abweichungen vom registrierten Plan

1. **`T3`-Geometrie: der Tail erreicht die Fenster nur über eine Schnittkante.** Registriert war
   `--short-read 6144` allein. Der erste `fp16`-Versuch tat genau das und schrieb in 979,6 s
   **0 Positionen**: der Prompt läuft als ein batched Chunk, und ohne Pin fällt kein Schnitt in die
   letzten 6.144 Token (`cuts = {reread_to, root_at, message_at, turn_at, pin_at, n−1}`). Gelöst mit
   `pin=124877` in der `GEN`-Zeile plus `--prompt-cache 3`: [0, 124.877) batched, der Tail durch die
   Verify-Fenster. Der 0-Positionen-Lauf bleibt als Beleg in den Rohdaten.
2. **Positionsbereich.** Registriert: 124.878…131.021. Gemessen: 124.877…131.020 (6.144
   Positionen; `pin` ist 0-basiert, die letzte geschriebene Position ist die vor dem letzten Token).
3. **Reuse der drei 4K-Arme** (`int8`, `winB512`, `fp16`), gedeckt durch `D0` (8/8 byte-identisch).
4. **`T3d` wie registriert gescheitert**, mit Ersatzbeleg statt nachträglicher Regeländerung.
5. **Die `T1`-Regel hat keine Kategorie für „Intervall enthält 0, |Mittel| über der Grenze"** —
   berichtet als „nicht getragen" mit Richtung, ohne neue Klausel.
6. **Der 512-Punkt des 4K-Satzes wird von einem Prompt dominiert** (s02: 0,0166 im Tag-Arm, 0,0178
   im Fenster-Arm, 0,0005 bei `int8`); die Segmentmittel bei 512 sind kein Längeneffekt.
7. **`--block` im Werkzeug.** `tools/kv_k1.py` hat die registrierte Moving-Block-Bootstrap erst mit
   diesem Lauf bekommen (`--block N`, Standard 0 = aus). Ohne `--block` druckt es dieselben Zahlen
   wie vorher (geprüft an `win`/`int8`: 0,00211 bzw. 0,00085, Positionsebene).

## Was das heißt

- **Der Einsatzpunkt ist 32K bis 256K, nicht 4K.** Gemessen ist dort bisher der 128K-Punkt, und
  dort ist der Effekt null (−0,00001 nats, Intervall ±0,0007). Die Abdeckung, die der Tag
  zusätzlich exakt liest, **fällt mit dem Horizont** (E1/S1 über die S-Leiter: 100 → 94,6 → 80,0 →
  48,6 %); 32K und 64K sind deshalb die einzigen Längen, an denen der Tag noch mehr zeigen könnte
  als bei 128K, und 256K kann 128K nicht übertreffen — der Ring hält dort 6.400 von 262.144 Zellen
  (2,4 % des Kontexts) statt 4,9 % bei 128K. **32K, 64K und 256K sind ungemessen**; das ist der
  nächste Schritt, nicht eine Nebenfrage.
- **Der 4K-Punkt ist ein Diagnose-Punkt.** Er ist billig (48 Requests in Minuten), die Verteilung
  ist eingefroren, und ein Unterschied, der hier nicht auftritt, tritt nirgends auf. Er zeigt, dass
  der Mechanismus wirkt — Top-1 48/48 statt 47/48, `int8`-Parität, 7/8 Segmente zugunsten des Tags
  —, und er zeigt **nicht**, dass dieser Gewinn bei 32K+ bleibt. Die 0,00198 sind kein
  Produktversprechen.
- **Für die Leiter ändert das nichts:** der Hebel ist gratis, am 128K-Punkt wirkungslos, und ein
  5,0-bit/Zelle-Ziel wird davon nicht erreicht; `B1` bleibt geschlossen. Ein Auslieferungs-Default
  ist der Tag auf dieser Evidenz **nicht** — dafür fehlt der Nachweis im Einsatzbereich.

## Reproduktion (die tatsächlich gelaufenen Befehle)

```powershell
# D0 zuerst (Gate), dann die beiden neuen 4K-Arme, dann die vier 128K-Arme — ein GPU-Job zur Zeit,
# 30 s zwischen zwei Starts. Die drei Reuse-Arme des 4K-Punkts kommen aus 2026-11-01-decode4k.
python tools\quant_quality.py prompt --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids data\quant_quality\cwe-4k\set.ids.json --lens 4056 --kv fp16 `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\4k\d0-fp16

python tools\quant_quality.py prompt --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids data\quant_quality\cwe-4k\set.ids.json --lens 256,512,768,1000,2048,4056 --kv q4lm32 `
    --prefill 512 --env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 `
    --env STRATA_KV_WINDOW_R=64 --env STRATA_KV_WINDOW_BATCH=512 --env STRATA_KV_WINDOW_TAG=1 `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\4k\tagB512

python tools\quant_quality.py prompt --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids data\quant_quality\cwe-4k\set.ids.json --lens 256,512,768,1000,2048,4056 --kv q4lm32 `
    --env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 --env STRATA_KV_WINDOW_R=64 `
    --env STRATA_KV_WINDOW_TAG=1 `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\4k\tagAuto

# 128K: je Arm einmal, --pin/--prompt-cache tragen die Schnittkante (Abweichung 1)
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\p128k.ids.json `
    --kv <fp16|int8|q4lm32> --prefill auto --short-read 6144 --topk 256 --allow-no-positions `
    --pin 124877 --prompt-cache 3 `
    [--env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 --env STRATA_KV_WINDOW_R=64 `
     [--env STRATA_KV_WINDOW_TAG=1]] `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\128k\<arm>.tsv

# Auswertung
python tools\kv_decode.py --ref ...\2026-11-01-decode4k\fp16 `
    --run ...\2026-11-01-decode4k\int8 --run ...\2026-11-01-decode4k\w256r64B512 `
    --run ...\2026-11-03-tag-arm\4k\tagB512 --run ...\2026-11-03-tag-arm\4k\tagAuto `
    --base int8 --base w256r64B512
python tools\kv_k1.py --ref ...\128k\fp16.tsv --run ...\128k\win.tsv --run ...\128k\tag.tsv `
    --run ...\128k\int8.tsv --kl-topk 256 --block 256 --json ...\128k\k1.json
#   Robustheit: dasselbe mit --block 512 und --block 1024 (nicht entscheidend)
```
