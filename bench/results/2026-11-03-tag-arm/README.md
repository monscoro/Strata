# T1/T2/T3 — der Tag-Arm: Lesepolitik am 4K-Decode-Punkt und teacher-forced am 128K-Punkt

**Status: vorregistriert (2026-11-03), vor dem ersten Arm-Lauf.** Der erste Arm-Lauf (auch der
Reuse-Gate-Lauf `D0`) darf erst nach dem Schreiben dieses Dokuments starten. Katalogzeile:
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
