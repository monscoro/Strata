# Decode bei 4K: der Kandidat am echten Kontext

**Status: gemessen (2026-11-01), sieben Arme, Ergebnis unten. Vorregistriert vor dem ersten
Arm-Lauf.** Anschluss an
`bench/results/2026-10-28-decode-probe/` (dort ausdrücklich offen gelassen: „Ein Lauf mit
4K-Prompts und einem Decode-Schritt wäre der nächste Arm — und ein anderer, weil er die
Ring-Verdrängung mitbringt"). Katalogzeile: `docs/paper/Ansatz_Improvements.md`, B-Term B4.

## Warum diese Messung

Die Sonde vom 2026-10-28 maß mit Prompts von höchstens 1.000 Token. Am Dump-Punkt las der Arm nur
`R + S` exakte Zellen (siehe unten) und den Rest quantisiert — aber aus einem Kontext, in dem der
Ring noch fast alles exakt hielt und die ~700 quantisierten Zellen den kleineren Teil des Kontexts
ausmachten. Beim echten 4K-Lesevorgang tragen ~3.700 quantisierte Zellen den Großteil des Kontexts.
Genau dort — am Antwortpunkt der cwe-4k-Segmente — liegt die Frage, die der teacher-forced-Lauf
nicht stellen kann: er mittelt KL über alle Prefill-Positionen; hier ist die Verteilung des
Decode-Pfades **an einer Position** gefragt, wenn fast der ganze Kontext aus dem Q4-Bulk kommt.

Anschluss an die vorhandenen Zahlen: der teacher-forced-Mittelwert für den Kandidaten (`w256r64`)
ist 0,01689 gegen `int8`s 0,01688 (top-64, cwe-4k) — Parität im Prefill. Die frühen Buckets
(0–1.023) lagen bei 0,00348/0,00413 und die Sonde fiel dort auf 0,00225. Die späten Buckets
(ab Position 2.048) liegen bei 0,0195–0,0396; das ist die Gegend, in die der Dump-Punkt an
Position ~4.056 fallen wird.

## Was die Engine am Decode-Punkt liest (statische Analyse, vorab registriert)

Aus `include/strata/kernels/qsa_decode_attn.hpp` (`qsa_win_slot`, `win_all = 0`, der publizierte
Horizont — aus dem Code gelesen, nicht gemessen):

- Zelle `c < R`: exakt (Sinks, zellenadressiert, werden nie überschrieben).
- Zelle `c ≥ n − S` (die letzten `S` Zellen vor Horizont `n`): exakt, solange der Ring den
  richtigen Tag führt.
- **alle anderen Zellen lesen quantisiert** — auch dann, wenn der Ring sie noch exakt hält.
  `B` (die Ringzeilen jenseits von `S`, für die Append-Batch) kommt im Lesepredikat **nicht** vor.

Daraus, vorab:

1. Am Dump-Punkt liest **jeder Fensterarm** `R + S` exakte Zellen, unabhängig von `B` und
   unabhängig davon, ob der Ring verdrängt hat. Die S-Achse (`Q2`) und die Sink-Achse (`Q3`) sind
   am Decode-Punkt damit **nicht** an Verdrängung geknüpft — die Aussage im Sonden-README
   („wird erst sichtbar, wenn Zellen tatsächlich aus dem Ring fallen") trifft nach dem Code so
   nicht zu; die Sonde scheiterte schlicht an der Auflösung. Dieser Lauf prüft es bei 4K.
2. `Q5` unten: mit `B = 512` fest (Ring unter echter Verdrängung) ändert sich an der exakten
   Lesemenge am Dump-Punkt **nichts**. Was sich ändert, ist der Prefill-Pfad selbst: unter
   `--prefill auto` (Chunk 4.056) liest jede Abfrage während des Prefills über ihren
   Chunkübergangspunkt hinaus exakt (Spanne `[pos0 − S, pos0 + B)` der Append-Batch deckt den
   ganzen Chunk ab), unter `--prefill 512` nur bis zum eigenen Chunkende. Die KV-Werte der späteren
   Positionen unterscheiden sich also mathematisch, nicht nur im Runden. Eine gemessene Lücke
   zwischen beiden Armen ist darum **kein** Decode-Leseunterschied, sondern ein
   Prefill-Unterschied — das gehört zur Auswertung von `Q5` dazu (die eigentliche Frage des Punkts:
   hängt die exakte Lesemenge am Decode-Punkt von `B` ab, oder nur vom Horizont?).
3. Verdrahtung: `qsa_set_kv_win_batch` hebt `B` nur an, wenn `STRATA_KV_WINDOW_BATCH` **nicht**
   gesetzt ist (`src/core/layer.cpp`); der Ring des Users wird nicht angehoben, und ein Chunk, den der
   Ring nicht trägt, wird von `kv_append_win` laut abgelehnt
   (`src/program/generate.cpp`, Kommentar zur gemessenen Ablehnung „do not fit the … recent
   ring", Exit 1). Ein `B = 512`-Arm bei 4K braucht darum `--prefill 512` (der Chunk passt
   in `S + B = 768`).

## Instrument

`tools/quant_quality.py prompt` mit dem neuen `--lens`-Flag (Standard bleibt 256/512/768/1000 —
die Sonden-Verzeichnisse sind von der Änderung nicht berührt):

```
--lens 256,512,768,1000,2048,4056
```

48 Requests = 8 Segmente × 6 verschachtelte Längen (256 ⊂ 512 ⊂ 768 ⊂ 1000 ⊂ 2048 ⊂ 4056, die
Sondenform, um zwei Stufen nach oben verlängert). 4056 sitzt am Antwortpunkt (`answer_at`
4.054–4.064) und bleibt für alle Segmente gültig (das kürzeste Segment hat 4.088 Token) — deshalb
4056 und nicht 4096. Wie in der Sonde: kein Prompt-Cache, `--prefill auto`, ein Decode-Schritt,
`STRATA_DUMP_FIRST_LOGITS` (float32, ganzes Vokabular). Auswertung mit `tools/kv_decode.py`:
`KL(fp16‖Arm)` über das ganze Vokabular, gepaart je Prompt, geclustert je Segment (df = 7,
`t_crit` 2,365), einseitige Grenze bei Parität.

## Arme

| Arm | `--kv` | Fenster-Umgebung | `--prefill` | Ringzeilen je Tensor | Zweck |
|---|---|---|---|---|---|
| `fp16` | `fp16` | — | auto | — | Referenz |
| `int8` | `int8` | — | auto | — | Qualitäts-Default, die Latte |
| `w256r64` | `q4lm32` | `WINDOW=1 S=256 R=64` | auto | `R+S+B` (B auf den Auto-Chunk gehoben) | der Kandidat |
| `w256r64B512` | `q4lm32` | `WINDOW=1 S=256 R=64 BATCH=512` | **512** | `64+256+512` = 832 (Ring verdrängt) | `Q5`: die B-Frage |
| `w64r64` | `q4lm32` | `WINDOW=1 S=64 R=64` | auto | wie `w256r64` | `Q2`: die S-Achse |
| `w256r0` | `q4lm32` | `WINDOW=1 S=256 R=0` | auto | wie `w256r64` | `Q3`: die Sink-Achse |
| `w8v` | `q4lm8v` | `WINDOW=1 S=256 R=64` | auto | wie `w256r64` | `Q4`: die 16/8-Frage am realen Maßstab |

Alle Arme sind bei denselben Bytes am 4K-Messpunkt vergleichbar wie in der Sonde:
`q4lm32`/`q4lm8v` mit Fenster liegen bei 7,39 bit/value gegen `int8`s 8,25 (gemessen in
`2026-10-28-ring-geometry`; hier nicht neu gemessen). Der `B512`-Arm verkürzt den Ring selbst
zusätzlich auf 832 Zeilen und ist damit der Arm, der dem 128K-Produktpunkt (Ring unter dauernder
Verdrängung) im Leseverhalten am nächsten liegt.

## Fragen und Entscheidungsregeln (vor dem Lauf festgelegt)

Form: gepaarte Differenz je Prompt, geclustert je Segment (`t` mit df = 7 — die 48 Requests sind
8 Segmente × 6 geschachtelte Längen, keine 48 unabhängigen Einheiten); positiv heißt „der Arm ist
schlechter"; dazu die Top-1-Übereinstimmung mit `fp16` (48 Requests je Arm).

1. **`Q1` — steht der Kandidat am 4K-Decode-Punkt auf `int8`?** Gepaart `w256r64 − int8`.
   Urteil „Parität" nur, wenn das 95-%-Intervall die 0 einschließt **oder** der Mittelwert ≤ 0 ist
   **und** die Top-1-Übereinstimmung mit `fp16` höchstens einen Request unter der von `int8` liegt.
   Liegt der Punktwert über der einseitigen Grenze bei Parität und in ≥ 6/8 Segmenten auf der
   schlechteren Seite, steht im Urteil ausdrücklich „trennbar, schlechter" — nicht die
   Sonden-Auflösung von 0,0013, sondern die dieses Paares hier.
2. **`Q2` — kostet die S-Achse am Decode-Punkt?** Gepaart `w64r64 − w256r64`. Urteil „S frei":
   Intervall enthält 0 **oder** Effekt unter der Auflösung. Urteil „S trägt": Intervall schließt
   0 aus **und** ≥ 6/8 Segmente stimmen zu (Codeleseseite erwartet „schlechter": beim Decode liest
   `S = 64` nur 128 exakte Zellen statt 320).
3. **`Q3` — tragen die Sinks am Decode-Punkt?** Gepaart `w256r0 − w256r64`, gleiche Regel wie
   `Q2` (Codeleseseite erwartet „schlechter"; das Prefill-Ergebnis enthielt beim `R = 0` +0,00394,
   8/8 Segmente).
4. **`Q4` — die 16/8-Frage am realen Maßstab (Wiederholung).** Gepaart `w8v − w256r64`, **deskriptiv**:
   die Sonde reproduzierte die Prefill-Reihenfolge nicht (−0,00074, nicht signifikant); bei 4K
   dominieren die ~3.700 quantisierten Zellen. Kein Produktschluss aus diesem Lauf — ob 16/16 am
   Punkt bleibt, ist den teacher-forced-Standard-Sicherungen vorbehalten; dieser Run liefert nur
   die Richtung (≥ 6/8 Segmente mit der Prefill-Ordnung 16/16, mit der Sonden-Ordnung 16/8 an
   dieser Stelle, oder unentschieden).
5. **`Q5` — die B-Frage am echten Antwortpunkt.** Gepaart `w256r64B512 − w256r64`. Vorhersehbar
   (statische Analyse oben): gleicher exakter Lesesatz, andere Prefill-Exaktheit. Urteil „der
   Ring selbst ist am Decode-Punkt unsichtbar", wenn |Δ| unter der Auflösung bleibt; jede
   Messlücke über der Auflösung wird als **Prefill-**Unterschied der beiden Chunk-Einstellungen
   gelesen, nicht als Decode-Leseunterschied (Kommentar dazu zwingend).
6. **`Q0` — Güte-Gatter vor jedem Urteil** (Eigenschaften des Laufs, nicht Messwerte): je Arm
   ist die `index.json`-Umgebung exakt die der Tabelle, zählen 48 Requests, kein „nonfinite",
   kein „do not fit" in stderr, keine leere `first.*`-Menge. Schlägt ein Gatter an, wird der
   Arm einmal wiederholt; zusammen mit dem Wiedergänger im Urteil der sonst vermerkten Paare
   erlauben sie Auskunft über die Verwerfung als Teil der Eine-Lauf-Sicherheit (wird vermerkt).

## Auflösung, vorab notiert

Die KLs an dieser Stelle liegen vermutlich eine Größenordnung über den Sonden-KLs (die frühen
Buckets des teacher-forced-Laufs liegen bei 0,003–0,004, die späten bei 0,02–0,04); die gepaarte
Streuung wächst vermutlich mit. Aus der Sonde als Richtwert: Trennbarkeit um 0,001–0,003 bei viel
kleineren KLs — hier also eine Größenordnung erwartet. Für jeden Paarvergleich wird die Auflösung
**nach** dem Lauf aus der beobachteten gepaarten Segmentstreuung ausgerechnet (einseitige Grenze
bei Parität, 1,645 · SEM über die 8 Segmentwerte) und im Ergebnis angegeben — vorab ist jede Zahl
dafür geraten.

## Aufbau

| Punkt | Wert |
|---|---|
| Engine | 0.1.47 Debug-Build `build-hip-port\strata.exe`, SHA-256 `CD5700EC…3D0394CA`, ROCm 7.1 (vor dem ersten Arm gemessen; der Build unterscheidet sich von der SHA `F9058D56…860F20` der Sonden-Läufe — der Vergleich zu deren Absolutwerten ist mit Vorsicht zu lesen, die gepaarten Vergleiche sind intern) |
| Modell | swift-1.5-iq3_xxs, `bench/configs/cwe4k-swift-iq3_xxs.json` (`env` bleibt leer; Fenster-Umgebung je Arm) |
| Prompts | cwe-4k, 8 Segmente × 6 Längen (256/512/768/1000/2048/4056) = 48 Requests je Arm |
| Pfad | batched Prompt-Pfad (kein `--short-read`), ein Decode-Schritt je Request |
| Metrik | `KL(fp16‖Arm)` über das ganze Vokabular, gepaart je Prompt, geclustert je Segment (df = 7) |

## Läufe

```sh
LEN=256,512,768,1000,2048,4056
tools/quant_quality.py prompt --config bench/configs/cwe4k-swift-iq3_xxs.json \
  --ids data/quant_quality/cwe-4k/set.ids.json --lens $LEN --kv <fmt> [--env K=V ...] \
  --out Strata-data/bench/2026-11-01-decode4k/<arm>
# nur der B512-Arm zusätzlich mit:
#   --prefill 512 --env STRATA_KV_WINDOW_BATCH=512
```

Sieben Arme, nacheinander, ein GPU-Job zur Zeit, 30 s zwischen zwei Engine-Starts (G3-Ordnung).
Reihenfolge: `fp16`, `int8`, `w256r64`, `w256r64B512`, `w64r64`, `w256r0`, `w8v`.

## Ergebnis

Gelaufen am 2026-11-01, sieben Arme, je 48 Requests, alle Güte-Gatter (`Q0`) bestanden: die
`index.json`-Umgebung exakt wie in der Tabelle, 48 `first.*`-Dateien je Arm, kein „nonfinite",
kein „do not fit"; der `B512`-Arm mit `--prefill 512` und ohne Ring-Anhebung (stderr: keine
Zeile „KV window ring sized"), die vier Auto-Arme mit „ring sized for the prompt chunk (6144)".
`KL(fp16‖Arm)` über das **volle Vokabular**, je Prompt:

| Arm | `--kv` | Fenster | mean KL | median | max | Top-1 = fp16 |
|---|---|---|---|---|---|---|
| `int8` | `int8` | aus | **0,00184** | 0,00014 | 0,04381 | **48/48** |
| `w256r64` | `q4lm32` | 256/64 | 0,00429 | 0,00018 | 0,11601 | 47/48 |
| `w256r64B512` | `q4lm32` | 256/64, `B=512`, `--prefill 512` | 0,00434 | 0,00018 | 0,09059 | 47/48 |
| `w64r64` | `q4lm32` | 64/64 | 0,00419 | 0,00029 | 0,07449 | 47/48 |
| `w256r0` | `q4lm32` | 256/0 | 0,00482 | 0,00018 | 0,11097 | 47/48 |
| `w8v` | `q4lm8v` | 256/64 | 0,00423 | 0,00018 | 0,13922 | 47/48 |

Gepaart, **je Segment** (df = 7, `t_crit` 2,365; positiv heißt „der Arm ist schlechter"; Grenze =
1,645 · SEM über die 8 Segmentwerte):

| Vergleich | Δ (Arm − Basis) | 95 % | `t` | Segmente schlechter | einseitige Grenze bei Parität |
|---|---|---|---|---|---|
| `w256r64` − `int8` | +0,00244 | [−0,00123, +0,00612] | +1,57 | 6/8 | 0,0026 |
| `w256r64B512` − `int8` | +0,00249 | [−0,00054, +0,00552] | +1,95 | 7/8 | 0,0021 |
| `w256r64B512` − `w256r64` | +0,00005 | [−0,00188, +0,00197] | +0,06 | 4/8 | 0,0013 |
| `w64r64` − `w256r64` | −0,00009 | [−0,00398, +0,00379] | −0,06 | 5/8 | 0,0027 |
| `w256r0` − `w256r64` | +0,00054 | [−0,00065, +0,00172] | +1,07 | 5/8 | 0,0008 |
| `w8v` − `w256r64` | −0,00006 | [−0,00191, +0,00178] | −0,08 | 3/8 | 0,0013 |

### `Q1` — steht der Kandidat am 4K-Decode-Punkt auf `int8`?

**Nach der vorregistrierten Regel: ja, Parität.** Das Intervall schließt die 0 ein (+0,00244
[−0,00123, +0,00612]), der Punktwert liegt **unter** der einseitigen Grenze von 0,0026, und die
Top-1-Übereinstimmung liegt mit 47/48 genau einen Request unter `int8`s 48/48 — beide Klauseln
der Regel sind erfüllt. Anders als in der Sonde (dort lautete die Netto-Aussage „nicht
unterscheidbar, mit Neigung zu schlechter", weil der Punktwert über der Grenze lag) trägt diesmal
auch die Richtung nicht über die Grenze hinaus.

Die per-Längen-Struktur gehört zur Lesung dazu (mean KL je verschachtelter Länge, darunter die
gepaarte Differenz dieser Länge, 8 Segmente je Zelle):

| Länge | `int8` | `w256r64` | gepaart (Δ) | Segment-sd |
|---|---|---|---|---|
| 256 | 0,0002 | 0,0003 | +0,00007 | 0,0002 |
| 512 | 0,0008 | 0,0036 | +0,00278 | 0,0059 |
| 768 | 0,0009 | 0,0042 | +0,00331 | 0,0079 |
| 1.000 | 0,0002 | 0,0009 | +0,00065 | 0,0014 |
| 2.048 | 0,0005 | 0,0005 | −0,00009 | 0,0003 |
| 4.056 | 0,0083 | 0,0163 | +0,00795 | 0,0262 |

Am Antwortpunkt (4.056) lesen beide Arme fast den ganzen Kontext quantisiert, und die KLs liegen
dort tatsächlich in der Gegend der späten teacher-forced-Buckets: `w256r64` 0,0163 gegen den
0,01689-Volltextwert aus `2026-10-28-ring-geometry`, `int8` 0,0083. Die gepaarte Differenz dort
ist nominell **+0,008** bei Segment-sd 0,026 — `t` +0,86, weit von trennbar entfernt. Das heißt
nicht „Kandidat gleich gut am Antwortpunkt", sondern: **die Antwortpunkt-Stelle allein löst mit
8 Segmenten keine 0,008 nats auf.** Die Sonde und dieser Lauf messen beide im frühen Bereich; wer
den Kandidaten am echten 4K-Decode verteidigen will, braucht mehr Segmente oder echte Generationen.

### `Q2` — kostet die S-Achse am Decode-Punkt?

**Nicht entscheidbar — und die codegeleitete Erwartung bestätigt sich nicht.** −0,00009
[−0,00398, +0,00379], 5/8 Segmente. Nach der statischen Analyse liest `S = 64` am Decode-Punkt nur
128 exakte Zellen statt 320, unabhängig von der Ring-Verdrängung; gemessen: kein Unterschied.
Die Aussage im Sonden-README stimmt nach dem Code nicht so, wie sie formuliert ist — der
Unterschied zwischen den `S`-Varianten hängt nicht an der Verdrängung, er ist an dieser Stelle
einfach nicht da.

### `Q3` — tragen die Sinks am Decode-Punkt?

**Nicht entscheidbar.** +0,00054 [−0,00065, +0,00172], 5/8 Segmente, Grenze 0,0008. Die Richtung
(ohne Sinks nominell schlechter) stimmt mit dem Prefill-Befund (+0,00394, 8/8) überein, aber am
Decode-Antwortpunkt bleibt sie unter der Auflösung.

### `Q4` — die 16/8-Frage am realen Maßstab (Wiederholung)

**Unentschieden.** `w8v` − `w256r64` = −0,00006, 3/8 Segmente. Weder die Prefill-Reihenfolge
(16/16 besser, gesichert) noch die Nominalordnung der Sonde (16/8 nominell besser) reproduziert
sich: an dieser Decode-Stelle sind die beiden bei denselben Bytes gleich. Der Produktpunkt
(`w256r64`) bleibt, wie die teacher-forced-Sicherungen ihn stehen haben.

### `Q5` — die B-Frage am echten Antwortpunkt

**Der Ring selbst ist am Decode-Punkt unsichtbar.** `w256r64B512` − `w256r64` = +0,00005
[−0,00188, +0,00197], 4/8 Segmente, Grenze 0,0013. Der Auto-Arm hielt am 4K-Punkt einen
6.400-Zeilen-recent-Ring (`B` auf 6.144 gehoben, 12,6 MiB pro QSA-Layer, stderr-Zeile) und las
dabei den **ganzen Prompt exakt**; der `B512`-Arm verdrängte in einem 832-Zeilen-Ring mit
`--prefill 512` — und die Decode-Verteilung am Antwortpunkt unterscheidet sich trotzdem um
0,00005 nats.

Damit ist der statisch abgeleitete Befund am echten Lauf bestätigt, und er ist der Grundbaustein
für die Byte-Seite: **die exakte Lesemenge des Decode-Pfades hängt vom Horizont ab (`R` Sinks +
`S` Zellen davor), nicht von `B`.** Der Ring braucht `B` nur, um eine ganze Append-Batch zu
fassen; am Decode-Punkt darf er Zellen in den quantisierten Bulk entlassen, ohne dass sich die
Verteilung messbar ändert. Für das korrigierte Byte-Modell (`R + S + B`) heißt das: `B` muss nur
den Prefill-Chunk decken, nicht den Kontext — der Produktpunkt mit `B = 6.144` reserviert am
Decode mehr Zeilen, als gelesen werden. Was das zur Laufzeit- und Byte-A/B am 128K-Produktpunkt
wird (Katalogzeile `B4`), steht noch aus.

### Der gemeinsame Top-1-Flip

Alle vier `q4lm`-Arme (drei `q4lm32`-Geometrien plus `w8v`) verlieren **denselben** Request:
`s01` bei Länge 768. `int8` verliert keinen. Die vier teilen dasselbe Bulk-Quant (dort liegt der
Unterschied zu `int8`); die Ring-Geometrie ist daran unbeteiligt. Ein Request von 48 — ein
Hinweis, keine Urteilsgrundlage.

### Auflösung, aus der beobachteten Streuung

Einseitige Grenzen bei Parität über die Segmentwerte: 0,0008–0,0027 je nach Paar (Tabelle oben).
Die vorab notierte Erwartung war „Trennbarkeit bei ~0,005, Versagen bei ~0,002"; gemessen:
0,0008–0,0033 über die Paare. Die KLs am Antwortpunkt sind eine Größenordnung über den
Sonden-KLs, die gepaarte Auflösung ist es nicht — die Segmentstreuung wächst langsamer als die
KLs selbst.

### Was dieser Lauf nicht sagt

- `Q5` misst nur den Decode-Punkt an einer Position: der Prefill-Unterschied der beiden
  Chunk-Einstellungen (ganz-Prompt-exakt gegen verdrängend) ist in den Dump-Wert zwar
  eingerechnet, aber nicht als eigene Größe lesbar.
- Die 128K-Fragen (S/R-Aufteilung, `B` bei echten 6.144er-Chunks, Laufzeit- und Byte-A/B gegen
  `int8` am Produktpunkt) haben hier kein Objekt; dafür bleibt es bei
  `2026-10-28-ring-geometry`, `cwe-128k` und der Katalogzeile `B4`.
- Echte Generationen (nicht nur der erste Decode-Schritt) sind nicht gemessen.
