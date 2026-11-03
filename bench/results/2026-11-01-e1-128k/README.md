# E1 am 128K-Punkt: wo sitzt die Indexer-Selektion, wenn der Ring klein gegen den Kontext ist?

**Status: gemessen (2026-11-01), ein Lauf, Ergebnis unten. Vorregistriert vor dem ersten
Arm-Lauf.** Der erste Arm-Lauf darf erst nach dem
Schreiben dieses Dokuments starten. Katalogzeilen: `docs/paper/Ansatz_Improvements.md`, E1 (die
128K-Hälfte der Messdecke) und die Leiterzeile S1/WP-3a. Anschluss an
`bench/results/2026-10-25-s1-selstats/` — dort steht der 32K-Lauf mit denselben Zählern; seine
Abschlussnotizen sagen ausdrücklich: „32K, nicht 128K … Ob die 70 % bei 128K halten, ist
ungemessen; die Richtung des `outside`-Anteils ist offen." Genau diese Lücke ist der Lauf hier.

## Warum

S1 (32K, `n_kv` > 16.384) zählte für 12.601.344 gewählte Zellen: exakt (R+S) 11,7 %, bezahlt und
quantisiert (B) 59,2 %, außerhalb 29,1 %; die Tag-Lesung (win_all) würde 70,2 % exakt machen;
Recall der Sinks 28,9 % (fallend mit dem Horizont), Recall von S 86,2 % (über die Horizonte
stabil). Die Anteile hängen an der Länge: der exakte geometrische Satz (R=64 Sinks + die
letzten S=256 Zellen) ist eine feste Zellenzahl über einem wachsenden Kontext, der die Selektion
breit streut — der `outside`-Anteil kann nur steigen. Das Byte-Rezept hängt an denselben Anteilen:
der Ring costet `(R+S+B)·2.048 B` je QSA-Layer, und die heutige Geometrie liest davon
am Decode-Punkt nur `R+S`. Diese Messung liefert die 128K-Stütze der B-Terms (B1, B4-Byte-Seite),
kein Qualitätsurteil.

## Instrument (unverändert gegen 32K)

`STRATA_KV_SELSTATS` in `src/prefill/prefill.cpp` — eine Zeile je Stichprobe und QSA-Layer direkt
hinter `qsa_block_topk`:

```
layer pos n_kv width blocks in_r in_s in_b outside d2 d4 d8 d16 d32 older tag_s tag_b tag_out
min_sel max_unsel s_min s_max
```

Die Fächer folgen `qsa_win_slot` (`include/strata/kernels/qsa_decode_attn.hpp`): `in_r` = `c < R`
(exakt), `in_s` = `c ≥ n_kv − S` (exakt), `in_b` = Ringzeilen jenseits von S — die das Fenster
bezahlt, aber am Horizont quantisiert liest —, `outside` = nichts davon. `tag_s/tag_b/tag_out`
zählen von den dreien, wie viele Ringzeilen den Zelltag noch tragen (die win_all-Beute), das
Readback von `win_tag` ist in diesem Build **eingebaut** — der 32K-Lauf brauchte dafür noch einen
Zweiten. `d2…d32/older` zerlegen die nicht-exakten Treffer nach Abstand hinter dem Horizont in
Vielfachen von S (partitionieren `in_b + outside`), `d2` ist der Streifen, den ein doppelt so
großes S in den exakten Bereich umortet — Zellen, die schon gewählt sind, es ändert die Selektion
nicht, nur die Lesart.

## Aufbau

| Punkt | Wert |
|---|---|
| Engine | 0.1.47 Debug-Build `build-hip-port\strata.exe`, SHA-256 `CD5700EC0793F83B…3D0394CA` (identisch mit `2026-11-01-decode-4k`; vor dem Lauf erneut geprüft) |
| Modell | swift-1.5-iq3_xxs, Pack/Modelle wie in `strata-swift-iq3_xxs.json` |
| KV | `--kv q4lm32` (der Produktpunkt; die 32K-E1-Vorgänger lief mit `q4lm8v` — die Selektionsstatistik hängt am Codec nicht, die Lesung wird trotzdem offen gelegt, nicht still gemischt) |
| Fenster | `STRATA_KV_WINDOW=1`, `S=256`, `R=64` (Produktpunkt) |
| Prefill | `--prefill auto` (die Engine hebt `B` auf den Chunk — erwartet die stderr-Zeile „KV window ring sized for the prompt chunk (6144)“, am 2026-11-01 gemessen) |
| Passage | cwe-8k, alle 8 Segmente konkateniert, dann dieselben 8 noch einmal (s08–s15) — der Korpus wiederholt sich, weil kein 128K-Set existiert (siehe Abweichung 1): 65.511 + 65.511 = 131.022 Token |
| Schnitt | `--tokens 130048`: die letzten 974 Token der Passage fallen weg (mitten im Dokument s15) — deshalb `max-context = 130.560` (das Werkzeug rechnet `Tokens + 512`) ≤ die Pack-Kontext 131.072, der Lauf passt ohne weitere Knobs |
| Sampling | `STRATA_KV_SELSTATS_EVERY` 32 (Standard, wie 32K): jede 32. Abfrage je Selektions-Batch → ca. 4.000 Stichproben je QSA-Layer bei 132k Kontext |
| Pfad | batched Prefill (kein `--short-read`), ein `GEN 1`-Request, `QUIT` |

### Abweichungen von der 32K-Vorlage, offen genannt

1. **Wiederholte Dokumente.** Kein 128K-Korpus existiert; die Passage ist cwe-8k zweimal plus
   ein 974-Token-Schnitt. Die Selektionsstatistik sieht im zweiten Durchlauf dieselben
   Deskriptorstrukturen noch einmal. Für Anteilsmessung vermerkt, nicht kosmetisch weggeputzt.
2. **`q4lm32` statt `q4lm8v`.** Der Kandidat, nicht die 32K-Vorgänger-Codec-Wahl. Der Zähler
   hängt hinter der Selektion, nicht am Codec.
3. **974 Token Schnitt am Ende.** Der letzte Dokument-Tail fehlt; kein Qualitätsurteil hängt
   daran, das Positionsspektrum liegt nur minimal unter dem vollen Kontext.

## Fragen und Entscheidungsregeln (vor dem Lauf festgelegt)

Primärer Ausschnitt: `n_kv` > 16.384 (dort existieren alle Abstandsfächer; der 32K-Bericht
centralisierte dieselbe Stütze). Berichtete Größe: Anteil der gewählten Zellen, je QSA-Layer und
gepoolt, mit 95-%-Wilson-Intervall über die gezählten Zellen. Keine KL, kein Timing, kein
Produktpunkt-Urteil — **Anteile.**

Der folgende Vergleich nutzt die 32K-Zahlen als registrierte Vergleichsniederlage (nicht als
Nullhypothese mit Signifikanzmachinerie; Wilson gegen einen festen Wert ist die Form, keine
Lesehilfe):

| Befund | 32K-Vergleichswert | regel |
|---|---|---|
| `R1` exakt (in_r+in_s) Anteil, gepoolt `n_kv`>16k | 11,7 % | Intervall schließt 11,7 % ein → „hält“; sonst „gefallen/gestiegen“, Richtung ausdrücklich |
| `R2` in_b-Anteil (Bezahlt-quantisiert) | 59,2 % | dieselbe Regel |
| `R3` outside-Anteil | 29,1 % | dieselbe Regel (Katalogfrage war die Richtung: „open“) |
| `R4` Tag-Lesung (win_all → exakt lesbar) | 70,2 % | dieselbe Regel |
| `R5` Recall S sinks: R / S | 28,9 % / 86,2 % (32K, `n_kv`>2.051) | gepaarkte Richtung analogous pro Stütze; das 32K-Fallen des sink-Recall mit dem Horizont ist die Frage, ob es bei 4-fachem Kontext weiterfällt |
| `R6` d-Histogramm, Ubung gegen die Gleichverteilung über `R ≤ c < n−S−B`… | 32K-Multiple (8,9×/7,1×/4,6×/1,7×/0,68×/0,41×) | gleiche Ubung-Formel, gleicher Ausschnittstyp — eine schmalere oder breitere Nah-Konzentration wird genannt, keine, die der 32K-Beitrag geordnet gegen die 128K-Beitrag verwechselt |
| `R7` die S-Ladder aus d2/d4… (genau das vorhandene Zusätzliches) | 32K: S=512 → 20,3 %, 1.024 → 34,1 %, 2.048 → 52,1 %, 4.096 → 65,6 % | dieselbe Ableitung bei 130k; die Ladder-Anteile mit Vorbehalt (die Annahme „größeres S umortet nur gewählte Zellen“ ist wieder als Voraussetzung vermerkt, nicht als gemessen) |

Die Katalog-Zelle S1 hat den 50-%-Kill der B1-Zeile weiterhin geöffnet: bei der heutigen
Geometrie (Breite 2.051 sobald die Selektion wählt, R+S = 320) liegt die arithmetische Decke bei
15,6 % — der 50-%-Satz ist auch bei 128K unerreichbar, und die Kill-Lesung wird wie im 32K-Bericht
über Recall und über die d-Fächer geführt. Dieses Dokument re-registeriert den Kill nicht neu —
er folgt der Katalogzeile (S1) unverändert.

## Güte-Gatter vor der Auswertung

1. stderr enthält die Zeile „KV window ring sized for the prompt chunk (6144)“ und **keine** „do
   not fit“-Ablehnung.
2. Das `GEN` endet mit `DONE` ohne „nonfinite“.
3. Die `selstats.tsv` enthält Zeilen für alle 12 QSA-Layer und reicht bis `n_kv` > 130.000.
4. `index.json` speichert die Umgebung exakt so, wie sie in der Tabelle steht (die `--env`-Reihenfolge
   bleibt gleich).

Schlägt ein Gatter an, wird der Lauf einmal wiederholt; jede Wiederholung wird im Ergebnis
ausdrücklich vermerkt.

## Was das nicht ist

- **Prefill-Selektion, nicht Decode-Selektion.** Gleiche Prädikat-Funktion, anderer Lauf; die
  32K-Annahme „Prefill-Selektion ≈ Decode-Selektion" (Begründung dort) wird hier übernommen,
  nicht neu bemessen.
- **Anteile, keine Qualität.** Der Lauf schreibt keine Logits und keine KL; er beantwortet die
  Frage „wo landet die Auswahl", nicht „wie gut liest sie".
- **Wiederholung als Korpusfüller.** Siehe Abweichung 1 oben.
- **Kein Timing.** Debug-Build mit Blickprobe-Syncs.

## Reproduktion

```powershell
python tools\kv_selstats.py run --config strata-swift-iq3_xxs.json `
    --ids C:\Users\krone\git\Strata-data\bench\2026-11-01-e1-128k\cwe8k-x2.ids.json `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-01-e1-128k `
    --exe build-hip-port\strata.exe --kv q4lm32 --tokens 130048
python tools\kv_selstats.py report C:\Users\krone\git\Strata-data\bench\2026-11-01-e1-128k\selstats.tsv
```

Rohdaten (TSV, stderr, index.json) liegen außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-11-01-e1-128k\`; im Repo steht nur dieses Dokument und
der Aggregatblock. Der `report` druckt seit diesem Branch die d-Multiples und die S-Leiter für
beide Pools (`n_kv` > 2051 und > 16384); die publizierten 32K-Werte erscheinen unter der
> 16384-Zeile, und genau diese Zeile trägt die Regeln `R6`/`R7` oben.

## Ergebnis

Ein Lauf, keine Wiederholung — alle vier Güte-Gatter hielten beim ersten Versuch (Ring-Zeile
„KV window ring sized for the prompt chunk (6144 cells), 12.6 MiB per QSA layer"; kein „do not
fit"; kein „nonfinite"; Zeilen für alle 12 QSA-Layer bis `n_kv` 130.017; `index.json` mit
exakt der registrierten Umgebung). Gelesen für `n_kv` > 16.384: **42.624 Stichproben,
87.421.824 gewählte Zellen.** Der Prompt las 130.048 Token in 908 s (143 tok/s) — Debug-Build
mit Blickprobe-Syncs, keine Timing-Aussage.

**Die Aggregation ist am 32K-Baseline selbst validiert:** dasselbe Skript auf
`2026-10-25-s1-selstats/selstats.tsv` reproduziert 11,67 % exakt, `in_b` 59,18 %, außerhalb
29,15 %, Recall 28,9/86,2 und die publizierten d-Vielfachen (8,88×/7,11×/4,64×/1,73×/0,68×/0,41×)
sowie die Leiter (20,3/34,1/52,1/65,6/76,2) — die Formel der Multiples ist damit am Baseline
festgelegt, nicht nachträglich gewählt.

### Die sieben vorregistrierten Befunde

| Befund | 32K | E1 bei 128K (`n_kv` > 16.384) | Regel-Ergebnis |
|---|---|---|---|
| `R1` exakt (in_r+in_s) | 11,7 % | **8,55 %** [8,54; 8,56] | Intervall schließt 11,7 % **nicht** ein → **gefallen** (−3,1 pp) |
| `R2` in_b | 59,2 % | **40,82 %** [40,81; 40,83] | **gefallen** (−18,4 pp) |
| `R3` outside | 29,1 % | **50,63 %** [50,62; 50,64] | **gestiegen** (+21,5 pp) — die offene Richtung ist beantwortet |
| `R4` Tag-Lesung (win_all) | 70,2 % | **48,55 %** [48,54; 48,56] | **gefallen** (−21,6 pp) |
| `R5` Recall R / S | 28,9 % / 86,2 % | **20,3 % / 63,4 %** | beide **gefallen**; der S-Recall ist bei 128K nicht mehr horizontal-stabil |
| `R6` d-Vielfache gegen Gleichverteilung | 8,9× / 7,1× / 4,6× / 1,7× / 0,68× / 0,41× | **11,74× / 8,97× / 6,45× / 3,23× / 1,25× / 0,61×** | Nah-Bias **steiler** (d2 8,9→11,7×), Alt-Masse **flacher** (older 0,41→0,61×) |
| `R7` S-Leiter (d2…kumuliert) | 20,3 / 34,1 / 52,1 / 65,6 / 76,2 % | **13,6 / 21,3 / 32,4 / 43,6 / 52,1 %** | bei **jedem** S **niedriger** |

Alle Wilson-Intervalle schließen den 32K-Wert aus; die Richtungen stehen damit ohne
Signifikanz-Maschinerie fest. Der Vergleich hat aber eine offen genannte Konfundierung: der
32K-Pool umfasste 16–32K, der 128K-Pool 16–130K und ist damit horizont-gewichtet. Die
horizontgetrennte Tafel unten trennt das: bei 8–16K liegt der exakte Anteil bei 11,36 % (nahe
dem 32K-Wert), bei ≥16K bei 8,55 %. Ein Teil von `R1` ist Komposition; die Richtung bleibt.

### Der Mechanismus: eine feste Zellenzahl über wachsendem Kontext

| Horizont | Stichproben | exakt | in R | in S | in B | außerhalb | Tag-Lesung |
|---|---|---|---|---|---|---|---|
| ≤ 319 (Selektion = Identität) | 120 | 100,0 % | 37,7 % | 62,3 % | 0,0 % | 0,0 % | 100,0 % |
| 320–2.051 | 660 | 27,0 % | 5,4 % | 21,6 % | 73,0 % | 0,0 % | 100,0 % |
| 2.052–8.191 | 2.292 | 12,0 % | 2,2 % | 9,8 % | 84,6 % | 3,4 % | 94,6 % |
| 8.192–16.383 | 3.072 | 11,4 % | 1,2 % | 10,1 % | 70,7 % | 18,0 % | 80,0 % |
| ≥ 16.384 | 42.624 | 8,6 % | 0,6 % | 7,9 % | 40,8 % | 50,6 % | 48,6 % |

Das ist die erwartete Geometrie, jetzt am 128K-Punkt gemessen: `R+S` ist eine feste Zellenzahl
(320), `B` eine feste (6.144); der Kontext wächst auf 130k. Der `outside`-Anteil klettert
0 → 3,4 → 18,0 → 50,6 %, der Ringanteil `in_b` fällt entsprechend, und die Tag-Lesung — die
win_all-Beute, die 32K noch 70 % deckte — deckt am 128K-Ende **48,6 %**: der Ring, selbst mit
Tag-Lesung, deckt weniger als die Hälfte der Auswahl. Die Layer-Streuung bleibt (exakt 6,5 %
in Layer 4 bis 13,2 % in Layer 0), also ist auch hier keine Zahl die ganze Geschichte.

### Lesart für S1, B1, E3

- **Die 10-%-Linie greift jetzt.** 8,55 % ≤ 10 %: das positionsfeste Overlay schützt am
  128K-Punkt überwiegend Zellen, die der Indexer nicht wählt. Bei 32K lag der Wert mit 11,7 %
  knapp darüber („dazwischen"), und der Rand entschied — der bleibt auch hier zu dünn für einen
  Score-Pool (p50 0,01 % der Spreizung, p90 0,04 %, 7 Gleichstände in 42.624 Stichproben; die
  5-%-Regel scheitert um zwei Größenordnungen). Der auswahlfolgende Pool ist damit weiter nicht
  baubar — die Begründung ist jetzt an zwei Punkten dieselbe.
- **Die Leiter kauft am 128K-Punkt weniger pro S.** S=512 → 13,6 % (32K: 20,3), S=4.096 →
  43,6 % (65,6), S=8.192 → 52,1 % (76,2). Um die Hälfte der Auswahl exakt zu lesen, bräuchte es
  am 128K-Punkt ein S um 8.192 — 32× das heutige —, und das Byte-Modell preist diesen Ring mit
  B = 512 auf 64 + 8.192 + 512 = 8.768 Zeilen: 4,50 + 16·8.768/131.072 = **5,57 bit/Zelle**
  (mit B = 6.144: 6,26) — beide über dem Produktpunkt (5,29). Die
  registrierte S1-Lesung (Recall und d-Fächer, nicht der unerreichbare 50-%-Satz) steht damit
  auch für die 128K-Hälfte: **B1 öffnet sich nicht** — im Gegenteil, es kauft hier weniger pro
  S als bei 32K. Das war die offene Frage des 32K-Berichts, und die Antwort ist „nein".
- **Der Hebel bleibt die Tag-Lesung, jetzt mit Zahl am 128K-Punkt.** 48,6 % statt 8,6 % exakt,
  bei null zusätzlichen Bytes — aber unter der Hälfte, und der Qualitätsbeweis (schließt das die
  KL-Lücke zu `int8`?) bleibt der nächste Arm, wie S1 ihn aufgeschrieben hat.

### Grenzen

- Prefill-Selektion, nicht Decode-Selektion; die Annahme ist übernommen, nicht neu bemessen.
- Anteile, keine Qualität — keine Logits, keine KL.
- Der Korpus wiederholt sich (kein 128K-Set); für Auswahlstatistik unkritisch, für
  Qualitätsurteile nicht — dieser Lauf fällt keins.
- `q4lm32` statt `q4lm8v` (32K-Vorgänger); der Zähler sitzt hinter der Selektion, nicht am Codec.
- Die Poolung über `n_kv` > 16.384 ist horizont-gewichtet; die Tafel oben trennt das aus.
- Debug-Build mit zwei Readbacks und Sync je Stichprobe: kein Timing-Lauf.
