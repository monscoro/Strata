# G3: unterscheiden sich die Arme funktional?

**Status: fertig (2026-10-31).** Neun Läufe bei Budget 256. Gate 1 besteht, das Ersatz-Gate fällt und
liefert damit die Rauschgrenze von einem Wort. **Kein Arm liegt jenseits der Grenze.** Der Befund: die
Unterschiede zwischen den Armen sitzen im zehnten Platz, nicht in der Antwort — das Modell selbst irrt
sich auf vier Segmenten, für alle Arme mit demselben Wort. Katalogzeile:
`docs/paper/Ansatz_Improvements.md`, Arbeitsplan G3 (`docs/research/BEYOND_KL_QUANT_FIDELITY.md` §4).
Vorläufer: G1 (`bench/results/2026-10-28-g1-features/README.md`), G2
(`bench/results/2026-10-28-g2-full-vocab/README.md`).

Dieses Dokument legt **vor** dem ersten Lauf fest, was gemessen wird, wie gezählt wird und wann das
Ergebnis „der Maßstab hat keine Auflösung" heißt. Drei Nachträge (unten) halten fest, was die Pilotläufe
am Anker davor geändert haben.

## Ergebnis

### Die Zahlen

Neun Läufe, je ein Engine-Start, je acht Segmente, Budget 256, Denkblock geschlossen
(`bench/configs/cwe4k-swift-iq3_xxs.json`, HIP, gfx1200, 4,0–4,8 min je Lauf):

| Lauf | `recall` | je Segment (Treffer von 10) |
|---|---|---|
| `fp16` (Anker) | 0,950 | 9 9 10 10 9 9 10 10 |
| `int8` | 0,938 | 9 9 10 10 9 9 10 9 |
| `q4lm32w` | 0,963 | 10 9 10 10 9 9 10 10 |
| `s320r0` | 0,950 | 9 9 10 10 9 9 10 10 |
| `q4lm32` | 0,938 | 9 9 10 9 9 9 10 10 |
| `q4lm8v` | 0,875 | 9 9 10 9 9 9 **6** 9 |
| `q4lm8k` | 0,950 | 9 10 10 9 9 9 10 10 |
| `q4lm8` | 0,950 | 9 9 10 10 9 9 10 10 |

Gepaart gegen `int8` (Grenze 1,895·sd/√8) liegt **kein** Arm jenseits der Grenze. Der größte Abstand ist
`q4lm8v` mit −0,062 (Grenze 0,094, in 2 von 8 Segmenten schlechter, in keinem besser); die anderen liegen
zwischen 0,000 und +0,025 mit Grenzen von 0,024 bis 0,043. `q4lm8v` gegen `fp16` ist −0,075 (Grenze 0,093).
Die zweite Fassung (`recall_first10`, die ersten zehn genannten Wörter) ändert keine Zahl außer bei
`q4lm8k` (0,925 statt 0,950) — die Gold-Wörter stehen in beiden Antwortformen vorn.

Über alle 8 × 8 = 64 Arm-Segment-Zellen: **28 nennen alle zehn Gold-Wörter, 35 nennen neun, eine nennt
sechs.** Kein Segment hat in allen acht Armen dieselben zehn Wörter — die Unterschiede sitzen im zehnten
Platz.

### Was die Verluste sind

Der größte Teil der fehlenden Wörter ist **das Modell, nicht das KV**, und zwar für alle Arme gleich:

| Segment | alle Arme nennen | Gold wäre | Ausnahme |
|---|---|---|---|
| s00 | `lywgne` | `atiyuw` | nur `q4lm32w` nennt `atiyuw` |
| s01 | `bmvgdb` | `clldqg` | nur `q4lm8k` nennt `clldqg` |
| s03 | `vmnarn` | `vszdcd` | nur `q4lm32`, `q4lm8v`, `q4lm8k`; die anderen nennen `vszdcd` |
| s04 | `iwwqby` | `hlczmi` | — |
| s05 | `benmuv` | `bkacgz` | — |

Diese fünf Wörter sind Häufigkeitsfehler des Modells auf dieser Aufgabe — `fp16` macht sie genauso. Sie
sind der Grund, warum der Anker bei 9,50/10 steht und nicht bei 10/10.

Der Arm mit dem größten Abstand, `q4lm8v`, verliert fünf Wörter auf zwei Segmenten:

- **s03, ein Wort:** `vmnarn` statt `vszdcd` — derselbe Fehler wie `q4lm32` und `q4lm8k`.
- **s06, vier Wörter:** `q4lm8v` antwortet dort nicht als Liste, sondern schreibt die Häufigkeiten aus
  („**bzqjcf**: This word appears moderately. A manual count reveals it appears **21** times"), und in
  256 Token stehen dann nur sechs der zehn Wörter. Es kennt die anderen; das Budget schneidet sie ab.
  Die nachträgliche Form-Analyse (`tools/kv_g3_forms.py`) zeigt es: im ganzen Text nennt `q4lm8v` auf s06
  sechs verschiedene Kunstwörter, die anderen Arme zehn.

Die Antwortform ist überhaupt nicht stabil: `fp16` und `int8` antworten auf 6 von 8 Segmenten als
Kommaliste (41–46 Token, endet mit dem Ende-Token) und auf 2 als Fließtext mit Nummerierung (256 Token,
ins Budget geschnitten); `q4lm8k` wechselt auf 4 Segmenten in den Fließtext, `q4lm8` auf 3, die übrigen
auf 1–2. Das ist eine Eigenschaft des Modells und für alle Arme dieselbe Größenordnung — aber es heißt,
dass jede Metrik über ein festes Token-Budget die Verbosität mitmisst. Deshalb steht die Form in der
Tabelle der Form-Analyse und nicht im Urteil.

### Das Urteil

Die vorregistrierte Bedingung —

> mindestens ein Arm liegt in `recall` gepaart über die acht Segmente jenseits der Grenze von `int8`

— ist **nicht erfüllt**. Auf dieser Aufgabe und dieser Länge (4.096 Token, acht Segmente) unterscheiden
sich die Arme funktional nicht messbar; was sich unterscheidet, sind einzelne Wörter im zehnten Platz
und die Antwortform.

Die Decke blockiert nicht (Nachtrag 2), und die 8K-Fassung greift, wenn **kein** Arm ein Wort verliert:
`q4lm8v` verliert fünf und `q4lm32` eins, also greift sie nicht. Dass diese Verluste unter der Grenze
liegen, ist eine Aussage über n = 8 und 10 Wörter je Segment, nicht über die Arme.

### Was daraus für G4 folgt

G4 sollte Merkmale aus G1 mit den Ereignisraten aus G3 korrelieren. Die Ereignisraten sind hier 0 bis 5
Wörter je 80, davon die meisten das Modell selbst, und kein Arm ist entscheidbar. Eine Korrelation
zwischen einem Merkmal und einer Rate, die sich nicht von null unterscheidet, ist keine Messung. G4 hat
auf diesem Instrument **keinen Gegenstand**; das ist das Kill-Kriterium der Stufe, und es wird nicht
weggerechnet.

Drei Wege, alle außerhalb dieser Vorregistrierung und deshalb eine Entscheidung, keine Fortsetzung:

1. **Härter messen, gleiche Aufgabe:** 8K-Fassung (`--seq-tokens 8192`). Mehr Wörter zu zählen heißt
   mehr Gelegenheit zu verlieren — aber auch ein schwererer Anker.
2. **Andere Aufgabe:** die Dreiwort-Fassung (`data/quant_quality/fwe-2k`) ist leichter, nicht schwerer;
   sie wäre die Gegenprobe „verliert ein Arm überhaupt je ein Wort", nicht die härtere Fassung.
3. **Annehmen, was gemessen ist:** bei 4K sind die 4-Bit-Arme von `int8` funktional nicht zu trennen. Das
   ist ein Ergebnis und passt zu G2 (die KL-Rangfolge ist auf dieser Stichprobe nicht entscheidbar).

## Nachtrag 4 (2026-10-31, vor dem ersten 8K-Lauf): die Fortsetzung

**Warum überhaupt.** G4 hat bei 4K keinen Gegenstand (oben). Die 8K-Fassung ist der Weg 1 aus der Liste;
sie wird gefahren, weil mehr Wörter über mehr Token zu zählen mehr Gelegenheit zum Verlieren ist — und
weil die KV-Kompression genau dort sitzen sollte: weit weg, viele Token, ein seltenes Detail.

**Das Set.** `data/quant_quality/cwe-8k`, gebaut mit demselben Generator, demselben Seed (42) und
demselben Alpha (2,0), nur `--seq-tokens 8192`:

| | 4K (`cwe-4k`) | 8K (`cwe-8k`) |
|---|---|---|
| Token je Segment | 4.088–4.096 | 8.187–8.191 |
| Prompt (`answer_at`) | 4.054–4.064 | 8.152–8.161 |
| gezogene Wörter | 1.967–2.293 | 4.158–4.605 |
| Kunstwörter im Vokabular | 81 | **163** |

Der Generator skaliert das Vokabular mit (`seq_tokens // 50`), also ist die 8K-Fassung nicht nur länger,
sondern auch schwerer: 163 Kandidaten statt 81, doppelt so viele gezogene Wörter. Das ist die
vorregistrierte Fassung („gleicher Seed, gleiche Alpha"), und es wird so berichtet.

**Die Läufe.** Zuerst nur `fp16` und `int8` — die Frage dieser Runde ist, ob das Instrument bei 8K
überhaupt etwas sieht, nicht die Rangfolge der Arme. Alles andere wie bei 4K: `--close-thinking`,
Budget 256, `--short-read 64`, `--prefill 512`, ein Engine je Lauf.

**Die Entscheidung, vor dem ersten 8K-Lauf festgelegt:**

| Befund bei 8K | Folge |
|---|---|
| Anker ≤ 2,5/10 | Boden: G3 hat auch bei 8K keinen Gegenstand, keine Arme, G4 bleibt verschoben |
| Anker ≥ 9,5/10 **und** `int8` auf allen acht Segmenten identisch zum Anker | das Instrument sieht bei 8K nichts, was es bei 4K nicht sah; keine Arme, Ergebnis „bei 4K und 8K nicht trennbar" |
| sonst | die sechs Arme laufen bei 8K |

**Kosten.** Ein 8K-Lauf ist rund doppelt so teuer wie ein 4K-Lauf (Prefill 8K statt 4K); geschätzt
**8–10 min je Lauf**, zwei Läufe also rund **20 min GPU**. Die sechs Arme wären danach rund **1 h**.

## Frage

Bisher ist jede Aussage dieser Arbeitslinie eine Zahl über die **Verteilung** des nächsten Tokens
(`KL(fp16‖Arm)`). G2 hat gezeigt, dass die trunkierte KL die Rangfolge trägt. G3 fragt das andere:

> **Verlieren die Arme, die auf der KL schlechter aussehen, auch Wörter, die ein Nutzer sehen würde?**

Das ist die erste Stufe mit einer **funktionalen** Zielgröße statt einer Verteilungsdistanz. Sie
entscheidet, ob G4 überhaupt einen Gegenstand hat: G4 korreliert Merkmale mit den Ereignisraten, die
hier entstehen.

## Die Aufgabe

RULERs *frequent words extraction* in der Zehn-Wort-Fassung (*common words extraction*, arXiv:2404.06654
§3.3), aus dem Generator, der `data/quant_quality/cwe-4k` gebaut hat (`tools/quant_quality.py fwe`,
`--seq-tokens 4096 --samples 8 --answer-words 10 --alpha 2.0 --seed 42`):

- 81 sechsbuchstabige Kunstwörter, das häufigste ist `...` (der Rauschanteil, den die Frage zu ignorieren
  heißt), die übrigen nach einer Zeta(2)-Rangverteilung gezogen — 1.967 bis 2.293 gezogene Wörter je
  Segment, gemischt.
- Danach die Frage und `Answer: According to the coded text above, the ten most frequently appeared
  words are:` — an dieser Stelle endet der Prompt (`answer_at`).
- Gold sind die zehn häufigsten Wörter. Die Antwortregion im Text ist 32–36 Token lang (Mittel 33,75).

Acht Segmente, je 4.088–4.096 Token. Die Aufgabe verlangt Zählen über den ganzen Prompt; sie ist die
natürliche Gegenprobe zu einer KV-Kompression, weil deren Verlust genau dort sitzt: weit weg, viele
Token, ein seltenes Detail.

## Was gemessen wird

| Punkt | Festlegung |
|---|---|
| Prompt | `ids[:answer_at]` aus `data/quant_quality/cwe-4k/set.ids.json` (4.054–4.064 Token) plus der Assistenten-Turn mit geschlossenem Denkblock, `<|im_start|>assistant\n thinking\n\n</think>\n\n` (7 Token) — der Pfad, den die App im Nicht-Denken-Modus bedient |
| Dekodierung | **greedy** (`GEN` ohne Sampling-Schlüssel; der Engine-Default ist argmax, keine Strafterme), `ckpt=0` je Segment |
| Budget | **256 neue Token** oder das Ende-Token des Modells, was zuerst kommt |
| Engine | `bench/configs/cwe4k-swift-iq3_xxs.json`, ein Engine-Start je Lauf, alle acht Segmente darin |
| Prompt-Pfad | `--short-read 64` (Engine- und App-Default), `--prefill 512` (Chunk des gepinnten Configs) |
| Build | `build-hip-port\strata.exe`, HIP, ROCm 7.1, gfx1200 — derselbe Build wie G2 |

**Warum 256 Token:** das Modell antwortet in einer von zwei Formen, und beide sind zwischen `fp16` und
`int8` auf denselben Segmenten identisch (Nachtrag 3): als Kommaliste (zehn Wörter in 41–46 Token) oder
nummeriert mit einem Satz je Wort (rund 15 Token je Wort, also rund 170 für zehn). 256 trägt beide; 64
schnitt die nummerierte Form nach ein bis zwei Wörtern ab. Ein falsches Wort trifft ein Gold-Wort mit
10/2.000 Wahrscheinlichkeit (2.000 gezogene Wörter, 10 davon Gold), also erwartet 0,15 Fehltreffer je
Segment über die Luft — dieselbe Größenordnung für jeden Arm.

**Die Arme** (dieselben sieben wie G2, plus der Anker):

| Lauf | KV | Umgebung | Rolle |
|---|---|---|---|
| `fp16` | `fp16` | — | Anker: was die Aufgabe ohne KV-Quantisierung hergibt |
| `int8` | `int8` | — | der Default, Vergleichspunkt jeder gepaarten Differenz |
| `q4lm32w` | `q4lm32` | `STRATA_KV_WINDOW=1 S=256 R=64` | Leiter |
| `s320r0` | `q4lm32` | `STRATA_KV_WINDOW=1 S=320 R=0` | Fenster-Verlustpunkt aus §3 |
| `q4lm32` | `q4lm32` | — | Leiter |
| `q4lm8v` | `q4lm8v` | — | Leiter |
| `q4lm8k` | `q4lm8k` | — | Leiter |
| `q4lm8` | `q4lm8` | — | Leiter, schlechtestes KL-Niveau |

**Ein Unterschied zu S3, der vorab festgehalten wird:** die Fenster-Arme halten auf diesem Pfad
`S + R + 512` Zeilen (der Chunk des Prompt-Pfads), nicht `S + R + 1` wie in den teacher-forced
S3-Läufen, wo jedes Prompt-Token ein eigener Dekodierschritt war. Bei 4.096 Token Prompt sieht
`q4lm32w` hier also rund 768 statt rund 321 Zeilen. Das ist der Pfad, den die App bedient; G3 misst
ihn, nicht den S3-Pfad.

## Die Metrik

Je Segment: `recall = |Gold ∩ Wörter(Text)| / 10`, wobei `Wörter(Text)` die Mengen der `[a-z]+`-Läufe
des generierten Textes in Kleinschreibung ist. Ein Gold-Wort zählt, wenn es irgendwo in den 64 Token
steht; die **Reihenfolge wird nicht bewertet** (RULER bewertet sie auch nicht).

- **primär:** `recall` über alle 64 Token.
- **sekundär:** `recall_first10` — dieselbe Zählung, aber nur über die ersten zehn **verschiedenen**
  sechsbuchstabigen Wörter des Textes. Das ist die Fassung „hat es die richtigen zehn genannt".

Beide werden berichtet. Die Aussage der Stufe hängt an `recall`.

## Die Auswertung

- Mittel über die acht Segmente, je Arm.
- **Gepaart über Segmente** gegen `int8` und gegen `fp16`: je Segment die Differenz, dann Mittel, sd,
  Grenze `1,895 · sd/√8` (dieselbe Grenze wie G1 und G2, t mit df = 7), plus die Zahl der Segmente, in
  denen der Arm schlechter liegt. **Cluster ist das Segment, nie die Position** (§3 des Plans).
- Bei 8 Segmenten und 10 Wörtern ist ein Wort 0,1 im Mittel; die Auflösung der gepaarten Grenze liegt
  bei rund 0,1. Kleinere Unterschiede sind mit diesem Instrument nicht messbar, und das wird berichtet,
  nicht weggerechnet.

## Das Gate

Zwei Prüfungen **vor** jeder Armaussage:

| Prüfung | Erwartung | Wenn sie fällt |
|---|---|---|
| `int8` zweimal, zwei getrennte Engine-Starts | dieselben 8 × 256 Token-Ids | die Armvergleiche warten, bis die Ursache benannt ist; die Nichtdeterminismus-Breite wird berichtet |
| `int8` ohne `--spec`/`--spec-min-p` | dieselben Ids wie mit (der gepinnte Config dekodiert spekulativ) | siehe Nachtrag 1: für dieses Modell nicht durchführbar, ersetzt durch `--spec 2` gegen `--spec 4` |

**Beide Prüfungen sind gelaufen.** Die erste besteht bei 256 Token: zwei Engine-Starts, 8 von 8 Segmenten
mit denselben Ids und demselben Text. Die zweite fällt — siehe Nachtrag 1 und die Grenze unten.

Die zweite Prüfung ist keine Formsache: spekulative Dekodierung ist nur dann verlustfrei, wenn die
Verifikation das greedy-Ergebnis exakt reproduziert. Weicht sie ab, ist das eine Eigenschaft des
Produkts, die in den Bericht gehört.

## Sättigungs- und Bodenregel (geändert in Nachtrag 2)

Der Anker entscheidet, ob das Instrument überhaupt etwas sieht.

| Befund am Anker `fp16` | Lesart | Folge |
|---|---|---|
| Mittel `recall` ≤ 2,5/10 | Boden: der Anker kann die Aufgabe nicht | einmal neu bauen mit `--seq-tokens 2048` (gleicher Seed, gleiche Alpha), dann nur `fp16` + `int8`; bleibt es beim Boden, hat G3 keinen Gegenstand und G4 wird verschoben |
| Mittel `recall` ≥ 9,5/10 | Decke | **kein Abbruch**: eine Decke ist der Ausgangspunkt, den man zum Messen von Verlusten will; die sieben Arme laufen trotzdem |
| dazwischen | das Instrument hat Auflösung | alle sieben Arme werden berichtet |

Verliert bei einer Decke **kein** Arm ein Wort, ist die 8K-Fassung (`--seq-tokens 8192`, gleicher Seed, gleiche
Alpha) die vorregistrierte Fortsetzung, nicht der erste Schritt.

Warum die Decke nicht mehr blockiert (Nachtrag 2): G3 fragt nach **Verlusten**. Ein Anker bei 10/10 ist dafür
der beste Ausgangspunkt — jedes verlorene Wort ist sichtbar. Die ursprüngliche Fassung hätte bei einer Decke
zuerst 8K gebaut, ohne dass ein Arm gemessen wäre; das wäre GPU-Zeit für eine Frage, die die Armläufe selbst
beantworten.

## Was „unterscheiden" heißt

G3 ist kein Annahme- oder Ablehnungstest für einen Arm. Das Urteil der Stufe ist:

> **Mindestens ein Arm liegt in `recall` gepaart über die acht Segmente jenseits der Grenze von `int8`
> — und in welcher Richtung.**

Ein Arm, der Wörter verliert, wird hier **nicht** abgelehnt; die verlorenen Wörter sind die
Ereignisrate, mit der G4 die Merkmale aus G1 korreliert. Ein Arm, der nichts verliert, ist die
Gegenprobe: ein Merkmal, das dort einen Verlust meldet, ist ein Fehlalarm.

## Was das nicht sagt

- **Acht Segmente, ein Korpus, eine Aufgabe.** Die Auflösung ist rund 0,1 im Mittel; ein Unterschied von
  einem Wort je Segment ist nicht messbar.
- **Die Rauschgrenze ist ein Wort.** Zwei Konfigurationen derselben KV-Form (`--spec 2` gegen `--spec 4`)
  unterscheiden sich um ein Wort auf einem Segment (Nachtrag 1). Ein Unterschied von einem Wort zwischen
  zwei Armen ist deshalb keine Aussage.
- **`recall` ist grob.** Es zählt Wörter, nicht Reihenfolge, nicht Sicherheit, nicht Begründung.
- **Die Antwortform mischt mit.** Eine feste Token-Zahl misst auch, wie viel das Modell schreibt; auf s06
  kostet das `q4lm8v` vier Wörter, die es kennt (Ergebnis-Abschnitt). Die wörterbuchbeschränkte
  Nachauswertung (`tools/kv_g3_forms.py`) ändert an keiner Armzahl etwas, aber sie ist **nach** den Läufen
  entstanden und nicht vorregistriert.
- **Greedy ist eine Stichprobe der Größe eins** aus der Antwortverteilung des Modells. Deterministisch
  heißt nicht repräsentativ.
- **Ein entarteter Lauf zählt wie er ist.** Wiederholt sich ein Arm in eine Schleife, ist das ein
  funktionaler Verlust und wird nicht durch eine Umschreibung des Budgets repariert.
- **Nur HIP gebaut.** CUDA, SYCL und Metal sind hier nicht gemessen.
- **Der Fenster-Arm ist nicht der S3-Arm**, siehe oben: `S + R + 512` statt `S + R + 1`.
- **Gemessen wird der Nicht-Denken-Pfad.** Mit offenem Denkblock zählt das Modell den Text auf und antwortet
  nicht (Nachtrag 2); ein Denk-Modus ist mit diesem Instrument nicht gemessen.

## Kosten

Gemessen bei den neun Läufen (4.067-Token-Prompt, geschlossener Denkblock, Budget 256, HIP, gfx1200):
**4,0 bis 4,8 min je Lauf** (schnellster `q4lm8` 4:03, langsamster `q4lm32w` 4:47), zusammen **41 min
GPU** für neun Läufe; der `--spec 2`-Lauf kam mit 4:02 dazu. Die Auswertung ist offline.

## Dateien

| Datei | Inhalt |
|---|---|
| `tools/kv_g3_run.py` | ein Engine je Lauf, `GEN` je Segment, schreibt Ids und Text |
| `tools/kv_g3_score.py` | `recall`, `recall_first10`, gepaarte Differenzen, Sättigungs-/Bodenurteil |
| `tools/kv_g3_forms.py` | nachträglich: Antwortform, wörterbuchbeschränkte Zählung, gepaart |
| `score.json` | die vorregistrierte Auswertung der acht Läufe |
| `forms.json` | die nachträgliche Form-Analyse |
| `data/quant_quality/cwe-4k/gold.json` | die zehn Gold-Wörter je Segment |
| `data/quant_quality/cwe-4k/set.ids.json` | Prompt-Ids und `answer_at` |

Die Rohläufe (Ids und Text je Segment, je Lauf eine JSON) liegen in
`C:\Users\krone\git\Strata-data\bench\2026-10-31-g3\`; die 64-Token- und Pilotläufe in dessen
Unterordner `pilot\`.

## Nachtrag (2026-10-31, vor dem Anker-Lauf)

**Die zweite Gate-Prüfung ist nicht durchführbar.** Ein nativer IQ-Pack verlangt `--spec T` mit `T >= 2`;
die Engine verweigert ohne das den Start (gemessen: der `int8`-Lauf ohne `--spec`, Exit 2, stderr
`strata generate: … is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) and --prefill
CHUNK`; statische Analyse dazu: `src/program/generate.cpp:3752-3757` prüft `o.spec < 2`). Ein Lauf ohne
spekulative Dekodierung existiert für dieses Modell also nicht.

Ersetzt wird sie durch **`int8` mit `--spec 2` gegen den gepinnten `--spec 4`**, gleiche Ids erwartet. Das
ist schwächer als geplant: es zeigt, dass die Fenstergröße der Verifikation die Token nicht ändert, nicht,
dass die Verifikation das greedy-Ergebnis eines ungedrafteten Laufs reproduziert. Der Bericht nennt es so.

**Das Ersatz-Gate fällt, auch auf der Metrik.** Mit dem endgültigen Prompt und Budget 256 (`--spec 2`
gegen `--spec 4`, je ein Engine-Start, acht Segmente):

| Segment | `--spec 4` (der Default) | `--spec 2` |
|---|---|---|
| s03 | Fließtext, 256 Token | Fließtext, 256 Token, anderer Wortlaut |
| s05 | `… xppqmo, njyajr, awnpwh …` | `… njyajr, xppqmo, awnpwh …` (vertauscht) |
| s07 | `… ydfaud, bilnka` | `… ydfaud, lnnjgo` |

3 von 8 Segmenten haben andere Token, und auf s07 ändert sich die Metrik: `--spec 2` nennt `lnnjgo`
(das Gold-Wort) statt `bilnka`, also 10/10 statt 9/10. Mittel: `--spec 2` 0,950, `--spec 4` 0,938,
Differenz +0,012 (Grenze 0,024) — **nicht jenseits der Grenze, aber in derselben Größenordnung wie die
Armunterschiede.**

Das ist die gemessene **Rauschgrenze des Instruments**: zwei Konfigurationen derselben KV-Form
unterscheiden sich um ein Wort auf einem Segment. Ein Unterschied von einem Wort zwischen zwei Armen ist
damit nicht von einem Konfigurationseffekt zu trennen; erst mehrere Wörter über mehrere Segmente sind
eine Aussage. Für die Armvergleiche ist es unschädlich, weil alle Arme dieselbe Konfiguration benutzen —
es begrenzt aber die Auflösung, und der Bericht nennt es so.

**Gate 1 besteht:** `int8` zweimal in zwei getrennten Engine-Starts, 8 von 8 Segmenten mit denselben
Token-Ids und demselben Text — bei 64 und bei 256 Token.

## Nachtrag 2 (2026-10-31, vor dem ersten Arm-Lauf): der Prompt und das Budget

Drei Pilotläufe am Anker `fp16`, alle **vor** dem ersten Arm:

| Pilot | Prompt | Budget | Ergebnis |
|---|---|---|---|
| 1 | roh, `ids[:answer_at]` | 64 | `recall` 0,325; 6 von 8 Segmenten öffnen ` thinking`, das Budget endet darin; 2 von 8 antworten direkt, 10/10 |
| 2 | roh | 2.048 | 10/10 auf allen acht — **aber die Zahl ist falsch**: auf 6 von 8 Segmenten zählt das Modell den Text Wort für Wort auf („83. gywwje - 1", 381 Wörter, 2.048 Token) und antwortet nie; die Gold-Wörter stehen in der Aufzählung, nicht als Antwort. Nur s06/s07 antworten (47 Token, 10/10) |
| 3 | roh + Assistenten-Turn mit geschlossenem Denkblock | 128 | s00 46 Token, **9/10** (sagt `lywgne` statt `atiyuw`); s02 42 Token, **10/10**. Die Chat-Umhüllung (`<|im_start|>user … <|im_end|>`) ist schlechter: auf s00 erklärt das Modell die Methode, statt zu antworten |

Daraus drei Änderungen, alle vor dem ersten Arm festgelegt:

1. **Der Prompt bekommt den Assistenten-Turn mit geschlossenem Denkblock**
   (`<|im_start|>assistant\n thinking\n\n</think>\n\n`, 7 Token). Das ist der Pfad, den die App im
   Nicht-Denken-Modus bedient. Ohne ihn ist die Aufgabe für dieses Modell nicht messbar: es zählt lieber
   den ganzen Text auf, und keine Budgetgröße macht daraus eine Antwort.
2. **Budget 64** statt 2.048: mit geschlossenem Denkblock ist die Antwort 42–46 Token lang.
3. **Die Decke blockiert die Arme nicht mehr** (siehe oben): sie ist der Ausgangspunkt zum Messen von
   Verlusten.

Was der Pilot **nicht** zeigt: er lief nur auf `fp16` und nur auf zwei bzw. acht Segmenten; er begründet
Prompt-Form und Budget, nicht das Ergebnis. Die `recall`-Zahl über die ganze Ausgabe bleibt anfällig für
eine Aufzählung — deshalb steht in jeder Tabelle die Spalte `distinct6`; ein Lauf, der den Text aufzählt,
zeigt dort mehr als 20 verschiedene sechsbuchstabige Wörter statt der 10 einer Antwort.

## Nachtrag 3 (2026-10-31, vor dem ersten Arm-Lauf): das Budget trägt beide Antwortformen

Der Gate- und Ankerlauf mit dem neuen Prompt (Budget 64) zeigt zwei Antwortformen, die beide zwischen
`fp16` und `int8` **auf demselben Segment identisch** sind:

| Segment | Form | Länge | Treffer bei 64 |
|---|---|---|---|
| s00, s01, s02, s04, s05, s07 | Kommaliste (`mfkbjz, zvzywe, …`) | 41–46 Token, endet mit dem Ende-Token | 9–10 von 10 |
| s03, s06 | nummeriert mit einem Satz je Wort (`1. **rpwqxq**: This word appears very frequently …`) | rund 15 Token je Wort, also rund 170 für zehn | 1 bzw. 2 von 10 |

Die nummerierte Form ist keine Entartung, sondern die zweite Antwortform des Modells; sie wird vom Budget
abgeschnitten, nicht vom KV. Anker und Default messen dort dasselbe (s03: `fp16` 1, `int8` 2; s06: beide 2),
also ist der Fehler für den gepaarten Vergleich unschädlich — aber er kostet 8 der 10 Wörter und damit
Auflösung.

**Deshalb: Budget 256.** Das trägt die nummerierte Form (rund 170 Token) mit Luft und ändert an der
Kommaliste nichts, weil sie mit dem Ende-Token endet.

**Gemessen bei 256** (die Vorhersage an dieser Stelle war der 64-Token-Wert 0,738 und damit zu
pessimistisch): der Anker steht bei **0,950** (9 9 10 10 9 9 10 10). Die nummerierte Form nennt die
Gold-Wörter, wenn sie Platz hat — sie braucht ihn nur, um sie auszuschreiben. Damit ist der Anker an der
Decke (9,50 von 10, Schwelle 9,5), nicht zwischen Boden und Decke.


