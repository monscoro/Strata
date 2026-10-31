# G3: unterscheiden sich die Arme funktional?

**Status: vorregistriert (2026-10-31), Läufe stehen aus.** Katalogzeile:
`docs/paper/Ansatz_Improvements.md`, Arbeitsplan G3 (`docs/research/BEYOND_KL_QUANT_FIDELITY.md` §4).
Vorläufer: G1 (`bench/results/2026-10-28-g1-features/README.md`), G2
(`bench/results/2026-10-28-g2-full-vocab/README.md`).

Dieses Dokument legt **vor** dem ersten Lauf fest, was gemessen wird, wie gezählt wird und wann das
Ergebnis „der Maßstab hat keine Auflösung" heißt.

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
| Prompt | `ids[:answer_at]` aus `data/quant_quality/cwe-4k/set.ids.json` — 4.054–4.064 Token |
| Dekodierung | **greedy** (`GEN` ohne Sampling-Schlüssel; der Engine-Default ist argmax, keine Strafterme), `ckpt=0` je Segment |
| Budget | **64 neue Token** oder das Ende-Token des Modells, was zuerst kommt |
| Engine | `bench/configs/cwe4k-swift-iq3_xxs.json`, ein Engine-Start je Lauf, alle acht Segmente darin |
| Prompt-Pfad | `--short-read 64` (Engine- und App-Default), `--prefill 512` (Chunk des gepinnten Configs) |
| Build | `build-hip-port\strata.exe`, HIP, ROCm 7.1, gfx1200 — derselbe Build wie G2 |

**Warum 64 Token:** die Gold-Antwort ist 32–36 Token lang, 64 ist die Antwort plus Luft. Ein falsches
Wort trifft ein Gold-Wort mit 10/2.000 Wahrscheinlichkeit (2.000 gezogene Wörter, 10 davon Gold), also
erwartet 0,15 Fehltreffer je Segment über die rund 30 Token Luft — dieselbe Größenordnung für jeden Arm.

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
| `int8` zweimal, zwei getrennte Engine-Starts | dieselben 8 × 64 Token-Ids | die Armvergleiche warten, bis die Ursache benannt ist; die Nichtdeterminismus-Breite wird berichtet |
| `int8` ohne `--spec`/`--spec-min-p` | dieselben Ids wie mit (der gepinnte Config dekodiert spekulativ) | siehe Nachtrag: für dieses Modell nicht durchführbar, ersetzt durch `--spec 2` gegen `--spec 4` |

Die zweite Prüfung ist keine Formsache: spekulative Dekodierung ist nur dann verlustfrei, wenn die
Verifikation das greedy-Ergebnis exakt reproduziert. Weicht sie ab, ist das eine Eigenschaft des
Produkts, die in den Bericht gehört.

## Sättigungs- und Bodenregel (vorab)

Der Anker entscheidet, ob das Instrument überhaupt etwas sieht. Beides wird **einmal** verschärft:

| Befund am Anker `fp16` | Lesart | eine Verschärfung |
|---|---|---|
| Mittel `recall` ≥ 9,5/10 | Decke: die Aufgabe ist zu leicht | neu bauen mit `--seq-tokens 8192` (gleicher Seed, gleiche Alpha), dann nur `fp16` + `int8` |
| Mittel `recall` ≤ 2,5/10 | Boden: die Aufgabe ist zu schwer | neu bauen mit `--seq-tokens 2048`, dann nur `fp16` + `int8` |
| dazwischen | das Instrument hat Auflösung | alle sieben Arme werden berichtet |

Bleibt es nach der Verschärfung bei Decke oder Boden, hat G3 **keinen Gegenstand**; dann wird G4
verschoben und nicht ergebnislos gestartet — so steht es im Kill der Stufe. Die Verschärfung läuft nur
mit `fp16` und `int8`, weil sie nur die Frage „sieht das Instrument etwas?" beantwortet; die sieben
Arme laufen dort, wo die Antwort ja ist.

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
- **`recall` ist grob.** Es zählt Wörter, nicht Reihenfolge, nicht Sicherheit, nicht Begründung.
- **Greedy ist eine Stichprobe der Größe eins** aus der Antwortverteilung des Modells. Deterministisch
  heißt nicht repräsentativ.
- **Ein entarteter Lauf zählt wie er ist.** Wiederholt sich ein Arm in eine Schleife, ist das ein
  funktionaler Verlust und wird nicht durch eine Umschreibung des Budgets repariert.
- **Nur HIP gebaut.** CUDA, SYCL und Metal sind hier nicht gemessen.
- **Der Fenster-Arm ist nicht der S3-Arm**, siehe oben: `S + R + 512` statt `S + R + 1`.

## Kosten

Gemessen an einem 4.060-Token-Prompt mit 64 neuen Token auf `fp16`, HIP, gfx1200: **32,3 s** je
Segment. Acht Segmente und ein Engine-Start (rund 14 s) sind also **rund 4,5 min je Lauf**; acht Läufe
plus die beiden Gate-Läufe rund **45 min GPU**. Die Auswertung ist offline.

## Dateien

| Datei | Inhalt |
|---|---|
| `tools/kv_g3_run.py` | ein Engine je Lauf, `GEN` je Segment, schreibt Ids und Text |
| `tools/kv_g3_score.py` | `recall`, `recall_first10`, gepaarte Differenzen, Sättigungs-/Bodenurteil |
| `data/quant_quality/cwe-4k/gold.json` | die zehn Gold-Wörter je Segment |
| `data/quant_quality/cwe-4k/set.ids.json` | Prompt-Ids und `answer_at` |

## Nachtrag (2026-10-31, vor dem Anker-Lauf)

**Die zweite Gate-Prüfung ist nicht durchführbar.** Ein nativer IQ-Pack verlangt `--spec T` mit `T >= 2`;
die Engine verweigert ohne das den Start (gemessen: der `int8`-Lauf ohne `--spec`, Exit 2, stderr
`strata generate: … is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) and --prefill
CHUNK`; statische Analyse dazu: `src/program/generate.cpp:3752-3757` prüft `o.spec < 2`). Ein Lauf ohne
spekulative Dekodierung existiert für dieses Modell also nicht.

Ersetzt wird sie durch **`int8` mit `--spec 2` gegen den gepinnten `--spec 4`**, gleiche Ids erwartet. Das
ist schwächer als geplant: es zeigt, dass die Fenstergröße der Verifikation die Token nicht ändert, nicht,
dass die Verifikation das greedy-Ergebnis eines ungedrafteten Laufs reproduziert. Der Bericht nennt es so.

**Gate 1 besteht:** `int8` zweimal in zwei getrennten Engine-Starts, 8 von 8 Segmenten mit denselben
Token-Ids und demselben Text.
