# Bewertung: „Beyond KL" – Eignung für unseren Vergleichsmaßstab

**Status: Bewertung (2026-10-28). Keine neue Messung außer der Decode-Sonde vom selben Tag; jede
Zahl ist zitiert oder als Rechnung markiert.** Gegenstand:
`docs/research/BEYOND_KL_QUANT_FIDELITY_ASSIGNMENT.md` (der Auftrag, wörtlich) und
`docs/research/BEYOND_KL_QUANT_FIDELITY_PLAN.md` (der Begleitplan). Ergebnis dieser Bewertung ist
der überarbeitete Arbeitsplan in `docs/research/BEYOND_KL_QUANT_FIDELITY.md`.

## 0. Kurzurteil

Der Auftrag fragt das Richtige, und seine Prinzipien (§13) sind unsere: keine Metrik ohne gemessenen
Zusatznutzen, negative Ergebnisse zählen, Pre-Registration, keine Architektur aus dem Aufgabentext
ableiten. Aber er ist als Programm für ein Labor mit Benchmark-Suite geschnitten — nicht für einen
Branch mit einer GPU, acht Segmenten und fünfzehn Arm-Konfigurationen.

**Als Landkarte gut. Als Arbeitsplan etwa eine Größenordnung zu groß, und an drei Stellen am
falschen Objekt.** Sein kritischer Pfad ist nicht die Metrik, sondern das Ground Truth — und das ist
bei uns näher, als der Plan denkt.

## 1. Was der Auftrag richtig sieht — und was wir schon gemessen haben

Der stärkste Teil ist Auftragsteil B („Failure Analysis of Existing Metrics"). Er ist bei uns zur
Hälfte **gemessen**, nicht behauptet:

| Auftragsbehauptung | unser Befund | Quelle |
|---|---|---|
| A: Mittelwertverdünnung | die schlimmsten 1 % der Positionen tragen 85,5 / 81,6 / 70,9 % der KL (`int8` / Fenster / plain) | `bench/results/2026-10-25-e4-error-shape/` |
| B: fehlende Entscheidungsrelevanz | 0,7–1,6 % Kopf-Substitutionen tragen ~56 % der `int8`-Summe | ebd. |
| CB4: PPL invertiert die Rangfolge | 2,4052 / 2,4079 / 2,4029 gegen fp16 2,4149, bei durchweg positiver KL | `Ansatz_Improvements.md` §4 |
| Ankerabhängigkeit | die LS-Skala sah gegen `int8` +0,0028 „besser" aus, gegen `fp16` bleiben +0,0003 | S3-Bericht, M11 |
| CB5: Definitionsartefakt im Tail | top-64 → top-256 hebt das Niveau um +5,1 / +6,4 / +6,4 %, die Rangfolge bleibt | S0-Bericht, M1/M7 |
| CB6: zeitliche Struktur unsichtbar | Beweis, keine Messung — und der Ring-Lauf zeigt, dass Regionslabels genau das erklären | `2026-10-28-ring-geometry` |
| Auflösung | 0,0020 nats einseitig bei echter Parität, 32.746 Positionen | S3-Bericht, M10 |

Dazu die Decode-Sonde vom 2026-10-28: volles Vokabular, 32 Prompts, `int8` 0,00055, der
Fensterkandidat 0,00225. Die Parität zum Kandidaten ist **nicht bestätigt, nur nicht ausgeschlossen**
(+0,0017, Intervall berührt die 0, 6/8 Segmente schlechter). Das ist genau der Fall, den der Auftrag
„Fall E: unnötiger Fehlalarm" nennt — und wir können ihn heute nicht auflösen.

Damit ist Auftragsteil B aus vorhandenen Messungen beantwortbar. Der Plan sagt das selbst (F0
„katalogisiert nur") und hat recht.

## 2. Wo der Auftrag für diesen Tree zu groß oder falsch geschnitten ist

1. **§10 Armtabelle: drei von sieben Zeilen existieren nicht.** `K8V4-NL` gibt es in diesem Tree
   nicht (kein Flag, kein `--kv`-Wert, kein Kernel). `K8V4` existiert als Format (816 B je Zelle),
   wurde aber nie auf cwe gemessen. „QSA-aware Varianten" sind kein Arm. Der Auftrag verlangt
   selbst, zu prüfen, was implementiert ist — das leert die Tabelle.
2. **§4.4 C/D (quantisierter Indexer) ist nicht ausführbar.** Der Indexer ist FP32 per Vertrag, und
   seine Keys werden aus dem Hidden State `x` projiziert, nicht aus dem K/V-Cache (statische
   Analyse, `src/core/layer.cpp:1351-1358`). Für die erste QSA-Schicht existiert der direkte Weg
   „KV-Fehler → andere Selektion" damit nicht; ab der zweiten ist er indirekt. Die QSA-Dimension ist
   ein **Instrumentierungsprojekt**, keine Metrikfrage.
3. **§8.1 Full-Vocabulary pro Position: ~32 GB je Arm** (32.746 × 248.320 × 4 B, Rechnung). Der
   Engine-Deckel ist 256. „Research Mode" ist ein Engine-Projekt, kein Analyse-Modus.
4. **§6 QBF als Score ist bei uns nicht fittbar.** Fünfzehn Arm-Konfigurationen in drei
   Build-Clustern und acht Segmente tragen kein gelerntes Modell, und §13.2 verbietet willkürliche
   Gewichte. Verteidigbar ist ein Dashboard mit Regeln — was der Auftrag selbst als einfachste
   Lösung bezeichnet.
5. **§7.1 zehn Benchmark-Suiten sind die falsche Reihenfolge.** Wir brauchen **eine**, die trennt.
   Die Nadel ist gesättigt (18/18); neun weitere Suiten zu bauen, bevor man weiß, ob irgendeine
   trennt, ist Arbeit ohne Rückmeldung.

## 3. Der kritische Pfad ist das Ground Truth — und näher als der Plan denkt

Der Plan behandelt F2 (Free-Running-Suite) als Blocker und listet sechs Task-Familien zum Bauen.
Aber `data/quant_quality/cwe-4k` ist kein Code-Korpus, sondern **RULER FWE**: acht Segmente,
~2.000 gezogene Wörter, **zehn Gold-Wörter** bei Position ~4.060. `set.ids.json` trägt `answer_at`,
`gold.json` die zehn Wörter.

Das funktionale Ereignis ist damit von den Daten schon definiert: die ersten `answer_at` Token als
Prompt, greedy generieren, zählen, wie viele der zehn Gold-Wörter vorkommen. Acht Prompts mal N
Arme, Minuten GPU je Arm. Das kann sättigen — dafür ist die Sättigungsregel da —, aber es ist der
billigste mögliche Test der Frage „unterscheiden sich die Arme funktional überhaupt?", und er sollte
laufen, **bevor** eine Metrik entworfen wird.

**Wichtige Abgrenzung:** die Decode-Sonde ist **nicht** dieser Test. Ihre Prompts sind die ersten
256/512/768/1.000 Token — Präfixe der Passage, nicht die Retrieval-Frage bei 4.060. Sie misst den
Decode-Pfad, nicht die Aufgabe. Und ihre Prompts sind zu kurz, als dass der Ring verdrängt: bei
1.000 Token liegt der größte Teil des Kontexts noch im Ring.

## 4. Was zu übernehmen ist, nach Wert pro Aufwand

### Stufe 1 — offline, kein GPU, aus den TSVs die wir haben

- **ΔNLL auf dem Ziel-Token.** `logp(target)` liegt für beide Arme an jeder Position vor und wurde
  nie benutzt. Das ist §3.3 und die einzige Größe, die „sagt das Modell den echten Text schlechter
  voraus" direkt misst, unabhängig vom Argmax.
- **Margin-bedingte Flips.** Aus den top-256-Paaren: `m_t = lp₁ − lp₂`. „Flip ∧ m ≥ 1 nat" trennt
  harmlose Gleichstände von echten Verdrängungen (§3.1).
- **Regionslabels.** Position × S/R/B aus `.seg.json` → Sink / Horizontkante / recent / Ring /
  outside. Das Distanzprofil des Ring-Laufs hat gezeigt, was das leistet: es hat den
  0,0039-Effekt verortet, den der Aggregatwert allein nicht platzieren konnte.
- **CVaR99 plus Floor-Ereignisrate.** p99 ist floor-dominiert; CVaR plus getrennte Floor-Zählung ist
  die ehrliche Tail-Zusammenfassung.

### Stufe 2 — ein Engine-Schritt

- **Full-Vocabulary-KL auf einer geschichteten Stichprobe** (Top-KL + zufällig + Flip-Positionen),
  nicht auf allen 32.746. Das ist der einzige Weg, die Definitionsfrage M1/CB5 zu erledigen, und
  kostet ~1 GB je Arm statt 32 GB. Die Decode-Sonde liefert das schon für 32 Prompts — aber auf dem
  Decode-Pfad, nicht an den teacher-forced-Positionen, an denen alle unsere Vergleiche hängen.

### Stufe 3 — das Ground Truth

- Das FWE-Ereignis, vorregistriert, mit Sättigungsregel.

## 5. Was der Maßstab nicht kann — und der Test, den wir schon haben

**Die Auflösung.** 0,0020 nats bei 32.746 Positionen ist eine Eigenschaft des Korpus (gepaarte
Streuung 0,22), nicht der Metrik. Eine bessere Metrik auf denselben Positionen hat dieselbe Grenze
für die Frage „ist der Arm schlechter"; sie kann sagen *wo* und *wie*, nicht *ob ein 0,001-Effekt
real ist*. Der Wert des Programms ist **Zuordnung und Frühwarnung, nicht Empfindlichkeit**. Wer sich
davon auflösbare 0,001-Effekte verspricht, wird enttäuscht — und das ist keine Metrik-Schwäche,
sondern eine Korpus-Größe.

Und wir haben einen Kalibrierungstest, den kein synthetisches Gegenbeispiel ersetzt: **bekannte
Paare aus eigenen Messungen**.

| Paar | Wahrheit | Was ein Kandidat sagen muss |
|---|---|---|
| `q4lm32w` vs `int8` | Parität (+0,00002 top-64 / +0,00034 top-256; `int8` führt 4/8 bzw. 5/8) | Parität |
| `s320r0` vs 256/64 | Verlust (+0,00394 [+0,00097, +0,00691], der Arm besser in 0/8) | Unterschied, richtige Richtung |
| 16/8 vs 8/16 | kein Unterschied (−0,00030, Intervall um 0) | keinen erfinden |
| Leiter: 8/8 0,04127 > 8/16 0,03212 ≈ 16/8 0,03182 > 16/16 0,02512 > 16/16+Fenster 0,01689 ≈ `int8` 0,01688 | Ordnung | Ordnung erhalten |

Kein Validierungsnachweis — es fehlt das funktionale Ground Truth —, aber ein harter Filter: ein
Merkmal, das auf dem Paritätspaar einen Unterschied meldet oder die Leiter umdreht, ist Rauschen
oder nur umskalierte KL. Genau der „unnötige Fehlalarm" aus §14 Fall E, operationalisiert auf echten
Daten. Vier Checks, kostenlos, aus Daten die wir haben.

## 6. Empfehlung

Nicht das Programm ausführen, sondern drei Sätze daraus:

1. **Anker und Auflösung immer mitliefern** (M11/M10). Das ist keine neue Metrik, sondern die
   Bedingung dafür, dass jede Metrik vergleichbar ist — kostenlos, und behebt den größten gemessenen
   Fehler der Vergangenheit.
2. **Drei Offline-Merkmale** (ΔNLL, Margin-Flip, Region) gegen die vier Kalibrierungspaare prüfen;
   was die Leiter umdreht oder das Paritätspaar trennt, fällt sofort raus. Danach Redundanzmatrix
   gegen Mean KL: |ρ| ≥ 0,9 ohne Zusatz → nicht aufnehmen.
3. **Das FWE-Ereignis** als Ground Truth. Erst danach die Frage „sagt ein Merkmal den Ereignisverlust
   besser voraus als Mean KL" — und mit acht Prompts mal sechs Armen wird die Antwort schwach sein.
   Das zu sagen ist ein gültiges Ergebnis (§13.10).

**Nicht bauen:** QBF-Score, QSA-Selektionsmetriken (vor der Instrumentierung), Zehn-Suiten-Matrix,
Full-Vocab pro Position, K8V4-NL.

## 7. Der wichtigste Einzelpunkt

Der Auftrag behandelt „KL gegen bessere Metrik" als Achse. Unsere Messgeschichte sagt, dass die
dominante Fehlerquelle **definitionell** ist: top-64-Truncation mit Floor-Substitution, Supportbreite,
Ankerwahl. Eine neue Metrik auf demselben trunkierten Support erbt alle drei.

Deshalb ist Schritt null nicht eine neue Metrik, sondern ein **Full-Vocabulary-Anker auf einer
Stichprobe**. Sonst werden alle Kandidaten durch dieselbe verzerrte Linse verglichen, und der
Vergleich zwischen ihnen ist konfundiert — man misst dann, welches Merkmal am besten mit dem
Artefakt harmoniert.

## 8. Verhältnis zu den anderen Dokumenten

| Dokument | Rolle |
|---|---|
| `BEYOND_KL_QUANT_FIDELITY_ASSIGNMENT.md` | der Auftrag, wörtlich und unverändert. Bleibt so. |
| `BEYOND_KL_QUANT_FIDELITY_PLAN.md` | Begleitplan und **Inventar** (M1–M12, CB1–CB7, Rohdatenlage, Build-Cluster). Bleibt als Herkunftsnachweis stehen. |
| `BEYOND_KL_QUANT_FIDELITY_ASSESSMENT.md` | dieses Dokument: warum der Plan nicht eins zu eins der Arbeitsplan ist. |
| `BEYOND_KL_QUANT_FIDELITY.md` | der überarbeitete Arbeitsplan: was wir tatsächlich tun, in welcher Reihenfolge, mit welchen Kills. |

## Quellen

- `bench/results/2026-10-27-s3-level-split/README.md` — neun Arme, Anker, Auflösung, Byte-Modell
- `bench/results/2026-10-28-ring-geometry/README.md` — Sinks, Distanzprofil, Prefill/Decode-Grenze
- `bench/results/2026-10-28-decode-probe/README.md` — volles Vokabular, Decode-Pfad
- `bench/results/2026-10-25-e4-error-shape/README.md` — Fehlerform, Kopf/Tail
- `bench/results/2026-10-25-s0-kl-support/README.md` — Supportbreite, top-64/256
- `docs/paper/Ansatz_Improvements.md` §7 — K1–K6, PPL-Inversion, Ziel-Zurückstellung
- `docs/research/BEYOND_KL_QUANT_FIDELITY_PLAN.md` §1.4 — M1–M12 mit Fundstellen
