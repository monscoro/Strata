# G1: neun Merkmale gegen drei Kalibrierungspaare

**Status: vorregistriert (2026-10-28), Läufe sind Analyse vorhandener Daten, kein GPU-Job.**
Katalogzeile: `docs/paper/Ansatz_Improvements.md`, Arbeitsplan G1
(`docs/research/BEYOND_KL_QUANT_FIDELITY.md` §3 und §4).

Dieses Dokument legt **vor** der Auswertung fest, was gerechnet wird, wie gepaart wird und wann ein
Merkmal fällt. Die Daten liegen seit dem 2026-10-25/27/28 in `Strata-data/bench/` und wurden für
andere Fragen schon ausgewertet; neu ist nur, dass dieselben Positionen gegen die Kalibrierungspaare
gestellt werden.

## Frage

Tragen ΔNLL, Margin-Flip und Region etwas, das Mean KL nicht schon sagt? Und wenn nicht: reichen
Mean KL und P99, wie der Arbeitsplan es als mögliches Ergebnis vorsieht?

## Daten

Zwanzig Läufe, fünfzehn Konfigurationen, 32.746 Positionen je Arm, `cwe-4k`, teacher-forced,
`--topk 256`. Anker ist `fp16` (`2026-10-27-s3/fp16.tsv`). Die Zuordnung Name → Datei steht in
`bench/fixtures/beyond-kl-calibration.json`, weil die Rohdaten außerhalb des Repos liegen.

| Verzeichnis | Arme |
|---|---|
| `2026-10-25-s0` | `int8`, `plain`, `win256`, `fp16` |
| `2026-10-27-s3` | `fp16`, `int8`, `q4lm32`, `q4lm32s`, `q4lm32sw`, `q4lm32w`, `q4lm8`, `q4lm8k`, `q4lm8v` |
| `2026-10-28-ring` | `s320r0`, `s192r128`, `s64r256` |
| `2026-10-25-tag-quality` | `int8`, `tag`, `win256`, `fp16` |

Für die Paare wird durchweg der **S3-Satz** benutzt (gleicher Build, gleiche Positionen). `int8`
existiert dreimal; die drei Dateien sind gleich groß, aber nur die S3-Datei ist mit der Leiter
zusammen entstanden.

## Definitionen (je Position q, Referenz `fp16`)

| Merkmal | Definition | Einheit |
|---|---|---|
| `kl64` / `kl256` | `KL(fp16‖Arm)` über die top-64 / top-256 der Referenz, Floor-Substitution wie in `kl_between` | nats |
| `dnll` | `logp_fp16(y) − logp_Arm(y)`, `y` = das Text-Token. Positiv = Arm schlechter | nats |
| `flip` | 1, wenn `argmax(Arm) ≠ argmax(fp16)` | Rate |
| `flip_m1` | 1, wenn `flip` **und** der Referenz-Margin `m = lp₁ − lp₂` ≥ 1 nat | Rate |
| `floor` | 1, wenn die von der Referenz auf die Substitution umgelegte Masse `Σ p_ref(K_ref \\ K_arm)` ≥ 0,01 ist | Rate |
| `cvar99` | Mittel der schlechtesten 1 % der `kl256`-Werte eines Segments | nats |
| `p99` | 99. Perzentil der `kl256`-Werte eines Segments | nats |
| `region_excess` | Mittel `kl256` **im verdrängten** Bereich `[R, n−S)` minus Mittel **im exakten** (Sinks `q < R` plus letzte `S`) | nats |

**Zur Region, offen gesagt:** das Label ist eine **Positionsfunktion**, keine Eigenschaft des Arms.
`R = 64`, `S = 256` sind die Geometrie des Referenz-Fensterarms; `n` ist die Segmentlänge. Damit ist
„außerhalb" bei festem `n` gleichbedeutend mit „weiter hinten", und `region_excess` kann sich als
bloß umskalierte Position erweisen. Genau das prüft die Redundanzmatrix. Ein Arm ohne Fenster
bekommt dasselbe Label — dann ist es ein reiner Distanz-Split, und das steht im Ergebnis.

**Eine Definition wurde nach dem ersten Probelauf korrigiert, und zwar bevor ein Merkmal bewertet
war.** Die erste Fassung von `floor` fragte, ob die top-256-Menge der Referenz in der des Arms
enthalten ist. Das ist auf den ersten zwei geprüften Armen an **100 %** der Positionen wahr: zwei
Listen zu je 256 aus 248.320 Token stimmen praktisch nie als Menge überein. Eine Konstante kann
nicht diskriminieren, also misst die Definition jetzt die **Masse**, die auf die Substitution
umgelegt wird, und zählt Positionen ab 1 %. Das ist eine Korrektur der Definition, kein Ergebnis —
kein Merkmal war zu diesem Zeitpunkt gegen ein Paar gerechnet.

## Aggregation und Statistik

- **Einheit ist das Segment** (n = 8), nie die Position. Je Segment wird der Merkmalswert gemittelt
  (bei Raten: der Anteil), dann werden die acht Segmentwerte gepaart.
- Gepaarte Differenz `d_s = F(A)_s − F(B)_s`; `t` mit df = 7; 95-%-Intervall mit `t_{0.975,7} =
  2,365`; einseitige Grenze bei echter Parität `1,895 · sd/√8`.
- Ein Merkmal ist **aufgelöst** für ein Paar, wenn `|Δ|` die einseitige Grenze überschreitet.

## Die Prüfregel (unverändert aus dem Arbeitsplan §3)

Ein Merkmal wird nur aufgenommen, wenn es **alle vier** besteht:

| Paar | Wahrheit | Prüfung |
|---|---|---|
| `q4lm32w` vs `int8` | Parität (+0,00002 top-64 / +0,00034 top-256; `int8` führt 4/8 bzw. 5/8) | 95-%-Intervall enthält die 0 |
| `s320r0` vs `q4lm32w` | Verlust (+0,00394 [+0,00097, +0,00691] top-256, Arm besser in 0/8) | Δ > 0 **und** Intervall schließt die 0 aus |
| `q4lm8v` vs `q4lm8k` | kein Unterschied (−0,00030, Intervall um 0) | 95-%-Intervall enthält die 0 |
| Leiter `q4lm8` > `q4lm8k` ≈ `q4lm8v` > `q4lm32` > `q4lm32w` ≈ `int8` | Ordnung (top-64: 0,04127 / 0,03212 / 0,03182 / 0,02512 / 0,01689 / 0,01688) | jede der fünf Stufen richtig: `>` wenn Δ über der Grenze, `≈` wenn darunter |

Danach die **Redundanzprüfung**: `|Spearman ρ| ≥ 0,9` gegen Mean KL über die fünfzehn
Konfigurationen → nicht aufnehmen, es sei denn der Zusatzbeitrag ist in der Partialbetrachtung
nachweisbar.

## Was ein Nullresultat heißt

Fällt jedes Merkmal, lautet das Ergebnis: **Mean KL und P99 reichen für die Fragen, die dieser Korpus
beantworten kann**, und das Dashboard entfällt. Das ist ein gültiges Ergebnis (Auftrag §13.10), kein
Fehlschlag — und es ist nach Lage der Daten das wahrscheinlichere, weil die Auflösung 0,0020 nats
beträgt und die Merkmale auf denselben Positionen mit derselben Streuung leben.

## Was diese Stufe nicht kann

- **Kein funktionales Ground Truth.** Alle vier Paare sind KL-Aussagen. Ein Merkmal, das sie besteht,
  ist damit kalibriert, nicht validiert. Das ist G3/G4.
- **Keine Auflösung unter 0,0020.** Ein Merkmal kann die Grenze nicht verschieben, nur die Zuordnung
  verbessern.
- **Trunkierung.** Alle Merkmale außer `dnll` leben auf der top-256-Stütze und erben M1/CB5. Der
  Vollvokabular-Anker ist G2, nicht G1.
- **Acht Segmente.** Jedes „besteht" heißt: in dieser Stichprobe, mit df = 7.

---

# Ergebnis (2026-10-28)

**Kein neues Merkmal besteht die Prüfregel.** Von den neun gerechneten Merkmalen bestehen nur `kl64`
und `kl256` alle vier Prüfungen — also genau die beiden, die schon die Baseline sind. Damit ist der
in der Vorregistrierung genannte Fall eingetreten: **Mean KL bleibt die Baseline, ein Dashboard
entfällt.** Es fällt aber nicht so aus, wie die Vorregistrierung es als wahrscheinlichsten Fall
beschrieben hat — P99 besteht die Regel ebenfalls nicht (siehe unten), und zwei Merkmale sind
nicht redundant, obwohl sie fallen.

Der Arbeitsplan nennt drei Merkmale (`dnll`, Margin-Flip, Region); die Definitionstabelle oben listet
neun, und alle neun standen vor dem Lauf fest: `kl64`/`kl256` als Baseline, `p99`/`cvar99` als die
bisherigen Tail-Zahlen, `floor`/`flip_m1` als Ratenvarianten. Der Titel dieses Dokuments nannte
lange nur die drei aus dem Plan; er ist nach dem Lauf berichtigt.

## Der Lauf

`python tools/kv_g1.py --json …`, **248 s**, kein GPU-Job, 17 Läufe plus Anker, 32.746 Positionen je
Arm, top-256. Die Rohdaten liegen in `Strata-data/bench/`, die Zuordnung im Fixture.

Zwei Vorabprüfungen, die vor jeder Zahl stehen:

- **Fixture:** alle 11 gepinnten Armmittel reproduzieren innerhalb 2e-05. Die Läufe sind also
  dieselben, gegen die S3 und der Ring-Lauf gerechnet haben.
- **Doppelläufe:** `int8_s0`, `int8_tag` und `win256_tag` sind dieselbe Konfiguration in einem
  anderen Verzeichnis. Alle neun Merkmale stimmen auf **0,0** überein. Dasselbe gilt für `plain`
  (S0) und `q4lm8v` (S3) — dieselbe `--kv q4lm8v`, zwei Sitzungen zwei Tage auseinander. Drei
  Konfigurationen, sieben Läufe, jedes Paar exakt aufeinander: der Messpunkt ist deterministisch,
  und die Zahlen aus verschiedenen Berichten sind vergleichbar.

## Die Arme

Anker `fp16`; gegen sich selbst sind alle Merkmale 0 und die Zeile steht nicht in der Tabelle.

| Arm | kl64 | kl256 | dnll | flip | flip@1nat | floor | p99 | cvar99 | region_excess |
|---|---|---|---|---|---|---|---|---|---|
| `int8` | 0,01688 | 0,01774 | +0,00092 | 0,95 % | 0,20 % | 1,2 % | 0,2772 | 1,4250 | −0,02111 |
| `q4lm8` | 0,04127 | 0,04414 | +0,02031 | 1,91 % | 0,62 % | 2,3 % | 1,0299 | 2,5970 | −0,04623 |
| `q4lm8k` | 0,03212 | 0,03417 | +0,00980 | 1,48 % | 0,41 % | 1,9 % | 0,7488 | 2,2592 | −0,04164 |
| `q4lm8v` | 0,03182 | 0,03387 | +0,01075 | 1,60 % | 0,42 % | 2,2 % | 0,7401 | 2,1410 | −0,03170 |
| `q4lm32` | 0,02512 | 0,02667 | +0,00234 | 1,26 % | 0,27 % | 1,7 % | 0,5314 | 1,9197 | −0,01527 |
| `q4lm32s` | 0,02420 | 0,02634 | −0,00676 | 1,18 % | 0,24 % | 1,6 % | 0,4912 | 1,9645 | −0,01565 |
| `q4lm32sw` | 0,01881 | 0,02017 | +0,00048 | 1,07 % | 0,22 % | 1,2 % | 0,3513 | 1,5687 | −0,01330 |
| `q4lm32w` | 0,01689 | 0,01809 | −0,00176 | 1,11 % | 0,24 % | 1,1 % | 0,3202 | 1,3715 | −0,01676 |
| `plain` | 0,03182 | 0,03387 | +0,01075 | 1,60 % | 0,42 % | 2,2 % | 0,7401 | 2,1410 | −0,03170 |
| `win256` | 0,02090 | 0,02223 | +0,00041 | 1,16 % | 0,29 % | 1,3 % | 0,3967 | 1,6707 | −0,02106 |
| `tag` | 0,01964 | 0,02107 | −0,00068 | 1,09 % | 0,27 % | 1,3 % | 0,3802 | 1,6267 | −0,02317 |
| `s320r0` | 0,02053 | 0,02203 | +0,00027 | 1,11 % | 0,24 % | 1,3 % | 0,3959 | 1,6859 | −0,01272 |
| `s192r128` | 0,01757 | 0,01871 | −0,00208 | 1,05 % | 0,24 % | 1,2 % | 0,3638 | 1,4183 | −0,00940 |
| `s64r256` | 0,01882 | 0,01995 | −0,00198 | 1,02 % | 0,23 % | 1,2 % | 0,3238 | 1,5541 | −0,00961 |

## Die Prüfregel, Merkmal für Merkmal

Parität = `q4lm32w − int8`, zertifizierter Verlust = `s320r0 − q4lm32w`, kein Unterschied =
`q4lm8v − q4lm8k`, jeweils mit 95-%-Intervall über die acht Segmente (t, df = 7). „Leiter" = wie viele
der fünf Stufen richtig fallen. `ρ` = Spearman gegen Mean KL über die 15 Konfigurationen inklusive
Anker.

| Merkmal | Parität | zert. Verlust | kein Unterschied | Leiter | ρ | Urteil |
|---|---|---|---|---|---|---|
| `kl64` | +0,00002 [−0,00183, +0,00186] ✓ | +0,00364 [+0,00091, +0,00637] ✓ | −0,00030 [−0,00462, +0,00402] ✓ | 5/5 | +0,996 | **besteht** |
| `kl256` | +0,00034 [−0,00142, +0,00211] ✓ | +0,00394 [+0,00096, +0,00693] ✓ | −0,00030 [−0,00532, +0,00472] ✓ | 5/5 | — | **besteht** |
| `dnll` | −0,00268 [−0,00401, −0,00135] ✗ | +0,00203 [−0,00404, +0,00811] ✗ | +0,00095 [−0,00967, +0,01157] ✓ | 2/5 | +0,642 | fällt |
| `flip` | +0,00156 [+0,00018, +0,00294] ✗ | +0,00006 [−0,00070, +0,00082] ✗ | +0,00128 [−0,00090, +0,00347] ✓ | 4/5 | +0,950 | fällt |
| `flip_m1` | +0,00040 [−0,00028, +0,00108] ✓ | +0,00003 [−0,00025, +0,00031] ✗ | +0,00009 [−0,00060, +0,00078] ✓ | 4/5 | +0,932 | fällt |
| `floor` | −0,00064 [−0,00146, +0,00018] ✓ | +0,00195 [+0,00128, +0,00263] ✓ | +0,00281 [+0,00164, +0,00397] ✗ | 4/5 | +0,971 | fällt |
| `p99` | +0,04309 [−0,00995, +0,09613] ✓ | +0,07562 [−0,00177, +0,15302] ✗ | −0,00874 [−0,10301, +0,08553] ✓ | 4/5 | +0,989 | fällt |
| `cvar99` | −0,05349 [−0,26514, +0,15816] ✓ | +0,31441 [+0,05743, +0,57140] ✓ | −0,11822 [−0,46031, +0,22388] ✓ | 4/5 | +0,982 | fällt |
| `region_excess` | +0,00435 [−0,00157, +0,01028] ✓ | +0,00404 [−0,01128, +0,01936] ✗ | +0,00994 [−0,00494, +0,02482] ✓ | 2/5 | −0,707 | fällt |

## Warum jedes Merkmal fällt

**`dnll` scheitert an der Parität, und zwar mit dem falschen Vorzeichen.** Auf dem Paar, das KL als
gleich erklärt, ist der Fensterarm beim Text-Token **besser** (−0,00268, Intervall ganz unter 0). Das
ist kein Rauschen, sondern ein Muster über mehrere Arme: negativ sind `q4lm32s` (−0,00676), `s192r128`
(−0,00208), `s64r256` (−0,00198), `q4lm32w` (−0,00176) und `tag` (−0,00068); positiv ist jeder Arm der
max-abs-Familie ohne Fenster (`q4lm32` +0,00234, `q4lm8k` +0,00980, `q4lm8` +0,02031). Auf der Leiter
fällt `dnll` an drei Stufen (2/5), weil zwei „>"-Stufen unter der Auflösung bleiben und die letzte ins
„<" kippt.

**Ein einfaches Muster ist es aber nicht, und das gehört dazu.** Das Fenster allein erklärt das
Vorzeichen nicht: es dreht `q4lm32` von +0,00234 auf −0,00176 (`q4lm32w`) und `q4lm32s` von −0,00676
auf +0,00048 (`q4lm32sw`) — einmal ins Negative, einmal ins Positive. Und `q4lm32s` liest gar keinen
Ring. Der Mechanismus ist damit **nicht** identifiziert; gemessen ist nur, dass ΔNLL und KL auf
denselben Positionen in der Richtung auseinanderfallen können.

**Das ist der strukturelle Vorbehalt, und er gehört hierher:** die Wahrheiten sind **KL-Aussagen**,
also gegen die `fp16`-Verteilung gerichtet; `dnll` ist gegen **das Text-Token** gerichtet. Die beiden
können in der Richtung auseinanderfallen, wenn eine Quantisierung die realisierte Fortsetzung besser
trifft und die Verteilung trotzdem weiter verschiebt. Ein Merkmal, das nicht KL ist, kann an einer
KL-Wahrheit scheitern, ohne falsch zu sein. Was hier gemessen ist, ist die **Untauglichkeit als
KL-Proxy**, nicht die Untauglichkeit als Qualitätsmaß — das wäre G3. Der Mechanismus (warum genau
diese Arme beim Text-Token gewinnen) ist **nicht** gemessen und bleibt offen.

**`flip` meldet auf dem Paritätspaar einen Unterschied** (+0,00156, 1,11 % gegen 0,95 %) und sieht
den zertifizierten Sink-Verlust **überhaupt nicht** (+0,00006, Intervall um 0). Ein Top-1-Wechsel
reagiert also auf die feine Bulk-Quantisierung und nicht auf den groben Sink-Fehler. `flip` ist
zusätzlich redundant (ρ +0,950).

**`flip_m1` repariert das eine und verliert das andere.** Die Margin-Bedingung (Referenz-Margin
≥ 1 nat) nimmt den Fehlalarm auf dem Paritätspaar heraus (+0,00040, Intervall enthält 0) — und damit
auch den zertifizierten Verlust (+0,00003, kein Effekt) und eine Leiterstufe. Das ist ein **Tausch,
keine Verbesserung**: die Bedingung schärft, indem sie Auflösung wegwirft. ρ +0,932.

**`floor` ist das einzige Merkmal, das nicht aus `kl256` gebaut ist und den Sink-Verlust sieht** (+0,00195
[+0,00128, +0,00263], Parität sauber) — und es sieht ihn auch dort, wo KL keinen hat: `q4lm8v` gegen
`q4lm8k` +0,00281 [+0,00164, +0,00397], obwohl KL dort −0,00030 sagt. Eine Substitutionsrate von
1,9 % gegen 2,2 % ist ein Unterschied in der Stütze, kein Unterschied im Fehler. Damit fällt `floor`
an der dritten Prüfung, und mit ρ +0,971 wäre es ohnehin keine neue Dimension.

**`p99` verliert um 1,4 %.** Die Parität besteht es (Intervall enthält 0), den zertifizierten Verlust
nicht (+0,07562 [−0,00177, +0,15302], Intervall berührt die 0), und auf der letzten Leiterstufe
meldet es einen Unterschied, wo Gleichstand verlangt ist: +0,04309 gegen eine einseitige Grenze von
0,04250. Das ist genau der Fehlalarm, den die Prüfregel abfangen soll — und er entsteht hier an der
**Auflösungsgrenze**, nicht an einem großen Effekt. ρ +0,989.

**`cvar99` besteht drei von vier Prüfungen** und fällt an der ersten Leiterstufe (`q4lm8` gegen
`q4lm8k`: +0,33775 gegen die Grenze 0,43302). Den zertifizierten Sink-Verlust sieht es mit klarem
Intervall (+0,31441 [+0,05743, +0,57140]) — deutlich, wo P99 ihn nicht sieht —, aber mit ρ +0,982
ordnet es die Arme wie Mean KL. Als **Rang**instrument bringt es nichts; als **Tail-Zahl neben** Mean
KL ist es die bessere Wahl als P99 (siehe unten).

**`region_excess` fällt am zertifizierten Verlust und an drei Leiterstufen — und der Grund ist die
Definition, nicht die Messung.** Das Merkmal ist für **alle 14 Arme negativ**, auch für die ohne
Fenster (`int8` −0,02111, `q4lm8v` −0,03170). Da KL nicht negativ werden kann, heißt das: die
als „exakt" gelabelte Seite hat **höheren** Fehler als die verdrängte Mitte. Das Label ist eine
Positionsfunktion (so stand es in der Vorregistrierung) und enthält am Ende die letzten 256
Positionen — dort hat ein Fensterarm die längste Historie und damit den größten aufgelaufenen Fehler,
während der Kopf fast keine Historie hat. Die verdrängte Mitte mischt alles dazwischen. Die
Vorregistrierung hat diesen Ausgang als Möglichkeit benannt; eingetreten ist der schlechtere Fall:
das Merkmal ist nicht einmal eine monotone Funktion der Distanz, es kippt auf der Leiter
(`q4lm8v` gegen `q4lm32` −0,01643 gegen eine Grenze von 0,00929).

## Zwei Merkmale sind nicht redundant — und genau die fallen

Die Redundanzprüfung ist der Grund, warum das Ergebnis nicht einfach „alles doppelt" lautet:

| Merkmal | ρ (15 inkl. Anker) | ρ (14 ohne Anker) | |
|---|---|---|---|
| `dnll` | +0,642 | +0,665 | neue Dimension |
| `region_excess` | −0,707 | −0,639 | neue Dimension |
| `flip_m1` | +0,932 | +0,916 | redundant |
| `flip` | +0,950 | +0,938 | redundant |
| `floor` | +0,971 | +0,965 | redundant |
| `cvar99` | +0,982 | +0,978 | redundant |
| `p99` | +0,989 | +0,987 | redundant |
| `kl64` | +0,996 | +0,996 | dieselbe Zahl, andere Stütze |

`dnll` und `region_excess` tragen also Information, die Mean KL nicht hat — sie sind nur **nicht
kalibriert**: `dnll`, weil es eine andere Frage beantwortet (Text-Token statt Verteilung), und
`region_excess`, weil sein Label nicht das trennt, was es trennen sollte. Das ist der Unterschied
zwischen „redundant" und „falsch", und er ist der einzige Ansatzpunkt, den G1 offenlässt.

## Zwei Nachträge zur Auswertungsmenge (kein Einfluss auf ein Urteil)

Beides betrifft nur, **über welche Punkte** die Redundanz gerechnet wird, nicht die Prüfregel:

1. Die Vorregistrierung sagt „fünfzehn Konfigurationen". Die drei Doppelläufe (`int8_s0`, `int8_tag`,
   `win256_tag`) sind dieselbe Konfiguration in einem anderen Verzeichnis; ein Duplikat ist eine
   Gewichtung, kein Arm. Gerechnet ist beides: über die 14 Arme ohne Anker und über alle 15
   inklusive Anker. Beide Zahlen stehen in der Tabelle oben und führen zum selben Urteil.
2. Die erste Fassung von `floor` (Mengen-Enthaltensein) war auf 100 % der Positionen konstant; die
   Korrektur auf die Substitutionsmasse steht in der Vorregistrierung und erfolgte **vor** der
   Bewertung eines Merkmals gegen ein Paar.

## Was das Ergebnis heißt

1. **Mean KL bleibt die Baseline**, in beiden Stützen. Kein Merkmal aus §3 des Auftrags ersetzt sie
   für die Fragen, die dieser Korpus beantworten kann.
2. **P99 bleibt nicht als zweite Zahl stehen.** Die Vorregistrierung hat den Fall „Mean KL und P99
   reichen" genannt; gemessen sieht P99 den einzigen zertifizierten Verlust nicht. Wenn eine
   Tail-Zahl daneben stehen soll, dann **CVaR99** — aber als Beschreibung, nicht als neue Dimension.
   Warum die beiden so weit auseinanderliegen, steht im Nachtrag unten.
3. **Kein Dashboard, kein Score.** Das ist das Ergebnis, das der Auftrag selbst als gültig bezeichnet
   (§13.10), und es ist die billigere Lösung.
4. **Die Prüfregel hat drei Fehlalarme abgefangen**, die eine reine „größerer Unterschied = bessere
   Metrik"-Lesart übernommen hätte: `flip` auf der Parität, `floor` auf dem Paar ohne Unterschied und
   `p99` an der Auflösungsgrenze.

## Was dieses Ergebnis nicht sagt

- **Nichts über Modellqualität.** Alle Wahrheiten sind KL-Aussagen. Ob `dnll` die funktional bessere
  Zahl ist, entscheidet **G3**, nicht G1. G1 sagt nur: es ist kein KL-Proxy.
- **Nichts unterhalb von 0,0020 nats.** Die Merkmale leben auf denselben Positionen mit derselben
  Streuung; keines kann die Auflösung verschieben.
- **Nichts über das Vollvokabular.** Alle Merkmale außer `dnll` erben die Trunkierung (M1/CB5). Ob
  die trunkierte KL die KL ist, prüft **G2**.
- **Acht Segmente.** Jedes Urteil gilt in dieser Stichprobe, mit df = 7. `dnll` mit ρ 0,642 ist eine
  Richtung, kein Nachweis einer neuen Dimension.

## Reproduktion

```sh
python tools/kv_g1.py --json <out>.json        # 248 s, kein GPU-Job
```

Fixture: `bench/fixtures/beyond-kl-calibration.json` (18 Läufe, 11 gepinnte Armmittel, drei Paare,
Leiter, drei Doppelläufe). Die Rohdaten liegen außerhalb des Repos unter `Strata-data/bench/`.

---

## Nachtrag: warum P99 und CVaR99 auseinanderfallen (nachträglich, deskriptiv)

**Dieser Abschnitt ist nach dem Ergebnis geschrieben und entscheidet nichts.** Die Prüfregel ist
angewandt, `cvar99` ist gefallen; hier steht nur, *warum* die beiden Tail-Zahlen auf denselben
Positionen so verschiedene Antworten geben. Werkzeug: `tools/kv_tail.py`, 58 s, kein GPU-Job.

| Arm | Mittel | P99 | CVaR99 | CVaR/P99 | Summe im schlechtesten 1 % | Summe im Kopf (q < 64) |
|---|---|---|---|---|---|---|
| `int8` | 0,01775 | 0,2772 | 1,4250 | 5,1 | 77,8 % | 3,0 % |
| `q4lm32w` | 0,01809 | 0,3202 | 1,3715 | 4,3 | 74,0 % | 2,0 % |
| `s320r0` | 0,02203 | 0,3959 | 1,6859 | 4,3 | 74,4 % | 1,6 % |
| `q4lm8k` | 0,03418 | 0,7488 | 2,2592 | 3,0 | 64,1 % | 2,7 % |
| `q4lm8` | 0,04415 | 1,0299 | 2,5970 | 2,5 | 57,1 % | 2,3 % |

Die Mittel weichen in der letzten Stelle von der Tabelle oben ab: hier steht das ungewichtete Mittel
über die acht Segmente, dort das positionsgewichtete. Die Segmente sind fast gleich lang.

**1. Der Fehler ist extrem kopflastig, und zwar umso mehr, je besser der Arm ist.** Im `int8`-Arm
tragen die schlechtesten 1 % der Positionen **77,8 %** der gesamten KL-Summe, im schlechtesten Arm
`q4lm8` noch 57,1 %. Das ist E4 auf Arnebene: ein guter Arm ist *meistens fast fehlerfrei und an
wenigen Stellen katastrophal*, ein schlechter ist gleichmäßig schlecht. Zwei Arme mit demselben
Mittel können also zwei verschiedene Verteilungen sein.

**2. P99 liest den Rand des Tails, CVaR den Tail.** P99 ist die 42.-schlechteste von ~4.095
Positionen je Segment, CVaR99 der Mittelwert der schlechtesten 41. Der Unterschied ist keine
Feinheit: bei `int8` liegen **81 %** des CVaR-Werts (1,4250 − 0,2772 = 1,1478) **oberhalb** der
P99-Linie, bei `q4lm8` noch 60 %. Eine Quantilzahl kann per Konstruktion nicht sehen, dass *mehr*
Positionen über einer alten Linie liegen — sie liest nur, wo die Linie steht.

**3. Genau das ist der zertifizierte Verlust.** `s320r0` gegen `q4lm32w`, an derselben Linie
gemessen (der P99 des Vergleichsarms, je Segment): **0,86 % → 1,16 %** der Positionen liegen
darüber, und die Masse darüber steigt von 0,0099 auf 0,0138 nats/Position. Die Zahl der Extremfälle
wächst also um gut ein Drittel, während der *Rand* nur um 0,0756 nats wandert. P99 sieht den Rand
(+0,0756, Intervall berührt die 0), CVaR99 sieht Rand **und** Masse (+0,3144, Intervall klar
darüber).

**4. Es ist nicht einfach Rauschen.** Die gepaarte Streuung ist für CVaR99 mit 0,3073 absolut
**3,3-mal größer** als für P99 (0,0926) — CVaR ist also nicht die „stabilere" Zahl. Der Effekt ist
aber 4,2-mal größer, deshalb t 2,89 gegen 2,31 und 8/8 gegen 7/8 Segmente. Der Unterschied liegt in
der Größe des Signals, nicht in seiner Streuung.

**5. Der Verlust sitzt nicht im Kopf.** Die ersten 64 Positionen tragen nur 1,6–3,0 % der KL-Summe.
Der Sink-Verlust ist kein Fehler *an* den Sink-Positionen, sondern einer an *jeder* Position, die die
Sink-Zeilen liest — deshalb hebt er den ganzen Tail leicht an, statt ein paar Positionen zu
explodieren. Das ist die plausible Erklärung dafür, dass `floor` (Substitutionsmasse) ihn sieht und
`flip` nicht; der Mechanismus selbst ist hier nicht nachgewiesen.

**Was daraus folgt, und was nicht.** P99 ist als zweite Zahl die falsche Wahl, CVaR99 die richtige —
aber nur als **Beschreibung der Konzentration**, nicht als Ranginstrument: ρ 0,982 gegen Mean KL, und
die Leiter besteht es nicht. Ein Merkmal, das die *Form* des Tails beschreibt (Anteil der Summe im
schlechtesten 1 %, Verhältnis CVaR/P99), wäre als Nächstes prüfbar — aber nur gegen die Prüfregel,
und die ist dafür noch nicht geschrieben. Über die fünf Arme dieser Tabelle fällt der Summenanteil
monoton mit Mean KL, also ist eher Redundanz zu erwarten als ein neuer Freiheitsgrad.
