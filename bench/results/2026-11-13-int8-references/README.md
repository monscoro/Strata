# Die zwei INT8-Referenzen auf cwe-4k (Vorregistrierung, 2026-11-13)

Diese Datei legt fest, **was gemessen wird und wann es zählt**, bevor der Lauf startet. Sie schließt
die erste Lücke der Kollegen-Mail („zwei INT8-Referenzen: den heutigen Produktpfad und einen
Kontrollarm mit vergleichbarer Kalibrierung und Vorverarbeitung") und benutzt die
Nichtunterlegenheitsgrenze, die in `bench/results/2026-11-13-kv-offset-tail/README.md` §3 **vorher**
festgelegt wurde. Sie wird hier **nicht** neu verhandelt.

## Warum das der richtige nächste Schritt ist

Von den neun Armen, die für den Vergleich nötig sind, sind **sieben gemessen** (2026-11-12, derselbe
Build 0.1.59.1, derselbe Text, dieselben `set.ids.json`). Es fehlen genau die zwei, die die
Vorverarbeitung fair machen:

| # | Arm | Stand |
|---|---|---|
| 1 | `fp16` (Referenz der absoluten KL) | vorhanden (2026-11-12) |
| 2 | **R1** `int8` (heutiger Produktpfad) | vorhanden |
| 3 | **R2a** `int8` + Rotation (`STRATA_KV_ROT=1`) | **neu** |
| 4 | **R2b** `int8` + Rotation + Kalibrierung (`STRATA_KV_CENTER`) | **neu** |
| 5 | `q4lm32w` (Produkt, 16/16 + Fenster) | vorhanden |
| 6 | `q4lm32w` + C | vorhanden |
| 7 | `q4u32` (uniformes Gitter, Schritt per LS, + Fenster) | vorhanden |
| 8 | `q4u32` + C | vorhanden |
| 9 | `q4u32z` (Shear, 20-B-Block = 5,0 bit) | vorhanden |

**Dazu ein Wiederholungslauf `q4lm32w-repro`** — nicht als Ergebnis, sondern als **Gültigkeitsprüfung
der Paarung**: sein SHA-256 muss exakt `d6e6514…` (der Kampagnen-Arm) sein. Weicht er ab, ist der
Binärstand ein anderer und die vorhandenen Arme sind **nicht** gepaart vergleichbar; dann zählt nur
der Vergleich der neuen Arme untereinander, und das wird so berichtet.

**R2a ist der Trenner.** Die Rotation ist selbst eine Behandlung und verändert `int8` messbar (frühere
Messung: erster Token, KL 0,0066 → 0,0051 auf einem NVFP4-Pack und 0,0022 → 0,0054 auf IQ2_XS).
Ohne R2a würde man der Kalibrierung zuschreiben, was die Rotation tut.

## Warum der Fixed-Code-Offset-Arm **nicht** läuft

Er ist gemessen, nicht geschätzt: der reine Restfehler-Offset trägt **0,14 dB ≈ 0,0002 nats**
(`bench/results/2026-11-13-kv-offset-tail/`, H0). Das ist **ein Zehntel** der Paritätsauflösung
(0,00200) und ein Zehntel des Shear-Effekts. Ein Arm mit erwarteter Wirkung 0,0002 kann bei dieser
Auflösung **kein** Ergebnis liefern — er wäre ein Nullbefund per Konstruktion. Er bleibt als
**Writer-Kostenfrage** offen (der billige Fit braucht keine Iterationen) und ist **kein**
Qualitätsarm. Der Shear wird mit seinem vorhandenen 20-Byte-Block gefahren (5,0 bit); dass der
6-Bit-Offset (4,6875 bit) denselben Hebel trägt, ist auf Blockebene gemessen (96,8–98,3 % des exakten
Gewinns) und wird als **Format**-Aussage geführt, nicht als zweiter Arm.

## Die Endpunkte

Alle Differenzen sind **gepaart über Positionen** (dieselben `ids`), Segmentmittel über 8 Segmente,
95-%-Intervall aus der t-Verteilung über die Segmente; zusätzlich das per-Position-Intervall.

| | Frage | Endpunkt | Gate |
|---|---|---|---|
| **E1** | was ist die Vorverarbeitung auf `int8` wert? | `R2b − R1`, `R2a − R1` | informativ, kein Gate |
| **E2** | ist der **Produktarm** dem fairen `int8` nicht unterlegen? | `q4lm32w+C − R2b` | obere 95-%-Grenze **< +0,0020**, Punkt **≤ +0,0010** |
| **E3** | ist der **Fast-Path-Kandidat** dem fairen `int8` nicht unterlegen? | `q4u32 − R2b` | wie E2 |
| **E4** | reproduziert der Shear seinen 4K-Befund? | `q4u32z − q4u32` | **negativ**, Größe in [0,001; 0,003] nats |
| **E5** | wirkt die Kalibrierung auf `int8`? | `R2b − R2a` | informativ |

**Nebenbefunde, keine Endpunkte:** das 1-%-Tail (int8-definiert), Top-1 mit exaktem McNemar gegen
`fp16`, p99. Laufzeit wird **nicht** in diesem Lauf gemessen (eigene Stufe, getrennte Writer-/Reader-
Zeit).

## Die Fallen, die vorher benannt sind

* **R2 kann kaputt sein.** `int8` + Rotation ist ein Pfad, der bisher nur über den ersten Token
  gemessen wurde. Ist die KL von R2a **mehr als 3×** die von R1, gilt der Pfad als **defekt**, nicht
  als Ergebnis; dann ist E2/E3 mit diesem Lauf **nicht entscheidbar** und das wird so berichtet.
* **Die Kalibrierungstabelle ist für `kLloydWin` gebaut** (`center-table.bin` aus dem S2-Probe). Sie
  ist formal modellweit (Layer × KV-Kopf × Kanal) und wird auf `int8` genauso angewandt; ob sie dort
  *hilft*, ist E5 und keine Voraussetzung.
* **`q4u32` und `q4u32z` tragen das Fenster** (`STRATA_KV_WINDOW=1`, S 256 / R 64 / Chunk 512), der
  Produktarm trägt es im Modus. Der Vergleich ist damit auf allen q4-Armen dieselbe Geometrie.
* **Ein Lauf, keine Parallele.** Neun Arme à ~17 min sind ~2,6 h; gestartet werden nur die drei
  neuen (zwei INT8 + der Reproduktionsarm), ~52 min, weil die sieben vorhandenen Arme auf demselben
  Build und demselben Text liegen. 30 s zwischen Engine-Starts.
* **Kein Build während des Laufs.**

## Was diese Stufe nicht behauptet

* Keine Aussage über den **Reader-Preis**. Ein Qualitätsarm sagt nichts über Laufzeit.
* Keine Aussage über **andere Texte**: cwe-4k, 32.746 Positionen, 8 Segmente.
* Keine Aussage über **128K**. Der Abstand zu `int8` ist dort größer (−0,00741 nats); dieser Lauf
  entscheidet die **Richtung** und die Fairness der Referenz, nicht die 128K-Frage.

---

# Nachtrag 2026-11-13, 14:20: das Ergebnis

Der Lauf ist durch: drei Arme, 17,4 / 17,4 / 18,2 min, 30 s Pause dazwischen. Der Reproduktionsarm
ist **byte-identisch** zum Kampagnenarm (`d6e65140272b…`, 32.746 Zeilen, 8 Segmente), also sind die
sieben Altarme und die drei neuen in **einem** gepaarten Vergleich gültig — alle Zahlen unten kommen
aus einem einzigen `compare`-Durchlauf über zehn Arme mit `fp16` als Referenz.

## Die absoluten Zahlen (cwe-4k, 32.746 Positionen, KL gegen fp16)

| Arm | mittlere KL | Top-1 | p99 |
|---|---:|---:|---:|
| `fp16` | 0,00000 | 100,00 % | 0,0000 |
| **R1** `int8` | 0,01688 | 99,05 % | 0,1888 |
| **R2a** `int8`+ROT | **0,01579** | 99,06 % | 0,1671 |
| **R2b** `int8`+ROT+C | **0,01476** | 99,03 % | 0,1575 |
| `q4lm32w` (Produkt) | 0,01689 | 98,89 % | 0,2410 |
| `q4lm32w`+C | 0,01768 | 98,97 % | 0,2263 |
| `q4u32` | 0,01739 | 98,89 % | 0,2653 |
| `q4u32`+C | 0,01782 | 98,96 % | 0,2062 |
| `q4u32z` (Shear) | 0,01877 | 98,89 % | 0,2212 |

## Die Gates, wie sie vorher festgelegt waren

| | Differenz | Mittel | 95-%-Intervall (8 Segmente, zweiseitig) | einseitige obere Grenze | t | Gate |
|---|---|---:|---|---:|---:|---|
| **E1a** | R2a − R1 | −0,00108 | [−0,00321, +0,00104] | +0,00063 | −1,18 | informativ |
| **E1b** | R2b − R1 | −0,00212 | [−0,00481, +0,00057] | +0,00005 | −1,82 | informativ |
| **E5** | R2b − R2a | −0,00103 | [−0,00481, +0,00274] | +0,00201 | −0,63 | informativ |
| **E2** | `q4lm32w`+C − R2b | **+0,00293** | [−0,00117, +0,00702] | **+0,00623** | +1,65 | **NICHT BESTANDEN** |
| **E3** | `q4u32` − R2b | **+0,00263** | [−0,00084, +0,00611] | **+0,00544** | +1,75 | **NICHT BESTANDEN** |
| **E4** | `q4u32z` − `q4u32` (Ganztext) | +0,00138 | [−0,00056, +0,00332] | +0,00295 | +1,64 | Vorzeichen falsch |
| **E4** | `q4u32z` − `q4u32` (Antwort-Region) | −0,00193 | [−0,00409, +0,00022] | −0,00020 | −2,07 | einseitig ja, zweiseitig grenzwertig |

## Der Hauptbefund: die alte Parität war gegen eine unbehandelte Referenz

Die Kampagne hatte `q4lm32w` gegen `int8` mit **+0,00002** nats als Parität gelesen. Das war gegen
`int8` **ohne** Rotation und **ohne** Kalibrierung. Mit derselben Vorverarbeitung und derselben
Kalibrierung auf beiden Seiten ist die Referenz **besser**:

* `q4lm32w`+C − R2b = **+0,00293** nats (per Position +0,00293, t 2,42, Intervall [+0,00056, +0,00529] —
  schließt 0 aus),
* der **ausgelieferte** Arm ohne Kalibrierung, `q4lm32w` − R2b = **+0,00213** nats (Segment-t 1,67;
  per Position t 1,71, Intervall [−0,00032, +0,00459]).

Damit ist die **Nichtunterlegenheitsgrenze nicht gezeigt** — weder über die Segment-CI (obere Grenze
+0,00623 bzw. +0,00544 gegen die vorab fixierten +0,0020) noch über den Punkt-Schätzer (+0,00293 bzw.
+0,00263 gegen +0,0010). *„Nicht gezeigt" ist nicht „widerlegt":* die Segment-Intervalle enthalten 0,
die Auflösung des Textes (0,00200) ist selbst von derselben Größe wie der Effekt. Was der Lauf sagt,
ist präzise: **die Parität ist nicht belegt, und der Punkt-Schätzer zeigt in die falsche Richtung.**

Das ist genau der Punkt der Kollegen-Mail. Er ist mit *einem* zusätzlichen Arm-Paar entschieden, nicht
mit einer neuen Behauptung über den Codec.

**Und die Vorverarbeitung selbst ist nicht aufgelöst:** R2a − R1 = −0,00108 (t −1,18) und
R2b − R2a = −0,00103 (t −0,63). Die Rotation und die Kalibrierung *helfen* `int8` in der Tendenz
(0,01688 → 0,01579 → 0,01476), aber keiner der beiden Schritte ist einzeln signifikant. Der faire
Vergleich ist trotzdem der richtige: die Referenz darf nicht schlechter sein als sie sein kann, nur
weil ihr eine Behandlung fehlt.

## Der Widerspruch im Tail — und er ist echt

Die Nebenbefunde drehen die Richtung **auf den schweren Positionen** (Tail definiert über die KL von
R2a, also armunabhängig):

| A gegen R2a | Tail-Mittel A | Δ im Tail | t | Top-1 (McNemar) |
|---|---:|---:|---:|---|
| `q4lm32w` | 1,018 | **−0,329** | −3,04 | R2a besser, p = 0,0025 |
| `q4u32` | 1,089 | **−0,258** | −2,24 | R2a besser, p = 0,0028 |
| `q4u32z` | 1,144 | **−0,203** | −2,08 | R2a besser, p = 0,0020 |

Im schlimmsten 1 % **von `int8`** ist jeder q4-Arm deutlich besser (bis 0,33 nats), und trotzdem
gewinnt `int8` das Top-1 über alle Positionen und liegt bei p99 vorn (0,167 gegen 0,241). Das ist keine
Widersprüchlichkeit, sondern die Fehlerform: **`int8` hat wenige katastrophale Positionen und viele
fast fehlerfreie; q4 hat mehr kleine Fehler und weniger Ausreißer.** Für die 128K-Frage ist das die
relevante Spur — dort war der Abstand ebenfalls tail-getrieben.

*Grenze dieses Befunds:* der Tail ist über `int8` definiert. Er sagt „auf den Positionen, wo `int8` am
schlechtesten ist, ist q4 besser", **nicht** „q4 hat keine schlechten Positionen" — p99 und Top-1 sagen
das Gegenteil.

## Ein Fehler in dieser Vorregistrierung (E4)

Die Zahl, aus der E4 gebaut wurde (+0,00193, t 2,07), steht in der **Antwort-Region**, nicht im
Ganztext. Die Vorregistrierung nannte die Tabelle nicht. Im Ganztext ist das Vorzeichen umgekehrt
(`q4u32z` − `q4u32` = **+0,00138**, der Shear also *schlechter*), und das war am 2026-11-12 genauso —
der Fehler ist meiner beim Abschreiben, nicht im Codec. E4 wird deshalb **nicht als bestanden
gezählt**: in der Antwort-Region reproduziert der Shear seinen Befund exakt (byte-gleiche JSON-Werte:
−0,00193, t −2,07 über dieselben 8 Segmente), im Ganztext trägt er nicht. Auch in der Antwort-Region
ist der Befund nur **einseitig** fest (obere Grenze −0,00020); zweiseitig berührt das Intervall die 0
(+0,00022).

## Was dieser Lauf nicht sagt

* Nichts über **Laufzeit** (eigene Stufe, getrennte Writer-/Reader-Zeit).
* Nichts über **128K** und nichts über andere Texte.
* Nichts darüber, **warum** die Kalibrierung auf `int8` so wenig auflöst (E5, t −0,63) — sie ist für
  `kLloydWin` gebaut, und das ist eine Formatfrage, keine Wirkungsfrage.
* `q4lm32w`+C ist im Mittel **schlechter** als `q4lm32w` (0,01768 gegen 0,01689), bei besserem p99 und
  Top-1. Die Kampagne hatte das schon so gemessen; dieser Lauf bestätigt es gegen die neue Referenz.

