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
