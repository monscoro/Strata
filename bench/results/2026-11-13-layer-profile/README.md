# Dynamikbereich und Komprimierbarkeit pro QSA-Layer (2026-11-13)

Die Frage: **unterscheiden sich die 12 QSA-Layer im Dynamikbereich und damit in der
Komprimierbarkeit der KV-Werte — und ergibt das einen Ansatz, die Quantisierung zu verbessern?**

Die kurze Antwort: **nein, kein per-Layer-Ansatz.** Die Layer sind im Codec-Maß weitgehend
austauschbar (Faktor 1,14 auf K, 1,40 auf V), und es gibt keinen Tiefen-Trend. Was sich
unterscheidet, ist nicht die Tiefe, sondern **K gegen V** und die **Größe der Kanal-Offsets**.

## Was daran neu ist — und was nicht

Drei der vier Fragen waren schon beantwortet. Das ist wichtig, weil sonst eine alte Zahl als neuer
Befund gelesen wird:

| Frage | Stand vor dieser Stufe | Quelle |
|---|---|---|
| **Sensitivität pro Layer** (wo tut der Fehler weh?) | gemessen: die Bisektion zweier Layer-Hälften ergibt `max/(A+F) = 0,515` gegen die vorab fixierte Schwelle 0,65 → **keine Hälfte trägt den Tail**, B1 bleibt geschlossen. Einzelschichten: nur L3 `0,001241`, nur L47 `0,000008` (praktisch `int8`) | `bench/results/2026-11-08-bisect`, `docs/paper/dave_diskussion/PLAN.md` Z. 261–329 |
| **Fehleranteil pro Layer** | gemessen: „der KV-Fehler je Schicht ist fast gleich, 6,3 … 12,9 %" | ebd. Z. 265 |
| **Kanalstatistik pro Layer** (kurtosis, crest, Varianz-Spreizung, je Kopf-Gruppe) | gemessen auf 27.10., 12 Layer, 3.072 Kanäle | `tools/kv_s2_channels.py`, `Strata-data\bench\2026-11-10-centering\s2-cwe4k-merged.json` |
| **Centering-Gewinn pro Layer** | gemessen: **K −1,70 dB, V −4,63 dB**, je Layer von **−1,03 dB (L7, K)** bis **−7,02 dB (L47, V)** | `tools/kv_center_attrib.py`, `bench/results/2026-11-10-centering/README.md` Z. 273 |
| **Komprimierbarkeit pro Layer bei festen 4,5 bit** | **neu** (diese Stufe): `rmse/rms` je Layer, also die Fehlerrate **ohne** den Energieunterschied der Layer | `tools/kv_layer_profile.py` |
| **Block-Dynamikbereich pro Layer** | **neu** (diese Stufe): kurtosis und crest je Layer auf Blockebene, mit dem `q4u32`-Writer | ebd. |

Der Unterschied zwischen „Fehleranteil" (6,3–12,9 %) und „Fehlerrate" ist der Punkt: der Anteil
mischt die Energie des Layers mit seiner Komprimierbarkeit. Diese Stufe teilt die Energie heraus —
erst dann ist die Frage „ist Layer X schwerer zu komprimieren?" überhaupt eine Codec-Frage.

## Der Aufbau

* Werkzeug: `tools/kv_layer_profile.py` (host-seitig, kein GPU-Job, kein Build).
* Daten: `cells-cwe4k.bin` aus `2026-11-10-centering` — die **rotierte** K/V-Zeile des gebatchten
  Prompt-Pfads, 12 QSA-Layer (3, 7, … 47), je 32.746 Token, `nkv = 2`, `hd = 256`, 8 Blöcke je
  Token und Kopf.
* Stichprobe: **jeder 8. Block jeder Schicht**, fortlaufend über die Records gezählt →
  **65.492 Blöcke je Seite und Layer** (785.904 insgesamt).
* Block = 32 Werte entlang `head_dim`, wie im Kernel. `d` wird auf **fp16** gerundet, wie im
  Kernel. Drei Writer-Varianten: **uniform + LS** (`q4u32`), **Lloyd-16 + LS** (Produktform) und
  **Shear** (affiner Fit je Block, 6 Passaden = Obergrenze des Hebels).
* Offset: Tabelle **je Kopf und Kanal** — so wie `center-table.bin` gebaut ist, nicht als ein
  Vektor je Layer. Die Tabelle wird **in-sample** aus demselben Dump geschätzt, ihr Gewinn ist
  also eine **Obergrenze**.

**Zwei Fehler unterwegs, beide benannt.** (1) Eine erste Fassung des Werkzeugs mittelte den
Offset über Blockpositionen statt über Kanäle — korrigiert, bevor gemessen wurde. (2) Eine zweite
Fassung nahm an, der Dump sei **layerweise** geordnet; er ist es nicht (die Records **zyklieren**
3, 7, … 47, 3, 7, …). Diese Fassung lieferte Hunderte Teilzeilen und ist **verworfen**; das
Ergebnis oben stammt aus der Fassung, die zwei Durchgänge über den Dump macht (die Offset-Tabelle
braucht den ganzen Layer, und alle 12 Layer gleichzeitig zu halten wären ~6 GB).

**Die Querprobe ist exakt:** die über die 12 Layer gepoolte Block-MSE ist **K 1,4683e-02** und
**V 2,5892e-03** — dieselben Zahlen wie `tools/kv_q4_passes.py` (`p1`) auf denselben Zellen, auf
alle gedruckten Stellen. Das Werkzeug liest also dieselben Zellen und rechnet denselben Writer.

## Das Ergebnis

Fehlerrate = `rmse / rms` des Layers; Gewinne in dB gegen den uniformen LS-Writer, positiv = besser.

| Layer | K rate | K kurt | K crest | K lloyd | K shear | K calib | V rate | V kurt | V crest | V lloyd | V shear | V calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0,0864 | +0,03 | 2,417 | +0,535 | +1,334 | +1,503 | 0,0806 | +0,10 | 2,242 | +0,508 | +1,354 | +1,477 |
| 7 | 0,0864 | −0,05 | 2,418 | +0,451 | +1,250 | +1,263 | 0,0838 | −0,04 | 2,363 | +0,359 | +1,504 | +3,612 |
| 11 | 0,0857 | −0,03 | 2,398 | +0,438 | +1,261 | +1,678 | 0,0805 | +0,55 | 2,196 | +0,419 | +1,153 | +2,990 |
| 15 | 0,0838 | +0,04 | 2,342 | +0,547 | +1,313 | +1,953 | 0,0900 | +0,14 | 2,482 | +0,949 | +1,774 | +7,438 |
| 19 | 0,0807 | +0,03 | 2,257 | +0,402 | +1,199 | +1,476 | 0,0752 | +0,38 | 2,083 | +0,510 | +1,490 | +3,431 |
| 23 | 0,0816 | −0,00 | 2,284 | +0,413 | +1,308 | +1,176 | 0,0835 | +0,13 | 2,281 | +0,467 | +1,388 | +5,057 |
| 27 | 0,0839 | +0,02 | 2,342 | +0,432 | +1,255 | +1,368 | 0,0886 | +0,56 | 2,384 | +0,624 | +1,305 | +5,208 |
| 31 | 0,0829 | +0,01 | 2,319 | +0,518 | +1,344 | +2,331 | 0,0930 | +0,19 | 2,599 | +0,911 | +1,649 | +7,134 |
| 35 | **0,0766** | −0,01 | 2,138 | +0,407 | +1,188 | +1,193 | **0,0743** | +0,91 | 2,034 | +0,338 | +1,339 | +2,493 |
| 39 | 0,0841 | −0,00 | 2,351 | +0,385 | +1,242 | +1,899 | 0,0805 | +0,51 | 2,179 | +0,256 | +1,350 | +3,591 |
| 43 | 0,0843 | +0,04 | 2,349 | +0,404 | +1,203 | +1,574 | **0,1043** | +0,26 | **2,937** | +0,937 | +1,981 | +6,495 |
| 47 | **0,0873** | +0,02 | 2,445 | +0,490 | +1,326 | +2,445 | 0,0905 | +0,33 | 2,515 | +0,790 | +1,922 | +7,441 |

### F1 — die Komprimierbarkeit streut kaum: **nicht bestanden**

Vorab fixiert: ein Faktor **> 1,5** zwischen dem leichtesten und dem schwersten Layer.

* **K: 0,0766 … 0,0873 — Faktor 1,14** (entspricht **0,19 bit**).
* **V: 0,0743 … 0,1043 — Faktor 1,40** (entspricht **0,49 bit**), also **unter** der Schwelle.

Das ist der Kern der Antwort: eine per-Layer-Bitbreite hätte auf K **0,2 bit** und auf V **0,5 bit**
Material, und das nur, wenn die schwersten Layer auch die empfindlichsten wären. Der schwerste
V-Layer ist 43 (0,1043), der leichteste 35 (0,0743) — beide sind QSA-Layer, aber 35 liegt in der
Mitte und 43 am Ende; es ist kein Muster, sondern ein Einzelwert.

### F2 — kein Tiefen-Trend: **nicht bestanden**

Vorab fixiert: Spearman |ρ| ≥ 0,7 gegen den Layer-Index.

* K: **−0,10**, V: **+0,34** (bei 12 Punkten entspricht +0,34 einem p ≈ 0,28).

Es gibt also **keine** Regel „die tiefen Layer sind teurer" oder „die frühen sind teurer". Die
leichte Tendenz auf V (die drei schwersten sind 31, 43, 47) ist statistisch nicht von 0 zu
unterscheiden; auf ihr eine Allokation zu bauen wäre geraten.

### F3 — der Shear unterscheidet die Layer kaum: **auf K nein, auf V ja, aber ohne Produktwert**

Vorab fixiert: Spannweite > 0,5 dB.

* K: **+1,188 … +1,344 dB — 0,16 dB** Spannweite.
* V: **+1,153 … +1,981 dB — 0,83 dB**.

Der Shear ist kein Produktpfad (`q4u32z`, nur Referenzhebel). Sein Muster sagt aber etwas über die
Daten: auf V liegen die Blöcke in den späten Layern schiefer, auf K überall gleich. Das ist die
**einzige** Stelle, an der die Layer auf V wirklich auseinandergehen.

### F4 — der Centering-Gewinn unterscheidet die Layer deutlich: **bestanden, aber schon gemessen**

Vorab fixiert: Spannweite > 0,5 dB.

* K: **+1,176 … +2,445 dB — 1,27 dB**.
* V: **+1,477 … +7,441 dB — 5,96 dB** (Faktor 5 zwischen Layer 3 und Layer 47).

Das ist die **größte** per-Layer-Variation in diesen Daten — und sie ist **nicht neu**: dieselbe
Attribution steht in `bench/results/2026-11-10-centering/README.md` (Z. 273) mit K −1,70 dB,
V −4,63 dB und der Spanne −1,03 dB (L7, K) bis −7,02 dB (L47, V). Die Zahlen hier weichen um
Zehntel ab (anderer Writer: uniform+LS statt Lloyd+max, andere Stichprobe), das **Muster ist
dasselbe**: V trägt den Offset, die späten V-Layer tragen ihn am stärksten.

**Und daraus folgt kein Allokationshebel**, aus zwei Gründen:

1. Der Offset ist **kostenlos** (12 KB je Modell) — man wendet ihn überall an. Eine per-Layer-
   Auswahl würde den Gewinn nicht vergrößern, sondern nur dort abschalten, wo er ohnehin nichts
   kostet.
2. Der Blockgewinn ist **nicht** der End-zu-End-Gewinn. Genau dieser Schluss ist schon gemessen und
   dokumentiert (C4): auf dem 16/16-Arm machte dasselbe Centering den Arm **schlechter**
   (0,01689 → 0,01768), obwohl es im Blockmaß 1,91 dB wert war. Der 128K-Lauf von heute zeigt
   dasselbe Bild von der anderen Seite: `q4lm32w` **+0,00314 besser** mit Centering, `int8`
   **+0,00181 schlechter**. Der Blockgewinn ist codec- und seitenabhängig.

## Der Dynamikbereich: überall derselbe

Das ist der sauberste negative Befund dieser Stufe.

* **Block-Kurtosis:** K −0,05 … +0,04, V −0,04 … +0,91. Kein Layer ist stachelig; alle liegen um
  oder unter Gauß. Der höchste Wert gehört zu V/Layer 35 (+0,91) — und Layer 35 ist der
  **leichteste** Layer beider Seiten. Die Kurtosis und die Komprimierbarkeit zeigen also nicht
  einmal in dieselbe Richtung.
* **Block-Crest** (mittleres Block-absmax / rms): K 2,138 … 2,445 (**Faktor 1,14**),
  V 2,034 … 2,937 (**Faktor 1,44**). Für 32 gaußsche Werte wäre ≈ 2,0 zu erwarten; alle Layer
  liegen darüber, aber **alle im selben Band**.
* **Kanal-Ebene** (S2-Sonde, vorhandene Daten, 27.10.): Kurtosis **−0,07 … −0,18** (also
  *leichter* als Gauß in jedem Layer), Crest **3,81 … 4,08** (Faktor **1,05**).

Ein Layer mit einem deutlich schwereren Rand gibt es in diesem Modell **nicht**. Das ist die
codec-seitige Bestätigung des A7-Befunds (Rotation ist eine Versicherung gegen schwere Ränder, und
diese Versicherung wird hier nicht gebraucht): es gibt keinen Layer, für den sie nötig wäre.

## Was daraus folgt

1. **Kein per-Layer-Ansatz.** Die Komprimierbarkeit streut unter der vorab fixierten Schwelle
   (K 1,14, V 1,40 gegen 1,5), es gibt keinen Tiefen-Trend, und die Sensitivitätsseite ist
   unabhängig davon schon geschlossen (Bisektion 0,515 < 0,65). Beide Achsen zeigen in dieselbe
   Richtung: **die Layer sind austauschbar**.
2. **Die relevante Achse ist K gegen V, nicht die Tiefe.** V hat die größere Streuung in jeder
   Spalte (Rate 1,40 gegen 1,14, Crest 1,44 gegen 1,14, Calib 5,96 dB gegen 1,27 dB). Wer
   asymmetrisch allozieren will, muss das zwischen den Seiten tun — und das ist bereits gemessen:
   der 16/8-Arm (weniger Stufen auf V) ist auf 128K **schlechter** (`q4lm8v-win − q4lm32w` =
   −0,00348 im Vortest). V ist die **schwerere** Seite, nicht die billigere.
3. **Der einzige per-Layer-Hebel, den die Daten zeigen, ist der Offset** — und er ist gratis und
   wird schon global genutzt. Ihn zu „allokieren" wäre eine Lösung ohne Problem.

## Was diese Stufe nicht sagt

* **Nichts über Qualität.** Block-MSE misst, wo der Codec seinen Fehler macht, nicht wo er weh tut.
  Die Sensitivitätsfrage braucht einen Arm, der einen Layer ändert (GPU), und der ist gemessen und
  negativ.
* **Nichts über die Out-of-Sample-Gültigkeit der Offsets.** Die Tabelle ist hier **in-sample** aus
  demselben Dump geschätzt; die Out-of-Sample-Frage (cwe-4k → cwe-8k) ist separat gemessen
  (Korrelation 0,9489, `2026-11-10-centering`).
* **Nichts über den Decode-Pfad.** Der Dump ist der gebatchte Prompt-Pfad (Mode 4).
* **Nichts über andere Modelle oder Texte**: ein Modell (Swift-Qwen3.8-Flash-Next, IQ3_XXS),
  cwe-4k, rotierter Pfad, Stichprobe jeder 8. Block.
* **Nichts über die Draft-Layer** (der MTP rotiert seine eigenen K/V und ist keiner der 12
  QSA-Layer).

## Rohdaten

`C:\Users\krone\git\Strata-data\bench\2026-11-10-centering\cells-cwe4k.bin` (vorhanden) und
`C:\Users\krone\AppData\Local\Temp\opencode\layer-profile-s8.json` → beim Abschluss nach
`Strata-data\bench\2026-11-13-layer-profile\` kopiert. Reproduzieren:

```sh
python tools/kv_layer_profile.py --dump <cells-cwe4k.bin> --stride 8 --json out.json
```
