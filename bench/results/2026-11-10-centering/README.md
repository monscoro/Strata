# Miniprojekt „Centering": der feste Kanal-Offset vor der Rotation

**Status: vorregistriert (2026-11-10). Kein Arm gelaufen, kein Kernel gebaut.** Dieses Dokument
steht vor dem ersten Lauf; die Gatter stehen fest. Anlass ist die Recherche in
`docs/research/KV_ROTATION_SURVEY.md` und die Auswertung der S2-Rohdaten aus
`bench/results/2026-10-27-s2-channels/` (Rohdaten: `Strata-data/bench/2026-10-27-s2-channels.txt`).
Katalogzeile: `docs/paper/Ansatz_Improvements.md`, **A6** (Per-Kanal-Offset vor der Rotation); die
Rotations-Variante **A7** (randomisierter Hadamard H·D) läuft im selben Harness mit, ist aber
**nicht** Gegenstand dieses Projekts.

## 1. Warum

**Der Befund.** Die S2-Sonde akkumuliert je Kanal unter anderem den Mittelwert der K-Werte über
32.746 Token (12 QSA-Layer × 256 Kanäle). Ausgewertet am 2026-11-10:

| | vor FWHT | nach FWHT |
|---|---:|---:|
| standardisierter Kanal-Mittelwert \|mean\|/sd, Median | **0,2673** | **0,2825** |
| p90 / Maximum | 0,7337 / 3,5778 | 0,6967 / 1,6561 |
| Sampling-Erwartung bei n = 32.746 | 0,0044 | 0,0044 |

Der Kanal-Mittelwert ist rund 60-mal größer als das Sampling-Rauschen: **er ist echt**. Und die
Rotation **beseitigt ihn nicht**, sie verteilt ihn: aus einem festen Offset je Kanal wird ein
fester Bias in *jedem* rotierten Koordinatenpaar (dieselben Offsets addieren sich mit den
Hadamard-Vorzeichen). Er ist **kein gemeinsamer DC** — der Mittelwert der Kanalmittelwerte je Layer
liegt bei −0,016 bis +0,086 bei einer Streuung der Kanalmittel von 0,43 bis 0,64, also
\|mean\|/sd_typ = 0,007 bis 0,068: die Offsets sind **kanalspezifisch**.

**Warum das bei uns teuer ist.** Unsere Skala ist eine **Max-Statistik**: `d = amax / lmax`, mit
`amax` als Shuffle-Maximum über die 32 Werte eines Blocks (`src/kernels/cuda/kv_q4.cu:138-144`;
`lmax` = 87 für 16 Stufen, 69 für 8 Stufen). Ein fester Bias bläht dieses Maximum auf — und die
Skala wird an **alle** Werte des Blocks bezahlt, nicht nur an den verrauschten.

**Was das für die Spannungsnutzung heißt.** Die Zahl der effektiv genutzten Stufen je Block ist
das Signal, an dem das Centering zu messen ist. Für einen Block mit 32 Werten und Stufentabelle L
gilt: die Stufenweite ist 2·d, das Signal nutzt die Spanne seiner eigenen Streuung. **Definition
des Werkzeugs:** die *Spannungsnutzung* u eines Blocks = (Streuung des Signals im Block) / d, in
Stufen. Ohne Bias ist u der Quotient aus Signalstreuung und Blockmaximum; mit Bias wächst das
Maximum, d wächst, u sinkt. **Die Hypothese: das Centering hebt u messbar (die Skala wird enger),
und die MSE folgt daraus.**

**Größenordnung (Schätzung, keine Messung).** Nach der Rotation: sd ≈ 1,1 bis 1,67, Bias
|mean_post| ≈ 0,283 · sd; Signalmaximum über 32 Koordinaten ≈ 2,1 · sd, Biasmaximum über denselben
Block ≈ 0,6 · sd → die Skala ist um grob **15 bis 30 %** zu groß, der MSE skaliert mit d² → **etwa
+1 bis +2 dB**. Zum Vergleich: der Rest der Bulk-Seite bis zum konvergierten per-Block-Fit ist
**0,17 dB** (`kv_q4_parity`: q4_0 2,938e-02 = 0 dB, kLloyd 2,740e-02 = −0,30 dB, kLloydLs
2,551e-02 = −0,61 dB, konvergierter Fit 2,456e-02 = −0,78 dB).

**Was die Literatur sagt (Belege in `docs/research/KV_ROTATION_SURVEY.md`).** OptR (arXiv:2608.02691)
zentriert die Keys vor der Rotation; ein öffentlicher `llama.cpp`-Fork hat ein K-Cache-Mean-Centering
mit **Calibration-Basis-Guard** (ein Versatz aus der falschen Basis verschlechtert messbar).
NSNQuant (arXiv:2505.18231) zentriert kalibrierungsfrei über eine laufende Schätzung.

## 2. Die Korrektur, so einfach wie möglich

Die Korrektur ist **eine Subtraktion vor dem FWHT** — kein neues Cache-Format, kein neues Feld im
Block, kein zusätzliches Byte im Cache, **kein Umbau des Lesers**:

1. **K:** `k' = k − m_K` je Kanal, **vor** dem FWHT. Der Writer bekommt eine Offset-Tabelle
   `m_K` (12 Layer × 2 KV-Köpfe × 256 Kanäle = 6.144 Werte, 12 KB in fp16).
2. **V:** `v' = v − m_V` je Kanal, vor dem FWHT. Der Leser muss `m_V` **einmal** auf die
   Aufmerksamkeits-Ausgabe addieren (ein Vektoradd pro Token und Kopf): mit
   `Σ aᵢ = 1` ist das exakt: `Σ aᵢ (vᵢ − m_V) = (Σ aᵢ vᵢ) − m_V`.
3. **Die Scores bleiben unangetastet:** `⟨q, k⟩ = ⟨q, k′ + m_K⟩ = ⟨q, k′⟩ + ⟨q, m_K⟩` — der zweite
   Teil ist **pro Query konstant** (der Query ist pro Token fix) und verschiebt alle Logits dieser
   Query gleich, der Softmax bleibt exakt. Also: **kein Add-Back auf dem Score-Pfad.**
4. **Reihenfolge auf dem Ausgabepfad:** erst un-rotieren (H), **dann** `m_V` addieren — im
   rotierten Raum wäre der Add-Back `H·m_V`. Eine vertauschte Reihenfolge bricht das Modell
   still; das ist dieselbe Klasse wie die Inverse-Falle beim randomisierten Hadamard (A7).
5. **Das Fenster bleibt exakt:** der Ring speichert den fp16-Zwilling der (zentrierten, rotierten)
   Zellen — dieselben Werte, die der Writer schreibt. Die Genauigkeit des Rings ändert sich nicht.
6. **Die Selektion sieht nichts:** der Indexer-Key ist die Projektion von `x`
   (`src/core/layer.cpp:1399-1407`), **vor** dem FWHT ab 1472.

**Was die Korrektur NICHT braucht:** keine Änderung am Blockformat (18 Byte bleiben 18 Byte), keine
Änderung am Leser außer dem einen Vektoradd, keine Änderung an der Level-Tabelle, keine Änderung
am Ring, keine Änderung an der Streaming-Granularität.

**Was sie braucht (und das ist der Preis):** eine Offset-Tabelle pro Modell (12 KB) — und die
Frage, woher sie kommt:
- **(a) Kalibriert und ausgeliefert:** ein Kalibrierlauf über einen Text (analog zum
  Expert-Profil), die Tabelle ins Pack. Robust, aber ein Artefakt im Modell.
- **(b) Online geschätzt (kalibrierungsfrei):** ein laufender Mittelwert je Kanal (EMA) über die
  ersten Zellen; die Zellen vor der Schätzung werden ohne Versatz quantisiert (die ältesten, also
  die Sinks — der Bereich, den die Auswahl ohnehin am wenigsten liest). Kein Artefakt, aber
  komplexer (ein Schwellen- und ein Nachlaufverhalten, das der Plan festlegen muss).
- **(c) Aus dem Modellgewicht abgeleitet:** die K-Kanäle sind `w_norm · x̃`; falls der Offset aus
  einem nicht-null Erwartungswert der normalisierten Aktivierungen stammt, wäre er über
  `E[x̃]` und den Norm-Gewichtsvektor bestimmbar — **ohne** Kalibrierung. Ob das trägt, ist offen
  (der Offset ist in den Daten real, die Kette Normgewicht → Aktivierungs-Erwartung → K-Mittel ist
  nicht gemessen). Wenn (c) trägt, ist das die einfachste Korrektur überhaupt: eine Tabelle, die
  aus dem Gewicht abgeleitet wird.

**Die Entscheidung zwischen (a), (b) und (c) fällt in Stufe C0** — sie hängt an der Stabilität des
Offsets über Texte und Längen.

## 3. Der Basis-Vertrag (der teuerste Teil)

Ein Versatz ist **basisgebunden**: gemessen im unrotierten Kanalraum, angewendet vor dem FWHT.
Der `llama.cpp`-Guard existiert genau deshalb. Für uns heißt das:

- Der Offset wird **im unrotierten Kanalraum** gemessen (das ist die Ebene, in der die Tabelle
  lebt) und **vor** dem FWHT abgezogen.
- **Snapshot:** `conversation_snapshot.cpp` verweigert heute windowed States („exact side pool
  outside the image"). Ein zentrierter Zustand gehört in dieselbe Klasse: das Snapshot-Format
  braucht ein **eigens Kennzeichen** (analog zum vorhandenen `rotated`-Flag), und ein Snapshot eines
  zentrierten Zustands darf **nicht** in einen unzentrierten zurückgespielt werden (sonst fehlt
  `m_K`). Das ist ein Schema-Flag, kein Formatumbau.
- **Streaming:** die Host-Kopie trägt dieselben zentrierten Werte; der `m_V`-Add-Back ist
  pfadunabhängig (er hängt an der Attention, nicht am Speicherort).
- **Der `int8`-Pfad bleibt byte-identisch:** das Centering ist opt-in (eigener Schalter, kein
  Default-Pfad).

## 4. Die Stufen und ihre Gatter (vor dem Lauf festgelegt)

| Stufe | Was | Gatter (Kill) | Kosten |
|---|---|---|---|
| **C0** | **Offset-Stabilität.** Die S2-Sonde auf einem **zweiten Text und einer zweiten Länge** (z. B. cwe-8k, 32K) → die Offsets je Kanal vergleichen (Korrelation, relative Abweichung in sd-Einheiten) | **G1:** Korrelation der standardisierten Kanalmittelwerte ≥ 0,90 **und** mittlere relative Abweichung ≤ 0,25 sd. Darunter: (a) wird unbrauchbar, es bliebe (b) oder (c) — und (b) ist erst nach C1 zu bauen | 1 Lauf, ~3,5 min GPU (S2-Rohzeit laut `2026-10-27-s2-channels`: „ein Lauf, 3,5 min GPU-Zeit, kein Kernel geändert") |
| **C1** | **Der Effekt, ohne Engine.** `kv_q4_parity` wird um eine Variante „mit Offsets" erweitert; gleiche Zellen, gleicher Tag, gleicher Build: MSE in dB gegen die Baseline **und** die Spannungsnutzung u je Block (Median, p10, p90) | **G2:** MSE-Gewinn ≥ 0,05 dB (der Rauschteppich des Werkzeugs). **G3:** die Spannungsnutzung u steigt um ≥ 10 % (Median). Beides fällt, ist die Achse geschlossen | Minuten, kein Modelllauf |
| **C2** | **Die Korrektur, gebaut.** Writer-Änderung (Subtraktion vor dem FWHT), `m_V`-Add-Back auf dem Ausgabepfad, Offset-Tabelle im Weg der Wahl (a)/(b)/(c), Snapshot-Kennzeichen, opt-in-Schalter | **G4:** `int8`-Pfad byte-identisch (SHA-Gatter wie in der Kampagne), `kv_q4_parity` und `kv_win_parity` grün, `kv_window_parity` unverändert | **M** (Writer + Leser + Pack) |
| **C3** | **Die Wirkung.** Ein cwe-4k-Arm (gepaart gegen `int8` und gegen den Window-Arm) | **G5:** die gepaarte KL verbessert sich; 128K nur bei einem 4K-Unterschied | **M** (ein Arm) |

**Reihenfolge:** C0 → C1 → C2 → C3. C0 und C1 sind billig und entscheiden, ob C2 überhaupt
gebaut wird. **C0 ist der eigentliche Kill**: ein Offset, der mit dem Text wandert, ist kein
Offset, sondern Rauschen — und dann hilft auch die schönste Tabelle nichts.

**Der Rotations-Vergleich läuft im selben Harness mit** (H·D, drei Vorzeichenzüge; A7), weil der
Harness ihn für einen Bruchteil der Kosten mitmessen kann — aber **G2 und G3 gehören dem
Centering**; die RHT-Varianten haben eine eigene, unbestätigte Erwartung (≈ 0).

## 5. Was dieses Projekt nicht ist

- Kein Ergebnis. C0 bis C3 sind nicht gelaufen; die Schätzung in §1 ist eine Schätzung.
- Kein Ersatz für die Level-Tabelle (0,30 dB) und nicht für die per-Block-Skala (0,31 dB). Das
  Centering ist eine **dritte, orthogonale** Hebelwirkung auf derselben Bulk-Seite — wenn es
  trägt, verschiebt es den Rest (0,17 dB) nicht, es kommt dazu.
- Kein 5.0-Bit-Ziel. Der Byte-Budget wird durch das Centering **nicht** berührt (kein zusätzliches
  Bit im Cache; die 12 KB liegen im Modell, nicht im KV).
- Kein Ersatz für den Window-Overlay. Das Fenster lieferte den großen Sprung (KL 0,0318 → 0,0209
  auf cwe-4k, gemessen 2026-11-10); das Centering ist eine Bulk-Hebelwirkung.

## 6. Reproduktion (nach dem Bau)

```powershell
# C0: der zweite S2-Lauf (Text und Länge variieren), dann der Vergleich
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs-merged.json `
    --ids <zweiter Text> --kv q4lm8v --prefill 512 --short-read 0 --allow-no-positions `
    --env STRATA_KV_S2_STATS=<datei> --out <leer>
python tools\kv_s2_channels.py --tsv <datei> --json <datei.json>

# C1: der Parity-Harness mit und ohne die Offsets (gleiche Zellen, gleicher Build)
ctest --test-dir build-hip-port -R kv_q4_parity --output-on-failure

# C2/C3: der cwe-4k-Arm mit dem Schalter, gepaart ausgewertet
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs-merged.json `
    --ids data\quant_quality\cwe-4k\set.ids.json --kv q4lm8v --env STRATA_KV_CENTER=1 `
    --out ...\c3\q4lm8v-center.tsv
```

Rohdaten (TSVs, stderr, `seg.json`, S2-Dateien) liegen außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-11-10-centering\`.

## 7. Grenzen, die schon jetzt gelten

- Die Stabilität des Offsets über Texte und Längen ist **ungeprüft** (G1).
- Ob der Offset aus dem Modellgewicht ableitbar ist (Variante c), ist **ungeprüft**; ohne (c)
  bleibt ein Artefakt oder eine Onlineschätzung.
- Der Schätzwert der Skaleninflation (15 bis 30 %) ist eine **Rechnung aus Momenten**, keine
  Messung; G2/G3 sind die Messung.
- Die Wirkung auf die **128K**-Statistiken ist offen; der Plan (K2/K3 der Leiter) misst 128K erst,
  wenn ein 4K-Unterschied vorliegt.
- Die Basis-Regeln (Snapshot-Kennzeichen, Streaming, `m_V`-Add-Back in der richtigen Reihenfolge)
  sind **statisch** abgeleitet; ihre Prüfung ist Teil von C2 (Parität und ein Test, der die
  Reihenfolge absichtlich bricht und das Ergebnis prüft).
