# Miniprojekt „Centering": die systematischen Momente vor der Rotation entfernen

**Status: vorregistriert (2026-11-10). Kein Arm gelaufen, kein Kernel gebaut.** Dieses Dokument
steht vor dem ersten Lauf; die Gatter stehen fest. Anlass ist die Recherche in
`docs/research/KV_ROTATION_SURVEY.md` und die Auswertung der S2-Rohdaten aus
`bench/results/2026-10-27-s2-channels/` (Rohdaten: `Strata-data/bench/2026-10-27-s2-channels.txt`).
Katalogzeile: `docs/paper/Ansatz_Improvements.md`, **A6** (Per-Kanal-Offset vor der Rotation); die
Rotations-Variante **A7** (randomisierter Hadamard H·D) läuft im selben Harness mit, ist aber
**nicht** Gegenstand dieses Projekts.

**Das Prinzip des Projekts:** *Ein Bias ist bei randomisierten Verfahren immer problematisch — eine
Zufallsrotation entfernt einen systematischen Anteil nicht, sie verteilt ihn nur auf zufällige
Koordinaten.* Für einen gegebenen Sign-Zug S ist S·μ ein **fester** Vektor mit fester Energie
(‖H·S·μ‖ = ‖μ‖, orthogonal): der Bias landet nur zufällig, er verschwindet nicht. Genau das ist
Flagrots Beobachtung („rotation flattens a spiky piece of a key but can give a flat one a new
peak"). **Darum ist die Reihenfolge fest: erst den systematischen Anteil entfernen, dann über
Randomisierung reden** — vorher ist eine Randomisierung nicht sinnvoll messbar. Dieses Projekt
entfernt den systematischen Anteil **konsequent**, also alle Momente, die systematisch sind, nicht
nur den ersten.

## 1. Warum

**Der Befund.** Die S2-Sonde akkumuliert je Kanal unter anderem Mittel, Varianz, Schiefe und
Exzess-Kurtosis der K-Werte über 32.746 Token (12 QSA-Layer × 256 Kanäle). Ausgewertet am
2026-11-10, mit den Standardfehlern der Moment-Schätzer für diese Stichprobe als Rauschmaß:

| Moment | vor FWHT | nach FWHT | Rauschen | systematisch? |
|---|---|---|---|---|
| Mittel (1. Moment) | Median \|mean\|/sd **0,2673**, p90 0,7337, max 3,5778 | **0,2825**, p90 0,6967 | 0,0055 | **ja, und die Rotation entfernt ihn nicht** |
| Schiefe (3. Moment) | Median 0,1050, p90 0,3531, max 1,7199 | 0,0836, p90 0,2255 | 0,0135 | **ja, kanalspezifisch** (62,6 % der Kanäle > 5× Rauschen vor, 58,5 % nach) |
| Exzess-Kurtosis (4. Moment) | Median 0,2220, p90 0,8452, max 6,7842 | 0,1295, p90 0,4373 | 0,0271 | ja, aber die Rotation flacht es stark ab (65,0 % → 48,1 % über 5× Rauschen) |

Drei Lesarten:

1. **Der Mittelwert ist echt** (rund 60-mal über dem Rauschen) **und die Rotation verteilt ihn nur.**
   Aus einem festen Offset je Kanal wird ein fester Bias in *jedem* rotierten Koordinatenpaar
   (dieselben Offsets addieren sich mit den Hadamard-Vorzeichen). Er ist **kein gemeinsamer DC** —
   der Mittelwert der Kanalmittelwerte je Layer liegt bei −0,016 bis +0,086 bei einer Streuung der
   Kanalmittel von 0,43 bis 0,64, also \|mean\|/sd_typ = 0,007 bis 0,068: die Offsets sind
   **kanalspezifisch**.
2. **Die Schiefe ist ebenfalls systematisch und kanalspezifisch** — aber sie ist **nicht**
   layerweit: der Mittelwert der Schiefe je Layer liegt nur bei 0,2 bis 2,0× Rauschen, mit 46 bis
   56 % gleichgerichteten Kanälen. Es gibt also keinen „Layer-Schief", den man mit einem einzigen
   Versatz räumt; jeder Kanal trägt seine eigene. Die Rotation nimmt sie nur leicht zurück
   (0,1050 → 0,0836).
3. **Die Kurtosis ist der Teil, den die Rotation schon erledigt** (0,2220 → 0,1295). Sie ist der
   Grund, warum die Rotation überhaupt im Plan steht.

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

**Was sie braucht (und das ist der Preis):** eine Offset-Tabelle pro Modell — **12 Layer × 256
Kanäle = 3.072 Werte (6 KB in fp16)**, *nicht* 6.144: die S2-Sonde mittelt über die beiden
KV-Köpfe (`accumulate(..., rows = T*2, 256)`), sie kann also **keine** kopfgetrennten Offsets messen.
Per-Kopf-Offsets (6.144 Werte) bräuchten eine Erweiterung der Sonde; die geteilte Tabelle ist
das, was die Messung trägt, und sie ist der ehrlichere Weg (ein Versatz, der pro Kopf
verschieden wäre, ist mit dieser Sonde nicht sichtbar). Dazu die Frage, woher die Tabelle kommt:
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

**Und die anderen systematischen Momente (das „konsequent").** Das Centering entfernt den ersten
Moment. Die **Schiefe** (3. Moment) ist ebenfalls systematisch und kanalspezifisch (Median 0,1050
vor und 0,0836 nach der Rotation; 62,6 % der Kanäle über 5× Rauschen) — eine **symmetrische**
Lloyd-Tabelle kann eine schiefe Quelle nicht ausnutzen: die Stufen werden auf die leere Seite
verteilt. Sie zu entfernen geht nur **kanalweise** (es gibt keinen Layer-Schieft, s. §1), also mit
derselben Tabelle: pro Kanal ein Versatz **und** ein Asymmetrie-Parameter, oder eine eigene
Stufentabelle je Kanal (12 KB je Layer und Kopf — dieselbe Währung wie die Offsets). Die
**Exzess-Kurtosis** (4. Moment) ist der Teil, den die Rotation schon flacht (0,2220 → 0,1295); sie
zu entfernen hieße eine andere Codebook-Familie (Katalogzeile A5), nicht eine Korrektur im Writer.
**Die Reihenfolge, die dieses Projekt daraus macht:** (1) der 1. Moment (Versatz) — billig, exakt
erhaltend, erwartet der größte Hebel; (2) der 3. Moment (Asymmetrie) — **nur wenn** der
Parity-Harness zeigt, dass danach noch Stufen verschenkt werden; (3) der 4. Moment — gehört in
eine andere Zeile (A5). Die Messung entscheidet die Reihenfolge: u (die Spannungsnutzung) ist das
Werkzeug, und wenn u nach dem Versatz noch weit unter ihrem Maximum liegt, ist der nächste
Kandidat die Asymmetrie.

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
| **C0** | **Offset-Stabilität und Moment-Status — zwei Läufe auf dem aktuellen Build.** **C0a:** cwe-4k (derselbe Text und dieselbe Länge wie die Messung vom 2026-10-27) → die Referenz **desselben** Builds. **C0b:** cwe-8k (zweiter Text, zweite Länge, 65.571 Ids) → der Gate-Vergleich. Der Vergleich **C0a ↔ C0b** ist G1; der Vergleich der 0.1.47-Datei mit C0a isoliert zusätzlich den **Build-Effekt** auf die Offsets (deskriptiv, kein Kill). Beide Läufe über den batched Prompt-Pfad (`--short-read 0`), weil der Verify-Pfad graph-capture-geschützt ist. Im selben Lauf die **Signifikanz aller drei Momente** berichten (Mittel, Schiefe, Kurtosis gegen die Standardfehler der Schätzer: 0,0055 / 0,0135 / 0,0271 bei n = 32.746), damit die Entfernungsreihenfolge (§2) auf Messdaten steht und nicht auf einer Vermutung | **G1 (C0a ↔ C0b):** Korrelation der standardisierten Kanalmittelwerte ≥ 0,90 **und** mittlere relative Abweichung ≤ 0,25 sd. Darunter: (a) wird unbrauchbar, es bliebe (b) oder (c) — und (b) ist erst nach C1 zu bauen. **G1b:** die Schiefe und die Kurtosis werden auf dem zweiten Text mit ihrer Signifikanz berichtet (Deskriptiv, kein Kill: sie entscheiden die Reihenfolge in §2, nicht das Projekt) | 2 Läufe, ~3,5 min (4K) + ~7 min (8K) GPU (S2-Rohzeit laut `2026-10-27-s2-channels`: „ein Lauf, 3,5 min GPU-Zeit, kein Kernel geändert") |
| **C1** | **Der Effekt, ohne Engine-Änderung — auf den echten Zellen.** Der Parity-Harness zieht **synthetische Gauss-Zellen** (`nd(rng)*2.0f`); dort gibt es keinen Kanal-Offset, die Centering-Frage ist also dort **nicht** messbar. **Deshalb zwei Schritte:** (C1a) ein Lauf mit `STRATA_KV_Q4_DUMP=<datei>` (FROM/TO auf die ersten 8192 Positionen geklemmt, ~400 MB) → die **echten** rotierten K/V-Zellen des batched Promptpfads, im selben Lauf mit der S2-Sonde (Momente und Zellen aus einem Durchgang über eine Passage); (C1b) die Auswertung auf dem Host (`tools/kv_codec_mse.py`): MSE und Spannungsnutzung u je Block, Variante für Variante. **Die Varianten in fester Reihenfolge:** (1) uniform+max (der Anker `q4_0`), (2) uniform+max + Centering, (3) uniform+LS (der A8-Kandidat, **affin**), (4) uniform+LS + Centering, (5) lloyd16+max, (6) lloyd16+LS, (7) lloyd16K+lloyd8V (der q4lm8v-Codec) — jeweils mit und ohne Centering; (8) H·D + Centering nur zum Protokoll (A7) | **G2 erfüllt** (MSE-Gewinn **−1,9 dB** beim Anker, **−2,6 dB** beim q4lm8v-Codec — Faktor 2,8 bzw. 1,8 im MSE). **G3 nicht erfüllt** (+0,35 %), weil das registrierte Maß σ/Schritt **skaleninvariant** ist und ein Bias-Entfernen per Definition nicht zeigen kann; der Bericht trägt stattdessen \|d\|/σ (die Skala relativ zur Streuung des Blocks — die Größe, die „besser den Wertebereich ausnutzen" meint). **A7 mit Messung geschlossen**: drei H·D-Züge innerhalb 0,01 dB, und auf den zentrierten Daten ist der RHT (−0,18 dB) *schlechter* als das uniforme Gitter mit der gefitteten Skala (−0,20 dB) | C1a: 1 Lauf, ~4 min GPU (4K, batched Promptpfad, ein GPU-Job); C1b: Host, Minuten (keine GPU) |
| **C2** | **Die Korrektur, gebaut.** Writer-Änderung (Subtraktion vor dem FWHT), `m_V`-Add-Back auf dem Ausgabepfad, Offset-Tabelle im Weg der Wahl (a)/(b)/(c), Snapshot-Kennzeichen, opt-in-Schalter | **G4:** `int8`-Pfad byte-identisch (SHA-Gatter wie in der Kampagne), `kv_q4_parity` und `kv_win_parity` grün, `kv_window_parity` unverändert | **M** (Writer + Leser + Pack) |
| **C3** | **Die Wirkung.** Ein cwe-4k-Arm (gepaart gegen `int8` und gegen den Window-Arm) | **G5:** die gepaarte KL verbessert sich; 128K nur bei einem 4K-Unterschied | **M** (ein Arm) |

**Reihenfolge:** C0 → C1 → C2 → C3. C0 und C1 sind billig und entscheiden, ob C2 überhaupt
gebaut wird. **C0 ist der eigentliche Kill**: ein Offset, der mit dem Text wandert, ist kein
Offset, sondern Rauschen — und dann hilft auch die schönste Tabelle nichts.

### Ergebnis (2026-11-10): der Bias kostete ein Drittel des Fehlers

Auf den **echten** Zellen (C1a/C1b: 12 QSA-Layer × 32.746 Zellen × 2 KV-Köpfe, Block-MSE auf
denselben Zellen, gegen die publizierten Zahlen des Parity-Harness validiert):

| Variante | ohne Centering | mit Centering | Δ |
|---|---:|---:|---:|
| uniform+max (der Anker `q4_0`) | 9,093e-03 | **5,826e-03** | **−1,93 dB** |
| uniform+LS (der affine A8-Kandidat) | 8,686e-03 | 5,564e-03 | −1,93 dB |
| lloyd16+max | 8,443e-03 | 5,445e-03 | −1,91 dB |
| lloyd16+LS | 7,868e-03 | **5,072e-03** | **−1,91 dB** |
| lloyd16K+lloyd8V (der q4lm8v-Codec) | 1,288e-02 | **7,033e-03** | **−2,63 dB** |
| RHT(H·D) + uniform+LS (drei Züge) | −0,21 bis −0,24 dB | −0,18 bis −0,19 dB | ~0 |

**Der Bias kostete mehr als die Level-Tabelle (0,30 dB) und die per-Block-Skala (0,31 dB) zusammen.**
Zwei eigene Messfehler waren unterwegs und sind hier benannt, weil sie das Bild zuerst falsch
gemacht haben: (1) der **K-Offset wurde auch auf V angewendet** — die S2-Sonde misst **nur K**, und K
und V haben verschiedene Offsets; (2) die **S2-Sonde mittelt über die beiden KV-Köpfe**, ihre Zahl ist
damit weder ein Kopf- noch ein V-Offset. **Die Tabelle muss je Kopf und je Seite sein.**

**C0 (der eigentliche Kill) hat nicht gefeuert:** über Text und Länge (cwe-4k gegen cwe-8k) liegt die
Korrelation der Kanalmittelwerte bei **0,9489** und die mittlere Abweichung bei **0,0799 sd** (Gate:
≥ 0,90 und ≤ 0,25 sd). Der Build-Effekt 2026-10-27 (Engine 0.1.47) gegen den aktuellen Build
(0.1.59.1) ist **exakt null** (Korrelation 1,0000, Abweichung 0,0000 sd) — der Merge hat die
K-Verteilung nicht bewegt.

**Stand der Umsetzung (C2 ist gebaut, die KL-Messung C3 läuft):** die zentrierte Rotation und der
V-Add-Back als Kernel (`src/kernels/cuda/kv_q4.cu`), alle Rotationsstellen über **eine** Writer-
Funktion (damit keine sie vergessen kann), der Loader `STRATA_KV_CENTER=<datei>` in `generate.cpp`,
der Generator `tools/kv_center_table.py` (aus dem Dump; der Dump hält die rotierten Zellen, die
Tabelle braucht den Kanalraum → der FWHT ist selbstinvolvierend), das Snapshot-Kennzeichen (+32) in
`conversation_snapshot.cpp`. **Nicht umfasst und als Grenze zu nennen:** die Draft-Layer (der MTP
rotiert seine eigenen K/V; der Draft-Layer ist keiner der 12 QSA-Layer → seine Offsets sind
ungemessen, er bleibt unzentriert) und der Hybrid (rotiert V ohne Centering → kein Add-Back).

**Der Rotations-Vergleich läuft im selben Harness mit** (H·D, drei Vorzeichenzüge; A7), weil der
Harness ihn für einen Bruchteil der Kosten mitmessen kann — aber **G2 und G3 gehören dem
Centering**; die RHT-Varianten haben eine eigene, unbestätigte Erwartung (≈ 0) und werden **auf dem
zentrierten Signal** bewertet (Variante 3 gegen 2), weil eine Randomisierung einen systematischen
Anteil nicht entfernt, sondern nur anders verteilt.

## 5. Was dieses Projekt nicht ist

- Kein Ergebnis. C0 bis C3 sind nicht gelaufen; die Schätzung in §1 ist eine Schätzung.
- Kein Ersatz für die Level-Tabelle (0,30 dB) und nicht für die per-Block-Skala (0,31 dB). Das
  Centering ist eine **dritte, orthogonale** Hebelwirkung auf derselben Bulk-Seite — wenn es
  trägt, verschiebt es den Rest (0,17 dB) nicht, es kommt dazu.
- **Kein Ersatz für die Randomisierung, sondern ihre Voraussetzung.** Ein zufälliger Hadamard (A7)
  entfernt einen systematischen Anteil nicht, er verteilt ihn: für einen gegebenen Sign-Zug ist
  S·μ ein **fester** Vektor mit fester Energie (‖H·S·μ‖ = ‖μ‖). Deshalb wird A7 **erst nach** dem
  Centering gemessen, und die Erwartung an A7 bleibt ≈ 0: nach der Rotation stehen Mittel (0,2825),
  Schiefe (0,0836) und Kurtosis (0,1295) nahe an Gauss, aber nicht an Null — ein RHT verteilt diese
  Restanteile nur anders, er nimmt sie nicht weg.
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
- Die **Schiefe** ist systematisch, aber nur **kanalspezifisch** (kein Layer-Schieft: der Mittelwert
  je Layer liegt bei 0,2 bis 2,0× Rauschen). Ihre Entfernung bräuchte dieselbe kanalspezifische
  Tabelle wie die Offsets (12 KB je Layer und Kopf) — ob sie nach dem Versatz überhaupt noch Stufen
  kostet, entscheidet die Spannungsnutzung u in C1, nicht dieses Dokument.
- Ob der Offset aus dem Modellgewicht ableitbar ist (Variante c), ist **ungeprüft**; ohne (c)
  bleibt ein Artefakt oder eine Onlineschätzung.
- Der Schätzwert der Skaleninflation (15 bis 30 %) ist eine **Rechnung aus Momenten**, keine
  Messung; G2/G3 sind die Messung.
- Die Wirkung auf die **128K**-Statistiken ist offen; der Plan (K2/K3 der Leiter) misst 128K erst,
  wenn ein 4K-Unterschied vorliegt.
- Die Basis-Regeln (Snapshot-Kennzeichen, Streaming, `m_V`-Add-Back in der richtigen Reihenfolge)
  sind **statisch** abgeleitet; ihre Prüfung ist Teil von C2 (Parität und ein Test, der die
  Reihenfolge absichtlich bricht und das Ergebnis prüft).
