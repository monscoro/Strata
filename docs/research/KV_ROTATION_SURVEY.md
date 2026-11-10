# Rotationsachse im KV-Codec: Hadamard gegen randomisierten Hadamard (RHT) — und der feste Kanal-Offset

**Status: Literaturrecherche mit einer Auswertung eigener Rohdaten (2026-11-10). Keine neue
Messung.** Jede Zahl aus der Literatur ist mit Quelle genannt; die eigenen Zahlen stammen aus
`bench/results/2026-10-27-s2-channels/` (Rohdaten `Strata-data/bench/2026-10-27-s2-channels.txt`).
Der eine Schätzwert (Skaleninflation) ist als Schätzung markiert und **nicht gemessen**.

**Worum es geht:** Unser Codec rotiert K und V vor dem Schreiben mit dem Walsh–Hadamard über
`head_dim = 256` (`src/kernels/cuda/kv_q4.cu:26`, `fwht256_kernel`, orthonormal, Skala 1/16). Die
Frage war, ob es ein besseres Diagonalisierungs-/Rotationsverfahren gibt. Antwort der Recherche:
**als Kandidat für den Codec nein (Erwartung ≈ 0 dB), als Hinweis auf eine andere, billigere
Hebelwirkung ja** — der feste Kanal-Offset, den die Rotation nicht entfernt, sondern in einen
festen Bias verwandelt.

## 1. Die Literatur meint mit „randomized Hadamard" etwas anderes als „FHRT"

| Begriff | Definition | Quellen |
|---|---|---|
| **RHT** (Randomized Hadamard Transform) | **H·D**: Hadamard × zufällige Vorzeichendiagonale | QuIP# (ICML 2024, arXiv:2402.04396), QuaRot (NeurIPS 2024, arXiv:2404.00456), QTIP, SpinQuant (ICLR 2025, arXiv:2405.16406), HARP (EMNLP 2026, arXiv:2605.29843) |
| **FHRT** (Fast Randomized Hadamard Transform) | **H·D₁·Π·D₂** mit zusätzlicher zufälliger Permutation Π (Haar-Approximation) | Ho et al.; im LLM-Quantisierungsbereich **nicht verwendet** |
| URR (Uniform Random Rotation) | echte Zufallsrotation | O(d²) — zu teuer, genau deshalb der RHT |

**Befund 1: Unser Kandidat war das falsche Objekt.** QuIP# nennt „Randomized Hadamard Transform"
und meint die sign-only-Variante H·D; die Permutation π fehlt dort. Die Theoriearbeit „Quantizing
With Randomized Hadamard Transforms: Efficient Heuristic Now Proven" (Ben-Basat et al., UCL /
Broadcom / CMU / Harvard / Microsoft / VMware Research, 2026-10) beweist, dass diese Variante
hinreicht, und begründet, warum man bei URRs nicht bleibt (O(d²)). Die Permutation π erscheint in
der Literatur nur in **adaptiven** Varianten: RotateKV (IJCAI 2025) nutzt ein outlier-bewusstes
**Channel-Reordering** (offline Statistiken, bleibt FWHT-schnell), OSCAR (2026) leitet die
Rotationen aus offline gemessenen aufmerksamkeitsgewichteten Kovarianzen ab. Beides braucht
Kalibrierdaten und ein Artefakt im Modell — bei uns die teuerste denkbare Währung.

Für unsere Implementierung heißt das: **H·D ist der Kandidat, nicht das volle FHRT.** Der
Vorzeichenteil ist bei uns praktisch gratis (unser Ladevorgang multipliziert schon mit der Skala
`scale`, kv_q4.cu:41: `reg[i] = row_src[i * warp_size + lane] * scale` → die Vorzeichen falten
sich in dieselbe Multiplikation), der Permutationsteil wäre der teure (gestreute Gather statt
gestreuter Loads).

## 2. Befund 2: der Transform ist nicht mehr selbst-invers — und unser Code benutzt ihn an vier Stellen

Unser FWHT ist selbst-invers und wird an vier Stellen aufgerufen: K/V-Schreibpfad
(`src/core/layer.cpp:1472-1513`), Query (`layer.cpp:1559`), Aufmerksamkeits-Ausgabe
(`layer.cpp:1588`), plus die Ring-Schreiber. Mit T = H·D gilt:

- Schreiben und Query: **erst D, dann H** (dieselbe Reihenfolge)
- Ausgabe: o = T⁻¹o′ = **D·H** o′ → **erst H, dann D** (umgekehrte Reihenfolge)

Ein naives „Vorzeichen überall einbauen" bricht die Ausgabe **still**: das Modell läuft weiter, nur
schlechter. Im Butterfly-Netz ist das billig lösbar — die Vorzeichen gehören an den Eingang der
ersten Stufe (→ H·D) oder an den Ausgang der letzten (→ D·H); es braucht also **zwei** Varianten
des Netzes und einen Schalter im Kernel. Die Literatur umgeht das lieber, indem sie die
V-Rotation samt Inversem in die Projektion und W_O absorbiert (OptR, arXiv:2608.02691: „The value
rotation and its inverse are absorbed into the value projection and W_O, respectively, avoiding
separate value-side rotation kernels") — für K können wir das nicht (Post-RoPE), für V vielleicht.

## 3. Befund 3: „Rotation hilft nicht immer" — und unsere eigene Messung sieht genauso aus

**FlagRot** (arXiv:2611.03992, 2026-11-02) formuliert die Gegenposition: „The usual remedy
rotates keys before quantizing, **assuming rotation always helps. However, we observe that it
does not: rotation flattens a spiky piece of a key but can give a flat one a new peak**, and both
coexist within one key." FlagRot entscheidet deshalb je 32-Kanal-Stück (rotiert oder nicht) und
speichert die Wahl in vorhandenen Sign-Bits; bei per-channel Caches genügt eine Entscheidung je
Kopf.

Das ist der **theoretische Schatten unseres eigenen WP §7.9** (Hadamard an/aus = Gleichstand:
cwe-2k t +0,21; cwe-4k t −1,31 / −0,17; nirgends signifikant) und unser S2-Befund erklärt ihn:
der Crest (max|x|/sd) ist **pre 3,72 kleiner als post 3,93** — die Rotation **erhöht** bei unseren
Daten den Spitzenwert, statt ihn zu senken.

**Einschränkung der Evidenzbasis:** Die Rotationsgewinne der Literatur gelten überwiegend der
**per-token/per-vector**-Quantisierung (KIVI+Had in NSNQuant, arXiv:2505.18231: „Hadamard
transform is known to reduce errors in token-wise quantization a lot"). Unser Codec ist
**blockweise** (32 Werte, eine Skala je Block); dort ist der Dynamikbereich-Gewinn strukturell
viel kleiner. Unsere eigene Bilanz auf der Bulk-Seite stützt das: Level-Tabelle 0,30 dB,
per-Block-Skala 0,31 dB, Rest bis zum konvergierten per-Block-Fit **0,17 dB**
(`kv_q4_parity`, gfx1200: q4_0 2,938e-02 = 0 dB, kLloyd 2,740e-02 = −0,30 dB, kLloydLs 2,551e-02 =
−0,61 dB, konvergierter Fit 2,456e-02 = −0,78 dB).

**Unabhängige Gegenprobe zu unserem S2:** Die Interface-aware-NVM-Arbeit (arXiv:2609.05764,
2026-09) misst „after the Hadamard transform and normalization, a key coordinate is close to
Gaussian, with measured excess kurtosis ≈ −0.16". Unser S2 maß −0,14 (Mittel über 3.072 Kanäle,
12 Layer) — dieselbe Größenordnung, von einer anderen Gruppe, an anderen Daten.

## 4. Befund 4: der feste Kanal-Offset — die interessantere Hebelwirkung

### 4.1 Was die Literatur 2026 dazu sagt

**OptR** (arXiv:2608.02691, 2026) zentriert die Keys **vor** der Rotation: „OptR first centers the
keys before rotation and quantization. **This shifts all logits for a query by the same constant and
therefore leaves the softmax distribution unchanged.**" Ihre Rotation ist per Kopf gelernt, der
Mittelwert wird offline berechnet und nach der Rotation abgezogen (mathematisch dasselbe wie ein
Centering davor). Der llama.cpp-Fork (PrismML-Eng, PR 2026-07) führt ein K-Cache-Mean-Centering
mit einem **Calibration-Basis-Guard** ein: „a bias calibrated with the Hadamard K-cache rotation
active must be applied with the rotation active, and vice versa. a mismatched basis measurably
degrades quantization quality instead of improving it" — der Versatz wird im rotierten Fall also
**abgelehnt**, nicht still angewendet.

Warum das erhaltend ist, formal: K → K − m verschiebt alle Logits **einer** Query um dieselbe
Konstante ⟨q, m⟩ (Softmax unverändert); V → V − m verschiebt die Ausgabe um m, weil die
Aufmerksamkeitsgewichte Σaᵢ = 1 erfüllen. Die Selektion sieht nichts davon (unser Indexer-Key
ist die Projektion von `x`, layer.cpp:1399-1407, **vor** dem FWHT ab 1472).

### 4.2 Was unsere eigenen Rohdaten sagen (Auswertung vom 2026-11-10)

Quelle: `bench/results/2026-10-27-s2-channels/` (Rohdaten `Strata-data/bench/2026-10-27-s2-channels.txt`,
377 kB, 3.072 Kanäle = 12 Layer × 256, je 32.746 Token). Die S2-Sonde akkumuliert je Kanal unter
anderem Mittel und Varianz — der Offset ist also **bereits gemessen**, er war nur nie gelesen
worden. Auswertung: standardisierter Kanal-Mittelwert |mean|/sd.

| | vor FWHT | nach FWHT |
|---|---:|---:|
| Median | **0,2673** | **0,2825** |
| p90 | 0,7337 | 0,6967 |
| Maximum | 3,5778 | 1,6561 |
| Sampling-Erwartung bei n = 32.746 | 0,0044 | 0,0044 |

**Der Kanal-Mittelwert ist echt, kein Rauschen** — der Median liegt rund 60-mal über der
Erwartung eines mittelwertsfreien Kanals. Und die Rotation **beseitigt ihn nicht**, sie verteilt
ihn: aus einem festen Offset je Kanal wird ein fester Bias in *jedem* rotierten Koordinatenpaar
(die Offsets addieren sich mit den Hadamard-Vorzeichen). Er ist auch **kein gemeinsamer DC**: der
Mittelwert der Kanalmittelwerte je Layer liegt bei −0,016 bis +0,086 bei einer Streuung der
Kanalmittel von 0,43 bis 0,64, also |mean|/sd_typ = 0,007 bis 0,068 — die Offsets sind
**kanalspezifisch**.

### 4.3 Warum das bei uns teuer ist (Schätzung, keine Messung)

Unsere Skala ist eine **Max-Statistik**: `d = amax / lmax` mit `amax` als Shuffle-Maximum über den
32er-Block (`src/kernels/cuda/kv_q4.cu:138-144`), `lmax` = 87 (16 Stufen) bzw. 69 (8 Stufen). Ein
fester Bias je Koordinate bläht dieses Maximum auf.

- Nach der Rotation: sd ≈ 1,1 bis 1,67 (Var-Spreizung 1,2 bis 2,8), Bias |mean_post| ≈ 0,283 · sd
- Signalmaximum über 32 Koordinaten ≈ 2,1 · sd; Biasmaximum über dieselben Block ≈ 2,1 · 0,283 · sd
  ≈ 0,6 · sd
- → die Skala ist um grob **15 bis 30 %** zu groß, der MSE skaliert mit d² → **etwa +1 bis +2 dB**

Das ist **eine Schätzung aus den Momenten, keine Messung**: die Blockmaxima addieren sich nicht
exakt, und der MSE-Effekt ist am echten Zellensatz zu messen. Aber die Größenordnung ist
auffällig: sie ist **größer als der gesamte bisherige Hebel auf der Bulk-Seite** (Level-Tabelle
0,30 dB + Skala 0,31 dB = 0,61 dB) und um Größenordnungen größer als die Erwartung für einen RHT.

### 4.4 Die Kosten

- Die Offset-Tabelle ist winzig: 12 Layer × 2 KV-Köpfe × 256 Kanäle = 6.144 Werte (12 KB in fp16).
- Aber sie ist eine **kalibrierte** Größe (Text wie cwe) und damit ein echtes Pack-Artefakt —
  dieselbe Währung, die eine gelernte Rotation bräuchte, aber um Größenordnungen billiger.
- Die Basis-Falle aus dem llama.cpp-PR gilt: die Offsets müssen in der Basis gemessen werden, in
  der sie angewendet werden (unrotiert, je Kanal) — und jeder Zustand, der einen K-Zustand
  weiterreicht (Snapshot, Streaming, das Ringfenster), muss dieselbe Basis mitführen.
- Das Ringfenster trägt die exakte fp16-Zwillinge: sie müssten die zentrierten Werte führen
  (konsistent, weil der Ring dieselben rotierten Werte spiegelt).

## 5. Was daraus folgt

1. **Die Rotationsachse (RHT / FHRT) hat für unseren Codec Erwartung ≈ 0 dB.** Belege: unsere
   eigene Hadamard-an/aus-Ablation ist ein Gleichstand; unser Crest steigt durch die Rotation
   (3,72 → 3,93); die Literatur warnt explizit, dass Rotation nicht immer hilft (FlagRot); die
  literarischen Gewinne gelten per-token/per-vector, unser Codec ist blockweise. Das ist **kein
   Kill** (es ist nicht gemessen), aber eine Erwartung, die man vor dem Bauen festlegt.
2. **Die Offsetachse (Per-Kanal-Centering vor der Rotation) ist der interessantere Kandidat.**
   Erwartung aus den Momenten: 1 bis 2 dB MSE, also mehr als alles, was die Bulk-Seite bisher
   gebracht hat; exakt erhaltend (Softmax und Ausgabe); billig (12 KB, ein Kalibrierlauf).
3. **Beide brauchen denselben billigen Test:** `kv_q4_parity` (487 Zellen × 2 KV-Köpfe × 256
   Werte = 498.688 rotierte Werte, 15.584 Blöcke) vergleicht die Block-MSE auf **denselben**
   Zellen. Vier Varianten: (1) H (Baseline), (2) H·D mit **drei** Vorzeichenzügen (SpinQuants
   Lehre: der Zug ist eine Lotterie, bis 13 Accuracy-Punkte Streuung — also Streuung berichten,
   nicht ein Zug), (3) H + Per-Kanal-Centering, (4) H·D + Centering. Vorab festzulegende Regel:
   ein Gewinn unter ≈0,05 dB (Rauschteppich des Werkzeugs) schließt die Achse; ein Centering-Gewinn
   über ≈0,3 dB (Größenordnung der Level-Tabelle) geht in die KL-Kette (cwe-4k, dann 128K).

## 6. Grenzen

- Die Offset-Auswertung ist eine Auswertung **einer** Passage (cwe-4k, 4K) an **einem** Layer-Set.
  Ob der Offset über Positionen, Passagen und Kontextlängen stabil ist, ist **ungeprüft**; ein
  Versatz, der mit dem Kontext wandert, wäre kein Versatz, sondern Rauschen. Das ist vor einem
  Bau zu messen (z. B. zwei Texte, zwei Längen).
- Die Skaleninflation ist eine Schätzung aus Momenten, keine Messung.
- Die Rotationsachse ist hier **nicht** gemessen; die Erwartung stützt sich auf fremde Messungen
  (unsere §7.9-Ablation) und fremde Modelle (FlagRot, SpinQuant).
- Nichts davon ist in den Engine gebaut. Kein Default-Pfad ist angefasst; der `int8`-Pfad bleibt
  byte-identisch.

## 7. Quellen

| Quelle | Was wir daraus nehmen |
|---|---|
| QuIP# (Tseng et al., ICML 2024, arXiv:2402.04396) | RHT = H·S; die Definition, die in der LLM-Quantisierung „randomized Hadamard" heißt; die Inkoherenz-Schranke μ = 2 log(4mn/δ) |
| QuaRot (Ashkboos et al., NeurIPS 2024, arXiv:2404.00456) | H̃ = H·diag(s) orthogonal; **online** Hadamard im Attention für K und V (KV-Cache); Fusion in die Gewichte via Computational Invariance |
| QTIP (NeurIPS 2024) | RHT, damit die Gewichte ≈ i.i.d.-Gaussian sind |
| SpinQuant (Liu et al., ICLR 2025, arXiv:2405.16406) | Zufallsrotationen streuen stark (bis 13 Punkte Accuracy); gelernte Rotationen schlagen sie; R3/R4 als **online** Hadamards für den KV-Cache; nach Cayley-Optimierung zählt die Initialisierung weniger |
| HARP (EMNLP 2026, arXiv:2605.29843) | gelernte strukturierte Rotation; bei Θ = 0 exakt QuIP#s RHT; die Ausweitung auf den KV-Cache verlangt graph-level placement-Koordination |
| „Quantizing With Randomized Hadamard Transforms: Efficient Heuristic Now Proven" (2026-10) | RHT = H·D ist ausreichend; URRs kosten O(d²) |
| FlagRot (arXiv:2611.03992, 2026-11-02) | Rotation hilft nicht immer; Entscheidung je 32-Kanal-Stück; bei per-channel Caches je Kopf |
| RotateKV (Su et al., IJCAI 2025) | outlier-bewusstes Channel-Reordering (der adaptive Ersatz für π) + Pre-RoPE-Grouped-Head-Rotation + Sink-Schutz |
| OptR (arXiv:2608.02691, 2026) | Mean-Centering vor der Rotation ist softmax-erhaltend; per-Kopf-Korrekturen; V-Rotation in Projektion/W_O absorbiert |
| OSCAR (2026) | Rotationen aus offline gemessenen aufmerksamkeitsgewichteten Kovarianzen |
| NSNQuant (arXiv:2505.18231) | Hadamard **nach** RoPE (wie bei uns); der Hadamard-Gewinn gilt der tokenweisen Quantisierung |
| KVQuant (Hooper et al., NeurIPS 2024, arXiv:2401.18079) | per-channel Keys vor RoPE; NUQ; dense-and-sparse |
| Interface-aware NVM KV (arXiv:2609.05764, 2026-09) | RHT = Hadamard + random sign flip; post-Hadamard Key-Exzesskurtosis ≈ −0,16 (unabhängige Gegenprobe zu unserem −0,14); dieselbe Matrix rotiert den Query |
| llama.cpp PR (PrismML-Eng, 2026-07) | K-Cache-Mean-Centering **mit** Calibration-Basis-Guard: ein Versatz aus der falschen Basis verschlechtert messbar |
