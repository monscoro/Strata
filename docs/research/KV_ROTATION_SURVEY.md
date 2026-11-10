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

| Moment | vor FWHT | nach FWHT | Rauschen | systematisch? |
|---|---|---|---|---|
| Mittel (1. Moment), \|mean\|/sd | Median **0,2673**, p90 0,7337, max 3,5778 | **0,2825**, p90 0,6967 | 0,0055 | **ja — und die Rotation entfernt ihn nicht** |
| Schiefe (3. Moment) | Median 0,1050, p90 0,3531, max 1,7199 | 0,0836, p90 0,2255 | 0,0135 | **ja, kanalspezifisch** (62,6 % der Kanäle > 5× Rauschen vor, 58,5 % nach) |
| Exzess-Kurtosis (4. Moment) | Median 0,2220, p90 0,8452, max 6,7842 | 0,1295, p90 0,4373 | 0,0271 | ja, aber die Rotation flacht es stark ab (65,0 % → 48,1 % über 5× Rauschen) |

**Der Kanal-Mittelwert ist echt, kein Rauschen** — der Median liegt rund 60-mal über der
Erwartung eines mittelwertsfreien Kanals. Und die Rotation **beseitigt ihn nicht**, sie verteilt
ihn: aus einem festen Offset je Kanal wird ein fester Bias in *jedem* rotierten Koordinatenpaar
(die Offsets addieren sich mit den Hadamard-Vorzeichen). Er ist auch **kein gemeinsamer DC**: der
Mittelwert der Kanalmittelwerte je Layer liegt bei −0,016 bis +0,086 bei einer Streuung der
Kanalmittel von 0,43 bis 0,64, also |mean|/sd_typ = 0,007 bis 0,068 — die Offsets sind
**kanalspezifisch**.

**Und die Konsequenz für randomisierte Verfahren:** Ein zufälliger Hadamard kann einen systematischen
Anteil nicht entfernen, er verteilt ihn nur. Für einen gegebenen Sign-Zug S ist S·μ ein **fester**
Vektor mit fester Energie (‖H·S·μ‖ = ‖μ‖, orthogonal) — der Bias landet zufällig, er verschwindet
nicht. Das ist der mathematische Kern von Flagrots Beobachtung („rotation flattens a spiky piece of
a key but can give a flat one a new peak"). **Darum ist die Reihenfolge fest: erst den
systematischen Anteil entfernen, dann über Randomisierung reden** — und die Randomisierung ist
danach erst sinnvoll messbar. Das Miniprojekt `bench/results/2026-11-10-centering/` folgt genau
dieser Reihenfolge und entfernt **alle** systematischen Momente, nicht nur den ersten (der 3. Moment
ist ebenfalls systematisch, aber nur kanalspezifisch — die Schiefe je Layer liegt bei 0,2 bis 2,0×
Rauschen, es gibt keinen „Layer-Schieft", den ein einziger Versatz räumen könnte).

**Eine Präzisierung zu §3:** Nach der Rotation stehen die Momente nahe an Gauss, aber nicht an Null
(Kurtosis 0,1295, Schiefe 0,0836 gegen Rauschen 0,0271 bzw. 0,0135). Die Aussage „praktisch
Gaussian" bleibt für die *Marginale* richtig; sie ist aber nicht „exakt Gauß", und die Restabweichung
ist per Kanal systematisch — sie ist also mit einem kanalspezifischen Codebook (Katalogzeile A5) zu
erreichen, nicht mit einer besseren Rotation.

### 4.3 Was der Bias kostet (gemessen am 2026-11-10, nicht geschätzt)

Auf den **echten** Zellen (Dump des Engines: 12 QSA-Layer × 32.746 Zellen × 2 KV-Köpfe; Block-MSE,
dieselben Zellen in jeder Variante; `tools/kv_codec_mse.py`, gegen die publizierten Zahlen des
Parity-Harness validiert):

| Variante | ohne Centering | mit Centering | Δ |
|---|---:|---:|---:|
| uniform+max (der Anker `q4_0`) | 9,093e-03 | **5,826e-03** | **−1,93 dB** |
| uniform+LS (der affine Kandidat) | 8,686e-03 | 5,564e-03 | −1,93 dB |
| lloyd16+max | 8,443e-03 | 5,445e-03 | −1,91 dB |
| lloyd16+LS | 7,868e-03 | **5,072e-03** | **−1,91 dB** |
| lloyd16K+lloyd8V (der q4lm8v-Codec) | 1,288e-02 | **7,033e-03** | **−2,63 dB** |
| lloyd16K+lloyd8V+LS | 1,192e-02 | 6,527e-03 | −2,62 dB |
| RHT(H·D) + uniform+LS (drei Züge) | −0,21 bis −0,24 dB | −0,18 bis −0,19 dB | ~0 |

**Der Bias kostete ein Drittel des Quantisierungsfehlers** — mehr als die Level-Tabelle (0,30 dB) und
die per-Block-Skala (0,31 dB) zusammen und mehr als der Rest bis zum konvergierten per-Block-Fit
(0,17 dB). Die frühere Schätzung in diesem Dokument (1–2 dB) war damit richtig; die dazwischen
gemessenen „kein Effekt"-Werte waren **zwei eigene Fehler**: der K-Offset wurde auch auf V angewendet
(K und V haben verschiedene Offsets), und die S2-Sonde mittelt über die beiden KV-Köpfe (sie misst
weder je Kopf noch V). **Die Tabelle muss je Kopf und je Seite sein.**

Warum die Max-Statistik so empfindlich ist: `d = amax / lmax` mit `amax` als Shuffle-Maximum über den
32er-Block (`src/kernels/cuda/kv_q4.cu:138-144`); ein fester Bias je Koordinate bläht dieses Maximum
auf, und die Skala wird an **alle** Werte des Blocks bezahlt. Die Messung bestätigt die Größenordnung
der Schätzung (Skala grob 15–30 % zu groß, MSE entsprechend).

## 4a. Die drei Rollen: Rotation, Centering, Randomisierung

| Verfahren | Achse | Was es leistet | Status |
|---|---|---|---|
| **Die feste Hadamard-Rotation** | **Ausreißer / Tail** | „incoherence processing … a form of outlier suppression" (QuIP 2023); „apply **online Hadamard transformations to the attention module to remove outlier features in keys and values**" (QuaRot 2024); „**outlier removal using rotation enables accurate token-wise or tensor-wise quantization**" (SpinQuant 2025) | **unverzichtbar** (Post-FWHT-Clip schadet: +34 % / +111 % KL, WP §7.11) **und** schnellpfad-kompatibel (die Abbildung ist affin) |
| **Das Centering (der Nullpunkt)** | **Mittelwert / Bias** | gemessen: **−1,9 bis −2,6 dB** MSE. In der Literatur ist derselbe Eingriff der **Nullpunkt** der asymmetrischen Quantisierung: „The outlier x₁ is spread across all channels as a constant offset x₁/√2, **which can be absorbed into the zero-point under asymmetric quantization**" (OffQ, arXiv:2606.07116) | **der größte offene Hebel** — von keiner Rotation ersetzbar |
| **Die Randomisierung (H·D)** | **Ausrichtungsrisiko / Schwanz** | schützt den *festen* Hadamard vor der unglücklichen Ausrichtung auf die Datenstruktur; die Theorie gibt die Inkoherenz-Schranke mit Wahrscheinlichkeit ≥ 1−δ **über den Zug** (QuIP#, Halko et al. 2011; „Quantizing With Randomized Hadamard Transforms: Efficient Heuristic Now Proven", 10/2026). Der kontrollierte KV-Vergleich (arXiv:2607.09683) findet: **auf isotropen oder niedrigrangigen Daten hilft die Rotation gar nicht** („the rotation neither helps nor hurts"), **auf schweren Rändern ist sie der einzige viable Weg** („TQ-KV is the only viable scheme") | **gemessen ohne Wirkung** (drei Züge innerhalb 0,01 dB; auf zentrierten Daten −0,18 gegen −0,20 dB) → **geschlossen für dieses Modell**, mit dem **Regime-Gate: die prä-Rotation-Exzesskurtosis** (hier −0,18, also *leichter* als Gauß → die Versicherung wird nicht gebraucht; für ein Modell mit Massenaktivierungen wäre der RHT die richtige Wahl) |

**Die Kernaussage, die alles trägt:** eine orthogonale Transformation **verteilt** die Energie eines
Vektors, sie **entfernt** sie nicht: ‖H·μ‖ = ‖μ‖. Ein fester Bias bleibt ein fester Bias, nur an
einer anderen Koordinate; ein zufälliger Zug verteilt ihn lediglich auf eine zufällige Koordinate
(FlagRot: „rotation flattens a spiky piece of a key but can give a flat one a new peak"). Die
Literatur ist sich einig, dass die **Rotation** die Ausreißer-Achse bedient und dass für die
**Mittelwert-Achse** ein **Nullpunkt** (asymmetrische Quantisierung) oder ein explizites Centering
das richtige Mittel ist — unsere Messung trennt beide Achsen erstmals am selben Codec.

**Was daraus für den Plan folgt:** Window-Overlay und Centering sind **orthogonale** Hebel (der
Window lieferte den KL-Sprung 0,0318 → 0,0209, das Centering den MSE-Gewinn); Level-Tabelle und
per-Block-Skala sind es auch (0,30 bzw. 0,31 dB); die Randomisierung ist auf unseren Daten **kein**
Hebel (gemessen). Die Rotation selbst bleibt, weil sie die Ausreißer-Achse bedient **und** weil nur
eine affine Abbildung in den Tensor-Core-Promptpfad passt (der Lloyd-Modus nicht: der Kern rechnet
`Σ q_i·(code−8)` und wendet die Skala danach an — eine nicht-affine Level-Tabelle lässt sich so nicht
rechnen; der Preis ist der Decode-Kern als Rückfall, im B1-Pfad als +12 % gemessen, im 128K-Prompt
**mit Fremdlast im Verdacht und deshalb zurückgezogen**).

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
   **Und sie ist die Voraussetzung der Rotationsachse, nicht deren Alternative:** ein zufälliger
   Hadamard entfernt einen systematischen Anteil nicht, er verteilt ihn nur (§4.2). Die Reihenfolge
   ist damit fest: **erst den Bias entfernen, dann die Randomisierung messen** — und die
   Randomisierung wird auf dem zentrierten Signal gemessen (Variante 3 gegen 2), weil man sonst den
   Bias misst und nicht den Nutzen der Vorzeichen. Konsequent heißt auch: der 3. Moment (Schiefe)
   ist ebenfalls systematisch und wird mitentfernt oder als Rest benannt (§4.2).
3. **Beide brauchen denselben billigen Test:** `kv_q4_parity` (487 Zellen × 2 KV-Köpfe × 256
   Werte = 498.688 rotierte Werte, 15.584 Blöcke) vergleicht die Block-MSE auf **denselben**
   Zellen. Vier Varianten in fester Reihenfolge: (1) H (Baseline), (2) H + Per-Kanal-Centering,
   (3) H·D + Centering, (4) H·D allein nur zum Protokoll. Vorab festzulegende Regel:
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
| **OffQ** (arXiv:2606.07116, 2026-06) | die Hadamard-Streuung eines Ausreißers ist ein **konstanter Offset**, der in den **Nullpunkt** der asymmetrischen Quantisierung wandert; und: Bias-Terme in den Attention-Schichten (Qwen 2.5 u. a.) schiefen die Aktivierungen und bringen Ausreißer in den KV-Cache — dieselbe Fehlerklasse wie unser Kanal-Offset |
| **Transforms for LLM Quantization: The Great Inversion and Format Co-Design** (arXiv:2608.25188, 2026-08) | der erste Survey nur über die Transformationsstufe; „a transform … **redistributes** the outlier's energy without discarding any of it, which is why flattening rather than clipping is the mechanism" |
| **TORQUE** (arXiv:2609.36032, 2026-09) | die Rotation **verdeckt** nützliche Struktur (ein Ausreißer wird über alle Kanäle gestreut); die Erhaltung der Ausreißer **vor** der Rotation ist eine eigene Entscheidung |
| **BATQuant** (ECCV 2026) | „global orthogonal rotations inadvertently **transfer outlier energy across quantization blocks**, inducing new outliers that disrupt local block-wise scaling" — die blockweise Sicht auf genau unser Problem (das Blockmaximum) |
| **DRIVE / Ben-Basat et al.** (arXiv:2605.06014, 2026-05) | die RHT ist **jetzt bewiesen** statt heuristisch; und die Arbeit zerlegt den Fehler in Varianz und **Bias** („the NMSE decomposes into the aggregated variance and squared bias") — genau unsere zwei Achsen |
| **Ablation, Statistical Inference, and Validation for KV-Cache Compression** (D'Alberto, Sirasao, Delaye, Patwari; AMD; arXiv:2607.09683, 2026-06) | der kontrollierte Vergleich von **TurboQuant** (randomisierter Walsh-Hadamard + analytisches Beta-Lloyd-Max-Codebook) gegen **SpectralQuant** (pro-Kopf-Eigenbasis + Water-Filling): „**the rotation neither helps nor hurts**" auf isotropen/niedrigrangigen Daten, „**TQ-KV is the only viable scheme**" auf schweren Rändern; die FULL-Projektion (statt RHT) ist **dominiert**; die Eigenbasis ist **fragil** (ein spurious Eigenwert → Ausgabe antikorrelliert) und der Schwanz leckt in die Ausgabe; Water-Filling bringt **nichts**; die Empfehlung für 4 Bit und lange Sequenzen ist der strukturierte Hadamard. TQ rotiert übrigens `WHT(k ⊙ s)/√d` mit **pro-Kopf**-Zeichen — genau unser Messobjekt. Und: eine Restfehler-Korrektur auf dem **K**-Pfad (selbst ein randomisierter Hadamard) wird durch den Softmax **exponentiell** verstärkt (Jensen), auf dem V-Pfad nicht → eine Warnung für eigene K-Pfad-Korrekturen |
| **TurboQuant** (Zandieh et al., ICLR 2026, arXiv:2504.19874) | unser Lloyd-Maxi-Codec ist die TQ-Idee mit einer **Gauß-** statt einer Beta-Tabelle (TQ: ein analytisches Lloyd-Max-Codebook für die rotierte Verteilung) |
| **SpectrumQuant** (arXiv:2610.20303, 2026-10) | die Rotations**auswahl** aus der Kovarianz: eine quantisierungsfreundliche Transformation ist „alignment + mixing", der Hadamard ist ein Spezialfall; ein kovarianzbewusster Transform schlägt den Hadamard im MSE (rund 2 %) |
