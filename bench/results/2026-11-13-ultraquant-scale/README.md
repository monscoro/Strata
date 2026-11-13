# UltraQuants Skalenmechanismus, auf unserem Gitter nachgemessen (2026-11-13)

Diese Stufe prüft **einen** Literaturmechanismus an unseren eigenen Zellen, ohne GPU und ohne Build:
den Skalenmechanismus von **UltraQuant** (Chakrabarti et al., AMD, arXiv:2606.20474v3, EMNLP 2026
Industry Track). Die Datei `docs/research/KV_ZERO_POINT_SURVEY.md` (Z. 152–190) beschreibt das
Paper bereits; **was dort fehlt, ist der Preis des Mechanismus auf unserem Gitter.** Genau den
misst diese Stufe.

## Was die Quelle sagt, und was sie nicht sagt

| Aussage | Stelle | Status |
|---|---|---|
| Skala ist eine Zweierpotenz, `E = round(log2(c·m))`, `m` = Absmax der Gruppe | §5.2/§5.3 | belegt (Volltext) |
| `c = 0,156`, eine globale Konstante, modellunabhängig | §5.3, App. B.2 | belegt (Volltext) |
| 17 B je 32 Kanäle = 4,25 bit | §5.1 | belegt (Volltext) |
| Per-Block-Absmax trägt, das Codebook „barely matters" | App. B.1 | belegt (Volltext) |
| **0,001543** | — | **nicht belegt.** Weder im Paper (Volltextsuche über 248.923 Zeichen: kein `0.0015`, kein `1543`) noch in `KV_ZERO_POINT_SURVEY.md`. Es ist ein **Skalenwert** (c = 0,156), kein KL-Wert, und nicht direkt mit unseren nats vergleichbar. Die Zahl wird **nicht** verwendet. |

## Die drei Fragen

1. **Die zweite LS-Passade.** `q4u32` (`q4u_ls_group`, `src/kernels/cuda/kv_q4.cu:238`–278) startet
   bei `d = mval/-8`, weist die Codes zu, macht **eine** LS-Passade und weist die Codes neu zu — dann
   ist Schluss. Der Shear läuft `kShearPasses = 2` alternierende Passaden. Kostet eine zweite Passade
   nur Writer-Butterflies (kein Leser, kein Format), dann ist die Frage allein: **was bringt sie?**
2. **Die Exponent-Skala.** UltraQuant erzwingt die Zweierpotenz, weil die Hardware (UE8M0,
   scaled-MFMA) sie so liest. Bei uns ist sie **optional**. Ein Exponent statt fp16 spart **1 Byte je
   18-Byte-Block** (18 → 17 B, 4,5 → 4,25 bit). Die Frage: **was kostet der Snap?**
3. **Die Konstante c = 0,156.** Sie ist für das **E2M1-Gitter** optimiert, dessen weitester Codepunkt 6
   ist. Unser Gitter reicht bis 8 (`(code-8)*d`), also wäre der gittereigene Wert 0,125. Die Frage:
   **trägt c = 0,156 auf unser Gitter?**

## Der Aufbau

* Werkzeug: `tools/kv_q4_passes.py` (neu, host-seitig, kein GPU-Job).
* Daten: `cells-cwe4k.bin` aus `2026-11-10-centering` — die **rotierte** K/V-Zeile des gebatchten
  Prompt-Pfads (Mode 4), also genau die Zellen, für die `q4u32` gebaut ist.
* Stichprobe: **1.571.808 Blöcke** (`--stride 8`), K und V getrennt, keine Zentrierung, kein Fenster.
  Block = 32 Werte entlang `head_dim`, wie im Kernel.
* `d` wird überall auf **fp16** gerundet, wo der Kernel es speichert (`f16_from_f32`).
* Anschlussprüfung: `p1-f64` = **8,635958e-03** gegen die dokumentierten **8,693901e-03**
  (`bench/results/2026-11-13-kv-offset-tail/README.md`, `uniform+LS`) — **0,7 %** Abweichung, die von
  der Stichprobe kommt. Das Werkzeug liest also dieselben Zellen wie die frühere Messung.

## Die Entscheidung, **vor** dem ersten Lauf festgelegt

| Gewinn gegen den heutigen Writer | Folge |
|---|---|
| ≥ 1,0 dB | Build: eigener End-to-End-Arm |
| 0,3 – 1,0 dB | Build nur, wenn zusätzlich ein Format- oder Lesergewinn herauskommt |
| < 0,3 dB | kein Build, Nullbefund dokumentieren |

Für den Snap gilt dieselbe Skala mit umgekehrtem Vorzeichen (Verlust statt Gewinn).

## Das Ergebnis

Gewinn in dB gegen **p1** (heute: eine LS-Passade, `d` als fp16); positiv = besser.

| Variante | Block-MSE | K | V | Gewinn |
|---|---:|---:|---:|---:|
| `p1-f64` (ohne fp16-Rundung) | 8,635958e-03 | 1,468271e-02 | 2,589208e-03 | +0,000 |
| **`p1` (heute)** | 8,635983e-03 | 1,468275e-02 | 2,589214e-03 | 0 |
| `p2` (zweite LS-Passade) | 8,479628e-03 | 1,441756e-02 | 2,541693e-03 | **+0,079** |
| `p3` (dritte) | 8,419549e-03 | 1,431547e-02 | 2,523625e-03 | **+0,110** |
| `p1-snap` (2^round) | 1,509879e-02 | 2,571878e-02 | 4,478798e-03 | **−2,426** |
| `p2-snap` | 1,516888e-02 | 2,583254e-02 | 4,505230e-03 | −2,446 |
| `p1-snap2` (bester aus 2^floor/2^ceil) | 1,310373e-02 | 2,218030e-02 | 4,027166e-03 | **−1,811** |
| `p2-snap2` | 1,310373e-02 | 2,218030e-02 | 4,027166e-03 | −1,811 |
| `uq156` (d = 0,156·absmax, kein LS) | 1,461594e-02 | 2,479320e-02 | 4,438687e-03 | −2,285 |
| `uq156-snap` | 1,502303e-02 | 2,529464e-02 | 4,751411e-03 | −2,404 |
| `uq125` (d = 0,125·absmax, unser Gitter) | 1,081535e-02 | 1,830888e-02 | 3,321820e-03 | −0,977 |
| `uq125-snap` | 1,685176e-02 | 2,848983e-02 | 5,213699e-03 | −2,903 |

### 1. Die zweite Passade: **0,079 dB — kein Build**

Die LS-Passade ist nach **einer** Iteration praktisch erschöpft. Die dritte Passade gewinnt gegenüber
der zweiten nur noch 0,031 dB. Nach der vorab fixierten Skala (< 0,3 dB) ist das ein **Nullbefund**:
die Hypothese „eine zweite LS-Passade nach der Neu-Zuweisung holt Numerik" ist **widerlegt**.

Zum Maßstab: der **ganze** Shear ist 1,29 dB Block-MSE wert und war end-to-end ~0,0014 nats. 0,079 dB
sind damit ~0,00009 nats — **ein Zwanzigstel** der 4K-Paritätsauflösung (0,00200) und ein
Neunzehntel der 128K-Auflösung (0,00254). Eine solche Änderung ist mit keinem Text dieses Projekts
nachweisbar. Der Writer-Aufwand (eine weitere Passade = eine Neu-Zuweisung + zwei Butterfly-Summen)
wäre real, der Gewinn nicht.

### 2. Die Exponent-Skala: **−1,81 dB im besten Fall — kein Build**

Der Snap ist der teure Teil, nicht das Byte. `2^round` kostet **2,43 dB**; selbst wenn der Writer den
besseren von `2^floor` und `2^ceil` wählt (zwei Code-Zuweisungen, ein MSE-Vergleich — immer noch reine
Writer-Arbeit), bleiben **1,81 dB** Verlust. Das entspricht ~**0,0020 nats**, also **einer vollen
128K-Paritätsauflösung** — für **1 Byte je 18-Byte-Block** (5,6 % weniger KV-Bytes).

Damit ist die Aussage in `KV_ZERO_POINT_SURVEY.md` Z. 181–184 („die Skala als Exponent ist eine
konkrete Verbesserung unseres Formats") **auf unserem Gitter nicht bestätigt**: sie ist eine
Verbesserung des **Formats** (1 Byte) und ein **Verlust an Numerik** (1,81 dB). Bei UltraQuant ist der
Snap durch die Hardware erzwungen; bei uns wäre er freiwillig, und freiwillig ist er zu teuer.

Die Pointe steht in der ersten Zeile: unser **fp16**-`d` kostet gegen die ungerundete Rechnung
**0,000 dB** (8,635958e-03 gegen 8,635983e-03, 3·10⁻⁶ relativ). Das zweite Byte ist numerisch gratis;
das gesparte Byte wäre es nicht.

`p1-snap2` und `p2-snap2` sind **bitgleich**: sobald `d` gesnappt ist, liegt es auf demselben
Zweierpotenz-Intervall, die Wahl `floor`/`ceil` fällt gleich aus, und die Codes sind dieselben. Der
Unterschied zwischen p1 und p2 (0,079 dB) verschwindet im Snap-Raster — ein weiterer Beleg, dass die
Passaden-Frage unterhalb der Snap-Auflösung liegt.

### 3. Die Konstante c: **nicht übertragbar**

* `c = 0,156` (UltraQuant, E2M1-Spitze 6): **2,29 dB schlechter** als unser LS-Fit.
* `c = 0,125` (unser Gitter, Spitze 8): **0,98 dB schlechter**.
* Mit Snap: 2,40 bzw. 2,90 dB schlechter.

Der Grund ist strukturell und nicht überraschend: c ist ein **Startwert** für die Skala. Wir bezahlen
ihn nicht, weil wir ihn im Writer durch **eine** LS-Passade ersetzen, und die ist auf unserem Gitter um
0,98–2,29 dB besser. UltraQuants „eine globale Konstante, modellunabhängig" ist bei ihnen die Folge
der Hardware (die Skala muss vor dem Kernel feststehen); bei uns ist derselbe Hebel **kostenlos
nachziehbar** und wird deshalb besser.

Was von UltraQuant **bleibt**, ist die Bestätigung, nicht der Mechanismus: ihre Ablation B.1 („per-block
absmax ist die tragende Änderung, das Codebook barely matters") ist unser Arm-Befund von der anderen
Seite (Lloyd-Form gegen uniformes Gitter: +0,00050 nats, n.s.). Und ihre Kernaussage „4-bit KV buys
capacity, not kernel speed" ist unser Fenster-Argument.

## Was diese Stufe nicht sagt

* **Nichts über Laufzeit.** Der Snap würde den Leser vielleicht billiger machen (Exponent-Shift statt
  fp16-Konvertierung), aber das ist eine Kernel-Frage und hier nicht gemessen. Der Qualitätspreis ist
  gemessen und entscheidet die Frage vorher.
* **Nichts über End-to-End-nats.** Die dB→nats-Brücke (1 dB ≈ 0,0011 nats) ist eine Übersetzung über
  den Shear-Punkt, keine zweite Messung.
* **Nichts über andere Texte oder Modelle**: cwe-4k-Zellen, ein Modell (Swift-Qwen3.8-Flash-Next,
  IQ3_XXS), rotierter Prompt-Pfad.
* **Nichts über eine zweite Passade im Decode-Pfad**: der Dump ist der gebatchte Prompt-Pfad (Mode 4).
  Der Decode-Pfad hat dieselbe Blockgeometrie, ist aber nicht Teil dieser Messung.

## Rohdaten

`C:\Users\krone\git\Strata-data\bench\2026-11-13-ultraquant-scale\` — `q4passes-s8.json` (alle
Varianten, K/V getrennt) und `q4passes-s8.log` (der Lauf). Reproduzieren:

```sh
python tools/kv_q4_passes.py --dump <cells-cwe4k.bin> --stride 8
```
