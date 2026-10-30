# G2: ist die trunkierte KL die KL?

**Status: vorregistriert (2026-10-28), Referenzlauf gelaufen (2026-10-28), Arm-Läufe stehen aus.**
Build (HIP) und der Referenzlauf mit `STRATA_LOGPOS_TOPK=4096` sind fertig; der Nachtrag am Ende dieses
Dokuments (2026-10-30) korrigiert drei Punkte der Messung **vor** dem ersten Arm-Lauf. Katalogzeile:
`docs/paper/Ansatz_Improvements.md`, Arbeitsplan G2
(`docs/research/BEYOND_KL_QUANT_FIDELITY.md` §4). Vorläufer: G1
(`bench/results/2026-10-28-g1-features/README.md`).

Dieses Dokument legt **vor** dem Build und vor dem ersten Lauf fest, was gemessen wird, wie die Stichprobe
gezogen wird und wann das Ergebnis „top-256 ist nicht die KL" heißt. Es ist die einzige G-Stufe, die einen
Build braucht.

## Frage

Alle Zahlen dieser Arbeitslinie sind `KL(fp16‖Arm)` über die **top-256 der Referenz**, mit
Floor-Substitution für Token außerhalb. Trägt diese Stütze die Aussage? Genauer, in der Form, die
entscheidbar ist:

> **Ordnen zwei Arme auf der vollen Stütze anders als auf top-256?**

Nicht gefragt ist, ob der *Betrag* der KL stimmt. Ein konstanter Bias wäre unschädlich, solange er alle
Arme gleich trifft. Gefragt ist die **Rangfolge**, weil jede Empfehlung dieser Arbeitslinie eine
Rangfolge ist.

## Warum das ohne Build nicht geht (geprüft am Code, nicht behauptet)

- Der Dump schreibt je Position acht feste Spalten und dann `id:logprob`-Paare; die Zahl der Paare ist
  `STRATA_LOGPOS_TOPK`, **hart auf 256 gedeckelt** (statische Analyse,
  `src/core/verify.cpp:3242-3245`: `std::min(256, …)`). Der Deckel ist eine Policy, keine Puffergrenze.
- Die **volle Logit-Zeile liegt bereits auf dem Host**: `verify.cpp:3212-3219` kopiert `T × n_vocab`
  floats und indiziert sie Zeile für Zeile. Ein Vollvokabular-Dump braucht also **keine** zusätzliche
  GPU-Arbeit, nur das Schreiben eines größeren Ausschnitts derselben Zeile.
- Die Stütze des einen Arms ist nicht die des anderen. Zwei Arme, die je ihre eigenen top-N dumpen,
  liefern für `KL(p‖q)` **nicht** dasselbe `v`-Set; genau die Token, an denen sich die Arme
  unterscheiden, fehlen dann. Deshalb reicht es nicht, den Deckel zu erhöhen — der Arm muss auf einer
  **vorgegebenen** Token-Menge ausgeben.

## Die Änderung (opt-in, Default unberührt)

Zwei Umgebungsvariablen, beide ohne Wirkung, wenn sie nicht gesetzt sind:

| Variable | Wirkung |
|---|---|
| `STRATA_LOGPOS_TOPK` | Deckel von 256 auf 4096 erhöht (nur für den Referenzlauf nötig) |
| `STRATA_LOGPOS_IDS=<datei>` | je Position `pos id id id …`: es werden **genau diese** Token-Ids ausgegeben, statt der top-k |

Die acht festen Spalten bleiben unverändert, ebenso das `id:logprob`-Format (`%.6f`). Damit liest
`tools/quant_quality.py` die Datei ohne Änderung. Ohne `STRATA_LOGPOS_IDS` ist die Ausgabe
byte-identisch zu heute.

Betroffene Dateien: `src/core/verify.cpp` (CUDA/HIP) und `sycl/src/core/verify.cpp` (Kopie).
**Gebaut und geprüft wird hier nur HIP** (ROCm 7.1, `build-hip-port`); der SYCL-Text ist ungebaut,
und Metal hat mit `STRATA_METAL_LOGPOS` eine eigene Implementierung, die hier **nicht** angefasst wird
(G2 läuft auf HIP). Beides wird im Ergebnisbericht als „nicht gebaut" ausgewiesen, nicht als geprüft.

Nebeneffekt, der die Identitätsprüfung stärker macht als geplant: weil die Ids-Variante nur die
`id:logprob`-Paare ersetzt, schreibt ein Arm-Lauf die **acht festen Spalten für alle 32.746
Positionen** — vergleichbar mit der bestehenden TSV Zeile für Zeile.

## Stichprobe

**Eine feste Positionsmenge `S`, für jedes Segment dieselbe.** Der Dump schreibt `pos` **relativ zum
Segment** — jedes Segment beginnt wieder bei 0 (gemessen an der `int8`-TSV: Zeile 4.095 hat wieder
`pos 0`). Ein Ids-File ist deshalb nur eindeutig, wenn alle Segmente dieselben Positionen dumpen. Das
ist auch statistisch das Richtige: jede Segmentmitte steht dann auf denselben Positionen, und der
gepaarte Vergleich bleibt balanciert.

| Schicht | Zahl | Warum |
|---|---|---|
| **flach** | Vereinigung der 40 flachsten Positionen **je Segment** (≤ 320) | kleinste retained mass der Referenz; dort ist der Trunkierungsfehler am größten, dort entscheidet sich die Frage |
| **zufällig** | 640, global ohne Zurücklegen | der einzige unverzerrte Blick auf die Population |
| **Flip** | bis 64, `argmax(fp16) ≠ argmax(int8)` | die Ereignisse, an denen die bisherigen Metriken hingen |

Summe **1.024 Positionen**. Jedes der acht Segmente dumpt zusätzlich zu seinen ~4.095 Positionen
genau diese 1.024. Die Auswahl benutzt **nur die Referenz** (retained mass) und einen festen Arm
(`int8` für die Flips); sie ist damit für alle anderen Arme dieselbe Menge — Voraussetzung dafür,
dass ein gepaarter Vergleich überhaupt etwas heißt.

## Stütze und Tail-Schranke

Der Referenzlauf dumpt seine **top-4096**; diese Ids sind die Stütze für alle Arme. Je Position wird
berichtet:

- `retained = Σ exp(logp)` über die 4.096 Token (gemessen, nicht geschätzt),
- `tail_bound = (1 − retained) · max|log p − log q|` über die behaltenen Token — die gröbste, aber
  ehrliche Schranke für den Beitrag des weggelassenen Rests.

**Abbruchregel:** ist `tail_bound` auf irgendeiner Position > 1e-4 nats, wird N erhöht (4.096 → 16.384)
und die Stichprobe **vor** jeder Bewertung neu gezogen. Das ist eine Definitionskorrektur, kein
Ergebnis, und wird als solche berichtet.

## Arme (acht Läufe)

`fp16` (Referenz, dumpt top-4096) und sieben Arme mit Ids-Datei: `int8`, `q4lm32w`, `q4lm32`,
`q4lm8v`, `q4lm8k`, `q4lm8`, `s320r0`. Das sind genau die sechs Arme der S3-Leiter plus der
Ring-Arm — also alle fünf Leiterstufen und das zertifizierte Verlustpaar.

Gleiche Einstellungen wie S3: `cwe-4k`, teacher-forced, `--prefill 512`, Fenster nur bei den
`*w`-Armen, `env: {}` sonst.

## Auswertung

Auf **denselben 1.024 Positionen** werden zwei Zahlen je Arm gerechnet:

- `kl256` — die heutige Metrik (Floor-Substitution wie in `kl_between`),
- `klfull` — `Σ_v p_v (log p_v − log q_v)` über die 4.096 Token, ohne Substitution. Wo der Arm einem
  behaltenen Token keine Masse gibt, ist `log q_v` endlich und groß-negativ; es wird **nicht**
  ersetzt. Genau das ist der Unterschied, den G2 messen soll.

Dazu je Arm der **Trunkierungsbias** `kl256 − klfull` (Mittel über die Positionen, mit gepaartem
Intervall über die acht Segmente).

## Entscheidungsregel (vor dem Lauf festgelegt)

**Primär — die fünf Leiterstufen.** Für jedes benachbarte Paar der S3-Leiter
(`q4lm8 > q4lm8k ≈ q4lm8v > q4lm32 > q4lm32w ≈ int8`) wird auf der Stichprobe geprüft:

1. das **Vorzeichen** von `klfull(A) − klfull(B)` stimmt mit dem von `kl256(A) − kl256(B)` überein,
   **und**
2. die **Entscheidung** bleibt dieselbe: `>` wenn die gepaarte Differenz über der einseitigen Grenze
   liegt, `≈` wenn darunter — mit derselben Grenze wie in G1 (1,895 · sd/√8).

**Bestehen alle fünf Stufen beide Prüfungen, trägt die top-256-Stütze die Rangfolge.** Dann ist G2
bestanden und die bisherigen Aussagen sind nicht bloß relativ.

**Sekundär, nur berichtet, nicht entscheidend:**

- der Trunkierungsbias je Arm (Vorzeichen und Größe),
- der Vergleich der Stichprobenmittel gegen die Populationsmittel aus den bestehenden Läufen — die
  Zahl, die sagt, wie weit die geschichtete Stichprobe von der Population entfernt ist,
- `klfull` auf der Zufallsschicht allein (unverzerrt, aber mit der Auflösung von ~480 Positionen:
  grob 0,02 nats — sie kann große Effekte sehen und kleine nicht, und das steht so im Bericht).

**Kippt eine Stufe**, lautet das Ergebnis nach dem Kill des Arbeitsplans: *die trunkierte KL ist nicht
die KL, alle bisherigen Aussagen sind relativ.* Dann wird nicht „die volle KL“ zur neuen Metrik
erklärt (sie ist mit diesem Aufwand nicht für 20 Läufe zu haben), sondern jede Empfehlung bekommt
diesen Vorbehalt, und die Leiter wird auf der vollen Stütze neu aufgestellt, soweit die acht Arme
reichen.

## Identitätsprüfung des Builds (SHA-Gate)

Der Build darf die Rechnung nicht ändern. Geprüft wird in zwei Stufen:

1. **Je Arm, an allen Positionen:** die acht festen Spalten (`logp(target)` auf neun
   Nachkommastellen, `argmax`, `logp(argmax)`) müssen **zeichengleich** mit der bestehenden TSV sein —
   32.746 Zeilen je Arm, acht Arme.
2. **`int8` zusätzlich:** ein Lauf ohne `STRATA_LOGPOS_IDS` und ohne erhöhten Deckel muss die
   bestehende `int8`-TSV **byte-identisch** reproduzieren (volle Zeile, inklusive top-256-Paare).

Schlägt eine der beiden fehl, ist der Build nicht verwendbar und die Ursache zuerst zu klären.

## Kosten

Ein Build (eine Datei je Backend, HIP: Minuten) und acht Läufe über `cwe-4k` — dieselbe Größenordnung
wie der S3-Satz, also ein halber Tag GPU. Die Stichprobe spart **Rechenzeit nicht**: der Prefill läuft
ohnehin über alle Positionen je Segment. Sie spart Schreibmenge bei den Armen: 1.024 × 8 × 4.096 ×
~12 B ≈ 400 MB je Arm. Der Referenzlauf dumpt top-4096 für **alle** Positionen (≈ 1,6 GB), weil die
Ids erst aus ihm entstehen; danach wird er auf die Stichprobe reduziert und die Rohdatei gelöscht.

## Was diese Stufe nicht kann

- **Kein funktionales Ground Truth.** G2 sagt, ob die *Verteilung* richtig gemessen ist, nicht ob die
  Verteilung die *Aufgabe* vorhersagt. Das ist G3.
- **Keine Auflösung unter 0,0020** auf der vollen Population. Die Stichprobe ist klein; sie kann die
  Rangfolge prüfen, nicht die Beträge fein auflösen.
- **Nur ein Korpus, acht Segmente.** Die Rangfolge wird auf `cwe-4k` geprüft, nicht allgemein.
- **Nur HIP gebaut.** CUDA, SYCL und Metal bleiben ungeprüfte Textänderungen.

## Nachtrag vor den Läufen (2026-10-30): die Abbruchregel war unerreichbar

Alles oben ist vor dem Build geschrieben. Dieser Abschnitt ist **vor dem ersten Arm-Lauf** geschrieben
und ändert drei Dinge an der Messung, nicht an der Frage. Er ist eine Korrektur der Definition, kein
Ergebnis; die Frage, die Entscheidungsregel und die Stichprobe bleiben, wie sie vorregistriert sind.

### 1. `tail_bound > 1e-4` kann nicht die Abbruchregel sein (gemessen)

Die Regel oben verlangt, dass `tail_bound` auf **jeder** Position ≤ 1e-4 nats bleibt. Gemessen auf dem
bestehenden `int8`-Lauf (32.746 Positionen, Stütze top-256, also die *kleinere* der beiden Stützen):

| | `tail_bound` (nats) |
|---|---|
| Median | 0,0006 |
| p99 | 0,21 |
| Maximum | 10,73 |
| Positionen > 1e-4 | mehr als die Hälfte |

Die Schranke ist auf top-256 allein dadurch groß, dass ein flaches Modell seine Masse breit verteilt:
bei einer Zeile mit `retained = 0,297` ist `tail_bound ≥ 0,703 · max|Δ|`, also schon bei einem
Unterschied von 0,0002 nats zwischen zwei Armen über 1e-4. Eine Regel, die auf über der Hälfte der
Positionen auslöst, misst nichts. Sie hätte außerdem auf top-4096 erneut nicht gehalten: 53 von 8.192
Stichprobenzeilen halten dort weniger als 0,95 (Abschnitt „Die exakten Zeilen").

**Ersatz (vor dem Lauf festgelegt):** der **direkte Stabilitätstest der Leiterentscheidung** — genau
die Entscheidungsregel oben, auf `kl256` gegen `kl4096`. Das ist die Eigenschaft, die zählt („ordnen
zwei Arme auf der vollen Stütze anders als auf top-256?"), und sie ist ohne Schranke prüfbar.
`tail_bound` bleibt als **beschreibende** Zahl im Bericht, ohne Trigger.

### 2. Die Vereinigungsstütze ist verworfen (gemessen)

Naheliegend war eine Stütze je Position als Vereinigung der top-4096 **aller acht** Segmente, damit
jedes Segment dieselbe Ids-Datei liest. Gemessen an der Referenz: im Mittel **16.672** Ids je Position
(Median 17.036, Maximum 24.536) — rund **2,2 GB je Arm**, und der Zusatznutzen für ein gegebenes
Segment ist aus der Referenz **nicht messbar**: die zusätzlichen Ids liegen außerhalb der top-4096
*dieses* Segments, ihr `p_v` ist dort also unbekannt. Die acht Segmente einer Position sind
verschiedene Texte; ihre Köpfe decken die Schwänze der anderen schlecht.

### 3. Stattdessen: Stütze je Segment, ein Engine-Start je Segment

`STRATA_LOGPOS_IDS` wird **beim Start** gelesen, also braucht jede verschiedene Ids-Datei einen eigenen
Start. Acht Starts kosten rund zwei Minuten Ladezeit je Arm — der Prefill läuft ohnehin. Dafür bleibt
`tools/quant_quality.py` unverändert: ein Ein-Segment-Ids-JSON, `--env STRATA_LOGPOS_IDS=<datei>`,
danach zusammenführen (`tools/kv_g2_run.py`). Drei Stützen:

| Stütze | Definition | Wofür |
|---|---|---|
| **256** | Präfix der Referenz-top-4096 | `kl256`, die heutige Metrik |
| **4096** | die top-4096 **des eigenen Segments** aus der Referenz | `kl4096`, die volle Stütze dieser Stufe |
| **voll** | alle 248.320 Ids, an den Zeilen mit `retained(4096) < 0,95` | die exakte Probe, ohne jede Trunkierung |

Die Ids-Dateien entstehen aus der Referenz (`tools/kv_g2_support.py`, `support.json`): je Segment
1.024 Zeilen, 25–72 MB, zusammen 277 MB. Der reduzierte Referenzlauf auf der Stichprobe ist 561 MB.

### Die exakten Zeilen (53 von 8.192)

`retained(4096)` der Referenz auf der Stichprobe: Minimum 0,481, p01 0,979, p10 0,9991, Median 0,9998,
Mittel 0,9980; ≥ 0,999 in 90,8 %, ≥ 0,99 in 97,7 %. Unter 0,95 liegen **53 Zeilen** (47 Positionen):
s00 1, s01 3, s02 30, s03 1, s04 1, s05 11, s06 2, s07 4. Dort dumpen Arm **und** Referenz das ganze
Vokabular (eigener Referenzlauf `ref-exact`), und `kl4096` wird gegen `klfull` geprüft — die Probe
sitzt genau dort, wo die Trunkierung am meisten kostet.

### Die top-256-Prüfung ist eine Mengenprüfung, keine Zeichenprüfung

Der Nachtrag zum SHA-Gate: die ersten 256 Paare der Referenz sind **nicht** zeichengleich mit der
S3-`fp16`-TSV. Gemessen: 8.155 Zeilen identisch, **37** Zeilen mit denselben Paaren in anderer
Reihenfolge (exakte Gleichstände, die bei k=256 und k=4096 anders sortiert werden — die Folge der
Logprob-Werte ist identisch, die Zuordnung Id → Logprob auch), **0** Zeilen mit anderem Inhalt. Die
Prüfung vergleicht deshalb die Zuordnung Id → Logprob; die Reihenfolge sieht keine KL.

### Kosten (gemessen bzw. aus den Dateigrößen gerechnet)

| Posten | Menge |
|---|---|
| Ids-Dateien | 277 MB |
| reduzierte Referenz (Stichprobe) | 561 MB |
| je Arm: 1.024 Zeilen × 4.096 Paare + 53 Zeilen × 248.320 Paare | ≈ 0,77 GB |
| sieben Arme + `ref-exact` | ≈ 5,6 GB |
| GPU-Zeit | wie S3, plus ~2 min Start je Arm |

Damit bleibt der Satz bei **rund 6,5 GB** statt 18–29 GB für die Vereinigungsstütze, bei gleicher
Aussagekraft für die gestellte Frage.

Eine Falle, die der erste Arm-Lauf gezeigt hat (gemessen, abgebrochen und neu gestartet): der
Durchfall-Top-k muss **0** sein. Mit `--topk 4096` schreibt jede Position, die die Ids-Datei nicht nennt,
trotzdem ihre 4.096 Paare — 2,3 GB je Arm statt 0,77 GB, für Paare, die niemand liest. Die Ids-Datei
ersetzt den Top-k an den genannten Positionen unabhängig davon; der Wert gilt nur für die anderen.

