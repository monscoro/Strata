Hier ist der Survey zu L3 (per-channel K-Skalierung vor der Hadamard-Rotation). Die Literatur stützt den Mechanismus, zeigt aber auch die genaue Lücke, in die dein Vorschlag stößt – und liefert ein wichtiges Gegenargument, das du in deinem Kill-Experiment adressieren musst.

---

## 1. Per-channel Key-Quantisierung: Warum K anders behandelt werden muss als V

**KVQuant (NeurIPS 2024)** ist der zentrale Referenzpunkt. Die Arbeit zeigt, dass für Keys per-channel-Quantisierung (entlang der Kanal-Dimension) besser ist als per-token, weil „different channels and tokens have different average magnitudes“ – ein Element, das in einem Kanal ein Ausreißer ist, ist in einem anderen Kanal keiner. KVQuant behandelt Keys **Pre-RoPE** und per-channel, Values dagegen per-token. Der entscheidende Befund für deine Frage: Per-channel Key-Quantisierung ist keine Option, sondern eine Notwendigkeit, wenn man niedrige Bitbreiten anstrebt. KVQuant erreicht 3-bit mit sub-0.1 Perplexity-Degradation nur durch diese Behandlung.

**KVarN und die Qwen-spezifische Evidenz** bestätigen das für deine Modellfamilie. Qwen-Architekturen haben „extreme channel outlier distributions that cause catastrophic quality loss under naive int4 KV quantization“ – Qwen2.5-7B verliert **+638 PPL** bei int4 ohne Rotation. KVarN quantisiert K per-channel und V per-token, mit Hadamard-Rotation als Vorverarbeitung. Das ist exakt die Kombination, die dein L3 vorschlägt – nur dass KVarN die per-channel-Skalierung offenbar **nach** der Rotation oder als Teil der Rotation anwendet, nicht als separaten Schritt davor. Das ist die Lücke.

---

## 2. Hadamard-Rotation: Was sie leistet und wo sie ansetzt

**QuaRot / Ashkboos et al. (NeurIPS 2024)** und **SpinQuant** etablieren die rotate-then-quantize-Logik: Eine orthogonale Transformation \(H\) verteilt Energie über Dimensionen, reduziert channel-wise Varianz und glättet Ausreißer – „the main source of degradation under low-bit quantization“. Wichtig für deine Argumentation: Die Rotation wirkt **nach** der Datenerhebung. Wenn die Heterogenität bereits im Rohsignal extrem ist, muss die Rotation sie „einsammeln“ und über alle 256 Dimensionen verschmieren. Das funktioniert, aber es erhöht den gemeinsamen Skalierungsfaktor für den gesamten Block – genau das, was KVLinC als Problem identifiziert (siehe Abschnitt 4).

**OSCAR (arXiv 2605.17757)** liefert die mathematische Präzisierung: Hadamard mischt Outlier-Energie über Kanäle, sodass der per-group min–max-Quantizer „comparable per-channel magnitudes“ sieht. Das ist die Standard-Outlier-Suppression. OSCAR zeigt aber auch die Grenze: Nach der Rotation sind die Kanäle **gleichmäßig** (Lemma 1: Diagonale wird \(1/d \cdot \mathrm{tr}(\Lambda)\)). Das bedeutet: Per-channel-Skalierung **nach** der Rotation ist sinnlos, weil alle Kanäle bereits dieselbe Importance haben. Der einzige Ort, an dem per-channel-Skalierung noch etwas bewegen kann, ist **vor** der Rotation. Das ist das strukturelle Argument für L3.

**RotateKV (IJCAI 2025)** ist der nächste Verwandte deines Vorschlags. RotateKV führt „Outlier-Aware Rotation“ ein, die channel-reordering nutzt, um die Rotation an variierende channel-wise Outlier-Verteilungen anzupassen – **ohne** die FWHT-Effizienz zu opfern. Der Unterschied zu L3: RotateKV **permutiert** Kanäle vor der Rotation, um Ausreißer in bestimmte Positionen zu bringen, skaliert sie aber nicht. Dein L3 skaliert jeden Kanal auf eine gemeinsame Zielvarianz. Das ist invasiver, aber potenziell wirksamer, weil es die Outlier-Energie tatsächlich reduziert, statt sie nur umzuordnen. RotateKV erreicht damit <0.3 PPL-Degradation bei 2-bit auf LLaMA-2-13B – ein starkes Existenzbeweis, dass channel-wise Vorbehandlung vor Hadamard bei extrem niedrigen Bitbreiten funktioniert.

---

## 3. Der direkte Beleg für „Skalierung vor Rotation“ – und die Lücke

**The Token-Wise Imperative (arXiv 2604.19157)** formuliert das Problem präzise: „high-magnitude outliers in specific channels artificially expand the quantization grid, leading to massive quantization errors across the rest of the vector“ – und die Lösung ist eine orthonormale Transformation \(H\). Aber die Arbeit entscheidet sich dann für **block-diagonal rotation** statt per-channel-Skalierung, weil sie token-wise Quantisierung als systemfreundlicher bevorzugt. Sie räumt jedoch ein: „larger \(h\) improves cross-channel mixing and robustness“ – also mehr Rotation hilft, aber sie kostet. L3 ist der Versuch, die Robustheit **ohne** größeres \(h\) zu bekommen, indem man die Heterogenität vor der Rotation reduziert.

**Die eigentliche Forschungslücke:** Keine der gefundenen Arbeiten untersucht explizit eine **kalibrierte, statische per-channel-Skalierung von K vor der Hadamard-Rotation** als separaten Schritt, der die Rotation auf ein bereits homogenisiertes Signal wirken lässt. KVQuant macht per-channel-Quantisierung *statt* Rotation. RotateKV macht channel-reordering *innerhalb* der Rotationsvorbereitung. KVarN macht Hadamard + Varianz-Normalisierung, aber die genaue Reihenfolge und der Beitrag der per-channel-Skalierung vor der Rotation werden nicht isoliert. **Das ist die Nische, in der dein L3 sitzt** – und genau deshalb ist das Kill-Experiment so billig: Du testest eine Kombination, die in der Literatur als naheliegend gilt, aber nicht isoliert gemessen wurde.

---

## 4. Das Gegenargument, das du kennen musst: KVLinC

**KVLinC (arXiv 2510.05373)** ist der wichtigste Kontrapunkt. Die Autoren beobachten: „Quantizing raw keys channel-wise together with Hadamard rotated values token-wise emerges as the optimal configuration... the application of Hadamard rotation to keys redistributes each outlier dimension, thereby increasing the scaling factor of quantization leading to higher error“. Mit anderen Worten: **Rotation auf Keys kann schädlich sein**, wenn man nicht vorher channel-wise vorskaliert. Die Rotation verteilt den Ausreißer über alle Kanäle und **erhöht dadurch den gemeinsamen Skalierungsfaktor** für den 18-Byte-Block – der Quantisierungsfehler steigt für alle 32 Werte im Block, nicht nur für den Ausreißer.

Das ist kein Argument gegen L3 – im Gegenteil: Es ist das stärkste Argument **für** L3. KVLinC zeigt, dass Rotation ohne Vorsk alierung kontraproduktiv sein kann. Dein aktuelles q4lm8v rotiert **ohne** Vorsk alierung. L3 würde genau das beheben: Erst die Kanäle ausgleichen, dann rotieren, dann quantisieren. Der Skalierungsfaktor im 18-Byte-Block würde sinken, weil die Rotation nicht mehr einen riesigen Ausreißer über 256 Dimensionen verschmieren muss. KVLinC wählt stattdessen den Ausweg, Keys **gar nicht** zu rotieren und stattdessen channel-wise zu quantisieren. Das ist eine andere Design-Entscheidung, die mit deinem bestehenden 18-Byte-Block-Format (das Rotation erfordert, um die Nibbles sinnvoll zu füllen) nicht kompatibel ist. L3 ist der Weg, die Rotation **zu behalten** und trotzdem den KVLinC-Effekt zu vermeiden.

---

## 5. Einordnung für dein Kill-Experiment

Die Literatur stützt drei Vorhersagen, die dein Experiment testen kann:

1. **Wenn L3 wirkt, dann muss es den p99 und die letzte-1.024 bewegen, nicht den 4K-Mittelwert.** KVQuant und KVarN zeigen, dass per-channel-Behandlung von K-Outliers den **Tail** der Fehlerverteilung adressiert, nicht den Durchschnitt. Der 4K-Mittelwert ist bei dir bereits nahe int8 – dort gibt es wenig zu gewinnen. Die Tail-Zellen außerhalb des Fensters sind die, deren Skalierungsfaktor durch unbehandelte K-Outliers aufgebläht wird.

2. **Wenn L3 das Kill-Kriterium reißt, dann liegt es wahrscheinlich an der statischen Kalibrierung.** KVQuant warnt: „In per-channel quantization, it is challenging to update scaling factors online since the scaling factors corresponding to each incoming channel would potentially need to be updated whenever a new token is added“. Deine Skalierung wäre **statisch** (offline kalibriert), was das Problem vermeidet – aber wenn die K-Verteilung über 128K driften sollte, könnte eine statische Skalierung nicht ausreichen. Dein Kill-Experiment sollte daher nicht nur „L3 vs. Baseline“ messen, sondern auch prüfen, ob die kalibrierten Skalen über die 128K-Positionen stabil bleiben. Wenn sie driften, ist L3 in dieser Form tot – aber dann weißt du, dass du eine **dynamische** per-channel-Skalierung brauchst, was ein anderer (teurerer) Hebel wäre.

3. **Der billigste Test ist ein einziger 128K-Lauf, weil die Tail-Statistiken dort entstehen, wo die Outliers wirken.** Du brauchst keine 4K- oder 32K-Läufe für das Kill. Die 128K-Tail-Zellen sind die einzigen, die von den 50,6 % Indexer-Selektionen außerhalb des Fensters dominiert werden – und genau dort sitzen die K-Outlier-Kanäle, die L3 adressiert.

---

**Fazit des Surveys:** L3 ist literaturgestützt, aber nicht literaturbestätigt. KVQuant liefert die Notwendigkeit per-channel K-Behandlung, RotateKV den Existenzbeweis für channel-wise Vorbehandlung vor Hadamard, OSCAR das mathematische Argument, dass Skalierung *nach* Rotation nutzlos ist, und KVLinC das Gegenargument, das L3 gerade deshalb plausibel macht. Die Kombination „statische per-channel-Skalierung vor Hadamard, dann bestehender 18-Byte-Block“ ist in dieser exakten Form nicht publiziert – das ist deine Lücke, und dein Kill-Experiment ist der günstigste Weg, sie zu schließen oder zu dokumentieren, dass sie geschlossen ist.