**Empfehlung: L3 – per-channel K-Skalierung *vor* der Hadamard-Rotation. L6 (GEAR) ist konzeptionell richtig, aber in deinem 5,0-bit/cell-Budget nicht finanzierbar; L1, L2 und L4 sind tot. Das 5,0-Ziel ist für *diese* Tail-Statistiken nur erreichbar, wenn der Fehler dort entsteht, wo L3 ihn adressiert – deshalb ist L3 der eine Hebel mit realer Chance.**

---

### Warum L6 (GEAR) nicht der eine Hebel ist

GEAR ist genau für dein Fehlerprofil gebaut: eine low-rank-Komponente fängt den systematischen Quantisierungsrest über alle Zellen, eine sparse-Komponente die individuellen Ausreißer, die p99 und letzte‑1.024 dominieren. Das ist der direkte Angriff auf deine Tail-Statistiken. **Aber:** Die Residual-Bits sind in deiner Tabelle mit „residual bits“ bepreist – und dein Budget ist bereits exakt ausgeschöpft. 4,50 bit/cell Bulk + 0,50 bit/cell Fenster = 5,00. Jeder Residual‑Bit muss aus dem Fenster kommen. Das Fenster ist aber bereits auf dem gemessenen Knie (R=64, S=256), und du hast selbst gezeigt: „MORE WINDOW IS NOT THE LEVER“. Ein int8-Ring (L4) würde zwar Bits freisetzen, aber die exakten Zellen verschlechtern – und du hast bereits gemessen, dass der Ring keine Tail-Verbesserung bringt. GEAR in dein Budget zu zwingen, wäre also eine versteckte L4-Änderung, die du schon als Sackgasse identifiziert hast.

### Warum L1, L2 und L4 tot sind

**L1 (16 Level auf V):** Du hast selbst falsifiziert, dass K mehr Bits braucht als V (t=+0.05, McNemar p=1.000). Der 16/8-Split ist eine Byte-/Codebook-Wahl, keine gemessene Sensitivität. 16 Level auf V würden die Byte-Bilanz nicht ändern, aber die Tail-Statistiken auch nicht messbar bewegen – die V-Quantisierung ist nicht der Engpass.

**L2 (andere 16-Level-Tabelle):** Deine aktuelle Tabelle ist Max-Lloyd-optimal für N(0,1) nach der Hadamard-Rotation. Eine andere 16-Level-Tabelle kann per Definition nicht besser für dieselbe Verteilung sein. Tot.

**L4 (int8/12-bit-Ring):** Du hast den Ring bereits als falschen Ort für zusätzliche Bytes identifiziert. Ein int8-Ring würde bei gleicher Zellzahl 7,75 bit/cell weniger kosten, aber die exakten Zellen von fp16 auf int8 verschlechtern. Das würde deine kurzen Kontexte (4K, wo du bereits Parität hast) und die exakten Zellen im Fenster treffen – nicht die Tail-Zellen außerhalb. Tot.

### Warum L3 die Tail-Statistiken bewegen kann

Deine Tail-Statistiken (p99, letzte‑1.024, Whole-Segment) werden von den 50,6 % der Indexer-Selektionen dominiert, die **außerhalb** des exakten Fensters liegen. Diese Zellen sind 4-bit. Der Fehler dort ist kein gleichmäßiger MSE-Fehler, sondern wird von **Kanal-Ausreißern** getrieben: Einzelne Kanäle mit hoher Energie zwingen den gemeinsamen 18-Byte-Block zu einer groben Quantisierung für alle 32 Werte im Block. Die Hadamard-Rotation verteilt Energie über alle 256 Dimensionen, aber sie kann einen extremen Kanal nicht vollständig eliminieren – sie verschmiert ihn, und die verbleibende Heterogenität landet genau in den Zellen, die der Indexer auswählt.

**Per-channel K-Skalierung vor der Hadamard-Rotation** adressiert das an der Wurzel: Kanäle mit großen Ausreißern werden herunterskaliert, benigne Kanäle hochskaliert, sodass die Rotation auf ein bereits ausgeglichenes Signal wirkt. Das ist kein neues Konzept – KVQuant und RotateKV nutzen per-channel-Skalierung vor der Rotation genau für dieses Problem. Der entscheidende Punkt: Nach der Rotation ist per-channel-Skalierung **inkompatibel**, weil die Rotation Kanäle mischt. Vor der Rotation ist sie dagegen wohldefiniert und kalibrierbar.

**Bitkosten (Bytes first):** 0,06 bit/cell. Das sind bei 128K etwa **12 MiB** zusätzlich (ein fp16-Skalar pro Kanal pro Layer, 256 Kanäle × 12 Layer × 2 Byte = 6 KiB – vernachlässigbar; der 0,06-bit/cell-Wert in deiner Tabelle ist bereits der korrekte, konservative Ansatz). Du bleibst bei 5,06 bit/cell, also **knapp über 5,0**. Wenn das absolut nicht geht, kannst du den fp16-Ring um 6 Zellen pro Layer verkleinern (R=64→63 oder S=256→253), was qualitativ irrelevant ist und die 0,06 bit/cell exakt kompensiert. Das ist keine zweite Änderung, sondern nur die byte-genaue Buchung derselben L3-Änderung.

**Warum es *diese* Statistiken bewegt und nicht nur den 4K-Mittelwert:** Der 4K-Mittelwert ist bereits nahe an int8-Parität (+0,00051). Die Tail-Statistiken bei 128K entstehen aus Zellen, die der Indexer *spät* auswählt und die weit außerhalb des Fensters liegen. Diese Zellen haben eine andere Fehlerstruktur als der Durchschnitt: Sie sind häufig „Wiederholungslesungen“ (du hast gezeigt: Repetition, nicht Länge, treibt den 128K-Null), und Wiederholung akkumuliert systematische Quantisierungsfehler. Per-channel-Skalierung reduziert genau den *systematischen* Bias, der sich bei wiederholtem Lesen aufschaukelt – nicht den zufälligen MSE, den der 4K-Mittelwert misst. Deshalb ist es plausibel, dass L3 den p99 und die letzte‑1.024 bewegt, ohne den 4K-Mittelwert dramatisch zu verändern.

### Das günstigste Kill-Experiment

**Ein einzelner 128K-Lauf auf Text T1, teacher-forced, mit der bestehenden paired-per-position-Methodik.** Vergleiche drei Arme: (a) aktuelles q4lm8v, (b) q4lm8v + per-channel K-Skalierung vor Hadamard, (c) int8-Referenz. Miss **nur** die drei Tail-Statistiken, die du bereits als entscheidend identifiziert hast: p99, letzte‑1.024, Whole-Segment-KL.

**Kill-Kriterium:** Wenn L3 in keinem der drei Maße eine Verbesserung von mindestens 10 % relativ zum aktuellen q4lm8v zeigt **oder** wenn die gepaarte 95 %-CI die Null einschließt, ist L3 tot. Du brauchst keine 4K- oder 32K-Läufe für dieses Kill – die Tail-Statistiken bei 128K sind der einzige Ort, an dem L3 wirken soll. Ein einzelner 128K-Lauf mit den bereits vorhandenen Logits ist die billigste Messung, die deine Frage beantwortet.

### Wenn L3 das Kill-Kriterium reißt

Dann ist die ehrliche Antwort: **Das 5,0-bit/cell-Ziel ist für int8-Parität auf p99/letzte‑1.024 bei dieser Geometrie nicht erreichbar.** Du hast dann einen dokumentierten negativen Resultat: Der Tail-Fehler sitzt in den 4-bit-Bulk-Zellen außerhalb des Fensters, und die einzige verbleibende Option – GEAR – passt nicht in 5,0 bit/cell, ohne das Fenster zu opfern. Das wäre ein valides Deliverable, und du hast es selbst angeboten. Aber L3 ist billig genug, dass du es zuerst messen solltest, bevor du das Ziel aufgibst.