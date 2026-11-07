**Betreff: Re: Danke – L1/16/16 akzeptiert; P0a zeigt einen Tail, keine Mittelwert-Verschiebung**

M.,

die Antwort auf deine eine Frage ist eindeutig, und sie ist negativ: **Es gibt keine veröffentlichte p99-, letzte-1.024- oder Worst-Position-Zahl für `turboquant_4bit_nc`. Nur PPL.** Die vLLM-Studie vom 11. Mai 2026 testet MRCR und vier Reasoning-Benchmarks über vier Modelle (30B–200B+) und berichtet Durchsatz, Latenz und aggregierte Genauigkeit – keine Positionsverteilung. Die 0,8 %-Zahl, die ich dir genannt hatte, ist eine Mittelwert-/Perplexity-Größe. Sie sagt nichts über deine 18 Vektoren. Du wirst diese Zahl selbst produzieren müssen, und dein P0b ist der richtige Ort dafür.

Aber es gibt einen indirekten Befund, der genau gegen deinen Tail-Fall spricht – und den du in P0b direkt mitprüfen kannst.

**Der entscheidende Fund: TurboQuant kollabiert bei Qwen im Worst-Case.** Die Arbeit „Worst-Case Quality Collapse in KV Cache Quantization“ (Villines, Mai/Juni 2026) führt eine Per-Head-Minimum-Cosine-Sonde ein, die aus einem einzigen FP16-Forward-Pass den **minimalen** Cosinus über das gesamte Attention-Grid berichtet – nicht den Mittelwert. Das Ergebnis für TurboQuant: „Uniform TurboQuant collapses to **near-zero or negative minimum cosine on both Qwen models** while matching its published fidelity on Llama-3.1-8B and Mistral-7B-v0.3.“ Das ist kein PPL-Befund. Es ist ein Worst-Case-Befund auf genau deiner Modellfamilie. Und er sagt: Die mittlere Cosinus-Ähnlichkeit kann nominal sein, während einzelne (Layer, Head)-Einheiten nahezu zufälligen Output produzieren – ein Fehlermodus, den Mittelwerte und PPL systematisch verbergen. Deine 18 Positionen mit Delta > 0,1 nats sind exakt das, was diese Sonde aufdeckt, wenn man sie auf die Positionsebene herunterbricht.

**Was das für deine Frage bedeutet:** Wenn uniform TurboQuant bei Qwen im Worst-Case kollabiert, dann ist die Wahrscheinlichkeit hoch, dass die per-Token-Normkorrektur **genau die Vektoren nicht rettet, die deinen Tail tragen**. Die Normkorrektur skaliert die Länge des rekonstruierten Vektors korrekt, aber sie kann die Richtung nicht wiederherstellen, wenn die 4-Bit-Zentroide den Richtungsraum zu grob abdecken. Deine KVarN-Zerlegung wird das zeigen: Wenn der tangentiale Term an den Top-Gap-Positionen dominiert, ist die Normkorrektur machtlos – sie skaliert einen Vektor, dessen Richtung bereits falsch ist.

**Deshalb eine Ergänzung zu P0b, die ich für notwendig halte:** Deine radiale/tangentiale Zerlegung beantwortet die halbe Frage. Sie kann radial dominiert sein, während der Cosinus an den 18 Positionen bereits kollabiert ist. Miss **zusätzlich den minimalen Cosinus pro (Layer, Head) an genau den 18 Top-Gap-Positionen** – nicht nur den MSE. Die N+D-Dekomposition (8-Bit-Magnitude + Low-Bit-Richtung, Villines) zeigt, dass der minimale Cosinus die Statistik ist, die den Tail-Kollaps sichtbar macht, den MSE und PPL verbergen. Ohne diese Zahl kannst du nicht unterscheiden, ob die Normkorrektur einen Tail-Vektor rettet oder nur einen Mittelwert verschiebt.

**Zu deiner Entscheidungsregel:** r < 5 % tot, 5–20 % marginal, r ≥ 20 % überlebt – das ist vernünftig, aber nur, wenn du r **an den 18 Top-Gap-Positionen** rechnest, nicht gemittelt über die 6.144. Ein hohes mittleres r bei r < 5 % an den Top-Gap-Positionen ist kein Tail-Gewinn. Dein P0a zeigt: 54 % des gesamten Abstands kommen von fünf Positionen. Die Zerlegung muss dort genau sein, nicht im Durchschnitt.

**Der zweite Hebel, den du in derselben Capture mitprüfen kannst:** KVarN adressiert explizit „outlying token-scale errors“ – Vektoren, deren Magnitude oder Richtung so aus dem Rahmen fällt, dass der blockweite Skalar sie nicht erfasst. Das ist mechanistisch **anders** als die Normkorrektur: KVarN korrigiert die Skala des Tokens **vor** der Quantisierung (per-Token-Skala, gespeichert), nicht die Norm des rekonstruierten Vektors **nach** der Quantisierung. Dein P0b kann diesen Unterschied direkt sichtbar machen, wenn du die Zerlegung für beide Arme rechnest – die per-Token-Skalierung ist im selben Capture darstellbar, ohne Engine-Änderung.

**Zusammengefasst, was ich dir schicke:**
1. Die Zahl, die du suchst, existiert nicht – du produzierst sie. Kein PPL-Ersatz.
2. Der Qwen-spezifische Worst-Case-Kollaps ist das stärkste indirekte Argument dagegen, dass `norm_correction` ein Tail-Hebel ist. Er verschiebt die Erwartung, nicht die Entscheidung – die entscheidest du mit P0b.
3. P0b ist richtig gebaut. Ergänze den minimalen Cosinus an den 18 Positionen und, wenn billig, die KVarN-per-Token-Skalierung im selben Capture. Dann hast du in einem Lauf: radiale/tangentiale Zerlegung, Worst-Case-Cosinus und den mechanistisch alternativen Hebel.
4. Deine Schwellen (r < 5 % / 5–20 % / ≥ 20 %) sind gut – aber nur positionsspezifisch an den Top-Gap-Vektoren, nicht gemittelt.
5. P1 (16/16 ohne Fenster) bleibt die fehlende Kontrolle, aber erst nach P0b. P2 startet nur, wenn P0b überlebt. Kein 5,0-Budget, kein Default – einverstanden.

Schick mir die Zerlegung, sobald der Capture durch ist. Wenn der tangentiale Term an den 18 Positionen dominiert, ist die Normkorrektur tot, und wir sparen das 5,0-Budget. Wenn der radiale Term dominiert, bauen wir die minimale 18-Byte-Version und messen sie auf 4K. Beides ist ein Ergebnis.

Best,
Dave