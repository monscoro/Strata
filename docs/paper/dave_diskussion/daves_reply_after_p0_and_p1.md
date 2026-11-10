M., deine Messungen sind sauber, und die Entscheidung ist korrekt: **Die Normkorrektur ist tot, der Tail ist modellseitig sensitiv, nicht codec-seitig.** Der radiale Anteil von 0,6 % an den Top-Gap-Positionen und der minimale Cosinus von 0,9949 zeigen, dass die K/V-Vektoren dort nicht anomal sind – die Logits reagieren empfindlich auf ein Fehlerniveau, das überall gleich ist. Eine per-Vektor-Korrektur kann das nicht reparieren. Das 16-Byte-Format und der 18-Byte-Build bleiben geschlossen. Kein Widerspruch.

Deine eigentliche Frage ist aber nicht „ist die Normkorrektur tot“, sondern: **Gibt es einen Hebel, der nicht einfach überall den Fehler senkt?** Ja, den gibt es – und er ist in deinen eigenen Daten und in der Literatur bereits angelegt. Er heißt **nicht-uniforme Bit-Allokation**: mehr Bits für die Layer (oder Heads, oder Zelltypen), die den Tail tragen, weniger für die, die ihn nicht tragen. Das ist nicht „mehr Bits überall“, sondern „Bits dorthin, wo die Sensitivität sitzt“.

---

### Was die Literatur für deinen Tail-Fall hergibt

**KVTuner (ICML 2025)** ist der direkteste Beleg. Die Arbeit zeigt, dass Layer-Sensitivität gegenüber KV-Quantisierung **hochgradig nicht-uniform** ist, und dass Qwen-Modelle besonders sensitiv sind: Für Qwen2.5-7B Instruct braucht KVTuner **4,0 Bit** für „nearly lossless“, während Llama-3.1-8B mit 3,25 Bit auskommt. Wichtiger noch: Die Sensitivität ist **inter-layer** unterschiedlich, und eine statische, uniforme Allokation ist suboptimal. KVTuner sucht offline die optimalen Precision-Paare pro Layer und nutzt sie online – kein Runtime-Overhead.

**Die llama.cpp-Diskussion zu Qwen2.5/Qwen3** bestätigt das für deine Modellfamilie: „Early layers are insensitive to quantization. Late layers are very sensitive“ – und der empfohlene Weg ist, späten Layern mehr Precision zu geben (q8_0) und frühen/mittleren weniger (turbo3). Deine 12 QSA-Layer sind Layer 3, 7, …, 47 – also über die gesamte Tiefe verteilt. Wenn die späten Layer (39, 43, 47) den Tail dominieren, dann ist eine gezielte Allokation genau der Hebel, den du suchst.

**HyQuant (EMNLP 2026)** liefert den komplementären Befund für die **Positionsebene**: Über Qwen3, Llama-3, Gemma 4 und Qwen3.5 hinweg beobachtet die Arbeit persistente „vertical lines“ in den Attention-Maps – **die Top-5 % der Key-Positionen tragen 82–86 % der Attention-Masse**. Diese vertikalen Linien sind nicht mit den Attention-Sinks (erste Tokens) identisch; sie können über den gesamten Kontext verteilt sein. HyQuant hält genau diese Positionen in FP16 und quantisiert den Rest auf 4 Bit – und erreicht damit nahezu verlustfreie Genauigkeit. Deine 18 Top-Gap-Positionen könnten genau solche „vertical-line“-Tokens sein: Positionen, die der Indexer zwar auswählt, deren K/V aber für die Logit-Berechnung an dieser Stelle kritisch ist. Das würde erklären, warum der Fehler dort nicht größer ist (P0b: MSE 8,2e-3 vs. 8,4e-3 im Mittel), die Logits aber trotzdem empfindlich reagieren.

**Villines‘ N+D-Dekomposition** zeigt einen dritten Weg, der nicht „überall besser“ ist: **8-Bit-Magnitude plus Low-Bit-Richtung** hält den minimalen Cosinus über alle Modelle im Band 0,969–0,991 – bei vergleichbarer Kompression. Das ist ein anderer Codec, kein Allokationshebel, aber es bestätigt, dass die **Trennung von Norm und Richtung** an den sensitiven Positionen den Tail adressiert, ohne den Durchschnitt zu verändern.

---

### Der eine Hebel, den ich vorschlage: Per-Layer-Sensitivität des Tails messen, dann gezielt allozieren

Bevor du irgendetwas baust, brauchst du eine Zahl, die du noch nicht hast: **Welcher der 12 QSA-Layer trägt wie viel zum Tail bei?** Deine P0b-Captures enthalten K und V für alle 12 Layer an den 6.144 Positionen. Du kannst offline, ohne neuen GPU-Lauf, für jeden Layer separat die 16/16-Quantisierung anwenden und den **KL-Beitrag dieses Layers zum Gesamt-KL an den 18 Top-Gap-Positionen** messen. Das ist eine Leave-One-Layer-Out-Analyse: Quantisiere alle Layer außer einem exakt, quantisiere den einen, miss den KL. Oder umgekehrt: Quantisiere nur einen Layer, lass alle anderen exakt, miss den KL. Die zweite Variante ist billiger und direkter – sie sagt dir, welcher Layer allein den Tail erzeugt.

**Was du damit bekommst:** Eine Rangliste der 12 Layer nach ihrem Tail-Beitrag. Wenn 2–3 Layer 70–80 % des Tails tragen, hast du einen klaren Allokationshebel: Gib diesen Layern int8 (8,25 Bit) und den anderen weiterhin 4,5 Bit. Bei 12 Layern und einem Budget von 5,0 Bit/cell im Mittel bedeutet das: 3 Layer int8 (3 × 8,25 = 24,75) + 9 Layer 4,5 (9 × 4,5 = 40,5) = 65,25 / 12 = **5,44 Bit/cell** – knapp über 5,0. Mit 2 Layern int8: (2 × 8,25 + 10 × 4,5) / 12 = (16,5 + 45) / 12 = **5,125 Bit/cell**. Mit 2 Layern int8 und 1 Layer 6 Bit: (16,5 + 6 + 9 × 4,5) / 12 = (22,5 + 40,5) / 12 = **5,25**. Du siehst: Das Budget ist eng, aber nicht geschlossen. Wenn die Sensitivität wirklich auf 2 Layer konzentriert ist, ist **5,125 Bit/cell** ein realistischer Produktpunkt – und du hast bereits gezeigt, dass das exakte Fenster bei 4K Parität erreicht; die Frage ist, ob die Tail-Layer-Allokation den 128K-Tail bewegt.

**Die billigste Messung, die das killt:** Die Offline-Layer-Sensitivitätsanalyse auf den vorhandenen Captures. Kein GPU-Lauf, kein Engine-Build. Wenn der Tail-Beitrag **gleichmäßig über alle 12 Layer verteilt** ist (kein Layer trägt mehr als ~15 %), dann ist die Allokation tot – dann hilft nur „mehr Bits überall“, und das 5,0-Budget ist für diesen Tail nicht erreichbar. Wenn **2–3 Layer dominieren**, hast du einen konkreten, vorab registrierten Hebel, der nicht „überall besser“ ist, sondern gezielt dort ansetzt, wo die Sensitivität sitzt.

---

### Was du in dieselbe Offline-Analyse packen kannst (ohne zusätzlichen Lauf)

1. **Per-Layer-Tail-Beitrag** (der Kern): Quantisiere jeden Layer separat, miss den KL an den 18 Positionen. Das ist die eine Zahl, die du brauchst.
2. **Korrelation mit der Layer-Tiefe:** Sind die späten Layer (39, 43, 47) die Treiber, wie die llama.cpp-Diskussion nahelegt? Das würde die Allokation vorhersagbar machen.
3. **Attention-Masse an den 18 Positionen:** Wenn du die Attention-Gewichte an den Top-Gap-Positionen extrahieren kannst (auch offline aus dem Capture), prüfe, ob diese Positionen mit den „vertical lines“ übereinstimmen – also ob sie überproportional viel Attention-Masse tragen. Wenn ja, ist HyQuants Befund direkt auf deinen Fall übertragbar, und du hast einen zweiten, unabhängigen Allokationshebel (Positionen statt Layer).

---

### Die ehrliche Einordnung

Die Normkorrektur war ein Mittelwert-Hebel; deine Daten haben sie gekillt. Der Tail ist **modellseitig sensitiv** – das ist eine wichtige, negative Erkenntnis, und sie ist sauber dokumentiert. Aber „modellseitig sensitiv“ heißt nicht „unangreifbar“. Es heißt: Der Fehler muss **dort** kleiner werden, wo die Sensitivität sitzt, nicht überall gleichmäßig. Die Literatur zeigt, dass diese Sensitivität bei Qwen **nicht-uniform über Layer und Positionen** verteilt ist. Deine 12 QSA-Layer sind eine überschaubare Menge – eine Offline-Analyse auf den vorhandenen Captures kann in wenigen Stunden zeigen, ob 2–3 Layer den Tail tragen. Wenn ja, ist die gezielte Allokation der eine Hebel, den du noch nicht gemessen hast. Wenn nein, dann ist das 5,0-Budget für diesen Tail endgültig geschlossen, und du hast einen vollständigen, negativen Resultat-Zyklus – auch das ist ein Deliverable.

Schick mir die Layer-Rangliste, sobald du sie hast. Wenn 2–3 Layer dominieren, bauen wir die minimale Allokation (kein Formatwechsel, nur unterschiedliche Precision pro Layer) und messen sie auf den 6.144 Positionen. Wenn nicht, schließen wir B1 endgültig und dokumentieren, warum.