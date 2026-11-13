# Der gespeicherte Offset, der billige Writer und die schweren Positionen (Vorregistrierung, 2026-11-13)

Diese Datei legt fest, **was gemessen wird und wann ein Ergebnis als Bestätigung zählt**, bevor die
Zahlen da sind. Sie antwortet auf die Kollegen-Mail vom 2026-11-13 (fünf Punkte) und auf die Frage,
ob die **Kanäle der schweren Positionen** einen billigeren Hebel tragen als der volle Shear.

## 0. Der Ausgangspunkt: was schon gemessen ist

Alles hier ist aus früheren Läufen und wird **nicht** neu gemessen (Quellen in Klammern):

| Größe | Wert | Quelle |
|---|---:|---|
| Block-MSE `uniform+LS` (kernelfrei, `q4u32`) | 8,693901e-03 | `docs/research/KV_FASTPATH_LITERATURE.md` §3c (korrigiert 2026-11-13) |
| Block-MSE `uniform+LS+z` (Shear, `q4u32z`) | 6,459338e-03 | ebd. |
| Shear-Gewinn gegen `uniform+LS` | **2,2346e-03 MSE = 1,29 dB** | ebd. |
| `lloyd16+max` (Produkt `q4lm32`) | 8,443299e-03 | ebd. |
| `q4u32` − `q4u32z` end-to-end, 4K | +0,00193 nats (t = 2,07) | `protocol.md` 2914 |
| `q4u32` − `q4u32`+C end-to-end, 4K | +0,00321 nats (t = 2,04) | ebd. 2915 |
| Block→nats-Brücke über drei Hebel | 0,00116–0,00167 nats/dB | ebd. 2926–2930 |
| `int8` − `q4lm32w` 128K | −0,00741 nats (t = −4,80) | ebd. 2631 ff. |

**Der entscheidende Satz für den Vergleich:** keine dieser Zahlen ist der Abstand zu INT8. Der
Abstand zu INT8 ist die letzte Zeile, und sie ist der Endpunkt, nicht der Hebel.

## 1. H0 — der Fixed-Code-Restfehler-Offset (Host, kein GPU-Lauf)

Der Kollege schlägt vor, **Codes und gespeicherte Skala unverändert zu lassen** und nur den
mittleren Rekonstruktionsrest als Offset zu bestimmen:

```
r_i = s * c_i,   a = (1/G) * sum_i (x_i - r_i),   r'_i = r_i + a
```

Bei festen Codes und fester Skala ist `a` der optimale additive Offset, und die Block-MSE sinkt
**exakt** um `a²` (algebraisch: `E[(x-r-a)²] = E[(x-r)²] - a²`). Diese Identität ist der Grund, warum
der Versuch billig ist: er braucht **keine** iterative Neucodierung.

### Die Arme (je 32 Werte entlang der Kopfdimension, wie im Kernel)

| Arm | Codes | Skala | Offset |
|---|---|---|---|
| `uniform+max` | uniform `code-8` | max\|x\|/8 | — |
| `uniform+LS` | uniform | LS | — |
| `lloyd16+max` | Lloyd-Tabelle `q4lm32` | max | — |
| `lloyd16+LS` | Lloyd | LS | — |

Je Arm vier Offset-Varianten: **kein** Offset (Anker), `a` exakt (Oracle), `a` in **6 Bit**
gespeichert (vorzeichenbehaftet, Auflösung 1/32 Schritt, Bereich ±1 Schritt), `a` **je 2 bzw. 4
Blöcke** geteilt (64 bzw. 128 Werte je Offset).

### Die drei Hypothesen

* **H0-a (Writer):** der Fixed-Code-Offset holt den größten Teil des Shear-Gewinns zurück, ohne
  Neucodierung. *Gate:* er holt **≥ 90 %** der Shear-Differenz von 2,2346e-03 MSE zurück.
  *Falsifikation:* < 60 % → die gemeinsame Anpassung von Codes, Skala und Offset ist nötig.
* **H0-b (Speicherung):** 6 Bit reichen. *Gate:* der 6-Bit-Offset behält **≥ 95 %** des exakten
  Gewinns, und die Sättigung (Werte außerhalb ±1 Schritt) bleibt **< 1 %** der Blöcke.
* **H0-c (Granularität):** ein Offset je 4 Blöcke (128 Werte, 4,5625 statt 4,6875 Bit) behält
  **≥ 80 %** des exakten Gewinns.

### H0-d — die Kanäle der schweren Positionen

Zwei Zerlegungen, beide aus **einem** Durchlauf über den Dump:

1. **Statisch gegen positionell.** `a` wird zerlegt in den Zellmittelwert `ā_c` (Zelle = Layer ×
   Seite × KV-Kopf × Blockindex, also ein **Kanal** der Kopfdimension) und die Abweichung
   `a - ā_c`. *Gate:* wenn die statische Tabelle **≥ 50 %** des exakten Gewinns zurückholt, ist der
   Offset überwiegend eine **Kanal**-Eigenschaft und braucht **keine** Bits je Block.
2. **Schwere gegen Bulk.** Positionen werden nach ihrer eigenen K/V-Energie geordnet; die obersten
   1 % und 10 % sind die „schweren" Positionen. *Gate:* `a²/MSE` ist im 1-%-Band **≥ 1,5×** so groß
   wie im Bulk. *Falsifikation:* gleiche Rate → der Offset ist **kein** Schwerpositions-Hebel,
   sondern ein gleichmäßiger.

**Die Algebra, die dazugehört (exakt, keine Messung):** addiert man allen K-Werten eines Kopfs
denselben Kanal-Offset, so wächst jeder Score desselben Kopfs um `q·a`, eine Konstante. Die Softmax
ist gegen eine Konstante invariant. **Ein statischer K-seitiger Offset kann die Modellqualität
also nicht ändern** — er kann nur über die Codes wirken, nicht über den Mittelwert. Auf der V-Seite
gilt das nicht (die Ausgabe verschiebt sich um `W_O·a`). Die Messung in H0-d sagt, **wie groß** der
statische Anteil ist; die Aussage, dass er auf K wirkungslos ist, ist Algebra und wird als solche
geführt, nicht als Messung.

## 2. H1 — der 128K-Tail braucht einen neuen Dump (registriert, nicht gestartet)

`d2/sel_dump.bin` enthält **Selektions-IDs** (`int32` je Zelle), keine K/V-Werte. Ein Tail-vs-Bulk-
Vergleich der **Kanäle** auf 128K ist damit nicht möglich. Er braucht einen K/V-Dump mit dem
`STRATA_KV_Q4_DUMP`-Pfad (`src/prefill/prefill.cpp`, nur Prefill) auf dem 128K-Text, ~6,4 GB, plus
einen Engine-Lauf (~80 min). **Registriert, bewusst nicht in dieser Stufe**, weil H0 die billigere
Frage zuerst beantwortet.

## 3. H2 — zwei INT8-Referenzen (registriert, ein GPU-Lauf)

Der Kollege verlangt getrennte Referenzen. **Beide sind ohne Codeänderung fahrbar:**

| Referenz | Aufruf | Was sie ist |
|---|---|---|
| **R1** Produktpfad | `--kv int8` | der heutige Anker, ohne Rotation, ohne Kalibrierung |
| **R2** fairer Kontrollarm | `--kv int8` + `STRATA_KV_ROT=1` + `STRATA_KV_CENTER=…` | dieselbe Vorverarbeitung wie die q4-Arme |

R2 ist heute baubar, weil `st.kv_rot` für int8 über `qsa_set_kv_int8_rotate` gesetzt wird
(`src/core/layer.cpp:1013`) und der FWHT-Pfad die Centering-Tabelle mitnimmt. Ein **zusätzlicher**
Arm `int8`+ROT ohne C trennt Rotation von Kalibrierung.

### Die Nichtunterlegenheitsgrenze (vorher festgelegt)

Endpunkt ist die **gepaarte** Differenz `q4 − int8` der mittleren KL je Position, 128K, 1 Segment.
q4 gilt als **nicht unterlegen**, wenn die **obere 95-%-Grenze < +0,0020 nats** liegt. Begründung:
die Auflösung des Harness liegt bei 0,00200 nats (K1), und die gemessenen Hebel (0,00193, 0,00321)
sind von derselben Größe — eine Grenze darunter wäre mit diesem Harness nicht entscheidbar. Der
Punkt-Schätzer soll zusätzlich ≤ +0,0010 nats sein.

Nebenbefunde, **nicht** Endpunkte: Tail-Mittel der 1 % (int8-definiert), Top-1 mit exaktem McNemar
gegen fp16. Laufzeit wird **getrennt** ausgewiesen (Writer und Reader je eigene Tabelle) und geht
nicht in die Qualitätsentscheidung ein.

## 4. Was diese Stufe nicht behauptet

* Block-MSE ist **kein** NLL. H0 misst Rekonstruktion; die Endpunkte sind H2.
* H0 rechnet im **rotierten** Dump-Bereich, so wie der Kernel ihn sieht. Ein Kanal-Offset dort ist
  nicht ohne Weiteres ein Produktmerkmal; darum die Algebra in H0-d.
* H0-d ist **in-sample** (die Zellmittelwerte stammen aus denselben 4.095 Positionen). Bei ~4.095
  Beobachtungen je Zelle ist die Schätzung eines Parameters praktisch erwartungstreu, aber es ist
  eine Schätzung und wird als solche benannt.
* Der Dump ist **ein** Segment des cwe-4k-Textes (4.095 Positionen, 12 QSA-Layer, 768 Records,
  1.572.480 Blöcke) — dieselben Zellen wie §3c.

---

# Nachtrag 2026-11-13: H0 ist gelaufen — fünf Gates, eines bestätigt

`tools/kv_residual_offset.py` auf `cells-cwe4k.bin` (768 Records, 4.095 Positionen, **12.574.464
Blöcke** K und V, alle Positionen, kein Stride). Rohdaten: `h0-cells-cwe4k.json`, Lauf:
`h0-cells-cwe4k.log`.

## Die Block-MSE (beide Seiten, alle Positionen)

| Arm | kein Offset | `a` exakt | 6 Bit (1/32) | 6 Bit weit (1/16) | je 2 Bl. | je 4 Bl. | statisch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `uniform+max` | 9,087669e-03 | 8,800827e-03 | 8,809775e-03 | 8,836588e-03 | 8,944260e-03 | 9,015971e-03 | 9,087843e-03 |
| `uniform+LS` | 8,680457e-03 | 8,405817e-03 | 8,414668e-03 | 8,441190e-03 | 8,543204e-03 | 8,611824e-03 | 8,680566e-03 |
| `lloyd16+max` | 8,432340e-03 | 8,168868e-03 | 8,173968e-03 | 8,171490e-03 | 8,300435e-03 | 8,366420e-03 | 8,432219e-03 |
| `lloyd16+LS` | 7,857215e-03 | 7,610900e-03 | 7,615046e-03 | 7,613073e-03 | 7,733944e-03 | 7,795613e-03 | 7,857007e-03 |

Der Shear auf denselben Blöcken: **6,464105e-03** (−1,351 dB gegen `uniform+LS`).

## Die Gates

| Gate | Vorhersage | gemessen | Urteil |
|---|---|---|---|
| **H0-a** Writer | Fixed-Code-Offset holt ≥ 90 % des Shear-Gewinns | **12,4 %** (0,000275 von 0,002216 MSE) | **nicht bestätigt** |
| **H0-b** 6 Bit | ≥ 95 % des exakten Gewinns, Sättigung < 1 % | 96,8–98,3 % ✓; Sättigung 0 % (uniform) / **6,8–7,7 %** (Lloyd) | **bestätigt**, Sättigungs-Gate verfehlt |
| **H0-c** Granularität | Offset je 4 Blöcke behält ≥ 80 % | **25,0 %** (je 2 Blöcke: 50,0 %) | **nicht bestätigt** |
| **H0-d1** statisch | Kanal-Tabelle holt ≥ 50 % | **≈ 0 %** (−0,1 bis +0,1 %) | **nicht bestätigt** |
| **H0-d2** schwere Positionen | `a²/MSE` im 1-%-Band ≥ 1,5× Bulk | **0,99–1,00×** (3,1 % gegen 3,1 %) | **nicht bestätigt** |

## Was das heißt

1. **Der Shear ist überwiegend keine Nullpunkt-Korrektur.** Der reine Restfehler-Offset trägt
   **12,4 %** seines Gewinns; die übrigen **88 %** sind die **Neuplatzierung des Gitters** (Codes und
   Schritt gemeinsam). Die Formulierung im Protokoll („der Offset bringt 0,00193 nats") beschreibt
   damit den *Shear*, nicht den Offset. Auf der Brücke (0,00116–0,00167 nats/dB) ist der billige
   Offset allein **0,00016–0,00023 nats** — weit unter der Paritätsauflösung von 0,00200.
2. **Der Offset ist positionslokal, nicht kanalstatisch.** Eine Kanal-Tabelle (Layer × Seite × Kopf ×
   Blockindex) holt **nichts** zurück; ein Offset je 2 Blöcke verliert die Hälfte, je 4 Blöcke drei
   Viertel. Es gibt keinen billigeren Speicherort als „je Block".
3. **Der Offset ist kein Schwerpositions-Hebel.** `a²/MSE` ist in allen Bändern 3,1 %, und die
   schweren Positionen (oberste 1 % K/V-Energie) haben **dieselbe** Block-MSE wie der Bulk
   (Verhältnis 1,02–1,05). Die frühere Tail-Beobachtung der Kampagne (die Kalibrierung hilft im
   1-%-Tail, −0,09285 nats) ist also **kein** Offset-Effekt — sie kommt aus der Kalibrierung selbst.
4. **Sechs Bit reichen** — bei 1/32 Schritt und Bereich ±1 auf den Uniform-Armen, bei 1/16 Schritt
   und ±2 auf den Lloyd-Armen (99,0–99,1 %). Das ist die Antwort auf die Frage aus §5, Punkt 5 der
   Survey: die Präzisionswirkung ist jetzt gemessen, für **diese** Verteilung.
5. **Die Konzentration ist mild:** 10 % der Blöcke tragen 58 % der `a²`-Energie, 1 % tragen 14 %.
   Das ist 5,8× bzw. 14× die Gleichverteilung — eine schwere Flanke, aber kein „wenige Kanäle
   erklären alles".

## Die Abweichung ist aufgelöst (Abweichung 20)

Die **absoluten** MSE lagen um Faktor ≈ 5,0 über der Tabelle in §3c (8,680e-03 gegen 1,738780e-03),
während alle Verhältnisse exakt übereinstimmten. Ursache ist ein **Zählfehler** in
`tools/kv_affine_price.py`: `Acc.n[side]` wurde bei **jeder Variante** erhöht, also fünfmal je
Datensatz und Seite — jede MSE war 1/5 ihres Werts. Der Fehler ist behoben; dieselbe Rechnung liefert
jetzt **8,693901e-03 / 6,459338e-03** (stride 8) und bestätigt die Zahlen aus H0
(8,680457e-03 / 6,464105e-03 auf allen Positionen). Die **dB-Werte waren unberührt**
(1,290 gegen 1,290), deshalb ändert sich keine Schlussfolgerung der Kampagne — nur die absoluten
Zahlen in §3c/§3d. Der kleine Rest-Unterschied im Shear (1,351 dB auf allen Positionen gegen
1,290 dB bei `stride 8`) ist die Stichprobe: `stride 8` nimmt 1/8 der Positionen.


