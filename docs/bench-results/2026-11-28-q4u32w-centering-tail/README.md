# 128K-Tail: `q4u32w` ohne gegen mit Mittelwert-Kalibrierung (vorab festgelegt, 2026-11-28)

Diese Datei legt fest, **was gemessen wird und wann ein Ergebnis als Gewinn zählt**, bevor die Zahlen da
waren. Die Frage stand vor dem PR mit den `q4u32`/`q4u32w`-Formaten: verschenkt `q4u32w` ohne
Kalibrierung bei 128K Kontext P99? Das Fat-Tail-Problem der KV-Quantisierung soll weitergeschoben werden,
nicht stehen bleiben.

Der Lauf lief auf dem Fork-Build `0.1.63-q4u32` (SHA-256 `3f3a336a…`, 45.682.688 Byte). In diesem Baum
ist derselbe Punkt `--kv q4u32w`; der Port hat die Gates neu gefahren, nicht die KL-Werte. Die
Reproduktion unten zeigt, dass der Lauf zum A/B passt.

## 1. Die Frage

Gibt die **Mittelwert-Kalibrierung** (die Kanal-Offset-Tabelle vor der Rotation, `STRATA_KV_CENTER`, aus
`tools/kv_center_table.py` aus dem cwe-4k-Dump) dem Fensterformat `q4u32w` (256/64, B = 6144) bei **128K
Kontext im Tail** etwas zurück: kleineres **KL-p99** gegen fp16, ohne dass der Mittelwert nachgibt?
Endpunkt ist `q4u32w` mit Kalibrierung gegen `q4u32w` ohne — beide auf einem Binary, dieselben Ids,
gepaart wie im A/B.

Warum offen: 4K ist die einzige bisher gemessene Paarung (`q4u32`±C: **+0,00321 nats zugunsten C,
t = 2,04**, Quelle: die 4K-Messung vom 13.11.2026, kv-offset-tail §0, Zitat aus protocol.md 2915); beim
Lloyd-Codec half C nur im p99 (0,2410 → 0,2263) und dort **nicht gepaart signifikant** (McNemar
p = 0,18, die Centering-Kampagne vom 10.11.2026, C4/D1-Nachtrag). Die dafür registrierten 128K-Stufen —
**D1** (10.11., out-of-sample) und **H1/H2** (13.11.) — sind vorregistriert und **nie gefahren**.

## 2. Was schon da ist (wird nicht neu gemessen)

128K ohne Kalibrierung (A/B 2026-11-25, `cwe-128k/t1`, 131.060 Positionen, fp16-Referenz,
`compare-fp16.json`):

| Arm | mean KL | **KL p99** | Top-1 |
|---|---:|---:|---:|
| `fp16` (Referenz) | 0 (PPL 2.750587) | — | — |
| `q4u32` (ohne Fenster) | 0.078700 | **2.356977** | 98.173 % |
| `q4u32w` (ohne C) | 0.069493 | **1.983749** | 98.308 % |

**Erwartung (Rechnung, keine Hoffnung):** die Wirkung der Kalibrierung ist pro Zelle und damit
kontextunabhängig (D1-Nachtrag 2026-11-11): ≈ 0,002–0,003 nats im Mittel bei 128K. Die Auflösung des
Paares liegt bei 0,00254 nats (A/B-README) — das ist knapp oberhalb der Auflösung entscheidbar. Für den
**p99** gibt es keine gepaarte Auflösung; deshalb ist er als *relatives* Kriterium (T1 unten) festgelegt
und nicht als punktweiser Beleg.

## 3. Entscheidungsregel (vor dem Lauf fixiert)

Zwei Arme, **ein** Binary, dieselben Ids, gepaart über 131.060 Positionen.

- **T1 — Tail-Gewinn (primär):** `p99(ohne C) − p99(mit C) > 0` **und** der Gewinn ist
  ≥ **5 %** des ohne-C-Werts (1,983749 × 0,95 ≈ 1,885). Unterhalb von 5 % gilt der Tail als
  nicht bewegt, weil p99 ohne gepaartes Maß zwischen Einzelläufen schwer zu fassen ist.
- **T2 — kein Regress im Mittel (Gegenprobe):** die gepaarte mean-KL-Differenz
  `Δ = KL(mit C) − KL(ohne C)` hat eine obere 95-%-Grenze **≤ +0,0010 nats**. Die Erwartung
  aus 4K (−0,00321 zugunsten C) darf nicht umkippen: schließt das Intervall 0 ein, ist der
  Effekt im Mittel „nicht unterscheidbar", positiv mit obere Grenze > +0,0010 ist er ein
  **Regress** und T1 damit wertlos (Tail auf Kosten des Mittels).
- **T3 — Nebenbeobachtung (kein Tor):** Top-1 gegen fp16 mit exaktem McNemar
  (`tools/kv_tail_top1.py`, sobald der int8-Arm als Tail-Definition läuft).
- **Wirksamkeitsgatter:** (a) beide Arme liefern 131.060 Positionen; (b) die SHAs der beiden
  Arme sind verschieden (wie C3: ein Arm, der „erfolgreich" nichts tut, fällt nur im
  Zahlenvergleich auf — die Grenze, an der die Zentrierung im Verify-Pfad 2026-11-10 versagt
  hat); (c) die Offset-Tabelle ist aus **cwe-4k**, die Texte aus **cwe-128k**: der Lauf ist
  out-of-sample, genau das ist die Frage, nicht die Voraussetzung.

**Entscheidung:** erfüllen T1 **und** T2, ist `q4u32w + C` der empfohlene Punkt. Erfüllt nur T2,
bleibt C draußen und der Fat-Tail-Hebel ist bei diesem Codec **ausgereizt** (Deckung mit C4: C ist kein
Produktpfad). Erfüllt weder T1 noch T2: protokollieren und den Hebel als verschwendet benennen.

**Gestuft:** erst Text `t1` (der A/B-Text); ein zweiter Text (`t2`) nur, wenn T1 offen bleibt.
Ein `int8`-Arm als Tail-Definition (4. Arm) wird ergänzt, sobald T3 gebraucht wird — er ist
nicht Teil von T1/T2.

## 4. Arme und Reproduktion

Arme (sonst identisch zum A/B: `quant_quality.py run`, `cwe-128k/t1.ids.json`, `--prefill 6144`,
`STRATA_KV_WINDOW_S=256`/`_R=64`/`_BATCH=6144`, `STRATA_VERIFY_ASYNC_LAUNCH=1`, kein
`--kv-resident`, CPU-Experten, ein Slot, `--max-context 131072`, RX 9060 XT / gfx1200, ROCm 7.2):

| # | Arm | `--kv` | Env |
|---|---|---|---|
| 1 | Reproduktion `q4u32w` | `q4u32` | `STRATA_KV_WINDOW=1` |
| 2 | **Kalibriert** | `q4u32` | `STRATA_KV_WINDOW=1`, `STRATA_KV_CENTER=<center-table.bin>` |
| 3 | Reproduktion der Referenz | `fp16` | — |
| 4 (gestuft) | Tail-Anker | `int8` | — |

Der Lauf predatet den Namen `q4u32w`: auf dem Fork-Build hieß der Punkt `--kv q4u32` plus
`STRATA_KV_WINDOW=1`. In diesem Baum füllt `--kv q4u32w` dieselbe Env selbst; `run-center.py` behält die
Schreibweise, die tatsächlich lief.

- **Tabelle bauen:** ein 4k-Lauf mit `STRATA_KV_Q4_DUMP` (Range `..._FROM`/`..._TO`) auf
  `terminal/data/quant_quality/cwe-4k/`, dann `python tools\kv_center_table.py --dump <dump> --out center-table.bin`.
  Achtung: die Tabelle ist **je Kopf und je Seite** (Abweichung 15) und der Dump die rotierten
  Zellen; ein trunkierter Dump verfälscht sie (Abweichung 11/15).
- **Auswertung:** `python terminal\tools\quant_quality.py compare <fp16.tsv> <ohneC.tsv> <mitC.tsv> --paired`
  für mean/p99 und die gepaarte Differenz; `evaluate.py` setzt T1/T2 mechanisch um.
- **Kosten:** der Tabellen-Dump ≈ ein 4k-Lauf (Minuten), die drei 128K-Arme ≈ 72–75 min je Arm,
  zusammen ≈ **4 h GPU**. Ein vierter Arm addiert ~75 min.

## 5. Ergebnis (28.11.2026)

Die drei Arme liefen in einer Sitzung auf dem Fork-Build `3f3a336a…`:

| Arm | Positionen | mean KL | **KL p99** | Top-1 | Dauer | SHA-256 (16) |
|---|---:|---:|---:|---:|---:|---|
| `q4u32w` ohne C | 131.060 | 0.069493 | **1.983749** | 98.308 % | 4.520 s | `debdee3a8cd31011` |
| `q4u32w` mit C | 131.060 | 0.069453 | **2.018592** | 98.304 % | 4.329 s | `eb3c5952f9f08c75` |

**Gatter:** (a) beide Arme 131.060 Positionen — erfüllt. (b) SHAs verschieden — erfüllt; die
Kalibrierung hat den Lauf also verändert. (c) Tabelle aus cwe-4k, Texte aus cwe-128k — erfüllt. Die
Reproduktion ist zusätzlich belegt: der ohne-C-Arm ist **byte-identisch** zum A/B-`q4u32w`
(`debdee3a…`), und der fp16-Referenzarm ist **byte-identisch** zum A/B-fp16 (`19d98337…`). Also sind
Binary, Env, Ids und Referenz derselbe Satz wie am 25.11.

**T1 — Tail-Gewinn: nicht erfüllt.** `p99(ohne C) − p99(mit C) = −0,034842` — der p99 wird **1,76 %
schlechter**, nicht besser. Die Regel verlangte einen Gewinn von ≥ 5 % (p99 ≤ 1,885).

**T2 — kein Regress im Mittel: nicht erfüllt.** `Δ = KL(mit C) − KL(ohne C) = −0,000040 nats`,
95 %-Intervall `[−0,003174, +0,003093]`, `t = +0,03` über 131.060 gepaarte Positionen. Die obere Grenze
liegt mit +0,0031 über der geforderten +0,0010; das Paar kann einen Regress bis +0,0031 nicht ausschließen.
Die Auflösung bei wahrer Parität liegt bei 0,00263 nats — der Lauf ist für einen Effekt dieser Größe
schlicht nicht empfindlich genug, und der primäre Tail-Endpunkt zeigt in die falsche Richtung.

**Entscheidung nach der vorregistrierten Regel: weder T1 noch T2.** Die Kalibrierung bleibt draußen; der
Fat-Tail-Hebel ist bei diesem Codec ausgereizt. Das ist die dritte Messung in dieselbe Richtung (4K
+0,00321 zugunsten C an der Harness-Auflösung 0,00200; Lloyd nur im p99, McNemar p = 0,18).

## 6. Was diese Stufe nicht behauptet

- Kein Wechsel des Default: die Stufe misst einen Hebel; `q4u32w` bleibt eine Opt-in-Wahl.
- Ein Text (`t1`), eine Karte (RX 9060 XT gfx1200, Windows, ROCm 7.2), ein Binary; die Aussage gilt
  für diesen Satz.
- p99 hat kein gepaartes Auflösungsmaß — T1 ist als relative Regel (5 %) gesetzt, nicht als
  Signifikanztest. Der gepaarte Beleg liegt im Mittel (T2).
- Die Offset-Tabelle stammt aus **einem** cwe-4k-Segment (4.095 Zellen je Layer, 12 QSA-Layer, 2 Heads
  × 256 Kanäle). Out-of-sample wie vorregistriert, aber schmaler als „mehrere 4k-Texte".
- Die Draft-Layer bleibt unzentriert (MTP misst seine eigenen K/V; C2-Grenze), der Hybrid nicht
  betroffen (rotiert V ohne Centering). Der MTP-Draft läuft im A/B-Pfad ohnehin nicht.
- Mit der Kalibrierung kämen ein Artefakt (die Tabelle) und die Snapshot-Basisdisziplin (+32) zurück —
  das ist Produktarbeit, unabhängig vom Messergebnis.
