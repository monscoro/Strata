# B4 — das Byte-Modell neu gerechnet, und der Prototyp statisch bewertet

**Status: offline gerechnet, 2026-11-01, vor jeder A/B-Messung. Kein Produktionscode, kein
GPU-Lauf.** Katalogzeile: `docs/paper/Ansatz_Improvements.md`, B-Term B4 („Byte-Modell neu
rechnen, dann Same-Day A/B gegen Paper Tab. 4"). Grundbaustein ist die Messung vom 2026-11-01
(`bench/results/2026-11-01-decode-4k/`, Q5): **die exakte Lesemenge des Decode-Pfades hängt vom
Horizont ab (`R` Sinks + `S` Zellen davor), nicht von `B`** — der Ring braucht `B` nur, um eine
ganze Append-Batch zu fassen. Was hier steht, ist Rechnung und statische Analyse; wo eine Stelle
aus dem Code gelesen statt gemessen ist, steht das dabei.

## Das Modell

Ein Ringzeile kostet pro QSA-Layer **2.048 B beider fp16-Zelltensoren** (`qsa_win_tensor_bytes` × 2,
`src/core/layer.cpp:236`: n_head_kv 2 × head_dim 256 × fp16, K **und** V). Dazu die Zelltags des
recent-Rings: 4 B je (S+B)-Zeile (`qsa_win_bytes`, `layer.cpp:241`; Ring = S + B, `layer.cpp:234`).

Die Zelleneinheit ist die skalare KV-Zelle: 1.024 je Position (2 KV-Köpfe × 256 dim × K+V), also
L·1.024 Zellen je Layer bei Kontext L. Daraus der Ringaufschlag:

```
bit/Zelle ≈ 4,50 (Bulk, 16 Level auf K und V, S3(a)) + 16·(R+S+B)/L   mit L = 131.072
```

Die Konstante ist hier nicht Numerologie: 16.384 bit (2.048 B) je Ringzeile geteilt durch die
1.024 Zellen der eigenen Position = 16 bit/Zelle je reservierter Zeile; über den Kontext verteilt
gibt das exakt `16·(R+S+B)/L`. Die Tags sind auf dieser Ebene zweitrangig: +0,0015 bit/Zelle bei
B = 6.144 (0,03 %) — unter der Granularität der Anker, im Folgenden nicht mitgereist.

## Validierung gegen die veröffentlichten Anker

Zwei Anker sind aus Messläufen publiziert; beide treffen ohne freien Parameter:

| Anker | Publikation | Modell | |
|---|---|---|---|
| Produktarm, B = 6.144 | **5,29 bit/Zelle** (`2026-10-28-ring-geometry`) | 4,50 + 16·6.464/131.072 = 5,289 | ✓ |
| Geometrie-Arm, B = 512 | **4,60 bit/Zelle** (ebenda) | 4,50 + 16·832/131.072 = 4,602 | ✓ |
| Ringanteil B = 6.144 | **151 MB** (6.144·2.048·12, ebenda) | 150,99 MB | ✓ |
| Aufschlag 0,79 bit/Zelle am 128K-Punkt, davon 0,75 durch `B` (Paper §3, Katalogzeile) | 16·6.464/L = 0,79; 16·6.144/L = 0,75 | ✓ beides exakt |

Zusätzlich das Byte-Soll des Kataloghebers, dieselbe Formel weitergeschoben:

| `B` | Ringzeilen (64+256+B) | bit/Zelle | Ring-Bytes, 12 Layer | Tags, 12 Layer |
|---|---|---|---|---|
| 512 | 832 | **4,60** | 20,4 MB | 0,04 MB |
| 1.024 | 1.344 | 4,66 | 33,0 MB | 0,06 MB |
| 2.048 | 2.368 | 4,79 | 58,2 MB | 0,11 MB |
| 4.096 | 4.416 | 5,04 | 108,5 MB | 0,21 MB |
| 6.144 (Produktpunkt) | 6.464 | **5,29** | 158,9 MB | 0,31 MB |

Delta 512 → 6.144: **138,4 MB** (5632 Zeilen · 2.048 B · 12). Der `B`-Anteil steckt praktisch den
kompletten Aufschlag zwischen 4,60 und 5,29 auf; „mehr recent (`S`)" dagegen kostet 16·256/131.072
= 0,031 bit/Zelle je Schritt — der Ring ist der teure Hebel, `S` ist der billige. (Das ist dieselbe
Lesung wie in der Ring-Geometrie: bei 4.608 Zellen ist der Ring der knappe Teil.)

## Der Prototyp, statisch bewertet

Katalogoption: *die fp16-Kopie nur für den laufenden Append-Batch resident halten und nach der
Attention wieder freigeben, sodass mehrere kleinere Chunks vorprefillt werden*. Bewertung aus dem
Codestand (statisch, unbestätigt, außer wo ein Messverweis steht):

1. **Was der Reader liest.** Decode liest exakt `c < R` oder `c ≥ n − S` (`qsa_win_slot`,
   `include/strata/kernels/qsa_decode_attn.hpp`); Ringzeilen jenseits der letzten `S` kommen
   im Prädikat **nicht** vor. Gemessen (nicht nur statisch) am 4K-Antwortpunkt: Q5,
   +0,00005 [−0,00188, +0,00197], 4/8 Segmente. Am Decode-Punkt sind die `B`-Zeilen also
   bezahlt und ungelesen — sie entliegen zu lassen ändert nichts an dem, was Decode liest.
   **Statisch sauber für Decode; an 128K ungemessen** (dieser Beweisträger ist E1, nicht B4).
2. **Was Prefill braucht.** Eine Prefill-Query liest `[pos0 − S, pos0 + B)` exakt
   (`2026-10-28-ring-geometry`); pro Append-Batch dient dieselbe `B`-Zeilen-Spanne genau
   einem Batch. Zwischen Batches ist sie veraltet — der SELSTATS-Zähler nennt Selektionstreffer
   dort `in_b`. **Während eines Batches kann die Spanne nicht entlassen werden**: der Ring muss
   die ganze Batch tragen (`kv_q4.cu:483`: `T ≤ S + B`, sonst laut `do not fit` und Exit 1;
   am 2026-11-01 am echten Lauf reproduziert).
3. **Wo der Ring sitzt.** Er ist einmalig bei Init pro Zustand carved, beide Tensoren + Ring-Tag
   (`layer.cpp:998–1002`, `take()` in Elementen), Gitter nach dem Prompt-Chunk eingefroren
   (`layer.cpp:220–223`); Ringe sind nie mit elastischem K/V kombinierbar (`layer.cpp:994`).
   Ein „entleeren, danach wieder befüllen"-Ablauf braucht einen Carve-Plan, der Freigabe und
   Neubelegung trägt (die Frozen-Grid-Invariante bleibt gewahrt, solange der Carve selbst besteht
   und nur der Inhalt entleert wird) — **ein solcher Ablauf ist nicht gebaut**; der Prototyp
   ist hier bewertet, nicht vorhanden.
4. **Was die Freigabe wirklich spart.** Zurück kommt der `B`-Anteil am Ende des Prefills: am
   Produktpunkt **151 MB** (der `B`-Anteil des 158,9-MB-Gesamtrings); gegenüber dem 512er-Arm
   sind es **138,4 MB**. Diese Zeilen können zwischen letzter Append-Batch und den
   Decode-Schritten frei sein; ein 512er-Ring wiegt auch während des Decode nur 12,6 MB.
   Während des Prefills spart die Variante nichts: die Zeilen braucht genau der Batch, in dem
   sie geschaffen werden. Die Luft am Decode-Punkt könnte Expert-Cache tragen (statisch;
   unbestätigt, bis gemessen).
5. **Was der Prototyp nicht schenkt.** Bei Batch gleich Chunk (Produktpunkt-Form) ändert sich
   an den gelesenen Zellen nichts — es verschiebt nur Residenz. „Mehrere kleinere Chunks bei
   gleicher Latenz" geht an genau dem Punkt vorbei, an dem eines mit `T > S + B` heute laut
   abgelehnt wird: ein
   832-Zeilen-Ring mit 6.144er-GPU-Chunk müsste die Chunk-Append in 12 Ring-Batches zu 512
   spalten — und Queries, deren Horizont über die letzten `S` Zellen des Chunks hinausreicht,
   lesen den älteren Teil **des eigenen Chunks quantisiert**. Das wäre eine Qualitätseinbuße,
   die der heutige Prefill nicht zahlt, und die in der Tabelle oben nicht steckt: 4,60 bit/Zelle
   steht für „832 Ringzeilen resident", nicht für „Prefill liest weniger exakt". Gleiche Latenz,
   derselbe Satz exakter Reads, weniger Zeilen — das Trio ist im heutigen Code nicht gleichzeitig
   verfügbar. Der Prototyp liefert davon zwei (weniger residente Zeilen, Produktpunkt-Latenz),
   der dritte Weg bleibt eine Messung.

## Fazit (Rechnung + statische Analyse; jede Messung ist als solche gekennzeichnet)

- Das Byte-Modell ist neu gerechnet und trägt beide publizierten Anker exakt; der Aufschlag
  zwischen 4,60 und 5,29 bit/Zelle ist der `B`-Anteil (0,69 bit/Zelle = 16·5632/131.072).
- Die Prototyp-Option spart am Decode-Punkt bis zu **151 MB** (Produktpunkt) bzw. **138,4 MB**
  gegenüber dem 512er-Arm; die Decode-Verteilung ändert sich dabei nicht, gemessen am 4K-Punkt
  (Q5) und für 128K statisch geschlossen. **Sie kostet** dieselbe Prefill-Latenz wie heute und
  ändert nichts an bit/Zelle. „Beliebig kleinere
  Chunks ohne Kosten" existiert im Code nicht; der freie Weg dorthin läuft über die Kauf-Kante
  „ältere Teile des eigenen Chunks quantisiert", die dieses Modell nicht anpreist.
- Offen bleibt die Laufzeit-/Byte-A/B gegen Paper Tab. 4 (Katalogzeile B4), nicht hier gelaufen —
  ein GPU-Gerät nach dem anderen, und B1 ist in dieser Runde per Auftrag ausgeschlossen.

## Dateien

- Keine neuen Rohdaten; eine Rechnung, deren Referenzstellen im Text stehen:
  `src/core/layer.cpp` (234, 236, 241, 994–1002), `src/kernels/cuda/kv_q4.cu` (483–489),
  `src/program/generate.cpp` (3993, 4007), `include/strata/kernels/qsa_decode_attn.hpp`
  (`qsa_win_slot`).
- Messverweise: `bench/results/2026-10-28-ring-geometry/` (Anker 4,60 / 5,29 / 151 MB),
  `bench/results/2026-11-01-decode-4k/` (Q5: der Ring ist am Decode-Punkt unsichtbar).
