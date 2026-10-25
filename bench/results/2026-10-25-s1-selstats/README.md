# S1 — wo landet die Indexer-Selektion?

**Status: vorregistriert. Der Lauf wartet auf die GPU-Freigabe; es gibt noch keine Zahlen.**
Katalogzeile: `docs/paper/Ansatz_Improvements.md`, E1 („Deckt das Overlay die gelesenen Zellen?").

## Frage

Das Fenster (`q4lm8v` + `STRATA_KV_WINDOW=1`, S=256, R=64) liest zwei Zellmengen exakt: die ersten
R Zellen (Sinks) und die letzten S Zellen (recent). Alles andere liest quantisiert. Der Indexer wählt
pro Query höchstens 2.051 Zellen aus. **Ob diese ausgewählten Zellen in R oder S fallen, ist
ungemessen.** Genau das entscheidet, ob „mehr S/R" (Kataloghebel B1) der richtige Ausgabeort für Bits
ist, oder ob das Overlay Zellen schützt, die niemand liest.

## Aufbau

| Punkt | Wert |
|---|---|
| Engine | 0.1.47 + S1-Zähler (Debug-Build, `build-hip-port\strata.exe`, SHA-256 `44DFF00E…BBA506`) |
| Modell | swift-1.5-iq3_xxs, `--kv q4lm8v`, `--prefill auto` |
| Fenster | `STRATA_KV_WINDOW=1`, `S=256`, `R=64` |
| Passage | cwe-4k, alle 8 Segmente in Reihenfolge konkateniert = **32.754 Token**, kein Trenner |
| Pfad | batched Prefill (`--short-read` entfernt), ein `GEN 1`-Request, `QUIT` |
| Sampling | jede 32. Query je Selektions-Batch → 1.024 Stichproben je QSA-Layer |
| zusätzlich | `STRATA_SEL_OVERLAP` (vorhandener Haken, Schicht 0, 16er-Kacheln) für die Persistenz |

Der Zähler (`src/prefill/prefill.cpp`, hinter `STRATA_KV_SELSTATS`) hängt direkt hinter
`qsa_block_topk` und schreibt eine Zeile je Stichprobe:

```
layer  pos  n_kv  width  blocks  in_r  in_s  in_b  outside  min_sel  max_unsel  s_min  s_max
```

Die vier Fächer folgen **dem Prädikat des Kernels** (`qsa_win_slot`,
`include/strata/kernels/qsa_decode_attn.hpp:73`):

- `in_r` — `c < R`: Sink-Zeile, wird exakt gelesen.
- `in_s` — `c >= n_kv - S`: recent-Zeile, wird exakt gelesen (Tag-Prüfung besteht im Prefill, weil die
  Zelle zu den letzten S angehängten gehört).
- `in_b` — `n_kv - S - B <= c < n_kv - S`: liegt im Ring, **hat ein bezahltes fp16-Overlay, wird aber
  quantisiert gelesen** (`qsa_win_slot` gibt -1 zurück).
- `outside` — außerhalb des Rings: kein Overlay, quantisiert.

`min_sel`/`max_unsel` sind die Blockscores an der Selektionsgrenze. Die Selektion läuft über **ganze
Blöcke** (`include/strata/kernels/qsa_select.hpp:12`), deshalb ist `min_sel - max_unsel` exakt der
Rand: positiv = die Selektion trennt sauber, ≤ 0 = ein Gleichstand, den die Blockreihenfolge
entscheidet.

## Entscheidungsregeln (vor dem Lauf festgelegt)

| Befund | Lesart | nächster Schritt |
|---|---|---|
| Anteil `in_r + in_s` **≥ 50 %** der gewählten Zellen | Das Overlay sitzt auf dem gelesenen Satz | **B1** (mehr S/R) ist der richtige Hebel, Byte-Modell für R+S ≤ 4.096 |
| Anteil **≤ 10 %** | Das Overlay schützt Zellen, die niemand liest | positionsfestes Overlay ist der falsche Ort; Score-getriebener Pool prüfen |
| dazwischen | unentschieden | der Rand entscheidet mit (nächste Zeile) |
| `in_b` **deutlich > 0** | Der Indexer liest Zellen, deren fp16-Kopie bezahlt und nie exakt gelesen wird | Beleg für **B4** (Ringgröße), *nicht* für B1 |
| Rand `p10(margin/(s_max-s_min))` **≥ 5 %**, Inversionen ≤ 1 % | Selektion ist entscheidbar | ein Score-getriebener Pool wäre stabil |
| Inversionen **> 10 %** oder `p50 ≈ 0` | Grenze ist beliebig | Score-Pool würde flackern; **B1** (mehr Zellen) ist die robustere Antwort |

Ein Gleichstand („dazwischen") zählt **nicht** als Erfolg; dann entscheidet der Rand, und ohne klaren
Rand ist das Ergebnis „kein Hebel gefunden".

## Konfundierungen (ausdrücklich)

- **32K statt 128K.** Die Anteile sind längensensitiv: `in_s` ist eine feste Zellenzahl (256) über
  einem wachsenden `n_kv`, also fällt der `in_s`-Anteil mit der Länge, sofern die Selektion breit
  streut. 32K zeigt die Größenordnung, nicht den 128K-Wert.
- **Acht Dokumente aneinandergehängt, ohne Trenner.** Die Nahtstellen sind Dokumentgrenzen, die das
  Modell nie gesehen hat. Für Auswahlstatistik unkritisch, für Qualitätsurteile nicht — dieser Lauf
  fällt kein Qualitätsurteil.
- **Annahme, nicht geprüft:** die Prefill-Selektion einer Query entspricht der Decode-Selektion an
  derselben Position. Begründung: die Scores kommen aus `qsa_block_scores` über die gepoolten
  Indexer-Keys, die nur von den Token-Keys abhängen, nicht von der Attention. Der Prefill geht durch
  einen anderen Attention-Kernel (`qsa_prompt_attn`), was die Scores nicht berührt. Ein eigener
  Prefill-gegen-Decode-Vergleich ist **nicht** Teil dieses Laufs.
- **Debug-Build.** Zwei Readbacks und ein Sync je Stichprobe. Kein Timing-Lauf; die Laufzeit sagt
  nichts über den Produktionspfad.
- **`B = 6.144`** stammt aus `--prefill auto` (Chunk-Obergrenze in `generate.cpp`), nicht aus einer
  Qualitätsentscheidung. Siehe `in_b`.
- Der `int8`-Pfad ist unberührt: der Zähler läuft nur, wenn `STRATA_KV_SELSTATS` gesetzt ist. Ein
  Byte-Identitätsnachweis für `int8` mit diesem Build steht noch aus (Gate vor einem Merge, nicht vor
  diesem Lauf).

## Reproduktion

```powershell
# 1. Bauen (ROCm-clang braucht die MSVC-Umgebung, docs/AMD_HIP.md)
& "$env:TEMP\opencode\build-selstats.ps1"

# 2. Rauchtest: 2.048 Token, prüft nur, dass die TSV entsteht und plausibel ist
python tools\kv_selstats.py run --out C:\Users\krone\git\Strata-data\bench\2026-10-25-s1-smoke `
    --exe build-hip-port\strata.exe --tokens 2048 --overlap

# 3. Der Lauf
python tools\kv_selstats.py run --out C:\Users\krone\git\Strata-data\bench\2026-10-25-s1-selstats `
    --exe build-hip-port\strata.exe --overlap

# 4. Auswertung
python tools\kv_selstats.py report C:\Users\krone\git\Strata-data\bench\2026-10-25-s1-selstats\selstats.tsv
```

Rohdaten (TSV, `stderr.txt`, `index.json`) liegen außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-10-25-s1-selstats\`; hier im Repo stehen nur die
Aggregate dieses Berichts.
