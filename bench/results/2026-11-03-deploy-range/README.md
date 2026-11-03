# Einsatzbereich 32K und 64K: trägt die Tag-Lesung dort, wo das Produkt läuft?

**Status: vorregistriert (2026-11-03), vor dem ersten Arm-Lauf.** Der erste Arm-Lauf darf erst nach
dem Schreiben dieses Dokuments starten. Anschluss an `bench/results/2026-11-03-tag-arm/` (dort 4K
als Diagnose-Punkt und 128K als erster Punkt im Einsatzbereich; 4K ist **kein** Produktpunkt).
Katalogzeilen: `docs/paper/Ansatz_Improvements.md`, E1/S1.

## Warum

Der Einsatzpunkt der Maschine ist ein Kontextfenster von 32K bis 256K, nicht 4K. Gemessen ist in
diesem Bereich bisher nur 128K, und dort ist der Effekt der Tag-Lesung null: `tag − win` =
−0,00001 nats, Block-95 % [−0,00066; +0,00069], bei 48,55 % exakt gelesener Auswahl statt 8,55 %.
Die Abdeckung, die der Tag zusätzlich exakt liest, **fällt mit dem Horizont** (E1/S1 über die
S-Leiter: 100 → 94,6 → 80,0 → 48,6 %). Wenn der Hebel irgendwo im Einsatzbereich etwas bewirkt,
dann bei der kürzesten Länge des Bereichs. Genau das prüft dieser Lauf: **32K und 64K**, mit
demselben Protokoll wie der 128K-Punkt, damit die drei Längen vergleichbar sind.

Die Frage ist ausdrücklich **nicht** „ist 4K besser als 128K", sondern: trägt der Tag bei 32K
und/oder 64K, und wie groß ist dort die Lücke zu `int8`?

## Aufbau

Derselbe teacher-forced Aufbau wie `T3` in `2026-11-03-tag-arm/`: ein Segment, ein `GEN 1`,
per-Position-Logits aus den Verify-Fenstern am Prompt-Ende, `--short-read 6144`, `--topk 256`,
`KL(fp16‖Arm)` über `fp16`s top-256, gepaart je Position, Moving-Block-Bootstrap.

| Punkt | Wert |
|---|---|
| Passagen | `p32k` = die ersten **4** Segmente der E1-Passage (32.756 Token), `p64k` = die ersten **8** (65.511 Token); dieselbe cwe-8k-Passage wie beim 128K-Punkt, verschachtelte Präfixe |
| Warum Präfixe | dieselben Texteinheiten wie 128K, nur kürzer; der Tail jedes Punktes ist das Ende eines anderen Segments, also anderer Text — die Längen sind drei Punkte, kein geschachtelter Satz von Teilmengen desselben Tails |
| Schnitt | `pin = seg_tokens − 6144` (32K: 26.612, 64K: 59.367) plus `--prompt-cache 3`: alles davor batched, die letzten 6.144 Token durch die Fenster |
| Fenster | `STRATA_KV_WINDOW=1 S=256 R=64`, kein `BATCH` (der Ring folgt dem Chunk: 6.144 Zellen, `S+B` = 6.400 Zeilen) |
| Prefill | `--prefill auto` |
| Positionen | 6.144 je Arm und Länge (der Dokument-Tail), `--allow-no-positions` |
| Arme | `fp16`, `int8`, `win` (`q4lm32`, 256/64), `tag` (`q4lm32`, 256/64 + `STRATA_KV_WINDOW_TAG=1`) |

Ein GPU-Job zur Zeit, 30 s zwischen zwei Engine-Starts, Engine-SHA
`CD5700EC0793F83B…3D0394CA` vor dem Lauf geprüft.

## Entscheidungsregeln (vor dem Lauf festgelegt)

Je Länge getrennt, ein Segment, also **keine** Segment-Clusterung; Moving-Block-Bootstrap über
Blöcke von 256 aufeinanderfolgenden Positionen (24 Blöcke, 2.000 Resamples, Seed 20261103).
Δ = KL(fp16‖Arm) − KL(fp16‖Basis), negativ = der Arm ist besser.

- **`D1` — trägt der Tag bei dieser Länge?** Gepaart `tag − win`.
  **„Tag trägt"** nur wenn das Bootstrap-95-%-Intervall die 0 ausschließt, der Mittelwert negativ
  ist **und** |Mittelwert| > 0,0013 nats — dieselbe vor dem 128K-Lauf fixierte Schwelle wie in
  `T3b`, unverändert übernommen. Sonst **„trägt nicht"**, mit Zahl und Richtung.
- **`D2` — wie groß ist die Lücke zu `int8`?** Gepaart `tag − int8` und `win − int8`.
  **„geschlossen"** wenn der Mittelwert von `tag − int8` ≤ 0 ist **und** die einseitige
  95-%-Bootstrap-Obergrenze unter **+0,001 nats** liegt (die K1-Grenze der Leiter). Sonst
  „nicht geschlossen", mit beiden Zahlen.
- **`D3` — ist der Arm informativ?** `KL(fp16‖win)` > 0 mit Bootstrap-95 % ohne 0. Andernfalls hat
  der Ring den Prompt gehalten und `D1` ist für diese Länge **ungültig**.
- **`D4` — Verlauf über die Längen.** 32K, 64K und 128K werden **einzeln** berichtet; keine
  gepoolte Zahl über Längen (verschiedene Positionen, verschiedene Texte — das wäre
  Pseudo-Replikation). Eine Aussage „der Effekt fällt mit der Länge" wird nur gemacht, wenn die
  Vorzeichen der drei Einzelmittel übereinstimmen; sonst bleibt es bei drei Einzelbefunden.

### Güte-Gatter (vor der Auswertung, je Arm)

- `DONE` ohne „nonfinite"; kein „do not fit"; kein `ERR`; Engine-SHA unverändert.
- TSV mit **6.144** Positionen; innerhalb einer Länge in allen vier Armen **dieselbe**
  Positionsmenge; `win`/`tag` mit der Ring-Zeile (6.144 Zellen), `tag` zusätzlich mit der
  `STRATA_KV_WINDOW_TAG=1`-Zeile.
- `seg.json` mit `pin` und `prompt_cache` = 3.

### Abweichungen und Grenzen, offen genannt

1. **Wiederholter Korpus, andere Tails.** 32K endet in Segment s03, 64K in s07, 128K in s15 (dort
   zweimal gelesen). Die drei Punkte sind damit nicht derselbe Text; für die gepaarte Differenz je
   Punkt ist das unkritisch, für einen Längenvergleich ist es eine Grenze und wird so berichtet.
2. **Kein 256K.** Der obere Rand des Bereichs ist mit diesem Lauf nicht abgedeckt; bei 262.144
   Zellen passt `fp16` voraussichtlich nicht in die 16 GB, und der Ring hält dort 2,4 % des
   Kontexts (gegen 4,9 % bei 128K). Eigener Lauf, eigene Entscheidung.
3. **`--short-read 6144`** ist ein Fenster, kein Voll-Prompt-Durchlauf; die 6.144 Positionen sind
   der Dokument-Tail.
4. **Kein Timing, kein Byte-A/B** — der Arm kostet statisch 0 Bytes.

### Kosten

Acht Läufe (2 Längen × 4 Arme), je ~32K: 245 s Prefill + 6.144 Fenster-Schritte, ~64K: 490 s +
6.144 Schritte; nach der 128K-Erfahrung (1.193 s) zusammen etwa 75 Minuten GPU, kein Engine-Bau.

## Reproduktion

```powershell
# ids: die ersten 4 bzw. 8 Segmente von 2026-11-01-e1-128k/cwe8k-x2.ids.json, ein Segment
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids C:\Users\krone\git\Strata-data\bench\2026-11-03-deploy-range\p32k.ids.json `
    --kv <fp16|int8|q4lm32> --prefill auto --short-read 6144 --topk 256 --allow-no-positions `
    --pin 26612 --prompt-cache 3 `
    [--env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 --env STRATA_KV_WINDOW_R=64 `
     [--env STRATA_KV_WINDOW_TAG=1]] `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-03-deploy-range\32k\<arm>.tsv
#   p64k genauso, --pin 59367, Ausgabe ...\64k\<arm>.tsv

python tools\kv_k1.py --ref ...\32k\fp16.tsv --run ...\32k\win.tsv --run ...\32k\tag.tsv `
    --run ...\32k\int8.tsv --kl-topk 256 --block 256 --json ...\32k\k1.json
```

Rohdaten liegen außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-11-03-deploy-range\`; im Repo steht dieses Dokument und
nach dem Lauf der Ergebnisabschnitt darunter.
