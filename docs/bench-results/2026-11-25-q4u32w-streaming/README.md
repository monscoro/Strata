# q4u32w mit `--kv-resident`: die Korrektheits-Gates (2026-11-25, portiert auf 0.1.70)

Diese Datei hält fest, was das Fenster aus `q4u32` zusammen mit KV-Streaming freigibt. Die Reihenfolge war
vereinbart: erst die zwei Korrektheits-Gates, danach Laufzeit. Beide Gates sind grün. Laufzeitzahlen stehen
hier bewusst nicht.

## Maschine und Build

* Windows, ROCm 7.2, eine AMD Radeon RX 9060 XT (gfx1200), 16 GiB VRAM.
* Engine `0.1.70` aus diesem Baum, Basis `4560d0379`.
* Modell: `qwen-iq3_xxs` mit nativem IQ3_XXS-Shard 1 und PLE-Shard 2, Experten aus den GGUF-Dateien
  (`--mmap-experts`), fester 256-Slot-Cache, kein MTP-Ring (der MTP trägt kein Fenster).
* `STRATA_KV_WINDOW=1`, S/R 256/64. Der fp16-Seitenspeicher des Fensters liegt in beiden Armen im VRAM.
* `run_gate.py <resident|streamed> <tag> [reserve]` liest `gate-config.json` aus diesem Ordner; `{ROOT}` wird
  zum Checkout-Wurzelverzeichnis aufgelöst. Der Prompt sind die ersten 256 IDs aus
  `terminal/data/quant_quality/cwe-4k/set.ids.json`, gespeichert in `prompt256.tokens`; Kontext 32.768 Zellen.

## Gate 1 — Kernel-Parität

`ctest -R "kv_q4_parity|kv_stream_parity|kv_win_parity"`: **6/6 bestanden** (mit `_lanecell`, `_win_all`,
`_refuse`). Der `q4u32w`-Arm in `engine/src/kernels/kv_stream_parity.cpp` baut den q4-Blockpool mit Seitenpools
und Tags auf, schreibt über `kv_append_win_steps` in den gestreamten Zustand und vergleicht die
Attention-Ausgabe bitweise mit dem vollständig residenten Zustand.

## Gate 2 — Ende-zu-Ende-Logits

Gleicher Prompt, gleicher Build, gleicher Rest der Argumente. Einziger Unterschied: die gestreamten Arme
setzen `--kv-resident` und nehmen echte Evictions und Host-Nachläufe in Kauf.

| Arm | `--kv-resident` | Bedeutung |
|---|---|---|
| `resident-a`, `resident-b` | nicht gesetzt | Referenz, vollständig resident; zwei Läufe |
| `streamed-evict` | 20.480 von 32.768 Zellen | die Stream-Map gibt Blöcke aus und holt sie aus der Host-Kopie zurück |
| `streamed-noevict` | 32.768 von 32.768 Zellen | Host-Kopie vorhanden, es wird nichts ausgegeben |

Alle vier Arme schreiben **bytegleiche** Logits: 993.288 Byte, SHA-256 `cdeb5cc7…`, erstes Greedy-Token
`cfeaae10…`. Der Hash unterscheidet sich vom Fork-Lauf (`75cb396e…`), weil die Basis sich bewegt hat;
bytegleich **innerhalb dieses Builds** ist genau das, was das Gate zeigen soll.

## Die Ablehnung bleibt

`--kv q4lm8v` mit Fenster und `--kv-resident` endet mit Exit 2 und dem Hinweis, dass nur `q4u32`s Fenster
bitweise geprüft ist — vor dem Laden des Modells; `kv_win_parity_refuse` hält das fest. Ein Snapshot eines
fensterbehafteten Zustands wird weiterhin abgelehnt. Für `q4u32w` ist das Paar erlaubt, weil Gate 1 und 2 es
decken.

## Was diese Datei nicht behauptet

* Keine Laufzeitmessung. Streaming kostet Zeit und spart VRAM; die Zahlen dazu wurden auf einem anderen Build
  gemessen und stehen deshalb nicht hier.
* Keine Qualitätsmessung. Die KL-/Top-1-Werte für `q4u32w` wurden vollständig resident gemessen
  (`2026-11-25-q4u32-window-ab/`) und gelten hier nur, weil Gate 1 und 2 die Bitgleichheit mit dem gestreamten
  Pfad zeigen.
* Kein Vergleich gegen `oct3` oder `q4lm8v`.
