# Tag-Read — Datenerhebung, nicht ausgewertet

**Status: gesammelt, nicht ausgewertet.** Vier Arme, 2026-10-25, 15:19–16:49, ein Prozess
nach dem anderen. Jeder TSV hat 32.746 Zeilen, kein `nonfinite`. Der Tag-Arm hat die Zeile
`STRATA_KV_WINDOW_TAG=1` in `tag.tsv.stderr.txt`, der Fenster-Arm nicht. Die KL ist nicht gerechnet.

Frage, die diese Läufe später beantworten sollen: ändert `STRATA_KV_WINDOW_TAG=1` die cwe-4k-KL
gegenüber dem heutigen Fenster und gegenüber `int8`? Das Flag lässt `qsa_win_slot` jede Ringzeile
exakt lesen, deren Tag die Zelle noch nennt, nicht nur die letzten S. Aus ist der bisherige
Horizont-Check, unverändert.

## Aufbau

Alle vier Arme sind **derselbe Build**. Ein Vergleich gegen die S0-Logs (Engine `40942104…`) wäre
ein Build-Unterschied, kein Arm-Unterschied.

| | |
|---|---|
| Maschine | AMD Radeon RX 9060 XT (gfx1200, 16 GB), Windows 11, ROCm 7.1 |
| Engine | `build-hip-port\strata.exe`, 0.1.47 + S1-Zähler (aus, solange `STRATA_KV_SELSTATS` nicht gesetzt ist) + Tag-Flag, SHA-256 `6F78D941…946AD4` |
| Satz | cwe-4k, `--topk 256` (beide Supports später offline), `--kv` je Arm |
| Pfad | der Yardstick (`tools/quant_quality.py run`), also `--short-read`, nicht der Prefill-Zähler |

| Arm | `--kv` | Env |
|---|---|---|
| fp16 | fp16 | — |
| int8 | int8 | — |
| win256 | q4lm8v | `STRATA_KV_WINDOW=1` `S=256` `R=64` |
| tag | q4lm8v | dasselbe plus `STRATA_KV_WINDOW_TAG=1` |

Ein Arm nach dem anderen. Rohdaten außerhalb des Repos:
`C:\Users\krone\git\Strata-data\bench\2026-10-25-tag-quality\`.

Wanduhr aus dem Laufprotokoll, nicht als K5 gelesen (Debug-Build, Syncs des Zählers aus, aber nicht
der Produktions-Build): fp16 15:19–15:39, int8 15:39–15:58, win256 15:58–16:19, tag 16:19–16:49.
Die Segmentzeiten des Tag-Arms liegen bei 191–227 s, die von int8 bei 128–140 s. Was das bedeutet,
steht hier nicht.

## Reproduktion

```powershell
$exe = "build-hip-port\strata.exe"
$raw = "C:\Users\krone\git\Strata-data\bench\2026-10-25-tag-quality"
$common = @("tools\quant_quality.py", "run", "--config", "strata-swift-iq3_xxs.json",
            "--ids", "data\quant_quality\cwe-4k\set.ids.json", "--topk", "256", "--exe", $exe)
python @common --kv fp16 --out "$raw\fp16.tsv"
python @common --kv int8 --out "$raw\int8.tsv"
python @common --kv q4lm8v --env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 --env STRATA_KV_WINDOW_R=64 --out "$raw\win256.tsv"
python @common --kv q4lm8v --env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 --env STRATA_KV_WINDOW_R=64 --env STRATA_KV_WINDOW_TAG=1 --out "$raw\tag.tsv"
```

Der Tag-Arm muss in `stderr.txt` die Zeile `STRATA_KV_WINDOW_TAG=1` haben. Fehlt sie, hat das Flag
nicht gegriffen und der Arm ist kein Tag-Arm.
