# q4u32w mit `--kv-resident`: Korrektheits-Gates und erste Laufzeitmessung (2026-11-25)

Hier steht, was die Kombination aus `q4u32`-Fenster und KV-Streaming freigibt und was nicht.
Die Reihenfolge war vereinbart: erst die zwei Korrektheits-Gates, danach Laufzeit. Beide Gates
sind grün. Die Laufzeitwerte unten sind eine erste Orientierung, kein Freigabe-Benchmark.

## Maschine und Build

* Windows, ROCm 7.2, eine AMD Radeon RX 9060 XT (gfx1200), 16 GiB VRAM.
* Engine `0.1.63-q4u32`, Checkout `2cf00b099` plus die in dieser Arbeitskopie geänderten Dateien.
  Installierte `terminal\engine\strata.exe`: 45,681,664 Byte, SHA-256
  `e1be2bc764570709dfb77e075066b474b8099654c09edabef33e0bf5d2fb4532`.
* Modell: `qwen-iq3_xxs` mit nativem IQ3_XXS-Shard 1 und PLE-Shard 2. Experten aus den GGUF-Dateien
  (`--mmap-experts`), Expert-Profil, feste 256-Slot-Caches (der Motor meldet 336 Slots). Kein MTP;
  `--spec 4` läuft nur als Suffix-Drafter. Das hält die verglichenen Arme identisch.
* `STRATA_KV_WINDOW=1`, S/R 256/64. Der fp16-Siebenspeicher des Fensters liegt in beiden Armen im VRAM.
* `STRATA_PREFILL_CPU_SHARE=0`, `STRATA_PREFILL_WARM=0`, `STRATA_WDDM_BUDGET=0`. Der letzte Schalter
  nimmt die künstliche WDDM-Budgetgrenze aus der Messung; die freie VRAM-Zeile stammt weiterhin von
  `hipMemGetInfo`, nicht aus einer Prozess-RSS-Zählung.
* Reproduktionsskripte: `run_gate.py`, `refusal_check.py`, `perf_probe.py`, `perf_run.py`. Sie lesen alle
  `gate-config.json` aus diesem Ordner; `{ROOT}` wird zum Checkout-Wurzelverzeichnis aufgelöst, die Modelle
  liegen relativ dazu in `..\Strata-data` und `..\flash-next`. Die Prompts liegen als Token-IDs bei:
  `prompt256.tokens`, `prompt4096.tokens`, `prompt128k.tokens`, `prompt256k.tokens`.

## Gate 1 — Kernel-Parität

`ctest -R "kv_stream_parity|kv_win_parity"` auf gfx1200: **5/5 bestanden**, 255,75 s. Rohausgabe in
`ctest.txt`. Der neue Arm `kv_stream_parity_lanecell` läuft denselben Test mit
`STRATA_ATTN_LANECELL=1`, also über den zweiten Fensterleser.

Der `q4u32w`-Arm in `engine/src/kernels/kv_stream_parity.cpp` baut den q4-Blockpool mit
Seitenpools und Tags auf, schreibt über `kv_append_win_steps` in den Streamed-Zustand und vergleicht
die Attention-Ausgabe bitweise mit dem vollständig residenten Zustand. Bei der ursprünglichen
Ausführung: 306 Batches, 706.019 Blockzugriffe, 72.175 Misses (89,8 % Treffer), kein Overflow,
keine Abweichung. Der Ring-Restore-Pfad wird in diesem Test ausgelassen; MTP-Ringe tragen kein Fenster.

## Gate 2 — Ende-zu-Ende-Logits

Gleicher Prompt, gleicher Build, gleicher Rest der Argumente. Einziger Unterschied: der streaming-Arm
setzt `--kv-resident` und nimmt damit echte Evictions und Host-Nachläufe in Kauf.

* Prompt: die ersten 256 IDs aus `terminal/data/quant_quality/cwe-4k/set.ids.json`, gespeichert in
  `prompt256.tokens` (SHA-256 `88e11621955e7c21ea2336c4c0440812ae5b65f6e037e7d9176b86e1f50fc31b`).
* Kontext: 32.768 Zellen. Beide Arme laufen mit `--max-new 1`.
* Resident-Arme (`resident-a`, `resident-b`, `resident-evict`): kein `--kv-resident`.
* Streaming ohne Eviction (`streamed-noevict`): `--kv-resident 32768` bei 32.768 Kontextzellen — die Host-Kopie
  existiert, es wird nichts ausgegeben.
* Streaming mit Eviction (`streamed-a`, `streamed-evict`): `--kv-resident 20480` bei 32.768 Kontextzellen; die
  Stream-Map muss Blöcke ausgeben und aus der Host-Kopie zurückholen (die Startzeile sagt `20480 of 32768 cells`).

| Datei | Größe | SHA-256 |
|---|---:|---|
| `resident-a-logits.bin` | 993.288 | `75cb396e76efb4848298f82ecff0c65f6757879c4331ce9acfc8b6c99e3be066` |
| `resident-b-logits.bin` | 993.288 | `75cb396e76efb4848298f82ecff0c65f6757879c4331ce9acfc8b6c99e3be066` |
| `streamed-a-logits.bin` | 993.288 | `75cb396e76efb4848298f82ecff0c65f6757879c4331ce9acfc8b6c99e3be066` |
| `streamed-noevict-logits.bin` | 993.288 | `75cb396e76efb4848298f82ecff0c65f6757879c4331ce9acfc8b6c99e3be066` |
| `resident-evict-logits.bin` | 993.288 | `75cb396e76efb4848298f82ecff0c65f6757879c4331ce9acfc8b6c99e3be066` |
| `streamed-evict-logits.bin` | 993.288 | `75cb396e76efb4848298f82ecff0c65f6757879c4331ce9acfc8b6c99e3be066` |

Alle sechs Dateien sind **bytegleich**. Das gleiche gilt für die sechs `*-first.bin`-Dateien
(993.280 Byte, SHA-256 `1624cc6d367b18881211555baec379d5db88fe6e3b04c0ccf22220cb0cb7e3a7`,
Greedy-Token 2423). Damit deckt der Test sowohl den Streamed-Pfad ohne Ausgabe als auch echte
Evictions ab. Die Dateien enthalten Header und eine Logits-Zeile: ein native IQ-Pack läuft nicht
durch die per-Token-Dump-Schleife. `--dump-logits` schreibt für diesen Pfad jetzt die Zeile des
ersten Verify-Fensters; vor der Änderung war die Datei trotz Erfolgsmeldung leer.

Die Ablehnung bleibt für die anderen Fensterarme aktiv. `refusal_check.py` startet denselben Build
mit `--kv q4lm8v`, `STRATA_KV_WINDOW=1` und `--kv-resident 20480`; `refusal.log` zeigt Exit 2 und
den Hinweis, dass nur `q4u32` bitweise geprüft ist (was vor dem Laden des Modells passiert). Ein
Snapshot eines fensterbehafteten Zustands wird weiterhin in
`engine/src/core/conversation_snapshot.cpp` abgelehnt. Setup schreibt `--kv-resident` jetzt für die
`q4u32`-Fensterwahl, wenn RAM vorhanden ist und `--kv-streaming` nicht `off` ist; für andere
Fensterarme bleibt die Ablehnung. Setup schreibt das Paar erst ab Engine `0.1.63-q4u32`; ein älterer
q4u32-Build oder ein Upstream-Build bekommt weiterhin keine Config mit `--kv-resident` neben dem Fenster.
`--kv-check` prüft zusätzlich die Engine-Version: eine Config mit `q4u32`-Fenster und `--kv-resident`
gilt erst ab `0.1.63-q4u32` als startbar. `terminal\tools\test_setup_choices.py` läuft ohne Fehler
durch (75 Tests; `pytest` ist auf dieser Maschine nicht installiert).

## Default-Pfad nach dem Rebuild

Der Rebuild nach der Default-Änderung (Engine-Auto-Pfad und Setup schreiben `q4u32` + `STRATA_KV_WINDOW=1`) hat
SHA-256 `0a3b299211320432ef3060593534525740cf509102452aded35f22900b16198f` (45.682.688 Byte,
`0.1.63-q4u32`). Mit ihm wurden Gate 1 (5/5, 159 s) und Gate 2 erneut gefahren: alle fünf Wiederholungen
(`reg-resident-a/-b`, `reg-streamed-a`, `reg-resident-evict`, `reg-streamed-evict`) reproduzieren die
dokumentierten Dateien byteidentisch (`75cb396e…` bzw. `1624cc6d…`); die `reg-*`-Dateien sind danach gelöscht,
die Zahlen stehen hier. Der zuvor fehlende Arm ohne Eviction (`streamed-noevict`, `--kv-resident 32768`) wurde
mit demselben Binary neu gemessen und ist ebenfalls bytegleich.

`default_check.py` startet die Gate-Argumente einmal **ohne `--kv`** (der neue Default) und einmal mit
`--kv q4u32` (reines Grid), 255-Token-Prompt, 32K Kontext:

| Arm | Startzeile | Freies VRAM (1-Token-Fenster) |
|---|---|---|
| ohne `--kv` | `KV storage: q4u32 with the exact-cell window (q4u32w) …` | 11.261 MiB |
| `--kv q4u32` | keine KV-Zeile (explizit benannt) | 11.280 MiB |

Die 19 MiB sind der Ring bei B = 256 (der Prompt hat 255 Token; bei B = 6.144 sind es 152 MiB, siehe den
128K-A/B). Rohausgaben: `default-check-auto-default.log`, `default-check-plain-grid.log`.

## Erste Laufzeit- und Speichermessung

Diese Zahlen stammen aus einem 4.095-Token-Prompt bei **deklariertem** Kontext von 128K oder 256K.
Sie messen also Prefill und Decode mit der vollen KV-Reservierung, aber **nicht** einen vollständig
gefüllten 128K/256K-Kontext. Zwei Runden in umgekehrter Reihenfolge; die erste Runde war auf dieser
nicht ganz ruhigen Maschine teils deutlich langsamer. Deshalb werden beide Werte gezeigt.

| Kontext | Arm | Prefill tok/s, Runde 1 / 2 | Decode tok/s, 1 / 2 | Freies VRAM bei 4-Token-Fenster | Pinned RAM für KV |
|---|---|---:|---:|---:|---:|
| 128K | voll resident | 60,8 / 90,1 | 1,24 / 1,82 | 7.752 MiB | — |
| 128K | `--kv-resident 32768` | 43,8 / 85,8 | 0,81 / 2,01 | 8.326 MiB | 0,84 GiB |
| 256K | voll resident | 78,8 / 63,3 | 1,94 / 2,28 | 6.661 MiB | — |
| 256K | `--kv-resident 32768` | 113,5 / 89,6 | 1,64 / 2,51 | 8.027 MiB | 1,69 GiB |

Streaming lässt in beiden Kontexten und beiden Runden mehr VRAM frei: 574 MiB bei 128K und
1.366 MiB bei 256K. Bei 256K ist der Prefill in beiden Runden schneller. Bei 128K liegen die Arme
in der zweiten Runde nahe beieinander; aus der ersten Runde würde man fälschlich einen großen
Nachteil lesen. Das ist der Grund, warum hier keine einzelne Zahl als Ergebnis herausgegriffen
wird. Die Rohzeilen stehen in `perf-<kontext>-4k-<arm>.log` und `...-r2.log`.

`verify window ... pool` meldet 378,0 ms/Runde (128K streaming, Runde 2) gegen 424,6 ms/Runde
(voll resident) und 377,8 gegen 357,3 ms/Runde bei 256K. Eine Runde fasst bis zu sechs Draft-Token
zusammen; das ist nicht dasselbe wie eine einzelne Plain-Forward-Zeit. Die `decode`-Zeile ist die
Wanduhr über die erzeugten Token.

## Volle 128K-/256K-Prefills: resident gegen Streaming

Das sind echte Vollläufe, nicht nur 4K-Prompts mit reserviertem Kontext. Jeder der vier gezeigten
Läufe lief allein; gleiche Maschine, gleicher Build, gleicher Prompt, 6144-Token-Chunks.

| Kontext | Arm | Prompt-Token | Chunks | Prefill | tok/s | Time-to-first-token | 4 Decode-Token | Freies VRAM (4-Token-Fenster) | Pinned RAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K | voll resident | 131.060 | 22 | 210.768,3 ms | 621,8 | 213.081,8 ms | 1.916,0 ms (2,09 tok/s) | 7.752 MiB | — |
| 128K | `--kv-resident 32768` | 131.060 | 22 | 262.146,9 ms | 499,9 | 265.518,9 ms | 2.625,7 ms (1,52 tok/s) | 8.326 MiB | 0,84 GiB |
| 256K | voll resident | 262.116 | 43 | 393.012,9 ms | 666,9 | 395.269,0 ms | 1.862,8 ms (2,15 tok/s) | 6.661 MiB | — |
| 256K | `--kv-resident 32768` | 262.116 | 43 | 441.201,0 ms | 594,1 | 445.175,6 ms | 2.229,7 ms (1,79 tok/s) | 8.027 MiB | 1,69 GiB |

Gegenüber dem voll residenten Arm kostet das Streaming bei 128K **+24,4 % Prefill-Zeit**
(−19,6 % tok/s) und **−27,3 % Decode-Rate**; bei 256K sind es **+12,3 % Prefill-Zeit**
(−10,9 % tok/s) und **−16,7 % Decode-Rate**. Dafür bleiben 574 MiB (128K) beziehungsweise
1.366 MiB (256K) mehr VRAM frei, zu einem Preis von 0,84 beziehungsweise 1,69 GiB pinned RAM.
Das ist der gemessene Preis der Speicherersparnis auf dieser Karte, kein allgemeiner Prozentsatz.

Die Rohzeilen stehen in `perf-128k-fullprompt-full.log`, `perf-128k-fullprompt-stream.log`,
`perf-256k-fullprompt-full.log` und `perf-256k-fullprompt-stream-r2.log`. `verify window ... pool`
meldet resident 395,2/381,0 ms und streamend 491,6/410,0 ms (128K/256K). Ein erster 256K-Stream-
Versuch war nach einem Shell-Timeout kurz doppelt gelaufen und wurde verworfen; nur der saubere
`r2`-Lauf liegt hier und geht in die Zahlen ein.

Die meiste Prefill-Zeit kam in allen Läufen nicht aus der KV-Host-Kopie, sondern aus dem
Experten-Streaming aus den GGUF-Dateien: die Engine meldet 207,9/389,9 s resident gegen
258,3/436,2 s streamend (128K/256K) Host-Zeit für Experten. Die KV-Streaming-Kosten sind nicht
separat getaktet; die Tabellenwerte sind Gesamt-Prefills, kein isolierter KV-Vergleich.

## Was dieses Dokument nicht behauptet

* Keine neue Qualitätsmessung. Die bekannten KL-/Top-1-Werte für `q4u32w` wurden vollständig resident
  gemessen und gelten hier nur, weil Gate 1 und 2 die Bitgleichheit mit dem Streamed-Pfad zeigen.
* Kein Vergleich gegen `oct3` oder `q4lm8v` in diesem Lauf. Für `q4lm8v` bleibt die Kombination mit
  `--kv-resident` ausdrücklich abgelehnt.
* Kein Freigabe-Benchmark: die 4K-Prefills sind mit fixen 336 Experten-Slots, ohne MTP und auf einer
  nicht ruhigen Maschine gemessen. Auch die Vollläufe sind Eine-Maschine-Ergebnisse; die Tabelle oben
  zeigt resident gegen Streaming, aber keine zweimal wiederholte Volllauf-Messung pro Arm.
