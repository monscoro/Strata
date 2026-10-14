# Arbeitspaket: die KV-Cache-Quantisierung verbessern (Stand 2026-10-14)

Status: **Plan, kein Ergebnis.** Abgeleitet aus [KV_QUANT_RESEARCH.md](KV_QUANT_RESEARCH.md) (Survey, Stand der
Technik, Code-Deep-Dive #1 über quant.cpp). Wird noch verfeinert. Alle Wirkungsangaben sind **Erwartungen aus
fremden Messungen**, keine Strata-Ergebnisse. Branch `KV_QUANT_research`.

## 1. Ziel

Die Qualitätslücke des opt-in `--kv q4_0` verkleinern — mit einem **datenblinden, modell-agnostischen** Verfahren,
das mit **Qwen3.8-Flash-Next** (per-Head `q_norm`/`k_norm`, `head_dim` 256) funktioniert, ohne Kalibrierung und
ohne den Default-Pfad (`int8`) in irgendeiner Weise zu berühren.

## 2. Nicht-Ziele

- Kein TurboQuant-/SpectralQuant-Import: kein Kalibrierungs-Subsystem, keine Eigenbasis, keine Lern-Rotationen.
- Kein QJL-Residuum (in zwei unabhängigen Nachbauten ohne Effekt, auf K sogar schädlich — §5.2 des Surveys).
- Keine Änderung an Default, Byte-Zahl oder Byte-Gleichheit der bestehenden `int8`/`q4_0`/`k8v4`-Pfade.
- Keine neue Abhängigkeit.

## 3. Ausgangslage (gemessen, alles aus diesem Repo)

| Größe | Wert | Quelle |
| --- | --- | --- |
| **int8 gegen fp16** (neues fixes Set, 5×999) | **−0,10 % PPL**, KL 0,0136 | `bench/results/2026-10-14-kv-first-run/README.md` |
| **q4_0 gegen fp16** (neues fixes Set, 5×999) | **+1,12 % PPL**, KL 0,0364 (**2,7× int8**), Top-1 94,15 % | ebd. |
| int8 gegen fp16 (alter Doc-Probe) | ununterscheidbar (±2 Standardfehler) | `bench/results/2026-09-27-kv-q4/README.md` |
| q4_0 Document-PPL (alter Doc-Probe) | **+8 % @1K, +12 % @8K** | ebd. |
| q4_0 KL gegen fp16 (alt) | 0,039 (1K) / 0,297 (8K); 2,6-4,2× int8 | ebd. |
| q4_0 Top-1 (alt) | 91,3 % (Doc 1K), 88,0 % (8K) | ebd. |
| q4_0 Needles | 5/5 (1K-262K) | ebd. |
| q4_0 Decode @128K | 67,4 t/s, 0,83 GiB K/V (`--kv-resident 32768`) | ebd. |
| Layout | 18 B je 32 Werte, 144 B/Head, **4,5 bit/Wert** | `include/strata/kernels/kv_q4.hpp` |

Die beiden neuen Zeilen sind der **erste Meßpunkt auf dem festen Set** (§7.1) und der Maßstab für WP-1. Die alten
Doc-Probe-Zahlen („+8 % @1K") stammen von einem anderen, inzwischen entfernten Werkzeug auf einem einzelnen langen
Dokument und sind mit dem neuen Set **nicht** vergleichbar — der `docs`-Abschnitt zeigt hier bei ähnlicher KL nur
+0,09 % PPL.

Der Kern der Lücke: wir rotieren bereits (festes, orthonormales H) und legen dann das **uniforme** ggml-`q4_0`-Gitter
darüber, mit **einer Skala je 32 Werte**. Nach der Rotation sind die Werte aber näherungsweise gaußverteilt — ein
uniformer Gitter-Quantisierer mit 16 Stufen ist dafür nicht der passende (und der per-32-Skalierer re-normalisiert
jeden Block noch einmal, was den Verteilungsbezug zerstört).

## 4. Die Kandidaten (Wirkung gegen Aufwand)

| # | Kandidat | Belegte Wirkung | Aufwand | Datenblind? | Schon woanders? |
| --- | --- | --- | --- | --- | --- |
| **A** | `q4lm`: Rotation + **eine max-abs-Skala je Head** + 16-Level-Lloyd-Max-Codebook | klein (~1 % rel. in quant.cpp) | klein | **ja** | nein, nirgends |
| **B** | `--kv-window N` + `--kv-sink S`: die jüngsten/ersten Zellen exakt | **groß** (nimmt fast die ganze 4-bit-Strafe) | mittel bis groß | ja | offener PR **#2318** |
| **C** | SRHT: Zufallsvorzeichen auf H | unklar, theorie-nah | winzig | ja | in unserem Survey vorgeschlagen |

**Empfehlung:** **A zuerst.** Es ist der kleinste Eingriff, unser eigener Beitrag, und respektiert die Gate-Policy
(neues Format = opt-in, Default byte-identisch). **B direkt danach** — es hat den größeren Beleg, berührt aber
Selektion/Streaming/Snapshot und überschneidet sich mit dem offenen PR #2318 (dort abstimmen oder messen statt
doppeln). **C** als billiger Toggle, sobald A oder B steht.

## 5. Design

### 5.1 `--kv q4lm` (Kandidat A)

Neues, opt-in Format mit eigener Format-ID (`kKvQ4lm`). Die Rotation bleibt `fwht256` (orthonormal, selbstinvers)
für K, V und Q — damit bleiben die Scores exakt `<Hq, Hk> = <q, k>`, und der Ausgang wird mit demselben H
zurückgedreht.

Block = **ein Head-Vektor (256 Werte)** statt 8 × 32:

| Feld | Bytes | Inhalt |
| --- | --- | --- |
| `s` | 2 | fp16-Skala `s = ‖x‖ · amax(H·x̂) / C16`, `C16` = äußerster Codebook-Wert (≈ 2,7326) |
| `qs` | 128 | 256 4-bit-Indizes, LSB-first |
| **Σ** | **130** | **4,06 bit/Wert** (statt 144 B / 4,5 bit) |

Quantisieren: `norm = ‖x‖`, `x̂ = x/norm`, `y = H·x̂`, `amax = max|y|`, `s = norm·amax/C16`,
`idx[j] = argmin_k |y[j]·(C16/amax) − cb[k]|`.
Dequantisieren: `value = cb[idx] · s` (die Norm ist in `s` eingefaltet, ein einziger Skalenwert je Head).

Codebook: 16 feste Konstanten (Max-Lloyd für N(0,1), dieselben, die quant.cpp benutzt), als `constexpr`/`__constant__`
Tabelle. Der Aufmerksamkeitspfad dequantisiert genau wie heute bei `q4_0` (`KV_MODE 4`); nur die Abbildung
Nibble→Wert wird `(q − 8)·d` → `cb[q]·s`.

Byte-Bilanz je Zelle (2 KV-Heads, K+V): `q4lm` **520 B** gegen `q4_0` 576 B (−9,7 %) und int8 1.056 B.

Zwei Kontroll-Varianten, um Ursachen zu trennen (nur intern, kein Extra-Format):
- `q4lu`: wie `q4lm`, aber **uniformes** 16-Level-Gitter — isoliert den Skalenwechsel 32→Head;
- `q4lm32`: per-32-Block wie heute, nur Gitter→Codebook — isoliert den Codebook-Effekt.

### 5.2 `--kv-window N` / `--kv-sink S` (Kandidat B)

Auf `q4_0` und `k8v4`: die jüngsten `N` Zellen (und `S` Sink-Zellen am Anfang) liegen **fp16** vor, der Rest
4-bit. Kontext-invariant: `N = 128` bleibt bei 8K, 32K und 128K gleich; der Zuschlag ist `128/Zellen` (bei 32K
0,4 %). Beleg: quant.cpp, 3.970 Tokens — 4-bit flach +3,1 %, 4-bit + 128-Token-Fenster −0,1 %, FP32-Referenz.

Umsetzung mit dem, was schon da ist: `kv_stream` hat Host-Kopie, Block-Residency, `page_table` und die Selektion.
Die Auswahl bildet jüngste/erste Zellen auf einen kleinen fp16-Pool ab; alles andere läuft unverändert. Genau der
Ansatz von **#2318** (dort `--kv-window`/`--kv-sink` für `q2k`) — vor Baubeginn dort abstimmen.

### 5.3 SRHT (Kandidat C)

`STRATA_KV_SRHT=1`: eine feste ±1-Vorzeichen-Permutation (aus festem Seed) vor und nach `fwht256`, eigenes Gate,
Default aus. Beachte: quant.cpp fährt die Rotation mit **konstantem** Seed, ist also selbst kein echtes SRHT
(§6.4 des Surveys).

## 6. Berührte Dateien und Backends (Landkarte am Beispiel `q4_0`)

- Kern/API: `include/strata/kernels/kv_q4.hpp`, `include/strata/kernels/kv_stream.hpp` (Enum),
  `include/strata/core/layer.hpp` (`qsa_kv_format`)
- **CUDA und HIP** (dieselbe `.cu`): `src/kernels/cuda/kv_q4.cu`, `kv_stream.cu`, `qsa_prompt_attn.cu`
  (`KV_MODE 4`), `qsa_decode_attn.cu`
- **SYCL**: `sycl/src/kernels/cuda/kv_q4.dp.cpp`, `kv_stream.dp.cpp`, `qsa_prompt_attn.dp.cpp`,
  `qsa_decode_attn.dp.cpp`
- Dispatch/Layout/Streaming/State: `src/core/layer.cpp` / `.hpp`, `src/core/conversation_snapshot.cpp`,
  `src/core/mtp.cpp`
- CLI/UI: `src/program/generate.cpp` (`--kv`-Liste und Usage), ggf. `serve/web/app.js`
- Tests: `src/kernels/kv_q4_parity.cpp`, `kv_stream_parity.cpp`, `kv_hybrid_parity.cpp`, `kv_steps_parity.cpp`
  (jeweils mit SYCL-Zwilling)
- **Metal**: nur Stub (`metal/src/engine/stubs.cpp`) — keine Arbeit

Regel aus [AGENTS.md](../AGENTS.md): **jede** von einer Datei berührte Backend-Art (CUDA, HIP, SYCL) bauen, bevor
Review.

## 7. Arbeitspakete

| WP | Inhalt | Deliverable | Aufwand |
| --- | --- | --- | --- |
| **WP-0** | KV-Qualitäts-Harness: **steht** — `tools/quant_quality.py --kv FMT` (§7.1). Offen: Baselines fp16/int8/q4_0/k8v4 bei 1K/8K/32K auf {Q2_0, IQ3_S} fixieren (Top-1, KL, PPL + `needle_bench.py` + Speed-A/B) | `bench/results/<datum>-kv-baseline/` | S |
| **WP-1** | `--kv q4lm` (+ interne Kontrollen `q4lu`, `q4lm32`); CUDA/HIP + SYCL; Parity-Tests | Format + Tests + Messbericht | M |
| **WP-2** | `--kv-window` / `--kv-sink` auf `q4_0` (und `k8v4`), mit #2318 abgestimmt | Format + Tests + Messbericht | M-L |
| **WP-3** | `STRATA_KV_SRHT=1` | Toggle + Messbericht | S |
| **WP-4** (später) | Delta-Modus für 3-bit; Per-Layer-Bit-Zuteilung nach Kurtosis | — | L |

Reihenfolge: **WP-0 → WP-1 → (WP-2 ‖ WP-3) → WP-4.** Jedes WP ist **ein** PR (AGENTS.md: eine Änderung je PR).

### 7.1 WP-0: der Harness steht, der erste Meßpunkt liegt vor

`tools/quant_quality.py` pinnte `--kv` fix auf `fp16` und warf `--kv`/`--kv-resident` aus der Config — damit war **kein**
KV-Format meßbar. Seit Commit `f275f3cf` ist `--kv FMT` ein Lauf-Argument (Default `fp16`, also unverändertes
Verhalten), und die Engine-Umgebung benutzt unter Windows `PATH` statt `LD_LIBRARY_PATH` — wie `serve/server.py`s
`child_env` (`94d31866`). Beides sind nur Werkzeug-Änderungen; der Default-Pfad der Engine bleibt unberührt.

Erster Meßpunkt (Gewichte fest, nur die K/V-Ablage variiert):

    python tools/quant_quality.py run --config strata-q2_0.json --kv fp16 --out kv-fp16.tsv
    python tools/quant_quality.py run --config strata-q2_0.json --kv int8 --out kv-int8.tsv
    python tools/quant_quality.py run --config strata-q2_0.json --kv q4_0 --out kv-q4_0.tsv
    python tools/quant_quality.py compare kv-fp16.tsv kv-int8.tsv kv-q4_0.tsv --segments

Referenz ist der **fp16-KV-Arm bei gleicher Config**, **nicht** Q8_0: sonst mischt der Vergleich Gewichts- und
KV-Fehler. `run` hat keine Segment-Auswahl (jeder Arm = 5×999 Positionen teacher-forced, ~2,5 min), das ist der Preis
für den festen, committeten Tokensatz.

**Ergebnis** (Q2_0, gfx1100, 0.1.44.1 HIP; 4.995 Positionen; `bench/results/2026-10-14-kv-first-run/README.md`):

| KV | Top-1 | mittlere KL | PPL | gegen fp16 |
| --- | ---: | ---: | ---: | ---: |
| int8 | 96,48 % | 0,0136 | 4,469 | **−0,10 %** |
| q4_0 | 94,15 % | 0,0364 | 4,524 | **+1,12 %** |

Das ist die Zahl, die WP-1 schlagen muß (q4_0-Lücke, hier +1,12 % PPL / KL 0,0364 = 2,7× int8). Achtung: die
September-Zahl „+8 % @1K“ (`2026-09-27-kv-q4`) stammt von einem anderen, seitentfernten Werkzeug auf einem einzelnen
langen Dokument und ist mit diesem Set **nicht** vergleichbar — deshalb jetzt zwei Zahlen statt einer.

Zweiter, **modellfreier** Meßpunkt: die KV-Parity-Tests (`kv_q4_parity`, `kv_q8_parity`, `kv_hybrid_parity`,
`kv_stream_parity`, `kv_steps_parity`) — GPU, synthetisch, in ctest. `kv_hybrid_parity` boundet die quantisierte
Attention **schon gegen die fp32-true** Attention; dort gehört die entsprechende Schranke für `q4lm`/`q4lu` hin.
Sie sind das Korrektheits-Gate, nicht der Qualitätsmaßstab.

## 8. Messplan und Gate

Für jedes WP:

1. **Identitäts-Gate.** Mit dem neuen Format **aus** muss die Ausgabe byte-identisch sein (Default `int8` und die
   bestehenden `q4_0`/`k8v4` unverändert). Ein neues opt-in-Format ist per Konstruktion default-neutral — das
   **zeigen**, nicht behaupten.
2. **Qualität.** `tools/quant_quality.py run` / `compare` — **gegen den fp16-KV-Arm bei gleicher Config** (für
   Gewichts-Quants wäre Q8_0 die Referenz; für KV wäre das eine Vermischung beider Fehler) — Top-1, mean/p99 KL,
   PPL auf dem festen Set `data/quant_quality/v1` (code, docs, english, multilingual, agent), je Format und Kontext
   (1K/8K/32K); `tools/needle_bench.py` für Recall (32K/128K/262K, Tiefen 10/50/90 %).
3. **Speed und Speicher.** Decode-A/B mit `--kv-resident`, wie in `bench/results/2026-09-27-kv-q4`.
4. **Akzeptanz.** Ein WP wird nur vorgeschlagen, wenn es die q4_0-Lücke bei **gleicher oder kleinerer** Byte-Zahl
   messbar schließt und die Speed-A/B nicht verliert. Sonst bleibt es opt-in/experimentell oder wird verworfen.

## 9. Risiken und offene Fragen

- **Transfer der Literatur.** quant.cpp ist Llama (`head_dim` 128, **kein** QK-Norm); wir sind `head_dim` 256 **mit**
  QK-Norm. Der Codebook-Gewinn (~1 % rel.) kann bei uns kleiner oder null sein — deshalb WP-0 vor WP-1 und `q4lu`
  als Kontrolle.
- **max-abs je Head** ist eine Heuristik (wie quant.cpp Variante B). Die echte Standardabweichung oder die
  Beta-Verteilung der rotierten Einheitsvektor-Koordinaten kann besser sein — in WP-1 mitmessen.
- **QK-Norm.** Wird K per Head normiert (Qwen3.8-Flash-Next tut das), ist `‖k‖` fast konstant; dann ist der eine
  Skalenwert je Head fast redundant und die gesparten Bits könnten eine 5-bit-Stufe tragen. Hypothese, messen.
- **B überschneidet sich mit #2318** (offen, `q2k`). Vorher im PR-Thread fragen, ob dort gemessen/beigetragen wird.
- **Streaming und STATE_HASH.** Eine neue Format-ID muss in den Snapshot-Hash und die Suffix-Reuse-Formatliste
  aufgenommen werden (`docs/DETAILS.md:1008`, PR #2351).
- **Gate-Modelle.** Die genaue Vierer-Liste des Identitäts-Gates ist hier noch offen — vor WP-0 festzurren.

## 10. Was wir bewusst nicht machen

- Kein QJL-Residuum.
- Kein Kalibrierungs-/Eigenbasis-Default (SpinQuant/OSCAR/OptR) — wenn überhaupt, nur opt-in in einem späteren WP.
- Kein 2-bit; 3-bit nur später mit Delta (in quant.cpp eine sauber dokumentierte Sackgasse).
