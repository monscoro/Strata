# 256K: trägt die Tag-Lesung am oberen Rand des Einsatzbereichs?

**Status: vorregistriert (2026-11-03), gemessen (2026-11-04).** Die Regeln unten standen vor dem
ersten Arm-Lauf; das Ergebnis steht darunter. Anschluss an `bench/results/2026-11-03-deploy-range/`
(32K, 64K) und `2026-11-03-tag-arm/` (4K, 128K).

## Warum

Der Einsatzpunkt der Maschine ist ein Kontextfenster von 32K bis 256K. 32K und 64K sind gemessen
(der Tag trägt nach der registrierten Regel an keinem der beiden), 128K ist gemessen und mit einer
zweiten Lesung derselben Passage konfundiert, 256K ist **ungemessen**. Am oberen Rand ist der
Hebel am kleinsten: der Ring hält 6.144 von 262.144 Zellen (2,3 % des Kontexts, gegen 4,9 % bei
128K und 18,8 % bei 32K). Wenn der Tag dort noch etwas bewirkt, dann trotz dieser Abdeckung; wenn
nicht, ist die Aussage „kein Hebel über 32K" vollständig.

## Aufbau

Derselbe teacher-forced Aufbau wie die Punkte 32K/64K/128K: ein Segment, ein `GEN 1`,
per-Position-Logits aus den Verify-Fenstern am Prompt-Ende, `--short-read 6144`, `--topk 256`,
gepaart je Position, Moving-Block-Bootstrap.

| Punkt | Wert |
|---|---|
| Passage | `p256k` = cwe-8k **viermal** (die acht Segmente in vier Kopien), 262.044 Token |
| Warum viermal | 262.144 Token sind mit 65.511 Token distinctem Text nicht ohne Wiederholung baubar; die Wiederholungsfrage klärt `bench/results/2026-11-03-repeat-control/` bei gleicher Länge, nicht dieser Punkt |
| Schnitt | `pin = 262.044 − 6.144 = 255.900` plus `--prompt-cache 3`: alles davor batched, die letzten 6.144 Token durch die Fenster |
| Kontext | `--max-context 262144` (neuer Werkzeug-Flag; die GGUF nennt `qwen4exp.context_length = 262144`, Prompt 262.044 + 1 Token passt hinein) |
| Fenster | `STRATA_KV_WINDOW=1 S=256 R=64`, kein `BATCH` (der Ring folgt dem Chunk: 6.144 Zellen) |
| Prefill | `--prefill auto` |
| Positionen | 6.143 je Arm (der Dokument-Tail), `--allow-no-positions` |
| Arme | `int8` (**Referenz**), `win` (`q4lm32`, 256/64), `tag` (`q4lm32`, 256/64 + `STRATA_KV_WINDOW_TAG=1`) |

### Warum `int8` und nicht `fp16` die Referenz ist

Ein Start mit `--max-context 262656 --kv fp16` lädt (Probe 2026-11-03, `READY`) und lässt
**561 MiB** VRAM frei — unter der 1.024-MiB-Warnschwelle des Motors („when the card is this full
the driver can page GPU memory to system RAM"). Bei 128K ließ `fp16` noch 2.383 MiB frei. `fp16`
ist damit als 256K-Referenz nicht vertretbar; die KL wird gegen `int8` genommen. Die gepaarte
Differenz `tag − win` bleibt gültig, weil beide Arme dieselbe Referenzverteilung sehen; die
**absoluten** KL-Werte sind mit den fp16-referenzierten Punkten nicht vergleichbar.

Ein GPU-Job zur Zeit, 30 s zwischen zwei Engine-Starts, Engine-SHA `CD5700EC0793F83B…3D0394CA`
vor dem Lauf geprüft.

## Entscheidungsregeln (vor dem Lauf festgelegt)

Δ = KL(int8‖Arm), negativ = der Arm ist besser. Gepaart je Position, Moving-Block-Bootstrap über
Blöcke von 256 aufeinanderfolgenden Positionen (24 Blöcke, 2.000 Resamples, Seed 20261103).

- **`Q1` — trägt der Tag bei 256K?** Gepaart `tag − win`. **„Tag trägt"** nur wenn das
  Bootstrap-95-%-Intervall die 0 ausschließt, der Mittelwert negativ ist **und** |Mittelwert| >
  0,0013 nats — dieselbe Schwelle wie `D1`/`T3b`, unverändert.
- **`Q2` — ist die Lücke zu `int8` geschlossen?** `KL(int8‖tag)` ist mit dieser Referenz direkt
  der Abstand. **„geschlossen"** wenn der Mittelwert ≤ 0,001 nats ist **und** die einseitige
  95-%-Bootstrap-Obergrenze unter +0,001 nats liegt. `KL(int8‖win)` wird daneben berichtet.
- **`Q3` — ist der Arm informativ?** `KL(int8‖win)` > 0 mit Bootstrap-95 % ohne 0. Andernfalls hat
  der Ring den Prompt gehalten und `Q1` ist **ungültig**.
- **`Q4` — ein Punkt, kein Verlauf.** 256K wird einzeln berichtet; keine gepoolte Zahl mit
  32K/64K/128K und keine Aussage „der Effekt fällt mit der Länge".

### Güte-Gatter (vor der Auswertung, je Arm)

- `DONE` ohne „nonfinite"; kein „do not fit"; kein `ERR`; Engine-SHA unverändert.
- `--max-context 262144` in der Engine-Zeile, Ring-Zeile mit 6.144 Zellen, `tag` zusätzlich mit
  `STRATA_KV_WINDOW_TAG=1`.
- TSV mit **6.143** Positionen; in allen drei Armen **dieselbe** Positionsmenge; `seg.json` mit
  `pin` 255.900 und `prompt_cache` = 3.

### Grenzen, offen genannt

1. **Keine fp16-Referenz** (siehe oben). Die Aussage ist „Abstand zu `int8`", nicht „Abstand zu
   fp16".
2. **Vierfache Wiederholung.** Der Tail ist die vierte Lesung desselben Textes (Abstände 65,5K,
   131K, 196,5K). Ob Wiederholung die KL senkt, misst `2026-11-03-repeat-control/` bei 64K; dieser
   Punkt kann von einer solchen Senkung nur profitieren oder sie nicht zeigen.
3. **Ring = 2,3 % des Kontexts.** Der Tag kann nur die 6.144 gelesenen Zellen betreffen; bei
   262.144 Zellen ist das der kleinste Anteil aller Punkte.
4. **Kein Timing, kein Byte-A/B.**

### Kosten

Drei Läufe, je ~262K: ~1.950 s Prefill (bei 134 tok/s) + 6.144 Fenster-Schritte, nach der
128K-Erfahrung (1.193 s) zusammen etwa 1,8 Stunden GPU, kein Engine-Bau.

---

# Ergebnis

**Status: gemessen (2026-11-04).** Alle Güte-Gatter bestanden: je Arm **6.143** Positionen, in
allen drei Armen dieselbe Positionsmenge, `win`/`tag` mit der Ring-Zeile, nur `tag` mit der
`STRATA_KV_WINDOW_TAG=1`-Zeile, kein „nonfinite", kein „do not fit", kein `ERR`, Engine-SHA
unverändert. Alle drei Engine-Zeilen tragen `--max-context 262144`, `--short-read 6144` und
`--prompt-cache 3`; `seg.json` meldet `pin` 255.900. Die vier Kopien der Passage sind Token für
Token identisch (262.044 Token, geprüft).

## Verteilung

`KL(int8‖Arm)` je Position, 6.143 Positionen pro Arm:

| Arm | Mittel | Median | p90 | p99 | max | > 0,01 nat | > 0,1 nat |
|---|---|---|---|---|---|---|---|
| `win` | 0,000405 | 0,00000064 | 0,000006 | 0,000128 | 0,9210 | 10 | 5 |
| `tag` | 0,000103 | 0,00000073 | 0,000006 | 0,000078 | 0,3062 | 7 | 2 |

`KL(int8‖win)` ist an **allen** 6.143 Positionen größer als 0, `KL(int8‖tag)` ebenso: der Ring
liest überall etwas anderes als `int8`, nur sehr wenig. Keine Position liegt über 1 nat.

## `Q3` — ist der Arm informativ? **Ja, knapp.**

`KL(int8‖win)` = **0,000405**, Block-95 % [+0,0000023; +0,001210]. Das Intervall schließt die 0
aus, aber die untere Grenze liegt nur 2,3·10⁻⁶ über ihr. Die Klausel ist damit erfüllt — der Ring
hat den Prompt nicht einfach gehalten —, und `Q1` ist gültig; viel Spielraum lässt der Punkt aber
nicht.

## `Q1` — trägt der Tag bei 256K? **Nein.**

Gepaart `tag − win`, negativ = Tag besser:

| `tag − win` | Block-95 % | einseitige Grenze | besser / schlechter |
|---|---|---|---|
| **−0,000302** | [−0,000905; +0,0000001] | +0,00000006 | 2.886 / 3.257 |

Der Mittelwert zeigt zum Tag, aber zwei der drei Klauseln scheitern: das Intervall schließt die 0
nicht aus (die Obergrenze liegt 1·10⁻⁷ über ihr), und 0,0003 nats liegt weit unter der
0,0013-Schwelle. Der Median ist praktisch null (+0,00000001). Die Regel „Tag trägt" ist **nicht**
erfüllt.

## `Q2` — ist die Lücke zu `int8` geschlossen? **Ja.**

| Arm | `KL(int8‖Arm)` | Block-95 % | einseitige Grenze | Regel |
|---|---|---|---|---|
| `tag` | **0,000103** | [+0,0000023; +0,0003048] | +0,0003046 | **geschlossen** (Mittel ≤ 0,001, Obergrenze < +0,001) |
| `win` | 0,000405 | [+0,0000023; +0,0012098] | +0,0012096 | — |

Beide Klauseln halten deutlich: der Mittelwert liegt bei einem Zehntel der Schwelle, die
einseitige Obergrenze bei einem Drittel. Der Tag ist an diesem Punkt sogar näher an `int8` als das
Fenster — aber der gepaarte Unterschied (0,0003 nats) ist zu klein, um nach `Q1` etwas zu tragen.

## Beschreibend, nicht registriert

Es gibt am 256K-Punkt **keine** Position über 1 nat; die post-hoc-Trimmung, die an den anderen
Punkten zwischen 6 und 220 Offsets entfernt, ist hier leer und ändert keine Zahl. Der Grund ist derselbe wie
in der Wiederholungs-Kontrolle: der Tail ist die vierte Lesung desselben Textes, und Wiederholung
senkt den Fehler.

## Was das heißt

- **Registriert:** Der Tag trägt bei 256K nicht (`Q1`), die Lücke zu `int8` ist geschlossen
  (`Q2`), der Arm ist informativ (`Q3`). Damit ist die Aussage „kein Hebel über 32K" für alle
  Punkte des Einsatzbereichs belegt: 32K, 64K, 128K und 256K.
- **Der Punkt ist der am stärksten wiederholte.** Der Ring hält 6.144 von 262.144 Zellen
  (2,3 %), und `KL(int8‖win)` liegt bei 0,0004 nats — der niedrigste Absolutwert der Reihe,
  allerdings gegen `int8` statt `fp16` gemessen und damit nicht direkt mit den anderen Punkten
  vergleichbar (siehe Aufbau). Der Vergleich, der zählt, ist der gepaarte `tag − win`, und der
  ist null.
- **Wiederholung, nicht Länge.** Der Tail ist die vierte Lesung; die Wiederholungs-Kontrolle
  (`bench/results/2026-11-03-repeat-control/`) zeigt bei gleicher Länge, dass die zweite Lesung
  die KL um 7,6-mal (Mittel) senkt und den Tag-Effekt umdreht. `Q4` verbietet, daraus einen
  Verlauf über die Länge zu machen; es bleibt: an diesem Punkt hat der Tag nichts zu geben.
- **Für das Produkt:** kein Auslieferungs-Default, kein 5,0-bit/Zelle-Hebel; `B1` bleibt
  geschlossen.

## Abweichungen vom registrierten Plan

1. **6.143 statt 6.144 Positionen** — dieselbe Pin-Arithmetik wie an allen anderen Punkten
   (`pin = n − 6144` ergibt `n − 1 − pin = 6.143`); alle drei Arme teilen dieselben Positionen,
   keine Regel hängt an der Zahl.
2. **Keine fp16-Referenz** — wie vorregistriert; `int8` ist die Referenz, die absoluten KLs sind
   mit den fp16-Punkten nicht vergleichbar.
3. **Die post-hoc-Trimmung ist leer** (keine Position über 1 nat); sie wird berichtet, entscheidet
   aber nichts.
4. **Kein Timing, kein Byte-A/B.**

## Grenzen, offen genannt

1. **Keine fp16-Referenz** (siehe Aufbau). Die Aussage ist „Abstand zu `int8`".
2. **Vierfache Wiederholung.** Der Tail ist die vierte Lesung desselben Textes (Abstände 65,5K,
   131K, 196,5K). Der Punkt kann von der Wiederholungs-Senkung nur profitieren (die KL fällt),
   und er trennt Länge und Wiederholung nicht.
3. **Ring = 2,3 % des Kontexts**, der kleinste Anteil aller Punkte.
4. **Ein Segment**, keine Segment-Clusterung; die 6.143 Positionen sind geschachtelt.
5. **`Q3` ist knapp** — die untere Intervallgrenze liegt 2,3·10⁻⁶ über 0. Der Punkt ist
   informativ, aber nur eben.

## Reproduktion

```powershell
python tools\quant_quality.py run --config bench\configs\cwe4k-swift-iq3_xxs.json `
    --ids C:\Users\krone\git\Strata-data\bench\2026-11-03-256k\p256k.ids.json `
    --kv <int8|q4lm32> --prefill auto --short-read 6144 --topk 256 --allow-no-positions `
    --pin 255900 --prompt-cache 3 --max-context 262144 `
    [--env STRATA_KV_WINDOW=1 --env STRATA_KV_WINDOW_S=256 --env STRATA_KV_WINDOW_R=64 `
     [--env STRATA_KV_WINDOW_TAG=1]] `
    --out C:\Users\krone\git\Strata-data\bench\2026-11-03-256k\<arm>.tsv
```

Rohdaten liegen außerhalb des Repos unter
`C:\Users\krone\git\Strata-data\bench\2026-11-03-256k\`; im Repo steht dieses Dokument und nach dem
Lauf der Ergebnisabschnitt darunter.
