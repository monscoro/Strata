# Peer Review: "Hardening the Exact-Cell KV Window for `--kv q4lm8v`"

**Role:** anonymous reviewer. **Recommendation:** Major revision.

**Verfasser / Provenance:** Review und nachfolgende Revision wurden von
**GLM-5.3** erzeugt (opencode agent, Model `glm-5.3-flash-free`, Sitzung vom
2026-10-24). Das ist ein **Selbst-Review im selben Modellkreis**, kein
unabhängiges Fremd-Gutachten: biases sind systematisch möglich (der Reviewer
kennen die Autorensicht, Engine-Details und die Repo-Dokumentation). Alle
externen Zahlen sind belegte Quellen; alle eigenen Empfehlungen, die auf
statischer Code-/Textanalyse beruhen, sind so markiert. Eine unabhängige
menschliche oder fremdmodellige Gegenprüfung bleibt offen und ist ausdrücklich
erwünscht.

This review reads the draft as a systems/efficiency paper. External numbers
cited below carry their sources; points that are my reading of the manuscript
rather than a measurement are marked as such.

---

## 1. Summary of the paper

The paper presents an opt-in KV-cache point ("exact-cell window") for Strata's
4-bit Lloyd K/V cache: the S=256 most recent cells and R=64 sink cells per
attention layer are kept in fp16 while the rest of the cache stays 4-bit,
yielding 5.29 bit/cell (1,016 MiB at 128K) versus the int8 default's 8.25
bit/cell (1,584 MiB). It documents five integration-boundary hardening
decisions (A1–A5), a teacher-forced evaluation methodology (mean KL, p99 KL,
top-1, PPL, paired per-segment statistics), and results at 2K/4K/128K on one
GPU/OS/software stack. Headline findings: the window closes 73% of the
plain-4-bit→int8 KL gap at 4K and 58% of the plain arm's mean KL at the 128K
tail, but does not overtake int8 anywhere; the int8 default stays
byte-identical.

## 2. Overall assessment

The paper is unusually honest for a systems write-up: results that favor the
competitor (int8) are reported prominently, the PPL trap (quantized arm scoring
*below* the fp16 reference while KL is positive) is called out explicitly, and
paired statistics with sign counts are a better habit than most engineering
reports show. The hardening table (problem → decision → evidence) is a good
pattern.

The two blocking weaknesses are (i) **no positioning against prior and current
work at all** — the paper cites nothing, and its core mechanism (recent-token
fp16 residual + retained attention sinks on top of a low-bit quantized base)
has direct precedents the community knows well; and (ii) a **single-task,
single-model evaluation** that cannot support the generality of the quality
claims ("at the 8-bit level in the answer region", "across lengths tested")
as currently phrased. Both are fixable without new GPU time (revision plan in
§6); one additional measured baseline would materially strengthen §5.

---

## 3. Strengths

1. **Decision-metric discipline.** KL as the decision metric with PPL flagged
   as context-only is methodologically correct, and the damping effect
   (quantized arm PPL < fp16 PPL with positive KL) is explained rather than
   hidden. Most runtime papers report PPL only.
2. **Paired statistics with sign counts.** Same-text-pairing, per-segment
   differences, t with 7 d.o.f., and the explicit warning that a large t with
   4/8 segments is not a result — good practice. Stating that the 128K blocks
   are adjacent and non-independent, and demoting those t values to
   descriptive, is honest.
3. **Byte-identical default gate.** The SHA-identity gate for the int8 path
   across every change is exactly the right engineering invariant for an
   opt-in feature, and it is checked 8/8.
4. **Silent-failure framing.** "A silent misread is worse than a loud refusal"
   is a defensible design principle, and A2's two-layer enforcement (setup +
   engine) is the correct belt-and-suspenders treatment for a config boundary.
5. **Honest runtime reporting.** Prefill parity and decode deltas inside
   run-to-run spread are reported with the caveat that signs changed across
   rounds, rather than claimed as wins.

## 4. Major concerns

**M1 — No related work; the window is not positioned (blocking).**
The mechanism — keep a short fp16 residual of recent tokens and a set of
sink tokens, quantize the rest — is a combination of two well-known results:

- *StreamingLLM* (Xiao et al., arXiv:2309.17453) established that the initial
  tokens act as attention sinks and must be retained; evicting them causes
  large perplexity spikes.
- *KIVI* (Liu et al., ICML 2024, arXiv:2402.02750) quantizes K per-channel at
  2 bits and V at 2 bits while keeping a fixed number of recent tokens in
  fp16 (`residual_length`); *KVQuant* (Hooper et al., NeurIPS 2024,
  arXiv:2401.18079) adds per-channel key quantization, pre-RoPE key
  quantization, and non-uniform datatypes to make sub-4-bit K work.

The draft's window is *Strata's point in this design space* — per-cell
post-rotation Lloyd K(4)/V(8) with an fp16 recent+sink overlay — and that is
fine, but a reviewer cannot judge the contribution without the map. The
Hadamard rotation already shipped in Strata's base format (`q4_0` after an
orthonormal Walsh–Hadamard rotation) belongs to the same family as the
rotations used by QuaRot (arXiv:2404.00456) and RotateKV (IJCAI 2025), which
should be named as the reason the per-cell (per-token) layout is viable at all.

**Required fix:** add a Related Work section (~0.75 column; draft text below)
and cite StreamingLLM, KIVI, KVQuant, GEAR (arXiv:2403.05527), SnapKV
(arXiv:2404.14469) / PyramidKV (arXiv:2406.02069) for the eviction side, and
KVTuner (ICML 2025, arXiv:2502.04420) for layer-wise mixed precision. Position
the window explicitly as: *quantization-side* (bits per cell) rather than
*eviction-side* (which cells survive) — the paper's current "buys memory"
claim is about coverage of the full context, which eviction methods give up.

**M2 — Single task family; quality claims outrun the evidence.**
All quality numbers come from one RULER "complex word extraction" task at
three lengths (different texts, correctly not pooled). The abstract says
"the answer region [is] at the 8-bit level" and "statistically
indistinguishable at 2K" — both statements are true for that one task on one
model, but a reader will take them as general. The community standard for
long-context KV-compression claims is a *suite*: RULER in full (especially
niah_multikey/multivalue and variable tracking, which stress different
attention patterns than cwe), LongBench subsets, or at minimum a second task
family. The in-repo needle tests exist (32K/128K, 6/6 per arm) but are
declared saturated in §5.5 themselves — a saturated test has no discriminating
power, so it neither supports nor refutes quality claims.

**Required fix (no GPU time needed):** scope every quality sentence to cwe
("on this task", "on the cwe corpora"), and add the multi-task run to Future
Work as an explicit gate. If any GPU time is available, one extra RULER task
family at 4K would be the highest-value single measurement.

**M3 — No community-baseline arm; the interesting comparison is missing.**
The paper compares Strata arms against each other (int8 / plain 4-bit /
window) but never lands the 5.29-bit point on the published trade-off curve.
Published points in the same regime:

- KVTuner reports "nearly lossless" 3.25-bit mixed-precision KV per-layer for
  Llama-3.1-8B and 4.0-bit for Qwen2.5-7B (their setting, not Strata's).
- KIVI runs 2-bit K / 2-bit V per-channel plus fp16 residual.
- vLLM's FP8 KV-cache + FP8-attention production path reports ≤1–2 accuracy
  points degradation across reasoning benchmarks and sustained long-context
  quality up to 1M-token prompts (vLLM blog, 2026-04-22). This is important
  context for the *default*: an 8-bit KV default is the conservative choice
  in 2026, and the paper's int8-vs-window motivation ("KV memory is the
  constraint") should acknowledge that halving KV with FP8 is already commodity
  on NVIDIA Hopper/Blackwell — Strata's audience (one consumer GPU, Windows
  HIP/AMD) is precisely where FP8 tensor-core paths do *not* exist, which
  actually strengthens the paper's motivation but is never stated.

**Required fix:** add one paragraph to the Introduction or Related Work that
(a) states FP8-KV as the comparable production baseline on datacenter
hardware, (b) explains that on the paper's target hardware (AMD HIP, consumer
VRAM) the block-quantized int8 path is the deployed 2× point and 4-bit is the
frontier, and (c) states which community arm would be the fair competitor
(per-channel K + fp16 residual, KIVI-style) and why it was or was not measured
on this engine. If the harness can score an arm with per-channel K quantized
instead of per-cell, that single extra arm in Table 4/5 would answer the
"how does per-cell compare to the community layout" question with data.

**M4 — KL direction and asymmetry are under-specified.**
§4.3 says "divergence from the fp16 distribution to the arm's distribution".
KL is not symmetric; D(fp16‖arm) ≈ E_fp16[log fp16 − log arm] heavily penalizes
the *arm* placing small mass where fp16 has mass, which is exactly a
"rows-to-columns" bulk error. The observed damping (the arm's working
distribution is sharper; PPL falls below the reference) is a signature that
would read differently under D(arm‖fp16). State the argument order with the
standard notation D_KL(P‖Q), say why the chosen direction fits the use case
(distribution-matching for a downstream sampler), and note in one sentence
how the ranking could change under the other direction. [Static reading of
the manuscript; no measurement.]

**M5 — Answer-region claims rest on n=27 positions.**
Table 7's answer region has 27 positions; the paper properly flags the
96.30-vs-92.59 top-1 difference as one position, but elsewhere §5.2/§5.3 lean
on "indistinguishable in the answer region" (paired t=−0.51) and "the better
answer region (0.00526 against 0.00661)" as if the region were a robust
aggregate. With n=27, no percentile is meaningful. Either drop the answer-
region *comparisons* from the argument chain (keep the numbers as descriptive
with the n=27 caveat attached to every mention), or define a pooled answer
region across the eight segments at 4K where the region has real mass.

**M6 — Runtime statistics are thinner than the quality statistics.**
Table 3 medians come from two rounds. Decode "−1.0 token/s, changing signs"
is fine as a non-claim, but prefill "equal (526.1 vs 528.0)" and the 512/128
prefill "634.0" (a 20% faster value that §5.1 then says did not reproduce on
the needle prompts) leave the reader unsure which prefill numbers are
trustworthy. State explicitly: how many rounds, what the round-to-round spread
was, and mark the 512/128 prefill value as not reproduced *in the table
caption*, not only in the prose. The mixed-prompt set is described nowhere in
the text (composition, lengths, hit-rate definition) — §4.5 should define it,
or the table caption must.

**M7 — "14% more expert slots" and "2-point higher hit rate" are unearned.**
These system benefits are reported but never cashed out: what does +416
expert slots at 128K actually buy in throughput at a given concurrency, or in
maximum context per user? Either (a) add a two-line serving measurement
(two concurrent streams, tokens/s) or (b) keep the numbers but phrase them as
*capacity* outcomes ("frees 568 MiB at 128K, ~416 additional cached expert
slots") without implying a throughput win that was not measured. [Reviewer
reading; the engine runs single-stream in the reported runtime table.]

**M8 — Reproducibility statement is missing.**
No code/data availability, no harness description pointer, no statement of how
cwe texts were generated (are they public RULER instances? synthetic
templates? seeds?). For a methodology-first paper this should be §4.0 or a
final paragraph. The repo-documented measurement history
(`docs/KV_QUANT_WORKPACKAGE.md`) shows the numbers have provenance — say so
in the paper.

## 5. Minor points

1. **A-series numbering.** The table lists A1, A2, A3, A5; A4 appears only in
   the caption and §3's prose. A reader first scanning the table cannot tell
   whether A4 was skipped by accident. Either renumber 1–4 with the parity
   test as decision #4, or add an explicit one-line row "A4 — open item".
2. **Terminology.** "Cell" (per-token, per-layer K or V entry), "QSA layers"
   (12 of them), "row" (ring row vs cell row) are used before definition.
   One definitions sentence after the first use in §2.1 would remove three
   reader reversals. Define GQA/K/V sharing if applicable.
3. **Equation (1).** The formula is a context-average ("bits/cell ≈ 4.50 +
   16(R+S+B)/L") but B is transient (only held during prefill of the current
   chunk). Say "over a full prefill+decode pass at fixed L" and note the
   B-term is amortized, otherwise a careful reader will object that 16·(R+S+B)/L
   with B=6,144 dominates and is not really a *ring* cost.
4. **Footprint table sourcing.** Table 1's plain-arm 864 MiB is byte-model,
   the int8 row is runtime-observed; the caption says so, but the run-time
   table (Table 3) silently mixes a measured 1,016 with a modeled 884/864.
   One shared convention ("programmed bytes" vs "measured RSS") selected up
   front would prevent cross-table confusion.
5. **p99 without uncertainty.** Percentile-of-KL over ~16–131k positions has
   tight variance at the upper body but the p99 estimate itself has no CI.
   A one-line note (order-statistic variance, ~k/(k) relative for n≫100) or a
   bootstrap range on the headline p99 deltas would close the biggest
   remaining statistical hole cheaply. Compute offline, no GPU time.
6. **"8/8 runs".** Byte-identity across 8 runs is a great gate, but "8" is
   arbitrary in-text; say which (one per changed build? per scenario?). One
   parenthesis.
7. **Abstract length/numbers.** The abstract carries six t-statistics; four
   is the readability ceiling. Move the 128K or 2K numbers to §5 and keep
   the abstract to the two gap-closing numbers plus the memory number.
8. **`int8` naming.** The default is written `int8` but costs 8.25 bit/cell
   (block-scale overhead). One footnote ("int8 = 8-bit content plus block
   scales, 8.25 effective") prevents the recurring reader question "why 8.25?".
9. **Figures.** Zero figures. A single anatomy figure (cache layout: quantized
   bulk, fp16 ring, sinks, streamed region marked incompatible) would carry
   §2.1–2.2 better than the equation and reduce its text by ~1/4 column.
10. **"Buys memory, not accuracy".** Phrasing lands as if the window were a
    worse trade than it is; "a 36% memory saving with a measurable KL cost
    that vanishes at short context" is more precise and still conservative.

## 6. Concrete revision plan

Priority-ordered; items marked *(text)* need no new measurements.

| # | Change | Where | Cost |
|---|--------|-------|------|
| 1 | Add Related Work + 6–9 citations (draft below) | new §2 opener | text |
| 2 | Scope every quality claim to the cwe task | abstract, §5, conclusion | text |
| 3 | FP8-KV production context paragraph (Hopper/Blackwell vs HIP/consumer) | §1 ¶1 | text |
| 4 | Fix KL direction with D_KL(P‖Q) notation + one justification sentence | §4.3 | text |
| 5 | Answer-region n=27 caveat attached at every mention; drop from argument chain | §5.2, §5.3 | text |
| 6 | A4 as visible open row; renumber or annotate | Table 2 | text |
| 7 | Runtime table: rounds, spread, served-set definition, "not reproduced" moved into caption | Table 3, §4.5 | text |
| 8 | Capacity sentence ("frees 568 MiB ≈ 416 expert slots") without implied throughput | §5.1 | text |
| 9 | One anatomy figure of the window layout | §2 | small |
| 10 | Reproducibility/provenance paragraph (harness, data, repo docs) | new end of §4 or §7 | text |
| 11 | (if any GPU) one KIVI-style per-channel-K arm at 4K into Tables 4/5 | §5.2 | measurement |
| 12 | (if any GPU) one second RULER task family at 4K | §5.2 | measurement |

### Draft Related Work text (for the authors to adapt; claims are the cited papers')

> *KV-cache compression splits into eviction and quantization. Eviction methods
> keep a subset of cells: H2O tracks cumulative attention as heavy hitters;
> SnapKV selects per-head clusters from a prefill observation window; PyramidKV
> allocates layer-dependent budgets, keeping more cells in lower layers; SkiR /
> ChunkKV score whole chunks. Quantization methods reduce the bits per kept
> cell: KIVI shows 2-bit per-channel K and V with a fixed fp16 residual of
> recent tokens; KVQuant adds per-channel keys, pre-RoPE key quantization and
> non-uniform datatypes to reach <0.1 PPL degradation at 3 bits; GEAR adds
> low-rank plus sparse-outlier error correction; RotateKV and QuaRot use
> rotations to handle channel outliers. KVTuner searches layer-wise precision
> pairs offline and reports nearly lossless 3.25–4.0-bit mixed precision on
> reasoning tasks. In production, vLLM ships an FP8 KV-cache path with ≤1–2
> accuracy points on reasoning benchmarks on Hopper- and Blackwell-class
> hardware. The present window sits in the quantization family: it keeps the
> whole context addressable (no eviction), quantizes cells per token after a
> Hadamard rotation, and spends fp16 bytes on a small recent-plus-sink
> overlay — the same recent-residual and sink idea as StreamingLLM and KIVI,
> applied on top of a 4-bit K / 8-bit V Lloyd base.*

### Reference list (verified identifiers)

- Xiao et al., *Efficient Streaming Language Models with Attention Sinks*,
  arXiv:2309.17453 (2023; ICLR 2024).
- Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*,
  ICML 2024, arXiv:2402.02750.
- Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference
  with KV Cache Quantization*, NeurIPS 2024, arXiv:2401.18079.
- Kang et al., *GEAR: An Efficient KV Cache Compression Recipe for
  Near-Lossless Generative LLM Serving*, arXiv:2403.05527.
- Li et al., *SnapKV: LLM Knows What You Are Looking for Before
  Generation*, NeurIPS 2024, arXiv:2404.14469.
- Cai et al., *PyramidKV: Dynamic KV Cache Compression based on Pyramidal
  Information Funneling*, COLM 2025, arXiv:2406.02069.
- Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference
  of Large Language Models*, NeurIPS 2023, arXiv:2306.14048.
- Ashkbof et al., *RotateKV*, IJCAI 2025 (proceedings 2025/0690).
  *(Relevance: rotation-based K outlier handling; directly comparable to
  Strata's Hadamard pre-rotation.)*
- Li et al., *KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache
  Quantization*, ICML 2025, arXiv:2502.04420.
- *QuaRot: Outlier-Free 4-Bit Inference via Rotations*, arXiv:2404.00456.
- Feng et al./Tao et al., *More Tokens, Lower Precision: Towards the Optimal
  Token-Precision Trade-off in KV Cache Compression*, arXiv:2412.12706
  (2024/25). *(Layer- and token-precision co-tuning; useful to argue the
  4/8 K/V split.)*
- MiVoLM/vLLM team, *The State of FP8 KV-Cache and Attention Quantization in
  vLLM*, vLLM blog, 2026-04-22 (accessed 2026-10).
- (in-repo provenance) `docs/DETAILS.md` (KV quantization section),
  `docs/QUANTS.md` (yardstick KL table), `bench/results/2026-09-27-kv-q4`
  (document-perplexity 8–12% above 8-bit at 1K/8K for the earlier plain
  `q4_0` KV format — the source for the abstract's 8–12% claim).

## 7. Questions for the authors (to answer in the reply letter)

1. Was the plain-q4_0-for-K alternative (i.e., 4-bit for **both** K and V,
   symmetric) also measured, and how does the K4/V8 split compare? The KIVI
   result that K is more sensitive than V predicts K4/V8 as the right split;
   Strata's own numbers would either confirm or contradict that per-model.
2. Is the sink count R=64 a measured knee (the workpackage notes mention a
   knee near S≈512 for a different arm), or a carried default? §7.13-style
   knee data for the 4-bit base would justify 256/64 against the alternatives
   already in Table 3.
3. What does the window do for **multi-turn serving** (a second prompt appended
   to a 100K context)? The runtime A/B used a mixed prompt set; the quality
   runs are single-segment. Chat-style incremental growth is the window's home
   workload and is currently unmeasured — worth one line stating this.
4. Blocks at 128K: did you consider scoring the *middle* of the segment (e.g.
   positions 60–64K) as well as the tail? Tail-only scoring cannot detect
   mid-context degradation, which is where per-cell quantization historically
   hurts (needle depth-dependence).

---

## 8. Abarbeitungsstand (Stand 2026-10-24, nach Revision durch GLM-5.3)

Die Revision wurde direkt im Artikel `docs/paper/KV-Window-Hardening.tex`
vorgenommen und mit pdflatex kompiliert geprüft.

### Revisionsplan, Punkte 1–10

| Nr. | Original-Punkt | Status | Wo |
|-----|----------------|--------|----|
| 1 | Related Work + Zitate | **Umgesetzt.** Neuer Abschnitt §2 mit 17 Einträgen (arXiv-IDs verifiziert), Survey-Tabelle dokumentierter Betriebspunkte, Code-/Doku-Links in den Referenzen | Paper §2 |
| 2 | Qualitätsaussagen auf cwe begrenzen | **Umgesetzt.** Abstract, §6.2, §6.4, §9 (Schluss) tragen jetzt Task- und Modell-Scope | Paper global |
| 3 | FP8-Produktionskontext | **Umgesetzt.** Neuer Absatz §1 (Hopper/Blackwell ≤1–2 Punkte bis 1M Token vs. HIP/Consumer ohne FP8-Pfad) | Paper §1 |
| 4 | KL-Richtung präzisieren | **Umgesetzt.** $D_{\mathrm{KL}}(\mathrm{ref}\|\mathrm{arm})$, Begründung + Reverse-Ranking-Einschränkung | Paper §5.3 |
| 5 | Antwortregion (n=27) | **Umgesetzt.** Aus der Beweiskette entfernt; überall „descriptive, \ldots positions" | Paper §6.2, §6.3, §6.4 |
| 6 | A4 sichtbar | **Umgesetzt.** Eigene Tabellenzeile „Open item."; Zeilenreihenfolge A1–A5 numerisch | Table 2 |
| 7 | Laufzeit-Tabellen präzisieren | **Umgesetzt.** „Medians of two same-day alternating rounds", Mixed-Set/HIT-Definition, 512/128-Vorbehalt in der Caption. Konkrete Streuungswerte wurden bewusst **nicht** erfunden; sie liegen in den Messreports | Table 3, §6.1 |
| 8 | Kapazität statt Durchsatz | **Umgesetzt.** „frees 568 MiB ≈ 416 expert slots — capacity figures, not speed wins" | Paper §6.1 |
| 9 | Anatomie-Abbildung | **Umgesetzt.** TikZ-Layoutfigur (Hauptbar + fp16-Overlay + A2-Veto) | Paper §3 |
| 10 | Reproduzierbarkeit / Provenance | **Umgesetzt.** Neuer Unterabschnitt „Provenance and reproducibility" | Paper, Methodology |

### Major-/Minor-Punkte des Reviews

| Punkt | Status | Anmerkung |
|-------|--------|-----------|
| M1 Positionierung | **Behoben.** | Related Work §2 |
| M2 Task-Breite | **Teilweise abgemildert (Scope), offen (Messung).** | Claims auf cwe begrenzt; zweiter RULER-Task/LongBench steht im Follow-up, erfordert GPU-Zeit |
| M3 Community-Anker-Arm | **Teilweise (Kontext), offen (Messung).** | KIVI-Style-per-channel-Arm steht im Follow-up; keine neue Messung |
| M4 KL-Notation | **Behoben.** | §5.3 |
| M5 n=27 | **Behoben.** | s. Tables |
| M6 Runtime-Statistik | **Behoben (Dokumentation).** | Runden/Set-Definition in der Caption; konkrete Streuungsbeträge sind in den Messreports, nicht erfunden |
| M7 Slots/Hit | **Behoben.** | Kapazitätsformulierung ohne Durchsatzimplikation |
| M8 Reproduzierbarkeit | **Behoben.** | Paper §5 Provenance-Abschnitt |
| Minor 1–10 | **1,2,3,6(int8-Fußnote),9,10 umgesetzt; 4 (Bytes-Konvention) teilweise; 5 (p99-CI) offen; 7 (Abstract-Trimmung) teilweise — beide t-Werte verblieben; 8 (8/8-Run-Kontext) offen** | |
| Optional Messitems 11–12 | **Offen.** | Brauchen GPU-Zeit; als Follow-up im Papier verankert |
| Fragen an Autoren (Q1–Q4) | **Offen.** | Q1 (K4/V4-Symmetrie), Q2 (S/R-Knee), Q3 (Multi-Turn), Q4 (Mittelkontext) — die diskutierbaren Teile wandern in den Improvements-Katalog `Ansatz_Improvements.md` |

### Fakten, die der Review nur gelesen, nicht geprüft hat

- Der Review misst selbst nichts. Alle prüffähigen Kernbelege stammen aus den
  repo-dokumentierten Messberichten; die in-Review-Verweise
  (arXiv:2309.17453, 2402.02750, 2401.18079, 2403.05527, 2404.14469,
  2406.02069, 2306.14048, 2404.00456, IJCAI 2025/0690, 2502.04420,
  2412.12706, vLLM-Blog 2026-04-22) wurden gegen die Originale geprüft.
- Verbleibende offene Empfehlungen: P99-Bootstrap-CI (rechnerisch, offline,
  hochpriorisiert), KIVI-/per-channel-Arm und zweiter RULER-Task (GPU),
  Multi-Turn-Messung (GPU), A4-Paritätstest (Engineering).

### Nach dem zweiten, unabhängigen Review (grok-4.7, 2026-10-24)

Ein zweites, unabhängiges Desk-Review (`PEER_REVIEW_GROK.md`, grok-4.7 via
opencode; blind gegenüber dieser Datei, getrennte Sitzung, keine Messungen)
hat den revidierten Artikel gegen die Messquellen geprüft und acht
Major-Befunde geliefert, die alle gegen `DETAILS.md`, den Workpackage-Bericht
(Branch `KV_QUANT_research`) und den Code verifiziert wurden. Die ehrliche
Bilanz dieses Selbst-Reviews: Sechs der acht Befunde überdeckt die obige
Abarbeitungsliste nicht; die oben stehende Liste prüft Textaktivität
(„Abschnitt X neu"), nicht Inhaltstreue der gedruckten Zahlen gegen die
zitierten Quellen. Die Lücken, im Einzelnen:

- **Grok M1 (übersehen):** `DETAILS.md` — vom Paper selbst als
  Provenance-Quelle zitiert — enthält einen zweiten 128K-Text mit
  umgekehrter Tail-Rangfolge (int8 0.00324 / window 0.00437 / plain 0.00664
  auf den letzten 4.096; Ganzsegment int8 0.06435 / window 0.07524 /
  plain 0.08834); der Artikel druckte nur Text 1 (58 %-Claim). Dieses
  Review hat die eigene Provenance-Aussage nie gegen die zitierten Quellen
  ausgespielt — genau der Punkt, den „measurement-first" ausschließt.
- **Grok M2 (übersehen):** Die entscheidende KL ist laut Harness
  top-64-trunkiert, renormalisiert und mit Floor-Substitution
  (`quant_quality.py`, `kl_between`, in beiden Branches identisch);
  dieses Review präzisierte nur die KL-Richtung (M4 oben), las aber nie
  die Implementierung.
- **Grok M3 (übersehen):** Das gemessene Modell ist ein Hybrid (12
  QSA-Layer von 48, Indexer-Budget top-2048, `geometry.hpp`); die
  4,9 %-Zahl ist die Ring-Kapazität (R+S+B)/L, nicht die per-Query-exakte
  Abdeckung (R+S = 320 laut WP §7.21); „cell" hat drei Bedeutungen.
- **Grok M4–M6 (übersehen):** die nie auf 128K gemessene
  512/128-Qualitätsbehauptung samt falscher Tabellenverweise (M4), der
  Break-even ≈27.600 Zellen statt „32K" (M5, bei 32K ist das Fenster
  noch billiger als int8: 7,67 vs 8,25 Bit) und der Crossover 4,8–5,3 Bit,
  den derselbe Bericht §7.16 als Artefakt von R=8 eingezogen hat (M6).
- **Grok M7 (teilweise):** Dieses Review hatte in Minor 8 den „8/8"-Kontext
  als offen vermerkt, aber die untraceablen Counts (8/8, 59/59, „about one
  second") blieben bis zum Grok-Review im Text; die vier Window-Ordner
  unter `bench/results/2026-10-*` sind leer.
- **Grok M8 (übersehen):** „4-bit K / 8-bit V" ist falsch benannt — tatsächlich
  16-Level-K / 8-Level-V im selben 4-Bit-Block; der eigene Swift-K-vs-V-Test
  widerlegt die Asymmetrie (t +0,05, McNemar p=1,000). Dieses Review fragte
  in Q1 nach einer „K4/V4-Symmetrie" und wiederholte damit die falsche
  Rahmung statt sie zu hinterfragen.

Die Papier-Fixes zu den Revisionsplan-Items 1–10 des Grok-Reviews wurden am
2026-10-24 im Artikel umgesetzt (Text-only, inkl. Zweittext-Nachlese). Ein
anderes Modell (z. B. grok-4.7) als zweite Prüfinstanz ist aus denselben
Gründen empfohlen, aus denen dieser Caveat gilt: ein Selbst-Review desselben
Modells übersieht systematisch dieselben blinden Flecken.

---

*End of review. Nothing in §4–§8 requires reproducing the authors' runs;
every external number above is from the cited public source, and points
derived only from reading the manuscript are marked as static reading.*
