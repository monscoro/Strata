# Peer review: Hardening the Exact-Cell KV Window for `--kv q4lm8v`

- **Reviewer model:** grok-4.7 (xai), via OpenCode
- **Date:** 2026-10-24
- **Paper reviewed:** `docs/paper/KV-Window-Hardening.tex` (compiled PDF, 8 pages; TeX log reports no overfull boxes)
- **This review made no measurements and no GPU runs.** It is a desk review. Claims below that come from reading the manuscript, the work package, or engine headers are marked as static analysis. Numbers are confirmed against a cited source or listed as unconfirmed. No figure in this review was produced by running the engine.

**Sources used for the cross-check**

- `docs/DETAILS.md` (KV window paragraphs near the top, and the “KV cache quantization” section)
- `docs/QUANTS.md` (yardstick section)
- `docs/KV_QUANT_WORKPACKAGE.md` on branch `KV_QUANT_research` (`git show`), especially §7.13–§7.21 and §12
- `bench/results/2026-09-27-kv-q4/README.md`
- Static reading of `include/strata/core/geometry.hpp`, `include/strata/kernels/kv_q4.hpp`, `include/strata/kernels/qsa.hpp`, `include/strata/kernels/qsa_decode_attn.hpp`, `include/strata/kernels/kv_q8.hpp`, `src/core/layer.cpp`, `src/program/generate.cpp`, and the research-branch `tools/quant_quality.py` (`kl_between`)
- The vLLM FP8 blog cited by the paper, fetched 2026-10-24
- The window result directories named by the work package (`bench/results/2026-10-17-kv-wp2a-window`, `2026-10-17-kv-wp2b-window3bit`, `2026-10-18-kv-runtime-ab`, `2026-10-18-kv-wp2b-oscar-knee`) exist in the working tree and are **empty**. The numbers were checked against the work-package text, not against raw TSVs.

`docs/paper/PEER_REVIEW.md` and `docs/paper/Ansatz_Improvements.md` were not used. A repository search returned one line from each; those lines were ignored.

---

## 1. Summary

The paper describes an opt-in exact-cell overlay on Strata’s 4-bit Lloyd KV cache: a resident fp16 side pool for 64 sink cells and 256 recent cells, plus room for the current prompt chunk, on top of a quantized cache that still stores every cell. At 128K the production point is 5.29 bit/element and 1,016 MiB against the int8 default’s 8.25 bit/element and 1,584 MiB, with prefill and decode that the same-day A/B does not separate, and with a teacher-forced KL cost that the authors correctly refuse to call an accuracy win. The measurement discipline — paired KL, a tail percentile, and an explicit rejection of perplexity as the decision metric — is the paper’s real contribution. The draft undercuts that discipline by omitting a second 128K text already printed in a document it cites, by describing a truncated top-64 KL as a full-distribution KL, and by writing the design as if the measured model were a 12-layer dense-attention cache.

## 2. Strengths

- **The claim is the right size.** The abstract and §7.4 say the window is a 36% KV-memory saving with a small, measurable KL cost, not an accuracy win over int8. That matches the tables. Many KV-compression papers do not do this.
- **The decision metric is the right one, even though it is misdescribed (see M2).** Reporting perplexity beside KL, and pointing out that every 128K tail PPL sits below the fp16 reference while every KL is positive (Table `tab:q128k`, PPL 2.4052 / 2.4079 / 2.4029 against fp16 2.4149), is the correct warning. Those PPL figures match work-package §7.21.
- **The short-context statistics are traceable and mostly careful.** The 2K and 4K tables match §7.15, §7.16 and §7.20 to the printed digit. The 2K “not distinguishable” claim uses the right phrase (failure to reject, \(t=-1.92\)), not “equal”. The 4K answer-region \(t=-0.51\) is in §7.20. The 27-position 128K answer region is labeled descriptive, and the caption does the arithmetic \(25/27\) and \(26/27\). That is better than most systems papers.
- **Saturated tests are not sold as results.** Needle 18/18 is described as no regression and no separation. That matches §7.20 (“gesättigt”). A4 is an open review gate, not a shipped fix.
- **Section 4 is honest about what was not measured.** Labeling the alternatives as static analysis, and naming a per-channel-K arm as the missing community anchor, is the right posture. The threats section’s one-task, one-GPU, prefill-quantized limits are real and are stated.
- **The byte arithmetic of the production point checks.** With \(L=131{,}072\), \(R+S+B=6{,}464\), \(4.50 + 16(R+S+B)/L = 5.289\), and \(16 \times 6{,}144 / 131{,}072 = 0.75\). The 0.75-of-0.79 sentence is the chunk term, and it is right. \(1{,}584 - 1{,}016 = 568\) MiB, \(568/1{,}584 = 35.9\%\), and \(3{,}449 - 3{,}033 = 416\) slots. Those runtime medians (526.1 / 528.0 prefill, 35.2 / 34.2 decode, hit 0.74 / 0.76) are the §7.20 table, including the per-round decode signs \(-2.4 / +0.4\) and int8’s own 36.6 → 33.8 spread.
- **The hardening principle is the right one for this kind of bug.** A silent exact-read that is actually quantized is worse than a loud refusal. A2’s double enforcement (setup and engine, exit 2) is in `docs/DETAILS.md`. The empty-tag sentinel (`0xFF`, because a zero tag is indistinguishable from cell 0) is in `src/core/layer.cpp` (static analysis). The paper’s choice to keep file-level detail out is fine if the evidence column only claims what a reader can find.

## 3. Major concerns

### M1. The second 128K text is in a document the paper cites, and the paper does not report it

**Where:** Abstract (the 58% tail sentence and the “int8 keeps the better p99…” sentence); §7.3; §7.4; §9, which says the 128K result is one text; §6.6, which cites `docs/DETAILS.md` as provenance.

**What the source says:** `docs/DETAILS.md` (the `strata-details` citation), in the same paragraph as the 0.00663 / 0.00959 / 0.01587 numbers, reports a second 128K text from the same generator (same seed, alpha, and 2,621 code words, drawn separately):

| Region | int8 | window 256/64 | plain `q4lm8v` |
| --- | ---: | ---: | ---: |
| Last 4,096, mean KL | 0.00324 | 0.00437 | 0.00664 |
| Last 4,096, p99 | 0.0165 | 0.0210 | (not given) |
| Whole segment, mean KL | 0.06435 | 0.07524 | 0.08834 |

On that text the tail-mean ranking **reverses**: int8 is better than the window, and the window removes about a third of plain’s tail KL, not 58% (\((0.00664-0.00437)/0.00664 = 34\%\)). DETAILS’s own reading is the one the paper should have led with: the gap between the two quantized arms is inside the text-to-text spread; both texts agree that the window beats plain and that int8 wins the whole segment.

Work-package §7.21, which the paper treats as the 128K quality source, does **not** contain this second text. A search of the working tree finds the numbers only in `docs/DETAILS.md`. The raw TSV is not in the tree (unconfirmed beyond that paragraph).

**Why it matters:** The abstract’s 58% and the “31% lower mean, \(t=-0.74\)” sentence are true of text 1 and false of text 2. A measurement-first paper that cites DETAILS and then reports one of two texts has a selective-reporting problem, even though the robust conclusion (int8 wins p99 and the whole segment; the window beats plain) survives both texts. §9’s “one text” is true of what the paper printed, and incomplete relative to what the authors had already written down.

**Fix:** Put both texts in Table `tab:q128k` or in a two-row companion, with the same regions (last 4,096, last 1,024, whole segment, p99). State the generator settings. Replace the abstract’s 58% with the range across the two texts, or with the statement DETAILS already uses. Say where the second run’s log-probabilities live. If the second run was a different engine build than §7.21’s 0.1.44.1, say so (DETAILS attributes the paragraph to “port build 0.1.47”).

### M2. The decision metric is a top-64, renormalized, floor-substituted KL, and the paper describes a full-distribution KL

**Where:** §6.3, the formula \(D_{\mathrm{KL}}(p_{\mathrm{ref}}\,\|\,p_{\mathrm{arm}})\) and the sentence about “bulk distribution drift”; §6.6, which points at `docs/QUANTS.md` for the yardstick; the interpretation of p99 and of the single-position KLs 17.3, 17.1 and 15.8 in the caption of Table `tab:regions`.

**What the harness does** (static analysis of `KV_QUANT_research:tools/quant_quality.py`, `kl_between`, and of the working-tree `tools/quant_quality.py`, which is the same KL): KL is computed over the reference’s 64 most likely tokens. Both distributions are renormalized onto that set. A token the arm does not list is given the arm’s 64th log-probability as a floor. `docs/QUANTS.md` documents this for the weight-quant yardstick (KL against Q8_0 on `data/quant_quality/v1`). It does not document the cwe protocol. The working-tree harness has no cwe generator; that lives on the research branch.

This is not \(D_{\mathrm{KL}}\) of the two full next-token distributions. Mass the arm places outside the reference’s top 64 is invisible except through the floor. A KL of 17 nats is what a floor substitution produces when the arm’s top 64 misses a reference mode; it is not, by itself, evidence of an isolated hard token in the full distribution. The paper’s remark that the reverse direction was not measured is true and incomplete: the full-vocabulary forward KL was not measured either.

The numbers in the tables are still the harness’s numbers. They match the work package. They are comparable across arms **inside this definition**. They are not the quantity §6.3 interprets, and they are not comparable to a paper that reports full-vocabulary KL.

**Fix:** Define the metric as the harness defines it, in §6.3, before any interpretation. Say top-64, renormalized, floor = arm’s 64th log-probability. Drop or qualify “bulk distribution drift”. Note that p99 and the 17-nat positions can be floor events. If a full-vocabulary KL is wanted, it needs a new scoring pass (`STRATA_LOGPOS` does not record it). A bootstrap over positions, which the paper already lists as follow-up, should be computed on this truncated KL and should say so.

### M3. The design section describes a different machine from the one that was measured

**Where:** §3, first paragraph (“a cell is one token’s K or V entry”; “the reference model has 12 attention layers”); the bit/cell formula; §4, “full 16-bit precision on 4.9% of cells”; §9, the 20.3% / 4.9% exact-cell fractions; Figure `fig:anatomy`; related work’s comparison to StreamingLLM, H2O and SnapKV as if the attended set were the stored set.

**Static analysis of the compiled geometry** (`include/strata/core/geometry.hpp`, `qsa.hpp`, `qsa_decode_attn.hpp`, `kv_q4.hpp`):

- The model the kernels are compiled for is Qwen3.8-Flash-Next: 48 layers, of which every 4th is full attention (layers 3, 7, …, 47). That is 12 QSA layers and 36 GDN layers. The paper’s “12 attention layers” is the QSA count with the hybrid architecture omitted. The paper never names the model, only the Swift 1.5 IQ3_XXS weights.
- A QSA query does not read the cache. It reads a selection of width \(\min(n_{\mathrm{kv}}, \texttt{idx\_top\_k} + \texttt{idx\_block} - 1)\), with `idx_top_k = 2048` and `idx_block = 4`, so at most 2,051 cells. The window reader (`qsa_win_slot`) substitutes an fp16 row when a **selected** cell is a sink or a recent cell whose tag still matches. I found no reference to the window in the indexer headers I read. Static analysis, not a full audit: the overlay does not appear to change which cells are attended, and sinks are not obviously forced into the selection. If that reading is wrong, the paper should say where the force happens.
- “Cell” is three different things in the paper and the sources. The paper defines a cell as one token’s K **or** V in one layer. The byte model’s “576 B per cell and layer” is one token’s **K and V** in one QSA layer (`kv_q4_bytes_per_cell`: 2 KV heads × 256 × 2 tensors × 18-byte blocks / 32 = 576 B). The “5.29 bit/cell” rate is bits **per scalar element** (576 × 8 / 1024 = 4.50). The formula \(4.50 + 16(R+S+B)/L\) is consistent only with the third sense. A reader who uses the paper’s definition cannot interpret the formula.
- The 4.9% figure is the ring’s resident capacity, \((64+256+6144)/131072 = 4.93\%\), not the fraction of cells a decode query reads exact. At a decode step the exact set is \(R+S = 320\) cells, which is 0.24% of 128K, and only the subset of those 320 that the indexer selected is actually read exact. The 20.3% at 4K is \((64+256+512)/4096\), again ring capacity, not per-query coverage. Work-package §7.21 already separates these: quality runs use `--short-read` and `--prefill 512`, and “per query \(R+S = 320\) exact cells, independent of ring capacity \(B\)”. The paper’s threats section then treats 4.9% as the production coverage. That overstates the exact fraction a scored token sees by about twenty times, and it mixes a memory fact (the ring is resident and includes \(B\)) with a quality fact (a short-read query sees \(R+S\)).

The memory numbers do not depend on fixing this. The interpretation does. “Protect the cells that matter most”, the StreamingLLM analogy, and “no eviction, unlike H2O/SnapKV” are storage claims. On this model the attended set is already a top-2048 selection. Eviction papers drop cells from a dense read; this window changes the precision of selected cells that also fall in a 320-cell overlay. That may be a **better** story — a small exact overlay on an already-sparse read — but it is not the story written down. It also means the fair external comparison is not only a KIVI-style per-channel arm on a dense cache. It is that arm on this indexer.

**Fix:** Name the model and the hybrid split in §3. Define three words and keep them: element (the bit rate), token-position (the work package’s Zelle, K and V together), and K or V vector. State the indexer budget and that the window substitutes values inside the selection rather than changing it, or cite the code that forces sinks and recent tokens in. Split “ring holds 4.9% of positions” (memory) from “a decode query has at most 320 exact positions, of which the indexer reads some” (quality). Redraw Figure 1 so sinks sit at the start of the cache, and so the top bar does not look like an exact region inside the quantized cache. The caption already says the overlay is additional; the drawing does not.

### M4. Section 8.2 says the 512/128 grid matches 256/64 at 128K. That quality run was not done, and the table citations are wrong

**Where:** §8.2, “At 4K and at 128K the 512/128 pair scores the same quality as 256/64 (Tables `tab:q2k` and `tab:q128k`)”; also §7.1, which correctly does **not** put a 128K quality number on 512/128, and Table `tab:q128k`, which has no 512/128 row.

**What was measured:**

- cwe-4k, §7.20: 256/64 vs 512/128, paired \(t=+0.02\), 4/8. That is the equality. It lives in Table `tab:q4k`, which §8.2 does not cite.
- cwe-2k, Table `tab:q2k`: 0.00702 vs 0.00623. §7.16’s within-family step 256/64 → 512/128 is \(t=+1.98\), 7/8, not a settled tie. §8.2 cites this table for a 4K/128K claim.
- cwe-128k, §7.21: int8, 256/64, and plain `q4lm8v` only. No 512/128 quality arm.

The decision not to ship 512/128 is still supported by the 4K tie, the runtime cost, and the needle prefill not reproducing the 634 vs 528 tok/s (246.3 vs 247.7 s at depth 10, 241.5 vs 251.2 s at depth 50; §7.20). It is not supported by a 128K quality measurement. “Cheapest measured arm with indistinguishable quality” at 128K is true only because 256/64 is the only window arm whose 128K KL was measured. That is a weaker statement than the section makes, and the section already tries to be careful (“not a local optimum”). The false sentence undoes the care.

The “8–10 MiB more” in the same sentence does not match the paper’s own table (1,023 − 1,016 = 7 MiB). The work package uses both figures. See minor 2.

**Fix:** Delete the 128K half of the sentence. Cite Table `tab:q4k` and §7.20’s \(t=0.02\) for the 4K tie. Say explicitly that 512/128 was not quality-scored at 128K. If the sentence is meant to justify shipping 256/64, rest it on 4K quality plus the runtime A/B, which is what the work package actually decided on.

### M5. The operator rule uses 32K as the break-even. The break-even is about 28K, and at 32K the window is still cheaper than int8

**Where:** §3.2, “Below roughly 32K cells the ring costs more than the 4-bit base saves, which is why setup offers the window only from 32K upwards.” §8.1 repeats it as the decision rule.

**Source:** Work-package §7.20: with \(B=6{,}144\), break-even against int8 is about 27,600 cells; at 32K the window is 7.67 bit against int8’s 8.25; setup offers the arm from 32K because that is above the break-even. `docs/DETAILS.md` says the same thing in fewer words: “below the ~28K break-even … offered at 32K and more.” The commit that added the window says the same.

Arithmetic, static, from the paper’s own formula: the 4-bit base saves \(8.25-4.50 = 3.75\) bit/element. The ring costs \(16 \times 6{,}464 / L\). Those meet at \(L = 103{,}424 / 3.75 \approx 27{,}580\). At \(L=32{,}768\) the window is \(4.50 + 3.16 = 7.66\) bit/element, still under int8. The offer threshold and the break-even are different numbers. The paper merges them, and the merge makes the decision rule false at the threshold it names.

**Fix:** Say “break-even about 28K cells; setup offers the window from 32K, where it is still cheaper than int8 (7.67 vs 8.25 bit/element at \(B=6{,}144\))”. Recompute the §8.1 rule from that sentence. No new run is required.

### M6. The “cheaper overlay” paragraph cites a result the same work package later withdrew, and it names the wrong tensor

**Where:** §8.3.

The paragraph says the repository documents a “≤3-bit K overlay” with a measured crossover “between roughly 4.8 and 5.3 bit/cell” on “the Swift 2K/4K harnesses, under the \(R=8\) grid”.

What §7.15 actually measured is cwe-**2k** only, not 4K. The 3-bit side is 8-level **V** (`q4lm8v`: 16 reconstruction levels on K, 8 on V), or 8-level on both sides (`q4lm8`). It is not a 3-bit K overlay. Both still occupy the same 18-byte block; the level count does not change the stored 4.5 bit/element of the base (work-package §7.19: the old 4.0 figure dropped the 2-byte scale). The 4.8–5.3 bit numbers are that old, pre-ring model (`4.0 + 12(S+R)/L`), which §7.19 says was too optimistic twice (the exact cell is additional, and the span is \(R+S+B\)). §7.16 then says the crossover **does not reproduce** at \(R=64\): across five shapes the 3-bit and 4-bit bases are not separable (\(|t| \le 1.50\)), and the 4.8–5.3 crossover “was an artifact of too small a sink count”.

§8.2 cites §7.16 for the knee. §8.3 cites the crossover §7.16 withdrew, without the withdrawal. “Not a validated production point” is not enough: a reader will take 4.8–5.3 as a byte bound. Under the corrected model the 2K 256/64 point is about 11 bit/element, not 5.9, and it is more expensive than int8 at that length (§7.19).

**Fix:** Either delete the paragraph or rewrite it as: at \(R=8\), on cwe-2k, an old byte model showed a narrow region where 8-level V plus a small window beat 16-level on KL per byte; §7.16 did not reproduce a crossover at the production sink depth; the absolute bit numbers in §7.15 were superseded by §7.19; a 128K arm is unmeasured. Do not call it a K overlay.

### M7. Two correctness counts in the abstract and the body are not in the measurement history the paper cites, and the raw window artifacts are not in the tree

**Where:** Abstract, “byte-identical in 8/8 runs”; §5, Table `tab:decisions`, A1 and A2 evidence; §7.5, the SHA, “about one second”, “setup tests pass 59/59”.

**What traces:**

| Claim | Status |
| --- | --- |
| Dump-logits SHA prefix `A014F504…BA052` for int8 with the window code present | In `docs/DETAILS.md`. The adding commit says one SHA-256 of the first window’s logits, 993,280 bytes, byte-identical to v0.2.2 for `--kv int8` and `--kv fp16`. |
| **8/8 runs** | **Unconfirmed.** Not in DETAILS, not in the work package. |
| Setup tests **59/59** | **Unconfirmed.** Not in DETAILS, not in the work package. |
| Window-plus-streaming exits 2 before load | In `docs/DETAILS.md`. |
| “in about one second” | **Unconfirmed.** |
| Kernel parity 12/12 | In work-package §7.14 and §7.19. This is the synthetic `kv_q4_parity` suite (ring wrap, poisoned tag, sinks), not the A4 dump. The paper distinguishes these. Good. |
| Needle 18/18, 6/6 per arm, saturated | In §7.20. |
| Plain-arm hang, not reproduced in 6/6, window arms completed 3×128K | In §7.20. |
| A1, “an 8,192-token prompt completes with a 6,144 chunk” | The post-fix success is **unconfirmed** in the measurement history. A code comment in `generate.cpp` records the **pre-fix** failure on this PC: an 8,191-token prompt, `--prefill auto:16384`, exit 1, “8191 cells in one batch do not fit the 6400-row recent ring” (static analysis). |
| A3, “targeted probes pass” | **Unconfirmed** as a test result. The freeze exists (`qsa_kv_win_freeze`, static analysis). In a release build the `assert` is compiled out and the late call prints a warning and returns. The paper’s “assert” is a debug assert plus a release warning. |
| A5 mechanism | Confirmed by static reading of `layer.cpp` (sentinel `0xFF`). “Empty-state semantics are correct” as a passed test is unconfirmed in the reports. |

§6.6 says the measurement reports are plain-text files next to the code. The four window result directories are empty. A reader who follows the paper to `bench/results/2026-10-18-kv-runtime-ab` finds nothing. The work package on `KV_QUANT_research` is the real report, and the bibliography does say that. The “files next to the code” sentence overclaims.

**Fix:** Drop 8/8 and 59/59, or cite the log that contains them. Quote the SHA the way DETAILS does (one identity against the previous release), which is already a strong gate. Replace “about one second” with “before load, exit 2” unless a timed log exists. Point §6.6 at the work-package path and the branch, and either commit the READMEs or stop naming empty directories. For A1, cite the pre-fix failure that is actually written down, and mark the post-fix completion as the behavior of the cap in `generate.cpp` (static) unless a completion log exists. For A3, say the release build warns and ignores, it does not abort.

### M8. Related work misnames the production split, and it cites a K-versus-V asymmetry the authors’ own Swift run rejected

**Where:** §2, “the K-is-more-sensitive-than-V asymmetry behind its 4-bit K / 8-bit V split is the same observation KVTuner formalizes”; Table `tab:survey` is not the problem here. §3 correctly says K has 16 levels and V has 8, in 18-byte blocks.

`q4lm8v` stores both tensors in the same 4-bit block (4.5 bit/element including the scale). K uses a 16-level codebook and V an 8-level codebook (work-package §12.2 and §7.19: “16 K-Werte und 8 V-Stufen im selben 4-Bit-Code”). That is not a 4-bit K / 8-bit V split. The engine’s hybrid that **is** 8-bit K and 4-bit V is `k8v4`, which this paper is not about. The sentence also inverts the usual reading of the name: “8v” is 8-level V, not 8-bit V.

Work-package §9 and §11.2: the Q2_0 hint that K wants the bits (McNemar 0:5, \(p=0.062\)) did not survive Swift. Direct contrast on Swift: \(+0.00011\), \(t=+0.05\), 3/8, McNemar \(p=1.000\). The production choice of 8-level V is a byte/codebook choice that the authors’ own regime check did not justify as “K is more sensitive”. KVTuner’s layer-wise pairs are a different mechanism. Citing them as the same observation is not supported by the measurement history the paper tells the reader to trust.

**Fix:** Delete the sentence, or replace it with: the production base uses a 16-level table on K and an 8-level table on V inside the same 4-bit block; a K-versus-V bit allocation was not supported on Swift (\(t=+0.05\)); KVTuner is a different, unmeasured axis (already the right stance in §4). Section 3 can stay as it is.

## 4. Minor points

1. **The 8–12% sentence compares against the wrong baseline.** Introduction: “document perplexity 8–12% above the 8-bit default at 1K and 8K”. `bench/results/2026-09-27-kv-q4/README.md` says document perplexity +8% at 1K and +12% at 8K, and the table’s “vs fp16 (nats/token)” column is \(+0.078\) and \(+0.112\), i.e. about \(+8\%\) and \(+12\%\) versus **fp16** (\(\mathrm{e}^{0.078}-1\), \(\mathrm{e}^{0.112}-1\)), on **Q2_0**, for **`q4_0`**, not for `q4lm8v`. Against int8 the same table is 32.26/30.18 − 1 = 6.9% at 1K and 8.65/7.63 − 1 = 13.4% at 8K. Say “about 8% and 12% above fp16, on Q2_0, for `q4_0`”. The citation `strata-kv4` is the right file; the comparator in the sentence is not.

2. **8–10 MiB versus 7 MiB.** §7.1 and §8.2 say the 512/128 arm costs 8–10 MiB more. Table `tab:runtime` says 1,023 versus 1,016. The work package uses both. Pick the table, or show the runtime reading that is not 7 MiB.

3. **“Scan limit” is not what the code implements.** §3.2 and DETAILS call 6,144 a ring scan limit. Static analysis: `kKvWinBatchMax = 6144` in `generate.cpp` is a host policy cap, shared with the Windows HIP auto-chunk cap in `host_limits.hpp` (a commit-charge crash at 8,192 on an RX 7900 XTX, not a ring-reader scan). The comment in `generate.cpp` says the cap also keeps setup’s 152 MiB figure honest on hosts whose auto chunk is 8,192 or larger. Say “host cap”, not “scan limit”.

4. **Figure 1 puts the sinks at the tail.** The quantized bar draws sinks immediately left of “recent”, at the right edge. Sinks are the first \(R\) cells. The dashed arrows then map that tail region onto the left of the overlay. A reader can conclude that the exact region is a suffix of the cache. Move sinks to the left end, and do not paint them as a hole in the 4-bit bar. The caption’s “widths not to scale” does not fix the order.

5. **The method requires a sign count, and the results omit it.** §6.4 says a large \(t\) with 4/8 segments is not a result. The 4K headline \(t=-3.55\) is §7.20’s \(1/8\) positive (7 of 8 segments favor int8). Report that; it strengthens the sentence. Also report the §7.16 interval: “not distinguishable” at 2K, and the 95% interval of int8 − window still reaches about \(+0.0044\). Failure to reject is not a bound.

6. **The runtime table was taken with speculation on.** §7.20: `--spec 4`, `--expert-cache auto`, engine 0.1.44.1, mixed prompts including one prompt of about 123K tokens. The paper does not mention `--spec 4`. Absolute decode tok/s are serving rates under drafts, so the run-to-run spread includes acceptance, not only the KV format. Say so. Two rounds is a thin A/B for “equal”; the paper’s own spread argument is the right one, and it should be next to the word “equal” in the abstract, not only in §7.1.

7. **Engine build and OS are underspecified, and the two cited sources disagree.** Threats: Windows 11, ROCm/HIP 7.1. DETAILS says Windows and ROCm 7.1, and “port build 0.1.47” for the 128K KL paragraph. §7.20 and §7.21 say engine 0.1.44.1. The paper names no engine version. “Windows 11” specifically is unconfirmed (sources say Windows). Name the build that produced each table. If 0.1.47 reproduced 0.1.44.1 byte-for-byte on cwe-128k, that is worth one sentence; if the numbers were copied across a build label, say that too.

8. **Quality configuration is stated for 128K only.** §6.2 says the 128K quality runs use `--prefill 512`. §7.19 and §7.20 say cwe-2k and cwe-4k did too, and that the path is `--short-read` (decode windows), not the production batched prefill of 6,144. The paper’s “prefill KV is quantized” matches the work package’s language and hides the path difference: a production chunk holds the whole chunk exact while it is prefilled; a short-read step sees \(R+S\). §7.21 argues quality is independent of \(B\) for that reason. That argument is static, not an A/B of \(B=512\) against \(B=6{,}144\). State the path, and do not let a reader think Table `tab:q128k` is the production chunk’s quality.

9. **§8.4 says a mid-context region is unmeasured.** §7.21 already reports, for the window arm only, KL 0.054 on the first 4,096 positions and KL 0.164 on one middle 4K block (positions 65,536–69,631). A paired mid-segment comparison is unmeasured. A one-arm interior number exists. Say that, so the follow-up is “score the other arms on that block”, not “no interior number exists”.

10. **int8’s 8.25 is one fp16 scale per 64 elements, not the q4 block.** `kv_q8.hpp`: \(2 \times 256 \times 2 + 2 \times 4 \times 2 \times 2 = 1{,}056\) B per token per layer, and \(1{,}056 \times 8 / 1{,}024 = 8.25\). The paper’s “block scales” is close enough for the rate and wrong for the group size. One clause fixes it. DETAILS’s other section that says “int8 (default, 8.0 bit/cell)” is the payload-only figure; the paper correctly uses 8.25. Do not “fix” the paper toward 8.0.

11. **§8.2’s “the window first carries the load near \(S=512\)” is the \(R=8\), 16-level-base knee (§7.13), not the production base.** On `q4lm8v` with \(R=8\), 128/8 already removes about half the KL, and 512 vs 1024 is still significant (§7.15, \(t=+2.80\), 8/8). The production sink depth makes 256/64 indistinguishable from int8 on cwe-2k (§7.16). The second half of the sentence is right; the first half will be read as a statement about the shipped grid.

12. **Typesetting.** The TeX log writes 8 pages and reports no overfull boxes. Underfull hboxes are mostly in the bibliography. I did not rasterize the PDF (no `pdftoppm` / PDF library in this environment), so a collision between Figure 1 and the column, if any, was not visually checked. The abstract is a results section; a reader who stops there gets the 58% number that M1 asks you to qualify.

## 5. Revision plan

| Priority | Change | Where | Cost |
| --- | --- | --- | --- |
| 1 | Report both 128K texts, and make the abstract match the text-to-text spread DETAILS already states | Abstract, §7.3, §7.4, §9 | Text only, if the second TSV still exists. If it does not, say the second text is a DETAILS paragraph without a raw log, and downgrade text 1 to “one of two texts, raw file for the second not in the tree”. |
| 2 | Define KL as top-64, renormalized, with the floor. Stop calling it bulk drift | §6.3, Table `tab:regions` caption | Text. A full-vocabulary rescore is a new run; do not pretend the current files contain it. |
| 3 | Name the model, the 12-of-48 QSA split, the indexer budget, and the three meanings of “cell”. Split ring capacity from per-query exact cells | §3, §4, §9, Figure 1 | Text and a redraw. No new run. Confirm or correct the “window does not change selection” reading before freezing the sentence. |
| 4 | Remove the 128K 512/128 quality claim. Cite Table `tab:q4k` for the 4K tie | §8.2 | Text. A 128K 512/128 quality arm is a GPU run; it is follow-up, not a patch. |
| 5 | Replace “below roughly 32K” with the ~28K break-even and the 32K offer threshold | §3.2, §8.1 | Text. Arithmetic already in §7.20. |
| 6 | Rewrite or delete the 4.8–5.3 crossover paragraph. It is \(R=8\), cwe-2k, old byte model, 8-level V, and §7.16 did not reproduce it | §8.3 | Text. |
| 7 | Drop or source 8/8 and 59/59. Point provenance at the work package, not at empty directories. State engine build per table | Abstract, §5, §6.6, §7.5, threats | Text, plus committing the READMEs if they exist off-tree. |
| 8 | Delete the “4-bit K / 8-bit V” sentence. Point at the Swift K-vs-V falsification | §2 | Text. |
| 9 | Fix the 8–12% comparator, the 8–10 MiB prose, the scan-limit wording, and the figure’s sink placement | Intro, §3.2, §7.1, Figure 1 | Text and a redraw. |
| 10 | State `--short-read`, `--prefill 512`, and `--spec 4`. Report the 4K sign count (1/8) and the 2K interval | §6.2, §6.4, §7.1, §7.2 | Text. |

Items 1–8 should land before this is sent as a measurement-first paper. Items 9–10 are the same revision, not a second round. None of 1–8 requires a GPU except the optional new runs named in the cost column. Do not add those runs in order to avoid reporting the second text.

## 6. Questions for the authors

1. Where is the raw log for the second 128K text (int8 0.00324, window 0.00437, plain 0.00664 on the last 4,096)? Was it engine 0.1.44.1 or the 0.1.47 build DETAILS names? Why is it in DETAILS and not in §7.21, and why did the paper omit it?
2. Please confirm that every KL in Tables `tab:q2k`, `tab:q4k`, `tab:q128k` and `tab:regions` is `kl_between` (top 64, renormalized, floor = arm’s 64th log-probability), and that the positions with KL 17.3, 17.1 and 15.8 are floor substitutions rather than full-distribution events. If some table used a different reduction, which one?
3. Does any selection path force the \(R\) sinks or the \(S\) recent cells into the indexer set, or does attention see an exact cell only when the indexer already selected it? I did not find a force in the headers I read (static analysis). If there is one, where?
4. Was cwe-128k scored only through `--short-read` (one decode step at a time, exact set \(R+S\)), or did any scored token see a 6,144-token chunk held exact? If the former, Table `tab:q128k` is not the production prefill’s quality. Has that been A/B’d, or is independence of \(B\) an argument?
5. What are the eight runs behind “8/8”, and the fifty-nine tests behind “59/59”? A pointer to a log is enough. The SHA prefix alone is already in DETAILS.
6. The 1,016 MiB figure matches \(864 + 152\) from the byte model. §7.19 measured the ring at \(B=512\) and \(B=16{,}384\), and computed \(B=6{,}144\). Did the production arm also print a VRAM delta of 152 MiB for the ring, or is 1,016 the model? The table reads as a runtime reading.
7. §7.16’s “2/8” on int8 − window at 2K: is that 2 segments where int8 is better, or 2 where the window is better? The paper’s “2/8 segments positive” can be read either way, and the two readings are different papers.
8. The MTP drafter turns the window off around its own init (`qsa_set_kv_win(false)` in the comment in `layer.cpp`) rather than sharing the frozen grid. Is A3’s “same grid on the MTP/twin state” still true after that, or does the drafter deliberately not carry the window?

## 7. Reference list check

Checked against the cited works’ titles, abstracts, and (for vLLM) the page itself. I did not re-read the PDFs’ result tables except where a search snippet quoted the abstract. I am not inventing missing bibliographic fields; gaps are marked unverified.

| Key | Verdict |
| --- | --- |
| `streamingllm` | Authors, title, arXiv:2309.17453, ICLR 2024, and the GitHub path match the paper as I know it. The use (sink retention, not the eviction policy) is fair in §2’s “no eviction” sentence. StreamingLLM itself drops the middle. The paper says so. |
| `kivi` | Authors, title, ICML 2024, arXiv:2402.02750, and `jy-yuan/KIVI` match. **Misquote:** KIVI quantizes K per-channel and V per-token, both to 2 bits, plus a full-precision recent residual. §2 says “2-bit per-channel K and V”. The table’s “per-channel” without a V axis repeats the error. The residual and the tuning-free claim are right. |
| `kvquant` | Authors, title, NeurIPS 2024, arXiv:2401.18079, and the SqueezeAILab path match. The abstract’s “< 0.1 perplexity degradation with 3-bit quantization” supports the table. WikiText-2 and C4 are in secondary descriptions of that abstract (OpenReview / the paper’s own summaries); I did not re-read the PDF tables. Pre-RoPE and per-channel keys are the paper’s stated methods. Supported at abstract level. |
| `gear` | arXiv:2403.05527 is right. **Title is wrong:** the paper is “Near-Lossless Generative Inference of LLM”, not “Generative LLM Serving”. “H. Kang et al.” is incomplete, not false. “4-bit + low-rank + sparse” matches the abstract’s near-lossless 4-bit recipe. I did not check the exact residual rank. |
| `snapkv` | Title and arXiv:2404.14469 and NeurIPS 2024 match. **Authors are wrong.** The first author is Yuhong Li, not Z. Zhang. Z. Zhang is the H2O line. “Per-head clusters” is a loose gloss of an observation-window selection; the table states no number, so this is a labeling issue, not a stolen result. |
| `pyramidkv` | “12% of the cache matches the full cache on LongBench” is in the paper’s own summary (arXiv:2406.02069, “only 12% of the KV cache”). Z. Cai is the right initial (Zefan Cai). COLM 2025 was not independently re-checked beyond the citation; the arXiv id is right. |
| `h2o` | Title and arXiv:2306.14048, NeurIPS 2023, match. “Z. Zhang et al.” is the right first author here (Zhenyu Zhang). No numeric claim. H2O is cited in the text and absent from Table `tab:survey`. That is an omission, not a misquote. |
| `quarot` | Title and arXiv:2404.00456 match. **No authors.** QuaRot is a weight-and-activation rotation paper that also rotates the cache; grouping it with RotateKV as “rotations before quantization” is fair if one clause says it is not a KV-cache method. I did not re-check the author list, so I will not supply one. |
| `rotatekv` | Title matches. IJCAI 2025 proceedings `2025/690` matches the PDF URL (`2025/0690`). **No authors** (Zunhai Su and coauthors, from the proceedings page; I am not copying a full list I did not verify name by name). “2-bit, rotation-based” is a fair table cell. The table states no accuracy number. |
| `kvtuner` | Title, ICML 2025, arXiv:2502.04420, and the cmd2001 path match the pages I checked. “Nearly lossless 3.25-bit” and a 4.0-bit point are in the abstract / the ICML page. The range 3.25–4.0 in the table is supported. I did not verify every coauthor. The **use** of this citation in §2 is the problem (M8), not the bibliography line. |
| `top-k` | Title and arXiv:2412.12706 match. First author on the PDF is Jiebin Zhang; the paper omits authors. The “studies the trade-off” sentence is fair and makes no stolen number. Findings of EMNLP 2025 exists for this paper; the bibliography cites only the arXiv id, which is acceptable if labeled as a preprint. |
| `llama-cpp` | `--cache-type-k/v` with fp16, q8_0 and q4_0 is real. The table omits later types (q4_1, q5_0, q5_1, iq4_nl, and others). Not a misquote. A version or a commit would make the claim checkable; the repository URL alone does not. |
| `vllm-fp8` | The URL is live. The post is dated 2026-04-22, by Jonas Kübler, Eldar Kurtić, and coauthors, not an anonymous “vLLM team”. The phrase “at most 1–2 points” is in the post, on the reasoning suite for Qwen3-30B-A3B-Thinking (Figure 7). **The paper’s sentence is stronger than the post.** The 1M-token claim is one model (Qwen3.5-27B) on MRCR AUC, which the post says is recovered in aggregate, with visible variance in the longest buckets. Other long-context rows recover 93–98% of AUC, which is not “at most 1–2 points” if the baseline is not near 100. The post also says `head_dim = 256` is where FP8 prefill still regresses. That is this model’s head dimension (`geometry.hpp`, static analysis). Using the blog as “the datacenter two-times point, ≤1–2 points, up to 1M tokens, on Hopper and Blackwell” drops the caveat that matters most for the comparison. Cite the reasoning bound as a reasoning bound, and mention the head-dimension caveat or do not use the blog as an unqualified contrast. |
| `strata-details` | The file exists and contains the byte model, the 5.29 / 1,016 vs 8.25 / 1,584 point, the cwe-4k and cwe-128k numbers, **and the second 128K text the paper omits (M1)**. The citation is the right file. The paper does not follow it. |
| `strata-quants` | The file exists. It documents a **different** experiment: weight-quant KL against Q8_0 on `data/quant_quality/v1`, top-64. It does not document RULER cwe or an fp16-KV reference. Citing it as the cwe yardstick will send a reader to the wrong protocol. Cite the research-branch harness docstring, or add a cwe paragraph to QUANTS.md. |
| `strata-kv4` | The directory and README exist. The 8–12% figure is in the README and is versus fp16, not versus int8 (minor 1). The citation target is right. |
| `strata-wp` | The file exists on `KV_QUANT_research`, and §7.12–§7.21 are the right range for the window history. The bibliography says so. The working-tree bench directories it names are empty (M7). |

No citation looks invented. The errors are a wrong SnapKV author, a wrong KIVI axis, a wrong GEAR subtitle, missing authors on three entries, and one vLLM paraphrase that drops the source’s own caveats.

## 8. Number audit

Confirmed against the work package or DETAILS, to the printed digit, unless noted. “WP” is `KV_QUANT_research:docs/KV_QUANT_WORKPACKAGE.md`.

| Paper figure | Source | Status |
| --- | --- | --- |
| int8 8.25 bit, 1,584 MiB; plain 4.50 bit, 864 MiB; window 5.29 bit, 1,016 MiB at 128K, \(B=6{,}144\) | WP §7.19–§7.20; DETAILS | Confirmed as byte model. 1,016 = 864 + 152. Direct VRAM delta at \(B=6{,}144\) not in the §7.19 measurement table (that table measures \(B=512\) and \(B=16{,}384\)). |
| 36% and 568 MiB | 1,584 − 1,016 | Confirmed arithmetic. |
| Runtime table: slots 3,033 / 3,449 / 3,496 / 3,412; prefill 526.1 / 528.0 / 105.0 / 634.0; decode 35.2 / 34.2 / 34.5 / 33.35; hit 0.74 / 0.76 / 0.76 / 0.77 | WP §7.20 | Confirmed. Hardware: RX 9060 XT 16 GB, gfx1200, Windows HIP, Swift IQ3_XXS, engine 0.1.44.1, `--spec 4`. |
| Decode −1.0 tok/s, signs −2.4 / +0.4, int8 36.6 → 33.8 | WP §7.20 | Confirmed. |
| 512/128 prefill advantage not reproduced on needles (246.3 vs 247.7 s, 241.5 vs 251.2 s) | WP §7.20 | Confirmed. Paper does not print the seconds; the qualitative claim matches. |
| cwe-2k: 16,351 positions; fp16 PPL 2.999; int8 99.35% / 0.00503 / 0.0733 / 3.004; plain 98.92% / 0.01791 / 0.2793 / 3.024; 256/64 99.33% / 0.00702 / 0.0835 / 3.003; 512/128 99.39% / 0.00623 / 0.0743 / 2.995 | WP §7.15, §7.16 | Confirmed. |
| 2K paired \(t=-1.92\), 2/8 | WP §7.16 | \(t\) confirmed. Sign-count wording is ambiguous (question 7). |
| cwe-4k: 32,746 positions; int8 99.05% / 0.01688 / 0.1888 / 2.860; plain 98.40% / 0.03182 / 0.5785 / 2.888; 256/64 98.84% / 0.02090 / 0.2811 / 2.858; 512/128 98.95% / 0.02088 / 0.2822 / 2.858 | WP §7.20 | Confirmed. |
| 4K \(t=-3.55\); gap closed 0.01093/0.01494 = 73%; answer \(t=-0.51\); 256/64 vs 512/128 \(t=+0.02\) | WP §7.20 | Confirmed. Sign count 1/8 not in the paper. |
| cwe-128k text 1: 131,063 tokens, 131,062 scored; tail KL 0.00959 / 0.00663 / 0.01587; p99 0.01655 / 0.02977 / 0.03079; top-1 99.73 / 99.73 / 99.51; PPL 2.4052 / 2.4079 / 2.4029; fp16 PPL 2.4149; target top-1 79.74% | WP §7.21; DETAILS | Confirmed for text 1. |
| Region table, including 0.00265 / 0.00444 / 0.01096, whole-segment 0.10715 / 0.13312 / 0.15758, p99 3.65623 / 4.75603 / 5.55046, answer 0.00526 / 0.00861 / 0.01155, 27 positions, 96.30 / 92.59 / 96.30 | WP §7.21 | Confirmed. |
| Single-position KLs 17.3, 17.1, 15.8 | WP §7.21 (positions 3,033, 124,954, 18,995) | Confirmed as harness outputs. Interpretation as full-distribution events is not (M2). |
| Tail \(t=-0.74\); 58% of plain’s tail KL; 31% lower mean than int8 | WP §7.21 | Confirmed for text 1 only (M1). |
| Second text 0.00324 / 0.00437 / 0.00664 and whole-segment 0.06435 / 0.07524 / 0.08834 | DETAILS only | In a cited source. Not in the paper. Raw file not found in the tree. |
| 4K exact fraction 20.3% at \(B=512\); ring 4.9% at 128K with \(B=6{,}144\) | WP §7.20 | Arithmetic confirmed. Interpretation as per-query exact fraction is not (M3). Per-query exact count is 320 (WP §7.21). |
| Break-even “roughly 32K” | WP §7.20 says ~27,600; at 32K, 7.67 bit | Paper is wrong (M5). |
| SHA `A014F504…BA052` | DETAILS | Prefix confirmed. 8/8 unconfirmed. |
| Setup 59/59; refusal “about one second”; A1 post-fix 8,192-token completion; A3 probes pass | — | Unconfirmed. Exit code 2 confirmed (DETAILS). |
| Parity 12/12; needles 18/18; hang 6/6 non-reproduction; window 3×128K clean | WP §7.14, §7.19, §7.20 | Confirmed as the work package’s report. Not re-run here. |
| 0.79 bit/cell above the base, of which 0.75 is \(B\) | Paper formula | Confirmed arithmetic. |
| “8–12% above the 8-bit default” | `2026-09-27-kv-q4` README | The percents are real; the comparator is fp16, on Q2_0, for `q4_0` (minor 1). |
| Windows 11 | Threats | Unconfirmed. Sources say Windows. ROCm 7.1 is in DETAILS. |
| gfx1200, RX 9060 XT 16 GB, Swift 1.5 IQ3_XXS | WP §7.20; DETAILS | Confirmed. CPU (Ryzen 9 3900XT, 64 GB DDR4-3000) is in DETAILS and not in the paper. Not a numerical error. |

---

*End of review. No engine binary was started. No benchmark was run. No number in this review was measured by the reviewer.*
