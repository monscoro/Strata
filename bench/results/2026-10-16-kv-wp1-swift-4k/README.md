# WP-1 auf Swift, Power-Top-up: cwe-4k (die zwei Gleichstände)

Sibling: `../2026-10-16-kv-wp1-swift/` (cwe-2k: Bit tot, zwei Ties). Design: §5.5; Power-Regel §8.
Der Bit-Arm (`q4lm8v`) läuft hier nicht mehr - bei 2K entschieden verloren (+51 %, t −3,60).

- **Where:** one PC, RX 9060 XT (gfx1200, RDNA4, 16 GB), 63.9 GB RAM, HIP build of 2026-10-16
  (commit `6fc7b2f1`, kvstats parked/env-off).
- **What ran:** Qwen3.8-Flash-Next, **Swift IQ3_XXS** weight pack, weights fixed, only `--kv` changed.
- **Test:** RULER CWE at 4K, 8 segments, ~4,096 tokens each, `data/quant_quality/cwe-4k/`.
  Reference: the `fp16` arm of the same run (PPL 2.857). Segment times ~138-151 s.
- **Incident:** the first `q4lm32nr` attempt died silently after s00 (second silent death on this arm
  in two days, both early, no TDR/crash log, s00 times unremarkable); attempt 3 passed all 8 clean.
  2 of 4 norot runs died, 0 of 11 other runs - intermittent driver instability, no arm pattern that
  survives the two clean 8/8 norot passes. Noted, not chased.

## Whole segment (32,746 positions)

| arm | top-1 | mean KL | KL / `q4lm32` | KL p99 | PPL | vs fp16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp16` | - | 0 | - | 0 | 2.857 | - |
| `q4lm32` | 98.74% | 0.02512 | 1.00x | 0.3712 | 2.864 | +0.24% |
| `q4lm32t` (K max/V rms) | 98.74% | 0.02483 | 0.99x | 0.3764 | 2.862 | +0.17% |
| `q4lm32nr` (no Hadamard) | 98.38% | 0.02606 | 1.04x | 0.4088 | 2.876 | +0.67% |

## Paired over the 8 segments

| pair | ΔKL whole (t, up) | ΔKL answer (t, up) |
| --- | --- | --- |
| `q4lm32` − `q4lm32t` | +0.00029 (+0.21, 4/8) | −0.00082 (−0.23, 4/8) |
| `q4lm32` − `q4lm32nr` | −0.00094 (−1.31, 3/8) | −0.00063 (−0.17, 5/8) |
| `q4lm32t` − `q4lm32nr` | −0.00123 (−0.88, 4/8) | +0.00020 (+0.09, 3/8) |

McNemar (270 rows): all p ≥ 0.25.

## Verdict (2K + 4K, 16 segments)

- **Treatment: toter Gleichstand auf beiden Längen** (beide Male t +0.21 ganz, Antwortregion flach).
  V-RMS kostet nichts und kauft nichts. Kein Mandat für WP-2 - aber auch kein Veto: falls das Layout
  die RMS-Regel gratis mitnehmen kann, schadet sie nicht.
- **No-rotation: Gleichstand** (2K t +0.21, 4K t −1.31 ganz / −0.17 Antwort - nirgends signifikant).
  Die Hadamard-Transformation trägt bei 4-Bit-Lloyd auf Swift nichts Messbares. Produktionsnutzen wäre
  3 gesparte FWHTs pro Token - WP-2-/WP-4-Sache, kein WP-1-Urteil.
- **Machtfrage (§8):** 16 Segmente, alle |t| < 1,4 bei den Ties; McNemar kann prinzipiell nicht auflösen
  (98-99 % Antwort-Übereinstimmung). Das ist ein Urteil als Negativbefund: kein Arm rechtfertigt
  Komplexität. Die 87-Segment-Regel würde daran nichts ändern - sie könnte nur ein kleines |t| in ein
  signifikantes verwandeln, keinen praktisch relevanten Effekt herbeizaubern.
