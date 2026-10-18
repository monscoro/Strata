"""Read kvab.jsonl (tools/ab_engine.py --log) and print the per-arm medians as markdown.

Decode numbers move run to run (adaptive expert tier, OS), so the value is the median over the rounds and the
spread, not a single run. The window arms carry the WP-2a probe overhead (a full fp16 twin of *every* cell is
appended next to the q4 blocks, plus one extra launch per append), so their decode number is a PROBE cost, not
the delivery cost - that is written into the report, not into this script.

    python analyze.py kvab.jsonl [--markdown]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--markdown", action="store_true", help="print a markdown table instead of plain text")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.log) if l.strip()]
    arms, reqs = [], []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
        if r["req"] not in reqs:
            reqs.append(r["req"])

    piv: dict = defaultdict(lambda: defaultdict(lambda: {"dec": [], "prom": [], "pn": [], "n": []}))
    for r in rows:
        c = piv[r["arm"]][r["req"]]
        if r["decode"]:
            c["dec"].append(r["decode"])
        if r["prompt"]:
            c["prom"] = [r["prompt"]]          # the long prompt is the only meaningful prompt number
            c["pn"].append(r["prompt_n"])
        c["n"].append(r["n"])

    def med(v):
        return statistics.median(v) if v else float("nan")

    base = "fp16" if "fp16" in arms else arms[0]
    ref = med(piv[base]["long"]["dec"])
    order = sorted(arms, key=lambda x: -med(piv[x]["long"]["dec"]))

    if a.markdown:
        print("| Arm | Decode tok/s (Median der Runden) | Spanne | vs. fp16 | Prompt tok/s (long) |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for arm in order:
            d = piv[arm]["long"]["dec"]
            p = piv[arm]["long"]["prom"]
            print("| `%s` | %.1f | %.1f–%.1f | %+.1f %% | %.0f |" % (
                arm, med(d), min(d) if d else float("nan"), max(d) if d else float("nan"),
                100.0 * (med(d) / ref - 1.0) if ref else float("nan"), med(p)))
    else:
        for arm in order:
            d = piv[arm]["long"]["dec"]
            p = piv[arm]["long"]["prom"]
            print("%-16s decode %6.1f  (%s)  vs fp16 %+6.1f %%   prompt %7.1f" % (
                arm, med(d), " ".join("%.1f" % x for x in d),
                100.0 * (med(d) / ref - 1.0) if ref else float("nan"), med(p)))
    print("\nrounds seen: %d, arms: %d, rows: %d" % (len(set(r["round"] for r in rows)), len(arms), len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
