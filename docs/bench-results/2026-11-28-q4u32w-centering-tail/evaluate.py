#!/usr/bin/env python3
"""Apply the rule fixed in README.md section 3 to the three arms' outputs.

Reads what run-center.py produced:
  t1-fp16.tsv, t1-q4u32w-noc.tsv, t1-q4u32w-c.tsv
  compare-fp16.json  (per-arm mean/p99 KL against fp16)
  paired.json        (per-position paired differences between the arms)
and prints T1/T2 and the decision.  It changes nothing it reads.

T1 (primary, tail):  p99(noc) - p99(c) > 0  AND  p99(c) <= 0.95 * 1.983749  (the A/B's q4u32w p99,
                     the number the protocol fixes the 5 % against: 1.885).
T2 (guard, mean):    the upper 95 % bound of Delta = KL(c) - KL(noc) <= +0.0010 nats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-28-q4u32w-centering-tail")
P99_NOC_AB = 1.983749          # A/B 2026-11-25, q4u32w without calibration (README section 2)
T1_REL = 0.05                  # the pre-registered 5 %
T2_UPPER = 0.0010              # nats


def arm_stats(compare: Path, path: Path) -> dict:
    d = json.loads(compare.read_text(encoding="utf-8"))
    return d[str(path)]["all"]


def main() -> int:
    ap = argparse.ArgumentParser(description="apply README section 3; --compare/--paired name the outputs of "
                                             "`quant_quality.py compare --paired`")
    ap.add_argument("--compare", type=Path, default=DATA / "compare-fp16.json")
    ap.add_argument("--paired", type=Path, default=DATA / "paired.json")
    a = ap.parse_args()

    noc = DATA / "t1-q4u32w-noc.tsv"
    c = DATA / "t1-q4u32w-c.tsv"
    for p in (noc, c, a.compare, a.paired):
        if not p.exists():
            print(f"missing: {p}")
            return 1

    a_noc, a_c = arm_stats(a.compare, noc), arm_stats(a.compare, c)
    paired = json.loads(a.paired.read_text(encoding="utf-8"))["positions"]
    key = f"{noc.name} - {c.name}"
    if key not in paired:
        print(f"paired.json has no pair {key!r}; keys: {list(paired)}")
        return 1
    p = paired[key]

    print(f"{'arm':16s} {'positions':>9s} {'mean KL':>9s} {'KL p99':>9s} {'Top-1':>8s}")
    for tag, a in (("q4u32w-noc", a_noc), ("q4u32w-c", a_c)):
        print(f"{tag:16s} {a['positions']:9d} {a['kl']:9.6f} {a['kl_p99']:9.6f} {100 * a['top1']:7.3f}%")

    gain = a_noc["kl_p99"] - a_c["kl_p99"]
    rel = gain / P99_NOC_AB
    t1 = gain > 0 and a_c["kl_p99"] <= (1.0 - T1_REL) * P99_NOC_AB

    # p is KL(noc) - KL(c); Delta = KL(c) - KL(noc) = -p
    d_mean, d_lo, d_hi = -p["mean"], -p["ci95_high"], -p["ci95_low"]
    t2 = d_hi <= T2_UPPER

    print()
    print(f"T1  p99(noc) - p99(c) = {gain:+.6f}  ({100 * rel:+.2f} % of the A/B's {P99_NOC_AB})")
    print(f"    rule: gain > 0 and p99(c) <= {(1.0 - T1_REL) * P99_NOC_AB:.4f}  -> {'PASS' if t1 else 'FAIL'}")
    print(f"T2  Delta = KL(c) - KL(noc) = {d_mean:+.6f} nats, 95 % [{d_lo:+.6f}, {d_hi:+.6f}]")
    print(f"    rule: upper bound <= {T2_UPPER:+.4f}  -> {'PASS' if t2 else 'FAIL'}")
    print()
    print(f"paired n={p['n']}  t={p['t']:+.2f}  parity bound={p['parity_bound']:.5f}  "
          f"segments worse {p['segments_a_worse']}/{p['segments']}")
    print()
    if t1 and t2:
        print("decision: q4u32w + C goes into the PR (README section 3).")
    elif t2:
        print("decision: only T2 passes - centering stays out; the tail lever is exhausted at this codec.")
    else:
        print("decision: neither T1 nor T2 - record it and call the lever spent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
