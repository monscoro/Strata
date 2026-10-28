#!/usr/bin/env python3
"""Why P99 and CVaR99 disagree: the shape of the KL tail, not its size.

G1 found that P99 does not see the certified sink loss while CVaR99 does (both are functions of the same
per-segment `kl256` series).  This tool measures the three things that can explain such a split, on the runs
that already exist - no GPU, no new data:

  1. **Heaviness.**  CVaR99 / P99 per arm, and the share of a segment's KL *sum* that sits in its worst 1 %.
     A good arm is mostly tiny with a few catastrophic positions; a bad arm is uniformly bad.  The two have
     very different tails at the same mean.
  2. **Noise.**  P99 is a single order statistic (the 42nd-worst of 4,095 positions per segment); CVaR99 is
     the mean of the worst 41.  The paired standard deviation across the eight segments says how much of the
     interval width is estimator noise rather than arm difference.
  3. **Location.**  Where the extra loss of one arm over another sits: the head (q < 64, the sinks), the
     extreme positions above the other arm's P99, or the body.

    python tools/kv_tail.py [--fixture bench/fixtures/beyond-kl-calibration.json] [--bench-root DIR]
                            [--arms int8,q4lm32w,s320r0] [--pair a,b] [--kl-topk 256]

The numbers are descriptive: this tool decides nothing.  The acceptance rule of G1 is unchanged and already
applied - `cvar99` fails it (ladder) and is redundant (rho 0.982).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quant_quality as qq  # noqa: E402
from kv_g1 import paired_segments, T7_975  # noqa: E402

R_HEAD = 64   # the sink rows of the reference window geometry


def series(path: str, ref: dict, kl_topk: int) -> dict:
    """{segment: [(pos, kl256)]} - the per-position series, same definition as G1."""
    run = qq.read_run(path)
    out = {}
    for seg, rr in run.items():
        ra = ref.get(seg, {})
        pos = sorted(set(rr) & set(ra))
        if not pos:
            continue
        out[seg] = [(q, qq.kl_between(ra[q][4], rr[q][4], kl_topk)) for q in pos]
    return out


def shape(segs: dict) -> dict:
    """Per-segment tail numbers, averaged over the segments (the unit is the segment, as everywhere else)."""
    rows = []
    for seg, ps in segs.items():
        ks = sorted(k for _, k in ps)
        n = len(ks)
        k_cvar = max(1, int(round(0.01 * n)))
        total = sum(ks)
        head = sum(k for q, k in ps if q < R_HEAD)
        rows.append({
            "n": n,
            "mean": total / n,
            "p99": ks[int(0.99 * (n - 1))],
            "cvar99": sum(ks[-k_cvar:]) / k_cvar,
            "sum_share_top1": sum(ks[-k_cvar:]) / total if total else float("nan"),
            "sum_share_head": head / total if total else float("nan"),
        })
    return {f: sum(r[f] for r in rows) / len(rows) for f in rows[0]}


def excess_over(segs: dict, ref_segs: dict, kl_topk: int) -> dict:
    """How far above the *other* arm's P99 this arm goes: the mass P99 cannot see but CVaR99 can.

    Per segment the threshold is the reference arm's own 99th percentile, so the count answers "did this arm
    put more positions above the same line", and the excess answers "how far".  A diagnostic, not a metric.
    """
    above, mass, n = [], [], 0
    for seg, ps in segs.items():
        rs = ref_segs.get(seg)
        if not rs:
            continue
        rk = sorted(k for _, k in rs)
        thr = rk[int(0.99 * (len(rk) - 1))]
        above.append(sum(1 for _, k in ps if k > thr) / len(ps))
        mass.append(sum(max(0.0, k - thr) for _, k in ps) / len(ps))
        n += 1
    return {"segments": n, "above_rate": sum(above) / n, "excess": sum(mass) / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="bench/fixtures/beyond-kl-calibration.json")
    ap.add_argument("--bench-root")
    ap.add_argument("--kl-topk", type=int, default=256)
    ap.add_argument("--arms", default="int8,q4lm32w,s320r0,q4lm8,q4lm8k")
    ap.add_argument("--pair", help="a,b for the paired comparison (default: s320r0,q4lm32w)")
    ap.add_argument("--json")
    a = ap.parse_args()

    fx = json.load(open(a.fixture, encoding="utf-8"))
    root = pathlib.Path(a.bench_root) if a.bench_root else \
        pathlib.Path(__file__).resolve().parent.parent.parent / "Strata-data" / "bench"
    ref = qq.read_run(str(root / fx["runs"][fx["anchor"]]))

    names = [n for n in a.arms.split(",") if n]
    data = {n: series(str(root / fx["runs"][n]), ref, a.kl_topk) for n in names}

    print(f"anchor {fx['anchor']}, KL support top-{a.kl_topk}, head = q < {R_HEAD}, "
          f"worst 1 % = {max(1, int(round(0.01 * 4095)))} of ~4,095 positions per segment\n")
    print(f"  {'arm':10s} {'mean':>9s} {'p99':>8s} {'cvar99':>8s} {'cvar/p99':>8s} "
          f"{'sum in top 1%':>13s} {'sum in head':>11s}")
    out = {}
    for n in names:
        s = shape(data[n])
        out[n] = s
        print(f"  {n:10s} {s['mean']:9.5f} {s['p99']:8.4f} {s['cvar99']:8.4f} "
              f"{s['cvar99'] / s['p99']:8.1f} {100 * s['sum_share_top1']:12.1f}% "
              f"{100 * s['sum_share_head']:10.1f}%")

    pa, pb = (a.pair.split(",") if a.pair else ("s320r0", "q4lm32w"))
    if pa in data and pb in data:
        print(f"\npaired {pa} - {pb}, over the eight segments (t 97.5 %, df = 7):")
        for label, feat in (("p99", "p99"), ("cvar99", "cvar99"), ("mean", "mean")):
            st = {n: {s: shape({s: data[n][s]}) for s in data[n]} for n in (pa, pb)}
            p = paired_segments(st[pa], st[pb], feat)
            print(f"  {label:7s} {p['mean']:+9.4f} [{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}]  "
                  f"t {p['t']:+6.2f}  sd {p['sd']:.4f}  sem {p['sem']:.4f}  "
                  f"{p['segments_a_worse']}/{p['segments']}")
        ex_a = excess_over(data[pa], data[pb], a.kl_topk)
        ex_b = excess_over(data[pb], data[pa], a.kl_topk)
        print(f"\n  positions above the other arm's P99 (same line for both, per segment):")
        print(f"    {pa:10s} {100 * ex_a['above_rate']:5.2f}% of positions, excess mass {ex_a['excess']:.4f} nats/position")
        print(f"    {pb:10s} {100 * ex_b['above_rate']:5.2f}% of positions, excess mass {ex_b['excess']:.4f} nats/position")
        out["pair"] = {"a": pa, "b": pb, "above": {pa: ex_a, pb: ex_b}}

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
