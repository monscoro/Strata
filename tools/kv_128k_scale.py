#!/usr/bin/env python
"""The 128K scale check: our numbers against the predecessor's reported 128K table.

The predecessor's window sheet reports, for the second 128K text: int8's whole-segment KL
0.06435, p99 0.0165, tail mean (last 4,096) 0.00324, last 1,024 0.00265.  Our int8 arm on the
same text measures the whole segment at 0.06435 -- to the last digit -- and the p99 at 1.6615.

A uniform factor of about a hundred between the two sets of statistics would say: the texts and
the mean agree, and the tail statistics differ in scale, not in substance (the ratios would
survive).  This script computes our own tail statistics on the same definitions and prints the
ratios next to the predecessor's, so the question can be answered with numbers instead of
impressions.

Usage:
  python tools/kv_128k_scale.py --ref <fp16.tsv> --int8 <int8.tsv> [--arm <arm.tsv>]
"""
import argparse
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quant_quality", HERE / "quant_quality.py")
qq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qq)

# the predecessor's reported reading for the second 128K text (docs/paper/KV-Window-Hardening.tex,
# section "Quality at 128K"): int8 and the window arm
PRED = {
    "int8": {"whole": 0.06435, "p99": 0.0165, "tail4096": 0.00324, "last1024": 0.00265},
    "window": {"whole": 0.07524, "p99": 0.0210, "tail4096": 0.00437, "last1024": 0.00444},
}


def stats(ref_run, run, kl_topk):
    series = qq.kl_series(ref_run, run, kl_topk)
    keys = sorted(series)
    vals = [series[k] for k in keys]
    n = len(vals)
    p99 = sorted(vals)[int(0.99 * (n - 1))]
    # the tail statistics are POSITIONAL (the predecessor's wording: "the last 4,096 positions",
    # "the last 1,024"), not the largest values - the two differ by orders of magnitude here
    tail = sum(vals[-4096:]) / min(4096, n)
    last = sum(vals[-1024:]) / min(1024, n)
    # where the error mass sits: the share of the total that the top-k positions carry
    desc = sorted(vals, reverse=True)
    total = sum(vals) or 1.0
    shares = {k: sum(desc[:k]) / total for k in (10, 100, 1000, 10000)}
    return {"whole": sum(vals) / n, "p99": p99, "tail4096": tail, "last1024": last,
            "shares": shares}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--arm")
    ap.add_argument("--kl-topk", type=int, default=qq.TOPK)
    args = ap.parse_args()

    ref = qq.read_run(args.ref)
    out = {}
    out["int8"], n = stats(ref, qq.read_run(args.int8), args.kl_topk)
    if args.arm:
        out["window"], _ = stats(ref, qq.read_run(args.arm), args.kl_topk)

    print(f"positions: {n}")
    print()
    print(f"{'statistic':>12} {'ours (int8)':>14} {'pred (int8)':>14} {'factor':>9} | "
          f"{'ours (window)':>14} {'pred (window)':>15} {'factor':>9}")
    print("-" * 96)
    for k in ("whole", "p99", "tail4096", "last1024"):
        row = f"{k:>12} {out['int8'][k]:14.5f} {PRED['int8'][k]:14.5f} {out['int8'][k]/PRED['int8'][k]:9.3f}"
        if "window" in out:
            row += (f" | {out['window'][k]:14.5f} {PRED['window'][k]:15.5f} "
                    f"{out['window'][k]/PRED['window'][k]:9.3f}")
        print(row)
    print()
    print("where the error mass sits (the share of the total KL the top-k positions carry):")
    for name, st in out.items():
        sh = st["shares"]
        print(f"  {name:>8}: top 10 {100*sh[10]:5.1f} %   top 100 {100*sh[100]:5.1f} %   "
              f"top 1,000 {100*sh[1000]:5.1f} %   top 10,000 {100*sh[10000]:5.1f} %")
    print()
    print("Reading: a factor near 1 on the whole segment with a factor near 100 on the tail")
    print("statistics would mean the texts and the means agree and the tail statistics differ")
    print("in scale.  Equal factors on every statistic would mean a pure unit difference.")


if __name__ == "__main__":
    main()
