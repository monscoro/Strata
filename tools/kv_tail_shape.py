#!/usr/bin/env python
"""The shape of the error distribution: is the tail fat, and where does the mass sit?

The 128K int8 arm reproduces the predecessor's whole-segment mean exactly (0.06435) and has a p99
of 1.66 - a factor of a hundred above the published p99.  That is not a scale difference but a
shape statement: the error mass of a long context concentrates in a few hundred positions.

This script measures the shape, so the claim "fat tail" is a number and not an adjective:
  * the concentration: the share of the total error the top-k positions carry;
  * the effective number of positions: 1 / H of the normalized error mass (how many positions
    would carry the same mass if it were spread evenly);
  * the quantile ratios (p50/p90/p99 against the mean);
  * the excess kurtosis of the per-position KL distribution;
  * where the heavy positions sit along the text.

Usage:
  python tools/kv_tail_shape.py --ref <fp16.tsv> --int8 <int8.tsv> [--arm <arm.tsv>] [--block 4096]
"""
import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quant_quality", HERE / "quant_quality.py")
qq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qq)


def shape(vals):
    v = np.asarray(vals, dtype=np.float64)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std())
    desc = np.sort(v)[::-1]
    total = float(desc.sum()) or 1.0
    shares = {k: float(desc[:k].sum() / total) for k in (10, 100, 1000, 10000)}
    # the effective number of positions: exp(H) of the normalized error mass (how many positions
    # would carry the same mass if it were spread evenly)
    p = desc / total
    h = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    # the quantiles
    s = np.sort(v)
    q = {f"p{q}": float(s[min(n - 1, int(q / 100 * (n - 1)))]) for q in (50, 90, 99, 99.9)}
    # the excess kurtosis of the per-position KL distribution
    kurt = float(((v - mean) ** 4).mean() / (sd ** 4)) - 3.0 if sd > 0 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "shares": shares, "eff": math.exp(h), "q": q, "kurt": kurt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--arm")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--kl-topk", type=int, default=qq.TOPK)
    args = ap.parse_args()

    ref = qq.read_run(args.ref)
    arms = {"int8": qq.read_run(args.int8)}
    if args.arm:
        arms["arm"] = qq.read_run(args.arm)

    series = {name: qq.kl_series(ref, r, args.kl_topk) for name, r in arms.items()}
    keys = sorted(set.intersection(*[set(s) for s in series.values()]))

    print(f"positions: {len(keys)}")
    print()
    for name, s in series.items():
        vals = [s[k] for k in keys]
        st = shape(vals)
        print(f"=== {name} ===")
        print(f"  mean {st['mean']:.5f}   sd {st['sd']:.5f}   excess kurtosis {st['kurt']:.1f}")
        print(f"  quantiles: p50 {st['q']['p50']:.5f}   p90 {st['q']['p90']:.4f}   "
              f"p99 {st['q']['p99']:.4f}   p99.9 {st['q']['p99.9']:.3f}")
        print(f"  p99 / mean = {st['q']['p99']/st['mean']:.1f}    "
              f"p99.9 / mean = {st['q']['p99.9']/st['mean']:.0f}")
        print(f"  error mass: top 10 {100*st['shares'][10]:5.1f} %   top 100 {100*st['shares'][100]:5.1f} %   "
              f"top 1,000 {100*st['shares'][1000]:5.1f} %   top 10,000 {100*st['shares'][10000]:5.1f} %")
        print(f"  effective number of positions (exp(H) of the normalized mass): {st['eff']:.0f} "
              f"of {st['n']} ({100*st['eff']/st['n']:.2f} %)")
        print()

    # where the heavy positions sit
    name = "int8"
    s = series[name]
    order = sorted(keys, key=lambda k: s[k], reverse=True)
    top = order[:1000]
    per_block = {}
    for k in top:
        per_block[k[1] // args.block] = per_block.get(k[1] // args.block, 0) + 1
    print(f"where the top-1,000 positions of {name} sit (per {args.block} positions):")
    line = []
    for b in sorted(per_block):
        line.append(f"{b}:{per_block[b]}")
    print("  " + "  ".join(line))
    print()
    print("Reading: an effective number of positions far below 1 % of the corpus is the fat tail in")
    print("numbers: the mean KL is carried by a few hundred positions, so the mean is a statement")
    print("about those positions and the tail statistics are the ones that describe the codec.")


if __name__ == "__main__":
    main()
