#!/usr/bin/env python
"""The 128K gate, mechanically: K2 (whole segment) and K3 (p99) against int8, per text.

The ladder's criteria for the 128K texts (section 7 of docs/paper/Ansatz_Improvements.md):

  K2  the whole-segment KL (against the fp16 reference) must not be worse than int8's
  K3  the p99 of the KL (against the fp16 reference) must not be worse than int8's

and, as the calibration was nominated by the p99 and the top-1, the script also reports the
paired tail (a tail set defined once, by int8's own KL, so it is the same for every arm) and the
top-1 with an exact McNemar test between the two 16/16 arms.

Everything here is a comparison against the fp16 reference; the anchor for the *paired* numbers is
fp16 as well (the difference between two arms is the difference of their KLs against fp16 - the
int8-anchored series measure the KL between the arms, which is a different quantity).

Usage:
  python tools/kv_gate_128k.py --ref <fp16.tsv> --int8 <int8.tsv> --arm <q4lm32w.tsv> \
      [--arm-center <q4lm32w+center.tsv>] [--label t1]
"""
import argparse
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quant_quality", HERE / "quant_quality.py")
qq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qq)


def kl_stats(ref_run, run, kl_topk):
    """the per-position KL series against the fp16 reference, with the arm's own p99 and top-1"""
    series = qq.kl_series(ref_run, run, kl_topk)
    vals = sorted(series.values())
    p99 = vals[int(0.99 * (len(vals) - 1))]
    top1 = 0
    n = 0
    for seg, d in ref_run.items():
        for pos, rec in d.items():
            other = run.get(seg, {}).get(pos)
            if other is None:
                continue
            n += 1
            if rec[4][0][0] == other[4][0][0]:
                top1 += 1
    return series, p99, top1, n


def paired(a, b):
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    sem = sd / math.sqrt(len(d))
    return m, (m / sem if sem else float("nan")), m - 1.96 * sem, m + 1.96 * sem, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--arm-center")
    ap.add_argument("--label", default="text")
    ap.add_argument("--kl-topk", type=int, default=qq.TOPK)
    args = ap.parse_args()

    ref = qq.read_run(args.ref)
    runs = {"int8": qq.read_run(args.int8), "arm": qq.read_run(args.arm)}
    if args.arm_center:
        runs["arm+center"] = qq.read_run(args.arm_center)

    stats = {}
    for name, r in runs.items():
        stats[name] = kl_stats(ref, r, args.kl_topk)

    print(f"=== {args.label}: K2 (whole segment) and K3 (p99) against int8, fp16 as the reference ===")
    print(f"{'arm':>12} {'mean KL':>10} {'p99':>9} {'top-1':>9} | {'d vs int8 (mean)':>16} {'t':>7} "
          f"{'95 % CI':>22} | {'K2':>4} {'K3':>4}")
    print("-" * 116)
    verdict = {}
    for name in ("int8", "arm", "arm+center"):
        if name not in stats:
            continue
        series, p99, top1, n = stats[name]
        mean_kl = sum(series.values()) / len(series)
        if name == "int8":
            print(f"{name:>12} {mean_kl:10.5f} {p99:9.4f} {100.0*top1/n:8.2f} % | "
                  f"{'---':>16} {'---':>7} {'---':>22} | {'-':>4} {'-':>4}")
            continue
        m, t, lo, hi, _ = paired(series, stats["int8"][0])
        int8_mean = sum(stats["int8"][0].values()) / len(stats["int8"][0])
        k2 = "yes" if mean_kl <= int8_mean else "NO"
        k3 = "yes" if p99 <= stats["int8"][1] else "NO"
        verdict[name] = (k2, k3, m, t, lo, hi)
        print(f"{name:>12} {mean_kl:10.5f} {p99:9.4f} {100.0*top1/n:8.2f} % | {m:+16.5f} {t:+7.2f} "
              f"{f'[{lo:+.5f}, {hi:+.5f}]':>22} | {k2:>4} {k3:>4}")

    # the tail, defined once by int8's own KL, and the top-1 between the two 16/16 arms
    keys = sorted(set.intersection(*[set(s[0]) for s in stats.values()]))
    tail_n = max(1, int(round(len(keys) * 0.01)))
    tail = set(sorted(keys, key=lambda k: stats["int8"][0][k], reverse=True)[:tail_n])
    print()
    print(f"the tail (the 1 % of positions where int8's KL is largest, {tail_n} positions, "
          f"defined once and arm-independently):")
    for name in stats:
        s = stats[name][0]
        tm = sum(s[k] for k in tail) / len(tail)
        print(f"  {name:>12} mean KL {tm:9.5f}   d vs int8 {tm - sum(stats['int8'][0][k] for k in tail)/len(tail):+.5f}")

    if "arm+center" in stats:
        a, b = stats["arm"][0], stats["arm+center"][0]
        a_only = b_only = 0
        for seg, d in ref.items():
            for pos, rec in d.items():
                ra = runs["arm"].get(seg, {}).get(pos)
                rb = runs["arm+center"].get(seg, {}).get(pos)
                if not ra or not rb:
                    continue
                if rec[4][0][0] == ra[4][0][0] and rec[4][0][0] != rb[4][0][0]:
                    a_only += 1
                if rec[4][0][0] == rb[4][0][0] and rec[4][0][0] != ra[4][0][0]:
                    b_only += 1
        nn = a_only + b_only
        if nn:
            k = min(a_only, b_only)
            tail = sum(math.comb(nn, i) for i in range(k + 1))
            p = min(1.0, (2 * tail) / (2 ** nn))
            print()
            print(f"top-1, arm against arm+center: arm right where arm+center wrong: {a_only}, "
                  f"the other way: {b_only}  ->  exact McNemar p = {p:.4f}")
    print()
    for name, (k2, k3, *_rest) in verdict.items():
        print(f"{args.label} / {name}: K2 (whole segment not behind int8) {k2}, "
              f"K3 (p99 not behind int8) {k3}")


if __name__ == "__main__":
    main()
