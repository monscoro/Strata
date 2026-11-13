#!/usr/bin/env python
"""The 128K gates, mechanically (bench/results/2026-11-13-128k-fair/README.md).

Applies exactly the numbers fixed before the run:

  G1  q4lm32w+C - R2b  upper 95 % bound < +0.00254 and point <= +0.00127
  G2  R2b - R2a        point < 0 and the 95 % interval excludes 0   (the calibration prediction)
  G3  q4u32 - R2b      as G1
  P1  q4lm32w+C - q4lm32w must reproduce the pre-run test: +0.0031433 (t 2.0518)

Two intervals are computed for every pair, because `t1` is ONE segment and there is no
segment-t distribution:

  * per position (t over 131,060 positions),
  * a block bootstrap over 32 fixed blocks of 4,096 positions (10,000 resamples, seed 20261113,
    percentile interval) -- the only form that catches the serial correlation of the KL series.

The gate uses the WIDER upper bound of the two.  Nothing here is chosen after seeing the numbers.

Usage:
  python bench/results/2026-11-13-128k-fair/eval-128k-fair.py [--json gates.json]
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREV = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\d1")
NEW = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-13-128k-fair\arms")

spec = importlib.util.spec_from_file_location("quant_quality", REPO / "tools" / "quant_quality.py")
qq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qq)

# fixed before the run (README.md, "Die Endpunkte")
BOUND = 0.00254     # parity_bound of the reference pair (int8 - q4lm32w), rounded up
POINT = 0.00127     # half the bound

P1_MEAN = 0.003143258885202661   # q4lm32w - q4lm32w+C: the calibration is BETTER by this much
P1_T = 2.0518169057068594

BLOCK = 4096
RESAMPLES = 10000
SEED = 20261113

ARMS = {
    "fp16": PREV / "t1-fp16.tsv",
    "R1": PREV / "t1-int8.tsv",
    "R2a": NEW / "t1-int8-rot.tsv",
    "R2b": NEW / "t1-int8-rot-center.tsv",
    "q4lm32w": PREV / "t1-q4lm32w.tsv",
    "q4lm32w+C": PREV / "t1-q4lm32w-center.tsv",
    "q4lm8v-win": PREV / "t1-q4lm8v-win.tsv",
    "q4lm8v-win+C": PREV / "t1-q4lm8v-win-center.tsv",
    "q4u32": NEW / "t1-q4u32.tsv",
}


def tci(d):
    """mean, t, 95 % CI per position (the form d1/paired-all6.json uses)."""
    d = np.asarray(d, dtype=np.float64)
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    sem = sd / math.sqrt(len(d)) if len(d) > 1 else 0.0
    t = m / sem if sem else (0.0 if m == 0.0 else float("nan"))
    return m, t, m - 1.96 * sem, m + 1.96 * sem


def block_boot(d):
    """The percentile interval over 32 fixed blocks of 4,096 positions."""
    d = np.asarray(d, dtype=np.float64)
    nblk = math.ceil(len(d) / BLOCK)
    means = np.empty(nblk)
    for i in range(nblk):
        chunk = d[i * BLOCK:(i + 1) * BLOCK]
        means[i] = chunk.mean()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, nblk, size=(RESAMPLES, nblk))
    boot = means[idx].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), nblk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE / "gates.json"))
    args = ap.parse_args()

    missing = [k for k, p in ARMS.items() if not Path(p).exists()]
    if missing:
        print(f"missing arms: {', '.join(missing)}")
        print("(run run-128k-fair.py first; the six old arms live in d1)")
        return 1

    ref = qq.read_run(str(ARMS["fp16"]))
    series = {}
    for name, path in ARMS.items():
        series[name] = qq.kl_series(ref, qq.read_run(str(path)), qq.TOPK)
        m = sum(series[name].values()) / len(series[name])
        print(f"{name:14s} mean KL {m:.5f}  positions {len(series[name])}")

    keys = sorted(set.intersection(*[set(s) for s in series.values()]))
    print(f"\npaired over {len(keys)} positions, fp16 as the reference\n")
    print(f"{'difference':22s} {'mean':>10s} {'t':>7s} {'CI per position':>24s} "
          f"{'CI block bootstrap':>24s}")
    diffs = {}
    for a, b in (("R2a", "R1"), ("R2b", "R1"), ("R2b", "R2a"),
                 ("q4lm32w+C", "R2b"), ("q4u32", "R2b"),
                 ("q4lm32w", "q4lm32w+C"), ("q4lm8v-win", "q4lm32w")):
        d = [series[a][k] - series[b][k] for k in keys]
        m, t, lo, hi = tci(d)
        blo, bhi, nblk = block_boot(d)
        diffs[f"{a} - {b}"] = {"mean": m, "t": t, "ci_low": lo, "ci_high": hi,
                               "boot_low": blo, "boot_high": bhi, "blocks": nblk}
        print(f"{a + ' - ' + b:22s} {m:+10.5f} {t:+7.2f} {f'[{lo:+.5f}, {hi:+.5f}]':>24s} "
              f"{f'[{blo:+.5f}, {bhi:+.5f}]':>24s}")

    def upper(key):
        d = diffs[key]
        return max(d["ci_high"], d["boot_high"])

    gates = {}
    for g, key in (("G1", "q4lm32w+C - R2b"), ("G3", "q4u32 - R2b")):
        d = diffs[key]
        up = upper(key)
        ok = (up < BOUND) and (d["mean"] <= POINT)
        gates[g] = {"pair": key, "mean": d["mean"], "upper": up, "bound": BOUND,
                    "point_bound": POINT, "passed": bool(ok)}
    d = diffs["R2b - R2a"]
    up = upper("R2b - R2a")
    gates["G2"] = {"pair": "R2b - R2a", "mean": d["mean"], "upper": up,
                   "passed": bool(d["mean"] < 0.0 and up < 0.0)}
    d = diffs["q4lm32w - q4lm32w+C"]
    gates["P1"] = {"pair": "q4lm32w - q4lm32w+C", "mean": d["mean"], "t": d["t"],
                   "expected_mean": P1_MEAN, "expected_t": P1_T,
                   "passed": bool(abs(d["mean"] - P1_MEAN) < 1e-6 and abs(d["t"] - P1_T) < 1e-3)}

    print("\nDie Gates, wie sie vor dem Lauf festlagen:")
    for g in ("G1", "G2", "G3", "P1"):
        v = gates[g]
        if g == "P1":
            print(f"  {g}  {v['pair']}: {v['mean']:+.5f} (t {v['t']:+.2f}), erwartet "
                  f"{v['expected_mean']:+.5f} (t {v['expected_t']:+.2f}) -> "
                  f"{'BESTANDEN' if v['passed'] else 'NICHT BESTANDEN'}")
        elif g == "G2":
            print(f"  {g}  {v['pair']}: {v['mean']:+.5f}, obere Grenze {v['upper']:+.5f} "
                  f"(muss < 0 sein) -> {'BESTANDEN' if v['passed'] else 'NICHT BESTANDEN'}")
        else:
            print(f"  {g}  {v['pair']}: Punkt {v['mean']:+.5f} (Grenze {POINT:+.5f}), "
                  f"obere Grenze {v['upper']:+.5f} (muss < {BOUND:+.5f}) -> "
                  f"{'BESTANDEN' if v['passed'] else 'NICHT BESTANDEN'}")

    # the tail, defined once by R2a's own KL, and the top-1 with an exact McNemar test
    tail_n = max(1, int(round(len(keys) * 0.01)))
    tail = sorted(keys, key=lambda k: series["R2a"][k], reverse=True)[:tail_n]
    print(f"\nDas Tail (1 % der Positionen mit der groessten KL von R2a, {tail_n} Positionen):")
    base = sum(series["R2a"][k] for k in tail) / len(tail)
    for name in ("R1", "R2b", "q4lm32w", "q4lm32w+C", "q4u32"):
        tm = sum(series[name][k] for k in tail) / len(tail)
        print(f"  {name:12s} Tail-Mittel {tm:9.5f}   d gegen R2a {tm - base:+.5f}")

    def top1(name, path):
        run = qq.read_run(str(path))
        right = n = 0
        for seg, dd in ref.items():
            for pos, rec in dd.items():
                other = run.get(seg, {}).get(pos)
                if other is None:
                    continue
                n += 1
                right += rec[4][0][0] == other[4][0][0]
        return right, n

    t1 = {name: top1(name, ARMS[name]) for name in ("R1", "R2a", "R2b", "q4lm32w", "q4lm32w+C",
                                                   "q4u32")}
    print("\nTop-1 gegen fp16:")
    for name, (right, n) in t1.items():
        print(f"  {name:12s} {100.0 * right / n:.2f} %")

    def mcnemar(a, b):
        ra, rb = qq.read_run(str(ARMS[a])), qq.read_run(str(ARMS[b]))
        a_only = b_only = 0
        for seg, dd in ref.items():
            for pos, rec in dd.items():
                x, y = ra.get(seg, {}).get(pos), rb.get(seg, {}).get(pos)
                if not x or not y:
                    continue
                truth = rec[4][0][0]
                if truth == x[4][0][0] and truth != y[4][0][0]:
                    a_only += 1
                if truth == y[4][0][0] and truth != x[4][0][0]:
                    b_only += 1
        nn = a_only + b_only
        if not nn:
            return a_only, b_only, float("nan")
        # exact two-sided McNemar in the log domain: 2**nn overflows for nn ~ 2000
        k = min(a_only, b_only)
        logs = [math.lgamma(nn + 1) - math.lgamma(i + 1) - math.lgamma(nn - i + 1)
                for i in range(k + 1)]
        top = max(logs)
        log_sum = top + math.log(sum(math.exp(v - top) for v in logs))
        log_p = math.log(2.0) + log_sum - nn * math.log(2.0)
        return a_only, b_only, min(1.0, math.exp(log_p))

    print("\nTop-1, exakter McNemar (a richtig, wo b falsch | b richtig, wo a falsch -> p):")
    for a, b in (("q4lm32w+C", "R2b"), ("q4u32", "R2b"), ("R2b", "R2a"), ("R2b", "R1")):
        ao, bo, p = mcnemar(a, b)
        print(f"  {a} gegen {b}: {ao} | {bo}  ->  p = {p:.4f}")

    out = {"bound": BOUND, "point_bound": POINT, "positions": len(keys),
           "arms": {k: sum(v.values()) / len(v) for k, v in series.items()},
           "differences": diffs, "gates": gates,
           "top1": {k: v[0] / v[1] for k, v in t1.items()}}
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nJSON: {args.json}")
    return 0 if all(gates[g]["passed"] for g in ("G1", "G2", "G3", "P1")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
