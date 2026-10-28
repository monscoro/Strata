"""Paired decode-probe scoring: KL(fp16 || arm) per prompt, from `quant_quality.py prompt` dumps.

`pcompare` prints mean/median/max over the prompts, which is not enough to decide anything: the 32
prompts are 8 segments x 4 nested lengths, so the spread has to be paired, and the segment balance
has to be visible. This reads the same `first.*` files, computes KL over the whole vocabulary, and
prints the per-run means plus the paired difference against one or more base runs.

Usage:
  python tools/kv_decode.py --ref <fp16 dir> --run <int8 dir> --run <candidate dir> [--base int8]
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np


def logp(d: pathlib.Path, i: int) -> np.ndarray:
    x = np.fromfile(d / f"first.{i}", dtype=np.float32).astype(np.float64)
    x -= x.max()
    return x - np.log(np.exp(x).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--base", action="append", default=None,
                    help="run name(s) to compare against; default: the first --run")
    a = ap.parse_args()

    ref = pathlib.Path(a.ref)
    requests = json.loads((ref / "index.json").read_text())["requests"]
    n = len(requests)

    ref_lp = [logp(ref, i) for i in range(n)]
    ref_p = [np.exp(x) for x in ref_lp]
    ref_top = [int(x.argmax()) for x in ref_lp]

    kl = {}
    top1 = {}
    print(f"reference {ref.name}: {n} prompts "
          f"({len({r['segment'] for r in requests})} segments x "
          f"{len({r['tokens'] for r in requests})} lengths)")
    print(f"{'run':14s} {'prompts':>7s} {'mean KL':>9s} {'median':>9s} {'max':>8s} {'top-1':>7s}")
    for run in a.run:
        d = pathlib.Path(run)
        vals, agree = [], 0
        for i in range(n):
            if not (d / f"first.{i}").exists():
                continue
            lp = logp(d, i)
            vals.append(float((ref_p[i] * (ref_lp[i] - lp)).sum()))
            agree += int(ref_top[i] == int(lp.argmax()))
        kl[d.name] = vals
        s = sorted(vals)
        print(f"{d.name:14s} {len(vals):7d} {sum(vals) / len(vals):9.5f} "
              f"{s[len(s) // 2]:9.5f} {s[-1]:8.5f} {agree:5d}/{len(vals)}")
        top1[d.name] = agree

    bases = a.base or [pathlib.Path(a.run[0]).name]
    for base in bases:
        print(f"\npaired per prompt against {base} (negative = the run is better)")
        for run in a.run:
            name = pathlib.Path(run).name
            if name == base:
                continue
            d = [x - y for x, y in zip(kl[name], kl[base])]
            m = sum(d) / len(d)
            sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
            sem = sd / math.sqrt(len(d))
            seg = {}
            for i, x in enumerate(d):
                seg.setdefault(requests[i]["segment"], []).append(x)
            worse = sum(1 for s in seg if sum(seg[s]) / len(seg[s]) > 0)

            # The 32 prompts are 8 segments x 4 *nested* lengths, so they are not 32 independent units.
            # Cluster on the segment: average within each segment, then t with df = segments - 1.
            # This is the pre-registered unit (plan §7.4: never resample positions/prompts inside a cluster).
            per_seg = [sum(seg[s]) / len(seg[s]) for s in seg]
            ns = len(per_seg)
            ms = sum(per_seg) / ns
            sds = math.sqrt(sum((x - ms) ** 2 for x in per_seg) / (ns - 1))
            sems = sds / math.sqrt(ns)
            tcrit = 2.365 if ns == 8 else 1.96   # df = 7 for 8 segments
            print(f"  {name} - {base}\n"
                  f"    per prompt   {m:+.5f}  sd {sd:.4f}  t {m / sem:+.2f}  "
                  f"95% [{m - 1.96 * sem:+.5f}, {m + 1.96 * sem:+.5f}]  "
                  f"(one-sided bound at parity {1.645 * sem:.5f})\n"
                  f"    per segment  {ms:+.5f}  sd {sds:.4f}  t {ms / sems:+.2f}  "
                  f"95% [{ms - tcrit * sems:+.5f}, {ms + tcrit * sems:+.5f}]  "
                  f"worse in {worse}/{ns} segments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
