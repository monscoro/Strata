#!/usr/bin/env python3
"""P1: the pure 16/16 arm without the window, scored against fp16, int8 and the windowed arm.

P0b showed that the KV error at the P0a top-gap positions is normal, but those positions came from the
windowed `win` arm.  P1 removes the window (`--kv q4lm32`, `--short-read 6144`, no STRATA_KV_WINDOW) and
asks whether the tail was a codec effect or a read-path effect.  This tool reads the run TSVs and reports
the same shape P0a used, so the numbers are directly comparable:

  * per-arm KL(fp16 || arm) over fp16's top-256: mean, median, p99, max, mean without the worst 1 %;
  * paired pure - int8 and pure - win: mean, mean without the top 1 %, better/worse counts, worst five;
  * the last 1,024 positions;
  * the top-18 positions of the paired gap (arm - int8), the P0a analogue, and how many of P0a's 18 they
    overlap, plus the positions that are bad in every arm.

    python tools/kv_p1_pure.py --run DIR --fp16 DIR\\fp16.tsv --int8 DIR\\int8.tsv --win DIR\\win.tsv
                              [--pure DIR\\p1-pure16.tsv] [--json OUT.json]

`--run` is the directory that holds p1-pure16.tsv (default: the P1 bench directory).  No GPU, no model.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quant_quality as qq  # noqa: E402

TOPK = 256
LAST = 1024
P0A_TOP18 = [130972, 127887, 130973, 130952, 130942, 130968, 126191, 125074, 128793,
             130954, 128883, 124891, 127727, 128709, 128516, 125064, 128583, 130951]


def series(ref: dict, arm: dict, positions: list[int]) -> dict[int, float]:
    return {p: qq.kl_between(ref[p][4], arm[p][4], TOPK) for p in positions}


def shape(kl: dict[int, float]) -> dict:
    v = [kl[p] for p in kl]
    s = sorted(v)
    n = len(s)
    cut = max(1, int(round(0.01 * n)))
    return {"n": n, "mean": sum(v) / n, "median": s[n // 2], "p99": s[int(0.99 * (n - 1))],
            "max": max(v), "mean_no_top1": sum(s[:-cut]) / (n - cut)}


def paired(a: dict[int, float], b: dict[int, float]) -> dict:
    d = {p: a[p] - b[p] for p in a}
    v = sorted(d.values())
    n = len(v)
    cut = max(1, int(round(0.01 * n)))
    top = sorted(d.items(), key=lambda kv: -kv[1])[:5]
    return {"n": n, "mean": sum(v) / n, "mean_no_top1": sum(v[:-cut]) / (n - cut),
            "better": sum(1 for x in v if x < 0), "worse": sum(1 for x in v if x > 0),
            "top5": [[p, round(x, 4)] for p, x in top]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=r"C:\Users\krone\git\Strata-data\bench\2026-11-07-p1-pure16")
    ap.add_argument("--pure", help="default: RUN\\p1-pure16.tsv")
    ap.add_argument("--fp16", default=r"C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\128k\fp16.tsv")
    ap.add_argument("--int8", default=r"C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\128k\int8.tsv")
    ap.add_argument("--win", default=r"C:\Users\krone\git\Strata-data\bench\2026-11-03-tag-arm\128k\win.tsv")
    ap.add_argument("--json")
    a = ap.parse_args()

    pure_path = a.pure or str(pathlib.Path(a.run) / "p1-pure16.tsv")
    ref = qq.read_run(a.fp16)
    pure = qq.read_run(pure_path)
    int8 = qq.read_run(a.int8)
    win = qq.read_run(a.win)

    ref = next(iter(ref.values()))
    pure = next(iter(pure.values()))
    int8 = next(iter(int8.values()))
    win = next(iter(win.values()))
    pos = sorted(set(ref) & set(pure) & set(int8) & set(win))
    print(f"positions: {len(pos)} (fp16 {len(ref)}, pure {len(pure)}, int8 {len(int8)}, win {len(win)})")
    if len(pos) != 6144:
        print(f"warning: expected 6,144 positions, got {len(pos)}")

    kl = {"pure": series(ref, pure, pos), "int8": series(ref, int8, pos), "win": series(ref, win, pos)}
    print(f"\nKL(fp16 || arm), fp16's top-{TOPK}:")
    print(f"  {'arm':6s} {'mean':>10s} {'median':>10s} {'p99':>9s} {'max':>8s} {'mean no top 1%':>15s}")
    for name, k in kl.items():
        s = shape(k)
        print(f"  {name:6s} {s['mean']:10.6f} {s['median']:10.6f} {s['p99']:9.6f} {s['max']:8.4f} "
              f"{s['mean_no_top1']:15.6f}")

    print("\npaired:")
    pairs = {"pure - int8": paired(kl["pure"], kl["int8"]),
             "pure - win": paired(kl["pure"], kl["win"])}
    for name, p in pairs.items():
        print(f"  {name:12s} mean {p['mean']:+.6f}  mean no top 1% {p['mean_no_top1']:+.6f}  "
              f"better {p['better']}/{p['n']}, worse {p['worse']}  top5 {p['top5']}")

    last = pos[-LAST:]
    print(f"\nlast {LAST} positions:")
    for name, k in kl.items():
        s = shape({p: k[p] for p in last})
        print(f"  {name:6s} mean {s['mean']:.6f}  mean no top 1% {s['mean_no_top1']:.6f}  p99 {s['p99']:.6f}")
    for name in ("pure - int8", "pure - win"):
        pa, pb = name.split(" - ")
        p = paired({q: kl[pa][q] for q in last}, {q: kl[pb][q] for q in last})
        print(f"  {name:12s} mean {p['mean']:+.6f}  mean no top 1% {p['mean_no_top1']:+.6f}")

    # The direct analogue of P0a: the top positions of the *paired gap* (arm - int8), not of the arm's own KL.
    # P0a selected its 18 from win - int8; an arm's absolute KL top can be dominated by a position that every
    # arm gets wrong (130952 is bad in int8 too), so the gap is the comparable object.
    print(f"\npaired gap vs int8, top-18 (the P0a analogue):")
    gap_top = {}
    for arm in ("pure", "win"):
        gap = {p: kl[arm][p] - kl["int8"][p] for p in pos}
        top = sorted(gap, key=lambda p: -gap[p])[:18]
        gap_top[arm] = {"top18": top, "mean_gap": sum(gap.values()) / len(gap),
                        "mean_gap_p0a": sum(gap[p] for p in P0A_TOP18) / len(P0A_TOP18),
                        "top5_share": 100 * sum(gap[p] for p in top[:5]) / sum(v for v in gap.values() if v > 0)}
        print(f"  {arm:5s} mean gap {gap_top[arm]['mean_gap']:+.6f}  at P0a's 18 {gap_top[arm]['mean_gap_p0a']:+.6f}  "
              f"top-5 carry {gap_top[arm]['top5_share']:.1f}% of the positive gap")
        print(f"        top-18 {top}")
        print(f"        overlap with P0a's 18: {len(set(top) & set(P0A_TOP18))}/18")

    # The shared knife-edge: a position that is bad in every arm is model/reference sensitivity, not a codec.
    common = set(gap_top["pure"]["top18"]) & set(gap_top["win"]["top18"]) & set(P0A_TOP18)
    print(f"\n  positions in pure's 18, win's 18 and P0a's 18: {len(common)}")
    worst_abs = max(pos, key=lambda p: kl["pure"][p])
    print(f"  pure's worst absolute KL: {worst_abs} ({kl['pure'][worst_abs]:.4f}); int8 there "
          f"{kl['int8'][worst_abs]:.4f}, win {kl['win'][worst_abs]:.4f}")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"positions": len(pos), "arms": {n: shape(k) for n, k in kl.items()},
             "paired": pairs, "last": {n: shape({p: k[p] for p in last}) for n, k in kl.items()},
             "gap_vs_int8": gap_top, "shared_positions": sorted(common)}, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
