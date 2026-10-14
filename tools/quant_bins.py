#!/usr/bin/env python3
"""Where a KV format's error comes from: `quant_quality.py compare` pools every position of every segment into one
number, which cannot answer the question the 8K set exists for - **does the error grow with the context length?**

    python tools/quant_bins.py bins --ref kv-fp16.tsv --run kv-q4_0.tsv [...] [--bin 1000] [--json bins.json]
        per segment, one row per `--bin` positions of that segment: top-1 agreement and mean KL of each run against
        the reference, in the same sense as `quant_quality.py compare`.

    python tools/quant_bins.py segments --run RUN.tsv --pair a:0=b:0 --pair a:7000=b:0 [--n 1200] [--json x.json]
        one run against itself: `--pair` names two positions in two of its segments whose *tokens* are the same, and
        the report says whether the engine scored them the same.  That is how "is a segment's context its own tokens
        or everything before it" and "does 7000 tokens of preceding text change a position" were answered.

`quant_quality.py run` gives every segment its own `GEN` request and sizes `--max-context` for a single segment, so
position `p` of a segment is scored with that segment's first `p` tokens as its context - `bins` reads position as
context length for exactly that reason, and checks it with `segments` before believing it.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import quant_quality as qq


def binned(rows: dict, lo: int, hi: int) -> dict:
    out = {}
    for seg, d in rows.items():
        keep = {p: v for p, v in d.items() if lo <= p < hi}
        if keep:
            out[seg] = keep
    return out


def cmd_bins(a) -> int:
    ref = qq.read_run(a.ref)
    n = max(max(d) for d in ref.values()) + 1
    edges = list(range(0, n, a.bin)) + [n]
    res = {}
    print(f"reference: {a.ref}  ({n} positions per segment, bins of {a.bin})")
    for path in a.runs:
        run = qq.read_run(path)
        per = {}
        print(f"\n=== {pathlib.Path(path).name} ===")
        print(f"{'segment':13s} {'bin':>11s} {'pos':>5s} {'top1':>7s} {'KL':>9s}")
        for seg in ref:
            if seg not in run:
                continue
            per[seg] = {}
            for lo, hi in zip(edges, edges[1:]):
                v = qq.compare({seg: binned({seg: ref[seg]}, lo, hi)[seg]},
                               {seg: binned({seg: run[seg]}, lo, hi)[seg]})[seg]
                per[seg][f"{lo}-{hi}"] = v
                print(f"{seg:13s} {f'{lo}-{hi}':>11s} {v['positions']:5d} {100 * v['top1']:6.2f}% {v['kl']:9.6f}")
        res[path] = per
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


def rekey(rows: dict, seg: str, off: int, n: int) -> dict:
    d = rows[seg]
    return {i: d[off + i] for i in range(n) if off + i in d}


def cmd_segments(a) -> int:
    rows = qq.read_run(a.run)
    print(f"run: {a.run}  segments: " + ", ".join(f"{k}={len(v)}" for k, v in rows.items()))
    res = {}
    for pair in a.pair:
        left, right = pair.split("=")
        ls, lo = left.rsplit(":", 1)
        rs, ro = right.rsplit(":", 1)
        lo, ro = int(lo), int(ro)
        A, B = rekey(rows, ls, lo, a.n), rekey(rows, rs, ro, a.n)
        v = qq.compare({"x": A}, {"x": B})["x"]
        keys = [i for i in A if i in B]
        same = sum(1 for i in keys if A[i][3] == B[i][3])
        dmax = max((abs(A[i][1] - B[i][1]) for i in keys), default=0.0)
        res[pair] = {"positions": v["positions"], "same_topk": same, "same_argmax": v["top1"],
                     "kl": v["kl"], "max_abs_dlogp": dmax}
        print(f"\n{pair}   ({v['positions']} positions)")
        print(f"  same top-{qq.TOPK} list : {same}/{v['positions']}")
        print(f"  same argmax       : {100 * v['top1']:.2f}%")
        print(f"  KL                : {v['kl']:.6f}")
        print(f"  max |d logp(text)|: {dmax:.3e}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bins", help="one row per position bin, per segment")
    p.add_argument("--ref", required=True)
    p.add_argument("--run", dest="runs", action="append", required=True)
    p.add_argument("--bin", type=int, default=1000)
    p.add_argument("--json")
    p = sub.add_parser("segments", help="compare two of one run's segments at positions whose tokens are the same")
    p.add_argument("--run", required=True)
    p.add_argument("--pair", action="append", required=True, metavar="SEG:OFF=SEG:OFF")
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--json")
    a = ap.parse_args()
    return {"bins": cmd_bins, "segments": cmd_segments}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
