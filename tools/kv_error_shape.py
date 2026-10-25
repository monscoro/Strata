#!/usr/bin/env python3
"""E4 (catalog: docs/paper/Ansatz_Improvements.md, row E4): is the 4-bit arm's residual error concentrated on a few
positions or spread over the bulk?  Offline, from the S0 per-position logs (the paper's own logs never existed).

The object is the per-position KL against fp16, with the harness's own kl_between (64-token support, floor
substitution) so the numbers are the published ones.  Three things are reported:

  1. the shape of each arm's per-position KL and what share of the total the worst 0.1/1/5/10 % carry;
  2. the same split into HEAD (the reference's ranks 1-8) and TAIL (ranks 9-64) terms.  The split is exact: the KL is
     a sum over the support, so head + tail terms add up to it.  This is the part that separates a real disagreement
     (a token the reference thinks likely, which the arm has pushed to its floor) from a floor event at rank 60;
  3. the paired window gain (plain - win256) and where it comes from.

A *thin* residual (few positions carrying most of the KL, and still thin when only head terms count) points at sparse
outliers; a *dense* one at a low-rank or basis problem.  This is the output-space proxy the catalog prescribes; it is
not the K/V residual matrix itself.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from quant_quality import kl_between  # noqa: E402  the harness's own arithmetic

RAW = pathlib.Path(r"C:\Users\krone\git\Strata-data\bench\2026-10-25-s0")
WIDTH = 256
TOPK = 64
HEAD = 8          # the reference's ranks 1..8


def segment_rows(path: pathlib.Path, first: int, last: int):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < first:
                continue
            if i >= last:
                return
            p = line.split("\t")
            if len(p) < 8 + WIDTH:
                continue
            yield (int(p[0]), int(p[1]), float(p[2]), int(p[3]),
                   [(int(x), float(y)) for x, y in (z.split(":") for z in p[8:8 + WIDTH])])


def quantile(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def concentration(v, fracs=(0.001, 0.01, 0.05, 0.10, 0.50)):
    s = sorted(v, reverse=True)
    tot = sum(s)
    return [100 * sum(s[:max(1, int(f * len(s)))]) / tot for f in fracs]


def kl_parts(kr, ka, head=HEAD):
    """The harness's KL, split into the terms of the reference's ranks 1..head and the rest, and the count of
    reference tokens in each part that fell outside the arm's support (they took the arm's floor)."""
    kr, ka = kr[:TOPK], ka[:TOPK]
    da = dict(ka)
    floor = min(b for _, b in ka)
    pr = [__import__("math").exp(b) for _, b in kr]
    pa = [__import__("math").exp(da.get(i, floor)) for i, _ in kr]
    sr, sa = sum(pr), sum(pa)
    terms = [(x / sr) * __import__("math").log((x / sr) / (y / sa)) for x, y in zip(pr, pa)]
    sub = [i not in da for i, _ in kr]
    return (sum(terms), sum(terms[:head]), sum(terms[head:]),
            sum(1 for x in sub[:head] if x), sum(1 for x in sub[head:] if x))


def main() -> int:
    segs = json.loads((RAW / "fp16.tsv.seg.json").read_text())["segments"]
    for name in ("int8", "win256", "plain"):
        other = json.loads((RAW / f"{name}.tsv.seg.json").read_text())["segments"]
        if [(s["name"], s["first_line"], s["end_line"]) for s in other] != \
           [(s["name"], s["first_line"], s["end_line"]) for s in segs]:
            raise SystemExit(f"{name}: line ranges differ from fp16 - not the same run layout")

    arms = {n: RAW / f"{n}.tsv" for n in ("int8", "win256", "plain")}
    kl = {n: [] for n in arms}
    khead = {n: [] for n in arms}
    ktail = {n: [] for n in arms}
    sub_head = {n: [] for n in arms}
    paired = {n: [] for n in arms}          # plain - arm, per position, whole KL
    paired_head = {n: [] for n in arms}

    for s in segs:
        first, last = s["first_line"], s["end_line"]
        ref = {p[0]: p for p in segment_rows(RAW / "fp16.tsv", first, last)}
        got = {}
        for name, path in arms.items():
            got[name] = {}
            for p in segment_rows(path, first, last):
                if p[0] not in ref:
                    continue
                r = ref[p[0]]
                if r[1] != p[1]:
                    raise SystemExit(f"{name} {s['name']} pos {p[0]}: target mismatch")
                total, head, tail, sh, _st = kl_parts(r[4], p[4])
                kl[name].append(total)
                khead[name].append(head)
                ktail[name].append(tail)
                sub_head[name].append(sh)
                got[name][p[0]] = (total, head)
        for pos, (tw, hw) in got["win256"].items():
            tp, hp = got["plain"][pos]
            paired["win256"].append(tp - tw)
            paired_head["win256"].append(hp - hw)

    n = len(kl["int8"])
    print(f"positions: {n}   KL support {TOPK}, head = the reference's ranks 1..{HEAD}\n")
    print(f"{'arm':8s} {'KL mean':>9s} {'p50':>8s} {'p90':>8s} {'p99':>8s} {'max':>8s} "
          f"{'head':>8s} {'tail':>8s} {'head %':>7s} {'pos w/ head sub':>16s}")
    for name in arms:
        v, h, t, sh = kl[name], khead[name], ktail[name], sub_head[name]
        print(f"{name:8s} {sum(v)/n:9.5f} {quantile(v,0.5):8.5f} {quantile(v,0.9):8.5f} {quantile(v,0.99):8.5f} "
              f"{max(v):8.4f} {sum(h)/n:8.5f} {sum(t)/n:8.5f} {100*sum(h)/sum(v):6.1f}% "
              f"{100*sum(1 for x in sh if x)/n:15.1f}%")

    print("\nconcentration: share of the total carried by the worst 0.1 / 1 / 5 / 10 / 50 % of positions")
    print(f"{'arm':8s} {'KL':>26s} {'head terms':>26s} {'tail terms':>26s}")
    for name in arms:
        a, b, c = concentration(kl[name]), concentration(khead[name]), concentration(ktail[name])
        print(f"{name:8s} " + "  ".join(f"{x:5.1f}" for x in a) + "   " +
              "  ".join(f"{x:5.1f}" for x in b) + "   " + "  ".join(f"{x:5.1f}" for x in c))

    print("\nhead substitutions per position (a reference top-8 token pushed to the arm's floor)")
    print(f"{'arm':8s} " + " ".join(f"{k:>6s}" for k in ("0", "1", "2", "3+")) +
          f" {'mean':>6s} {'KL of positions with 0':>24s} {'n':>7s}")
    for name in arms:
        sh = sub_head[name]
        cnt = [sum(1 for x in sh if x == 0), sum(1 for x in sh if x == 1),
               sum(1 for x in sh if x == 2), sum(1 for x in sh if x >= 3)]
        zero_kl = [kl[name][i] for i, x in enumerate(sh) if x == 0]
        print(f"{name:8s} " + " ".join(f"{c:6d}" for c in cnt) + f" {sum(sh)/n:6.2f} "
              f"{(sum(zero_kl)/len(zero_kl) if zero_kl else float('nan')):24.5f} {len(zero_kl):7d}")

    for name, pv, ph in (("plain - win256", paired["win256"], paired_head["win256"]),):
        pos_n = sum(1 for d in pv if d > 0)
        neg_n = sum(1 for d in pv if d < 0)
        tot = sum(pv)
        print(f"\npaired {name}: {len(pv)} positions, total {tot:+.3f} nat")
        print(f"  window better on {pos_n} ({100*pos_n/len(pv):.1f}%), worse on {neg_n} "
              f"({100*neg_n/len(pv):.1f}%), mean {tot/len(pv):+.5f}")
        print(f"  per-position gain: p50 {quantile(pv,0.5):+.5f}  p90 {quantile(pv,0.9):+.5f}  "
              f"p99 {quantile(pv,0.99):+.5f}  max {max(pv):+.4f}")
        a, b = concentration(pv, (0.001, 0.01, 0.05)), concentration(ph, (0.001, 0.01, 0.05))
        print(f"  share of the total gain from the worst 0.1/1/5 %: {a[0]:.1f}% / {a[1]:.1f}% / {a[2]:.1f}%")
        print(f"  the same for the head terms only:                  {b[0]:.1f}% / {b[1]:.1f}% / {b[2]:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
