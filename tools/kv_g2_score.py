#!/usr/bin/env python3
"""G2 scoring: is the truncated KL the KL?  Streaming, one row of each side at a time.

The runs are large (a 4,096-pair row is ~67 kB, 8,192 of them are 561 MB) and the arm files carry every one of the
32,746 positions with pairs only on the 1,024 sample rows, so nothing is loaded whole: the reference and the arm are
walked together, in the order both wrote them, and only the per-segment sums are kept.

Three numbers per arm and position:

  kl256    the published yardstick - KL(fp16 || arm) over the reference's top-256, with the arm's own top-256 and
           the floor substitution of `quant_quality.kl_between`, computed from the existing S3 runs at the same
           1,024 positions.  This is the number every earlier result in this line used.
  kl256sup the same sum over the same 256 tokens but with the arm's true logprob for each of them: the G2 support
           gives the arm's own top-256 nothing to substitute, and this is the first 256 terms of kl4096.
  kl4096   sum_v p_v (log p_v - log q_v) over the reference's top-4,096, true values on both sides, no
           substitution and no renormalisation.  The support holds 99.80 % of the reference's mass on average.

The gate is part of the report, not a separate step: for every arm whose S3 run exists, the eight fixed columns of
all 32,746 rows must match character for character.  A failing arm is scored anyway and marked `gate: FAIL`.

    python tools/kv_g2_score.py --ref ...\\2026-10-28-g2\\ref-top4096.tsv \\
        --support ...\\2026-10-28-g2\\support.json \\
        --ref-exact ...\\2026-10-28-g2\\ref-exact.tsv \\
        --arm int8=...\\int8.tsv --arm q4lm8=...\\q4lm8.tsv ... \\
        --s3-dir C:\\Users\\krone\\git\\Strata-data\\bench\\2026-10-27-s3 \\
        --s3-arm s320r0=C:\\...\\2026-10-28-ring\\s320r0.tsv --out ...\\score.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quant_quality as qq  # noqa: E402

T7_95 = 1.895          # one-sided 95 % for df = 7, as in G1
LADDER = {"order": ["q4lm8", "q4lm8k", "q4lm8v", "q4lm32", "q4lm32w", "int8"],
          "steps": [">", "~", ">", ">", "~"]}


def rows(path: pathlib.Path, want=None, only_pairs: bool = False):
    """Yield (segment, pos, fields) in file order; `pos` restarts at 0 in every segment.

    `want` filters positions: a set applies to every segment, a dict `{segment: set}` per segment (the exact rows
    are a different set in each segment).
    """
    meta = json.load(open(str(path) + ".seg.json", encoding="utf-8"))
    bounds = [(s["name"], s["end_line"]) for s in meta["segments"]]
    si = r = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            while si < len(bounds) and r >= bounds[si][1]:
                si += 1
            if si >= len(bounds):
                break
            name = bounds[si][0]
            f8 = line.rstrip("\n").split("\t")
            p = int(f8[0])
            w = want.get(name) if isinstance(want, dict) else want
            if (not only_pairs or len(f8) > 8) and (w is None or p in w):
                yield name, p, f8
            r += 1


def pairs(fields: list) -> list:
    return [(int(z.partition(":")[0]), float(z.partition(":")[2])) for z in fields[8:]]


def gate(arm: pathlib.Path, ref: pathlib.Path) -> dict:
    """The eight fixed columns of every row, character for character."""
    bad, rowsn, per_col = 0, 0, [0] * 8
    first = []
    with open(arm, encoding="utf-8") as fa, open(ref, encoding="utf-8") as fr:
        for la, lr in zip(fa, fr):
            rowsn += 1
            a = la.rstrip("\n").split("\t")
            r = lr.rstrip("\n").split("\t")
            diff = [i for i in range(8) if a[i] != r[i]]
            if diff:
                bad += 1
                for i in diff:
                    per_col[i] += 1
                if len(first) < 10:
                    first.append({"row": rowsn, "columns": diff, "arm": a[:8], "ref": r[:8]})
    return {"rows": rowsn, "differing_rows": bad, "columns": per_col, "first": first}


def mean(xs: list) -> float:
    return math.fsum(xs) / len(xs)


def paired(a: dict, b: dict, key: str) -> dict | None:
    segs = [s for s in a if s != "__all__" and s in b and a[s].get(key) is not None
            and b[s].get(key) is not None]
    if len(segs) < 2:
        return None
    d = [a[s][key] - b[s][key] for s in segs]
    n = len(d)
    m = mean(d)
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in d) / (n - 1))
    sem = sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "sem": sem, "bound": T7_95 * sem,
            "ci95": [m - 2.365 * sem, m + 2.365 * sem],
            "a_worse": sum(1 for x in d if x > 0), "diff": d}


def decide(p: dict) -> str:
    if p["mean"] > p["bound"]:
        return ">"
    if p["mean"] < -p["bound"]:
        return "<"
    return "~"


def score_s3(ref: pathlib.Path, arm: pathlib.Path, want) -> dict:
    """The published kl256 at the sample positions, from the existing runs (streamed, not loaded)."""
    per = {}
    for (rn, rp, rf), (an, ap, af) in zip(rows(ref, want), rows(arm, want)):
        if (rn, rp) != (an, ap):
            raise SystemExit(f"S3 row order differs: ref {rn} {rp} vs arm {an} {ap}")
        per.setdefault(rn, []).append(qq.kl_between(pairs(rf)[:256], pairs(af)[:256], 256))
    out = {s: mean(v) for s, v in per.items()}
    out["__all__"] = mean([v for s, v in out.items()])
    return out


def score_arm(ref: pathlib.Path, arm: pathlib.Path, want: set | None) -> dict:
    """kl256sup and kl4096 per segment, streaming both sides together."""
    per = {}
    n = 0
    for (rn, rp, rf), (an, ap, af) in zip(rows(ref, want), rows(arm, want, only_pairs=True)):
        if (rn, rp) != (an, ap):
            raise SystemExit(f"row order differs: ref {rn} {rp} vs arm {an} {ap}")
        rpairs = pairs(rf)
        amap = dict(pairs(af))
        k256 = 0.0
        kfull = 0.0
        for i, (tid, lp) in enumerate(rpairs):
            lq = amap.get(tid)
            if lq is None:      # the arm was told this id; a missing one is a run that did not follow the support
                raise SystemExit(f"{an} pos {ap}: id {tid} missing from the arm")
            term = math.exp(lp) * (lp - lq)
            kfull += term
            if i < 256:
                k256 += term
        d = per.setdefault(rn, {"kl256sup": [], "kl4096": [], "n": 0})
        d["kl256sup"].append(k256)
        d["kl4096"].append(kfull)
        d["n"] += 1
        n += 1
    out = {s: {"kl256sup": mean(v["kl256sup"]), "kl4096": mean(v["kl4096"]), "positions": v["n"]}
           for s, v in per.items()}
    out["__all__"] = {"kl256sup": mean([x["kl256sup"] for x in out.values()]),
                      "kl4096": mean([x["kl4096"] for x in out.values()]), "positions": n}
    return out


def score_exact(ref: pathlib.Path, arm: pathlib.Path, exact: dict) -> dict:
    """The 53 rows whose top-4,096 holds less than 0.95: kl4096 against klfull on the whole vocabulary."""
    per = {}
    for (rn, rp, rf), (an, ap, af) in zip(rows(ref, exact), rows(arm, exact, only_pairs=True)):
        if (rn, rp) != (an, ap):
            raise SystemExit(f"exact row order differs: ref {rn} {rp} vs arm {an} {ap}")
        rpairs = pairs(rf)
        # The exact-row reference was dumped with the same id list as the arm, so its pairs are in ascending id
        # order, not in logprob order: sort once to get the reference's own top-4,096.
        rpairs.sort(key=lambda z: -z[1])
        amap = dict(pairs(af))
        k4096 = math.fsum(math.exp(lp) * (lp - amap[tid]) for tid, lp in rpairs[:4096])
        kfull = math.fsum(math.exp(lp) * (lp - amap[tid]) for tid, lp in rpairs)
        d = per.setdefault(rn, {"kl4096": [], "klfull": [], "retained": []})
        d["kl4096"].append(k4096)
        d["klfull"].append(kfull)
        d["retained"].append(math.fsum(math.exp(lp) for _, lp in rpairs[:4096]))
    out = {s: {"rows": len(v["kl4096"]), "kl4096": mean(v["kl4096"]), "klfull": mean(v["klfull"]),
               "retained4096": mean(v["retained"])} for s, v in per.items()}
    tot = [x for v in per.values() for x in v["kl4096"]]
    totf = [x for v in per.values() for x in v["klfull"]]
    out["__all__"] = {"rows": len(tot), "kl4096": mean(tot), "klfull": mean(totf),
                      "worst_row_gap": max(f - g for f, g in zip(tot, totf))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="ref-top4096.tsv on the 1,024 sample positions")
    ap.add_argument("--support", required=True, help="support.json (the exact rows and the provenance)")
    ap.add_argument("--ref-exact", help="the reference's dump on the exact rows, whole vocabulary")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=TSV")
    ap.add_argument("--s3-dir", help="the existing runs for kl256 and the gate: <dir>/<arm>.tsv")
    ap.add_argument("--s3-arm", action="append", default=[], metavar="NAME=TSV",
                    help="an S3 run outside --s3-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=4096)
    ap.add_argument("--population", action="store_true",
                    help="also the published kl256 over every position of the existing runs (a full read of each)")
    a = ap.parse_args()

    support = json.load(open(a.support, encoding="utf-8"))
    exact = {s: set(v["exact_positions"]) for s, v in support["ids"].items()}
    want = set(int(x) for x in pathlib.Path(support["positions"]["path"]).read_text(encoding="utf-8").split())
    arms = {}
    for spec in a.arm:
        name, _, path = spec.partition("=")
        arms[name] = pathlib.Path(path)
    if not arms:
        raise SystemExit("no --arm given")
    s3 = {}
    for name in arms:
        p = pathlib.Path(a.s3_dir) / f"{name}.tsv" if a.s3_dir else None
        s3[name] = p if p is not None and p.exists() else None
    for spec in a.s3_arm:
        name, _, path = spec.partition("=")
        s3[name] = pathlib.Path(path)

    res = {"params": {"topk": a.topk, "ref": str(a.ref), "support": str(a.support),
                      "positions": len(want), "exact_rows": support["exact_rows"],
                      "ladder": LADDER},
           "gate": {}, "arms": {}, "exact": {}, "ladder": {}}

    # --- kl256 from the existing runs, and the gate on the fixed columns
    for name, path in arms.items():
        if s3.get(name) is not None and a.s3_dir:
            res["gate"][name] = gate(path, s3[name])
        arm = score_arm(pathlib.Path(a.ref), path, want)
        if a.s3_dir and s3.get(name) is not None:
            k256 = score_s3(pathlib.Path(a.s3_dir) / "fp16.tsv", s3[name], want)
            for seg, v in k256.items():
                if seg == "__all__":
                    arm["__all__"]["kl256"] = v
                else:
                    arm[seg]["kl256"] = v
            if a.population:
                arm["__all__"]["kl256_population"] = score_s3(
                    pathlib.Path(a.s3_dir) / "fp16.tsv", s3[name], None)["__all__"]
        res["arms"][name] = arm

    # --- the exact rows
    if a.ref_exact:
        for name, path in arms.items():
            res["exact"][name] = score_exact(pathlib.Path(a.ref_exact), path, exact)

    # --- the ladder, on both metrics
    for key in ("kl256", "kl256sup", "kl4096"):
        steps, ok = [], True
        for (x, y), want_step in zip(zip(LADDER["order"], LADDER["order"][1:]), LADDER["steps"]):
            ax, ay = res["arms"].get(x), res["arms"].get(y)
            px = paired(ax, ay, key) if ax and ay else None
            got = decide(px) if px else "n/a"
            steps.append({"a": x, "b": y, "want": want_step, "got": got,
                          "mean": px["mean"] if px else None, "bound": px["bound"] if px else None,
                          "a_worse": px["a_worse"] if px else None, "n": px["n"] if px else 0})
            ok = ok and got == want_step
        res["ladder"][key] = {"steps": steps, "holds": ok}

    # --- the pre-registered rule: same sign and same decision on kl4096 as on kl256
    rule, ok = [], True
    for (x, y) in zip(LADDER["order"], LADDER["order"][1:]):
        ax, ay = res["arms"].get(x), res["arms"].get(y)
        p256 = paired(ax, ay, "kl256") if ax and ay else None
        p4096 = paired(ax, ay, "kl4096") if ax and ay else None
        if p256 is None or p4096 is None:
            rule.append({"a": x, "b": y, "pass": False, "why": "no data"})
            ok = False
            continue
        same_sign = (p256["mean"] > 0) == (p4096["mean"] > 0)
        same_decision = decide(p256) == decide(p4096)
        rule.append({"a": x, "b": y, "kl256": p256["mean"], "kl4096": p4096["mean"],
                     "kl256_decision": decide(p256), "kl4096_decision": decide(p4096),
                     "same_sign": same_sign, "same_decision": same_decision,
                     "pass": same_sign and same_decision})
        ok = ok and same_sign and same_decision
    res["rule"] = {"steps": rule, "passes": ok}

    pathlib.Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")

    # --- the report
    print(f"{len(want)} positions x {len(support['ids'])} segments, exact rows {support['exact_rows']}\n")
    print(f"{'arm':10s} {'kl256':>9s} {'kl256sup':>9s} {'kl4096':>9s} {'bias':>9s}  gate")
    for name in arms:
        d = res["arms"][name]["__all__"]
        k2, ks, k4 = d.get("kl256"), d["kl256sup"], d["kl4096"]
        g = res["gate"].get(name)
        gs = "n/a" if g is None else ("ok" if g["differing_rows"] == 0 else
                                     f"FAIL {g['differing_rows']}/{g['rows']}")
        bias = f"{k2 - k4:9.5f}" if k2 is not None else f"{'-':>9s}"
        k2s = f"{k2:9.5f}" if k2 is not None else f"{'-':>9s}"
        print(f"{name:10s} {k2s} {ks:9.5f} {k4:9.5f} {bias}  {gs}")
    if res["exact"]:
        print(f"\nexact rows: {'arm':10s} {'kl4096':>9s} {'klfull':>9s} {'rows':>5s}")
        for name, d in res["exact"].items():
            x = d["__all__"]
            print(f"{'':12s}{name:10s} {x['kl4096']:9.5f} {x['klfull']:9.5f} {x['rows']:5d}")
    if a.population:
        print(f"\nsample against the population: {'arm':10s} {'sample':>9s} {'all':>9s} {'ratio':>6s}")
        for name in arms:
            d = res["arms"][name]["__all__"]
            if d.get("kl256") is not None and d.get("kl256_population"):
                print(f"{'':30s}{name:10s} {d['kl256']:9.5f} {d['kl256_population']:9.5f} "
                      f"{d['kl256'] / d['kl256_population']:6.2f}")
    print("\nladder (kl256 -> kl4096):")
    for s in res["rule"]["steps"]:
        if "kl256" in s:
            print(f"  {s['a']:8s} {s['b']:8s}  kl256 {s['kl256']:+.5f} {s['kl256_decision']}   "
                  f"kl4096 {s['kl4096']:+.5f} {s['kl4096_decision']}   "
                  f"{'pass' if s['pass'] else 'FAIL'}")
        else:
            print(f"  {s['a']:8s} {s['b']:8s}  {s['why']}")
    print(f"\nrule: {'PASSES' if res['rule']['passes'] else 'FAILS'}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
