#!/usr/bin/env python3
"""G1 of the beyond-KL work plan: do delta NLL, the margin-conditioned flip and the region label carry anything
that mean KL does not already say?

The definitions and the acceptance rule were frozen in `bench/results/2026-10-28-g1-features/README.md` before
this script ran; the four calibration pairs come from `bench/fixtures/beyond-kl-calibration.json`.  Nothing here
measures a model - it rescores the twenty teacher-forced runs that already exist (32,746 positions each, cwe-4k,
top-256), so it costs no GPU.

    python tools/kv_g1.py [--fixture bench/fixtures/beyond-kl-calibration.json] [--bench-root DIR]
                          [--kl-topk 256] [--arms a,b,c] [--json OUT.json]

Per position q, with the fp16 row as reference:

    kl64 / kl256   KL(fp16 || arm) over the reference's top-64 / top-256, floor substitution as in kl_between
    dnll           logp_fp16(target) - logp_arm(target); positive = the arm is worse
    flip           1 if argmax(arm) != argmax(fp16)
    flip_m1        1 if flip and the reference margin m = lp1 - lp2 >= 1 nat
    floor          1 if the reference's top-256 set is not contained in the arm's (a substitution happened)
    p99, cvar99    the 99th percentile and the mean of the worst 1 % of a segment's kl256
    region_excess  mean kl256 in the evicted middle [R, n-S) minus the exact edges (sinks q < R, last S)

Aggregation is per segment and pairing is over the eight segments (t with df = 7), never over positions: the
positions of one text are correlated, and the resolution at true parity (1.895 * sd / sqrt(8)) is the number a
difference has to beat.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quant_quality as qq  # noqa: E402

FEATURES = ["kl64", "kl256", "dnll", "flip", "flip_m1", "floor", "p99", "cvar99", "region_excess"]
R_DEFAULT, S_DEFAULT = 64, 256   # the reference window geometry, frozen with the definitions
T7_975, T7_95 = 2.365, 1.895     # two-sided 97.5 % and one-sided 95 % for df = 7


def arm_features(path: str, ref: dict, kl_topk: int, r_win: int, s_win: int) -> dict:
    """{segment: {feature: value}} for one arm, plus the position-weighted means under '__all__'."""
    run = qq.read_run(path)
    out, sums, counts = {}, {}, {}
    for seg, rr in run.items():
        ra = ref.get(seg, {})
        pos = sorted(set(rr) & set(ra))
        if not pos:
            continue
        n = len(pos)
        kl64, kl256, dnll, flip, flip_m1, floor, floor_mass = [], [], [], [], [], [], []
        for q in pos:
            tr, lr, topr, _hitr, kr = ra[q]
            ta, la, topa, _hita, ka = rr[q]
            if tr != ta:
                raise SystemExit(f"{seg} pos {q}: target {tr} != {ta} (not the same text)")
            kl64.append(qq.kl_between(kr, ka, 64))
            kl256.append(qq.kl_between(kr, ka, kl_topk))
            dnll.append(lr - la)
            changed = topr != topa
            flip.append(1.0 if changed else 0.0)
            if kr[0][1] >= kr[1][1]:
                margin = kr[0][1] - kr[1][1]
            else:   # defensive: the engine writes the list descending, but the margin must not depend on that
                two = sorted((b for _, b in kr), reverse=True)[:2]
                margin = two[0] - two[1] if len(two) > 1 else 0.0
            flip_m1.append(1.0 if (changed and margin >= 1.0) else 0.0)
            ids_r = {i for i, _ in kr[:kl_topk]}
            ids_a = {i for i, _ in ka[:kl_topk]}
            missing_mass = sum(math.exp(b) for i, b in kr[:kl_topk] if i not in ids_a)
            floor_mass.append(missing_mass)
            floor.append(1.0 if missing_mass >= 0.01 else 0.0)

        ks = sorted(kl256)
        k_cvar = max(1, int(round(0.01 * len(ks))))
        evicted = [k for q, k in zip(pos, kl256) if r_win <= q < n - s_win]
        exact = [k for q, k in zip(pos, kl256) if not (r_win <= q < n - s_win)]
        rec = {
            "positions": n,
            "kl64": sum(kl64) / n,
            "kl256": sum(kl256) / n,
            "dnll": sum(dnll) / n,
            "flip": sum(flip) / n,
            "flip_m1": sum(flip_m1) / n,
            "floor": sum(floor) / n,
            "floor_mass": sum(floor_mass) / n,
            "p99": ks[int(0.99 * (len(ks) - 1))],
            "cvar99": sum(ks[-k_cvar:]) / k_cvar,
            "region_excess": (sum(evicted) / len(evicted) - sum(exact) / len(exact))
                              if evicted and exact else None,
            "evicted": len(evicted),
            "exact": len(exact),
        }
        out[seg] = rec
        for f in ("kl64", "kl256"):
            sums[f] = sums.get(f, 0.0) + rec[f] * n
            counts[f] = counts.get(f, 0) + n
    segs = [s for s in out]
    allrec = {}
    for f in FEATURES:
        vals = [out[s][f] for s in segs if out[s].get(f) is not None]
        allrec[f] = sum(vals) / len(vals) if vals else None
    for f in ("kl64", "kl256"):
        allrec[f] = sums[f] / counts[f]
    allrec["positions"] = counts.get("kl256", 0)
    out["__all__"] = allrec
    return out


def paired_segments(a: dict, b: dict, feature: str) -> dict | None:
    """The eight segment differences a - b, paired; None when a segment has no value for the feature."""
    segs = [s for s in a if s != "__all__" and s in b and a[s].get(feature) is not None
            and b[s].get(feature) is not None]
    if len(segs) < 2:
        return None
    d = [a[s][feature] - b[s][feature] for s in segs]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    sem = sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "sem": sem, "t": m / sem if sem else float("nan"),
            "ci95_low": m - T7_975 * sem, "ci95_high": m + T7_975 * sem,
            "parity_bound": T7_95 * sem,
            "segments_a_worse": sum(1 for x in d if x > 0),
            "segments": n}


def rule_verdict(p: dict, truth: str) -> str:
    if truth == "parity":
        return "pass" if p["ci95_low"] <= 0 <= p["ci95_high"] else "FAIL"
    if truth == "a_worse":
        return "pass" if p["ci95_low"] > 0 else "FAIL"
    raise SystemExit("unknown truth " + truth)


def ladder_check(stats: dict, ladder: dict, feature: str) -> tuple[bool, list]:
    steps = []
    order = ladder["order"]
    for (a, b), want in zip(zip(order, order[1:]), ladder["steps"]):
        if a not in stats or b not in stats:
            steps.append((a, b, want, "n/a", None))
            continue
        p = paired_segments(stats[a], stats[b], feature)
        if p is None:
            steps.append((a, b, want, "n/a", None))
            continue
        if p["mean"] > p["parity_bound"]:
            got = ">"
        elif p["mean"] < -p["parity_bound"]:
            got = "<"
        else:
            got = "~"
        steps.append((a, b, want, got, p))
    return all(w == g for _, _, w, g, _ in steps), steps


def ranks(values: list) -> list:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def spearman(x: list, y: list) -> float:
    rx, ry = ranks(x), ranks(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="bench/fixtures/beyond-kl-calibration.json")
    ap.add_argument("--bench-root", help="where the runs live (default: <repo>/../Strata-data/bench)")
    ap.add_argument("--kl-topk", type=int, default=256)
    ap.add_argument("--arms", help="comma-separated subset of fixture arm names, for a quick check")
    ap.add_argument("--json")
    a = ap.parse_args()

    fx = json.load(open(a.fixture, encoding="utf-8"))
    root = pathlib.Path(a.bench_root) if a.bench_root else \
        pathlib.Path(__file__).resolve().parent.parent.parent / "Strata-data" / "bench"
    if not root.is_dir():
        raise SystemExit(f"bench root not found: {root} (pass --bench-root)")

    names = [n for n in fx["runs"] if n != fx["anchor"]]
    if a.arms:
        names = [n for n in a.arms.split(",") if n]
    print(f"anchor {fx['anchor']}, {len(names)} arms, KL support top-{a.kl_topk}, "
          f"region geometry R={R_DEFAULT} S={S_DEFAULT}")

    ref = qq.read_run(str(root / fx["runs"][fx["anchor"]]))
    stats = {}
    for name in names:
        path = root / fx["runs"][name]
        if not path.exists():
            raise SystemExit(f"missing run {name}: {path}")
        stats[name] = arm_features(str(path), ref, a.kl_topk, R_DEFAULT, S_DEFAULT)
        v = stats[name]["__all__"]
        print(f"  {name:12s} pos {v['positions']:6d}  kl64 {v['kl64']:.5f}  kl256 {v['kl256']:.5f}  "
              f"dnll {v['dnll']:+.5f}  flip {100 * v['flip']:.2f}%  flip@1nat {100 * v['flip_m1']:.2f}%  "
              f"floor {100 * v['floor']:.1f}%", flush=True)

    # --- the same configuration measured twice must land on the same numbers, or the corpus is not one corpus
    dups = fx.get("duplicate_runs", {})
    distinct = [n for n in names if n not in dups]
    for dup, twin in dups.items():
        if dup not in stats or twin not in stats:
            continue
        d = max(abs(stats[dup]["__all__"][f] - stats[twin]["__all__"][f])
                for f in FEATURES if stats[dup]["__all__"][f] is not None)
        print(f"  duplicate {dup} vs {twin}: max |delta| over the features {d:.2e}"
              f"{'' if d < 1e-9 else '  (NOT identical)'}")

    # --- the fixture's pinned means must still reproduce, or nothing below is comparable
    tol = fx["pinned_tolerance"]
    bad = []
    for name, pin in fx["pinned_arm_means"].items():
        if name not in stats:
            continue
        for f in ("kl64", "kl256"):
            got, want = stats[name]["__all__"][f], pin[f]
            if abs(got - want) > tol:
                bad.append(f"{name}.{f}: got {got:.5f}, pinned {want:.5f}")
    if bad:
        print("\nFIXTURE MISMATCH - the runs or the definitions drifted, stopping:")
        for b in bad:
            print("  " + b)
        return 2
    print(f"\nfixture check: {len([n for n in fx['pinned_arm_means'] if n in stats])} pinned arms reproduce "
          f"within {tol}")

    # --- the four calibration pairs
    print("\nthe rule (bench/results/2026-10-28-g1-features/README.md):")
    verdicts = {}
    for feat in FEATURES:
        line = []
        ok = True
        for pair in fx["pairs"]:
            if pair["a"] not in stats or pair["b"] not in stats:
                line.append(f"{pair['id']}: n/a")
                ok = False
                continue
            p = paired_segments(stats[pair["a"]], stats[pair["b"]], feat)
            if p is None:
                line.append(f"{pair['id']}: n/a")
                ok = False
                continue
            v = rule_verdict(p, pair["truth"])
            ok = ok and v == "pass"
            line.append(f"{pair['id']}: {p['mean']:+.5f} [{p['ci95_low']:+.5f}, {p['ci95_high']:+.5f}] "
                        f"{v}")
        ladder_ok, steps = ladder_check(stats, fx["ladder"], feat)
        ok = ok and ladder_ok
        verdicts[feat] = {"pairs": line, "ladder": ladder_ok, "all": ok}
        print(f"  {feat:14s} " + " | ".join(line) + f" | ladder {'pass' if ladder_ok else 'FAIL'} "
              f"-> {'PASS' if ok else 'fail'}")

    # --- redundancy against mean KL over the arms (duplicates removed: a repeated run is a weight, not an arm).
    # The anchor belongs in the set the pre-registration names ("fifteen configurations"); its features are 0 by
    # construction, so the same number is also printed without it, which is the stricter of the two.
    ref_feats = arm_features(str(root / fx["runs"][fx["anchor"]]), ref, a.kl_topk, R_DEFAULT, S_DEFAULT)
    print(f"\nredundancy: Spearman rho against mean kl256 (|rho| >= 0.9 = nothing new)")
    for label, group in (("15 configs incl. anchor", [fx["anchor"]] + distinct),
                         (f"{len(distinct)} arms excl. anchor", distinct)):
        print(f"  over {label}")
        base = [(ref_feats if n == fx["anchor"] else stats[n])["__all__"]["kl256"] for n in group]
        for feat in FEATURES:
            if feat == "kl256":
                continue
            vals = [(ref_feats if n == fx["anchor"] else stats[n])["__all__"][feat] for n in group]
            if any(v is None for v in vals):
                print(f"    {feat:14s} n/a (undefined for some arm)")
                continue
            rho = spearman(vals, base)
            verdicts[feat][f"rho_vs_kl256_{len(group)}"] = rho
            print(f"    {feat:14s} rho {rho:+.3f}  "
                  f"{'redundant' if abs(rho) >= 0.9 else 'adds a dimension'}")

    passed = [f for f in FEATURES if verdicts[f]["all"]]
    print("\nfeatures that pass all four checks: " + (", ".join(passed) if passed else "NONE"))
    if not passed:
        print("  -> the pre-registered outcome: mean KL and P99 stay the baselines, no dashboard.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"fixture": a.fixture, "kl_topk": a.kl_topk, "arms": names, "stats": stats,
             "verdicts": verdicts, "ladder_steps": {f: ladder_check(stats, fx["ladder"], f)[1]
                                                    for f in FEATURES}}, indent=1), encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
