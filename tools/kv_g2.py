#!/usr/bin/env python3
"""G2 sampler: which 1,024 positions the eight G2 runs dump log-probabilities for.

The strata, the sizes and the rules were frozen in `bench/results/2026-10-28-g2-full-vocab/README.md`
before this script ran.  It measures no model and evaluates no arm: it reads two teacher-forced runs that
already exist (the fp16 anchor and int8, both cwe-4k, top-256) and selects positions from them.

    python tools/kv_g2.py [--ref .../fp16.tsv] [--arm .../int8.tsv] [--seed 20261028]
                          [--positions-out g2-positions.txt] [--json g2-sample.json]

What it reads, per position: the retained mass of the reference's top-256 (`sum exp(logp)`, measured, not
estimated) and the argmax.  Nothing else - the KL is not computed here, so the selection cannot depend on
the quantity the sample is meant to test.

The three strata (union, then filled to exactly 1,024 from the remaining pool):

    flat    union of the 40 positions with the smallest retained mass, per segment (<= 320)
    random  640 drawn uniformly without replacement - the only unbiased look at the population
    flip    up to 64 positions where argmax(ref) != argmax(arm) in at least one segment; a position index
            means different text in each segment, so the flip set is per segment in principle, but the id
            file is one shared set - the stratum is the union, and the per-segment counts are reported

Positions 0 .. 4,086 only: the shortest segment (s05) has 4,087 positions, and the id file is one line per
position for all eight segments, so a position that one segment lacks would unbalance the paired test.
Each stratum has its own RNG derived from the seed, so a change in one stratum does not shift the others.

Why the deliberate bias is harmless: the sample is not an estimator of the population mean KL.  It is the
same set of positions for every arm and for both estimators (kl256 and klfull), so the bias cancels in the
paired difference, which is the quantity G2 tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quant_quality as qq  # noqa: E402

# frozen with the pre-registration
N_SAMPLE, N_FLAT_PER_SEG, N_RANDOM, N_FLIP = 1024, 40, 640, 64
SEED = 20261028


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: pathlib.Path) -> dict:
    """{segment: {pos: (retained_mass, argmax, target, pairs)}} from a run's TSV and its .seg.json."""
    run = qq.read_run(str(path))
    out = {}
    for seg, rr in run.items():
        d = {}
        for pos, (tgt, _lp, top, _hit, pairs) in rr.items():
            d[pos] = (math.fsum(math.exp(b) for _, b in pairs), top, tgt, len(pairs))
        out[seg] = d
    return out


def dist(values: list) -> dict:
    v = sorted(values)
    n = len(v)

    def q(p: float) -> float:
        return v[min(n - 1, max(0, int(p * (n - 1))))]

    return {"n": n, "min": v[0], "p01": q(0.01), "p10": q(0.10), "median": q(0.50),
            "mean": math.fsum(v) / n, "p90": q(0.90), "max": v[-1]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="fp16.tsv", help="the anchor run (fp16), retained mass and argmax come from it")
    ap.add_argument("--arm", default="int8.tsv", help="the fixed arm for the flip stratum (int8)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--positions-out", default="g2-positions.txt")
    ap.add_argument("--json", default="g2-sample.json")
    a = ap.parse_args()

    ref_path, arm_path = pathlib.Path(a.ref), pathlib.Path(a.arm)
    for p in (ref_path, arm_path):
        if not p.exists():
            raise SystemExit(f"missing run: {p} (pass --ref / --arm)")

    ref, arm = load(ref_path), load(arm_path)
    if sorted(ref) != sorted(arm):
        raise SystemExit(f"segment sets differ: {sorted(ref)} vs {sorted(arm)}")

    segs = sorted(ref)
    lengths = {s: len(ref[s]) for s in segs}
    hi = min(max(ref[s]) for s in segs)          # 4,086 for the S3 runs
    valid = list(range(hi + 1))
    for s in segs:
        missing = [p for p in valid if p not in ref[s]]
        if missing:
            raise SystemExit(f"{s}: {len(missing)} positions of 0..{hi} missing from the reference")
        if any(p not in arm[s] for p in valid):
            raise SystemExit(f"{s}: the arm does not cover 0..{hi}")
        if any(ref[s][p][2] != arm[s][p][2] for p in valid):
            raise SystemExit(f"{s}: target differs between the runs - not the same text")
    widths = {v[3] for s in segs for v in ref[s].values()}
    print(f"segments {', '.join(f'{s} {lengths[s]}' for s in segs)}")
    print(f"shared positions 0..{hi} ({hi + 1}), reference pairs per position {sorted(widths)}")

    # --- stratum 1: the flattest positions of each segment (union)
    flat_by_seg, cut = {}, {}
    for s in segs:
        order = sorted(valid, key=lambda p: (ref[s][p][0], p))[:N_FLAT_PER_SEG]
        flat_by_seg[s] = order
        cut[s] = ref[s][order[-1]][0]
    flat = sorted({p for ps in flat_by_seg.values() for p in ps})

    # --- stratum 2: uniform without replacement over the shared positions
    rng_random = random.Random(f"{a.seed}:random")
    drawn_random = sorted(rng_random.sample(valid, N_RANDOM))

    # --- stratum 3: argmax flips, capped, drawn uniformly from the flip set.
    # A position index refers to different text in each segment, so the flip is a per-segment event; the id file
    # is one shared set of positions, so the stratum is the union over the segments ("flips somewhere").  How
    # many of the drawn positions flip in each segment is reported, because that is what the dumps will show.
    flips_by_seg = {s: [p for p in valid if ref[s][p][1] != arm[s][p][1]] for s in segs}
    pool = sorted({p for v in flips_by_seg.values() for p in v})
    rng_flip = random.Random(f"{a.seed}:flip")
    drawn_flip = sorted(rng_flip.sample(pool, min(N_FLIP, len(pool))))
    drawn_flip_per_seg = {s: sum(1 for p in drawn_flip if p in set(flips_by_seg[s])) for s in segs}

    # --- union, then fill to exactly N_SAMPLE from the remaining pool
    chosen = set(flat) | set(drawn_random) | set(drawn_flip)
    remaining = [p for p in valid if p not in chosen]
    rng_fill = random.Random(f"{a.seed}:fill")
    n_fill = N_SAMPLE - len(chosen)
    if n_fill < 0:
        raise SystemExit(f"the strata already hold {len(chosen)} positions, more than {N_SAMPLE}")
    drawn_fill = sorted(rng_fill.sample(remaining, n_fill)) if n_fill else []
    positions = sorted(chosen | set(drawn_fill))
    if len(positions) != N_SAMPLE:
        raise SystemExit(f"sample has {len(positions)} positions, {N_SAMPLE} asked")

    # --- provenance and the distributions the README asks to report
    overlap = {
        "flat_x_random": len(set(flat) & set(drawn_random)),
        "flat_x_flip": len(set(flat) & set(drawn_flip)),
        "random_x_flip": len(set(drawn_random) & set(drawn_flip)),
    }
    pop_mass = [ref[s][p][0] for s in segs for p in valid]
    sample_mass = [ref[s][p][0] for s in segs for p in positions]
    per_stratum = {
        "flat": [ref[s][p][0] for s in segs for p in flat_by_seg[s]],
        "random": [ref[s][p][0] for s in segs for p in drawn_random],
        "flip": [ref[s][p][0] for s in segs for p in drawn_flip],
        "fill": [ref[s][p][0] for s in segs for p in drawn_fill],
    }

    out = {
        "seed": a.seed,
        "inputs": {
            "reference": {"path": str(ref_path), "sha256": sha256(ref_path)},
            "arm": {"path": str(arm_path), "sha256": sha256(arm_path)},
        },
        "segments": {s: lengths[s] for s in segs},
        "shared_positions": {"low": 0, "high": hi, "count": hi + 1},
        "params": {"n_sample": N_SAMPLE, "flat_per_segment": N_FLAT_PER_SEG,
                   "random": N_RANDOM, "flip": N_FLIP},
        "strata": {
            "flat": {"count": len(flat), "per_segment": N_FLAT_PER_SEG,
                     "flattest_cut_by_segment": cut},
            "random": {"count": len(drawn_random)},
            "flip": {"count": len(drawn_flip), "available": len(pool),
                     "union_rate": len(pool) / len(valid),
                     "mean_segment_rate": sum(len(v) for v in flips_by_seg.values()) / (len(segs) * len(valid)),
                     "available_per_segment": {s: len(v) for s, v in flips_by_seg.items()},
                     "drawn_per_segment": drawn_flip_per_seg},
            "fill": {"count": len(drawn_fill)},
            "overlap": overlap,
        },
        "retained_mass": {
            "population": dist(pop_mass),
            "sample": dist(sample_mass),
            "by_stratum": {k: dist(v) for k, v in per_stratum.items() if v},
        },
        "positions": positions,
        "membership": {"flat": flat, "random": drawn_random, "flip": drawn_flip, "fill": drawn_fill},
    }

    pathlib.Path(a.positions_out).write_text("".join(f"{p}\n" for p in positions), encoding="utf-8")
    pathlib.Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"\nstrata: flat {len(flat)} (union of {N_FLAT_PER_SEG} per segment), random {len(drawn_random)}, "
          f"flip {len(drawn_flip)} of {len(pool)} available (union over the segments; "
          f"{100 * out['strata']['flip']['mean_segment_rate']:.2f} % per segment on average), "
          f"fill {len(drawn_fill)}")
    print(f"overlap: {overlap} -> union {len(chosen)}, filled to {len(positions)}")
    print("\nretained mass of the reference's top-256 (population, then the sample and its strata):")
    for label, d in [("population", out["retained_mass"]["population"]),
                     ("sample", out["retained_mass"]["sample"])] + \
                    [(f"  {k}", v) for k, v in out["retained_mass"]["by_stratum"].items()]:
        print(f"  {label:12s} n {d['n']:6d}  min {d['min']:.6f}  p01 {d['p01']:.6f}  p10 {d['p10']:.6f}  "
              f"median {d['median']:.6f}  mean {d['mean']:.6f}  max {d['max']:.6f}")
    print("\nflattest cut per segment (the 40th smallest retained mass):")
    for s in segs:
        print(f"  {s}  {cut[s]:.6f}   flips {len(flips_by_seg[s])} of {len(valid)} "
              f"({100 * len(flips_by_seg[s]) / len(valid):.2f} %)")
    print(f"\nwrote {a.positions_out} ({len(positions)} lines) and {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
