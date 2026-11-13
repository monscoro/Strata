#!/usr/bin/env python
"""E1-E5 of bench/results/2026-11-13-int8-references/README.md, applied to the run's paired tables.

The gates were fixed BEFORE the run; this script only applies them, so the bound cannot be chosen after
the numbers are known.  It reads `arms/paired-int8refs.json` (written by quant_quality.py compare --paired)
and uses the whole-segment paired differences: mean, sd, sem over the 8 segments of cwe-4k.  The
one-sided 95 % upper bound is mean + t(0.95, 7) * sem with t = 1.8946, and the two-sided 95 % interval is
mean +- t(0.975, 7) * sem with t = 2.3646 -- both fixed here, not read off the run.

Usage:  python bench/results/2026-11-13-int8-references/eval-int8-refs.py [--arity 8]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-13-int8-references")

# the pre-declared non-inferiority bound (README.md, fixed 2026-11-13 before the run)
BOUND_UPPER = 0.0020     # the one-sided 95 % upper bound of q4 - int8 must be below this
BOUND_POINT = 0.0010     # and the point estimate at or below this

T975 = {7: 2.3646, 9: 2.2622}   # two-sided, per segment count
T950 = {7: 1.8946, 9: 1.8125}   # one-sided

CASES = [
    ("E1a", "R2a - R1  (was ist die Rotation auf int8 wert?)", "informativ (kein Gate)",
     "cwe4k-int8-rot.tsv", "cwe4k-int8.tsv"),
    ("E1b", "R2b - R1  (Rotation + Kalibrierung gegen den Produktpfad)", "informativ (kein Gate)",
     "cwe4k-int8-rot-center.tsv", "cwe4k-int8.tsv"),
    ("E5", "R2b - R2a  (wirkt die Kalibrierung auf int8?)", "informativ (kein Gate)",
     "cwe4k-int8-rot-center.tsv", "cwe4k-int8-rot.tsv"),
    ("E2", "q4lm32w+C - R2b  (ist der Produktarm dem fairen int8 nicht unterlegen?)", "NICHTUNTERLEGENHEIT",
     "cwe4k-q4lm32w-center.tsv", "cwe4k-int8-rot-center.tsv"),
    ("E3", "q4u32 - R2b  (ist der Fast-Path-Kandidat dem fairen int8 nicht unterlegen?)", "NICHTUNTERLEGENHEIT",
     "cwe4k-q4u32.tsv", "cwe4k-int8-rot-center.tsv"),
    ("E4", "q4u32z - q4u32  (reproduziert der Shear seinen 4K-Befund?)", "Shear: negativ, 0,001-0,003",
     "cwe4k-q4u32z.tsv", "cwe4k-q4u32.tsv"),
]


def verdict(kind, mean, lo, hi):
    if kind == "NICHTUNTERLEGENHEIT":
        ok = (hi < BOUND_UPPER) and (mean <= BOUND_POINT)
        return ("BESTANDEN" if ok else "NICHT BESTANDEN"), \
            f"obere 95-%-Grenze {hi:+.5f} < {BOUND_UPPER:+.4f}? {'ja' if hi < BOUND_UPPER else 'nein'}; " \
            f"Punkt {mean:+.5f} <= {BOUND_POINT:+.4f}? {'ja' if mean <= BOUND_POINT else 'nein'}"
    if kind == "Shear: negativ, 0,001-0,003":
        neg = hi < 0.0
        mag = 0.001 <= -mean <= 0.003
        return ("BESTANDEN" if (neg and mag) else "NICHT BESTANDEN"), \
            f"ganzes Intervall negativ? {'ja' if neg else 'nein'}; |Mittel| in [0,001; 0,003]? " \
            f"{'ja' if mag else 'nein'} ({-mean:.5f})"
    return "informativ", "kein Gate"


def pair_of(tab, a, b):
    """A - B from the table, whichever order the compare pass wrote it in.

    quant_quality.py names a pair after the order of the arms it was given, so half of the questions here
    are stored the other way round.  The reversal flips the sign of the mean, the t and the end points; the
    spread (sd, sem) does not change."""
    s = tab.get(f"{a} - {b}")
    if s is not None:
        return s
    s = tab.get(f"{b} - {a}")
    if s is None:
        return None
    out = dict(s)
    out["mean"], out["t"] = -s["mean"], -s["t"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DATA),
                    help="the bench directory holding arms/paired-int8refs.json; the raw data live "
                         "outside the repo, in Strata-data")
    ap.add_argument("--table", default="whole", choices=["whole", "answer"],
                    help="the whole-segment table (the campaign's primary) or the answer region")
    args = ap.parse_args()
    d = Path(args.dir)
    p = d / "arms" / "paired-int8refs.json"
    if not p.exists():
        raise SystemExit(f"no {p} - the compare step has not written it")
    tab = json.loads(p.read_text(encoding="utf-8"))[args.table]
    if tab is None:
        raise SystemExit("the answer-region table is missing (was --ids passed?)")

    print(f"paired table: {args.table}; gates from README.md, fixed before the run")
    print(f"non-inferiority: one-sided 95 % upper < {BOUND_UPPER:+.4f}, point <= {BOUND_POINT:+.4f}\n")
    n = None
    for tag, question, kind, a, b in CASES:
        key = f"{a} - {b}"
        s = pair_of(tab, a, b)
        if s is None:
            print(f"{tag}: MISSING pair {key}\n")
            continue
        n = s["n"]
        mean, sem = s["mean"], s["sem"]
        lo = mean - T975.get(n, 2.306) * sem
        hi2 = mean + T975.get(n, 2.306) * sem
        hi1 = mean + T950.get(n, 1.8595) * sem
        v, why = verdict(kind, mean, lo, hi1)
        print(f"{tag}  {question}")
        print(f"     {a} - {b}")
        print(f"     mean {mean:+.5f}  sem {sem:.5f}  sd {s['sd']:.5f}  t {s['t']:+.2f}  "
              f"segments up {s['up']}/{s['n']}")
        print(f"     two-sided 95 % interval [{lo:+.5f}, {hi2:+.5f}];  "
              f"one-sided 95 % upper {hi1:+.5f}")
        print(f"     -> {v}: {why}\n")
    if n and n != 8:
        print(f"note: {n} segments, not 8 -- the t values above assume the campaign's geometry")


if __name__ == "__main__":
    main()
