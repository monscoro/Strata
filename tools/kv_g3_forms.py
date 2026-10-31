#!/usr/bin/env python3
"""G3, post hoc: which form does the answer take, and does the form carry the difference?

    python tools/kv_g3_forms.py --set data/quant_quality/cwe-4k \\
        --base int8=runs\\int8.json --run fp16=runs\\fp16.json --run q4lm8v=runs\\q4lm8v.json ...

**This is not the pre-registered metric.** `tools/kv_g3_score.py` counts the gold words anywhere in the
generated text, as registered. That number moves with how much the model writes: a run that answers as a
comma list names ten words in 42 tokens, a run that writes "1. **word**: this word appears 26 times" needs
about 15 tokens per word and is cut by the budget. This tool reports the confound, it does not replace
the registered result:

- `form`: the shape of the answer (`list`, `numbered`, `prose`), a heuristic on the first characters;
- `distinct_vocab`: how many of the set's own coded words the text names (the 80 words of the eight gold
  lists; English six-letter words are excluded, which `recall_first10` in the scorer does not do);
- `first10`: the gold words among the first ten distinct coded words — the answer reading, robust to a
  longer answer but not to a cut one;
- `anywhere`: the registered reading, for comparison.

Differences are paired over segments with the same 1.895 sd/sqrt(8) bound as the scorer.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re

T7_95 = 1.895

WORD = re.compile(r"[a-z]+")
NUMBERED = re.compile(r"^\s*(\d+\.|[-*]\s+\*\*)")
PROSE = re.compile(r"^\s*(To determine|Here is|The user|We need|I need|Let me|Let's|First,)")


def form(text: str) -> str:
    if PROSE.match(text):
        return "prose"
    if NUMBERED.match(text):
        return "numbered"
    return "list"


def paired(a: dict, b: dict, key: str) -> dict | None:
    segs = [s for s in a if s in b]
    if len(segs) < 2:
        return None
    d = [a[s][key] - b[s][key] for s in segs]
    n = len(d)
    m = math.fsum(d) / n
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in d) / (n - 1))
    sem = sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "bound": T7_95 * sem,
            "a_worse": sum(1 for x in d if x < 0), "a_better": sum(1 for x in d if x > 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/quant_quality/cwe-4k/gold.json")
    ap.add_argument("--set", default="data/quant_quality/cwe-4k")
    ap.add_argument("--base", default="int8", help="the arm every difference is paired against")
    ap.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    gold = {g["name"]: g["answer"] for g in json.load(open(a.gold, encoding="utf-8"))}
    vocab = {w for words in gold.values() for w in words}   # the set's own coded words, 80 of them
    runs, res = {}, {"vocab": sorted(vocab), "runs": {}, "paired": {}, "note": "post hoc, see the docstring"}
    print(f"{'run':9s} {'seg':4s} {'form':8s} {'tok':>4s} {'words':>5s} {'distinct':>8s} "
          f"{'first10':>7s} {'anywhere':>8s}")
    for spec in a.run:
        name, _, path = spec.partition("=")
        d = json.load(open(path, encoding="utf-8"))
        per = {}
        for s in d["segments"]:
            ws = WORD.findall(s["text"].lower())
            coded = [w for w in ws if w in vocab]
            first = list(dict.fromkeys(coded))[:10]
            g = gold[s["name"]]
            f10 = sum(1 for w in g if w in set(first)) / len(g)
            anyw = sum(1 for w in g if w in set(ws)) / len(g)
            per[s["name"]] = {"form": form(s["text"]), "tokens": len(s["ids"]), "words": len(ws),
                              "distinct_vocab": len(set(coded)), "first10": f10, "anywhere": anyw,
                              "stopped": s["stopped"]}
            print(f"{name:9s} {s['name']:4s} {per[s['name']]['form']:8s} {len(s['ids']):4d} {len(ws):5d} "
                  f"{len(set(coded)):8d} {f10:7.2f} {anyw:8.2f}")
        runs[name] = per
        segs = sorted(per)
        res["runs"][name] = {"per_segment": per,
                             "mean_first10": math.fsum(per[s]["first10"] for s in segs) / len(segs),
                             "mean_anywhere": math.fsum(per[s]["anywhere"] for s in segs) / len(segs),
                             "forms": {f: sum(1 for s in segs if per[s]["form"] == f)
                                       for f in ("list", "numbered", "prose")}}

    print(f"\n{'run':9s} {'first10':>8s} {'anywhere':>9s}  forms (list/numbered/prose)")
    for name in runs:
        r = res["runs"][name]
        f = r["forms"]
        print(f"{name:9s} {r['mean_first10']:8.3f} {r['mean_anywhere']:9.3f}  {f['list']}/{f['numbered']}/{f['prose']}")

    if a.base in runs:
        print(f"\npaired against {a.base} (mean, bound = 1.895 sd/sqrt(8), segments worse/better):")
        for name in runs:
            if name == a.base:
                continue
            for key in ("first10", "anywhere"):
                p = paired(runs[name], runs[a.base], key)
                res["paired"].setdefault(name, {})[key] = p
                flag = ">" if p["mean"] > p["bound"] else "<" if p["mean"] < -p["bound"] else "~"
                print(f"  {name:9s} {key:9s} {p['mean']:+.3f} {flag}  bound {p['bound']:.3f}  "
                      f"{p['a_worse']}/{p['a_better']}")

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
