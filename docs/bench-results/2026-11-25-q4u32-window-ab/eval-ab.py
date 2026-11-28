#!/usr/bin/env python3
"""The decision rule fixed in README.md, computed mechanically.

`d = KL(q4u32w) - KL(q4u32)`, paired over the same fp16-scored positions. Negative means the window helps.
The window is established only when mean(d) < 0 AND the upper 95% bound is < 0 by both the per-position
t interval and the 32x4096 block bootstrap (10,000 resamples, seed 20261125).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QQ_PATH = ROOT / "terminal" / "tools" / "quant_quality.py"
DATA = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-25-q4u32-window-ab")

spec = importlib.util.spec_from_file_location("quant_quality", QQ_PATH)
qq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qq)

BLOCK = 4096
RESAMPLES = 10000
SEED = 20261125


def tci(d: list[float]) -> tuple[float, float, float, float]:
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    t = m / sem if sem else (0.0 if m == 0.0 else float("nan"))
    return m, t, m - 1.96 * sem, m + 1.96 * sem


def block_boot(d: list[float]) -> tuple[float, float, int]:
    nblk = math.ceil(len(d) / BLOCK)
    means = [sum(d[i * BLOCK:(i + 1) * BLOCK]) / len(d[i * BLOCK:(i + 1) * BLOCK])
             for i in range(nblk)]
    rng = __import__("random").Random(SEED)
    # numpy is not guaranteed on the Terminal box; pure Python is enough for 32 blocks.
    boot = []
    for _ in range(RESAMPLES):
        pick = [means[rng.randrange(nblk)] for _ in range(nblk)]
        boot.append(sum(pick) / nblk)
    boot.sort()
    return boot[int(0.025 * RESAMPLES)], boot[int(0.975 * RESAMPLES) - 1], nblk


def top1(run: dict, ref: dict) -> tuple[int, int]:
    right = n = 0
    for seg, dd in ref.items():
        for pos, rec in dd.items():
            other = run.get(seg, {}).get(pos)
            if other is None:
                continue
            n += 1
            right += rec[4][0][0] == other[4][0][0]
    return right, n


def mcnemar(a: dict, b: dict, ref: dict) -> tuple[int, int, float]:
    """a only right | b only right, two-sided exact p."""
    a_only = b_only = 0
    for seg, dd in ref.items():
        for pos, rec in dd.items():
            x, y = a.get(seg, {}).get(pos), b.get(seg, {}).get(pos)
            if not x or not y:
                continue
            truth = rec[4][0][0]
            a_only += truth == x[4][0][0] and truth != y[4][0][0]
            b_only += truth == y[4][0][0] and truth != x[4][0][0]
    nn = a_only + b_only
    if not nn:
        return a_only, b_only, float("nan")
    k = min(a_only, b_only)
    logs = [math.lgamma(nn + 1) - math.lgamma(i + 1) - math.lgamma(nn - i + 1)
            for i in range(k + 1)]
    top = max(logs)
    log_sum = top + math.log(sum(math.exp(v - top) for v in logs))
    return a_only, b_only, min(1.0, math.exp(math.log(2.0) + log_sum - nn * math.log(2.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DATA / "eval.json"))
    args = ap.parse_args()
    ref = qq.read_run(str(DATA / "t1-fp16.tsv"))
    plain = qq.read_run(str(DATA / "t1-q4u32.tsv"))
    win = qq.read_run(str(DATA / "t1-q4u32w.tsv"))
    sp, sw = qq.kl_series(ref, plain), qq.kl_series(ref, win)
    keys = sorted(set(sp) & set(sw))
    d = [sw[k] - sp[k] for k in keys]
    m, t, lo, hi = tci(d)
    blo, bhi, nblk = block_boot(d)
    passed = bool(m < 0 and hi < 0 and bhi < 0)

    tp, np_ = top1(plain, ref)
    tw, nw = top1(win, ref)
    ao, bo, p = mcnemar(win, plain, ref)

    print(f"paired positions: {len(keys)}")
    print(f"mean KL q4u32  {sum(sp.values()) / len(sp):.5f}   top-1 {100 * tp / np_:.2f}%")
    print(f"mean KL q4u32w {sum(sw.values()) / len(sw):.5f}   top-1 {100 * tw / nw:.2f}%")
    print(f"d = q4u32w - q4u32: mean {m:+.5f}, t {t:+.2f}, 95% [{lo:+.5f}, {hi:+.5f}], "
          f"block boot [{blo:+.5f}, {bhi:+.5f}] ({nblk} blocks)")
    print(f"top-1 McNemar (window only right | plain only right): {ao} | {bo}, p = {p:.4f}")
    print("Entscheidung: " + ("Fenster etablieren" if passed else
                              "kein nachgewiesener Qualitätsgewinn; q4u32 bleibt der Parameter"))

    out = {"positions": len(keys), "blocks": nblk, "seed": SEED,
           "kl": {"q4u32": sum(sp.values()) / len(sp), "q4u32w": sum(sw.values()) / len(sw)},
           "top1": {"q4u32": tp / np_, "q4u32w": tw / nw},
           "difference": {"mean": m, "t": t, "ci_low": lo, "ci_high": hi,
                          "boot_low": blo, "boot_high": bhi},
           "mcnemar": {"window_only": ao, "plain_only": bo, "p": p},
           "window_established": passed}
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
