#!/usr/bin/env python
"""The 128K error profile: where along the text does the error sit?

The 128K int8 arm reproduces the predecessor's whole-segment mean exactly (0.06435 on the second
text) but its tail statistics are one to two orders of magnitude larger, and the last 1,024
positions carry a mean KL of about six nats.  This script prints the mean KL per block of 4,096
positions and the position of the answer region, so the shape of the error can be read off: a
tail that grows towards the end of the text is a recall failure (the code word the task asks for
is scattered through the text and the quantized cache loses it), a flat tail is a codec that is
uniformly worse.

Usage:
  python tools/kv_128k_profile.py --ref <fp16.tsv> --int8 <int8.tsv> [--arm <arm.tsv>] [--ids <ids.json>]
"""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quant_quality", HERE / "quant_quality.py")
qq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--arm")
    ap.add_argument("--ids")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--kl-topk", type=int, default=qq.TOPK)
    args = ap.parse_args()

    ref = qq.read_run(args.ref)
    arms = {"int8": qq.read_run(args.int8)}
    if args.arm:
        arms["arm"] = qq.read_run(args.arm)

    answer_at = None
    if args.ids:
        st = json.load(open(args.ids))
        answer_at = st["segments"][0].get("answer_at")

    series = {name: qq.kl_series(ref, r, args.kl_topk) for name, r in arms.items()}
    keys = sorted(set.intersection(*[set(s) for s in series.values()]))
    n = len(keys)
    print(f"positions: {n}   block: {args.block}   answer region starts at: {answer_at}")
    print()
    print(f"{'block':>6} {'positions':>12} {'int8 mean KL':>14} {'arm mean KL':>14} {'int8 max':>12}")
    print("-" * 64)
    for i in range(0, n, args.block):
        chunk = keys[i:i + args.block]
        row = f"{i//args.block:6d} {chunk[0][1]:8d}-{chunk[-1][1]:<8d}"
        for name in ("int8", "arm"):
            if name in series:
                vals = [series[name][k] for k in chunk]
                row += f" {sum(vals)/len(vals):14.5f}"
            else:
                row += f" {'---':>14}"
        mx = max(series["int8"][k] for k in chunk)
        print(f"{row} {mx:12.3f}")
    print()
    if answer_at:
        tail = [k for k in keys if k[1] >= answer_at]
        if tail:
            v = [series["int8"][k] for k in tail]
            print(f"answer region ({len(tail)} positions): int8 mean KL {sum(v)/len(v):.4f}, max {max(v):.3f}")
            pre = [k for k in keys if k[1] < answer_at]
            vp = [series["int8"][k] for k in pre]
            print(f"before the answer region ({len(pre)} positions): int8 mean KL {sum(vp)/len(vp):.5f}")


if __name__ == "__main__":
    main()
