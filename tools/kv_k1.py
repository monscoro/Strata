"""K1 of the ladder for the S3 arms: the per-position paired test, not the 8-segment one.

KL(arm) per position is KL(int8 || arm) over int8's top-k tokens, and the ladder's K1 asks for the
mean of that over cwe-4k, its one-sided 95 % upper bound against int8 (+0.001 nats) and how many of
the 8 segments int8 leads.

Usage: python tools/kv_k1.py --ref int8.tsv --run a.tsv [--run b.tsv ...] [--kl-topk 256]
"""

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quant_quality as qq  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--kl-topk", type=int, default=256)
    ap.add_argument("--json")
    a = ap.parse_args()
    ref = qq.read_run(a.ref)
    out = {"ref": a.ref, "kl_topk": a.kl_topk, "runs": {}, "pairs": {}}
    print(f"reference {a.ref}, KL support: REF's {a.kl_topk} most likely tokens")
    print(f"{'run':34s} {'pos':>6s} {'mean KL':>9s} {'sd':>8s} {'t':>7s} {'95% up':>9s} {'seg int8 leads':>15s}")
    per_pos = {}
    for run in a.run:
        r = qq.read_run(run)
        vals, seg_mean, keyed = [], {}, {}
        for seg in ref:
            if seg not in r:
                continue
            v = []
            for pos, kr in ref[seg].items():
                if pos not in r[seg]:
                    continue
                ka = r[seg][pos][4]
                k = qq.kl_between(kr[4], ka, a.kl_topk)
                keyed[(seg, pos)] = k
                v.append(k)
            if v:
                seg_mean[seg] = sum(v) / len(v)
                vals.extend(v)
        per_pos[pathlib.Path(run).name] = keyed
        n = len(vals)
        mean = sum(vals) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1))
        sem = sd / math.sqrt(n)
        t = mean / sem
        # one-sided 95 % upper bound (normal approximation; n is 32,746, so t and z agree to 4 decimals)
        up = mean + 1.645 * sem
        leads = sum(1 for s in seg_mean if seg_mean[s] > 0)
        print(f"{pathlib.Path(run).name:34s} {n:6d} {mean:9.5f} {sd:8.4f} {t:7.2f} {up:9.5f} "
              f"{leads:>10d}/8")
        out["runs"][pathlib.Path(run).name] = {"n": n, "mean_kl": mean, "sd": sd, "sem": sem, "t": t,
                                               "upper95": up, "segments_int8_leads": leads,
                                               "segment_mean_kl": seg_mean,
                                               "k1": bool(mean <= 0.0 and up < 0.001 and leads <= 4)}
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")
    if len(a.run) > 1:
        print("\npaired per-position difference (same positions, so the text cancels):")
        names = [pathlib.Path(x).name for x in a.run]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ka, kb = per_pos[names[i]], per_pos[names[j]]
                keys = [k for k in ka if k in kb]
                d = [ka[k] - kb[k] for k in keys]
                n = len(d)
                mean = sum(d) / n
                sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
                sem = sd / math.sqrt(n)
                t = mean / sem
                up = mean + 1.645 * sem
                seg = {}
                for (s, _), x in ((k, ka[k] - kb[k]) for k in keys):
                    seg.setdefault(s, []).append(x)
                leads = sum(1 for s in seg if sum(seg[s]) / len(seg[s]) > 0)
                print(f"  {names[i]:30s} - {names[j]:30s} {mean:+.5f}  sd {sd:.4f}  t {t:+.2f}  "
                      f"95% CI [{mean - 1.96 * sem:+.5f}, {mean + 1.96 * sem:+.5f}]  "
                      f"{names[i]} worse in {leads}/8 segments")
                out["pairs"][f"{names[i]} - {names[j]}"] = {
                    "n": n, "mean": mean, "sd": sd, "sem": sem, "t": t,
                    "ci95_low": mean - 1.96 * sem, "ci95_high": mean + 1.96 * sem,
                    "segments_a_worse": leads}
        if a.json:
            pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
