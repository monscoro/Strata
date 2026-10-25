#!/usr/bin/env python3
"""S1 (catalog: docs/paper/Ansatz_Improvements.md, row E1): where does the indexer's selection land relative to the
window, and how close is the selection boundary?

One instrumented prefill of a long prompt, two debug hooks, no new kernel and no change to the default path:

  STRATA_KV_SELSTATS=<path>      (src/prefill/prefill.cpp) one line per sampled query and QSA layer: the share of the
                                 selected cells in R (the sinks), in S (the recent window) and in neither, plus the
                                 block-score margin at the selection boundary.
  STRATA_SEL_OVERLAP             (already in the engine) the union of 16 adjacent queries' selections against the sum
                                 of their widths - how much the selection persists across neighbouring queries.

The buckets follow the kernel's own predicate (include/strata/kernels/qsa_decode_attn.hpp, qsa_win_slot): a cell `c` at
horizon `n` is read exact when `c < R` (sink) or `c >= n - S` (recent, tag-checked); everything else reads quantized.
The selection itself is over whole blocks (include/strata/kernels/qsa_select.hpp), so `min_sel - max_unsel` is exactly
the boundary margin: positive = the selection separates cleanly, zero or negative = a tie the order of the block list
decides.

  python tools/kv_selstats.py run --out <dir> [--exe <path>] [--env K=V ...]
  python tools/kv_selstats.py report <dir>/selstats.tsv
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from quant_quality import FIXED_ENV, engine_args, mem_available_gb  # noqa: E402  one definition of the engine's args

WINDOW_ENV = {"STRATA_KV_WINDOW": "1", "STRATA_KV_WINDOW_S": "256", "STRATA_KV_WINDOW_R": "64"}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def cmd_run(a) -> int:
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    st = json.load(open(a.ids, encoding="utf-8"))
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # One passage: the set's segments in order, no separator.  Each segment is its own cwe document, so the seams are
    # document boundaries the model never saw in training - fine for selection statistics, and said so in the report.
    prompt = [i for s in st["segments"] for i in s["ids"]]
    if a.tokens and a.tokens < len(prompt):
        prompt = prompt[:a.tokens]

    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    if cfg.get("lib_dirs"):
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(cfg["lib_dirs"] + ([env[var]] if env.get(var) else []))
    env.update(FIXED_ENV)
    env.update(WINDOW_ENV)
    env["STRATA_KV_SELSTATS"] = str(out / "selstats.tsv")
    env["STRATA_KV_SELSTATS_EVERY"] = str(a.every)
    if a.overlap:
        env["STRATA_SEL_OVERLAP"] = "1"
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v

    if mem_available_gb() < a.min_free_gb:
        raise SystemExit(f"only {mem_available_gb():.0f} GB available (< {a.min_free_gb}): not starting an engine")

    args = engine_args(cfg, a.gpu_experts, len(prompt), a.kv)
    if a.exe:
        args[0] = a.exe
    k = args.index("--short-read")
    del args[k:k + 2]                      # the default (64): this prompt takes the batched prefill path
    k = args.index("--prefill")
    args[k + 1] = "auto"                   # let the engine size the chunk (generate.cpp caps it at 6,144 cells)
    k = args.index("--max-context")
    args[k + 1] = str(len(prompt) + 512)

    if a.dry_run:
        print("engine:", " ".join(args))
        print(f"prompt: {len(prompt)} tokens from {len(st['segments'])} segments of {st['version']}")
        print("env:", json.dumps({k: env[k] for k in sorted(env) if k.startswith("STRATA_")}, indent=1))
        print("selstats:", env["STRATA_KV_SELSTATS"])
        return 0

    errf = open(out / "stderr.txt", "w")
    print("engine:", " ".join(args), flush=True)
    print(f"prompt: {len(prompt)} tokens from {len(st['segments'])} segments of {st['version']}", flush=True)
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                         cwd=cfg.get("cwd"), env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY (exit {p.wait()}); see {out}/stderr.txt")

    p.stdin.write("GEN 1 " + ",".join(map(str, prompt)) + "\n")
    p.stdin.flush()
    done = None
    for line in p.stdout:
        if line.startswith(("DONE", "ERR")):
            done = line.strip()
            break
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    if done is None or not done.startswith("DONE") or "nonfinite" in done:
        raise SystemExit(f"the engine answered {done!r}; see {out}/stderr.txt")

    (out / "index.json").write_text(json.dumps({
        "prompt_tokens": len(prompt), "segments": [s["name"] for s in st["segments"]], "set": st["version"],
        "args": args, "env": sorted(k for k in env if k.startswith("STRATA_")),
        "window": WINDOW_ENV, "every": a.every,
        "exe_sha256": sha256(pathlib.Path(args[0])),
    }, indent=1))
    print("done; selstats.tsv written", flush=True)
    return 0


def cmd_report(a) -> int:
    rows = []
    with open(a.tsv, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != len(head):
                continue
            rows.append(dict(zip(head, p)))
    if not rows:
        raise SystemExit("no rows")

    def num(r, k):
        return float(r[k])

    layers = sorted({int(r["layer"]) for r in rows})
    print(f"{a.tsv}: {len(rows)} samples, layers {layers}, "
          f"n_kv {int(min(num(r, 'n_kv') for r in rows))}..{int(max(num(r, 'n_kv') for r in rows))}")

    buckets = [(0, 319, "<=319 (selection = identity)"), (320, 2051, "320..2051 (below the 2,051 block bound)"),
               (2052, 8191, "2052..8191"), (8192, 16383, "8192..16383"), (16384, 10 ** 9, ">=16384")]

    W = 42
    print("\nshare of the selected CELLS, by horizon (all layers pooled)")
    print(f"{'horizon':{W}s} {'samples':>8s} {'cells':>10s} {'in R':>7s} {'in S':>7s} {'in B':>7s} {'outside':>8s}")
    for lo, hi, label in buckets:
        sel = [r for r in rows if lo <= num(r, "n_kv") <= hi]
        if not sel:
            continue
        tot = sum(num(r, "in_r") + num(r, "in_s") + num(r, "in_b") + num(r, "outside") for r in sel)
        ir = sum(num(r, "in_r") for r in sel)
        is_ = sum(num(r, "in_s") for r in sel)
        ib = sum(num(r, "in_b") for r in sel)
        ou = sum(num(r, "outside") for r in sel)
        print(f"{label:{W}s} {len(sel):8d} {int(tot):10d} {100*ir/tot:6.1f}% {100*is_/tot:6.1f}% "
              f"{100*ib/tot:6.1f}% {100*ou/tot:7.1f}%")

    print("\nthe same per QSA layer, horizon >= 320 (where the selection actually selects)")
    print(f"{'layer':6s} {'samples':>8s} {'cells':>10s} {'in R':>7s} {'in S':>7s} {'in B':>7s} {'outside':>8s}")
    for l in layers:
        sel = [r for r in rows if int(r["layer"]) == l and num(r, "n_kv") >= 320]
        if not sel:
            continue
        tot = sum(num(r, "in_r") + num(r, "in_s") + num(r, "in_b") + num(r, "outside") for r in sel)
        ir = sum(num(r, "in_r") for r in sel)
        is_ = sum(num(r, "in_s") for r in sel)
        ib = sum(num(r, "in_b") for r in sel)
        ou = sum(num(r, "outside") for r in sel)
        print(f"{l:6d} {len(sel):8d} {int(tot):10d} {100*ir/tot:6.1f}% {100*is_/tot:6.1f}% "
              f"{100*ib/tot:6.1f}% {100*ou/tot:7.1f}%")

    print("\nselection boundary: margin = min(score of a selected block) - max(score of an unselected block)")
    print(f"{'horizon':{W}s} {'n':>6s} {'p1':>10s} {'p10':>10s} {'p50':>10s} {'p90':>10s} {'<=0':>6s}")
    for lo, hi, label in buckets:
        sel = [r for r in rows if lo <= num(r, "n_kv") <= hi and num(r, "max_unsel") > -1e29]
        if not sel:
            continue
        m = sorted(num(r, "min_sel") - num(r, "max_unsel") for r in sel)
        q = lambda f: m[min(len(m) - 1, int(f * (len(m) - 1) + 0.5))]  # noqa: E731
        neg = sum(1 for x in m if x <= 0.0)
        print(f"{label:{W}s} {len(m):6d} {q(0.01):10.4g} {q(0.10):10.4g} {q(0.50):10.4g} {q(0.90):10.4g} "
              f"{100*neg/len(m):5.1f}%")

    print("\nmargin relative to the score spread of the query (margin / (s_max - s_min))")
    print(f"{'horizon':{W}s} {'n':>6s} {'p1':>9s} {'p10':>9s} {'p50':>9s} {'p90':>9s}")
    for lo, hi, label in buckets:
        sel = [r for r in rows if lo <= num(r, "n_kv") <= hi and num(r, "max_unsel") > -1e29
               and num(r, "s_max") > num(r, "s_min")]
        if not sel:
            continue
        m = sorted((num(r, "min_sel") - num(r, "max_unsel")) / (num(r, "s_max") - num(r, "s_min")) for r in sel)
        q = lambda f: m[min(len(m) - 1, int(f * (len(m) - 1) + 0.5))]  # noqa: E731
        print(f"{label:{W}s} {len(m):6d} {q(0.01):8.1%} {q(0.10):8.1%} {q(0.50):8.1%} {q(0.90):8.1%}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default="strata-swift-iq3_xxs.json")
    r.add_argument("--ids", default="data/quant_quality/cwe-4k/set.ids.json")
    r.add_argument("--out", required=True)
    r.add_argument("--exe", default=None, help="override the config's engine (e.g. a debug build)")
    r.add_argument("--kv", default="q4lm8v")
    r.add_argument("--tokens", type=int, default=0, help="cut the passage to this many tokens (0 = all)")
    r.add_argument("--every", type=int, default=32, help="sample every Nth query of a selection batch")
    r.add_argument("--overlap", action="store_true", help="also run the existing STRATA_SEL_OVERLAP hook")
    r.add_argument("--gpu-experts", default=None)
    r.add_argument("--min-free-gb", type=float, default=8.0)
    r.add_argument("--dry-run", action="store_true", help="print the engine args and env, start nothing")
    r.add_argument("--env", action="append", help="KEY=VALUE, repeatable")
    r.set_defaults(func=cmd_run)
    q = sub.add_parser("report")
    q.add_argument("tsv")
    q.set_defaults(func=cmd_report)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
