#!/usr/bin/env python3
"""G3: the KV arms, generating instead of scoring.

    python tools/kv_g3_run.py --config bench/configs/cwe4k-swift-iq3_xxs.json \\
        --set data/quant_quality/cwe-4k --kv int8 --new 64 --out RUN.json

One `strata --serve` engine per run (the config's exe and args plus the yardstick's fixed ones, exactly as
`quant_quality.py run` starts it), then one `GEN` per segment of the set: the prompt is the set's own token
ids up to `answer_at`, greedy (no sampling keys, so the engine's default), `ckpt=0` so no segment resumes
from another, and the engine's `T <id>` lines are the answer.  Generation stops at the model's end-of-text
token or after `--new` tokens, whichever comes first.

The prompt is read on the serving path: `--short-read 64` (the engine's default, the one the app uses) and
`--prefill 512` (the pinned config's chunk).  A window arm therefore keeps `S + R + 512` rows at a 4,096-token
prompt, not the `S + R + 1` rows the teacher-forced S3 runs saw - see the G3 pre-registration.

RUN.json: the kv format, the environment, the engine's argv, and per segment the prompt length, the
generated ids, the decoded text and the seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quant_quality as QQ  # noqa: E402

DROP_SPEC_VALUE = {"--spec", "--spec-min-p"}
# The assistant turn with the thinking block already closed: the product's non-thinking path. Without it the model
# opens a ` thinking` block, and on this task it then enumerates the whole coded text word by word instead of answering
# (measured on the anchor: 6 of 8 segments at 2,048 tokens).
CLOSE_THINKING = "<|im_start|>assistant\n thinking\n\n</think>\n\n"


def drop_spec(args: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(args):
        if args[i] in DROP_SPEC_VALUE:
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def set_spec(args: list[str], window: int) -> list[str]:
    out, i = list(args), 0
    while i < len(out) - 1:
        if out[i] == "--spec":
            out[i + 1] = str(window)
            return out
        i += 1
    return out + ["--spec", str(window)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", default="data/quant_quality/cwe-4k")
    ap.add_argument("--ids", help="set.ids.json (default: <set>/set.ids.json)")
    ap.add_argument("--kv", default="fp16")
    ap.add_argument("--env", action="append", metavar="K=V",
                    help="an environment variable for the engine, e.g. the window arm's STRATA_KV_WINDOW=1")
    ap.add_argument("--new", type=int, default=64, help="generated tokens per segment (64 = the gold answer's "
                                                        "32-36 tokens plus slack; see the pre-registration)")
    ap.add_argument("--short-read", type=int, default=64,
                    help="prompt tokens read through the decode windows (64 = the engine's and the app's default)")
    ap.add_argument("--segments", help="comma list of segment names (default: all)")
    ap.add_argument("--drop-spec", action="store_true", help="drop --spec/--spec-min-p (the greedy cross-check)")
    ap.add_argument("--spec", type=int, help="replace the config's --spec window (a native IQ pack needs >= 2)")
    ap.add_argument("--close-thinking", action="store_true",
                    help="append the assistant turn with the thinking block closed (the product's non-thinking path)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    ids_path = pathlib.Path(a.ids or (pathlib.Path(a.set) / "set.ids.json"))
    st = json.load(open(ids_path))
    tok = QQ.load_tokenizer(cfg["tokenizer"])
    out = pathlib.Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    base = drop_spec(list(cfg["args"])) if a.drop_spec else list(cfg["args"])
    if a.spec is not None:
        base = set_spec(base, a.spec)
    # The context must hold the prompt *and* the generated tokens: engine_args sizes it from the segment length
    # alone, which was enough for scoring (one token per row) but not for a 1,024-token generation.
    args = QQ.engine_args({**cfg, "args": base}, None, st["seg_tokens"] + a.new, a.kv, a.short_read)
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    if cfg.get("lib_dirs"):
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(cfg["lib_dirs"] + ([env[var]] if env.get(var) else []))
    env.update(QQ.FIXED_ENV)
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v

    want = set(a.segments.split(",")) if a.segments else None
    segs = [s for s in st["segments"] if want is None or s["name"] in want]
    errf = open(str(out) + ".stderr.txt", "w")
    print("engine:", " ".join(args), flush=True)
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                         cwd=cfg.get("cwd"), env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY (exit {p.wait()}); see {out}.stderr.txt")

    rows = []
    suffix = tok.encode(CLOSE_THINKING, parse_special=True) if a.close_thinking else []
    for s in segs:
        prompt = s["ids"][:s.get("answer_at", len(s["ids"]))] + suffix
        t0 = time.time()
        p.stdin.write(f"GEN {a.new} ckpt=0 " + ",".join(map(str, prompt)) + "\n")
        p.stdin.flush()
        got, done = [], None
        for line in p.stdout:
            if line.startswith("T "):
                got.append(int(line.split()[1]))
            elif line.startswith(("DONE", "ERR")):
                done = line.strip()
                break
        if done is None or not done.startswith("DONE"):
            p.kill()
            raise SystemExit(f"{s['name']}: the engine answered {done!r}; see {out}.stderr.txt")
        text = tok.decode(got)
        rows.append({"name": s["name"], "answer_at": len(prompt), "prompt": len(prompt),
                     "generated": len(got), "ids": got, "text": text,
                     "seconds": round(time.time() - t0, 1),
                     "stopped": "eos" if len(got) < a.new else "budget"})
        print(f"{s['name']}: {len(got)} tokens in {rows[-1]['seconds']:.1f} s  {text[:110]!r}", flush=True)

    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    out.write_text(json.dumps({"kv": a.kv, "env": a.env or [], "new": a.new, "short_read": a.short_read,
                               "drop_spec": a.drop_spec, "spec": a.spec,
                               "close_thinking": a.close_thinking, "ids": str(ids_path), "config": a.config,
                               "args": args, "segments": rows}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
