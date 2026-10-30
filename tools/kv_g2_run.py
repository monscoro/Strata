#!/usr/bin/env python3
"""G2 runner: one engine start per segment, because the id set is fixed at startup.

`STRATA_LOGPOS_IDS` is read when the engine starts, so a run that wants a different token set per segment needs one
engine start per segment.  Eight starts cost about two minutes of loading on top of the scoring, and they keep
`quant_quality.py` unchanged: each start gets a one-segment ids JSON and its own `--out`, and the parts are merged
into one TSV and one `.seg.json` afterwards.

    python tools/kv_g2_run.py --config bench/configs/cwe4k-swift-iq3_xxs.json \\
        --support C:\\...\\2026-10-28-g2\\support.json --kv int8 --out C:\\...\\int8.tsv

    python tools/kv_g2_run.py ... --ids-template '{seg}-full.txt' --allow-few-positions --out ...\\ref-exact.tsv

`--ids-template` names the per-segment file inside `support.json`'s `ids/` directory: `{seg}.txt` is the arm side
(the segment's own top-4,096, the whole vocabulary at the exact rows), `{seg}-full.txt` is the reference side of the
exact rows (those positions only, so the run scores a handful of rows per segment and `--allow-few-positions` is
needed to let it pass the "fewer than half the positions" check).

The fall-through top-k defaults to 0: with an id file the top-k only fills the positions the file does not name,
and at 4,096 that is 4,096 pairs on each of the other ~3,000 rows per segment - measured 2.3 GB per arm instead of
0.77 GB, for pairs nothing reads.

The merged `.seg.json` keeps the parts' own args, so a run can be reproduced segment by segment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def positions(path: pathlib.Path) -> list[int]:
    return [int(line.split(" ", 1)[0]) for line in open(path, encoding="utf-8") if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", default=str(HERE.parent / "data" / "quant_quality" / "cwe-4k" / "set.ids.json"))
    ap.add_argument("--support", required=True, help="support.json written by kv_g2_support.py")
    ap.add_argument("--kv", default="fp16")
    ap.add_argument("--topk", type=int, default=0,
                    help="the fall-through top-k for positions the id file does not name; 0 by default, because "
                         "the id file is the support and a top-k there would only add 4,096 pairs to every other "
                         "position (measured: 2.3 GB per arm instead of 0.77 GB)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-template", default="{seg}.txt")
    ap.add_argument("--segments", help="comma-separated subset (default: every segment in --ids)")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--min-free-gb", type=float, default=16)
    ap.add_argument("--allow-few-positions", action="store_true",
                    help="pass --allow-no-positions: the exact-row reference run scores only a few rows")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    support = json.load(open(a.support, encoding="utf-8"))
    ids_dir = pathlib.Path(support["ids"][next(iter(support["ids"]))]["path"]).parent
    st = json.load(open(a.ids, encoding="utf-8-sig"))
    want = set(a.segments.split(",")) if a.segments else None
    segs = [s for s in st["segments"] if want is None or s["name"] in want]
    if not segs:
        raise SystemExit("no segment selected")

    out = pathlib.Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = out.with_name(out.name + ".parts")
    parts_dir.mkdir(exist_ok=True)
    for f in (out, out.with_name(out.name + ".seg.json")):
        if f.exists():
            f.unlink()

    print(f"{a.kv} top{a.topk}: {len(segs)} segments, one engine start each, parts in {parts_dir}", flush=True)
    merged, parts = [], []
    first = 0
    with open(out, "w", encoding="utf-8", newline="\n") as dst:
        for s in segs:
            name = s["name"]
            ids_file = ids_dir / a.ids_template.format(seg=name)
            if not ids_file.exists():
                raise SystemExit(f"missing {ids_file}")
            pos = positions(ids_file)
            one = parts_dir / f"set.{name}.ids.json"
            one.write_text(json.dumps({"seg_tokens": st["seg_tokens"],
                                       "segments": [{"name": name, "ids": s["ids"]}]}, indent=1),
                           encoding="utf-8")
            part = parts_dir / f"{name}.tsv"
            cmd = [sys.executable, str(HERE / "quant_quality.py"), "run", "--config", a.config,
                   "--ids", str(one), "--out", str(part), "--kv", a.kv, "--topk", str(a.topk),
                   "--env", f"STRATA_LOGPOS_IDS={ids_file}",
                   "--min-free-gb", str(a.min_free_gb)]
            if a.exe:
                cmd += ["--exe", a.exe]
            if a.allow_few_positions:
                cmd += ["--allow-no-positions"]
            print(f"\n[{name}] {len(pos)} positions from {ids_file.name}", flush=True)
            if a.dry_run:
                print(" ".join(cmd), flush=True)
                continue
            r = subprocess.run(cmd)
            if r.returncode != 0:
                raise SystemExit(f"{name}: quant_quality.py exited {r.returncode}")

            meta = json.load(open(str(part) + ".seg.json", encoding="utf-8"))
            pm = meta["segments"][0]
            if pm["end_line"] - pm["first_line"] != len(pos):
                raise SystemExit(f"{name}: {pm['end_line'] - pm['first_line']} rows dumped, {len(pos)} positions "
                                 f"asked for")
            with open(part, encoding="utf-8") as src:
                n = 0
                for line in src:
                    dst.write(line)
                    n += 1
            if n != len(pos):
                raise SystemExit(f"{name}: part holds {n} lines, expected {len(pos)}")
            merged.append({"name": name, "first_line": first, "end_line": first + n,
                           "ids_file": str(ids_file), "ids_sha256": sha256(ids_file), "positions": len(pos)})
            parts.append({"segment": name, "part": str(part), "part_sha256": sha256(part),
                          "args": meta["args"], "env": meta["env"], "expert_cache_lines":
                          meta.get("expert_cache_lines", [])})
            first += n
            print(f"[{name}] merged, {n} rows, {first} so far", flush=True)

    if a.dry_run:
        return 0

    (out.with_name(out.name + ".seg.json")).write_text(json.dumps(
        {"segments": merged, "config": a.config, "kv": a.kv, "topk": a.topk,
         "ids_template": a.ids_template, "support": {"path": str(pathlib.Path(a.support).resolve()),
                                                     "sha256": sha256(pathlib.Path(a.support))},
         "parts": parts}, indent=1), encoding="utf-8")
    print(f"\nwrote {out} ({first} rows) and {out.name}.seg.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
