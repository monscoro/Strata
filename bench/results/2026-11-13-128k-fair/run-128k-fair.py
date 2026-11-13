#!/usr/bin/env python
"""The 128K question against the fair reference (bench/results/2026-11-13-128k-fair/).

Three new arms, about 4.4 hours.  The six arms of 2026-11-10/11 are NOT re-run: they were measured on
the same build (0.1.59.1), the same text and the same `t1.ids.json`, and the pairing is already
proven -- `d2/t1-int8.tsv` (2026-11-13 12:03) is byte-identical to `d1/t1-int8.tsv` (2026-11-11
10:43), SHA-256 f18551dfd9ef... on both.  So this run needs no reproduction arm.

  int8-rot           R2a  int8 + the Hadamard rotation (STRATA_KV_ROT=1).
  int8-rot-center    R2b  the same + the calibration (STRATA_KV_CENTER=center-table.bin).
  q4u32                   the fast-path candidate: the uniform grid with a fitted step, plus the
                          window in its 128K geometry (S 256, R 64, chunk 6144).

Geometry as in run-d1.py: cwe-128k/t1, 131,063 tokens, --prefill 6144, the verify-window launch
switch every campaign arm needs.  The INT8 arms do not carry the window -- they are INT8, not a q4
mode, and they are the reference the question is asked against.

Usage:
  python bench/results/2026-11-13-128k-fair/run-128k-fair.py            # all three
  RUN_128KFAIR_ONLY=q4u32 python .../run-128k-fair.py                   # a subset
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable
DATA = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-13-128k-fair")
PREV = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\d1")
OUT = DATA / "arms"
CENTER = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-centering\center-table.bin")
CFG = REPO / "bench" / "configs" / "cwe4k-swift-iq3_xxs-merged.json"
IDS = REPO / "data" / "quant_quality" / "cwe-128k"

# 128K geometry: the production window with the paper's chunk (6144), not the 4K chunk (512)
BASE_ENV = ["STRATA_KV_WINDOW_S=256", "STRATA_KV_WINDOW_R=64", "STRATA_KV_WINDOW_BATCH=6144",
            "STRATA_VERIFY_ASYNC_LAUNCH=1"]

WIN = {"STRATA_KV_WINDOW": "1"}
ROT = {"STRATA_KV_ROT": "1"}

# tag, --kv, calibrated, extra env
ARMS = [
    ("int8-rot", "int8", False, ROT),
    ("int8-rot-center", "int8", True, ROT),
    ("q4u32", "q4u32", False, WIN),
]

# the six arms of 2026-11-10/11, same build, same text, same ids -- reused, not re-run
PREVIOUS = ["fp16", "int8", "q4lm32w", "q4lm32w-center", "q4lm8v-win", "q4lm8v-win-center"]

ONLY = [a for a in os.environ.get("RUN_128KFAIR_ONLY", "").split(",") if a]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strata_running():
    raw = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True).stdout
    return any(line.lower().lstrip().startswith("strata.exe")
               for line in raw.decode("utf-8", errors="replace").splitlines())


def run_arm(tag, kv, centered, extra=None):
    extra = extra or {}
    out = OUT / f"t1-{tag}.tsv"
    env = list(BASE_ENV)
    if centered:
        env.append(f"STRATA_KV_CENTER={CENTER}")
    for k, v in extra.items():
        env.append(f"{k}={v}")
    cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "run",
           "--config", str(CFG), "--ids", str(IDS / "t1.ids.json"), "--out", str(out),
           "--kv", kv, "--prefill", "6144"]
    for e in env:
        cmd += ["--env", e]
    print(f"=== arm t1/{tag} ({kv}{' + centering' if centered else ''}"
          f"{' + rotation' if extra.get('STRATA_KV_ROT') else ''}"
          f"{' + window' if extra.get('STRATA_KV_WINDOW') else ''}) ===", flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        print(f"arm t1/{tag}: quant_quality.py exited with {rc}", flush=True)
        return rc
    lines = sum(1 for _ in open(out, encoding="utf-8", errors="replace"))
    seg = Path(str(out) + ".seg.json")
    segments = len(json.loads(seg.read_text(encoding="utf-8"))["segments"]) if seg.exists() else 0
    print(f"arm t1/{tag}: {lines} lines, {segments} segments, sha256 {sha256(out)}", flush=True)
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if strata_running():
        print("a strata.exe is running; refusing to start (one GPU job at a time)", flush=True)
        return 1
    rc = 0
    for tag, kv, centered, extra in ARMS:
        if ONLY and tag not in ONLY:
            continue
        rc = run_arm(tag, kv, centered, extra) or rc
        time.sleep(30)   # the campaign rule: 30 s between engine starts

    # ONE compare over all nine arms with the fp16 reference (d1's, never the arm under test), so
    # every pair in the README's G1-G3 comes out of the same read.
    ref = str(PREV / "t1-fp16.tsv")
    runs = [str(PREV / f"t1-{t}.tsv") for t in PREVIOUS if t != "fp16"]
    runs += [str(OUT / f"t1-{tag}.tsv") for tag, _kv, _c, _e in ARMS
             if (not ONLY or tag in ONLY) and (OUT / f"t1-{tag}.tsv").exists()]
    runs = [r for r in runs if Path(r).exists()]
    print("=== compare against the fp16 reference (absolute numbers AND paired differences) ===",
          flush=True)
    cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "compare", ref] + runs + \
          ["--paired", "--ids", str(IDS / "t1.ids.json"), "--segments",
           "--json", str(DATA / "compare-fp16.json"),
           "--paired-json", str(DATA / "paired-fp16.json")]
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode or rc

    print("=== the gates, as fixed in README.md before the run ===", flush=True)
    # eval-128k-fair.py reads the arms itself (it needs the per-position series for the block
    # bootstrap and the tail); compare-fp16.json is only the absolute-number record.
    cmd = [PY, str(Path(__file__).resolve().parent / "eval-128k-fair.py"),
           "--json", str(DATA / "gates.json")]
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
