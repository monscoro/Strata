#!/usr/bin/env python
"""The shear arm on cwe-4k (bench/results/2026-11-11-shear-codec/).

The seven arms in ONE run, so the paired differences are comparable: the anchor, the product arm, and
the 2x2 matrix {with, without} the per-block offset x {with, without} the per-channel calibration.

  int8               the K1 anchor
  q4lm32w            the product arm (Lloyd 16/16 + the exact-cell window)
  q4lm32w+C          the product arm + the calibration
  q4u32              the uniform grid with the step fitted per block (the affine sibling)
  q4u32+C            the same + the calibration
  q4u32z             the shear: the uniform grid + the per-block offset
  q4u32z+C           the shear + the calibration

Geometry: cwe-4k, the same ids and the same build as C3/C4 (the merged engine 0.1.59.1), the
window in its production geometry (S 256, R 64, chunk 512) and the verify-window switch every
campaign arm needs (STRATA_VERIFY_ASYNC_LAUNCH=1, without it the verify path dies of #267).
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
DATA = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-11-shear-codec")
CENTER = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-centering\center-table.bin")
CFG = REPO / "bench" / "configs" / "cwe4k-swift-iq3_xxs-merged.json"
IDS = REPO / "data" / "quant_quality" / "cwe-4k" / "set.ids.json"
OUT = DATA / "arms"

BASE_ENV = ["STRATA_KV_WINDOW_S=256", "STRATA_KV_WINDOW_R=64", "STRATA_KV_WINDOW_BATCH=512",
            "STRATA_VERIFY_ASYNC_LAUNCH=1"]

WIN = {"STRATA_KV_WINDOW": "1"}

# tag, --kv, calibrated, extra env
ARMS = [
    ("fp16", "fp16", False, {}),          # the reference the absolute KL is measured against
    ("int8", "int8", False, {}),
    ("q4lm32w", "q4lm32w", False, {}),          # the mode carries the window
    ("q4lm32w-center", "q4lm32w", True, {}),
    ("q4u32", "q4u32", False, WIN),
    ("q4u32-center", "q4u32", True, WIN),
    ("q4u32z", "q4u32z", False, WIN),
    ("q4u32z-center", "q4u32z", True, WIN),
]

ONLY = [a for a in os.environ.get("RUN_SHEAR_ONLY", "").split(",") if a]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_arm(tag, kv, centered, extra):
    extra = extra or {}
    out = OUT / f"cwe4k-{tag}.tsv"
    env = list(BASE_ENV)
    if centered:
        env.append(f"STRATA_KV_CENTER={CENTER}")
    for k, v in extra.items():
        env.append(f"{k}={v}")
    cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "run",
           "--config", str(CFG), "--ids", str(IDS), "--out", str(out), "--kv", kv,
           "--prefill", "512"]
    for e in env:
        cmd += ["--env", e]
    print(f"=== arm {tag} ({kv}{' + centering' if centered else ''}) ===", flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        print(f"arm {tag}: quant_quality.py exited with {rc}", flush=True)
        return rc
    lines = sum(1 for _ in open(out, encoding="utf-8", errors="replace"))
    seg = Path(str(out) + ".seg.json")
    segments = len(json.loads(seg.read_text(encoding="utf-8"))["segments"]) if seg.exists() else 0
    print(f"arm {tag}: {lines} lines, {segments} segments, sha256 {sha256(out)}", flush=True)
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rc = 0
    for tag, kv, centered, extra in ARMS:
        if ONLY and tag not in ONLY:
            continue
        rc = run_arm(tag, kv, centered, extra) or rc
        time.sleep(30)   # the campaign rule: 30 s between engine starts
    print("=== compare against the fp16 reference (all arms in one pass) ===", flush=True)
    runs = [str(OUT / f"cwe4k-{t}.tsv") for t, kv, c, e in ARMS
            if (not ONLY or t in ONLY) and (OUT / f"cwe4k-{t}.tsv").exists()]
    if len(runs) >= 2:
        # the reference is the fp16 arm (the first one): the absolute KL and the paired differences
    # are measured against it
        cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "compare"] + runs + \
              ["--paired", "--ids", str(IDS), "--segments",
               "--json", str(OUT / "compare-all.json"), "--paired-json", str(OUT / "paired-all.json")]
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode or rc
    print("=== the tail and the top-1, paired ===", flush=True)
    a = OUT / "cwe4k-q4lm32w.tsv"
    b = OUT / "cwe4k-q4u32z.tsv"
    if a.exists() and b.exists():
        cmd = [PY, str(REPO / "tools" / "kv_tail_top1.py"), "--ref", str(OUT / "cwe4k-fp16.tsv"),
               "--int8", str(OUT / "cwe4k-int8.tsv"), "--a", str(a), "--b", str(b),
               "--ids", str(IDS)]
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
