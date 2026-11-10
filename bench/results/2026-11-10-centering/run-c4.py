#!/usr/bin/env python
"""C4: the product codec (16 levels on both sides) with and without the centering.

Registered in bench/results/2026-11-10-centering/README.md (stage C4) before the run.

Why this arm: the q4lm8v arm carries 8 levels on the value side -- a 3-bit codebook inside 4-bit
bytes.  It saves no byte and costs 1.10 dB of block error (the parity harness, with the
centering: lloyd16K+lloyd8V+LS 6.527e-03 against lloyd16+LS 5.072e-03).  The product codec is
`--kv q4lm32w` (16/16, with the exact-cell window).  Two arms on cwe-4k, same ids, same build:

  a) q4lm32w            the 16/16 window arm, no centering
  b) q4lm32w + center   the same, with the calibrated per-channel offset

b against a isolates the centering; a against the q4lm8v window arm (b1/q4lm8v-win.tsv) isolates
the level split.  Both against int8 (b1/int8.tsv) is the K1 gate of the ladder (section 7).

The effectiveness gate from the campaign: the two arms must NOT be byte-identical (an arm that
runs and does nothing is only caught by a number comparison -- that is how the first centering
run failed, see the protocol).
"""
import hashlib
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable
DATA = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval")
B1 = DATA / "b1"
OUT = DATA / "b4"
CENTER = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-centering\center-table.bin")
CFG = REPO / "bench" / "configs" / "cwe4k-swift-iq3_xxs-merged.json"
IDS = REPO / "data" / "quant_quality" / "cwe-4k" / "set.ids.json"

# the window geometry of the production arm (b1/q4lm8v-win.tsv): S=256, R=64, chunk 512
WINDOW = ["STRATA_KV_WINDOW_S=256", "STRATA_KV_WINDOW_R=64", "STRATA_KV_WINDOW_BATCH=512"]

ARMS = [
    ("q4lm32w", ["--kv", "q4lm32w"] + [], False),
    ("q4lm32w-center", ["--kv", "q4lm32w"], True),
]


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


def run_arm(tag, kv, centered):
    out = OUT / f"cwe4k-{tag}.tsv"
    env = list(WINDOW)
    if centered:
        env.append(f"STRATA_KV_CENTER={CENTER}")
    cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "run",
           "--config", str(CFG), "--ids", str(IDS), "--out", str(out), "--kv", kv]
    for e in env:
        cmd += ["--env", e]
    print(f"=== arm {tag}: {' '.join(cmd[3:])} ===", flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        print(f"arm {tag}: quant_quality.py exited with {rc}", flush=True)
        return rc
    lines = sum(1 for _ in open(out, encoding="utf-8", errors="replace"))
    print(f"arm {tag}: {lines} lines, sha256 {sha256(out)}", flush=True)
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if strata_running():
        print("a strata.exe is running; refusing to start (one GPU job at a time)", flush=True)
        return 1
    rc = 0
    for tag, kv, centered in ARMS:
        rc = run_arm(tag, kv, centered) or rc
        time.sleep(30)   # the campaign rule: 30 s between engine starts
    a, b = OUT / "cwe4k-q4lm32w.tsv", OUT / "cwe4k-q4lm32w-center.tsv"
    if a.exists() and b.exists():
        sa, sb = sha256(a), sha256(b)
        print(f"effectiveness gate: the two arms must differ: {sa} / {sb} -> "
              f"{'OK' if sa != sb else 'FAILED (identical: the centering did nothing)'}", flush=True)
        if sa == sb:
            rc = 1
    print("=== compare against int8 (the K1 gate of the ladder) ===", flush=True)
    cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "compare",
           str(B1 / "int8.tsv"), str(B1 / "q4lm8v-win.tsv"), str(OUT / "cwe4k-q4lm32w.tsv"),
           str(OUT / "cwe4k-q4lm32w-center.tsv"),
           "--paired", "--ids", str(IDS), "--segments",
           "--json", str(OUT / "compare.json"), "--paired-json", str(OUT / "paired.json")]
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
