#!/usr/bin/env python3
"""Run the 128K tail comparison fixed in README.md: q4u32w without and with mean calibration.

The recorded run used `--kv q4u32` plus `STRATA_KV_WINDOW=1`, because the fork build it ran on predates the
`q4u32w` name.  In this tree `--kv q4u32w` fills the same window env, so the two spell the same point; the
runner keeps the spelling that was actually run.

Usage:
  python docs/bench-results/2026-11-28-q4u32w-centering-tail/run-center.py
  CT_ONLY=q4u32w-c python .../run-center.py     # resume one arm
  CT_FORCE=1 python .../run-center.py           # overwrite existing TSVs
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QQ = ROOT / "terminal" / "tools" / "quant_quality.py"
CFG = HERE / "config.json"
IDS = ROOT / "terminal" / "data" / "quant_quality" / "cwe-128k" / "t1.ids.json"
DATA = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-28-q4u32w-centering-tail")
CENTER = HERE / "center-table.bin"

# Identical to the A/B campaign (2026-11-25): same env, same corpus, same geometry.
BASE_ENV = ["STRATA_KV_WINDOW=1", "STRATA_KV_WINDOW_S=256", "STRATA_KV_WINDOW_R=64",
            "STRATA_KV_WINDOW_BATCH=6144", "STRATA_VERIFY_ASYNC_LAUNCH=1"]

# tag, --kv, extra env, what it is
ARMS = [
    ("q4u32w-noc", "q4u32", [], "q4u32w without mean calibration (the pair's baseline)"),
    ("q4u32w-c", "q4u32", [f"STRATA_KV_CENTER={CENTER}"], "q4u32w with mean calibration (the arm)"),
    ("fp16", "fp16", [], "the reference (its reproduction is the gate for cross-checking the A/B TSV)"),
]
ONLY = [x for x in os.environ.get("CT_ONLY", "").split(",") if x]
FORCE = os.environ.get("CT_FORCE", "") == "1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strata_running() -> bool:
    raw = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True).stdout
    return any(line.lower().lstrip().startswith("strata.exe")
               for line in raw.decode("utf-8", errors="replace").splitlines())


def run_arm(tag: str, kv: str, extra: list[str], what: str) -> dict:
    out = DATA / f"t1-{tag}.tsv"
    if out.exists() and (DATA / f"t1-{tag}.tsv.seg.json").exists() and not FORCE:
        print(f"=== arm {tag}: exists, skipping ({sha(out)}) ===", flush=True)
        return {"tag": tag, "what": what, "skipped": True, "sha256": sha(out),
                "lines": sum(1 for _ in open(out, encoding="utf-8", errors="replace"))}
    cmd = [sys.executable, str(QQ), "run", "--config", str(CFG), "--ids", str(IDS),
           "--out", str(out), "--kv", kv, "--prefill", "6144", "--topk", "64"]
    for e in BASE_ENV + extra:
        cmd += ["--env", e]
    print(f"=== arm {tag}: --kv {kv} {' '.join(extra)} ===", flush=True)
    print("cmd:", " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc:
        raise SystemExit(f"arm {tag}: quant_quality.py exited with {rc}")
    lines = sum(1 for _ in open(out, encoding="utf-8", errors="replace"))
    print(f"arm {tag}: {lines} lines, {time.time() - t0:.1f} s, sha256 {sha(out)}", flush=True)
    return {"tag": tag, "kv": kv, "what": what, "skipped": False,
            "seconds": round(time.time() - t0, 1), "lines": lines, "sha256": sha(out)}


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    if not CENTER.exists():
        print(f"the center table is missing: {CENTER}\n"
              f"build it first (see README section 4): a cwe-4k run with STRATA_KV_Q4_DUMP, then "
              f"python tools\\kv_center_table.py --dump <dump> --out {CENTER}", flush=True)
        return 1
    if strata_running():
        print("a strata.exe is running; refusing to start (one GPU job at a time)", flush=True)
        return 1
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    manifest = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "engine": cfg["exe"], "engine_sha256": sha(Path(cfg["exe"])),
                "config": str(CFG), "config_sha256": sha(CFG),
                "ids": str(IDS), "ids_sha256": sha(IDS),
                "center_table_sha256": sha(CENTER),
                "runner": str(Path(__file__).resolve()), "runner_sha256": sha(Path(__file__).resolve()),
                "common_env": BASE_ENV, "arms": []}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    for i, (tag, kv, extra, what) in enumerate(ARMS):
        if ONLY and tag not in ONLY:
            continue
        manifest["arms"].append(run_arm(tag, kv, extra, what))
        (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        if i < len(ARMS) - 1 and not (ONLY and ONLY[-1] == tag):
            time.sleep(30)   # the campaign rule between engine starts

    runs = [str(DATA / f"t1-{t}.tsv") for t in ("q4u32w-noc", "q4u32w-c")]
    print("=== compare against the fresh fp16 reference ===", flush=True)
    cmd = [sys.executable, str(QQ), "compare", str(DATA / "t1-fp16.tsv")] + runs + \
          ["--paired", "--ids", str(IDS), "--json", str(DATA / "compare-fp16.json"),
           "--paired-json", str(DATA / "paired.json")]
    print("cmd:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc:
        raise SystemExit(f"compare exited with {rc}")
    print("done:", DATA, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
