#!/usr/bin/env python3
"""Run the 128K q4u32 vs q4u32w A/B fixed in README.md.

Usage:
  python docs/bench-results/2026-11-25-q4u32-window-ab/run-ab.py
  AB_ONLY=q4u32w python .../run-ab.py              # resume one arm
  AB_FORCE=1 python .../run-ab.py                  # overwrite existing TSVs
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
DATA = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-25-q4u32-window-ab")

# Common to all arms. The window arm alone adds STRATA_KV_WINDOW=1.
BASE_ENV = ["STRATA_KV_WINDOW_S=256", "STRATA_KV_WINDOW_R=64", "STRATA_KV_WINDOW_BATCH=6144",
            "STRATA_VERIFY_ASYNC_LAUNCH=1"]
ARMS = [
    ("fp16", "fp16", False),
    ("q4u32", "q4u32", False),
    ("q4u32w", "q4u32", True),
]
ONLY = [x for x in os.environ.get("AB_ONLY", "").split(",") if x]
FORCE = os.environ.get("AB_FORCE", "") == "1"


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


def run_arm(tag: str, kv: str, window: bool) -> dict:
    out = DATA / f"t1-{tag}.tsv"
    if out.exists() and (DATA / f"t1-{tag}.tsv.seg.json").exists() and not FORCE:
        print(f"=== arm {tag}: exists, skipping ({sha(out)}) ===", flush=True)
        return {"tag": tag, "skipped": True, "sha256": sha(out),
                "lines": sum(1 for _ in open(out, encoding="utf-8", errors="replace"))}
    cmd = [sys.executable, str(QQ), "run", "--config", str(CFG), "--ids", str(IDS),
           "--out", str(out), "--kv", kv, "--prefill", "6144", "--topk", "64"]
    for e in BASE_ENV + (["STRATA_KV_WINDOW=1"] if window else []):
        cmd += ["--env", e]
    print(f"=== arm {tag}: --kv {kv}, window={'1' if window else '0'} ===", flush=True)
    print("cmd:", " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc:
        raise SystemExit(f"arm {tag}: quant_quality.py exited with {rc}")
    lines = sum(1 for _ in open(out, encoding="utf-8", errors="replace"))
    print(f"arm {tag}: {lines} lines, {time.time() - t0:.1f} s, sha256 {sha(out)}", flush=True)
    return {"tag": tag, "kv": kv, "window": window, "skipped": False,
            "seconds": round(time.time() - t0, 1), "lines": lines, "sha256": sha(out)}


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    if strata_running():
        print("a strata.exe is running; refusing to start (one GPU job at a time)", flush=True)
        return 1
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    manifest = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "engine": cfg["exe"], "engine_sha256": sha(Path(cfg["exe"])),
                "config": str(CFG), "config_sha256": sha(CFG),
                "ids": str(IDS), "ids_sha256": sha(IDS),
                "runner": str(Path(__file__).resolve()), "runner_sha256": sha(Path(__file__).resolve()),
                "common_env": BASE_ENV, "arms": []}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    for i, (tag, kv, window) in enumerate(ARMS):
        if ONLY and tag not in ONLY:
            continue
        manifest["arms"].append(run_arm(tag, kv, window))
        (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        if i < len(ARMS) - 1 and not (ONLY and ONLY[-1] == tag):
            time.sleep(30)   # the campaign rule between engine starts

    if set(ONLY) in (set(), {"fp16", "q4u32", "q4u32w"}):
        runs = [str(DATA / f"t1-{t}.tsv") for t in ("q4u32", "q4u32w")]
        print("=== compare against the fresh fp16 reference ===", flush=True)
        cmd = [sys.executable, str(QQ), "compare", str(DATA / "t1-fp16.tsv")] + runs + \
              ["--paired", "--ids", str(IDS), "--json", str(DATA / "compare-fp16.json"),
               "--paired-json", str(DATA / "paired-fp16.json")]
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc:
            print(f"compare exited with {rc}", flush=True)
            return rc
        cmd = [sys.executable, str(HERE / "eval-ab.py"), "--json", str(DATA / "eval.json")]
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc:
            print(f"eval exited with {rc}", flush=True)
            return rc
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
