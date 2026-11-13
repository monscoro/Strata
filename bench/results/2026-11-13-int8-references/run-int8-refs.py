#!/usr/bin/env python
"""The two fair INT8 references on cwe-4k (bench/results/2026-11-13-int8-references/).

Three arms, about 52 minutes.  The seven arms of 2026-11-11/12 are NOT re-run: they were measured on the
same build (0.1.59.1), the same text and the same `set.ids.json`, and the reproduction arm below proves
that the pairing is still valid.

  int8-rot           R2a  int8 + the Hadamard rotation (STRATA_KV_ROT=1).  The separator: the rotation is
                          itself a treatment, so without this arm the calibration would be credited with it.
  int8-rot-center    R2b  the same + the xx calibration (STRATA_KV_CENTER=center-table.bin).
  q4lm32w-repro           a byte reproduction of the product arm, NOT a result: its SHA-256 must equal the
                          campaign arm's (d6e6514...).  If it does not, the existing seven are not paired
                          comparable with the new ones and only the new arms are compared with each other.

The compare step runs in ONE pass over all ten arms with fp16 as the reference, so every pair in
README.md's table E1-E5 comes out of the same read.

Geometry: cwe-4k, the same config and ids as C3/C4 (the merged engine 0.1.59.1), the window in its
production geometry (S 256, R 64, chunk 512) for the q4 arms and the verify-window switch every campaign
arm needs (STRATA_VERIFY_ASYNC_LAUNCH=1; without it the verify path dies of #267).  The INT8 arms do not
carry the window - they are INT8, not a q4 mode, and that is the reference the question is asked against.
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
DATA = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-13-int8-references")
PREV = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-11-shear-codec\arms")
CENTER = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-centering\center-table.bin")
CFG = REPO / "bench" / "configs" / "cwe4k-swift-iq3_xxs-merged.json"
IDS = REPO / "data" / "quant_quality" / "cwe-4k" / "set.ids.json"
OUT = DATA / "arms"

# the campaign arm this run must reproduce, byte for byte (its own SHA-256, 2026-11-12)
REPRO_ARM = "q4lm32w"
REPRO_SHA = "d6e65140272b050b68f3d57da547b628c0320cdd7a8be5cfdb371cc9ec338f58"

BASE_ENV = ["STRATA_KV_WINDOW_S=256", "STRATA_KV_WINDOW_R=64", "STRATA_KV_WINDOW_BATCH=512",
            "STRATA_VERIFY_ASYNC_LAUNCH=1"]

WIN = {"STRATA_KV_WINDOW": "1"}
ROT = {"STRATA_KV_ROT": "1"}

# tag, --kv, calibrated, extra env
ARMS = [
    ("int8-rot", "int8", False, ROT),
    ("int8-rot-center", "int8", True, ROT),
    ("q4lm32w-repro", "q4lm32w", False, {}),
]

# the seven arms of 2026-11-11/12, same build, same text, same ids -- reused, not re-run
PREVIOUS = ["fp16", "int8", "q4lm32w", "q4lm32w-center", "q4u32", "q4u32-center", "q4u32z"]

ONLY = [a for a in os.environ.get("RUN_INT8REFS_ONLY", "").split(",") if a]


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
    print(f"=== arm {tag} ({kv}{' + centering' if centered else ''}"
          f"{' + rotation' if extra.get('STRATA_KV_ROT') else ''}) ===", flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        print(f"arm {tag}: quant_quality.py exited with {rc}", flush=True)
        return rc
    lines = sum(1 for _ in open(out, encoding="utf-8", errors="replace"))
    seg = Path(str(out) + ".seg.json")
    segments = len(json.loads(seg.read_text(encoding="utf-8"))["segments"]) if seg.exists() else 0
    print(f"arm {tag}: {lines} lines, {segments} segments, sha256 {sha256(out)}", flush=True)
    return 0


def check_repro() -> bool:
    """Is the new q4lm32w arm the same bytes as the campaign's?  This is the validity of the pairing."""
    new = OUT / f"cwe4k-{REPRO_ARM}-repro.tsv"
    old = PREV / f"cwe4k-{REPRO_ARM}.tsv"
    if not new.exists() or not old.exists():
        print(f"repro check: missing {new.name} or {old.name}", flush=True)
        return False
    got, want = sha256(new), sha256(old)
    same = got == want
    print(f"repro check: new {got}", flush=True)
    print(f"repro check: old {want}", flush=True)
    print(f"repro check: {'IDENTICAL' if same else 'DIFFERS'}"
          f"{'' if want == REPRO_SHA else ' (the recorded campaign SHA is ' + REPRO_SHA + '!)'}",
          flush=True)
    if not same:
        print("  -> the seven existing arms are NOT paired comparable with the new ones.  Report only the\n"
              "     new arms against each other; E2 (product vs fair int8) and E3 (fast path vs fair int8)\n"
              "     are NOT decidable in this run.", flush=True)
    (OUT / "repro.json").write_text(json.dumps(
        {"tag": REPRO_ARM, "new_sha256": got, "old_sha256": want,
         "recorded_sha256": REPRO_SHA, "identical": same}, indent=1), encoding="utf-8")
    return same


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rc = 0
    for tag, kv, centered, extra in ARMS:
        if ONLY and tag not in ONLY:
            continue
        rc = run_arm(tag, kv, centered, extra) or rc
        time.sleep(30)   # the campaign rule: 30 s between engine starts

    paired_ok = check_repro()

    fp16 = PREV / "cwe4k-fp16.tsv"
    new_runs = [OUT / f"cwe4k-{t}.tsv" for t, kv, c, e in ARMS]
    if not fp16.exists():
        print(f"compare: no fp16 reference at {fp16}", flush=True)
        return rc or 1

    print("=== compare (one pass, fp16 as the reference) ===", flush=True)
    if paired_ok:
        # the seven existing arms + the three new ones: every pair of README.md's table E1-E5
        runs = [PREV / f"cwe4k-{t}.tsv" for t in PREVIOUS] + new_runs
        runs = [p for p in runs if p.exists()]
        if len(runs) != len(PREVIOUS) + len(new_runs):
            print("  warning: a run is missing from the pass; the table is incomplete", flush=True)
    else:
        # only what the new run can answer: the rotation, the calibration on int8, the product's own KL
        runs = [p for p in new_runs if p.exists()] + [PREV / "cwe4k-int8.tsv"]
        runs = [p for p in runs if p.exists()]
    if len(runs) >= 2:
        cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "compare", str(fp16)] + \
              [str(p) for p in runs] + \
              ["--paired", "--ids", str(IDS), "--segments",
               "--json", str(OUT / "compare-int8refs.json"),
               "--paired-json", str(OUT / "paired-int8refs.json")]
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode or rc

    print("=== the tail and the top-1 (Nebenbefunde, paired) ===", flush=True)
    ref = str(fp16)
    int8_rot = OUT / "cwe4k-int8-rot.tsv"
    if int8_rot.exists():
        for label, a in (("q4u32", PREV / "cwe4k-q4u32.tsv"),
                         ("q4u32z", PREV / "cwe4k-q4u32z.tsv"),
                         ("q4lm32w", PREV / "cwe4k-q4lm32w.tsv")):
            if not a.exists():
                continue
            print(f"\n--- tail: A = {label}, B = int8-rot, int8 reference = int8-rot ---", flush=True)
            cmd = [PY, str(REPO / "tools" / "kv_tail_top1.py"), "--ref", ref,
                   "--int8", str(int8_rot), "--a", str(a), "--b", str(int8_rot),
                   "--ids", str(IDS)]
            rc = subprocess.run(cmd, cwd=str(REPO)).returncode or rc
    else:
        print("  skipped: no int8-rot arm", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
