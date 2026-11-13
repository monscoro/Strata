#!/usr/bin/env python
"""D2: das Auswahlverhalten des Indexers (registriert in bench/results/2026-11-11-ring-selection/README.md).

Ein einzelner 128K-Lauf (`int8`, keine Codec-Änderung) mit aktiviertem
QSA-Selektions-Dump (STRATA_QSA_SELECT_DUMP=1). Der Dump schreibt je Fenster und
QSA-Layer einen Datensatz `[int64 pos0, layer, T, cap]` + `T*cap` int32 (Format:
src/core/verify.cpp) nach sel_dump.bin. Bei 131.060 Token, 12 Layern und
cap = 2.051 Zellen sind das ~12,9 GB (die "13 MB" der Registrierung waren ein
Faktor-1000-Fehler; die Registrierung ist unten als Nachtrag korrigiert).

Die Auswertung macht bench/results/2026-11-11-ring-selection/analyze-d2.py
(Vereinigung, Churn, Profil, tail-Überdeckung gegen G9/G10).
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
DATA = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval")
OUT = DATA / "d2"
CFG = REPO / "bench" / "configs" / "cwe4k-swift-iq3_xxs-merged.json"
IDS = REPO / "data" / "quant_quality" / "cwe-128k" / "t1.ids.json"

BASE_ENV = [
    "STRATA_KV_WINDOW_S=256",
    "STRATA_KV_WINDOW_R=64",
    "STRATA_KV_WINDOW_BATCH=6144",
    "STRATA_VERIFY_ASYNC_LAUNCH=1",
    "STRATA_QSA_SELECT_DUMP=1",
    "STRATA_QSA_SELECT_DUMP_PATH=" + str(OUT / "sel_dump.bin"),
]

# text, tag, --kv, extra env
ARMS = [
    ("t1", "int8", "int8", {}),
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


def run_arm(text, tag, kv, extra=None):
    extra = extra or {}
    out = OUT / f"{text}-{tag}.tsv"
    env = list(BASE_ENV)
    for k, v in extra.items():
        env.append(f"{k}={v}")
    cmd = [PY, str(REPO / "tools" / "quant_quality.py"), "run",
           "--config", str(CFG), "--ids", str(IDS), "--out", str(out), "--kv", kv,
           "--prefill", "6144"]
    for e in env:
        cmd += ["--env", e]
    print(f"=== arm {text}/{tag} ===", flush=True)
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        print(f"arm {text}/{tag}: quant_quality.py exited with {rc}", flush=True)
        return rc
    lines = sum(1 for _ in open(out, encoding="utf-8", errors="replace"))
    seg = Path(str(out) + ".seg.json")
    segments = len(json.loads(seg.read_text(encoding="utf-8"))["segments"]) if seg.exists() else 0
    dump_path = OUT / "sel_dump.bin"
    dump_size = dump_path.stat().st_size if dump_path.exists() else 0
    print(f"arm {text}/{tag}: {lines} lines, {segments} segments, "
          f"sha256 {sha256(out)}, dump {dump_size} bytes", flush=True)
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if strata_running():
        print("a strata.exe is running; refusing to start (one GPU job at a time)", flush=True)
        return 1
    rc = 0
    for text, tag, kv, extra in ARMS:
        rc = run_arm(text, tag, kv, extra) or rc
        time.sleep(30)   # the campaign rule: 30 s between engine starts
    return rc


if __name__ == "__main__":
    sys.exit(main())
