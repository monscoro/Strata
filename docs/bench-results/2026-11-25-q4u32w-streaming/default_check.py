"""Start the engine once with NO --kv (the new auto default) and once with --kv q4u32 (the plain grid),
same everything else, and show the KV lines + free VRAM from each log.  Not a benchmark: startup only."""
import json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(r"C:\Users\krone\git\q4u32-stabilize")
OUT = ROOT / "docs" / "bench-results" / "2026-11-25-q4u32w-streaming"
cfg = json.loads((OUT / "gate-config.json").read_text(encoding="utf-8-sig"))
expand = lambda v: os.path.normpath(str(v).replace("{ROOT}", str(ROOT)))
exe, cwd = expand(cfg["exe"]), expand(cfg["cwd"])
args = [expand(v) for v in cfg["args"]]
args = [a for a in args if a != "--vision"]
for flag in ("--mtp", "--mtp-q4"):
    if flag in args:
        i = args.index(flag); del args[i:i + 2]
args[args.index("--max-context") + 1] = "32768"
args[args.index("--prefill") + 1] = "256"
args[args.index("--expert-cache") + 1] = "256"
if "--kv-resident" in args:
    i = args.index("--kv-resident"); del args[i:i + 2]
if "--kv" in args:                      # the gate config names the format; the auto arm must leave it out
    i = args.index("--kv"); del args[i:i + 2]
args += ["--vram-reserve-mib", "700", "--mmap-experts",
         "--tokens-file", str(OUT / "prompt256.tokens"), "--max-new", "1"]

env = dict(os.environ)
env["PATH"] = os.pathsep.join([str(ROOT / "terminal" / "engine" / "rocm" / "bin"),
                               r"C:\Program Files\AMD\ROCm\7.2\bin", env["PATH"]])
env["STRATA_PREFILL_CPU_SHARE"] = "0"
env["STRATA_PREFILL_WARM"] = "0"
env["STRATA_WDDM_BUDGET"] = "0"

for tag, extra in (("auto-default", []), ("plain-grid", ["--kv", "q4u32"])):
    e = dict(env)
    if tag == "plain-grid":
        pass
    cmd = [exe, *args, *extra]
    log = OUT / f"default-check-{tag}.log"
    with log.open("w", encoding="utf-8", errors="replace") as f:
        r = subprocess.run(cmd, cwd=cwd, env=e, stdout=f, stderr=subprocess.STDOUT)
    text = log.read_text(encoding="utf-8", errors="replace")
    print(f"=== {tag}: exit {r.returncode}")
    for line in text.splitlines():
        if re.search(r"KV storage|KV window|window|pinned|VRAM free|decode", line) and "verify: captur" not in line:
            print("   ", line[:200])
