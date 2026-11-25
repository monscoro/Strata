"""Run one q4u32w timing arm with a fixed token prompt.

Usage: perf_run.py <context> <full|stream> <prompt-file> <prefill-chunk> <max-new> <tag>
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CFG = OUT / "gate-config.json"


def expand(value):
    text = str(value).replace("{ROOT}", str(ROOT))
    return os.path.normpath(text) if "{ROOT}" in str(value) else text


ctx, arm, prompt, chunk, max_new, tag = sys.argv[1:]
cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
exe = expand(cfg["exe"])
cwd = expand(cfg["cwd"])
args = [expand(v) for v in cfg["args"] if v != "--vision"]
for flag in ("--mtp", "--mtp-q4", "--kv-resident"):
    if flag in args:
        i = args.index(flag)
        del args[i:i + 2]
args[args.index("--max-context") + 1] = ctx
args[args.index("--expert-cache") + 1] = "256"
args[args.index("--prefill") + 1] = chunk
args += ["--vram-reserve-mib", "700", "--mmap-experts", "--tokens-file", str(OUT / prompt),
         "--max-new", max_new]
if arm == "stream":
    args += ["--kv-resident", "32768"]
elif arm != "full":
    raise SystemExit("arm must be full or stream")

env = dict(os.environ)
env["PATH"] = os.pathsep.join([str(ROOT / "terminal" / "engine" / "rocm" / "bin"),
                               r"C:\Program Files\AMD\ROCm\7.2\bin", env["PATH"]])
env["STRATA_KV_WINDOW"] = "1"
env["STRATA_PREFILL_CPU_SHARE"] = "0"
env["STRATA_PREFILL_WARM"] = "0"
env["STRATA_WDDM_BUDGET"] = "0"

cmd = [exe, *args]
print("cmd:", " ".join(cmd), flush=True)
with (OUT / f"{tag}.log").open("w", encoding="utf-8", errors="replace") as log:
    r = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
print("exit", r.returncode, flush=True)
sys.exit(r.returncode)
