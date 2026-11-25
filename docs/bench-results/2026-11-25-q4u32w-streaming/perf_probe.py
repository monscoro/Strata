"""Short memory/decode probe at a declared context: full q4u32w vs streamed q4u32w.

This does NOT prefill the declared context.  It starts the engine with that --max-context so the KV
reservation is real, runs the fixed 256-token prompt and prints the engine's own VRAM/RAM lines.  It is a
startup/memory gate, not a 128K/256K prompt-rate measurement.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CFG = OUT / "gate-config.json"
TOKENS = OUT / "prompt256.tokens"


def expand(value):
    text = str(value).replace("{ROOT}", str(ROOT))
    return os.path.normpath(text) if "{ROOT}" in str(value) else text


ctx, arm = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else f"probe-{ctx}-{arm}"
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
args[args.index("--prefill") + 1] = "256"
args += ["--vram-reserve-mib", "700", "--mmap-experts", "--tokens-file", str(TOKENS), "--max-new", "4"]
if arm == "stream":
    args += ["--kv-resident", "32768"]
elif arm == "full":
    if "--kv-resident" in args:
        i = args.index("--kv-resident")
        del args[i:i + 2]
else:
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
