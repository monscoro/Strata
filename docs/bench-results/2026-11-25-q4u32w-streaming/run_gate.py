import hashlib
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


arm = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else arm
cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
exe = expand(cfg["exe"])
cwd = expand(cfg["cwd"])
args = [expand(v) for v in cfg["args"]]

# Teacher-forced parity: no image path, no server, no MTP ring (the MTP never carries a window).
args = [a for a in args if a != "--vision"]
for flag in ("--mtp", "--mtp-q4"):
    if flag in args:
        i = args.index(flag)
        del args[i:i + 2]
# Fix everything that otherwise varies with free RAM/VRAM between two launches.
args[args.index("--max-context") + 1] = "32768"
args[args.index("--prefill") + 1] = "256"
args[args.index("--expert-cache") + 1] = "256"
args += ["--vram-reserve-mib", "700", "--mmap-experts"]
# Native IQ packs require --spec T and --prefill CHUNK; both stay identical in the two arms.
args += ["--tokens-file", str(TOKENS), "--max-new", "1",
         "--dump-logits", str(OUT / f"{tag}-logits.bin")]
if arm == "resident":
    i = args.index("--kv-resident")
    del args[i:i + 2]
else:
    # Below the context length: the stream map really evicts and restores blocks from the host copy.
    args[args.index("--kv-resident") + 1] = "20480"

env = dict(os.environ)
env["PATH"] = os.pathsep.join([str(ROOT / "terminal" / "engine" / "rocm" / "bin"),
                               r"C:\Program Files\AMD\ROCm\7.2\bin", env["PATH"]])
env["STRATA_KV_WINDOW"] = "1"
env["STRATA_PREFILL_CPU_SHARE"] = "0"
env["STRATA_PREFILL_WARM"] = "0"
# WDDM gave this process a ~2.4 GiB budget, so the fixed expert cache closed intermittently.  Count real VRAM.
env["STRATA_WDDM_BUDGET"] = "0"
env["STRATA_DUMP_FIRST_LOGITS"] = str(OUT / f"{tag}-first.bin")

cmd = [exe, *args]
print("cmd:", " ".join(cmd), flush=True)
with (OUT / f"{tag}.log").open("w", encoding="utf-8", errors="replace") as log:
    r = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
print("exit", r.returncode, flush=True)
for name in (f"{tag}-logits.bin", f"{tag}-first.bin"):
    p = OUT / name
    if p.exists():
        print(name, p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest(), flush=True)
    else:
        print(name, "MISSING", flush=True)
sys.exit(r.returncode)
