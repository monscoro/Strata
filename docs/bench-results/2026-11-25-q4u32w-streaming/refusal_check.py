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


cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
exe = expand(cfg["exe"])
cwd = expand(cfg["cwd"])
args = [expand(v) for v in cfg["args"]]
# Same fixed, teacher-forced shape as run_gate.py; only the KV arm and the window reader change.
args = [a for a in args if a != "--vision"]
for flag in ("--mtp", "--mtp-q4"):
    if flag in args:
        i = args.index(flag)
        del args[i:i + 2]
args[args.index("--max-context") + 1] = "32768"
args[args.index("--expert-cache") + 1] = "256"
args[args.index("--kv") + 1] = "q4lm8v"
args[args.index("--kv-resident") + 1] = "20480"
args += ["--tokens-file", str(OUT / "prompt256.tokens"), "--max-new", "1"]

env = dict(os.environ)
env["PATH"] = os.pathsep.join([str(ROOT / "terminal" / "engine" / "rocm" / "bin"),
                               r"C:\Program Files\AMD\ROCm\7.2\bin", env["PATH"]])
env["STRATA_KV_WINDOW"] = "1"

cmd = [exe, *args]
with (OUT / "refusal.log").open("w", encoding="utf-8", errors="replace") as log:
    log.write("cmd: " + " ".join(cmd) + "\n")
    r = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"exit {r.returncode}\n")
print("exit", r.returncode)
sys.exit(0 if r.returncode == 2 else 1)
