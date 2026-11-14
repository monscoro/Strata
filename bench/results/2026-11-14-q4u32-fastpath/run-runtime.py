# Serie #1 des Branches feature_q4u32_as_an_int8_replacement_for_all: die Laufzeitmessung
# (Baseline A1).  Vorregistrierung: bench/results/2026-11-14-q4u32-fastpath/README.md.
#
# Was gemessen wird: die Prompt-Read-Rate des Engine-Logs fuer einen 32K-Nadel-Prompt
# (Tiefen 10/50/90 %), je Arm eine Engine, 30 s zwischen den Engine-Starts.  Der Harness
# ist der unveraenderte tools/needle_bench.py, der Server der unveraenderte serve/server.py.
#
# Die Gates, die dieser Treiber selbst entscheiden kann:
#   G-R5  der Fenster-Arm muss die "ring sized"-Zeile im Engine-Log zeigen, die
#         uebrigen nicht;
#   G-R6  jeder Arm muss "the prompt attention on matrix cores on" zeigen (der Schalter
#         ist gesetzt) - sonst gilt der Lauf als FP32-Lauf;
#   R Recall: der Harness muss alle drei Nadeln finden (er selbst prueft das und endet
#         mit Code 1, wenn nicht).
# G-R1 (der Fast-Path meldet sich), G-R2 (q4u32 nicht langsamer als int8), G-R3 (der
# Kontrollarm muss den Fallback zeigen) und G-R7 (zwei Zahlen je Arm) entscheidet der
# Auswerfer, nicht der Treiber.
#
# Rohdaten (Logs, needles-JSON, Server-Ausgabe) liegen ausserhalb des Repos unter
# C:\\Users\\krone\\git\\Strata-data\\bench\\2026-11-14-q4u32-fastpath\\.
import hashlib
import json
import pathlib
import re
import signal
import subprocess
import sys
import time
import urllib.request

REPO = pathlib.Path(r"C:\Users\krone\git\kv-quant-strata-0.2")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")   # das Repo-venv: serve braucht `regex`
DATA = pathlib.Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-14-q4u32-fastpath")
LOGS = DATA / "logs"
URL = "http://127.0.0.1:7603"
PORT = "7603"

# (Tag, Config, Fenster erwartet)
ARMS = [
    ("int8", "needle-fastpath-int8.json", False),
    ("q4_0", "needle-fastpath-q4_0.json", False),
    ("q4lm8v", "needle-fastpath-q4lm8v.json", False),
    ("q4u32", "needle-fastpath-q4u32.json", False),
    ("q4u32win", "needle-fastpath-q4u32win.json", True),
    ("q4lm32w", "needle-fastpath-q4lm32w.json", True),
]

RATE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\)")


def wait_up(seconds=900):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(3)
    return False


def stop(server):
    if server.poll() is None:
        server.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.kill()


def strata_running():
    # Das Ein-GPU-Job-Gate.  tasklists Ausgabe kann ein Zeichen ausserhalb der Codepage
    # halten (ein Prozessname), das cp1252 nicht decodieren kann - deshalb lenient.
    try:
        raw = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True).stdout
    except OSError as e:
        print(f"tasklist failed ({e}); the one-GPU-job gate cannot be checked", flush=True)
        return True
    out = raw.decode("utf-8", errors="replace")
    return any(line.lower().lstrip().startswith("strata.exe") for line in out.splitlines())


def wait_engine_gone(seconds=180):
    waited = 0
    while strata_running() and waited < seconds:
        time.sleep(5)
        waited += 5
    return not strata_running()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arm(tag, cfg_name, window):
    cfg = REPO / "bench" / "configs" / cfg_name
    print(f"=== arm {tag}: config {cfg_name} ===", flush=True)
    c = json.loads(cfg.read_text(encoding="utf-8"))
    exe = c["exe"]
    print(f"arm {tag}: engine {exe} sha256 {sha256(exe)}", flush=True)
    print(f"arm {tag}: env {c['env']}", flush=True)
    if strata_running():
        print(f"arm {tag}: a strata.exe is already running; refusing to start (one GPU job at a time)", flush=True)
        return 1
    print(f"arm {tag}: starting server", flush=True)
    server = subprocess.Popen([PY, str(REPO / "serve" / "server.py"), "--engine", "strata",
                               "--config", str(cfg), "--port", PORT, "--no-api-key"],
                              cwd=str(REPO), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    rc = 0
    if not wait_up():
        print(f"arm {tag}: server never came up", flush=True)
        stop(server)
        return 1
    print(f"arm {tag}: server up, running needle_bench (32k x depths 10/50/90)", flush=True)
    t0 = time.time()
    done = subprocess.run([PY, str(REPO / "tools" / "needle_bench.py"), "--url", URL,
                           "--lengths", "32k", "--depths", "10,50,90",
                           "--out", str(DATA / f"needles-{tag}.json")], cwd=str(REPO))
    print(f"arm {tag}: needle_bench exit {done.returncode}, wall {int(time.time() - t0)} s", flush=True)
    stop(server)
    if not wait_engine_gone():
        print(f"arm {tag}: the engine did not exit after the server stopped", flush=True)
        rc = rc or 1

    log = LOGS / f"needle-{tag}.engine.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    ring = "ring sized" in text
    wmma = "matrix cores on" in text
    rates = [(int(m.group(1)), float(m.group(4)) / 1000.0, float(m.group(5)))
             for m in RATE.finditer(text)]
    print(f"arm {tag}: window ring line: {'present' if ring else 'absent'} (expected "
          f"{'present' if window else 'absent'})", flush=True)
    print(f"arm {tag}: matrix-core prompt path: {'on' if wmma else 'OFF'} (expected on)", flush=True)
    print(f"arm {tag}: prompt read lines: {len(rates)}", flush=True)
    for n, secs, tps in rates:
        print(f"    prompt {n} tokens: {secs:.1f} s  ({tps:.1f} tok/s)", flush=True)
    if ring != window:
        rc = rc or 1
    if not wmma:
        rc = rc or 1
    if len(rates) != 3:
        rc = rc or 1
    if done.returncode != 0:
        rc = rc or 1
    if log.exists():
        print(f"arm {tag}: engine log sha256 {sha256(log)}", flush=True)
    time.sleep(30)   # die Kampagnenregel: 30 s zwischen Engine-Starts
    return rc


LOGS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
rc = 0
for tag, cfg_name, window in ARMS:
    rc = arm(tag, cfg_name, window) or rc

print("=== SUMMARY ===", flush=True)
print(f"{'arm':>10}  {'prompt tok/s (10/50/90 %)':>28}  {'Median':>8}  {'Fenster':>8}  {'WMMA':>5}", flush=True)
for tag, _cfg, window in ARMS:
    p = DATA / f"needles-{tag}.json"
    log = LOGS / f"needle-{tag}.engine.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    rates = [float(m.group(5)) for m in RATE.finditer(text)]
    med = sorted(rates)[len(rates) // 2] if rates else float("nan")
    cells = " ".join(f"{v:7.1f}" for v in rates) if rates else "-"
    found = "-"
    if p.exists():
        rows = json.loads(p.read_text(encoding="utf-8"))
        found = f"{sum(1 for r in rows if r.get('found'))}/{len(rows)}"
    print(f"{tag:>10}  {cells:>28}  {med:8.1f}  {'ja' if window else 'nein':>8}  "
          f"{'an' if 'matrix cores on' in text else 'aus':>5}  recall {found}", flush=True)
sys.exit(rc)
