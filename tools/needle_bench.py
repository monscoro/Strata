"""tools/needle_bench.py - needle-in-a-haystack recall through a running Strata server.

A long text is built from files in this repository (its docs and source code, and the llama.cpp docs setup downloads
into third_party/), code words are hidden in it at chosen depths, and the model is asked for them through the normal
API (thinking off, greedy). One line per graded needle: length, depth, found or not, prompt tokens, time.

    python tools/needle_bench.py                                  # 32K at depths 10/50/90, http://127.0.0.1:8080
    python tools/needle_bench.py --lengths 32k,128k,262k --depths 10,50,90 --url http://127.0.0.1:8081
    python tools/needle_bench.py --out needles.json --api-key KEY

Hardening (the validity gate for KV formats, workpackage 8.8 - all opt-in, defaults reproduce the old behavior):

    python tools/needle_bench.py --repeats 3                     # fresh word per repeat, same depths
    python tools/needle_bench.py --needle-types code,uuid,number # UUID-ish codes are the hardest (RULER 5)
    python tools/needle_bench.py --distractors 3                 # hard distractor needles, asked by label
    python tools/needle_bench.py --queries 3                     # several needles, graded each (MV/MQ-analogue)

The lengths are targets, 2% under so that "128k" fits a 128K context: the text is cut by characters (about 3.2 per
token), and the line reports the prompt's real token count from the server. A length the server's context cannot hold is skipped.
Exit code is a strict gate: 0 only if every graded needle is found and no request failed; --out JSON keeps the rows.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [("docs", "*.md"), ("src", "*.cpp"), ("src", "*.cu"), ("include", "*.hpp"), ("serve", "*.py"), ("tools", "*.py"),
           ("third_party/llama.cpp/docs", "*.md")]
WORDS = ["amber", "falcon", "quartz", "willow", "copper", "harbor", "saffron", "glacier", "orchid", "lantern",
         "meadow", "cobalt", "juniper", "tundra", "velvet", "ember"]
NEEDLE_TYPES = ("code", "uuid", "number")
CHARS_PER_TOKEN = 3.2


def haystack(n_chars: int) -> str:
    """Deterministic filler: the repository's text files in a fixed order, repeated if needed."""
    files = []
    for d, pat in SOURCES:
        files += sorted((ROOT / d).rglob(pat))
    parts, total = [], 0
    while total < n_chars:
        for f in files:
            try:
                t = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append(f"\n\n=== {f.relative_to(ROOT).as_posix()} ===\n{t}")
            total += len(parts[-1])
            if total >= n_chars:
                break
        if not files:
            raise SystemExit("no text files found to build the haystack from")
    return "".join(parts)[:n_chars]


def needle_value(rnd: random.Random, kind: str) -> str:
    """A fresh needle value. `code` is the historical word-word-NNN; `uuid` is RULER's hardest type;
    `number` is the 7-digit passkey style."""
    if kind == "uuid":
        h = "".join(rnd.choice("0123456789abcdef") for _ in range(32))
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
    if kind == "number":
        return str(rnd.randint(1000000, 9999999))
    if kind == "code":
        return f"{rnd.choice(WORDS)}-{rnd.choice(WORDS)}-{rnd.randint(100, 999)}"
    raise ValueError(f"unknown needle type {kind!r} (want one of {NEEDLE_TYPES})")


def insert_at(text: str, frac_percent: float, sentence: str) -> tuple[str, int]:
    """Insert `sentence` at a line start near `frac_percent` % of `text`. Pure; returns (new text, char pos)."""
    cut = int(len(text) * frac_percent / 100)
    cut = text.rfind("\n", 0, cut) + 1 or cut
    return text[:cut] + sentence + text[cut:], cut


def place_needles(text: str, items: list[tuple[float, str]]) -> tuple[str, list[int]]:
    """Insert several needle sentences into `text`. `items` is (depth %, sentence); insertion runs deepest
    first so earlier cuts stay valid. Returns (full text, char positions in input order). Pure."""
    order = sorted(range(len(items)), key=lambda i: items[i][0], reverse=True)
    pos: list[int] = [0] * len(items)
    for i in order:
        text, pos[i] = insert_at(text, items[i][0], items[i][1])
    return text, pos


def grade(answer: str, values: list[str]) -> dict[str, bool]:
    """Exact-substring grading per needle value. Pure."""
    return {v: (v in answer) for v in values}


def ask(url: str, key: str, prompt: str, timeout: float) -> tuple[str, int, float]:
    body = {"model": "strata", "max_tokens": 40, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    t0 = time.time()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return (out["choices"][0]["message"].get("content") or ""), out["usage"]["prompt_tokens"], time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--lengths", default="32k", help="comma-separated, e.g. 1k,32k,128k,262k")
    ap.add_argument("--depths", default="10,50,90", help="where a single needle sits, percent of the text")
    ap.add_argument("--repeats", type=int, default=1, help="fresh word per repeat, same depths")
    ap.add_argument("--needle-types", default="code", help=f"comma-separated of {NEEDLE_TYPES}")
    ap.add_argument("--distractors", type=int, default=0, help="hard distractor needles (other labels), asked by label")
    ap.add_argument("--queries", type=int, default=1, help="true needles per prompt, spread over 10-90%% (--depths ignored)")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed; words/positions are reproducible per seed")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", help="also write the results as JSON")
    a = ap.parse_args()
    kinds = [k.strip() for k in a.needle_types.split(",") if k.strip()]
    if not kinds or any(k not in NEEDLE_TYPES for k in kinds):
        raise SystemExit(f"--needle-types wants a comma list of {NEEDLE_TYPES}")
    if a.repeats < 1 or a.queries < 1 or a.distractors < 0:
        raise SystemExit("--repeats/--queries >= 1, --distractors >= 0")
    if a.queries + a.distractors > len(WORDS):
        raise SystemExit(f"--queries + --distractors <= {len(WORDS)} (distinct labels)")
    try:
        with urllib.request.urlopen(a.url.rstrip("/") + "/metrics", timeout=10) as r:
            ctx = int((json.loads(r.read()).get("engine") or {}).get("max_context") or 0)
    except (OSError, ValueError):
        ctx = 0
    rnd = random.Random(a.seed)
    rows, found, total = [], 0, 0
    for L in a.lengths.split(","):
        tokens = int(float(L.lower().rstrip("k")) * 1024) if L.lower().endswith("k") else int(L)
        tokens = int(tokens * 0.98)                        # "128k" fits a 128K context with room to answer
        if ctx and tokens + 200 > ctx:
            print(f"{L:>5}: skipped (the server's context is {ctx})")
            continue
        text = haystack(int(tokens * CHARS_PER_TOKEN))
        depths = [int(x) for x in a.depths.split(",")]
        for kind in kinds:
            for rep in range(a.repeats):
                if a.queries == 1:
                    cells = [(d, [d]) for d in depths]
                else:
                    spread = [10 + 80 * i / max(a.queries - 1, 1) for i in range(a.queries)]
                    cells = [(round(sum(spread) / len(spread)), spread)]
                    if depths != [10, 50, 90]:
                        print(f"{L:>5}: note: --queries {a.queries} spreads needles over 10-90%, --depths ignored")
                        depths = [10, 50, 90]  # print once
                for cell_d, qs in cells:
                    labels = rnd.sample(WORDS, a.queries + a.distractors)
                    truesty = [(q, labels[i], needle_value(rnd, kind)) for i, q in enumerate(qs)]
                    dist = [(rnd.uniform(5, 95), labels[a.queries + j], needle_value(rnd, kind))
                            for j in range(a.distractors)]
                    items = [(q, f"\nThe secret code word for {lab} is: {val}. Remember it.\n")
                             for q, lab, val in truesty]
                    items += [(f, f"\nThe secret code word for {lab} is: {val}. Remember it.\n")
                              for f, lab, val in dist]
                    prompt_text, _ = place_needles(text, items)
                    if a.queries == 1:
                        (q0, lab0, val0), = truesty
                        question = (f"\n\nWhat is the secret code word for {lab0} mentioned in the text "
                                    f"above? Reply with the code word only.")
                    else:
                        labs = ", ".join(lab for _, lab, _ in truesty)
                        question = (f"\n\nWhat are the secret code words for {labs} mentioned in the text "
                                    f"above? Reply with the code words only, separated by commas.")
                    try:
                        answer, n, secs = ask(a.url, a.api_key, prompt_text + question, a.timeout)
                    except (OSError, urllib.error.HTTPError) as e:
                        print(f"{L:>5} depth {cell_d:>3}%: request failed ({e})")
                        rows.append({"length": L, "depth": cell_d, "repeat": rep, "needle_type": kind,
                                     "error": str(e)})
                        continue
                    res = grade(answer, [val for _, _, val in truesty])
                    for (q, lab, val), ok in zip(truesty, res.values()):
                        total += 1
                        found += ok
                        tag = "FOUND" if ok else "missed"
                        print(f"{L:>5} depth {q:>3}% rep {rep} {kind:6s} {lab:8s}: {tag}  "
                              f"({n:,} prompt tokens, {secs:.0f} s)"
                              + ("" if ok else f"  answer: {answer.strip()[:80]!r}"), flush=True)
                        rows.append({"length": L, "depth": q, "repeat": rep, "needle_type": kind,
                                     "label": lab, "queries": a.queries, "distractors": a.distractors,
                                     "prompt_tokens": n, "seconds": round(secs, 1), "found": ok,
                                     "word": val, "answer": answer.strip()[:200]})
    done = [r for r in rows if "found" in r]
    print(f"{found} of {total} needles found")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    clean = all("error" not in r for r in rows)
    return 0 if done and found == total and clean else 1


if __name__ == "__main__":
    sys.exit(main())
