#!/usr/bin/env python
"""Which positions are hard, and why: the heavy positions against the task's own words.

The 128K int8 arm's error mass sits in a few hundred positions (the fat tail, protocol entry of
2026-11-11 11:45).  The heavy positions come in runs of one to three consecutive positions, which
is the token length of a six-letter coded word.  The task (RULER FWE: "read the following coded
text and track the frequency of each coded word ... find the ten most frequently appeared coded
words") has ten such words --- the answer words, which are the most frequent ones in the text.

This script decodes the prompt's token ids with the pack's GPT-2-style byte-level BPE (no
tokenizer library needed: the byte-level decoder is enough), locates the answer words in the
decoded text, and compares the KL at those positions with the KL everywhere else.

Usage:
  python tools/kv_heavy_positions.py --ids <t1.ids.json> --ref <fp16.tsv> --int8 <int8.tsv> \
      --tokenizer <pack>/tokenizer [--gold <gold.json>] [--text <s00.txt>]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quant_quality", HERE / "quant_quality.py")
qq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qq)


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--gold")
    ap.add_argument("--kl-topk", type=int, default=qq.TOPK)
    args = ap.parse_args()

    tok = Path(args.tokenizer)
    # the GPT-2 vocab.json maps the token STRING to the id; we need the other direction
    vocab = {v: k for k, v in json.load(open(tok / "vocab.json", encoding="utf-8")).items()}
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    st = json.load(open(args.ids, encoding="utf-8"))
    ids = st["segments"][0]["ids"]

    # decode the prompt: the text and the character span of every token
    text_parts = []
    spans = []
    pos = 0
    for t in ids:
        s = vocab.get(int(t), "")
        raw = bytes(u2b.get(c, 32) for c in s)
        txt = raw.decode("utf-8", errors="replace")
        text_parts.append(txt)
        spans.append((pos, pos + len(txt)))
        pos += len(txt)
    text = "".join(text_parts)
    print(f"prompt tokens: {len(ids)}   decoded characters: {len(text)}")

    # the answer words: from the gold file, or the most frequent words of the text
    words = None
    if args.gold:
        gold = json.load(open(args.gold, encoding="utf-8"))
        for g in gold:
            if g.get("tokens") == len(ids) or g.get("name") == st["segments"][0]["name"]:
                words = g["answer"]
                break
        if words is None:
            words = gold[0]["answer"]
    print(f"answer words: {words}")

    # the token positions that belong to an answer word
    answer_positions = []
    for w in words:
        start = 0
        while True:
            i = text.find(w, start)
            if i < 0:
                break
            for t, (a, b) in enumerate(spans):
                if a <= i < b:
                    answer_positions.append(t)
                    break
            start = i + 1
    answer_positions = sorted(set(answer_positions))
    print(f"token positions of the answer words: {len(answer_positions)}")

    ref = qq.read_run(args.ref)
    series = qq.kl_series(ref, qq.read_run(args.int8), args.kl_topk)
    keys = sorted(series)
    vals = np.array([series[k] for k in keys])
    ap = set(answer_positions)
    is_answer = np.array([k[1] in ap for k in keys])

    print()
    print(f"{'group':>28} {'positions':>10} {'mean KL':>12} {'p99':>12} {'share of the error mass':>24}")
    print("-" * 92)
    for label, mask in (("at the answer words", is_answer),
                        ("everywhere else", ~is_answer)):
        v = vals[mask]
        if len(v) == 0:
            continue
        s = np.sort(v)
        print(f"{label:>28} {len(v):10d} {v.mean():12.5f} {s[int(0.99*(len(s)-1))]:12.4f} "
              f"{100*v.sum()/vals.sum():23.1f} %")
    print()
    top = sorted(keys, key=lambda k: series[k], reverse=True)[:1000]
    n_top = sum(1 for k in top if k[1] in ap)
    print(f"of the top-1,000 positions, {n_top} are answer-word positions "
          f"({100*n_top/len(ap):.1f} % of the {len(ap)} answer-word positions, "
          f"{100*n_top/1000:.1f} % of the top-1,000)")
    if len(ap):
        share = vals[is_answer].sum() / vals.sum()
        base = len(ap) / len(vals)
        print(f"the answer words are {100*base:.2f} % of the positions and carry "
              f"{100*share:.1f} % of the error mass (a share ratio of {share/base:.0f})")


if __name__ == "__main__":
    main()
