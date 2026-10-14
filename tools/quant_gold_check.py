"""Check that a built FWE/CWE set's gold answer really is the realized most-frequent word order.

The gold is `vocab[1:1+k]`, i.e. the *theoretical* Zipf order. `fwe_context` draws keyword counts as
`int(num_words * k**-alpha / zeta)`, which is deterministic, so the realized order should equal it - unless two ranks
have the same integer count, in which case the order is a tie and the gold is ambiguous. This prints the realized
counts for the asked-about ranks and the first rank that ties with the one before it, so a set that is about to be
used for a measurement can be shown to have an unambiguous answer order.

Usage: quant_gold_check.py data/quant_quality/<set>
"""
import collections
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quant_quality as qq  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    set_dir = pathlib.Path(sys.argv[1])
    ids = json.loads((set_dir / "set.ids.json").read_text(encoding="utf-8"))
    gold = json.loads((set_dir / "gold.json").read_text(encoding="utf-8"))
    k = ids.get("answer_words") or 3
    alpha, seed, vocab_size = ids["alpha"], ids["seed"], ids["vocab_size"]
    zeta = qq.riemann_zeta(alpha)
    ok = True
    for i, g in enumerate(gold):
        vocab = qq.fwe_vocab(vocab_size, random.Random(seed + i))
        counts = collections.Counter(qq.fwe_context(vocab, g["drawn_words"], alpha, zeta, seed * 1000 + i).split())
        realized = [w for w, _ in counts.most_common(1 + k)][1:1 + k]
        c = [counts[w] for w in realized]
        tie = next((j for j in range(2, vocab_size) if counts[vocab[j - 1]] == counts[vocab[j]]), None)
        same = realized == g["answer"]
        ok &= same
        print(f"{g['name']}: asked {k}, gold {'==' if same else '!='} realized, "
              f"counts {c}, rank {k} = {counts[realized[k - 1]]}, rank {k + 1} = {counts[vocab[k + 1]]}, "
              f"first tie at rank {tie}")
    print("gold is the realized order:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
