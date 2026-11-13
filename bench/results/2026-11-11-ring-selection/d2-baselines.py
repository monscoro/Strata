#!/usr/bin/env python
"""D2, Nachtrag: die Zufalls-Baselines zu den gemessenen Zahlen - ohne den Dump zu lesen.

Der Dump beantwortet, was der Indexer tut. Er beantwortet nicht, ob eine Zahl
ueberhaupt etwas aussagt: eine Auswahl von 2.051 Zellen aus bis zu 131.060 ist so
breit, dass "irgendeine spaetere Query hat die Zelle gelesen" fast zwangslaeufig
gilt. Dieses Skript rechnet die Baseline einer GLEICHVERTEILTEN Auswahl derselben
Breite aus - analytisch, aus der Positionszahl allein, kein GPU-Lauf, kein Dump:

  Churn(L)        = 1 - w/n                      (zwei Zuege ohne Zuruecklegen)
  |U ueber N|     = n * (1 - (1 - w/n)^N)
  read_later(p)   = 1 - (1 - w/n)^(N_p)          je Layer, 12 Layer als ODER

n = p + 1 (die Zellen, die es zur Query p gibt), w = min(n, 2051) (die Breite der
Auswahl, wie D2-a sie misst), N_p = die Queries nach p.

  python bench/results/2026-11-11-ring-selection/d2-baselines.py \
      --ref  .../d1/t1-fp16.tsv --int8 .../d2/t1-int8.tsv
"""
import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("quant_quality", REPO / "tools" / "quant_quality.py")
qq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qq)

CAP = 2051      # qsa_selection_width: 2.048 Top-K-Zellen + 3 Block-Ids (D2-a, gemessen)
LAGS = (1, 8, 64)
UNION_NS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
LAYERS = 12


def width(p):
    return min(p + 1, CAP)


def churn(p, lag):
    n, w = p + 1, width(p)
    if n <= 0 or lag <= 0:
        return float("nan")
    return max(0.0, 1.0 - w / n)


def union(p, N):
    n, w = p + 1, width(p)
    return n * (1.0 - (1.0 - w / n) ** N)


def read_later_one_layer(p, n_later, w):
    n = p + 1
    if n_later <= 0:
        return 0.0
    return 1.0 - (1.0 - w / n) ** n_later


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--tail", type=float, default=0.01)
    a = ap.parse_args()

    ref = qq.read_run(a.ref)
    run = qq.read_run(a.int8)
    series = qq.kl_series(ref, run)
    pos = sorted(q for (_seg, q) in series)
    kl = [series[(_seg, q)] for (_seg, q) in series]
    n_pos = pos[-1] + 1
    order = sorted(range(len(pos)), key=lambda i: -kl[i])
    tail = [pos[i] for i in order[: max(1, int(round(len(pos) * a.tail)))]]
    print(f"positions {len(pos)} (0..{n_pos - 1}); tail {a.tail:.0%}: {len(tail)}")

    # Der Anker fuer die Mengen: die Mitte des Textes, wo eine Query die ganze
    # Breite hat (p >> cap) - dort ist die Baseline nicht durch die kurze Historie verzerrt.
    anchor = n_pos // 2
    print(f"\nBaseline bei einer Query der Laenge p = {anchor} (n = {anchor + 1} Zellen, w = {width(anchor)}):")
    for lag in LAGS:
        print(f"  Churn Lag {lag:3d}: Zufall {100 * churn(anchor, lag):5.2f} % neu")
    for N in UNION_NS:
        print(f"  |U| ueber {N:5d} Queries: Zufall {union(anchor, N):9.1f} Zellen "
              f"({100 * union(anchor, N) / (anchor + 1):5.1f} % des Kontexts)")

    # read_later: die tail-Positionen selbst - die schwaechste Stelle der Metrik
    worst = None
    for p in tail:
        n_later = n_pos - 1 - p
        one = read_later_one_layer(p, n_later, width(p))
        any_layer = 1.0 - (1.0 - one) ** LAYERS
        if worst is None or any_layer < worst[1]:
            worst = (p, any_layer, one, n_later)
    print(f"\nread_later, tail {a.tail:.0%}: die unguenstigste tail-Position (die wenigsten Queries danach)")
    print(f"  p = {worst[0]}, {worst[3]} Queries danach: Zufall je Layer {100 * worst[2]:.2f} %, "
          f"12 Layer als ODER {100 * worst[1]:.2f} %")
    print("  -> eine Metrik, die dort schon unter Zufall ~100 % liegt, kann den Indexer nicht loben.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
