#!/usr/bin/env python
"""D2: die Auswertung des Selektions-Dumps (Registrierung: README.md in diesem Ordner).

Liest `sel_dump.bin` (Format: `[int64 pos0, layer, T, cap]` + `T*cap` int32 je Datensatz, siehe
src/core/verify.cpp) und rechnet die sechs vorregistrierten Groessen D2-a ... D2-f aus. Die
tail-Positionen kommen aus den bestehenden 128K-Armen (`d1/t1-int8.tsv` gegen `d1/t1-fp16.tsv`),
mit derselben KL-Definition wie tools/kv_tail_top1.py (`qq.kl_series`, TOPK 64).

  python bench/results/2026-11-11-ring-selection/analyze-d2.py \
      --dump C:/Users/krone/git/Strata-data/bench/2026-11-10-merge-reeval/d2/sel_dump.bin \
      --ref  C:/Users/krone/git/Strata-data/bench/2026-11-10-merge-reeval/d1/t1-fp16.tsv \
      --int8 C:/Users/krone/git/Strata-data/bench/2026-11-10-merge-reeval/d1/t1-int8.tsv \
      --json C:/Users/krone/git/Strata-data/bench/2026-11-10-merge-reeval/d2/analysis.json

Der Dump wird EINMAL sequentiell gelesen (~13 GB), alle Groessen entstehen in diesem einen Durchlauf:
der Ring der letzten 4096 Queries je Layer traegt D2-c und D2-d, die Zaehler tragen a/b/e/f.
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("quant_quality", REPO / "tools" / "quant_quality.py")
qq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qq)

RING = 4096            # Queries, die fuer Churn und Vereinigung gehalten werden (= Pool-Kapazitaet, G9)
LAGS = (1, 8, 64)      # D2-d: Churn gegen die Auswahl von vor 1, 8, 64 Queries
UNION_NS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
ANCHOR = 4096          # D2-c wird an jeder 4096. Query ausgewertet (32 Anker im 128K-Lauf)
AGE_BINS = 20          # floor(log2(alter+1)); 131071 -> 17
R_DEFAULT, S_DEFAULT = 64, 256   # der positionale Ring aus D1 (STRATA_KV_WINDOW_R/S)


def load_tail(ref_path, int8_path, frac):
    """Die tail-Positionen: die `frac` Queries mit der groessten int8-KL (arm-unabhaengig definiert)."""
    ref = qq.read_run(ref_path)
    run = qq.read_run(int8_path)
    series = qq.kl_series(ref, run)
    if len(ref) != 1:
        raise SystemExit(f"{int8_path}: {len(ref)} segments, the dump is one sequence; the position mapping is ambiguous")
    pos = np.array([q for (_seg, q) in series], dtype=np.int64)
    kl = np.array([series[k] for k in series], dtype=np.float64)
    order = np.argsort(-kl, kind="stable")
    n = max(1, int(round(len(pos) * frac)))
    tail = pos[order[:n]]
    return pos, kl, np.sort(tail)


class Layer:
    def __init__(self, n_pos, cap, ring, union_ns):
        self.ring = np.zeros((ring, cap), dtype=np.int32)
        self.ring_w = np.zeros(ring, dtype=np.int32)
        self.q = 0
        self.mask = np.zeros(n_pos, dtype=bool)          # scratch fuer die Churn-Lags
        self.last_sel = np.full(n_pos, -1, dtype=np.int64)   # letzte Query, die die Zelle gewaehlt hat
        self.any_sel = np.zeros(n_pos, dtype=bool)
        self.width_sum = 0
        self.rows = 0
        self.at_cap = 0
        self.identity = 0
        self.cells = 0
        self.ring_hits = 0
        self.sink_hits = 0
        self.win_hits = 0
        self.age = np.zeros(AGE_BINS, dtype=np.int64)
        self.churn_new = {L: 0 for L in LAGS}
        self.churn_tot = {L: 0 for L in LAGS}
        self.union = {N: [] for N in union_ns}
        self.tail_sel = 0
        self.own_hits = 0
        self.self_sel = np.zeros(n_pos, dtype=bool)   # the query AT p selected the cell p (the tail-bias artefact)


def analyze(dump, n_pos, cap_expected, tails, r, s, ring, anchor, union_ns, progress):
    layers = {}
    recs = 0
    windows = set()
    bad = 0
    with open(dump, "rb") as f:
        while True:
            hdr = np.fromfile(f, dtype="<i8", count=4)
            if hdr.size == 0:
                break
            if hdr.size < 4:
                print(f"warning: trailing partial header after {recs} records", file=sys.stderr)
                break
            pos0, layer, T, cap = (int(x) for x in hdr)
            if cap != cap_expected:
                raise SystemExit(f"record {recs}: cap {cap} != {cap_expected} (mixed dumps?)")
            ids = np.fromfile(f, dtype="<i4", count=T * cap)
            if ids.size < T * cap:
                print(f"warning: trailing partial record after {recs} records", file=sys.stderr)
                break
            ids = ids.reshape(T, cap)
            recs += 1
            windows.add(pos0)
            st = layers.get(layer)
            if st is None:
                st = layers[layer] = Layer(n_pos, cap, ring, union_ns)
            if T <= 0 or pos0 < 0:
                bad += 1
                continue
            for row_i in range(T):
                p = pos0 + row_i
                if p < 0 or p >= n_pos:
                    bad += 1
                    continue
                w = min(p + 1, cap)
                row = ids[row_i, :w]
                if row.size and (row.min() < 0 or row.max() >= n_pos):
                    bad += 1
                    continue
                qi = st.q
                slot = qi % ring
                st.ring[slot, :w] = row
                st.ring_w[slot] = w
                st.q += 1

                # D2-a: Groesse der Auswahl
                st.width_sum += w
                st.rows += 1
                st.at_cap += (w >= cap)
                st.identity += (p + 1 <= cap)      # unter der Schwelle ist die Auswahl die Identitaet

                # D2-b: der positionale Ring (Kopf c < R, Fenster c >= p-S+1)
                tail_start = max(0, p - s + 1)
                sink = row < r
                win = row >= tail_start
                st.sink_hits += int(sink.sum())
                st.win_hits += int(win.sum())
                st.ring_hits += int((sink | win).sum())
                st.cells += w

                # D2-e: Alter der gelesenen Zelle
                age = p - row
                b = np.minimum(np.log2(age + 1).astype(np.int32), AGE_BINS - 1)
                st.age += np.bincount(b, minlength=AGE_BINS)

                # D2-f: tail-Zellen in dieser Auswahl, und "wird die eigene Zelle gelesen"
                if tails is not None:
                    st.tail_sel += int(tails[row].sum())
                st.own_hits += int((row == p).any())
                if (row == p).any():
                    st.self_sel[p] = True   # the incomplete-tail bias makes this almost always true; say how often
                st.any_sel[row] = True
                st.last_sel[row] = qi

                # D2-d: Churn gegen die Auswahl von vor L Queries
                for lag in LAGS:
                    if qi < lag:
                        continue
                    ps = (qi - lag) % ring
                    prev = st.ring[ps, :st.ring_w[ps]]
                    st.mask[prev] = True
                    st.churn_new[lag] += int(np.count_nonzero(~st.mask[row]))
                    st.mask[prev] = False
                    st.churn_tot[lag] += w

                # D2-c: Wachstum der Vereinigung ueber die letzten N Queries
                if qi >= ring and qi % anchor == 0:
                    u = np.zeros(n_pos, dtype=bool)
                    idx = 0
                    for N in union_ns:
                        while idx < N:
                            qq_slot = (qi - 1 - idx) % ring   # rueckwaerts: die letzte Query ist qi-1
                            u[st.ring[qq_slot, :st.ring_w[qq_slot]]] = True
                            idx += 1
                        st.union[N].append(int(u.sum()))
            if progress and recs % progress == 0:
                print(f"  {recs} records, {len(windows)} windows, last pos0 {pos0}", file=sys.stderr, flush=True)
    return layers, recs, windows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ref", required=True, help="fp16-Referenz des 128K-Laufs (d1/t1-fp16.tsv)")
    ap.add_argument("--int8", required=True, help="int8-Arm des 128K-Laufs (d1/t1-int8.tsv)")
    ap.add_argument("--json", help="Ergebnis als JSON hierher schreiben")
    ap.add_argument("--tail", type=float, nargs="+", default=[0.01, 0.02])
    ap.add_argument("--window-r", type=int, default=R_DEFAULT)
    ap.add_argument("--window-s", type=int, default=S_DEFAULT)
    ap.add_argument("--progress", type=int, default=0)
    ap.add_argument("--ring", type=int, default=RING, help="Queries im Ring (Default 4096 = Pool-Kapazitaet)")
    ap.add_argument("--anchor", type=int, default=ANCHOR, help="D2-c an jeder N-ten Query")
    ap.add_argument("--allow-short-ring", action="store_true", help="Testlauf: Ring <= max(Lags) zulassen (verfaelscht D2-d)")
    a = ap.parse_args()
    if a.ring <= max(LAGS) and not a.allow_short_ring:
        raise SystemExit(f"--ring {a.ring} <= max lag {max(LAGS)}: the churn lags would read the slot just written")

    pos_all, kl_all, t1 = load_tail(a.ref, a.int8, a.tail[0])
    n_pos = int(pos_all.max()) + 1
    print(f"positions with a KL: {len(pos_all)} (0..{n_pos - 1}); tail {a.tail[0]:.0%}: {len(t1)}")

    tails_masks = {}
    for frac in a.tail:
        _p, _kl, t = load_tail(a.ref, a.int8, frac)
        m = np.zeros(n_pos, dtype=bool)
        m[t] = True
        tails_masks[frac] = (t, m)
        print(f"tail {frac:.0%}: {len(t)} positions, KL >= {np.sort(kl_all)[-len(t)]:.5f}")

    # cap aus dem ersten Datensatz: 4 int64 Kopf, dann T*cap int32
    with open(a.dump, "rb") as f:
        hdr = np.fromfile(f, dtype="<i8", count=4)
    if hdr.size < 4:
        raise SystemExit(f"{a.dump}: empty")
    cap = int(hdr[3])
    print(f"dump: cap {cap} cells per row, first record pos0 {int(hdr[0])} layer {int(hdr[1])} T {int(hdr[2])}")
    if int(hdr[0]) != 0:
        print(f"warning: the first window starts at {int(hdr[0])}, not 0 (the prompt path was not dumped?)", file=sys.stderr)

    tail_frac = a.tail[0]
    tail_pos, tail_mask = tails_masks[tail_frac]
    union_ns = tuple(n for n in UNION_NS if n <= a.ring) or (1,)
    layers, recs, windows, bad = analyze(a.dump, n_pos, cap, tail_mask, a.window_r, a.window_s,
                                         a.ring, a.anchor, union_ns, a.progress)
    if not layers:
        raise SystemExit("no records")
    print(f"records: {recs}, windows: {len(windows)}, qsa layers: {len(layers)}, skipped: {bad}")

    rep = {"dump": str(a.dump), "cap": cap, "records": recs, "windows": len(windows), "skipped": bad,
           "n_positions": n_pos, "window_r": a.window_r, "window_s": a.window_s,
           "tail": {}, "layers": {}}

    def agg(fn):
        return [fn(st) for _l, st in sorted(layers.items())]

    # ---- D2-a ----
    widths = agg(lambda st: st.width_sum / st.rows)
    at_cap = agg(lambda st: st.at_cap / st.rows)
    ident = agg(lambda st: st.identity / st.rows)
    print("\nD2-a  Auswahlgroesse je Query (Zellen)")
    print(f"  mean {np.mean(widths):.1f}  min {np.min(widths):.1f}  max {np.max(widths):.1f}  "
          f"(cap {cap}); am cap {100 * float(np.mean(at_cap)):.2f} %; Identitaet (<={cap} Zellen) {100 * float(np.mean(ident)):.2f} %")
    rep["D2-a"] = {"width_mean": float(np.mean(widths)), "width_min": float(np.min(widths)),
                   "width_max": float(np.max(widths)), "at_cap": float(np.mean(at_cap)),
                   "identity": float(np.mean(ident)), "per_layer_mean": [float(x) for x in widths]}

    # ---- D2-b ----
    ring = agg(lambda st: st.ring_hits / st.cells)
    sink = agg(lambda st: st.sink_hits / st.cells)
    win = agg(lambda st: st.win_hits / st.cells)
    print("\nD2-b  Ueberdeckung des positionalen Rings (R+S) durch die Auswahl")
    print(f"  Ring gesamt {100 * float(np.mean(ring)):.2f} %  (Kopf c<{a.window_r}: {100 * float(np.mean(sink)):.2f} %, "
          f"Fenster c>={a.window_s}: {100 * float(np.mean(win)):.2f} %)")
    print(f"  je Layer: " + " ".join(f"{100 * x:.2f}" for x in ring) + " %")
    rep["D2-b"] = {"ring": float(np.mean(ring)), "sink": float(np.mean(sink)), "window": float(np.mean(win)),
                   "per_layer": [float(x) for x in ring]}

    # ---- D2-c ----
    print("\nD2-c  Wachstum der Vereinigung |U der letzten N Queries| (Zellen), Mittel ueber die Anker")
    print(f"  {'N':>5} {'Mittel':>9} {'max':>8}   Reserve gegen {a.ring}")
    union_curves = {}
    for N in sorted(next(iter(layers.values())).union):
        vals = [v for st in layers.values() for v in st.union[N]]
        if not vals:
            continue
        m, mx = float(np.mean(vals)), float(np.max(vals))
        union_curves[N] = {"mean": m, "max": mx, "anchors": len(vals) // max(1, len(layers))}
        print(f"  {N:5d} {m:9.1f} {mx:8.1f}   {a.ring / m:6.2f}x")
    rep["D2-c"] = union_curves
    if a.ring in union_curves:
        g9 = a.ring / union_curves[a.ring]["mean"]
        verdict = "offen (Reserve >= 4)" if g9 >= 4 else ("marginal (1..4)" if g9 >= 1 else "geschlossen (< 1)")
        print(f"  G9 bei N={a.ring}: Reserve {g9:.2f}x -> {verdict}")
        rep["G9"] = {"reserve": g9, "verdict": verdict}

    # ---- D2-d ----
    print("\nD2-d  Churn: Anteil der Zellen, die die Auswahl der Query vor L NICHT enthielt")
    churn = {}
    for L in LAGS:
        vals = agg(lambda st, L=L: st.churn_new[L] / st.churn_tot[L])
        churn[str(L)] = float(np.mean(vals))
        print(f"  Lag {L:3d}: {100 * float(np.mean(vals)):5.2f} % neu (je Layer "
              + " ".join(f"{100 * x:.1f}" for x in vals) + ")")
    rep["D2-d"] = churn

    # ---- D2-e ----
    print("\nD2-e  Zeitprofil der gelesenen Zellen: Alter = p - c")
    hist = np.zeros(AGE_BINS, dtype=np.int64)
    for st in layers.values():
        hist += st.age
    tot = int(hist.sum())
    med = None
    acc = 0
    for b in range(AGE_BINS):
        acc += int(hist[b])
        if med is None and acc >= tot / 2:
            med = 2 ** b - 1
    print(f"  Median-Alter ~{med} Zellen; Alter < {a.window_s}: {100 * float(hist[:8].sum() / tot):.1f} % "
          f"(log2-Bins 0..7), Anteil > 4096: {100 * float(sum(hist[13:]) / tot):.1f} %")
    print(f"  Alterverteilung je log2-Bin: " + " ".join(f"2^{b}={100 * x / tot:.1f}" for b, x in enumerate(hist) if x))
    rep["D2-e"] = {"median_age": med, "hist": [int(x) for x in hist],
                   "share_young": float(hist[:8].sum() / tot), "share_old_4096": float(sum(hist[13:]) / tot)}

    # ---- D2-f ----
    print("\nD2-f  tail-Positionen unter den Auswahlen")
    f_all = {}
    for frac, (tp, _m) in tails_masks.items():
        own = float(np.mean([st.self_sel[tp].mean() for st in layers.values()]))
        # "gelesen": die Zelle p wird von einer SPAETEREN Query gelesen (irgendein Layer)
        later = np.zeros(len(tp), dtype=bool)
        anyl = np.zeros(len(tp), dtype=bool)
        lastq = np.zeros(len(tp), dtype=bool)
        for l, st in layers.items():
            later |= st.last_sel[tp] > tp
            anyl |= st.last_sel[tp] >= 0
            lastq |= st.last_sel[tp] == st.q - 1
        share = sum(st.tail_sel for st in layers.values()) / sum(st.cells for st in layers.values())
        d = {"n": int(len(tp)),
             "ever_selected": float(anyl.mean()),       # irgendeine Query ueberhaupt
             "read_later": float(later.mean()),         # eine spaetere Query (die Bruecke)
             "read_by_last": float(lastq.mean()),       # die letzte Query des Laufs
             "own_query": float(own),                   # die Query an p selbst (fast immer wahr: Tail-Bias)
             "share_of_selections": float(share)}       # reverse Lesart: Anteil der tail-Zellen an allen Auswahlen
        f_all[f"{frac:.2f}"] = d
        print(f"  tail {frac:.0%} ({len(tp)} Positionen): in mind. einem Layer ausgewaehlt "
              f"{100 * d['ever_selected']:5.1f} %; von einer spaeteren Query gelesen "
              f"{100 * d['read_later']:5.1f} %; von der letzten Query "
              f"{100 * d['read_by_last']:5.1f} %; eigene Query "
              f"{100 * d['own_query']:5.1f} %; Anteil an allen Auswahlen {100 * d['share_of_selections']:.2f} %")
    rep["D2-f"] = f_all
    primary = f_all[f"{tail_frac:.2f}"]["read_later"]
    rep["G10"] = {"metric": "read_later", "value": primary,
                  "verdict": "Bruecke traegt (>= 50 %)" if primary >= 0.5 else "Bruecke traegt nicht (< 50 %)"}
    print(f"\nG10 (read_later, tail {tail_frac:.0%}): {100 * primary:.2f} % -> {rep['G10']['verdict']}")
    print("  (G11 ist keine Messung: ein auswahlfolgender Pool ist eine Vermutung, keine Garantie.)")

    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"\nJSON: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
