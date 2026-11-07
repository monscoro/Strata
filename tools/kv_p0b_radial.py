#!/usr/bin/env python3
"""P0b: is the 16/16 error radial (a norm error) or tangential (a direction error)?

Reads the raw, post-rotation / pre-quantization K/V that `STRATA_KV_Q4_DUMP` writes from the batched
prefill path and splits each 256-value vector's squared error exactly as KVarN does:

    x_dq    = quantize(x)
    n       = ||x||,  n_dq = ||x_dq||
    cos     = <x, x_dq> / (n * n_dq)
    radial  = (n - n_dq)^2
    tangent = 2 * n * n_dq * (1 - cos)
    total   = ||x - x_dq||^2 = radial + tangent

The oracle norm correction is `x_corr = x_dq * (n / n_dq)`; afterwards only `tangent` is left.  The radial
share `r = radial / total` is the number the pre-registered decision rule asks for: r < 5 % kills the
correction, 5...20 % is marginal (no build), r >= 20 % survives.  Per Dave's third reply the rule is read at
the 18 top-gap positions, not averaged over the 6,144; the mean is reported beside it.

Dave also points at Villines' per-head minimum-cosine probe (uniform TurboQuant collapses to a near-zero or
negative minimum cosine on Qwen while the mean stays nominal).  So the tool reports the minimum cosine per
(layer, head, K/V), over the whole dump and at the top-gap positions, for both codecs.  It is a diagnostic,
not a second decision rule.

Arms (all on the same vectors, so the comparison is paired by construction):

    max87       per 32-value block, d = max|x| / 87, the 16-level Lloyd table.  The current 18-byte block.
    rms32       per 256-value vector, d = ||x|| / 512 = RMS / 32.  The 16-byte candidate: the block stores
                codes only, the scale comes from the per-vector norm Dave's format stores anyway.
    max87+norm  oracle correction on max87 (the minimal 18-byte build).
    rms32+norm  oracle correction on rms32 (the 16-byte format with the correction).
    int8        per 64 values, fp16 d = amax / 127, round-to-nearest.  The codec yardstick.
    rms32blk    per 32-value block, d = RMS / 32.  Diagnostic only: the same rule per block, the arm the
                kv_q4.hpp header measured at 3.547e-02 on the synthetic Gaussian, kept for continuity.

Scale handling follows the kernels, not an ideal quantizer: q4 codes are chosen against the fp32 d and the
value is reconstructed with the fp16-stored d (kv_q4.cu's q4l_group); int8 is quantized against the stored
fp16 scale (kv_q8.cu).

    python tools/kv_p0b_radial.py --dump D:\\bench\\p0b\\kv.bin --json D:\\bench\\p0b\\p0b.json

    # optional: take the top-gap positions from the KL series instead of the five known from P0a
    python tools/kv_p0b_radial.py --dump kv.bin --top-n 18 \
        --arm-tsv ...\\probe.tsv --ref-tsv ...\\fp16.tsv --base-tsv ...\\int8.tsv

No GPU, no model: it only reads the dump and the TSVs.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# The 16-level Max-Lloyd table (kv_q4.hpp's kQ4lMap), integer, in units of d.
LEVELS = np.array([-87, -66, -52, -40, -30, -21, -12, -4, 4, 12, 21, 30, 40, 52, 66, 87], dtype=np.float64)
# The kernel's boundaries (q4l_level): midpoints, `>=` goes to the higher level.
MIDS = (LEVELS[:-1] + LEVELS[1:]) / 2.0

MAGIC = 0x51345042
HDR = np.dtype([("magic", "<i4"), ("layer", "<i4"), ("p0", "<i8"),
                ("T", "<i4"), ("nkv", "<i4"), ("hd", "<i4"), ("pad", "<i4")])
assert HDR.itemsize == 32

# P0a's five worst positions (win - int8, KL top-256), the ones that carry 54 % of the gap.
TOP5 = [130972, 127887, 130973, 130952, 130942]

ARMS = ["max87", "rms32", "max87+norm", "rms32+norm", "int8", "rms32blk"]
BITS = {"max87": 18 * 8 / 32, "rms32": 16 * 8 / 32 + 32 / 256,
        "max87+norm": 18 * 8 / 32 + 32 / 256, "rms32+norm": 16 * 8 / 32 + 32 / 256,
        "int8": 8.0 + 16 / 64, "rms32blk": 16 * 8 / 32}


def q4_levels(y: np.ndarray) -> np.ndarray:
    """The nearest level for values already divided by d, with the kernel's >= boundaries."""
    return LEVELS[np.searchsorted(MIDS, y, side="right")]


def _q4_with_scale(x: np.ndarray, d32: np.ndarray) -> np.ndarray:
    """Codes against the fp32 d, reconstruction with the fp16-stored d (kv_q4.cu)."""
    d16 = d32.astype(np.float16).astype(np.float32)
    y = np.divide(x, d32[:, None], out=np.zeros_like(x), where=d32[:, None] != 0)
    return q4_levels(y) * d16[:, None]


def q4_block(x: np.ndarray) -> np.ndarray:
    """32-value blocks, d = max|x| / 87."""
    return _q4_with_scale(x, (np.max(np.abs(x), axis=1) / 87.0).astype(np.float32))


def q4_block_rms(x: np.ndarray) -> np.ndarray:
    """32-value blocks, d = RMS / 32 (diagnostic; no stored scale in a 16-byte block)."""
    return _q4_with_scale(x, (np.sqrt(np.mean(x * x, axis=1)) / 32.0).astype(np.float32))


def q4_vector_rms(x: np.ndarray) -> np.ndarray:
    """256-value vectors, one d = RMS / 32 = ||x|| / 512, the 16-byte candidate."""
    return _q4_with_scale(x, (np.sqrt(np.mean(x * x, axis=1)) / 32.0).astype(np.float32))


def int8_block(x: np.ndarray) -> np.ndarray:
    """64-value groups, fp16 d = amax / 127, rint and clamp (kv_q8.cu)."""
    sf = (np.max(np.abs(x), axis=1).astype(np.float32) / 127.0).astype(np.float16).astype(np.float32)
    y = np.divide(x, sf[:, None], out=np.zeros_like(x), where=sf[:, None] != 0)
    q = np.clip(np.rint(y), -127.0, 127.0)
    return q * sf[:, None]


def decompose(x: np.ndarray, xdq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(radial, tangential, cos) per vector, in squared units over the whole 256 values."""
    n = np.linalg.norm(x, axis=1)
    nd = np.linalg.norm(xdq, axis=1)
    radial = (n - nd) ** 2
    dot = np.sum(x * xdq, axis=1)
    denom = n * nd
    cos = np.divide(dot, denom, out=np.ones_like(dot), where=denom > 0)
    tangent = 2.0 * denom * (1.0 - cos)
    return radial, tangent, cos


def corrected(x: np.ndarray, xdq: np.ndarray) -> np.ndarray:
    """x_dq * (||x|| / ||x_dq||): the oracle norm correction."""
    n = np.linalg.norm(x, axis=1)
    nd = np.linalg.norm(xdq, axis=1)
    scale = np.divide(n, nd, out=np.ones_like(n), where=nd > 0)
    return xdq * scale[:, None]


def quantize(x: np.ndarray, arm: str) -> np.ndarray:
    if arm == "max87":
        return q4_block(x.reshape(-1, 32)).reshape(x.shape)
    if arm == "rms32":
        return q4_vector_rms(x)
    if arm == "rms32blk":
        return q4_block_rms(x.reshape(-1, 32)).reshape(x.shape)
    if arm == "int8":
        return int8_block(x.reshape(-1, 64)).reshape(x.shape)
    if arm == "max87+norm":
        return corrected(x, q4_block(x.reshape(-1, 32)).reshape(x.shape))
    if arm == "rms32+norm":
        return corrected(x, q4_vector_rms(x))
    raise ValueError(arm)


def gap_series(arm_tsv: str, ref_tsv: str, base_tsv: str) -> dict:
    """{position: paired KL gap (arm - base)} from existing runs."""
    import quant_quality as qq
    arm, ref, base = (qq.read_run(p) for p in (arm_tsv, ref_tsv, base_tsv))
    gaps = {}
    for seg, ra in ref.items():
        aa, ba = arm.get(seg, {}), base.get(seg, {})
        for pos in set(ra) & set(aa) & set(ba):
            k_arm = qq.kl_between(ra[pos][4], aa[pos][4], 256)
            k_base = qq.kl_between(ra[pos][4], ba[pos][4], 256)
            gaps[pos] = k_arm - k_base
    return gaps


def gap_positions(arm_tsv: str, ref_tsv: str, base_tsv: str, n: int) -> list[int]:
    """The `n` positions with the largest paired KL gap (arm - base), from existing runs."""
    gaps = gap_series(arm_tsv, ref_tsv, base_tsv)
    return [p for p, _ in sorted(gaps.items(), key=lambda kv: -kv[1])[:n]]


class Acc:
    """Sums per arm: globally, per K/V, and per position (for the top-gap subset)."""

    def __init__(self, npos: int):
        self.n = 0                     # values
        self.sse = {a: 0.0 for a in ARMS}
        self.rad = {a: 0.0 for a in ARMS}
        self.tan = {a: 0.0 for a in ARMS}
        self.kv = {t: {"n": 0, **{a: 0.0 for a in ARMS}} for t in ("K", "V")}
        self.kvrad = {t: {a: 0.0 for a in ARMS} for t in ("K", "V")}
        self.pos_sse = {a: np.zeros(npos) for a in ARMS}
        self.pos_rad = {a: np.zeros(npos) for a in ARMS}
        self.pos_n = np.zeros(npos)
        self.pos_norm = np.zeros(npos)      # sum of ||x|| per vector
        self.pos_vecs = np.zeros(npos)
        # Villines probe: min cosine per (layer, tensor, head), over all positions and at the top-gap set
        self.cos_min = {}
        self.cos_top = {}
        self.cos_top_vals = {a: [] for a in ("max87", "rms32")}

    def add(self, x: np.ndarray, is_k: np.ndarray, positions: np.ndarray, base_pos: int,
            layer: int, nkv: int, top_pos: np.ndarray) -> None:
        idx = positions - base_pos
        heads = np.arange(len(x)) % nkv
        in_top = np.isin(positions, top_pos)
        self.n += x.size
        self.pos_n += np.bincount(idx, minlength=len(self.pos_n))
        norms = np.linalg.norm(x, axis=1)
        self.pos_norm += np.bincount(idx, weights=norms, minlength=len(self.pos_n))
        self.pos_vecs += np.bincount(idx, minlength=len(self.pos_n))
        for a in ARMS:
            dq = quantize(x, a)
            radial, tangent, cos = decompose(x, dq)
            err = np.sum((x - dq) ** 2, axis=1)
            self.sse[a] += float(err.sum())
            self.rad[a] += float(radial.sum())
            self.tan[a] += float(tangent.sum())
            for t, m in (("K", is_k), ("V", ~is_k)):
                self.kv[t][a] += float(err[m].sum())
                self.kv[t]["n"] += int(m.sum()) * x.shape[1]
                self.kvrad[t][a] += float(radial[m].sum())
            self.pos_sse[a] += np.bincount(idx, weights=err, minlength=len(self.pos_n))
            self.pos_rad[a] += np.bincount(idx, weights=radial, minlength=len(self.pos_n))
            if a not in ("max87", "rms32"):
                continue
            for t, m in (("K", is_k), ("V", ~is_k)):
                for h in range(nkv):
                    mh = m & (heads == h)
                    if not mh.any():
                        continue
                    key = (a, layer, t, h)
                    c = cos[mh]
                    self.cos_min[key] = min(self.cos_min.get(key, np.inf), float(c.min()))
                    ct = c[in_top[mh]]
                    if ct.size:
                        self.cos_top[key] = min(self.cos_top.get(key, np.inf), float(ct.min()))
                        self.cos_top_vals[a].extend(ct.tolist())


def metrics(acc: Acc, sel: np.ndarray | None = None) -> dict:
    out = {}
    for a in ARMS:
        sse = acc.sse[a] if sel is None else float(acc.pos_sse[a][sel].sum())
        rad = acc.rad[a] if sel is None else float(acc.pos_rad[a][sel].sum())
        n = acc.n if sel is None else float(acc.pos_n[sel].sum()) * 256.0
        out[a] = {"mse": sse / n if n else float("nan"),
                  "radial_share": rad / sse if sse else float("nan"),
                  "bits": BITS[a]}
    base = out["max87"]["mse"]
    for a in ARMS:
        out[a]["gain_db_vs_max87"] = (10 * math.log10(base / out[a]["mse"])
                                      if out[a]["mse"] > 0 else float("nan"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--json")
    ap.add_argument("--top-positions", help=f"comma list (default: {','.join(map(str, TOP5))})")
    ap.add_argument("--top-n", type=int, default=18)
    ap.add_argument("--arm-tsv")
    ap.add_argument("--ref-tsv")
    ap.add_argument("--base-tsv")
    ap.add_argument("--chunk", type=int, default=8192, help="vectors per numpy batch (default 8192)")
    a = ap.parse_args()

    top = [int(p) for p in a.top_positions.split(",")] if a.top_positions else list(TOP5)
    if a.arm_tsv and a.ref_tsv and a.base_tsv:
        top = gap_positions(a.arm_tsv, a.ref_tsv, a.base_tsv, a.top_n)
        print(f"top {len(top)} gap positions from the KL series: {top}")

    # first pass: the position range, so the per-position arrays are dense
    lo = hi = None
    with open(a.dump, "rb") as f:
        while True:
            h = np.fromfile(f, dtype=HDR, count=1)
            if h.size == 0:
                break
            r = h[0]
            if int(r["magic"]) != MAGIC:
                raise SystemExit(f"bad magic: {int(r['magic']):#x}")
            p0, T = int(r["p0"]), int(r["T"])
            lo = p0 if lo is None else min(lo, p0)
            hi = p0 + T if hi is None else max(hi, p0 + T)
            f.seek(2 * T * int(r["nkv"]) * int(r["hd"]) * 4, 1)   # K and V
    if lo is None:
        raise SystemExit("empty dump")
    print(f"dump {a.dump}: positions {lo}...{hi - 1} ({hi - lo}), "
          f"{pathlib.Path(a.dump).stat().st_size / 1e6:.0f} MB")

    acc = Acc(hi - lo)
    layers = set()
    top_pos = np.array(top, dtype=np.int64)
    with open(a.dump, "rb") as f:
        while True:
            h = np.fromfile(f, dtype=HDR, count=1)
            if h.size == 0:
                break
            r = h[0]
            p0, T, nkv, hd = int(r["p0"]), int(r["T"]), int(r["nkv"]), int(r["hd"])
            if hd != 256:
                raise SystemExit(f"head_dim {hd}: the decomposition is written for 256")
            layers.add(int(r["layer"]))
            cnt = T * nkv * hd
            K = np.fromfile(f, dtype=np.float32, count=cnt).reshape(T * nkv, hd)
            V = np.fromfile(f, dtype=np.float32, count=cnt).reshape(T * nkv, hd)
            positions = np.repeat(np.arange(p0, p0 + T, dtype=np.int64), nkv)
            is_k = np.concatenate([np.ones(len(K), dtype=bool), np.zeros(len(V), dtype=bool)])
            x = np.concatenate([K, V], axis=0)
            pos_all = np.concatenate([positions, positions])
            for s in range(0, len(x), a.chunk):
                sl = slice(s, min(s + a.chunk, len(x)))
                acc.add(x[sl], is_k[sl], pos_all[sl], lo, int(r["layer"]), nkv, top_pos)

    res = metrics(acc)
    sel_top = np.array([p - lo for p in top if lo <= p < hi], dtype=np.int64)
    sub = metrics(acc, sel_top) if len(sel_top) else None

    print(f"\n{len(layers)} layers, {acc.n // 256} vectors, {acc.n} values\n")
    print(f"  {'arm':11s} {'bits':>6s} {'MSE':>11s} {'radial':>8s} {'gain dB':>8s}")
    for a_ in ARMS:
        m = res[a_]
        print(f"  {a_:11s} {m['bits']:6.2f} {m['mse']:11.4e} {m['radial_share']:7.1%} "
              f"{m['gain_db_vs_max87']:+8.2f}")
    print("\n  K / V split (MSE / radial share):")
    for t in ("K", "V"):
        n = acc.kv[t]["n"]
        row = "  ".join(f"{a_}: {acc.kv[t][a_] / n:.3e} / "
                        f"{(acc.kvrad[t][a_] / acc.kv[t][a_] if acc.kv[t][a_] else float('nan')):.1%}"
                        for a_ in ("max87", "rms32"))
        print(f"    {t}: {row}")
    if sub is not None:
        print(f"\n  top-gap positions {top} ({len(sel_top)} in the dump):")
        print(f"  {'arm':11s} {'MSE':>11s} {'radial':>8s} {'gain dB':>8s}")
        for a_ in ARMS:
            m = sub[a_]
            print(f"  {a_:11s} {m['mse']:11.4e} {m['radial_share']:7.1%} {m['gain_db_vs_max87']:+8.2f}")

    # The tail mechanism: a large-norm vector with the normal relative error has a large absolute
    # error.  Per-position mean ||x|| says whether the top-gap positions are simply the loud ones.
    mean_norm = acc.pos_norm / np.maximum(acc.pos_vecs, 1)
    norm_out = {"mean": float(mean_norm.mean())}
    if len(sel_top):
        norm_out["mean_top"] = float(mean_norm[sel_top].mean())
        norm_out["ratio_top"] = norm_out["mean_top"] / norm_out["mean"]
        print(f"\n  mean vector norm: {norm_out['mean']:.3f} overall, "
              f"{norm_out['mean_top']:.3f} at the top-gap positions "
              f"({norm_out['ratio_top']:.1f}x)")
    if a.arm_tsv and a.ref_tsv and a.base_tsv:
        gaps = gap_series(a.arm_tsv, a.ref_tsv, a.base_tsv)
        g = np.array([gaps.get(p, np.nan) for p in range(lo, hi)])
        m = ~np.isnan(g) & (acc.pos_vecs > 0)
        r_gap = float(np.corrcoef(mean_norm[m], g[m])[0, 1])
        norm_out["pearson_r_gap"] = r_gap
        norm_out["positions_in_gap"] = int(m.sum())
        print(f"  Pearson r(mean norm, P0a gap) = {r_gap:+.3f} over {int(m.sum())} positions")

    # Villines' probe: the minimum cosine over the grid per (layer, head, K/V), not the mean.
    print("\n  min cosine per (layer, head, K/V) - the Villines probe:")
    cos_out = {}
    for a_ in ("max87", "rms32"):
        mins = {k[1:]: v for k, v in acc.cos_min.items() if k[0] == a_}
        if not mins:
            continue
        worst = min(mins, key=mins.get)
        neg = sum(1 for v in mins.values() if v < 0.0)
        low = sum(1 for v in mins.values() if v < 0.5)
        vals = np.array(acc.cos_top_vals[a_]) if acc.cos_top_vals[a_] else np.array([])
        cos_out[a_] = {"min_all": min(mins.values()), "worst": worst,
                       "units_negative": neg, "units_below_0.5": low, "units": len(mins),
                       "min_top": float(vals.min()) if len(vals) else float("nan"),
                       "p1_top": float(np.percentile(vals, 1)) if len(vals) else float("nan"),
                       "median_top": float(np.median(vals)) if len(vals) else float("nan")}
        c = cos_out[a_]
        print(f"    {a_:6s} all positions: min {c['min_all']:+.4f} at layer/tensor/head "
              f"{worst[0]}/{worst[1]}/{worst[2]}, {neg}/{c['units']} units < 0, "
              f"{low}/{c['units']} < 0.5")
        print(f"    {'':6s} top-gap: min {c['min_top']:+.4f}, p1 {c['p1_top']:+.4f}, "
              f"median {c['median_top']:+.4f} ({len(vals)} vectors)")

    # the pre-registered decision: r at the top-gap positions is primary (Dave's third reply)
    r = res["max87"]["radial_share"]
    r_top = sub["max87"]["radial_share"] if sub is not None else float("nan")
    gate = r_top if not math.isnan(r_top) else r
    if gate < 0.05:
        verdict = "dead"
    elif gate < 0.20:
        verdict = "marginal"
    else:
        verdict = "survives"
    build18 = res["max87+norm"]["mse"] <= 0.95 * res["max87"]["mse"]
    fmt16 = res["rms32+norm"]["mse"] < res["max87"]["mse"]
    print(f"\n  radial share max87: {r:.1%} overall, {r_top:.1%} at the top-gap positions "
          f"-> {verdict} (the top-gap share is the gate)")
    print(f"  18-byte build (MSE(max87+norm) <= 0.95 * MSE(max87)): {'yes' if build18 else 'no'}")
    print(f"  16-byte format (MSE(rms32+norm) < MSE(max87)):         {'yes' if fmt16 else 'no'}")

    if a.json:
        out = {"dump": a.dump, "positions": [lo, hi - 1], "layers": sorted(layers),
               "values": acc.n, "arms": res, "top_gap": sub, "top_positions": top,
               "cosine": cos_out, "norm": norm_out,
               "verdict": verdict, "build_18": bool(build18), "format_16": bool(fmt16)}
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
