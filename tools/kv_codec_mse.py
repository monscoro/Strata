#!/usr/bin/env python
"""The codec decision measurement of the centering mini-project (bench/results/2026-11-10-centering/).

Reads the real cells the engine dumped (STRATA_KV_Q4_DUMP: the rotated K/V rows of the batched
prompt path) and the per-channel moments of the S2 probe (the per-channel means before the
rotation), and computes for every codec variant:

  * the block MSE (the quantity the parity harness reports in dB) against the true values;
  * the per-block resolution u = sigma(block) / effective step, in steps per sigma - the "how well
    does the codebook use the value range" measure: with a fixed bias in the block the spread (and
    so sigma) grows while the step is set by the maximum, and u falls.

The variants:
  uniform+max   the q4_0 codec: value = (code - 8) * d, d = max|x| / 87            (affine)
  uniform+LS    the same grid, but d fitted once per block to the block's own values   (affine)
  lloyd16+max   Max-Lloyd's 16 levels, d = max|x| / 87                              (not affine)
  lloyd16+LS    Max-Lloyd's 16 levels, d fitted once per block                        (not affine)
  lloyd8+...    the 8-level ladder (V side of q4lm8v)

With --center the measured per-channel offset is subtracted before the quantization; in the
rotated domain that is the rotated offset H*m (the rotation is linear, so H(k) - H(m) = H(k-m)).
"""
import argparse
import math
import struct

import numpy as np

KQ4 = 32
L16 = np.array([-87, -66, -52, -40, -30, -21, -12, -4, 4, 12, 21, 30, 40, 52, 66, 87], dtype=np.float64)
L8 = np.array([-69, -43, -24, -8, 8, 24, 43, 69], dtype=np.float64)
L16_MAX = 87.0
L8_MAX = 69.0


def fwht(x):
    """The same Sylvester construction the engine uses, scale 1/16 (orthonormal)."""
    a = np.array(x, dtype=np.float64, copy=True)
    n = a.shape[-1]
    s = 1
    while s < n:
        idx = np.arange(n)
        left = idx[(idx // s) % 2 == 0]
        right = left + s
        tmp = a[..., left].copy()
        a[..., left] = tmp + a[..., right]
        a[..., right] = tmp - a[..., right]
        s *= 2
    return a / 16.0


def read_s2_means(path):
    """{layer: (256,) means before the rotation} (the S2 probe merges the two KV heads)."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.split()
            if len(p) < 14:
                continue
            layer, ch, mean_pre = int(p[0]), int(p[1]), float(p[4])
            v = out.setdefault(layer, np.zeros(256, dtype=np.float64))
            v[ch] = mean_pre
    return out


def read_dump(path, max_records=None, want_layers=None):
    """The dumped records: (layer, p0, T, nkv, hd, K[T, nkv, hd], V[T, nkv, hd])."""
    recs = []
    hdr = struct.Struct("<iiqiiii")
    with open(path, "rb") as f:
        while True:
            head = f.read(hdr.size)
            if len(head) < hdr.size:
                break
            magic, layer, p0, T, nkv, hd, _pad = hdr.unpack(head)
            if magic != 0x51345042:
                raise SystemExit(f"bad magic {magic:#x} at record {len(recs)}")
            n = T * nkv * hd
            k = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(T, nkv, hd).astype(np.float64)
            v = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(T, nkv, hd).astype(np.float64)
            if want_layers is None or layer in want_layers:
                recs.append((layer, p0, T, nkv, hd, k, v))
            if max_records and len(recs) >= max_records:
                break
    return recs


def quantize_uniform(x, fit_scale):
    """The q4_0 grid: code = round(x/d) + 8 clamped to 0..15, value = (code-8)*d.

    fit_scale: False -> d = max|x|/87 (the engine's rule); True -> one least-squares pass over the
    block's own values against the uniform code values (the candidate of catalog line A8)."""
    amax = np.max(np.abs(x))
    d = amax / L16_MAX if amax > 0 else 0.0
    if d == 0.0:
        return np.zeros_like(x), 0.0
    code = np.clip(np.rint(x / d) + 8.0, 0.0, 15.0)
    if fit_scale:
        c = code - 8.0
        den = float(np.dot(c, c))
        if den > 0:
            d = float(np.dot(c, x) / den)
            if d < 0:
                d = 0.0
            else:
                code = np.clip(np.rint(x / d) + 8.0, 0.0, 15.0)
    recon = (code - 8.0) * d
    return recon, d


def quantize_lloyd(x, levels, fit_scale):
    """Max-Lloyd's table: the nearest level to x/d, value = level * d (d = max|x|/max|level|)."""
    lmax = float(np.max(np.abs(levels)))
    amax = np.max(np.abs(x))
    d = amax / lmax if amax > 0 else 0.0
    if d == 0.0:
        return np.zeros_like(x), 0.0
    y = x / d
    idx = np.abs(y[:, None] - levels[None, :]).argmin(axis=1)
    if fit_scale:
        l = levels[idx]
        den = float(np.dot(l, l))
        if den > 0:
            d = float(np.dot(l, x) / den)
            if d < 0:
                d = 0.0
            else:
                y = x / d
                idx = np.abs(y[:, None] - levels[None, :]).argmin(axis=1)
    recon = levels[idx] * d
    return recon, d


def blocks(arr, nblk):
    """(rows, nkv, hd) -> (rows*nkv*nblk, KQ4) blocks along the head dimension."""
    rows, nkv, hd = arr.shape
    b = arr.reshape(rows * nkv, nblk, KQ4)
    return b.reshape(-1, KQ4)


def evaluate(name, krec, vrec, kvar, vvar, centered, offsets_rot):
    """One variant: the MSE over K and V, and the per-block resolution u."""
    out = {}
    for side, arr, var in (("K", krec, kvar), ("V", vrec, vvar)):
        if centered:
            layer = var.pop("_layer")
            arr = arr - offsets_rot[layer][None, None, :]
        bl = blocks(arr, 8)
        err = np.empty(bl.shape[0])
        u = np.empty(bl.shape[0])
        step = np.empty(bl.shape[0])
        for i in range(bl.shape[0]):
            x = bl[i]
            if var["kind"] == "uniform":
                recon, d = quantize_uniform(x, var["fit"])
            else:
                recon, d = quantize_lloyd(x, var["levels"], var["fit"])
            err[i] = float(np.mean((recon - x) ** 2))
            # the effective step: the reconstruction range over the levels the block actually used
            used = np.unique(np.rint(recon / d) if d > 0 else recon)
            step[i] = (recon.max() - recon.min()) / max(len(used) - 1, 1) if d > 0 else 0.0
            u[i] = float(x.std()) / step[i] if step[i] > 0 else 0.0
        out[side] = (float(err.mean()), float(np.median(u)), float(np.median(np.abs(err))))
    mse = 0.5 * (out["K"][0] + out["V"][0])
    umed = 0.5 * (out["K"][1] + out["V"][1])
    return name, mse, umed, out["K"][0], out["V"][0], out["K"][1], out["V"][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--layers", type=int, default=None, help="only the first N layers")
    ap.add_argument("--max-cells", type=int, default=None, help="only N cells per layer")
    ap.add_argument("--center", action="store_true", help="subtract the measured per-channel offset")
    args = ap.parse_args()

    means = read_s2_means(args.s2)
    offsets_rot = {layer: fwht(v) for layer, v in means.items()}

    recs = read_dump(args.dump, want_layers=set(range(args.layers)) if args.layers else None)
    if not recs:
        raise SystemExit("no records in the dump")
    print(f"dump: {len(recs)} records, layers {sorted(set(r[0] for r in recs))}")

    # one variant set over all records
    variants = [
        ("uniform+max (q4_0, der Anker)", dict(kind="uniform", fit=False), dict(kind="uniform", fit=False)),
        ("uniform+LS  (A8-Kandidat, affin)", dict(kind="uniform", fit=True), dict(kind="uniform", fit=True)),
        ("lloyd16+max", dict(kind="lloyd", levels=L16, fit=False), dict(kind="lloyd", levels=L16, fit=False)),
        ("lloyd16+LS", dict(kind="lloyd", levels=L16, fit=True), dict(kind="lloyd", levels=L16, fit=True)),
        ("lloyd16K+lloyd8V (q4lm8v-Codec)", dict(kind="lloyd", levels=L16, fit=False),
         dict(kind="lloyd", levels=L8, fit=False)),
        ("lloyd16K+lloyd8V+LS", dict(kind="lloyd", levels=L16, fit=True), dict(kind="lloyd", levels=L8, fit=True)),
    ]

    base = None
    rows = []
    for name, kv, vv in variants:
        tot_err = 0.0
        tot_n = 0
        us = []
        for layer, p0, T, nkv, hd, karr, varr in recs:
            if args.max_cells:
                karr, varr = karr[:args.max_cells], varr[:args.max_cells]
                T = args.max_cells
            bl_k = blocks(karr, 8)
            bl_v = blocks(varr, 8)
            for side, bl, var in (("K", bl_k, kv), ("V", bl_v, vv)):
                if args.center:
                    bl = bl - offsets_rot[layer][None, :]
                for i in range(bl.shape[0]):
                    x = bl[i]
                    if var["kind"] == "uniform":
                        recon, d = quantize_uniform(x, var["fit"])
                    else:
                        recon, d = quantize_lloyd(x, var["levels"], var["fit"])
                    tot_err += float(np.mean((recon - x) ** 2))
                    tot_n += 1
                    used = np.unique(np.rint(recon / d) if d > 0 else recon)
                    st = (recon.max() - recon.min()) / max(len(used) - 1, 1) if d > 0 else 0.0
                    if st > 0:
                        us.append(float(x.std()) / st)
        mse = tot_err / max(tot_n, 1)
        if base is None:
            base = mse
        db = 10.0 * math.log10(mse / base) if mse > 0 and base > 0 else float("nan")
        umed = float(np.median(us)) if us else float("nan")
        rows.append((name, mse, db, umed))
        print(f"{name:34s} MSE {mse:.6e}  {db:+6.2f} dB  u(Median) {umed:7.2f}")

    print()
    print(f"Zentriert: {'ja' if args.center else 'nein'}   Werte je Block: {KQ4}")
    print("u = sigma(Block) / effektiver Schritt (Schritte je sigma; groesser = feinere Aufloesung)")


if __name__ == "__main__":
    main()
