#!/usr/bin/env python
"""Where in the network is the KV cache hard to compress? -- measured per QSA layer, host-side.

The dump `cells-cwe4k.bin` holds the rotated K/V rows of the batched prompt path per LAYER (the
QSA layers 3, 7, ... 47), 32,746 tokens each, 2 KV heads, head_dim 256.  So the codec side of the
layer question can be answered on our own cells: does the *compressibility* differ between layers,
and does any of the three levers (Lloyd shape, shear/offset, calibration) differ between layers?

FOUR QUESTIONS, FIXED BEFORE THE FIRST RUN (exploratory, no gates -- this is a screening):

  F1  spread of the normalised error rate (rmse / rms of the layer) over the 12 layers:
      a factor > 1.5 between the easiest and the hardest layer means the layers are NOT
      interchangeable, and a mixed-precision (per-layer bit width) question has a basis.
  F2  monotone trend over the layer index (Spearman |rho| >= 0.7): a rule like "the deep layers
      are cheaper" would be usable without a table.
  F3  spread of the shear gain > 0.5 dB: the per-block offset is a per-layer lever (the shear is
      not a product path, but its magnitude tells where the blocks are skewed).
  F4  spread of the calibration gain > 0.5 dB: the channel-mean table is a per-layer lever.

THE LIMIT, named before the numbers: block MSE measures where the CODEC makes its error, not where
the error HURTS.  A layer can be hard to compress and still be unimportant for the output.  The
per-layer *sensitivity* is a different question and was already measured on the GPU
(`tools/kv_bisect_layers.py`, 2026-11-08: neither half of the layers carries the tail).  Nothing
here is a claim about quality.

The dump is NOT ordered layer by layer -- the records cycle through the 12 layers (3, 7, ... 47, 3,
7, ...).  A first version assumed grouping and produced one partial row per run of layers; that
output is discarded.  The offsets (the calibration table) need the whole layer, so this tool reads
the dump TWICE: pass 1 accumulates per-head channel means and the uncalibrated codec errors, pass 2
repeats the codec errors on the centred data.  One pass would have to hold all 12 layers in memory
(~6 GB).

Everything is measured on the rotated data (the domain the codec quantises in), `d` as fp16, and
the offset is a table per head and channel, estimated from this dump (in-sample), i.e. the same
thing `--center` does for `center-table.bin`.  Blocks are 32 values along head_dim, as in the
kernel; every `--stride`-th block of each layer is used, counted continuously across the records.

Usage:
  python tools/kv_layer_profile.py --dump <cells-cwe4k.bin> --stride 8 [--json out.json]
"""
import argparse
import json
import math
import struct

import numpy as np

KQ4 = 32
HDR = struct.Struct("<iiqiiii")
MAGIC = 0x51345042
L16 = np.array([-87, -66, -52, -40, -30, -21, -12, -4, 4, 12, 21, 30, 40, 52, 66, 87],
               dtype=np.float64)
B16 = np.array([-76.5, -59.0, -46.0, -35.0, -25.5, -16.5, -8.0, 0.0, 8.0, 16.5, 25.5, 35.0,
                46.0, 59.0, 76.5])


def iter_records(path):
    """Yield (layer, p0, T, nkv, hd, K[T,nkv,hd], V[T,nkv,hd]) one record at a time."""
    with open(path, "rb") as f:
        while True:
            h = f.read(HDR.size)
            if len(h) < HDR.size:
                break
            magic, layer, p0, T, nkv, hd, _pad = HDR.unpack(h)
            if magic != MAGIC:
                raise SystemExit("bad magic")
            n = T * nkv * hd
            raw = f.read(n * 8)
            if len(raw) < n * 8:
                raise SystemExit("truncated record")
            k = np.frombuffer(raw[: n * 4], dtype="<f4").reshape(T, nkv, hd)
            v = np.frombuffer(raw[n * 4:], dtype="<f4").reshape(T, nkv, hd)
            yield layer, p0, T, nkv, hd, k, v


def assign(x, d):
    with np.errstate(divide="ignore", invalid="ignore"):
        idd = np.where(d != 0.0, 1.0 / d, 0.0)
    return np.clip(np.trunc(x * idd[:, None] + 8.5), 0.0, 15.0)


def uniform_ls(x):
    """The q4u32 writer: max init, one LS pass, codes reassigned, d stored as fp16."""
    amax = np.max(np.abs(x), axis=1)
    idx = np.argmax(np.abs(x), axis=1)
    mval = x[np.arange(x.shape[0]), idx]
    d0 = np.where(amax > 0.0, mval / -8.0, 0.0)
    code = assign(x, d0)
    c = code - 8.0
    den = np.sum(c * c, axis=1)
    num = np.sum(c * x, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), d0)
    d = np.where((np.sign(d) == np.sign(d0)) & (d != 0.0), d, d0)
    d = np.asarray(d, dtype=np.float16).astype(np.float64)
    code = assign(x, d)
    return (code - 8.0) * d[:, None]


def lloyd_ls(x):
    """The product codec's shape (16 Lloyd levels) with the fitted step."""
    amax = np.max(np.abs(x), axis=1)
    d = np.where(amax > 0.0, amax / 87.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = x / np.where(d > 0.0, d, 1.0)[:, None]
    lv = L16[np.searchsorted(B16, y, side="right")]
    den = np.sum(lv * lv, axis=1)
    num = np.sum(lv * x, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_ls = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), d)
    d = np.where(d_ls > 0.0, d_ls, d)
    d = np.asarray(d, dtype=np.float16).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = x / np.where(d > 0.0, d, 1.0)[:, None]
    lv = L16[np.searchsorted(B16, y, side="right")]
    return lv * d[:, None]


def shear(x, iters=6):
    """The best affine map per block: fitted step AND fitted zero point (the shear's ceiling)."""
    lo = np.min(x, axis=1)
    hi = np.max(x, axis=1)
    s = np.where((hi - lo) > 0, (hi - lo) / 15.0, 1.0)
    o = lo.copy()
    for _ in range(iters):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(s > 0, 1.0 / s, 0.0)
        code = np.clip(np.rint((x - o[:, None]) * inv[:, None]), 0.0, 15.0)
        cm, xm = code.mean(axis=1), x.mean(axis=1)
        dc, dx = code - cm[:, None], x - xm[:, None]
        den = np.sum(dc * dc, axis=1)
        num = np.sum(dc * dx, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_new = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), s)
        s = np.where(s_new > 0, s_new, s)
        o = xm - s * cm
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(s > 0, 1.0 / s, 0.0)
    code = np.clip(np.rint((x - o[:, None]) * inv[:, None]), 0.0, 15.0)
    return o[:, None] + s[:, None] * code


class Acc:
    """Streaming accumulator: enough for the pooled moments and the pooled codec MSEs."""

    def __init__(self, nkv, hd):
        self.nkv, self.hd = nkv, hd
        self.n = 0                      # rows seen
        self.s1 = self.s2 = self.s3 = self.s4 = 0.0
        self.head_sum = np.zeros((nkv, hd))
        self.n_head = 0                 # rows per head
        self.amax_sum = 0.0
        self.nblk = 0
        self.err = {"uniform": 0.0, "lloyd": 0.0, "shear": 0.0}
        self.nelem = 0
        self.nblk_used = 0

    def add(self, x, stride, counters, key):
        n, hd = x.shape
        head = np.arange(n) % self.nkv
        for h in range(self.nkv):
            self.head_sum[h] += x[head == h].sum(axis=0)
        self.n += n
        self.n_head += n // self.nkv
        self.s1 += float(x.sum())
        self.s2 += float((x * x).sum())
        self.s3 += float((x ** 3).sum())
        self.s4 += float((x ** 4).sum())

        bl = x.reshape(-1, KQ4)
        nb = bl.shape[0]
        c0 = counters.get(key, 0)
        counters[key] = c0 + nb
        if stride > 1:
            keep = ((c0 + np.arange(nb)) % stride) == 0
            bl = bl[keep]
        if bl.shape[0] == 0:
            return
        self.amax_sum += float(np.max(np.abs(bl), axis=1).sum())
        self.nblk += bl.shape[0]
        self.nblk_used += bl.shape[0]
        self.err["uniform"] += float(((uniform_ls(bl) - bl) ** 2).sum())
        self.err["lloyd"] += float(((lloyd_ls(bl) - bl) ** 2).sum())
        self.err["shear"] += float(((shear(bl) - bl) ** 2).sum())
        self.nelem += bl.size

    # -- the pooled statistics -------------------------------------------------
    def rms(self):
        return math.sqrt(self.s2 / (self.n * self.hd))

    def kurtosis(self):
        ne = self.n * self.hd
        m = self.s1 / ne
        m2 = self.s2 / ne - m * m
        m4 = (self.s4 / ne - 4.0 * m * self.s3 / ne + 6.0 * m * m * self.s2 / ne
              - 3.0 * m ** 4)
        return m4 / (m2 * m2) - 3.0 if m2 > 0 else float("nan")

    def crest(self):
        r = self.rms()
        return (self.amax_sum / self.nblk) / r if r > 0 else float("nan")

    def mse(self, name):
        return self.err[name] / self.nelem

    def rmse_rate(self):
        r = self.rms()
        return math.sqrt(self.mse("uniform")) / r if r > 0 else float("nan")


def run(path, stride):
    acc, counters, offsets = {}, {}, {}
    for layer, p0, T, nkv, hd, k, v in iter_records(path):
        for side, arr in (("K", k), ("V", v)):
            key = (layer, side)
            a = acc.setdefault(key, Acc(nkv, hd))
            x = np.asarray(arr, dtype=np.float64).reshape(-1, hd)
            a.add(x, stride, counters, key)
    for (layer, side), a in acc.items():
        offsets[(layer, side)] = a.head_sum / max(a.n_head, 1)

    centred, counters2 = {}, {}
    for layer, p0, T, nkv, hd, k, v in iter_records(path):
        for side, arr in (("K", k), ("V", v)):
            key = (layer, side)
            a = centred.setdefault(key, Acc(nkv, hd))
            x = np.asarray(arr, dtype=np.float64).reshape(-1, hd)
            off = offsets[key]
            a.add(x - off[np.arange(x.shape[0]) % nkv], stride, counters2, key)
    return acc, centred, offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    acc, centred, offsets = run(args.dump, args.stride)

    layers = sorted({l for l, _s in acc})
    print(f"records: {len(layers)} layers x 2 sides, stride {args.stride}, "
          f"{acc[(layers[0], 'K')].n} rows per side per layer")
    print()
    print(f"{'layer':>5} | {'K rmse/rms':>10} {'K kurt':>7} {'K crest':>7} {'K off/rms':>9} "
          f"{'K lloyd':>8} {'K shear':>8} {'K calib':>8} | {'V rmse/rms':>10} {'V kurt':>7} "
          f"{'V crest':>7} {'V off/rms':>9} {'V lloyd':>8} {'V shear':>8} {'V calib':>8}")
    print("-" * 172)
    rows = []
    for layer in layers:
        entry = {"layer": layer}
        for side in ("K", "V"):
            a, c = acc[(layer, side)], centred[(layer, side)]
            off = offsets[(layer, side)]
            entry[side] = {
                "rows": a.n, "blocks": a.nblk_used,
                "rms": a.rms(), "kurtosis": a.kurtosis(), "crest": a.crest(),
                "rmse_rate": a.rmse_rate(),
                "offset_rms_over_rms": float(np.sqrt(np.mean(off ** 2)) / a.rms()),
                "lloyd_gain_db": 10.0 * math.log10(a.mse("uniform") / a.mse("lloyd")),
                "shear_gain_db": 10.0 * math.log10(a.mse("uniform") / a.mse("shear")),
                "calib_gain_uniform_db": 10.0 * math.log10(a.mse("uniform") / c.mse("uniform")),
                "calib_gain_lloyd_db": 10.0 * math.log10(a.mse("lloyd") / c.mse("lloyd")),
                "mse_uniform": a.mse("uniform"),
            }
        rows.append(entry)
        kk, vv = entry["K"], entry["V"]
        print(f"{layer:5d} | {kk['rmse_rate']:10.4f} {kk['kurtosis']:7.2f} {kk['crest']:7.3f} "
              f"{kk['offset_rms_over_rms']:9.4f} {kk['lloyd_gain_db']:8.3f} "
              f"{kk['shear_gain_db']:8.3f} {kk['calib_gain_uniform_db']:8.3f} | "
              f"{vv['rmse_rate']:10.4f} {vv['kurtosis']:7.2f} {vv['crest']:7.3f} "
              f"{vv['offset_rms_over_rms']:9.4f} {vv['lloyd_gain_db']:8.3f} "
              f"{vv['shear_gain_db']:8.3f} {vv['calib_gain_uniform_db']:8.3f}")

    def spread(key, side):
        vals = [r[side][key] for r in rows]
        return min(vals), max(vals), (max(vals) / min(vals) if min(vals) > 0 else float("nan"))

    idx = np.array([r["layer"] for r in rows], dtype=float)

    def rho(key, side):
        val = np.array([r[side][key] for r in rows], dtype=float)
        return float(np.corrcoef(np.argsort(np.argsort(idx)),
                                 np.argsort(np.argsort(val)))[0, 1])

    print()
    print("F1/F2  normierte Fehlerrate (rmse/rms) -- die Komprimierbarkeit bei festen 4,5 bit:")
    for side in ("K", "V"):
        lo, hi, fac = spread("rmse_rate", side)
        bits = 20.0 * math.log10(fac) / 6.02
        print(f"  {side}: {lo:.4f} .. {hi:.4f}  Faktor {fac:.2f} = {bits:.2f} bit   "
              f"(Spearman gegen Layer-Index {rho('rmse_rate', side):+.2f})")
    print("F3  Shear-Gewinn (dB) -- wie schief die Bloecke liegen:")
    for side in ("K", "V"):
        lo, hi, _ = spread("shear_gain_db", side)
        print(f"  {side}: {lo:+.3f} .. {hi:+.3f}  Spannweite {hi - lo:.2f} dB   "
              f"Spearman {rho('shear_gain_db', side):+.2f}")
    print("F4  Kalibrierungs-Gewinn, uniformes Gitter (dB):")
    for side in ("K", "V"):
        lo, hi, _ = spread("calib_gain_uniform_db", side)
        print(f"  {side}: {lo:+.3f} .. {hi:+.3f}  Spannweite {hi - lo:.2f} dB   "
              f"Spearman {rho('calib_gain_uniform_db', side):+.2f}")
    print("Nebenwerte:")
    for side in ("K", "V"):
        lo, hi, fac = spread("crest", side)
        print(f"  {side} crest (mittleres Block-absmax / rms): {lo:.3f} .. {hi:.3f}  "
              f"Faktor {fac:.2f}")
    for side in ("K", "V"):
        lo, hi, _ = spread("lloyd_gain_db", side)
        print(f"  {side} Lloyd-Form gegen uniformes Gitter: {lo:+.3f} .. {hi:+.3f} dB")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"stride": args.stride, "layers": rows}, f, indent=1)
        print(f"\nJSON: {args.json}")


if __name__ == "__main__":
    main()
