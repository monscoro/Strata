#!/usr/bin/env python
"""The `q4u32` writer, pass by pass, and the UltraQuant scale rule -- measured on the host.

Two questions, one tool, no GPU and no build:

  (1) THE SECOND PASS.  The engine's `q4u32` writer (`q4u_ls_group`, kv_q4.cu:238) starts at
      `d = mval / -8`, assigns the codes, solves ONE least-squares step for `d`, and reassigns the
      codes with the new step -- and stops there.  The shear (`q4u_group`, kShearPasses = 2) runs two
      alternating passes.  A second pass costs the writer only butterflies (the reader and the byte
      layout are untouched), so the question is what it buys.

  (2) THE ULTRQUANT SCALE.  UltraQuant (arXiv:2606.20474v3, Sec. 5.2/5.3) stores one UE8M0 scale
      per group of 32: `s = 2^round(log2(c * m))` with `m` the group's absmax and `c = 0.156` a
      single model-agnostic constant.  Two of its claims land on our path: the per-block absmax is
      the load-bearing part (the codebook barely matters -- App. B.1), and the scale is a power of
      two so the dequantisation is an exponent shift.  Our `q4u32` already has the per-block absmax
      (as the init of a fitted step) but stores `d` as fp16.  So the only thing left to borrow is the
      SNAP: does forcing `d` onto 2^k cost enough MSE to matter, and what would it buy?
      It would buy one byte per 18-byte block (an exponent instead of an fp16), i.e. 17 B / 4.25 bit
      per value.

DECISION, fixed before the first run (so the measurement can only answer it):

  gain >= 1.0 dB vs. today's writer   -> build: a second pass is worth an end-to-end arm
  0.3 dB <= gain < 1.0 dB             -> build only if it also buys a format gain (byte or reader)
  gain < 0.3 dB                       -> no build; document as a null result

  The same three bands apply to the snap, with the sign reversed: a snap loss below 0.3 dB is
  "free enough to try if the byte is worth something", 0.3-1.0 dB is "only with a measured reader
  gain", above 1.0 dB is "no".

Everything is read from the engine's own cell dump (the rotated K/V rows of the batched prompt
path), so the numbers describe the engine's data.  `d` is rounded to fp16 wherever the engine would
store it; the fp64 row is only there to join the older `uniform+LS` reading of kv_affine_price.py.

Usage:
  python tools/kv_q4_passes.py --dump <cells-cwe4k.bin> --stride 8
"""
import argparse
import json
import math
import struct

import numpy as np

KQ4 = 32
# ggml's q4_0 grid: code 0 sits on the block's absmax (d carries the sign), code 15 seven steps
# the other way.  So the grid is [-8d, 7d] and the absmax is always code 0.
HDR = struct.Struct("<iiqiiii")
MAGIC = 0x51345042


def iter_dump(path, stride=1):
    """Stream the dump: one record at a time, K/V as float64 blocks of 32 along the head dimension.

    The file is 1.6 GB; read_dump() in kv_codec_mse.py holds all of it as float64 (3.2 GB).  This
    reader yields (side, blocks) and lets the caller accumulate, so the footprint stays at one
    record.
    """
    with open(path, "rb") as f:
        rec = 0
        while True:
            head = f.read(HDR.size)
            if len(head) < HDR.size:
                break
            magic, layer, p0, T, nkv, hd, _pad = HDR.unpack(head)
            if magic != MAGIC:
                raise SystemExit(f"bad magic {magic:#x} at record {rec}")
            n = T * nkv * hd
            raw = f.read(n * 8)  # K then V, both float32
            if len(raw) < n * 8:
                raise SystemExit(f"truncated record {rec}")
            k = np.frombuffer(raw[: n * 4], dtype="<f4").reshape(T, nkv, hd)
            v = np.frombuffer(raw[n * 4 :], dtype="<f4").reshape(T, nkv, hd)
            nblk = hd // KQ4
            for side, arr in (("K", k), ("V", v)):
                bl = np.asarray(arr, dtype=np.float64).reshape(T * nkv, nblk, KQ4)
                bl = bl.reshape(-1, KQ4)
                if stride > 1:
                    bl = bl[::stride]
                yield side, bl
            rec += 1
            del raw, k, v


def assign(x, d):
    """The engine's code assignment: code = trunc(x/d + 8.5), clipped to 0..15 (kv_q4.cu)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        idd = np.where(d != 0.0, 1.0 / d, 0.0)
    return np.clip(np.trunc(x * idd[:, None] + 8.5), 0.0, 15.0)


def ls_step(x, code, d0):
    """One least-squares step for d given the codes, with the engine's sign guard."""
    c = code - 8.0
    den = np.sum(c * c, axis=1)
    num = np.sum(c * x, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), d0)
    same = (np.sign(d) == np.sign(d0)) & (d != 0.0)
    return np.where(same, d, d0)


def init_max(x):
    """q4u32's init: d = mval / -8 (the block's signed absmax), amax == 0 stays 0."""
    amax = np.max(np.abs(x), axis=1)
    idx = np.argmax(np.abs(x), axis=1)
    mval = x[np.arange(x.shape[0]), idx]
    d = np.where(amax > 0.0, mval / -8.0, 0.0)
    return d


def to_f16(d):
    """The engine stores d as fp16 (f16_from_f32); every pass reads it back that way."""
    return np.asarray(d, dtype=np.float16).astype(np.float64)


def recon(x, code, d):
    return (code - 8.0) * d[:, None]


def mse(x, code, d):
    return float(np.mean((recon(x, code, d) - x) ** 2))


def snap_pow2(d, mode):
    """The UltraQuant UE8M0 snap: s = 2^round(log2|d|), sign kept.  mode: round | floor | ceil."""
    a = np.abs(d)
    out = np.zeros_like(d)
    nz = a > 0.0
    if not np.any(nz):
        return out
    lg = np.log2(a[nz])
    e = {"round": np.rint(lg), "floor": np.floor(lg), "ceil": np.ceil(lg)}[mode]
    out[nz] = np.sign(d[nz]) * np.exp2(e)
    return out


def snap_pick(x, code, d):
    """The snap a writer would ship if it may choose: the better of 2^floor and 2^ceil.

    Two candidate steps, one code assignment each and one MSE comparison per block -- still only
    writer work, and it halves the price of a pure round()."""
    lo = snap_pow2(d, "floor")
    hi = snap_pow2(d, "ceil")
    c_lo = assign(x, lo)
    c_hi = assign(x, hi)
    e_lo = np.mean((recon(x, c_lo, lo) - x) ** 2, axis=1)
    e_hi = np.mean((recon(x, c_hi, hi) - x) ** 2, axis=1)
    take_lo = e_lo <= e_hi
    d_out = np.where(take_lo, lo, hi)
    code_out = np.where(take_lo[:, None], c_lo, c_hi)
    return code_out, d_out


def variant_curves(x):
    """All variants on one batch of blocks.  Returns {name: (code, d)} plus a few raw d rows."""
    out = {}
    d0 = init_max(x)
    code = assign(x, d0)

    # p1-f64: the older reading (no fp16 round trip), the join to uniform+LS = 8.693901e-03
    d1f = ls_step(x, code, d0)
    code1f = assign(x, d1f)
    out["p1-f64"] = (code1f, d1f)

    # p1: today's writer -- one LS pass, codes reassigned, d stored as fp16
    d1 = to_f16(d1f)
    code1 = assign(x, d1)
    out["p1"] = (code1, d1)

    # p2 / p3: one more (two more) alternating pass(es) on the stored fp16 step
    d2 = to_f16(ls_step(x, code1, d1))
    code2 = assign(x, d2)
    out["p2"] = (code2, d2)
    d3 = to_f16(ls_step(x, code2, d2))
    out["p3"] = (assign(x, d3), d3)

    # the snap on top of p1 and p2 (round), and the two-candidate pick
    d1s = snap_pow2(d1, "round")
    out["p1-snap"] = (assign(x, d1s), d1s)
    d2s = snap_pow2(d2, "round")
    out["p2-snap"] = (assign(x, d2s), d2s)
    out["p1-snap2"] = snap_pick(x, code1, d1)
    out["p2-snap2"] = snap_pick(x, code2, d2)

    # UltraQuant's own rule, for the record: s = c * amax, with and without the snap.  c = 0.156 is
    # tuned to the E2M1 grid whose widest codepoint is 6; our grid's widest is 8, so the grid's own
    # constant is 1/8 = 0.125.  Both are here so the constant's transfer can be read off directly.
    amax = np.max(np.abs(x), axis=1)
    du = np.where(amax > 0.0, 0.156 * amax, 0.0)
    out["uq156"] = (assign(x, du), du)
    dus = snap_pow2(du, "round")
    out["uq156-snap"] = (assign(x, dus), dus)
    dg = np.where(amax > 0.0, 0.125 * amax, 0.0)
    out["uq125"] = (assign(x, dg), dg)
    dgs = snap_pow2(dg, "round")
    out["uq125-snap"] = (assign(x, dgs), dgs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--stride", type=int, default=8, help="use every Nth block (blocks stay whole)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    acc = {}
    nblk = 0
    for side, bl in iter_dump(args.dump, args.stride):
        if bl.size == 0:
            continue
        nblk += bl.shape[0]
        for name, (code, d) in variant_curves(bl).items():
            s = acc.setdefault(name, {"K": [0.0, 0], "V": [0.0, 0]})
            e = float(np.sum((recon(bl, code, d) - bl) ** 2))
            s[side][0] += e
            s[side][1] += bl.size

    def mse_of(name, side=None):
        if side:
            return acc[name][side][0] / max(acc[name][side][1], 1)
        return 0.5 * (mse_of(name, "K") + mse_of(name, "V"))

    base = mse_of("p1")
    print(f"dump: {args.dump}")
    print(f"blocks: {nblk} (stride {args.stride}), d as the engine stores it (fp16)")
    print()
    print(f"{'Variante':12s} {'MSE':>13s} {'K':>13s} {'V':>13s} {'dB gegen p1':>13s}")
    order = ["p1-f64", "p1", "p2", "p3", "p1-snap", "p2-snap", "p1-snap2", "p2-snap2",
             "uq156", "uq156-snap", "uq125", "uq125-snap"]
    for name in order:
        m = mse_of(name)
        db = 10.0 * math.log10(m / base) if m > 0 and base > 0 else float("nan")
        print(f"{name:12s} {m:13.6e} {mse_of(name, 'K'):13.6e} {mse_of(name, 'V'):13.6e} {db:+13.3f}")

    print()
    print("Die Fragen, in dB (positiv = besser als der heutige Writer p1):")
    for name in ("p2", "p3", "p1-snap2", "p2-snap2", "p1-snap", "p2-snap"):
        m = mse_of(name)
        print(f"  {name:10s} {10.0 * math.log10(base / m):+6.3f} dB"
              f"   {'Gewinn' if mse_of(name) < base else 'Verlust'}")

    # the 1.16 dB shear / 0.0014 nats bridge from the protocol, for scale
    print()
    print("Zum Massstab: der ganze Shear (1.29 dB Block-MSE) war end-to-end ~0,0014 nats wert;")
    print("die Paritaetsaufloesung ist 0,00200 nats (4K) / 0,00254 nats (128K).")
    print("1 dB Block-MSE entspricht damit grob 0,0011 nats.")

    if args.json:
        out = {
            "blocks": nblk,
            "stride": args.stride,
            "mse": {k: mse_of(k) for k in order},
            "mse_per_side": {k: {"K": mse_of(k, "K"), "V": mse_of(k, "V")} for k in order},
            "db_vs_p1": {k: 10.0 * math.log10(mse_of(k) / base) for k in order},
            "gain_db_vs_p1": {k: 10.0 * math.log10(base / mse_of(k)) for k in order},
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"\nJSON: {args.json}")


if __name__ == "__main__":
    main()
