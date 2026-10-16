"""Offline tests for tools/quant_quality.py's kvstats aggregation (no engine, no GPU).

Run directly:  python tools/test_kvstats.py
or with pytest: pytest tools/test_kvstats.py
"""
import math
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant_quality import KVSTATS_MAGIC, kvstats_aggregate

GROUP = struct.Struct("<5d 24d 32Q 256f")


def fold(xs):
    """A KvStatsAccum-equivalent group body for one list of values (single block position 0)."""
    n = len(xs)
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    s4 = sum(x ** 4 for x in xs)
    mx = max(abs(x) for x in xs)
    bs = [s1] + [0.0] * 7
    bsq = [s2] + [0.0] * 7
    bx = [mx] + [0.0] * 7
    hist = [0] * 32
    for x in xs:
        ax = abs(x)
        if ax <= 0.0:
            hb = 0
        elif ax >= 1e30:
            hb = 31
        else:
            l = math.log2(ax)
            hb = 0 if l < -8.0 else (31 if l >= 24.0 else math.floor(l) + 8)  # floor, not int()
        hist[hb] += 1
    ch = [0.0] * 256
    for i, x in enumerate(xs[:256]):
        ch[i % 256] = max(ch[i % 256], abs(x))
    return (float(n), s1, s2, s4, mx, *bs, *bsq, *bx, *hist, *ch)


def gidx(layer, head, is_v, tap):
    return ((layer * 8 + head) * 2 + is_v) * 2 + tap


def synthetic(n_tokens=64, seed=1):
    """Sections of N(0,1) groups: pre and post identical (norm conserved exactly)."""
    rnd = random.Random(seed)
    buf = bytearray(KVSTATS_MAGIC)
    for w in range(n_tokens // 16):
        pos0, n = w * 16, 16
        present = []
        for head in range(2):
            for is_v in range(2):
                xs = [rnd.gauss(0.0, 1.0) for _ in range(n * 256)]
                for tap in (0, 1):
                    present.append((gidx(0, head, is_v, tap), fold(xs)))
        buf += struct.pack("<q q I", pos0, n, len(present))
        for gi, body in present:
            buf += struct.pack("<H", gi) + GROUP.pack(*body)
    return bytes(buf), n_tokens


def test_gaussian_kurtosis_near_zero():
    raw, n = synthetic()
    agg = kvstats_aggregate(raw, n)
    assert len(agg["groups"]) == 2 * 2 * 2, len(agg["groups"])
    for name, g in agg["groups"].items():
        assert abs(g["kurtosis_excess"]) < 0.5, (name, g["kurtosis_excess"])
        assert abs(g["rms"] - 1.0) < 0.05, (name, g["rms"])
        assert abs(g["mean"]) < 0.05, (name, g["mean"])
        assert g["crest"] > 2.0, (name, g["crest"])


def test_fwht_self_check_exact():
    raw, n = synthetic()
    agg = kvstats_aggregate(raw, n)
    assert agg["checks"]["fwht_norm_matched"] == 2 * 2, agg["checks"]["fwht_norm_matched"]
    assert agg["checks"]["fwht_norm_maxreldiff"] == 0.0, agg["checks"]["fwht_norm_maxreldiff"]
    assert agg["checks"]["covered_tokens"] == n, agg["checks"]["covered_tokens"]


def test_unclipped_and_exceed():
    raw, n = synthetic(n_tokens=32)
    agg = kvstats_aggregate(raw, n)
    for name, g in agg["groups"].items():
        u = [g["unclipped_kappa"][f"{0.80 + 0.02 * i:.2f}"] for i in range(11)]
        assert u[-1] == 1.0, (name, u[-1])
        assert all(a <= b for a, b in zip(u, u[1:])), (name, u)
        e = g["exceed_rms"]
        assert set(e) == {"2", "3", "4", "5"}
        assert 0.02 < e["2"] < 0.08 and e["5"] < 0.001, (name, e["2"], e["5"])
        assert 0.0 < g["channels"]["top1_mass"] < 0.2, (name, g["channels"]["top1_mass"])


def test_bad_magic():
    try:
        kvstats_aggregate(b"NOPE1234" + b"\x00" * 24, 1)
    except SystemExit:
        return
    raise AssertionError("bad magic must SystemExit")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


if __name__ == "__main__":
    bad = 0
    for t in TESTS:
        try:
            t()
            print(f"ok {t.__name__}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"{len(TESTS) - bad}/{len(TESTS)} passed")
    sys.exit(1 if bad else 0)
