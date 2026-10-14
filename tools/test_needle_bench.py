"""Offline tests for tools/needle_bench.py's pure helpers (no server, no GPU).

Run directly:  python tools/test_needle_bench.py
or with pytest: pytest tools/test_needle_bench.py
"""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from needle_bench import grade, insert_at, needle_value, place_needles


def test_insert_at_line_start():
    text = "aaa\nbbb\nccc\nddd\n"
    out, pos = insert_at(text, 50, "NEEDLE\n")
    assert out[pos:pos + 7] == "NEEDLE\n"
    assert out[:pos].endswith("\n") or pos == 0
    assert "aaa\nbbb\nccc\nddd\n" in out.replace("NEEDLE\n", "")


def test_place_needles_depth_order():
    text = "\n".join(f"line{i}" for i in range(200)) + "\n"
    items = [(10.0, "FIRST\n"), (50.0, "SECOND\n"), (90.0, "THIRD\n")]
    out, pos = place_needles(text, items)
    assert len(pos) == 3
    assert out.index("FIRST\n") < out.index("SECOND\n") < out.index("THIRD\n")
    # ascending input works too (insertion runs deepest first internally)
    out2, _ = place_needles(text, list(reversed(items)))
    assert out2.index("FIRST\n") < out2.index("SECOND\n") < out2.index("THIRD\n")


def test_place_needles_all_present_once():
    text = "x\ny\nz\n" * 50
    items = [(20.0, "A\n"), (80.0, "B\n")]
    out, _ = place_needles(text, items)
    assert out.count("A\n") == 1 and out.count("B\n") == 1


def test_grade_substring():
    assert grade("the code is amber-falcon-123 ok", ["amber-falcon-123"]) == {"amber-falcon-123": True}
    assert grade("nope", ["amber-falcon-123"]) == {"amber-falcon-123": False}
    r = grade("a 111 and b 222", ["111", "222", "333"])
    assert r == {"111": True, "222": True, "333": False}


def test_value_formats():
    rnd = random.Random(7)
    for _ in range(20):
        assert re.fullmatch(r"[a-z]+-[a-z]+-\d{3}", needle_value(rnd, "code"))
        assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                            needle_value(rnd, "uuid"))
        assert re.fullmatch(r"\d{7}", needle_value(rnd, "number"))


def test_values_differ_per_draw():
    rnd = random.Random(7)
    assert len({needle_value(rnd, "code") for _ in range(10)}) > 1


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
