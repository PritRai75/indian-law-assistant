"""Tests for app/citations.py on the committed index. The strings are the forms the first real run
showed (session 14) plus edge cases; none comes from a golden question."""
import json
from pathlib import Path

import pytest

from app import citations as C

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "index" / "chunks.json").exists(), reason="index not built yet")


@pytest.fixture(scope="module")
def prov():
    chunks = json.loads((ROOT / "index" / "chunks.json").read_text(encoding="utf-8"))
    return {p.lower(): p for c in chunks for p in c["provision_ids"]}


def ids(text, prov):
    return C.read_citation(text, prov).ids


def test_one_id_in_any_spelling(prov):
    assert ids("§ 1026.52(b)(1)(ii)(a)", prov) == ["1026.52(b)(1)(ii)(A)"]      # index spelling back
    assert ids("12 CFR 1026.12-1", prov) == ["1026.12-1"]
    assert ids("comment 1005.6(b)-1", prov) == ["1005.6(b)-1"]


def test_packed_ids(prov):
    assert ids("1026.13(c)(1), § 1026.13(c)(2)", prov) == ["1026.13(c)(1)", "1026.13(c)(2)"]
    assert ids("1005.11(c)(1); § 1005.11(c)(2)(i)", prov) == ["1005.11(c)(1)", "1005.11(c)(2)(i)"]
    assert ids("1005.6(b)(1), 1005.6(b)(1)", prov) == ["1005.6(b)(1)"]         # no repeats


def test_comment_subpoints_are_the_comment(prov):
    assert ids("1005.12(a)-5.iv.B", prov) == ["1005.12(a)-5"]
    assert ids("comment 1005.6(b)(1)-3.ii", prov) == ["1005.6(b)(1)-3"]


def test_ranges(prov):
    assert ids("1005.11(c)(2)(iii)-(iv)", prov) == ["1005.11(c)(2)(iii)", "1005.11(c)(2)(iv)"]
    assert ids("1005.6(b)(1)-(3)", prov) == ["1005.6(b)(1)", "1005.6(b)(2)", "1005.6(b)(3)"]
    assert ids("1026.52(b)(1)(i) to (ii)", prov) == ["1026.52(b)(1)(i)", "1026.52(b)(1)(ii)"]
    assert ids("1026.13(c)(1) through (c)(2)", prov) == ["1026.13(c)(1)"]      # not a one-label range


def test_sequence_kinds():
    has_ii = lambda label: True
    assert C._sequence("i", "iv", has_ii) == ["i", "ii", "iii", "iv"]
    assert C._sequence("a", "c", has_ii) == ["a", "b", "c"]
    assert C._sequence("i", "iii", lambda label: False) == ["i", "ii", "iii"]   # (iii) can only be roman
    assert C._sequence("i", "k", lambda label: False) == ["i", "j", "k"]       # no (ii) there: letters
    assert C._sequence("i", "v", lambda label: False) == ["i", "v"]            # letters i..v: 14, too long
    assert C._sequence("1", "40", has_ii) == ["1", "40"]                       # too long: the two ends
    assert C._sequence("3", "1", has_ii) == ["3", "1"]                         # backwards: the two ends


def test_ids_inside_a_phrase(prov):
    assert ids("see 1026.9(c)(2)(i) and the notice rules", prov) == ["1026.9(c)(2)(i)"]


def test_what_is_not_an_index_id_is_sorted(prov):
    r = C.read_citation("1005.11", prov)
    assert r.ids == [] and r.too_broad == ["1005.11"] and r.problems == ["1005.11"]
    r = C.read_citation("1026.99(z)", prov)
    assert r.not_found == ["1026.99(z)"]
    r = C.read_citation("comment 12(b)-1", prov)                               # exists in both parts
    assert r.no_part == ["comment 12(b)-1"] and r.ids == []
    r = C.read_citation("visa-rules", prov)
    assert r.unreadable and r.problems == ["visa-rules"]


def test_read_all_and_marker(prov):
    found, readings = C.read_all(["1005.6(b)(1)", "1005.6(b)(1); 1005.6(b)-1", "1005.11"], prov)
    assert found == ["1005.6(b)(1)", "1005.6(b)-1"] and readings[2].too_broad == ["1005.11"]
    assert C.marker("1005.6(b)(1)") == "§ 1005.6(b)(1)" and C.marker("1005.6(b)-1") == "comment 1005.6(b)-1"
    assert C.marker("1026.12-1") == "comment 1026.12-1" and C.marker("1005.2") == "§ 1005.2"
