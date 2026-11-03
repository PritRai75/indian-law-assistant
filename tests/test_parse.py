"""Tests for ingest/parse.py, run on the committed eCFR XML.

The counts below are pinned to the corpus as of 2026-10-23. If the corpus is
re-downloaded, they change on purpose: update them in the same commit.
"""

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import pytest

from ingest import parse as P

REPO = P.REPO


@pytest.fixture(scope="module")
def records():
    recs, _ = P.parse_all()
    return recs


@pytest.fixture(scope="module")
def by_id(records):
    return {r["id"]: r for r in records}


def ids_of(records):
    return [r["id"] for r in records]


# ------------------------------------------------------------------ overall shape

def test_counts_pinned_to_corpus(records):
    c = Counter((r["reg"], r["kind"]) for r in records)
    assert c == {("E", "rule"): 625, ("E", "commentary"): 470,
                 ("Z", "rule"): 1202, ("Z", "commentary"): 1103}


def test_ids_unique(records):
    dupes = [i for i, n in Counter(ids_of(records)).items() if n > 1]
    assert dupes == []


def test_id_formats(records):
    for r in records:
        if r["kind"] == "rule":
            assert re.fullmatch(r"10(05|26)\.\d+(\([A-Za-z0-9]{1,6}\))*", r["id"]), r["id"]
        else:
            assert re.fullmatch(r"10(05|26)\.\d+(\([A-Za-z0-9]{1,6}\))*-\d+", r["id"]), r["id"]


def test_only_sections_in_scope(records):
    for r in records:
        part, sec = r["section"].split(".")
        assert int(sec) in P.SCOPE[part]


def test_every_parent_exists(by_id):
    for r in by_id.values():
        if r["kind"] == "rule" and r["parent"]:
            assert r["parent"] in by_id, r["id"]


def test_subpart_on_every_record(records):
    expected = {"1005": {"A"}, "1026": {"A", "B", "G"}}
    for r in records:
        assert r["subpart"] in expected[r["part"]], r["id"]


def test_no_markup_left_in_text(records):
    for r in records:
        assert "<" not in r["text"] and "(_" not in r["text"], r["id"]


# ------------------------------------------------- nothing dropped, nothing repeated

def _norm(s):
    return re.sub(r"[\s—]", "", s)


@pytest.mark.parametrize("part", ["1005", "1026"])
def test_rule_text_conserved(records, part):
    """Every character of in-scope rule text appears once, in order, in the provisions."""
    root = ET.parse(P.RAW_DIR / f"part-{part}.xml").getroot()
    for d8 in root.iter("DIV8"):
        sec = d8.get("N")
        if int(sec.split(".")[1].split("-")[0]) not in P.SCOPE[part]:
            continue
        src = _norm("".join(P.text_of(e) for e in d8 if e.tag == "P"))
        out = _norm("".join(r["text"] for r in records if r["section"] == sec and r["kind"] == "rule"))
        assert src == out, sec


@pytest.mark.parametrize("part", ["1005", "1026"])
def test_commentary_text_conserved(records, part):
    root = ET.parse(P.RAW_DIR / f"part-{part}.xml").getroot()
    supp = next(d for d in root.iter("DIV9") if d.get("N", "").startswith("Supplement I"))
    sec, body = None, defaultdict(str)
    for el in supp:
        if el.tag in ("HEAD", "CITA", "XREF", "EDNOTE", "FP"):
            continue
        t = P.text_of(el)
        if not t:
            continue
        if P.is_heading(el, t):
            m = P.SEC_HEAD.match(t)
            if m:
                sec = int(m.group(2))
            elif P.OTHER_HEAD.match(t):
                sec = None
            continue
        if sec in P.SCOPE[part]:
            body[sec] += t
    for s, src in body.items():
        out = "".join(r["text"] for r in records if r["kind"] == "commentary" and r["section"] == f"{part}.{s}")
        if (part, s) == ("1026", 11):  # the deliberate 11(c) renumbering changes digits only
            assert len(_norm(src)) == len(_norm(out))
        else:
            assert _norm(src) == _norm(out), f"{part}.{s}"


# ------------------------------------------------------------ rule-text cases

def test_section_intro_text(by_id):
    assert by_id["1005.2"]["text"].startswith("Except as otherwise provided in subpart B")
    assert by_id["1005.2"]["depth"] == 0


def test_letter_i_after_h(records, by_id):
    """'(i)' is the letter after (h) here, not roman numeral one."""
    for sec in ("1005.2", "1026.3", "1026.13", "1026.56"):
        assert by_id[f"{sec}(i)"]["depth"] == 1, sec
    assert "1026.13(i)(1)" in by_id


def test_roman_i_under_h(by_id):
    assert by_id["1005.18(h)(2)(i)"]["depth"] == 3
    assert by_id["1005.20(h)(1)(i)"]["parent"] == "1005.20(h)(1)"
    # roman (i) with its own children before (ii)
    assert by_id["1026.3(h)(5)(i)(A)"]["text"] == "(A) Recording fees;"
    assert "1026.3(h)(5)(ii)" in by_id


def test_heading_then_label_in_one_paragraph(by_id):
    """'(a) Heading—(1) Heading. Text' is two provisions."""
    assert by_id["1026.52(a)"]["text"] == "(a) Limitations during first year after account opening"
    assert by_id["1026.52(a)(1)"]["text"].startswith("(1) General rule. Except as provided")
    assert "1026.52(b)(2)(i)(A)" in by_id  # "(2) ...—(i) ...—(A) Generally."
    assert by_id["1026.52(b)(3)(i)"]["text"].startswith("(i) Except as provided")


def test_labels_back_to_back(by_id):
    assert by_id["1026.12(b)(1)"]["text"] == "(1)"
    assert by_id["1026.12(b)(1)(i)"]["text"].startswith("(i) Definition of unauthorized use.")
    assert by_id["1005.2(a)(1)"]["text"].startswith("(1) “Access device” means")


def test_italic_labels(by_id):
    r = by_id["1026.52(b)(2)(i)(B)(1)"]
    assert r["text"] == "(1) Transactions that the card issuer declines to authorize;"
    assert r["depth"] == 5


def test_known_paragraphs(by_id):
    assert by_id["1026.52(b)(1)(ii)(A)"]["text"] == "(A) $32;"
    assert "two business days" in by_id["1005.6(b)(1)"]["text"]
    assert by_id["1026.62(a)"]["section_heading"].startswith("Overdraft credit")


# ------------------------------------------------------------- commentary cases

def test_comment_numbering_consecutive(records):
    seen = defaultdict(list)
    for r in records:
        if r["kind"] == "commentary":
            key, n = r["id"].rsplit("-", 1)
            seen[key].append(int(n))
    for key, nums in seen.items():
        assert nums == list(range(1, len(nums) + 1)), key


def test_missing_heading_fixed(by_id):
    assert by_id["1005.17(b)(3)-1"]["text"].startswith("1. Variations in terms, conditions, or features.")
    assert by_id["1005.17(b)(3)-1"]["target"] == "1005.17(b)(3)"
    assert "1005.17(b)(2)-3" not in by_id


def test_numbering_slip_fixed(by_id):
    assert by_id["1026.11(c)-3"]["text"].startswith("3. Request by an administrator of an estate.")
    assert by_id["1026.11(c)-7"]["text"].startswith("7. Application to joint accounts.")
    assert "1026.11(c)-8" not in by_id


def test_comment_ids_and_targets(by_id):
    c = by_id["1026.12(b)-1"]
    assert c["comment"] == "12(b)-1" and c["target"] == "1026.12(b)" and c["kind"] == "commentary"
    assert by_id["1026.12-1"]["target"] is None  # section-level comment


def test_italic_number_stays_inside_comment(records):
    """'<I>1.</I> Calculate ...' is a sub-item of comment 14(c)-5, not a new comment."""
    hits = [r["id"] for r in records if "1. Calculate the annual percentage rate" in r["text"]]
    assert hits == ["1026.14(c)-5"]


def test_subitem_tagged_as_heading_is_body(by_id):
    """A comment sub-item marked up as HD3 in 52(b)(1) commentary stays inside its comment."""
    assert "\nii. Relationship between § 1026.52(b)(1) and (2)." in by_id["1026.52(b)(1)-1"]["text"]


def test_plain_paragraph_headings(by_id):
    """'<P>Paragraph 51(b)(1)(i).</P>' is a heading, not text."""
    assert by_id["1026.51(b)(1)(i)-1"]["text"].startswith("1. Consideration of income and assets")


# ------------------------------------------------------------------ the tree

def test_tree_rejects_impossible_label():
    t = P.Tree("1026.99", {})
    t.place("a")
    with pytest.raises(P.ParseError):
        t.place("c")


def test_tree_letter_i_when_h_has_no_children():
    t = P.Tree("1026.99", {})
    for lab in "abcdefgh":
        t.place(lab)
    pid, level, _ = t.place("i")
    assert pid == "1026.99(i)" and level == P.LETTER


def test_tree_roman_i_must_be_followed_by_ii():
    t = P.Tree("1026.99", {})
    for lab in "abcdefgh":
        t.place(lab)
    t.place("1")
    t.place("i")  # could be (h)(1)(i) or the letter (i)
    with pytest.raises(P.ParseError):
        t.place("j")


def test_tree_unfinished_roman_i_is_an_error():
    t = P.Tree("1026.99", {})
    for lab in "abcdefgh":
        t.place(lab)
    t.place("1")
    t.place("i")
    with pytest.raises(P.ParseError):
        t.finish()


# ------------------------------------------------------------------- status

def _status(tmp_path, entries):
    p = tmp_path / "status.json"
    p.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return p


def _recs(*ids):
    return [{"id": i} for i in ids]


def test_status_subtree_and_most_specific(tmp_path):
    entries = P.load_status(_status(tmp_path, [
        {"id": "1026.62", "subtree": True, "status": "nullified", "note": "n", "source": "s"},
        {"id": "1026.52(b)(1)(ii)", "subtree": True, "status": "vacated", "note": "v", "source": "s"},
        {"id": "1026.52(b)(1)(ii)(C)", "status": "in_force", "note": "unchanged", "source": "s"},
    ]))
    recs = _recs("1026.62(a)", "1026.62(a)-1", "1026.52(b)(1)(ii)(A)", "1026.52(b)(1)(ii)(C)",
                 "1026.52(b)(1)(ii)-1", "1026.52(b)(1)", "1026.6")
    P.apply_status(recs, entries)
    got = {r["id"]: r["status"] for r in recs}
    assert got == {"1026.62(a)": "nullified", "1026.62(a)-1": "nullified",
                   "1026.52(b)(1)(ii)(A)": "vacated", "1026.52(b)(1)(ii)(C)": "in_force",
                   "1026.52(b)(1)(ii)-1": "vacated", "1026.52(b)(1)": "in_force", "1026.6": "in_force"}


def test_status_without_subtree_is_exact(tmp_path):
    entries = P.load_status(_status(tmp_path, [
        {"id": "1026.2(a)(15)", "status": "mixed", "note": "n", "source": "s"},
    ]))
    recs = _recs("1026.2(a)(15)", "1026.2(a)(15)(i)")
    P.apply_status(recs, entries)
    assert [r["status"] for r in recs] == ["mixed", "in_force"]


def test_status_entry_matching_nothing_is_an_error(tmp_path):
    entries = P.load_status(_status(tmp_path, [
        {"id": "1026.99(z)", "status": "vacated", "note": "n", "source": "s"},
    ]))
    with pytest.raises(P.ParseError):
        P.apply_status(_recs("1026.6"), entries)


def test_status_rejects_unknown_value_and_missing_fields(tmp_path):
    with pytest.raises(P.ParseError):
        P.load_status(_status(tmp_path, [{"id": "1026.6", "status": "maybe", "note": "n", "source": "s"}]))
    with pytest.raises(P.ParseError):
        P.load_status(_status(tmp_path, [{"id": "1026.6", "status": "vacated", "note": "n"}]))


def test_committed_status_file_applies(records):
    """If corpus/status.json exists, every entry matches something (checked in apply_status)."""
    assert {r["status"] for r in records} <= P.STATUSES
