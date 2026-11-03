"""Tests for ingest/chunk.py, run on the committed provisions and status register.

The counts below are pinned to the corpus as of 2026-10-23 and the session-10
rules (4,000-character limit, no mixed statuses). If the corpus or a rule
changes, they change on purpose: update them in the same commit.
"""

import json
from collections import Counter

import pytest

from ingest import chunk as K

LIMIT = K.LIMIT


@pytest.fixture(scope="module")
def provisions():
    return K.load_provisions()


@pytest.fixture(scope="module")
def missing():
    return K.load_missing()


@pytest.fixture(scope="module")
def chunks(provisions, missing):
    return K.build(provisions, missing)


@pytest.fixture(scope="module")
def by_id(chunks):
    return {c["id"]: c for c in chunks}


def chunk_of(chunks, pid):
    return next(c for c in chunks if pid in c["provision_ids"])


def prov(pid, text, status="in_force", parent=None, kind="rule", target=None, section="1026.9"):
    """A small fake provision for unit tests of the splitting rules."""
    return {"id": pid, "kind": kind, "reg": "Z", "part": "1026", "subpart": "B", "section": section,
            "section_heading": "Test section.", "parent": parent, "depth": pid.count("("),
            "text": text, "status": status, "status_note": None if status == "in_force" else f"note {status}",
            "as_of": "2026-10-23", "comment": pid.split(".", 1)[1] if kind == "commentary" else None,
            "target": target, "paragraph_heading": "9(a) Test" if target else None}


# ------------------------------------------------------------------ overall shape

def test_counts_pinned(chunks):
    s = K.summary(chunks)
    assert s["chunks"] == 1115
    assert s["by_kind"] == {"rule": 379, "commentary": 736}
    assert s["by_status"] == {"in_force": 1068, "mixed": 32, "nullified": 8, "vacated": 7}
    assert s["single_provisions_over_limit"] == 33
    assert s["whole_sections"] == 13


def test_every_provision_once_in_order(chunks, provisions, missing):
    got = [pid for c in chunks for pid in c["provision_ids"]]
    extra = {m["id"] for m in missing}
    assert Counter(got) == Counter([p["id"] for p in provisions] + list(extra))
    assert [pid for pid in got if pid not in extra] == [p["id"] for p in provisions]


def test_chunk_ids_unique_and_shaped(chunks):
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        base = c["id"].split("#")[0]
        assert base == c["unit_id"]
        assert " " not in c["id"]
        if c["kind"] == "commentary":
            assert base.endswith("-cmt")
        else:
            assert "-" not in base
        assert c["id"].startswith(c["section"])


def test_pieces_numbered(chunks):
    units = Counter(c["unit_id"] for c in chunks)
    for c in chunks:
        if c["piece"]:
            n, of = c["piece"]
            assert c["id"] == f"{c['unit_id']}#{n}" and of == units[c["unit_id"]] and of >= 2
        else:
            assert units[c["unit_id"]] == 1


def test_size_limit(chunks):
    for c in chunks:
        assert c["chars"] == len(c["text"])
        if c["chars"] > LIMIT:
            assert len(c["provision_ids"]) == 1, c["id"]   # only a single long provision may pass the limit
        assert len(c["embed_text"]) <= K.EMBED_MAX


def test_fields_embed_py_needs(chunks):
    for c in chunks:
        assert isinstance(c["id"], str) and c["id"]
        assert c["embed_text"].endswith(c["text"])


def test_commentary_never_mixes_paragraphs(chunks, provisions):
    target = {p["id"]: p["target"] for p in provisions if p["kind"] == "commentary"}
    for c in chunks:
        if c["kind"] == "commentary":
            assert len({target[pid] for pid in c["provision_ids"]}) == 1, c["id"]


def test_chunks_stay_in_one_section_and_kind(chunks, provisions):
    meta = {p["id"]: (p["section"], p["kind"]) for p in provisions}
    for c in chunks:
        assert {meta.get(pid, (c["section"], c["kind"])) for pid in c["provision_ids"]} == {(c["section"], c["kind"])}


# ------------------------------------------------------------------ status

def test_no_chunk_mixes_statuses(chunks, provisions):
    st = {p["id"]: (p["status"], p["status_note"]) for p in provisions}
    for c in chunks:
        found = {st[pid] for pid in c["provision_ids"] if pid in st}
        assert len(found) <= 1, c["id"]
        if found:
            assert found == {(c["status"], c["status_note"])}


def test_status_in_every_header(chunks):
    for c in chunks:
        head = c["embed_text"][: -len(c["text"])]
        assert f"Status: {K.STATUS_LINE[c['status']]}" in head
        assert f"Text as of: {c['as_of']}" in head
        if c["status"] != "in_force":
            assert c["status_note"] and c["status_note"] in head


def test_flagged_text_never_inside_another_chunk(chunks, provisions):
    """No vacated, nullified or mixed provision's text shows up in any other chunk (header included)."""
    flagged = [p for p in provisions if p["status"] != "in_force" and len(p["text"]) > 20]
    for c in chunks:
        for p in flagged:
            if p["id"] not in c["provision_ids"]:
                assert p["text"][:80] not in c["embed_text"], (c["id"], p["id"])


def test_late_fee_amounts_are_their_own_vacated_chunk(chunks):
    c = chunk_of(chunks, "1026.52(b)(1)(ii)(A)")
    assert c["status"] == "vacated"
    assert c["provision_ids"] == ["1026.52(b)(1)(ii)(A)", "1026.52(b)(1)(ii)(B)"]
    assert "VACATED" in c["embed_text"]


def test_overdraft_section_all_nullified(chunks):
    cs = [c for c in chunks if c["section"] == "1026.62"]
    assert cs and {c["status"] for c in cs} == {"nullified"}


def test_flagged_parent_named_not_quoted(by_id):
    head = by_id["1026.52(b)(1)#5"]["embed_text"]
    assert "Part of § 1026.52(b)(1)(ii)" in head
    assert "$8" not in head


# ------------------------------------------------------------------ structure on the real corpus

def test_short_section_is_one_chunk(by_id):
    c = by_id["1005.6"]
    assert c["unit_id"] == "1005.6" and c["provision_ids"][0] == "1005.6(a)"
    assert c["provision_ids"][-1].startswith("1005.6(b)")


def test_long_section_splits_at_first_level(by_id):
    assert "1026.13" not in by_id
    assert "1026.13(c)" in by_id or any(k.startswith("1026.13(c)") for k in by_id)


def test_second_level_split(by_id):
    assert "1005.18(b)" not in by_id
    assert "1005.18(b)(1)" in by_id
    assert by_id["1005.18(b)(2)#1"]["piece"] == [1, 5]


def test_intro_goes_with_first_piece(chunks):
    c = chunk_of(chunks, "1005.2")      # section intro text of the definitions section
    assert c["provision_ids"][0] == "1005.2" and len(c["provision_ids"]) > 1


def test_lead_in_for_piece_inside_a_paragraph(by_id):
    c = by_id["1005.18(b)(2)#2"]          # starts at (b)(2)(ix)(A), so the lead-in is its parent (ix)
    assert c["provision_ids"][0] == "1005.18(b)(2)(ix)(A)"
    assert "Lead-in from § 1005.18(b)(2)(ix): (ix) Disclosure of additional fee types" in c["embed_text"]
    assert "Lead-in" not in by_id["1005.18(b)(1)"]["embed_text"]   # starts at its own paragraph


def test_section_level_comments(by_id):
    c = next(v for k, v in by_id.items() if k.startswith("1026.12-cmt"))
    assert c["provision_ids"][0] == "1026.12-1"
    assert "General comments on § 1026.12" in c["embed_text"]


def test_commentary_header(by_id):
    head = by_id["1026.12(b)-cmt"]["embed_text"]
    assert "Official interpretation (Supplement I) of § 1026.12" in head
    assert "Paragraph: 12(b) Liability of Cardholder for Unauthorized Use" in head
    assert "Covers: comments 12(b)-1 to 12(b)-4" in head


def test_paragraph_word_not_doubled(chunks):
    assert not any("Paragraph: Paragraph" in c["embed_text"] for c in chunks)
    c = chunk_of(chunks, "1026.55(b)(7)(ii)-3")
    assert "Paragraph: 55(b)(7)(ii)" in c["embed_text"]


def test_longest_comment_kept_whole(chunks):
    c = chunk_of(chunks, "1005.18(g)-5")
    assert c["provision_ids"] == ["1005.18(g)-5"] and c["chars"] == 15545


# ------------------------------------------------------------------ missing in-force paragraph

def test_missing_paragraph_chunk(chunks, by_id):
    c = by_id["1026.2(a)(15)(ii)(C)"]
    assert c["status"] == "in_force" and c["provision_ids"] == ["1026.2(a)(15)(ii)(C)"]
    assert "not printed in the eCFR" in c["embed_text"] and "Pub. L. 119-10" in c["embed_text"]
    assert c["text"].startswith("(C) An overdraft line of credit")
    i = chunks.index(c)
    assert "1026.2(a)(15)(ii)(B)" in chunks[i - 1]["provision_ids"]


def test_missing_paragraph_already_present_is_an_error(provisions, missing):
    entry = dict(missing[0], id="1026.2(a)(15)(ii)(B)")
    with pytest.raises(ValueError, match="already a provision"):
        K.build(provisions, [entry])


# ------------------------------------------------------------------ splitting rules on small inputs

def test_status_change_starts_a_new_chunk():
    ps = [prov("1026.9(a)", "(a) a"), prov("1026.9(a)(1)", "(1) b", parent="1026.9(a)"),
          prov("1026.9(a)(2)", "(2) c", status="vacated", parent="1026.9(a)"),
          prov("1026.9(a)(3)", "(3) d", parent="1026.9(a)")]
    units = K.rule_units("1026.9", ps)
    assert [(cid, [p["id"] for p in g]) for cid, g in units] == [
        ("1026.9(a)#1", ["1026.9(a)", "1026.9(a)(1)"]),
        ("1026.9(a)#2", ["1026.9(a)(2)"]),
        ("1026.9(a)#3", ["1026.9(a)(3)"])]


def test_pack_keeps_a_long_provision_whole():
    ps = [prov("1026.9(a)", "x" * 50), prov("1026.9(b)", "y" * 300), prov("1026.9(c)", "z" * 50)]
    assert [[p["id"] for p in g] for g in K.pack(ps, limit=100)] == [["1026.9(a)"], ["1026.9(b)"], ["1026.9(c)"]]


def test_small_section_one_chunk():
    ps = [prov("1026.9(a)", "(a) a"), prov("1026.9(b)", "(b) b")]
    assert [cid for cid, _ in K.rule_units("1026.9", ps)] == ["1026.9"]


def test_split_levels():
    ps = [prov("1026.9", "intro"), prov("1026.9(a)", "a" * 60),
          prov("1026.9(b)", "(b) head"), prov("1026.9(b)(1)", "b" * 60, parent="1026.9(b)"),
          prov("1026.9(b)(2)", "c" * 60, parent="1026.9(b)")]
    units = K.rule_units("1026.9", ps, limit=100)
    assert [(cid, [p["id"] for p in g]) for cid, g in units] == [
        ("1026.9(a)", ["1026.9", "1026.9(a)"]),
        ("1026.9(b)(1)", ["1026.9(b)", "1026.9(b)(1)"]),
        ("1026.9(b)(2)", ["1026.9(b)(2)"])]


def test_mixed_statuses_in_one_chunk_is_an_error():
    ps = [prov("1026.9(a)", "a"), prov("1026.9(b)", "b", status="vacated")]
    with pytest.raises(ValueError, match="mixes statuses"):
        K.make_chunk("1026.9", "1026.9", ps, "Test", None)


def test_preceding_sibling():
    assert K.preceding_sibling("1026.2(a)(15)(ii)(C)") == "1026.2(a)(15)(ii)(B)"
    assert K.preceding_sibling("1005.6(b)(3)") == "1005.6(b)(2)"
    with pytest.raises(ValueError):
        K.preceding_sibling("1026.2(a)(15)(ii)(A)")


# ------------------------------------------------------------------ the committed file

def test_committed_chunks_match_a_fresh_build(chunks):
    """corpus/txt/chunks.jsonl is what chunk.py produces now (re-run it after changing the
    provisions, the status register or chunk.py)."""
    if not K.OUT.exists():
        pytest.skip("corpus/txt/chunks.jsonl not built yet")
    with open(K.OUT, encoding="utf-8") as f:
        committed = [json.loads(line) for line in f if line.strip()]
    assert committed == chunks
