"""Tests for app/retrieve.py. No AWS call: vector and hybrid searches are given a ready-made query vector."""
from pathlib import Path

import numpy as np
import pytest

from app import retrieve as rt

ROOT = Path(__file__).resolve().parents[1]
needs_index = pytest.mark.skipif(not (ROOT / "index" / "embeddings.npy").exists(), reason="index not built yet")


def read(text):
    """read_refs without the written text: [(parts, path), ...]."""
    return [(parts, path) for _, parts, path in rt.read_refs(text)]


# ---------------------------------------------------------------- pieces, no index needed

def test_tokens_split_provision_ids():
    assert rt.tokens("What does 1026.13(c)(2) require?") == ["what", "does", "1026", "13", "c", "2", "require"]
    assert rt.tokens("comment 12(b)-1") == ["comment", "12", "b", "1"]


def test_full_references_carry_their_part():
    assert read("What does 1026.13(c)(2) require?") == [(["1026"], "13(c)(2)")]
    assert read("Is § 1005.6 about liability?") == [(["1005"], "6")]
    assert read("12 CFR 1026.12 unauthorized use") == [(["1026"], "12")]
    assert read("Reg E 1005.11(c)(3)(i)") == [(["1005"], "11(c)(3)(i)")]
    assert read("comment 1026.13(c)(2)-1") == [(["1026"], "13(c)(2)-1")]
    assert read("1026.52(b)(1)(ii)(A) late fee") == [(["1026"], "52(b)(1)(ii)(a)")]
    assert read("§§ 1026.12–1026.13") == [(["1026"], "12"), (["1026"], "13")]


def test_short_references_take_the_named_regulation():
    assert read("Under Reg E 6(b), what is the cap?") == [(["1005"], "6(b)")]
    assert read("Regulation Z § 13(c)(2) billing cycles") == [(["1026"], "13(c)(2)")]
    assert read("Reg. E section 11 timing") == [(["1005"], "11")]
    assert read("reg z 7(B)(1)") == [(["1026"], "7(b)(1)")]
    assert read("section 6 of Reg E") == [(["1005"], "6")]
    assert read("paragraph 7(b)(9) of Reg Z") == [(["1026"], "7(b)(9)")]
    assert read("What does § 13(c)(2) mean under Reg Z?") == [(["1026"], "13(c)(2)")]
    assert read("Under Reg Z what does 9(c)(2) say?") == [(["1026"], "9(c)(2)")]
    assert read("Under the EFTA, see comment 6(b)(1)-3") == [(["1005"], "6(b)(1)-3")]
    assert read("Truth in Lending § 13(c)(2)") == [(["1026"], "13(c)(2)")]


def test_short_references_without_a_regulation_try_both_parts():
    assert read("What does comment 12(b)-1 say?") == [(["1005", "1026"], "12(b)-1")]
    assert read("13(c)(2) for a credit card") == [(["1005", "1026"], "13(c)(2)")]
    assert read("Reg E 6(b) or Reg Z 12(b)?") == [(["1005"], "6(b)"), (["1026"], "12(b)")]


def test_comment_forms():
    assert read("comment 6(b)(1)-3.ii on business days") == [(["1005", "1026"], "6(b)(1)-3")]
    assert read("Reg Z comment 13(c)(2)–1") == [(["1026"], "13(c)(2)-1")]    # en dash
    assert read("Reg Z comment 12-1") == [(["1026"], "12-1")]
    # the comment is read once, not also as the rule paragraph 13(c)(2)
    assert read("Reg Z comment 13(c)(2)-1 and 13(c)(2)") == [(["1026"], "13(c)(2)-1"), (["1026"], "13(c)(2)")]


def test_things_that_are_not_references():
    assert read("Reg E, 2 business days after learning of the loss") == []
    assert read("section 5 of my cardholder agreement says") == []
    assert read("I was charged $50 (fifty) within 2 (two) days") == []
    assert read("card ending 4417 used on 2026-10-23 at 9:30 (am)") == []
    assert read("My card was stolen. What can I do?") == []
    assert read("12 CFR Part 1026") == []              # a whole part names nothing to rank first


def test_covered_by():
    assert rt.covered_by("1026.13(c)", "1026.13(c)")
    assert rt.covered_by("1026.13(c)", "1026.13(c)(2)")
    assert rt.covered_by("1026.13(c)", "1026.13(c)(2)-1")         # comments on paragraphs under it
    assert rt.covered_by("1026.13", "1026.13-1")                  # section-level comments
    assert not rt.covered_by("1026.1", "1026.13(c)")              # 1026.1 is not a prefix of section 13
    assert not rt.covered_by("1026.13(c)", "1026.13(d)")
    assert not rt.covered_by("1026.13(c)(2)-1", "1026.13(c)(2)")  # a comment covers only itself
    assert not rt.covered_by("1026.13(c)(2)-1", "1026.13(c)(2)-10")


def test_resolve_spelling_both_parts_and_misses():
    provisions = ["1005.12(b)-1", "1026.12(b)-1", "1026.52(b)(1)(ii)(A)", "1005.6(b)(1)", "1026.13-1"]
    assert rt.resolve([("x", ["1026"], "52(b)(1)(ii)(a)")], provisions) == (["1026.52(b)(1)(ii)(A)"], [])
    assert rt.resolve([("x", ["1005", "1026"], "12(b)-1")], provisions) == (["1005.12(b)-1", "1026.12(b)-1"], [])
    assert rt.resolve([("§ 1005.6", ["1005"], "6")], provisions) == (["1005.6"], [])    # a section, by its paragraphs
    assert rt.resolve([("1026.18", ["1026"], "18")], provisions) == ([], ["1026.18"])
    assert rt.resolve([("a", ["1026"], "13"), ("b", ["1026"], "13")], provisions) == (["1026.13"], [])


def test_bm25_prefers_rare_matching_words():
    docs = [rt.tokens(t) for t in ("billing error notice within sixty days",
                                   "notice of a change in terms", "the notice")]
    s = rt.BM25(docs).scores(rt.tokens("billing error notice"))
    assert s.argmax() == 0 and s[1] > 0 and s[2] > 0


def test_bm25_unknown_words_score_zero():
    assert not rt.BM25([["card"]]).scores(["mortgage"]).any()


def test_rrf_rewards_agreement():
    top, fused = rt.rrf([[1, 2, 3], [3, 1, 4]], 4)
    assert top[0] == 1                       # 1st and 2nd beats 3rd and 1st
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert set(top) == {1, 2, 3, 4}


def test_ranked_ties_keep_index_order():
    assert rt.ranked(np.array([0.5, 0.9, 0.5, 0.9]), 4) == [1, 3, 0, 2]


def test_bad_mode_is_refused():
    with pytest.raises(ValueError):
        rt.search("x", mode="keyword")


# ---------------------------------------------------------------- on the real index

def vec_of(chunk_id):
    chunks, vecs, _, _ = rt.load()
    return vecs[[c["id"] for c in chunks].index(chunk_id)]


@needs_index
def test_every_provision_is_found_by_its_own_id():
    """Each of the 3,401 provision IDs, written as a question would write it, names exactly itself
    and lands on the one chunk that holds it."""
    chunks = rt.load()[0]
    for i, c in enumerate(chunks):
        for p in c["provision_ids"]:
            written = f"comment {p}" if "-" in p else f"§ {p}"
            assert rt.query_refs(written) == [p], written
            if "-" in p:
                assert np.flatnonzero(rt.named_mask([p])).tolist() == [i], p


@needs_index
def test_short_forms_reach_the_index():
    assert rt.query_refs("Under Reg E 6(b), what is the cap?") == ["1005.6(b)"]
    assert rt.query_refs("What does comment 12(b)-1 say?") == ["1005.12(b)-1", "1026.12(b)-1"]
    assert rt.query_refs("13(c)(2) for a credit card") == ["1026.13(c)(2)"]    # no 1005.13(c)(2)
    assert rt.query_refs("Reg Z 1026.18 APR disclosure") == []                # excluded subpart C
    assert rt.query_refs("comment 2(a)(14)-4") == []                           # no longer in the eCFR text


@needs_index
def test_named_set_is_the_provision_its_paragraphs_and_their_comments():
    chunks = rt.load()[0]
    ids = {chunks[i]["id"] for i in np.flatnonzero(rt.named_mask(["1005.6(b)"]))}
    assert "1005.6" in ids and "1005.6(b)-cmt" in ids and "1005.6(b)(1)-cmt" in ids
    assert all(i.startswith("1005.6") for i in ids)
    assert not {i for i in ids if i.startswith("1005.6(a)") or i.startswith("1005.6(c)")}
    assert rt.named_mask([]).sum() == 0


@needs_index
def test_chunk_vector_finds_its_own_chunk():
    assert rt.search("", k=1, mode="vector", query_vec=vec_of("1005.6(b)-cmt"))[0]["id"] == "1005.6(b)-cmt"


@needs_index
def test_bm25_puts_named_provisions_first():
    top = rt.search("What does 1026.13(c)(2) require?", k=5, mode="bm25")
    assert [r["id"] for r in top[:2]] == ["1026.13(c)(2)-cmt", "1026.13(c)"]
    assert [r["named"] for r in top] == [True, True, False, False, False]
    assert all(r["named"] for r in rt.search("Under Reg E 6(b), what is the most a customer can lose?", 5, "bm25"))


@needs_index
def test_named_provision_first_in_hybrid_too():
    # A stand-in question vector that favours Reg Z unauthorized use, as a vague question about a
    # card might: the named Reg E paragraph must still come first, and the fused list fills the rest.
    q = vec_of("1026.12(b)-cmt")
    top = rt.search("Under Reg E 6(b), what is the cap?", k=5, query_vec=q)
    assert top[0]["named"] and top[0]["chunk"]["section"] == "1005.6"
    plain = rt.search("What is the cap on liability?", k=5, query_vec=q)
    assert not any(r["named"] for r in plain)                  # nothing named, nothing promoted
    assert "1026.12(b)-cmt" in [r["id"] for r in plain]


@needs_index
def test_hybrid_is_deterministic_and_ranked():
    q = vec_of("1026.13(c)(2)-cmt")
    a = rt.search("billing error resolution two complete billing cycles", k=5, query_vec=q)
    b = rt.search("billing error resolution two complete billing cycles", k=5, query_vec=q)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["rank"] for r in a] == [1, 2, 3, 4, 5]
    assert "1026.13(c)(2)-cmt" in [r["id"] for r in a[:3]]


@needs_index
def test_flagged_chunks_stay_searchable():
    # Vacated and nullified text must be found, so the assistant can say why it won't state it.
    statuses = {r["chunk"]["status"] for r in rt.search("§ 1026.52(b)(1)(ii)", k=10, mode="bm25")}
    assert "vacated" in statuses


@needs_index
def test_command_line_prints_all_three_modes(monkeypatch, capsys):
    monkeypatch.setattr(rt, "embed_query", lambda text: vec_of("1005.6(b)-cmt"))
    rt.main(["-k", "3", "Under Reg E 6(b) and 1026.18, what is the cap?"])
    out = capsys.readouterr().out
    assert "Named: 1005.6(b)" in out and "Not in the index: 1026.18" in out
    assert out.count("\n") == 4 + 3          # question, named line, blank, header, then k rows


# ---------------------------------------------------------------- rule text only (experiment 1)

@needs_index
def test_rule_text_only_search_returns_rule_chunks_only():
    chunks, vecs, _, bm25 = rt.load_rules()
    assert chunks and all(c["kind"] == "rule" for c in chunks) and len(vecs) == len(chunks) == len(bm25.tfs)
    assert len(chunks) == sum(c["kind"] == "rule" for c in rt.load()[0])
    q = vec_of("1026.13(c)(2)-cmt")
    for mode in ("vector", "bm25", "hybrid"):
        top = rt.search("billing error resolution two complete billing cycles", 5, mode, query_vec=q,
                        interpretations=False)
        assert len(top) == 5 and all(r["chunk"]["kind"] == "rule" for r in top)
    with_comments = rt.search("billing error resolution two complete billing cycles", 5, query_vec=q)
    assert any(r["chunk"]["kind"] == "commentary" for r in with_comments)     # the default is unchanged


@needs_index
def test_rule_text_only_still_puts_named_provisions_first():
    top = rt.search("What does 1026.13(c)(2) require?", k=5, mode="bm25", interpretations=False)
    assert top[0]["id"] == "1026.13(c)" and top[0]["named"]
    assert [r["named"] for r in top].count(True) == sum(rt.named_in("What does 1026.13(c)(2) require?",
                                                                   rt.load_rules()[0]))
