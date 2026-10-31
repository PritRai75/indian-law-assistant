"""Tests for app/retrieve.py. No AWS call: vector and hybrid searches are given a ready-made query vector."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import retrieve as rt  # noqa: E402

needs_index = pytest.mark.skipif(not (ROOT / "index" / "embeddings.npy").exists(), reason="index not built yet")


# ---------------------------------------------------------------- pieces, no index needed

def test_tokens_split_section_numbers():
    assert rt.tokens("Is BNS 318(4) bailable?") == ["is", "bns", "318", "4", "bailable"]
    assert rt.tokens("BNSS-S2(1)(g)") == ["bnss", "s", "2", "1", "g"]
    assert rt.tokens("Order VIII, Rule 1") == ["order", "viii", "rule", "1"]


def test_reference_word_of_each_id_shape():
    assert rt.ref_word("BNS-S318") == "bns318"
    assert rt.ref_word("BNSS-S2") == "bnss2"
    assert rt.ref_word("CPC-S35A") == "cpc35a"
    assert rt.ref_word("CPC-OVIII-R1@commercial") == "cpc_oviii_r1"
    assert rt.ref_word("CPC-OXIII-A-R1") == "cpc_oxiii_a_r1"
    assert rt.ref_word("CPC-OXXI-R46I") == "cpc_oxxi_r46i"


def test_references_written_in_questions():
    assert rt.query_refs("Is an offence under BNS 318(4) bailable?") == ["bns318"]
    assert rt.query_refs("What does section 63 of the BSA require?") == ["bsa63"]
    assert rt.query_refs("Is s. 318(4) of BNS bailable?") == ["bns318"]
    assert rt.query_refs("BNSS s.173 and CPC Section 35A") == ["bnss173", "cpc35a"]
    assert rt.query_refs("Order VIII Rule 1 written statement") == ["cpc_oviii_r1"]
    assert rt.query_refs("O.8 r.1 and Order XIII-A rule 3") == ["cpc_oviii_r1", "cpc_oxiii_a_r3"]
    assert rt.query_refs("The police won't write down my complaint") == []
    assert rt.query_refs("abns 3 and IPC 420") == []


def test_bm25_prefers_rare_matching_words():
    docs = [rt.tokens(t) for t in ("cheating and dishonestly inducing delivery of property",
                                   "theft of property", "the property of the State")]
    s = rt.BM25(docs).scores(rt.tokens("cheating property"))
    assert s.argmax() == 0 and s[1] > 0 and s[2] > 0


def test_bm25_unknown_words_score_zero():
    assert not rt.BM25([["theft"]]).scores(["murder"]).any()


def test_rrf_rewards_agreement():
    top, fused = rt.rrf([[1, 2, 3], [3, 1, 4]], 4)
    assert top[0] == 1                       # 1st and 2nd beats 3rd and 1st
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert set(top) == {1, 2, 3, 4}


def test_ranked_ties_keep_index_order():
    assert rt.ranked(np.array([0.5, 0.9, 0.5, 0.9]), 4) == [1, 3, 0, 2]


# ---------------------------------------------------------------- on the real index

@needs_index
def test_chunk_vector_finds_its_own_chunk():
    chunks, vecs, _, _ = rt.load()
    i = [c["id"] for c in chunks].index("BNS-S318")
    assert rt.search("", k=1, mode="vector", query_vec=vecs[i])[0]["id"] == "BNS-S318"


@needs_index
def test_bm25_finds_section_by_number_and_act():
    ids = [r["unit_id"] for r in rt.search("BNS section 318 cheating", k=3, mode="bm25")]
    assert ids[0] == "BNS-S318"


@needs_index
def test_bm25_finds_cpc_rule():
    ids = [r["unit_id"] for r in rt.search("Order VIII Rule 1 written statement", k=3, mode="bm25")]
    assert "CPC-OVIII-R1" in ids


@needs_index
def test_hybrid_runs_without_aws_and_is_deterministic():
    chunks, vecs, _, _ = rt.load()
    q = vecs[[c["id"] for c in chunks].index("BNSS-S173")]
    a = rt.search("information in cognizable cases FIR", k=5, query_vec=q)
    b = rt.search("information in cognizable cases FIR", k=5, query_vec=q)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    # vector ranks s.173 first (its own vector); BM25 prefers s.174 for these words; fusion keeps both near the top
    assert {"BNSS-S173", "BNSS-S174"} <= {r["unit_id"] for r in a[:3]}
    assert [r["rank"] for r in a] == [1, 2, 3, 4, 5]


@needs_index
def test_lawyer_reference_beats_same_number_in_other_acts():
    # Without reference words, BNSS s.318 (same number, shorter chunk) outranked BNS s.318.
    assert rt.search("BNS section 318 cheating", k=1, mode="bm25")[0]["unit_id"] == "BNS-S318"
    assert rt.search("Is an offence under BNS 318(4) bailable?", k=1, mode="bm25")[0]["unit_id"] == "BNS-S318"
    assert rt.search("Is BNSS 318 about the same thing?", k=1, mode="bm25")[0]["unit_id"] == "BNSS-S318"


def test_bad_mode_is_refused():
    with pytest.raises(ValueError):
        rt.search("x", mode="keyword")
