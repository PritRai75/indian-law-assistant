"""The CI retrieval check (eval/retrieval.py, session 18). The first test is the gate itself: it searches
the dev questions with the committed vectors and fails if any rank got worse than eval/baseline.json.
The rest use made-up results and a fake embedder, so no AWS call is ever made."""
import json

import numpy as np
import pytest

from app import retrieve as rt
from eval import golden as G
from eval import retrieval as RV


# ------------------------------------------------------------------ the gate

def test_dev_retrieval_has_not_dropped_from_the_baseline():
    fails, notes, per_q = RV.check()
    assert fails == [], "\n".join(fails)
    assert len(per_q) == sum(not i.should_abstain for i in RV.dev_items())


def test_the_committed_vectors_are_unit_length_and_dev_only():
    info = json.loads(RV.VECTORS_INFO.read_text(encoding="utf-8"))
    vecs = np.load(RV.VECTORS)
    assert info["split"] == "dev" and len(vecs) == len(info["questions"]) == len(RV.dev_items())
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)
    test_ids = {i.id for i in G.load() if i.split == "test"}
    assert not test_ids & {q["id"] for q in info["questions"]}                  # CI never sees test questions


# Experiment 1's search arm (session 22): hybrid k5 on the rule text only, the dev questions' first gold
# ranks with the committed vectors (12 of 16 in the top 5, MRR 0.573; the default gives 9 and 0.438).
RULES_ONLY_RANKS = {"q001": None, "q002": None, "q003": 3, "q004": 4, "q005": 1, "q006": 4, "q007": None,
                    "q016": 3, "q018": 1, "q020": None, "q023": 1, "q034": 1, "q038": 1, "q049": 1,
                    "q052": 1, "q053": 1}


def test_rule_text_only_dev_ranks_are_as_measured():
    from eval import run as R
    items = G.load()
    vecs = RV.load_vectors(items)
    ix = R.index_view()
    got = {}
    for i in RV.dev_items(items):
        if not i.should_abstain:
            found = [r["id"] for r in rt.search(i.question, 5, "hybrid", query_vec=vecs[i.id], interpretations=False)]
            got[i.id] = R.retrieval_metrics(i.gold_ids or [], found, ix)["first_gold_rank"]
    assert got == RULES_ONLY_RANKS


# ------------------------------------------------------------------ comparing with a baseline

SETTINGS = {"k": 5, "mode": "hybrid", "rrf_k": 60, "pool": 50}


def per_q(ranks):
    return {q: {"persona": "frontline" if n % 2 else "compliance", "type": "dispute", "first_gold_rank": r,
                "gold_found": int(r is not None), "gold_total": 1, "retrieved": ["a", "b", "c", "d", "e"]}
            for n, (q, r) in enumerate(sorted(ranks.items()))}


BASE = RV.baseline_doc(per_q({"q901": 1, "q902": 3, "q903": None}), SETTINGS)


def test_same_results_pass():
    assert RV.compare(BASE, per_q({"q901": 1, "q902": 3, "q903": None}), SETTINGS) == ([], [])


def test_a_worse_rank_or_a_lost_hit_fails():
    fails, _ = RV.compare(BASE, per_q({"q901": 2, "q902": None, "q903": None}), SETTINGS)
    assert len(fails) == 2
    assert fails[0].startswith("q901 (compliance): first gold chunk rank 1 -> rank 2")
    assert fails[1].startswith("q902 (frontline): first gold chunk rank 3 -> not in the top 5")


def test_a_trade_still_fails_though_the_totals_hold():
    fails, notes = RV.compare(BASE, per_q({"q901": 1, "q902": None, "q903": 3}), SETTINGS)
    assert [f.split(":")[0] for f in fails] == ["q902 (frontline)"] and notes[0].startswith("q903: better")


def test_a_better_rank_passes_with_a_note():
    fails, notes = RV.compare(BASE, per_q({"q901": 1, "q902": 1, "q903": 4}), SETTINGS)
    assert fails == [] and len(notes) == 3 and "re-write the baseline" in notes[-1]


def test_changed_settings_or_questions_fail():
    assert "search settings changed" in RV.compare(BASE, per_q({"q901": 1, "q902": 3, "q903": None}),
                                                   {**SETTINGS, "k": 10})[0][0]
    assert "questions changed" in RV.compare(BASE, per_q({"q901": 1, "q902": 3}), SETTINGS)[0][0]


def test_tally_counts_hits_mrr_and_personas():
    t = RV.tally(per_q({"q901": 1, "q902": 3, "q903": None, "q904": 2}))
    assert t["overall"]["hit_at_1"] == [1, 4] and t["overall"]["hit_at_3"] == [3, 4] and t["overall"]["hit_at_5"] == [3, 4]
    assert t["overall"]["mrr"] == round((1 + 1 / 3 + 0 + 1 / 2) / 4, 3)
    assert t["by_persona"]["compliance"]["hit_at_5"] == [1, 2] and t["by_persona"]["frontline"]["hit_at_5"] == [2, 2]


# ------------------------------------------------------------------ the question vectors

def fake_embed(question):
    v = np.random.default_rng(len(question)).standard_normal(rt.load()[1].shape[1]).astype(np.float32)
    return v / np.linalg.norm(v)


def test_embed_then_load_round_trips_and_notices_changes(tmp_path):
    items = G.load()
    vec, info = tmp_path / "v.npy", tmp_path / "v.json"
    n = RV.embed_questions(items, embed=fake_embed, vectors=vec, info=info)
    assert n == len(RV.dev_items(items))
    loaded = RV.load_vectors(items, vec, info)
    first = RV.dev_items(items)[0]
    assert np.allclose(loaded[first.id], fake_embed(first.question))

    edited = [i.model_copy(update={"question": i.question + " Thanks."}) if i.id == first.id else i for i in items]
    with pytest.raises(ValueError, match=f"question text changed since embedding: {first.id}"):
        RV.load_vectors(edited, vec, info)
    moved = [i.model_copy(update={"split": "test"}) if i.id == first.id else i for i in items]
    with pytest.raises(ValueError, match=f"no longer dev: {first.id}"):
        RV.load_vectors(moved, vec, info)
    np.save(vec, np.load(vec)[:-1])
    with pytest.raises(ValueError, match="rows"):
        RV.load_vectors(items, vec, info)
    with pytest.raises(ValueError, match="missing"):
        RV.load_vectors(items, tmp_path / "none.npy", info)
