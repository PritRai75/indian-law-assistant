"""Tests for eval/report.py: the error tags and their order, the outside-law keywords, the slices and
the author's review. No AWS call: answers are made up, graded by eval.run.grade and judged by hand."""
import json
import tomllib

import pytest

from app.tools import dispute_timeline
from eval import golden as G
from eval import judge as J
from eval import report as P
from eval import run as R

IX = R.index_view()
BY_ID = {i.id: i for i in G.load()}
PLAIN = BY_ID["q006"]          # Z, frontline, one gold ID (1026.10(b)(2)(ii)), two key facts, no tool
TWO_GOLD = BY_ID["q005"]       # Z, compliance, gold 1026.9(c)(2)(i) and 1026.55(b)(3)
TOOL = BY_ID["q004"]           # E, a dispute_timeline call with two deadlines
NETWORK = BY_ID["q008"]        # decline, network_rules
STATE = BY_ID["q024"]          # decline, state_law
OTHER_CHUNK = "1026.4(a)"      # holds no gold ID of these items
NOT_IN_FORCE = "1026.2(a)(15)(i)"


def resp(retrieved=("1026.10(b)", OTHER_CHUNK), cites=("1026.10(b)(2)(ii)",), **over):
    """A made-up app.schema.Response as a dict."""
    r = {"answer": "An answer.", "citations": list(cites), "tool_calls": [], "abstained": False,
         "abstain_reason": None, "retrieved_ids": list(retrieved), "regulations_as_of": "2026-10-23",
         "disclaimer": "x", "raw_answer": "An answer.", "raw_citations": list(cites), "invalid_citations": [],
         "flags": [], "prompt_version": "p2", "config_hash": "abc", "model_id": "m",
         "usage": {"inputTokens": 1000, "outputTokens": 100}, "model_calls": 1, "latency_ms": 4000}
    r.update(over)
    return r


def tool_resp(args=None, answer=None, **over):
    args = args or dict(TOOL.expected_tool_call.args)
    dates = " and ".join(d.date.strftime("%B %d, %Y") for d in TOOL.expected_result.deadlines)
    return resp(retrieved=("1005.11(c)", OTHER_CHUNK), cites=("1005.11(c)(1)",),
                tool_calls=[{"name": "dispute_timeline", "args": args, "result": dispute_timeline(**args)}],
                answer=answer or f"Investigate by {dates}.", **over)


def judged(label="faithful", covered=None, n=2, unsupported=()):
    covered = n if covered is None else covered
    return P.Verdict("judged", label, covered, n, list(unsupported))


def tags(item, r, v=None):
    return [t for t, _ in P.tags_for(item, R.grade(item, r, IX), v or judged(n=len(item.key_facts or [])))]


def test_the_items_used_here_are_what_the_tests_assume():
    assert (PLAIN.regulation, PLAIN.persona, PLAIN.gold_ids, len(PLAIN.key_facts)) == ("Z", "frontline", ["1026.10(b)(2)(ii)"], 2)
    assert TWO_GOLD.gold_ids == ["1026.9(c)(2)(i)", "1026.55(b)(3)"] and TOOL.expected_tool_call.args["regulation"] == "E"
    assert (NETWORK.abstain_category, STATE.abstain_category) == ("network_rules", "state_law")
    assert OTHER_CHUNK in IX.chunk_provisions and IX.status[NOT_IN_FORCE] != "in_force"
    assert NOT_IN_FORCE in IX.chunk_provisions["1026.2(a)(15)#2"]


def test_tags_are_the_spec_taxonomy_plus_incomplete():
    spec = {"retrieval_miss", "partial_retrieval", "wrong_reg", "status_error", "tool_misuse", "calculation_error",
            "memory_leak", "unsupported_claim", "wrong_citation", "missed_abstention", "over_abstention"}
    assert set(P.TAGS) == spec | {"incomplete"} and len(P.TAGS) == 12 and P.PASS not in P.TAGS


# ---------------------------------------------------------------- outside-law keywords

@pytest.mark.parametrize("claim", [
    "Visa's zero liability policy means she owes nothing", "Under the network rules the bank must refund",
    "State law may give the customer more time", "Under California law the deadline is longer",
    "The Fair Credit Reporting Act governs the dispute with the bureau", "FCRA requires the bureau to investigate",
    "the Act requires a written notice", "Regulation CC sets the hold period", "Reg F covers the collector",
    "a court could hold the issuer liable", "the issuer's own terms waive the fee", "the cardholder agreement allows it",
    "the statute of limitations is three years", "she can file a chargeback with Mastercard",
])
def test_outside_law_finds(claim):
    assert P.outside_law(claim)


@pytest.mark.parametrize("claim", [
    "She discovered the loss on Friday", "The creditor must state that the rate may increase",
    "the consumer acted within 60 days", "Regulation E caps liability at $50", "Under Regulation Z the issuer must",
    "Reg Z requires an acknowledgement", "the institution must act on the notice", "a courtesy notice is not required",
    "the prepaid account agreement must disclose fees", "subject to the terms and conditions of the account",
    "the transfer was made through a debit card network",
])
def test_outside_law_leaves(claim):
    assert not P.outside_law(claim)


# ---------------------------------------------------------------- answerable questions: each tag and the order

def test_a_clean_answer_passes():
    assert tags(PLAIN, resp()) == []
    assert tags(TOOL, tool_resp()) == []


def test_retrieval_miss_comes_before_the_decline_it_causes():
    r = resp(retrieved=(OTHER_CHUNK,), cites=(), abstained=True)
    assert tags(PLAIN, r, judged(covered=0)) == ["retrieval_miss", "over_abstention"]


def test_over_abstention_records_its_cause_and_hides_its_consequences():
    m = R.grade(TOOL, resp(retrieved=("1005.11(c)",), cites=(), abstained=True, flags=["no_valid_citation"]), IX)
    got = P.tags_for(TOOL, m, P.Verdict("skipped", kf_total=None))
    assert got == [("over_abstention", "declined (cause: no_valid_citation)")]   # no missed call, no key facts


def test_wrong_reg_and_status_error():
    assert tags(PLAIN, resp(cites=("1026.10(b)(2)(ii)", "1005.6(b)(1)")))[0] == "wrong_reg"
    assert tags(PLAIN, resp(retrieved=("1026.10(b)", "1026.2(a)(15)#2"), cites=("1026.10(b)(2)(ii)", NOT_IN_FORCE))) \
        == ["status_error"]


def test_tool_misuse_and_calculation_error():
    wrong = dict(TOOL.expected_tool_call.args, notice_date="2026-11-06")
    got = tags(TOOL, tool_resp(args=wrong, answer="Investigate by November 23, 2026."))
    assert got == ["tool_misuse", "calculation_error"]
    assert tags(TOOL, tool_resp(answer="Investigate by November 20, 2026.")) == ["calculation_error"]
    m = R.grade(TOOL, tool_resp(args=wrong), IX)
    assert P.tags_for(TOOL, m, judged())[0][1] == "wrong_args (notice_date)"


def test_judge_claims_split_into_memory_leak_and_unsupported_claim():
    v = judged("partially_faithful", unsupported=["Visa's zero liability covers her", "the fee is $40"])
    assert tags(PLAIN, resp(), v) == ["memory_leak", "unsupported_claim"]
    assert tags(PLAIN, resp(), judged("unfaithful", unsupported=["the bank must reply in 5 days"])) == ["unsupported_claim"]


def test_wrong_citation_but_not_for_a_citation_form():
    assert tags(PLAIN, resp(cites=("1026.10(b)(2)(ii)", "1026.11(a)(1)"))) == ["wrong_citation"]   # real ID, not given
    assert tags(PLAIN, resp(cites=("1026.10(b)(2)(ii)", "1026.99(z)"))) == ["wrong_citation"]      # not in the index
    assert tags(PLAIN, resp(cites=("1026.10(b)(2)(ii)", "§ 1026.10"))) == []                       # a whole section


def test_missing_key_facts_partial_retrieval_or_incomplete():
    both = ("1026.9(c)(2)#1", "1026.55(b)(3)")
    assert tags(TWO_GOLD, resp(retrieved=both, cites=("1026.55(b)(3)",)), judged(covered=1)) == ["incomplete"]
    assert tags(TWO_GOLD, resp(retrieved=("1026.55(b)(3)", OTHER_CHUNK), cites=("1026.55(b)(3)",)),
                judged(covered=1)) == ["partial_retrieval"]
    assert tags(TWO_GOLD, resp(retrieved=both, cites=("1026.55(b)(3)",)), judged(covered=2)) == []


def test_no_key_fact_tag_after_a_retrieval_miss():
    assert tags(PLAIN, resp(retrieved=(OTHER_CHUNK,), cites=()), judged(covered=0)) == ["retrieval_miss"]


def test_an_unneeded_call_is_checked_last():
    call = {"name": "dispute_timeline", "args": {"regulation": "Z"}, "result": {"error": "x"}}
    assert tags(PLAIN, resp(tool_calls=[call])) == ["tool_misuse"]
    got = P.tags_for(PLAIN, R.grade(PLAIN, resp(tool_calls=[call]), IX), judged(covered=1))
    assert [t for t, _ in got] == ["incomplete", "tool_misuse"] and "no call was needed" in got[1][1]


def test_without_a_verdict_the_judge_tags_cannot_fire():
    assert tags(PLAIN, resp(), P.Verdict("missing")) == []
    assert tags(PLAIN, resp(), P.Verdict("error")) == []


# ---------------------------------------------------------------- should-decline questions

def test_declines():
    no = resp(retrieved=("1005.6", OTHER_CHUNK), cites=(), abstained=True)
    assert tags(NETWORK, no) == []
    # decision 79: a correct decline that names an outside law is a memory_leak, whatever the words
    assert tags(NETWORK, no, judged("partially_faithful", unsupported=["governed by Subpart C"])) == ["memory_leak"]
    yes = resp(retrieved=("1005.6", OTHER_CHUNK), cites=("1005.6(b)(1)",))
    assert tags(NETWORK, yes, judged("unfaithful", unsupported=["Visa's zero liability"])) \
        == ["missed_abstention", "memory_leak"]
    m = R.grade(STATE, yes, IX)
    assert P.tags_for(STATE, m, judged())[0] == ("missed_abstention", "answered; should decline (state_law)")


# ---------------------------------------------------------------- a whole run

RUN = "r20990101T000000Z"


def add_verdict(db, qid, *, status="judged", label="faithful", covered=2, total=2, unsupported=(), version="j2"):
    claims = [{"claim": "ok", "supported": True, "source": "passage 1"}] + [
        {"claim": c, "supported": False, "source": None} for c in unsupported]
    vj = None if status != "judged" else json.dumps({
        "claims": claims, "faithfulness": label, "rationale": "why",
        "key_facts": [{"n": n, "covered": n <= covered} for n in range(1, total + 1)]})
    db.execute("INSERT INTO judgements (run_id, qid, judge_version, status, faithfulness, kf_covered, kf_total, "
               "verdict_json) VALUES (?,?,?,?,?,?,?,?)",
               (RUN, qid, version, status, label if status == "judged" else None,
                covered if status == "judged" else None, total if status == "judged" else None, vj))
    db.commit()


@pytest.fixture
def db(tmp_path):
    d = J.connect(tmp_path / "runs.db")
    d.execute("INSERT INTO runs (run_id, split, git_commit, git_dirty, config_json, prompt_version, grader_version, "
              "n_items, today_default) VALUES (?,?,?,?,?,?,?,?,?)",
              (RUN, "dev", "437f410abc", 0, json.dumps({"search_mode": "hybrid", "k": 5, "tool_on": True}), "p2",
               "g4", 6, "2026-11-03"))
    answers = {
        "q006": resp(),                                                                  # passes
        "q005": resp(retrieved=("1026.55(b)(3)", OTHER_CHUNK), cites=("1026.55(b)(3)",)),   # partial_retrieval
        "q004": tool_resp(answer="Investigate by November 20, 2026."),                    # calculation_error
        "q016": resp(retrieved=(OTHER_CHUNK,), cites=(), abstained=True, flags=["no_valid_citation"]),
        "q008": resp(retrieved=("1005.6",), cites=(), abstained=True),                     # memory_leak (decline)
        "q024": resp(retrieved=(OTHER_CHUNK,), cites=(), abstained=True),                  # passes
    }
    for qid, r in answers.items():
        item = BY_ID[qid]
        R.log_result(d, RUN, item, r, None, R.grade(item, r, IX))
    R.log_result(d, RUN, BY_ID["q049"], None, "ThrottlingException: slow down", None)
    add_verdict(d, "q006")
    add_verdict(d, "q005", label="partially_faithful", covered=1, unsupported=["the fee is $40"])
    add_verdict(d, "q004", covered=1)
    add_verdict(d, "q016", status="skipped")
    add_verdict(d, "q008", label="partially_faithful", unsupported=["Visa's zero liability covers her"])
    # q024: no verdict yet
    return d


def test_build_counts_slices_judge_and_tags(db):
    rep = P.build(db, RUN, ix=IX)
    o = rep["overall"]
    assert (o["n"], o["answerable"], o["declines"]) == (6, 4, 2)
    assert o["faithful"] == (2, 4)                 # judged: q006, q005, q004, q008; q016 skipped; q024 none
    assert o["key_facts"] == (2 + 1 + 1 + 0, 2 + 2 + 2 + 3)   # q016's fallback covers 0 of its 3
    assert o["failed"] == (4, 6) and o["abstain_correct"] == (5, 6)
    assert {f["id"]: f["primary"] for f in rep["failures"]} == {
        "q004": "calculation_error", "q005": "unsupported_claim", "q008": "memory_leak", "q016": "retrieval_miss"}
    assert rep["tag_counts"]["memory_leak"] == 1
    assert rep["declines_by_category"] == {"network_rules": (1, 1, 1), "state_law": (1, 1, 0)}
    assert (rep["by_persona"]["frontline"]["n"], rep["by_persona"]["compliance"]["n"]) == (3, 3)
    assert rep["no_verdict"] == [("q024", "missing")] and rep["not_run"][0][0] == "q049"
    assert rep["run_name"] == "hybrid k5"


def test_q005_tags_in_order(db):
    f = next(f for f in P.build(db, RUN, ix=IX)["failures"] if f["id"] == "q005")
    assert [t for t, _ in f["tags"]] == ["unsupported_claim", "partial_retrieval"]


def test_type_slices_follow_the_golden_order(db):
    rep = P.build(db, RUN, ix=IX)
    assert list(rep["by_type"]) == [t for t in G.TYPES if t in {"payments_disclosures", "fees_terms", "deadline",
                                                                 "decline"}]      # q049 (dispute) failed to run
    assert list(rep["by_persona"]) == ["frontline", "compliance"]


def test_an_earlier_rubric_reads_its_own_verdicts(db):
    add_verdict(db, "q006", label="partially_faithful", unsupported=["x"], version="j1")
    rep = P.build(db, RUN, version="j1", ix=IX)
    assert rep["overall"]["judged"] == 1 and "this report uses j1" in P.text(rep)


def test_text_and_markdown(db):
    rep = P.build(db, RUN, ix=IX)
    t, md = P.text(rep), P.markdown(rep)
    for out in (t, md):
        assert "upper bound" in out and "lower bound" in out and "q024 (missing)" in out
        assert "Graded again with g5 (the run was logged with g4)." in out
        assert "too few to compare" in out
    assert "| all | 6 |" in md and "| q008 | frontline | decline | memory_leak |" in md
    assert "4 of 6 questions failed" in t


def test_show(db):
    out = P.show(db, RUN, "q005")
    assert TWO_GOLD.question.strip()[:40] in out and "unsupported_claim, partial_retrieval" in out
    assert "partially_faithful" in out
    assert "Failed to run" in P.show(db, RUN, "q049")
    with pytest.raises(SystemExit):
        P.show(db, RUN, "q001")


# ---------------------------------------------------------------- the author's review

def write(path, rep, fill=None):
    text = P.review_sheet(rep)
    for qid, (tag, note) in (fill or {}).items():
        text = text.replace(f'id = "{qid}"', f'id = "{qid}"\n#x', 1)
        head, rest = text.split(f'id = "{qid}"\n#x', 1)
        rest = rest.replace('tag = ""', f'tag = "{tag}"', 1).replace('note = ""', f'note = "{note}"', 1)
        text = head + f'id = "{qid}"' + rest
    path.write_text(text, encoding="utf-8")
    return path


def test_review_sheet_lists_every_failure_with_its_automatic_tag(db, tmp_path):
    rep = P.build(db, RUN, ix=IX)
    sheet = P.review_sheet(rep)
    assert "For declines: missed_abstention, then memory_leak." in sheet
    blocks = tomllib.loads(sheet)["review"]
    assert [(b["id"], b["auto"], b["tag"]) for b in blocks] == [
        ("q004", "calculation_error", ""), ("q005", "unsupported_claim", ""),
        ("q008", "memory_leak", ""), ("q016", "retrieval_miss", "")]


def test_review_in_progress_then_complete(db, tmp_path):
    rep = P.build(db, RUN, ix=IX)
    path = write(tmp_path / "r.toml", rep, {"q004": ("calculation_error", "")})
    assert P.read_review(path, rep) == (None, "review in progress: 1 of 4 reviewed (r.toml)")
    path = write(tmp_path / "r.toml", rep, {"q004": ("calculation_error", ""), "q005": ("incomplete", "fee is in passage 2"),
                                            "q008": ("pass", "judge misread"), "q016": ("retrieval_miss", "")})
    review, state = P.read_review(path, rep)
    assert state == "reviewed by the author (r.toml)"
    done = P.build(db, RUN, review=review, ix=IX)
    assert done["review"]["counts"] == {"calculation_error": 1, "incomplete": 1, "retrieval_miss": 1}
    assert done["review"]["passed"] == ["q008"]
    assert done["review"]["changed"] == [("q005", "unsupported_claim", "incomplete", "fee is in passage 2"),
                                         ("q008", "memory_leak", "pass", "judge misread")]
    t = P.text(done)
    assert "after the review 3" in t and "q005: unsupported_claim -> incomplete (fee is in passage 2)" in t


def test_review_refuses_a_sheet_that_no_longer_matches(db, tmp_path):
    rep = P.build(db, RUN, ix=IX)
    path = write(tmp_path / "r.toml", rep)
    add_verdict(db, "q024", label="partially_faithful", unsupported=["state law gives more"])   # a new failure
    with pytest.raises(SystemExit, match="missing: q024"):
        P.read_review(path, P.build(db, RUN, ix=IX))
    bad = tmp_path / "bad.toml"
    bad.write_text(P.review_sheet(rep).replace('tag = ""', 'tag = "hallucination"', 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown tag 'hallucination'"):
        P.read_review(bad, rep)


def test_command_line_review_sheet_and_markdown(db, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(P, "REVIEW_DIR", tmp_path / "review")
    dbfile = tmp_path / "runs.db"
    assert P.main([RUN, "--review-sheet", "--db", str(dbfile)]) == 0
    sheet = tmp_path / "review" / f"{RUN}.toml"
    assert sheet.exists() and "4 failures to review" in capsys.readouterr().out
    assert P.main([RUN, "--review-sheet", "--db", str(dbfile)]) == 2          # never overwritten
    md = tmp_path / "new folder" / "report.md"
    assert P.main([RUN, "--db", str(dbfile), "--markdown", str(md)]) == 0
    out = capsys.readouterr().out
    assert "review in progress: 0 of 4 reviewed" in out and md.read_text(encoding="utf-8").startswith("# Evaluation report")
    assert P.main([RUN, "--db", str(tmp_path / "none.db")]) == 2
