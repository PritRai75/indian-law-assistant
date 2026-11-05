"""Tests for eval/judge.py and the --k / --tool-off run flags. No AWS call: a fake Bedrock client
replies with made-up verdicts."""
import hashlib
import json
import re
from collections import Counter
from datetime import date
from types import SimpleNamespace

import pytest

from app.config import DEFAULT as CONFIG
from eval import golden as G
from eval import judge as J
from eval import run as R

# A rubric edit changes verdicts, so it must get a new JUDGE_VERSION. If this fails after an intended
# edit: bump JUDGE_VERSION in eval/judge.py and record the new version's hash here.
RUBRIC_HASHES = {
    "j1": "3cdb949fd58fc0b28f651ea63ae0583b24e936d50ccc5d631db0d93d18cd62da",   # S19: first rubric
}


def test_rubric_text_matches_its_version():
    digest = hashlib.sha256(J.JUDGE_PROMPT.encode("utf-8")).hexdigest()
    assert RUBRIC_HASHES.get(J.JUDGE_VERSION) == digest, (
        f"JUDGE_PROMPT changed: bump JUDGE_VERSION and record {digest} for it")


# ---------------------------------------------------------------- key facts without citations

ITEMS = G.load()
BY_ID = {i.id: i for i in ITEMS}
FACTS = [(i.id, f) for i in ITEMS for f in (i.key_facts or [])]
CITATION = re.compile(r"§|10(?:05|26)\.\d")
# Words a removed citation may contain: never a word of the fact itself.
CITATION_WORDS = {"under", "per", "pursuant", "to", "see", "and", "through", "cfr", "regulation", "e", "z",
                  "comment", "comments"}


def words(text):
    return Counter(w.lower() for w in re.findall(r"[A-Za-z]+|\d+", text))


@pytest.mark.parametrize("qid,fact", FACTS)
def test_strip_removes_only_citation_words(qid, fact):
    removed = words(fact) - words(J.strip_citations(fact))
    odd = [w for w in removed if not (w in CITATION_WORDS or w.isdigit() or len(w) == 1
                                      or re.fullmatch(r"[ivx]+", w))]
    assert not odd, f"{qid}: stripping removed {odd} from {fact!r}"


def test_strip_leaves_only_the_in_sentence_citations():
    left = sorted({q for q, f in FACTS if CITATION.search(J.strip_citations(f))})
    assert left == ["q028", "q045"]          # "12 CFR § 1005.7(a) governs ...": the rubric covers these
    changed = sum(J.strip_citations(f) != f for _, f in FACTS)
    assert changed == 41


@pytest.mark.parametrize("fact,want", [
    ("The bank must resolve it within two billing cycles (§ 1026.13(c)(2)).",
     "The bank must resolve it within two billing cycles."),
    ("may withhold payment under 12 CFR § 1026.12(c)", "may withhold payment"),
    ("Regulation Z (§ 1026.13) governs billing error rights.", "Regulation Z governs billing error rights."),
    ("cannot increase liability (Comment 1005.6(b)-2).", "cannot increase liability."),
    ("governed by Subpart F (12 CFR §§ 1026.46–1026.48).", "governed by Subpart F."),
    ("investigate within 10 business days (20 November 2026)", "investigate within 10 business days (20 November 2026)"),
    ("no citation here", "no citation here"),
])
def test_strip_examples(fact, want):
    assert J.strip_citations(fact) == want


# ---------------------------------------------------------------- the judge's message

CHUNKS = J.chunk_lookup()
SOME_CHUNKS = list(CHUNKS)[:3]


def response(**kw):
    r = {"answer": "The bank has 10 business days [§ 1005.11(c)(1)].", "abstained": False,
         "retrieved_ids": SOME_CHUNKS, "flags": [],
         "tool_calls": [{"name": "dispute_timeline", "args": {"regulation": "E"}, "result": {"deadlines": []}}]}
    r.update(kw)
    return r


def test_message_has_everything_the_spec_lists():
    item = BY_ID["q004"]
    msg = J.judge_message(item, response(), date(2026, 11, 5), CHUNKS)
    assert item.question.strip() in msg
    assert "Today's date: 2026-11-05 (Thursday)" in msg
    positions = [msg.index(f'n="{n}" chunk="{cid}"') for n, cid in enumerate(SOME_CHUNKS, 1)]
    assert positions == sorted(positions)                     # passages in the assistant's order
    assert '"tool": "dispute_timeline"' in msg and '"result": {"deadlines": []}' in msg
    assert '<answer declined="no">' in msg and "[§ 1005.11(c)(1)]" in msg
    assert item.reference_answer.strip() in msg
    for n, fact in enumerate(item.key_facts, 1):
        assert f"{n}. {J.strip_citations(fact)}" in msg


def test_message_strips_key_fact_citations_and_marks_declines():
    msg = J.judge_message(BY_ID["q038"], response(abstained=True, tool_calls=[]), date(2026, 11, 3), CHUNKS)
    facts = msg.split("<key_facts>")[1]
    assert "§" not in facts
    assert '<answer declined="yes">' in msg and "(no tool calls)" in msg


def test_message_refuses_a_chunk_missing_from_the_index():
    with pytest.raises(ValueError, match="not in the current index"):
        J.judge_message(BY_ID["q004"], response(retrieved_ids=["9999.1(a)"]), date(2026, 11, 3), CHUNKS)


# ---------------------------------------------------------------- reading the verdict

def verdict(n_facts=2, label="faithful", supported=(True,), covered=None, **extra):
    v = {"claims": [{"claim": f"claim {i}", "supported": s, "source": "passage 1" if s else None}
                    for i, s in enumerate(supported)],
         "faithfulness": label,
         "key_facts": [{"n": n, "covered": c} for n, c in enumerate(covered or [True] * n_facts, 1)],
         "rationale": "why"}
    v.update(extra)
    return json.dumps(v)


def test_parse_good_verdict_and_code_fence():
    assert J.parse_verdict(verdict(), 2).faithfulness == "faithful"
    assert J.parse_verdict("```json\n" + verdict() + "\n```", 2).key_facts[1].n == 2
    assert J.parse_verdict(verdict(supported=()), 2).claims == []        # no claims: faithful


@pytest.mark.parametrize("text,n,reason", [
    ("not json", 2, "no JSON object"),
    (verdict(n_facts=1), 2, "facts 1 to 2"),
    (verdict(label="partially_faithful"), 2, "must be faithful"),
    (verdict(supported=(True, False)), 2, "can't be faithful"),
    (verdict(label="mostly"), 2, "faithfulness"),
    (verdict(extra_field=1), 2, "extra_field"),
])
def test_parse_rejects(text, n, reason):
    with pytest.raises(ValueError, match=reason):
        J.parse_verdict(text, n)


# ---------------------------------------------------------------- calling the judge (fake Bedrock)

class FakeBedrock:
    """Replies with the given texts in turn and records every request."""

    def __init__(self, *texts):
        self.texts, self.requests = list(texts), []

    def converse(self, **req):
        self.requests.append(json.loads(json.dumps(req)))
        return {"output": {"message": {"role": "assistant", "content": [{"text": self.texts.pop(0)}]}},
                "usage": {"inputTokens": 100, "outputTokens": 20}, "stopReason": "end_turn"}




def test_judge_one_uses_the_judge_model_at_temperature_zero():
    fake = FakeBedrock(verdict())
    out = J.judge_one("msg", 2, config=CONFIG, client=fake)
    assert out["verdict"].faithfulness == "faithful" and out["error"] is None and out["calls"] == 1
    req = fake.requests[0]
    assert req["modelId"] == CONFIG.judge_model != CONFIG.generator_model
    assert req["inferenceConfig"]["temperature"] == 0.0
    assert req["system"][0]["text"] == J.JUDGE_PROMPT and "toolConfig" not in req


def test_judge_one_repairs_once():
    fake = FakeBedrock("oops", verdict())
    out = J.judge_one("msg", 2, config=CONFIG, client=fake)
    assert out["verdict"] and out["calls"] == 2 and out["usage"] == {"inputTokens": 200, "outputTokens": 40}
    assert "could not be used (no JSON object found)" in fake.requests[1]["messages"][-1]["content"][0]["text"]


def test_judge_one_gives_up_after_the_repair():
    out = J.judge_one("msg", 2, config=CONFIG, client=FakeBedrock("oops", verdict(n_facts=1)))
    assert out["verdict"] is None and out["calls"] == 2
    assert out["error"].startswith("no usable verdict after one repair try: key_facts")


# ---------------------------------------------------------------- judging a logged run

RUN = "r20990101T000000Z"


@pytest.fixture
def db(tmp_path):
    d = J.connect(tmp_path / "runs.db")
    d.execute("INSERT INTO runs (run_id, today_default, split) VALUES (?, ?, ?)", (RUN, "2026-11-03", "dev"))
    for qid, resp in (("q004", response()), ("q009", response(abstained=True, tool_calls=[])),
                      ("q038", response(flags=["invalid_citations", "no_valid_citation"], abstained=True))):
        item = BY_ID[qid]
        R.log_result(d, RUN, item, resp, None, {"should_abstain": item.should_abstain})
    R.log_result(d, RUN, BY_ID["q040"], None, "ThrottlingException: slow down", None)   # failed to run
    return d


def test_judge_run_judges_skips_and_summarises(db):
    fake = FakeBedrock(verdict(covered=[True, False]),
                       verdict(label="partially_faithful", supported=(True, False)))
    lines = []
    J.judge_run(db, RUN, config=CONFIG, client=fake, chunks=CHUNKS, progress=lines.append)
    assert len(fake.requests) == 2                         # q004, q009; q038 skipped; q040 never answered
    assert "skipped (no_valid_citation)" in lines[2]
    assert "Today's date: 2026-11-05" in fake.requests[0]["messages"][0]["content"][0]["text"]   # q004's own today
    assert "Today's date: 2026-11-03" in fake.requests[1]["messages"][0]["content"][0]["text"]   # run default
    s = J.summary(db, RUN)
    o = s["overall"]
    assert (o["n"], o["judged"], o["skipped"], o["errors"]) == (3, 2, 1, 0)
    assert (o["faithful"], o["partially_faithful"], o["unfaithful"]) == (1, 1, 0)
    assert o["key_facts"] == (3, 4) and o["key_fact_share_mean"] == 0.75
    assert s["answerable"]["judged"] == 1 and s["declines"]["judged"] == 1
    assert s["skipped"] == [("q038", "no_valid_citation")]
    text = J.report(s)
    assert "faithful 1, partially 1, unfaithful 0" in text and "q038 no_valid_citation" in text


def test_judge_run_does_not_judge_twice_unless_asked(db):
    J.judge_run(db, RUN, config=CONFIG, client=FakeBedrock(verdict(), verdict()), chunks=CHUNKS, progress=lambda _: None)
    lines = []
    J.judge_run(db, RUN, config=CONFIG, client=FakeBedrock(), chunks=CHUNKS, progress=lines.append)
    assert lines and lines[0].startswith("Nothing to judge")
    fake = FakeBedrock(verdict(label="unfaithful", supported=(False,)))
    J.judge_run(db, RUN, config=CONFIG, client=fake, ids=["q004"], rejudge=True, chunks=CHUNKS, progress=lambda _: None)
    assert J.summary(db, RUN)["overall"]["unfaithful"] == 1
    assert db.execute("SELECT COUNT(*) FROM judgements").fetchone()[0] == 3


def test_judge_run_logs_an_unusable_verdict_as_an_error(db):
    J.judge_run(db, RUN, config=CONFIG, client=FakeBedrock("x", "y", verdict(), verdict()), chunks=CHUNKS,
                progress=lambda _: None)
    s = J.summary(db, RUN)
    assert s["errors"] == ["q004"] and s["overall"]["judged"] == 1


def test_show_prints_claims_and_key_facts(db):
    J.judge_run(db, RUN, config=CONFIG, client=FakeBedrock(verdict(covered=[True, False]), verdict()),
                chunks=CHUNKS, progress=lambda _: None)
    text = J.show(db, RUN, "q004")
    assert text.startswith("q004: faithful. why")
    assert "[covered]" in text and "[NOT covered]" in text and BY_ID["q004"].key_facts[1][:30] in text
    assert J.show(db, RUN, "q038") == "q038: skipped (no_valid_citation)"


def test_unknown_run_or_question(db):
    with pytest.raises(SystemExit, match="no run"):
        J.judge_run(db, "r0", config=CONFIG, client=FakeBedrock(), chunks=CHUNKS)
    with pytest.raises(SystemExit, match="not in run"):
        J.judge_run(db, RUN, config=CONFIG, client=FakeBedrock(), ids=["q059"], chunks=CHUNKS)


def test_smoke_example_uses_no_golden_question():
    fake = FakeBedrock(verdict(n_facts=1, label="partially_faithful", supported=(True, False)))
    out = J.smoke(config=CONFIG, client=fake)
    assert out.startswith("The judge works. Verdict on the made-up example: partially_faithful")
    sent = fake.requests[0]["messages"][0]["content"][0]["text"]
    assert all(i.question.strip() not in sent for i in ITEMS)
    assert "1005.6(b)(1)" in sent


# ---------------------------------------------------------------- eval.run --k / --tool-off

def args(k=None, tool_off=False):
    return SimpleNamespace(k=k, tool_off=tool_off)


def test_run_flags_change_and_log_the_config():
    from app.config import DEFAULT, config_hash
    weak = R.config_from_args(args(k=2, tool_off=True))
    assert (weak.k, weak.tool_on) == (2, False) and weak.judge_model == DEFAULT.judge_model
    assert R.config_from_args(args()) == DEFAULT
    assert config_hash(weak, "2026-10-23") != config_hash(DEFAULT, "2026-10-23")
    with pytest.raises(SystemExit, match="at least 1"):
        R.config_from_args(args(k=0))
