"""Tests for app/answer.py. No AWS call: a fake Bedrock client plays back scripted model replies, and
search gets a ready-made question vector."""
import copy
import json
from datetime import date
from pathlib import Path

import pytest

from app import answer as an
from app import retrieve as rt
from app.config import DEFAULT
from app.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from app.schema import ANSWER_JSON_SCHEMA
from app.tools import TOOL_SPEC

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "index" / "embeddings.npy").exists(), reason="index not built yet")

Q = "Under Reg E 6(b), what is the most a customer can lose?"
TODAY = date(2026, 11, 3)


class FakeClient:
    """Bedrock stand-in: returns the scripted replies in order and keeps a copy of each request."""
    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def converse(self, **request):
        self.requests.append(copy.deepcopy(request))
        return self.replies.pop(0)


def text(body, stop="end_turn"):
    body = body if isinstance(body, str) else json.dumps(body)
    return {"output": {"message": {"role": "assistant", "content": [{"text": body}]}},
            "stopReason": stop, "usage": {"inputTokens": 1000, "outputTokens": 50}}


def tool(args, use_id="t1", name="dispute_timeline"):
    return {"output": {"message": {"role": "assistant", "content": [
                {"text": "Let me work out the dates."},
                {"toolUse": {"toolUseId": use_id, "name": name, "input": args}}]}},
            "stopReason": "tool_use", "usage": {"inputTokens": 1200, "outputTokens": 80}}


def reply(answer="At most $50 if reported within 2 business days [§ 1005.6(b)(1)].",
          citations=("1005.6(b)(1)",), abstained=False, reason=None):
    return {"answer": answer, "citations": list(citations), "abstained": abstained, "abstain_reason": reason}


def run(*replies, config=DEFAULT, question=Q):
    client = FakeClient(*replies)
    chunks, vecs, _, _ = rt.load()
    vec = vecs[[c["id"] for c in chunks].index("1005.6(b)-cmt")]
    return an.answer(question, TODAY, config=config, client=client, query_vec=vec), client


# ---------------------------------------------------------------- the request

def test_request_has_prompt_tool_and_settings():
    resp, client = run(text(reply()))
    req = client.requests[0]
    assert req["modelId"] == "global.anthropic.claude-sonnet-4-6"
    assert req["system"] == [{"text": SYSTEM_PROMPT}]
    assert req["toolConfig"] == {"tools": [TOOL_SPEC]}
    assert req["inferenceConfig"] == {"temperature": 0.0, "maxTokens": 1024}
    assert "outputConfig" not in req
    user = req["messages"][0]["content"][0]["text"]
    assert user.startswith("<passages>") and "Today's date: 2026-11-03 (Tuesday)" in user and user.endswith(f"{Q}\n</question>")
    assert resp.retrieved_ids[0].startswith("1005.6")      # the named paragraph comes first


def test_structured_mode_sends_the_schema():
    _, client = run(text(reply()), config=DEFAULT.with_(output_mode="structured"))
    fmt = client.requests[0]["outputConfig"]["textFormat"]
    assert fmt["type"] == "json_schema"
    assert json.loads(fmt["structure"]["jsonSchema"]["schema"]) == ANSWER_JSON_SCHEMA


def test_tool_off_sends_no_tool():
    _, client = run(text(reply()), config=DEFAULT.with_(tool_on=False))
    assert "toolConfig" not in client.requests[0]


# ---------------------------------------------------------------- a plain answer

def test_plain_answer():
    resp, _ = run(text(reply()))
    assert resp.answer == "At most $50 if reported within 2 business days [§ 1005.6(b)(1)]."
    assert resp.citations == ["1005.6(b)(1)"] and resp.raw_citations == ["1005.6(b)(1)"]
    assert not resp.abstained and resp.abstain_reason is None and resp.flags == []
    assert len(resp.retrieved_ids) == 5 and resp.regulations_as_of == "2026-10-23"
    assert resp.prompt_version == PROMPT_VERSION and resp.model_calls == 1
    assert resp.usage == {"inputTokens": 1000, "outputTokens": 50}
    assert resp.disclaimer.startswith("General information") and len(resp.config_hash) == 12


def test_code_fence_is_tolerated():
    resp, _ = run(text("```json\n" + json.dumps(reply()) + "\n```"))
    assert resp.flags == [] and resp.citations == ["1005.6(b)(1)"]


# ---------------------------------------------------------------- the tool loop

ARGS = {"regulation": "E", "event": "unauthorized", "notice_date": "2026-11-03", "learned_of_loss": "2026-10-30"}


def test_tool_call_runs_and_its_result_goes_back():
    final = reply("Liability is at most $50 [§ 1005.6(b)(1)]; deadlines count business days [§ 1005.2(d)].",
                  ["1005.6(b)(1)", "1005.2(d)"])
    resp, client = run(tool(ARGS), text(final))
    assert len(client.requests) == 2 and resp.model_calls == 2
    assert [c.name for c in resp.tool_calls] == ["dispute_timeline"] and resp.tool_calls[0].args == ARGS
    assert resp.tool_calls[0].result["liability_tiers"] == ["max_50"]
    sent = client.requests[1]["messages"]
    assert "toolUse" in sent[1]["content"][1]                              # the model's own turn, kept
    result = sent[2]["content"][0]["toolResult"]
    assert result["toolUseId"] == "t1" and result["status"] == "success"
    assert result["content"][0]["json"]["liability_tiers"] == ["max_50"]
    # 1005.2(d) is in no retrieved chunk but is named in the tool result, so it is allowed
    assert "1005.2(d)" not in {p for r in rt.search(Q, 5, "bm25") for p in r["chunk"]["provision_ids"]}
    assert resp.citations == ["1005.6(b)(1)", "1005.2(d)"] and resp.flags == []
    assert resp.usage == {"inputTokens": 2200, "outputTokens": 130}


def test_tool_input_errors_go_back_to_the_model():
    bad = {**ARGS, "regulation": "Z"}                       # learned_of_loss is Reg E only
    resp, client = run(tool(bad), tool({**ARGS, "colour": "red"}, "t2"), text(reply()))
    statuses = [m["content"][0]["toolResult"]["status"] for m in client.requests[2]["messages"][2::2]]
    assert statuses == ["error", "error"]
    assert resp.tool_calls[0].result["error"]["argument"] == "learned_of_loss"
    assert "colour" in resp.tool_calls[1].result["error"]["message"]
    assert an.run_tool("other_tool", {})["error"]["argument"] == "name"


def test_tool_limit_ends_in_an_abstention():
    resp, client = run(tool(ARGS, "t1"), tool(ARGS, "t2"), tool(ARGS, "t3"))
    assert resp.model_calls == 3 and len(resp.tool_calls) == 2       # the third request is not run
    assert resp.abstained and resp.flags == ["tool_limit"] and resp.citations == []
    assert resp.raw_answer is None


# ---------------------------------------------------------------- repair try

def test_bad_reply_gets_one_repair_try():
    resp, client = run(text("The answer is $50."), text(reply()))
    assert resp.flags == ["output_repaired"] and resp.citations == ["1005.6(b)(1)"] and resp.model_calls == 2
    repair = client.requests[1]["messages"][-1]["content"][0]["text"]
    assert repair.startswith("Your last reply was not a valid answer object (no JSON object found)")


def test_extra_or_missing_fields_are_bad_replies():
    resp, client = run(text({**reply(), "confidence": "high"}), text({"answer": "x"}))
    assert "confidence" in client.requests[1]["messages"][-1]["content"][0]["text"]
    assert resp.flags == ["output_invalid"] and resp.abstained and resp.answer == an.OUTPUT_INVALID


# ---------------------------------------------------------------- citation guard

def test_invented_citation_is_removed_and_flagged():
    final = reply("Cap is $50 [§ 1005.6(b)(1)] and see [§ 1026.99(z)] and [comment 1005.6(b)-1].",
                  ["§ 1005.6(b)(1)", "1026.99(z)"])
    resp, _ = run(text(final))
    assert resp.answer == "Cap is $50 [§ 1005.6(b)(1)] and see [citation removed] and [comment 1005.6(b)-1]."
    assert resp.citations == ["1005.6(b)(1)", "1005.6(b)-1"]                # inline comment counted too
    assert resp.raw_citations == ["§ 1005.6(b)(1)", "1026.99(z)", "1005.6(b)(1)", "1005.6(b)-1"]
    assert resp.invalid_citations == ["1026.99(z)"] and resp.flags == ["invalid_citations"]
    assert not resp.abstained


def test_answer_with_no_valid_citation_becomes_an_abstention():
    resp, _ = run(text(reply("Visa says zero liability.", ["visa-rules"])))
    assert resp.abstained and resp.answer == an.NO_CITATION and resp.citations == []
    assert resp.raw_answer == "Visa says zero liability." and resp.flags == ["invalid_citations", "no_valid_citation"]


def test_model_abstention_without_citations_stays_as_it_is():
    resp, _ = run(text(reply("Network rules are outside the regulations I can answer from.", [],
                             abstained=True, reason="card network rules")))
    assert resp.abstained and resp.abstain_reason == "card network rules" and resp.flags == []


def test_the_guard_reads_every_spelling_of_one_id():          # was test_normalize_id (guard c1)
    from app.citations import read_citation
    prov = an.index_spelling()
    assert read_citation("§ 1005.6(b)(1)", prov).ids == ["1005.6(b)(1)"]
    assert read_citation("comment 1005.6(b)-1", prov).ids == ["1005.6(b)-1"]
    assert read_citation("12 CFR 1026.52(b)(1)(ii)(A)", prov).ids == ["1026.52(b)(1)(ii)(A)"]
    assert read_citation(" §§1026.13 (c)(2) ", prov).ids == ["1026.13(c)(2)"]


# ---------------------------------------------------------------- input

def test_empty_or_long_question_is_refused():
    with pytest.raises(ValueError):
        an.answer("   ", TODAY, client=FakeClient())
    with pytest.raises(ValueError):
        an.answer("x" * 2001, TODAY, client=FakeClient())


# ---------------------------------------------------------------- guard c2: every citation form (session 15)

def test_packed_range_and_subpoint_citations_are_kept():
    final = reply("Cap is $50 [§ 1005.6(b)(1), § 1005.6(b)(2)], tiers in [§ 1005.6(b)(1)-(3)], "
                  "example in [comment 1005.6(b)(1)-3.ii].",
                  ["1005.6(b)(1); § 1005.6(b)(2)"])
    resp, _ = run(text(final))
    assert "[citation removed]" not in resp.answer and resp.answer == final["answer"]
    assert resp.citations == ["1005.6(b)(1)", "1005.6(b)(2)", "1005.6(b)(3)", "1005.6(b)(1)-3"]
    assert resp.invalid_citations == [] and resp.flags == [] and not resp.abstained
    assert resp.raw_citations == ["1005.6(b)(1); § 1005.6(b)(2)", "1005.6(b)(1), § 1005.6(b)(2)",
                                  "1005.6(b)(1)-(3)", "1005.6(b)(1)-3.ii"]        # raw strings, as written


def test_marker_with_some_valid_ids_keeps_only_those():
    final = reply("Cap is $50 [§ 1005.6(b)(1), § 1005.11(c)(1)] and [comment 1005.6(b)-1; § 1026.99(z)].",
                  ["1005.6(b)(1)"])
    resp, _ = run(text(final))
    assert resp.answer == "Cap is $50 [§ 1005.6(b)(1)] and [comment 1005.6(b)-1]."
    assert resp.invalid_citations == ["1005.11(c)(1)", "1026.99(z)"]           # real but not retrieved; no such ID
    assert resp.flags == ["invalid_citations"] and not resp.abstained


def test_whole_section_citation_is_too_broad_to_verify():
    resp, _ = run(text(reply("Cap is $50 [§ 1005.6].", ["1005.6"])))
    assert resp.invalid_citations == ["1005.6"] and resp.answer == an.NO_CITATION and resp.abstained


def test_guard_version_is_in_the_config_hash(monkeypatch):
    from app import config as cfg
    before = cfg.config_hash(DEFAULT, "2026-10-23")
    monkeypatch.setattr(cfg, "GUARD_VERSION", "c-test")
    assert cfg.config_hash(DEFAULT, "2026-10-23") != before
