"""Tests for app/prompts.py. No AWS call."""
import hashlib
from datetime import date
from pathlib import Path

import pytest

from app import prompts as pr
from app.tools import TOOL_SPEC

ROOT = Path(__file__).resolve().parents[1]
needs_index = pytest.mark.skipif(not (ROOT / "index" / "chunks.json").exists(), reason="index not built yet")

# Changing the prompt changes eval results, so it must get a new PROMPT_VERSION. If this fails after
# an intended edit: bump PROMPT_VERSION in app/prompts.py and add the new version's hash here.
PROMPT_HASHES = {
    "p1": "f64d0e22177ff96a184f68bb713c85ab93ba1aba591131784f43bf8a52dee426",
}


def test_prompt_text_matches_its_version():
    digest = hashlib.sha256(pr.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert PROMPT_HASHES.get(pr.PROMPT_VERSION) == digest, (
        f"SYSTEM_PROMPT changed: bump PROMPT_VERSION and record {digest} for it")


def test_prompt_names_the_real_tool():
    assert TOOL_SPEC["toolSpec"]["name"] in pr.SYSTEM_PROMPT


def test_prompt_describes_every_answer_field():
    for field in ("answer:", "citations:", "abstained:", "abstain_reason:"):
        assert f"- {field}" in pr.SYSTEM_PROMPT


def fake_chunk(**kw):
    c = {"id": "1005.6(b)-cmt", "kind": "commentary", "provision_ids": ["1005.6(b)-1", "1005.6(b)-2"],
         "embed_text": "Regulation E ...\nStatus: in force\n\n1. Text."}
    c.update(kw)
    return c


def test_passage_shows_id_source_provisions_and_header():
    p = pr.format_passage(2, fake_chunk())
    assert p.startswith('<passage n="2" chunk="1005.6(b)-cmt" source="official interpretation">')
    assert "Provisions: 1005.6(b)-1, 1005.6(b)-2\n" in p
    assert "Status: in force" in p and p.endswith("</passage>")
    rule = pr.format_passage(1, fake_chunk(id="1005.6(b)", kind="rule", provision_ids=["1005.6(b)(1)"]))
    assert 'source="regulation"' in rule


def test_user_message_order_and_date():
    results = [{"rank": 1, "chunk": fake_chunk()}, {"rank": 2, "chunk": fake_chunk(id="x", provision_ids=["x"])}]
    msg = pr.build_user_message("  Card stolen Friday, called Tuesday?  ", results, date(2026, 11, 3))
    assert msg.index("<passages>") < msg.index('n="1"') < msg.index('n="2"') < msg.index("Today's date")
    assert "Today's date: 2026-11-03 (Tuesday)" in msg
    assert msg.endswith("<question>\nCard stolen Friday, called Tuesday?\n</question>")


def test_today_must_be_a_date():
    with pytest.raises(TypeError):
        pr.build_user_message("q", [], "2026-11-03")


# ---------------------------------------------------------------- the prompt matches the real chunk headers

@pytest.fixture(scope="module")
def chunks():
    from app.retrieve import load
    return load()[0]


@needs_index
def test_status_words_in_the_prompt_are_the_ones_headers_use(chunks):
    words = {"vacated": "VACATED", "nullified": "OVERTURNED BY CONGRESS", "mixed": "PARTLY NOT IN FORCE"}
    for w in words.values():
        assert w in pr.SYSTEM_PROMPT
    for c in chunks:
        status_line = next(l for l in c["embed_text"].splitlines() if l.startswith("Status: "))
        if c["status"] == "in_force":
            assert status_line == "Status: in force", c["id"]
        else:
            assert status_line.startswith(f"Status: {words[c['status']]}"), c["id"]


@needs_index
def test_plan_type_line_named_in_the_prompt_exists(chunks):
    lines = {l for c in chunks for l in c["embed_text"].splitlines() if l.startswith("Applies to: ")}
    assert any("home-equity plans" in l for l in lines)
    assert any("open-end (not home-secured) plans" in l for l in lines)
    assert '"Applies to:"' in pr.SYSTEM_PROMPT


@needs_index
def test_example_ids_in_the_prompt_are_real(chunks):
    ids = {p for c in chunks for p in c["provision_ids"]}
    assert {"1005.6(b)(1)", "1005.6(b)-1"} <= ids


@needs_index
def test_every_passage_has_provisions_and_a_status_line(chunks):
    for i, c in enumerate(chunks):
        p = pr.format_passage(i + 1, c)
        assert "Provisions: " + c["provision_ids"][0] in p and "\nStatus: " in p, c["id"]
