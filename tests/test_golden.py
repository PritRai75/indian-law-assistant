"""The golden set file (eval/golden.toml) and its checker (eval/golden.py)."""
import tomllib

import pytest

from app.tools import dispute_timeline
from eval import golden as G

STATUSES = G.provision_statuses()


def items_from(toml_text):
    items, problems = G.parse(tomllib.loads(toml_text)["question"])
    return items, problems + G.check(items, STATUSES)


DRAFT = '''
[[question]]
id = "q900"
stage = "draft"
persona = "frontline"
type = "reg_e_account"
question = """A customer asks about her overdraft opt-in."""
'''

LABELLED = '''
[[question]]
id = "q901"
stage = "labelled"
persona = "compliance"
type = "reg_e_account"
question = """Under Reg E, what must the opt-in notice say?"""
regulation = "E"
gold_ids = ["1005.17(b)(1)", "1005.17(b)(1)(i)"]
reference_answer = """Two to four sentences."""
key_facts = ["one fact"]
should_abstain = false
'''


# ------------------------------------------------------------------ the real file

def test_golden_file_has_no_problems():
    items, problems = G.parse(G.read())
    problems += G.check(items, STATUSES)
    problems += [x for i in items for x in G.compare_with_tool(i)]
    assert problems == [], "\n".join(problems)
    assert len(items) >= 10


def test_load_returns_items():
    assert [i.id for i in G.load()][:3] == ["q001", "q002", "q003"]


def test_header_lists_every_type_and_category():
    header = G.GOLDEN_FILE.read_text(encoding="utf-8").split("\n[[question]]")[0]   # up to the first real block
    for name in (*G.TYPES, *G.ABSTAIN_CATEGORIES):
        assert name in header, name


# ------------------------------------------------------------------ drafts

def test_minimal_draft_is_fine():
    items, problems = items_from(DRAFT)
    assert problems == [] and items[0].question == "A customer asks about her overdraft opt-in."


def test_draft_with_labels_is_flagged():
    _, problems = items_from(DRAFT + 'gold_ids = ["1005.17(b)(1)"]\n')
    assert problems == ['q900: draft has pass-2 fields [\'gold_ids\']; set stage = "labelled" when you label it']


def test_misspelled_field_is_an_error():
    _, problems = items_from(DRAFT + 'gold_id = ["1005.17(b)(1)"]\n')
    assert len(problems) == 1 and "gold_id" in problems[0] and "Extra inputs" in problems[0]


def test_bad_values_are_named():
    bad = DRAFT.replace('"frontline"', '"lawyer"').replace('"q900"', '"Q900"')
    _, problems = items_from(bad)
    assert any(p.startswith("Q900: persona:") for p in problems)
    assert any(p.startswith("Q900: id:") for p in problems)


def test_repeated_id_and_question():
    _, problems = items_from(DRAFT + DRAFT)
    assert "q900: id used 2 times" in problems
    assert any(p.startswith("question used 2 times") for p in problems)


def test_unexpected_top_level_key(tmp_path):
    f = tmp_path / "g.toml"
    f.write_text('title = "x"\n' + DRAFT, encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected top-level keys"):
        G.read(f)


# ------------------------------------------------------------------ labelled

def test_valid_labelled_answerable():
    items, problems = items_from(LABELLED)
    assert problems == [] and items[0].gold_ids == ["1005.17(b)(1)", "1005.17(b)(1)(i)"]


def test_missing_label_fields():
    text = LABELLED.replace('key_facts = ["one fact"]\n', "").replace("should_abstain = false\n", "")
    _, problems = items_from(text)
    assert problems == ["q901: labelled but key_facts is missing", "q901: labelled but should_abstain is missing"]


def test_gold_id_spelling_and_typos():
    _, problems = items_from(LABELLED.replace('"1005.17(b)(1)(i)"', '"1005.17(B)(1)(i)"'))
    assert problems == ["q901: gold_ids: '1005.17(B)(1)(i)' is spelled '1005.17(b)(1)(i)' in the index"]
    _, problems = items_from(LABELLED.replace('"1005.17(b)(1)(i)"', '"1005.17(b)(1)(x)"'))
    assert len(problems) == 1 and "not a provision ID" in problems[0] and "did you mean" in problems[0]


def test_answerable_gold_id_must_be_in_force():
    _, problems = items_from(LABELLED.replace('"1005.17(b)(1)(i)"', '"1026.52(b)(3)"'))
    assert len(problems) == 1 and "status 'vacated'" in problems[0] and "not_in_force" in problems[0]


def test_answerable_needs_regulation_ids_and_facts():
    text = (LABELLED.replace('regulation = "E"\n', "")
            .replace('gold_ids = ["1005.17(b)(1)", "1005.17(b)(1)(i)"]', "gold_ids = []")
            .replace('key_facts = ["one fact"]', 'key_facts = ["a", "b", "c", "d"]'))
    _, problems = items_from(text)
    assert problems == ["q901: regulation (E, Z or both) is needed for an answerable question",
                        "q901: an answerable question needs at least one gold ID",
                        "q901: key_facts: 1 to 3 for an answerable question"]


def test_decline_rules():
    text = LABELLED.replace("should_abstain = false", "should_abstain = true")
    _, problems = items_from(text)
    assert 'q901: should_abstain is true, so type must be "decline"' in problems
    assert any("abstain_category is needed" in p for p in problems)
    ok = (text.replace('type = "reg_e_account"', 'type = "decline"')
          .replace('regulation = "E"\n', 'abstain_category = "network_rules"\n')
          .replace('gold_ids = ["1005.17(b)(1)", "1005.17(b)(1)(i)"]', "gold_ids = []")
          .replace('key_facts = ["one fact"]', "key_facts = []"))
    assert items_from(ok)[1] == []


def test_decline_type_needs_should_abstain():
    _, problems = items_from(LABELLED.replace('type = "reg_e_account"', 'type = "decline"'))
    assert problems == ['q901: type is "decline" but should_abstain is false']


# ------------------------------------------------------------------ tool labels

CALL = '''
[question.expected_tool_call]
name = "dispute_timeline"
args = { regulation = "E", event = "error", notice_date = 2026-12-01, transfer_kind = "pos_debit" }
'''


def labelled_with_tool(result_toml):
    base = LABELLED.replace('type = "reg_e_account"', 'type = "deadline"')
    return base + CALL + "\n[question.expected_result]\n" + result_toml


def tool_result():
    return dispute_timeline(regulation="E", event="error", notice_date="2026-12-01", transfer_kind="pos_debit")


def as_toml(result, change=None):
    rows = []
    for d in result["deadlines"]:
        rule = d["rule"].split(",")[0]
        day = change if change and rule == "1005.11(c)(3)(ii)" else d["date"]
        rows.append(f'  {{ what = "x", date = {day}, rule = "{rule}" }},')
    return "deadlines = [\n" + "\n".join(rows) + "\n]\n"


def test_call_and_result_go_together():
    _, problems = items_from(LABELLED + CALL)
    assert problems == ["q901: expected_tool_call and expected_result go together: give both or neither"]


def test_tool_arguments_are_checked():
    bad = CALL.replace('transfer_kind = "pos_debit"', 'transfer_kind = "debit", colour = "red"')
    _, problems = items_from(LABELLED + bad + "\n[question.expected_result]\n")
    assert "q901: expected_tool_call: unknown argument 'colour'; known: " in problems[0]
    assert any("transfer_kind must be one of" in p for p in problems)
    missing = CALL.replace("notice_date = 2026-12-01, ", "")
    _, problems = items_from(LABELLED + missing + "\n[question.expected_result]\n")
    assert problems == ["q901: expected_tool_call: missing required argument 'notice_date'"]


def test_tool_rejecting_the_arguments_is_reported():
    rejected = CALL.replace('transfer_kind = "pos_debit"', 'transfer_kind = "pos_debit", learned_of_loss = 2026-11-30')
    _, problems = items_from(LABELLED + rejected + "\n[question.expected_result]\n")
    assert len(problems) == 1 and "the tool rejects these arguments (learned_of_loss:" in problems[0]


def test_deadline_rule_must_be_a_provision_id():
    text = labelled_with_tool('deadlines = [ { what = "x", date = 2026-12-15, rule = "1005.11(c)(9)" } ]\n')
    _, problems = items_from(text)
    assert len(problems) == 1 and problems[0].startswith("q901: expected_result deadline 2026-12-15:")


def test_label_matching_the_tool_passes_and_a_wrong_date_is_caught():
    items, problems = items_from(labelled_with_tool(as_toml(tool_result())))
    assert problems == [] and G.compare_with_tool(items[0]) == []
    items, _ = items_from(labelled_with_tool(as_toml(tool_result(), change="2027-01-01")))
    (mismatch,) = G.compare_with_tool(items[0])
    assert mismatch.startswith("q901: deadline 1005.11(c)(3)(ii): label 2027-01-01, tool ")


def test_wrong_tier_is_caught():
    items, _ = items_from(labelled_with_tool('liability_tiers = ["max_50"]\n'))
    assert G.compare_with_tool(items[0]) == ["q901: liability_tiers: label ['max_50'], tool None"]


def test_rule_matches_combined_rule_fields():
    assert G.rule_matches("1005.11(c)(2)(i)", "1005.11(c)(2)(i), (c)(3)(i)")
    assert G.rule_matches("1005.11(c)(3)(i)", "1005.11(c)(2)(i), (c)(3)(i)")
    assert not G.rule_matches("1005.11(c)(2)", "1005.11(c)(2)(i), (c)(3)(i)")
    assert G.rule_matches("1026.13(c)(2)", "1026.13(c)(2)")


def test_summary_counts():
    items, _ = items_from(DRAFT + LABELLED)
    text = G.summary(items)
    assert text.startswith("2 questions: 1 draft, 1 labelled")
    assert "labelled: 1 answerable, 0 declines, 0 uncertain, 0 expecting the tool" in text
