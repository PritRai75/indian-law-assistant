"""Tests for app/schema.py and app/config.py. No AWS call."""
import pytest
from pydantic import ValidationError

from app.config import DEFAULT, Config, config_hash
from app.schema import ANSWER_JSON_SCHEMA, FLAGS, ModelAnswer

GOOD = {"answer": "x", "citations": ["1005.6(b)(1)"], "abstained": False, "abstain_reason": None}


def test_model_answer_accepts_the_four_fields():
    assert ModelAnswer.model_validate(GOOD).citations == ["1005.6(b)(1)"]


@pytest.mark.parametrize("bad", [
    {**GOOD, "extra": 1},                         # unknown field
    {k: v for k, v in GOOD.items() if k != "abstain_reason"},   # missing field (null must be explicit)
    {**GOOD, "answer": ""},
    {**GOOD, "citations": "1005.6(b)(1)"},
])
def test_model_answer_rejects(bad):
    with pytest.raises(ValidationError):
        ModelAnswer.model_validate(bad)


def test_json_schema_matches_the_pydantic_model():
    assert set(ANSWER_JSON_SCHEMA["properties"]) == set(ModelAnswer.model_fields)
    assert set(ANSWER_JSON_SCHEMA["required"]) == set(ModelAnswer.model_fields)


def test_json_schema_uses_only_what_bedrock_accepts():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            for banned in ("minLength", "maxLength", "minimum", "maximum", "multipleOf"):
                assert banned not in node
            if "minItems" in node:
                assert node["minItems"] in (0, 1)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(ANSWER_JSON_SCHEMA)


def test_every_flag_is_explained():
    import app.answer as an
    import inspect
    used = set(__import__("re").findall(r'flags\.append\("(\w+)"\)', inspect.getsource(an)))
    assert used <= set(FLAGS)


def test_config_checks_and_hash():
    with pytest.raises(ValueError):
        Config(search_mode="keyword")
    with pytest.raises(ValueError):
        Config(output_mode="xml")
    a = config_hash(DEFAULT, "2026-10-23")
    assert a == config_hash(Config(), "2026-10-23")
    assert a != config_hash(DEFAULT.with_(k=8), "2026-10-23")
    assert a != config_hash(DEFAULT, "2026-11-01")
