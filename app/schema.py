"""Shapes of the model's reply and of the API response (spec: response schema, Pydantic).

ModelAnswer is what the model writes (the four fields the prompt describes). Response is what the API
returns: the model's answer after the citation guard, plus what the code adds (tool calls, retrieved
chunks, the "as of" date, the disclaimer) and what the eval needs (the model's raw citations before
the guard, flags, prompt version, tokens, latency).
"""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = ("General information from 12 CFR Parts 1005 and 1026 as of the date shown, not legal or "
              "compliance advice. Card network rules, state law and the account terms may give the "
              "customer more protection.")


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)
    citations: list[str]
    abstained: bool
    abstain_reason: str | None


# The same four fields for Bedrock structured outputs. Written out by hand because Bedrock accepts
# only a subset of JSON Schema (every object needs "additionalProperties": false; no minLength);
# tests/test_schema.py checks it matches ModelAnswer.
ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "Two to six sentences, citations inline."},
        "citations": {"type": "array", "items": {"type": "string"},
                      "description": "Every provision ID cited, as listed in the passages or tool results."},
        "abstained": {"type": "boolean"},
        "abstain_reason": {"type": ["string", "null"]},
    },
    "required": ["answer", "citations", "abstained", "abstain_reason"],
    "additionalProperties": False,
}

# Flags a response can carry (all are logged; the eval counts them)
FLAGS = {
    "invalid_citations": "the model cited an ID that is in no retrieved passage or tool result; removed",
    "no_valid_citation": "an answer was left with no valid citation, so it became an abstention",
    "output_repaired": "the first reply wasn't a valid answer object; a second try was",
    "output_invalid": "no valid answer object after the repair try; returned as an abstention",
    "tool_limit": "the model asked for the tool after its last allowed tool turn; returned as an abstention",
}


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


class Response(BaseModel):
    answer: str
    citations: list[str]
    tool_calls: list[ToolCall]
    abstained: bool
    abstain_reason: str | None
    retrieved_ids: list[str]
    regulations_as_of: str
    disclaimer: str = DISCLAIMER
    # for the eval and the logs
    raw_answer: str | None            # the model's answer text before the guard (None if there was none)
    raw_citations: list[str]          # every citation string the model wrote (field and inline), before the guard
    invalid_citations: list[str]
    flags: list[str]
    prompt_version: str
    config_hash: str
    model_id: str
    usage: dict[str, int]             # inputTokens, outputTokens over all model calls
    model_calls: int
    latency_ms: int
