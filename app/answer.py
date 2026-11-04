"""Answer one question: search, the model with the dispute_timeline tool, the citation guard, the schema.

    from datetime import date
    from app.answer import answer
    answer("Customer's debit card was stolen Friday and she called today. Most she can lose?", date(2026, 11, 3))

From the project root (needs `aws login`): python -m app.answer "your question" [--today 2026-11-03] [--structured]

Steps:
1. Search: the top k chunks (app.retrieve).
2. The model (Bedrock Converse): the system prompt, then the passages and the question (app.prompts).
   If it asks for dispute_timeline, the code runs it and sends the result back. At most
   `max_tool_turns` turns may use the tool; a request after that ends the loop with an abstention.
3. Its final reply must be the four-field JSON object (app.schema.ModelAnswer). If it isn't, the
   model gets one repair try; if that fails too, the response is an abstention.
4. Citation guard: every ID the model cited, in the citations field or inline as [§ ...] or
   [comment ...], must be a provision in a retrieved chunk or appear in a tool result. Each citation
   is read with app.citations, so a packed string, a range or a comment sub-point counts as the IDs
   it names. Others are removed (inline ones become "[citation removed]") and flagged. An answer
   left with no valid citation becomes an abstention. The eval scores the raw citations, from
   before the guard, read the same way.
"""
import argparse
import json
import re
import time
from collections import Counter
from datetime import date
from functools import lru_cache

from app import citations as cit
from app import retrieve as rt
from app.config import DEFAULT, config_hash
from app.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from app.schema import ANSWER_JSON_SCHEMA, ModelAnswer, Response, ToolCall
from app.tools import TOOL_SPEC, dispute_timeline

TOOL_NAME = TOOL_SPEC["toolSpec"]["name"]
REPAIR = ("Your last reply was not a valid answer object ({error}). Reply again with only the JSON object "
          "with the fields answer, citations, abstained and abstain_reason.")
NO_CITATION = ("I couldn't find support for an answer in the regulation passages I retrieved, so I can't "
               "answer this reliably.")
OUTPUT_INVALID = "I couldn't produce a valid answer to this question."
TOOL_LIMIT = "I couldn't finish the date calculation this question needs."

PROVISION = re.compile(r"\b10(?:05|26)\.\d{1,2}(?:\([0-9A-Za-z]{1,4}\))*(?:-\d{1,2})?")
INLINE = re.compile(r"\[\s*(§+|comments?)\s*([^\]]+?)\s*\]", re.IGNORECASE)


@lru_cache(maxsize=1)
def _client(read_timeout):
    import boto3
    from botocore.config import Config as BotoConfig
    region = rt.load()[2]["region"]
    return boto3.client("bedrock-runtime", region_name=region,
                        config=BotoConfig(read_timeout=read_timeout, retries={"max_attempts": 5, "mode": "adaptive"}))


# ---------------------------------------------------------------- pieces

def run_tool(name, args):
    """Run a tool call from the model. Never raises: bad input comes back as {"error": ...} for the model."""
    if name != TOOL_NAME:
        return {"error": {"argument": "name", "message": f"unknown tool {name!r}"}}
    if not isinstance(args, dict):
        return {"error": {"argument": "input", "message": "tool input must be an object"}}
    try:
        return dispute_timeline(**args)
    except TypeError as e:                     # an argument the tool doesn't take, or a missing one
        return {"error": {"argument": "input", "message": str(e)}}


def parse_reply(text):
    """The model's final text as a ModelAnswer. Raises ValueError with a short reason."""
    t = text.strip()
    if t.startswith("```"):                    # tolerate a code fence despite the prompt
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    try:
        data = json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e.msg}") from None
    try:
        return ModelAnswer.model_validate(data)
    except Exception as e:                     # pydantic.ValidationError, shortened for the repair message
        first = e.errors()[0] if hasattr(e, "errors") else {"loc": (), "msg": str(e)}
        raise ValueError(f"field {'.'.join(map(str, first['loc'])) or '?'}: {first['msg']}") from None


def allowed_ids(results, tool_calls):
    """Lower-case ID -> spelling, for every provision in a retrieved chunk or mentioned in a tool result."""
    allowed = {p.lower(): p for r in results for p in r["chunk"]["provision_ids"]}
    for call in tool_calls:
        for p in PROVISION.findall(json.dumps(call.result)):
            allowed.setdefault(p.lower(), p)
    return allowed


@lru_cache(maxsize=1)
def index_spelling():
    """Lower-case provision ID -> the index's spelling, for every provision in the index."""
    return {p.lower(): p for c in rt.load()[0] for p in c["provision_ids"]}


def guard(model_answer, allowed, provisions=None):
    """Apply the citation guard (GUARD_VERSION c2). Returns (answer text, valid IDs, raw strings, invalid).

    Each citation string, in the citations field or inline, is read with app.citations, so one string
    may name several IDs ("1026.13(c)(1), § 1026.13(c)(2)", a range, a comment sub-point). An ID is
    valid if `allowed` holds it (a retrieved chunk or a tool result). Invalid: IDs read but not allowed,
    and whatever the reader couldn't turn into an index ID (too broad, not found, no part, unreadable).
    An inline marker whose every ID is valid stays as written; one with some valid IDs is rewritten to
    those only, one per marker part ("[§ A; comment B]"); one with none becomes "[citation removed]"."""
    provisions = provisions or index_spelling()
    raw = [c.strip() for c in model_answer.citations]
    raw += [m.group(2) for m in INLINE.finditer(model_answer.answer)]
    raw = list(dict.fromkeys(c for c in raw if c))                     # dedupe, keep order
    valid, invalid = [], []
    for c in raw:
        reading = cit.read_citation(c, provisions)
        for i in reading.ids:
            (valid if i.lower() in allowed else invalid).append(allowed.get(i.lower(), i))
        invalid += reading.problems
    valid, invalid = list(dict.fromkeys(valid)), list(dict.fromkeys(invalid))

    def inline(m):
        reading = cit.read_citation(m.group(2), provisions)
        ok = [allowed[i.lower()] for i in reading.ids if i.lower() in allowed]
        if ok and len(ok) == len(reading.ids) and not reading.problems:
            return m.group(0)
        return "[" + "; ".join(cit.marker(i) for i in ok) + "]" if ok else "[citation removed]"
    return INLINE.sub(inline, model_answer.answer), valid, raw, invalid


def converse_request(config):
    req = {"modelId": config.generator_model,
           "system": [{"text": SYSTEM_PROMPT}],
           "inferenceConfig": {"temperature": config.temperature, "maxTokens": config.max_tokens}}
    if config.tool_on:
        req["toolConfig"] = {"tools": [TOOL_SPEC]}
    if config.output_mode == "structured":
        req["outputConfig"] = {"textFormat": {"type": "json_schema", "structure": {"jsonSchema": {
            "schema": json.dumps(ANSWER_JSON_SCHEMA), "name": "card_rules_answer",
            "description": "The answer to a staff question about Regulations E and Z"}}}}
    return req


def reply_text(message):
    return "".join(b["text"] for b in message["content"] if "text" in b)


# ---------------------------------------------------------------- the flow

def answer(question, today, *, config=DEFAULT, client=None, query_vec=None):
    """Answer one question. Returns a Response. Raises ValueError for an empty or over-long question."""
    started = time.perf_counter()
    question = (question or "").strip()
    if not question:
        raise ValueError("question is empty")
    if len(question) > config.max_question_chars:
        raise ValueError(f"question is over {config.max_question_chars} characters")
    client = client or _client(config.read_timeout)

    results = rt.search(question, config.k, config.search_mode, query_vec=query_vec)
    as_of = results[0]["chunk"]["as_of"] if results else rt.load()[0][0]["as_of"]
    req = converse_request(config)
    messages = [{"role": "user", "content": [{"text": build_user_message(question, results, today)}]}]
    usage, calls, tool_calls, flags = Counter(), 0, [], []

    def call():
        nonlocal calls
        resp = client.converse(messages=messages, **req)
        calls += 1
        for key in ("inputTokens", "outputTokens"):
            usage[key] += resp.get("usage", {}).get(key, 0)
        messages.append(resp["output"]["message"])
        return resp

    # 1. tool loop
    resp = call()
    tool_turns = 0
    while resp["stopReason"] == "tool_use":
        if tool_turns == config.max_tool_turns:
            flags.append("tool_limit")
            break
        tool_turns += 1
        blocks = []
        for b in resp["output"]["message"]["content"]:
            if "toolUse" in b:
                use = b["toolUse"]
                result = run_tool(use["name"], use.get("input"))
                tool_calls.append(ToolCall(name=use["name"], args=use.get("input") or {}, result=result))
                blocks.append({"toolResult": {"toolUseId": use["toolUseId"], "content": [{"json": result}],
                                              "status": "error" if "error" in result else "success"}})
        messages.append({"role": "user", "content": blocks})
        resp = call()

    # 2. the final reply, with one repair try
    model_answer = None
    if "tool_limit" not in flags:
        try:
            model_answer = parse_reply(reply_text(resp["output"]["message"]))
        except ValueError as e:
            messages.append({"role": "user", "content": [{"text": REPAIR.format(error=e)}]})
            resp = call()
            if resp["stopReason"] != "tool_use":
                try:
                    model_answer = parse_reply(reply_text(resp["output"]["message"]))
                    flags.append("output_repaired")
                except ValueError:
                    pass
            if model_answer is None:
                flags.append("output_invalid")

    # 3. the guard
    if model_answer is None:
        text, valid, raw, invalid = (TOOL_LIMIT if "tool_limit" in flags else OUTPUT_INVALID), [], [], []
        abstained, reason, raw_answer = True, flags[-1].replace("_", " "), None
    else:
        text, valid, raw, invalid = guard(model_answer, allowed_ids(results, tool_calls))
        abstained, reason, raw_answer = model_answer.abstained, model_answer.abstain_reason, model_answer.answer
        if invalid:
            flags.append("invalid_citations")
        if not abstained and not valid:
            flags.append("no_valid_citation")
            text, abstained, reason = NO_CITATION, True, "no valid citation"
        if not abstained:
            reason = None

    return Response(
        answer=text, citations=valid, tool_calls=tool_calls, abstained=abstained, abstain_reason=reason,
        retrieved_ids=[r["id"] for r in results], regulations_as_of=as_of,
        raw_answer=raw_answer, raw_citations=raw, invalid_citations=invalid, flags=flags,
        prompt_version=PROMPT_VERSION, config_hash=config_hash(config, as_of), model_id=config.generator_model,
        usage=dict(usage), model_calls=calls, latency_ms=round((time.perf_counter() - started) * 1000))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Answer one question and print the response as JSON.")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--today", type=date.fromisoformat, default=date.today())
    ap.add_argument("--structured", action="store_true", help="use Bedrock structured outputs")
    args = ap.parse_args(argv)
    config = DEFAULT.with_(output_mode="structured") if args.structured else DEFAULT
    print(answer(" ".join(args.question), args.today, config=config).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
