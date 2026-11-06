"""The LLM judge (spec: Evaluation method, layer 2; tracker t6-judge). Grades answers already logged
in runs.db by eval/run.py: is every claim supported by the passages, and does the answer state the
question's key facts? It never asks the assistant again, so a run can be re-judged after a rubric
change without new answers.

From the repo root (needs `aws login` except --summary and --show):
    python -m eval.judge --smoke                 one made-up example: checks AWS access and the reply format
    python -m eval.judge RUN_ID                  judge every answer in the run not yet judged with this rubric
    python -m eval.judge RUN_ID --ids q004 q009  only these questions
    python -m eval.judge RUN_ID --rejudge        judge again, replacing this rubric version's verdicts
    python -m eval.judge RUN_ID --rubric j1      use an earlier rubric (verdicts are stored per version)
    python -m eval.judge RUN_ID --summary        the counts, from stored verdicts (no model calls)
    python -m eval.judge RUN_ID --show q004      one stored verdict in full

Calibration is blind: the judge refuses a run on the calibration sheet (eval/calibration/key.json) until
every label is filled in and eval/calibration/ is committed (eval/calibrate.py). The --smoke example uses
nothing from the golden set.

What the judge (Sonnet 4.5, config.judge_model, temperature 0) receives, per the spec: the question,
today's date, the passages the assistant was given (rebuilt from the index by their chunk IDs, in the
assistant's own passage format), every tool call with its result, the answer as the user sees it
(after the citation guard), whether it was a decline, the reference answer and the numbered key facts.
It returns one JSON object: every claim marked supported or not with its source, a faithfulness label,
each key fact covered or not, and a one-line rationale (JudgeVerdict). Code checks the object: one
verdict per key fact, and a label that agrees with the claims (all supported -> faithful; any
unsupported -> not faithful). A bad reply gets one repair try; a second bad reply is logged as an
error, not guessed.

Choices (Prit, session 19):
- Key facts are graded without their citations, since citations are scored by code: strip_citations()
  removes "(§ 1026.13(c)(2))", "under 12 CFR § 1026.12(c)" and the like (41 of the 43 facts with a
  citation); the rubric tells the judge to ignore any citation still left (q028, q045 use one inside
  the sentence).
- Every answer is judged, declines included, except the guard's fixed fallback texts (no valid
  citation, invalid output, tool limit): nothing in them can be judged. Those are logged as skipped.

Choices marked * are Claude's (session 19), open to Prit's push-back:
- * Statements that something is outside the regulations or the passages ("credit bureau deadlines
  aren't set by Regulation Z") aren't claims for faithfulness: abstention is graded by code. They
  still count for key-fact coverage.
- * The reference answer is never evidence for faithfulness (only the passages and tool results are);
  it only explains what the key facts mean.
- * Key-fact coverage is all or nothing per fact: wording may differ, but every date, number, amount,
  party and condition in the fact must be there.
- * Verdicts are stored per rubric version (JUDGE_VERSION); a rubric edit fails tests/test_judge.py
  until the version is bumped, so verdicts from different rubrics are never mixed.

Rubric j2 (session 20), the spec's one revision. Prit chose all four changes, which Claude proposed
after reading the disagreements on calibration set 1 (40 answers, set aside as a measure after a
labelling mix-up and used only to design j2). j2 is fixed before set 2 (30 fresh answers) is labelled,
so set 2 measures j1 and j2 on answers neither was written from. j1 stays runnable (--rubric j1).
1. Naming a law, part, subpart or provision as governing something, with no passage or tool result
   saying so ("governed by the Fair Credit Reporting Act", "see Subpart B"), is an unsupported claim;
   in a decline it is a side claim (so partially_faithful, not unfaithful). Prompt rule 4 and
   decision 48: declines say the topic is outside Z and E, without naming the outside law.
2. Dates: a calendar-day count from a date in the question or a tool result, by a period a passage or
   tool result gives, is supported if correct (the judge checks it). Business-day counts and
   holiday/business-day statements need a tool result: only the tool has the holiday table.
   Facts the question states count as given.
3. Suggestions about what to do or review next ("consult § 1026.13(a)") aren't claims unless they
   state what a rule requires.
4. A claim is judged at the scope it states ("not prohibited by § 1026.10(d)" against that passage);
   something left out is not an unsupported claim, since omissions are graded by key facts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import statistics
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from eval import golden as G
from eval import run as R

JUDGE_VERSION = "j2"
# j1 first rubric (session 19); j2 the one revision (session 20, Prit chose all four changes from the
# disagreements on calibration set 1): naming an outside law is an unsupported claim (a side claim in
# a decline); calendar-day counts may be worked out if correct, business days need a tool result;
# suggestions about what to do next aren't claims; a claim is judged at the scope it states and an
# omission is a key-fact matter.

GUARD_CAUSES = R.GUARD_CAUSES          # the guard's fixed fallback texts are not judged
MAX_TOKENS = 3000                       # a claims list for a six-sentence answer fits in well under this

JUDGE_PROMPT_J1 = """\
You grade one answer written by an assistant for staff at a US bank and credit card issuer. The \
assistant answers questions about Regulation E (12 CFR Part 1005) and Regulation Z (12 CFR Part 1026) \
using only the numbered passages it was given and the results of a date tool, dispute_timeline. You \
receive what the assistant received (the question, today's date, the passages and the tool results), \
its answer, and a reference answer with numbered key facts written by a human expert.

Do three things.

1. Claims. Split the answer into its claims: each separate statement of a rule, fact, date, amount, \
deadline or conclusion. A sentence may hold several claims. For each, decide whether the passages or \
tool results support it.
- Supported: a passage or tool result states it, or it follows directly from one; different wording \
is fine. A date, deadline or amount is supported only if a tool result or passage gives it.
- Not supported: no passage or tool result states it, even if it is true in real life (for example a \
card network rule, state law or a statute stated from memory); it contradicts a passage; it changes \
a rule's strength or scope ("must" turned into "may", a condition or exception dropped or added); or \
it presents text from a passage whose Status line is not "in force" as law in force.
- The reference answer is not evidence. Judge support only against the passages and tool results.
- Not claims: citations themselves, "[citation removed]" markers, and statements that something is \
outside the regulations or the passages, or that the assistant can't answer it. Leave those out of \
the claims list.
For each claim give its source: "passage N", "tool", or null when not supported.

2. Faithfulness label.
- faithful: every claim is supported (also when the answer makes no claims).
- partially_faithful: at least one claim is not supported, but every claim that answers the main \
thing the question asks is supported.
- unfaithful: a claim that answers the main thing the question asks is not supported.

3. Key facts. For each numbered key fact, decide whether the answer states it. Wording may differ, but \
every date, number, amount, party and condition in the fact must be there; a fact stated only in part, \
or with a different date or amount, is not covered. Ignore any citation inside a key fact: the fact is \
covered if the answer states its substance, whatever the answer cites. Coverage is about what the \
answer says, not whether the passages support it. Use the reference answer only to understand what \
a key fact means.

Reply with one JSON object and nothing else, no text before or after it and no code fence:
{"claims": [{"claim": "...", "supported": true, "source": "passage 2"}],
 "faithfulness": "faithful",
 "key_facts": [{"n": 1, "covered": true}],
 "rationale": "one sentence"}
List every key fact once, by its number, in order.
"""

JUDGE_PROMPT_J2 = """\
You grade one answer written by an assistant for staff at a US bank and credit card issuer. The \
assistant answers questions about Regulation E (12 CFR Part 1005) and Regulation Z (12 CFR Part 1026) \
using only the numbered passages it was given and the results of a date tool, dispute_timeline. You \
receive what the assistant received (the question, today's date, the passages and the tool results), \
its answer, and a reference answer with numbered key facts written by a human expert.

Do three things.

1. Claims. Split the answer into its claims: each separate statement of a rule, fact, date, amount, \
deadline or conclusion. A sentence may hold several claims. For each, decide whether the passages or \
tool results support it.
- Supported: a passage or tool result states it, or it follows directly from one; different wording \
is fine. Facts the question itself states (dates, weekdays, amounts, what happened) count as given.
- Dates and day counts: a date, deadline or amount is supported when a tool result or passage gives \
it. A date reached by counting calendar days from a date in the question or a tool result, by a \
period a passage or tool result states, is also supported if the count is right: check the count \
yourself. A count of business days, or a statement that a day is or isn't a business day or a \
holiday, is supported only when a tool result gives it.
- Not supported: no passage or tool result states it, even if it is true in real life (for example a \
card network rule, state law or a statute stated from memory); it contradicts a passage; it changes \
a rule's strength or scope ("must" turned into "may", a condition or exception dropped or added); or \
it presents text from a passage whose Status line is not "in force" as law in force.
- Naming a law, regulation part, subpart or provision as the one that governs something, when no \
passage or tool result says so ("that is governed by the Fair Credit Reporting Act", "see Subpart B", \
"the rule is in § 1005.5"), is a claim, and it is not supported. In an answer that declines, such a \
statement is a side claim: it does not answer the main thing the question asks.
- Judge a claim at the scope it states. A conclusion limited to one provision ("this is not \
prohibited by § 1026.10(d)") is supported when that provision's passage supports it, even if other \
rules not in the passages might also bear on the question. Something the answer leaves out is not an \
unsupported claim: omissions are graded by the key facts.
- The reference answer is not evidence. Judge support only against the passages and tool results.
- Not claims: citations themselves, "[citation removed]" markers, statements that something is \
outside the regulations or the passages or that the assistant can't answer it, and suggestions about \
what to do or review next ("consult § 1026.13(a)", "the procedures would need to be evaluated") \
unless they state what a rule requires. Leave those out of the claims list.
For each claim give its source: "passage N", "tool", "question", or null when not supported.

2. Faithfulness label.
- faithful: every claim is supported (also when the answer makes no claims).
- partially_faithful: at least one claim is not supported, but every claim that answers the main \
thing the question asks is supported.
- unfaithful: a claim that answers the main thing the question asks is not supported.

3. Key facts. For each numbered key fact, decide whether the answer states it. Wording may differ, but \
every date, number, amount, party and condition in the fact must be there; a fact stated only in part, \
or with a different date or amount, is not covered. Ignore any citation inside a key fact: the fact is \
covered if the answer states its substance, whatever the answer cites. Coverage is about what the \
answer says, not whether the passages support it. Use the reference answer only to understand what \
a key fact means.

Reply with one JSON object and nothing else, no text before or after it and no code fence:
{"claims": [{"claim": "...", "supported": true, "source": "passage 2"}],
 "faithfulness": "faithful",
 "key_facts": [{"n": 1, "covered": true}],
 "rationale": "one sentence"}
List every key fact once, by its number, in order.
"""

RUBRICS = {"j1": JUDGE_PROMPT_J1, "j2": JUDGE_PROMPT_J2}
JUDGE_PROMPT = RUBRICS[JUDGE_VERSION]

REPAIR = ("Your last reply could not be used ({error}). Reply again with only the JSON object described: "
          "claims, faithfulness, key_facts (one entry per key fact, numbered from 1) and rationale.")

SCHEMA = """
CREATE TABLE IF NOT EXISTS judgements (
    run_id TEXT, qid TEXT, judge_version TEXT, judge_model TEXT, judged_at TEXT,
    status TEXT, skip_reason TEXT, error TEXT, faithfulness TEXT, kf_covered INTEGER, kf_total INTEGER,
    verdict_json TEXT, input_sha256 TEXT, input_tokens INTEGER, output_tokens INTEGER,
    model_calls INTEGER, latency_ms INTEGER,
    PRIMARY KEY (run_id, qid, judge_version)
);
"""


# ================================================================ key facts without citations

_ID = r"10(?:05|26)\.\d+(?:\([0-9A-Za-z]{1,4}\))*(?:-\d{1,2}(?:\([0-9a-z]{1,4}\))?)?"
_REF = (r"(?:12\s+CFR\s+)?(?:§§?|[Cc]omments?)\s*" + _ID
        + r"(?:\s*(?:[–-]|through)\s*(?:§§?\s*)?" + _ID + r")?")
_REFS = _REF + r"(?:\s*(?:,|;|and)\s*" + _REF + r")*"
_PAREN = re.compile(r"\s*\(\s*(?:(?:see|under)\s+)?(?:Regulation\s+[EZ]\s*,?\s*)?" + _REFS + r"\s*\)")
_TRAIL = re.compile(r",?\s+(?:under|per|pursuant\s+to)\s+(?:Regulation\s+[EZ]\s*,?\s*)?" + _REFS
                    + r"(?=\s*[.;,]?\s*$)")


def strip_citations(fact: str) -> str:
    """A key fact without its citations: a bracketed one anywhere, "(§ 1026.13(c)(2))", or one that ends
    the sentence, "under 12 CFR § 1026.12(c)". A citation used inside the sentence ("12 CFR § 1005.7(a)
    governs ...") is left for the rubric's "ignore any citation" rule, since removing it breaks the sentence."""
    out = _TRAIL.sub("", _PAREN.sub("", fact)).strip()
    return re.sub(r"\s+([.,;])", r"\1", out)


# ================================================================ what the judge sees

class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    supported: bool
    source: str | None


class KeyFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int
    covered: bool


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim]
    faithfulness: Literal["faithful", "partially_faithful", "unfaithful"]
    key_facts: list[KeyFact]
    rationale: str


def chunk_lookup() -> dict:
    from app import retrieve as rt
    return {c["id"]: c for c in rt.load()[0]}


def judge_message(item: G.GoldenItem, resp: dict, today: date, chunks: dict) -> str:
    """The judge's user message for one logged answer."""
    from app.prompts import format_passage
    missing = [cid for cid in resp["retrieved_ids"] if cid not in chunks]
    if missing:
        raise ValueError(f"retrieved chunk(s) not in the current index: {', '.join(missing)}")
    passages = "\n\n".join(format_passage(n, chunks[cid]) for n, cid in enumerate(resp["retrieved_ids"], 1))
    calls = "\n".join(json.dumps({"tool": c["name"], "arguments": c["args"], "result": c["result"]},
                                 ensure_ascii=False) for c in resp["tool_calls"]) or "(no tool calls)"
    facts = "\n".join(f"{n}. {strip_citations(f)}" for n, f in enumerate(item.key_facts or [], 1))
    return (f"<question>\n{item.question.strip()}\n</question>\n\n"
            f"Today's date: {today.isoformat()} ({today.strftime('%A')})\n\n"
            f"<passages>\n{passages or '(no passages)'}\n</passages>\n\n"
            f"<tool_results>\n{calls}\n</tool_results>\n\n"
            f"<answer declined=\"{'yes' if resp['abstained'] else 'no'}\">\n{resp['answer'].strip()}\n</answer>\n\n"
            f"<reference_answer>\n{(item.reference_answer or '').strip()}\n</reference_answer>\n\n"
            f"<key_facts>\n{facts or '(none)'}\n</key_facts>")


def parse_verdict(text: str, n_facts: int) -> JudgeVerdict:
    """The judge's reply as a JudgeVerdict, checked. Raises ValueError with a short reason."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    try:
        data = json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e.msg}") from None
    try:
        v = JudgeVerdict.model_validate(data)
    except Exception as e:                      # pydantic.ValidationError, shortened for the repair message
        first = e.errors()[0] if hasattr(e, "errors") else {"loc": (), "msg": str(e)}
        raise ValueError(f"field {'.'.join(map(str, first['loc'])) or '?'}: {first['msg']}") from None
    if [k.n for k in v.key_facts] != list(range(1, n_facts + 1)):
        raise ValueError(f"key_facts must list facts 1 to {n_facts} once each, in order")
    all_supported = all(c.supported for c in v.claims)
    if all_supported and v.faithfulness != "faithful":
        raise ValueError("every claim is supported, so faithfulness must be faithful")
    if not all_supported and v.faithfulness == "faithful":
        raise ValueError("a claim is not supported, so faithfulness can't be faithful")
    return v


def skip_reason(resp: dict) -> str | None:
    """Why an answer isn't judged: it is one of the guard's fixed fallback texts."""
    return next((f for f in resp["flags"] if f in GUARD_CAUSES), None)


# ================================================================ one call

def _client(config):
    from app.answer import _client as answer_client
    return answer_client(config.read_timeout)


def judge_one(message: str, n_facts: int, *, config, client, rubric: str = JUDGE_PROMPT) -> dict:
    """Ask the judge. Returns {"verdict": JudgeVerdict or None, "error", "usage", "calls", "latency_ms"}."""
    started = time.perf_counter()
    req = {"modelId": config.judge_model, "system": [{"text": rubric}],
           "inferenceConfig": {"temperature": 0.0, "maxTokens": MAX_TOKENS}}
    messages = [{"role": "user", "content": [{"text": message}]}]
    usage, calls, verdict, problem = {"inputTokens": 0, "outputTokens": 0}, 0, None, None
    for attempt in range(2):                     # the first try, then one repair try
        resp = client.converse(messages=messages, **req)
        calls += 1
        for key in usage:
            usage[key] += resp.get("usage", {}).get(key, 0)
        reply = resp["output"]["message"]
        messages.append(reply)
        try:
            verdict = parse_verdict("".join(b["text"] for b in reply["content"] if "text" in b), n_facts)
            break
        except ValueError as e:
            problem = e
            if attempt == 0:
                messages.append({"role": "user", "content": [{"text": REPAIR.format(error=e)}]})
    error = None if verdict else f"no usable verdict after one repair try: {problem}"
    return {"verdict": verdict, "error": error, "usage": usage, "calls": calls,
            "latency_ms": round((time.perf_counter() - started) * 1000)}


# ================================================================ runs.db

def connect(path: Path = R.DB_FILE) -> sqlite3.Connection:
    db = R.connect(path)
    db.executescript(SCHEMA)
    return db


def logged_answers(db, run_id: str, ids: list[str] | None = None):
    """(item, response dict, today) for each answered question in the run, in question order."""
    head = db.execute("SELECT today_default FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if head is None:
        raise SystemExit(f"no run {run_id!r} in the database; see python -m eval.run --runs")
    default_today = date.fromisoformat(head[0])
    rows = db.execute("SELECT qid, item_json, response_json FROM results WHERE run_id = ? ORDER BY qid",
                      (run_id,)).fetchall()
    known = {r[0] for r in rows}
    if ids and (missing := [q for q in ids if q not in known]):
        raise SystemExit(f"not in run {run_id}: {', '.join(missing)}")
    out = []
    for qid, item_json, resp_json in rows:
        if ids and qid not in ids:
            continue
        if resp_json is None:            # the question failed to run; nothing to judge
            continue
        item = G.GoldenItem.model_validate_json(item_json)
        out.append((item, json.loads(resp_json), item.today or default_today))
    return out


def save(db, run_id, qid, *, config, status, skip=None, result=None, message=None, version=JUDGE_VERSION):
    v = (result or {}).get("verdict")
    usage = (result or {}).get("usage", {})
    db.execute("INSERT OR REPLACE INTO judgements VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        run_id, qid, version, config.judge_model, datetime.now(timezone.utc).isoformat(),
        status, skip, (result or {}).get("error"),
        v.faithfulness if v else None,
        sum(k.covered for k in v.key_facts) if v else None, len(v.key_facts) if v else None,
        v.model_dump_json() if v else None,
        hashlib.sha256(message.encode("utf-8")).hexdigest() if message else None,
        usage.get("inputTokens"), usage.get("outputTokens"), (result or {}).get("calls"),
        (result or {}).get("latency_ms")))
    db.commit()


def judge_run(db, run_id, *, config, client=None, ids=None, rejudge=False, chunks=None, progress=print,
              version: str = JUDGE_VERSION) -> None:
    """Judge every logged answer in the run (or `ids`) that this rubric version hasn't judged yet."""
    if version not in RUBRICS:
        raise SystemExit(f"no rubric {version!r}; there are {', '.join(RUBRICS)}")
    done = {q for (q,) in db.execute("SELECT qid FROM judgements WHERE run_id = ? AND judge_version = ?",
                                     (run_id, version))}
    todo = [(i, r, t) for i, r, t in logged_answers(db, run_id, ids) if rejudge or i.id not in done]
    if not todo:
        progress(f"Nothing to judge: every answer in {run_id} already has a {version} verdict "
                 f"(use --rejudge to replace them).")
        return
    chunks = chunks or chunk_lookup()
    client = client or _client(config)
    called = False
    for n, (item, resp, today) in enumerate(todo, 1):
        if (why := skip_reason(resp)):
            save(db, run_id, item.id, config=config, status="skipped", skip=why, version=version)
            progress(f"  [{n}/{len(todo)}] {item.id}: skipped ({why})")
            continue
        message = judge_message(item, resp, today, chunks)
        try:
            result = judge_one(message, len(item.key_facts or []), config=config, client=client,
                               rubric=RUBRICS[version])
        except Exception as e:                  # noqa: BLE001 - an error on the first call stops the run
            if not called:
                raise
            result = {"verdict": None, "error": f"{type(e).__name__}: {e}"}
        called = True
        status = "judged" if result["verdict"] else "error"
        save(db, run_id, item.id, config=config, status=status, result=result, message=message, version=version)
        v = result["verdict"]
        progress(f"  [{n}/{len(todo)}] {item.id}: " + (
            f"judged ({sum(k.covered for k in v.key_facts)}/{len(v.key_facts)} key facts)" if v else
            f"error: {result['error']}"))


# ================================================================ summary

def summary(db, run_id: str, version: str = JUDGE_VERSION) -> dict:
    rows = db.execute(
        "SELECT j.qid, j.status, j.skip_reason, j.faithfulness, j.kf_covered, j.kf_total, r.persona, r.should_abstain "
        "FROM judgements j JOIN results r ON r.run_id = j.run_id AND r.qid = j.qid "
        "WHERE j.run_id = ? AND j.judge_version = ? ORDER BY j.qid", (run_id, version)).fetchall()

    def tally(rs):
        judged = [r for r in rs if r[1] == "judged"]
        labels = {k: sum(r[3] == k for r in judged) for k in ("faithful", "partially_faithful", "unfaithful")}
        shares = [r[4] / r[5] for r in judged if r[5]]
        return {"n": len(rs), "judged": len(judged), "skipped": sum(r[1] == "skipped" for r in rs),
                "errors": sum(r[1] == "error" for r in rs), **labels,
                "key_facts": (sum(r[4] or 0 for r in judged), sum(r[5] or 0 for r in judged)),
                "key_fact_share_mean": round(statistics.fmean(shares), 3) if shares else None}

    out = {"run_id": run_id, "judge_version": version, "overall": tally(rows), "by_persona": {},
           "answerable": tally([r for r in rows if not r[7]]), "declines": tally([r for r in rows if r[7]]),
           "skipped": [(r[0], r[2]) for r in rows if r[1] == "skipped"],
           "errors": [r[0] for r in rows if r[1] == "error"]}
    for p in sorted({r[6] for r in rows}):
        out["by_persona"][p] = tally([r for r in rows if r[6] == p])
    return out


def report(s: dict) -> str:
    def line(name, t):
        if not t["n"]:
            return f"  {name:12} none"
        return (f"  {name:12} judged {t['judged']}/{t['n']}: faithful {t['faithful']}, partially "
                f"{t['partially_faithful']}, unfaithful {t['unfaithful']}; key facts {t['key_facts'][0]}/"
                f"{t['key_facts'][1]} (mean share {t['key_fact_share_mean']})")
    lines = [f"Judge {s['judge_version']} on run {s['run_id']}", line("all", s["overall"]),
             line("answerable", s["answerable"]), line("declines", s["declines"]), "", "By persona"]
    lines += [line(p, t) for p, t in s["by_persona"].items()]
    if s["skipped"]:
        lines += ["", "Not judged (the guard's fallback text): " + ", ".join(f"{q} {w}" for q, w in s["skipped"])]
    if s["errors"]:
        lines += ["", "No usable verdict: " + ", ".join(s["errors"]) + " (judge them again with --ids)"]
    return "\n".join(lines)


def show(db, run_id: str, qid: str, version: str = JUDGE_VERSION) -> str:
    row = db.execute("SELECT status, skip_reason, error, verdict_json FROM judgements "
                     "WHERE run_id = ? AND qid = ? AND judge_version = ?", (run_id, qid, version)).fetchone()
    if row is None:
        return f"No {version} verdict for {qid} in {run_id}."
    status, skip, error, vj = row
    if status != "judged":
        return f"{qid}: {status}" + (f" ({skip})" if skip else "") + (f": {error}" if error else "")
    v = JudgeVerdict.model_validate_json(vj)
    item = next(i for i, _, _ in logged_answers(db, run_id, [qid]))
    out = [f"{qid}: {v.faithfulness}. {v.rationale}", "", "Claims:"]
    out += [f"  [{'supported' if c.supported else 'NOT supported'}{', ' + c.source if c.source else ''}] {c.claim}"
            for c in v.claims] or ["  (none)"]
    out += ["", "Key facts:"]
    out += [f"  {k.n}. [{'covered' if k.covered else 'NOT covered'}] {strip_citations(f)}"
            for k, f in zip(v.key_facts, item.key_facts or [])]
    return "\n".join(out)


# ================================================================ the smoke example

SMOKE_ITEM = G.GoldenItem.model_validate({
    "id": "q000", "stage": "labelled", "persona": "frontline", "type": "unauthorized",
    "question": "A made-up example to check the judge works: what is the most a debit card holder can lose?",
    "regulation": "E", "gold_ids": ["1005.6(b)(1)"],
    "reference_answer": "If the consumer tells the bank within two business days of learning of the loss, "
                        "liability is at most $50.",
    "key_facts": ["at most $50 if notice is given within two business days (§ 1005.6(b)(1))"],
    "should_abstain": False})
SMOKE_RESPONSE = {
    "answer": "If she tells the bank within two business days of learning of the loss, the most she can lose "
              "is $50 [§ 1005.6(b)(1)]. Her card network will also refund everything.",
    "abstained": False, "retrieved_ids": [], "tool_calls": [], "flags": []}


def smoke(*, config, client=None) -> str:
    """Judge one made-up answer: one claim the 1005.6(b)(1) passage supports, and one card-network claim
    from memory that no passage supports. Checks AWS access and that the reply parses; uses nothing from
    the golden set, so it shows Prit no verdict on an answer he may label."""
    chunks = chunk_lookup()
    first = next(c for c in chunks.values() if "1005.6(b)(1)" in c["provision_ids"])
    resp = dict(SMOKE_RESPONSE, retrieved_ids=[first["id"]])
    message = judge_message(SMOKE_ITEM, resp, date(2026, 11, 5), chunks)
    result = judge_one(message, 1, config=config, client=client or _client(config))
    if not result["verdict"]:
        return f"The judge replied, but not usably: {result['error']}"
    v = result["verdict"]
    return (f"The judge works. Verdict on the made-up example: {v.faithfulness}; key fact covered: "
            f"{v.key_facts[0].covered}; claims: " + "; ".join(
                f"{'supported' if c.supported else 'NOT supported'}: {c.claim}" for c in v.claims)
            + f"\n(Expected: partially_faithful, the network claim not supported, key fact covered. "
              f"{result['usage']['inputTokens']} tokens in, {result['usage']['outputTokens']} out.)")


# ================================================================ command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.judge", description=__doc__.split("\n\n")[0])
    ap.add_argument("run_id", nargs="?", help="a run from python -m eval.run --runs")
    ap.add_argument("--ids", nargs="+", metavar="QID", help="only these questions")
    ap.add_argument("--rubric", choices=list(RUBRICS), default=JUDGE_VERSION,
                    help=f"which rubric version (default {JUDGE_VERSION}, the current one)")
    ap.add_argument("--rejudge", action="store_true", help="replace this rubric version's existing verdicts")
    ap.add_argument("--summary", action="store_true", help="print the counts from stored verdicts, no model calls")
    ap.add_argument("--show", metavar="QID", help="print one stored verdict, no model calls")
    ap.add_argument("--smoke", action="store_true", help="judge one made-up example (checks AWS access)")
    ap.add_argument("--db", type=Path, default=R.DB_FILE, help="the SQLite file (default runs.db)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from app.config import DEFAULT
    if args.smoke:
        print(smoke(config=DEFAULT))
        return 0
    if not args.run_id:
        ap.error("give a RUN_ID (python -m eval.run --runs lists them), or --smoke")
    db = connect(args.db)
    if args.show:
        print(show(db, args.run_id, args.show, args.rubric))
        return 0
    if not args.summary:
        from eval.calibrate import blind_block
        if (why := blind_block(args.run_id)):
            print(f"Not judged: {why}")
            return 2
        print(f"Judging run {args.run_id} with rubric {args.rubric} ({DEFAULT.judge_model}).")
        try:
            judge_run(db, args.run_id, config=DEFAULT, ids=args.ids, rejudge=args.rejudge, version=args.rubric)
        except Exception as e:                  # noqa: BLE001 - the first call failed
            print(f"\nThe first judge call failed, so judging stopped: {type(e).__name__}: {e}")
            print("This is usually the AWS login: run `aws login`, check `aws sts get-caller-identity`, and try again.")
            return 1
        print()
    print(report(summary(db, args.run_id, args.rubric)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
