"""The report for one logged run (spec: Evaluation method and Error taxonomy; tracker t6-report): the
code-graded metrics and the judge's verdicts side by side, by persona and by type, and one primary
error tag for every question that failed. It reads runs.db only: no model calls, no `aws login`.

From the repo root:
    python -m eval.report RUN_ID                     the report (judge rubric j2)
    python -m eval.report RUN_ID --judge j1          with an earlier rubric's verdicts
    python -m eval.report RUN_ID --markdown FILE     also write it as Markdown (for the README)
    python -m eval.report RUN_ID --show q004         one question: its answer, tags and the judge's verdict
    python -m eval.report RUN_ID --review-sheet      write eval/review/RUN_ID.toml for the author's review

Judge the run first (python -m eval.judge RUN_ID); questions with no verdict are listed, and the tags
that need the judge can't fire for them.

How the numbers are counted:
- Code metrics are graded again from the stored answers with the current grader (eval.run.grade), as
  --regrade does; the report says when the run was logged with another grader version.
- Faithfulness counts judged answers only. The guard's fallback texts (no valid citation, invalid
  output, tool limit) are not judged: they hold no claims.
- Key-fact coverage is over answerable questions: covered facts / all their facts. A fallback text
  covers none of its question's facts (counted 0 of n). Declines' key facts are left out, because the
  judge's agreement with Prit on them was weak (session 20: kappa 0.19 on 10 answers).
- Every judge faithfulness number is an upper bound (session 20, decision 76): of the 14 answers j2
  called faithful in calibration set 2, Prit agreed on 7; of the 16 it flagged, on 15. It misses
  dropped or bent conditions. So the judge-based tags (memory_leak, unsupported_claim) are a lower bound.

Error tags (session 21; Prit chose the automatic tags with a review of the final run, the new tag
`incomplete`, and counting a decline that names an outside law as memory_leak; after the first real
report he chose to split every decline's unsupported claims the same way as an answer's). A question fails when
any tag applies. Tags are checked in pipeline order, cause before symptom, and the first one that
applies is the primary tag; the others are kept and shown.

Should-decline questions:
  1. missed_abstention    it answered
  2. memory_leak          a claim the judge marks unsupported names something outside Z and E (the
                          same keyword test as step 7 below; decision 48: a decline says the topic is
                          outside Z and E without naming the law)
     unsupported_claim    any other claim the judge marks unsupported (j2 also counts naming a part
                          of Z or E as governing, "see Subpart B", as one)
  Both apply to a correct decline and, as further tags, to a missed abstention.
Answerable questions:
  1. retrieval_miss       no passage the model was given holds a gold paragraph (first_gold_rank None)
  2. over_abstention      it declined (with the cause: the model, or the guard's no_valid_citation,
                          output_invalid or tool_limit)
  3. wrong_reg            it cites the other regulation (code: wrong_reg)
  4. status_error         it cites a provision that isn't in force (code: status_error)
  5. tool_misuse          a call was needed and missed, or made with wrong arguments
  6. calculation_error    an expected date, or a Reg E $50 / $500 tier, is missing from the answer
  7. memory_leak          a claim the judge marks unsupported names something outside Z and E: a card
                          network, state law, a statute, another regulation, a court, the issuer's
                          own terms (OUTSIDE_LAW below; a keyword match, so a heuristic)
     unsupported_claim    any other claim the judge marks unsupported
  8. wrong_citation       it cites a real ID it wasn't given (invented), an ID not in the index, or a
                          string the reader can't read. A whole section or an ID with no part is a
                          form problem, counted in citation precision but not tagged.
  9. partial_retrieval    a key fact is missing and a gold paragraph wasn't retrieved (some were)
     incomplete           a key fact is missing although every gold paragraph was retrieved
 10. tool_misuse          the only problem: a tool call that wasn't needed
Key-fact tags (9) aren't checked after an over-abstention (the decline explains them) or a
retrieval miss, and a missed call isn't tagged after an over-abstention either.
Decisions 77-79 are Prit's; the order above and the keyword list are Claude's (session 21), open to
his push-back.

The review (decision 77): for the final test run, the author reviews every failure in
eval/review/RUN_ID.toml (--review-sheet writes it) and confirms or changes its primary tag, or marks
it "pass". The README's counts and three worked examples come from the reviewed tags. The review covers
the failures found here only: the judge's missed conditions (above) stay a stated limit.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from eval import golden as G
from eval import judge as J
from eval import run as R

REVIEW_DIR = R.REPO / "eval" / "review"
TAGS = ("retrieval_miss", "over_abstention", "wrong_reg", "status_error", "tool_misuse", "calculation_error",
        "memory_leak", "unsupported_claim", "wrong_citation", "partial_retrieval", "incomplete",
        "missed_abstention")
PASS = "pass"                      # a review's verdict that a flagged question didn't fail
PERSONAS = ("frontline", "compliance")

# What marks a claim as stated from outside Regulations Z and E (decline categories network_rules,
# state_law, issuer_terms, other_law). Word boundaries keep "discovered", "state that", "acted" and
# "courtesy" out; "Act" must be capitalised. Words that are ordinary Z/E vocabulary are left out
# (checked against the corpus in session 21): "account agreement" (71 chunks), "terms and conditions"
# (18), "card network" (3, Reg E's debit card networks), "issuer's policies" (1). "State law" and "the
# Act" stay although the corpus uses them: the spec's memory_leak is such a statement *not in the
# passages*, which is what an unsupported claim is. "Federal law / rules / statutes / regulations"
# counts unless Regulation E or Z (or 12 CFR 1005 / 1026) follows: "federal law (Regulation E)" is Z/E
# itself (run r20261106T081540Z's q008), "the federal rules that regulate calling hours" isn't (q044).
OUTSIDE_LAW = re.compile("|".join([
    r"\b(?:visa|mastercard|american express|amex)\b", r"\bnetwork(?:'s)? rules?\b",
    r"\bzero[- ]liability\b", r"\bnacha\b", r"\bchargebacks?\b",
    r"\bstate (?:laws?|statutes?|regulators?)\b", r"(?-i:\b[Uu]nder [A-Z][a-z]+ law\b)",
    r"\bstatutes?\b", r"\bstatutory\b",
    r"\bfederal (?:laws?|rules?|statutes?|regulations?)\b"
    r"(?!\s*[(,]?\s*(?:(?:in|at)\s+)?(?:(?:Regulation|Reg\.?)\s+[EZ]\b|12 CFR (?:Part )?10(?:05|26)))",
    r"\b(?:FCRA|FCBA|TILA|EFTA|UDAAP|UDAP|CARD Act|Dodd-Frank|UCC)\b", r"\bUniform Commercial Code\b",
    r"\b(?:Truth in Lending|Electronic Fund Transfer|Fair Credit (?:Billing|Reporting)) Act\b",
    r"(?-i:\bAct\b)", r"(?-i:\bReg(?:ulation)?\.? (?!E\b|Z\b)[A-Z]{1,2}\b)",
    r"\bcourts?\b", r"\bcase law\b", r"\blawsuits?\b",
    r"\b(?:issuer|bank)'s (?:own )?(?:terms|fee schedule)\b", r"\bcardholder agreements?\b",
]), re.IGNORECASE)


def outside_law(text: str) -> bool:
    return bool(OUTSIDE_LAW.search(text))


# ================================================================ one question

@dataclass
class Verdict:
    """What the report needs from one stored judgement."""
    status: str                                 # judged, skipped, error, or "missing" (no row)
    faithfulness: str | None = None
    kf_covered: int | None = None
    kf_total: int | None = None
    unsupported: list[str] = field(default_factory=list)


def verdict_from_row(row) -> Verdict:
    """row: (status, faithfulness, kf_covered, kf_total, verdict_json), or None for no verdict."""
    if row is None:
        return Verdict("missing")
    status, faith, cov, tot, vj = row
    claims = json.loads(vj)["claims"] if vj else []
    return Verdict(status, faith, cov, tot, [c["claim"] for c in claims if not c["supported"]])


def short(text: str, n: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 1] + "…"


def claims_note(claims: list[str]) -> str:
    return "; ".join(f'"{short(c, 80)}"' for c in claims[:2]) + (f" (+{len(claims) - 2} more)" if len(claims) > 2 else "")


def tags_for(item: G.GoldenItem, m: dict, v: Verdict) -> list[tuple[str, str]]:
    """Every error tag that applies, as (tag, evidence), in precedence order: the first is the primary
    tag. An empty list means the question passed."""
    out: list[tuple[str, str]] = []

    def add(tag, why):
        if tag not in (t for t, _ in out):
            out.append((tag, why))

    unsupported = v.unsupported if v.status == "judged" else []
    leaks = [c for c in unsupported if outside_law(c)]
    others = [c for c in unsupported if not outside_law(c)]

    def claims_tags():
        if leaks:
            add("memory_leak", "judge: unsupported " + claims_note(leaks))
        if others:
            add("unsupported_claim", "judge: unsupported " + claims_note(others))

    if item.should_abstain:
        if not m["abstained"]:
            add("missed_abstention", f"answered; should decline ({item.abstain_category})")
        claims_tags()
        return out

    if m["first_gold_rank"] is None:
        add("retrieval_miss", f"none of the {m['k']} passages given holds a gold paragraph")
    if m["abstained"]:
        add("over_abstention", f"declined (cause: {m['abstain_cause']})")
    if m["wrong_reg"]:
        add("wrong_reg", f"cites the other regulation ({item.regulation} question): {', '.join(m['wrong_reg_ids'])}")
    if m["status_error"]:
        add("status_error", "cites a provision not in force: " + ", ".join(m["not_in_force_ids"]))
    if m["tool_outcome"] == "wrong_args" or (m["tool_outcome"] == "missed_call" and not m["abstained"]):
        add("tool_misuse", m["tool_outcome"] + (f" ({', '.join(m['arg_diffs'])})" if m["arg_diffs"] else ""))
    if m["calc_exact"] is False and not m["abstained"]:
        add("calculation_error", "missing from the answer: " + ", ".join(m["calc_missing"]))
    claims_tags()
    bad = []
    if m["invented_ids"]:
        bad.append("not given to it: " + ", ".join(m["invented_ids"]))
    if m["cited_not_in_index"]:
        bad.append(f"{m['cited_not_in_index']} not in the index")
    if m["cited_unreadable"]:
        bad.append(f"{m['cited_unreadable']} unreadable")
    if bad:
        add("wrong_citation", "; ".join(bad))
    if (v.status in ("judged", "skipped") and not m["abstained"] and m["first_gold_rank"] is not None
            and v.kf_total and (v.kf_covered or 0) < v.kf_total):
        facts = f"key facts {v.kf_covered} of {v.kf_total}"
        if m["gold_found"] < m["gold_total"]:
            add("partial_retrieval", f"{facts}; gold paragraphs retrieved {m['gold_found']} of {m['gold_total']}")
        else:
            add("incomplete", f"{facts}; every gold paragraph was retrieved")
    if m["tool_outcome"] == "unneeded_call":
        add("tool_misuse", "called the tool, though no call was needed")
    return out


# ================================================================ one run

@dataclass
class Row:
    item: G.GoldenItem
    metrics: dict | None              # None when the question failed to run
    error: str | None
    verdict: Verdict
    tags: list[tuple[str, str]]


def load_rows(db, run_id: str, version: str = J.JUDGE_VERSION, ix: R.IndexView | None = None) -> tuple[dict, list[Row]]:
    """The run's logged settings, and one Row per question, graded again with the current grader."""
    head = db.execute("SELECT run_id, split, ids, git_commit, git_dirty, config_json, prompt_version, grader_version, "
                      "n_items, note FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if head is None:
        raise SystemExit(f"no run {run_id!r} in the database; see python -m eval.run --runs")
    run = dict(zip(("run_id", "split", "ids", "git_commit", "git_dirty", "config_json", "prompt_version",
                    "grader_version", "n_items", "note"), head))
    ix = ix or R.index_view()
    verdicts = {qid: (status, faith, cov, tot, vj) for qid, status, faith, cov, tot, vj in db.execute(
        "SELECT qid, status, faithfulness, kf_covered, kf_total, verdict_json FROM judgements "
        "WHERE run_id = ? AND judge_version = ?", (run_id, version))}
    rows = []
    for qid, item_json, resp_json, error in db.execute(
            "SELECT qid, item_json, response_json, error FROM results WHERE run_id = ? ORDER BY qid", (run_id,)):
        item = G.GoldenItem.model_validate_json(item_json)
        if resp_json is None:
            rows.append(Row(item, None, error, Verdict("missing"), []))
            continue
        m = R.grade(item, json.loads(resp_json), ix)
        v = verdict_from_row(verdicts.get(qid))
        rows.append(Row(item, m, None, v, tags_for(item, m, v)))
    return run, rows


def stats(rows: list[Row]) -> dict:
    """One slice: eval.run's code metrics, the judge's counts, key-fact coverage and failures."""
    ran = [r for r in rows if r.metrics is not None]
    out = R.tally([r.metrics for r in ran])
    judged = [r for r in ran if r.verdict.status == "judged"]
    out["judged"] = len(judged)
    out["faithful"] = (sum(r.verdict.faithfulness == "faithful" for r in judged), len(judged))
    out["faith_labels"] = {k: sum(r.verdict.faithfulness == k for r in judged)
                           for k in ("faithful", "partially_faithful", "unfaithful")}
    kf = [r for r in ran if r.item.should_abstain is False and r.verdict.status in ("judged", "skipped")]
    out["key_facts"] = (sum(r.verdict.kf_covered or 0 for r in kf),
                        sum(r.verdict.kf_total if r.verdict.status == "judged" else len(r.item.key_facts or [])
                            for r in kf))
    out["abstain_correct"] = (sum(r.metrics["abstain_outcome"] in ("correct_answer", "correct_decline") for r in ran),
                              len(ran))
    out["failed"] = (sum(bool(r.tags) for r in ran), len(ran))
    return out


def build(db, run_id: str, version: str = J.JUDGE_VERSION, review: dict | None = None,
          ix: R.IndexView | None = None) -> dict:
    """Everything the report prints. `review`: qid -> (tag, note), a complete review of this run."""
    from eval.calibrate import run_name
    run, rows = load_rows(db, run_id, version, ix)
    ran = [r for r in rows if r.metrics is not None]
    rep = {"run": run, "run_name": run_name(db, run_id), "judge_version": version,
           "grader_now": R.GRADER_VERSION, "overall": stats(rows), "by_persona": {}, "by_type": {},
           "declines_by_category": {}, "failures": [], "not_run": [(r.item.id, r.error) for r in rows if r.metrics is None],
           "no_verdict": [(r.item.id, r.verdict.status) for r in ran if r.verdict.status in ("missing", "error")],
           "review": None}
    for p in PERSONAS:
        if any(r.item.persona == p for r in ran):
            rep["by_persona"][p] = stats([r for r in ran if r.item.persona == p])
    for t in G.TYPES:
        if any(r.item.type == t for r in ran):
            rep["by_type"][t] = stats([r for r in ran if r.item.type == t])
    for cat in sorted({r.item.abstain_category for r in ran if r.item.should_abstain}):
        group = [r for r in ran if r.item.should_abstain and r.item.abstain_category == cat]
        rep["declines_by_category"][cat] = (sum(r.metrics["abstained"] for r in group), len(group),
                                           sum(any(t == "memory_leak" for t, _ in r.tags) for r in group))
    for r in ran:
        if r.tags:
            rep["failures"].append({"id": r.item.id, "persona": r.item.persona, "type": r.item.type,
                                    "primary": r.tags[0][0], "tags": r.tags,
                                    "question": r.item.question})
    rep["tag_counts"] = Counter(f["primary"] for f in rep["failures"])
    if review is not None:
        changed = [(f["id"], f["primary"], review[f["id"]][0], review[f["id"]][1]) for f in rep["failures"]
                   if review[f["id"]][0] != f["primary"]]
        rep["review"] = {"counts": Counter(t for t, _ in review.values() if t != PASS),
                         "passed": sorted(q for q, (t, _) in review.items() if t == PASS), "changed": changed}
    return rep


# ================================================================ the author's review

def review_path(run_id: str) -> Path:
    return REVIEW_DIR / f"{run_id}.toml"


def review_sheet(rep: dict) -> str:
    """The review file's text: one block per failure, with the automatic tag and an empty `tag`."""
    rid, v = rep["run"]["run_id"], rep["judge_version"]
    lines = [f"# Review of the automatic error tags for run {rid} ({rep['run_name']}, judge {v}).",
             "# For each failure: read it with",
             f"#     python -m eval.report {rid} --show QID",
             "# then set `tag` to the primary tag you think is right, or \"pass\" if it didn't fail, and add",
             "# a short note when you change it. Don't edit `id` or `auto`.",
             "# Tags for answerable questions, in the order they are checked: "
             + ", ".join(t for t in TAGS if t != "missed_abstention") + ".",
             "# For declines: missed_abstention, then memory_leak, then unsupported_claim.",
             f"# Check any time: python -m eval.report {rid}  (it says how many are reviewed)", ""]
    for f in rep["failures"]:
        others = ", ".join(t for t, _ in f["tags"][1:])
        lines += ["[[review]]", f'id = "{f["id"]}"  # {" ".join(f["question"].split()[:10])} …',
                  f'auto = "{f["primary"]}"' + (f"  # also: {others}" if others else ""),
                  'tag = ""', 'note = ""', ""]
    return "\n".join(lines)


def read_review(path: Path, rep: dict) -> tuple[dict | None, str]:
    """(qid -> (tag, note) if the review is complete, else None; a one-line state). Raises SystemExit
    when the file no longer matches the run's automatic tags or holds an unknown tag."""
    with open(path, "rb") as f:
        blocks = tomllib.load(f).get("review", [])
    auto = {f["id"]: f["primary"] for f in rep["failures"]}
    got = {b.get("id"): b for b in blocks}
    problems = []
    if len(got) != len(blocks):
        problems.append("a question is listed twice")
    if set(got) != set(auto):
        problems.append("the failures differ from the run's (missing: "
                        + (", ".join(sorted(set(auto) - set(got))) or "none") + "; extra: "
                        + (", ".join(sorted(q for q in set(got) - set(auto) if q)) or "none") + ")")
    for qid, b in got.items():
        if qid in auto and b.get("auto") != auto[qid]:
            problems.append(f"{qid}: auto is {b.get('auto')!r} in the file but {auto[qid]!r} now")
        if b.get("tag") and b["tag"] not in TAGS + (PASS,):
            problems.append(f"{qid}: unknown tag {b['tag']!r}")
    if problems:
        raise SystemExit(f"{path.name} doesn't match run {rep['run']['run_id']}: " + "; ".join(problems)
                         + ". The automatic tags have changed since the sheet was written (a new grader or"
                           " judge verdicts?); move the old file aside and write a new sheet.")
    done = [q for q, b in got.items() if b.get("tag")]
    if len(done) < len(got):
        return None, f"review in progress: {len(done)} of {len(got)} reviewed ({path.name})"
    return {q: (b["tag"], b.get("note", "")) for q, b in got.items()}, f"reviewed by the author ({path.name})"


# ================================================================ printing

CAVEAT = ("Judge faithfulness is an upper bound: in calibration (set 2, j2) Prit agreed with 7 of the 14 answers "
          "the judge called faithful and 15 of the 16 it flagged; it misses dropped or bent conditions. So the "
          "judge-based tags (memory_leak, unsupported_claim) are a lower bound.")
LEGEND = ("Columns: hit@5 = a gold paragraph in the top 5 (answerable); MRR = mean of 1/rank of the first gold "
          "passage; cit. prec. = cited IDs in a gold family / all cited; abst. ok = declined or answered as it "
          "should; tool ok = right call or right non-call; calc = dates and Reg E tiers all in the answer; "
          "faithful* = judged faithful / judged (see the note); key facts = covered / all (answerable); "
          "failed = questions with any error tag.")
COLUMNS = ("n", "hit@5", "MRR", "cit. prec.", "abst. ok", "tool ok", "calc", "faithful*", "key facts", "failed")


def frac(pair) -> str:
    n, d = pair
    return f"{n}/{d}" if d else "-"


def row_cells(name: str, t: dict) -> list[str]:
    return [name, str(t["n"]), frac(t["hit_at_5"]), "-" if t["mrr"] is None else f"{t['mrr']:.2f}",
            frac(t["citation_precision"]), frac(t["abstain_correct"]), frac(t["tool_correct"]),
            frac(t["calc_exact"]), frac(t["faithful"]), frac(t["key_facts"]), frac(t["failed"])]


def table_rows(rep: dict) -> list[tuple[str, list[list[str]]]]:
    return [("Overall", [row_cells("all", rep["overall"])]),
            ("By persona", [row_cells(p, t) for p, t in rep["by_persona"].items()]),
            ("By type (counts only: too few to compare)", [row_cells(k, t) for k, t in rep["by_type"].items()])]


def header_lines(rep: dict) -> list[str]:
    run = rep["run"]
    what = run["ids"] or run["split"] or "?"
    lines = [f"Run {run['run_id']} ({rep['run_name']}), {what}, {run['n_items']} questions; commit "
             f"{(run['git_commit'] or '?')[:7]}{' (uncommitted changes)' if run['git_dirty'] else ''}, prompt "
             f"{run['prompt_version']}; judge {rep['judge_version']}"]
    if run["grader_version"] != rep["grader_now"]:
        lines.append(f"Graded again with {rep['grader_now']} (the run was logged with {run['grader_version']}).")
    if rep["judge_version"] != "j2":
        lines.append(f"Calibration figures below are for j2; this report uses {rep['judge_version']}.")
    return lines


def notes_lines(rep: dict) -> list[str]:
    o = rep["overall"]
    lines = [f"* {CAVEAT}",
             f"  Judged {o['judged']}: faithful {o['faith_labels']['faithful']}, partially "
             f"{o['faith_labels']['partially_faithful']}, unfaithful {o['faith_labels']['unfaithful']}.",
             "  Key facts: answerable questions only; a guard fallback text covers none."]
    if rep["no_verdict"]:
        lines.append(f"  No {rep['judge_version']} verdict for " + ", ".join(f"{q} ({s})" for q, s in rep["no_verdict"])
                     + f": the judge-based tags can't fire for them (python -m eval.judge {rep['run']['run_id']}).")
    if rep["not_run"]:
        lines.append("  Failed to run (not counted): " + ", ".join(q for q, _ in rep["not_run"]))
    return lines


def tag_lines(rep: dict) -> tuple[list[str], list[list[str]]]:
    """(intro lines, rows of [tag, automatic count, reviewed count])."""
    o = rep["overall"]
    rv = rep["review"]
    intro = [f"{o['failed'][0]} of {o['failed'][1]} questions failed (one primary tag each)"
             + (f"; after the review {sum(rv['counts'].values())}" if rv else "")]
    rows = [[t, str(rep["tag_counts"].get(t, 0))] + ([str(rv["counts"].get(t, 0))] if rv else [])
            for t in TAGS if rep["tag_counts"].get(t) or (rv and rv["counts"].get(t))]
    return intro, rows


def text(rep: dict) -> str:
    lines = header_lines(rep)
    for title, rows in table_rows(rep):
        lines += ["", title, "  " + f"{'':22}" + " ".join(f"{c:>10}" for c in COLUMNS)]
        lines += ["  " + f"{r[0]:22}" + " ".join(f"{c:>10}" for c in r[1:]) for r in rows]
    if rep["declines_by_category"]:
        lines += ["", "Declines by category (declined / n, memory_leak): " + ", ".join(
            f"{c} {n}/{d}" + (f" ({leak} leak)" if leak else "") for c, (n, d, leak) in rep["declines_by_category"].items())]
    lines += ["", LEGEND, ""] + notes_lines(rep)
    intro, rows = tag_lines(rep)
    lines += ["", "Error tags"] + ["  " + intro[0]]
    head = ["automatic"] + (["reviewed"] if rep["review"] else [])
    if rows:
        lines.append("  " + f"{'':20}" + " ".join(f"{h:>10}" for h in head))
        lines += ["  " + f"{r[0]:20}" + " ".join(f"{c:>10}" for c in r[1:]) for r in rows]
    if rep["review"]:
        rv = rep["review"]
        if rv["changed"]:
            lines += ["  Changed by the review:"] + [f"    {q}: {a} -> {t}" + (f" ({n})" if n else "")
                                                     for q, a, t, n in rv["changed"]]
    if rep.get("review_state"):
        lines.append("  " + rep["review_state"])
    if rep["failures"]:
        lines += ["", "Failures (primary tag first)"]
        for f in rep["failures"]:
            lines.append(f"  {f['id']} {f['persona']:10} {f['type']:20} " + ", ".join(t for t, _ in f["tags"]))
            lines += [f"      {t}: {why}" for t, why in f["tags"]]
    return "\n".join(lines)


def md_table(head: list[str], rows: list[list[str]]) -> list[str]:
    esc = lambda c: c.replace("|", "\\|")          # noqa: E731
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(esc, r)) + " |" for r in rows]


def markdown(rep: dict) -> str:
    lines = ["# Evaluation report", ""] + [line + "  " for line in header_lines(rep)]
    for title, rows in table_rows(rep):
        lines += ["", f"## {title}", ""] + md_table(["", *COLUMNS], rows)
    if rep["declines_by_category"]:
        lines += ["", "## Declines by category", ""] + md_table(
            ["category", "declined", "memory_leak"],
            [[c, f"{n}/{d}", str(leak)] for c, (n, d, leak) in rep["declines_by_category"].items()])
    lines += ["", "## Notes", "", LEGEND, ""] + [line.lstrip(" *") for line in notes_lines(rep)]
    intro, rows = tag_lines(rep)
    lines += ["", "## Error tags", "", intro[0], ""]
    if rows:
        lines += md_table(["tag", "automatic"] + (["reviewed"] if rep["review"] else []), rows)
    if rep.get("review_state"):
        lines += ["", rep["review_state"]]
    if rep["failures"]:
        lines += ["", "## Failures", ""] + md_table(
            ["question", "persona", "type", "tags", "evidence"],
            [[f["id"], f["persona"], f["type"], ", ".join(t for t, _ in f["tags"]),
              "; ".join(f"{t}: {why}" for t, why in f["tags"])] for f in rep["failures"]])
    return "\n".join(lines) + "\n"


def show(db, run_id: str, qid: str, version: str = J.JUDGE_VERSION) -> str:
    """One question in full: the question, the answer, the tags with their evidence and the judge's verdict."""
    _, rows = load_rows(db, run_id, version)
    row = next((r for r in rows if r.item.id == qid), None)
    if row is None:
        raise SystemExit(f"not in run {run_id}: {qid}")
    out = [f"{qid} ({row.item.persona}, {row.item.type}" + (f", {row.item.abstain_category}" if row.item.should_abstain else "")
           + ")", "", "Question:", "  " + row.item.question.strip()]
    if row.metrics is None:
        return "\n".join(out + ["", f"Failed to run: {row.error}"])
    resp = json.loads(db.execute("SELECT response_json FROM results WHERE run_id = ? AND qid = ?",
                                 (run_id, qid)).fetchone()[0])
    m = row.metrics
    out += ["", "Answer" + (" (declined)" if resp["abstained"] else "") + ":", "  " + resp["answer"].strip(), "",
            f"Retrieval: first gold rank {m['first_gold_rank']}, gold paragraphs {m['gold_found']} of {m['gold_total']}; "
            f"tool {m['tool_outcome']}; calculation {m['calc_exact']}", "",
            "Tags: " + (", ".join(t for t, _ in row.tags) or "none (passed)")]
    out += [f"  {t}: {why}" for t, why in row.tags]
    out += ["", J.show(db, run_id, qid, version)]
    return "\n".join(out)


# ================================================================ command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.report", description=__doc__.split("\n\n")[0])
    ap.add_argument("run_id", help="a logged run (python -m eval.run --runs)")
    ap.add_argument("--judge", default=J.JUDGE_VERSION, choices=list(J.RUBRICS),
                    help=f"which rubric's verdicts to use (default {J.JUDGE_VERSION})")
    ap.add_argument("--markdown", type=Path, metavar="FILE", help="also write the report as Markdown")
    ap.add_argument("--show", metavar="QID", help="one question in full, and stop")
    ap.add_argument("--review-sheet", action="store_true", help="write eval/review/RUN_ID.toml for the author's review")
    ap.add_argument("--db", type=Path, default=R.DB_FILE, help="the SQLite file (default runs.db)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to another encoding

    if not args.db.exists():
        print(f"No {args.db.name} here: run this from the repo root, where eval.run logs runs.")
        return 2
    db = J.connect(args.db)
    if args.show:
        print(show(db, args.run_id, args.show, args.judge))
        return 0
    rep = build(db, args.run_id, args.judge)
    path = review_path(args.run_id)
    shown = path.relative_to(R.REPO).as_posix() if path.is_relative_to(R.REPO) else str(path)
    if args.review_sheet:
        if path.exists():
            print(f"{shown} already exists; it isn't overwritten.")
            return 2
        if not rep["failures"]:
            print("No failures, so there is nothing to review.")
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(review_sheet(rep), encoding="utf-8", newline="\n")
        print(f"Wrote {shown}: {len(rep['failures'])} failures to review.")
        return 0
    if path.exists():
        review, state = read_review(path, rep)
        if review is not None:
            rep = build(db, args.run_id, args.judge, review=review)
        rep["review_state"] = state
    print(text(rep))
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(rep), encoding="utf-8", newline="\n")
        print(f"\nWrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
