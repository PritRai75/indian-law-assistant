"""The golden set: format, loading and checks (spec: Golden set; tracker t3-format).

The questions live in eval/golden.toml, one [[question]] block each, written and labelled by hand.
TOML has no null, so a field that doesn't apply is simply left out.

Two stages:
  draft     pass 1: id, persona, type (a guess), question, and optionally a private trap note
  labelled  pass 2: adds regulation, gold_ids, reference_answer, key_facts, should_abstain,
            abstain_category, expected_tool_call / expected_result, uncertain, today

After all questions were labelled, `python -m eval.split --write` gave each a `split` (dev or test,
session 18); from then on every question must have one, and a split never changes.

`check()` returns every problem it finds as a readable line, so one run shows them all;
tests/test_golden.py fails while the real file has any. `python -m eval.golden` prints the counts
and the problems.

`python -m eval.golden --skeleton q011 q012` (or `q011-q020`) starts pass 2 (session 16): each draft
becomes stage "labelled" with empty pass-2 fields (SKELETON) that fail the checks until filled in.
Only the stage line and the added lines change; the file must read back as exactly that.

Gold IDs match as a family (Prit's choice, session 14): a gold ID covers itself, the paragraphs
under it and the comments on any of them, the same rule as the search's named sets and
`python -m eval.lookup`. `in_family()` is that rule; eval/run.py scores retrieval and citations
with it. `warnings()` flags a gold ID whose family spans more than one rule chunk, since a broad
ID makes retrieval easy to hit; a warning never fails the check.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import tomllib
from collections import Counter
from datetime import date as Date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

REPO = Path(__file__).resolve().parents[1]
GOLDEN_FILE = REPO / "eval" / "golden.toml"
CHUNKS_FILE = REPO / "index" / "chunks.json"

TYPES = ("unauthorized", "dispute", "deadline", "fees_terms", "payments_disclosures",
         "reg_e_account", "cross_reg", "decline")
ABSTAIN_CATEGORIES = {
    "network_rules": "card network rules (Visa, Mastercard), including 'zero liability'",
    "state_law": "state law",
    "issuer_terms": "the issuer's or bank's own account terms and fees",
    "other_law": "the statutes behind Z and E, other regulations (FCRA, Reg F, Reg CC, UDAAP), court decisions",
    "excluded_part": "parts of Z and E not indexed (Z subparts C-F, E subpart B remittances, appendices)",
    "not_in_force": "a vacated, overturned or not-yet-effective rule",
    "sign_off": "a compliance sign-off or an outcome prediction",
}
LIABILITY_TIERS = ("max_50", "max_500", "later_transfers_unprotected", "none")
MAX_QUESTION_CHARS = 2000          # app/config.py refuses longer questions

class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ToolCall(Model):
    name: Literal["dispute_timeline"]
    args: dict


class Deadline(Model):
    what: str = Field(min_length=1)          # your own words, for reading the file
    date: Date
    rule: str = Field(min_length=1)          # one provision ID, e.g. 1005.11(c)(1)


class ToolResult(Model):
    liability_tiers: list[Literal[LIABILITY_TIERS]] = []
    deadlines: list[Deadline] = []
    notice_timely: bool | None = None


class GoldenItem(Model):
    id: str = Field(pattern=r"^q\d{3}$")
    stage: Literal["draft", "labelled"]
    persona: Literal["frontline", "compliance"]
    type: Literal[TYPES]
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    trap: str | None = None
    # pass 2
    regulation: Literal["E", "Z", "both"] | None = None
    gold_ids: list[str] | None = None
    reference_answer: str | None = None
    key_facts: list[str] | None = None
    should_abstain: bool | None = None
    abstain_category: Literal[tuple(ABSTAIN_CATEGORIES)] | None = None
    expected_tool_call: ToolCall | None = None
    expected_result: ToolResult | None = None
    uncertain: bool = False
    today: Date | None = None                 # the date the question treats as "today", if it says
    split: Literal["dev", "test"] | None = None


# ------------------------------------------------------------------ loading

def read(path: Path = GOLDEN_FILE) -> list[dict]:
    """The raw [[question]] blocks from the TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    extra = set(data) - {"question"}
    if extra:
        raise ValueError(f"{path.name}: unexpected top-level keys {sorted(extra)}; every block must be [[question]]")
    return data.get("question", [])


def load(path: Path = GOLDEN_FILE) -> list[GoldenItem]:
    """The items, validated. Raises ValueError listing every problem if there are any."""
    items, problems = parse(read(path))
    problems += check(items)
    if problems:
        raise ValueError(f"{path.name} has {len(problems)} problem(s):\n" + "\n".join(problems))
    return items


def parse(raw: list[dict]) -> tuple[list[GoldenItem], list[str]]:
    items, problems = [], []
    for n, block in enumerate(raw, 1):
        label = block.get("id", f"block {n}")
        try:
            items.append(GoldenItem(**block))
        except ValidationError as e:
            for err in e.errors():
                where = ".".join(str(x) for x in err["loc"]) or "(block)"
                problems.append(f"{label}: {where}: {err['msg']}")
    return items, problems


# ------------------------------------------------------------------ checks

def provision_statuses(path: Path = CHUNKS_FILE) -> dict[str, str]:
    """Every provision ID in the index, mapped to its chunk's status (a chunk never mixes statuses)."""
    with open(path, encoding="utf-8") as f:
        return {p: c["status"] for c in json.load(f) for p in c["provision_ids"]}


def check_id(pid: str, statuses: dict[str, str]) -> str | None:
    """None if the ID exists exactly as written, else a message with the likely intended ID."""
    if pid in statuses:
        return None
    by_lower = {p.lower(): p for p in statuses}
    if pid.lower() in by_lower:
        return f"{pid!r} is spelled {by_lower[pid.lower()]!r} in the index"
    close = difflib.get_close_matches(pid, list(statuses), n=3, cutoff=0.8)
    hint = f"; did you mean {', '.join(map(repr, close))}?" if close else ""
    return f"{pid!r} is not a provision ID in the index{hint}"


def in_family(gold_id: str, provision_id: str) -> bool:
    """True if `provision_id` is `gold_id`, a paragraph under it, or a comment on either (any case).
    A comment gold ID ("1005.6(b)-1") covers only that comment."""
    from app.retrieve import covered_by
    return covered_by(gold_id.lower(), provision_id.lower())


def rule_chunk_owners(path: Path = CHUNKS_FILE) -> dict[str, str]:
    """Every provision ID in a rule chunk, mapped to that chunk's ID."""
    with open(path, encoding="utf-8") as f:
        return {p: c["id"] for c in json.load(f) if c["kind"] == "rule" for p in c["provision_ids"]}


BROAD_AFTER = 1          # rule chunks a gold ID's family may span before it is called broad


def warnings(items: list[GoldenItem], owners: dict[str, str] | None = None) -> list[str]:
    """Labelled items whose gold IDs are broad. Advice only: never a problem."""
    owners = rule_chunk_owners() if owners is None else owners
    out = []
    for item in items:
        if item.stage != "labelled" or not item.gold_ids:
            continue
        for gid in item.gold_ids:
            chunks = sorted({c for p, c in owners.items() if in_family(gid, p)})
            if len(chunks) > BROAD_AFTER:
                shown = ", ".join(chunks[:4]) + (", ..." if len(chunks) > 4 else "")
                out.append(f"{item.id}: gold ID {gid} covers {len(chunks)} rule chunks ({shown}), so almost any of "
                           "them counts as a retrieval hit. Use a narrower ID if one answers the question; "
                           "keep it if the whole paragraph is the answer.")
    return out


def tool_spec():
    from app.tools import TOOL_SPEC
    return TOOL_SPEC["toolSpec"]["inputSchema"]["json"]


def check(items: list[GoldenItem], statuses: dict[str, str] | None = None) -> list[str]:
    """Every rule the file must follow, beyond the field types. One line per problem."""
    statuses = provision_statuses() if statuses is None else statuses
    problems = []
    for pid, n in Counter(i.id for i in items).items():
        if n > 1:
            problems.append(f"{pid}: id used {n} times")
    for text, n in Counter(" ".join(i.question.split()).lower() for i in items).items():
        if n > 1:
            problems.append(f"question used {n} times: {text[:60]!r}")
    for item in items:
        problems += [f"{item.id}: {p}" for p in check_item(item, statuses)]
    if any(i.split for i in items):                 # session 18: once the split exists, it covers everything
        problems += [f"{i.id}: no split; every question needs one (dev or test) once the split exists"
                     for i in items if not i.split]
    return problems


PASS_2_FIELDS = ("regulation", "gold_ids", "reference_answer", "key_facts", "should_abstain",
                 "abstain_category", "expected_tool_call", "expected_result", "today", "split")


def check_item(item: GoldenItem, statuses: dict[str, str]) -> list[str]:
    p = []
    if item.stage == "draft":
        filled = [f for f in PASS_2_FIELDS if getattr(item, f) is not None]
        if filled:
            p.append(f"draft has pass-2 fields {filled}; set stage = \"labelled\" when you label it")
        return p

    # labelled
    for f in ("gold_ids", "reference_answer", "key_facts", "should_abstain"):
        if getattr(item, f) is None:
            p.append(f"labelled but {f} is missing")
    if p:
        return p
    if not item.reference_answer:
        p.append("reference_answer is empty")
    for pid in item.gold_ids:
        if (msg := check_id(pid, statuses)):
            p.append(f"gold_ids: {msg}")
    if len(set(item.gold_ids)) < len(item.gold_ids):
        p.append("gold_ids has a repeat")
    if any(not f for f in item.key_facts):
        p.append("key_facts has an empty entry")

    if item.should_abstain:
        if item.type != "decline":
            p.append("should_abstain is true, so type must be \"decline\"")
        if item.abstain_category is None:
            p.append("should_abstain is true, so abstain_category is needed: one of " + ", ".join(ABSTAIN_CATEGORIES))
        if len(item.key_facts) > 3:
            p.append("key_facts: at most 3")
        if item.expected_tool_call or item.expected_result:
            p.append("a decline expects no tool call or result")
    else:
        if item.type == "decline":
            p.append("type is \"decline\" but should_abstain is false")
        if item.abstain_category is not None:
            p.append("abstain_category is only for declines")
        if item.regulation is None:
            p.append("regulation (E, Z or both) is needed for an answerable question")
        if not item.gold_ids:
            p.append("an answerable question needs at least one gold ID")
        if not 1 <= len(item.key_facts) <= 3:
            p.append("key_facts: 1 to 3 for an answerable question")
        for pid in item.gold_ids:
            if statuses.get(pid, "in_force") != "in_force":
                p.append(f"gold_ids: {pid} has status {statuses[pid]!r}; the assistant declines on it "
                         "(decision 27), so this question is a decline (category not_in_force) or needs another ID")

    if (item.expected_tool_call is None) != (item.expected_result is None):
        p.append("expected_tool_call and expected_result go together: give both or neither")
    if item.expected_tool_call is not None:
        p += check_tool_args(item.expected_tool_call.args)
    if item.expected_result is not None:
        for d in item.expected_result.deadlines:
            if (msg := check_id(d.rule, statuses)):
                p.append(f"expected_result deadline {d.date}: {msg}")
    return p


def as_strings(args: dict) -> dict:
    """TOML reads unquoted dates (2026-11-05) as dates; the tool takes 'YYYY-MM-DD' strings, as the model sends."""
    def s(v):
        return v.isoformat() if isinstance(v, Date) else v
    return {k: [s(x) for x in v] if isinstance(v, list) else s(v) for k, v in args.items()}


def check_tool_args(args: dict) -> list[str]:
    """The arguments must fit the tool's input schema and be accepted by the tool itself."""
    from app.tools import dispute_timeline
    schema, p = tool_spec(), []
    props = schema["properties"]
    for name in args:
        if name not in props:
            p.append(f"expected_tool_call: unknown argument {name!r}; known: {', '.join(props)}")
    for name in schema["required"]:
        if name not in args:
            p.append(f"expected_tool_call: missing required argument {name!r}")
    for name, value in args.items():
        enum = props.get(name, {}).get("enum")
        if enum and value not in enum:
            p.append(f"expected_tool_call: {name} must be one of {enum}, got {value!r}")
    if p:
        return p
    result = dispute_timeline(**as_strings(args))
    if "error" in result:
        e = result["error"]
        p.append(f"expected_tool_call: the tool rejects these arguments ({e['argument']}: {e['message']})")
    return p


def compare_with_tool(item: GoldenItem) -> list[str]:
    """Hand-computed expected_result against what the tool returns for expected_tool_call.

    Labels are computed by hand from the regulation text (labeling rule 3); this is the cross-check
    afterwards. A mismatch means the label or the tool is wrong: investigate before trusting either.
    """
    from app.tools import dispute_timeline
    if item.expected_tool_call is None or item.expected_result is None:
        return []
    if check_tool_args(item.expected_tool_call.args):
        return []                                 # unknown, missing or rejected arguments: reported by check()
    got = dispute_timeline(**as_strings(item.expected_tool_call.args))
    if "error" in got:
        return []                                 # reported by check()
    return [f"{item.id}: {x}" for x in compare_result(item.expected_result, got)]


def compare_result(want: ToolResult, got: dict) -> list[str]:
    """Differences between a labelled result and one dispute_timeline result (empty: they agree)."""
    if "error" in got:
        return [f"the tool returned an error ({got['error'].get('message', '')})"]
    p = []
    if sorted(got["liability_tiers"] or []) != sorted(want.liability_tiers):
        p.append(f"liability_tiers: label {want.liability_tiers}, tool {got['liability_tiers']}")
    if want.notice_timely is not None and got["notice_timely"] != want.notice_timely:
        p.append(f"notice_timely: label {want.notice_timely}, tool {got['notice_timely']}")
    for d in want.deadlines:
        same_rule = [g for g in got["deadlines"] if rule_matches(d.rule, g["rule"])]
        if not same_rule:
            p.append(f"deadline {d.rule}: the tool gives no deadline under this rule")
        elif d.date.isoformat() not in {g["date"] for g in same_rule}:
            p.append(f"deadline {d.rule}: label {d.date}, tool {', '.join(g['date'] for g in same_rule)}")
    return p


def rule_matches(label_rule: str, tool_rule: str) -> bool:
    """The tool's rule field can name more than one paragraph ("1005.11(c)(2)(i), (c)(3)(i)")."""
    first = tool_rule.split(",")[0].strip()
    section = re.match(r"\d{4}\.\d+", first)
    rules = [first] + [section.group(0) + r.strip() for r in tool_rule.split(",")[1:]] if section else [first]
    return label_rule in rules


# ------------------------------------------------------------------ skeleton for pass 2

# The lines `--skeleton` adds to a draft. The three plain fields hold empty values that the checks
# reject (an empty reference answer, an empty key fact), and should_abstain is left as a comment, so
# python -m eval.golden lists every skeleton item until it is really labelled: nothing passes by
# being forgotten. gold_ids = [] is valid for a decline, which may list none.
SKELETON = f'''\
# --- pass 2: fill these in, then run python -m eval.golden (it lists what is still missing) ---
# regulation = "E"          # "E", "Z" or "both"; leave out for most declines
# today = 2026-11-03        # only if the question names its own "today"
gold_ids = []
reference_answer = """
"""
key_facts = [""]
# should_abstain = false    # true for a decline, and then also abstain_category
# abstain_category = ""     # {", ".join(ABSTAIN_CATEGORIES)}
# uncertain = true          # only when unsure: leaves the question out of scoring
# Tool call and result, only if the answer needs dispute_timeline: see the example at the top of the file.'''

SKELETON_FIELDS = {"gold_ids": [], "reference_answer": "", "key_facts": [""]}
_DRAFT_STAGE = re.compile(r'^\s*stage\s*=\s*"draft"\s*(#.*)?$')


def expand_ids(args: list[str]) -> list[str]:
    """"q011", and ranges "q011-q020" written out. Raises ValueError on anything else."""
    out = []
    for a in args:
        m = re.fullmatch(r"q(\d{3})(?:-q(\d{3}))?", a.strip())
        if not m:
            raise ValueError(f"{a!r}: give question ids like q011, or a range like q011-q020")
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if hi < lo:
            raise ValueError(f"{a!r}: the range runs backwards")
        out += [f"q{n:03d}" for n in range(lo, hi + 1)]
    return list(dict.fromkeys(out))


def add_skeletons(text: str, ids: list[str]) -> str:
    """The file's text with each listed draft turned into a labelled block with empty pass-2 fields.

    Only two things change per block: `stage = "draft"` becomes "labelled", and SKELETON goes in after
    the block's last field (before any comment lines that lead into the next block). Everything else,
    line endings included, is kept byte for byte. All or nothing: any problem raises ValueError
    listing them all, and nothing is changed. The result is re-read with tomllib and must equal the
    old file plus exactly the skeleton fields, or the change is refused.
    """
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    old = tomllib.loads(text).get("question", [])
    starts = [n for n, line in enumerate(lines) if line.strip() == "[[question]]"]
    if len(starts) != len(old):
        raise ValueError(f"found {len(starts)} [[question]] lines but {len(old)} blocks; "
                         "is a [[question]] line inside a text field?")
    where = {block.get("id"): k for k, block in enumerate(old)}
    problems, edits = [], []
    for qid in ids:
        if qid not in where:
            problems.append(f"{qid}: not in the file")
            continue
        k = where[qid]
        if old[k].get("stage") != "draft":
            problems.append(f"{qid}: stage is {old[k].get('stage')!r}; the skeleton is only for drafts")
            continue
        s, e = starts[k], (starts[k + 1] if k + 1 < len(starts) else len(lines))
        stage = [n for n in range(s, e) if _DRAFT_STAGE.match(lines[n])]
        if len(stage) != 1:
            problems.append(f"{qid}: expected one line stage = \"draft\" in the block, found {len(stage)}")
            continue
        if any(lines[n].lstrip().startswith("[") for n in range(s + 1, e)):
            problems.append(f"{qid}: a draft block should have no [question....] tables")
            continue
        last = e - 1
        while last > s and (not lines[last].strip() or lines[last].lstrip().startswith("#")):
            last -= 1
        edits.append((qid, stage[0], last))
    if problems:
        raise ValueError("\n".join(problems))

    for qid, stage_line, last in sorted(edits, key=lambda x: -x[2]):
        lines[stage_line] = lines[stage_line].replace('"draft"', '"labelled"', 1)
        lines[last + 1:last + 1] = SKELETON.split("\n")
    new_text = nl.join(lines)

    expected = [dict(block) for block in old]
    for qid in ids:
        expected[where[qid]].update(stage="labelled", **SKELETON_FIELDS)
    if tomllib.loads(new_text).get("question", []) != expected:
        raise ValueError("the edited file doesn't read back as expected; nothing was changed")
    return new_text


def skeleton(path: Path, ids: list[str]) -> list[str]:
    """Apply add_skeletons to the file at `path` in place. Returns the ids changed."""
    ids = expand_ids(ids)
    text = path.read_bytes().decode("utf-8")
    new_text = add_skeletons(text, ids)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(new_text.encode("utf-8"))
    tmp.replace(path)
    return ids


# ------------------------------------------------------------------ summary

def summary(items: list[GoldenItem]) -> str:
    lines = [f"{len(items)} questions: " + ", ".join(f"{n} {s}" for s, n in sorted(Counter(i.stage for i in items).items()))]
    for field in ("persona", "type"):
        lines.append(f"  by {field}: " + ", ".join(f"{k} {n}" for k, n in sorted(Counter(getattr(i, field) for i in items).items())))
    labelled = [i for i in items if i.stage == "labelled"]
    if labelled:
        unfinished = [i for i in labelled if i.should_abstain is None]       # e.g. a skeleton not yet filled in
        declines = [i for i in labelled if i.should_abstain]
        lines.append(f"  labelled: {len(labelled) - len(declines) - len(unfinished)} answerable, {len(declines)} declines, "
                     f"{sum(i.uncertain for i in labelled)} uncertain, "
                     f"{sum(i.expected_tool_call is not None for i in labelled)} expecting the tool"
                     + (f", {len(unfinished)} unfinished (should_abstain not set)" if unfinished else ""))
        if declines:
            lines.append("  declines by category: " + ", ".join(
                f"{k} {n}" for k, n in sorted(Counter(i.abstain_category for i in declines).items())))
    if any(i.split for i in items):
        lines.append("  split: " + ", ".join(f"{k} {n}" for k, n in sorted(Counter(i.split or "none" for i in items).items())))
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m eval.golden",
                                 description="Check the golden set; with --skeleton, start pass 2 on drafts.")
    ap.add_argument("path", nargs="?", type=Path, default=GOLDEN_FILE, help="default: eval/golden.toml")
    ap.add_argument("--skeleton", nargs="+", metavar="ID",
                    help="turn these drafts (q011 q012, or q011-q020) into labelled blocks with empty "
                         "pass-2 fields to fill in")
    args = ap.parse_args(argv)
    path = args.path
    if args.skeleton:
        try:
            done = skeleton(path, args.skeleton)
        except ValueError as e:
            print(f"Nothing changed:\n{e}")
            return 1
        print(f"{len(done)} draft(s) now labelled, with empty pass-2 fields: {', '.join(done)}")
        print("Fill them in. Until then the check below lists them as problems, and pytest fails.\n")
    items, problems = parse(read(path))
    problems += check(items)
    problems += [x for i in items for x in compare_with_tool(i)]
    print(summary(items))
    advice = warnings(items)
    if advice:
        print(f"\n{len(advice)} warning(s), advice only:")
        print("\n".join(f"  {x}" for x in advice))
    if problems:
        print(f"\n{len(problems)} problem(s):")
        print("\n".join(f"  {x}" for x in problems))
        return 1
    print("\nNo problems.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
