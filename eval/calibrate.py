"""Judge calibration, part 1: the blind labelling sheet (spec: judge calibration; tracker t6-calib).
Prit labels about 40 logged answers himself, before any judge verdict on them exists, so agreement
measures the judge against an independent reading.

From the repo root (no AWS call):
    python -m eval.calibrate --sheet RUN_A RUN_B --set set2   pick the answers (30 by default), write the sheet and labels file
    (--set set2, set3, ... puts a new set in its own folder; without --set, commands mean set 1)
    python -m eval.calibrate --check                        check labels.toml: what's filled in, what's wrong
    python -m eval.calibrate --compare                      the judge against the labels (after judging both runs)
    python -m eval.calibrate --compare --save               the same, also saved as eval/calibration/compare-j1.json
    python -m eval.calibrate --compare --judge-version j1   an earlier rubric's verdicts (both rounds are reported)

Sets (session 20). Set 1 (session 19: 40 answers from r20261105T140818Z and r20261105T141055Z) is kept
as committed but set aside as a measure: items c28-c40 were labelled against the wrong questions (their
notes describe q028-q040). Its disagreements were used to design rubric j2. Set 2 (Prit: 30 fresh
answers from two new dev runs with settings different from set 1's, so the answers differ at
temperature 0) is labelled blind with j2 already fixed, then judged with j1 and j2. --sheet refuses a
run that another set already uses. The labels file shows each question's first words above its block,
so a mix-up shows while labelling.

Files:
- scratch/calibration/sheet.html   read this in a browser (git-ignored; it holds the passages, so it's big)
- eval/calibration/labels.toml     fill this in, then commit it (the spec publishes the calibration set)
- eval/calibration/key.json        which run and question each item came from. Don't open it while
                                   labelling: it says which answers came from the weak run.

What Prit labels for each item, the same two things the judge decides (eval/judge.py, rubric j1):
- faithfulness: faithful, partially_faithful or unfaithful, against the passages and tool results shown;
- key_facts: one true or false per key fact, in order: does the answer state it?
Optional: note, a few words on anything odd.

How the 40 are picked (Prit's choice, session 19: 40 by seeded pick; details Claude's, open to push-back):
1. Every logged answer in the two runs, except questions that failed to run and the guard's fallback
   texts (eval/judge.py doesn't judge those either).
2. 20 per run; within each run, answerable and declines in the dev split's proportion (16 : 8, so 13
   and 7). If a group has too few answers, the rest comes from the run's other group, then the other run.
3. Inside each group, a seeded shuffle (CALIBRATION_SEED) picks which; then the 40 are shuffled again
   and numbered c01-c40 in that order, so neither the run nor the question order shows.
The sheet never shows the run, the question ID, code-graded metrics or any judge output.
The rules at the top of the sheet follow the current rubric (j2 from session 20; set 1's sheet
showed j1's rules).

Part 2, --compare (Claude's design, session 20, open to Prit's push-back). It runs only once the labels
are complete and committed, and reads the judge's stored verdicts for the same 40 answers (runs.db,
one rubric version). It reports:
- faithfulness, 3 labels: agreement with a Wilson 95% interval; Cohen's kappa with a bootstrap 95%
  interval; the 3x3 confusion matrix (rows Prit, columns judge); how many answers each called faithful;
  where the judge was more lenient or stricter (faithful > partially_faithful > unfaithful);
- faithful or not, 2 labels (partially and unfaithful together, the README's "fully faithful" share):
  agreement and kappa, the same way;
- key facts: agreement over every fact and kappa, both with bootstrap intervals, and the 2x2 counts;
- the same numbers by run, answerable/decline and persona (no intervals: the slices are small);
- every disagreement with Prit's note, the judge's rationale and the command to see the verdict.
Kappa is plain (unweighted) Cohen's kappa: a faithful-vs-partially split counts as much as
faithful-vs-unfaithful. The bootstrap resamples whole questions (2000 times, seeded), not answers or
facts: 40 answers cover 23 questions, and answers to one question share its key facts, so they aren't
independent. The Wilson interval (session 19's plan) treats the 40 answers as independent, so it is a
little narrower than it should be. The spec asks for an interval on the confusion matrix too; its
cells (often 0 to 3 answers) are shown as counts only.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import sys
import tomllib
from datetime import date
from pathlib import Path

from eval import golden as G
from eval import judge as J
from eval import run as R

CALIBRATION_SEED = 20261105
TARGET = 30                                 # set 2 (Prit, session 20); set 1 had 40
ANSWERABLE_SHARE = 16 / 24                  # the dev split: 16 answerable, 8 declines
LABELS = ("faithful", "partially_faithful", "unfaithful")

CAL_DIR = R.REPO / "eval" / "calibration"
SHEET_FILE = R.REPO / "scratch" / "calibration" / "sheet.html"      # set 1's files (session 19)
LABELS_FILE = CAL_DIR / "labels.toml"
KEY_FILE = CAL_DIR / "key.json"


def set_paths(name: str = "set1") -> dict:
    """The sheet, labels and key of one calibration set. Set 1 (session 19) sits at the top of
    eval/calibration/; later sets each have a folder, eval/calibration/set2/ and so on."""
    if name == "set1":
        return {"sheet": SHEET_FILE, "labels": LABELS_FILE, "key": KEY_FILE}
    if not name.startswith("set") or not name[3:].isdigit():
        raise SystemExit(f"a set is named set1, set2, ...; not {name!r}")
    return {"sheet": R.REPO / "scratch" / "calibration" / name / "sheet.html",
            "labels": CAL_DIR / name / "labels.toml", "key": CAL_DIR / name / "key.json"}


def all_keys(cal_dir: Path | None = None) -> list[Path]:
    """Every calibration set's key.json."""
    cal_dir = cal_dir or CAL_DIR
    return [k for k in [cal_dir / "key.json", *sorted(cal_dir.glob("set*/key.json"))] if k.exists()]


def run_name(db, run_id: str) -> str:
    """A short name from the run's logged settings, e.g. "hybrid k5", "vector k3 no tool" or "hybrid k5
    prompt p3" (the prompt only when it isn't the default)."""
    from app.prompts import PROMPT_VERSION
    row = db.execute("SELECT config_json, prompt_version FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    try:
        c = json.loads(row[0])
        return (f"{c['search_mode']} k{c['k']}" + ("" if c["tool_on"] else " no tool")
                + ("" if row[1] in (None, PROMPT_VERSION) else f" prompt {row[1]}"))
    except (TypeError, KeyError, ValueError):
        return run_id


# ================================================================ picking the 40

def candidates(db, run_id: str) -> list[tuple]:
    """(run_id, item, response, today) for every judgeable answer in one run."""
    return [(run_id, item, resp, today) for item, resp, today in J.logged_answers(db, run_id)
            if not J.skip_reason(resp)]


def targets(n_runs: int, total: int = TARGET) -> list[tuple[int, int]]:
    """(answerable, declines) to take from each run."""
    per_run = [total // n_runs + (r < total % n_runs) for r in range(n_runs)]
    return [(round(n * ANSWERABLE_SHARE), n - round(n * ANSWERABLE_SHARE)) for n in per_run]


def pick(pools: list[list[tuple]], seed: int = CALIBRATION_SEED, total: int = TARGET) -> list[tuple]:
    """The calibration items, in sheet order. `pools` holds one list of candidates per run."""
    rng = random.Random(seed)
    groups = []                                   # per run: [answerable, declines], each shuffled
    for pool in pools:
        ans = sorted((c for c in pool if not c[1].should_abstain), key=lambda c: c[1].id)
        dec = sorted((c for c in pool if c[1].should_abstain), key=lambda c: c[1].id)
        rng.shuffle(ans)
        rng.shuffle(dec)
        groups.append([ans, dec])
    chosen, short = [], 0
    for (want_a, want_d), (ans, dec) in zip(targets(len(pools), total), groups):
        take_a, take_d = min(want_a, len(ans)), min(want_d, len(dec))
        gap = (want_a - take_a) + (want_d - take_d)
        extra_a = min(gap, len(ans) - take_a)     # fill from the run's other group first
        take_a += extra_a
        take_d += min(gap - extra_a, len(dec) - take_d)
        chosen += ans[:take_a] + dec[:take_d]
        del ans[:take_a], dec[:take_d]
        short += want_a + want_d - take_a - take_d
    leftovers = [c for ans, dec in groups for c in ans + dec]   # then from the other run, in shuffled order
    chosen += leftovers[:short]
    rng.shuffle(chosen)
    return chosen


# ================================================================ the sheet

CSS = """
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a64; --line:#dedcd4; --card:#ffffff; --accent:#2f5d8a;
        --soft:#eef3f8; --warn:#8a4b2f; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ecebe6; --muted:#a3a29b; --line:#34332f;
        --card:#1f1f1d; --accent:#8fb6dc; --soft:#22303d; --warn:#e0a47f; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.55 system-ui, sans-serif; }
main { max-width: 900px; margin: 0 auto; padding: 24px 16px 80px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; } h2 { font-size: 1.25rem; margin: 0; }
h3 { font-size: .95rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 20px 0 6px; }
.item { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 20px; margin: 28px 0; }
.rules { background: var(--soft); border-radius: 10px; padding: 14px 18px; }
.rules li { margin: 4px 0; }
.answer { border-left: 4px solid var(--accent); padding: 8px 14px; background: var(--soft); border-radius: 0 8px 8px 0; }
.declined { color: var(--warn); font-weight: 600; }
pre { white-space: pre-wrap; word-wrap: break-word; font: 14px/1.5 ui-monospace, Consolas, monospace;
      background: var(--bg); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; margin: 6px 0; }
details { margin: 6px 0; } summary { cursor: pointer; color: var(--accent); }
nav { position: sticky; top: 0; background: var(--bg); padding: 8px 0; border-bottom: 1px solid var(--line);
      font-size: .9rem; line-height: 1.9; }
nav a { color: var(--accent); margin-right: 10px; }
ol.facts li { margin: 4px 0; }
.muted { color: var(--muted); font-size: .9rem; }
"""

RULES = """
<ul>
<li><b>Judge only against the passages and tool results shown</b>, not what you know or the reference
answer. A claim is supported when a passage or tool result states it or it follows directly from one.
Facts the question states (dates, weekdays, amounts) count as given.</li>
<li><b>Dates:</b> a date a passage or tool result gives is supported. So is a date counted in calendar
days from a date in the question or a tool result, by a period a passage or tool result states, if the
count is right (check it). Business-day counts, and "that day is a holiday / a business day", need a
tool result.</li>
<li><b>Not supported:</b> stated from memory (network rules, state law, statutes), contradicting a passage,
"must" turned into "may", a condition dropped or added, or text whose Status line isn't "in force"
presented as law in force. <b>Naming the law or provision that governs something</b> when no passage
says so ("that's the Fair Credit Reporting Act", "see Subpart B") is not supported either; in a decline
it is a side claim.</li>
<li><b>Judge a claim at the scope it states:</b> "not prohibited by § 1026.10(d)" is supported if that
passage supports it. Something left out is not an unsupported claim; it shows in the key facts.</li>
<li><b>Not claims:</b> citations, "[citation removed]", "this is outside the regulations / the passages",
and suggestions about what to do or review next ("consult § 1026.13(a)") unless they state what a rule
requires.</li>
<li><b>faithful</b>: every claim supported (also when there are no claims).
<b>partially_faithful</b>: something unsupported, but everything that answers the main ask is supported.
<b>unfaithful</b>: something that answers the main ask is unsupported.</li>
<li><b>Key facts:</b> true when the answer states it, in any wording, with every date, number, amount,
party and condition; a part stated, or a different date, is false. Ignore citations in a key fact.
Use the reference answer only to understand what a key fact means.</li>
</ul>
"""


def item_html(cid: str, item: G.GoldenItem, resp: dict, today: date, chunks: dict) -> str:
    e = html.escape
    passages = "".join(
        f'<details open><summary>Passage {n} · {e(chunks[cid_]["id"])} · '
        f'{"official interpretation" if chunks[cid_]["kind"] == "commentary" else "regulation"}</summary>'
        f'<pre>Provisions: {e(", ".join(chunks[cid_]["provision_ids"]))}\n{e(chunks[cid_]["embed_text"])}</pre></details>'
        for n, cid_ in enumerate(resp["retrieved_ids"], 1)) or "<p>(no passages)</p>"
    tools = "".join(f'<pre>{e(json.dumps({"tool": c["name"], "arguments": c["args"], "result": c["result"]}, indent=2, ensure_ascii=False))}</pre>'
                    for c in resp["tool_calls"]) or "<p>(no tool calls)</p>"
    facts = "".join(f"<li>{e(J.strip_citations(f))}</li>" for f in item.key_facts or []) or "<li>(none)</li>"
    declined = '<p class="declined">The assistant declined this question.</p>' if resp["abstained"] else ""
    n = len(item.key_facts or [])
    hint = f'In labels.toml: block {cid}, key_facts with {n} true/false value{"s" if n != 1 else ""}.'
    return (f'<section class="item" id="{cid}"><h2>{cid}</h2><p class="muted">{hint}</p>'
            f'<h3>Question</h3><p>{e(item.question.strip())}</p>'
            f'<p class="muted">Today\'s date: {today.isoformat()} ({today.strftime("%A")})</p>'
            f'<h3>Answer</h3>{declined}<div class="answer">{e(resp["answer"].strip())}</div>'
            f'<h3>Key facts</h3><ol class="facts">{facts}</ol>'
            f'<h3>Reference answer</h3><p>{e((item.reference_answer or "").strip())}</p>'
            f'<h3>Tool results</h3>{tools}'
            f'<h3>Passages the assistant was given</h3>{passages}</section>')


def sheet_html(chosen: list[tuple], ids: list[str], chunks: dict) -> str:
    nav = " ".join(f'<a href="#{c}">{c}</a>' for c in ids)
    body = "".join(item_html(c, item, resp, today, chunks) for c, (_, item, resp, today) in zip(ids, chosen))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Calibration Sheet</title><style>{CSS}</style></head><body><main>'
            f'<h1>Judge calibration: {len(ids)} answers</h1>'
            f'<p class="muted">Label each one in eval/calibration/labels.toml. Check progress any time with '
            f'<code>python -m eval.calibrate --check</code>.</p><div class="rules">{RULES}</div>'
            f'<nav>{nav}</nav>{body}</main></body></html>')


def preview(question: str, words: int = 10) -> str:
    """The question's first words, for a comment in the labels file (no quotes or line breaks)."""
    w = question.replace('"', "'").split()
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


def labels_template(chosen: list[tuple], ids: list[str], sheet: str = "scratch/calibration/sheet.html",
                    check_flag: str = "") -> str:
    head = (f"# Judge calibration labels (rules of rubric {J.JUDGE_VERSION}). Written by Prit Raithatha.\n"
            f"# Read each item in {sheet} and fill in its block here.\n"
            "#   faithfulness  \"faithful\", \"partially_faithful\" or \"unfaithful\"\n"
            "#   key_facts     one true or false per key fact, in the sheet's order, e.g. [true, false]\n"
            "#   note          optional, a few words\n"
            "# Label only from the sheet. Each block starts with the first words of its question: check\n"
            "# they match the sheet's item before you fill it in.\n"
            f"# Check any time: python -m eval.calibrate --check{check_flag}. Labels are final once committed:\n"
            "# don't change them after seeing the judge's verdicts.\n")
    blocks = [f'\n# {c}: "{preview(item.question)}"\n[[label]]\nid = "{c}"\nfaithfulness = ""\n'
              f'key_facts = []        # {len(item.key_facts or [])} key fact{"s" if len(item.key_facts or []) != 1 else ""}\n'
              f'note = ""\n' for c, (_, item, _, _) in zip(ids, chosen)]
    return head + "".join(blocks)


def write_sheet(db, runs: list[str], *, seed=CALIBRATION_SEED, chunks=None, force=False, size=TARGET,
                sheet=SHEET_FILE, labels=LABELS_FILE, key=KEY_FILE, other_keys=None) -> dict:
    """Pick the items and write the sheet, the labels file and the key. Returns the key."""
    if len(set(runs)) != len(runs):
        raise SystemExit("give two different runs")
    for other in (all_keys() if other_keys is None else other_keys):
        if other.resolve() != key.resolve():
            used = set(runs) & set(json.loads(other.read_text(encoding="utf-8")).get("runs", []))
            if used:
                raise SystemExit(f"{', '.join(sorted(used))} already used by calibration set "
                                 f"{other.parent.name if other.parent != CAL_DIR else 'set1'}: use fresh runs")
    if labels.exists() and not force:
        raise SystemExit(f"{labels.relative_to(R.REPO) if labels.is_relative_to(R.REPO) else labels} already "
                         "exists; it may hold your labels. Delete it yourself first if you mean to start again.")
    pools = [candidates(db, r) for r in runs]
    chosen = pick(pools, seed, size)
    if len(chosen) < size:
        print(f"Only {len(chosen)} judgeable answers in these runs; the sheet has all of them.")
    ids = [f"c{n:02d}" for n in range(1, len(chosen) + 1)]
    chunks = chunks or J.chunk_lookup()
    sheet.parent.mkdir(parents=True, exist_ok=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_text(sheet_html(chosen, ids, chunks), encoding="utf-8")
    flag = "" if labels.parent == CAL_DIR else f" --set {labels.parent.name}"
    template = labels_template(chosen, ids, sheet.relative_to(R.REPO).as_posix() if sheet.is_relative_to(R.REPO)
                               else sheet.as_posix(), flag)
    labels.write_text(template, encoding="utf-8", newline="\n")
    names = [run_name(db, r) for r in runs]
    if len(set(names)) < len(names):
        names = [f"{n} #{i}" for i, n in enumerate(names, 1)]
    k = {"seed": seed, "size": size, "runs": runs, "run_names": names, "judge_version": J.JUDGE_VERSION,
         "golden_sha256": hashlib.sha256(G.GOLDEN_FILE.read_bytes()).hexdigest(),
         "items": {c: {"run_id": r, "qid": item.id, "key_facts": len(item.key_facts or []),
                       "should_abstain": item.should_abstain, "persona": item.persona}
                   for c, (r, item, _, _) in zip(ids, chosen)}}
    key.write_text(json.dumps(k, indent=2) + "\n", encoding="utf-8", newline="\n")
    return k


def spread(k: dict) -> str:
    """What the sheet holds, without saying which item is which."""
    items = list(k["items"].values())
    by_run = {r: sum(i["run_id"] == r for i in items) for r in k["runs"]}
    ans = sum(not i["should_abstain"] for i in items)
    fl = sum(i["persona"] == "frontline" for i in items)
    qids = {i["qid"] for i in items}
    return (f"{len(items)} items: " + ", ".join(f"{n} from {r}" for r, n in by_run.items())
            + f"; {ans} answerable, {len(items) - ans} declines; {fl} frontline, {len(items) - fl} compliance; "
              f"{len(qids)} different questions")


# ================================================================ checking the labels

def check(labels=LABELS_FILE, key=KEY_FILE) -> tuple[int, int, list[str]]:
    """(labelled, total, problems). An item is labelled when faithfulness and key_facts are both filled in."""
    if not labels.exists() or not key.exists():
        raise SystemExit("no labels file yet: run python -m eval.calibrate --sheet RUN_DEFAULT RUN_WEAK first")
    k = json.loads(key.read_text(encoding="utf-8"))["items"]
    try:
        data = tomllib.loads(labels.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        return 0, len(k), [f"labels.toml isn't valid TOML: {e}"]
    blocks, problems, done = data.get("label", []), [], 0
    seen = [b.get("id") for b in blocks]
    if sorted(seen, key=str) != sorted(k):
        missing, extra = sorted(set(k) - set(seen)), sorted(set(map(str, seen)) - set(k))
        dupes = sorted({s for s in seen if seen.count(s) > 1}, key=str)
        problems += [f"missing blocks: {', '.join(missing)}"] * bool(missing)
        problems += [f"blocks not on the sheet: {', '.join(extra)}"] * bool(extra)
        problems += [f"blocks listed twice: {', '.join(map(str, dupes))}"] * bool(dupes)
    for b in blocks:
        cid = b.get("id")
        if cid not in k:
            continue
        if extra := sorted(set(b) - {"id", "faithfulness", "key_facts", "note"}):
            problems.append(f"{cid}: unexpected field(s) {', '.join(extra)}")
        f, facts = b.get("faithfulness", ""), b.get("key_facts", [])
        if f == "" and facts == []:
            continue                                         # not labelled yet
        if f not in LABELS:
            problems.append(f"{cid}: faithfulness must be one of {', '.join(LABELS)}, not {f!r}")
        if not isinstance(facts, list) or not all(isinstance(x, bool) for x in facts):
            problems.append(f"{cid}: key_facts must be a list of true/false")
        elif len(facts) != k[cid]["key_facts"]:
            problems.append(f"{cid}: key_facts has {len(facts)} value(s); this item has {k[cid]['key_facts']} key fact(s)")
        if not isinstance(b.get("note", ""), str):
            problems.append(f"{cid}: note must be text in quotes")
        if f in LABELS and isinstance(facts, list) and len(facts) == k[cid]["key_facts"] \
                and all(isinstance(x, bool) for x in facts):
            done += 1
    return done, len(k), problems


# ================================================================ keeping the labels blind

def git_dirty(path: Path) -> bool | None:
    """True if git sees uncommitted changes (or untracked files) under path; None if git can't tell."""
    import subprocess
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", str(path)], cwd=R.REPO,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.strip())


def blind_block(run_id: str, labels=None, key=None, dirty=git_dirty, cal_dir=None) -> str | None:
    """Why the judge must not judge this run yet, or None. A run on any calibration set's sheet is
    judged only once that set's labels are all filled in, well-formed and committed, so Prit never sees
    a verdict first. With labels and key given, only that set is checked."""
    if labels is None or key is None:
        for k in all_keys(cal_dir):
            if (why := blind_block(run_id, k.parent / "labels.toml", k, dirty)):
                return why
        return None
    if not key.exists():
        return None
    if run_id not in json.loads(key.read_text(encoding="utf-8")).get("runs", []):
        return None
    done, total, problems = check(labels, key)
    if problems or done < total:
        return (f"{run_id} is on the calibration sheet and only {done} of {total} labels are done"
                + (f" ({len(problems)} problem(s); see python -m eval.calibrate --check)" if problems else "")
                + ". Judging it now would show you verdicts before your labels are final.")
    if any(dirty(f) for f in (labels, key)):        # the two files only: a saved compare-*.json is fine
        return (f"{run_id} is on a calibration sheet: commit {labels.parent.relative_to(R.REPO).as_posix() if labels.is_relative_to(R.REPO) else labels.parent}/ "
                "first, so the labels are fixed before any verdict exists.")
    return None


# ================================================================ part 2: the judge against Prit's labels

ORDER = {"unfaithful": 0, "partially_faithful": 1, "faithful": 2}   # for "more lenient" / "stricter"
RUN_NAMES = ("default", "weak")              # set 1's runs (its key.json has no run_names)
BOOTSTRAP_REPS = 2000
Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score 95% interval for k successes in n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def kappa(pairs: list[tuple]) -> float | None:
    """Cohen's kappa for (Prit, judge) pairs: agreement corrected for the agreement chance alone would
    give, from each rater's own label counts. None when it's undefined (both used one same label)."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    cats = {a for a, _ in pairs} | {b for _, b in pairs}
    chance = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    if chance >= 1:
        return None
    return (observed - chance) / (1 - chance)


def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default)."""
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap(items: list[dict], stat, *, reps: int = BOOTSTRAP_REPS, seed: int = CALIBRATION_SEED) -> dict:
    """Percentile 95% interval for stat(items), resampling whole questions (clusters), not single
    answers: the 40 answers cover 23 questions, and two answers to one question share its key facts,
    so they aren't independent. Resamples where stat is undefined are left out and counted."""
    by_q: dict[str, list[dict]] = {}
    for it in items:
        by_q.setdefault(it["qid"], []).append(it)
    clusters = [by_q[q] for q in sorted(by_q)]
    rng = random.Random(seed)
    vals, undefined = [], 0
    for _ in range(reps):
        sample = [it for _ in clusters for it in clusters[rng.randrange(len(clusters))]]
        v = stat(sample)
        if v is None:
            undefined += 1
        else:
            vals.append(v)
    vals.sort()
    if not vals:
        return {"lo": None, "hi": None, "undefined": undefined, "reps": reps}
    return {"lo": percentile(vals, 0.025), "hi": percentile(vals, 0.975), "undefined": undefined, "reps": reps}


def faith_pairs(items, binary=False):
    if binary:
        return [(it["prit"] == "faithful", it["judge"] == "faithful") for it in items]
    return [(it["prit"], it["judge"]) for it in items]


def fact_pairs(items):
    return [(p, j) for it in items for p, j in zip(it["prit_kf"], it["judge_kf"])]


def paired_items(db, *, judge_version: str = J.JUDGE_VERSION, labels=LABELS_FILE, key=KEY_FILE):
    """(items, not_compared). Each item joins one label block to the judge's verdict on the same answer.
    not_compared lists (cid, reason) for answers without a usable verdict."""
    k = json.loads(key.read_text(encoding="utf-8"))
    names = k.get("run_names") or RUN_NAMES
    blocks = {b["id"]: b for b in tomllib.loads(labels.read_text(encoding="utf-8"))["label"]}
    items, missing = [], []
    for cid, info in k["items"].items():
        row = db.execute("SELECT status, skip_reason, error, verdict_json, judge_model FROM judgements "
                         "WHERE run_id = ? AND qid = ? AND judge_version = ?",
                         (info["run_id"], info["qid"], judge_version)).fetchone()
        if row is None:
            missing.append((cid, f"no {judge_version} verdict yet"))
            continue
        status, skip, error, vj, model = row
        if status != "judged":
            missing.append((cid, f"{status}" + (f" ({skip})" if skip else "") + (f": {error}" if error else "")))
            continue
        v = J.JudgeVerdict.model_validate_json(vj)
        b = blocks[cid]
        if len(v.key_facts) != len(b["key_facts"]):
            missing.append((cid, f"the verdict has {len(v.key_facts)} key facts, the label {len(b['key_facts'])}"))
            continue
        items.append({"cid": cid, "qid": info["qid"], "run_id": info["run_id"],
                      "run": names[k["runs"].index(info["run_id"])],
                      "decline": bool(info["should_abstain"]), "persona": info["persona"],
                      "prit": b["faithfulness"], "judge": v.faithfulness,
                      "prit_kf": list(b["key_facts"]), "judge_kf": [x.covered for x in v.key_facts],
                      "note": b.get("note", ""), "rationale": v.rationale, "judge_model": model})
    return items, missing


def _slice(items) -> dict:
    fp, kp = faith_pairs(items), fact_pairs(items)
    return {"n": len(items), "agree": sum(a == b for a, b in fp),
            "kappa": kappa(fp), "kappa_binary": kappa(faith_pairs(items, binary=True)),
            "facts": len(kp), "facts_agree": sum(a == b for a, b in kp), "facts_kappa": kappa(kp)}


def compare(items: list[dict], *, reps: int = BOOTSTRAP_REPS, seed: int = CALIBRATION_SEED) -> dict:
    """Every number --compare prints, as a dict (saved with --save)."""
    n = len(items)
    fp, bp, kp = faith_pairs(items), faith_pairs(items, binary=True), fact_pairs(items)
    agree, b_agree, k_agree = (sum(a == b for a, b in p) for p in (fp, bp, kp))
    bs = lambda stat: bootstrap(items, stat, reps=reps, seed=seed)          # noqa: E731
    lenient = [it["cid"] for it in items if ORDER[it["judge"]] > ORDER[it["prit"]]]
    strict = [it["cid"] for it in items if ORDER[it["judge"]] < ORDER[it["prit"]]]
    slices = {}
    runs = list(dict.fromkeys(it["run"] for it in items))
    for name, test in [*[(f"run: {r}", lambda it, r=r: it["run"] == r) for r in runs],
                       ("answerable", lambda it: not it["decline"]), ("declines", lambda it: it["decline"]),
                       ("frontline", lambda it: it["persona"] == "frontline"),
                       ("compliance", lambda it: it["persona"] == "compliance")]:
        slices[name] = _slice([it for it in items if test(it)])
    disagreements = []
    for it in items:
        facts = [i for i, (p, j) in enumerate(zip(it["prit_kf"], it["judge_kf"]), 1) if p != j]
        if it["prit"] != it["judge"] or facts:
            disagreements.append({"cid": it["cid"], "qid": it["qid"], "run": it["run"], "run_id": it["run_id"],
                                  "prit": it["prit"], "judge": it["judge"],
                                  "facts": [{"n": i, "prit": it["prit_kf"][i - 1], "judge": it["judge_kf"][i - 1]}
                                            for i in facts],
                                  "note": it["note"], "rationale": it["rationale"]})
    return {
        "n": n, "questions": len({it["qid"] for it in items}),
        "faithfulness": {
            "agree": agree, "agreement": agree / n if n else None, "wilson95": wilson(agree, n),
            "kappa": kappa(fp), "kappa95": bs(lambda s: kappa(faith_pairs(s))),
            "confusion": {p: {j: sum(it["prit"] == p and it["judge"] == j for it in items) for j in LABELS}
                          for p in LABELS},
            "judge_more_lenient": lenient, "judge_stricter": strict,
            "prit_faithful": sum(it["prit"] == "faithful" for it in items),
            "judge_faithful": sum(it["judge"] == "faithful" for it in items)},
        "faithful_or_not": {
            "agree": b_agree, "agreement": b_agree / n if n else None, "wilson95": wilson(b_agree, n),
            "kappa": kappa(bp), "kappa95": bs(lambda s: kappa(faith_pairs(s, binary=True)))},
        "key_facts": {
            "n": len(kp), "agree": k_agree, "agreement": k_agree / len(kp) if kp else None,
            "agreement95": bs(lambda s: (sum(a == b for a, b in fact_pairs(s)) / len(fact_pairs(s)))
                              if fact_pairs(s) else None),
            "kappa": kappa(kp), "kappa95": bs(lambda s: kappa(fact_pairs(s))),
            "confusion": {"prit_true": {"judge_true": sum(p and j for p, j in kp), "judge_false": sum(p and not j for p, j in kp)},
                          "prit_false": {"judge_true": sum(j and not p for p, j in kp), "judge_false": sum(not p and not j for p, j in kp)}},
            "prit_covered": sum(p for p, _ in kp), "judge_covered": sum(j for _, j in kp)},
        "slices": slices, "disagreements": disagreements,
        "bootstrap": {"reps": reps, "seed": seed, "resampled": "questions"},
    }


def _pct(x):
    return "n/a" if x is None or x != x else f"{100 * x:.1f}%"


def _num(x):
    return "undefined" if x is None else f"{x:.2f}"


def _ci(b, pct=False):
    if b["lo"] is None:
        return "interval undefined"
    f = _pct if pct else _num
    extra = f"; {b['undefined']} of {b['reps']} resamples undefined" if b["undefined"] else ""
    return f"95% {f(b['lo'])} to {f(b['hi'])}{extra}"


def compare_report(c: dict, *, judge_version: str, missing: list[tuple]) -> str:
    f, b, kf = c["faithfulness"], c["faithful_or_not"], c["key_facts"]
    short = {"faithful": "faithful", "partially_faithful": "partial", "unfaithful": "unfaithful"}
    lines = [f"Judge {judge_version} against Prit's labels: {c['n']} answers to {c['questions']} questions"]
    if missing:
        lines.append(f"  Not compared ({len(missing)}): " + "; ".join(f"{cid} {why}" for cid, why in missing))
    lines += [
        "",
        "Faithfulness (3 labels)",
        f"  agreement   {f['agree']}/{c['n']} = {_pct(f['agreement'])} "
        f"(Wilson 95% {_pct(f['wilson95'][0])} to {_pct(f['wilson95'][1])})",
        f"  kappa       {_num(f['kappa'])} ({_ci(f['kappa95'])})",
        f"  faithful    Prit {f['prit_faithful']}/{c['n']}, judge {f['judge_faithful']}/{c['n']}",
        f"  judge more lenient on {len(f['judge_more_lenient'])}, stricter on {len(f['judge_stricter'])}",
        "",
        "  Confusion (rows Prit, columns judge)",
        "  " + " " * 12 + "".join(f"{short[j]:>12}" for j in LABELS),
    ]
    lines += ["  " + f"{short[p]:<12}" + "".join(f"{f['confusion'][p][j]:>12}" for j in LABELS) for p in LABELS]
    lines += [
        "",
        "Faithful or not (2 labels: partial and unfaithful together)",
        f"  agreement   {b['agree']}/{c['n']} = {_pct(b['agreement'])} "
        f"(Wilson 95% {_pct(b['wilson95'][0])} to {_pct(b['wilson95'][1])})",
        f"  kappa       {_num(b['kappa'])} ({_ci(b['kappa95'])})",
        "",
        f"Key facts ({kf['n']} facts in {c['n']} answers)",
        f"  agreement   {kf['agree']}/{kf['n']} = {_pct(kf['agreement'])} ({_ci(kf['agreement95'], pct=True)})",
        f"  kappa       {_num(kf['kappa'])} ({_ci(kf['kappa95'])})",
        f"  covered     Prit {kf['prit_covered']}/{kf['n']}, judge {kf['judge_covered']}/{kf['n']}",
        f"  Prit true, judge false: {kf['confusion']['prit_true']['judge_false']}; "
        f"Prit false, judge true: {kf['confusion']['prit_false']['judge_true']}",
        "",
        "By slice (small samples: read these as pointers, not results)",
        f"  {'':16}{'n':>4}{'agree':>9}{'kappa':>8}{'f/not k':>9}{'facts agree':>14}{'kappa':>8}",
        "  (f/not k: kappa on faithful or not)",
    ]
    for name, s in c["slices"].items():
        lines.append(f"  {name[:16]:16}{s['n']:>4}{s['agree']:>5}/{s['n']:<3}{_num(s['kappa']):>8}"
                     f"{_num(s['kappa_binary']):>9}{s['facts_agree']:>9}/{s['facts']:<4}{_num(s['facts_kappa']):>8}")
    lines += ["", f"Disagreements ({len(c['disagreements'])}): see one in full with "
                  "python -m eval.judge RUN_ID --show QID"]
    for d in c["disagreements"]:
        what = []
        if d["prit"] != d["judge"]:
            what.append(f"faithfulness Prit {short[d['prit']]}, judge {short[d['judge']]}")
        what += [f"key fact {x['n']} Prit {str(x['prit']).lower()}, judge {str(x['judge']).lower()}" for x in d["facts"]]
        lines.append(f"  {d['cid']} {d['qid']} {d['run']:<7} {'; '.join(what)}")
        lines.append(f"      Prit:  {d['note']}" if d["note"] else "      Prit:  (no note)")
        lines.append(f"      judge: {d['rationale']}")
        lines.append(f"      python -m eval.judge {d['run_id']} --show {d['qid']}")
    lines += ["", f"Intervals: Wilson for agreement on answers; the rest by bootstrap ({c['bootstrap']['reps']} "
                  "resamples of whole questions, seeded), since answers to the same question aren't independent."]
    return "\n".join(lines)


def compare_refusal(labels=LABELS_FILE, key=KEY_FILE, dirty=git_dirty) -> str | None:
    """Why --compare must not run yet, or None: the labels must be complete and committed, so every
    comparison is against the labels as they were before any verdict existed."""
    if not labels.exists() or not key.exists():
        return "no calibration labels yet (python -m eval.calibrate --sheet first)"
    done, total, problems = check(labels, key)
    if problems or done < total:
        return f"only {done} of {total} labels are done; see python -m eval.calibrate --check"
    if any(dirty(f) for f in (labels, key)):
        return ("labels.toml or key.json has uncommitted changes. Labels don't change after verdicts exist; "
                "restore the committed ones (git restore eval/calibration/labels.toml eval/calibration/key.json)")
    return None


def compare_file(judge_version: str, labels: Path = LABELS_FILE) -> Path:
    return labels.parent / f"compare-{judge_version}.json"


# ================================================================ command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.calibrate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--sheet", nargs=2, metavar=("RUN_A", "RUN_B"),
                    help="pick the items from these two runs and write the sheet and labels file")
    ap.add_argument("--set", default="set1", help="which calibration set: set1 (session 19), set2, ... (default set1)")
    ap.add_argument("--size", type=int, default=TARGET, help=f"with --sheet: how many answers (default {TARGET})")
    ap.add_argument("--check", action="store_true", help="check labels.toml")
    ap.add_argument("--compare", action="store_true", help="the judge's verdicts against the labels")
    ap.add_argument("--judge-version", default=J.JUDGE_VERSION, help=f"with --compare: whose verdicts (default {J.JUDGE_VERSION})")
    ap.add_argument("--save", action="store_true", help="with --compare: also write eval/calibration/compare-VERSION.json")
    ap.add_argument("--db", type=Path, default=R.DB_FILE, help="the SQLite file (default runs.db)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    P = set_paths(args.set)
    rel = lambda path: path.relative_to(R.REPO).as_posix() if path.is_relative_to(R.REPO) else str(path)  # noqa: E731
    flag = "" if args.set == "set1" else f" --set {args.set}"
    if args.sheet:
        db = J.connect(args.db)
        judged = db.execute("SELECT COUNT(*) FROM judgements WHERE run_id IN (?, ?)", args.sheet).fetchone()[0]
        if judged:
            print(f"Refused: {judged} judge verdict(s) already exist for these runs, so labelling them "
                  "wouldn't be blind. Use two fresh runs.")
            return 2
        k = write_sheet(db, list(args.sheet), size=args.size, **P)
        print(spread(k))
        print(f"\nRead:   {rel(P['sheet'])}\nLabel:  {rel(P['labels'])}\n"
              f"Check:  python -m eval.calibrate --check{flag}\n(Don't open {rel(P['key'])} while labelling.)")
        return 0
    if args.check:
        done, total, problems = check(P["labels"], P["key"])
        print(f"{done} of {total} labelled.")
        for p in problems:
            print(f"  {p}")
        if not problems and done == total:
            print(f"All labelled and well-formed: commit {rel(P['labels'].parent)}/ before running the judge on these runs.")
        return 1 if problems else 0
    if args.compare:
        if (why := compare_refusal(P["labels"], P["key"])):
            print(f"Not compared: {why}.")
            return 2
        key = json.loads(P["key"].read_text(encoding="utf-8"))
        if key.get("golden_sha256") != hashlib.sha256(G.GOLDEN_FILE.read_bytes()).hexdigest():
            print("Note: eval/golden.toml has changed since the sheet was made; key facts are compared by "
                  "position, so check any item whose facts changed.\n")
        items, missing = paired_items(J.connect(args.db), judge_version=args.judge_version,
                                      labels=P["labels"], key=P["key"])
        if not items:
            print(f"Not compared: no {args.judge_version} verdicts on the calibration answers yet. Judge both runs "
                  "first: " + " and ".join(f"python -m eval.judge {r} --rubric {args.judge_version}" for r in key["runs"]))
            return 2
        c = compare(items)
        print(f"Calibration {args.set}\n" + compare_report(c, judge_version=args.judge_version, missing=missing))
        if args.save:
            out = {"set": args.set, "judge_version": args.judge_version,
                   "judge_models": sorted({it["judge_model"] for it in items}),
                   "labels_sha256": hashlib.sha256(P["labels"].read_bytes()).hexdigest(),
                   "runs": dict(zip(key.get("run_names") or RUN_NAMES, key["runs"])), "not_compared": missing, **c}
            path = compare_file(args.judge_version, P["labels"])
            path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
            print(f"\nSaved {rel(path)}")
        return 1 if missing else 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
