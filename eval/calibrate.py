"""Judge calibration, part 1: the blind labelling sheet (spec: judge calibration; tracker t6-calib).
Prit labels about 40 logged answers himself, before any judge verdict on them exists, so agreement
measures the judge against an independent reading.

From the repo root (no AWS call):
    python -m eval.calibrate --sheet RUN_DEFAULT RUN_WEAK   pick the 40 answers, write the sheet and the labels file
    python -m eval.calibrate --check                        check labels.toml: what's filled in, what's wrong

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
TARGET = 40
ANSWERABLE_SHARE = 16 / 24                  # the dev split: 16 answerable, 8 declines
LABELS = ("faithful", "partially_faithful", "unfaithful")

SHEET_FILE = R.REPO / "scratch" / "calibration" / "sheet.html"
LABELS_FILE = R.REPO / "eval" / "calibration" / "labels.toml"
KEY_FILE = R.REPO / "eval" / "calibration" / "key.json"


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
A date or amount needs a passage or tool result that gives it.</li>
<li><b>Not supported:</b> stated from memory (network rules, state law, statutes), contradicting a passage,
"must" turned into "may", a condition dropped or added, or text whose Status line isn't "in force"
presented as law in force.</li>
<li><b>Not claims:</b> citations, "[citation removed]", and "this is outside the regulations / the passages".</li>
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


def labels_template(chosen: list[tuple], ids: list[str]) -> str:
    head = ("# Judge calibration labels (rubric j1). Written by Prit Raithatha.\n"
            "# Read each item in scratch/calibration/sheet.html and fill in its block here.\n"
            "#   faithfulness  \"faithful\", \"partially_faithful\" or \"unfaithful\"\n"
            "#   key_facts     one true or false per key fact, in the sheet's order, e.g. [true, false]\n"
            "#   note          optional, a few words\n"
            "# Check any time: python -m eval.calibrate --check. Labels are final once committed:\n"
            "# don't change them after seeing the judge's verdicts.\n")
    blocks = [f'\n[[label]]\nid = "{c}"\nfaithfulness = ""\n'
              f'key_facts = []        # {len(item.key_facts or [])} key fact{"s" if len(item.key_facts or []) != 1 else ""}\n'
              f'note = ""\n' for c, (_, item, _, _) in zip(ids, chosen)]
    return head + "".join(blocks)


def write_sheet(db, runs: list[str], *, seed=CALIBRATION_SEED, chunks=None, force=False,
                sheet=SHEET_FILE, labels=LABELS_FILE, key=KEY_FILE) -> dict:
    """Pick the items and write the sheet, the labels file and the key. Returns the key."""
    if len(set(runs)) != len(runs):
        raise SystemExit("give two different runs")
    if labels.exists() and not force:
        raise SystemExit(f"{labels.relative_to(R.REPO) if labels.is_relative_to(R.REPO) else labels} already "
                         "exists; it may hold your labels. Delete it yourself first if you mean to start again.")
    pools = [candidates(db, r) for r in runs]
    chosen = pick(pools, seed)
    if len(chosen) < TARGET:
        print(f"Only {len(chosen)} judgeable answers in these runs; the sheet has all of them.")
    ids = [f"c{n:02d}" for n in range(1, len(chosen) + 1)]
    chunks = chunks or J.chunk_lookup()
    sheet.parent.mkdir(parents=True, exist_ok=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_text(sheet_html(chosen, ids, chunks), encoding="utf-8")
    template = labels_template(chosen, ids)
    labels.write_text(template, encoding="utf-8", newline="\n")
    k = {"seed": seed, "runs": runs, "judge_version": J.JUDGE_VERSION,
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


def blind_block(run_id: str, labels=LABELS_FILE, key=KEY_FILE, dirty=git_dirty) -> str | None:
    """Why the judge must not judge this run yet, or None. A run on the calibration sheet is judged only
    once every label is filled in, well-formed and committed, so Prit never sees a verdict first."""
    if not key.exists():
        return None
    if run_id not in json.loads(key.read_text(encoding="utf-8")).get("runs", []):
        return None
    done, total, problems = check(labels, key)
    if problems or done < total:
        return (f"{run_id} is on the calibration sheet and only {done} of {total} labels are done"
                + (f" ({len(problems)} problem(s); see python -m eval.calibrate --check)" if problems else "")
                + ". Judging it now would show you verdicts before your labels are final.")
    if dirty(labels.parent):
        return (f"{run_id} is on the calibration sheet: commit eval/calibration/ first, so the labels are "
                "fixed before any verdict exists.")
    return None


# ================================================================ command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.calibrate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--sheet", nargs=2, metavar=("RUN_DEFAULT", "RUN_WEAK"),
                    help="pick the items from these two runs and write the sheet and labels file")
    ap.add_argument("--check", action="store_true", help="check labels.toml")
    ap.add_argument("--db", type=Path, default=R.DB_FILE, help="the SQLite file (default runs.db)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.sheet:
        db = J.connect(args.db)
        judged = db.execute("SELECT COUNT(*) FROM judgements WHERE run_id IN (?, ?)", args.sheet).fetchone()[0]
        if judged:
            print(f"Refused: {judged} judge verdict(s) already exist for these runs, so labelling them "
                  "wouldn't be blind. Use two fresh runs.")
            return 2
        k = write_sheet(db, list(args.sheet))
        print(spread(k))
        print(f"\nRead:   {SHEET_FILE.relative_to(R.REPO)}\nLabel:  {LABELS_FILE.relative_to(R.REPO)}\n"
              f"Check:  python -m eval.calibrate --check\n(Don't open {KEY_FILE.relative_to(R.REPO)} while labelling.)")
        return 0
    if args.check:
        done, total, problems = check()
        print(f"{done} of {total} labelled.")
        for p in problems:
            print(f"  {p}")
        if not problems and done == total:
            print("All labelled and well-formed: commit eval/calibration/ before running the judge on these runs.")
        return 1 if problems else 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
