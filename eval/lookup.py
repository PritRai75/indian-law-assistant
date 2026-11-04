"""Lookup helper for labelling the golden set (tracker t3-lookup).

Prints the corpus text behind a provision ID, so gold_ids, reference answers and key facts can be
written from the exact text the assistant answers from, with each provision's status.

    python -m eval.lookup "1005.6(b)"                 the paragraph, the paragraphs under it, and the comments on any of them,
                                                      then, as one line each, the comments filed on paragraphs above it
    python -m eval.lookup 1005.6                      a whole section
    python -m eval.lookup "comment 13(c)(2)-1" "Reg E 11(c)"   several at once, in any form a question can use
    python -m eval.lookup --outline 1026.13           one line per provision: ID, status, opening words
    python -m eval.lookup --find "provisional credit" [--reg E]   plain text match across the corpus
    python -m eval.lookup --sections [--reg Z]        the table of contents: every indexed section and its title

References are read the way the assistant's search reads them (app.retrieve.read_refs), so a short
form without the part ("13(c)(2)") shows both parts when both exist. --find is a plain substring
match in document order, deliberately not the assistant's search: labels must not depend on what
that search happens to rank highly. Long output reads more easily saved to a file:
    python -m eval.lookup 1005.6 > lookup.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROVISIONS_FILE = REPO / "corpus" / "txt" / "provisions.jsonl"
CHUNKS_FILE = REPO / "index" / "chunks.json"
REG_NAME = {"E": "Regulation E", "Z": "Regulation Z"}
WIDTH = 100


@lru_cache(maxsize=1)
def provisions() -> dict[str, dict]:
    """Every provision in the index, in document order (the order of index/chunks.json).

    Text and fields come from corpus/txt/provisions.jsonl. A provision that exists only in the index
    (1026.2(a)(15)(ii)(C), restored by chunk.py because the eCFR doesn't print it) is taken from its chunk."""
    with open(PROVISIONS_FILE, encoding="utf-8") as f:
        parsed = {r["id"]: r for r in map(json.loads, f)}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    out = {}
    for c in chunks:
        for pid in c["provision_ids"]:
            if pid in parsed:
                out[pid] = parsed[pid]
            else:
                out[pid] = {"id": pid, "kind": c["kind"], "reg": c["reg"], "section": c["section"],
                            "section_heading": c["section_heading"], "parent": None, "text": c["text"],
                            "status": c["status"], "status_note": c["status_note"]}
    return out


def label(p: dict) -> str:
    return f"comment {p['id']}" if p["kind"] == "commentary" else f"§ {p['id']}"


def status_text(p: dict) -> str:
    from ingest.chunk import STATUS_LINE
    if p["status"] == "in_force":
        return "in force" + (f". Note: {p['status_note']}" if p.get("status_note") else "")
    return f"{STATUS_LINE[p['status']]}. {p['status_note'] or ''}".strip()


def wrap(text: str, indent: str = "    ") -> str:
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, WIDTH, initial_indent=indent, subsequent_indent=indent) or [""]
    return "\n".join(lines)


def opening(text: str, n: int) -> str:
    one = " ".join(text.split())
    return one if len(one) <= n else one[:n - 1].rstrip() + "…"


def context(p: dict, all_: dict[str, dict]) -> list[str]:
    """For a deep paragraph, the opening words of each paragraph above it; for a comment, of the rule it explains."""
    out, seen = [], set()
    up = p.get("target") if p["kind"] == "commentary" else p.get("parent")
    while up and up in all_ and up not in seen:
        seen.add(up)
        q = all_[up]
        out.append(f"  under {label(q)}: {opening(q['text'], 160)}")
        up = q.get("parent")
    return list(reversed(out))


def covered(ref: str) -> list[dict]:
    """The provisions a reference covers, as the search's named set: the provision, paragraphs under it,
    and comments on either (a comment reference covers only itself). Rules first, then comments."""
    from app.retrieve import covered_by
    all_ = provisions()
    hits = [p for pid, p in all_.items() if covered_by(ref.lower(), pid.lower())]
    return [p for p in hits if p["kind"] == "rule"] + [p for p in hits if p["kind"] == "commentary"]


def resolve(text: str) -> tuple[list[str], list[str]]:
    """References written in `text`, checked against the index: (found IDs, references matching nothing).
    A bare full ID ("1005.6", "1026.13(c)(2)", "1005.6(b)-1") is also accepted."""
    from app.retrieve import read_refs, resolve as resolve_refs
    ids = list(provisions())
    refs = read_refs(text)
    if not refs and re.fullmatch(r"\s*\d{4}\.\d+\S*\s*", text):
        refs = read_refs(f"§ {text.strip()}")
    found, not_found = resolve_refs(refs, ids)
    if not refs:
        not_found = [text]
    return found, not_found


def above(ref: str) -> list[str]:
    """The paragraphs above a rule ID, nearest first, ending with its section:
    "1005.12(a)(1)(iv)(A)" -> ["1005.12(a)(1)(iv)", "1005.12(a)(1)", "1005.12(a)", "1005.12"]."""
    if "-" in ref.split(".", 1)[1]:
        return []                                        # a comment has no paragraphs above it
    out, cur = [], ref
    while (m := re.fullmatch(r"(.+)\([^()]+\)", cur)):
        cur = m.group(1)
        out.append(cur)
    return out


def comments_above(ref: str) -> list[tuple[str, list[dict]]]:
    """Comments filed on a paragraph above `ref` (or on its section), which the gold-ID family of `ref`
    doesn't cover: [(that paragraph, its comments)], nearest paragraph first (session 15)."""
    by_target: dict[str, list[dict]] = {}
    for pid, p in provisions().items():
        if p["kind"] == "commentary":
            by_target.setdefault(pid.rsplit("-", 1)[0].lower(), []).append(p)
    return [(a, by_target[a.lower()]) for a in above(ref) if a.lower() in by_target]


def show_above(ref: str) -> list[str]:
    groups = comments_above(ref)
    if not groups:
        return []
    n = sum(len(cs) for _, cs in groups)
    lines = ["", f"-- {n} comment(s) filed on paragraphs above {ref}. Its gold-ID family doesn't cover them: "
                 "if one is what answers, list it in gold_ids."]
    for target, cs in groups:
        lines.append(f"  on § {target}:")
        for p in cs:
            flag = "" if p["status"] == "in_force" else f"[{p['status'].upper()}] "
            lines.append(f"    {label(p):<30}{flag}{opening(p['text'], 70)}")
    return lines


def show(ref: str, outline: bool = False) -> str:
    all_ = provisions()
    items = covered(ref)
    if not items:
        return f"== {ref}: nothing in the index"
    first = items[0]
    rules = sum(p["kind"] == "rule" for p in items)
    head = (f"== {ref}  ({REG_NAME[first['reg']]}, § {first['section']} {first['section_heading'].rstrip('.')}): "
            f"{rules} rule paragraph(s), {len(items) - rules} comment(s)")
    lines = [head]
    if not outline:
        lines += context(all_[ref], all_) if ref in all_ else []
    for p in items:
        flag = "" if p["status"] == "in_force" else f"  [{p['status'].upper()}]"
        if outline:
            lines.append(f"  {label(p):<28}{flag.strip() + ' ' if flag else ''}{opening(p['text'], 70)}")
        else:
            lines += ["", f"{label(p)}  ({status_text(p)})", wrap(p["text"])]
    lines += show_above(ref)
    return "\n".join(lines)


def find(phrase: str, reg: str | None = None, limit: int = 40) -> str:
    """Every provision whose text contains `phrase` (case and line breaks ignored), in document order."""
    want = " ".join(phrase.lower().split())
    hits = []
    for p in provisions().values():
        if reg and p["reg"] != reg:
            continue
        flat = " ".join(p["text"].split())
        at = flat.lower().find(want)
        if at >= 0:
            start = max(0, at - 60)
            snippet = ("…" if start else "") + flat[start:at + len(want) + 60] + ("…" if at + len(want) + 60 < len(flat) else "")
            flag = "" if p["status"] == "in_force" else f" [{p['status'].upper()}]"
            hits.append(f"  {label(p)}{flag}\n      {snippet}")
    shown = hits[:limit]
    more = f"\n  … and {len(hits) - limit} more; narrow the phrase or add --reg" if len(hits) > limit else ""
    return f"{len(hits)} provision(s) contain {phrase!r}" + (f" in {REG_NAME[reg]}" if reg else "") + \
           (":\n" + "\n".join(shown) + more if hits else ".")


# Sections the index holds (spec: Corpus). Everything else in Parts 1005 and 1026 is left out on purpose.
INDEXED = {"1005": [(1, 20)], "1026": [(1, 16), (51, 62)]}


def outside_scope(guess: str) -> bool:
    m = re.match(r"(1005|1026)\.(\d+)", guess)
    return bool(m) and not any(a <= int(m.group(2)) <= b for a, b in INDEXED[m.group(1)])


def suggest(written: str) -> str:
    from eval.golden import check_id
    guess = re.sub(r"^(§|comment|section)\s*", "", written.strip(), flags=re.I)
    if outside_scope(guess):
        return (f"== {written}: not in the index: that section is outside the indexed subparts "
                "(Reg E §§ 1005.1-1005.20; Reg Z §§ 1026.1-1026.16 and 1026.51-1026.62)")
    hint = check_id(guess, {pid: "" for pid in provisions()}) if re.match(r"\d{4}\.", guess) else None
    return f"== {written}: not found in the index" + (f" ({hint})" if hint else ". Check the spelling.")


def sections(reg: str | None = None) -> str:
    """The table of contents: every indexed section with its title, in order, for one or both regulations."""
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    seen = {}
    for c in chunks:
        if reg in (None, c["reg"]) and c["section"] not in seen:
            seen[c["section"]] = (c["reg"], c["section_heading"])
    order = sorted(seen, key=lambda s: (s.split(".")[0], int(s.split(".")[1])))
    lines, last = [], None
    for s in order:
        r, heading = seen[s]
        if r != last:
            lines.append(("" if last is None else "\n") + f"{REG_NAME[r]} (12 CFR {s.split('.')[0]}), the sections indexed:")
            last = r
        lines.append(f"  § {s}  {heading}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.lookup", description=__doc__.split("\n\n")[0])
    ap.add_argument("refs", nargs="*", help="provision references, e.g. 1005.6(b) or \"comment 13(c)(2)-1\"")
    ap.add_argument("--outline", action="store_true", help="one line per provision instead of the full text")
    ap.add_argument("--find", metavar="PHRASE", help="list provisions whose text contains PHRASE")
    ap.add_argument("--reg", choices=["E", "Z"], help="with --find or --sections: one regulation only")
    ap.add_argument("--sections", action="store_true", help="the table of contents: every indexed section and its title")
    args = ap.parse_args(argv)
    if not args.refs and not args.find and not args.sections:
        ap.print_help()
        return 2
    out, missing = [], False
    if args.sections:
        out.append(sections(args.reg))
    if args.find:
        out.append(find(args.find, args.reg))
    for written in args.refs:
        found, not_found = resolve(written)
        out += [show(pid, args.outline) for pid in found]
        out += [suggest(w) for w in not_found]
        missing |= bool(not_found)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to another encoding
    print("\n\n".join(out))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
