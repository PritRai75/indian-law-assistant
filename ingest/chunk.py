"""Group provisions into the chunks that are embedded and searched.

Run from the repo root after parse.py:

    python -m ingest.chunk            # writes corpus/txt/chunks.jsonl and prints a summary

A *chunk* is what search returns and what Claude reads. Rules (decided in
session 10):

1. **Size limit: 4,000 characters** of provision text (about 1,000 tokens).
2. **Rule text.** A section that fits is one chunk (ID `1005.3`). A longer
   section is split at its first-level paragraphs (`1026.13(c)`); a
   first-level paragraph that is still too long is split at its second-level
   paragraphs (`1005.18(b)(2)`). Text above the first paragraph (a section's
   intro, or a paragraph's own heading line) goes with the first piece.
3. **Official interpretations.** Grouped by the paragraph they interpret
   (`1026.12(b)-cmt` holds comments 12(b)-1, 12(b)-2 ...); section-level
   comments such as 12-1 form `1026.12-cmt`.
4. **Never mix statuses.** Inside a group, a change of `status` (or of
   `status_note`) starts a new chunk, so a vacated or nullified provision
   never sits inside an in-force chunk.
5. **Still too long:** consecutive provisions are packed into pieces of at most
   the limit. Pieces of one unit get `#1`, `#2` ... (`1005.18(b)(2)#1`).
   A single provision longer than the limit stays whole: it is one citable
   unit, and Titan accepts it (the longest is about 15,500 characters).
6. **Missing in-force text.** `corpus/status.json` lists provisions that are
   law but that the eCFR no longer prints (`missing_in_force`). Each becomes
   its own chunk, placed right after its preceding sibling, with a header
   saying why it is not in the eCFR.

Every chunk's `embed_text` is a header plus the provision texts. The header
names the regulation, subpart, section, the paragraphs or comments covered,
and the status, so a chunk read on its own still says where it comes from and
whether it is law in force. A rule chunk that starts inside a paragraph (say
at "(D)") also gets a "Lead-in" line quoting the start of its parent paragraph
("(ii) Safe harbors. A card issuer may impose a fee ..."), with that
parent's status when it is not in force.

Regulation Z states some rules twice: once for home-equity plans and once for
other open-end plans such as credit cards (1026.6(a)/(b), 1026.7(a)/(b),
1026.9(c)(1)/(2)), and the official interpretations of the two are often word
for word the same. So every chunk inside one of these six paragraphs gets an
"Applies to" line naming the paragraph and its heading ("Applies to: § 1026.7(a)
Rules affecting home-equity plans"), unless it already holds that paragraph or
its Lead-in quotes it (added in session 11).

Chunk IDs never contain a space. Rule chunk IDs never contain "-"; commentary
chunk IDs end in "-cmt" (before any "#n"); so the two kinds can't collide.
Answers cite provisions (`provision_ids`), never chunk IDs.

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROVISIONS = REPO / "corpus" / "txt" / "provisions.jsonl"
STATUS = REPO / "corpus" / "status.json"
OUT = REPO / "corpus" / "txt" / "chunks.jsonl"

LIMIT = 4000           # characters of provision text per chunk (decided in session 10)
EMBED_MAX = 50_000     # Titan Text Embeddings V2 refuses longer input (embed.py checks it too)

REG_NAME = {
    "E": "Regulation E (Electronic Fund Transfers, 12 CFR Part 1005)",
    "Z": "Regulation Z (Truth in Lending, 12 CFR Part 1026)",
}
# Subpart titles as the eCFR XML prints them (only the subparts in scope).
SUBPART = {
    ("1005", "A"): "General",
    ("1026", "A"): "General",
    ("1026", "B"): "Open-End Credit",
    ("1026", "G"): "Special Rules Applicable to Credit Card Accounts and Open-End Credit Offered to College Students",
}
STATUS_LINE = {
    "in_force": "in force",
    "vacated": "VACATED, not law in force",
    "nullified": "OVERTURNED BY CONGRESS, never took effect",
    "mixed": "PARTLY NOT IN FORCE (contains wording that is not law in force)",
    "not_yet_effective": "NOT YET EFFECTIVE",
}


# ------------------------------------------------------------------ input

def load_provisions(path: Path = PROVISIONS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_missing(path: Path = STATUS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("missing_in_force", [])


# ------------------------------------------------------------------ helpers

def size(ps: list[dict]) -> int:
    """Characters of provision text, joined by newlines (what goes into the chunk)."""
    return sum(len(p["text"]) for p in ps) + max(len(ps) - 1, 0)


def paragraph_prefix(pid: str, section: str, levels: int) -> str:
    """The first `levels` paragraph labels of a rule ID: ('1026.13(c)(2)(i)', '1026.13', 2) -> '1026.13(c)(2)'.
    Returns the ID itself when it has fewer labels."""
    rest = pid[len(section):]
    labels = [lab + ")" for lab in rest.split(")")[:-1]]
    return section + "".join(labels[:levels])


def group_by(ps: list[dict], key) -> list[tuple[str, list[dict]]]:
    """Consecutive runs of provisions sharing key(p), in input order."""
    return [(k, list(g)) for k, g in itertools.groupby(ps, key=key)]


def status_runs(ps: list[dict]) -> list[list[dict]]:
    """Split where status or status_note changes."""
    return [g for _, g in group_by(ps, lambda p: (p["status"], p["status_note"]))]


def pack(ps: list[dict], limit: int = LIMIT) -> list[list[dict]]:
    """Consecutive provisions into pieces of at most `limit` characters (a single longer provision stays whole)."""
    out, cur = [], []
    for p in ps:
        if cur and size(cur + [p]) > limit:
            out.append(cur)
            cur = []
        cur.append(p)
    if cur:
        out.append(cur)
    return out


def pieces(unit_id: str, ps: list[dict], limit: int = LIMIT) -> list[tuple[str, list[dict]]]:
    """A unit's provisions as (chunk_id, provisions): one chunk if it fits and has one status, else
    status runs packed to the limit, numbered unit#1, unit#2 ..."""
    parts = [piece for run in status_runs(ps) for piece in pack(run, limit)]
    if len(parts) == 1:
        return [(unit_id, parts[0])]
    return [(f"{unit_id}#{n}", piece) for n, piece in enumerate(parts, 1)]


# ------------------------------------------------------------------ grouping

def split_level(ps: list[dict], section: str, level: int) -> list[tuple[str, list[dict]]]:
    """Group a section's (or paragraph's) rule provisions by their paragraph at `level`
    (1 = '(c)', 2 = '(c)(2)'). Provisions above that level (the section intro, a paragraph's own
    heading line) join the first group."""
    groups = group_by(ps, lambda p: paragraph_prefix(p["id"], section, level))
    head = []
    out = []
    for key, g in groups:
        if key.count("(") < level and not out:      # text above the first paragraph at this level
            head += g
            continue
        if head:
            g, head = head + g, []
        out.append((key, g))
    if head:                                        # nothing at this level: keep it as one group
        out.append((head[0]["id"], head))
    return out


def rule_units(section: str, ps: list[dict], limit: int = LIMIT) -> list[tuple[str, list[dict]]]:
    """Split one section's rule provisions into units (spec rule 2), then into pieces (rules 4-5)."""
    if size(ps) <= limit and len(status_runs(ps)) == 1:
        return [(section, ps)]
    out = []
    for key1, g1 in split_level(ps, section, 1):
        if size(g1) <= limit:
            out += pieces(key1, g1, limit)
            continue
        for key2, g2 in split_level(g1, section, 2):
            out += pieces(key2, g2, limit)
    return out


def commentary_units(section: str, ps: list[dict], limit: int = LIMIT) -> list[tuple[str, list[dict]]]:
    """Group one section's comments by the paragraph they interpret (spec rule 3)."""
    out = []
    for target, g in group_by(ps, lambda p: p["target"] or section):
        out += pieces(f"{target}-cmt", g, limit)
    return out


# ------------------------------------------------------------------ chunk records

def comment_range(ps: list[dict]) -> str:
    first, last = ps[0]["comment"], ps[-1]["comment"]
    return f"comment {first}" if first == last else f"comments {first} to {last}"


def paragraph_range(ps: list[dict]) -> str:
    first, last = ps[0]["id"], ps[-1]["id"]
    return f"§ {first}" if first == last else f"§ {first} to § {last}"


LEAD_IN_CHARS = 300

# Paragraphs that split a rule by plan type (session 11). Found by their wording; the set is checked
# against this list, so a new eCFR date that adds or drops one stops the build instead of passing silently.
PLAN_SCOPE_RE = re.compile(r"\((\w+)\) (Rules affecting (?:home-equity plans|open-end \(not home-secured\) plans))")
PLAN_SCOPE_IDS = ("1026.6(a)", "1026.6(b)", "1026.7(a)", "1026.7(b)", "1026.9(c)(1)", "1026.9(c)(2)")


def plan_scopes(provisions: list[dict]) -> dict[str, str]:
    """{paragraph ID: its heading} for the paragraphs that split a rule by plan type."""
    found = {}
    for p in provisions:
        m = PLAN_SCOPE_RE.match(p["text"]) if p["kind"] == "rule" else None
        if m:
            if p["status"] != "in_force":
                raise ValueError(f"plan-type paragraph {p['id']} is not in force; its heading can't be quoted")
            found[p["id"]] = m.group(2)
    if sorted(found) != sorted(PLAN_SCOPE_IDS):
        raise ValueError(f"plan-type paragraphs changed: found {sorted(found)}, expected {sorted(PLAN_SCOPE_IDS)}")
    return found


def is_within(pid: str | None, para: str) -> bool:
    """True for the paragraph itself or anything below it: ('1026.7(b)(1)', '1026.7(b)') -> True."""
    return bool(pid) and (pid == para or pid.startswith(para + "("))


def applies_to(ps: list[dict], parent: dict | None, scopes: dict[str, str]) -> str | None:
    """'Applies to: § X <heading>' for a chunk inside a plan-type paragraph that doesn't already
    show that heading (in its own text, or quoted by its Lead-in)."""
    p0 = ps[0]
    key = p0["id"] if p0["kind"] == "rule" else p0["target"]
    for para, heading in scopes.items():
        if not is_within(key, para):
            continue
        if para in {p["id"] for p in ps} or (parent is not None and parent["id"] == para):
            return None
        return f"Applies to: § {para} {heading}"
    return None


def lead_in(parent: dict) -> str:
    """'Lead-in from § X: <start of the parent's text>' for a chunk that starts inside a paragraph,
    so a piece such as '(D) ...' still says what it belongs to. The parent's own text stays in its
    own chunk (that is where it is cited from).

    A parent that is not in force is named but never quoted: its eCFR text can carry the vacated or
    nullified wording (1026.52(b)(1)(ii) states the vacated $8 late fee), and quoting it here would
    put that wording inside a chunk of a different status."""
    if parent["status"] != "in_force":
        return (f"Part of § {parent['id']}, whose text is {STATUS_LINE[parent['status']]}; "
                f"see that paragraph's own chunk")
    t = " ".join(parent["text"].split())
    if len(t) > LEAD_IN_CHARS:
        t = t[:LEAD_IN_CHARS].rsplit(" ", 1)[0] + " …"
    return f"Lead-in from § {parent['id']}: {t}"


def header(kind: str, ps: list[dict], section_heading: str, status: str, note: str | None,
           as_of: str, extra: str | None = None) -> str:
    p0 = ps[0]
    lines = [
        REG_NAME[p0["reg"]],
        f"Subpart {p0['subpart']}: {SUBPART[(p0['part'], p0['subpart'])]}",
    ]
    if kind == "rule":
        lines.append(f"§ {p0['section']} {section_heading}")
        lines.append(f"Covers: {paragraph_range(ps)}")
    else:
        lines.append(f"Official interpretation (Supplement I) of § {p0['section']} {section_heading}")
        # Some source headings already start "Paragraph 55(b)(7)(ii)"; don't print the word twice.
        ph = re.sub(r"^Paragraphs?\s+", "", p0["paragraph_heading"] or "")
        lines.append(f"Paragraph: {ph}" if p0["target"] else f"General comments on § {p0['section']}")
        lines.append(f"Covers: {comment_range(ps)}")
    lines.append(f"Text as of: {as_of}")
    lines.append(f"Status: {STATUS_LINE[status]}" + (f". {note}" if note else ""))
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def make_chunk(chunk_id: str, unit_id: str, ps: list[dict], section_heading: str,
               piece: tuple[int, int] | None, extra: str | None = None,
               parent: dict | None = None, scope: str | None = None) -> dict:
    p0 = ps[0]
    statuses = {(p["status"], p["status_note"]) for p in ps}
    if len(statuses) != 1:
        raise ValueError(f"{chunk_id}: mixes statuses {sorted(statuses, key=str)}")
    status, note = statuses.pop()
    text = "\n".join(p["text"] for p in ps)
    extra = "\n".join(x for x in (scope, lead_in(parent) if parent is not None else None, extra) if x) or None
    head = header(p0["kind"], ps, section_heading, status, note, p0["as_of"], extra)
    return {
        "id": chunk_id,
        "unit_id": unit_id,
        "piece": list(piece) if piece else None,
        "kind": p0["kind"],
        "reg": p0["reg"],
        "part": p0["part"],
        "subpart": p0["subpart"],
        "section": p0["section"],
        "section_heading": section_heading,
        "provision_ids": [p["id"] for p in ps],
        "status": status,
        "status_note": note,
        "as_of": p0["as_of"],
        "chars": len(text),
        "text": text,
        "embed_text": head + "\n\n" + text,
    }


def missing_chunk(entry: dict, sibling: dict, section_heading: str, parent: dict | None) -> dict:
    """A chunk for law in force that the eCFR no longer prints (status.json `missing_in_force`)."""
    rec = {k: sibling[k] for k in ("reg", "part", "subpart", "section", "as_of", "kind")}
    rec.update(id=entry["id"], text=entry["text"], status="in_force",
               status_note=None, parent=sibling["parent"], depth=sibling["depth"])
    extra = ("Note: not printed in the eCFR as of " + sibling["as_of"] + ". The 2024 overdraft lending rule "
             "(89 FR 106768) removed this paragraph, but Congress overturned that rule (Pub. L. 119-10), "
             "so the paragraph remains in force. Text from the eCFR before that rule.")
    ch = make_chunk(entry["id"], entry["id"], [rec], section_heading, None, extra, parent)
    ch["status_note"] = "Not printed in the eCFR; still in force because the rule that removed it was overturned by Congress (Pub. L. 119-10)."
    ch["source"] = "status.json missing_in_force (eCFR text before the 2024 overdraft rule)"
    return ch


def preceding_sibling(pid: str) -> str:
    """'1026.2(a)(15)(ii)(C)' -> '1026.2(a)(15)(ii)(B)' (single-letter or single-digit last label only)."""
    base, last = pid[: pid.rindex("(")], pid[pid.rindex("(") + 1:-1]
    if len(last) != 1 or last in "aA1i":
        raise ValueError(f"{pid}: can't work out the paragraph before it")
    return f"{base}({chr(ord(last) - 1)})"


# ------------------------------------------------------------------ main build

def build(provisions: list[dict], missing: list[dict], limit: int = LIMIT) -> list[dict]:
    headings = {}
    for p in provisions:
        if p["kind"] == "rule":
            headings.setdefault(p["section"], p["section_heading"].rstrip(". "))
    by_id = {p["id"]: p for p in provisions}
    scopes = plan_scopes(provisions)

    chunks = []
    for (section, kind), g in group_by(provisions, lambda p: (p["section"], p["kind"])):
        units = rule_units(section, g, limit) if kind == "rule" else commentary_units(section, g, limit)
        counts = Counter(cid.split("#")[0] for cid, _ in units)
        seen = Counter()
        for cid, ps in units:
            unit_id = cid.split("#")[0]
            seen[unit_id] += 1
            piece = (seen[unit_id], counts[unit_id]) if "#" in cid else None
            ids = {p["id"] for p in ps}
            par = ps[0].get("parent")
            parent = by_id[par] if par and par not in ids else None
            chunks.append(make_chunk(cid, unit_id, ps, headings[section], piece, parent=parent,
                                     scope=applies_to(ps, parent, scopes)))

    for entry in missing:
        if entry["id"] in by_id:
            raise ValueError(f"missing_in_force {entry['id']} is already a provision in the corpus")
        sib = by_id[preceding_sibling(entry["id"])]
        at = next(i for i, c in enumerate(chunks) if sib["id"] in c["provision_ids"])
        par = by_id.get(sib["parent"]) if sib["parent"] else None
        chunks.insert(at + 1, missing_chunk(entry, sib, headings[sib["section"]], par))

    check(chunks, provisions, missing, limit)
    return chunks


def check(chunks: list[dict], provisions: list[dict], missing: list[dict], limit: int = LIMIT) -> None:
    """Stop on anything that would make the chunks untrustworthy."""
    ids = Counter(c["id"] for c in chunks)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise ValueError(f"duplicate chunk IDs: {dup}")
    got = [pid for c in chunks for pid in c["provision_ids"]]
    want_set = {p["id"] for p in provisions} | {m["id"] for m in missing}
    if len(got) != len(want_set) or set(got) != want_set:
        raise ValueError("chunks don't hold every provision exactly once")
    order = [pid for pid in got if pid not in {m["id"] for m in missing}]
    if order != [p["id"] for p in provisions]:
        raise ValueError("chunks changed the order of provisions")
    for c in chunks:
        if c["chars"] > limit and len(c["provision_ids"]) > 1:
            raise ValueError(f"{c['id']}: over {limit} characters with more than one provision")
        if len(c["embed_text"]) > EMBED_MAX:
            raise ValueError(f"{c['id']}: embed_text over {EMBED_MAX} characters")
        if " " in c["id"]:
            raise ValueError(f"{c['id']}: space in chunk ID")


def summary(chunks: list[dict], limit: int = LIMIT) -> dict:
    sizes = sorted(c["chars"] for c in chunks)
    n = len(sizes)
    return {
        "chunks": n,
        "by_kind": dict(Counter(c["kind"] for c in chunks)),
        "by_status": dict(Counter(c["status"] for c in chunks)),
        "chars": {"median": sizes[n // 2], "p90": sizes[int(0.9 * n)], "max": sizes[-1]},
        "single_provisions_over_limit": sum(1 for c in chunks if c["chars"] > limit),
        "split_into_pieces": sum(1 for c in chunks if c["piece"]),
        "whole_sections": sum(1 for c in chunks if c["kind"] == "rule" and c["unit_id"] == c["section"]),
    }


def write(chunks: list[dict], path: Path = OUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    try:
        chunks = build(load_provisions(), load_missing())
    except ValueError as e:
        sys.exit(f"chunk.py stopped: {e}")
    write(chunks, args.out)
    print(json.dumps(summary(chunks), indent=1))
    print(f"Wrote {len(chunks)} chunks to {args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out}")


if __name__ == "__main__":
    main()
