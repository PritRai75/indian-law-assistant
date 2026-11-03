"""Turn the eCFR XML for Regulations E and Z into a list of provisions.

Run from the repo root:

    python -m ingest.parse            # writes corpus/txt/provisions.jsonl and prints a summary

A *provision* is the smallest piece a citation can point to:

- a **rule paragraph**, such as 1026.52(b)(1)(ii)(A), with only its own text
  (its children are separate provisions), or
- a **numbered official interpretation** (a "comment"), such as comment
  52(b)(1)(ii)-1, with all its sub-items (i., ii., A., B. ...).

Only the sections in scope are kept: Reg E subpart A (1005.1-1005.20) and
Reg Z subparts A, B and G (1026.1-1026.16 and 1026.51-1026.62), plus the
Supplement I commentary on those sections. chunk.py later groups provisions
into the units that are embedded and searched.

ID scheme (fixed in session 9):

| Kind       | ID                       | How it is cited                   |
| ---------- | ------------------------ | --------------------------------- |
| rule       | 1026.52(b)(1)(ii)(A)     | 12 CFR 1026.52(b)(1)(ii)(A)       |
| commentary | 1026.52(b)(1)(ii)-1      | Reg Z comment 52(b)(1)(ii)-1      |
| commentary | 1026.13-1 (section-level)| Reg Z comment 13-1                |

A rule ID never contains "-", and a commentary ID always does, so the two
can't collide. Italic labels such as (<I>1</I>) are written as (1), the way
the CFR cites them.

The XML has no nesting below the section: every paragraph is a flat <P>
starting with its label, and one <P> can hold several levels
("(a) Heading—(1) Heading. Text"). So the tree is rebuilt from the labels.
A label is accepted only where it can legally come next (the first item of a
deeper level, or the successor of the current item at its level); anything
else stops the run with an error, so a parsing mistake can't pass silently.

Status: corpus/status.json (hand-checked) marks provisions that the eCFR
prints but that are not law in force. Each record gets `status` and
`status_note` from it; everything else is "in_force".

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "corpus" / "raw"
MANIFEST = REPO / "corpus" / "manifest.json"
STATUS = REPO / "corpus" / "status.json"
OUT = REPO / "corpus" / "txt" / "provisions.jsonl"

REG = {"1005": "E", "1026": "Z"}
SCOPE = {
    "1005": set(range(1, 21)),
    "1026": set(range(1, 17)) | set(range(51, 63)),
}
STATUSES = {"in_force", "vacated", "nullified", "mixed", "not_yet_effective"}


class ParseError(Exception):
    pass


# ---------------------------------------------------------------- text helpers

def clean(s: str) -> str:
    """Collapse all whitespace (including newlines inside the XML) to single spaces."""
    return " ".join(s.split())


def text_of(e: ET.Element) -> str:
    return clean("".join(e.itertext()))


TAG_RE = re.compile(r"<[^>]+>")


def strip_tags(s: str) -> str:
    """Remove XML tags from a serialised fragment and undo the entity escapes."""
    s = TAG_RE.sub("", s)
    s = re.sub(r"\(_([0-9]+|[ivx]+)\)", r"(\1)", s)  # undo the italic-label marker in cross-references
    s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return clean(s)


def inner_xml(e: ET.Element) -> str:
    """The element's content as a string, without its own start and end tag."""
    raw = ET.tostring(e, encoding="unicode")
    raw = raw[raw.index(">") + 1:]
    raw = raw[: raw.rindex("</")]
    return clean(raw)


# ------------------------------------------------------- paragraph label levels

# CFR paragraph levels: (a) (1) (i) (A) (italic 1) (italic i)
LETTER, NUMBER, ROMAN, UPPER, INUMBER, IROMAN = 1, 2, 3, 4, 5, 6
ROMAN_RE = re.compile(r"^(?=[ivxl]+$)l?x{0,3}(ix|iv|v?i{0,3})$")


def roman_to_int(s: str) -> int:
    vals = {"i": 1, "v": 5, "x": 10, "l": 50}
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = vals[a]
        total += -v if b != " " and vals.get(b, 0) > v else v
    return total


def letter_index(s: str) -> int:
    """a=1 ... z=26, aa=27 ... (the CFR doubles letters after z)."""
    if len(set(s)) != 1:
        return -1
    return (len(s) - 1) * 26 + (ord(s[0]) - ord("a") + 1)


def candidates(raw: str) -> list[tuple[int, int]]:
    """Possible (level, ordinal) readings of a label; '_' marks an italic label."""
    italic = raw.startswith("_")
    lab = raw.lstrip("_")
    out = []
    if lab.isdigit():
        out.append((INUMBER if italic else NUMBER, int(lab)))
    elif lab.isupper() and lab.isalpha() and len(set(lab)) == 1:
        out.append((UPPER, letter_index(lab.lower())))
    elif lab.islower() and lab.isalpha():
        if ROMAN_RE.match(lab):
            out.append((IROMAN if italic else ROMAN, roman_to_int(lab)))
        if not italic and letter_index(lab) > 0:
            out.append((LETTER, letter_index(lab)))
    return out


class Tree:
    """Tracks the current path of labels in one section and places new labels."""

    def __init__(self, section: str, overrides: dict[str, int]):
        self.section = section
        self.stack: list[tuple[int, int, str]] = []  # (level, ordinal, label)
        self.overrides = overrides  # provision id -> forced level (for the rare true ambiguity)
        self.pending: str | None = None  # id of an ambiguous label read as roman, awaiting "(ii)"
        self.pending_next: str | None = None

    def _valid(self, level: int, ordinal: int) -> bool:
        deepest = self.stack[-1][0] if self.stack else 0
        if ordinal == 1 and (level == deepest + 1 or (deepest == 0 and level == LETTER)):
            return True  # first item of the next level down
        for lv, od, _ in self.stack:
            if lv == level:
                return ordinal == od + 1  # next sibling at that level
        return False

    def place(self, raw: str) -> tuple[str, int, list[str]]:
        """Place a label; return (provision id, level, list of ambiguity notes)."""
        lab = raw.lstrip("_")
        options = [c for c in candidates(raw) if self._valid(*c)]
        notes = []
        if len(options) > 1:
            # Two readings fit, e.g. "(i)" after "(h)" when (h) has children.
            # Try each and use an explicit override if one names the result.
            ids = {}
            for lv, od in options:
                ids[lv] = self._id_if(lv, lab)
            forced = [lv for lv, pid in ids.items() if self.overrides.get(pid) == lv]
            if len(forced) == 1:
                options = [c for c in options if c[0] == forced[0]]
            else:
                # Take the deeper reading (a new roman list), but only on condition
                # that the next label at its level or above is its "(ii)"; checked
                # below and at the end of the section, so a wrong guess stops the run.
                options = [max(options)]
                self.pending_next = ids[options[0][0]]
        if not options:
            path = self._path()
            raise ParseError(
                f"{self.section}: label ({lab}) cannot follow {path or 'the section start'}"
            )
        level, ordinal = options[0]
        if self.pending and level <= ROMAN:  # its own children (A), (1) ... may come first
            if not (level == ROMAN and ordinal == 2):
                raise ParseError(
                    f"{self.pending}: read as roman (i), but the next label is ({lab}), "
                    "not (ii). Add an entry to OVERRIDES."
                )
            self.pending = None
        if self.pending_next:
            self.pending, self.pending_next = self.pending_next, None
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, ordinal, lab))
        return self._path(), level, notes

    def finish(self) -> None:
        if self.pending:
            raise ParseError(f"{self.pending}: read as roman (i), but no (ii) follows. Add an entry to OVERRIDES.")

    def _path(self) -> str:
        return self.section + "".join(f"({lab})" for _, _, lab in self.stack)

    def _id_if(self, level: int, lab: str) -> str:
        kept = [s for s in self.stack if s[0] < level]
        return self.section + "".join(f"({l})" for _, _, l in kept) + f"({lab})"


# ------------------------------------------------------------- rule paragraphs

# A label at the current position: "(b)" or italic "(<I>1</I>)" (already
# rewritten to "(_1)" before matching).
LABEL_AT = re.compile(r"^\s*\((_?[A-Za-z0-9]{1,6})\)\s*")
# An italic heading right after a label, followed by another label:
# "<I>Heading</I>—(1) ..." or "<I>Heading.</I> (i) ..."
HEADING_THEN_LABEL = re.compile(r"^(<I>[^<]*</I>)\s*—?\s*(?=\(_?[A-Za-z0-9]{1,6}\)[\s(])")


def segments_of_p(p: ET.Element) -> list[tuple[str | None, str]]:
    """Split one <P> into (label, xml-fragment) pieces. Label None = no label."""
    s = re.sub(r"\(<I>([0-9]+|[ivx]+)</I>\)", r"(_\1)", inner_xml(p))
    pieces: list[tuple[str | None, str]] = []
    pos = 0
    m = LABEL_AT.match(s)
    if not m:
        return [(None, s)]
    while True:
        m = LABEL_AT.match(s[pos:])
        if not m:
            break
        label = m.group(1)
        pos += m.end()
        start = pos
        # Labels back to back, e.g. "(a)(1) Account means ...": empty text for (a).
        if LABEL_AT.match(s[pos:]) and s[pos:pos + 1] == "(":
            pieces.append((label, ""))
            continue
        h = HEADING_THEN_LABEL.match(s[pos:])
        if h:
            pos += h.end()
            pieces.append((label, s[start:start + len(h.group(1))]))
            continue
        pieces.append((label, s[start:]))
        pos = len(s)
        break
    return pieces


def label_display(raw: str) -> str:
    return f"({raw.lstrip('_')})"


def parse_rules(part: str, root: ET.Element, overrides: dict[str, int],
                notes: list[str]) -> list[dict]:
    out = []
    for d6 in root.iter("DIV6"):
        sub = d6.get("N")
        for d8 in d6.iter("DIV8"):
            sec_id = d8.get("N")
            sec_no = int(sec_id.split(".")[1].split("-")[0])
            if sec_no not in SCOPE[part]:
                continue
            head = text_of(d8.find("HEAD"))
            heading = re.sub(r"^§\s*[\d.]+\s*", "", head)
            tree = Tree(sec_id, overrides)
            for el in d8:
                if el.tag in ("HEAD", "CITA", "SECAUTH", "EDNOTE"):
                    continue
                if el.tag != "P":
                    raise ParseError(f"{sec_id}: unexpected <{el.tag}> in rule text")
                for raw_label, frag in segments_of_p(el):
                    text = strip_tags(frag)
                    if raw_label is None:
                        if not out or out[-1]["section"] != sec_id:
                            # Introductory text before the first label: the section itself.
                            out.append({
                                "id": sec_id, "kind": "rule", "reg": REG[part], "part": part,
                                "subpart": sub, "section": sec_id, "section_heading": heading,
                                "parent": None, "depth": 0, "text": text,
                            })
                            continue
                        out[-1]["text"] += "\n" + text
                        notes.append(f"{out[-1]['id']}: unlabelled paragraph joined to it ({text[:50]}...)")
                        continue
                    pid, level, amb = tree.place(raw_label)
                    notes.extend(amb)
                    parent = pid[: pid.rindex("(")] if "(" in pid else None
                    out.append({
                        "id": pid,
                        "kind": "rule",
                        "reg": REG[part],
                        "part": part,
                        "subpart": sub,
                        "section": sec_id,
                        "section_heading": heading,
                        "parent": parent if parent and parent != sec_id else None,
                        "depth": pid.count("("),
                        "text": f"{label_display(raw_label)} {text}".strip(),
                    })
            tree.finish()
    return out


# ------------------------------------------------------------------ commentary

SEC_HEAD = re.compile(r"^Section (\d{4})\.(\d+)\b")
# "13(c)(2) Time for resolution", "Paragraph 2(b)(2).", "Paragraphs 4(b)(7) and (b)(8)."
PARA_HEAD = re.compile(r"^(?:Paragraphs? )?(\d+)((?:\([A-Za-z0-9]{1,6}\))+)(?=[\s.,—]|$)")
_REF = r"\d+(?:\([A-Za-z0-9]{1,6}\))+"
P_BARE_REF = re.compile(rf"^(?:Paragraphs? )?{_REF}(?:,? (?:and|through) (?:\([A-Za-z0-9]{{1,6}}\))+)*\.?$")
P_REF_TITLE = re.compile(rf"^{_REF}\s+[A-Z][^.;:]{{0,100}}\.?$")
OTHER_HEAD = re.compile(r"^(Subpart [A-Z]\b|Introduction\b|Appendi(?:x|ces)\b)")
COMMENT_START = re.compile(r"^(\d+)\.\s")  # "1. ..." (not italic)
SUBITEM = re.compile(r"^(?:[ivxl]+|[A-Z]|\d+)\.\s")

# Paragraph headings the eCFR XML leaves out, found because comment numbering
# restarts at "1." without a heading: (part, heading in force, expected next
# number) -> the heading that belongs there. Each was checked against the rule
# text the comments interpret.
MISSING_HEADINGS = {
    # Comments "1. Variations in terms, conditions, or features" interpret
    # 1005.17(b)(3) "Same account terms, conditions, and features". The CFPB's
    # copy has the heading (consumerfinance.gov/rules-policy/regulations/1005/17,
    # checked 3 Nov 2026).
    ("1005", "17(b)(2)", 3): "17(b)(3)",
}

# Numbering slips in the eCFR XML: (part, heading, expected number, number
# printed) -> how much to add to that number and every later one under the same
# heading, so IDs match the CFPB's published numbering.
NUMBERING_FIXES = {
    # eCFR prints "2." twice under 11(c); the CFPB numbers the comments 1-7
    # (consumerfinance.gov/rules-policy/regulations/1026/interp-11/, checked 3 Nov 2026).
    ("1026", "11(c)", 3, 2): 1,
}


def is_heading(el: ET.Element, t: str) -> bool:
    """Headings come as HD1-HD3 tags, as an italic <P>, or as a short plain <P>."""
    if el.tag in ("HD1", "HD2", "HD3"):
        return not SUBITEM.match(t)  # a few comment sub-items are tagged HD3
    if el.tag != "P":
        return False
    # "Paragraph 51(b)(1)(i)." or "Paragraphs 4(b)(7) and (b)(8)." (italic or plain)
    if P_BARE_REF.match(t):
        return True
    # "2(a)(27)(i) Successor in interest": a reference plus a short title, no sentence
    return bool(P_REF_TITLE.match(t))


def parse_commentary(part: str, root: ET.Element, notes: list[str]) -> list[dict]:
    supp = next(d for d in root.iter("DIV9") if d.get("N", "").startswith("Supplement I"))
    out: list[dict] = []
    sec: int | None = None
    sec_heading = ""
    key: str | None = None  # "13(c)(2)" or "13" for section-level comments
    para_heading = ""
    current: dict | None = None
    expected = Counter()  # heading -> last comment number seen
    offsets = Counter()  # heading -> renumbering shift from NUMBERING_FIXES

    def close():
        nonlocal current
        if current:
            current["text"] = "\n".join(current.pop("_lines"))
            out.append(current)
        current = None

    for el in supp:
        if el.tag in ("HEAD", "CITA", "XREF", "EDNOTE", "FP"):
            continue
        t = text_of(el)
        if not t:
            continue
        if is_heading(el, t):
            close()
            m = SEC_HEAD.match(t)
            if m:
                if m.group(1) != part:
                    raise ParseError(f"Supplement I of {part}: heading for another part: {t}")
                sec = int(m.group(2))
                sec_heading = re.sub(r"^Section \d{4}\.\d+\s*[—-]?\s*", "", t)
                key, para_heading = str(sec), ""
                continue
            m = PARA_HEAD.match(t)
            if m:
                if sec is None or int(m.group(1)) != sec:
                    if sec is not None and sec in SCOPE[part]:
                        raise ParseError(f"{part} Supp I: paragraph heading {t!r} under section {sec}")
                    continue
                key = m.group(1) + m.group(2)
                para_heading = t
                if t.startswith("Paragraphs"):
                    notes.append(f"{part}.{key}: heading covers several paragraphs: {t!r}")
                continue
            if OTHER_HEAD.match(t):
                sec, key = None, None
                continue
            if sec is not None and sec in SCOPE[part]:
                raise ParseError(f"{part} Supp I, section {sec}: unrecognised heading <{el.tag}> {t[:80]!r}")
            continue  # free-form headings in sections we don't index

        # Body text
        if sec is None or sec not in SCOPE[part]:
            continue
        raw = inner_xml(el)
        m = COMMENT_START.match(t)
        # "<I>1.</I> Assume ..." (only the number italic) is a sub-item below "A.";
        # "<I>1. Example ...</I>" (number and heading italic) starts a comment.
        if m and not re.match(r"^<I>\d+\.\s*</I>", raw):
            printed = int(m.group(1))
            want = expected[key] + 1
            shift = NUMBERING_FIXES.get((part, key, want, printed))
            if shift is not None:
                offsets[key] = shift
                notes.append(f"{part}.{key}-{want}: eCFR prints {printed}.; renumbered from here "
                             f"to match the CFPB's numbering")
            n = printed + offsets[key]
            if n != printed:
                t = f"{n}." + t[len(m.group(0)) - 1:]  # keep the text, correct the number
            fix = MISSING_HEADINGS.get((part, key, want))
            if n == 1 and fix:
                notes.append(f"{part} Supp I: heading {fix} is missing from the eCFR XML; "
                             f"comments after {key}-{want - 1} assigned to {fix}")
                key, para_heading, want = fix, f"{fix} (heading missing from the eCFR XML)", 1
            if n != want:
                raise ParseError(f"{part} comment {key}-{n}: expected number {want}")
            expected[key] = n
            close()
            target = f"{part}.{key}"
            current = {
                "id": f"{part}.{key}-{n}",
                "kind": "commentary",
                "reg": REG[part],
                "part": part,
                "subpart": None,  # filled from the rule sections below
                "section": f"{part}.{sec}",
                "section_heading": sec_heading,
                "comment": f"{key}-{n}",
                "target": target if "(" in key else None,
                "paragraph_heading": para_heading or None,
                "_lines": [t],
            }
            continue
        if current is None:
            raise ParseError(f"{part} Supp I, section {sec}, under {key!r}: text before comment 1: {t[:80]!r}")
        current["_lines"].append(t)
    close()
    return out


# ---------------------------------------------------------------------- status

def load_status(path: Path = STATUS) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data["entries"]
    for e in entries:
        if e["status"] not in STATUSES:
            raise ParseError(f"status.json: unknown status {e['status']!r} for {e['id']}")
        for k in ("id", "status", "note", "source"):
            if not e.get(k):
                raise ParseError(f"status.json: entry {e} is missing {k!r}")
    return entries


def covers(entry: dict, pid: str) -> bool:
    """An entry covers its exact ID, and with "subtree": true, everything under it."""
    if pid == entry["id"]:
        return True
    if not entry.get("subtree"):
        return False
    return pid.startswith(entry["id"] + "(") or pid.startswith(entry["id"] + "-")


def apply_status(records: list[dict], entries: list[dict]) -> None:
    used = Counter()
    for r in records:
        hits = [e for e in entries if covers(e, r["id"])]
        if len(hits) > 1:
            # The most specific entry wins (the longest ID).
            hits.sort(key=lambda e: len(e["id"]), reverse=True)
        if hits:
            e = hits[0]
            used[e["id"]] += 1
            r["status"], r["status_note"] = e["status"], e["note"]
        else:
            r["status"], r["status_note"] = "in_force", None
    unused = [e["id"] for e in entries if not used[e["id"]]]
    if unused:
        raise ParseError(f"status.json entries that match no provision: {unused}")


# ------------------------------------------------------------------------ main

def parse_part(part: str, path: Path, overrides: dict[str, int] | None = None,
               notes: list[str] | None = None) -> list[dict]:
    notes = notes if notes is not None else []
    root = ET.parse(path).getroot()
    rules = parse_rules(part, root, overrides or {}, notes)
    comments = parse_commentary(part, root, notes)
    subparts = {r["section"]: r["subpart"] for r in rules}
    for c in comments:
        c["subpart"] = subparts.get(c["section"])
    ids = Counter(r["id"] for r in rules + comments)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        raise ParseError(f"duplicate provision IDs in {part}: {dupes[:10]}")
    return rules + comments


# Labels that read both ways and are settled by hand: provision id -> level.
OVERRIDES: dict[str, int] = {}


def parse_all(raw_dir: Path = RAW_DIR, status_path: Path = STATUS) -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    records = []
    for part in ("1005", "1026"):
        records += parse_part(part, raw_dir / f"part-{part}.xml", OVERRIDES, notes)
    apply_status(records, load_status(status_path))
    return records, notes


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true", help="don't print the notes")
    args = ap.parse_args(argv)
    try:
        records, notes = parse_all()
    except ParseError as e:
        raise SystemExit(f"PARSE ERROR: {e}")
    as_of = json.loads(MANIFEST.read_text(encoding="utf-8"))["as_of"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            r["as_of"] = as_of
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter((r["reg"], r["kind"]) for r in records)
    s = Counter(r["status"] for r in records)
    print(f"As of {as_of}: wrote {len(records):,} provisions to {OUT.relative_to(REPO).as_posix()}")
    for (reg, kind), n in sorted(c.items()):
        print(f"  Reg {reg} {kind:<10} {n:>5}")
    print("  status: " + ", ".join(f"{k} {v}" for k, v in sorted(s.items())))
    if notes and not args.quiet:
        print(f"\nNotes ({len(notes)}):")
        for n in notes:
            print(f"  - {n}")


if __name__ == "__main__":
    main(sys.argv[1:])
