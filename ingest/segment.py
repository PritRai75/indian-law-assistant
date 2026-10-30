# Corpus pipeline (Project 1). v0 from session 3; v1 (session 4, 30 Oct 2026) fixes state-amendment
# leaks and 8 missed CPC units. Tests: tests/test_segment.py. Run from a folder containing raw/BNS.pdf,
# raw/BNSS.pdf, raw/BSA.pdf, raw/CPC.pdf (the India Code PDFs) and an empty txt/ folder.
# Order: python extract.py BNS BNSS BSA CPC  ->  python segment.py BNS BNSS BSA CPC  ->  python schedule.py
# Needs: pip install pdfplumber. See corpus-audit.md for what each rule fixes.
"""Segment extracted body text into: contents / body / schedule / appendix / state_amendment.
Then parse provision units and report token stats. Tokens are estimated as chars / 4
(no tokenizer reachable offline here; treat as +-15%)."""
import json, re, sys, collections


tok = lambda s: round(len(s) / 4)

# Any run of footnote markers, stars and opening brackets before a number: "⟨8⟩60.", "*[4.",
# "⟨1⟩[ ⟨2⟩5." (two nested amendments, CPC O.XVIII r.5).
MARK = r"(?:⟨[^⟩]*⟩\s*|\*|\[\s*)*"
# A provision number: "35", "35A", and "46-I" (the CPC body prints O.XXI r.46I with a hyphen).
# num() drops the hyphen so the number matches the contents list.
NUM = r"(\d+(?:-?[A-Z]{1,3})?)"
num = lambda s: s.replace("-", "")
SEC = re.compile(rf"^{MARK}{NUM}\.\s*(?:\[)?([A-Z“\"‘'(][^—–]*?)\s*\.?\s*[—–-]{{1,2}}")
ORDER = re.compile(rf"^{MARK}\s*[“\"]?ORDER\s+([IVXL]+(?:\s*-?\s*[A-Z])?)\b")
norm_order = lambda o: re.sub(r"\s*-?\s*([A-Z])$", r"-\1", o) if re.search(r"[IVXL]\s*-?\s*[A-HJ-UWYZ]$", o) else o


def load(act):
    rows = [l.split("\t") for l in open(f"txt/{act}.body.tsv").read().split("\n")]
    return [(r[0], int(r[1])) for r in rows]


# A line that starts a numbered provision: "116. Part to…", "⟨8⟩60. Property…", "110. [Value…] Rep.",
# "255.—Public servant…". Looser than SEC on purpose: it only needs the start of the line, so a
# heading that wraps before its dash still counts.
HEAD_NUM = re.compile(rf"^{MARK}{NUM}\.\s*(?:\[|[—–]|[A-Z“\"‘'(])")
# An ORDER heading with no opening quote mark. A quoted “ORDER … inside a state block is state text.
ORDER_HEAD = re.compile(r"^(?:⟨[^⟩]*⟩|\*)?\[?\s*ORDER\s+([IVXL]+(?:\s*-?\s*[A-Z])?)\b")
# How far ahead in the Act's own contents list a heading may be and still count as the next one.
# Central text: up to 10 ahead, so a missed heading (e.g. an unparsed repealed stub) can't stall
# the tracker. Inside a state block: up to 3 ahead, so quoted text rarely ends a block by accident.
BODY_WINDOW, STATE_WINDOW = 10, 3
# The opening words of an amending Act's own section ("Amendment of section 115 of Act V of 1908.—",
# "Substitution of section 115.—"). In the CPC body these lines are always state text, including a
# state block that resumes after central text without repeating its STATE AMENDMENT heading.
AMENDING_FORMULA = re.compile(r"^(?:Amendment|Insertion|Substitution|Omission)s? (?:of|in|to)\b")


def segment(act, rows):
    """Label each line: contents, body, schedule, appendix, objects_and_reasons or state_amendment.

    A state-amendment block starts at a "STATE AMENDMENT(S)" heading, or (CPC) at an amending Act's
    formula line ("Amendment of section 115.—"). It ends only when central text resumes: the next
    central heading in the Act's own contents order, a new ORDER, a starred Commercial Courts Act
    heading ("*[4." or "*[ORDER XV-A"), or a change of part (appendix, schedule, objects and reasons). v0 ended a block at the first line
    after a "[Vide …]" line that wasn't a state name, which leaked consecutive amendments by the
    same state into central units (CPC ss.9, 35A, 60, 115, O.IV r.2, O.XIX r.4).
    """
    text = [r[0] for r in rows]
    # body starts at the enacting formula (after the arrangement of sections)
    start = next(i for i, t in enumerate(text) if t.startswith("BE it enacted") or t.startswith("WHEREAS"))
    seg = ["contents"] * start
    toc = toc_of(act, rows, seg)          # zip() stops at len(seg), so this reads the contents only
    order, ptr = None, -1                 # current Order (None = sections) and last central heading's toc index
    mode, in_state = "body", False
    for i in range(start, len(text)):
        t = text[i]
        prev_mode = mode
        if re.match(r"^(⟨[^⟩]*⟩)?\[?THE (FIRST|SECOND) SCHEDULE", t) and act in ("BNSS", "BSA"):
            mode = "schedule"
        if act == "CPC" and re.match(r"^(⟨[^⟩]*⟩|\*)?\[?\s*APPENDIX", t) and i > start:
            mode = "appendix"
        if act == "CPC" and t.startswith("THE SECOND SCHEDULE"):
            mode = "schedule"
        if t.startswith("STATEMENT OF OBJECTS AND REASONS"):
            mode = "objects_and_reasons"
        if mode != prev_mode:
            in_state = False
        starred = t.startswith("*[")   # Commercial Courts Act text: always central
        if re.match(r"^STATE AMENDMENTS?\b", t) or (act == "CPC" and AMENDING_FORMULA.match(t)):
            in_state = True
        elif mode == "body":
            om = ORDER_HEAD.match(t) if act == "CPC" else None
            if om and (norm_order(om.group(1)) in toc or starred):
                order, ptr, in_state = norm_order(om.group(1)), -1, False
            else:
                hm = HEAD_NUM.match(t)
                if hm:
                    ahead = toc.get(order, [])[ptr + 1: ptr + 1 + (STATE_WINDOW if in_state else BODY_WINDOW)]
                    if num(hm.group(1)) in ahead:
                        ptr += 1 + ahead.index(num(hm.group(1)))
                        in_state = False
                    elif starred:
                        in_state = False
        seg.append("state_amendment" if in_state else mode)
    return seg


def toc_of(act, rows, seg):
    """Section/rule numbers listed in the Act's own arrangement, keyed by Order (None = sections)."""
    toc, order = collections.defaultdict(list), None
    # Start at the arrangement heading: the CPC prints a numbered "LIST OF AMENDING ACTS" before it,
    # and those numbers ("35. The Code of Civil Procedure (Amendment) Act, 1949") are not sections.
    first = next((i for i, ((t, _), s) in enumerate(zip(rows, seg))
                  if s == "contents" and t.startswith("ARRANGEMENT OF")), 0)
    for (t, _), s in list(zip(rows, seg))[first:]:
        if s != "contents":
            continue
        m = ORDER.match(t)
        if act == "CPC" and m:
            order = norm_order(m.group(1)); continue
        m = re.match(r"^(?:⟨[^⟩]*⟩|\*)?\[?(\d+[A-Z]{0,3})\.\s", t)
        if m:
            toc[order].append(m.group(1))
    return toc


# A repealed or omitted stub: "110. [Value of subject matters.] Rep. by …", "7. ⟨1⟩[No security …] Rep.",
# "17A. [Production of evidence …] omitted by …" (the last after joining its wrapped second line).
# Only the keyword is case-insensitive; the section letters stay upper-case.
REPEALED = re.compile(rf"^{MARK}{NUM}\.\s*(?:⟨[^⟩]*⟩\s*)*\[?\s*(?:⟨[^⟩]*⟩\s*)*\[([^\]]+)\]\.?\]?\s*\.?\s*(?:(?i:rep\.|omitted|repealed)|\*)")
SEC_DASH_FIRST = re.compile(rf"^{MARK}{NUM}\.\s*[—–]+\s*([A-Z].{{0,80}})")  # e.g. BNS "255.—Public servant..."


def units(act, rows, seg):
    """Split body into provision units: section N (BNS/BNSS/BSA/CPC sections) or ORDER/RULE (CPC).
    A heading is accepted only if its number appears in the Act's own arrangement for that Order and
    has not been seen yet -- this rejects line-initial years ("2023. A may be...") and list items."""
    toc = toc_of(act, rows, seg)
    star_notes = collections.defaultdict(list)
    for l in open(f"txt/{act}.notes.jsonl"):
        n = json.loads(l)
        if n["text"].startswith("*"):
            star_notes[n["page"]].append(n["text"])

    def star_kind(pg):
        """Classify a '*' marker by its footnote verb: Ins./Subs. -> commercial text; Omitted -> general
        text that does not apply to commercial disputes."""
        for txt in star_notes.get(pg, []) + star_notes.get(pg + 1, []):
            if "commercial" in txt.lower():
                return "omitted_for_commercial" if "Omitted" in txt[:20] else "commercial"
        return "commercial"

    seen, seen_orders = set(), set()
    out, cur, order, order_regime = [], None, None, "general"
    for i, ((t, pg), s) in enumerate(zip(rows, seg)):
        if s != "body":
            continue
        m = ORDER.match(t)
        if act == "CPC" and m:
            order = norm_order(m.group(1))
            # "*[ORDER XI" = Commercial Courts Act 2016 version (footnote "*")
            order_regime = star_kind(pg) if t.startswith("*[") else "general"
            if order_regime == "commercial" and order in seen_orders:
                seen = {k for k in seen if k[0] != order + "@commercial"}
            seen_orders.add(order)
            cur = None
            continue
        head = t   # the heading text, joined with its continuation lines when it wraps
        m = SEC.match(t) or REPEALED.match(t) or SEC_DASH_FIRST.match(t)
        if not m and re.match(rf"^{MARK}{NUM}\.\s", t):
            # heading (or repealed stub) wrapped onto the next line(s) before its dash or "Rep."
            joined = t
            for j in (1, 2):
                if i + j < len(rows):
                    joined += " " + rows[i + j][0]
                    m = SEC.match(joined) or REPEALED.match(joined)
                    if m:
                        head = joined
                        break
        regime = "general"
        if m:
            n = num(m.group(1))
            starred = t.startswith("*[")
            kind = star_kind(pg) if starred else order_regime
            regime = "commercial" if kind == "commercial" else "general"
            okey = order
            if regime == "commercial" and (order, n) in seen:
                okey = (order or "") + "@commercial"   # a second, commercial-dispute version
            allowed = toc.get(order)
            if (okey, n) in seen or (allowed and n not in allowed and regime == "general"):
                m = None
        if m:
            seen.add((okey, n))
            suffix = "@commercial" if okey != order else ""
            uid = (f"{act}-O{order}-R{n}" if order else f"{act}-S{n}") + suffix
            cur = {"id": uid, "page": pg, "heading": m.group(2)[:80], "regime": regime,
                   "applies_to_commercial": kind != "omitted_for_commercial",
                   "status": "repealed" if REPEALED.match(head) else "in_force", "text": t}
            out.append(cur)
        elif cur:
            cur["text"] += "\n" + t
    return out


if __name__ == "__main__":
    report = {}
    for act in sys.argv[1:]:
        rows = load(act)
        seg = segment(act, rows)
        notes = [json.loads(l)["text"] for l in open(f"txt/{act}.notes.jsonl")]
        by = collections.defaultdict(list)
        for (t, _), s in zip(rows, seg):
            by[s].append(t)
        stats = {k: tok("\n".join(v)) for k, v in by.items()}
        stats["footnotes"] = tok("\n".join(notes))
        u = units(act, rows, seg)
        json.dump(u, open(f"txt/{act}.units.json", "w"), ensure_ascii=False, indent=0)
        json.dump(seg, open(f"txt/{act}.seg.json", "w"))
        sizes = sorted(tok(x["text"]) for x in u)
        q = lambda p: sizes[min(len(sizes) - 1, int(p * len(sizes)))]
        report[act] = dict(tokens=stats, units=len(u), unit_tokens=dict(
            p50=q(.5), p90=q(.9), p99=q(.99), max=sizes[-1], over_1k=sum(s > 1000 for s in sizes),
            over_2k=sum(s > 2000 for s in sizes)),
            biggest=[(x["id"], tok(x["text"])) for x in sorted(u, key=lambda x: -tok(x["text"]))[:5]])
    print(json.dumps(report, indent=1))
