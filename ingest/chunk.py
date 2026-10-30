# Corpus pipeline (Project 1), session 5. Turns provision units into retrieval chunks.
# Run from corpus/ after segment.py:  python ../ingest/chunk.py BNS BNSS BSA CPC
# Reads txt/<ACT>.units.json, writes txt/chunks.jsonl (one chunk per line, all Acts) and prints a report.
"""Chunk provision units for retrieval (spec: "Chunking rule" and "Chunk IDs").

- One chunk per section or rule. Repealed stubs stay as their own short chunks.
- Definition sections (BNS, BNSS, BSA and CPC s.2) are split one clause per chunk: BNS-S2(8),
  BNSS-S2(1)(g), CPC-S2(7A). Text before the first clause is kept with the first clause.
- Other units over MAX_TOKENS are split at numbered sub-sections: BNSS-S187(3).
- A unit or sub-section that is still over MAX_TOKENS (no sub-sections to split at) is cut into parts
  at block starts (Explanation, Exception, Illustrations, Provided that, a lettered clause), packed to
  about PART_TOKENS each: BNS-S101#2. A part is not a legal unit, so answers cite the unit ID.

Every chunk keeps its unit's metadata, the footnotes whose markers fall inside the chunk, and an
`embed_text`: the Act, chapter and heading prefixed to the text, so a chunk cut from the middle of a
section still says where it comes from. Footnote text is not embedded (spec: footnotes are passed to
the model with the passage, not searched).
"""
import json, re, sys, collections

tok = lambda s: round(len(s) / 4)   # chars / 4, as in segment.py (+-15%)
MAX_TOKENS = 1000
PART_TOKENS = 800

ACTS = {
    "BNS": ("Bharatiya Nyaya Sanhita, 2023", "2025-10-06"),
    "BNSS": ("Bharatiya Nagarik Suraksha Sanhita, 2023", "2025-10-06"),
    "BSA": ("Bharatiya Sakshya Adhiniyam, 2023", "2025-10-06"),
    "CPC": ("Code of Civil Procedure, 1908", "2026-01-10"),
}
DEFINITIONS = {"BNS-S2", "BNSS-S2", "BSA-S2", "CPC-S2"}

LEAD = r"^(?:⟨[^⟩]*⟩\s*|\*|\[\s*)*"                      # footnote markers and amendment brackets
NUMBERED = re.compile(LEAD + r"\((\d+)([A-Z]?)\)")         # "(3)", "⟨2⟩[(4A)"
LETTERED = re.compile(LEAD + r"\(([a-z]{1,2})\)\s*[“\"]")  # "(g) “cognizable offence” means"
BLOCK = re.compile(LEAD + r"(?:Explanation|Exception|Illustrations?\b|Provided\b|\([a-z]{1,2}\)\s)")
# Preferred cut points: a new Explanation, Exception, Illustrations or proviso, not a lettered item.
STRONG_BLOCK = re.compile(LEAD + r"(?:Explanation|Exception|Illustrations?\b|Provided\b)")


def letter_after(s):
    """'a' -> 'b', 'z' -> 'za', 'za' -> 'zb': the order Indian statutes use for clauses."""
    return "za" if s == "z" else s[:-1] + chr(ord(s[-1]) + 1)


def sub_sections(lines):
    """Indexes where numbered sub-sections start, as (index, label). A label must be higher than the
    previous one (1 < 1A < 2), so an Explanation's own "(1)" inside a sub-section is not a split.
    The heading line counts as "(1)" when it ends "…—(1) …"."""
    out, last = [], (0, "")
    # "(1)" right after the heading's closing dash, on the first line or, when the heading wraps, a
    # later one ("164. Procedure … breach of" / "peace.—(1) Whenever …")
    for i, l in enumerate(lines[:3]):
        if i and NUMBERED.match(l):
            break
        if re.search(r"[—–]\s*(?:⟨[^⟩]*⟩\s*|\[\s*)*\(1\)", l):
            out, last = [(0, "1")], (1, "")
            break
    for i, l in enumerate(lines[1:], 1):
        m = NUMBERED.match(l)
        if m and (int(m.group(1)), m.group(2)) > last:
            last = (int(m.group(1)), m.group(2))
            out.append((i, m.group(1) + m.group(2)))
    return out


def definition_clauses(lines):
    """Clause starts in a definition section, as (index, label). Numbered clauses "(8) “act” …" (BNS,
    CPC) and lettered clauses "(g) “cognizable offence” …" (BNSS, BSA, inside sub-section (1)), then
    any later sub-section ("(2) Words and expressions …"). A lettered clause must be the next letter and
    open a quoted term, which keeps roman sub-items ("(i) in relation to any State") inside their clause
    even where "(i)" could be read as a letter. A numbered label must be the next number or a lettered
    insert (7 -> 7A -> 7B -> 8), so a list inside a clause doesn't split it."""
    out, last, nxt_let = [], (0, ""), "a"
    for i, l in enumerate(lines):
        m = LETTERED.match(l)
        if m and m.group(1) == nxt_let:
            out.append((i, m.group(1)))
            nxt_let = letter_after(nxt_let)
            last = max(last, (1, ""))   # lettered clauses sit inside sub-section (1), opened on the heading line
            continue
        m = NUMBERED.match(l)
        if m:
            key = (int(m.group(1)), m.group(2))
            if key > last and key[0] <= last[0] + 1:
                out.append((i, m.group(1) + m.group(2)))
                last = key
    return out


def parts(lines, budget=PART_TOKENS):
    """Cut lines into consecutive groups of about `budget` tokens.

    When the next line would take a group past the budget, the cut goes at the latest block start
    that leaves at least a third of the budget before it: an Explanation, Exception, Illustrations or
    proviso if there is one, else a lettered clause. So an "Illustrations." line stays with its
    illustrations. With no
    such block start, the group grows until it would pass MAX_TOKENS and is then cut at any line (the
    table in BNSS s.359(1)). A last group under a fifth of the budget is folded into the one before it
    when they fit together.
    """
    size = lambda ls: tok("\n".join(ls))
    groups, cur = [], []
    for l in lines:
        if cur and size(cur + [l]) > budget:
            if BLOCK.match(l) and size(cur) >= budget // 3:
                groups.append(cur)
                cur = []
            else:
                ok = [k for k in range(1, len(cur)) if BLOCK.match(cur[k]) and size(cur[:k]) >= budget // 3]
                cuts = [k for k in ok if STRONG_BLOCK.match(cur[k])] or ok
                if cuts:
                    groups.append(cur[:cuts[-1]])
                    cur = cur[cuts[-1]:]
                elif size(cur + [l]) > MAX_TOKENS:
                    groups.append(cur)
                    cur = []
        cur.append(l)
    if cur:
        groups.append(cur)
    if len(groups) > 1 and size(groups[-1]) < budget // 5 and size(groups[-2] + groups[-1]) <= MAX_TOKENS:
        groups[-2:] = [groups[-2] + groups[-1]]
    return groups


def ref_of(unit_id):
    """'BNSS-S187' -> 'Section 187'; 'CPC-OXXI-R46I@commercial' -> 'Order XXI, Rule 46I'."""
    m = re.match(r"^[A-Z]+-(?:O(.+)-R(\w+)|S(\w+))(@commercial)?$", unit_id)
    return f"Order {m.group(1)}, Rule {m.group(2)}" if m.group(1) else f"Section {m.group(3)}"


def make_chunk(act, unit, cid, label, lines, line_pages, part=None):
    title, as_on = ACTS[act]
    text = "\n".join(lines)
    marks = {(pg, k) for l, pg in zip(lines, line_pages)
             for mk in re.findall(r"⟨([^⟩]*)⟩", l) for k in re.findall(r"\d+", mk)}
    notes = [f for f in unit["footnotes"] if (f["page"], f["marker"]) in marks]
    ref = ref_of(unit["id"]) + (label or "")
    head = f"{ref}: {unit['heading']}" if unit["heading"] else ref
    if part:
        head += f" (part {part[0]} of {part[1]})"
    if unit["regime"] == "commercial":
        head += " (version for commercial disputes)"
    if unit["status"] == "repealed":
        head += " (repealed)"
    prefix = "\n".join(x for x in (title, unit["chapter"], head) if x)
    return {"id": cid, "unit_id": unit["id"], "act": act, "chapter": unit["chapter"], "heading": unit["heading"],
            "label": label, "part": part[0] if part else None, "status": unit["status"], "regime": unit["regime"],
            "applies_to_commercial": unit["applies_to_commercial"], "page": line_pages[0], "as_on": as_on,
            "footnotes": notes, "text": text, "embed_text": prefix + "\n\n" + text, "tokens": tok(text)}


def chunk_unit(act, unit):
    lines, pages = unit["text"].split("\n"), unit["line_pages"]
    base = unit["id"].replace("@commercial", "")
    sfx = "@commercial" if unit["id"].endswith("@commercial") else ""
    if unit["id"] in DEFINITIONS:
        # one chunk per clause; words before the first clause ("In this Sanhita, unless …") go with it
        starts = definition_clauses(lines)
        ends = [i for i, _ in starts[1:]] + [len(lines)]
        pieces = [(f"(1)({lab})" if lab.isalpha() else f"({lab})", 0 if k == 0 else i, end)
                  for k, ((i, lab), end) in enumerate(zip(starts, ends))]
    elif tok(unit["text"]) > MAX_TOKENS and len(sub_sections(lines)) >= 2:
        starts = sub_sections(lines)
        if starts[0][0] > 0:                          # text before "(1)" goes with the first sub-section
            starts[0] = (0, starts[0][1])
        pieces = [(f"({lab})", i, nxt) for (i, lab), nxt in zip(starts, [s[0] for s in starts[1:]] + [len(lines)])]
    else:
        pieces = [(None, 0, len(lines))]

    chunks = []
    for lab, a, b in pieces:
        seg_lines, seg_pages = lines[a:b], pages[a:b]
        cid = base + (lab or "") + sfx
        if tok("\n".join(seg_lines)) > MAX_TOKENS:
            done, groups = 0, parts(seg_lines)
            for n, grp in enumerate(groups, 1):
                chunks.append(make_chunk(act, unit, f"{cid}#{n}", lab, grp, seg_pages[done:done + len(grp)],
                                         part=(n, len(groups))))
                done += len(grp)
        else:
            chunks.append(make_chunk(act, unit, cid, lab, seg_lines, seg_pages))
    return chunks


if __name__ == "__main__":
    all_chunks, report = [], {}
    for act in sys.argv[1:]:
        units = json.load(open(f"txt/{act}.units.json", encoding="utf-8"))
        cs = [c for u in units for c in chunk_unit(act, u)]
        all_chunks += cs
        sizes = sorted(c["tokens"] for c in cs)
        report[act] = dict(units=len(units), chunks=len(cs),
                           split_units=sum(1 for u in units if len(chunk_unit(act, u)) > 1),
                           tokens=dict(p50=sizes[len(sizes) // 2], p90=sizes[int(.9 * len(sizes))], max=sizes[-1],
                                       over_1k=sum(s > MAX_TOKENS for s in sizes)))
    ids = collections.Counter(c["id"] for c in all_chunks)
    report["duplicate_ids"] = [k for k, v in ids.items() if v > 1]
    report["total_chunks"] = len(all_chunks)
    with open("txt/chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1))
