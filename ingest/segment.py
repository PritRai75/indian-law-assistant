# Corpus pipeline v0 (Project 1, session 3, 30 Oct 2026). Run from a folder containing raw/BNS.pdf,
# raw/BNSS.pdf, raw/BSA.pdf, raw/CPC.pdf (the India Code PDFs) and an empty txt/ folder.
# Order: python extract.py BNS BNSS BSA CPC  ->  python segment.py BNS BNSS BSA CPC  ->  python schedule.py
# Needs: pip install pdfplumber. See corpus-audit.md for what each rule fixes.
"""Segment extracted body text into: contents / body / schedule / appendix / state_amendment.
Then parse provision units and report token stats. Tokens are estimated as chars / 4
(no tokenizer reachable offline here; treat as +-15%)."""
import json, re, sys, collections


tok = lambda s: round(len(s) / 4)

STATES = {"Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
          "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
          "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Orissa", "Punjab",
          "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
          "West Bengal", "Delhi", "Chandigarh", "Pondicherry", "Puducherry", "Jammu and Kashmir",
          "Union territory of Jammu and Kashmir and Union territory of Ladakh", "Ladakh",
          "Lakshadweep", "Dadra and Nagar Haveli", "Andaman and Nicobar Islands", "Madras", "Bombay",
          "Mysore", "Uttaranchal"}
MARK = r"(?:⟨[^⟩]*⟩|\*)?\[?"
SEC = re.compile(rf"^{MARK}(\d+[A-Z]{{0,3}})\.\s*(?:\[)?([A-Z“\"‘'(][^—–]*?)\s*\.?\s*[—–-]{{1,2}}")
ORDER = re.compile(rf"^{MARK}\s*[“\"]?ORDER\s+([IVXL]+(?:\s*-?\s*[A-Z])?)\b")
norm_order = lambda o: re.sub(r"\s*-?\s*([A-Z])$", r"-\1", o) if re.search(r"[IVXL]\s*-?\s*[A-HJ-UWYZ]$", o) else o


def load(act):
    rows = [l.split("\t") for l in open(f"txt/{act}.body.tsv").read().split("\n")]
    return [(r[0], int(r[1])) for r in rows]


def segment(act, rows):
    text = [r[0] for r in rows]
    # body starts at the enacting formula (after the arrangement of sections)
    start = next(i for i, t in enumerate(text) if t.startswith("BE it enacted") or t.startswith("WHEREAS"))
    seg = ["contents"] * start
    mode, in_state, after_vide = "body", False, False
    for i in range(start, len(text)):
        t = text[i]
        if re.match(r"^(⟨[^⟩]*⟩)?\[?THE (FIRST|SECOND) SCHEDULE", t) and act in ("BNSS", "BSA"):
            mode = "schedule"
        if act == "CPC" and re.match(r"^(⟨[^⟩]*⟩|\*)?\[?\s*APPENDIX", t) and i > start:
            mode = "appendix"
        if act == "CPC" and t.startswith("THE SECOND SCHEDULE"):
            mode = "schedule"
        if t.startswith("STATEMENT OF OBJECTS AND REASONS"):
            mode = "objects_and_reasons"
        if re.match(r"^STATE AMENDMENTS?\b", t):
            in_state, after_vide = True, False
        elif in_state and after_vide:
            if t.strip(" :.") in STATES or t.startswith("[Vide") or t.startswith("Vide"):
                pass
            else:
                in_state = False
        if in_state:
            after_vide = t.startswith("[Vide") or t.rstrip().endswith("]") and "Vide" in t
        seg.append("state_amendment" if in_state else mode)
    return seg


def toc_of(act, rows, seg):
    """Section/rule numbers listed in the Act's own arrangement, keyed by Order (None = sections)."""
    toc, order = collections.defaultdict(list), None
    for (t, _), s in zip(rows, seg):
        if s != "contents":
            continue
        m = ORDER.match(t)
        if act == "CPC" and m:
            order = norm_order(m.group(1)); continue
        m = re.match(r"^(?:⟨[^⟩]*⟩|\*)?\[?(\d+[A-Z]{0,3})\.\s", t)
        if m:
            toc[order].append(m.group(1))
    return toc


REPEALED = re.compile(r"(?i)" + rf"^{MARK}(\d+[A-Z]{{0,3}})\.\s*\[?\s*\[([^\]]+)\]\.?\]?\s*\.?\s*(?:Rep\.|Omitted|Repealed|\*)")
SEC_DASH_FIRST = re.compile(rf"^{MARK}(\d+[A-Z]{{0,3}})\.\s*[—–]+\s*([A-Z].{{0,80}})")  # e.g. BNS "255.—Public servant..."


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
        m = SEC.match(t) or REPEALED.match(t) or SEC_DASH_FIRST.match(t)
        if not m and re.match(rf"^{MARK}\d+[A-Z]{{0,3}}\.\s+[A-Z“\"‘'(]", t):
            # heading wrapped onto the next line(s) before its dash
            joined = t
            for j in (1, 2):
                if i + j < len(rows):
                    joined += " " + rows[i + j][0]
                    m = SEC.match(joined)
                    if m:
                        break
        regime = "general"
        if m:
            num = m.group(1)
            starred = t.startswith("*[")
            kind = star_kind(pg) if starred else order_regime
            regime = "commercial" if kind == "commercial" else "general"
            okey = order
            if regime == "commercial" and (order, num) in seen:
                okey = (order or "") + "@commercial"   # a second, commercial-dispute version
            allowed = toc.get(order)
            if (okey, num) in seen or (allowed and num not in allowed and regime == "general"):
                m = None
        if m:
            seen.add((okey, num))
            suffix = "@commercial" if okey != order else ""
            uid = (f"{act}-O{order}-R{num}" if order else f"{act}-S{num}") + suffix
            cur = {"id": uid, "page": pg, "heading": m.group(2)[:80], "regime": regime,
                   "applies_to_commercial": kind != "omitted_for_commercial",
                   "status": "repealed" if REPEALED.match(t) else "in_force", "text": t}
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
