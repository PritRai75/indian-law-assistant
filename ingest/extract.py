# Corpus pipeline v0 (Project 1, session 3, 30 Oct 2026). Run from a folder containing raw/BNS.pdf,
# raw/BNSS.pdf, raw/BSA.pdf, raw/CPC.pdf (the India Code PDFs) and an empty txt/ folder.
# Order: python extract.py BNS BNSS BSA CPC  ->  python segment.py BNS BNSS BSA CPC  ->  python schedule.py
# Needs: pip install pdfplumber. See corpus-audit.md for what each rule fixes.
"""Font-aware extraction of India Code PDFs.

Per page:
  - drop the diagonal "IndiaCode" watermark (chars > 20pt)
  - mark raised small chars (superscripts) as ⟨n⟩ so footnote refs never fuse
    with section numbers ("date1" -> "date⟨1⟩")
  - split body lines from the footnote block at the bottom of the page
Writes <ACT>.body.txt and <ACT>.notes.jsonl (page, text) plus a page map.
"""
import json, re, sys, statistics
import pdfplumber

FOOT_START = re.compile(r"^(\d+|\*+)\.?\s")
HEADING = re.compile(r"^(?:⟨[^⟩]*⟩)?\[?\d+[A-Z]{0,3}\.\s+\[?[A-Z][a-z][^.]{3,}\.?\]?\s*[—–]")


def is_super(c, base_bottom):
    return c["size"] <= 8.5 and c["bottom"] < base_bottom - 1.5


def line_text(chars):
    sizes = [c["size"] for c in chars if c["text"] != " "] or [11]
    big = [c for c in chars if c["size"] >= 9.5 and c["text"] != " "] or chars
    base = statistics.median(c["bottom"] for c in big)
    out, prev_x1, in_sup = [], None, False
    for c in sorted(chars, key=lambda c: c["x0"]):
        if c["text"] == " ":
            if in_sup:
                out.append("⟩"); in_sup = False
            out.append(" "); prev_x1 = c["x1"]; continue
        if prev_x1 is not None and c["x0"] - prev_x1 > c["size"] * 0.2 and not in_sup:
            out.append(" ")
        sup = is_super(c, base)
        if sup and not in_sup:
            out.append("⟨"); in_sup = True
        elif not sup and in_sup:
            out.append("⟩"); in_sup = False
            if c["x0"] - prev_x1 > c["size"] * 0.25:
                out.append(" ")
        out.append(c["text"])
        prev_x1 = c["x1"]
    if in_sup:
        out.append("⟩")
    s = re.sub(r" {2,}", " ", "".join(out)).strip()
    return s, statistics.median(sizes)


def run(act):
    body, notes = [], []
    with pdfplumber.open(f"raw/{act}.pdf") as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            p = page.filter(lambda o: o.get("object_type") != "char" or o["size"] < 20)
            lines = p.extract_text_lines(return_chars=True)
            allchars = p.chars
            h = page.height
            in_foot = False
            built = []
            for l in lines:
                # re-collect chars from the page so explicit space chars are kept
                ids = {id(c) for c in l["chars"]}
                xs0 = min(c["x0"] for c in l["chars"]); xs1 = max(c["x1"] for c in l["chars"])
                lc = [c for c in allchars if id(c) in ids or (
                    c["text"] == " " and l["top"] - 1 <= (c["top"] + c["bottom"]) / 2 <= l["bottom"] + 1
                    and xs0 <= c["x0"] <= xs1)]
                txt, med = line_text(lc)
                built.append((l, txt, med))
            # footnote numbers referenced on this page (superscripts, plus bare "*[" / "***" markers)
            markers = set()
            for _, txt, _ in built:
                for m in re.findall(r"⟨([^⟩]*)⟩", txt):
                    markers.update(re.findall(r"\d+|\*+", m))
                if re.search(r"\*\[|\*\*\*", txt):
                    markers.add("*")
            for l, txt, med in built:
                if not txt:
                    continue
                if re.fullmatch(r"\d{1,3}", txt) and l["top"] > h * 0.88:
                    continue  # page number
                # footnote block: small-type "n. ..." line whose n is a marker used on this page,
                # and which is not itself a provision heading ("59. Release on ground of illness.—")
                fm = FOOT_START.match(txt)
                if (not in_foot and med <= 9.4 and fm and l["top"] > h * 0.35
                        and (fm.group(1) in markers or fm.group(1)[0] == "*" and "*" in markers)
                        and not HEADING.match(txt)):
                    in_foot = True
                if in_foot and med <= 9.4:
                    if FOOT_START.match(txt) or not notes:
                        notes.append({"page": pno, "text": txt})
                    else:
                        notes[-1]["text"] += " " + txt
                    continue
                in_foot = False
                body.append(f"{txt}\t{pno}\t{med:.0f}")
    with open(f"txt/{act}.body.tsv", "w") as f:
        f.write("\n".join(body))
    with open(f"txt/{act}.notes.jsonl", "w") as f:
        for n in notes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    print(act, "body lines", len(body), "footnotes", len(notes))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        run(a)
