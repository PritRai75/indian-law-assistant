# Corpus pipeline v0 (Project 1, session 3, 30 Oct 2026). Run from a folder containing raw/BNS.pdf,
# raw/BNSS.pdf, raw/BSA.pdf, raw/CPC.pdf (the India Code PDFs) and an empty txt/ folder.
# Order: python extract.py BNS BNSS BSA CPC  ->  python segment.py BNS BNSS BSA CPC  ->  python schedule.py
# Needs: pip install pdfplumber. See corpus-audit.md for what each rule fixes.
"""Rebuild BNSS First Schedule Part I (classification of BNS offences) from word x-positions.
Columns: section | offence | punishment | cognizable | bailable | court."""
import json, re, collections
import pdfplumber

COLS = [("section", 0), ("offence", 100), ("punishment", 220), ("cognizable", 318),
        ("bailable", 382), ("court", 452)]


WINDOWS = {"offence": (100, 115), "punishment": (200, 236), "cognizable": (316, 336),
           "bailable": (380, 396), "court": (450, 464)}


def page_cols(words):
    """Left margin of each column on this page = modal word x0 inside that column's window."""
    edges = [("section", 0)]
    for name, (lo, hi) in WINDOWS.items():
        c = collections.Counter(round(w["x0"]) for w in words if lo <= w["x0"] <= hi)
        edges.append((name, (c.most_common(1)[0][0] if c else lo + 8) - 1.5))
    return edges


def col_of(x, edges):
    name = edges[0][0]
    for n, x0 in edges:
        if x >= x0:
            name = n
    return name


rows, cur, started, done = [], None, False, False
with pdfplumber.open("raw/BNSS.pdf") as pdf:
    for pno in range(174, len(pdf.pages)):
        p = pdf.pages[pno].filter(lambda o: o.get("object_type") != "char" or o["size"] < 20)
        text = p.extract_text() or ""
        if "CLASSIFICATION OF OFFENCES" in text:
            started = True
        if not started:
            continue
        words = [w for w in p.extract_words(keep_blank_chars=False, x_tolerance=1.5)
                 if w["top"] < p.height * 0.9]
        # drop the repeated column-number header row ("1 2 3 4 5 6")
        hdr_tops = {round(w["top"]) for w in words if w["text"] == "6" and w["x0"] > 480}
        words = [w for w in words if round(w["top"]) not in hdr_tops]
        joined = " ".join(w["text"] for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])))
        if "AGAINST OTHER LAWS" in joined:
            cut = min(w["top"] for w in words if "CLASSIFICATION" in w["text"] and w["top"] > 0)
            words = [w for w in words if w["top"] < cut - 3]
            done = True
        edges = page_cols(words)
        # row anchors: section-number words in column 1
        anchors = sorted(w["top"] for w in words if col_of(w["x0"], edges) == "section"
                         and re.fullmatch(r"\d+[A-Z]?(\(\d+\))*(\([a-z]+\))?", w["text"]))
        page_rows = []
        for a in anchors:
            cur = {k: [] for k, _ in COLS}; cur["page"] = pno + 1; cur["_top"] = a
            page_rows.append(cur); rows.append(cur)
        for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
            # a word belongs to the last row whose anchor is at or above it (+4pt for baseline offset);
            # words above the first anchor continue the previous page's last row
            owner = None
            for r in page_rows:
                if r["_top"] <= w["top"] + 4:
                    owner = r
            if owner is None:
                owner = rows[len(rows) - len(page_rows) - 1] if len(rows) > len(page_rows) else None
            if owner is None:
                continue
            owner[col_of(w["x0"], edges)].append(w["text"])
        if done:
            break

clean = []
for r in rows:
    d = {k: re.sub(r"\s+", " ", " ".join(r[k])).strip() for k, _ in COLS}
    for k in ("cognizable", "bailable"):
        d[k] = re.sub(r"(?<=\w) (?=\w)", "", d[k]) if len(d[k]) < 40 else d[k]  # "C o g nizable" -> "Cognizable"
    d["page"] = r["page"]
    clean.append(d)
json.dump(clean, open("txt/BNSS.schedule1.json", "w"), indent=1, ensure_ascii=False)
vals = lambda k: collections.Counter(x[k] for x in clean).most_common(8)
print("rows", len(clean))
print("cognizable", vals("cognizable"))
print("bailable", vals("bailable"))
print("court", vals("court"))
for x in clean:
    if x["section"].startswith(("103", "318(4)", "303(2)", "64(1)")):
        print(x)
