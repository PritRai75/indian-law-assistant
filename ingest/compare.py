"""List which provisions of Regs E and Z changed between eCFR versions.

Run from the repo root:

    python -m ingest.compare                      # every change in scope since 2024-01-01
    python -m ingest.compare --since 2025-01-01   # a later start
    python -m ingest.compare --dates 2024-05-14,2025-10-01   # only these change dates
    python -m ingest.compare --pairs 2024-05-13:2024-05-20 --pack   # any two dates; zip the results

Downloads are kept in scratch/compare/ and reused, so re-running is quick.
Older eCFR versions are parsed leniently (markup slips become notes in the
report); the corpus itself is always parsed strictly by ingest/parse.py.

Why it exists: the eCFR prints amendments as of their effective date, even
when a court later vacates them or Congress overturns them, and nothing in
the XML says so. To mark such text in corpus/status.json, we need the exact
provisions each amendment touched. This script finds them by comparing the
eCFR's own text from the day before each change with the text on the day.
Later, the same comparison shows what changed when the corpus is refreshed.

What it does, in order:
1. Asks the eCFR's versions list which dates a section in scope (or a
   Supplement I) changed between --since and the corpus as-of date.
2. For each date D, downloads Parts 1005 and 1026 as of D minus one day and
   as of D into scratch/compare/ (git ignores scratch/; nothing in corpus/
   changes).
3. Parses both with ingest/parse.py and lists provisions added, removed and
   changed, with the changed words.
4. Prints a summary and writes the full list to scratch/compare/report.md,
   and the same with complete texts to scratch/compare/changes.json.

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from ingest.fetch import API, PARTS, TITLE, http_get, part_url
from ingest.parse import SCOPE, ParseError, parse_part

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "scratch" / "compare"
MANIFEST = REPO / "corpus" / "manifest.json"

# Used only if the eCFR's versions list can't be read: the effective dates of
# the two 2024 rules found in the session 8 audit.
KNOWN_DATES = ["2024-05-14", "2025-10-01"]


# ------------------------------------------------------------- change dates

def in_scope(part: str, entry: dict) -> bool:
    """True for an in-scope section or a Supplement I of this part."""
    ident = str(entry.get("identifier", ""))
    name = str(entry.get("name", ""))
    m = re.fullmatch(rf"{part}\.(\d+)", ident)
    if m:
        return int(m.group(1)) in SCOPE[part]
    return "Supplement I" in ident or "Supplement I" in name


def change_dates(versions: dict, part: str, since: str, until: str) -> dict[str, list[str]]:
    """From one versions-list response, map each change date to what changed."""
    entries = versions.get("content_versions")
    if not isinstance(entries, list):
        raise ValueError(f"no 'content_versions' list; keys are {sorted(versions)}")
    out: dict[str, list[str]] = {}
    for e in entries:
        d = e.get("amendment_date") or e.get("date")
        if not d:
            raise ValueError(f"entry without a date: {e}")
        if since <= d <= until and in_scope(part, e):
            out.setdefault(d, []).append(str(e.get("identifier")))
    return out


def fetch_change_dates(since: str, until: str) -> dict[str, list[str]]:
    dates: dict[str, list[str]] = {}
    for part in PARTS:
        url = f"{API}/versions/title-{TITLE}.json?part={part}"
        body = json.loads(http_get(url))
        for d, idents in change_dates(body, part, since, until).items():
            dates.setdefault(d, []).extend(idents)
        time.sleep(1)
    return dict(sorted(dates.items()))


# ------------------------------------------------------------------ diffing

def word_diff(a: str, b: str, limit: int = 400) -> str:
    """Words removed as [-...-] and added as {+...+}, with a little context."""
    aw, bw = a.split(), b.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=aw, b=bw, autojunk=False).get_opcodes():
        if op == "equal":
            seg = aw[i1:i2]
            out.append(" ".join(seg) if len(seg) <= 6 else " ".join(seg[:3] + ["..."] + seg[-3:]))
        else:
            if i2 > i1:
                out.append("[-" + " ".join(aw[i1:i2]) + "-]")
            if j2 > j1:
                out.append("{+" + " ".join(bw[j1:j2]) + "+}")
    s = " ".join(out)
    return s if len(s) <= limit else s[:limit] + " ..."


def diff_provisions(before: list[dict], after: list[dict]) -> dict:
    a = {r["id"]: r["text"] for r in before}
    b = {r["id"]: r["text"] for r in after}
    return {
        "added": sorted(set(b) - set(a), key=list(b).index),
        "removed": sorted(set(a) - set(b), key=list(a).index),
        "changed": [i for i in b if i in a and " ".join(a[i].split()) != " ".join(b[i].split())],
        "texts_before": a,
        "texts_after": b,
    }


def raw_paragraphs(part: str, path: Path) -> list[dict]:
    """Fallback when parsing fails: every <P> or heading in scope, keyed by position.

    Rule text is keyed "<section> #<n>" (the n-th paragraph of the section) and
    commentary "Supp I <section> #<n>", so a diff still shows which section and
    roughly where, with the paragraph's own label at the start of its text.
    """
    import xml.etree.ElementTree as ET
    from ingest.parse import SEC_HEAD, text_of

    root = ET.parse(path).getroot()
    out = []
    for d8 in root.iter("DIV8"):
        sec = d8.get("N")
        if int(sec.split(".")[1].split("-")[0]) not in SCOPE[part]:
            continue
        for n, el in enumerate((e for e in d8 if e.tag == "P"), 1):
            out.append({"id": f"{sec} #{n}", "text": text_of(el)})
    supp = next(d for d in root.iter("DIV9") if d.get("N", "").startswith("Supplement I"))
    sec, n = None, 0
    for el in supp:
        t = text_of(el)
        m = SEC_HEAD.match(t)
        if m:
            sec, n = int(m.group(2)), 0
        if sec in SCOPE[part] and t:
            n += 1
            out.append({"id": f"Supp I {part}.{sec} #{n}", "text": t})
    return out


def raw_diff(before: list[dict], after: list[dict]) -> dict:
    """Align paragraphs by text (not position), so one insertion doesn't shift everything."""
    a, b = [r["text"] for r in before], [r["text"] for r in after]
    res = {"added": [], "removed": [], "changed": [], "texts_before": {}, "texts_after": {}}
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        pairs = list(zip(range(i1, i2), range(j1, j2))) if op == "replace" else []
        for i, j in pairs:
            key = after[j]["id"]
            res["changed"].append(key)
            res["texts_before"][key], res["texts_after"][key] = a[i], b[j]
        for i in range(i1 + len(pairs), i2):
            res["removed"].append(before[i]["id"])
            res["texts_before"][before[i]["id"]] = a[i]
        for j in range(j1 + len(pairs), j2):
            res["added"].append(after[j]["id"])
            res["texts_after"][after[j]["id"]] = b[j]
    return res


def compare_files(part: str, path_before: Path, path_after: Path) -> dict:
    """Parse two downloads of one part and diff them.

    Older eCFR versions have markup slips the current one doesn't, so they are
    parsed leniently: problems become notes (listed in the report) instead of
    errors. Only if even that fails does it compare raw paragraphs.
    """
    notes: list[str] = []
    try:
        before = parse_part(part, path_before, notes=notes, strict=False)
        after = parse_part(part, path_after, notes=notes, strict=False)
    except ParseError as e:
        res = raw_diff(raw_paragraphs(part, path_before), raw_paragraphs(part, path_after))
        res["warning"] = f"parse error ({e}); compared raw paragraphs instead"
        return res
    res = diff_provisions(before, after)
    lenient = sorted(set(n for n in notes if n.startswith("LENIENT")))
    if lenient:
        res["warning"] = f"{len(lenient)} markup problem(s) handled leniently: " + " | ".join(lenient)
    return res


def parse_pairs(s: str) -> list[tuple[str, str]]:
    """'2024-05-13:2024-05-20,2025-09-30:2025-10-01' -> [(before, after), ...]"""
    pairs = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        a, sep, b = item.partition(":")
        for d in (a, b):
            date.fromisoformat(d)  # raises ValueError on a bad date
        if not sep or a >= b:
            raise ValueError(f"bad pair {item!r}: use BEFORE:AFTER with BEFORE earlier")
        pairs.append((a, b))
    if not pairs:
        raise ValueError("no pairs given")
    return pairs


def changes_json(results: list[tuple[str, str, str, dict]]) -> list[dict]:
    """Machine-readable version of the comparison, with full before/after texts."""
    out = []
    for before, after, part, res in results:
        out.append({
            "before": before, "after": after, "part": part,
            "warning": res.get("warning"),
            "added": {i: res["texts_after"][i] for i in res["added"]},
            "removed": {i: res["texts_before"][i] for i in res["removed"]},
            "changed": {i: {"before": res["texts_before"][i], "after": res["texts_after"][i]}
                        for i in res["changed"]},
        })
    return out


def pack(zip_path: Path, files: list[Path]) -> None:
    """Zip files (compressed) so they can be attached to a chat as one file."""
    import zipfile

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.write(f, f.name)


# ------------------------------------------------------------------- report

def report_lines(d: str, part: str, res: dict, full: bool) -> list[str]:
    lines = []
    if "error" in res:
        return [f"  Part {part}: {res['error']}"]
    n = len(res["added"]) + len(res["removed"]) + len(res["changed"])
    lines.append(f"  Part {part}: {len(res['added'])} added, {len(res['removed'])} removed, "
                 f"{len(res['changed'])} changed")
    if res.get("warning"):
        lines.append(f"    WARNING: {res['warning']}")
    if not n:
        return lines
    cap = None if full else 60
    for i in res["added"][:cap]:
        lines.append(f"    + {i}: {res['texts_after'][i][:150 if not full else None]}")
    for i in res["removed"][:cap]:
        lines.append(f"    - {i}: {res['texts_before'][i][:150 if not full else None]}")
    for i in res["changed"][:cap]:
        lines.append(f"    ~ {i}: {word_diff(res['texts_before'][i], res['texts_after'][i], 10**6 if full else 300)}")
    if not full and n > 3 * 60:
        lines.append("    (list cut; the full list is in the report file)")
    return lines


def show(p: Path) -> str:
    """A path relative to the repo when it is inside it, for printing."""
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default="2024-01-01", help="first day to look for changes (YYYY-MM-DD)")
    ap.add_argument("--dates", help="comma-separated change dates; skips the versions lookup")
    ap.add_argument("--pairs", help="comma-separated BEFORE:AFTER dates to compare directly, "
                                    "e.g. 2024-05-13:2024-05-20")
    ap.add_argument("--pack", action="store_true",
                    help="also zip the XML files, report and changes.json into scratch/compare/compare.zip")
    args = ap.parse_args(argv)

    until = json.loads(MANIFEST.read_text(encoding="utf-8"))["as_of"]
    WORK.mkdir(parents=True, exist_ok=True)

    if args.pairs:
        try:
            pairs = parse_pairs(args.pairs)
        except ValueError as e:
            raise SystemExit(f"--pairs: {e}")
        print("Comparing the pairs given:")
        for a, b in pairs:
            print(f"  {a} -> {b}")
    else:
        if args.dates:
            dates = {d.strip(): ["(given)"] for d in args.dates.split(",") if d.strip()}
        else:
            try:
                dates = fetch_change_dates(args.since, until)
            except (ValueError, SystemExit) as e:
                print(f"Could not read the eCFR versions list ({e}).")
                print(f"Falling back to the two known dates: {', '.join(KNOWN_DATES)}")
                dates = {d: ["(known date)"] for d in KNOWN_DATES}
        print(f"Changes in scope between {args.since} and the corpus date {until}:")
        for d, idents in dates.items():
            print(f"  {d}: {', '.join(sorted(set(idents)))}")
        pairs = [((date.fromisoformat(d) - timedelta(days=1)).isoformat(), d) for d in dates]

    summary, full, results, used = [], [], [], []
    for prev, d in pairs:
        summary.append(f"\n=== {prev} -> {d}")
        full.append(f"\n## {prev} -> {d}\n")
        for part in PARTS:
            paths = []
            for day in (prev, d):
                p = WORK / f"part-{part}-{day}.xml"
                if not p.exists():
                    print(f"Downloading Part {part} as of {day} ...")
                    p.write_bytes(http_get(part_url(day, part)))
                    time.sleep(1)
                paths.append(p)
                if p not in used:
                    used.append(p)
            res = compare_files(part, *paths)
            results.append((prev, d, part, res))
            summary += report_lines(d, part, res, full=False)
            full += report_lines(d, part, res, full=True)

    report = WORK / "report.md"
    report.write_text("# eCFR changes in scope\n" + "\n".join(full) + "\n", encoding="utf-8", newline="\n")
    changes = WORK / "changes.json"
    changes.write_text(json.dumps(changes_json(results), indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8", newline="\n")
    print("\n".join(summary))
    print(f"\nFull list: {show(report)} (open it in VS Code with: code {show(report)})")
    if args.pack:
        z = WORK / "compare.zip"
        pack(z, used + [report, changes])
        print(f"Packed {len(used)} XML files, the report and changes.json into "
              f"{show(z)} ({z.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1:])
