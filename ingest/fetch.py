"""Download Regulation E (12 CFR 1005) and Regulation Z (12 CFR 1026) as eCFR XML.

Run from the repo root:

    python -m ingest.fetch                     # uses the eCFR's latest issue date for Title 12
    python -m ingest.fetch --date 2026-10-30   # or a date you choose (YYYY-MM-DD)

What it does, in order:
1. Asks the eCFR which date its newest copy of Title 12 is from (titles.json),
   unless you pass --date. That date becomes the corpus "as of" date.
2. Downloads Part 1005 and Part 1026 for that one date, as XML.
3. Checks each file is well-formed XML and really contains the part asked for.
4. Saves them to corpus/raw/ byte for byte, and writes corpus/manifest.json with
   the URL, as-of date, download time, size and SHA-256 fingerprint of each file.

Only the Python standard library is used, so nothing new needs installing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

API = "https://www.ecfr.gov/api/versioner/v1"
TITLE = 12
PARTS = ["1005", "1026"]  # Regulation E, Regulation Z
REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "corpus" / "raw"
MANIFEST = REPO / "corpus" / "manifest.json"
USER_AGENT = "card-rules-portfolio-project/0.1 (github.com/PritRai75)"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def http_get(url: str, tries: int = 3, timeout: int = 120) -> bytes:
    """GET a URL and return the body, unzipped if the server gzipped it.

    The eCFR's full-XML endpoint refuses requests that don't accept compression
    (HTTP 406), so we always ask for gzip. Retries on rate limits, server errors
    and dropped connections, waiting 5 s then 10 s; other errors stop at once.
    """
    req = urllib.request.Request(
        url, headers={"Accept-Encoding": "gzip", "User-Agent": USER_AGENT}
    )
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                    body = gzip.decompress(body)
                return body
        except urllib.error.HTTPError as e:
            retryable = e.code == 429 or e.code >= 500
            detail = e.read()[:300].decode("utf-8", "replace")
            if not retryable or attempt == tries:
                raise SystemExit(f"HTTP {e.code} from {url}\n{detail}") from None
        except urllib.error.URLError as e:
            if attempt == tries:
                raise SystemExit(f"Could not reach {url}: {e.reason}") from None
        wait = 5 * attempt
        print(f"  retrying in {wait} s (attempt {attempt} of {tries} failed)")
        time.sleep(wait)
    raise AssertionError("unreachable")


def latest_issue_date() -> tuple[str, dict]:
    """Return Title 12's latest issue date and its full titles.json entry."""
    data = json.loads(http_get(f"{API}/titles.json"))
    for t in data.get("titles", []):
        if t.get("number") == TITLE:
            date = t.get("latest_issue_date")
            if not date or not DATE_RE.match(date):
                raise SystemExit(
                    f"titles.json has no usable latest_issue_date for Title {TITLE}: {t}\n"
                    "Re-run with --date YYYY-MM-DD."
                )
            return date, t
    raise SystemExit(f"Title {TITLE} not found in titles.json")


def part_url(date: str, part: str) -> str:
    return f"{API}/full/{date}/title-{TITLE}.xml?part={part}"


def check_part_xml(body: bytes, part: str) -> list[str]:
    """Fail if the body isn't XML; return warnings if it doesn't look like the part.

    The eCFR marks a part as an element with TYPE="PART" and N="<part number>".
    That layout is from the eCFR docs and hasn't been seen on a real download yet,
    so a mismatch is a warning for the audit, not a crash.
    """
    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        snippet = body[:300].decode("utf-8", "replace")
        raise SystemExit(f"Part {part}: response is not valid XML ({e}). Starts with:\n{snippet}")
    warnings = []
    if not any(el.get("TYPE") == "PART" and el.get("N") == part for el in root.iter()):
        warnings.append(f'no element with TYPE="PART" N="{part}" (root tag <{root.tag}>)')
    if b"Supplement I" not in body:
        warnings.append('the text "Supplement I" (official interpretations) does not appear')
    return warnings


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", help="as-of date YYYY-MM-DD (default: eCFR latest issue date)")
    ap.add_argument("--force", action="store_true", help="replace an existing corpus/manifest.json")
    args = ap.parse_args(argv)

    if MANIFEST.exists() and not args.force:
        raise SystemExit(
            f"{MANIFEST.relative_to(REPO).as_posix()} already exists. The corpus is pinned to "
            "its as-of date; re-run with --force only if you mean to change it."
        )

    title_info = None
    if args.date:
        if not DATE_RE.match(args.date):
            raise SystemExit("--date must look like 2026-10-30")
        as_of = args.date
    else:
        as_of, title_info = latest_issue_date()
    print(f"As-of date: {as_of}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files, all_warnings = [], []
    for part in PARTS:
        url = part_url(as_of, part)
        print(f"Downloading Part {part} ...")
        body = http_get(url)
        for w in check_part_xml(body, part):
            all_warnings.append(f"Part {part}: {w}")
        path = RAW_DIR / f"part-{part}.xml"
        path.write_bytes(body)  # bytes exactly as served, so the hash is reproducible
        files.append({
            "part": part,
            "regulation": {"1005": "E", "1026": "Z"}[part],
            "path": path.relative_to(REPO).as_posix(),
            "url": url,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        })
        print(f"  saved {files[-1]['path']} ({len(body):,} bytes)")
        time.sleep(1)  # be polite to a free public API

    manifest = {
        "source": "eCFR versioner API (www.ecfr.gov), US Government Publishing Office / Office of the Federal Register",
        "title": TITLE,
        "as_of": as_of,
        "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "title_info": title_info,  # titles.json entry for Title 12 (null when --date was given)
        "files": files,
        "warnings": all_warnings,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {MANIFEST.relative_to(REPO).as_posix()}")
    if all_warnings:
        print("\nWARNINGS (files were saved; the audit will look at these):")
        for w in all_warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main(sys.argv[1:])
