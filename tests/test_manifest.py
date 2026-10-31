"""Tests for corpus/manifest.json: the record of which India Code PDFs the corpus was built from.

The manifest's fingerprints (sha256) and sizes must match the PDFs in corpus/raw/, and the "as on"
dates that chunk.py stamps on every chunk must match the manifest's. If India Code's file is ever
replaced, or a PDF is damaged on checkout, these tests fail instead of the corpus changing silently.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
sys.path.insert(0, str(ROOT / "ingest"))
import chunk as ck  # noqa: E402

DOCS = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))["documents"]


def test_one_entry_per_act():
    assert sorted(d["act"] for d in DOCS) == sorted(ck.ACTS)


def test_fields_are_filled_in():
    for d in DOCS:
        assert re.fullmatch(r"[0-9a-f]{64}", d["sha256"]), d["act"]
        for key in ("as_on", "downloaded"):
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", d[key]), (d["act"], key)
        assert re.match(r"https://(www\.)?indiacode\.(gov|nic)\.in/", d["pdf_url"]), d["act"]
        assert d["file"] == f"raw/{d['act']}.pdf", d["act"]


def test_pdfs_match_their_fingerprints():
    for d in DOCS:
        data = (CORPUS / d["file"]).read_bytes()   # bytes, exactly as stored: no text decoding
        assert len(data) == d["bytes"], d["act"]
        assert hashlib.sha256(data).hexdigest() == d["sha256"], d["act"]


def test_chunk_dates_match_manifest():
    for d in DOCS:
        assert ck.ACTS[d["act"]][1] == d["as_on"], d["act"]
