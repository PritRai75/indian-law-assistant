"""The committed corpus must match corpus/manifest.json exactly.

If any of these fail, a raw XML file was changed, re-downloaded or altered by git
line-ending conversion after the manifest was written.
"""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "corpus" / "manifest.json"


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST.exists(), "corpus/manifest.json is missing: run python -m ingest.fetch"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_top_level_fields(manifest):
    assert manifest["title"] == 12
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", manifest["as_of"])
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", manifest["downloaded_at"])


def test_both_parts_listed_once(manifest):
    assert sorted(f["part"] for f in manifest["files"]) == ["1005", "1026"]


def test_urls_use_the_as_of_date(manifest):
    for f in manifest["files"]:
        assert f["url"] == (
            f"https://www.ecfr.gov/api/versioner/v1/full/{manifest['as_of']}"
            f"/title-12.xml?part={f['part']}"
        )


@pytest.mark.parametrize("part", ["1005", "1026"])
def test_file_matches_size_and_hash(manifest, part):
    entry = next(f for f in manifest["files"] if f["part"] == part)
    body = (REPO / entry["path"]).read_bytes()
    assert len(body) == entry["bytes"], (
        f"{entry['path']} size changed. If git converted line endings, check .gitattributes"
    )
    assert hashlib.sha256(body).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("part", ["1005", "1026"])
def test_file_is_well_formed_xml(manifest, part):
    entry = next(f for f in manifest["files"] if f["part"] == part)
    ET.parse(REPO / entry["path"])  # raises if not well-formed
