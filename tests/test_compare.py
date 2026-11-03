"""Tests for ingest/compare.py and lenient parsing. No network: older eCFR
versions are simulated by editing a copy of the committed XML."""

import json
import re
import zipfile

import pytest

from ingest import compare as C
from ingest import parse as P
from ingest.compare import change_dates, diff_provisions, parse_pairs, word_diff

RAW_1026 = P.RAW_DIR / "part-1026.xml"


def test_word_diff_marks_changes():
    assert word_diff("(A) $32;", "(A) $30;") == "(A) [-$32;-] {+$30;+}"


def test_diff_provisions():
    before = [{"id": "x(a)", "text": "same"}, {"id": "x(b)", "text": "old words"}, {"id": "x(c)", "text": "gone"}]
    after = [{"id": "x(a)", "text": "same"}, {"id": "x(b)", "text": "new words"}, {"id": "x(d)", "text": "new"}]
    d = diff_provisions(before, after)
    assert (d["added"], d["removed"], d["changed"]) == (["x(d)"], ["x(c)"], ["x(b)"])


def test_whitespace_only_is_not_a_change():
    d = diff_provisions([{"id": "x", "text": "a  b"}], [{"id": "x", "text": "a b"}])
    assert d["changed"] == []


def test_change_dates_filters_scope_and_window():
    versions = {"content_versions": [
        {"amendment_date": "2024-05-14", "identifier": "1026.52", "name": "§ 1026.52"},
        {"amendment_date": "2024-05-14", "identifier": "Supplement I to Part 1026", "name": "Supplement I"},
        {"amendment_date": "2024-05-14", "identifier": "1026.35", "name": "§ 1026.35"},  # out of scope
        {"amendment_date": "2023-12-31", "identifier": "1026.52", "name": "§ 1026.52"},  # too early
        {"amendment_date": "2027-01-01", "identifier": "1026.52", "name": "§ 1026.52"},  # after as-of
    ]}
    assert change_dates(versions, "1026", "2024-01-01", "2026-10-23") == {
        "2024-05-14": ["1026.52", "Supplement I to Part 1026"]
    }


def test_parse_pairs():
    assert parse_pairs("2024-05-13:2024-05-20, 2025-09-30:2025-10-01") == [
        ("2024-05-13", "2024-05-20"), ("2025-09-30", "2025-10-01")]
    for bad in ("2024-05-20:2024-05-13", "2024-05-13", "2024-13-01:2024-14-01", ""):
        with pytest.raises(ValueError):
            parse_pairs(bad)


# ------------------------------------------- an "older version" with known slips

@pytest.fixture(scope="module")
def old_version(tmp_path_factory):
    """The corpus Part 1026 with the two markup slips seen in 2024-2025 versions,
    plus one real wording change, so the expected diff is known exactly."""
    src = RAW_1026.read_text(encoding="utf-8")
    # 1. A pending-amendment link inside a section (seen in 1026.1, 1026.52, 1005.10).
    src, n1 = re.subn(r'(<DIV8 N="1026\.1" [^>]*>\s*<HEAD>[^<]*</HEAD>)',
                      r'\1<XREF ID="x" REFID="1">Link to an amendment published at 89 FR 64538, Aug. 7, 2024.</XREF>',
                      src, count=1)
    # 2. Comment 2(a)(20)-6 run into the end of the previous paragraph.
    src, n2 = re.subn(r"self-replenishing criterion\.</P>\s*<P>6\. <I>Verifications of collateral value",
                      "self-replenishing criterion.6. <I>Verifications of collateral value", src, count=1)
    # 3. A real change: the old safe harbor.
    src, n3 = re.subn(r"<P>\(A\) \$32;</P>", "<P>(A) $30;</P>", src, count=1)
    assert (n1, n2, n3) == (1, 1, 1)
    p = tmp_path_factory.mktemp("old") / "part-1026-old.xml"
    p.write_text(src, encoding="utf-8")
    return p


def test_old_version_fails_strict(old_version):
    with pytest.raises(P.ParseError):
        P.parse_part("1026", old_version)


def test_old_version_parses_leniently(old_version):
    notes = []
    recs = P.parse_part("1026", old_version, notes=notes, strict=False)
    assert any("pending amendment" in n and "1026.1" in n for n in notes)
    assert any(n.startswith("LENIENT") and "2(a)(20)-7" in n for n in notes)
    ids = {r["id"] for r in recs}
    assert "1026.2(a)(20)-6" not in ids and "1026.2(a)(20)-7" in ids


def test_compare_finds_exactly_the_planted_changes(old_version):
    res = C.compare_files("1026", old_version, RAW_1026)
    assert "LENIENT" in res["warning"]
    assert res["added"] == ["1026.2(a)(20)-6"]
    assert res["removed"] == []
    assert sorted(res["changed"]) == ["1026.2(a)(20)-5", "1026.52(b)(1)(ii)(A)"]
    assert word_diff(res["texts_before"]["1026.52(b)(1)(ii)(A)"],
                     res["texts_after"]["1026.52(b)(1)(ii)(A)"]) == "(A) [-$30;-] {+$32;+}"


def test_main_with_pairs_and_pack(tmp_path, monkeypatch, old_version):
    """End to end with files already downloaded: no network call may happen."""
    monkeypatch.setattr(C, "WORK", tmp_path)
    monkeypatch.setattr(C, "http_get", lambda url: pytest.fail(f"tried to download {url}"))
    for part in ("1005", "1026"):
        cur = (P.RAW_DIR / f"part-{part}.xml").read_bytes()
        old = old_version.read_bytes() if part == "1026" else cur
        (tmp_path / f"part-{part}-2024-05-13.xml").write_bytes(old)
        (tmp_path / f"part-{part}-2024-05-20.xml").write_bytes(cur)
    C.main(["--pairs", "2024-05-13:2024-05-20", "--pack"])
    changes = json.loads((tmp_path / "changes.json").read_text(encoding="utf-8"))
    z1026 = next(c for c in changes if c["part"] == "1026")
    assert z1026["changed"]["1026.52(b)(1)(ii)(A)"] == {"before": "(A) $30;", "after": "(A) $32;"}
    assert next(c for c in changes if c["part"] == "1005")["changed"] == {}
    with zipfile.ZipFile(tmp_path / "compare.zip") as z:
        assert sorted(z.namelist()) == sorted([
            "part-1005-2024-05-13.xml", "part-1005-2024-05-20.xml",
            "part-1026-2024-05-13.xml", "part-1026-2024-05-20.xml",
            "report.md", "changes.json"])
