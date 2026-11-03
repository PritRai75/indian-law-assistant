"""Tests for ingest/compare.py that need no network: the diff and the date filter."""

from ingest.compare import change_dates, diff_provisions, word_diff


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
