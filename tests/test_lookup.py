"""The labelling lookup helper (eval/lookup.py)."""
import json

from eval import lookup as L


def test_every_index_provision_is_available_in_document_order():
    with open(L.CHUNKS_FILE, encoding="utf-8") as f:
        order = [p for c in json.load(f) for p in c["provision_ids"]]
    assert list(L.provisions()) == order and len(order) == 3401


def test_provision_restored_by_the_chunker_is_included():
    p = L.provisions()["1026.2(a)(15)(ii)(C)"]
    assert p["status"] == "in_force" and "Pub. L. 119-10" in p["status_note"]
    assert "Note: Not printed in the eCFR" in L.show("1026.2(a)(15)(ii)(C)")


def test_references_in_every_form():
    cases = {
        "1005.6(b)": ["1005.6(b)"],
        "1005.6": ["1005.6"],
        "1005.6(B)(1)": ["1005.6(b)(1)"],
        "comment 13(c)(2)-1": ["1026.13(c)(2)-1"],
        "Reg E 11(c)": ["1005.11(c)"],
        "1026.12-1": ["1026.12-1"],
        "§ 1026.52(b)(1)(ii)(A)": ["1026.52(b)(1)(ii)(A)"],
    }
    for written, ids in cases.items():
        assert L.resolve(written) == (ids, []), written
    assert L.resolve("1026.18") == ([], ["1026.18"])
    assert L.resolve("nothing here") == ([], ["nothing here"])


def test_a_paragraph_covers_what_is_under_it_and_comments_rules_first():
    items = [p["id"] for p in L.covered("1005.6(b)")]
    assert items[0] == "1005.6(b)" and "1005.6(b)(1)" in items and "1005.6(b)(1)-3" in items
    kinds = [L.provisions()[i]["kind"] for i in items]
    assert kinds == sorted(kinds, key=lambda k: k != "rule")          # all rules, then all comments
    assert all(i == "1005.6(b)" or i.startswith(("1005.6(b)(", "1005.6(b)-")) for i in items)


def test_a_comment_reference_covers_only_itself():
    assert [p["id"] for p in L.covered("1005.6(b)(1)-3")] == ["1005.6(b)(1)-3"]


def test_show_gives_context_status_and_counts():
    out = L.show("1026.52(b)(1)(ii)")
    assert out.startswith("== 1026.52(b)(1)(ii)  (Regulation Z, § 1026.52 Limitations on fees): ")
    assert "  under § 1026.52(b)(1): (1) General rule." in out
    assert "§ 1026.52(b)(1)(ii)(A)  (VACATED, not law in force." in out
    assert "§ 1026.52(b)(1)(ii)(D)  (in force)" in out


def test_comment_context_is_the_rule_it_explains():
    out = L.show("1005.6(b)(1)-3")
    assert "  under § 1005.6(b): " in out and "  under § 1005.6(b)(1): " in out


def test_outline_is_one_line_per_provision():
    out = L.show("1005.6(b)", outline=True).splitlines()
    assert len(out) == 1 + len(L.covered("1005.6(b)"))
    assert out[1].startswith("  § 1005.6(b) ")


def test_find_ignores_case_and_line_breaks_and_filters_by_regulation():
    text = L.provisions()["1005.6(b)(1)-3"]["text"]
    words = " ".join(text.split()[:6]).upper()
    assert "comment 1005.6(b)(1)-3" in L.find(words)
    both = L.find("billing cycle")
    only_e = L.find("billing cycle", reg="E")
    assert int(both.split()[0]) > int(only_e.split()[0]) > 0
    assert "comment 1026." not in only_e and "§ 1026." not in only_e
    assert L.find("no such phrase anywhere") == "0 provision(s) contain 'no such phrase anywhere'."


def test_find_marks_provisions_not_in_force():
    assert "§ 1026.52(b)(1)(ii)(A) [VACATED]" in L.find("$32;")


def test_suggestions_for_missing_references():
    assert "outside the indexed subparts" in L.suggest("1026.18")
    assert "outside the indexed subparts" in L.suggest("§ 1005.33(a)")
    assert "did you mean" in L.suggest("1005.6(b)(9)")


def test_main_exit_codes(capsys):
    assert L.main(["1005.6(b)(1)"]) == 0
    assert "§ 1005.6(b)(1)  (in force)" in capsys.readouterr().out
    assert L.main(["1005.6(b)(1)", "1026.18"]) == 1
    assert L.main([]) == 2


def test_sections_lists_every_indexed_section_in_order(capsys):
    assert L.main(["--sections"]) == 0
    out = capsys.readouterr().out
    lines = [x.strip() for x in out.splitlines() if x.strip().startswith("§")]
    assert len(lines) == 20 + 28                                  # E 1-20; Z 1-16 and 51-62
    assert lines[0].startswith("§ 1005.1 ") and lines[-1].startswith("§ 1026.62 ")
    assert "§ 1005.10 " in out and out.index("§ 1005.9 ") < out.index("§ 1005.10 ")   # numeric order, not text order
    assert "§ 1026.17 " not in out                                # closed-end credit is not indexed


def test_sections_for_one_regulation(capsys):
    assert L.main(["--sections", "--reg", "Z"]) == 0
    out = capsys.readouterr().out
    assert "1005." not in out and "§ 1026.13 " in out
