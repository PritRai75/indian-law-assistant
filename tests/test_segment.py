"""Tests for ingest/segment.py.

Each test builds a tiny fake Act from text lines: a contents list, the "WHEREAS" preamble, then
the body. The lines are copied from the real India Code CPC (as on 10 Jan 2026), so the
segmenter meets the same patterns it meets in the PDF. Every test here pins down a failure
that was seen in the real corpus.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
import segment as sg  # noqa: E402

PREAMBLE = "WHEREAS it is expedient to consolidate and amend the laws relating to the procedure."

CONTENTS = [
    "ARRANGEMENT OF SECTIONS",
    "114. Review.",
    "115. Revision.",
    "116. Part to apply only to certain High Courts.",
    "ORDER XV",
    "1. Parties or their pleaders to be examined.",
    "2. Judgment at first hearing.",
    "3. Where parties at issue.",
    "4. Failure to produce evidence.",
    "ORDER XVI",
    "1. List of witnesses and summons to witnesses.",
]


def rows_of(contents, body):
    """Contents + body lines as (text, page) rows, the shape load() returns."""
    return [(t, 1) for t in contents + body]


def labels(body, contents=CONTENTS, act="CPC"):
    """Map each body line to the label segment() gives it."""
    seg = sg.segment(act, rows_of(contents, body))
    return dict(zip(body, seg[len(contents):]))


@pytest.fixture
def units(tmp_path, monkeypatch):
    """Run units() on a fake Act (CPC by default). units() reads txt/<ACT>.notes.jsonl, so give
    each Act an empty one."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "txt").mkdir()
    for act in ("BNS", "BNSS", "BSA", "CPC"):
        (tmp_path / "txt" / f"{act}.notes.jsonl").write_text("", encoding="utf-8")

    def run(body, contents=CONTENTS, act="CPC"):
        rows = rows_of(contents, body)
        return {x["id"]: x for x in sg.units(act, rows, sg.segment(act, rows))}
    return run


# ---------------------------------------------------------------- state-amendment blocks

# s.115: two Orissa amendments in a row, then Uttar Pradesh. The second Orissa block does not
# repeat the state name, which is what fooled v0 (it ended a block at the first non-state-name
# line after a "[Vide …]" line).
S115_BODY = [
    PREAMBLE,
    "115. Revision.—(1) The High Court may call for the record of any case which has been decided",
    "made, or any order deciding an issue in the course of a suit or other proceeding.]",
    "STATE AMENDMENTS",
    "Orrisa",
    "Amendment of section 115.—In the Code of Civil Procedure, 1908 (5 of 1908) for section 115 the",
    "115. Revision.—The High Court, in cases arising out of original suits or other proceedings",
    "[Vide the Orissa Act 26 of 1991, s. 2]",
    "Amendment of section 115.—In the Code of Civil Procedure, 1908 (5 of 1908), for section 115, the",
    "Procedure ( Orissa Amendment) Act, 2010, may call for the record of any case",
    "[Vide the Orissa Act 14 of 2010, s. 2]",
    "Uttar Pradesh",
    "Amendment of section 115 of Act V of 1908.—In section 115 of the said Code—",
    "[Vide Uttar Pradesh Act 14 of 1970, s. 3]",
    "Amendment of section 115 of Act V of 1908.— In section 115 of the Code of Civil",
    "[Vide Uttar Pradesh Act 37 of 1972, s. 6]",
    "116. Part to apply only to certain High Courts.—This Part applies only to High Courts",
]


def test_consecutive_blocks_without_state_name_stay_state():
    lab = labels(S115_BODY)
    assert [t for t in S115_BODY[3:16] if lab[t] != "state_amendment"] == []


def test_central_text_around_block_stays_body():
    lab = labels(S115_BODY)
    for t in (S115_BODY[1], S115_BODY[2], S115_BODY[-1]):
        assert lab[t] == "body", t


def test_quoted_heading_inside_block_does_not_end_it():
    # Orissa's replacement text begins "115. Revision.—", but s.115 was already seen.
    assert labels(S115_BODY)[S115_BODY[6]] == "state_amendment"


def test_unit_s115_has_no_state_text(units):
    u = units(S115_BODY)
    assert "Vide" not in u["CPC-S115"]["text"]
    assert "Orissa" not in u["CPC-S115"]["text"]
    assert u["CPC-S116"]["text"].startswith("116. Part to apply")


# A state block inside an Order that quotes a rule the central Order does not have
# (the audit's "invented central O.XV r.5" case). The block ends at the next ORDER heading.
ORDER_BODY = [
    PREAMBLE,
    "ORDER XV",
    "4. Failure to produce evidence.—Where the summons has been issued for the final disposal",
    "STATE AMENDMENTS",
    "Uttar Pradesh",
    "Order XV.—In Order XV, after rule 4, the following rule shall be inserted, namely:—",
    "5. Striking off defence on failure to deposit admitted rent, etc.—(1) In any suit by a lessor",
    "[Vide Uttar Pradesh Act 57 of 1976, s. 9]",
    "Insertion of rule 5.—After rule 4 of Order XV the following rule shall be inserted",
    "[Vide Uttar Pradesh Act 31 of 1975, s. 2]",
    "ORDER XVI",
    "1. List of witnesses and summons to witnesses.—(1) On or before such date as the Court may",
]


def test_quoted_rule_not_in_contents_stays_state():
    lab = labels(ORDER_BODY)
    assert all(lab[t] == "state_amendment" for t in ORDER_BODY[3:10])


def test_next_order_heading_ends_block():
    lab = labels(ORDER_BODY)
    assert lab["ORDER XVI"] == "body"
    assert lab[ORDER_BODY[-1]] == "body"


def test_appendix_ends_block():
    body = [
        PREAMBLE,
        "ORDER XVI",
        "1. List of witnesses and summons to witnesses.—(1) On or before such date",
        "STATE AMENDMENTS",
        "Kerala",
        "Amendment of rule 1.—In rule 1 of Order XVI, in sub-rule (1)",
        "[Vide Kerala Act 13 of 1957, s. 3]",
        "APPENDIX A",
        "PLEADINGS",
    ]
    lab = labels(body)
    assert lab[body[5]] == "state_amendment"
    assert lab["APPENDIX A"] == "appendix"
    assert lab["PLEADINGS"] == "appendix"


def test_act_without_state_amendments_is_all_body():
    body = [
        "BE it enacted by Parliament in the Seventy-fourth Year of the Republic of India",
        "114. Review.—Subject as aforesaid, any person considering himself aggrieved",
        "115. Revision.—(1) The High Court may call for the record",
        "116. Part to apply only to certain High Courts.—This Part applies",
    ]
    assert set(labels(body, act="BNS").values()) == {"body"}


# ---------------------------------------------------------------- contents list

# The CPC prints a numbered LIST OF AMENDING ACTS before its arrangement of sections. Reading those
# numbers as sections sent the heading tracker to the wrong "35" and swallowed ss.35A-158.
AMENDING_LIST_CONTENTS = [
    "LIST OF AMENDING ACTS",
    "34. The Code of Civil Procedure (Amendment) Act, 1946 (11 of 1946).",
    "35. The Code of Civil Procedure (Amendment) Act, 1949 (32 of 1949).",
    "36. The Adaptation of Laws (No. 2) Order, 1950.",
    "ARRANGEMENT OF SECTIONS",
    "35. Costs.",
    "35A. Compensatory costs in respect of false or vexatious claims or defences.",
    "35B. Costs for causing delay.",
    "36. Application to orders.",
]


def test_contents_start_at_arrangement():
    rows = rows_of(AMENDING_LIST_CONTENTS, [PREAMBLE])
    toc = sg.toc_of("CPC", rows, ["contents"] * len(AMENDING_LIST_CONTENTS))
    assert toc[None] == ["35", "35A", "35B", "36"]


def test_amending_acts_list_does_not_derail_tracker():
    body = [
        PREAMBLE,
        "35. Costs.—(1) Subject to such conditions and limitations as may be prescribed",
        "(g) interest on costs from or until a certain date.]",
        "STATE AMENDMENTS",
        "Jammu and Kashmir and Ladakh (UTs).—",
        "In Section 35, in sub-section (1) omit “Commercial”.",
        "[Vide the Jammu and Kashmir Reorganisation (Adaptation of Central Laws) Order, 2020,",
        "⟨1⟩[35A. Compensatory costs in respect of false or vexatious claims or defences.—(1) If in",
        "⟨1⟩[35B. Costs for causing delay.—(1) If, on any date fixed for the hearing of a suit or f",
        "⟨2⟩[36. Application to orders.—The provisions of this Code relating to the execution of",
    ]
    lab = labels(body, contents=AMENDING_LIST_CONTENTS)
    assert lab[body[5]] == "state_amendment"
    assert [lab[t] for t in body[7:]] == ["body", "body", "body"]


# ---------------------------------------------------------------- Commercial Courts Act text

O19_CONTENTS = ["ARRANGEMENT OF SECTIONS", "ORDER XIX", "1. Power to order any point to be proved by affidavit.",
                "2. Power to order attendance of deponent for cross-examination.",
                "3. Matters to which affidavits shall be confined.", "ORDER XX", "1. Judgment when pronounced."]

# O.XIX: a J&K block inserts rules 4-6, then the central Commercial Courts rules 4-6 follow
# ("*[4." is not in the contents), then an Uttar Pradesh amendment resumes with no new heading.
O19_BODY = [
    PREAMBLE,
    "ORDER XIX",
    "3. Matters to which affidavits shall be confined.—(1) Affidavits shall be confined to such facts",
    "STATE AMENDMENT",
    "Jammu and Kashmir and Ladakh (UTs).—",
    "In Order XIX of the Code, after Rule 3, insert the following new rules, namely-",
    "4. Court may control evidence.—(1) The court may, by directions, regulate the evidence as to issues",
    "Central Laws) Order, 2020, Notification No. S.O. 3774(E), dated (23-10-2020)].",
    "*[4. Court may control evidence.—(1) The Court may, by directions, regulate the evidence as to",
    "relied upon.]",
    "Amendment of Order XIX.—In the Schedule, in Order XIX, in rule 1, for the existing proviso, the",
    "[Vide Uttar Pradesh Act 57 of 1976, s. 9]",
    "ORDER XX",
]


def test_starred_rule_ends_block():
    lab = labels(O19_BODY, contents=O19_CONTENTS)
    assert lab[O19_BODY[6]] == "state_amendment"
    assert lab[O19_BODY[8]] == "body"
    assert lab["relied upon.]"] == "body"


def test_amending_formula_restarts_block():
    lab = labels(O19_BODY, contents=O19_CONTENTS)
    assert lab[O19_BODY[10]] == "state_amendment"
    assert lab[O19_BODY[11]] == "state_amendment"
    assert lab["ORDER XX"] == "body"


def test_starred_order_not_in_contents_ends_block():
    body = [
        PREAMBLE,
        "ORDER XV",
        "4. Failure to produce evidence.—Where the summons has been issued for the final disposal",
        "STATE AMENDMENTS",
        "Uttar Pradesh",
        "[Vide Uttar Pradesh Act 31 of 1975, s. 2]",
        "*[ORDER XV-A",
        "CASE MANAGEMENT HEARING",
    ]
    lab = labels(body)
    assert lab["*[ORDER XV-A"] == "body"
    assert lab["CASE MANAGEMENT HEARING"] == "body"


# ---------------------------------------------------------------- headings units() must find

O18_CONTENTS = ["ARRANGEMENT OF SECTIONS", "ORDER XVIII", "4. Recording of evidence.",
                "5. How evidence shall be taken in appealable cases.", "6. When deposition to be interpreted.",
                "17. Court may recall and examine witness.",
                "17A. [Omitted.]", "18. Power of Court to inspect."]


def test_heading_with_two_footnote_markers(units):
    # "⟨1⟩[ ⟨2⟩5." was not recognised, so v0 glued O.XVIII r.5 onto r.4.
    body = [
        PREAMBLE,
        "ORDER XVIII",
        "4. Recording of evidence.—(1) In every case, the examination-in-chief of a witness shall be on",
        "⟨1⟩[ ⟨2⟩5. How evidence shall be taken in appealable cases.—In case in which an appeal is allowed,",
        "the Court in the presence of the Judge.]",
        "⟨3⟩6. When deposition to be interpreted.—Where the evidence is taken down in a language different",
    ]
    u = units(body, contents=O18_CONTENTS)
    assert "How evidence" not in u["CPC-OXVIII-R4"]["text"]
    assert u["CPC-OXVIII-R5"]["text"].endswith("presence of the Judge.]")
    assert u["CPC-OXVIII-R5"]["status"] == "in_force"


def test_repealed_stub_wrapped_onto_next_line(units):
    body = [
        PREAMBLE,
        "ORDER XVIII",
        "17. Court may recall and examine witness.—The Court may at any stage of a suit recall any witness",
        "17A. [Production of evidence not previously known or which could not be produced despite due diligence.]",
        "omitted by the Code of Civil Procedure (Amendment) Act, 1999 (46 of 1999), s. 27 (w.e.f. 1-7-2002).",
        "18. Power of Court to inspect.—The Court may at any stage of a suit inspect any property",
    ]
    u = units(body, contents=O18_CONTENTS)
    assert u["CPC-OXVIII-R17A"]["status"] == "repealed"
    assert "omitted by" not in u["CPC-OXVIII-R17"]["text"]


def test_repealed_stub_with_marker_before_bracket(units):
    contents = ["ARRANGEMENT OF SECTIONS", "ORDER XLI", "6. Stay of execution.", "7. [Repealed.]", "8. Exercise of powers."]
    body = [
        PREAMBLE,
        "ORDER XLI",
        "6. Stay of execution.—An appeal shall not operate as a stay of proceedings under a decree",
        "7. ⟨1⟩[No security to be required from the Government or a public officer in certain cases.] Rep. by the",
        "A.O. 1937.",
        "8. Exercise of powers in appeal from order made in execution of decree.—The powers conferred",
    ]
    u = units(body, contents=contents)
    assert u["CPC-OXLI-R7"]["status"] == "repealed"
    assert "A.O. 1937." in u["CPC-OXLI-R7"]["text"]


def test_hyphenated_rule_number(units):
    # The contents print "46I." and the body prints "46-I.".
    contents = ["ARRANGEMENT OF SECTIONS", "ORDER XXI", "46H. Appeals.", "46I. Application to negotiable instruments.",
                "47. Attachment of share in movables."]
    body = [
        PREAMBLE,
        "ORDER XXI",
        "46H. Appeals.—An order made under rule 46B, rule 46C or rule 46D shall be appealable as if",
        "46-I. Application to negotiable instruments.—The provisions of rules 46A to 46H (both inclusive)",
        "47. Attachment of share in movables.—Where the property to be attached consists of a share",
    ]
    u = units(body, contents=contents)
    assert u["CPC-OXXI-R46I"]["text"].startswith("46-I. Application")
    assert "negotiable" not in u["CPC-OXXI-R46H"]["text"]


def test_repealed_keyword_case_does_not_loosen_section_letters():
    # Only "Rep./Omitted/Repealed" is case-insensitive; a lower-case letter suffix is not a section.
    assert sg.REPEALED.match("17A. [Production of evidence.] omitted by Act 46 of 1999.")
    assert not sg.REPEALED.match("17a. [Production of evidence.] omitted by Act 46 of 1999.")


# ---------------------------------------------------------------- heading metadata (session 5)

BNS_PREAMBLE = "BE it enacted by Parliament in the Seventy-fourth Year of the Republic of India as follows:—"


def test_hyphen_inside_heading_does_not_close_it(units):
    # v1 closed a heading on any hyphen, so BNSS s.174 was stored as "Information as to non".
    contents = ["ARRANGEMENT OF SECTIONS", "173. Information in cognizable cases.",
                "174. Information as to non-cognizable cases and investigation of such cases.",
                "175. Police officer’s power to investigate cognizable case."]
    body = [
        BNS_PREAMBLE,
        "174. Information as to non-cognizable cases and investigation of such cases.—(1) When",
        "175. Police officer’s power to investigate cognizable case.—(1) Any officer in charge of a police",
    ]
    u = units(body, contents=contents, act="BNSS")
    assert u["BNSS-S174"]["heading"] == "Information as to non-cognizable cases and investigation of such cases"


def test_heading_that_starts_with_a_hyphenated_word(units):
    contents = ["ARRANGEMENT OF SECTIONS", "ORDER VIII", "6. Particulars of set-off to be given in written statement.",
                "6A. Counter-claim by defendant."]
    body = [
        PREAMBLE,
        "ORDER VIII",
        "6A. Counter-claim by defendant.—(1) A defendant in a suit may, in addition to his right of pleading",
    ]
    assert units(body, contents=contents)["CPC-OVIII-R6A"]["heading"] == "Counter-claim by defendant"


def test_wrapped_heading_joins_hyphenated_word_without_space(units):
    # BNS s.179's heading breaks the line inside "bank-notes".
    contents = ["ARRANGEMENT OF SECTIONS",
                "179. Using as genuine, forged or counterfeit coin, Government stamp, currency-notes or bank-notes.",
                "180. Possession of forged or counterfeit coin, Government stamp, currency-notes or bank-notes."]
    body = [
        BNS_PREAMBLE,
        "179. Using as genuine, forged or counterfeit coin, Government stamp, currency-notes or bank-",
        "notes.—Whoever imports or exports, or sells or delivers to, or buys or receives from, any other person, or",
    ]
    u = units(body, contents=contents, act="BNS")
    assert u["BNS-S179"]["heading"] == ("Using as genuine, forged or counterfeit coin, Government stamp, "
                                        "currency-notes or bank-notes")
    assert u["BNS-S179"]["text"].endswith("any other person, or")   # the unit text itself is untouched


def test_long_heading_is_not_cut(units):
    # BNS s.196 has the longest heading in the four Acts (167 characters). v1 cut every heading at 80.
    contents = ["ARRANGEMENT OF SECTIONS", "195. Assaulting or obstructing public servant when suppressing riot, etc.",
                "196. Promoting enmity between different groups on grounds of religion, race, place of birth,",
                "197. Imputations, assertions prejudicial to national integration."]
    body = [
        BNS_PREAMBLE,
        "196. Promoting enmity between different groups on grounds of religion, race, place of birth,",
        "residence, language, etc., and doing acts prejudicial to maintenance of harmony.—(1) Whoever—",
    ]
    h = units(body, contents=contents, act="BNS")["BNS-S196"]["heading"]
    assert h.endswith("doing acts prejudicial to maintenance of harmony")
    assert len(h) == 167


def test_date_hyphen_in_repealed_stub_is_not_a_heading_end(units):
    # v1 took "… (w.e.f. 1" as the end of the heading, at the hyphen in the date.
    contents = ["ARRANGEMENT OF SECTIONS", "47. Questions to be determined by the Court executing decree.",
                "48. [Repealed.]", "49. Transferee."]
    body = [
        PREAMBLE,
        "48. [Execution barred in certain cases.] Rep. by the Limitation Act, 1963 (36 of 1963), s. 28 (w.e.f. 1-1-1964).",
        "49. Transferee.—Every transferee of a decree shall hold the same subject to the equities",
    ]
    u = units(body, contents=contents)
    assert u["CPC-S48"]["heading"] == "Execution barred in certain cases"
    assert u["CPC-S48"]["status"] == "repealed"


def test_repealed_stub_with_bracket_before_number(units):
    # CPC O.V r.20A: the bracket opens before the number, so v1 marked it in force and its heading ran
    # into rule 21.
    contents = ["ARRANGEMENT OF SECTIONS", "ORDER V", "20. Substituted service.", "20A. [Repealed.]",
                "21. Service of summons where defendant resides within jurisdiction of another Court."]
    body = [
        PREAMBLE,
        "ORDER V",
        "⟨1⟩[20A. Service of summons by post.] Rep. by the Code of Civil Procedure (Amendment) Act, 1976",
        "(104 of 1976), s. 55 (w.e.f. 1-2-1977)].",
        "21. Service of summons where defendant resides within jurisdiction of another Court.—A",
    ]
    u = units(body, contents=contents)
    assert u["CPC-OV-R20A"]["status"] == "repealed"
    assert u["CPC-OV-R20A"]["heading"] == "Service of summons by post"
    assert u["CPC-OV-R21"]["status"] == "in_force"


def test_in_force_heading_never_matches_outer_stub_pattern():
    assert not sg.is_repealed("174. Information as to non-cognizable cases and investigation of such cases.—(1) When")
    assert not sg.is_repealed("⟨1⟩[35A. Compensatory costs in respect of false or vexatious claims or defences.—(1) If")


# ---------------------------------------------------------------- file encoding (session 5)

def test_every_text_open_names_utf8():
    # Windows opens files as cp1252 unless told otherwise, which crashes on the ⟨ ⟩ markers.
    # Every built-in open() in ingest/ must say encoding="utf-8" (binary modes excepted).
    import ast
    missing = []
    for path in sorted((Path(__file__).resolve().parents[1] / "ingest").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
                if "b" not in mode and not any(k.arg == "encoding" for k in node.keywords):
                    missing.append(f"{path.name}:{node.lineno}")
    assert missing == []


def test_dash_before_heading_typo_reads_whole_heading(units):
    # BNS s.255 is printed "255.—Public servant …" with a dash before the heading too. v1 kept the first
    # 80 characters of the line as the heading.
    contents = ["ARRANGEMENT OF SECTIONS", "254. Harbouring offender who has escaped from custody.",
                "255. Public servant disobeying direction of law with intent to save person from punishment or",
                "256. Public servant framing incorrect record or writing with intent to save person."]
    body = [
        BNS_PREAMBLE,
        "255.—Public servant disobeying direction of law with intent to save person from punishment or",
        "property from forfeiture.—Whoever, being a public servant, knowingly disobeys any direction of the law",
    ]
    u = units(body, contents=contents, act="BNS")
    assert u["BNS-S255"]["heading"] == ("Public servant disobeying direction of law with intent to save person "
                                        "from punishment or property from forfeiture")
