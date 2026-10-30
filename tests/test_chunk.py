"""Tests for ingest/chunk.py. Lines are copied from the real India Code text (BNS, BNSS, BSA as on
6 Oct 2025; CPC as on 10 Jan 2026)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
import chunk as ck  # noqa: E402


def unit(uid, lines, heading="Heading", chapter="Chapter I: Preliminary", footnotes=(), regime="general",
         status="in_force", pages=None):
    return {"id": uid, "heading": heading, "chapter": chapter, "regime": regime, "status": status,
            "applies_to_commercial": True, "text": "\n".join(lines), "line_pages": pages or [1] * len(lines),
            "footnotes": list(footnotes)}


# ---------------------------------------------------------------- sub-sections

S164 = [
    "164. Procedure where dispute concerning land or water is likely to cause breach of",
    "peace.—(1) Whenever an Executive Magistrate is satisfied from a report of a police officer or upon other",
    "information that a dispute likely to cause a breach of the peace exists concerning any land or water",
    "(2) For the purposes of this section, the expression “land or water” includes buildings, markets,",
    "(3) A copy of the order shall be served in the manner provided by this Sanhita for the service",
]


def test_wrapped_heading_still_counts_as_subsection_1():
    # BNSS s.164: "(1)" follows the heading's dash on its second line. v0 of the chunker labelled
    # the first chunk "(2)".
    assert ck.sub_sections(S164) == [(0, "1"), (3, "2"), (4, "3")]


def test_explanation_numbering_does_not_split():
    lines = ["60. Property liable to attachment and sale in execution of decree.—(1) The following property",
             "(a) lands, houses or other buildings,",
             "Explanation I.—The particulars mentioned in clauses (g), (h), (i), (j), (l) and (o) are exempt",
             "(1) is the first item of a list inside the Explanation",
             "(1A) Notwithstanding anything contained in clause (c) of the proviso to sub-section (1),",
             "(2) Nothing in this section shall be deemed to exempt houses and other buildings"]
    assert [lab for _, lab in ck.sub_sections(lines)] == ["1", "1A", "2"]


def test_split_unit_ids_and_coverage():
    filler = "word " * 400                            # each sub-section stays under MAX_TOKENS, the unit doesn't
    lines = S164[:3] + [filler] + S164[3:4] + [filler] + S164[4:]
    chunks = ck.chunk_unit("BNSS", unit("BNSS-S164", lines))
    assert [c["id"] for c in chunks] == ["BNSS-S164(1)", "BNSS-S164(2)", "BNSS-S164(3)"]
    assert "\n".join(c["text"] for c in chunks) == "\n".join(lines)
    assert chunks[1]["embed_text"].startswith("Bharatiya Nagarik Suraksha Sanhita, 2023\nChapter I: Preliminary\n"
                                              "Section 164(2): Heading")


def test_small_unit_is_one_chunk():
    chunks = ck.chunk_unit("BNSS", unit("BNSS-S164", S164))
    assert [c["id"] for c in chunks] == ["BNSS-S164"]


# ---------------------------------------------------------------- definition sections

BNSS_S2 = [
    "2. Definitions.—(1) In this Sanhita, unless the context otherwise requires,—",
    "(h) “complaint” means any allegation made orally or in writing to a Magistrate,",
    "(i) “electronic communication” means the communication of any written, verbal, pictorial information",
    "(j) “High Court” means,—",
    "(i) in relation to any State, the High Court for that State;",
    "(ii) in relation to a Union territory to which the jurisdiction of the High Court for a State",
    "(k) “inquiry” means every inquiry, other than a trial, conducted under this Sanhita",
    "(2) Words and expressions used herein and not defined but defined in the Information Technology Act,",
]


def test_lettered_clauses_keep_roman_items_inside():
    # Start the letters at (h) for the test: the real section begins at (a).
    # (i) after (h) is the clause; (i) after (j) is a roman sub-item of clause (j)
    seq = ["(a) “a” means a"] + ["(%s) “%s” means x" % (c, c) for c in "bcdefg"] + BNSS_S2[1:]
    labs = [lab for _, lab in ck.definition_clauses(seq)]
    assert labs == list("abcdefghijk") + ["2"]


def test_numbered_definitions_with_inserted_clauses():
    lines = ["2. Definitions.—In this Act, unless there is anything repugnant in the subject or context,—",
             "(6) “foreign judgment” means the judgment of a foreign Court;",
             "(7) “Government Pleader” includes any officer appointed by the State Government",
             "(7A) “High Court”, in relation to the Andaman and Nicobar Islands, means the High Court",
             "(7B) “India”, except in sections 1, 29, 43, 44, ⟨1⟩[44A], 78, 79, 82, 83 and 87A, means",
             "(8) “Judge” means the presiding officer of a Civil Court;"]
    seq = ["(%d) “x” means y" % n for n in range(1, 6)] + lines[1:]
    assert [lab for _, lab in ck.definition_clauses(seq)] == ["1", "2", "3", "4", "5", "6", "7", "7A", "7B", "8"]


def test_definition_chunk_ids():
    chunks = ck.chunk_unit("BNSS", unit("BNSS-S2", BNSS_S2[:1] + ["(a) “audio-video electronic” means x"] + BNSS_S2[-1:]))
    assert [c["id"] for c in chunks] == ["BNSS-S2(1)(a)", "BNSS-S2(2)"]
    assert chunks[0]["text"].startswith("2. Definitions.")     # the opening words stay with clause (a)


# ---------------------------------------------------------------- parts

def test_parts_cut_at_block_starts_and_fold_tiny_tail():
    para = "A kills Z by a blow. " * 120                        # about 600 tokens
    lines = [para, "Illustrations.", para, "Explanation.—short"]
    groups = ck.parts(lines)
    assert groups[0] == [para]                               # cut before "Illustrations."
    assert groups[-1][-1] == "Explanation.—short"            # tiny tail folded into the last part
    assert sum(len(g) for g in groups) == len(lines)


def test_parts_hard_cut_without_block_starts():
    rows = ["318(4) | Cheating and dishonestly inducing delivery of property. | The person deceived."] * 80
    groups = ck.parts(rows)
    assert all(ck.tok("\n".join(g)) <= ck.MAX_TOKENS for g in groups) and len(groups) >= 2


def test_part_ids_and_headings():
    para = "Except in the cases hereinafter excepted, culpable homicide is murder. " * 30   # about 540 tokens
    chunks = ck.chunk_unit("BNS", unit("BNS-S101", ["101. Murder.—" + para, "Illustrations.", para], heading="Murder"))
    assert [c["id"] for c in chunks] == ["BNS-S101#1", "BNS-S101#2"]
    assert "Section 101: Murder (part 2 of 2)" in chunks[1]["embed_text"]


# ---------------------------------------------------------------- metadata

def test_footnotes_follow_their_markers():
    notes = [{"marker": "1", "page": 93, "text": "1. Subs. by Act 46 of 1999."},
             {"marker": "2", "page": 94, "text": "2. Ins. by Act 22 of 2002."}]
    u = unit("CPC-OV-R9", ["9. Delivery of summons by Court.—⟨1⟩[(1) Where the defendant resides", "(2) The service",
                           "⟨2⟩[(3) The services of summons may be made"], footnotes=notes, pages=[93, 93, 94])
    c = ck.make_chunk("CPC", u, "CPC-OV-R9(3)", "(3)", ["⟨2⟩[(3) The services of summons may be made"], [94])
    assert c["footnotes"] == [notes[1]]


def test_references_and_flags():
    assert ck.ref_of("BNSS-S187") == "Section 187"
    assert ck.ref_of("CPC-OXXI-R46I@commercial") == "Order XXI, Rule 46I"
    u = unit("CPC-OV-R20A", ["⟨1⟩[20A. Service of summons by post.] Rep. by the Code of Civil Procedure"],
             heading="Service of summons by post", status="repealed", chapter="Order V: Issue and Service of Summons")
    c = ck.chunk_unit("CPC", u)[0]
    assert c["id"] == "CPC-OV-R20A" and c["as_on"] == "2026-01-10"
    assert "Order V, Rule 20A: Service of summons by post (repealed)" in c["embed_text"]
    s35 = ck.chunk_unit("CPC", unit("CPC-S35@commercial", ["*[35. Costs.— (1) In relation to any Commercial dispute"],
                                    heading="Costs", regime="commercial"))[0]
    assert s35["id"] == "CPC-S35@commercial" and "(version for commercial disputes)" in s35["embed_text"]
