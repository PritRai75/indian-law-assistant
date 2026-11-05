"""Tests for eval/calibrate.py (the blind labelling sheet) and the judge's blind guard. No AWS call:
two made-up runs over the real dev questions, with made-up answers."""
import json
import tomllib

import pytest

from eval import calibrate as C
from eval import golden as G
from eval import judge as J
from eval import run as R

DEV = [i for i in G.load() if i.split == "dev"]
CHUNKS = J.chunk_lookup()
SOME = list(CHUNKS)[:2]
RUNS = ["r20990101T000000Z", "r20990101T000100Z"]
FALLBACK = {"q020": "no_valid_citation", "q040": "output_invalid"}      # in the weak run only


def answer(qid, run):
    flags = [FALLBACK[qid]] if run == RUNS[1] and qid in FALLBACK else []
    return {"answer": "A made-up answer.", "abstained": False, "retrieved_ids": SOME,
            "tool_calls": [], "flags": flags}


@pytest.fixture
def db(tmp_path):
    d = J.connect(tmp_path / "runs.db")
    for run in RUNS:
        d.execute("INSERT INTO runs (run_id, today_default) VALUES (?, ?)", (run, "2026-11-03"))
        for item in DEV:
            R.log_result(d, run, item, answer(item.id, run), None, {"should_abstain": item.should_abstain})
    R.log_result(d, RUNS[0], G.GoldenItem.model_validate({**DEV[0].model_dump(), "id": "q999"}), None,
                 "ThrottlingException", None)                     # failed to run: never on the sheet
    return d


@pytest.fixture
def files(tmp_path):
    return {"sheet": tmp_path / "scratch" / "sheet.html", "labels": tmp_path / "cal" / "labels.toml",
            "key": tmp_path / "cal" / "key.json"}


def test_targets_follow_the_dev_proportion():
    assert C.targets(2) == [(13, 7), (13, 7)]
    assert C.targets(2, 41) == [(14, 7), (13, 7)]


def test_pick_is_balanced_seeded_and_skips_fallbacks(db):
    pools = [C.candidates(db, r) for r in RUNS]
    assert [len(p) for p in pools] == [24, 22]                    # q999 failed; two fallbacks in the weak run
    chosen = C.pick(pools)
    assert len(chosen) == 40
    for run in RUNS:
        mine = [c for c in chosen if c[0] == run]
        assert len(mine) == 20
        assert sum(not c[1].should_abstain for c in mine) == 13
    assert not any(c[0] == RUNS[1] and c[1].id in FALLBACK for c in chosen)
    assert len({(c[0], c[1].id) for c in chosen}) == 40             # no answer twice
    assert [(c[0], c[1].id) for c in C.pick(pools)] == [(c[0], c[1].id) for c in chosen]   # seeded
    assert [(c[0], c[1].id) for c in C.pick(pools, seed=1)] != [(c[0], c[1].id) for c in chosen]
    order = [c[0] for c in chosen]
    assert order != sorted(order)                                  # runs are mixed, not in blocks


def test_pick_fills_a_short_group_from_the_rest(db):
    pools = [C.candidates(db, r) for r in RUNS]
    few_declines = [c for c in pools[1] if not c[1].should_abstain] + \
                   [c for c in pools[1] if c[1].should_abstain][:3]
    assert len(few_declines) == 18                                 # 15 answerable (q020 fell back) + 3 declines
    chosen = C.pick([pools[0], few_declines])
    weak = [c for c in chosen if c[0] == RUNS[1]]
    assert len(chosen) == 40 and len(weak) == 18                   # all it had: 15 answerable filled in for declines
    assert sum(c[1].should_abstain for c in weak) == 3
    assert sum(c[0] == RUNS[0] for c in chosen) == 22              # the other run made up the last 2


def test_sheet_is_blind_and_complete(db, files):
    k = C.write_sheet(db, RUNS, chunks=CHUNKS, **files)
    page = files["sheet"].read_text(encoding="utf-8")
    for run in RUNS:
        assert run not in page
    assert "weak" not in page.lower() and "verdict" not in page.lower()
    for cid, info in k["items"].items():
        assert f'id="{cid}"' in page
        assert f'"{info["qid"]}"' not in page and f">{info['qid']}<" not in page   # question IDs never shown
    first = next(i for i in DEV if i.id == k["items"]["c01"]["qid"])
    assert first.question.strip()[:60] in page
    assert J.strip_citations(first.key_facts[0])[:40] in page
    assert "Passage 1 · " + SOME[0] in page
    assert C.spread(k).startswith("40 items: 20 from r20990101T000000Z, 20 from r20990101T000100Z; 26 answerable")


def test_labels_template_parses_and_starts_empty(db, files):
    C.write_sheet(db, RUNS, chunks=CHUNKS, **files)
    data = tomllib.loads(files["labels"].read_text(encoding="utf-8"))
    assert [b["id"] for b in data["label"]] == [f"c{n:02d}" for n in range(1, 41)]
    assert C.check(files["labels"], files["key"]) == (0, 40, [])


def fill(files, edit=None):
    """Fill in every block with valid labels, then apply `edit` (cid -> replacement block text)."""
    k = json.loads(files["key"].read_text(encoding="utf-8"))["items"]
    blocks = []
    for cid, info in k.items():
        facts = ", ".join(["true"] * info["key_facts"])
        blocks.append((edit or {}).get(cid) or
                      f'[[label]]\nid = "{cid}"\nfaithfulness = "faithful"\nkey_facts = [{facts}]\nnote = ""\n')
    files["labels"].write_text("\n".join(blocks), encoding="utf-8")


def test_check_counts_and_finds_problems(db, files):
    C.write_sheet(db, RUNS, chunks=CHUNKS, **files)
    fill(files)
    assert C.check(files["labels"], files["key"]) == (40, 40, [])
    fill(files, {"c01": '[[label]]\nid = "c01"\nfaithfulness = "mostly"\nkey_facts = [true]\n',
                 "c02": '[[label]]\nid = "c02"\nfaithfulness = "faithful"\nkey_facts = [true, true, true, true]\n',
                 "c03": '[[label]]\nid = "c03"\nfaithfulness = "faithful"\nkey_facts = ["yes", true]\nscore = 1\n'})
    done, total, problems = C.check(files["labels"], files["key"])
    assert (done, total) == (37, 40)
    text = "\n".join(problems)
    assert "c01: faithfulness must be one of" in text
    assert "c02: key_facts has 4 value(s)" in text
    assert "c03: unexpected field(s) score" in text and "c03: key_facts must be a list of true/false" in text
    files["labels"].write_text('[[label]]\nid = "c01"\nfaithfulness = "faithful"\n', encoding="utf-8")
    assert "missing blocks: c02" in C.check(files["labels"], files["key"])[2][0]


def test_sheet_never_overwrites_labels(db, files):
    C.write_sheet(db, RUNS, chunks=CHUNKS, **files)
    with pytest.raises(SystemExit, match="already exists"):
        C.write_sheet(db, RUNS, chunks=CHUNKS, **files)
    with pytest.raises(SystemExit, match="two different runs"):
        C.write_sheet(db, [RUNS[0], RUNS[0]], chunks=CHUNKS, force=True, **files)


def test_sheet_refused_once_a_run_has_verdicts(db, tmp_path, capsys):
    db.execute("INSERT INTO judgements (run_id, qid, judge_version, status) VALUES (?, 'q001', 'j1', 'judged')",
               (RUNS[0],))
    db.commit()
    assert C.main(["--sheet", *RUNS, "--db", str(tmp_path / "runs.db")]) == 2
    assert "wouldn't be blind" in capsys.readouterr().out


def test_judge_waits_for_complete_committed_labels(db, files):
    C.write_sheet(db, RUNS, chunks=CHUNKS, **files)
    kw = {"labels": files["labels"], "key": files["key"]}
    assert C.blind_block("r20990102T000000Z", **kw, dirty=lambda _: False) is None     # not on the sheet
    assert "only 0 of 40 labels are done" in C.blind_block(RUNS[0], **kw, dirty=lambda _: False)
    fill(files)
    assert "commit eval/calibration/ first" in C.blind_block(RUNS[1], **kw, dirty=lambda _: True)
    assert C.blind_block(RUNS[1], **kw, dirty=lambda _: False) is None
    assert C.blind_block(RUNS[1], **kw, dirty=lambda _: None) is None                  # git can't tell: allowed


def test_judge_command_refuses_a_run_on_the_sheet(db, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(C, "LABELS_FILE", tmp_path / "cal" / "labels.toml")
    monkeypatch.setattr(C, "KEY_FILE", tmp_path / "cal" / "key.json")
    C.write_sheet(db, RUNS, chunks=CHUNKS, sheet=tmp_path / "s.html", labels=C.LABELS_FILE, key=C.KEY_FILE)
    monkeypatch.setattr(C.blind_block, "__defaults__", (C.LABELS_FILE, C.KEY_FILE, C.git_dirty))
    assert J.main([RUNS[0], "--db", str(tmp_path / "runs.db")]) == 2
    assert "Not judged:" in capsys.readouterr().out
