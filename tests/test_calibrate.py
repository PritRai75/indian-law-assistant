"""Tests for eval/calibrate.py (the blind labelling sheet) and the judge's blind guard. No AWS call:
two made-up runs over the real dev questions, with made-up answers."""
import json
import tomllib
from dataclasses import asdict

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
NAMES = ["hybrid k5", "hybrid k2 no tool"]                                 # from each run's logged config


def answer(qid, run):
    flags = [FALLBACK[qid]] if run == RUNS[1] and qid in FALLBACK else []
    return {"answer": "A made-up answer.", "abstained": False, "retrieved_ids": SOME,
            "tool_calls": [], "flags": flags}


@pytest.fixture
def db(tmp_path):
    from app.config import DEFAULT
    d = J.connect(tmp_path / "runs.db")
    for run, config in zip(RUNS, [DEFAULT, DEFAULT.with_(k=2, tool_on=False)]):
        d.execute("INSERT INTO runs (run_id, today_default, config_json) VALUES (?, ?, ?)",
                  (run, "2026-11-03", json.dumps(asdict(config))))
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
    assert C.targets(2) == [(10, 5), (10, 5)]                      # set 2: 30 (Prit, session 20)
    assert C.targets(2, 40) == [(13, 7), (13, 7)]                  # set 1: 40
    assert C.targets(2, 41) == [(14, 7), (13, 7)]


def test_pick_is_balanced_seeded_and_skips_fallbacks(db):
    pools = [C.candidates(db, r) for r in RUNS]
    assert [len(p) for p in pools] == [24, 22]                    # q999 failed; two fallbacks in the weak run
    chosen = C.pick(pools, total=40)
    assert len(chosen) == 40
    for run in RUNS:
        mine = [c for c in chosen if c[0] == run]
        assert len(mine) == 20
        assert sum(not c[1].should_abstain for c in mine) == 13
    assert not any(c[0] == RUNS[1] and c[1].id in FALLBACK for c in chosen)
    assert len({(c[0], c[1].id) for c in chosen}) == 40             # no answer twice
    assert [(c[0], c[1].id) for c in C.pick(pools, total=40)] == [(c[0], c[1].id) for c in chosen]   # seeded
    assert [(c[0], c[1].id) for c in C.pick(pools, seed=1, total=40)] != [(c[0], c[1].id) for c in chosen]
    order = [c[0] for c in chosen]
    assert order != sorted(order)                                  # runs are mixed, not in blocks


def test_pick_fills_a_short_group_from_the_rest(db):
    pools = [C.candidates(db, r) for r in RUNS]
    few_declines = [c for c in pools[1] if not c[1].should_abstain] + \
                   [c for c in pools[1] if c[1].should_abstain][:3]
    assert len(few_declines) == 18                                 # 15 answerable (q020 fell back) + 3 declines
    chosen = C.pick([pools[0], few_declines], total=40)
    weak = [c for c in chosen if c[0] == RUNS[1]]
    assert len(chosen) == 40 and len(weak) == 18                   # all it had: 15 answerable filled in for declines
    assert sum(c[1].should_abstain for c in weak) == 3
    assert sum(c[0] == RUNS[0] for c in chosen) == 22              # the other run made up the last 2


def test_sheet_is_blind_and_complete(db, files):
    k = C.write_sheet(db, RUNS, chunks=CHUNKS, size=40, other_keys=[], **files)
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
    C.write_sheet(db, RUNS, chunks=CHUNKS, size=40, other_keys=[], **files)
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
    C.write_sheet(db, RUNS, chunks=CHUNKS, size=40, other_keys=[], **files)
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
    C.write_sheet(db, RUNS, chunks=CHUNKS, size=40, other_keys=[], **files)
    with pytest.raises(SystemExit, match="already exists"):
        C.write_sheet(db, RUNS, chunks=CHUNKS, size=40, other_keys=[], **files)
    with pytest.raises(SystemExit, match="two different runs"):
        C.write_sheet(db, [RUNS[0], RUNS[0]], chunks=CHUNKS, force=True, size=40, other_keys=[], **files)


def test_sheet_refused_once_a_run_has_verdicts(db, tmp_path, capsys):
    db.execute("INSERT INTO judgements (run_id, qid, judge_version, status) VALUES (?, 'q001', 'j1', 'judged')",
               (RUNS[0],))
    db.commit()
    assert C.main(["--sheet", *RUNS, "--db", str(tmp_path / "runs.db")]) == 2
    assert "wouldn't be blind" in capsys.readouterr().out


def test_judge_waits_for_complete_committed_labels(db, files):
    C.write_sheet(db, RUNS, chunks=CHUNKS, size=40, other_keys=[], **files)
    kw = {"labels": files["labels"], "key": files["key"]}
    assert C.blind_block("r20990102T000000Z", **kw, dirty=lambda _: False) is None     # not on the sheet
    assert "only 0 of 40 labels are done" in C.blind_block(RUNS[0], **kw, dirty=lambda _: False)
    fill(files)
    assert "first, so the labels are fixed" in C.blind_block(RUNS[1], **kw, dirty=lambda _: True)
    assert C.blind_block(RUNS[1], **kw, dirty=lambda _: False) is None
    assert C.blind_block(RUNS[1], **kw, dirty=lambda _: None) is None                  # git can't tell: allowed


def test_judge_command_refuses_a_run_on_the_sheet(db, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(C, "LABELS_FILE", tmp_path / "cal" / "labels.toml")
    monkeypatch.setattr(C, "KEY_FILE", tmp_path / "cal" / "key.json")
    C.write_sheet(db, RUNS, chunks=CHUNKS, size=40, other_keys=[], sheet=tmp_path / "s.html",
                  labels=C.LABELS_FILE, key=C.KEY_FILE)
    monkeypatch.setattr(C, "CAL_DIR", tmp_path / "cal")             # the judge looks at every set's key
    assert J.main([RUNS[0], "--db", str(tmp_path / "runs.db")]) == 2
    assert "Not judged:" in capsys.readouterr().out


# ---------------------------------------------------------------- part 2: --compare

SHORT = {"f": "faithful", "p": "partially_faithful", "u": "unfaithful"}


def test_wilson_matches_statsmodels():
    # proportion_confint(k, n, method="wilson") from statsmodels 0.15.0 (session 20)
    for k, n, want in [(37, 40, (0.8014, 0.9742)), (0, 10, (0.0, 0.2775)), (10, 10, (0.7225, 1.0)),
                       (5, 9, (0.2667, 0.8112))]:
        assert tuple(round(x, 4) for x in C.wilson(k, n)) == want


def test_kappa_matches_scikit_learn():
    # cohen_kappa_score from scikit-learn 1.9.1 (session 20)
    prit, judge = "f f f p p u f p f f".split(), "f f p p u u f f f u".split()
    assert C.kappa(list(zip(prit, judge))) == pytest.approx(0.34426229508196726)
    assert C.kappa(list(zip([1, 1, 0, 0, 1, 1, 1, 0], [1, 0, 0, 0, 1, 1, 1, 1]))) == pytest.approx(0.4666666666666667)
    assert C.kappa([("f", "f"), ("u", "u")]) == 1.0
    assert C.kappa([("f", "f")] * 5) is None                       # one label for both: undefined
    assert C.kappa([]) is None


def item(cid, qid, prit, judge, pk, jk, *, run="default", decline=False, persona="frontline"):
    return {"cid": cid, "qid": qid, "run_id": run, "run": run, "decline": decline, "persona": persona,
            "prit": SHORT[prit], "judge": SHORT[judge], "prit_kf": pk, "judge_kf": jk, "note": "",
            "rationale": "", "judge_model": "m"}


def test_bootstrap_resamples_whole_questions_and_is_seeded():
    items = [item(f"c{n}", f"q{n % 3}", "f", "f" if n % 4 else "u", [True], [True]) for n in range(12)]
    stat = lambda s: C.kappa(C.faith_pairs(s))                     # noqa: E731
    a, b = C.bootstrap(items, stat, reps=300), C.bootstrap(items, stat, reps=300)
    assert a == b                                                  # seeded
    seen = []
    C.bootstrap(items, lambda s: seen.append(sorted({it["qid"] for it in s})) or 0.0, reps=50)
    for qids in seen:                                              # a question comes with all its answers
        assert set(qids) <= {"q0", "q1", "q2"}
    sizes = set()
    C.bootstrap(items, lambda s: sizes.add(len(s)) or 0.0, reps=50)
    assert sizes == {12}                                           # every question here has 4 answers
    same = [item(f"c{n}", f"q{n}", "f", "f", [True], [True]) for n in range(5)]
    assert C.bootstrap(same, stat, reps=20) == {"lo": None, "hi": None, "undefined": 20, "reps": 20}


def test_compare_counts():
    items = [item("c01", "q001", "f", "f", [True, True], [True, True]),
             item("c02", "q001", "f", "p", [True], [False], run="weak"),
             item("c03", "q002", "p", "p", [False], [True], decline=True),
             item("c04", "q003", "u", "f", [True], [True], persona="compliance"),
             item("c05", "q004", "p", "u", [], [], run="weak", decline=True, persona="compliance")]
    c = C.compare(items, reps=200)
    f = c["faithfulness"]
    assert (c["n"], c["questions"], f["agree"]) == (5, 4, 2)
    assert f["confusion"]["faithful"] == {"faithful": 1, "partially_faithful": 1, "unfaithful": 0}
    assert f["confusion"]["unfaithful"]["faithful"] == 1
    assert f["judge_more_lenient"] == ["c04"] and f["judge_stricter"] == ["c02", "c05"]
    assert (f["prit_faithful"], f["judge_faithful"]) == (2, 2)
    assert c["faithful_or_not"]["agree"] == 3                     # c01, c03, c05 agree on faithful-or-not
    kf = c["key_facts"]
    assert (kf["n"], kf["agree"]) == (5, 3)
    assert kf["confusion"] == {"prit_true": {"judge_true": 3, "judge_false": 1},
                               "prit_false": {"judge_true": 1, "judge_false": 0}}
    assert c["slices"]["run: weak"]["n"] == 2 and c["slices"]["declines"]["agree"] == 1
    assert c["slices"]["compliance"]["facts"] == 1
    assert [d["cid"] for d in c["disagreements"]] == ["c02", "c03", "c04", "c05"]
    assert c["disagreements"][1]["facts"] == [{"n": 1, "prit": False, "judge": True}]
    text = C.compare_report(c, judge_version="j1", missing=[("c06", "no j1 verdict yet")])
    assert "agreement   2/5 = 40.0%" in text and "Not compared (1): c06 no j1 verdict yet" in text
    assert "c03 q002 default key fact 1 Prit false, judge true" in text


def verdict(facts, label="faithful"):
    return J.JudgeVerdict.model_validate({
        "claims": [{"claim": "x", "supported": label == "faithful", "source": None}], "faithfulness": label,
        "key_facts": [{"n": i, "covered": x} for i, x in enumerate(facts, 1)], "rationale": "r"}).model_dump_json()


def test_paired_items_joins_labels_and_verdicts(db, files):
    k = C.write_sheet(db, RUNS, chunks=CHUNKS, size=40, other_keys=[], **files)
    fill(files)
    items_ = k["items"]
    first, second, third = list(items_)[:3]
    for cid, label, status in [(first, "unfaithful", "judged"), (second, "faithful", "skipped")]:
        info = items_[cid]
        db.execute("INSERT INTO judgements (run_id, qid, judge_version, judge_model, status, skip_reason, verdict_json) "
                   "VALUES (?, ?, 'j1', 'm', ?, ?, ?)", (info["run_id"], info["qid"], status,
                   "no_valid_citation" if status == "skipped" else None, verdict([True] * info["key_facts"], label)))
    info = items_[third]
    db.execute("INSERT INTO judgements (run_id, qid, judge_version, judge_model, status, verdict_json) "
               "VALUES (?, ?, 'j1', 'm', 'judged', ?)", (info["run_id"], info["qid"], verdict([True] * (info["key_facts"] + 1))))
    db.commit()
    got, missing = C.paired_items(db, judge_version="j1", labels=files["labels"], key=files["key"])
    assert [it["cid"] for it in got] == [first]
    assert got[0]["prit"] == "faithful" and got[0]["judge"] == "unfaithful"
    assert got[0]["run"] == NAMES[RUNS.index(items_[first]["run_id"])]
    why = dict(missing)
    assert why[second] == "skipped (no_valid_citation)"
    assert "key facts" in why[third] and why["c40"] == "no j1 verdict yet" and len(missing) == 39
    assert C.paired_items(db, judge_version="j2", labels=files["labels"], key=files["key"])[0] == []


def test_compare_waits_for_complete_committed_labels(db, files):
    C.write_sheet(db, RUNS, chunks=CHUNKS, size=40, other_keys=[], **files)
    kw = {"labels": files["labels"], "key": files["key"]}
    assert "only 0 of 40 labels are done" in C.compare_refusal(**kw, dirty=lambda _: False)
    fill(files)
    assert "uncommitted changes" in C.compare_refusal(**kw, dirty=lambda p: p == files["labels"])
    assert C.compare_refusal(**kw, dirty=lambda _: False) is None


def test_a_saved_comparison_doesnt_block_the_judge(db, files):
    C.write_sheet(db, RUNS, chunks=CHUNKS, size=40, other_keys=[], **files)
    fill(files)
    only_folder = lambda p: p == files["labels"].parent          # noqa: E731 - e.g. an untracked compare-j1.json
    assert C.blind_block(RUNS[0], labels=files["labels"], key=files["key"], dirty=only_folder) is None
    assert C.compare_refusal(labels=files["labels"], key=files["key"], dirty=only_folder) is None


def test_compare_command_reports_and_saves(db, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(C, "LABELS_FILE", tmp_path / "cal" / "labels.toml")
    monkeypatch.setattr(C, "KEY_FILE", tmp_path / "cal" / "key.json")
    files = {"sheet": tmp_path / "s.html", "labels": C.LABELS_FILE, "key": C.KEY_FILE}
    k = C.write_sheet(db, RUNS, chunks=CHUNKS, size=40, other_keys=[], **files)
    fill(files)
    monkeypatch.setattr(C.compare_refusal, "__defaults__", (C.LABELS_FILE, C.KEY_FILE, lambda _: False))
    dbfile = str(tmp_path / "runs.db")
    assert C.main(["--compare", "--judge-version", "j1", "--db", dbfile]) == 2
    assert "Judge both runs first" in capsys.readouterr().out
    for cid, info in k["items"].items():
        db.execute("INSERT INTO judgements (run_id, qid, judge_version, judge_model, status, verdict_json) "
                   "VALUES (?, ?, 'j1', 'm', 'judged', ?)",
                   (info["run_id"], info["qid"], verdict([cid != "c01"] * info["key_facts"])))
    db.commit()
    assert C.main(["--compare", "--judge-version", "j1", "--save", "--db", dbfile]) == 0
    out = capsys.readouterr().out
    assert "agreement   40/40 = 100.0%" in out and "Disagreements (1)" in out
    saved = json.loads((tmp_path / "cal" / "compare-j1.json").read_text(encoding="utf-8"))
    assert saved["runs"] == dict(zip(NAMES, RUNS)) and saved["n"] == 40 and saved["set"] == "set1"
    assert saved["faithfulness"]["kappa"] is None                  # every label "faithful": kappa undefined
    assert saved["key_facts"]["agree"] == saved["key_facts"]["n"] - k["items"]["c01"]["key_facts"]


# ---------------------------------------------------------------- sets (session 20)

def test_set_paths():
    assert C.set_paths("set1") == {"sheet": C.SHEET_FILE, "labels": C.LABELS_FILE, "key": C.KEY_FILE}
    p2 = C.set_paths("set2")
    assert p2["labels"] == C.CAL_DIR / "set2" / "labels.toml" and p2["key"] == C.CAL_DIR / "set2" / "key.json"
    assert p2["sheet"].parts[-3:] == ("calibration", "set2", "sheet.html") and "scratch" in p2["sheet"].parts
    with pytest.raises(SystemExit, match="set1, set2"):
        C.set_paths("final")


def test_set2_sheet_has_30_named_runs_and_question_previews(db, tmp_path):
    files = {"sheet": tmp_path / "s2" / "sheet.html", "labels": tmp_path / "cal" / "set2" / "labels.toml",
             "key": tmp_path / "cal" / "set2" / "key.json"}
    k = C.write_sheet(db, RUNS, chunks=CHUNKS, other_keys=[], **files)
    assert k["size"] == 30 and len(k["items"]) == 30 and k["run_names"] == NAMES
    assert sum(not i["should_abstain"] for i in k["items"].values()) == 20
    text = files["labels"].read_text(encoding="utf-8")
    assert C.check(files["labels"], files["key"]) == (0, 30, [])
    by_id = {i.id: i for i in DEV}
    for cid, info in k["items"].items():                         # each block is headed by its own question
        assert f'# {cid}: "{C.preview(by_id[info["qid"]].question)}"\n[[label]]\nid = "{cid}"' in text
    assert C.preview('One "two" three four', words=3) == "One 'two' three ..."


def test_sheet_refuses_runs_another_set_uses(db, tmp_path, files):
    other = tmp_path / "cal" / "key.json"
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps({"runs": [RUNS[1], "r20990101T999999Z"]}), encoding="utf-8")
    files = {"sheet": tmp_path / "s2.html", "labels": tmp_path / "cal" / "set2" / "labels.toml",
             "key": tmp_path / "cal" / "set2" / "key.json"}
    with pytest.raises(SystemExit, match="already used by calibration set"):
        C.write_sheet(db, RUNS, chunks=CHUNKS, other_keys=[other], **files)
    assert not files["labels"].exists()


def test_judge_waits_for_every_set_holding_the_run(db, tmp_path):
    cal = tmp_path / "cal"
    one = {"sheet": tmp_path / "s1.html", "labels": cal / "labels.toml", "key": cal / "key.json"}
    C.write_sheet(db, RUNS, chunks=CHUNKS, size=40, other_keys=[], **one)
    fill(one)                                                     # set 1: complete
    d2 = J.connect(tmp_path / "runs.db")
    new = ["r20990102T000000Z", "r20990102T000100Z"]
    for run in new:
        d2.execute("INSERT INTO runs (run_id, today_default) VALUES (?, ?)", (run, "2026-11-03"))
        for item in DEV:
            R.log_result(d2, run, item, answer(item.id, RUNS[0]), None, {"should_abstain": item.should_abstain})
    two = {"sheet": tmp_path / "s2.html", "labels": cal / "set2" / "labels.toml", "key": cal / "set2" / "key.json"}
    C.write_sheet(d2, new, chunks=CHUNKS, other_keys=[one["key"]], **two)   # set 2: not labelled yet
    assert [k.name for k in C.all_keys(cal)] == ["key.json", "key.json"]
    clean = lambda _: False                                       # noqa: E731
    assert C.blind_block(RUNS[0], dirty=clean, cal_dir=cal) is None
    why = C.blind_block(new[0], dirty=clean, cal_dir=cal)
    assert "only 0 of 30 labels are done" in why
    assert C.blind_block("r20990103T000000Z", dirty=clean, cal_dir=cal) is None    # on no sheet
