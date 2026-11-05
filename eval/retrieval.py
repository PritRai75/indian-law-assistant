"""The CI retrieval check on the dev split (spec: CI gate; tracker t5-ci, t5-base; session 18).

From the repo root:
    python -m eval.retrieval --embed            embed the dev questions with Titan (needs `aws login`), once,
                                                and again whenever a dev question's text changes
    python -m eval.retrieval --write-baseline   record today's results as eval/baseline.json
    python -m eval.retrieval                    the check: search the dev questions with the committed vectors
                                                and compare with the baseline (no AWS call)
    python -m eval.retrieval --mode vector      look at another search mode or k (no comparison)

Why committed vectors: search needs each question as a vector, and making one is a Titan call. CI has no
AWS access (and shouldn't pay for calls), so the vectors are made once on the laptop and committed
(eval/question_embeddings.npy, with eval/question_embeddings.json saying which question each row is).
Search is deterministic, so the same code, index and vectors always give the same ranking.

What is checked: for each answerable dev question, the rank of the first retrieved chunk holding a
provision in a gold family (the same rule as eval/run.py; ranks past k count as a miss). The check
fails if any question's rank got worse than in eval/baseline.json, so no slice (overall, by persona)
can drop either. It also fails if the search settings, the dev questions or their vectors no longer
match what the baseline was made with. A better rank passes, with a note to record it.

Changing the baseline: only with a commit that says why (spec). Run --write-baseline, look at what
changed (git diff eval/baseline.json), and commit it on its own.

tests/test_retrieval.py runs the check inside pytest, so CI runs it on every push.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from app import retrieve as rt
from app.config import DEFAULT
from eval import golden as G
from eval import run as R

REPO = Path(__file__).resolve().parents[1]
VECTORS = REPO / "eval" / "question_embeddings.npy"
VECTORS_INFO = REPO / "eval" / "question_embeddings.json"
BASELINE = REPO / "eval" / "baseline.json"
SPLIT = "dev"                     # CI never touches test (spec)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dev_items(items: list[G.GoldenItem] | None = None) -> list[G.GoldenItem]:
    items = G.load() if items is None else items
    return [i for i in items if i.split == SPLIT and i.stage == "labelled" and not i.uncertain]


def search_settings(k: int = DEFAULT.k, mode: str = DEFAULT.search_mode) -> dict:
    return {"k": k, "mode": mode, "rrf_k": rt.RRF_K, "pool": rt.POOL}


# ------------------------------------------------------------------ the question vectors

def embed_questions(items: list[G.GoldenItem], embed=None, vectors: Path = VECTORS, info: Path = VECTORS_INFO) -> int:
    """Embed every dev question (declines too, for later use) and save the vectors. `embed` stands in
    for Titan in tests. Returns how many were embedded."""
    embed = embed or rt.embed_query
    _, _, meta, _ = rt.load()
    rows = dev_items(items)
    vecs = np.vstack([np.asarray(embed(i.question), dtype=np.float32) for i in rows])
    np.save(vectors, vecs)
    record = {"model_id": meta["model_id"], "dimensions": meta["dimensions"], "normalize": meta["normalize"],
              "split": SPLIT, "questions": [{"id": i.id, "question_sha256": sha(i.question)} for i in rows]}
    info.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return len(rows)


def load_vectors(items: list[G.GoldenItem], vectors: Path = VECTORS, info: Path = VECTORS_INFO) -> dict:
    """{question id: vector}. Raises ValueError if the files are missing or don't match the dev questions."""
    if not vectors.exists() or not info.exists():
        raise ValueError(f"{vectors.name} or {info.name} is missing: run python -m eval.retrieval --embed")
    record = json.loads(info.read_text(encoding="utf-8"))
    vecs = np.load(vectors)
    _, _, meta, _ = rt.load()
    problems = []
    if len(vecs) != len(record["questions"]):
        problems.append(f"{vectors.name} has {len(vecs)} rows, {info.name} lists {len(record['questions'])}")
    for key in ("model_id", "dimensions", "normalize"):
        if record.get(key) != meta[key]:
            problems.append(f"the vectors were made with {key} {record.get(key)!r}, the index with {meta[key]!r}")
    saved = {q["id"]: q["question_sha256"] for q in record["questions"]}
    now = {i.id: sha(i.question) for i in dev_items(items)}
    if set(saved) != set(now):
        gone, new = sorted(set(saved) - set(now)), sorted(set(now) - set(saved))
        problems.append("the dev questions changed" + (f"; no longer dev: {', '.join(gone)}" if gone else "")
                        + (f"; no vector yet: {', '.join(new)}" if new else ""))
    changed = sorted(q for q in set(saved) & set(now) if saved[q] != now[q])
    if changed:
        problems.append(f"question text changed since embedding: {', '.join(changed)}")
    if problems:
        raise ValueError("; ".join(problems) + ". Re-run python -m eval.retrieval --embed (and check why).")
    return {q["id"]: vecs[n] for n, q in enumerate(record["questions"])}


# ------------------------------------------------------------------ results

def results(items: list[G.GoldenItem], vectors: dict, k: int = DEFAULT.k, mode: str = DEFAULT.search_mode) -> dict:
    """Per answerable dev question: persona, first gold rank (None: not in the top k), the k chunk IDs."""
    ix = R.index_view()
    out = {}
    for i in dev_items(items):
        if i.should_abstain:
            continue
        found = [r["id"] for r in rt.search(i.question, k, mode, query_vec=vectors[i.id])]
        m = R.retrieval_metrics(i.gold_ids or [], found, ix)
        out[i.id] = {"persona": i.persona, "type": i.type, "first_gold_rank": m["first_gold_rank"],
                     "gold_found": m["gold_found"], "gold_total": m["gold_total"], "retrieved": found}
    return out


def tally(per_q: dict) -> dict:
    """hit@1/3/5 as [hits, questions], MRR, gold recall, for all questions and by persona."""
    def one(rows):
        ranks = [r["first_gold_rank"] for r in rows]
        hit = lambda n: [sum(x is not None and x <= n for x in ranks), len(ranks)]
        return {"hit_at_1": hit(1), "hit_at_3": hit(3), "hit_at_5": hit(5),
                "mrr": round(statistics.fmean(1 / x if x else 0.0 for x in ranks), 3) if ranks else None,
                "gold_recall": [sum(r["gold_found"] for r in rows), sum(r["gold_total"] for r in rows)]}
    rows = list(per_q.values())
    return {"overall": one(rows),
            "by_persona": {p: one([r for r in rows if r["persona"] == p]) for p in sorted({r["persona"] for r in rows})}}


def baseline_doc(per_q: dict, settings: dict) -> dict:
    return {"about": "Retrieval on the dev split with committed question vectors; CI fails if any question's "
                     "rank gets worse. Change only through a commit that says why (python -m eval.retrieval).",
            "search": settings, "split": SPLIT, **tally(per_q),
            "questions": {q: {k: v for k, v in r.items()} for q, r in sorted(per_q.items())}}


def worse(old: int | None, new: int | None, k: int) -> bool:
    return (new or k + 1) > (old or k + 1)


def compare(base: dict, per_q: dict, settings: dict) -> tuple[list[str], list[str]]:
    """(failures, notes) of the current results against the baseline."""
    fails, notes = [], []
    if base.get("search") != settings:
        fails.append(f"search settings changed: baseline {base.get('search')}, now {settings}. If this is "
                     "meant, re-write the baseline in a commit that says why.")
        return fails, notes
    k = settings["k"]
    old_q = base["questions"]
    if set(old_q) != set(per_q):
        fails.append(f"the answerable dev questions changed (baseline {len(old_q)}, now {len(per_q)}): "
                     "re-write the baseline in a commit that says why")
        return fails, notes
    show = lambda r: f"rank {r}" if r else f"not in the top {k}"
    for q in sorted(per_q):
        a, b = old_q[q]["first_gold_rank"], per_q[q]["first_gold_rank"]
        if worse(a, b, k):
            fails.append(f"{q} ({per_q[q]['persona']}): first gold chunk {show(a)} -> {show(b)}; "
                         f"retrieved {', '.join(per_q[q]['retrieved'])}")
        elif worse(b, a, k):
            notes.append(f"{q}: better, {show(a)} -> {show(b)}")
    if notes:
        notes.append("Retrieval improved. To record it, re-write the baseline in a commit that says why.")
    return fails, notes


def summary_lines(t: dict) -> list[str]:
    def row(name, s):
        return (f"  {name:11} hit@1 {s['hit_at_1'][0]}/{s['hit_at_1'][1]}   hit@3 {s['hit_at_3'][0]}/{s['hit_at_3'][1]}"
                f"   hit@5 {s['hit_at_5'][0]}/{s['hit_at_5'][1]}   MRR {s['mrr']}"
                f"   gold IDs retrieved {s['gold_recall'][0]}/{s['gold_recall'][1]}")
    return [row("all", t["overall"])] + [row(p, s) for p, s in t["by_persona"].items()]


def check(items=None, baseline: Path = BASELINE) -> tuple[list[str], list[str], dict]:
    """Run the gate. Returns (failures, notes, current results)."""
    items = G.load() if items is None else items
    try:
        vectors = load_vectors(items)
    except ValueError as e:
        return [str(e)], [], {}
    settings = search_settings()
    per_q = results(items, vectors)
    if not baseline.exists():
        return [f"{baseline.name} is missing: run python -m eval.retrieval --write-baseline"], [], per_q
    fails, notes = compare(json.loads(baseline.read_text(encoding="utf-8")), per_q, settings)
    return fails, notes, per_q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.retrieval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--embed", action="store_true", help="embed the dev questions with Titan (aws login)")
    ap.add_argument("--write-baseline", action="store_true", help="record the current results as eval/baseline.json")
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"], help="look at another search mode (no comparison)")
    ap.add_argument("-k", type=int, help="look at another k (no comparison)")
    args = ap.parse_args(argv)
    items = G.load()
    if args.embed:
        n = embed_questions(items)
        print(f"Embedded {n} dev questions into {VECTORS.relative_to(REPO)} and {VECTORS_INFO.relative_to(REPO)}.")
        return 0
    if args.mode or args.k:
        k, mode = args.k or DEFAULT.k, args.mode or DEFAULT.search_mode
        per_q = results(items, load_vectors(items), k, mode)
        print(f"Dev, {len(per_q)} answerable questions, search {mode}, k {k} (a look only: not compared)")
        print("\n".join(summary_lines(tally(per_q))))
        return 0
    if args.write_baseline:
        settings = search_settings()
        per_q = results(items, load_vectors(items))
        BASELINE.write_text(json.dumps(baseline_doc(per_q, settings), indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {BASELINE.relative_to(REPO)}: {len(per_q)} answerable dev questions, search {settings}")
        print("\n".join(summary_lines(tally(per_q))))
        print("Look at git diff eval/baseline.json, then commit it with a message that says why.")
        return 0
    fails, notes, per_q = check(items)
    if per_q:
        print(f"Dev, {len(per_q)} answerable questions, search {search_settings()}")
        print("\n".join(summary_lines(tally(per_q))))
    for n in notes:
        print(n)
    if fails:
        print(f"\nRETRIEVAL CHECK FAILED ({len(fails)}):")
        print("\n".join(f"  {f}" for f in fails))
        return 1
    print("\nRetrieval check passed: no question's rank got worse than the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
