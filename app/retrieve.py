"""Search the index: vector, BM25 and hybrid (spec step 4, experiment 2).

    from app.retrieve import search
    search("Under Reg E 6(b), what is the most a customer can lose?", k=5)    # hybrid (default)
    search("my card was stolen and used at a gas station", k=5, mode="vector")

From the project root, `python -m app.retrieve "your question"` prints the top 5 of each mode side
by side (`-k 10` for more). That call embeds the question with Titan, so it needs `aws login`.

- vector: the question is embedded with Titan V2 (same settings as ingest/embed.py) and compared
  with every chunk by cosine similarity. The vectors have length 1, so cosine is a dot product.
  Good at meaning: everyday words find regulation wording ("my card was stolen" -> unauthorized
  electronic fund transfer).
- bm25: classic keyword scoring over each chunk's embed_text (its header and text), with any
  provision the question names ("1026.13(c)(2)", "Reg E 6(b)", "comment 12(b)-1") ranked first.
  Good at exact strings that embeddings blur.
- hybrid: reciprocal rank fusion (RRF) of the two lists. Each chunk scores 1/(60 + rank) in each
  list it appears in, and the sums are ranked. It uses only ranks, so the two scorers' very
  different score scales never need to be made comparable. A provision the question names still
  comes first, as in bm25.

`query_vec` lets a caller pass a ready-made question vector (CI uses committed question embeddings),
so search runs without any AWS call. The BM25 statistics are built from index/chunks.json when the
index is first loaded (well under a second), so there is no separate BM25 file to keep in sync.
"""
import argparse
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index"
RRF_K = 60          # the usual constant: dampens the gap between rank 1 and rank 2
POOL = 50           # how deep each list goes into the fusion
BM25_K1, BM25_B = 1.5, 0.75
PARTS = {"E": "1005", "Z": "1026"}

# Words: runs of letters. Numbers stand alone, so "1026.13(c)(2)" -> "1026", "13", "c", "2".
TOKEN = re.compile(r"[a-z]+|\d+")


def tokens(text):
    return TOKEN.findall(text.lower())


# ---------------------------------------------------------------- named provisions
#
# A provision the question names comes first (decision 9). Keyword scoring alone can't do this:
# "13", "c" and "2" appear in hundreds of chunks, and BM25 scales a match down by chunk length.
# So a reference written in the question is read into a provision ID ("Reg Z 13(c)(2)" ->
# "1026.13(c)(2)"), checked against the provision IDs in the index, and every chunk holding that
# provision, a paragraph under it, or an official comment on either is ranked above the rest.
#
# Ways a reference can be written, read in this order (each match is blanked out before the next
# pattern runs, so "comment 13(c)(2)-1" is not also read as the rule paragraph "13(c)(2)"):
#   full      1026.13(c)(2)   § 1005.6   12 CFR 1026.12   comment 1026.13(c)(2)-1   1026.12-1
#   comment   comment 12(b)-1   comment 6(b)(1)-3.ii   comment 12-1
#   reg       Reg E 6(b)   Regulation Z § 13(c)(2)   Reg. E section 11
#   marker    § 13(c)(2)   section 6(b)   sec. 11   paragraph 9(c)(2)   section 6 of Reg E
#   bare      13(c)(2)   6(b)          (needs at least one bracket; a plain number is never read)
# Short forms carry no part number. The part comes from the regulation the question names
# ("Reg E", "Regulation Z", "12 CFR 1005", "EFTA", "Truth in Lending"); if it names neither or both,
# both parts are tried and whichever exists is kept.

LABEL = r"\s?\(\s?(?:[a-z]|[ivx]{2,5}|\d{1,2})\s?\)"        # (c) (2) (ii) (A) ( b ); not (two) or (am)
PATH = r"(\d{1,2})((?:" + LABEL + r")*)"                     # section number, then any brackets
PATH1 = r"(\d{1,2})((?:" + LABEL + r")+)"                    # ... with at least one bracket
DASH = r"\s?[-‐-―]\s?"                             # - and the Unicode dashes (en, em, ...)
COMMENT_NO = r"(\d{1,2})(?:\.(?:[ivx]+|\d+))*(?![\d(])"      # -1, -1.ii (sub-points are part of comment 1)
SECTION_WORD = r"(?:§+\s*|\bsec(?:tion|\.)?\s*)"
NOT_OF_OTHER = r"(?!\s+of\s+(?!(?:the\s+)?reg(?:ulation)?\b)\S)"   # "section 5 of my agreement" is not a reference

FULL = re.compile(r"(?<![\d.])(1005|1026)\.(\d{1,2})(?!\d)((?:" + LABEL + r")*)(?:" + DASH + COMMENT_NO + r")?")
COMMENT = re.compile(r"\bcomments?\s+" + PATH + r"(?!\.\d)" + DASH + COMMENT_NO)
# "Reg E 6(b)", "Reg E § 6", "Regulation Z section 13(c)"; "Reg E, 2 business days" has no bracket or § and is not read
REG = re.compile(r"\breg(?:ulation)?\.?\s?([ez])\b,?\s*(?:" + SECTION_WORD + PATH + "|" + PATH1 + r")(?![\d.])")
MARKER = re.compile(r"(?:" + SECTION_WORD + PATH + "|" + r"\bparagraph\s+" + PATH1 + r")(?![\d.])" + NOT_OF_OTHER
                    + r"(?:\s+of\s+(?:the\s+)?reg(?:ulation)?\.?\s?([ez])\b)?")
BARE = re.compile(r"(?<![\w.§\-])" + PATH1 + r"(?![\w(])")

REG_CONTEXT = [(re.compile(r"\breg(?:ulation)?\.?\s?e\b|\b1005\b|\befta\b|electronic fund transfer act"), "E"),
               (re.compile(r"\breg(?:ulation)?\.?\s?z\b|\b1026\b|\btila\b|truth in lending"), "Z")]


def _clean(labels):
    return re.sub(r"\s", "", labels)


def _path(m, g):
    """Section number and brackets from whichever of two alternative (number, brackets) group pairs
    matched, starting at group g."""
    if m.group(g) is not None:
        return m.group(g) + _clean(m.group(g + 1))
    return m.group(g + 2) + _clean(m.group(g + 3))


def read_refs(text):
    """The references written in a question, before checking them against the index.

    Each is (written, parts, path): `parts` lists the part numbers to try, `path` is the rest of the
    ID in lower case, e.g. ("Reg Z 13(c)(2)", ["1026"], "13(c)(2)") or ("comment 12(b)-1",
    ["1005", "1026"], "12(b)-1"). In order of appearance."""
    t = text.lower()
    named = {reg for pattern, reg in REG_CONTEXT if pattern.search(t)}
    default = [PARTS[r] for r in sorted(named)] if len(named) == 1 else ["1005", "1026"]
    found = []

    def take(pattern, build):
        nonlocal t
        for m in pattern.finditer(t):
            ref = build(m)
            if ref:
                found.append((m.start(), text[m.start():m.end()].strip(), *ref))
        t = pattern.sub(lambda m: " " * (m.end() - m.start()), t)     # blank out, keep positions

    take(FULL, lambda m: ([m.group(1)], m.group(2) + _clean(m.group(3)) + (f"-{m.group(4)}" if m.group(4) else "")))
    take(COMMENT, lambda m: (default, m.group(1) + _clean(m.group(2)) + f"-{m.group(3)}"))
    take(REG, lambda m: ([PARTS[m.group(1).upper()]], _path(m, 2)))
    take(MARKER, lambda m: ([PARTS[m.group(5).upper()]] if m.group(5) else default, _path(m, 1)))
    take(BARE, lambda m: (default, _path(m, 1)))
    return [(w, parts, path) for _, w, parts, path in sorted(found, key=lambda r: r[0])]


def covered_by(ref, provision):
    """True if `provision` is the provision `ref` names, a paragraph under it, or a comment on either.
    Both in lower case. A comment reference ("1026.12(b)-1") covers only that comment."""
    if provision == ref:
        return True
    if "-" in ref:
        return False
    return provision.startswith(ref + "(") or provision.startswith(ref + "-")


def resolve(refs, provisions):
    """Check read references against the index's provision IDs (any case).

    Returns (found, not_found): `found` lists the IDs that exist, in the index's own spelling
    ("1026.52(b)(1)(ii)(A)", or a section such as "1005.6"); a short reference that exists in both
    parts gives both. `not_found` lists the written references that matched nothing (a provision
    outside the indexed subparts, a typo, or a number that only looked like a reference)."""
    lower = {p.lower(): p for p in provisions}
    found, not_found = [], []
    for written, parts, path in refs:
        hits = []
        for part in parts:
            ref = f"{part}.{path}"
            match = next((p for low, p in lower.items() if covered_by(ref, low)), None)
            if match:
                hits.append(match[:len(ref)])           # same length in any case: IDs are unique ignoring case
        if hits:
            found += [h for h in hits if h not in found]
        else:
            not_found.append(written)
    return found, not_found


# ---------------------------------------------------------------- scoring

class BM25:
    def __init__(self, docs):
        self.tfs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.lens.mean()
        n = len(docs)
        df = Counter(t for tf in self.tfs for t in tf)
        # This idf form stays positive even for a word in most chunks ("the", "shall").
        self.idf = {t: math.log(1 + (n - m + 0.5) / (m + 0.5)) for t, m in df.items()}

    def scores(self, query_tokens):
        out = np.zeros(len(self.tfs))
        norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lens / self.avg)
        for t in set(query_tokens):
            if t not in self.idf:
                continue
            tf = np.array([d.get(t, 0) for d in self.tfs], dtype=float)
            out += self.idf[t] * tf * (BM25_K1 + 1) / (tf + norm)
        return out


@lru_cache(maxsize=1)
def load():
    """The index, loaded once per process (a Lambda keeps it between warm calls)."""
    with open(INDEX / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    vecs = np.load(INDEX / "embeddings.npy")
    with open(INDEX / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    bm25 = BM25([tokens(c["embed_text"]) for c in chunks])
    return chunks, vecs, meta, bm25


@lru_cache(maxsize=1)
def provision_owner():
    """Lower-case provision ID -> index of the one chunk that holds it (each provision is in exactly one)."""
    chunks = load()[0]
    return {p.lower(): i for i, c in enumerate(chunks) for p in c["provision_ids"]}


def query_refs(text):
    """The provisions a question names that exist in the index, in the index's spelling."""
    return resolve(read_refs(text), [p for c in load()[0] for p in c["provision_ids"]])[0]


def named_mask(refs):
    """Boolean per chunk: holds a named provision, a paragraph under it, or a comment on either."""
    owner = provision_owner()
    mask = np.zeros(len(load()[0]), dtype=bool)
    for ref in (r.lower() for r in refs):
        for p, i in owner.items():
            if covered_by(ref, p):
                mask[i] = True
    return mask


@lru_cache(maxsize=1)
def _client():
    import boto3
    from botocore.config import Config
    _, _, meta, _ = load()
    return boto3.client("bedrock-runtime", region_name=meta["region"],
                        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}))


def embed_query(text):
    """The question as a unit vector, made exactly the way the chunks were (meta.json)."""
    _, _, meta, _ = load()
    body = json.dumps({"inputText": text, "dimensions": meta["dimensions"], "normalize": meta["normalize"]})
    resp = _client().invoke_model(modelId=meta["model_id"], body=body,
                                  contentType="application/json", accept="application/json")
    return np.asarray(json.loads(resp["body"].read())["embedding"], dtype=np.float32)


def ranked(scores, n):
    """Indexes of the n highest scores, best first; ties go to the earlier chunk, so results are deterministic."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    return [int(i) for i in order[:n]]


def rrf(lists, n):
    fused = Counter()
    for lst in lists:
        for rank, i in enumerate(lst, 1):
            fused[i] += 1 / (RRF_K + rank)
    return sorted(fused, key=lambda i: (-fused[i], i))[:n], fused


def search(query, k=5, mode="hybrid", query_vec=None):
    """Top k chunks for the query: dicts with rank, score, id, named (the question names a provision
    in it) and the chunk itself."""
    if mode not in ("vector", "bm25", "hybrid"):
        raise ValueError(f"mode must be vector, bm25 or hybrid, not {mode!r}")
    chunks, vecs, _, bm25 = load()
    named = named_mask(query_refs(query))
    if mode in ("vector", "hybrid"):
        q = embed_query(query) if query_vec is None else np.asarray(query_vec, dtype=np.float32)
        v_scores = vecs @ q
    if mode in ("bm25", "hybrid"):
        b_scores = bm25.scores(tokens(query))
        b_scores = b_scores + named * (b_scores.max() + 1)     # named provisions first

    if mode == "vector":
        top, score = ranked(v_scores, k), v_scores           # pure meaning: names get no help here
    elif mode == "bm25":
        top, score = ranked(b_scores, k), b_scores
    else:
        order, fused = rrf([ranked(v_scores, POOL), ranked(b_scores, POOL)], 2 * POOL)
        # Named provisions first here too: a named chunk the vector list never ranked gets only one of
        # its two fused scores and can sink below unnamed chunks both lists liked (session 6).
        top = ([i for i in order if named[i]] + [i for i in order if not named[i]])[:k]
        score = np.array([fused.get(i, 0.0) for i in range(len(chunks))])
    return [{"rank": r, "score": round(float(score[i]), 4), "id": chunks[i]["id"],
             "named": bool(named[i]), "chunk": chunks[i]}
            for r, i in enumerate(top, 1)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Show the top results of each search mode for one question.")
    ap.add_argument("question", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args(argv)
    question = " ".join(args.question)
    found, not_found = resolve(read_refs(question), [p for c in load()[0] for p in c["provision_ids"]])
    vec = embed_query(question)
    cols = {m: search(question, args.k, m, query_vec=vec) for m in ("vector", "bm25", "hybrid")}
    print(f"Q: {question}")
    print(f"Named: {', '.join(found) or '-'}" + (f"   Not in the index: {', '.join(not_found)}" if not_found else ""))
    print(f"\n{'':4}{'vector':34}{'bm25':34}hybrid   (* = named)")
    for r in range(args.k):
        cell = lambda m: (lambda x: f"{x['id']}{'*' if x['named'] else ''} {x['score']:.3f}")(cols[m][r])
        print(f"{r + 1:<4}{cell('vector'):34}{cell('bm25'):34}{cols['hybrid'][r]['id']}"
              f"{'*' if cols['hybrid'][r]['named'] else ''}")


if __name__ == "__main__":
    main()
