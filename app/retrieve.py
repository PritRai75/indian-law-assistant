"""Search the index: vector, BM25 and hybrid (spec step 4, experiment 2).

    from app.retrieve import search
    search("Is an offence under BNS 318(4) bailable?", k=5)                 # hybrid (default)
    search("police refused to write my complaint", k=5, mode="vector")

From the project root, `python -m app.retrieve "your question"` prints the top 5 of each mode side
by side.

- vector: the question is embedded with Titan V2 (same settings as ingest/embed.py) and compared
  with every chunk by cosine similarity. The vectors have length 1, so cosine is a dot product.
  Good at meaning: everyday words find statute wording ("won't write down my complaint" -> FIR).
- bm25: classic keyword scoring over each chunk's Act code plus embed_text, with any provision the
  question names ("BNS 318(4)", "Order VIII rule 1") ranked first. Good at exact strings that
  embeddings blur.
- hybrid: reciprocal rank fusion (RRF) of the two lists. Each chunk scores 1/(60 + rank) in each
  list it appears in, and the sums are ranked. It uses only ranks, so the two scorers' very
  different score scales never need to be made comparable. A provision the question names still
  comes first, as in bm25.

`query_vec` lets a caller pass a ready-made question vector (CI uses committed question embeddings),
so search runs without any AWS call. The BM25 statistics are built from index/chunks.json when the
index is first loaded (well under a second), so there is no separate BM25 file to keep in sync.
"""
import json
import math
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index"
RRF_K = 60          # the usual constant: dampens the gap between rank 1 and rank 2
POOL = 50           # how deep each list goes into the fusion
BM25_K1, BM25_B = 1.5, 0.75

# Words: runs of letters. Numbers stand alone, so "318(4)" -> "318", "4" and "S.2(1)(g)" -> "s", "2", "1", "g".
TOKEN = re.compile(r"[a-z]+|\d+")

# Named provisions. "318" alone matches section 318 of every Act (BNSS s.318 outranked BNS s.318 for
# "BNS section 318 cheating"), "bns" is in all 440 BNS chunks, and BM25 scales a match down by chunk
# length, so the long BNS s.318 fell out of the top 5 for "Is an offence under BNS 318(4) bailable?".
# So a provision the question names comes first in the BM25 list: each chunk has a reference word
# ("bns318", "cpc_oviii_r1"), a reference written in a question becomes the same word ("BNS 318(4)",
# "section 318 of the BNS", "Order VIII rule 1", "O.8 r.1"), and matching chunks are ranked above the
# rest, in BM25 order among themselves.
ACT_CODES = r"(bnss|bns|bsa|cpc)"
SECTION_REF = [
    re.compile(r"\b" + ACT_CODES + r"\s*,?\s*(?:section|sec\.?|s\.)?\s*(\d+[a-z]?)\b"),             # BNS 318, BNSS s. 173
    re.compile(r"\b(?:section|sec\.?|s\.)\s*(\d+[a-z]?)\b(?:\s*\([^)]*\))*\s*(?:of\s+)?(?:the\s+)?" + ACT_CODES + r"\b"),
]
RULE_REF = re.compile(r"\b(?:order|o\.?)\s*([ivxl]+|\d+)(?:\s*-?\s*(a)\b)?\s*,?\s*(?:rule|r\.?)\s*(\d+[a-z]?)\b")
def roman(n):
    out = ""
    for v, s in ((50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= v:
            out, n = out + s, n - v
    return out


def ref_word(unit_id):
    """'BNS-S318' -> 'bns318'; 'CPC-OVIII-R1@commercial' -> 'cpc_oviii_r1'; 'CPC-OXIII-A-R1' -> 'cpc_oxiii_a_r1'."""
    m = re.match(r"([A-Z]+)-(?:O([IVXL]+)(-A)?-R(\w+)|S(\w+))", unit_id)
    act = m.group(1).lower()
    if m.group(2):
        return f"{act}_o{m.group(2).lower()}{'_a' if m.group(3) else ''}_r{m.group(4).lower()}"
    return f"{act}{m.group(5).lower()}"


def query_refs(text):
    """Reference words for the provisions a question names."""
    t, out = text.lower(), []
    for m in SECTION_REF[0].finditer(t):
        out.append(m.group(1) + m.group(2))
    for m in SECTION_REF[1].finditer(t):
        out.append(m.group(2) + m.group(1))
    for m in RULE_REF.finditer(t):
        order = m.group(1) if not m.group(1).isdigit() else roman(int(m.group(1)))
        out.append(f"cpc_o{order}{'_a' if m.group(2) else ''}_r{m.group(3)}")
    return out


def tokens(text):
    return TOKEN.findall(text.lower())




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
    bm25 = BM25([tokens(c["act"] + " " + c["embed_text"]) for c in chunks])
    bm25.refs = np.array([ref_word(c["unit_id"]) for c in chunks])
    return chunks, vecs, meta, bm25


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
    """Top k chunks for the query: dicts with rank, score, id, unit_id, piece_of and the chunk."""
    if mode not in ("vector", "bm25", "hybrid"):
        raise ValueError(f"mode must be vector, bm25 or hybrid, not {mode!r}")
    chunks, vecs, _, bm25 = load()
    if mode in ("vector", "hybrid"):
        q = embed_query(query) if query_vec is None else np.asarray(query_vec, dtype=np.float32)
        v_scores = vecs @ q
    if mode in ("bm25", "hybrid"):
        b_scores = bm25.scores(tokens(query))
        named = np.isin(bm25.refs, query_refs(query))
        b_scores = b_scores + named * (b_scores.max() + 1)     # named provisions first

    if mode == "vector":
        top, score = ranked(v_scores, k), v_scores
    elif mode == "bm25":
        top, score = ranked(b_scores, k), b_scores
    else:
        order, fused = rrf([ranked(v_scores, POOL), ranked(b_scores, POOL)], 2 * POOL)
        # named provisions first here too: fusion alone sank BNS s.318 for "Is an offence under
        # BNS 318(4) bailable?", because the vector list never ranked it (its fused score is shown as is)
        top = ([i for i in order if named[i]] + [i for i in order if not named[i]])[:k]
        score = np.array([fused.get(i, 0.0) for i in range(len(chunks))])
    return [{"rank": r, "score": round(float(score[i]), 4), "id": chunks[i]["id"],
             "unit_id": chunks[i]["unit_id"], "piece_of": chunks[i].get("piece_of"), "chunk": chunks[i]}
            for r, i in enumerate(top, 1)]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "The police won't write down my complaint. What can I do?"
    vec = embed_query(question)
    cols = {m: search(question, 5, m, query_vec=vec) for m in ("vector", "bm25", "hybrid")}
    print(f"Q: {question}\n")
    print(f"{'':3}{'vector':28}{'bm25':28}hybrid")
    for r in range(5):
        print(f"{r + 1:<3}" + "".join(f"{cols[m][r]['id'] + ' ' + format(cols[m][r]['score'], '.3f'):28}"
                                      for m in ("vector", "bm25")) + cols["hybrid"][r]["id"])
