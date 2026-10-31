"""Tests for the saved vector index (index/), made by ingest/embed.py. No AWS call is made.

The key check: every chunk in corpus/txt/chunks.jsonl has a row in the index made from its current
embed_text. After a chunker change, this fails until embed.py is re-run, so a stale index can't
go unnoticed. Skipped until the index has been built for the first time.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index"

pytestmark = pytest.mark.skipif(not (INDEX / "embeddings.npy").exists(), reason="index not built yet")


@pytest.fixture(scope="module")
def index():
    with open(INDEX / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(INDEX / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return chunks, np.load(INDEX / "embeddings.npy"), meta


def test_one_row_per_chunk(index):
    chunks, vecs, meta = index
    assert vecs.shape == (len(chunks), meta["dimensions"]) == (meta["chunks"], 1024)
    assert vecs.dtype == np.float32


def test_index_matches_current_chunks(index):
    chunks, _, _ = index
    with open(ROOT / "corpus" / "txt" / "chunks.jsonl", encoding="utf-8") as f:
        current = [json.loads(line) for line in f if line.strip()]
    assert [c["id"] for c in chunks] == [c["id"] for c in current], "index is stale or partial: run ingest/embed.py"
    stale = [c["id"] for c, cur in zip(chunks, current)
             if c["embed_sha256"] != hashlib.sha256(cur["embed_text"].encode("utf-8")).hexdigest()]
    assert stale == [], "chunks changed since they were embedded: run ingest/embed.py"


def test_vectors_have_length_one(index):
    _, vecs, _ = index
    norms = np.linalg.norm(vecs, axis=1)
    assert np.all(np.abs(norms - 1) < 1e-3)
