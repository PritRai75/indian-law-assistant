# Corpus pipeline (Project 1), session 6. Run from the project root after chunk.py:
#   python ingest/embed.py --limit 5     try a few chunks first
#   python ingest/embed.py               embed everything not already in index/
# Needs: pip install numpy, and a live `aws login` session.
"""Embed every chunk's `embed_text` with Amazon Titan Text Embeddings V2 and save the vector index.

Reads  corpus/txt/chunks.jsonl
Writes index/chunks.json     the chunks in row order, each with `embed_sha256` and `embed_tokens`
       index/embeddings.npy  float32 array, one row of 1,024 numbers per chunk, each row of length 1
       index/meta.json       the model and settings, so the index says how it was made

A chunk whose embed_text is unchanged since the last saved index (same SHA-256) reuses its saved
vector. So a re-run after a chunker change pays only for the chunks that changed, and an interrupted
run picks up where it stopped (the index is saved every SAVE_EVERY new vectors). The files hold no
timestamp, so re-running on unchanged chunks rewrites identical files and git shows no change.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import boto3
import numpy as np
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "corpus" / "txt" / "chunks.jsonl"
INDEX = ROOT / "index"
MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "ap-south-1"
DIMS = 1024
SAVE_EVERY = 100
MAX_CHARS = 50_000        # Titan V2 accepts at most 50,000 characters (and 8,192 tokens) per input


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_chunks():
    with open(CHUNKS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_previous():
    """The last saved index as {embed_sha256: (vector, tokens)}; empty if there is none yet."""
    if not (INDEX / "chunks.json").exists() or not (INDEX / "embeddings.npy").exists():
        return {}
    with open(INDEX / "chunks.json", encoding="utf-8") as f:
        old = json.load(f)
    vecs = np.load(INDEX / "embeddings.npy")
    if len(old) != len(vecs):
        sys.exit("index/chunks.json and index/embeddings.npy disagree in length: delete index/ and re-run")
    return {c["embed_sha256"]: (vecs[i], c["embed_tokens"]) for i, c in enumerate(old)}


def make_client():
    # "adaptive" retries: on ThrottlingException boto3 waits and retries (up to 10 tries) and slows
    # its own sending rate, so a burst over the account's limit costs time, not a crashed run.
    return boto3.client("bedrock-runtime", region_name=REGION,
                        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))


def embed(client, text):
    """One Titan call: text -> (1,024 float32 numbers, the input's token count as Titan counts it)."""
    body = json.dumps({"inputText": text, "dimensions": DIMS, "normalize": True})
    resp = client.invoke_model(modelId=MODEL_ID, body=body,
                               contentType="application/json", accept="application/json")
    out = json.loads(resp["body"].read())
    return np.asarray(out["embedding"], dtype=np.float32), out["inputTextTokenCount"]


def save(done, vecs):
    INDEX.mkdir(exist_ok=True)
    with open(INDEX / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=0)
    np.save(INDEX / "embeddings.npy", np.vstack(vecs).astype(np.float32))
    meta = {"model_id": MODEL_ID, "region": REGION, "dimensions": DIMS, "normalize": True,
            "source": "corpus/txt/chunks.jsonl", "field": "embed_text", "chunks": len(done),
            "embed_tokens_total": sum(c["embed_tokens"] for c in done)}
    with open(INDEX / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="embed at most this many new chunks, then stop")
    ap.add_argument("--rpm", type=float, help="pace new calls to at most this many per minute")
    args = ap.parse_args()

    chunks, previous = load_chunks(), load_previous()
    too_long = [c["id"] for c in chunks if len(c["embed_text"]) > MAX_CHARS]
    if too_long:
        sys.exit(f"embed_text over {MAX_CHARS} characters: {too_long}")

    client, gap = None, (60 / args.rpm if args.rpm else 0)
    done, vecs, new, reused, started, last_call = [], [], 0, 0, time.time(), 0.0
    for c in chunks:
        h = sha(c["embed_text"])
        if h in previous:
            vec, tokens = previous[h]
            reused += 1
        else:
            if args.limit is not None and new >= args.limit:
                break
            client = client or make_client()
            wait = gap - (time.time() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.time()
            vec, tokens = embed(client, c["embed_text"])
            new += 1
            if new % SAVE_EVERY == 0:
                save(done + [dict(c, embed_sha256=h, embed_tokens=tokens)], vecs + [vec])
                print(f"  {new} new so far, saved ({time.time() - started:.0f} s)", flush=True)
        done.append(dict(c, embed_sha256=h, embed_tokens=tokens))
        vecs.append(vec)

    if not done:
        sys.exit("nothing embedded")
    save(done, vecs)
    norms = np.linalg.norm(np.vstack(vecs), axis=1)
    print(json.dumps({"chunks_in_index": len(done), "of_chunks": len(chunks), "new_calls": new, "reused": reused,
                      "seconds": round(time.time() - started, 1),
                      "embed_tokens_total": sum(c["embed_tokens"] for c in done),
                      "embed_tokens_max": max(c["embed_tokens"] for c in done),
                      "vector_length_min_max": [round(float(norms.min()), 4), round(float(norms.max()), 4)]},
                     indent=1))
    if len(done) < len(chunks):
        print(f"Stopped early: {len(chunks) - len(done)} chunks still to embed. Run again to continue.")


if __name__ == "__main__":
    main()
