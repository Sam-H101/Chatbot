"""Search by meaning. Each chunk gets an embedding from a small local model (Ollama's
OLLAMA_EMBED_MODEL, qwen3-embedding:0.6b), stored as float32 in chunk_vectors. A question
is embedded too, and the chunks nearest it in meaning (cosine) are blended with the keyword
search (FTS5) by reciprocal rank fusion: "hydraulic oil capacity" then also finds the
passage that says "transmission fluid ... 38 litres", which shares none of its words.

Best effort throughout. When Ollama or the embedding model is unavailable, search is by
keywords only, an upload is added without vectors, and the job worker embeds whatever is
missing later (backfill), as it does for documents indexed before this existed. After a
failed call, embedding pauses for RETRY_SECONDS, so questions don't each wait for it.
Flask-free, like jobs.py."""
import logging
import os
import threading
import time

import numpy as np

from . import db, extract, ollama

log = logging.getLogger(__name__)

BATCH = 32           # texts per embedding call
RRF_K = 60           # reciprocal rank fusion: a chunk scores 1 / (RRF_K + rank) in each ranking it is in
RETRY_SECONDS = 120  # after a failed embedding call, how long search goes by keywords alone
# Passages less similar than this (cosine) to a question don't count as found by meaning.
# With qwen3-embedding:0.6b on manuals, off-topic questions ("hi", "What is 2 + 2?") came
# to 0.30 at most, and the passages that answered real ones to 0.47-0.69; without a floor,
# small talk would be sent the five nearest passages however far they are.
MIN_SIMILARITY = float(os.environ.get("SEARCH_MIN_SIMILARITY") or "0.35")
# qwen3-embedding matches a question best when told the task; passages are embedded as they are.
QUERY = "Instruct: Given a question, retrieve passages from the user's documents that answer it\nQuery: {}"

_cache = None        # ((db.index_version(), model), (chunk ids, document ids, unit-length vectors) or None)
_cache_lock = threading.Lock()
_retry_at = 0.0
_last_error = None


def ready():
    """Embedding is on, unless a call failed in the last RETRY_SECONDS."""
    return time.monotonic() >= _retry_at


def _failed(e):
    global _retry_at, _last_error
    _retry_at = time.monotonic() + RETRY_SECONDS
    why = str(e) or type(e).__name__
    # Warned once per outage, not every RETRY_SECONDS.
    (log.debug if why == _last_error else log.warning)(
        "Search by meaning paused, searching by keywords only (%s); trying again in %d s", why, RETRY_SECONDS)
    _last_error = why


def _worked():
    global _last_error
    if _last_error is not None:
        log.warning("Search by meaning works again")
        _last_error = None


def status():
    """For the page: the embedding model, chunks in the library, how many of them it has
    embedded, and why it is paused (None when it isn't)."""
    chunks, embedded = db.vector_counts(ollama.EMBED_MODEL)
    return {"model": ollama.EMBED_MODEL, "chunks": chunks, "embedded": embedded,
            "error": None if ready() else _last_error}


def _embed(texts, kind="embed", on_usage=None):
    """Vectors of texts as chunk_vectors keeps them (float32 bytes)."""
    return [np.asarray(v, dtype="<f4").tobytes() for v in ollama.embed(texts, kind=kind, on_usage=on_usage)]


def _log(calls, doc_id=None):
    """Embedding calls in the usage log, at no cost (a local model). Best effort, as
    jobs.log_usage."""
    try:
        db.add_usage([(*call, 0.0) for call in calls], doc_id)
    except Exception:
        log.exception("Could not log %d embedding call(s)", len(calls))


def for_chunks(chunks, on_usage=None, on_progress=None):
    """Vectors for a new document's chunks, as db.add_document takes them: (model, one
    per chunk). None when embedding is unavailable: the document is added without them,
    and backfill embeds it later. on_progress(done, total) follows each call, and may
    raise to stop (a cancelled job)."""
    if not chunks or not ready():
        return None
    out = []
    for start in range(0, len(chunks), BATCH):
        try:
            out += _embed([extract.plain(c) for c in chunks[start:start + BATCH]], on_usage=on_usage)
        except Exception as e:  # any failure: an upload is never lost for its vectors
            _failed(e)
            return None
        _worked()
        if on_progress:
            on_progress(len(out), len(chunks))
    return ollama.EMBED_MODEL, out


def backfill(batches=4):
    """Embed up to `batches` calls' worth of chunks that have no vector from the current
    model, one document at a time: documents indexed before search by meaning, while the
    model was unavailable, or by another embedding model. The job worker calls this
    whenever no job is queued, so they catch up between uploads. Returns how many chunks
    it embedded (0: none missing, or embedding unavailable)."""
    model, done = ollama.EMBED_MODEL, 0
    for _ in range(batches):
        rows = db.missing_vectors(model, BATCH) if ready() else []
        if not rows:
            break
        calls = []
        try:
            blobs = _embed([extract.plain(text) for _, _, text in rows], on_usage=lambda *call: calls.append(call))
        except Exception as e:
            _failed(e)
            break
        _worked()
        kept = db.store_vectors(model, [(cid, text, blob) for (cid, _, text), blob in zip(rows, blobs)])
        _log(calls, rows[0][1] if kept else None)  # none kept: the document went meanwhile
        if not kept:
            break
        done += kept
    if done:
        log.info("Embedded %d chunk(s) for search by meaning", done)
    return done


def _vectors():
    """Every vector from the current model as unit-length rows of one matrix, with their
    chunk and document ids: loaded once, and again after chunks or vectors change
    (db.index_version). None when there are none."""
    global _cache
    key = (db.index_version(), ollama.EMBED_MODEL)  # read first: a change while loading reloads next time
    cached = _cache
    if cached and cached[0] == key:
        return cached[1]
    with _cache_lock:  # one load at a time
        if _cache and _cache[0] == key:
            return _cache[1]
        rows = db.all_vectors(key[1])
        held = None
        if rows:
            size = len(rows[0][2])
            rows = [r for r in rows if len(r[2]) == size]
            matrix = np.frombuffer(b"".join(r[2] for r in rows), dtype="<f4").reshape(len(rows), -1).astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            held = (np.array([r[0] for r in rows], dtype=np.int64), np.array([r[1] for r in rows], dtype=np.int64),
                    matrix / norms)
        _cache = (key, held)
    return held


def _nearest(question, n, doc_ids=None, on_usage=None):
    """Ids of the `n` chunks nearest the question in meaning (cosine), nearest first, none
    less similar than MIN_SIMILARITY; only in the documents `doc_ids`, if given. [] when
    there are no vectors to compare it with or it can't be embedded."""
    if not ready():
        return []
    held = _vectors()
    if held is None:
        return []
    ids, docs, matrix = held
    if doc_ids is not None:
        keep = np.isin(docs, np.asarray(list(doc_ids), dtype=np.int64))
        ids, matrix = ids[keep], matrix[keep]
    if not len(ids):
        return []
    try:
        (q,) = ollama.embed([QUERY.format(question)], kind="embed-query", on_usage=on_usage)
    except Exception as e:
        _failed(e)
        return []
    _worked()
    q = np.asarray(q, dtype=np.float32)
    if q.shape != (matrix.shape[1],):
        log.warning("A question's embedding has %s values, the stored ones %d", q.shape, matrix.shape[1])
        return []
    scores = matrix @ (q / (float(np.linalg.norm(q)) or 1.0))
    near = scores >= MIN_SIMILARITY
    ids, scores = ids[near], scores[near]
    k = min(n, len(scores))
    if not k:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    return ids[top[np.argsort(-scores[top], kind="stable")]].tolist()


def search(question, limit, doc_ids=None, on_usage=None):
    """The `limit` chunks that best answer a question, best first: the keyword search's
    ranking and the meaning ranking blended by reciprocal rank fusion. Each chunk as
    db.search_chunks gives it, plus `match`: "keywords", "meaning" or "both". Only in the
    documents `doc_ids`, if given. By keywords alone when there are no vectors or the
    question can't be embedded. on_usage hears of the question's embedding call."""
    n = max(4 * limit, 20)  # candidates from each ranking
    keyword = db.search_chunks(question, n, doc_ids)
    by_words = {h["chunk_id"]: rank for rank, h in enumerate(keyword)}
    by_meaning = {cid: rank for rank, cid in enumerate(_nearest(question, n, doc_ids, on_usage))}
    score = {c: sum(1 / (RRF_K + 1 + r[c]) for r in (by_words, by_meaning) if c in r) for c in {*by_words, *by_meaning}}
    best = sorted(score, key=lambda c: (-score[c], by_words.get(c, n), by_meaning.get(c, n)))[:limit]
    rows = {h["chunk_id"]: h for h in keyword}
    rows.update(db.chunks_by_id([c for c in best if c not in rows]))
    hits = []
    for c in best:
        if c in rows:  # a chunk deleted since the vectors were loaded is skipped
            hits.append({**rows[c], "match": "both" if c in by_words and c in by_meaning
                         else "keywords" if c in by_words else "meaning"})
    return hits
