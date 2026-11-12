"""Search by meaning. Each chunk gets an embedding from a small local model (Ollama's
OLLAMA_EMBED_MODEL, qwen3-embedding:0.6b), stored as float32 in chunk_vectors. A question
is embedded too, and the chunks nearest it in meaning (cosine) are blended with the keyword
search (FTS5) by reciprocal rank fusion: "hydraulic oil capacity" then also finds the
passage that says "transmission fluid ... 38 litres", which shares none of its words.

Best effort throughout. When Ollama or the embedding model is unavailable, search is by
keywords only, an upload is added without vectors, and the job worker embeds whatever is
missing later (backfill), as it does for documents indexed before this existed. After a
question's embedding fails, questions go by keywords alone for RETRY_SECONDS, so they don't
each wait for it; after a passage fails to embed even on its own, uploads and the backfill
stop embedding for as long, and the backfill sets that passage aside, so it can't hold up
the rest of the library. Flask-free, like jobs.py."""
import logging
import os
import threading
import time

import numpy as np

from . import db, extract, ollama

log = logging.getLogger(__name__)

BATCH = 32            # texts per embedding call
RRF_K = 60            # reciprocal rank fusion: a chunk scores 1 / (RRF_K + rank) in each ranking it is in
RETRY_SECONDS = 120   # after a failed embedding call, how long that kind of embedding (questions, passages) pauses
ASIDE_SECONDS = 1800  # a passage that fails to embed on its own waits this long before the backfill tries it again
# Passages less similar than this (cosine) to a question don't count as found by meaning.
# With qwen3-embedding:0.6b on the live library (14 manuals, MIN_CHARS below), greetings,
# thanks and other small talk ("hi", "thanks, that helps", "How much is the rent?") came to
# 0.45 at most, questions about the bot itself ("What is this?") to 0.49, and real
# questions' nearest passages to 0.59-0.81, the passages that answer them from 0.49. Without
# a floor, small talk would be sent the five nearest passages however far they are. (0.35,
# set on a small test library, let every short question through.)
MIN_SIMILARITY = float(os.environ.get("SEARCH_MIN_SIMILARITY") or "0.47")
# Passages found by meaning alone that are this much less similar than the nearest one are
# left out too. In-domain passages nearly all clear MIN_SIMILARITY, so it alone let a
# workshop guide's parts lists fill the places of "How long is the warranty?" (0.39-0.41,
# its answer 0.57) and "How much is the rent?" (0.36, its answer 0.55). Not tighter: for
# "hydraulic oil capacity", the "transmission fluid ... 38.0 litres" passage was 0.10 behind.
MEANING_MARGIN = float(os.environ.get("SEARCH_MEANING_MARGIN") or "0.12")
# Chunks shorter than this (characters) are left out of search by meaning, unless they are
# their document's only chunk; keyword search still finds them. A scrap of a longer document
# (the tail of a table, a page footer: "h Squareness Item Factory Specification Allowable
# Limit", 57 characters) carries next to no meaning, and on the live library came nearest to
# nearly every short question ("hi" 0.54, "How much is the rent?" 0.56), taking a place
# among real questions' passages too. The 10 chunks there under 100 characters are all such
# scraps; from 101 on they hold figures (valve clearances, 105).
MIN_CHARS = 100
# qwen3-embedding matches a question best when told the task; passages are embedded as they are.
QUERY = "Instruct: Given a question, retrieve passages from the user's documents that answer it\nQuery: {}"

_cache = None        # ((db.index_version(), model), (chunk ids, document ids, unit-length vectors) or None)
_cache_lock = threading.Lock()
_paused = {}         # "questions" | "passages" -> (time.monotonic() it ends, why), until a call works again
_singly = set()      # documents whose batch failed: the backfill sends them a passage per call
_aside = {}          # chunk id -> time.monotonic() before which the backfill passes it over
_NAMES = {"questions": "Search by meaning", "passages": "Embedding passages for search by meaning"}


def _on(kind):
    p = _paused.get(kind)
    return p is None or time.monotonic() >= p[0]


def ready():
    """Questions are embedded, unless one failed in the last RETRY_SECONDS."""
    return _on("questions")


def indexing():
    """Passages are embedded (uploads, backfill), unless one failed on its own in the last
    RETRY_SECONDS."""
    return _on("passages")


def _failed(kind, e):
    why = str(e) or type(e).__name__
    was = _paused.get(kind)
    _paused[kind] = (time.monotonic() + RETRY_SECONDS, why)
    # Warned once per outage, not every RETRY_SECONDS.
    (log.debug if was and was[1] == why else log.warning)(
        "%s paused, keywords only meanwhile (%s); trying again in %d s", _NAMES[kind], why, RETRY_SECONDS)


def _worked(kind):
    if _paused.pop(kind, None):
        log.warning("%s works again", _NAMES[kind])


def status():
    """For the page: the embedding model, chunks in the library, how many of them it has
    embedded, and why embedding is paused (None when it isn't)."""
    chunks, embedded = db.vector_counts(ollama.EMBED_MODEL)
    now = time.monotonic()
    errors = [p[1] for p in (_paused.get("questions"), _paused.get("passages")) if p and now < p[0]]
    return {"model": ollama.EMBED_MODEL, "chunks": chunks, "embedded": embedded, "error": (errors or [None])[0]}


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
    per chunk). BATCH a call; after a call fails, a chunk a call (a batch of tables can take
    longer than EMBED_TIMEOUT where each passage alone doesn't). None when a chunk fails
    even alone: the document is added without vectors, and backfill embeds it later.
    on_progress(done, total) follows each call, and may raise to stop (a cancelled job)."""
    if not chunks or not indexing():
        return None
    out, size = [], BATCH
    while len(out) < len(chunks):
        texts = [extract.plain(c) for c in chunks[len(out):len(out) + size]]
        try:
            out += _embed(texts, on_usage=on_usage)
        except ollama.Aborted:  # the job was cancelled: not a failure of the embedding model
            raise
        except Exception as e:  # any failure: an upload is never lost for its vectors
            if len(texts) > 1:
                size = 1
                continue
            _failed("passages", e)
            return None
        _worked("passages")
        if on_progress:
            on_progress(len(out), len(chunks))
    return ollama.EMBED_MODEL, out


def backfill(batches=4):
    """Embed, in up to `batches` calls, chunks that have no vector from the current model,
    one document at a time: documents indexed before search by meaning, while the model was
    unavailable, or by another embedding model. The job worker calls this whenever no job
    is queued, so they catch up between uploads. A document whose batch fails is sent a
    chunk a call from then on; a chunk that fails even alone is set aside for ASIDE_SECONDS,
    so the chunks and documents after it go on, and passages pause (Ollama may be down).
    Returns how many chunks it embedded (0: none missing, or embedding unavailable)."""
    model, done = ollama.EMBED_MODEL, 0
    for _ in range(batches):
        if not indexing():
            break
        now = time.monotonic()
        for cid in [c for c, until in _aside.items() if until <= now]:
            del _aside[cid]
        rows = db.missing_vectors(model, BATCH, skip=_aside)
        if not rows:
            if not _aside:
                _singly.clear()
            break
        doc = rows[0][1]
        if doc in _singly:
            rows = rows[:1]
        calls = []
        try:
            blobs = _embed([extract.plain(text) for _, _, text in rows], on_usage=lambda *call: calls.append(call))
        except Exception as e:
            if len(rows) > 1:  # perhaps more than one call can do in EMBED_TIMEOUT: try this document a chunk a call
                _singly.add(doc)
                continue
            _aside[rows[0][0]] = now + ASIDE_SECONDS
            _failed("passages", e)
            break
        _worked("passages")
        kept = db.store_vectors(model, [(cid, text, blob) for (cid, _, text), blob in zip(rows, blobs)])
        _log(calls, doc if kept else None)  # none kept: the document went meanwhile
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
        # Loaded into arrays sized up front, row by row, and normalized in place, the old ones
        # let go first: a load takes about the room of the vectors once (50,000 chunks: 205
        # MB), not three times (four to reload).
        _cache = None
        rows = db.all_vectors(key[1], MIN_CHARS)
        try:  # closed however it ends: the read transaction would hold up writers
            most, n, matrix = next(rows), 0, None
            ids, docs = np.empty(most, dtype=np.int64), np.empty(most, dtype=np.int64)
            for cid, doc, blob in rows:
                v = np.frombuffer(blob, dtype="<f4")
                if matrix is None:
                    matrix = np.empty((most, len(v)), dtype=np.float32)
                elif len(v) != matrix.shape[1]:  # another size than the first: not comparable
                    continue
                ids[n], docs[n], matrix[n] = cid, doc, v
                n += 1
        finally:
            rows.close()
        held = None
        if n:
            matrix = matrix[:n]
            norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))[:, None]  # no squared copy, as linalg.norm makes
            norms[norms == 0] = 1
            matrix /= norms
            held = ids[:n], docs[:n], matrix
        _cache = (key, held)
    return held


def _nearest(question, n, floor, doc_ids=None, on_usage=None):
    """The `n` chunks nearest the question in meaning (cosine), nearest first, as [(chunk
    id, similarity)], none less similar than `floor` (None: however far); only in the
    documents `doc_ids`, if given. [] when there are no vectors to compare it with or it
    can't be embedded."""
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
        _failed("questions", e)
        return []
    _worked("questions")
    q = np.asarray(q, dtype=np.float32)
    if q.shape != (matrix.shape[1],):
        log.warning("A question's embedding has %s values, the stored ones %d", q.shape, matrix.shape[1])
        return []
    scores = matrix @ (q / (float(np.linalg.norm(q)) or 1.0))
    if floor is not None:
        near = scores >= floor
        ids, scores = ids[near], scores[near]
    k = min(n, len(scores))
    if not k:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return list(zip(ids[top].tolist(), scores[top].tolist()))


def search(question, limit, doc_ids=None, on_usage=None):
    """The `limit` chunks that best answer a question, best first: the keyword search's
    ranking and the meaning ranking blended by reciprocal rank fusion. Each chunk as
    db.search_chunks gives it, plus `match`: "keywords", "meaning", "both" or "opening". By
    keywords alone when there are no vectors or the question can't be embedded. on_usage
    hears of the question's embedding call.

    In all documents, a passage found by meaning alone must be at least MIN_SIMILARITY
    and within MEANING_MARGIN of the nearest. Limited to the documents `doc_ids`, the
    question is about them whatever it says ("What is this document about?" is near no
    passage in particular, and shares no keyword with them): their nearest passages count
    however far they are, and when fewer than `limit` are found, their opening passages
    ("opening") make up the rest."""
    scoped = doc_ids is not None
    n = max(4 * limit, 20)  # candidates from each ranking
    keyword = db.search_chunks(question, n, doc_ids, question=True)
    by_words = {h["chunk_id"]: rank for rank, h in enumerate(keyword)}
    near = _nearest(question, n, None if scoped else MIN_SIMILARITY, doc_ids, on_usage)
    if near and not scoped:
        near = [(c, s) for c, s in near if c in by_words or s >= near[0][1] - MEANING_MARGIN]
    by_meaning = {cid: rank for rank, (cid, _) in enumerate(near)}
    score = {c: sum(1 / (RRF_K + 1 + r[c]) for r in (by_words, by_meaning) if c in r) for c in {*by_words, *by_meaning}}
    best = sorted(score, key=lambda c: (-score[c], by_words.get(c, n), by_meaning.get(c, n)))[:limit]
    rows = {h["chunk_id"]: h for h in keyword}
    rows.update(db.chunks_by_id([c for c in best if c not in rows]))
    hits = []
    for c in best:
        if c in rows:  # a chunk deleted since the vectors were loaded is skipped
            hits.append({**rows[c], "match": "both" if c in by_words and c in by_meaning
                         else "keywords" if c in by_words else "meaning"})
    if scoped and len(hits) < limit:
        found = {h["chunk_id"] for h in hits}
        hits += [{**h, "match": "opening"} for h in db.opening_chunks(doc_ids, limit + len(hits))
                 if h["chunk_id"] not in found][:limit - len(hits)]
    return hits
