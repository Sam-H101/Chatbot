"""Search by meaning. Each chunk gets an embedding from a small local model (Ollama's
OLLAMA_EMBED_MODEL, qwen3-embedding:0.6b), stored in chunk_vectors as a pgvector vector
with an HNSW index (db.nearest_chunks). A question is embedded too, and the chunks nearest
it in meaning (cosine) are blended with the keyword search (Postgres full-text search) by
reciprocal rank fusion: "hydraulic oil capacity" then also finds the passage that says
"transmission fluid ... 38 litres", which shares none of its words.

Best effort throughout. When Ollama or the embedding model is unavailable, search is by
keywords only, an upload is added without vectors, and the job worker embeds whatever is
missing later (backfill), as it does for documents indexed before this existed. After a
question's embedding fails, questions go by keywords alone for RETRY_SECONDS, so they don't
each wait for it; after a passage fails to embed even on its own, uploads and the backfill
stop embedding for as long, and the backfill sets that passage aside, so it can't hold up
the rest of the library. Flask-free, like jobs.py."""
import logging
import os
import time

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
# A passage found by keywords alone must be at least this similar to the question: one word
# in common is not being about it. "How much have I spent on food this month" (spent, food,
# month) found one receipt by meaning, and filled the other four places with a tractor
# manual's "every 6 months" and four chapters' "Food Accelerator Rod" (0.13-0.19); "How long
# is the warranty?" gave three of five to passages with "long" in them (0.22-0.28). Keyword
# passages that do bear on their question came to 0.34 and more (the warranty's 0.34-0.40).
KEYWORD_MIN_SIMILARITY = float(os.environ.get("SEARCH_KEYWORD_MIN_SIMILARITY") or "0.30")
# A question whose nearest passage of all, at least this similar, is in a document with a
# total (a receipt, bill or invoice) is asked of all such documents, up to TOTALS_PASSAGES
# passages and TOTALS_CHARS characters: what was spent on a kind of thing is added up from
# every receipt's lines, and which receipts have such lines no embedding tells. "How much
# have I spent on medical supplies" was 0.44 at most to five receipts (the one with "FL ENEMA
# 2PK" and "DULCO SC 30" among pretzels and a pizza cutter), all under MIN_SIMILARITY, so
# the model was given a tractor manual's passage with "supplies" in it and no receipt at all;
# embedding receipts with their summaries and keywords brought them no nearer (0.38-0.44).
# But the receipts were nearer than any manual (0.33), as for eight questions about spending
# (their nearest receipt 0.44-0.60, the nearest manual 0.29-0.40), while small talk and
# questions about the manuals were nearer a manual. The floor is for a library of receipts
# alone, where small talk came to 0.31-0.41 ("hi" 0.33).
TOTALS_MIN_SIMILARITY = float(os.environ.get("SEARCH_TOTALS_MIN_SIMILARITY") or "0.40")
TOTALS_PASSAGES = 20
TOTALS_CHARS = 12000  # of receipts, some 5,000 tokens: room left in a local model's context for the conversation
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
    """Vectors of texts, each a list of floats, as db stores them."""
    return [[float(x) for x in v] for v in ollama.embed(texts, kind=kind, on_usage=on_usage)]


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
            vectors = _embed([extract.plain(text) for _, _, text in rows], on_usage=lambda *call: calls.append(call))
        except Exception as e:
            if len(rows) > 1:  # perhaps more than one call can do in EMBED_TIMEOUT: try this document a chunk a call
                _singly.add(doc)
                continue
            _aside[rows[0][0]] = now + ASIDE_SECONDS
            _failed("passages", e)
            break
        _worked("passages")
        kept = db.store_vectors(model, [(cid, text, v) for (cid, _, text), v in zip(rows, vectors)])
        _log(calls, doc if kept else None)  # none kept: the document went meanwhile
        if not kept:
            break
        done += kept
    if done:
        log.info("Embedded %d chunk(s) for search by meaning", done)
    return done


def _question_vector(question, doc_ids=None, on_usage=None):
    """The question's embedding, to compare with the chunks' (among the documents `doc_ids`,
    if given). None when there are no vectors to compare it with (the question isn't
    embedded then) or it can't be embedded."""
    if not ready():
        return None
    size = db.vector_size(ollama.EMBED_MODEL, MIN_CHARS, doc_ids)
    if size is None:
        return None
    try:
        (q,) = ollama.embed([QUERY.format(question)], kind="embed-query", on_usage=on_usage)
    except Exception as e:
        _failed("questions", e)
        return None
    _worked("questions")
    if len(q) != size:
        log.warning("A question's embedding has %d values, the stored ones %d", len(q), size)
        return None
    if not any(q):
        return None  # no direction: as near to everything as to nothing
    return q


def search(question, limit, doc_ids=None, on_usage=None):
    """The `limit` chunks that best answer a question, best first: the keyword search's
    ranking and the meaning ranking blended by reciprocal rank fusion. Each chunk as
    db.search_chunks gives it, plus `match`: "keywords", "meaning", "both" or "opening". By
    keywords alone when there are no vectors or the question can't be embedded. on_usage
    hears of the question's embedding call.

    In all documents, a passage found by meaning alone must be at least MIN_SIMILARITY
    and within MEANING_MARGIN of the nearest, and one found by keywords alone at least
    KEYWORD_MIN_SIMILARITY (one without a vector yet counts: nothing says it is far), so a
    library of manuals and receipts doesn't answer a question about one with passages of
    the other that share a word with it. Limited to the documents `doc_ids`, the
    question is about them whatever it says ("What is this document about?" is near no
    passage in particular, and shares no keyword with them): their nearest passages count
    however far they are, and when fewer than `limit` are found, their opening passages
    ("opening") make up the rest.

    A question about the documents with totals (TOTALS_MIN_SIMILARITY) is searched as if
    limited to them, with room for TOTALS_PASSAGES passages of up to TOTALS_CHARS in all (and
    never fewer than `limit`), so each receipt's opening passage is there to add up from. The
    result's `missed` is how many of those documents did not fit (0 otherwise)."""
    scoped = doc_ids is not None
    model, q = ollama.EMBED_MODEL, _question_vector(question, doc_ids, on_usage)
    totals = []
    if q and not scoped:
        nearest = db.nearest_chunks(model, q, 1, TOTALS_MIN_SIMILARITY, MIN_CHARS)
        totals = db.documents_with_totals(nearest[0][0]) if nearest else []
    if totals:
        hits = search_among(question, max(limit, TOTALS_PASSAGES), totals, q)
        size, kept = 0, Hits()
        for h in hits:
            size += len(h["text"])
            if len(kept) >= limit and size > TOTALS_CHARS:
                break
            kept.append(h)
        kept.missed = len(set(totals) - {h["doc_id"] for h in kept})
        return kept
    return Hits(search_among(question, limit, doc_ids, q))


class Hits(list):
    """A search's passages; `missed`: documents with totals that the question was about and
    that did not fit among them."""
    missed = 0


def search_among(question, limit, doc_ids, q):
    """search, given the question's embedding `q` (None: by keywords alone)."""
    scoped = doc_ids is not None
    n = max(4 * limit, 20)  # candidates from each ranking
    keyword = db.search_chunks(question, n, doc_ids, question=True)
    model = ollama.EMBED_MODEL
    near = db.nearest_chunks(model, q, n, None if scoped else MIN_SIMILARITY, MIN_CHARS, doc_ids) if q else []
    if q and not scoped:
        close = {c for c, _ in near}
        far = db.similarities(model, q, [h["chunk_id"] for h in keyword if h["chunk_id"] not in close])
        keyword = [h for h in keyword if far.get(h["chunk_id"], 1.0) >= KEYWORD_MIN_SIMILARITY]
    by_words = {h["chunk_id"]: rank for rank, h in enumerate(keyword)}
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
