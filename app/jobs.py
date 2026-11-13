"""Background reading. Uploads and re-reads wait in the jobs table, and one worker thread
per process works through them in order, saving progress as it goes. Pages a model has
read are kept with the job, so a job that failed, or was cut off by a restart, carries on
where it stopped instead of paying for those pages again. With no job to do, the worker
embeds chunks that have no vectors yet (vectors.backfill). Flask-free, like llm.py and
ollama.py: the routes that create jobs and report on them are in main.py."""
import logging
import math
import os
import re
import sqlite3
import threading
import time
import unicodedata

import anthropic
import pypdfium2

from . import db, extract, llm, ollama, sections, thumbs, units, vectors

log = logging.getLogger(__name__)


def _pages_mode(raw):
    """CLAUDE_PDF_PAGES, checked once here: a mistyped value would otherwise turn down
    every upload, local ones too, with an error that doesn't name the setting."""
    mode = (raw or "scanned").strip().lower()
    if mode not in ("scanned", "all"):
        log.warning("CLAUDE_PDF_PAGES=%r is neither 'scanned' nor 'all': using 'scanned'", raw)
        return "scanned"
    return mode


# "model" (default): the chosen model reads PDFs (Ollama's vision model only the scanned
# pages; Claude see CLAUDE_PDF_PAGES). "local": the PDF's embedded text layer only, so
# scanned pages come out empty.
PDF_EXTRACTOR = os.environ.get("PDF_EXTRACTOR", "model")
# Which PDF pages Claude reads. "scanned" (default): only pages with little or no text
# layer (extract.is_scan); the others' text layer is used as it is. "all": every page,
# as text and image, which is worth its cost for complex tables and layouts.
CLAUDE_PDF_PAGES = _pages_mode(os.environ.get("CLAUDE_PDF_PAGES"))
# A job whose Claude reading is estimated to cost at least this much (USD) waits for
# the user to confirm it.
CONFIRM_COST_USD = float(os.environ.get("CONFIRM_COST_USD") or "1.00")
# Rough sizes behind an estimate: tokens per page Claude reads (image plus text) and
# writes, the metadata reply, and the notes on each section of a long document. Analysis
# input is about 4 characters a token.
PAGE_INPUT_TOKENS, PAGE_OUTPUT_TOKENS, ANALYSIS_OUTPUT_TOKENS, NOTES_OUTPUT_TOKENS = 2000, 600, 500, 1000
# A photo: Claude bills an image by its pixels (about width x height / 750 tokens, at most
# about 1,600 at the size it is sent, extract.CLAUDE_PHOTO_SIDE), plus the prompt; the text
# of a receipt or a page runs to about PHOTO_OUTPUT_TOKENS, PHOTO_TEXT_CHARS characters.
PHOTO_MAX_TOKENS, PHOTO_PROMPT_TOKENS, PHOTO_OUTPUT_TOKENS, PHOTO_TEXT_CHARS = 1600, 300, 600, 2400
# Tests turn the thread off and run the queue in their own thread with drain().
BACKGROUND = True

_keys = {}  # job id -> the browser's Claude key: in memory only, never stored
_wake, _stop = threading.Event(), threading.Event()
_thread = None
_start_lock = threading.Lock()
_current, _current_lock = None, threading.Lock()  # (job id, thread id) of the job being read (interrupt)
# Seconds between tries to mark a job failed while the database stays locked (_fail).
FAIL_RETRY_SECONDS = (2, 10, 30)
# When this process started (UTC, as SQLite's datetime('now')): answers begun before it and
# never finished were cut off by a restart (recover).
STARTED = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


class NoText(ValueError):
    pass


class Cancelled(Exception):
    pass


class AwaitConfirmation(Exception):
    """The estimate is at or above CONFIRM_COST_USD: the job waits for the user."""


class AnalysisFailed(Exception):
    """The metadata call of a job that replaces documents failed (its cause is chained):
    the job fails, so they keep their title, summary and keywords."""


# ---- helpers shared with the routes ----

def label(backend, model):
    return model if backend == "ollama" else llm.label(model)


def choice_label(choice):
    """'auto' | 'ollama:<model>' | 'claude:<model>' -> what the page shows."""
    if choice == "auto":
        return "Auto"
    backend, _, model = choice.partition(":")
    return label(backend, model)


def used(backend_model, note):
    backend, model = backend_model
    return {"model": f"{backend}:{model}", "model_label": label(backend, model), "switched": note}


def priced(kind, backend, model, usage):
    """One usage-log entry. Local models cost nothing."""
    return kind, backend, model, usage, llm.cost(model, usage) if backend == "claude" else 0.0


def log_usage(calls, doc_id=None, job_id=None):
    """Best effort: if the log can't be written (say the database stays locked), the
    answer or document that already succeeded still counts, and the reason a job
    failed is not replaced by a logging error."""
    try:
        db.add_usage(calls, doc_id, job_id)
    except Exception:
        log.exception("Could not log %d model call(s)", len(calls))


def remember_key(job_id, key):
    """The browser's key for this job, until it ends; without one, a key left under this
    id from before goes. A restart forgets it: a job that then needs Claude fails asking
    for the key, and Retry sends it again."""
    if key:
        _keys[job_id] = key
    else:
        _keys.pop(job_id, None)


def prune_keys():
    """Forget the keys of jobs that won't run again: cancelled, ended, or never committed
    (SQLite gives that id to the next job, which must not spend this key). Under
    files_lock, which the routes hold while they keep a key for a job they are about to
    commit or queue, so no key on its way in is dropped."""
    with db.files_lock:
        for job_id in list(_keys):
            if db.job_status(job_id) not in db.ACTIVE_JOBS:
                _keys.pop(job_id, None)


def _claude_reader(model, mode):
    return f"claude:{model}:{mode}"


def _vision_reader():
    return f"ollama:{ollama.VISION_MODEL}"


def first_reader(job):
    """Whose kept pages a retry of this job uses first: Claude's own reading for a
    Claude model, the local vision model's for local models and Auto (which only gets
    to Claude's pages if the local model fails again)."""
    backend, _, model = job["model"].partition(":")
    if backend == "claude":
        return _claude_reader(model, (job.get("options") or {}).get("claude_pages") or CLAUDE_PDF_PAGES)
    return _vision_reader()


def describe(e, name):
    """A failed job's error, worded like the API's error responses."""
    if isinstance(e, AnalysisFailed):
        cause = e.__cause__
        why = describe(cause, name) if isinstance(cause, anthropic.APIError) else f"Could not analyze {name}: {cause}"
        return f"{why}. The library was left as it was"
    if isinstance(e, llm.MissingKey):
        return "No Claude API key: enter yours (Key, top right), then press Retry"
    if isinstance(e, anthropic.AuthenticationError):
        return "Invalid Claude API key"
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limited by the Claude API; press Retry shortly"
    if isinstance(e, anthropic.APIStatusError):
        return f"Claude API error: {e.message}"
    if isinstance(e, anthropic.APIConnectionError):
        return "Could not reach the Claude API"
    if isinstance(e, ollama.OllamaUnavailable):
        return f"{e}. Is it running?"
    if isinstance(e, ollama.OllamaError):
        return f"Ollama error: {e}"
    if isinstance(e, NoText):
        return str(e) or "No text found in file"
    if isinstance(e, db.JobGone):
        return str(e)
    return f"Could not read {name}: {e}"


# ---- the worker ----

def start():
    """Start the worker thread, once per process. main.py calls this on every request,
    so nothing runs at import (gunicorn forks its worker first) and a restart picks the
    queue up again with the first page load. Jobs a restart left 'running' go back in
    the queue first: no thread was running them any more."""
    global _thread
    if not BACKGROUND or (_thread and _thread.is_alive()):
        return
    with _start_lock:
        if _thread and _thread.is_alive():
            return
        _stop.clear()
        recover()
        _thread = threading.Thread(target=_loop, name="jobs", daemon=True)
        _thread.start()


def recover():
    """After a restart: jobs left running go back in the queue, and answers left being written
    fail, saying so (their questions stay; a page showing one stops waiting for it). Answers
    this process began are left alone, should this run again later."""
    n = db.requeue_running()
    if n:
        log.warning("Re-queued %d job(s) interrupted by a restart", n)
    cut = db.end_interrupted_answers(STARTED)
    if cut:
        log.warning("%d answer(s) were cut off by a restart", cut)
    return n


def wake():
    _wake.set()


def stop(timeout=10):
    """Stop the worker thread after its current job (tests)."""
    global _thread
    _stop.set()
    _wake.set()
    if _thread:
        _thread.join(timeout)
        _thread = None


def check_units(doc_id=None):
    """Run the unit check (units.check) on documents with none from this version of
    units.py: indexed before it ran at upload, or checked by an earlier units.py. Or on
    just `doc_id`. Quick: it reads their stored chunks, no model. Returns how many
    documents it checked."""
    ids = db.unchecked_documents(units.VERSION) if doc_id is None else [doc_id]
    for i in ids:
        db.set_unit_findings(i, units.check(*db.unit_source(i)))
    return len(ids)


def unit_check(doc_id):
    """A document's unit check for the API (db.unit_findings), made now if it has none
    from this version yet (the start-up check hasn't got to it). None: no such document."""
    doc = db.unit_findings(doc_id, units.VERSION)
    if doc and doc["findings"] is None:
        check_units(doc_id)
        doc = db.unit_findings(doc_id, units.VERSION)
    return doc


def _loop():
    try:  # once per start, before any job: older documents get their unit check
        if n := check_units():
            log.info("Checked units in %d document(s) with no check from this version of units.py", n)
    except Exception:
        log.exception("Could not check units in older documents")
    while not _stop.is_set():
        _wake.clear()  # before looking, so a job queued meanwhile still wakes the wait below
        try:
            if run_next():
                continue
            # No job: chunks without vectors (older documents, or uploads made while the embedding
            # model was unavailable) are embedded a few calls at a time, between jobs.
            if vectors.backfill():
                continue
        except Exception:  # say the database stayed locked: try again shortly
            log.exception("Job worker error")
        _wake.wait(30)


def run_next():
    """Run the oldest queued job in this thread. False if nothing was queued."""
    global _current
    job = db.claim_next_job()
    if not job:
        return False
    with _current_lock:
        _current = job["id"], threading.get_ident()
    try:
        _process(job)
    finally:
        with _current_lock:
            _current = None
    return True


def interrupt():
    """If the job being read is no longer running (cancelled, or its document deleted), cut
    off the Ollama call it waits on, so it stops now rather than when the call returns: up
    to OLLAMA_TIMEOUT later for a call Ollama never answers, with the queue held up behind
    it. Claude calls are left to finish. Under _current_lock, so the worker can't have moved
    on to another job meanwhile. Best effort: the job stops at its next step anyway."""
    with _current_lock:
        try:
            if _current and db.job_status(_current[0]) != "running":
                ollama.abort(_current[1])
        except sqlite3.Error:
            log.warning("Could not check whether job %d still runs", _current[0], exc_info=True)


def drain():
    """Run queued jobs here until none is left, as the worker thread would (tests)."""
    while run_next():
        pass


class _Progress:
    """A running job's stage, page counts and finished pages, written to its row. A
    write that finds the job no longer running (cancelled) stops the job."""

    def __init__(self, job):
        self.id, self.partial, self.last = job["id"], job["partial"] or {}, 0.0

    def stage(self, text, done=None, total=None, force=True):
        if not force and time.monotonic() - self.last < 1:  # pages with a text layer fly by
            return
        fields = {"stage": text}
        if done is not None:
            fields.update(progress=done, total=total or 0)
        self._write(fields)

    def pages(self, reader):
        """Pages this reader finished in an earlier attempt: {number: text}."""
        return {int(n): t for n, t in self.kept(reader).items()}

    def kept(self, key):
        """What an earlier attempt kept under `key`: a reader's pages, or a model's notes on
        the sections of a long document ({section key: notes})."""
        return dict(self.partial.get(key, {}))

    def keep(self, key, items, text=None, done=None, total=None):
        self.partial.setdefault(key, {}).update({str(n): t for n, t in items.items()})
        fields = {"partial": self.partial}
        if text is not None:
            fields.update(stage=text, progress=done, total=total)
        self._write(fields)

    def _write(self, fields):
        self.last = time.monotonic()
        if not db.update_job(self.id, **fields):
            raise Cancelled()


def _process(job):
    job_id, opts = job["id"], job["options"] or {}
    status = "failed"
    progress = _Progress(job)

    def on_usage(*call):
        # Logged at once, so a restart mid-job can't lose what was paid for. Finishing the
        # job links its calls to the document (db._finish_job).
        log_usage([priced(*call)], job_id=job_id)

    try:
        with open(os.path.join(db.files_dir(), os.path.basename(job["file_path"])), "rb") as f:
            data = f.read()
        text, meta, analysis, who, note = _read(job, opts, data, progress, on_usage)
        kind = extract.kind(job["filename"])
        chunks = extract.chunk_text(text)
        # A PDF's text has pages (page 1 too when no page break follows it): each chunk's are
        # kept for citations, and the unit check lists the litre/gallon misprints, that answers
        # will be corrected for, by page. Photos have none: they are cited by file name.
        paged = kind == "pdf"
        pages = extract.page_ranges(chunks) if paged else None
        check = units.check(chunks, paged=paged)
        vecs = _index(chunks, progress, on_usage)
        if kind in thumbs.KINDS:
            progress.stage("Making a thumbnail")
            thumbs.try_make(job["sha256"], data, kind)  # none (a typed icon) rather than a failed job
        progress.stage("Saving")
        info = _details(meta)
        result = {**used(who, note), "title": meta.get("title") or job["filename"], "chunks": len(chunks),
                  "analysis": analysis, "replaced": [], "unit_fixes": check["total"], "embedded": vecs is not None}
        if job["kind"] == "reread":
            db.replace_document(job["doc_id"], chunks, meta, analysis, label(*who), units=check, job=(job_id, result),
                                pages=pages, vectors=vecs, details=info, suggestion=info["folder"])
        else:
            db.add_document(job["filename"], chunks, meta, analysis, label(*who), sha=job["sha256"],
                            size=job["file_size"], mime=job["mime"], replace=opts.get("on_duplicate") == "replace",
                            units=check, job=(job_id, result), pages=pages, vectors=vecs, details=info,
                            folder_id=opts.get("folder_id"), auto_file=bool(opts.get("auto_file")),
                            suggestion=info["folder"], confidence=info["confidence"])
        status = "done"
    except AwaitConfirmation:
        status = "awaiting_confirmation"
    except Cancelled:
        status = "cancelled"
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            e = RuntimeError("its original file is no longer stored; upload it again")
        elif not isinstance(e, (NoText, extract.Unsupported, llm.MissingKey, db.JobGone, anthropic.APIError,
                                ollama.OllamaError, AnalysisFailed, pypdfium2.PdfiumError)):  # explained by the job's error
            log.exception("Job %d (%s) failed", job_id, job["filename"])
        _fail(job_id, describe(e, job["filename"]))
    finally:
        if status != "awaiting_confirmation":
            _keys.pop(job_id, None)
    if status != "done" and job["kind"] == "reread":  # its document's calls, unless it was deleted
        doc_id = (db.get_job(job_id) or {}).get("doc_id")
        if doc_id:
            try:
                db.link_usage(job_id, doc_id)
            except Exception:
                log.exception("Could not link the usage of job %d", job_id)


_AMOUNT = re.compile(r"-?\d{1,12}([.,]\d{1,4})?")


def _details(meta):
    """The key details and filing suggestion of an analysis (META_SCHEMA's strings), checked:
    {document_date, vendor, total_amount, currency, due_date, folder, confidence}, None for
    anything missing or not what it should be. Older readings and test fakes lack the keys.
    Dates must be real YYYY-MM-DD dates (1900-2100). An amount is digits with a dot, or a
    comma when there is no dot, for decimals (currency signs and spaces dropped); a comma
    before exactly three digits could be a thousands separator ("1,234"), so it is left
    out, as is "1.234,50". The currency, three letters, is kept only with an amount. The
    folder is a path of at most 3 clean names (db.clean_folder_name: bad levels dropped);
    its confidence is "high" only when the model said so."""
    def text(key):
        value = meta.get(key, "")
        if isinstance(value, (int, float)) and not isinstance(value, bool):  # an amount given as a number
            value = str(value)
        return value.strip() if isinstance(value, str) else ""

    raw = "".join(c for c in text("total_amount") if not c.isspace() and unicodedata.category(c) != "Sc")
    amount = None
    if _AMOUNT.fullmatch(raw) and not re.search(r",\d{3}$", raw):
        value = float(raw.replace(",", "."))
        amount = value if math.isfinite(value) and abs(value) < 1e12 else None
    currency = text("currency").upper()
    vendor = " ".join(text("vendor").split())[:200].strip()
    folder = []
    for segment in text("folder").split("/"):
        try:
            folder.append(db.clean_folder_name(segment))
        except db.BadFolderName:
            continue
    return {"document_date": db.parse_date(text("document_date")), "due_date": db.parse_date(text("due_date")),
            "vendor": vendor or None, "total_amount": amount,
            "currency": currency if amount is not None and re.fullmatch(r"[A-Z]{3}", currency) else None,
            "folder": "/".join(folder[:3]) or None, "confidence": "high" if meta.get("folder_confidence") == "high" else "low"}


def _fail(job_id, error):
    """Mark a job failed (unless it was cancelled meanwhile). While the database stays locked
    past its busy timeout (say a huge document being saved or deleted), try again a few
    times: a job left 'running' would show its last stage until a restart, with nothing
    working on it, since only queued jobs are picked up."""
    for wait in (*FAIL_RETRY_SECONDS, None):
        try:
            db.update_job(job_id, status="failed", stage="Failed", error=error)
            return
        except sqlite3.OperationalError as e:
            if wait is None:
                raise
            log.warning("Could not mark job %d failed (%s); trying again in %d s", job_id, e, wait)
            time.sleep(wait)


def _index(chunks, progress, on_usage):
    """The chunks' vectors for search by meaning (vectors.for_chunks), or None when the
    embedding model is unavailable: the document is still added, searchable by keywords,
    and the worker embeds it later (vectors.backfill)."""
    if not vectors.indexing():
        return None

    def say(done, total=len(chunks)):
        progress.stage(f"Indexing for search by meaning: {done} of {total} passages", done, total)

    say(0)
    return vectors.for_chunks(chunks, on_usage=on_usage, on_progress=say)


def _read(job, opts, data, progress, on_usage):
    """Read and analyze the file with the job's model choice. Auto works locally and
    switches to Claude when Ollama fails and a key is available, estimating Claude's
    cost first: Claude reads the file when the local reading failed, and only analyzes
    the text read here when just the local analysis failed. Returns (text, meta,
    analysis, (backend, model) that analyzed it, why it switched)."""
    name, key = job["filename"], _keys.get(job["id"]) or os.environ.get("ANTHROPIC_API_KEY", "")
    is_pdf = extract.kind(name) == "pdf" and PDF_EXTRACTOR != "local"
    mode = opts.get("claude_pages") or CLAUDE_PDF_PAGES
    backend, _, model = job["model"].partition(":")
    reading = (job, data, is_pdf, mode, progress, on_usage)

    def with_claude(model, text=None):
        """Claude analyzes `text`; with none, it reads the file first."""
        _check_cost(job, opts, data, model, is_pdf, mode, progress, text)
        if not key:
            raise llm.MissingKey()
        client = llm.client_for(key)
        if text is None:
            text = _text("claude", model, client, *reading)
        return (text, *_analyze("claude", model, client, job, text, progress, on_usage), ("claude", model),
                opts.get("switched"))

    if backend == "claude":
        return with_claude(model)
    if backend == "ollama":
        text = _text("ollama", model, None, *reading)
        return (text, *_analyze("ollama", model, None, job, text, progress, on_usage), ("ollama", model), None)
    # Auto. "fallback" says how it switched before the user confirmed Claude's cost.
    fallback = "read" if opts.get("fallback") is True else opts.get("fallback")
    text = None
    if fallback != "read":
        try:
            text = _text("ollama", ollama.MODEL, None, *reading)  # pages read before are kept: no new calls
            if fallback != "analyze":
                return (text, *_analyze("ollama", ollama.MODEL, None, job, text, progress, on_usage),
                        ("ollama", ollama.MODEL), None)
        except ollama.OllamaError as e:
            if not key:
                raise
            opts.update(fallback="read" if text is None else "analyze", switched=f"local model failed ({e})")
            if not db.update_job(job["id"], options=opts):
                raise Cancelled()
    return with_claude(llm.MODEL, text)


def _analysis_calls(chars, n, read):
    """(input, output) tokens of each call that analyzes `chars` characters with Claude in
    `n` sections, `read` of them read: one call, or one per section read plus one
    combining their notes."""
    if n == 1:
        return [(chars // 4 + 100, ANALYSIS_OUTPUT_TOKENS)]
    return [(chars // n // 4 + 400, NOTES_OUTPUT_TOKENS)] * read + [
        (read * NOTES_OUTPUT_TOKENS + 300, ANALYSIS_OUTPUT_TOKENS)]


def estimate(name, data, model, is_pdf, mode, text=None):
    """What reading a file with Claude `model` should cost: pages, pages Claude reads,
    tokens and USD, from about 2,000 input and 600 output tokens per page Claude reads
    plus the analysis (one call, or for a long text one per section and one more). A photo
    is one call reading it (its pixels, PHOTO_MAX_TOKENS at most, and its text) and the
    analysis of about PHOTO_TEXT_CHARS. With `text` (read already), just the analysis. Each
    call is priced on its own, as the API bills it. A model missing from llm.PRICES is
    priced like the dearest one there (and marked), so the confirmation still works."""
    pages = read = 0
    photo = None  # (input, output) tokens of reading a photo
    if text is not None:
        chars, is_pdf = len(extract.plain(text)), False
    elif extract.kind(name) == "image":
        try:
            w, h = extract.image_size(data)
        except Exception:  # checked at upload; should its header still not read, count the most
            w = h = extract.CLAUDE_PHOTO_SIDE
        s = min(1.0, extract.CLAUDE_PHOTO_SIDE / max(w, h, 1))
        photo = (round(min(PHOTO_MAX_TOKENS, w * s * h * s / 750)) + PHOTO_PROMPT_TOKENS, PHOTO_OUTPUT_TOKENS)
        chars, is_pdf = PHOTO_TEXT_CHARS, False
        text = "x" * chars
    elif is_pdf:
        texts = extract.pdf_texts(data)
        reads = [mode == "all" or extract.is_scan(t) for t in texts]
        pages, read = len(texts), sum(reads)
        chars = sum(len(t) for t, r in zip(texts, reads) if not r) + read * PAGE_OUTPUT_TOKENS * 4
        # Sections are counted on the text to come, each page Claude reads standing in as text
        # of the length it should write: split() keeps pages whole, which can take more
        # sections than the characters alone suggest.
        text = extract.join_pages("word " * (PAGE_OUTPUT_TOKENS * 4 // 5) if r else t for t, r in zip(texts, reads))
    else:
        text = extract.extract_text(name, data)
        chars = len(text)
    price = model if model in llm.PRICES else max(llm.PRICES, key=lambda m: llm.PRICES[m][1])
    parts = sections.plan(text, llm.ANALYZE_CHARS)
    calls = _analysis_calls(chars, *parts)
    cost = sum(llm.cost(price, {"input": i, "output": o}) for i, o in calls)
    for start in range(0, read, extract.PDF_PAGES_PER_CALL):
        n = min(extract.PDF_PAGES_PER_CALL, read - start)
        cost += llm.cost(price, {"input": n * PAGE_INPUT_TOKENS, "output": n * PAGE_OUTPUT_TOKENS})
    if photo:
        cost += llm.cost(price, {"input": photo[0], "output": photo[1]})
        calls = [photo, *calls]
    return {"model": model, "model_label": llm.label(model), "mode": mode if is_pdf else None,
            "pages": pages, "claude_pages": read, "image": photo is not None,
            "input_tokens": read * PAGE_INPUT_TOKENS + sum(i for i, _ in calls),
            "output_tokens": read * PAGE_OUTPUT_TOKENS + sum(o for _, o in calls), "cost_usd": round(cost, 6),
            "sections": list(parts), "priced": model in llm.PRICES,
            "confirm_at": CONFIRM_COST_USD}


def _check_cost(job, opts, data, model, is_pdf, mode, progress, text=None):
    """Estimate the Claude reading (or with `text`, just the analysis) and keep it on
    the job. At CONFIRM_COST_USD or more, the job stops here and waits for the user,
    unless they confirmed at least this much already ("confirmed" holds that amount: a
    retry that would cost more, say Auto now has Claude read instead of just analyze,
    asks again)."""
    est = job["estimate"]
    photo = text is None and extract.kind(job["filename"]) == "image"
    if not est or (est["model"], est["mode"], est.get("image", False)) != (
            model, mode if is_pdf and text is None else None, photo):
        progress.stage("Estimating the cost")
        est = estimate(job["filename"], data, model, is_pdf, mode, text)
    confirmed = opts.get("confirmed")
    ask = est["cost_usd"] >= CONFIRM_COST_USD and not (
        confirmed is True or (confirmed is not None and est["cost_usd"] <= confirmed))
    wait = {"status": "awaiting_confirmation", "stage": "Waiting for you to confirm the cost"} if ask else {}
    if not db.update_job(job["id"], estimate=est, **wait):
        raise Cancelled()
    if ask:
        raise AwaitConfirmation()


def _text(backend, model, client, job, data, is_pdf, mode, progress, on_usage):
    """The file's text, read by one model. PDF pages stay apart with "\\f". A photo is read
    by the local vision model, or by Claude: one call either way, and no pages."""
    if extract.kind(job["filename"]) == "image":
        if backend == "ollama":
            progress.stage(f"Reading the photo with {ollama.VISION_MODEL}")
            text = ollama.transcribe_image(extract.prepare_image(data, extract.OLLAMA_PHOTO_SIDE), on_usage)
        else:
            progress.stage(f"Reading the photo with {llm.label(model)}")
            text = llm.transcribe_image(client, extract.prepare_image(data, extract.CLAUDE_PHOTO_SIDE), model,
                                        on_usage)
        text = text.replace(extract.PAGE_BREAK, "\n")
        if not extract.chunk_text(text):
            raise NoText("No text found in the photo")
        return text
    if not is_pdf:
        progress.stage("Reading the file")
        text = extract.extract_text(job["filename"], data)
    elif backend == "ollama":
        text = _ollama_pages(data, progress, on_usage)
    else:
        text = _claude_pages(client, data, model, mode, progress, on_usage)
    if not extract.chunk_text(text):
        raise NoText()
    return text


def _analyze(backend, model, client, job, text, progress, on_usage):
    """(meta, analysis) for the text. Ollama errors are raised, so Auto can switch to
    Claude. Other failures (a refusal, a reply that isn't JSON, an API still overloaded
    after its retries) leave a new upload without metadata, its text still searchable.
    A job that replaces documents, a re-read or an upload told to replace its copies,
    fails instead: they keep their metadata, and a retry finds its pages kept."""
    name, who = job["filename"], label(backend, model)
    progress.stage(f"Analyzing with {who}")
    # A long document is analyzed section by section (sections.py): each step shows in the
    # job, and each section's notes are kept with it, so a retry doesn't ask for them again.
    notes = f"notes:{backend}:{model}"
    hooks = {"on_usage": on_usage, "kept": progress.kept(notes),
             "on_progress": lambda stage, done, total: progress.stage(f"{stage} with {who}", done, total),
             "on_notes": lambda key, got: progress.keep(notes, {key: got}),
             "folders": db.folder_paths()}  # where it may be filed (sections.analyze)
    try:
        if backend == "ollama":
            return ollama.analyze(None, name, text, model, **hooks)
        return llm.analyze(client, name, text, model, **hooks)
    except (anthropic.AuthenticationError, llm.MissingKey, ollama.OllamaError, Cancelled):
        raise
    except Exception as e:
        if job["kind"] == "reread" or (job["options"] or {}).get("on_duplicate") == "replace":
            raise AnalysisFailed() from e
        return {"title": name}, f"none: {e}"


def _ollama_pages(data, progress, on_usage):
    """Text layer pages as they are; the vision model reads the scanned ones, each kept
    as soon as it is read."""
    reader = _vision_reader()

    def on_page(n, total, text, read):
        if text is None:  # the vision model starts on page n
            progress.stage(f"Reading page {n} of {total}", n - 1, total)
        elif read:
            progress.keep(reader, {n: text}, f"Read page {n} of {total}", n, total)
        else:
            progress.stage(f"Reading page {n} of {total}", n, total, force=n == total)

    return extract.join_pages(ollama.transcribe_pdf(data, on_usage=on_usage, done=progress.pages(reader),
                                                    on_page=on_page))


def _claude_pages(client, data, model, mode, progress, on_usage):
    """Claude reads the scanned pages (or, in mode 'all', every page) in batches of new
    PDFs holding just those pages; the other pages keep their text layer. Each finished
    batch is kept, so a retry sends only the batches still missing."""
    reader = _claude_reader(model, mode)
    texts = extract.pdf_texts(data)
    pages = dict(enumerate(texts, 1))
    want = [n for n, t in pages.items() if mode == "all" or extract.is_scan(t)]
    done = {n: t for n, t in progress.pages(reader).items() if n in pages}
    pages.update(done)
    todo = [n for n in want if n not in done]
    if todo:
        count = len(done.keys() & set(want))
        what = f"{len(want)} {'' if mode == 'all' else 'scanned '}page{'s' if len(want) > 1 else ''}"

        def say(k):
            return f"Reading {what} with {llm.label(model)}: {k} done"

        def on_batch(got):
            nonlocal count
            count += len(got)
            progress.keep(reader, got, say(count), count, len(want))

        progress.stage(say(count), count, len(want))
        pages.update(llm.transcribe_pdf(client, extract.pdf_batches(data, pages=todo), model,
                                        on_usage=on_usage, on_batch=on_batch))
    return extract.join_pages(pages[n] for n in range(1, len(texts) + 1))
