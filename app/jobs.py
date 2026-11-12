"""Background reading. Uploads and re-reads wait in the jobs table, and one worker thread
per process works through them in order, saving progress as it goes. Pages a model has
read are kept with the job, so a job that failed, or was cut off by a restart, carries on
where it stopped instead of paying for those pages again. Flask-free, like llm.py and
ollama.py: the routes that create jobs and report on them are in main.py."""
import logging
import os
import threading
import time

import anthropic

from . import db, extract, llm, ollama

log = logging.getLogger(__name__)

# "model" (default): the chosen model reads PDFs (Ollama's vision model only the scanned
# pages; Claude see CLAUDE_PDF_PAGES). "local": the PDF's embedded text layer only, so
# scanned pages come out empty.
PDF_EXTRACTOR = os.environ.get("PDF_EXTRACTOR", "model")
# Which PDF pages Claude reads. "scanned" (default): only pages with little or no text
# layer (extract.is_scan); the others' text layer is used as it is. "all": every page,
# as text and image, which is worth its cost for complex tables and layouts.
CLAUDE_PDF_PAGES = os.environ.get("CLAUDE_PDF_PAGES", "scanned")
# A job whose Claude reading is estimated to cost at least this much (USD) waits for
# the user to confirm it.
CONFIRM_COST_USD = float(os.environ.get("CONFIRM_COST_USD", "1.00"))
# Rough sizes behind an estimate: tokens per page Claude reads (image plus text) and
# writes, and the metadata reply. The metadata call's input is about 4 characters a token.
PAGE_INPUT_TOKENS, PAGE_OUTPUT_TOKENS, ANALYSIS_OUTPUT_TOKENS = 2000, 600, 500
# Tests turn the thread off and run the queue in their own thread with drain().
BACKGROUND = True

_keys = {}  # job id -> the browser's Claude key: in memory only, never stored
_wake, _stop = threading.Event(), threading.Event()
_thread = None
_start_lock = threading.Lock()


class NoText(ValueError):
    pass


class Cancelled(Exception):
    pass


class AwaitConfirmation(Exception):
    """The estimate is at or above CONFIRM_COST_USD: the job waits for the user."""


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
    """The browser's key for this job, until it ends. A restart forgets it: a job that
    then needs Claude fails asking for the key, and Retry sends it again."""
    if key:
        _keys[job_id] = key


def describe(e, name):
    """A failed job's error, worded like the API's error responses."""
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
        return "No text found in file"
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
    n = db.requeue_running()
    if n:
        log.warning("Re-queued %d job(s) interrupted by a restart", n)
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


def _loop():
    while not _stop.is_set():
        _wake.clear()  # before looking, so a job queued meanwhile still wakes the wait below
        try:
            if run_next():
                continue
        except Exception:  # say the database stayed locked: try again shortly
            log.exception("Job worker error")
        _wake.wait(30)


def run_next():
    """Run the oldest queued job in this thread. False if nothing was queued."""
    job = db.claim_next_job()
    if not job:
        return False
    _process(job)
    return True


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
        return {int(n): t for n, t in self.partial.get(reader, {}).items()}

    def keep(self, reader, pages, text, done, total):
        self.partial.setdefault(reader, {}).update({str(n): t for n, t in pages.items()})
        self._write({"partial": self.partial, "stage": text, "progress": done, "total": total})

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
        progress.stage("Saving")
        chunks = extract.chunk_text(text)
        result = {**used(who, note), "title": meta.get("title") or job["filename"],
                  "chunks": len(chunks), "analysis": analysis, "replaced": []}
        if job["kind"] == "reread":
            db.replace_document(job["doc_id"], chunks, meta, analysis, label(*who), job=(job_id, result))
        else:
            db.add_document(job["filename"], chunks, meta, analysis, label(*who), sha=job["sha256"],
                            size=job["file_size"], mime=job["mime"], replace=opts.get("on_duplicate") == "replace",
                            job=(job_id, result))
        status = "done"
    except AwaitConfirmation:
        status = "awaiting_confirmation"
    except Cancelled:
        status = "cancelled"
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            e = RuntimeError("its original file is no longer stored; upload it again")
        elif not isinstance(e, (NoText, extract.Unsupported, llm.MissingKey, db.JobGone, anthropic.APIError,
                                ollama.OllamaError)):  # those are explained by the job's error
            log.exception("Job %d (%s) failed", job_id, job["filename"])
        db.update_job(job_id, status="failed", stage="Failed", error=describe(e, job["filename"]))
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


def _read(job, opts, data, progress, on_usage):
    """Read and analyze the file with the job's model choice. Auto reads locally and
    switches to Claude when Ollama fails and a key is available, estimating Claude's
    cost first. Returns (text, meta, analysis, (backend, model), why it switched)."""
    name, key = job["filename"], _keys.get(job["id"]) or os.environ.get("ANTHROPIC_API_KEY", "")
    is_pdf = name.lower().endswith(".pdf") and PDF_EXTRACTOR != "local"
    mode = opts.get("claude_pages") or CLAUDE_PDF_PAGES
    backend, _, model = job["model"].partition(":")
    args = (job, data, key, is_pdf, mode, progress, on_usage)
    if backend == "auto" and opts.get("fallback"):  # switched before, and the user confirmed the cost
        backend, model = "claude", llm.MODEL
    if backend == "claude":
        _check_cost(job, opts, data, model, is_pdf, mode, progress)
        return (*_read_with("claude", model, *args), ("claude", model), opts.get("switched"))
    if backend == "ollama":
        return (*_read_with("ollama", model, *args), ("ollama", model), None)
    try:
        return (*_read_with("ollama", ollama.MODEL, *args), ("ollama", ollama.MODEL), None)
    except ollama.OllamaError as e:
        if not key:
            raise
        note = f"local model failed ({e})"
        opts.update(fallback=True, switched=note)
        if not db.update_job(job["id"], options=opts):
            raise Cancelled()
        _check_cost(job, opts, data, llm.MODEL, is_pdf, mode, progress)
        return (*_read_with("claude", llm.MODEL, *args), ("claude", llm.MODEL), note)


def estimate(name, data, model, is_pdf, mode):
    """What reading a file with Claude `model` should cost: pages, pages Claude reads,
    tokens and USD, from about 2,000 input and 600 output tokens per page Claude reads
    plus the metadata call. Each call is priced on its own, as the API bills it."""
    pages = read = 0
    if is_pdf:
        texts = extract.pdf_texts(data)
        reads = [t for t in texts if mode == "all" or extract.is_scan(t)]
        pages, read = len(texts), len(reads)
        chars = sum(map(len, texts)) - sum(map(len, reads)) + read * PAGE_OUTPUT_TOKENS * 4
    else:
        chars = len(extract.extract_text(name, data))
    analysis_in = min(chars, llm.ANALYZE_CHARS) // 4 + 100
    cost = llm.cost(model, {"input": analysis_in, "output": ANALYSIS_OUTPUT_TOKENS})
    for start in range(0, read, extract.PDF_PAGES_PER_CALL):
        n = min(extract.PDF_PAGES_PER_CALL, read - start)
        cost += llm.cost(model, {"input": n * PAGE_INPUT_TOKENS, "output": n * PAGE_OUTPUT_TOKENS})
    return {"model": model, "model_label": llm.label(model), "mode": mode if is_pdf else None,
            "pages": pages, "claude_pages": read, "input_tokens": read * PAGE_INPUT_TOKENS + analysis_in,
            "output_tokens": read * PAGE_OUTPUT_TOKENS + ANALYSIS_OUTPUT_TOKENS, "cost_usd": round(cost, 6),
            "confirm_at": CONFIRM_COST_USD}


def _check_cost(job, opts, data, model, is_pdf, mode, progress):
    """Estimate the Claude reading and keep it on the job. At CONFIRM_COST_USD or more,
    and not confirmed yet, the job stops here and waits for the user."""
    est = job["estimate"]
    if not est or (est["model"], est["mode"]) != (model, mode if is_pdf else None):
        progress.stage("Estimating the cost")
        est = estimate(job["filename"], data, model, is_pdf, mode)
    ask = not opts.get("confirmed") and est["cost_usd"] >= CONFIRM_COST_USD
    wait = {"status": "awaiting_confirmation", "stage": "Waiting for you to confirm the cost"} if ask else {}
    if not db.update_job(job["id"], estimate=est, **wait):
        raise Cancelled()
    if ask:
        raise AwaitConfirmation()


def _read_with(backend, model, job, data, key, is_pdf, mode, progress, on_usage):
    """(text, meta, analysis) from one model. Text keeps PDF pages apart with "\\f"."""
    name, client = job["filename"], None
    if backend == "claude":
        if not key:
            raise llm.MissingKey()
        client = llm.client_for(key)
    if not is_pdf:
        progress.stage("Reading the file")
        text = extract.extract_text(name, data)
    elif backend == "ollama":
        text = _ollama_pages(data, progress, on_usage)
    else:
        text = _claude_pages(client, data, model, mode, progress, on_usage)
    if not extract.chunk_text(text):
        raise NoText()
    progress.stage(f"Analyzing with {label(backend, model)}")
    try:
        if backend == "ollama":
            meta, analysis = ollama.analyze(None, name, extract.plain(text), model, on_usage=on_usage)
        else:
            meta, analysis = llm.analyze(client, name, extract.plain(text), model, on_usage=on_usage)
    except (anthropic.AuthenticationError, llm.MissingKey, ollama.OllamaError, Cancelled):
        raise  # Ollama errors let Auto switch to Claude
    except Exception as e:  # still index the text so it is searchable
        meta, analysis = {"title": name}, f"none: {e}"
    return text, meta, analysis


def _ollama_pages(data, progress, on_usage):
    """Text layer pages as they are; the vision model reads the scanned ones, each kept
    as soon as it is read."""
    reader = f"ollama:{ollama.VISION_MODEL}"

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
    reader = f"claude:{model}:{mode}"
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
