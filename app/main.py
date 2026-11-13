"""HTTP API + single-page UI. Uploads and re-reads are queued as jobs (see jobs.py)."""
import itertools
import json
import logging
import math
import mimetypes
import os
import re
import threading
from urllib.parse import urlsplit

import anthropic
from flask import Flask, Response, jsonify, request, send_file, send_from_directory, stream_with_context
from werkzeug.routing import IntegerConverter

from . import auth, db, extract, jobs, llm, ollama, thumbs, vectors

log = logging.getLogger(__name__)


class RowId(IntegerConverter):
    """<id:...> in routes. SQLite ids are 64-bit, so a larger number names no row: it
    becomes 0, which no row has either, and the route answers its usual 404 instead of
    overflowing inside sqlite3."""
    def to_python(self, value):
        n = int(value)
        return n if n < 2**63 else 0


app = Flask(__name__, static_folder="static")
app.url_map.converters["id"] = RowId
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
# Default model choice when the browser doesn't send one:
#   "ollama": Auto. Local models on this machine's GPU (see app/ollama.py), switching to
#             Claude only when needed and a Claude key is available.
#   "claude": the Claude API (CLAUDE_MODEL), with the key from the browser or ANTHROPIC_API_KEY.
BACKEND = os.environ.get("LLM_BACKEND", "ollama")
# Kept originals shown in the browser: PDFs open in its viewer, plain text (Markdown too) as
# text, and photos as images. Everything else downloads (HEIC too, which browsers can't
# show), so an uploaded HTML or SVG file never runs script here.
INLINE_TYPES = {"application/pdf", "text/plain", "image/jpeg", "image/png", "image/webp"}
# Inline photos and text get a sandbox besides nosniff: nothing in them can run or load. Not
# PDFs, whose viewer it breaks.
SANDBOX_CSP = "sandbox; default-src 'none'; img-src 'self'; style-src 'unsafe-inline'"
# The page itself (GET /): its own scripts only (no inline script or handlers), no images or
# connections to other sites, so even a fault in the Markdown preview can't run or fetch anything.
PAGE_CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'; form-action 'self'")
# What the file picker offers (GET /api/models). The server takes any UTF-8 text file too.
UPLOAD_EXTS = (".pdf", ".docx", ".txt", ".md", ".markdown", ".csv", ".json", ".html", ".xml", ".log", ".py", ".js",
               ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")
# The slim image has no /etc/mime.types, and Python's own table lacks these two.
mimetypes.add_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx")
mimetypes.add_type("text/plain", ".log")
MissingKey = llm.MissingKey
# Most characters of a new conversation's title, taken from its first question.
TITLE_CHARS = 80
# Most characters of a question (about 5,000 tokens). Each one is kept, twice (as asked and
# as sent), and sent again with every later question of its conversation.
QUESTION_CHARS = 20000
# Streamed answers being written in this process, by (conversation id, answer id): how many
# characters of it the page showed when it stopped it (POST /api/chat/stop), None until then.
_live, _live_lock = {}, threading.Lock()


class BadChoice(ValueError):
    pass


def _key():
    # The key comes from the browser per request (or ANTHROPIC_API_KEY); never stored server-side.
    return request.headers.get("X-API-Key", "").strip()


def _has_key():
    return bool(_key() or os.environ.get("ANTHROPIC_API_KEY"))


def _client():
    if not _has_key():
        raise MissingKey()
    return llm.client_for(_key())


def _keep_key(job_id):
    """The browser's key goes with a job (in memory only), kept before the worker can
    pick the job up."""
    jobs.remember_key(job_id, _key())


def _add_job(*args, **kw):
    """db.add_job, the browser's key kept for the job before the worker can see it. The
    caller holds files_lock: if the job is never committed, its key goes before another
    job can be given the same id."""
    try:
        return db.add_job(*args, on_insert=_keep_key, **kw)
    except BaseException:
        jobs.prune_keys()
        raise


def _body():
    """The request's JSON object, or {} when it sent none. Anything else is a 400."""
    body = request.get_json(force=True, silent=True)
    if body is None:
        return {}
    if not isinstance(body, dict):
        raise BadChoice("Expected a JSON object")
    return body


def _choice(raw):
    """'auto' | 'ollama:<model>' | 'claude:<model>' -> (backend, model or None)."""
    if raw is not None and not isinstance(raw, str):
        raise BadChoice(f"Unknown model choice: {raw!r}")
    raw = (raw or "").strip() or ("auto" if BACKEND == "ollama" else f"claude:{llm.MODEL}")
    if raw == "auto":
        return "auto", None
    backend, _, model = raw.partition(":")
    if backend == "ollama" and model:
        return backend, model
    if backend == "claude" and model in dict(llm.MODELS):
        return backend, model
    raise BadChoice(f"Unknown model choice: {raw}")


def _choice_text(choice):
    """(backend, model) -> 'auto' | 'ollama:<model>' | 'claude:<model>', as jobs and conversations keep it."""
    backend, model = choice
    return "auto" if backend == "auto" else f"{backend}:{model}"


def _job_choice(raw):
    """A model choice for a job, checked now: a Claude model needs a key up front."""
    choice = _choice(raw)
    if choice[0] == "claude" and not _has_key():
        raise MissingKey()
    return _choice_text(choice)


def _claude_pages(raw):
    raw = raw or jobs.CLAUDE_PDF_PAGES
    if raw not in ("scanned", "all"):
        raise BadChoice("claude_pages must be 'scanned' or 'all'")
    return raw


def _label(backend, model):
    return jobs.label(backend, model)


def _run(choice, fn, fits_local=True):
    """Run fn(backend, model) for the chosen chat model. Auto runs locally and switches
    to Claude when the prompt is too long for the local model or Ollama fails, provided
    a Claude key is available. Returns (result, model used, why it switched or None).
    A streamed answer's fn starts the stream and waits for its first piece, so Ollama
    failing before it writes anything switches too (_answer_events). Uploads do the same
    in the background (jobs._read)."""
    backend, model = choice
    if backend != "auto":
        return fn(backend, model), (backend, model), None
    if not fits_local and _has_key():
        return fn("claude", llm.MODEL), ("claude", llm.MODEL), "too long for the local model"
    try:
        return fn("ollama", ollama.MODEL), ("ollama", ollama.MODEL), None
    except ollama.OllamaError as e:
        if not _has_key():
            raise
        return fn("claude", llm.MODEL), ("claude", llm.MODEL), f"local model failed ({e})"


def _job_out(j):
    """A job for the API: no stored path and no page texts, but how many pages a retry
    keeps: those of the reader it starts with (db.list_jobs counts them per reader)."""
    partial = j.pop("partial", None) or {}
    kept = j.pop("kept", None) or {reader: len(pages) for reader, pages in partial.items()}
    j.pop("file_path", None)
    j["model_label"] = jobs.choice_label(j["model"])
    j["pages_kept"] = kept.get(jobs.first_reader(j), 0)
    return j


def _clean_name(name):
    """Control characters (CR/LF from a crafted request) can't go into a response header."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", name or "").strip()


class NotFound(LookupError):
    pass


def _failure(e):
    """(what went wrong, HTTP status) for an error a request or a model call raises; None
    for an unexpected one. A streamed answer sends it as its error event."""
    if isinstance(e, MissingKey):
        return "Enter your Claude API key first", 401
    if isinstance(e, (BadChoice, extract.Unsupported)):
        return str(e), 400
    if isinstance(e, NotFound):
        return str(e), 404
    if isinstance(e, anthropic.AuthenticationError):
        return "Invalid Claude API key", 401
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limited by the Claude API, try again shortly", 429
    if isinstance(e, anthropic.APIStatusError):
        return f"Claude API error: {e.message}", 502
    if isinstance(e, anthropic.APIConnectionError):
        return "Could not reach the Claude API", 502
    if isinstance(e, llm.Refused):
        return f"{e}. Try rewording it, or another model.", 502
    if isinstance(e, ollama.OllamaUnavailable):
        return f"{e}. Is it running?", 502
    if isinstance(e, ollama.OllamaError):
        return f"Ollama error: {e}", 502
    return None


# An answer that failed for a reason _failure doesn't know: the details (paths, say) stay in
# the server log, as with /api/chat's plain 500.
UNEXPECTED = "The answer failed (the server log has the details)"


def _error_reply(e):
    error, status = _failure(e)
    return jsonify(error=error), status


for _cls in (MissingKey, BadChoice, extract.Unsupported, NotFound, anthropic.AuthenticationError,
             anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError, llm.Refused,
             ollama.OllamaError):  # the most specific handler wins, so each gets _failure's wording
    app.register_error_handler(_cls, _error_reply)


@app.errorhandler(db.FolderError)
def _folder_error(e):
    """A folder change that can't be made: a bad name (400), a folder that is gone (404), a
    folder put inside itself (400), a name already used there (409, naming that folder), a
    delete whose documents changed meanwhile (409, with the counts now)."""
    if isinstance(e, db.FolderConflict):
        return jsonify(error=str(e), conflict=e.existing), 409
    if isinstance(e, db.FolderChanged):
        return jsonify(error=str(e), counts=e.counts), 409
    status = 404 if isinstance(e, db.MissingFolder) else 409 if isinstance(e, db.NoSuggestion) else 400
    return jsonify(error=str(e)), status


@app.errorhandler(413)
def _too_large(e):
    return jsonify(error=f"File is larger than the {MAX_UPLOAD_MB} MB upload limit (MAX_UPLOAD_MB)"), 413


@app.before_request
def _same_origin():
    """A page on another site can make a browser send simple POSTs here (a form, or a
    fetch with a text body), say to start a job waiting for its cost to be confirmed, or
    to delete documents. Browsers name the sending page's origin on every POST; it must
    be this server. Clients without one (curl, scripts) are not browsers and pass."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    origin = request.headers.get("Origin")
    if origin is not None and urlsplit(origin).netloc != request.host:
        return jsonify(error="Requests from other sites are not accepted"), 403
    return None


@app.before_request
def _start_worker():
    jobs.start()  # once per process (gunicorn starts it at boot); queued jobs carry on after a restart


# The optional login (APP_PASSWORD): checked after the two hooks above, so a request from
# another site still gets its 403 and any request, the login page's too, starts the worker.
auth.setup(app)


@app.get("/")
def index():
    """The page: markup only, its script and styles in /static (app.css, md.js, app.js, docs.js),
    so the CSP can refuse inline script, and images and connections to other sites."""
    resp = send_from_directory(app.static_folder, "index.html")
    resp.headers["Content-Security-Policy"] = PAGE_CSP
    return resp


@app.get("/api/models")
def models():
    """What the model pickers offer. Claude needs a key: the browser's or the server's."""
    try:
        local, local_error = ollama.models(), None
    except ollama.OllamaError as e:
        local, local_error = [], str(e)
    default = "auto" if BACKEND == "ollama" else f"claude:{llm.MODEL}"
    return jsonify(
        default=default,
        auto={"local": ollama.MODEL, "claude": llm.label(llm.MODEL)},
        local=local, local_error=local_error, vision=ollama.VISION_MODEL,
        claude=[{"id": m, "label": name} for m, name in llm.MODELS],
        server_key=bool(os.environ.get("ANTHROPIC_API_KEY")),
        max_upload_mb=MAX_UPLOAD_MB, max_question_chars=QUESTION_CHARS,
        claude_pages=jobs.CLAUDE_PDF_PAGES, confirm_cost_usd=jobs.CONFIRM_COST_USD,
        upload_types={"extensions": [e for e in UPLOAD_EXTS if extract.HEIC or e not in (".heic", ".heif")],
                      "heic": extract.HEIC},
    )


def _int_arg(name, default, low, high):
    """A whole-number query parameter from low to high, `default` when absent; else a 400."""
    raw = request.args.get(name, "").strip()
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        n = None
    if n is None or not low <= n <= high:
        raise BadChoice(f"{name} must be a whole number from {low} to {high}")
    return n


def _row_id(value, what):
    """An id from a JSON body: a whole number (None for none). One too large for SQLite names
    no row: it becomes 0, which none has either, so the usual 404 follows (RowId)."""
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise BadChoice(f"{what} must be a folder id or null")
    return value if -2**63 <= value < 2**63 else 0


def _folder_filter():
    """The documents a listing or search is limited to by ?folder=<id> (and, with
    ?recursive=1, the default, the folders inside it) or ?folder=unfiled: their ids, or None
    without the parameter. A folder that doesn't exist is a 404."""
    raw = request.args.get("folder", "").strip()
    if not raw:
        return None
    recursive = request.args.get("recursive", "1").strip()
    if recursive not in ("0", "1"):
        raise BadChoice("recursive must be 1 or 0")
    if raw != "unfiled":
        try:
            raw = _row_id(int(raw), "folder")
        except ValueError:
            raise BadChoice("folder must be a folder id or 'unfiled'") from None
    return db.folder_documents([raw], recursive == "1")


@app.get("/api/documents")
def documents():
    """Every document, newest first; with q, those whose metadata matches it, best first (at
    most `limit`, 10 by default). Either limited to a folder: ?folder=<id>|unfiled, with
    ?recursive=0 for its own documents only."""
    q, only = request.args.get("q", "").strip(), _folder_filter()
    if q:
        return jsonify(db.search_documents(q, _int_arg("limit", 10, 1, 200), only))
    return jsonify(db.list_documents(only))


def _snippet(raw):
    """A passage's snippet as search gives it: `snippet_parts`, plain text alternating with
    the words found (extract.snippet_parts), and `snippet`, the same text with those words
    [in brackets], as before parts were given (a passage's own brackets look the same)."""
    parts = extract.snippet_parts(raw, db.MATCH_START, db.MATCH_END)
    return {"snippet": "".join(f"[{p}]" if i % 2 else p for i, p in enumerate(parts)), "snippet_parts": parts}


@app.get("/api/search")
def search():
    """Passages with the words of q, for the library's search box: by keywords only, since
    it marks the words it found. At most `limit` (10 by default), and limited to a folder as
    GET /api/documents is (?folder, ?recursive)."""
    only = _folder_filter()
    return jsonify([{**{k: h[k] for k in ("doc_id", "filename", "title", "idx", "page_start", "page_end")},
                     **_snippet(h["snippet"])}
                    for h in db.search_chunks(request.args.get("q", ""), limit=_int_arg("limit", 10, 1, 50),
                                              doc_ids=only)])


@app.get("/api/search/status")
def search_status():
    """Search by meaning: the embedding model, how many chunks it has embedded of how
    many, and why it is paused, if it is (keyword search only meanwhile)."""
    return jsonify(vectors.status())


@app.post("/api/upload")
def upload():
    """Keep the file and queue it for reading: 202 {job}. The page follows the job
    through GET /api/jobs. Duplicates are caught here, before any model spends time."""
    f = request.files.get("file")
    name = _clean_name(f.filename) if f else ""
    if not name:
        return jsonify(error="No file"), 400
    model = _job_choice(request.form.get("model"))
    # A file already in the library is refused (409) unless the browser says what to do:
    # "replace" deletes the earlier copies once the new one is read, "keep" adds it anyway.
    on_duplicate = request.form.get("on_duplicate", "")
    if on_duplicate not in ("", "replace", "keep"):
        raise BadChoice("on_duplicate must be 'replace' or 'keep'")
    # Where it goes: folder_id (empty: Unfiled), and, into Unfiled, whether a folder the
    # analysis is sure of may take it (auto_file, on unless "0"). See db.add_document.
    folder_id = request.form.get("folder_id", "").strip() or None
    if folder_id is not None:
        try:
            folder_id = _row_id(int(folder_id), "folder_id")
        except ValueError:
            raise BadChoice("folder_id must be a folder id, or empty for Unfiled") from None
        if not db.get_folder(folder_id):
            return jsonify(error="That folder was deleted", folder_gone=True), 404
    auto_file = request.form.get("auto_file", "1").strip() or "1"
    if auto_file not in ("1", "0"):
        raise BadChoice("auto_file must be 1 or 0")
    options = {"on_duplicate": on_duplicate, "claude_pages": _claude_pages(request.form.get("claude_pages")),
               "folder_id": folder_id, "auto_file": auto_file == "1"}
    data = f.read()
    photo = extract.check(name, data)  # a file that can never be read gets its 400 now, not a failed job later
    sha = db.file_hash(data)
    with db.files_lock:  # check and queue together: two uploads of one file can't both pass
        dups, reading = db.find_duplicates(sha), db.active_jobs(sha=sha)
        if reading and on_duplicate != "keep":  # a copy is still queued or being read: nothing to replace yet
            return jsonify(error=f"{name} is being read right now", duplicate={
                "in_progress": True, "filename": reading[0]["filename"], "started_at": reading[0]["created_at"],
                "job_id": reading[0]["id"], "copies": len(dups)}), 409
        if dups and not on_duplicate:
            d = dups[0]
            return jsonify(error=f"{name} is already in the library", duplicate={
                "id": d["id"], "filename": d["filename"], "title": d["title"], "created_at": d["uploaded_at"],
                "model": d["model"], "copies": len(dups), "folder_id": d["folder_id"]}), 409
        # A photo is kept as the type of its bytes (check), never what the browser said.
        job_id = _add_job("upload", name, sha, model, options, data=data,
                          mime=photo or mimetypes.guess_type(name)[0] or f.mimetype or "application/octet-stream")
    jobs.wake()
    return jsonify(job=_job_out(db.get_job(job_id))), 202


@app.post("/api/documents/<id:doc_id>/reread")
def reread(doc_id):
    """Queue a new reading of a document's kept original, with any model: when done it
    replaces the document's text, metadata and model in place (same id)."""
    body = _body()
    model = _job_choice(body.get("model"))
    options = {"claude_pages": _claude_pages(body.get("claude_pages"))}
    with db.files_lock:  # a delete can't slip in between the checks and the job
        f = db.stored_file(doc_id)
        if not f:
            if not db.document_exists(doc_id):
                return jsonify(error="Not found"), 404
            return jsonify(error="No original file is kept for this document, so it can't be re-read"), 409
        if db.active_jobs(doc_id=doc_id):
            return jsonify(error=f"{f['filename']} is already being re-read"), 409
        job_id = _add_job("reread", f["filename"], f["sha256"], model, options, size=f["size"],
                          mime=f["mime"], doc_id=doc_id)
    jobs.wake()
    return jsonify(job=_job_out(db.get_job(job_id))), 202


@app.get("/api/jobs")
def job_list():
    """Unfinished and failed jobs, and others that changed in the last hour."""
    return jsonify([_job_out(j) for j in db.list_jobs()])


@app.get("/api/jobs/<id:job_id>")
def job_get(job_id):
    j = db.get_job(job_id)
    return jsonify(_job_out(j)) if j else (jsonify(error="Not found"), 404)


@app.post("/api/jobs/<id:job_id>/cancel")
def job_cancel(job_id):
    """Cancel a job that hasn't finished, or dismiss a failed one. A running one stops at
    once if it waits on Ollama, else after its current step (a Claude call, say)."""
    j = db.cancel_job(job_id)
    jobs.interrupt()
    jobs.prune_keys()  # a job that won't run again needs no key
    if not j:
        return jsonify(error="Not found"), 404
    if j["status"] == "done":
        return jsonify(error="This job has already finished", job=_job_out(j)), 409
    return jsonify(job=_job_out(j))


# Confirm and Retry check the job's state, keep the key and queue the job under files_lock:
# a waiting or failed job only moves on through the routes, which all hold it, so the
# check still holds when the job is queued, and no other job's key is touched.

@app.post("/api/jobs/<id:job_id>/confirm")
def job_confirm(job_id):
    """Start a job that waits for its estimated Claude cost to be confirmed."""
    with db.files_lock:
        j = db.get_job(job_id)
        if not j:
            return jsonify(error="Not found"), 404
        if j["status"] != "awaiting_confirmation":
            return jsonify(error="This job is not waiting for a confirmation"), 409
        if not _has_key():  # it waits only to spend on Claude
            raise MissingKey()
        _keep_key(job_id)  # before it is queued
        # The amount confirmed: a retry that would cost more asks again (jobs._check_cost).
        db.update_job(job_id, when=("awaiting_confirmation",), status="queued", stage="Waiting to start",
                      options={**j["options"], "confirmed": (j["estimate"] or {}).get("cost_usd", 0)})
    jobs.wake()
    return jsonify(job=_job_out(db.get_job(job_id)))


@app.post("/api/jobs/<id:job_id>/retry")
def job_retry(job_id):
    """Queue a failed job again. Pages a model already read are kept and not paid for
    twice; Auto tries the local model first again."""
    with db.files_lock:
        j = db.get_job(job_id)
        if not j:
            return jsonify(error="Not found"), 404
        if j["status"] != "failed":
            return jsonify(error="Only a failed job can be retried"), 409
        if not os.path.isfile(os.path.join(db.files_dir(), os.path.basename(j["file_path"]))):
            return jsonify(error="The original file is no longer stored; upload it again"), 409
        if j["model"].startswith("claude:") and not _has_key():
            raise MissingKey()
        options = {k: v for k, v in j["options"].items() if k not in ("fallback", "switched")}
        _keep_key(job_id)  # before it is queued
        db.update_job(job_id, when=("failed",), status="queued", stage="Waiting to start", error=None,
                      options=options)
    jobs.wake()
    return jsonify(job=_job_out(db.get_job(job_id)))


@app.get("/api/documents/<id:doc_id>/file")
def document_file(doc_id):
    """A document's kept original: PDFs, plain text, Markdown (as plain text) and photos
    (not HEIC) open in the browser, everything else downloads."""
    f = db.stored_file(doc_id)
    if not f:  # also documents uploaded before originals were kept
        return jsonify(error="No original file is kept for this document"), 404
    mime = f["mime"]
    if mime == "text/markdown" or extract.kind(f["filename"]) == "markdown":
        mime = "text/plain"  # shown as it is written; the page renders it, sanitized
    inline = mime in INLINE_TYPES
    resp = send_file(f["path"], mimetype=mime, as_attachment=not inline, download_name=f["filename"], max_age=0)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    if inline and mime != "application/pdf":
        resp.headers["Content-Security-Policy"] = SANDBOX_CSP
    return resp


@app.get("/api/documents/<id:doc_id>/thumb")
def document_thumb(doc_id):
    """A document's thumbnail (thumbs.py), WebP: ?size=card (400 px wide, the default) or
    large (1200). Made now if missing. Cached for good when ?v= is its file's hash (as
    listings give it in `thumb`: ids are reused, hashes not), else revalidated by ETag. 404
    for a document with no kept file, of another kind than PDF or photo, or that can't be
    rendered: the page shows a typed icon instead. Never SVG."""
    size = request.args.get("size", "card")
    if size not in thumbs.SIZES:
        raise BadChoice("size must be 'card' or 'large'")
    f = db.stored_file(doc_id)
    kind = f and extract.kind(f["filename"])
    if not f or not f["sha256"] or kind not in thumbs.KINDS or thumbs.failed(f["sha256"]):
        return jsonify(error="No thumbnail"), 404
    sha = f["sha256"]
    tag = f"{sha[:16]}-{size}"
    if request.if_none_match.contains_weak(tag):
        resp = Response(status=304)
    else:
        target = thumbs.get(sha, f["path"], kind, size)
        if not target:
            return jsonify(error="No thumbnail"), 404
        resp = send_file(target, mimetype="image/webp", conditional=False, etag=False)
    resp.headers["ETag"] = f'"{tag}"'
    resp.headers["Cache-Control"] = ("private, max-age=31536000, immutable" if request.args.get("v") == sha[:12]
                                     else "private, no-cache")
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@app.get("/api/documents/<id:doc_id>/text")
def document_text_view(doc_id):
    """A document's text for its preview, at most `max` characters (200,000 by default):
    {text, truncated, source, kind, bytes}. Markdown and text files give their kept original
    as written ("file"); others (DOCX, PDFs, photos, or no kept file) the text indexed for
    them ("chunks"), as read."""
    most = _int_arg("max", 200_000, 1, 500_000)
    doc = db.get_document(doc_id)
    if not doc:
        return jsonify(error="Not found"), 404
    f = db.stored_file(doc_id) if doc["kind"] in ("markdown", "text") else None
    if f:
        with open(f["path"], "rb") as fh:
            raw = fh.read(most * 4 + 4)  # enough for `most` characters of any UTF-8
        size = os.path.getsize(f["path"])
        text = raw.decode("utf-8-sig", errors="replace")  # without the BOM some Windows editors start a file with
        return jsonify(text=text[:most], truncated=len(raw) < size or len(text) > most, source="file",
                       kind=doc["kind"], bytes=size)
    text = extract.plain(db.document_text(doc_id))
    return jsonify(text=text[:most], truncated=len(text) > most, source="chunks", kind=doc["kind"],
                   bytes=len(text.encode()))


@app.get("/api/documents/<id:doc_id>")
def document_get(doc_id):
    """A document as listed, plus pages (the last its text reaches; null without pages) and
    copies ([{id, folder_id}] of other documents holding the same file)."""
    doc = db.get_document(doc_id)
    return jsonify(doc) if doc else (jsonify(error="Not found"), 404)


def _text_field(body, key, most, empty=None, collapse=False):
    """body[key] trimmed (`collapse`: with each run of white space made one space), or BadChoice."""
    value = body[key]
    if value is None and empty is None:
        return None
    if not isinstance(value, str):
        raise BadChoice(f"{key} must be text")
    value = " ".join(value.split()) if collapse else value.strip()
    if len(value) > most:
        raise BadChoice(f"{key} is {most} characters at most")
    return value or empty


def _document_fields(body):
    """PATCH /api/documents/<id>'s body, checked: (fields for db.update_document, suggestion)."""
    unknown = sorted(set(body) - {*db.EDITABLE, "folder_id", "suggestion"})
    if unknown:
        raise BadChoice(f"Unknown field: {unknown[0]}")
    fields = {}
    if "title" in body:
        title = body["title"].strip() if isinstance(body["title"], str) else ""
        if not title:
            raise BadChoice("A title can't be empty")
        if len(title) > 300:
            raise BadChoice("A title is 300 characters at most")
        fields["title"] = title
    if "doc_type" in body:  # as the user typed it: analyses' types are tidied (jobs.doc_type), not theirs
        fields["doc_type"] = _text_field(body, "doc_type", 60, empty="", collapse=True)
    for key in ("document_date", "due_date"):
        if key in body:
            fields[key] = None if body[key] is None else db.parse_date(body[key])
            if body[key] is not None and fields[key] is None:
                raise BadChoice(f"{key} must be a date like 2026-11-03")
    if "vendor" in body:
        fields["vendor"] = _text_field(body, "vendor", 200)
    if "total_amount" in body:
        amount = body["total_amount"]
        if amount is not None and (not isinstance(amount, (int, float)) or isinstance(amount, bool)
                                   or not math.isfinite(amount) or abs(amount) >= 1e12):
            raise BadChoice("total_amount must be a number, or null")
        fields["total_amount"] = None if amount is None else float(amount)
    if "currency" in body:
        currency = _text_field(body, "currency", 3)
        currency = currency and currency.upper()
        if currency is not None and not re.fullmatch(r"[A-Z]{3}", currency):
            raise BadChoice("currency must be a three-letter code like GBP, or null")
        fields["currency"] = currency
    if "folder_id" in body:
        fields["folder_id"] = _row_id(body["folder_id"], "folder_id")
    suggestion = body.get("suggestion")
    if suggestion not in (None, "apply", "dismiss"):
        raise BadChoice("suggestion must be 'apply' or 'dismiss'")
    if suggestion == "apply" and "folder_id" in fields:
        raise BadChoice("Send folder_id or suggestion: apply, not both")
    return fields, suggestion


@app.patch("/api/documents/<id:doc_id>")
def document_edit(doc_id):
    """Change a document's title, type or key details (each then kept by re-reads), move it
    (folder_id, null for Unfiled), or accept ("apply", making its folders if need be) or
    dismiss its suggested folder. Returns it as listed, with `created` (folder ids) for apply."""
    fields, suggestion = _document_fields(_body())
    got = db.update_document(doc_id, fields, suggestion)
    if not got:
        return jsonify(error="Not found"), 404
    doc, created = got
    return jsonify({**doc, **({"created": created} if suggestion == "apply" else {})})


@app.post("/api/documents/move")
def documents_move():
    """Move documents: {"ids": [...], "folder_id": id or null (Unfiled)}, or {"ids": [...],
    "path": "Household/Bills"}, which makes the folders missing from the path. Returns
    {moved, folder_id, created, from: {id: the folder each was in}}, which an Undo needs."""
    body = _body()
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise BadChoice('Expected {"ids": [1, 2, ...]} and a folder_id or path')
    if ("folder_id" in body) == ("path" in body):
        raise BadChoice("Send a folder_id (null for Unfiled) or a path, not both")
    if "path" in body:
        out = db.move_documents(ids, path=db.clean_path(body["path"], most=db.FOLDER_DEPTH))
    else:
        out = db.move_documents(ids, folder_id=_row_id(body["folder_id"], "folder_id"))
    return jsonify({**out, "from": {str(k): v for k, v in out["from"].items()}})


# ---- folders ----

@app.get("/api/folders")
def folder_list():
    """Every folder, sorted by path: id, name, parent_id, path (names from the top), doc_count
    (its own documents), total_count (with its subfolders'), created_at, updated_at."""
    return jsonify(db.list_folders())


@app.post("/api/folders")
def folder_add():
    """Make a folder: {"name", "parent_id": id or null for the top level}. 201 with it."""
    body = _body()
    return jsonify(db.add_folder(body.get("name"), _row_id(body.get("parent_id"), "parent_id"))), 201


@app.patch("/api/folders/<id:folder_id>")
def folder_edit(folder_id):
    """Rename a folder ({"name"}) and/or move it ({"parent_id": id, or null for the top level})."""
    body = _body()
    changes = {}
    if "name" in body:
        changes["name"] = body["name"]
    if "parent_id" in body:
        changes["parent_id"] = _row_id(body["parent_id"], "parent_id")
    return jsonify(db.update_folder(folder_id, **changes))


@app.delete("/api/folders/<id:folder_id>")
def folder_delete(folder_id):
    """Delete a folder. ?contents=move (the default): its documents and folders go to its
    parent. ?contents=delete: everything in it is deleted too, documents included, only with
    ?confirm_documents= the number of documents that is (409 with the counts otherwise)."""
    contents = request.args.get("contents", "move")
    if contents not in ("move", "delete"):
        raise BadChoice("contents must be 'move' or 'delete'")
    confirm = request.args.get("confirm_documents", "").strip()
    try:
        confirm = int(confirm) if confirm else None
    except ValueError:
        raise BadChoice("confirm_documents must be a whole number") from None
    out = db.delete_folder(folder_id, contents, confirm)
    if contents == "delete":
        jobs.interrupt()  # re-reads of its documents were cancelled
        jobs.prune_keys()
    return jsonify(out)


@app.get("/api/documents/<id:doc_id>/units")
def document_units(doc_id):
    """The litre/gallon misprints found in a document (units.check): what each printed pair
    is corrected to before a model reads it. Documents listed by /api/documents give only
    their count, unit_fixes."""
    doc = jobs.unit_check(doc_id)
    return jsonify(doc) if doc else (jsonify(error="Not found"), 404)


@app.delete("/api/documents/<id:doc_id>")
def delete(doc_id):
    deleted = db.delete_document(doc_id)
    jobs.interrupt()  # its re-read, if any, was cancelled
    jobs.prune_keys()
    return ("", 204) if deleted else (jsonify(error="Not found"), 404)


@app.post("/api/documents/delete")
def delete_many():
    ids = _body().get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        return jsonify(error="Expected {\"ids\": [1, 2, ...]}"), 400
    deleted = db.delete_documents(ids)
    jobs.interrupt()
    jobs.prune_keys()
    return jsonify(deleted=deleted)


@app.get("/api/usage")
def usage_report():
    """Model calls, tokens and cost over the last `days` days, per model and per day."""
    try:
        days = int(request.args.get("days", "30"))
    except ValueError:
        days = 0
    if not 1 <= days <= 3660:
        raise BadChoice("days must be a whole number from 1 to 3660")
    summary = db.usage_summary(days)
    for m in summary["by_model"]:
        m["label"] = _label(m["backend"], m["model"])
    return jsonify(summary)


@app.get("/api/instructions")
def instructions():
    return jsonify(db.list_instructions())


def _rule_text(body):
    text = body.get("text")
    if text is None:
        return None
    if not isinstance(text, str) or not text.strip():
        raise BadChoice("Instruction text can't be empty")
    if len(text) > 4000:
        raise BadChoice("Instruction is too long (4,000 characters at most)")
    return text.strip()


@app.post("/api/instructions")
def add_instruction():
    body = _body()
    text = _rule_text(body)
    if text is None:
        raise BadChoice("Instruction text can't be empty")
    rule_id = db.add_instruction(text, bool(body.get("enabled", True)))
    return jsonify(next(r for r in db.list_instructions() if r["id"] == rule_id)), 201


@app.patch("/api/instructions/<id:rule_id>")
def edit_instruction(rule_id):
    body = _body()
    enabled = body.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise BadChoice("enabled must be true or false")
    if not db.update_instruction(rule_id, _rule_text(body), enabled):
        return jsonify(error="Not found"), 404
    return jsonify(next(r for r in db.list_instructions() if r["id"] == rule_id))


@app.delete("/api/instructions/<id:rule_id>")
def delete_instruction(rule_id):
    return ("", 204) if db.delete_instruction(rule_id) else (jsonify(error="Not found"), 404)


def _id_list(raw, error):
    """Ids from a JSON body, as given ([] when absent); those too large for SQLite are left
    out, as they can't name a row."""
    if raw is None:
        return [], []
    if not isinstance(raw, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in raw):
        raise BadChoice(error)
    return raw, [i for i in dict.fromkeys(raw) if -2**63 <= i < 2**63]


def _scope(raw_docs, raw_folders=None):
    """POST /api/chat's doc_ids and folder_ids: what a question is limited to, those deleted
    meanwhile left out. None (both absent or []) means every document. Else {ids: {id: file
    name} of every document searched, direct: those of the documents chosen themselves,
    folders: [{id, name, count}]}: a folder's documents, its subfolders' included, are
    looked up now, so documents added to it since it was chosen are searched too."""
    docs, doc_ids = _id_list(raw_docs, "doc_ids must be a list of document ids")
    folders, folder_ids = _id_list(raw_folders, "folder_ids must be a list of folder ids")
    if not docs and not folders:
        return None
    direct, chosen = db.document_names(doc_ids), db.scope_folders(folder_ids)
    if not direct and not chosen:
        if not folders:  # worded as before folders could be chosen
            raise BadChoice("None of the chosen documents is in the library any more: choose others, or ask all documents")
        raise BadChoice("None of the chosen folders or documents is in the library any more: choose others, "
                        "or ask all documents")
    ids = {**db.document_names([d for f in chosen for d in f["doc_ids"]]), **direct}
    if not ids:
        names = [f["name"] for f in chosen]
        raise BadChoice(f"{names[0]} has no documents yet: add some, or ask all documents" if len(names) == 1 else
                        f"These folders have no documents yet: {', '.join(names)}. Add some, or ask all documents")
    return {"ids": ids, "direct": direct, "folders": [{k: f[k] for k in ("id", "name", "count")} for f in chosen]}


def _cited(hits):
    """The excerpts a question was given, as the model read them, and their sources: one per
    document, in the order first retrieved, with its pages (none for documents read before
    pages were kept, and files without pages) and its excerpts (indexes into excerpts)."""
    excerpts, sources = [], {}
    for i, h in enumerate(hits):
        pages = list(range(h["page_start"], h["page_end"] + 1)) if h["page_start"] else []
        doc = {k: h[k] for k in ("doc_id", "filename", "title", "has_file", "pdf")}
        excerpts.append({**doc, "idx": h["idx"], "page_start": h["page_start"], "page_end": h["page_end"],
                         "match": h.get("match", "keywords"), "text": llm.excerpt_text(h)})
        source = sources.setdefault(h["doc_id"], {**doc, "pages": [], "excerpts": []})
        source["pages"] = sorted({*source["pages"], *pages})
        source["excerpts"].append(i)
    return excerpts, list(sources.values())


def _title(question):
    """A new conversation's title: its first question on one line, cut at a word."""
    title = " ".join(question.split())
    if len(title) <= TITLE_CHARS:
        return title
    return title[:TITLE_CHARS - 1].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"


def _question(body):
    """A question's request, checked: (question, model choice, scope, conversation or None,
    history). A saved conversation's history comes from the database, so a client's
    `history` is ignored then; without conversation_id (a new chat, or an API client that
    keeps its own), the client's is used: the turns earlier answers returned."""
    question = body.get("question")
    question = question.strip() if isinstance(question, str) else ""
    if not question:
        raise BadChoice("Empty question")
    if len(question) > QUESTION_CHARS:
        raise BadChoice(f"The question is too long ({QUESTION_CHARS:,} characters at most)")
    choice, scope = _choice(body.get("model")), _scope(body.get("doc_ids"), body.get("folder_ids"))
    if choice[0] == "claude" and not _has_key():  # before anything is kept
        raise MissingKey()
    conv_id = body.get("conversation_id")
    if conv_id is not None:
        if not isinstance(conv_id, int) or isinstance(conv_id, bool):
            raise BadChoice("conversation_id must be the id of a conversation")
        conv = -2**63 <= conv_id < 2**63 and db.get_conversation(conv_id, messages=False)
        if not conv:
            raise NotFound("This conversation was deleted.")
        return question, choice, scope, conv, db.history(conv_id)
    history = body.get("history") or []
    if not isinstance(history, list) or not all(
            isinstance(t, dict) and t.get("role") in ("user", "assistant") and isinstance(t.get("content"), str)
            for t in history):
        raise BadChoice("history must be a list of turns, as earlier answers returned them")
    return question, choice, scope, None, history


def _prompt(question, scope, history, on_usage):
    """What the model is sent for a question: (system prompt, the user turn with the passages
    that best match it, by keywords and by meaning (vectors.search), what the page shows of
    them (sources, excerpts, ...), and the history a local model is sent: the latest turns
    that fit its context with the question (ollama.recent), None if the question alone doesn't).
    Claude is sent all of it. So a long conversation stays on the local model with Auto,
    which switches to Claude only for a question too long for it however short the history."""
    system = llm.system_prompt(db.active_instructions())
    hits = vectors.search(question, int(os.environ.get("TOP_K", "5")), scope and list(scope["ids"]), on_usage=on_usage)
    # No passage at all ("Summarise everything in my library", or small talk): the model is
    # given the documents' titles and summaries instead, so it can say what the library holds.
    library = db.library_overview() if not hits and scope is None else None
    user_message = llm.build_user_message(question, hits, scope and list(scope["direct"].values()), library,
                                          scope and scope["folders"])
    excerpts, sources = _cited(hits)
    # What it was asked of: the documents chosen ([] when only folders were), their names, the
    # folders chosen, and how many documents that came to (all None: every document).
    found = {"sources": sources, "excerpts": excerpts,
             "overview": bool(library and library[1]),  # given the library's summaries instead of passages
             "scope": scope and list(scope["direct"]), "scope_names": scope and list(scope["direct"].values()),
             "scope_folders": scope and [{"id": f["id"], "name": f["name"]} for f in scope["folders"]],
             "scope_count": scope and len(scope["ids"])}
    local = ollama.recent(history, {"content": system}, {"content": user_message})
    return system, user_message, found, local


def _left_out(found, history, local, used):
    """found gains left_out: how many earlier questions of a long conversation the local
    model that answered was not given (0 for Claude, which is given them all). A local model
    picked by name gets none when the question alone is too long for it (local is None)."""
    cut = history[:len(history) - len(local or [])] if used[0] == "ollama" else []
    found["left_out"] = sum(t["role"] == "user" for t in cut)


def _begin(conv, question, choice, scope):
    """Keep a question as it is asked, in its conversation (a new one, titled by the question,
    when conv is None), its answer to be filled in by _finish: {conversation_id, title,
    question_id, answer_id}. NotFound if the conversation was deleted since _question."""
    folders = [f["id"] for f in scope["folders"]] if scope and scope["folders"] else None
    ex = db.begin_exchange(conv and conv["id"], question, _choice_text(choice), scope and list(scope["direct"]),
                           _title(question), folder_ids=folders)
    if not ex:
        raise NotFound("This conversation was deleted.")
    return ex


def _finish(ex, sent, text, found, used, note, usage, status="done", error=None):
    """Fill in a question's answer (_begin), however it ended. Returns {conversation_id,
    title}: the title as it is now (renamed while it was answered, say), both None when the
    conversation was deleted meanwhile. If the database stays locked, the answer is given all
    the same, and a restart marks it failed (db.end_interrupted_answers)."""
    answer = {"content": text, "sources": found, "usage": usage, "status": status, "error": error,
              **(jobs.used(used, note) if used else {})}
    try:
        title = db.finish_exchange(ex, sent, answer)
    except Exception:
        log.exception("Could not keep an answer in its conversation")
        title = ex["title"]
    return {"conversation_id": ex["conversation_id"] if title is not None else None, "title": title}


def _turns(user_message, text):
    # What an API client without conversation_id sends back as `history` next time.
    return [{"role": "user", "content": user_message}, {"role": "assistant", "content": text}]


@app.post("/api/chat")
def chat():
    """Answer a question from the passages that best match it, in all documents or only
    those in doc_ids and in the folders folder_ids (with their subfolders, as they are at
    the time of asking), and keep both in the conversation conversation_id (a new one without).
    The whole answer at once; /api/chat/stream sends it as it is written. The question is
    kept as it is asked; if it then fails, it is kept with the reason, and the error names
    the conversation (conversation_id, title) as an answer would."""
    question, choice, scope, conv, history = _question(_body())
    ex = _begin(conv, question, choice, scope)
    calls, user_message, found = [], None, None

    def on_usage(*call):
        calls.append(jobs.priced(*call))

    try:
        system, user_message, found, local = _prompt(question, scope, history, on_usage)

        def answer(backend, model):
            if backend == "ollama":
                return ollama.chat(None, local or [], user_message, model, system=system, on_usage=on_usage)
            return llm.chat(_client(), history, user_message, model, system=system, on_usage=on_usage)

        (text, usage), used, note = _run(choice, answer, fits_local=local is not None)
        _left_out(found, history, local, used)
    except Exception as e:
        error = _failure(e)
        kept = _finish(ex, user_message, "", found, None, None, None, "failed", (error or (UNEXPECTED, 500))[0])
        if not error:
            raise
        return jsonify(error=error[0], **kept), error[1]
    finally:
        jobs.log_usage(calls)  # the question's embedding, and a refused Claude answer (billed too)
    return jsonify(answer=text, usage=usage, **found, **jobs.used(used, note),
                   **_finish(ex, user_message, text, found, used, note, usage), turns=_turns(user_message, text))


@app.post("/api/chat/stream")
def chat_stream():
    """/api/chat, sent as it is written: Server-Sent Events (text/event-stream), each with
    a JSON data line. "start" at once: the conversation and the answer's id (to stop it
    with); "meta" when the model has begun: the model, why Auto switched, the sources;
    "delta": the next piece of the answer's text; then "done" (the answer, its status, usage
    and conversation) or "error" ({error, status, kept}: kept when the text so far stands).
    A request that is wrong (an empty question, an unknown model, a deleted conversation, a
    Claude model and no key) gets a plain HTTP error instead, as from /api/chat. Closing the
    stream stops the answer, which is kept as far as it got (see chat_stop). Nothing may
    buffer it on the way: no-cache, and X-Accel-Buffering for nginx-like proxies."""
    question, choice, scope, conv, history = _question(_body())
    events = _answer_events(question, choice, scope, conv, history)
    return Response(stream_with_context(_sse(events)), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(events):
    """Events as Server-Sent Events. A comment goes first, so the headers leave at once."""
    try:
        yield ": answering\n\n"
        for name, data in events:
            yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    finally:
        events.close()  # the page went away: the answer stops and is kept (_answer_events)


def _answer_events(question, choice, scope, conv, history):
    """A streamed answer's events (chat_stream). The question is kept first (_begin), so a
    page reloaded meanwhile still shows it, and "start" names it at once. Auto picks its model
    as _run does: by what fits the local model, and by switching to Claude when Ollama fails
    before its first words; a model that fails after them ends the answer where it got to.
    The answer is filled in however it ends: done; stopped (the page closed the stream, which
    is noticed as GeneratorExit at the next piece, or said so through chat_stop, checked
    before each piece); broken off part way (an error event with kept: true); or failed with
    no words (kept: false; a refusal's words are dropped too). A stop the page reported keeps
    exactly what it showed: a stream closed at once still had a piece or two on the way."""
    calls, pieces, gen, ex, key = [], [], None, None, None
    user_message = found = used = note = saved = None
    status, error = "failed", None

    def on_usage(*call):
        calls.append(jobs.priced(*call))

    def stopped():
        return _live.get(key) is not None

    try:
        ex = _begin(conv, question, choice, scope)
        key = ex["conversation_id"], ex["answer_id"]
        with _live_lock:
            _live[key] = None
        yield "start", {k: ex[k] for k in ("conversation_id", "title", "answer_id")}
        system, user_message, found, local = _prompt(question, scope, history, on_usage)

        def begin(backend, model):  # the model's stream, and its first piece ([] if it wrote nothing)
            if backend == "ollama":
                g = ollama.chat_stream(None, local or [], user_message, model, system=system, on_usage=on_usage)
            else:
                g = llm.chat_stream(_client(), history, user_message, model, system=system, on_usage=on_usage)
            return g, list(itertools.islice(g, 1))

        (gen, first), used, note = _run(choice, begin, fits_local=local is not None)
        _left_out(found, history, local, used)
        if not stopped():
            yield "meta", {"conversation_id": ex["conversation_id"], "title": ex["title"], **found,
                           **jobs.used(used, note)}
            for piece in itertools.chain(first, gen):
                if stopped():
                    break
                pieces.append(piece)
                yield "delta", piece
        status = "stopped" if stopped() else "done"
    except GeneratorExit:  # stopped: the page closed the stream
        status = "stopped"
        raise
    except Exception as e:
        status = "broken" if pieces and not isinstance(e, llm.Refused) else "failed"
        error = _failure(e)
        if not error:
            log.exception("A streamed answer failed")
            error = UNEXPECTED, 500
    finally:
        try:  # the model's stream ends with the answer, however it ended, and reports what it used
            if gen:
                gen.close()
        except Exception:
            log.exception("Could not close an answer's stream")
        jobs.log_usage(calls)
        text = "".join(pieces) if status != "failed" else ""
        usage = next((c[3] for c in reversed(calls) if c[0] == "chat"), None)
        if ex:
            with _live_lock:  # a stop sent meanwhile finds the answer either still live, or kept
                shown = _live.pop(key, None)
                if shown is not None:  # the page stopped it there: what it showed is what is kept
                    text, status, error = "".join(pieces)[:shown], "stopped", None
                saved = _finish(ex, user_message, text, found, used, note, usage, status, error and error[0])
    if status in ("done", "stopped"):
        yield "done", {"answer": text, "status": status, "usage": usage, **saved, "turns": _turns(user_message, text)}
    else:
        yield "error", {"error": error[0], "status": error[1], "kept": status == "broken", **(saved or {})}


@app.post("/api/chat/stop")
def chat_stop():
    """The page stopped a streamed answer after showing `shown` characters of it (Unicode
    code points): {conversation_id, answer_id, shown}, as its "start" event named them. The
    answer keeps that much and no more, whether it is still being written (it ends before
    its next piece, or as soon as the model begins) or was kept already. Otherwise the
    server only notices a stop when a piece fails to go out, and keeps the pieces sent just
    before it, which the page never showed: "Stopped before the answer began" with words."""
    body = _body()
    conv_id, answer_id, shown = (body.get(k) for k in ("conversation_id", "answer_id", "shown"))
    if not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 2**63 for v in (conv_id, answer_id, shown)):
        raise BadChoice('Expected {"conversation_id", "answer_id", "shown"}: whole numbers')
    with _live_lock:
        live = (conv_id, answer_id) in _live
        if live:
            _live[conv_id, answer_id] = shown
    if live or db.cut_answer(conv_id, answer_id, shown):
        return "", 204
    return jsonify(error="Not found"), 404


# ---- saved conversations ----

@app.get("/api/conversations")
def conversation_list():
    """Saved conversations, the one asked last first: id, title, created_at, updated_at,
    model (the choice of its last question), doc_ids and folder_ids (the documents and
    folders it was limited to: both null for all; [] when they were all deleted; doc_ids []
    when only folders were chosen) and questions (how many it has)."""
    return jsonify(db.list_conversations())


@app.get("/api/conversations/<id:conv_id>")
def conversation_get(conv_id):
    """A conversation and its messages, oldest first: each question as asked, and its
    answer with the sources, excerpts, model, usage and status: answering (still being
    written), done, stopped, broken (failed part way) or failed (no words), with the error."""
    c = db.get_conversation(conv_id)
    return jsonify(c) if c else (jsonify(error="Not found"), 404)


@app.patch("/api/conversations/<id:conv_id>")
def conversation_rename(conv_id):
    title = _body().get("title")
    title = " ".join(title.split()) if isinstance(title, str) else ""
    if not title:
        raise BadChoice("A conversation's title can't be empty")
    if len(title) > 200:
        raise BadChoice("Title is too long (200 characters at most)")
    if not db.rename_conversation(conv_id, title):
        return jsonify(error="Not found"), 404
    return jsonify(db.get_conversation(conv_id, messages=False))


@app.delete("/api/conversations/<id:conv_id>")
def conversation_delete(conv_id):
    """Delete a conversation and its messages; its model calls stay in the usage log."""
    return ("", 204) if db.delete_conversation(conv_id) else (jsonify(error="Not found"), 404)
