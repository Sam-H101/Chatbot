"""HTTP API + single-page UI. Uploads and re-reads are queued as jobs (see jobs.py)."""
import itertools
import json
import logging
import mimetypes
import os
import re
from urllib.parse import urlsplit

import anthropic
from flask import Flask, Response, jsonify, request, send_file, send_from_directory, stream_with_context
from werkzeug.routing import IntegerConverter

from . import db, extract, jobs, llm, ollama, vectors

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
# Kept originals shown in the browser: PDFs open in its viewer, plain text as text.
# Everything else downloads, so an uploaded HTML or SVG file never runs script here.
INLINE_TYPES = {"application/pdf", "text/plain"}
# The slim image has no /etc/mime.types, and Python's own table lacks these two.
mimetypes.add_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx")
mimetypes.add_type("text/plain", ".log")
MissingKey = llm.MissingKey
# Most characters of a new conversation's title, taken from its first question.
TITLE_CHARS = 80


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


def _error_reply(e):
    error, status = _failure(e)
    return jsonify(error=error), status


for _cls in (MissingKey, BadChoice, extract.Unsupported, NotFound, anthropic.AuthenticationError,
             anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError, llm.Refused,
             ollama.OllamaError):  # the most specific handler wins, so each gets _failure's wording
    app.register_error_handler(_cls, _error_reply)


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


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


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
        max_upload_mb=MAX_UPLOAD_MB,
        claude_pages=jobs.CLAUDE_PDF_PAGES, confirm_cost_usd=jobs.CONFIRM_COST_USD,
    )


@app.get("/api/documents")
def documents():
    q = request.args.get("q", "").strip()
    return jsonify(db.search_documents(q) if q else db.list_documents())


@app.get("/api/search")
def search():
    """Passages with the words of q, for the library's search box: by keywords only, since
    it marks the words it found."""
    return jsonify([{**{k: h[k] for k in ("doc_id", "filename", "title", "idx", "page_start", "page_end")},
                     "snippet": extract.plain(h["snippet"])}
                    for h in db.search_chunks(request.args.get("q", ""), limit=10)])


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
    options = {"on_duplicate": on_duplicate, "claude_pages": _claude_pages(request.form.get("claude_pages"))}
    data = f.read()
    extract.check(name, data)  # a file that can never be read gets its 400 now, not a failed job later
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
                "model": d["model"], "copies": len(dups)}), 409
        job_id = _add_job("upload", name, sha, model, options, data=data,
                          mime=mimetypes.guess_type(name)[0] or f.mimetype or "application/octet-stream")
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
    """Cancel a job that hasn't finished (a running one stops after its current page),
    or dismiss a failed one."""
    j = db.cancel_job(job_id)
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
    f = db.stored_file(doc_id)
    if not f:  # also documents uploaded before originals were kept
        return jsonify(error="No original file is kept for this document"), 404
    resp = send_file(f["path"], mimetype=f["mime"], as_attachment=f["mime"] not in INLINE_TYPES,
                     download_name=f["filename"], max_age=0)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


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
    jobs.prune_keys()  # its re-read, if any, was cancelled
    return ("", 204) if deleted else (jsonify(error="Not found"), 404)


@app.post("/api/documents/delete")
def delete_many():
    ids = _body().get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        return jsonify(error="Expected {\"ids\": [1, 2, ...]}"), 400
    deleted = db.delete_documents(ids)
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


def _scope(raw):
    """POST /api/chat's doc_ids: the documents a question is limited to, as {id: file
    name}, those deleted meanwhile left out. None (or []) means every document."""
    if raw is None or raw == []:
        return None
    if not isinstance(raw, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in raw):
        raise BadChoice("doc_ids must be a list of document ids")
    names = db.document_names([i for i in dict.fromkeys(raw) if -2**63 <= i < 2**63])  # others can't exist
    if not names:
        raise BadChoice("None of the chosen documents is in the library any more: choose others, or ask all documents")
    return names


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
    choice, scope = _choice(body.get("model")), _scope(body.get("doc_ids"))
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
    them (sources, excerpts, ...), whether it all fits the local model's context)."""
    system = llm.system_prompt(db.active_instructions())
    hits = vectors.search(question, int(os.environ.get("TOP_K", "5")), scope and list(scope), on_usage=on_usage)
    # No passage at all ("Summarise everything in my library", or small talk): the model is
    # given the documents' titles and summaries instead, so it can say what the library holds.
    library = db.library_overview() if not hits and scope is None else None
    user_message = llm.build_user_message(question, hits, scope and list(scope.values()), library)
    excerpts, sources = _cited(hits)
    found = {"sources": sources, "excerpts": excerpts,
             "overview": bool(library and library[1]),  # given the library's summaries instead of passages
             "scope": scope and list(scope), "scope_names": scope and list(scope.values())}
    fits = ollama.fits([{"content": system}, *history, {"content": user_message}])
    return system, user_message, found, fits


def _save(conv, question, user_message, text, found, choice, used, note, usage, status="done", error=None):
    """Keep a question and its answer in their conversation, a new one (titled by the
    question) when conv is None. Returns {conversation_id, title}, both None when it couldn't
    be kept (the conversation was deleted meanwhile, or the database stayed locked): the
    answer is given all the same."""
    answer = {"content": text, "sources": found, "usage": usage, "status": status, "error": error,
              **jobs.used(used, note)}
    try:
        conv_id = db.add_exchange(conv and conv["id"], question, user_message, answer, model=_choice_text(choice),
                                  doc_ids=found["scope"], title=_title(question))
    except Exception:
        log.exception("Could not keep an answer in its conversation")
        conv_id = None
    title = (conv["title"] if conv else _title(question)) if conv_id else None
    return {"conversation_id": conv_id, "title": title}


def _turns(user_message, text):
    # What an API client without conversation_id sends back as `history` next time.
    return [{"role": "user", "content": user_message}, {"role": "assistant", "content": text}]


@app.post("/api/chat")
def chat():
    """Answer a question from the passages that best match it, in all documents or only
    those in doc_ids, and keep both in the conversation conversation_id (a new one without).
    The whole answer at once; /api/chat/stream sends it as it is written."""
    question, choice, scope, conv, history = _question(_body())
    calls = []

    def on_usage(*call):
        calls.append(jobs.priced(*call))

    try:
        system, user_message, found, fits = _prompt(question, scope, history, on_usage)

        def answer(backend, model):
            if backend == "ollama":
                return ollama.chat(None, history, user_message, model, system=system, on_usage=on_usage)
            return llm.chat(_client(), history, user_message, model, system=system, on_usage=on_usage)

        (text, usage), used, note = _run(choice, answer, fits_local=fits)
    finally:
        jobs.log_usage(calls)  # the question's embedding, and a refused Claude answer (billed too)
    return jsonify(answer=text, usage=usage, **found, **jobs.used(used, note),
                   **_save(conv, question, user_message, text, found, choice, used, note, usage),
                   turns=_turns(user_message, text))


@app.post("/api/chat/stream")
def chat_stream():
    """/api/chat, sent as it is written: Server-Sent Events (text/event-stream), each with
    a JSON data line. "meta" when the model has begun: the conversation, the model, why
    Auto switched, the sources; "delta": the next piece of the answer's text; then "done"
    (the answer, its usage, the conversation) or "error" ({error, status, kept}: kept when
    the text so far stands). A request that is wrong (an empty question, an unknown model,
    a deleted conversation, a Claude model and no key) gets a plain HTTP error instead,
    as from /api/chat. Closing the stream stops the answer, which is kept as far as it got.
    Nothing may buffer it on the way: no-cache, and X-Accel-Buffering for nginx-like proxies."""
    question, choice, scope, conv, history = _question(_body())
    if choice[0] == "claude" and not _has_key():
        raise MissingKey()
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
    """A streamed answer's events (chat_stream). Auto picks its model as _run does: by the
    fits check, and by switching to Claude when Ollama fails before its first words; a model
    that fails after them ends the answer where it got to. The question and its answer are
    kept in the conversation when the answer is done, stopped (the stream closed:
    GeneratorExit) or broken off part way (an error event with kept: true); not when nothing
    was written, nor after a refusal, whose words are dropped (kept: false). A new
    conversation is made just before "meta", so the page has its id even if it stops the
    answer; it goes again if nothing ends up in it."""
    calls, pieces, gen, found, used, note, status, error, made = [], [], None, None, None, None, None, None, False

    def on_usage(*call):
        calls.append(jobs.priced(*call))

    try:
        system, user_message, found, fits = _prompt(question, scope, history, on_usage)

        def begin(backend, model):  # the model's stream, and its first piece ([] if it wrote nothing)
            if backend == "ollama":
                g = ollama.chat_stream(None, history, user_message, model, system=system, on_usage=on_usage)
            else:
                g = llm.chat_stream(_client(), history, user_message, model, system=system, on_usage=on_usage)
            return g, list(itertools.islice(g, 1))

        (gen, first), used, note = _run(choice, begin, fits_local=fits)
        if conv is None:
            conv = {"id": db.add_conversation(_title(question), _choice_text(choice), found["scope"]),
                    "title": _title(question)}
            made = True
        yield "meta", {"conversation_id": conv["id"], "title": conv["title"], **found, **jobs.used(used, note)}
        for piece in itertools.chain(first, gen):
            pieces.append(piece)
            yield "delta", piece
        status = "done"
    except GeneratorExit:  # stopped: the page closed the stream
        status = "stopped"
        raise
    except Exception as e:
        status = "broken" if pieces and not isinstance(e, llm.Refused) else "failed"
        error = _failure(e)
        if not error:
            log.exception("A streamed answer failed")
            error = f"The answer failed: {e}", 500
    finally:
        try:  # the model's stream ends with the answer, however it ended, and reports what it used
            if gen:
                gen.close()
        except Exception:
            log.exception("Could not close an answer's stream")
        jobs.log_usage(calls)
        text = "".join(pieces)
        usage = next((c[3] for c in reversed(calls) if c[0] == "chat"), None)
        if status == "done" or (status in ("stopped", "broken") and text):
            saved = _save(conv, question, user_message, text, found, choice, used, note, usage, status,
                          error and error[0])
        elif made:
            db.drop_empty_conversation(conv["id"])
    if status == "done":
        yield "done", {"answer": text, "usage": usage, **saved, "turns": _turns(user_message, text)}
    else:
        yield "error", {"error": error[0], "status": error[1], "kept": status == "broken"}


# ---- saved conversations ----

@app.get("/api/conversations")
def conversation_list():
    """Saved conversations, the one asked last first: id, title, created_at, updated_at,
    model (the choice of its last question), doc_ids (the documents it was limited to:
    null for all, [] when they were all deleted) and questions (how many it has)."""
    return jsonify(db.list_conversations())


@app.get("/api/conversations/<id:conv_id>")
def conversation_get(conv_id):
    """A conversation and its messages, oldest first: each question as asked, and its
    answer with the sources, excerpts, model, usage and status (done, stopped or broken)."""
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
