"""HTTP API + single-page UI. Uploads and re-reads are queued as jobs (see jobs.py)."""
import mimetypes
import os
import re
from urllib.parse import urlsplit

import anthropic
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.routing import IntegerConverter

from . import db, extract, jobs, llm, ollama


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


def _job_choice(raw):
    """A model choice for a job, checked now: a Claude model needs a key up front."""
    backend, model = _choice(raw)
    if backend == "claude" and not _has_key():
        raise MissingKey()
    return "auto" if backend == "auto" else f"{backend}:{model}"


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
    Uploads do the same in the background (jobs._read)."""
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


@app.errorhandler(MissingKey)
def _no_key(e):
    return jsonify(error="Enter your Claude API key first"), 401


@app.errorhandler(413)
def _too_large(e):
    return jsonify(error=f"File is larger than the {MAX_UPLOAD_MB} MB upload limit (MAX_UPLOAD_MB)"), 413


@app.errorhandler(BadChoice)
def _bad_choice(e):
    return jsonify(error=str(e)), 400


@app.errorhandler(extract.Unsupported)
def _unsupported(e):
    return jsonify(error=str(e)), 400


@app.errorhandler(anthropic.AuthenticationError)
def _bad_key(e):
    return jsonify(error="Invalid Claude API key"), 401


@app.errorhandler(anthropic.RateLimitError)
def _rate_limited(e):
    return jsonify(error="Rate limited by the Claude API, try again shortly"), 429


@app.errorhandler(anthropic.APIStatusError)
def _api_error(e):
    return jsonify(error=f"Claude API error: {e.message}"), 502


@app.errorhandler(anthropic.APIConnectionError)
def _no_network(e):
    return jsonify(error="Could not reach the Claude API"), 502


@app.errorhandler(llm.Refused)
def _refused(e):
    return jsonify(error=f"{e}. Try rewording it, or another model."), 502


@app.errorhandler(ollama.OllamaUnavailable)
def _ollama_down(e):
    return jsonify(error=f"{e}. Is it running?"), 502


@app.errorhandler(ollama.OllamaError)
def _ollama_error(e):
    return jsonify(error=f"Ollama error: {e}"), 502


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
    return jsonify([{**{k: h[k] for k in ("doc_id", "filename", "title", "idx")}, "snippet": extract.plain(h["snippet"])}
                    for h in db.search_chunks(request.args.get("q", ""), limit=10)])


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


@app.post("/api/chat")
def chat():
    body = _body()
    question = body.get("question")
    question = question.strip() if isinstance(question, str) else ""
    if not question:
        return jsonify(error="Empty question"), 400
    choice = _choice(body.get("model"))
    history = body.get("history") or []
    hits = db.search_chunks(question, limit=int(os.environ.get("TOP_K", "5")))
    user_message = llm.build_user_message(question, hits)

    system = llm.system_prompt(db.active_instructions())
    calls = []

    def on_usage(*call):
        calls.append(jobs.priced(*call))

    def answer(backend, model):
        if backend == "ollama":
            return ollama.chat(None, history, user_message, model, system=system, on_usage=on_usage)
        return llm.chat(_client(), history, user_message, model, system=system, on_usage=on_usage)

    fits = ollama.fits([{"content": system}, *history, {"content": user_message}])
    try:
        (text, usage), used, note = _run(choice, answer, fits_local=fits)
    finally:
        jobs.log_usage(calls)  # a refused Claude answer is billed too
    return jsonify(
        answer=text,
        usage=usage,
        sources=sorted({h["filename"] for h in hits}),
        **jobs.used(used, note),
        # The client stores these and sends them back as `history` next turn.
        turns=[{"role": "user", "content": user_message}, {"role": "assistant", "content": text}],
    )
