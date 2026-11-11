"""HTTP API + single-page UI."""
import os

import anthropic
from flask import Flask, jsonify, request, send_from_directory

from . import db, extract, llm, ollama

app = Flask(__name__, static_folder="static")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
# Default model choice when the browser doesn't send one:
#   "ollama": Auto. Local models on this machine's GPU (see app/ollama.py), switching to
#             Claude only when needed and a Claude key is available.
#   "claude": the Claude API (CLAUDE_MODEL), with the key from the browser or ANTHROPIC_API_KEY.
BACKEND = os.environ.get("LLM_BACKEND", "ollama")
# "model" (default): the chosen model reads PDFs. With Ollama only scanned pages go
# to the vision model; with Claude every page is read as text and image.
# "local": the PDF's embedded text layer only; scanned pages come out empty.
PDF_EXTRACTOR = os.environ.get("PDF_EXTRACTOR", "model")


class MissingKey(Exception):
    pass


class BadChoice(ValueError):
    pass


class NoText(ValueError):
    pass


def _has_key():
    return bool(request.headers.get("X-API-Key", "").strip() or os.environ.get("ANTHROPIC_API_KEY"))


def _client():
    # The key comes from the browser per request (or ANTHROPIC_API_KEY); never stored server-side.
    if not _has_key():
        raise MissingKey()
    return llm.client_for(request.headers.get("X-API-Key", "").strip())


def _choice(raw):
    """'auto' | 'ollama:<model>' | 'claude:<model>' -> (backend, model or None)."""
    raw = (raw or "").strip() or ("auto" if BACKEND == "ollama" else f"claude:{llm.MODEL}")
    if raw == "auto":
        return "auto", None
    backend, _, model = raw.partition(":")
    if backend == "ollama" and model:
        return backend, model
    if backend == "claude" and model in dict(llm.MODELS):
        return backend, model
    raise BadChoice(f"Unknown model choice: {raw}")


def _label(backend, model):
    return model if backend == "ollama" else llm.label(model)


def _run(choice, job, fits_local=True):
    """Run job(backend, model) for the chosen model. Auto runs locally and switches to
    Claude when the prompt is too long for the local model or Ollama fails, provided
    a Claude key is available. Returns (result, model used, why it switched or None)."""
    backend, model = choice
    if backend != "auto":
        return job(backend, model), (backend, model), None
    if not fits_local and _has_key():
        return job("claude", llm.MODEL), ("claude", llm.MODEL), "too long for the local model"
    try:
        return job("ollama", ollama.MODEL), ("ollama", ollama.MODEL), None
    except ollama.OllamaError as e:
        if not _has_key():
            raise
        return job("claude", llm.MODEL), ("claude", llm.MODEL), f"local model failed ({e})"


def _used(used, note):
    backend, model = used
    return {"model": f"{backend}:{model}", "model_label": _label(backend, model), "switched": note}


@app.errorhandler(MissingKey)
def _no_key(e):
    return jsonify(error="Enter your Claude API key first"), 401


@app.errorhandler(413)
def _too_large(e):
    return jsonify(error=f"File is larger than the {MAX_UPLOAD_MB} MB upload limit (MAX_UPLOAD_MB)"), 413


@app.errorhandler(BadChoice)
def _bad_choice(e):
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


@app.errorhandler(ollama.OllamaUnavailable)
def _ollama_down(e):
    return jsonify(error=f"{e}. Is it running?"), 502


@app.errorhandler(ollama.OllamaError)
def _ollama_error(e):
    return jsonify(error=f"Ollama error: {e}"), 502


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
    )


@app.get("/api/documents")
def documents():
    q = request.args.get("q", "").strip()
    return jsonify(db.search_documents(q) if q else db.list_documents())


@app.get("/api/search")
def search():
    return jsonify([{k: h[k] for k in ("doc_id", "filename", "title", "idx", "snippet")}
                    for h in db.search_chunks(request.args.get("q", ""), limit=10)])


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="No file"), 400
    choice = _choice(request.form.get("model"))
    name, data = f.filename, f.read()
    is_pdf = name.lower().endswith(".pdf") and PDF_EXTRACTOR != "local"

    def read(backend, model):
        if not is_pdf:
            text = extract.extract_text(name, data)
        elif backend == "ollama":
            text = ollama.transcribe_pdf(data)  # scanned pages go to the vision model
        else:
            text = llm.transcribe_pdf(_client(), extract.pdf_batches(data), model)
        if not extract.chunk_text(text):
            raise NoText()
        try:
            if backend == "ollama":
                meta, analysis = ollama.analyze(None, name, text, model)
            else:
                meta, analysis = llm.analyze(_client(), name, text, model)
        except (anthropic.AuthenticationError, MissingKey, ollama.OllamaError):
            raise  # Ollama errors let Auto switch to Claude
        except Exception as e:  # still index the text so it is searchable
            meta, analysis = {"title": name}, f"none: {e}"
        return text, meta, analysis

    try:
        (text, meta, analysis), used, note = _run(choice, read)
    except (MissingKey, anthropic.APIError, ollama.OllamaError):
        raise  # handled by the error handlers above
    except NoText:
        return jsonify(error="No text found in file"), 400
    except Exception as e:
        return jsonify(error=f"Could not read {name}: {e}"), 400
    chunks = extract.chunk_text(text)
    doc_id = db.add_document(name, chunks, meta, analysis, _label(*used))
    return jsonify(id=doc_id, chunks=len(chunks), analysis=analysis, **_used(used, note), **meta)


@app.delete("/api/documents/<int:doc_id>")
def delete(doc_id):
    return ("", 204) if db.delete_document(doc_id) else (jsonify(error="Not found"), 404)


@app.post("/api/documents/delete")
def delete_many():
    ids = (request.get_json(force=True) or {}).get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        return jsonify(error="Expected {\"ids\": [1, 2, ...]}"), 400
    return jsonify(deleted=db.delete_documents(ids))


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
    body = request.get_json(force=True) or {}
    text = _rule_text(body)
    if text is None:
        raise BadChoice("Instruction text can't be empty")
    rule_id = db.add_instruction(text, bool(body.get("enabled", True)))
    return jsonify(next(r for r in db.list_instructions() if r["id"] == rule_id)), 201


@app.patch("/api/instructions/<int:rule_id>")
def edit_instruction(rule_id):
    body = request.get_json(force=True) or {}
    enabled = body.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise BadChoice("enabled must be true or false")
    if not db.update_instruction(rule_id, _rule_text(body), enabled):
        return jsonify(error="Not found"), 404
    return jsonify(next(r for r in db.list_instructions() if r["id"] == rule_id))


@app.delete("/api/instructions/<int:rule_id>")
def delete_instruction(rule_id):
    return ("", 204) if db.delete_instruction(rule_id) else (jsonify(error="Not found"), 404)


@app.post("/api/chat")
def chat():
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify(error="Empty question"), 400
    choice = _choice(body.get("model"))
    history = body.get("history") or []
    hits = db.search_chunks(question, limit=int(os.environ.get("TOP_K", "5")))
    user_message = llm.build_user_message(question, hits)

    system = llm.system_prompt(db.active_instructions())

    def answer(backend, model):
        if backend == "ollama":
            return ollama.chat(None, history, user_message, model, system=system)
        return llm.chat(_client(), history, user_message, model, system=system)

    fits = ollama.fits([{"content": system}, *history, {"content": user_message}])
    (text, usage), used, note = _run(choice, answer, fits_local=fits)
    return jsonify(
        answer=text,
        usage=usage,
        sources=sorted({h["filename"] for h in hits}),
        **_used(used, note),
        # The client stores these and sends them back as `history` next turn.
        turns=[{"role": "user", "content": user_message}, {"role": "assistant", "content": text}],
    )
