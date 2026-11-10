"""HTTP API + single-page UI."""
import os

import anthropic
from flask import Flask, jsonify, request, send_from_directory

from . import db, extract, llm

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024


def _client():
    # The key comes from the browser per request (or ANTHROPIC_API_KEY); never stored server-side.
    key = request.headers.get("X-API-Key", "").strip()
    if not key and not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingKey()
    return llm.client_for(key)


class MissingKey(Exception):
    pass


@app.errorhandler(MissingKey)
def _no_key(e):
    return jsonify(error="Enter your Claude API key first"), 401


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


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


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
    try:
        text = extract.extract_text(f.filename, f.read())
    except Exception as e:
        return jsonify(error=f"Could not read {f.filename}: {e}"), 400
    chunks = extract.chunk_text(text)
    if not chunks:
        return jsonify(error="No text found in file (scanned PDFs are not supported)"), 400
    try:
        meta, analysis = llm.analyze(_client(), f.filename, text)
    except (anthropic.AuthenticationError, MissingKey):
        raise
    except Exception as e:  # still index the text so it is searchable
        meta, analysis = {"title": f.filename}, f"none: {e}"
    doc_id = db.add_document(f.filename, chunks, meta, analysis)
    return jsonify(id=doc_id, chunks=len(chunks), analysis=analysis, **meta)


@app.delete("/api/documents/<int:doc_id>")
def delete(doc_id):
    return ("", 204) if db.delete_document(doc_id) else (jsonify(error="Not found"), 404)


@app.post("/api/chat")
def chat():
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify(error="Empty question"), 400
    hits = db.search_chunks(question, limit=int(os.environ.get("TOP_K", "5")))
    user_message = llm.build_user_message(question, hits)
    answer, usage = llm.chat(_client(), body.get("history") or [], user_message)
    return jsonify(
        answer=answer,
        usage=usage,
        sources=sorted({h["filename"] for h in hits}),
        # The client stores these and sends them back as `history` next turn.
        turns=[{"role": "user", "content": user_message}, {"role": "assistant", "content": answer}],
    )
