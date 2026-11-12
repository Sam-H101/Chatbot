import hashlib
import io

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import db, llm, main
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setattr(db, "FILES_DIR", "")  # originals go to tmp_path / "files"
    monkeypatch.setattr(main, "BACKEND", "claude")
    monkeypatch.setattr(llm, "client_for", lambda key: object())
    monkeypatch.setattr(llm, "analyze", lambda c, name, text, model=None, **kw: (
        {"title": "Lease", "doc_type": "contract", "summary": "A flat lease.", "keywords": ["rent", "tenant"]}, "full"))
    seen = {}

    def fake_chat(c, history, msg, model=None, system=None):
        seen["history"], seen["msg"], seen["model"], seen["system"] = history, msg, model, system
        return "42", {"input": 1, "output": 1, "cache_read": 0, "cache_write": 0}

    monkeypatch.setattr(llm, "chat", fake_chat)
    c = main.app.test_client()
    c.seen = seen
    return c


H = {"X-API-Key": "k"}
LEASE = b"The monthly rent is 900 euros.\n\nPets are allowed."


def upload(client, name="lease.txt", body=LEASE, **form):
    return client.post("/api/upload", data={"file": (io.BytesIO(body), name), **form}, headers=H)


def test_upload_search_delete(client):
    r = upload(client)
    assert r.status_code == 200 and r.json["title"] == "Lease"
    assert client.get("/api/documents?q=tenant").json[0]["filename"] == "lease.txt"   # metadata search
    assert client.get("/api/documents?q=zebra").json == []
    hits = client.get("/api/search?q=renting pets").json                               # content search, stemmed
    assert hits and "[" in hits[0]["snippet"]
    assert client.get('/api/search?q=" OR ) NEAR(').status_code == 200                 # hostile FTS syntax
    assert client.delete(f"/api/documents/{r.json['id']}").status_code == 204
    assert client.get("/api/documents").json == []
    assert client.get("/api/search?q=rent").json == []


def test_chat_grounds_on_excerpts(client):
    upload(client)
    r = client.post("/api/chat", json={"question": "How much is the rent?", "history": []}, headers=H)
    assert r.json["answer"] == "42" and r.json["sources"] == ["lease.txt"]
    assert "900 euros" in client.seen["msg"]
    assert r.json["turns"][0]["content"] == client.seen["msg"]


def test_requires_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client.post("/api/chat", json={"question": "hi"}).status_code == 401


def test_chunking():
    from app.extract import chunk_text
    chunks = chunk_text("a" * 4000 + "\n\n" + "b " * 100, size=1500)
    assert all(len(c) <= 1500 for c in chunks) and "".join(chunks).count("a") == 4000


def blank_pdf(pages):
    import pypdfium2
    pdf = pypdfium2.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(612, 792)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def test_pdf_is_read_by_claude_in_page_batches(client, monkeypatch):
    from app import extract, llm
    monkeypatch.setattr(extract.pdf_batches, "__defaults__", (20, None))
    seen = []

    def fake_transcribe(c, batches, model=None, **kw):
        seen.extend((first, last, total) for first, last, total, _ in batches)
        return "The deposit is 1800 euros."

    monkeypatch.setattr(llm, "transcribe_pdf", fake_transcribe)
    assert upload(client, "scan.pdf", blank_pdf(45)).status_code == 200
    assert seen == [(1, 20, 45), (21, 40, 45), (41, 45, 45)]
    assert client.get("/api/search?q=deposit").json[0]["filename"] == "scan.pdf"


def test_local_pdf_mode_finds_no_text_in_a_scan(client, monkeypatch):
    from app import main
    monkeypatch.setattr(main, "PDF_EXTRACTOR", "local")
    assert upload(client, "scan.pdf", blank_pdf(2)).status_code == 400


@pytest.fixture
def local(client, monkeypatch):
    """The Ollama backend, with Ollama's HTTP API faked. Records each request."""
    from app import main, ollama
    monkeypatch.setattr(main, "BACKEND", "ollama")
    calls = []

    def fake_chat(model, messages, **kw):
        calls.append({"model": model, "messages": messages, **kw})
        if "format" in kw:
            return {"message": {"content": '{"title": "Scan", "doc_type": "notes", '
                                           '"summary": "s", "keywords": ["k"]}'}, "prompt_eval_count": 900, "eval_count": 60}
        if messages[-1].get("images"):
            return {"message": {"content": "The deposit is 1800 euros."}, "prompt_eval_count": 800, "eval_count": 9}
        return {"message": {"content": "1800"}, "prompt_eval_count": 7, "eval_count": 2}

    monkeypatch.setattr(ollama, "_chat", fake_chat)
    client.calls = calls
    return client


def test_ollama_reads_only_scanned_pages_with_vision_model(local, monkeypatch):
    from app import extract, ollama
    monkeypatch.setattr(extract, "pdf_pages", lambda data: [("Page one has real text.", None),
                                                           ("", b"png")])
    r = upload(local, "mixed.pdf", blank_pdf(2))
    assert r.status_code == 200 and r.json["title"] == "Scan"
    vision = [c for c in local.calls if c["messages"][-1].get("images")]
    assert len(vision) == 1 and vision[0]["model"] == ollama.VISION_MODEL
    assert local.get("/api/search?q=deposit").json[0]["filename"] == "mixed.pdf"
    assert local.get("/api/search?q=real text").json[0]["filename"] == "mixed.pdf"


def test_blank_pdf_page_is_rendered_for_the_vision_model():
    from app import extract
    (text, png), = extract.pdf_pages(blank_pdf(1))
    assert text.strip() == "" and png[1:4] == b"PNG"


def test_ollama_chat_needs_no_key(local, monkeypatch):
    from app import ollama
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    upload(local)
    r = local.post("/api/chat", json={"question": "How much is the rent?", "history": []})
    assert r.status_code == 200 and r.json["answer"] == "1800"
    assert r.json["usage"]["input"] == 7
    sent = local.calls[-1]
    assert sent["model"] == ollama.MODEL and sent["messages"][0]["role"] == "system"
    assert "900 euros" in sent["messages"][-1]["content"]


def test_ollama_down_is_reported(client, monkeypatch):
    from app import main, ollama
    monkeypatch.setattr(main, "BACKEND", "ollama")
    monkeypatch.setattr(ollama, "URL", "http://127.0.0.1:9")  # nothing listens here
    r = client.post("/api/chat", json={"question": "hi", "history": []})
    assert r.status_code == 502 and "Could not reach Ollama" in r.json["error"]


def test_models_lists_local_and_claude(local, monkeypatch):
    from app import ollama
    monkeypatch.setattr(ollama, "models", lambda: ["minicpm-v4.6:latest", "qwen3.5:9b"])
    r = local.get("/api/models").json
    assert r["default"] == "auto" and r["local"] == ["minicpm-v4.6:latest", "qwen3.5:9b"]
    assert {"id": "claude-opus-5-5", "label": "Claude Opus 5.5"} in r["claude"]


def test_models_reports_ollama_down(local, monkeypatch):
    from app import ollama
    monkeypatch.setattr(ollama, "URL", "http://127.0.0.1:9")
    r = local.get("/api/models").json
    assert r["local"] == [] and "Could not reach Ollama" in r["local_error"]


def test_explicit_claude_model_for_upload_and_chat(local, monkeypatch):
    from app import llm
    used = {}
    monkeypatch.setattr(llm, "analyze", lambda c, name, text, model=None, **kw: (
        used.setdefault("analyze", model), ({"title": "T", "doc_type": "d", "summary": "s", "keywords": []}, "full"))[1])
    r = local.post("/api/upload", headers=H, data={
        "file": (io.BytesIO(b"The rent is 900 euros."), "a.txt"), "model": "claude:claude-haiku-5-5"})
    assert r.status_code == 200 and r.json["model"] == "claude:claude-haiku-5-5"
    assert r.json["model_label"] == "Claude Haiku 5.5" and used["analyze"] == "claude-haiku-5-5"
    assert local.calls == []  # Ollama untouched
    r = local.post("/api/chat", headers=H, json={"question": "rent?", "model": "claude:claude-sonnet-5-5"})
    assert r.json["answer"] == "42" and local.seen["model"] == "claude-sonnet-5-5"


def test_explicit_local_model(local):
    r = local.post("/api/chat", json={"question": "rent?", "model": "ollama:llama9:latest"})
    assert r.status_code == 200 and r.json["model"] == "ollama:llama9:latest" and r.json["switched"] is None
    assert local.calls[-1]["model"] == "llama9:latest"


def test_unknown_model_choice_is_rejected(local):
    assert local.post("/api/chat", json={"question": "hi", "model": "claude:gpt-9"}).status_code == 400
    assert local.post("/api/chat", json={"question": "hi", "model": "nonsense"}).status_code == 400


def test_auto_switches_to_claude_when_ollama_fails(local, monkeypatch):
    from app import llm, ollama

    def broken(model, messages, **kw):
        raise ollama.OllamaError(f"{model}: model not found")

    monkeypatch.setattr(ollama, "_chat", broken)
    r = local.post("/api/chat", headers=H, json={"question": "hi", "model": "auto"})
    assert r.status_code == 200 and r.json["model"] == f"claude:{llm.MODEL}"
    assert "local model failed" in r.json["switched"]
    r = upload(local)  # uploads switch too
    assert r.status_code == 200 and r.json["model"] == f"claude:{llm.MODEL}"


def test_auto_without_key_reports_the_local_error(local, monkeypatch):
    from app import ollama
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def down(model, messages, **kw):
        raise ollama.OllamaUnavailable(f"Could not reach Ollama at {ollama.URL}")

    monkeypatch.setattr(ollama, "_chat", down)
    r = local.post("/api/chat", json={"question": "hi", "model": "auto"})
    assert r.status_code == 502 and "Could not reach Ollama" in r.json["error"]


def test_auto_sends_long_prompts_to_claude(local, monkeypatch):
    from app import llm, ollama
    monkeypatch.setattr(ollama, "NUM_CTX", 4096)
    long_history = [{"role": "user", "content": "x" * 20000}, {"role": "assistant", "content": "ok"}]
    r = local.post("/api/chat", headers=H, json={"question": "hi", "history": long_history, "model": "auto"})
    assert r.json["model"] == f"claude:{llm.MODEL}" and r.json["switched"] == "too long for the local model"
    assert local.calls == []


def test_mass_delete(client):
    ids = [upload(client, f"f{i}.txt", f"Rent note number {i}.".encode()).json["id"] for i in range(3)]
    assert client.get("/api/documents").json[0]["model"] == "Claude Opus 5.5"  # who read it
    r = client.post("/api/documents/delete", json={"ids": ids[:2] + [99999]})
    assert r.status_code == 200 and r.json["deleted"] == 2
    left = client.get("/api/documents").json
    assert [d["id"] for d in left] == [ids[2]]
    assert {h["filename"] for h in client.get("/api/search?q=rent").json} == {"f2.txt"}  # FTS rows gone too
    assert client.post("/api/documents/delete", json={"ids": "all"}).status_code == 400
    assert client.post("/api/documents/delete", json={"ids": []}).json["deleted"] == 0


def test_old_database_gains_model_column(tmp_path, monkeypatch):
    import sqlite3
    from app import db
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, filename TEXT NOT NULL, uploaded_at TEXT NOT NULL "
                "DEFAULT (datetime('now')), char_count INTEGER NOT NULL, title TEXT NOT NULL DEFAULT '', doc_type "
                "TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT '', keywords TEXT NOT NULL DEFAULT '', "
                "analysis TEXT NOT NULL DEFAULT '')")
    con.execute("INSERT INTO documents (filename, char_count) VALUES ('old.txt', 5)")
    con.commit()
    con.close()
    monkeypatch.setattr(db, "DB_PATH", str(path))
    doc = db.list_documents()[0]
    assert doc["model"] == "" and doc["sha256"] is None and doc["has_file"] is False


# The schema as released before originals, hashes and the usage log (commit 894b984):
# the live library was made with exactly this.
SCHEMA_894B984 = """
CREATE TABLE documents (
    id INTEGER PRIMARY KEY, filename TEXT NOT NULL, uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
    char_count INTEGER NOT NULL, title TEXT NOT NULL DEFAULT '', doc_type TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '', keywords TEXT NOT NULL DEFAULT '', analysis TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '');
CREATE TABLE chunks (id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL, text TEXT NOT NULL);
CREATE TABLE instructions (id INTEGER PRIMARY KEY, text TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')));
CREATE VIRTUAL TABLE docs_fts USING fts5(filename, title, doc_type, summary, keywords, tokenize='porter unicode61');
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, tokenize='porter unicode61');
INSERT INTO documents (id, filename, char_count, title, doc_type, summary, keywords, analysis, model)
    VALUES (1, 'manual.pdf', 41, 'Tractor manual', 'manual', 'How to service it.', 'oil, filter', 'full', 'qwen3.5:9b');
INSERT INTO docs_fts (rowid, filename, title, doc_type, summary, keywords)
    VALUES (1, 'manual.pdf', 'Tractor manual', 'manual', 'How to service it.', 'oil, filter');
INSERT INTO chunks (id, doc_id, idx, text) VALUES (1, 1, 0, 'Change the hydraulic oil every 400 hours.');
INSERT INTO chunks_fts (rowid, text) VALUES (1, 'Change the hydraulic oil every 400 hours.');
INSERT INTO instructions (text) VALUES ('Use litres.');
"""


def test_library_from_before_kept_files_still_works(client, tmp_path, monkeypatch):
    import sqlite3
    from app import db
    path = tmp_path / "live.db"
    con = sqlite3.connect(path)
    con.executescript(SCHEMA_894B984)
    con.close()
    monkeypatch.setattr(db, "DB_PATH", str(path))
    doc = client.get("/api/documents").json[0]
    assert doc["title"] == "Tractor manual" and doc["model"] == "qwen3.5:9b"
    assert doc["has_file"] is False and doc["sha256"] is None and doc["file_size"] is None
    assert client.get("/api/documents?q=tractor").json[0]["id"] == 1                    # metadata search
    assert client.get("/api/search?q=hydraulic").json[0]["doc_id"] == 1                 # content search
    r = client.post("/api/chat", json={"question": "When do I change the hydraulic oil?"}, headers=H)
    assert r.status_code == 200 and "400 hours" in client.seen["msg"] and "- Use litres." in client.seen["system"]
    assert client.get("/api/documents/1/file").status_code == 404                       # no original kept
    assert upload(client, "manual.pdf.txt", b"Change the hydraulic oil every 400 hours.").status_code == 200  # never a duplicate
    assert client.get("/api/usage").json["total"]["calls"] == 1                         # the chat above
    assert client.delete("/api/documents/1").status_code == 204
    assert [d["filename"] for d in client.get("/api/documents").json] == ["manual.pdf.txt"]


def test_upload_keeps_the_original_file(client, tmp_path):
    r = upload(client)
    assert r.status_code == 200 and r.json["has_file"] is True
    doc = client.get("/api/documents").json[0]
    sha = hashlib.sha256(LEASE).hexdigest()
    assert doc["has_file"] is True and doc["sha256"] == sha and doc["file_size"] == len(LEASE)
    assert doc["mime"] == "text/plain" and "file_path" not in doc
    assert (tmp_path / "files" / sha).read_bytes() == LEASE                              # stored by content hash
    f = client.get(f"/api/documents/{doc['id']}/file")
    assert f.status_code == 200 and f.data == LEASE
    assert f.headers["Content-Type"] == "text/plain; charset=utf-8"
    assert f.headers["Content-Disposition"] == "inline; filename=lease.txt"
    assert f.headers["X-Content-Type-Options"] == "nosniff"


def test_kept_pdf_opens_in_the_browser_and_html_downloads(client, monkeypatch):
    from app import llm
    monkeypatch.setattr(llm, "transcribe_pdf", lambda c, batches, model=None, **kw: "The deposit is 1800 euros.")
    pdf = blank_pdf(2)
    doc_id = upload(client, "Handbuch für Öl.pdf", pdf).json["id"]
    f = client.get(f"/api/documents/{doc_id}/file")
    assert f.data == pdf and f.mimetype == "application/pdf"
    disposition = f.headers["Content-Disposition"]
    assert disposition.startswith("inline;") and "filename*=UTF-8''Handbuch%20f%C3%BCr%20%C3%96l.pdf" in disposition
    doc_id = upload(client, "page.html", b"<script>alert(1)</script> The rent is 900 euros.").json["id"]
    f = client.get(f"/api/documents/{doc_id}/file")
    assert f.mimetype == "text/html" and f.headers["Content-Disposition"] == "attachment; filename=page.html"


def test_documents_without_a_kept_file(client):
    from app import db
    old = db.add_document("old.txt", ["An old rent note."], {"title": "Old"}, "full")
    assert db.stored_file(old) is None
    assert client.get(f"/api/documents/{old}/file").status_code == 404
    assert client.get("/api/documents/9999/file").status_code == 404
    assert client.get("/api/documents").json[0]["has_file"] is False


def test_identical_files_are_stored_once_and_removed_with_the_last_copy(client, tmp_path):
    from app import db
    a = upload(client).json["id"]
    b = upload(client, "copy.txt", on_duplicate="keep").json["id"]
    other = upload(client, "other.txt", b"A different rent note.").json["id"]
    files = tmp_path / "files"
    assert len(list(files.iterdir())) == 2                                               # a and b share one file
    assert db.stored_file(b)["path"] == db.stored_file(a)["path"]
    assert client.delete(f"/api/documents/{a}").status_code == 204
    assert len(list(files.iterdir())) == 2 and client.get(f"/api/documents/{b}/file").data == LEASE
    assert client.post("/api/documents/delete", json={"ids": [b, other]}).json["deleted"] == 2
    assert list(files.iterdir()) == []                                                   # last users gone


def test_duplicate_upload_waits_for_a_decision(client):
    first = upload(client).json
    r = upload(client, "again.txt")                                                      # same bytes, other name
    assert r.status_code == 409 and "already in the library" in r.json["error"]
    dup = r.json["duplicate"]
    assert dup["id"] == first["id"] and dup["filename"] == "lease.txt" and dup["model"] == "Claude Opus 5.5"
    assert dup["created_at"] and dup["copies"] == 1
    assert len(client.get("/api/documents").json) == 1                                   # nothing added
    assert upload(client, "again.txt", on_duplicate="maybe").status_code == 400
    kept = upload(client, "again.txt", on_duplicate="keep").json
    assert kept["replaced"] == [] and len(client.get("/api/documents").json) == 2
    r = upload(client, "third.txt", on_duplicate="replace")                              # replaces both copies
    assert r.status_code == 200 and sorted(r.json["replaced"]) == sorted([first["id"], kept["id"]])
    docs = client.get("/api/documents").json
    assert [d["filename"] for d in docs] == ["third.txt"]
    assert client.get(f"/api/documents/{docs[0]['id']}/file").data == LEASE              # file survived the replace
    assert {h["filename"] for h in client.get("/api/search?q=rent").json} == {"third.txt"}
    assert upload(client, "new.txt", b"Something else entirely.").status_code == 200     # other bytes: no duplicate


def test_replace_keeps_the_old_copy_when_the_new_read_fails(client, monkeypatch):
    from app import llm, main
    monkeypatch.setattr(llm, "transcribe_pdf", lambda c, batches, model=None, **kw: "The deposit is 1800 euros.")
    pdf = blank_pdf(1)
    first = upload(client, "scan.pdf", pdf).json["id"]
    monkeypatch.setattr(main, "PDF_EXTRACTOR", "local")                                  # the re-read finds no text
    assert upload(client, "scan.pdf", pdf, on_duplicate="replace").status_code == 400
    assert [d["id"] for d in client.get("/api/documents").json] == [first]
    assert client.get(f"/api/documents/{first}/file").data == pdf


def test_cost_uses_the_price_table():
    from app import llm

    def u(i=0, o=0, cr=0, cw=0):
        return {"input": i, "output": o, "cache_read": cr, "cache_write": cw}

    assert llm.cost("claude-opus-5-5", u(1_000_000, 1_000_000)) == pytest.approx(24.0)
    assert llm.cost("claude-sonnet-5-5", u(cr=1_000_000, cw=1_000_000)) == pytest.approx(0.2 + 2.5)  # 0.1x / 1.25x
    assert llm.cost("claude-haiku-5-5", u(50_000, 1_000)) == pytest.approx((50_000 * 0.10 + 1_000 * 0.50) / 1e6)
    assert llm.cost("claude-haiku-5-5", u(150_000, 1_000)) == pytest.approx((150_000 * 0.50 + 1_000 * 2.50) / 1e6)
    assert llm.cost("claude-fable-5-2", u(1_000, 1_000)) == pytest.approx(0.048)
    assert llm.cost("claude-unknown-9", u(1_000, 1_000)) == 0


def test_claude_calls_report_their_usage():
    from types import SimpleNamespace as NS
    from app import llm
    usage = NS(input_tokens=1200, output_tokens=300, cache_read_input_tokens=None, cache_creation_input_tokens=50)
    reply = NS(stop_reason="end_turn", usage=usage, content=[
        NS(type="text", text='{"title": "t", "doc_type": "d", "summary": "s", "keywords": []}')])

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return reply

    fake = NS(beta=NS(messages=NS(create=lambda **kw: reply, stream=lambda **kw: Stream())))
    seen = []
    llm.analyze(fake, "a.txt", "text", "claude-haiku-5-5", on_usage=lambda *call: seen.append(call))
    llm.transcribe_pdf(fake, [(1, 20, 40, b"x"), (21, 40, 40, b"y")], "claude-haiku-5-5",
                       on_usage=lambda *call: seen.append(call))
    assert [c[0] for c in seen] == ["analyze", "transcribe", "transcribe"]
    assert seen[0][1:] == ("claude", "claude-haiku-5-5", {"input": 1200, "output": 300, "cache_read": 0, "cache_write": 50})
    assert llm.chat(fake, [], "hi", "claude-haiku-5-5")[1]["cache_write"] == 50


def _usage_rows():
    from app import db
    with db.connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM usage ORDER BY id")]


def test_upload_and_chat_log_each_model_call(local, monkeypatch):
    from app import extract, ollama
    monkeypatch.setattr(extract, "pdf_pages", lambda data: [("", b"png"), ("Typed page.", None), ("", b"png")])
    doc_id = upload(local, "scan.pdf", blank_pdf(3)).json["id"]
    local.post("/api/chat", json={"question": "How much is the deposit?"})
    rows = _usage_rows()
    assert [(r["kind"], r["model"]) for r in rows] == [
        ("transcribe", ollama.VISION_MODEL), ("transcribe", ollama.VISION_MODEL),        # scanned pages only
        ("analyze", ollama.MODEL), ("chat", ollama.MODEL)]
    assert [r["doc_id"] for r in rows] == [doc_id, doc_id, doc_id, None]
    assert [(r["input_tokens"], r["output_tokens"]) for r in rows] == [(800, 9), (800, 9), (900, 60), (7, 2)]
    assert all(r["backend"] == "ollama" and r["cost_usd"] == 0 for r in rows)


def test_claude_chat_is_logged_with_its_cost(client):
    client.post("/api/chat", json={"question": "hi", "model": "claude:claude-sonnet-5-5"}, headers=H)
    (row,) = _usage_rows()
    assert (row["kind"], row["backend"], row["model"]) == ("chat", "claude", "claude-sonnet-5-5")
    assert row["cost_usd"] == pytest.approx((1 * 2.0 + 1 * 10.0) / 1e6)


def test_failed_upload_still_logs_what_it_cost(client, monkeypatch):
    from app import llm

    def empty_transcribe(c, batches, model=None, on_usage=None):
        on_usage("transcribe", "claude", model, {"input": 3000, "output": 10, "cache_read": 0, "cache_write": 0})
        return ""

    monkeypatch.setattr(llm, "transcribe_pdf", empty_transcribe)
    assert upload(client, "blank.pdf", blank_pdf(1)).status_code == 400
    (row,) = _usage_rows()
    assert row["doc_id"] is None and row["cost_usd"] == pytest.approx((3000 * 4 + 10 * 20) / 1e6)


def test_usage_report_totals_per_model_and_day(client):
    from app import db
    db.add_usage([
        ("chat", "claude", "claude-opus-5-5", {"input": 1000, "output": 500, "cache_read": 2000, "cache_write": 0}, 0.0108),
        ("analyze", "claude", "claude-haiku-5-5", {"input": 10_000, "output": 200}, 0.0011),
        ("chat", "ollama", "qwen3.5:9b", {"input": 700, "output": 50}, 0.0),
    ], doc_id=None)
    with db.connect() as con:
        con.execute("INSERT INTO usage (created_at, kind, backend, model, input_tokens, cost_usd) VALUES "
                    "(datetime('now', '-1 day'), 'chat', 'claude', 'claude-opus-5-5', 100, 0.5),"
                    "(datetime('now', '-40 days'), 'chat', 'claude', 'claude-opus-5-5', 999, 5.0)")
    r = client.get("/api/usage").json                                                    # 30 days by default
    assert r["days"] == 30 and r["total"]["calls"] == 4
    assert r["total"]["input_tokens"] == 1000 + 10_000 + 700 + 100 and r["total"]["cache_read_tokens"] == 2000
    assert r["total"]["cost_usd"] == pytest.approx(0.5119)
    assert [(m["label"], m["calls"]) for m in r["by_model"]] == [
        ("Claude Opus 5.5", 2), ("Claude Haiku 5.5", 1), ("qwen3.5:9b", 1)]           # costliest first
    assert [d["calls"] for d in r["by_day"]] == [3, 1] and r["by_day"][0]["day"] > r["by_day"][1]["day"]
    assert client.get("/api/usage?days=365").json["total"]["calls"] == 5
    assert client.get("/api/usage?days=1").json["total"]["calls"] == 3                  # today only
    assert client.get("/api/usage?days=0").status_code == 400
    assert client.get("/api/usage?days=week").status_code == 400


def test_usage_report_when_nothing_was_logged(client):
    r = client.get("/api/usage").json
    assert r["total"] == {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                          "cache_write_tokens": 0, "cost_usd": 0}
    assert r["by_model"] == [] and r["by_day"] == []


def test_instructions_reach_the_system_prompt(client):
    from app import llm
    r = client.post("/api/instructions", json={"text": "Give volumes in litres: 1 US gallon = 3.785 L."})
    assert r.status_code == 201 and r.json["enabled"] is True
    off = client.post("/api/instructions", json={"text": "Answer in French.", "enabled": False}).json
    client.post("/api/chat", json={"question": "How big is the tank?"}, headers=H)
    system = client.seen["system"]
    assert system.startswith(llm.SYSTEM) and "- Give volumes in litres: 1 US gallon = 3.785 L." in system
    assert "French" not in system
    assert client.patch(f"/api/instructions/{off['id']}", json={"enabled": True}).json["enabled"] is True
    client.post("/api/chat", json={"question": "How big is the tank?"}, headers=H)
    assert "French" in client.seen["system"]
    assert client.patch(f"/api/instructions/{off['id']}", json={"text": "Answer in Dutch."}).json["text"] == "Answer in Dutch."
    assert client.delete(f"/api/instructions/{r.json['id']}").status_code == 204
    assert [x["text"] for x in client.get("/api/instructions").json] == ["Answer in Dutch."]


def test_no_instructions_keeps_the_plain_system_prompt(client):
    from app import llm
    client.post("/api/chat", json={"question": "hi"}, headers=H)
    assert client.seen["system"] == llm.SYSTEM


def test_instruction_validation(client):
    assert client.post("/api/instructions", json={"text": "  "}).status_code == 400
    assert client.post("/api/instructions", json={}).status_code == 400
    rid = client.post("/api/instructions", json={"text": "x"}).json["id"]
    assert client.patch(f"/api/instructions/{rid}", json={"enabled": "yes"}).status_code == 400
    assert client.patch("/api/instructions/9999", json={"enabled": False}).status_code == 404
    assert client.delete("/api/instructions/9999").status_code == 404


def test_ollama_gets_the_instructions_too(local):
    local.post("/api/instructions", json={"text": "Use litres."})
    local.post("/api/chat", json={"question": "tank?", "model": "ollama:qwen3.5:9b"})
    assert "- Use litres." in local.calls[-1]["messages"][0]["content"]


def test_analysis_limits_differ_per_backend(monkeypatch):
    """Claude reads up to 500K characters for metadata; local models keep 40K."""
    from types import SimpleNamespace
    from app import llm, ollama
    assert llm.ANALYZE_CHARS == 500_000 and ollama.ANALYZE_CHARS == 40_000
    text = "word " * 20_000  # 100,000 characters
    sent = {}

    def create(**kw):
        sent["claude"] = kw["messages"][0]["content"]
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(
            type="text", text='{"title": "t", "doc_type": "d", "summary": "s", "keywords": []}')])

    fake = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert llm.analyze(fake, "big.txt", text, "claude-haiku-5-5")[1] == "full"
    assert text in sent["claude"]

    def local_chat(model, messages, **kw):
        sent["local"] = messages[0]["content"]
        return {"message": {"content": '{"title": "t", "doc_type": "d", "summary": "s", "keywords": []}'}}

    monkeypatch.setattr(ollama, "_chat", local_chat)
    assert ollama.analyze(None, "big.txt", text)[1] == "partial"
    assert text[:40_000] in sent["local"] and text[:40_001] not in sent["local"]


def test_too_large_upload_gets_a_json_error(client, monkeypatch):
    from app import main
    monkeypatch.setitem(main.app.config, "MAX_CONTENT_LENGTH", 1000)
    r = upload(client, "big.txt", b"x" * 5000)
    assert r.status_code == 413 and "upload limit" in r.json["error"]


def test_models_reports_upload_limit(client, monkeypatch):
    from app import ollama
    monkeypatch.setattr(ollama, "models", lambda: [])
    assert client.get("/api/models").json["max_upload_mb"] == 200


def test_pdf_batches_split_further_when_too_big():
    from app import extract
    pdf = blank_pdf(20)
    limit = int(len(pdf) * 0.75)  # the whole file doesn't fit; halves do
    batches = extract.pdf_batches(pdf, size=20, max_bytes=limit)
    assert len(batches) > 1 and all(len(b[3]) <= limit for b in batches)
    pages = [p for first, last, total, _ in batches for p in range(first, last + 1)]
    assert pages == list(range(1, 21)) and {b[2] for b in batches} == {20}  # every page once, in order


def test_single_page_too_big_is_reported():
    import pytest
    from app import extract
    with pytest.raises(ValueError, match="page 1 alone"):
        extract.pdf_batches(blank_pdf(2), size=20, max_bytes=10)
