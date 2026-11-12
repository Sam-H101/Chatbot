import base64
import hashlib
import io
import re

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import db, jobs, llm, main, ollama, vectors
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    # No embedding model unless a test brings one (fake_embedder): search is by keywords, and
    # vectors.py starts each test with no pause and nothing cached.
    monkeypatch.setattr(ollama, "embed", no_embedder)
    monkeypatch.setattr(vectors, "_retry_at", 0.0)
    monkeypatch.setattr(vectors, "_last_error", None)
    monkeypatch.setattr(vectors, "_cache", None)
    monkeypatch.setattr(db, "FILES_DIR", "")  # originals go to tmp_path / "files"
    monkeypatch.setattr(main, "BACKEND", "claude")
    # No worker thread: tests run the queue themselves (jobs.drain), see upload() below.
    monkeypatch.setattr(jobs, "BACKGROUND", False)
    monkeypatch.setattr(jobs, "_keys", {})
    monkeypatch.setattr(jobs, "CONFIRM_COST_USD", 100.0)  # tests of the confirmation set their own
    monkeypatch.setattr(llm, "client_for", lambda key: object())
    monkeypatch.setattr(llm, "analyze", lambda c, name, text, model=None, **kw: (
        {"title": "Lease", "doc_type": "contract", "summary": "A flat lease.", "keywords": ["rent", "tenant"]}, "full"))
    seen = {}

    def fake_chat(c, history, msg, model=None, system=None, on_usage=None):
        seen["history"], seen["msg"], seen["model"], seen["system"] = history, msg, model, system
        usage = {"input": 1, "output": 1, "cache_read": 0, "cache_write": 0}
        if on_usage:
            on_usage("chat", "claude", model, usage)
        return "42", usage

    monkeypatch.setattr(llm, "chat", fake_chat)
    c = main.app.test_client()
    c.seen = seen
    return c


H = {"X-API-Key": "k"}
LEASE = b"The monthly rent is 900 euros.\n\nPets are allowed."


def no_embedder(texts, model=None, kind="embed", on_usage=None):
    from app import ollama
    raise ollama.OllamaUnavailable("Could not reach Ollama (no embedding model in tests)")


# What the fake embedding model "understands": each word counts towards one meaning.
MEANINGS = [{"hydraulic", "oil", "fluid", "transmission", "lubricant"}, {"capacity", "litres", "fill", "quantity"},
            {"rent", "euros", "deposit"}, {"pets", "dog"}, {"brake", "brakes", "pedal"}]


def fake_embedder(monkeypatch, calls=None):
    """ollama.embed, faked: a vector of how many words of each meaning a text has (plus a
    little of none, so no vector is all zeros). Records (kind, texts) in `calls`."""
    from app import ollama

    def embed(texts, model=None, kind="embed", on_usage=None):
        if calls is not None:
            calls.append((kind, list(texts)))
        words = [re.findall(r"[a-z]+", t.lower()) for t in texts]
        if on_usage:
            on_usage(kind, "ollama", model or ollama.EMBED_MODEL, {"input": sum(map(len, words)), "output": 0,
                                                                  "cache_read": 0, "cache_write": 0})
        return [[float(sum(w in m for w in ws)) for m in MEANINGS] + [0.1] for ws in words]

    monkeypatch.setattr(ollama, "embed", embed)
    return calls


def post(client, name="lease.txt", body=LEASE, **form):
    """POST a file to /api/upload and nothing more: an accepted one waits in the queue."""
    return client.post("/api/upload", data={"file": (io.BytesIO(body), name), **form}, headers=H)


def upload(client, name="lease.txt", body=LEASE, **form):
    """POST a file; an accepted one (202) is then read, as the worker thread would."""
    from app import jobs
    r = post(client, name, body, **form)
    if r.status_code == 202:
        jobs.drain()
    return r


def add(client, name="lease.txt", body=LEASE, **form):
    """Upload a file and read it; returns its finished job."""
    r = upload(client, name, body, **form)
    assert r.status_code == 202, r.json
    return client.get(f"/api/jobs/{r.json['job']['id']}").json


def job_of(client, r):
    return client.get(f"/api/jobs/{r.json['job']['id']}").json


def test_upload_search_delete(client):
    job = add(client)
    assert job["status"] == "done" and job["result"]["title"] == "Lease" and job["kind"] == "upload"
    assert client.get("/api/documents?q=tenant").json[0]["filename"] == "lease.txt"   # metadata search
    assert client.get("/api/documents?q=zebra").json == []
    hits = client.get("/api/search?q=renting pets").json                               # content search, stemmed
    assert hits and "[" in hits[0]["snippet"]
    assert client.get('/api/search?q=" OR ) NEAR(').status_code == 200                 # hostile FTS syntax
    assert client.delete(f"/api/documents/{job['doc_id']}").status_code == 204
    assert client.get("/api/documents").json == []
    assert client.get("/api/search?q=rent").json == []


def test_upload_is_queued_and_reported_as_a_job(client):
    from app import jobs
    r = post(client)
    assert r.status_code == 202
    job = r.json["job"]
    assert job["status"] == "queued" and job["filename"] == "lease.txt" and job["model"] == "claude:claude-opus-5-5"
    assert job["model_label"] == "Claude Opus 5.5" and job["options"]["claude_pages"] == "scanned"
    assert "file_path" not in job and "partial" not in job
    assert [j["id"] for j in client.get("/api/jobs").json] == [job["id"]]
    assert client.get("/api/documents").json == []                                       # not read yet
    jobs.drain()
    done = client.get(f"/api/jobs/{job['id']}").json
    assert done["status"] == "done" and done["stage"] == "Done" and done["error"] is None
    assert done["result"]["model_label"] == "Claude Opus 5.5" and done["result"]["chunks"] == 1
    assert done["result"]["replaced"] == [] and done["result"]["switched"] is None
    assert done["doc_id"] == client.get("/api/documents").json[0]["id"]
    assert client.get("/api/jobs/9999").status_code == 404


def test_the_worker_thread_reads_queued_uploads(client, monkeypatch):
    import time
    from app import jobs
    monkeypatch.setattr(jobs, "BACKGROUND", True)
    monkeypatch.setattr(jobs, "_thread", None)
    try:
        r = post(client)                                                                  # the request starts the worker
        for _ in range(200):
            job = job_of(client, r)
            if job["status"] == "done":
                break
            time.sleep(0.05)
        assert job["status"] == "done" and job["result"]["title"] == "Lease"
        assert jobs._thread.is_alive() and jobs._thread.name == "jobs"
    finally:
        jobs.stop()


def test_chat_grounds_on_excerpts(client):
    doc_id = add(client)["doc_id"]
    r = client.post("/api/chat", json={"question": "How much is the rent?", "history": []}, headers=H)
    assert r.json["answer"] == "42" and r.json["scope"] is None
    assert r.json["sources"] == [{"doc_id": doc_id, "filename": "lease.txt", "title": "Lease", "has_file": True,
                                  "pdf": False, "pages": [], "excerpts": [0]}]               # a text file has no pages
    (e,) = r.json["excerpts"]
    assert e["text"] == LEASE.decode() and (e["match"], e["page_start"], e["idx"]) == ("keywords", None, 0)
    assert '<excerpt file="lease.txt">\nThe monthly rent is 900 euros.' in client.seen["msg"]
    assert r.json["turns"][0]["content"] == client.seen["msg"]
    assert client.post("/api/chat", json={"question": "hi", "history": "none"}, headers=H).status_code == 400
    assert client.post("/api/chat", json={"question": "hi", "history": ["x"]}, headers=H).status_code == 400


def test_requires_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client.post("/api/chat", json={"question": "hi"}).status_code == 401


def test_a_claude_upload_needs_a_key_before_it_is_queued(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/api/upload", data={"file": (io.BytesIO(LEASE), "a.txt"), "model": "claude:claude-haiku-5-5"})
    assert r.status_code == 401 and client.get("/api/jobs").json == []
    assert post(client, claude_pages="some").status_code == 400


def test_chunking():
    from app.extract import chunk_text
    chunks = chunk_text("a" * 4000 + "\n\n" + "b " * 100, size=1500)
    assert all(len(c) <= 1500 for c in chunks) and "".join(chunks).count("a") == 4000


def test_page_breaks_survive_chunking():
    """PDF text keeps its pages apart with form feeds, and chunks keep every one of them
    (empty pages too), so a chunk's page can be counted later. Readers never see them."""
    from app.extract import chunk_text, join_pages, plain
    text = join_pages(["First page.", "", "Third page, part one.\n\nThird page, part two.", "Fourth " + "x" * 3000])
    assert text.count("\f") == 3
    chunks = chunk_text(text, size=1500)
    assert chunks[0] == "First page.\n\n\f\fThird page, part one.\n\nThird page, part two."
    assert chunks[1].startswith("\fFourth x") and all(len(c) <= 1500 for c in chunks)
    assert "".join(chunks).count("\f") == 3 and "".join(chunks).count("x") == 3000
    assert chunk_text("\f \f\n\n\f") == []                                                # blank pages: no text
    assert plain(chunks[0]) == "First page.\n\nThird page, part one.\n\nThird page, part two."
    assert plain(chunks[1]).startswith("Fourth") and plain("one\ftwo") == "one\n\ntwo"
    assert join_pages(["a\fb", "c"]) == "a\nb\fc"                                         # a stray form feed isn't a page


def test_chunks_know_the_pages_they_span():
    """A chunk's pages come from the page breaks before it and in it: one that opens a page
    starts on it, one that runs onto the next pages gets a range, and empty pages count."""
    from app.extract import chunk_text, join_pages, page_ranges
    text = join_pages(["First page.", "", "Third page, part one.\n\nThird page, part two.", "Fourth " + "x" * 3000,
                       "", "Sixth page."])
    chunks = chunk_text(text, size=1500)
    assert chunks[3] == "x" * 8 + "\n\n\f\fSixth page."                                   # page 4's end, then page 6
    assert page_ranges(chunks) == [(1, 3), (4, 4), (4, 4), (4, 6)]
    assert page_ranges(chunk_text(join_pages(["", "", "Third page."]))) == [(3, 3)]       # leading empty pages
    assert page_ranges(chunk_text("No pages at all.")) == [(1, 1)] and page_ranges([]) == []
    assert page_ranges(["a\n\n\fb\n\n\fc", "\fd"]) == [(1, 3), (4, 4)]


def test_claude_page_marks_split_a_batch_into_pages():
    from app.llm import page_names, split_pages
    assert split_pages("<<<PAGE 3>>>\nA\n\n<<<PAGE 7>>>\nB", [3, 7]) == {3: "A", 7: "B"}
    assert split_pages("<<<PAGE 1>>>\nA\n<<<PAGE 2>>>\nB", [3, 7]) == {3: "A", 7: "B"}      # one mark each: order wins
    assert split_pages("Intro\n<<<PAGE 7>>>\nB", [3, 7]) == {3: "Intro", 7: "B"}         # by number otherwise
    assert split_pages("<<<PAGE 3>>>\nA\n<<<PAGE 9>>>\nA2\n<<<PAGE 7>>>\nB\n<<<PAGE 8>>>\nC", [3, 7, 8]) == {
        3: "A\n\nA2", 7: "B", 8: "C"}                                                      # stray mark: page before
    assert split_pages("No marks at all.", [3, 7]) == {3: "No marks at all.", 7: ""}       # nothing is lost
    assert page_names([5]) == "page 5" and page_names([21, 22, 23]) == "pages 21-23"
    assert page_names([2, 4]) == "pages 2 and 4" and page_names([3, 7, 8]) == "pages 3, 7 and 8"


def blank_pdf(pages):
    import pypdfium2
    pdf = pypdfium2.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(612, 792)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def text_pdf(lines):
    """A one-page PDF with a real text layer (Helvetica)."""
    content = "BT /F1 14 Tf 72 720 Td 20 TL " + " ".join(f"({line}) '" for line in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >>"
            " /Contents 5 0 R >>", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            f"<< /Length {len(content)} >>\nstream\n{content}\nendstream"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode() + "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    return out + f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


def mixed_pdf(layout):
    """Pages 'T' with a text layer, 'S' without one (a scan, as far as reading goes)."""
    import pypdfium2
    out = pypdfium2.PdfDocument.new()
    for n, kind in enumerate(layout, 1):
        if kind == "T":
            src = pypdfium2.PdfDocument(text_pdf([f"Page {n} has a real text layer about the pump."]))
            out.import_pages(src)
            src.close()
        else:
            out.new_page(612, 792)
    buf = io.BytesIO()
    out.save(buf)
    return buf.getvalue()


def claude_reader(calls, fail=lambda pages: False, during=lambda pages: None):
    """A fake Anthropic client. Streamed calls 'read' the PDF they get: one <<<PAGE n>>>
    mark per page the prompt names, recorded in `calls`; plain calls answer metadata."""
    import pypdfium2
    from types import SimpleNamespace as NS

    def numbers(which):
        if "-" in which:
            a, b = map(int, re.findall(r"\d+", which))
            return list(range(a, b + 1))
        return list(map(int, re.findall(r"\d+", which)))

    class Stream:
        def __init__(self, kw):
            self.kw = kw

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            doc, prompt = self.kw["messages"][0]["content"]
            pages = numbers(re.search(r"holds (.+?) of a \d+-page", prompt["text"]).group(1))
            pdf = pypdfium2.PdfDocument(base64.b64decode(doc["source"]["data"]))
            calls.append({"pages": pages, "prompt": prompt["text"], "pdf_pages": len(pdf), "model": self.kw["model"]})
            pdf.close()
            during(pages)
            if fail(pages):
                raise RuntimeError(f"batch {pages[0]} broke")
            text = "".join(f"<<<PAGE {n}>>>\nScanned page {n} says the deposit is {n}00 euros.\n\n" for n in pages)
            return NS(stop_reason="end_turn", content=[NS(type="text", text=text)], usage=NS(
                input_tokens=2000 * len(pages), output_tokens=600 * len(pages), cache_read_input_tokens=0,
                cache_creation_input_tokens=0))

    meta = NS(stop_reason="end_turn", usage=NS(input_tokens=500, output_tokens=50, cache_read_input_tokens=0,
                                                cache_creation_input_tokens=0), content=[NS(type="text", text=(
        '{"title": "Pump manual", "doc_type": "manual", "summary": "s", "keywords": ["pump"]}'))])
    return NS(beta=NS(messages=NS(create=lambda **kw: meta, stream=lambda **kw: Stream(kw))))


def doc_text(doc_id):
    from app import db
    with db.connect() as con:
        return "\n\n".join(r[0] for r in con.execute("SELECT text FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,)))


def test_pdf_is_read_by_claude_in_page_batches(client, monkeypatch):
    from app import extract, llm
    monkeypatch.setattr(extract.pdf_batches, "__defaults__", (20, None, None))
    seen = []

    def fake_transcribe(c, batches, model=None, **kw):
        seen.extend((pages[0], pages[-1], total) for pages, total, _ in batches)
        return {p: "The deposit is 1800 euros." if p == 1 else "" for pages, _, _ in batches for p in pages}

    monkeypatch.setattr(llm, "transcribe_pdf", fake_transcribe)
    assert add(client, "scan.pdf", blank_pdf(45))["status"] == "done"
    assert seen == [(1, 20, 45), (21, 40, 45), (41, 45, 45)]                             # blank pages are all scans
    assert client.get("/api/search?q=deposit").json[0]["filename"] == "scan.pdf"


def test_claude_reads_only_scanned_pages_by_default(client, monkeypatch):
    """Pages with a text layer keep it; only the scanned ones go to Claude, as a new PDF
    of just those pages, with their page numbers in the prompt. The document's text has
    every page in order, separated by form feeds."""
    from app import llm
    calls = []
    monkeypatch.setattr(llm, "client_for", lambda key: claude_reader(calls))
    job = add(client, "pump.pdf", mixed_pdf("TSTS"), model="claude:claude-haiku-5-5")
    assert job["status"] == "done" and job["result"]["model_label"] == "Claude Haiku 5.5"
    assert [c["pages"] for c in calls] == [[2, 4]] and calls[0]["pdf_pages"] == 2
    assert "holds pages 2 and 4 of a 4-page document" in calls[0]["prompt"]
    est = job["estimate"]
    assert (est["pages"], est["claude_pages"], est["mode"], est["model"]) == (4, 2, "scanned", "claude-haiku-5-5")
    text = doc_text(job["doc_id"])
    assert text.count("\f") == 3
    order = [text.index(s) for s in ("Page 1 has", "Scanned page 2", "Page 3 has", "Scanned page 4")]
    assert order == sorted(order)
    hit = client.get("/api/search?q=deposit").json[0]
    assert hit["filename"] == "pump.pdf" and "\f" not in hit["snippet"]
    client.post("/api/chat", json={"question": "What is the deposit?"}, headers=H)
    assert "Scanned page 2" in client.seen["msg"] and "\f" not in client.seen["msg"]    # excerpts without form feeds
    # The checkbox "Claude reads every page": all four pages, in one batch.
    job = add(client, "pump.pdf", mixed_pdf("TSTS"), model="claude:claude-haiku-5-5", claude_pages="all",
              on_duplicate="keep")
    assert calls[-1]["pages"] == [1, 2, 3, 4] and "holds pages 1-4 of a 4-page document" in calls[-1]["prompt"]
    assert job["estimate"]["claude_pages"] == 4 and job["options"]["claude_pages"] == "all"
    assert "Page 1 has" not in doc_text(job["doc_id"])                                  # Claude's reading instead


def test_a_text_pdf_needs_no_claude_reading_at_all(client, monkeypatch):
    from app import llm
    calls = []
    monkeypatch.setattr(llm, "client_for", lambda key: claude_reader(calls))
    job = add(client, "typed.pdf", mixed_pdf("TT"), model="claude:claude-haiku-5-5")
    assert job["status"] == "done" and calls == [] and job["estimate"]["claude_pages"] == 0
    assert doc_text(job["doc_id"]).count("\f") == 1


def test_estimate_for_claude(client):
    from app import jobs, llm
    pdf = blank_pdf(45)
    est = jobs.estimate("scan.pdf", pdf, "claude-opus-5-5", True, "scanned")
    analysis_in = 45 * 600 * 4 // 4 + 100
    assert (est["pages"], est["claude_pages"]) == (45, 45)
    assert est["input_tokens"] == 45 * 2000 + analysis_in and est["output_tokens"] == 45 * 600 + 500
    per_call = [20, 20, 5]  # priced call by call: Haiku's long-prompt price never applies to a 20-page batch
    expected = sum(llm.cost("claude-opus-5-5", {"input": n * 2000, "output": n * 600}) for n in per_call)
    expected += llm.cost("claude-opus-5-5", {"input": analysis_in, "output": 500})
    assert est["cost_usd"] == pytest.approx(expected, abs=1e-6) and est["cost_usd"] > 1
    typed = jobs.estimate("typed.pdf", mixed_pdf("TT"), "claude-haiku-5-5", True, "scanned")
    assert typed["claude_pages"] == 0 and typed["cost_usd"] < 0.001
    assert jobs.estimate("typed.pdf", mixed_pdf("TT"), "claude-haiku-5-5", True, "all")["claude_pages"] == 2
    txt = jobs.estimate("a.txt", b"x" * 4000, "claude-opus-5-5", False, "scanned")
    assert txt["pages"] == 0 and txt["input_tokens"] == 1100 and txt["mode"] is None


def test_a_costly_claude_job_waits_for_confirmation(client, monkeypatch):
    from app import jobs, llm
    monkeypatch.setattr(jobs, "CONFIRM_COST_USD", 1.00)                                   # the default
    calls = []
    monkeypatch.setattr(llm, "client_for", lambda key: claude_reader(calls))
    r = upload(client, "scan.pdf", blank_pdf(60), model="claude:claude-opus-5-5")
    job = job_of(client, r)
    assert job["status"] == "awaiting_confirmation" and calls == []                      # nothing spent yet
    assert job["estimate"]["cost_usd"] > 1 and job["estimate"]["confirm_at"] == 1.0
    assert client.get("/api/documents").json == []
    assert client.post(f"/api/jobs/{job['id']}/retry", headers=H).status_code == 409    # not failed
    assert client.post(f"/api/jobs/{job['id']}/confirm", headers=H).json["job"]["status"] == "queued"
    jobs.drain()
    job = job_of(client, r)
    assert job["status"] == "done" and [c["pages"][0] for c in sorted(calls, key=lambda c: c["pages"])] == [1, 21, 41]
    assert client.post(f"/api/jobs/{job['id']}/confirm", headers=H).status_code == 409
    # A cheap job starts at once, still with its estimate.
    job = add(client, "short.pdf", blank_pdf(2), model="claude:claude-haiku-5-5")
    assert job["status"] == "done" and 0 < job["estimate"]["cost_usd"] < 1


def test_the_browser_key_is_kept_before_the_worker_can_see_the_job(client, monkeypatch):
    """The worker may be awake already: the job must not be visible before its key is kept."""
    from app import db, jobs
    seen = []
    db.add_job("upload", "a.txt", db.file_hash(b"x"), "claude:claude-haiku-5-5", {}, data=b"x",
               on_insert=lambda job_id: seen.append(db.get_job(job_id)))
    assert seen == [None]                                                                # not committed yet
    monkeypatch.setattr(jobs, "CONFIRM_COST_USD", 0)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = upload(client, "b.txt", b"Rent is due.", model="claude:claude-haiku-5-5")
    assert job_of(client, r)["status"] == "awaiting_confirmation" and jobs._keys[r.json["job"]["id"]] == "k"
    assert client.post(f"/api/jobs/{r.json['job']['id']}/confirm").status_code == 401   # Start needs a key too


def test_a_job_waiting_for_confirmation_can_be_cancelled(client, monkeypatch, tmp_path):
    from app import jobs
    monkeypatch.setattr(jobs, "CONFIRM_COST_USD", 0)                                      # every Claude job asks
    r = upload(client, model="claude:claude-haiku-5-5")
    job = job_of(client, r)
    assert job["status"] == "awaiting_confirmation" and (tmp_path / "files").exists()
    assert client.post(f"/api/jobs/{job['id']}/cancel").json["job"]["status"] == "cancelled"
    assert not any((tmp_path / "files").iterdir())                                       # its file went with it
    assert client.post(f"/api/jobs/{job['id']}/confirm").status_code == 409
    assert client.post("/api/jobs/9999/cancel").status_code == 404
    assert client.get("/api/documents").json == []


def test_auto_estimates_claude_only_when_it_switches(local, monkeypatch):
    from app import jobs, llm, ollama
    monkeypatch.setattr(jobs, "CONFIRM_COST_USD", 0.001)
    job = add(local)                                                                     # read locally: no estimate
    assert job["status"] == "done" and job["estimate"] is None and job["model"] == "auto"

    def broken(model, messages, **kw):
        raise ollama.OllamaError(f"{model}: model not found")

    monkeypatch.setattr(ollama, "_chat", broken)
    r = upload(local, "other.txt", b"Another rent note, 900 euros.")
    job = job_of(local, r)
    assert job["status"] == "awaiting_confirmation" and job["estimate"]["model"] == llm.MODEL
    assert "local model failed" in job["options"]["switched"]
    local.post(f"/api/jobs/{job['id']}/confirm", headers=H)
    jobs.drain()
    job = job_of(local, r)
    assert job["status"] == "done" and job["result"]["model"] == f"claude:{llm.MODEL}"
    assert "local model failed" in job["result"]["switched"]


def test_a_failed_claude_job_resumes_without_reading_kept_batches_again(client, monkeypatch):
    from app import jobs, llm
    monkeypatch.setattr(llm, "PDF_CONCURRENCY", 1)                                       # batches in order
    calls, broken = [], {21}
    monkeypatch.setattr(llm, "client_for", lambda key: claude_reader(calls, fail=lambda pages: pages[0] in broken))
    r = upload(client, "scan.pdf", blank_pdf(45), model="claude:claude-haiku-5-5")
    job = job_of(client, r)
    first = [c["pages"][0] for c in calls]
    # The batch after the broken one is either never sent, or (already under way) kept.
    assert job["status"] == "failed" and "batch 21 broke" in job["error"] and first[:2] == [1, 21]
    assert job["pages_kept"] == (25 if 41 in first else 20)
    broken.clear()
    assert client.post(f"/api/jobs/{job['id']}/retry", headers=H).json["job"]["status"] == "queued"
    jobs.drain()
    job = job_of(client, r)
    retried = [p for c in calls[len(first):] for p in c["pages"]]
    assert job["status"] == "done" and retried == list(range(21, 41 if 41 in first else 46))
    text = doc_text(job["doc_id"])
    assert text.count("\f") == 44 and "Scanned page 1 " in text and "Scanned page 45 " in text
    assert len([r for r in _usage_rows() if r["kind"] == "transcribe"]) == 3             # each page paid for once
    assert {(r["doc_id"], r["job_id"]) for r in _usage_rows()} == {(job["doc_id"], job["id"])}  # both attempts


def test_jobs_left_running_by_a_restart_are_queued_again(client, monkeypatch):
    """A restart: the job was running with its first batch kept, and the browser's key is
    forgotten. It is queued again, fails asking for the key, and Retry finishes it
    without reading the kept pages again."""
    from app import db, jobs, llm
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = []
    monkeypatch.setattr(llm, "client_for", lambda key: claude_reader(calls))
    r = post(client, "scan.pdf", blank_pdf(45), model="claude:claude-haiku-5-5")
    job_id = r.json["job"]["id"]
    assert db.claim_next_job()["id"] == job_id
    db.update_job(job_id, stage="Reading", partial={
        "claude:claude-haiku-5-5:scanned": {str(p): f"Kept page {p} mentions the gearbox." for p in range(1, 21)}})
    jobs._keys.clear()
    assert jobs.recover() == 1
    assert job_of(client, r)["status"] == "queued"
    jobs.drain()
    job = job_of(client, r)
    assert job["status"] == "failed" and "No Claude API key" in job["error"] and job["pages_kept"] == 20
    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 401                   # still no key
    client.post(f"/api/jobs/{job_id}/retry", headers=H)
    jobs.drain()
    assert job_of(client, r)["status"] == "done"
    assert sorted(c["pages"][0] for c in calls) == [21, 41]                              # pages 1-20 were kept
    assert client.get("/api/search?q=gearbox").json[0]["filename"] == "scan.pdf"


def test_cancelling_a_running_job_stops_it_and_adds_nothing(local, monkeypatch, tmp_path):
    from app import extract, jobs, ollama
    monkeypatch.setattr(extract, "pdf_pages", lambda data, **kw: [("", b"png")] * 3)
    fake, read, ids = ollama._chat, [], []

    def chat(model, messages, **kw):
        if messages[-1].get("images"):
            read.append(model)
            if len(read) == 1:  # the user cancels while page 1 is being read
                assert local.post(f"/api/jobs/{ids[0]}/cancel").json["job"]["status"] == "cancelled"
        return fake(model, messages, **kw)

    monkeypatch.setattr(ollama, "_chat", chat)
    r = post(local, "scan.pdf", blank_pdf(3), model="ollama:qwen3.5:9b")
    ids.append(r.json["job"]["id"])
    jobs.drain()
    job = job_of(local, r)
    assert job["status"] == "cancelled" and len(read) == 1                               # stopped after that page
    assert local.get("/api/documents").json == [] and not any((tmp_path / "files").iterdir())
    (row,) = _usage_rows()
    assert row["kind"] == "transcribe" and row["doc_id"] is None                         # the page it read is logged
    assert local.post(f"/api/jobs/{job['id']}/retry").status_code == 409


def test_a_job_cancelled_at_the_last_moment_adds_nothing(client, monkeypatch):
    """A cancel that lands after reading but before saving: the document is not added."""
    from app import db
    real = db.add_document

    def cancelled_meanwhile(*a, **kw):
        db.cancel_job(kw["job"][0])
        return real(*a, **kw)

    monkeypatch.setattr(db, "add_document", cancelled_meanwhile)
    r = upload(client)
    assert job_of(client, r)["status"] == "cancelled" and client.get("/api/documents").json == []


def test_local_pdf_mode_finds_no_text_in_a_scan(client, monkeypatch):
    from app import jobs
    monkeypatch.setattr(jobs, "PDF_EXTRACTOR", "local")
    job = add(client, "scan.pdf", blank_pdf(2))
    assert job["status"] == "failed" and job["error"] == "No text found in file"
    assert client.get("/api/documents").json == []


@pytest.fixture
def local(client, monkeypatch):
    """The Ollama backend, with Ollama's HTTP API faked. Records each request."""
    from app import main, ollama
    monkeypatch.setattr(main, "BACKEND", "ollama")
    calls = []

    def fake_chat(model, messages, **kw):
        calls.append({"model": model, "messages": messages, **kw})
        if "format" in kw and "topics" in kw["format"]["properties"]:  # notes on a section of a long document
            return {"message": {"content": '{"topics": ["deposit"], "specifications": ["1800 euros"], "procedures": [],'
                                           ' "warnings": []}'}, "prompt_eval_count": 500, "eval_count": 30}
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
    monkeypatch.setattr(extract, "pdf_pages", lambda data, **kw: [("Page one has real text.", None), ("", b"png")])
    job = add(local, "mixed.pdf", blank_pdf(2))
    assert job["status"] == "done" and job["result"]["title"] == "Scan" and job["total"] == 1  # last: passages indexed
    vision = [c for c in local.calls if c["messages"][-1].get("images")]
    assert len(vision) == 1 and vision[0]["model"] == ollama.VISION_MODEL
    assert "page 2 of a 2-page document" in vision[0]["messages"][-1]["content"]
    assert local.get("/api/search?q=deposit").json[0]["filename"] == "mixed.pdf"
    assert local.get("/api/search?q=real text").json[0]["filename"] == "mixed.pdf"
    assert doc_text(job["doc_id"]) == "Page one has real text.\n\n\fThe deposit is 1800 euros."


def test_ollama_reading_skips_pages_read_before(local, monkeypatch):
    """Pages the vision model finished are kept with the job: a retry reads only the rest."""
    from app import extract, jobs, ollama
    monkeypatch.setattr(extract, "pdf_pages", lambda data, **kw: [("", None if n in kw["skip"] else b"png")
                                                                  for n in range(1, 4)])
    fake, read = ollama._chat, []

    def flaky(model, messages, **kw):
        if messages[-1].get("images"):
            n = int(re.search(r"page (\d+)", messages[-1]["content"]).group(1))
            read.append(n)
            if n == 2 and read.count(2) == 1:
                raise ollama.OllamaError(f"{model}: out of memory")
            return {"message": {"content": f"Vision text of page {n}."}, "prompt_eval_count": 800, "eval_count": 9}
        return fake(model, messages, **kw)

    monkeypatch.setattr(ollama, "_chat", flaky)
    r = upload(local, "scan.pdf", blank_pdf(3), model="ollama:qwen3.5:9b")
    job = job_of(local, r)
    assert job["status"] == "failed" and "out of memory" in job["error"] and job["pages_kept"] == 1
    local.post(f"/api/jobs/{job['id']}/retry")
    jobs.drain()
    job = job_of(local, r)
    assert job["status"] == "done" and read == [1, 2, 2, 3]                               # page 1 not read again
    assert doc_text(job["doc_id"]) == "Vision text of page 1.\n\n\fVision text of page 2.\n\n\fVision text of page 3."


def test_blank_pdf_page_is_rendered_for_the_vision_model():
    from app import extract
    (text, png), = extract.pdf_pages(blank_pdf(1))
    assert text.strip() == "" and png[1:4] == b"PNG"
    assert list(extract.pdf_pages(blank_pdf(2), skip={1}))[0][1] is None                 # a page read before
    assert extract.page_count(blank_pdf(3)) == 3


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
    assert r["claude_pages"] == "scanned" and r["confirm_cost_usd"] == 100.0


def test_models_reports_ollama_down(local, monkeypatch):
    from app import ollama
    monkeypatch.setattr(ollama, "URL", "http://127.0.0.1:9")
    r = local.get("/api/models").json
    assert r["local"] == [] and "Could not reach Ollama" in r["local_error"]


def test_explicit_claude_model_for_upload_and_chat(local, monkeypatch):
    from app import jobs, llm
    used = {}
    monkeypatch.setattr(llm, "analyze", lambda c, name, text, model=None, **kw: (
        used.setdefault("analyze", model), ({"title": "T", "doc_type": "d", "summary": "s", "keywords": []}, "full"))[1])
    r = local.post("/api/upload", headers=H, data={
        "file": (io.BytesIO(b"The rent is 900 euros."), "a.txt"), "model": "claude:claude-haiku-5-5"})
    assert r.status_code == 202 and r.json["job"]["model_label"] == "Claude Haiku 5.5"
    jobs.drain()
    job = job_of(local, r)
    assert job["result"]["model"] == "claude:claude-haiku-5-5"
    assert job["result"]["model_label"] == "Claude Haiku 5.5" and used["analyze"] == "claude-haiku-5-5"
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
    assert post(local, model="nonsense").status_code == 400 and local.get("/api/jobs").json == []


def test_auto_switches_to_claude_when_ollama_fails(local, monkeypatch):
    from app import llm, ollama

    def broken(model, messages, **kw):
        raise ollama.OllamaError(f"{model}: model not found")

    monkeypatch.setattr(ollama, "_chat", broken)
    r = local.post("/api/chat", headers=H, json={"question": "hi", "model": "auto"})
    assert r.status_code == 200 and r.json["model"] == f"claude:{llm.MODEL}"
    assert "local model failed" in r.json["switched"]
    job = add(local)  # uploads switch too
    assert job["status"] == "done" and job["result"]["model"] == f"claude:{llm.MODEL}"
    assert "local model failed" in job["result"]["switched"] and job["estimate"]["model"] == llm.MODEL


def test_auto_without_key_reports_the_local_error(local, monkeypatch):
    from app import jobs, ollama
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def down(model, messages, **kw):
        raise ollama.OllamaUnavailable(f"Could not reach Ollama at {ollama.URL}")

    monkeypatch.setattr(ollama, "_chat", down)
    r = local.post("/api/chat", json={"question": "hi", "model": "auto"})
    assert r.status_code == 502 and "Could not reach Ollama" in r.json["error"]
    r = local.post("/api/upload", data={"file": (io.BytesIO(LEASE), "a.txt"), "model": "auto"})
    jobs.drain()
    job = job_of(local, r)
    assert job["status"] == "failed" and "Could not reach Ollama" in job["error"] and "Is it running?" in job["error"]


def test_auto_sends_long_prompts_to_claude(local, monkeypatch):
    from app import llm, ollama
    monkeypatch.setattr(ollama, "NUM_CTX", 4096)
    long_history = [{"role": "user", "content": "x" * 20000}, {"role": "assistant", "content": "ok"}]
    r = local.post("/api/chat", headers=H, json={"question": "hi", "history": long_history, "model": "auto"})
    assert r.json["model"] == f"claude:{llm.MODEL}" and r.json["switched"] == "too long for the local model"
    assert local.calls == []


def test_mass_delete(client):
    ids = [add(client, f"f{i}.txt", f"Rent note number {i}.".encode())["doc_id"] for i in range(3)]
    assert client.get("/api/documents").json[0]["model"] == "Claude Opus 5.5"  # who read it
    r = client.post("/api/documents/delete", json={"ids": ids[:2] + [99999]})
    assert r.status_code == 200 and r.json["deleted"] == 2
    left = client.get("/api/documents").json
    assert [d["id"] for d in left] == [ids[2]]
    assert {h["filename"] for h in client.get("/api/search?q=rent").json} == {"f2.txt"}  # FTS rows gone too
    assert client.post("/api/documents/delete", json={"ids": "all"}).status_code == 400
    assert client.post("/api/documents/delete", json={"ids": []}).json["deleted"] == 0


def test_reread_replaces_a_document_in_place(local, monkeypatch):
    from app import jobs, llm
    job = add(local)                                                                     # read by the local model
    doc_id, before = job["doc_id"], local.get("/api/documents").json[0]
    assert before["title"] == "Scan" and before["model"] == "qwen3.5:9b"

    def analyze(c, name, text, model=None, on_usage=None, **hooks):
        on_usage("analyze", "claude", model, {"input": 100, "output": 10})
        return {"title": "Flat lease", "doc_type": "lease", "summary": "Re-read.", "keywords": ["deposit"]}, "full"

    monkeypatch.setattr(llm, "analyze", analyze)
    r = local.post(f"/api/documents/{doc_id}/reread", json={"model": "claude:claude-haiku-5-5"}, headers=H)
    assert r.status_code == 202 and r.json["job"]["kind"] == "reread" and r.json["job"]["doc_id"] == doc_id
    assert local.post(f"/api/documents/{doc_id}/reread", json={"model": "auto"}).status_code == 409   # one at a time
    jobs.drain()
    job = job_of(local, r)
    assert job["status"] == "done" and job["doc_id"] == doc_id and job["result"]["model_label"] == "Claude Haiku 5.5"
    (doc,) = local.get("/api/documents").json
    assert doc["id"] == doc_id and doc["title"] == "Flat lease" and doc["model"] == "Claude Haiku 5.5"
    assert doc["uploaded_at"] == before["uploaded_at"] and doc["has_file"] and doc["sha256"] == before["sha256"]
    assert local.get("/api/documents?q=deposit").json[0]["id"] == doc_id                # metadata index updated
    assert local.get("/api/documents?q=scan").json == []
    assert len(local.get("/api/search?q=rent").json) == 1                                # chunks replaced, not added
    assert [r["doc_id"] for r in _usage_rows() if r["backend"] == "claude"] == [doc_id]
    assert local.post("/api/documents/9999/reread", json={"model": "auto"}).status_code == 404
    assert local.post(f"/api/documents/{doc_id}/reread", json={"model": "claude:gpt-9"}).status_code == 400


def test_deleting_a_document_cancels_its_reread(local):
    doc_id = add(local)["doc_id"]
    r = local.post(f"/api/documents/{doc_id}/reread", json={"model": "auto"})
    assert local.delete(f"/api/documents/{doc_id}").status_code == 204
    job = job_of(local, r)
    assert job["status"] == "cancelled" and job["stage"] == "The document was deleted" and job["doc_id"] is None


def test_a_reread_whose_analysis_fails_leaves_the_document_as_it_was(local, monkeypatch):
    """A re-read replaces a document's metadata. If its analysis fails (a refusal, an API
    still overloaded after its retries, a local reply that isn't JSON), the job fails and
    the document keeps its title, summary and keywords. The pages read are kept, so Retry
    only redoes the analysis. The same goes for an upload told to replace its copies."""
    from app import jobs, llm, ollama
    calls = []
    monkeypatch.setattr(llm, "client_for", lambda key: claude_reader(calls))
    doc_id = add(local, "pump.pdf", mixed_pdf("TS"))["doc_id"]                          # read locally: "Scan"
    fields = ("title", "doc_type", "summary", "keywords", "analysis", "model")
    doc = lambda: {k: local.get("/api/documents").json[0][k] for k in fields}  # noqa: E731
    before, text = doc(), doc_text(doc_id)

    def refused(c, name, text, model=None, on_usage=None, **hooks):
        raise llm.Refused("Claude declined this request")

    monkeypatch.setattr(llm, "analyze", refused)
    r = local.post(f"/api/documents/{doc_id}/reread", json={"model": "claude:claude-haiku-5-5"}, headers=H)
    jobs.drain()
    job = job_of(local, r)
    assert job["status"] == "failed" and job["error"] == (
        "Could not analyze pump.pdf: Claude declined this request. The library was left as it was")
    assert doc() == before and doc_text(doc_id) == text and local.get("/api/documents?q=notes").json
    assert [c["pages"] for c in calls] == [[2]] and job["pages_kept"] == 1
    monkeypatch.setattr(llm, "analyze", lambda c, name, text, model=None, **kw: (
        {"title": "Pump manual", "doc_type": "manual", "summary": "Re-read.", "keywords": ["seal"]}, "full"))
    local.post(f"/api/jobs/{job['id']}/retry", headers=H)
    jobs.drain()
    assert job_of(local, r)["status"] == "done" and doc()["title"] == "Pump manual"
    assert [c["pages"] for c in calls] == [[2]]                                          # page 2 not read again
    # A local model whose reply isn't JSON.
    fake = ollama._chat

    def half_json(model, messages, **kw):
        if "format" in kw:
            return {"message": {"content": '{"title": "Sca'}, "prompt_eval_count": 1, "eval_count": 1}
        return fake(model, messages, **kw)

    monkeypatch.setattr(ollama, "_chat", half_json)
    before = doc()
    r = local.post(f"/api/documents/{doc_id}/reread", json={"model": "ollama:qwen3.5:9b"})
    jobs.drain()
    job = job_of(local, r)
    assert job["status"] == "failed" and job["error"].startswith("Could not analyze pump.pdf: Unterminated string")
    assert doc() == before and local.get("/api/documents?q=seal").json[0]["id"] == doc_id
    # Replacing a duplicate: the copy in the library stays.
    job = add(local, "pump (1).pdf", mixed_pdf("TS"), on_duplicate="replace")
    assert job["status"] == "failed" and "The library was left as it was" in job["error"]
    assert [d["id"] for d in local.get("/api/documents").json] == [doc_id] and doc() == before
    # A new upload is still added, without metadata, so its text is searchable.
    job = add(local, "notes.txt", b"Grease the bearings monthly.")
    assert job["status"] == "done" and job["result"]["analysis"].startswith("none: Unterminated string")


def test_when_only_the_local_analysis_fails_claude_just_analyzes(local, monkeypatch):
    """Auto on a scan: the vision model read the pages, then the local analysis failed.
    Claude analyzes the text read here instead of reading (and billing) the pages again,
    and the estimate covers just that call. Only a failed local reading has Claude read."""
    from app import jobs, llm, ollama
    monkeypatch.setattr(jobs, "CONFIRM_COST_USD", 0)                                      # every Claude job asks
    calls, fake = [], ollama._chat
    monkeypatch.setattr(llm, "client_for", lambda key: claude_reader(calls))

    def analysis_down(model, messages, **kw):
        if "format" in kw:
            raise ollama.OllamaError(f"{model}: out of memory")
        return fake(model, messages, **kw)

    monkeypatch.setattr(ollama, "_chat", analysis_down)
    r = upload(local, "scan.pdf", mixed_pdf("TSS"))
    job = job_of(local, r)
    est = job["estimate"]
    assert job["status"] == "awaiting_confirmation" and job["options"]["fallback"] == "analyze"
    assert (est["model"], est["mode"], est["claude_pages"]) == (llm.MODEL, None, 0)       # just the analysis
    local.post(f"/api/jobs/{job['id']}/confirm", headers=H)
    jobs.drain()
    job = job_of(local, r)
    assert job["status"] == "done" and calls == [] and job["result"]["model"] == f"claude:{llm.MODEL}"
    assert "local model failed" in job["result"]["switched"] and job["options"]["confirmed"] == est["cost_usd"]
    assert len([c for c in local.calls if c["messages"][-1].get("images")]) == 2          # each scan read once
    text = doc_text(job["doc_id"])
    assert "Page 1 has" in text and text.count("The deposit is 1800 euros.") == 2

    def vision_down(model, messages, **kw):  # the vision model fails: Claude reads the scanned pages
        if messages[-1].get("images"):
            raise ollama.OllamaError(f"{model}: out of memory")
        return fake(model, messages, **kw)

    monkeypatch.setattr(ollama, "_chat", vision_down)
    r = upload(local, "other.pdf", mixed_pdf("TSST"))
    job = job_of(local, r)
    assert job["options"]["fallback"] == "read" and job["estimate"]["claude_pages"] == 2
    local.post(f"/api/jobs/{job['id']}/confirm", headers=H)
    jobs.drain()
    assert job_of(local, r)["status"] == "done" and [c["pages"] for c in calls] == [[2, 3]]


def test_a_confirmation_covers_the_amount_confirmed(client, monkeypatch):
    """A retry that would cost more than the user agreed to asks again."""
    from app import db, jobs, llm
    monkeypatch.setattr(jobs, "CONFIRM_COST_USD", 0)
    calls, broken = [], [True]
    monkeypatch.setattr(llm, "client_for", lambda key: claude_reader(calls, fail=lambda pages: broken[0]))
    r = upload(client, "scan.pdf", blank_pdf(3), model="claude:claude-haiku-5-5")
    job_id = r.json["job"]["id"]
    client.post(f"/api/jobs/{job_id}/confirm", headers=H)
    jobs.drain()
    assert job_of(client, r)["status"] == "failed"
    broken[0] = False
    client.post(f"/api/jobs/{job_id}/retry", headers=H)
    jobs.drain()
    assert job_of(client, r)["status"] == "done"                                         # same cost: no new question
    r = upload(client, "scan2.pdf", blank_pdf(2), model="claude:claude-haiku-5-5")
    job = job_of(client, r)
    db.update_job(job["id"], when=("awaiting_confirmation",), status="queued",
                  options={**job["options"], "confirmed": job["estimate"]["cost_usd"] / 2})
    jobs.drain()
    assert job_of(client, r)["status"] == "awaiting_confirmation"                        # half of it: asks again


def test_browser_keys_are_dropped_for_jobs_that_will_not_run(local, monkeypatch):
    """The browser's key is held only while its job can still run. A job whose insert
    rolled back leaves no key: SQLite gives its id to the next job, which must not spend
    another user's key."""
    from app import db, jobs
    r = post(local, "a.txt", b"Rent note A.")
    a = r.json["job"]["id"]
    assert jobs._keys == {a: "k"}
    local.post(f"/api/jobs/{a}/cancel")                                                   # cancelled while queued
    assert jobs._keys == {}
    store = db._store

    def disk_full(sha, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(db, "_store", disk_full)
    r = local.post("/api/upload", data={"file": (io.BytesIO(b"User A's note."), "a.txt")},
                   headers={"X-API-Key": "sk-user-a"})
    assert r.status_code == 500 and jobs._keys == {}
    monkeypatch.setattr(db, "_store", store)
    r = local.post("/api/upload", data={"file": (io.BytesIO(b"User B's note."), "b.txt")})   # no key
    assert r.status_code == 202 and r.json["job"]["id"] not in jobs._keys
    # Confirm on a job that isn't waiting leaves its key alone; a deleted document's re-read drops it.
    r = post(local, "c.txt", b"Rent note C.", model="claude:claude-haiku-5-5")
    c = r.json["job"]["id"]
    assert local.post(f"/api/jobs/{c}/confirm").status_code == 409 and jobs._keys[c] == "k"
    jobs.drain()
    doc_id = job_of(local, r)["doc_id"]
    r = local.post(f"/api/documents/{doc_id}/reread", json={"model": "claude:claude-haiku-5-5"}, headers=H)
    assert jobs._keys == {r.json["job"]["id"]: "k"}
    local.delete(f"/api/documents/{doc_id}")
    assert jobs._keys == {}


def test_json_bodies_that_are_not_objects_get_a_400(client):
    doc_id = add(client)["doc_id"]
    for path in (f"/api/documents/{doc_id}/reread", "/api/documents/delete", "/api/chat", "/api/instructions"):
        r = client.post(path, data="[1]", content_type="application/json", headers=H)
        assert r.status_code == 400 and r.json["error"] == "Expected a JSON object", path
    r = client.post(f"/api/documents/{doc_id}/reread", json={"model": 5}, headers=H)
    assert r.status_code == 400 and r.json["error"] == "Unknown model choice: 5"
    assert client.post(f"/api/documents/{doc_id}/reread", json={"claude_pages": ["all"]}, headers=H).status_code == 400
    assert client.post("/api/chat", json={"question": 5}, headers=H).status_code == 400
    assert [j["kind"] for j in client.get("/api/jobs").json] == ["upload"]               # nothing queued


def test_a_mistyped_claude_pdf_pages_setting_falls_back_to_scanned(local, monkeypatch):
    from app import jobs
    assert jobs._pages_mode(None) == jobs._pages_mode("") == "scanned" and jobs._pages_mode(" All ") == "all"
    monkeypatch.setattr(jobs, "CLAUDE_PDF_PAGES", jobs._pages_mode("every"))
    assert jobs.CLAUDE_PDF_PAGES == "scanned"
    r = post(local, model="ollama:qwen3.5:9b")
    assert r.status_code == 202 and r.json["job"]["options"]["claude_pages"] == "scanned"


def test_retry_counts_the_pages_of_the_reader_it_starts_with(local, monkeypatch):
    """Auto read 3 pages locally, then Claude read 20 before a batch failed. A retry
    starts with the local model again, so it keeps 3 pages (Claude's 20 only if it has
    to switch again). The job list counts them without loading the page texts."""
    from app import llm, ollama
    fake, seen = ollama._chat, []

    def vision_stops(model, messages, **kw):
        if messages[-1].get("images"):
            seen.append(model)
            if len(seen) > 3:
                raise ollama.OllamaUnavailable("Could not reach Ollama")
        return fake(model, messages, **kw)

    monkeypatch.setattr(ollama, "_chat", vision_stops)
    monkeypatch.setattr(llm, "PDF_CONCURRENCY", 1)
    calls = []
    monkeypatch.setattr(llm, "client_for", lambda key: claude_reader(calls, fail=lambda pages: pages[0] >= 21))
    r = upload(local, "scan.pdf", blank_pdf(25))
    job = job_of(local, r)
    assert job["status"] == "failed" and job["pages_kept"] == 3
    (listed,) = local.get("/api/jobs").json
    assert listed["pages_kept"] == 3 and "partial" not in listed and "kept" not in listed


def test_files_that_can_never_be_read_are_refused_before_they_are_queued(client, tmp_path):
    import zipfile
    docx = io.BytesIO()
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/other.xml", "<x/>")
    for name, body, error in [
        ("empty.txt", b"", "No text found in file"),
        ("blank.txt", b"  \n\f ", "No text found in file"),
        ("picture.png", b"\x89PNG\r\n\x1a\n\x00\xff", "Unsupported file: expected PDF, DOCX or UTF-8 text"),
        ("broken.pdf", b"%PDF-1.4 not really", "Could not open this PDF: "),
        ("none.pdf", blank_pdf(0), "Could not open this PDF: "),                         # PDFium won't load 0 pages
        ("broken.docx", b"PK not a zip", "Could not open this DOCX file"),
        ("odd.docx", docx.getvalue(), "Could not open this DOCX file"),
    ]:
        r = post(client, name, body)
        assert r.status_code == 400 and r.json["error"].startswith(error), (name, r.json)
    assert client.get("/api/jobs").json == [] and not (tmp_path / "files").exists()


def test_a_form_feed_in_a_text_file_is_not_a_page_break(client):
    """Only PDFs have pages: a form feed in a text file (a printout, ^L in code) must not
    give it page numbers later."""
    job = add(client, "notes.txt", b"Bench grinder notes.\n\nGap under 3 mm.\fSecond part after a form feed.")
    text = doc_text(job["doc_id"])
    assert "\f" not in text and "Gap under 3 mm.\nSecond part" in text


def test_a_model_without_a_list_price_still_needs_a_confirmation(client, monkeypatch):
    """A Claude model missing from llm.PRICES would cost $0 by the table and never ask:
    it is estimated at the dearest listed price instead, and marked as such."""
    from app import jobs, llm
    monkeypatch.setattr(jobs, "CONFIRM_COST_USD", 1.00)
    monkeypatch.setattr(llm, "MODELS", [*llm.MODELS, ("claude-sonnet-5-6", "claude-sonnet-5-6")])
    pdf = blank_pdf(60)
    est = jobs.estimate("scan.pdf", pdf, "claude-sonnet-5-6", True, "scanned")
    dearest = jobs.estimate("scan.pdf", pdf, "claude-fable-5-2", True, "scanned")
    assert est["priced"] is False and dearest["priced"] is True and est["cost_usd"] == dearest["cost_usd"] > 1
    r = upload(client, "scan.pdf", pdf, model="claude:claude-sonnet-5-6")
    assert job_of(client, r)["status"] == "awaiting_confirmation"


def test_posts_from_other_sites_are_refused(client, monkeypatch):
    """A page elsewhere can make a browser send simple POSTs here (a form, or a text
    body): to start a job waiting for its cost, or to delete documents."""
    from app import jobs
    monkeypatch.setattr(jobs, "CONFIRM_COST_USD", 0)
    r = upload(client, model="claude:claude-haiku-5-5")
    job_id = r.json["job"]["id"]
    evil = {"Origin": "http://evil.example", **H}
    assert client.post(f"/api/jobs/{job_id}/confirm", headers=evil).status_code == 403
    assert job_of(client, r)["status"] == "awaiting_confirmation"
    assert client.post("/api/documents/delete", data='{"ids": [1]}', content_type="text/plain",
                       headers={"Origin": "null"}).status_code == 403
    assert post(client, "x.txt", b"Rent.").status_code == 202                             # no Origin: curl, scripts
    assert client.post("/api/documents/delete", json={"ids": []}, headers={"Origin": "http://localhost"}).status_code == 200
    assert client.get("/api/jobs", headers={"Origin": "http://evil.example"}).status_code == 200  # reading is fine
    assert client.post(f"/api/jobs/{job_id}/confirm", headers={"Origin": "http://localhost", **H}).status_code == 200


def test_gunicorn_starts_the_job_worker_at_boot(local, monkeypatch):
    """After a restart, jobs it cut off carry on without waiting for a page load."""
    import time
    from types import SimpleNamespace
    from app import db, gunicorn_conf, jobs
    job_id = db.add_job("upload", "lease.txt", db.file_hash(LEASE), "auto", {}, data=LEASE)
    db.claim_next_job()                                                                    # running when it stopped
    monkeypatch.setattr(jobs, "BACKGROUND", True)
    monkeypatch.setattr(jobs, "_thread", None)
    try:
        gunicorn_conf.post_worker_init(SimpleNamespace(log=None))                          # no request at all
        for _ in range(200):
            if db.job_status(job_id) == "done":
                break
            time.sleep(0.05)
        assert db.job_status(job_id) == "done"
    finally:
        jobs.stop()


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
# The schema of commit cd8c125 (kept files, hashes, usage log), before the jobs table.
SCHEMA_CD8C125_EXTRA = """
ALTER TABLE documents ADD COLUMN sha256 TEXT;
ALTER TABLE documents ADD COLUMN file_path TEXT;
ALTER TABLE documents ADD COLUMN file_size INTEGER;
ALTER TABLE documents ADD COLUMN mime TEXT;
CREATE TABLE usage (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL DEFAULT (datetime('now')), kind TEXT NOT NULL,
    backend TEXT NOT NULL, model TEXT NOT NULL, input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0, cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0, doc_id INTEGER);
"""
# The schema of commit dc8b86a (jobs, usage.job_id), before unit checks were stored.
SCHEMA_DC8B86A_EXTRA = """
ALTER TABLE usage ADD COLUMN job_id INTEGER;
CREATE TABLE jobs (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
    filename TEXT NOT NULL, sha256 TEXT NOT NULL, file_path TEXT NOT NULL, file_size INTEGER, mime TEXT,
    model TEXT NOT NULL, options TEXT NOT NULL DEFAULT '{}', progress INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0, stage TEXT NOT NULL DEFAULT '', error TEXT, estimate TEXT, result TEXT,
    partial TEXT, doc_id INTEGER, created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')));
"""
# The schema of commit 712f372 (unit checks stored), before chunk pages and vectors.
SCHEMA_712F372_EXTRA = "ALTER TABLE documents ADD COLUMN unit_findings TEXT;"
OLD_SCHEMAS = {"894b984": SCHEMA_894B984, "cd8c125": SCHEMA_894B984 + SCHEMA_CD8C125_EXTRA,
               "dc8b86a": SCHEMA_894B984 + SCHEMA_CD8C125_EXTRA + SCHEMA_DC8B86A_EXTRA,
               "712f372": SCHEMA_894B984 + SCHEMA_CD8C125_EXTRA + SCHEMA_DC8B86A_EXTRA + SCHEMA_712F372_EXTRA}


@pytest.mark.parametrize("schema", OLD_SCHEMAS.values(), ids=OLD_SCHEMAS.keys())
def test_library_from_before_kept_files_still_works(client, tmp_path, monkeypatch, schema):
    import sqlite3
    from app import db
    path = tmp_path / "live.db"
    con = sqlite3.connect(path)
    con.executescript(schema)
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
    assert client.post("/api/documents/1/reread", json={"model": "auto"}, headers=H).status_code == 409  # nothing to re-read
    assert client.get("/api/jobs").json == []                                            # the jobs table is new
    job = add(client, "manual.pdf.txt", b"Change the hydraulic oil every 400 hours.")   # never a duplicate
    assert job["status"] == "done"
    assert client.get("/api/usage").json["total"]["calls"] == 1                         # the chat above
    assert client.get("/api/documents/1/units").json["findings"] == []                   # checked when asked
    assert client.delete("/api/documents/1").status_code == 204
    assert [d["filename"] for d in client.get("/api/documents").json] == ["manual.pdf.txt"]


@pytest.mark.parametrize("schema", OLD_SCHEMAS.values(), ids=OLD_SCHEMAS.keys())
def test_documents_from_before_unit_checks_get_one(client, tmp_path, monkeypatch, schema):
    """The live library's documents have no unit check: the worker runs it once at start,
    from their stored chunks (no model), and the list of one is worked out when asked."""
    import sqlite3
    import time
    from app import db, jobs
    path = tmp_path / "live.db"
    con = sqlite3.connect(path)
    con.executescript(schema)
    kioti = "| Capacity | Transmission fluid | 38.0 ℓ |\n|---|---|---|\n| | | 40.1 U.S.gal. |"
    con.execute("INSERT INTO documents (id, filename, char_count, title) VALUES (2, 'Chapter 6.pdf', ?, 'KIOTI')",
                (len(kioti),))
    con.execute("INSERT INTO chunks (id, doc_id, idx, text) VALUES (2, 2, 0, ?)", (kioti,))
    con.execute("INSERT INTO chunks_fts (rowid, text) VALUES (2, ?)", (kioti,))
    con.commit()
    con.close()
    monkeypatch.setattr(db, "DB_PATH", str(path))
    assert [d["unit_fixes"] for d in client.get("/api/documents").json] == [None, None]   # not checked yet
    r = client.get("/api/documents/2/units").json                                        # worked out when asked
    assert r["has_file"] is False and r["pdf"] is False and [(f["page"], f["kind"], f["printed"]) for f in r["findings"]] == [
        (None, "quarts-printed-as-gallons", "38.0 ℓ 40.1 U.S.gal.")]                    # no page breaks: no page
    assert [d["unit_fixes"] for d in client.get("/api/documents").json] == [1, None]
    monkeypatch.setattr(jobs, "BACKGROUND", True)                                         # the worker checks the rest
    monkeypatch.setattr(jobs, "_thread", None)
    try:
        jobs.start()
        for _ in range(100):
            if client.get("/api/documents").json[1]["unit_fixes"] is not None:
                break
            time.sleep(0.05)
    finally:
        jobs.stop()
        monkeypatch.setattr(jobs, "BACKGROUND", False)                                    # requests below mustn't restart it
    assert [d["unit_fixes"] for d in client.get("/api/documents").json] == [1, 0]
    assert jobs.check_units() == 0
    client.post("/api/chat", json={"question": "transmission fluid capacity"}, headers=H)
    assert "10.0 U.S.gal. (40.1 U.S. qt)" in client.seen["msg"]                          # answers corrected as listed
    # A check made by another version of units.py is made again, so the list keeps matching the fix.
    with db.connect() as con:
        con.execute("""UPDATE documents SET unit_findings='{"version": "older", "findings": []}'""")
    assert client.get("/api/documents/2/units").json["findings"][0]["printed"] == "38.0 ℓ 40.1 U.S.gal."
    assert jobs.check_units() == 1 and jobs.check_units() == 0                             # the other one, at start
    assert [d["unit_fixes"] for d in client.get("/api/documents").json] == [1, 0]


def test_upload_keeps_the_original_file(client, tmp_path):
    job = add(client)
    assert job["status"] == "done"
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


def fake_pages(text):
    """llm.transcribe_pdf, faked: every page in the batches reads `text`."""
    return lambda c, batches, model=None, **kw: {p: text for pages, _, _ in batches for p in pages}


def test_kept_pdf_opens_in_the_browser_and_html_downloads(client, monkeypatch):
    from app import llm
    monkeypatch.setattr(llm, "transcribe_pdf", fake_pages("The deposit is 1800 euros."))
    pdf = blank_pdf(2)
    doc_id = add(client, "Handbuch für Öl.pdf", pdf)["doc_id"]
    f = client.get(f"/api/documents/{doc_id}/file")
    assert f.data == pdf and f.mimetype == "application/pdf"
    disposition = f.headers["Content-Disposition"]
    assert disposition.startswith("inline;") and "filename*=UTF-8''Handbuch%20f%C3%BCr%20%C3%96l.pdf" in disposition
    doc_id = add(client, "page.html", b"<script>alert(1)</script> The rent is 900 euros.")["doc_id"]
    f = client.get(f"/api/documents/{doc_id}/file")
    assert f.mimetype == "text/html" and f.headers["Content-Disposition"] == "attachment; filename=page.html"


def test_documents_without_a_kept_file(client):
    from app import db
    old = db.add_document("old.txt", ["An old rent note."], {"title": "Old"}, "full")
    assert db.stored_file(old) is None
    assert client.get(f"/api/documents/{old}/file").status_code == 404
    assert client.get("/api/documents/9999/file").status_code == 404
    assert client.get("/api/documents").json[0]["has_file"] is False
    r = client.post(f"/api/documents/{old}/reread", json={"model": "auto"})
    assert r.status_code == 409 and "can't be re-read" in r.json["error"]


def test_identical_files_are_stored_once_and_removed_with_the_last_copy(client, tmp_path):
    from app import db
    a = add(client)["doc_id"]
    b = add(client, "copy.txt", on_duplicate="keep")["doc_id"]
    other = add(client, "other.txt", b"A different rent note.")["doc_id"]
    files = tmp_path / "files"
    assert len(list(files.iterdir())) == 2                                               # a and b share one file
    assert db.stored_file(b)["path"] == db.stored_file(a)["path"]
    assert client.delete(f"/api/documents/{a}").status_code == 204
    assert len(list(files.iterdir())) == 2 and client.get(f"/api/documents/{b}/file").data == LEASE
    assert client.post("/api/documents/delete", json={"ids": [b, other]}).json["deleted"] == 2
    assert list(files.iterdir()) == []                                                   # last users gone


def test_a_queued_upload_keeps_its_file_when_the_last_document_with_it_goes(client, tmp_path):
    from app import jobs
    a = add(client)["doc_id"]
    r = post(client, "again.txt", on_duplicate="keep")                                   # queued, same bytes
    assert client.delete(f"/api/documents/{a}").status_code == 204
    assert len(list((tmp_path / "files").iterdir())) == 1                                # the job still needs it
    jobs.drain()
    assert job_of(client, r)["status"] == "done"


def test_duplicate_upload_waits_for_a_decision(client):
    first = add(client)
    r = upload(client, "again.txt")                                                      # same bytes, other name
    assert r.status_code == 409 and "already in the library" in r.json["error"]
    dup = r.json["duplicate"]
    assert dup["id"] == first["doc_id"] and dup["filename"] == "lease.txt" and dup["model"] == "Claude Opus 5.5"
    assert dup["created_at"] and dup["copies"] == 1
    assert len(client.get("/api/documents").json) == 1                                   # nothing added
    assert upload(client, "again.txt", on_duplicate="maybe").status_code == 400
    kept = add(client, "again.txt", on_duplicate="keep")
    assert kept["result"]["replaced"] == [] and len(client.get("/api/documents").json) == 2
    third = add(client, "third.txt", on_duplicate="replace")                             # replaces both copies
    assert third["status"] == "done" and sorted(third["result"]["replaced"]) == sorted([first["doc_id"], kept["doc_id"]])
    docs = client.get("/api/documents").json
    assert [d["filename"] for d in docs] == ["third.txt"]
    assert client.get(f"/api/documents/{docs[0]['id']}/file").data == LEASE              # file survived the replace
    assert {h["filename"] for h in client.get("/api/search?q=rent").json} == {"third.txt"}
    assert upload(client, "new.txt", b"Something else entirely.").status_code == 202     # other bytes: no duplicate


def test_replace_keeps_the_old_copy_when_the_new_read_fails(client, monkeypatch):
    from app import jobs, llm
    monkeypatch.setattr(llm, "transcribe_pdf", fake_pages("The deposit is 1800 euros."))
    pdf = blank_pdf(1)
    first = add(client, "scan.pdf", pdf)["doc_id"]
    monkeypatch.setattr(jobs, "PDF_EXTRACTOR", "local")                                  # the re-read finds no text
    assert add(client, "scan.pdf", pdf, on_duplicate="replace")["status"] == "failed"
    assert [d["id"] for d in client.get("/api/documents").json] == [first]
    assert client.get(f"/api/documents/{first}/file").data == pdf


def test_cost_uses_the_price_table():
    from app import llm

    def u(i=0, o=0, cr=0, cw=0):
        return {"input": i, "output": o, "cache_read": cr, "cache_write": cw}

    assert llm.cost("claude-opus-5-5", u(1_000_000, 1_000_000)) == pytest.approx(24.0)
    assert llm.cost("claude-sonnet-5-5", u(cr=1_000_000, cw=1_000_000)) == pytest.approx(0.10 + 2.5)  # 1.25x writes
    # Cache reads have their own list price per model: 0.05x on Opus and Sonnet 5.5, 0.025x on Fable 5.2.
    assert llm.cost("claude-opus-5-5", u(cr=1_000_000)) == pytest.approx(0.20)
    assert llm.cost("claude-fable-5-2", u(cr=1_000_000)) == pytest.approx(0.20)
    assert llm.cost("claude-haiku-5-5", u(50_000, 1_000)) == pytest.approx((50_000 * 0.10 + 1_000 * 0.50) / 1e6)
    assert llm.cost("claude-haiku-5-5", u(150_000, 1_000)) == pytest.approx((150_000 * 0.50 + 1_000 * 2.50) / 1e6)
    assert llm.cost("claude-haiku-5-5", u(1_000, cr=60_000)) == pytest.approx((1_000 * 0.10 + 60_000 * 0.01) / 1e6)
    assert llm.cost("claude-haiku-5-5", u(1_000, cr=200_000)) == pytest.approx((1_000 * 0.50 + 200_000 * 0.05) / 1e6)
    # A realistic later chat turn on Opus 5.5: mostly read from the cache.
    assert llm.cost("claude-opus-5-5", u(20, 300, 20_000, 2_500)) == pytest.approx(
        (20 * 4 + 300 * 20 + 20_000 * 0.20 + 2_500 * 5) / 1e6)
    assert llm.cost("claude-fable-5-2", u(1_000, 1_000)) == pytest.approx(0.048)
    assert llm.cost("claude-unknown-9", u(1_000, 1_000)) == 0


def fake_claude(reply):
    """An Anthropic client whose every call, plain or streamed, answers `reply`."""
    from types import SimpleNamespace as NS

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return reply

    return NS(beta=NS(messages=NS(create=lambda **kw: reply, stream=lambda **kw: Stream())))


def test_claude_calls_report_their_usage():
    from types import SimpleNamespace as NS
    from app import llm
    usage = NS(input_tokens=1200, output_tokens=300, cache_read_input_tokens=None, cache_creation_input_tokens=50)
    reply = NS(stop_reason="end_turn", usage=usage, content=[
        NS(type="text", text='{"title": "t", "doc_type": "d", "summary": "s", "keywords": []}')])
    fake = fake_claude(reply)
    seen = []
    llm.analyze(fake, "a.txt", "text", "claude-haiku-5-5", on_usage=lambda *call: seen.append(call))
    pages = llm.transcribe_pdf(fake, [(list(range(1, 21)), 40, b"x"), (list(range(21, 41)), 40, b"y")],
                               "claude-haiku-5-5", on_usage=lambda *call: seen.append(call))
    assert sorted(pages) == list(range(1, 41)) and pages[1].startswith('{"title"') and pages[2] == ""  # no marks
    assert llm.chat(fake, [], "hi", "claude-haiku-5-5", on_usage=lambda *call: seen.append(call))[1]["cache_write"] == 50
    assert [c[0] for c in seen] == ["analyze", "transcribe", "transcribe", "chat"]
    assert seen[0][1:] == ("claude", "claude-haiku-5-5", {"input": 1200, "output": 300, "cache_read": 0, "cache_write": 50})


def test_refused_claude_chat_is_logged_and_reported(tmp_path, monkeypatch):
    """A refusal is billed for its input, so it gets a usage row, and the page gets a
    readable error instead of an HTML 500."""
    from types import SimpleNamespace as NS
    from app import db, jobs, llm, main
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setattr(jobs, "BACKGROUND", False)
    usage = NS(input_tokens=4000, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    monkeypatch.setattr(llm, "client_for", lambda key: fake_claude(NS(stop_reason="refusal", usage=usage, content=[])))
    r = main.app.test_client().post("/api/chat", json={"question": "hi", "model": "claude:claude-haiku-5-5"}, headers=H)
    assert r.status_code == 502 and "declined" in r.json["error"]
    (row,) = _usage_rows()
    assert (row["kind"], row["model"], row["input_tokens"]) == ("chat", "claude-haiku-5-5", 4000)
    assert row["cost_usd"] == pytest.approx(4000 * 0.10 / 1e6)


def _usage_rows():
    from app import db
    with db.connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM usage ORDER BY id")]


def test_upload_and_chat_log_each_model_call(local, monkeypatch):
    from app import extract, ollama
    monkeypatch.setattr(extract, "pdf_pages", lambda data, **kw: [("", b"png"), ("Typed page.", None), ("", b"png")])
    doc_id = add(local, "scan.pdf", blank_pdf(3))["doc_id"]
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

    def empty_transcribe(c, batches, model=None, on_usage=None, on_batch=None):
        on_usage("transcribe", "claude", model, {"input": 3000, "output": 10, "cache_read": 0, "cache_write": 0})
        return {p: "" for pages, _, _ in batches for p in pages}

    monkeypatch.setattr(llm, "transcribe_pdf", empty_transcribe)
    job = add(client, "blank.pdf", blank_pdf(1))
    assert job["status"] == "failed" and job["error"] == "No text found in file"
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


def test_a_failed_usage_write_does_not_fail_the_request(client, monkeypatch):
    """Usage logging is best effort: a busy database must not throw away a paid answer,
    report a stored upload as failed, or hide why a read failed."""
    import sqlite3
    from app import db, jobs

    def locked(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "add_usage", locked)
    r = client.post("/api/chat", json={"question": "hi"}, headers=H)
    assert r.status_code == 200 and r.json["answer"] == "42"
    job = add(client)
    assert job["status"] == "done" and [d["id"] for d in client.get("/api/documents").json] == [job["doc_id"]]
    monkeypatch.setattr(jobs, "PDF_EXTRACTOR", "local")                                  # a scan: no text layer
    job = add(client, "blank.pdf", blank_pdf(1))
    assert job["status"] == "failed" and job["error"] == "No text found in file"


def test_usage_rows_lose_their_link_when_the_document_goes(local):
    """Document ids are reused after a delete, so a deleted document's usage rows keep
    their cost but must not point at whichever document gets its id next."""
    first = add(local)["doc_id"]
    second = add(local, on_duplicate="replace")
    assert second["result"]["replaced"] == [first]
    assert [r["doc_id"] for r in _usage_rows()] == [None, second["doc_id"]]             # one analyze call each
    assert local.delete(f"/api/documents/{second['doc_id']}").status_code == 204
    assert [r["doc_id"] for r in _usage_rows()] == [None, None]                          # still logged


def test_first_requests_at_once_bring_the_schema_up_to_date(tmp_path, monkeypatch):
    """The page loads documents and instructions in parallel. On a new database and on
    one from before kept files, every one of those first requests must succeed."""
    import sqlite3
    import threading
    from app import db
    errors = []
    for n in range(40):
        path = tmp_path / f"{n}.db"
        if n % 2:
            con = sqlite3.connect(path)
            con.executescript(SCHEMA_894B984)
            con.close()
        monkeypatch.setattr(db, "DB_PATH", str(path))
        gate = threading.Barrier(4)

        def hit(fn):
            gate.wait()
            try:
                fn()
            except Exception as e:
                errors.append(repr(e))

        threads = [threading.Thread(target=hit, args=(fn,))
                   for fn in (db.list_documents, db.list_instructions, db.list_jobs, db.list_documents)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with db.connect() as con:
            assert {"sha256", "file_path", "file_size", "mime"} <= {r[1] for r in con.execute("PRAGMA table_info(documents)")}
    assert errors == []


def test_a_copy_still_queued_or_being_read_counts_as_a_duplicate(client):
    """Picking a file again while the first copy waits in the queue (or is being read)
    must not add it twice."""
    from app import db, jobs
    r = post(client, "pump.txt")
    assert r.status_code == 202
    again = post(client, "pump (1).txt")
    dup = again.json["duplicate"]
    assert again.status_code == 409 and dup["in_progress"] is True and dup["filename"] == "pump.txt"
    assert dup["started_at"] and dup["job_id"] == r.json["job"]["id"] and dup["copies"] == 0
    db.claim_next_job()                                                                  # now being read
    assert post(client, "pump (1).txt").json["duplicate"]["in_progress"] is True
    assert post(client, "pump (1).txt", on_duplicate="replace").status_code == 409       # nothing to replace yet
    db.requeue_running()
    assert post(client, "pump (1).txt", on_duplicate="keep").status_code == 202
    jobs.drain()
    assert len(client.get("/api/documents").json) == 2
    r = post(client, "pump (2).txt")                                                     # both done: an ordinary duplicate
    assert r.status_code == 409 and "in_progress" not in r.json["duplicate"] and r.json["duplicate"]["copies"] == 2


def test_a_failed_insert_leaves_no_file_behind(client, tmp_path, monkeypatch):
    import sqlite3
    from app import db, llm
    files = tmp_path / "files"
    monkeypatch.setattr(llm, "analyze", lambda c, name, text, model=None, **kw: (
        {"title": "T", "doc_type": "d", "summary": "s", "keywords": [1, 2]}, "full"))     # not strings: the insert fails
    job = add(client)
    assert job["status"] == "failed" and "expected str" in job["error"]
    assert client.get("/api/documents").json == [] and len(list(files.iterdir())) == 1  # kept for a retry
    assert client.post(f"/api/jobs/{job['id']}/cancel").json["job"]["status"] == "cancelled"  # dismissed
    assert not any(files.glob("*"))
    real = db.connect

    class CommitFails:  # the rows are written, then the commit fails (say the database is locked)
        def __init__(self):
            self.con = real()

        def __getattr__(self, name):
            return getattr(self.con, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.con.rollback()
            if exc[0] is None:
                raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "connect", CommitFails)
    with pytest.raises(sqlite3.OperationalError):
        db.add_document("a.txt", ["Rent."], {"title": "A"}, "full", data=b"Rent.")
    with pytest.raises(sqlite3.OperationalError):
        db.add_job("upload", "a.txt", db.file_hash(b"Rent."), "auto", {}, data=b"Rent.")
    monkeypatch.setattr(db, "connect", real)
    assert db.list_documents() == [] and not any(files.glob("*"))


def test_control_characters_are_removed_from_file_names(client):
    """A crafted client can send CR/LF in a file name; it must not break the Open link."""
    from app import jobs
    body = (b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename*=UTF-8''evil%0D%0AX-Injected%3A%201.txt\r\n"
            b"Content-Type: text/plain\r\n\r\nThe rent is 900 euros.\r\n--b--\r\n")
    r = client.post("/api/upload", data=body, content_type="multipart/form-data; boundary=b", headers=H)
    assert r.status_code == 202
    jobs.drain()
    doc = client.get("/api/documents").json[0]
    assert doc["filename"] == "evil X-Injected: 1.txt"
    f = client.get(f"/api/documents/{doc['id']}/file")
    assert f.status_code == 200 and "X-Injected" not in f.headers


def test_ids_too_large_for_sqlite_are_not_found(client):
    huge = 2 ** 70
    assert client.get(f"/api/documents/{huge}/file").status_code == 404
    assert client.delete(f"/api/documents/{huge}").status_code == 404
    assert client.patch(f"/api/instructions/{huge}", json={"enabled": False}).status_code == 404
    assert client.delete(f"/api/instructions/{huge}").status_code == 404
    assert client.get(f"/api/jobs/{huge}").status_code == 404
    assert client.post(f"/api/jobs/{huge}/cancel").status_code == 404
    assert client.post(f"/api/documents/{huge}/reread", json={"model": "auto"}).status_code == 404
    r = client.post("/api/documents/delete", json={"ids": [huge, 2 ** 63 - 1]})
    assert r.status_code == 200 and r.json["deleted"] == 0


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
    """One analysis call carries up to 500K characters to Claude, 40K to a local model.
    A local model reads a 100K-character text in three sections instead of its start."""
    from types import SimpleNamespace
    from app import llm, ollama, sections
    assert llm.ANALYZE_CHARS == 500_000 and ollama.ANALYZE_CHARS == 40_000
    assert round(ollama.analyze_tokens()) == 12_576                                     # (16384 - 2000) / 1.1 - 500
    text = "word " * 20_000  # 100,000 characters, one paragraph
    sent = {"claude": [], "local": []}

    def create(**kw):
        sent["claude"].append(kw["messages"][0]["content"])
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(
            type="text", text='{"title": "t", "doc_type": "d", "summary": "s", "keywords": []}')])

    fake = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert llm.analyze(fake, "big.txt", text, "claude-haiku-5-5")[1] == "full"
    assert len(sent["claude"]) == 1 and text in sent["claude"][0]                       # one call, as before

    def local_chat(model, messages, **kw):
        sent["local"].append((messages, kw))
        notes = "topics" in kw["format"]["properties"]
        return {"message": {"content": '{"topics": ["words"], "specifications": [], "procedures": [], "warnings": []}'
                            if notes else '{"title": "t", "doc_type": "d", "summary": "s", "keywords": []}'}}

    monkeypatch.setattr(ollama, "_chat", local_chat)
    assert ollama.analyze(None, "big.txt", text) == ({"title": "t", "doc_type": "d", "summary": "s", "keywords": []}, "full")
    *parts, final = sent["local"]
    assert [kw["format"] for _, kw in parts] == [sections.NOTES_SCHEMA] * 3 and final[1]["format"] == sections.META_SCHEMA
    bodies = [re.search(r"<part>\n(.*)\n</part>", m[0]["content"], re.S).group(1) for m, _ in parts]
    assert all(len(b) <= 40_000 for b in bodies) and " ".join(bodies).split() == text.split()   # every word, once
    assert all(ollama.fits(m, ollama.REPLY_TOKENS) and kw["num_predict"] == ollama.REPLY_TOKENS for m, kw in sent["local"])
    # Ollama is told to refuse a prompt it would have to cut, and to stop a reply at the end
    # of the context rather than drop the start of the text (test_text_denser_than_reckoned_...).
    assert all(kw["truncate"] is False and kw["shift"] is False for _, kw in sent["local"])
    assert "## Part 3 of 3\nKey topics:\n- words" in final[0][0]["content"]


def long_pdf(local, monkeypatch, pages=6, chars=1000, notes=None, ctx=None, reps=30):
    """A `pages`-page PDF whose text layer says which page each is, and a local model that
    analyzes `chars` characters per call: each page is a section of its own. `notes(n, kw)`
    answers a section's notes call (n = its page; None: the fixture's answer). Records
    (kind, stage, progress, total, messages, options) at each analysis call, the first
    four of the job running then."""
    import json
    from app import db, extract, ollama
    monkeypatch.setattr(ollama, "ANALYZE_CHARS", chars)
    if ctx:
        monkeypatch.setattr(ollama, "NUM_CTX", ctx)
    texts = [f"Page {n} is about pump number {n}. " + "The pump needs oil. " * reps for n in range(1, pages + 1)]
    monkeypatch.setattr(extract, "pdf_pages", lambda data, **kw: [(t, None) for t in texts])
    if not hasattr(local, "base_chat"):  # the fixture's fake, not an earlier long_pdf's
        local.base_chat = ollama._chat
    fake, seen = local.base_chat, []

    def chat(model, messages, **kw):
        if "format" in kw:
            with db.connect() as con:
                row = con.execute("SELECT stage, progress, total FROM jobs WHERE status='running'").fetchone()
            body = messages[0]["content"]
            notes_call = "topics" in kw["format"]["properties"]
            kind = ("merge" if "Merge them" in body else "notes") if notes_call else "meta"
            seen.append((kind, *(tuple(row) if row else (None, None, None)), messages, kw))
            n = re.search(r"This is part \d+ of \d+ \(page (\d+)\)", body)
            if notes and n:
                got = notes(int(n.group(1)), kw)
                if got is not None:
                    return {"message": {"content": got if isinstance(got, str) else json.dumps(got)},
                            "prompt_eval_count": 400, "eval_count": 30}
        return fake(model, messages, **kw)

    monkeypatch.setattr(ollama, "_chat", chat)
    return blank_pdf(pages), seen


def test_a_long_document_is_analyzed_section_by_section(local, monkeypatch):
    """Text over one call's limit: the same model takes notes on each section (a page here),
    then combines them into the metadata. The job shows each step, every call fits the
    local context, and each call is in the usage log."""
    from app import ollama
    pdf, seen = long_pdf(local, monkeypatch, notes=lambda n, kw: {
        "topics": [f"pump {n}"], "specifications": [f"oil {n} L"], "procedures": [], "warnings": []})
    job = add(local, "manual.pdf", pdf, model="ollama:qwen3.5:9b")
    assert job["status"] == "done" and job["result"]["analysis"] == "full" and job["result"]["title"] == "Scan"
    assert [s[0] for s in seen] == ["notes"] * 6 + ["meta"]
    assert [s[1:4] for s in seen[:2]] == [("Analyzing section 1 of 6 with qwen3.5:9b", 0, 6),
                                          ("Analyzing section 2 of 6 with qwen3.5:9b", 1, 6)]
    assert seen[-1][1:4] == ("Combining the notes of 6 sections with qwen3.5:9b", 6, 6)
    third = seen[2][4][0]["content"]
    assert "This is part 3 of 6 (page 3)" in third and "Page 3 is about" in third and "Page 4" not in third
    final = seen[-1][4][0]["content"]
    assert "read in 6 parts" in final and "Only" not in final
    assert "## Part 1 of 6 (page 1)\nKey topics:\n- pump 1\nSpecifications and values:\n- oil 1 L" in final
    assert all(ollama.fits(s[4], ollama.REPLY_TOKENS) for s in seen)
    assert [r["kind"] for r in _usage_rows() if r["kind"].startswith("analyze")] == ["analyze-section"] * 6 + ["analyze"]
    assert all(r["doc_id"] == job["doc_id"] for r in _usage_rows())
    assert local.get("/api/documents").json[0]["analysis"] == "full"


def test_sections_past_the_cap_are_sampled_evenly(local, monkeypatch):
    from app import sections
    monkeypatch.setattr(sections, "MAX_SECTIONS", 3)
    pdf, seen = long_pdf(local, monkeypatch)
    job = add(local, "manual.pdf", pdf, model="ollama:qwen3.5:9b")
    assert job["result"]["analysis"] == "partial: 3 of 6 sections, 50% of the text"
    read = [re.search(r"\(page (\d+)\)", s[4][0]["content"]).group(1) for s in seen if s[0] == "notes"]
    assert read == ["1", "3", "6"]                                                       # first, middle, last
    assert seen[0][1] == "Analyzing section 1 of 3 (sampled from 6) with qwen3.5:9b"
    assert "Only 3 of its 6 parts were read" in seen[-1][4][0]["content"]
    assert local.get("/api/documents").json[0]["analysis"] == "partial: 3 of 6 sections, 50% of the text"
    assert sections.pick(61, 24)[:3] == [0, 3, 5] and sections.pick(61, 24)[-1] == 60 and len(set(sections.pick(61, 24))) == 24
    assert sections.pick(5, 24) == [0, 1, 2, 3, 4] and sections.pick(9, 1) == [0]


def test_notes_too_long_for_one_call_are_merged_first(local, monkeypatch):
    """A small context: the sections' notes don't fit the final call together, so
    neighbouring notes are merged first, as many to a call as fit."""
    from app import ollama
    pdf, seen = long_pdf(local, monkeypatch, chars=40_000, ctx=3000, reps=43, notes=lambda n, kw: LONG_NOTE)
    assert round(ollama.analyze_tokens()) == 409                                          # (3000 - 2000) / 1.1 - 500
    job = add(local, "manual.pdf", pdf, model="ollama:qwen3.5:9b")
    assert job["status"] == "done" and job["result"]["analysis"] == "full"
    kinds = [s[0] for s in seen]
    assert kinds.count("notes") == 6 and kinds.count("merge") == 2 and kinds[-1] == "meta"
    final = seen[-1][4][0]["content"]
    assert "## Parts 1-3 of 6 (pages 1-3)" in final and "## Parts 4-6 of 6 (pages 4-6)" in final
    assert all(ollama.fits(s[4], ollama.REPLY_TOKENS) for s in seen)                   # every call within the context
    assert [r["kind"] for r in _usage_rows()].count("analyze-merge") == 2
    # The final call has a stage of its own again after the merges.
    assert [s[1] for s in seen[-3:]] == ["Combining notes, group 1 of 2 with qwen3.5:9b",
                                         "Combining notes, group 2 of 2 with qwen3.5:9b",
                                         "Combining the notes of 6 sections with qwen3.5:9b"]


# Notes on a section that take 118 of the 409 tokens a call carries with ctx=3000: three
# fit one merge call, four don't.
LONG_NOTE = {"topics": ["x" * 600], "specifications": [], "procedures": [], "warnings": []}


def test_a_merge_whose_reply_is_unusable_leaves_its_notes_unmerged(local, monkeypatch, caplog):
    """A merge reply cut off at num_predict (or not JSON) used to fail the whole analysis
    after every section was read, and the same way on every retry. Its group's notes now
    stay as they were, and later rounds merge fewer at a time, so the final call runs."""
    from app import jobs, ollama
    pdf, seen = long_pdf(local, monkeypatch, chars=40_000, ctx=3000, reps=43, notes=lambda n, kw: LONG_NOTE)
    inner, broken = ollama._chat, {"cut off": 1}

    def chat(model, messages, **kw):
        if "format" in kw and "Merge them" in messages[0]["content"] and broken["cut off"]:
            broken["cut off"] -= 1
            return {"message": {"content": '{"topics": ["a", "b'}, "done_reason": "length",
                    "prompt_eval_count": 1, "eval_count": ollama.REPLY_TOKENS}
        if "format" in kw and "Merge them" in messages[0]["content"] and broken.get("not json"):
            return {"message": {"content": "not json"}, "prompt_eval_count": 1, "eval_count": 2}
        return inner(model, messages, **kw)

    monkeypatch.setattr(ollama, "_chat", chat)
    job = add(local, "manual.pdf", pdf, model="ollama:qwen3.5:9b")
    assert job["status"] == "done" and job["result"]["analysis"] == "full" and job["result"]["title"] == "Scan"
    final = seen[-1][4][0]["content"]
    assert seen[-1][0] == "meta" and [s[0] for s in seen].count("merge") == 1           # the second group's
    for n in (1, 2, 3):                                                                  # the first group's, as they were
        assert f"## Part {n} of 6 (page {n})\nKey topics:\n- {'x' * 600}" in final
    assert "## Parts 4-6 of 6 (pages 4-6)" in final
    assert all(ollama.fits(s[4], ollama.REPLY_TOKENS) for s in seen)
    assert "left unmerged" in caplog.text and "cut off at 2,000 tokens" in caplog.text
    # A re-read whose merges never work still ends: each round merges half as many, then the
    # notes are shortened to fit.
    broken["not json"] = True
    local.post(f"/api/documents/{job['doc_id']}/reread", json={"model": "ollama:qwen3.5:9b"})
    jobs.drain()
    assert local.get("/api/jobs").json[0]["status"] == "done" and seen[-1][0] == "meta"
    assert seen[-1][4][0]["content"].count("## Part ") == 6 and ollama.fits(seen[-1][4], ollama.REPLY_TOKENS)


def test_a_retry_keeps_the_notes_of_sections_already_analyzed(local, monkeypatch):
    from app import jobs, ollama
    broken = [True]

    def notes(n, kw):
        if n == 3 and broken[0]:
            raise ollama.OllamaError("qwen3.5:9b: out of memory")
        return None  # the fixture's notes

    pdf, seen = long_pdf(local, monkeypatch, notes=notes)
    r = upload(local, "manual.pdf", pdf, model="ollama:qwen3.5:9b")
    job = job_of(local, r)
    assert job["status"] == "failed" and "out of memory" in job["error"]
    assert job["stage"] == "Failed" and local.get("/api/documents").json == []
    broken[0] = False
    local.post(f"/api/jobs/{job['id']}/retry")
    jobs.drain()
    assert job_of(local, r)["status"] == "done"
    pages = [re.search(r"\(page (\d+)\)", s[4][0]["content"]).group(1) for s in seen if s[0] == "notes"]
    assert pages == ["1", "2", "3", "3", "4", "5", "6"]                                  # 1 and 2 not asked again


def test_a_section_whose_notes_are_unusable_is_left_out(local, monkeypatch):
    from app import jobs
    pdf, seen = long_pdf(local, monkeypatch, notes=lambda n, kw: '{"topics": ["cut off' if n == 2 else None)
    job = add(local, "manual.pdf", pdf, model="ollama:qwen3.5:9b")
    assert job["status"] == "done" and job["result"]["analysis"] == "partial: 5 of 6 sections, 83% of the text"
    assert "Part 2 of 6" not in seen[-1][4][0]["content"]
    # No section usable: the analysis fails, and the new upload is still added and searchable.
    pdf, seen = long_pdf(local, monkeypatch, notes=lambda n, kw: "not json")
    job = add(local, "other.pdf", pdf, model="ollama:qwen3.5:9b", on_duplicate="keep")
    assert job["status"] == "done" and job["result"]["analysis"].startswith("none: Expecting value")
    assert local.get("/api/search?q=pump").json
    # A cancel between sections stops the job.
    ids = []

    def cancel_at_3(n, kw):
        if n == 3:
            local.post(f"/api/jobs/{ids[0]}/cancel")
        return None

    pdf, seen = long_pdf(local, monkeypatch, notes=cancel_at_3)
    r = post(local, "third.pdf", pdf, model="ollama:qwen3.5:9b", on_duplicate="keep")
    ids.append(r.json["job"]["id"])
    jobs.drain()
    assert job_of(local, r)["status"] == "cancelled" and len([s for s in seen if s[0] == "notes"]) == 3
    assert len(local.get("/api/documents").json) == 2


def test_claude_analyzes_a_long_document_section_by_section(monkeypatch):
    from types import SimpleNamespace as NS
    from app import llm, sections
    monkeypatch.setattr(llm, "ANALYZE_CHARS", 1000)
    calls = []

    def create(**kw):
        calls.append(kw)
        schema = kw["output_config"]["format"]["schema"]
        body = kw["messages"][0]["content"]
        if schema is sections.NOTES_SCHEMA and "part 2 of" in body:
            return NS(stop_reason="refusal", content=[], usage=None)
        text = ('{"topics": ["t"], "specifications": [], "procedures": [], "warnings": []}' if schema is sections.NOTES_SCHEMA
                else '{"title": "Manual", "doc_type": "manual", "summary": "s", "keywords": ["pump"]}')
        return NS(stop_reason="end_turn", content=[NS(type="text", text=text)], usage=NS(
            input_tokens=300, output_tokens=40, cache_read_input_tokens=0, cache_creation_input_tokens=0))

    seen, steps = [], []
    text = "\f".join(f"Page {n}. " + "Torque the bolts. " * 40 for n in range(1, 5))
    meta, analysis = llm.analyze(NS(beta=NS(messages=NS(create=create))), "m.pdf", text, "claude-haiku-5-5",
                                 on_usage=lambda *c: seen.append(c[0]), on_progress=lambda *s: steps.append(s))
    assert meta["title"] == "Manual" and analysis == "partial: 3 of 4 sections, 75% of the text"   # part 2 refused
    assert [c["max_tokens"] for c in calls] == [8000] * 4 + [2000]
    assert seen == ["analyze-section", "analyze-section", "analyze-section", "analyze"]   # no usage on the refusal
    assert steps[0] == ("Analyzing section 1 of 4", 0, 4) and steps[-1] == ("Combining the notes of 3 sections", 4, 4)


def test_estimate_counts_one_call_per_section(client, monkeypatch):
    from app import jobs, llm, sections
    monkeypatch.setattr(llm, "ANALYZE_CHARS", 1000)
    est = jobs.estimate("a.txt", b"x" * 3000, "claude-opus-5-5", False, "scanned")
    assert est["sections"] == [3, 3]
    assert est["input_tokens"] == 3 * (3000 // 3 // 4 + 400) + 3 * 1000 + 300 and est["output_tokens"] == 3 * 1000 + 500
    monkeypatch.setattr(sections, "MAX_SECTIONS", 2)
    assert jobs.estimate("a.txt", b"x" * 3000, "claude-opus-5-5", False, "scanned")["sections"] == [3, 2]
    assert jobs.estimate("a.txt", b"x" * 900, "claude-opus-5-5", False, "scanned")["sections"] == [1, 1]


def test_split_keeps_pages_whole_and_sections_even():
    from app.sections import split
    pages = [f"Page {n}. " + "a" * 290 for n in range(1, 8)]                               # 7 pages of 298 characters
    parts = split("\f".join(pages), 1000)
    assert [(a, b) for a, b, _ in parts] == [(1, 2), (3, 5), (6, 7)]                       # not 3 + 3 + 1
    assert all(len(t) <= 1000 and "\f" not in t for _, _, t in parts)
    assert "\n\n".join(t for _, _, t in parts) == "\n\n".join(pages)                    # nothing lost or repeated
    parts = split("One.\n\nTwo " + "b " * 800, 600)                                       # no pages: paragraphs, then cuts
    assert all(a is None and len(t) <= 600 for a, _, t in parts) and len(parts) == 3
    assert split("\f \f", 100) == []


def test_unit_misprints_are_listed_at_upload(local, monkeypatch):
    """The unit check runs on the stored chunks, as the excerpts fix does, so the list is
    what answers are corrected for. PDF findings carry their page."""
    from app import extract, jobs
    job = add(local, "spec.txt", "Transmission fluid 38.0 ℓ / 40.1 U.S.gal\n\nFuel 50 L (13.2 US gal)".encode())
    assert job["result"]["unit_fixes"] == 1
    doc = local.get("/api/documents").json[0]
    assert doc["unit_fixes"] == 1 and "unit_findings" not in doc
    r = local.get(f"/api/documents/{doc['id']}/units").json
    assert r["filename"] == "spec.txt" and r["has_file"] and r["pdf"] is False
    assert r["findings"] == [{"page": None, "kind": "quarts-printed-as-gallons", "printed": "38.0 ℓ / 40.1 U.S.gal",
                              "corrected": "38.0 ℓ / 10.0 U.S.gal (40.1 U.S. qt)", "context": "Transmission fluid"}]
    pages = ["Contents and safety.", "Capacity\nTransmission fluid\n38.0 \n40.1 U.S.gal.\nIMPORTANT"]
    monkeypatch.setattr(extract, "pdf_pages", lambda data, **kw: [(t, None) for t in pages])
    job = add(local, "manual.pdf", blank_pdf(2), model="ollama:qwen3.5:9b")
    r = local.get(f"/api/documents/{job['doc_id']}/units").json
    assert r["pdf"] is True and [(f["page"], f["kind"], f["context"]) for f in r["findings"]] == [
        (2, "quarts-printed-as-gallons", "Capacity Transmission fluid")]
    local.post("/api/chat", json={"question": "transmission fluid capacity"})
    assert "10.0 U.S.gal. (40.1 U.S. qt)" in local.calls[-1]["messages"][-1]["content"]  # the fix answers get
    pages[1] = "Capacity\nTransmission fluid\n38.0 L\n10.0 U.S.gal."                      # a re-read finds none
    local.post(f"/api/documents/{job['doc_id']}/reread", json={"model": "ollama:qwen3.5:9b"})
    jobs.drain()
    assert local.get(f"/api/documents/{job['doc_id']}/units").json["findings"] == []
    assert [d["unit_fixes"] for d in local.get("/api/documents").json] == [0, 1]
    assert local.get("/api/documents/9999/units").status_code == 404


# Text whose qwen3.5:9b token count was measured on the shared Ollama (prompt_eval_count).
PROSE = ("Before starting the engine, make sure the PTO lever is in the OFF position and the range gear shift "
         "lever is in neutral. Check the engine oil level with the dipstick and add oil if necessary. ") * 20
TORQUE = "\n".join(f"| Bolt {i} | M{6 + i % 4 * 2} x 1.25 | {i * 7.31:.1f} to {i * 8.04:.1f} | "
                   f"{i * 0.746:.2f} to {i * 0.82:.2f} |" for i in range(1, 121))
FIGURES = "\n".join(",".join(f"{(i * 37 + j * 101) % 10000 / 7:.3f}" for j in range(8)) for i in range(100))


def test_tokens_are_estimated_with_a_token_for_every_digit():
    """qwen3.5 gives every digit a token: tables of figures run at 1-1.3 characters a token,
    prose at 4-5. The 3 characters a token assumed before was 3 times under for figures:
    a 31,728-character section of tables passed the old check but was about 17,000
    tokens, Ollama cut its first half off, and the notes missed a warning at its start."""
    from app import ollama
    for text, measured in ((PROSE, 821), (TORQUE, 5300), (FIGURES, 6320)):
        assert 0.99 <= ollama.tokens(text) / measured <= 1.1, text[:30]
    assert ollama.tokens("") == 0 and ollama.tokens("7" * 2_500_000) == 2_500_000         # a megabyte at a time
    assert abs(ollama.tokens("x" * 3_000_000) - 500_000) <= 3                             # a word cut at each megabyte
    table = "\n".join([TORQUE] * 5)                                                       # 35,000 characters of figures
    assert len(table) / 3 + 2000 <= ollama.NUM_CTX                                        # what the old check let through
    assert not ollama.fits([{"content": table}])                                          # Auto now takes it to Claude


def test_tables_of_figures_are_analyzed_in_smaller_sections(local, monkeypatch):
    """Sections are sized by the token estimate: 35,000 characters of prose are one call,
    of figures several, each prompt within the context."""
    from app import ollama
    sent = []

    def chat(model, messages, **kw):
        sent.append(messages)
        notes = "topics" in kw["format"]["properties"]
        return {"message": {"content": '{"topics": ["t"], "specifications": [], "procedures": [], "warnings": []}'
                            if notes else '{"title": "t", "doc_type": "d", "summary": "s", "keywords": []}'}}

    monkeypatch.setattr(ollama, "_chat", chat)
    prose = "\n\n".join([PROSE] * 9)                                                     # 34,600 characters
    assert ollama.analyze(None, "a.txt", prose)[1] == "full" and len(sent) == 1
    sent.clear()
    table = "\n\n".join([TORQUE] * 5)
    assert ollama.analyze(None, "b.txt", table)[1] == "full" and len(sent) == 4          # 3 sections and the final call
    assert all(ollama.fits(m, ollama.REPLY_TOKENS) for m in sent)


def _ollama_http(monkeypatch, answer):
    """Ollama's HTTP API faked one level down from _chat: answer(body) returns the reply
    dict, or raises (an urllib HTTPError, say). Returns the request bodies."""
    import io
    import json
    from app import ollama
    bodies = []

    def urlopen(req, timeout):
        bodies.append(json.loads(req.data))
        return io.BytesIO(json.dumps(answer(bodies[-1])).encode())

    monkeypatch.setattr(ollama.urllib.request, "urlopen", urlopen)
    return bodies


def test_text_denser_than_reckoned_is_read_in_smaller_pieces(monkeypatch):
    """Analysis calls ask Ollama to refuse a prompt it would have to cut (truncate=False:
    HTTP 400 "exceeds the available context size") and to stop a reply at the end of the
    context (shift=False) rather than drop the start of the text. Either is
    ContextExceeded, and that text is read again in smaller pieces, their notes put
    together: nothing is cut off unnoticed."""
    import io
    import json
    import urllib.error
    from app import ollama
    room = {"prompt": 9000, "reply": 99_999}  # the fake model's context, in prompt characters

    def answer(body):
        prompt = body["messages"][0]["content"]
        assert body["truncate"] is False and body["shift"] is False
        if len(prompt) > room["prompt"]:
            raise urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(
                b'{"error": "request (17012 tokens) exceeds the available context size (16384 tokens), try increasing it"}'))
        if len(prompt) > room["reply"]:                                                   # the context ran out mid-reply
            return {"message": {"content": '{"topics": ["cut'}, "done_reason": "length", "eval_count": 300}
        if "topics" in body["format"]["properties"]:
            steps = re.findall(r"Step (\d+)\.", prompt)
            return {"message": {"content": json.dumps({"topics": steps, "specifications": [], "procedures": [],
                                                       "warnings": []})}, "done_reason": "stop", "eval_count": 9}
        return {"message": {"content": '{"title": "t", "doc_type": "d", "summary": "s", "keywords": []}'},
                "done_reason": "stop", "eval_count": 9}

    bodies = _ollama_http(monkeypatch, answer)
    text = "\n\n".join(f"Step {i}. Check pump {i} and fill it with oil." + " Then wipe it clean." * 25
                       for i in range(1, 25))                                             # 13,100 characters
    meta, analysis = ollama.analyze(None, "m.txt", text)                                  # one call by the estimate
    prompts = [b["messages"][0]["content"] for b in bodies]
    assert analysis == "full" and len(prompts[0]) > 9000 and len(prompts) > 2             # ... which was refused
    final = prompts[-1]
    assert sorted(map(int, re.findall(r"- (\d+)", final))) == list(range(1, 25))          # every step's notes, once
    assert re.search(r"read in [23] parts", final)
    # Sections refused in turn are read in pieces; so are pieces whose reply ran into the
    # end of the context.
    bodies.clear()
    monkeypatch.setattr(ollama, "ANALYZE_CHARS", 12_000)                                  # sections of up to 12,000
    room.update(prompt=9000, reply=4000)
    meta, analysis = ollama.analyze(None, "m.txt", f"{text}\n\n{text}")
    prompts = [b["messages"][0]["content"] for b in bodies]
    answered = [p for p in prompts if len(p) <= 4000]
    assert analysis == "full" and all(", piece " in p for p in answered[:-1])             # all but the final call
    assert any(4000 < len(p) <= 9000 for p in prompts)                                    # one stopped mid-reply
    assert sorted(map(int, re.findall(r"- (\d+)", prompts[-1]))) == sorted(list(range(1, 25)) * 2)
    # Nothing left that fits: the error is the job's (Auto takes it to Claude).
    room.update(prompt=10)
    with pytest.raises(ollama.ContextExceeded, match="exceeds the available context"):
        ollama.analyze(None, "m.txt", text)


def test_a_reply_cut_off_at_its_token_limit_leaves_its_section_out(monkeypatch):
    from app import ollama

    def answer(body):
        prompt = body["messages"][0]["content"]
        if "topics" in body["format"]["properties"] and "page 2" in prompt:               # num_predict reached
            return {"message": {"content": '{"topics": ["cut'}, "done_reason": "length", "eval_count": 2000}
        notes = "topics" in body["format"]["properties"]
        return {"message": {"content": '{"topics": ["t"], "specifications": [], "procedures": [], "warnings": []}'
                            if notes else '{"title": "t", "doc_type": "d", "summary": "s", "keywords": []}'}}

    _ollama_http(monkeypatch, answer)
    monkeypatch.setattr(ollama, "ANALYZE_CHARS", 1000)
    text = "\f".join(f"Page {n}. " + "Torque the bolts. " * 40 for n in range(1, 5))
    assert ollama.analyze(None, "m.pdf", text)[1] == "partial: 3 of 4 sections, 75% of the text"


def test_long_file_names_are_cut_in_analysis_prompts(local, monkeypatch):
    """Upload names have no length limit: a 6,000-character one made every section prompt
    too long for the context, and the error blamed OLLAMA_NUM_CTX."""
    from app import ollama
    sent = []

    def chat(model, messages, **kw):
        sent.append(messages[0]["content"])
        notes = "topics" in kw["format"]["properties"]
        return {"message": {"content": '{"topics": ["t"], "specifications": [], "procedures": [], "warnings": []}'
                            if notes else '{"title": "t", "doc_type": "d", "summary": "s", "keywords": []}'}}

    monkeypatch.setattr(ollama, "_chat", chat)
    assert ollama.analyze(None, "n" * 6000 + ".txt", "word " * 40_000)[1] == "full"
    assert len(sent) > 2 and all(f"File name: {'n' * 199}…\n" in p and ollama.fits([{"content": p}], 2000) for p in sent)


def test_estimate_counts_the_sections_split_makes(monkeypatch):
    """Pages are kept whole, so a text can take more sections than its characters suggest
    (staging: "analysis in 3 of 5 sections" estimated, "partial: 3 of 6 sections" read).
    The estimate counts them as analyze() will split them."""
    from app import extract, jobs, llm, sections
    pages = [f"Page {n}. " + "a" * 590 for n in range(1, 6)]                              # 5 pages of 598 characters
    assert sections.plan("\f".join(pages), 1000) == (5, 5)                                # not ceil(3,000 / 1,000) = 3
    assert sections.plan("x" * 900, 1000) == (1, 1) and sections.plan("x" * 3000, 1000, cap=2) == (3, 2)
    monkeypatch.setattr(llm, "ANALYZE_CHARS", 1000)
    assert jobs.estimate("a.txt", None, "claude-opus-5-5", False, "scanned", text="\f".join(pages))["sections"] == [5, 5]
    monkeypatch.setattr(extract, "pdf_texts", lambda data: pages)                         # every page has its text layer
    assert jobs.estimate("a.pdf", b"", "claude-opus-5-5", True, "scanned")["sections"] == [5, 5]
    est = jobs.estimate("a.pdf", b"", "claude-opus-5-5", True, "all")                     # Claude reads every page
    assert est["sections"][0] == len(sections.split("\f".join(["word " * 480] * 5), 1000)) == 15


def test_a_one_page_pdf_has_page_numbers_too(local, monkeypatch):
    """chunk_text drops page breaks no text follows, so a one-page PDF (or one with text on
    its first page only) has none. Its findings still say page 1, and a later check of
    its chunks (say after units.py changed) remembers it is a PDF's text."""
    from app import db, extract, jobs
    monkeypatch.setattr(extract, "pdf_pages", lambda data, **kw: [("Fuel tank 38.0 ℓ / 40.1 U.S.gal", None), ("", None)])
    job = add(local, "spec.pdf", blank_pdf(2), model="ollama:qwen3.5:9b")
    r = local.get(f"/api/documents/{job['doc_id']}/units").json
    assert r["pdf"] is True and [f["page"] for f in r["findings"]] == [1]
    with db.connect() as con:                                                             # a check from another units.py
        con.execute("UPDATE documents SET unit_findings=json_set(unit_findings, '$.version', 'older')")
    assert jobs.check_units() == 1
    assert [f["page"] for f in local.get(f"/api/documents/{job['doc_id']}/units").json["findings"]] == [1]
    add(local, "spec.txt", b"Fuel tank 38.0 L / 40.1 U.S.gal")                           # text files have no pages
    assert [f["page"] for f in local.get("/api/documents/2/units").json["findings"]] == [None]


def test_a_long_list_of_unit_fixes_is_cut_and_counted(local, monkeypatch):
    """A table whose figures all disagree stored a list as long as the document, parsed on
    every library listing. The first MAX_FINDINGS are kept; all are counted, and listings
    read only the count."""
    from app import db, units
    monkeypatch.setattr(units, "MAX_FINDINGS", 3)
    rows = "\n".join(f"| Day {n} | {n + 10}.0 L | {n + 30}.0 gal |" for n in range(1, 8))
    job = add(local, "fuel log.txt", rows.encode())
    assert job["result"]["unit_fixes"] == 7
    assert local.get("/api/documents").json[0]["unit_fixes"] == 7
    assert local.get("/api/documents?q=fuel").json[0]["unit_fixes"] == 7                  # search lists it the same way
    r = local.get(f"/api/documents/{job['doc_id']}/units").json
    assert r["total"] == 7 and [f["context"][-5:] for f in r["findings"]] == ["Day 1", "Day 2", "Day 3"]
    assert "recalculated" not in r["findings"][1]["context"]                              # the row before, as printed
    with db.connect() as con:                                                             # a check from before the total
        con.execute("UPDATE documents SET unit_findings=json_remove(unit_findings, '$.total')")
    assert local.get("/api/documents").json[0]["unit_fixes"] == 3
    assert local.get(f"/api/documents/{job['doc_id']}/units").json["total"] == 3


def test_too_large_upload_gets_a_json_error(client, monkeypatch):
    from app import main
    monkeypatch.setitem(main.app.config, "MAX_CONTENT_LENGTH", 1000)
    r = post(client, "big.txt", b"x" * 5000)
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
    assert len(batches) > 1 and all(len(b[2]) <= limit for b in batches)
    pages = [p for numbers, total, _ in batches for p in numbers]
    assert pages == list(range(1, 21)) and {b[1] for b in batches} == {20}  # every page once, in order


def test_pdf_batches_of_chosen_pages():
    import pypdfium2
    from app import extract
    batches = extract.pdf_batches(blank_pdf(30), size=3, pages=[29, 2, 5, 7, 11])
    assert [b[0] for b in batches] == [[2, 5, 7], [11, 29]] and {b[1] for b in batches} == {30}
    assert [len(pypdfium2.PdfDocument(b[2])) for b in batches] == [3, 2]                  # just those pages


def test_single_page_too_big_is_reported():
    import pytest
    from app import extract
    with pytest.raises(ValueError, match="page 1 alone"):
        extract.pdf_batches(blank_pdf(2), size=20, max_bytes=10)


# ---- page-numbered citations, search by meaning, questions limited to chosen documents ----

ENGINE = b"Engine oil: change the engine oil every 250 hours and check the oil level daily."
GEARBOX = b"Transmission fluid: the gearbox case takes 38.0 litres of universal tractor fluid."
BRAKES = ["Chapter 7 Brakes. Contents.", "Brake pedal free travel is 30-40 mm. " * 50, "Adjust both brake rods equally."]


def _chunk_rows(doc_id=None):
    """Chunks with their pages, and the model and size (bytes) of their vectors."""
    from app import db
    with db.connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT c.id, c.doc_id, c.idx, c.page_start, c.page_end, v.model, length(v.vector) AS size FROM chunks c"
            " LEFT JOIN chunk_vectors v ON v.chunk_id = c.id WHERE ? IS NULL OR c.doc_id = ? ORDER BY c.id",
            (doc_id, doc_id))]


def test_answers_cite_the_pages_of_their_excerpts(local, monkeypatch):
    """A PDF's chunks keep the pages they span. The model is told each excerpt's page and
    asked to cite it; the answer lists its sources, one per document with its pages, and
    the excerpts as the model read them."""
    from app import extract, jobs
    monkeypatch.setattr(extract, "pdf_pages", lambda data, **kw: [(t, None) for t in BRAKES])
    doc_id = add(local, "Chapter 7 Brake.pdf", blank_pdf(3), model="ollama:qwen3.5:9b")["doc_id"]
    assert [(r["page_start"], r["page_end"]) for r in _chunk_rows(doc_id)] == [(1, 1), (2, 2), (2, 3)]
    from app import llm
    r = local.post("/api/chat", json={"question": "How much free travel has the brake pedal?"}).json
    msg = local.calls[-1]["messages"][-1]["content"]
    assert f"</excerpt>\n</excerpts>\n\n{llm.PAGE_NOTE}\n\nHow much free travel" in msg   # asked to cite pages
    assert "like (Chapter 7 Brake.pdf, p. 12)" in llm.PAGE_NOTE and "p. 12" not in llm.SYSTEM
    for label in ('page="1"', 'page="2"', 'pages="2-3"'):
        assert f'<excerpt file="Chapter 7 Brake.pdf" {label}>' in msg
    assert r["sources"] == [{"doc_id": doc_id, "filename": "Chapter 7 Brake.pdf", "title": "Scan", "has_file": True,
                             "pdf": True, "pages": [1, 2, 3], "excerpts": [0, 1, 2]}]
    assert sorted((e["page_start"], e["page_end"]) for e in r["excerpts"]) == [(1, 1), (2, 2), (2, 3)]
    assert all("\f" not in e["text"] and e["text"] in msg for e in r["excerpts"])          # what the model read
    (spans,) = [e["text"] for e in r["excerpts"] if e["page_end"] == 3]                    # where page 3 starts in it
    assert spans.endswith("30-40 mm.\n\n[Page 3]\nAdjust both brake rods equally.") and spans.count("[Page") == 1
    assert "[Page" not in "".join(e["text"] for e in r["excerpts"] if e["page_end"] < 3)
    hit = local.get("/api/search?q=adjust rods").json[0]                                  # the library's search too
    assert (hit["page_start"], hit["page_end"]) == (2, 3)
    # A re-read keeps them; a text file has none.
    monkeypatch.setattr(extract, "pdf_pages", lambda data, **kw: [("", None), (BRAKES[2], None), ("", None)])
    local.post(f"/api/documents/{doc_id}/reread", json={"model": "ollama:qwen3.5:9b"})
    jobs.drain()
    assert [(r["page_start"], r["page_end"]) for r in _chunk_rows(doc_id)] == [(2, 2)]
    other = add(local, "notes.txt", b"Brake notes.\n\nMore brake notes.")["doc_id"]
    assert [(r["page_start"], r["page_end"]) for r in _chunk_rows(other)] == [(None, None)]


def test_questions_are_searched_without_their_stopwords(client):
    """"what is the" matched nearly every passage, ranked them by how often they say "the",
    and in the blend with search by meaning counted as a keyword match."""
    from app.db import fts_query
    assert fts_query("What is the hydraulic oil capacity?") == '"hydraulic" OR "oil" OR "capacity"'
    assert fts_query("What is 2 + 2?") == ""                                              # nothing to search for
    add(client, "gearbox.txt", GEARBOX)
    assert client.get("/api/search?q=What is the oil capacity?").json == []               # "the" alone doesn't match
    assert client.get("/api/search?q=what is the").json == []
    r = client.post("/api/chat", json={"question": "What is the use of it?"}, headers=H).json
    assert r["excerpts"] == [] and client.seen["msg"] == "What is the use of it?"


def test_search_by_meaning_finds_what_keywords_miss(client, monkeypatch):
    """"hydraulic oil capacity" shares no word with the gearbox passage, so the keyword
    search can't find it; the passages nearest in meaning include it. The two rankings are
    blended by reciprocal rank fusion: a passage found both ways comes first."""
    from app import vectors
    calls = fake_embedder(monkeypatch, [])
    for name, body in (("engine.txt", ENGINE), ("gearbox.txt", GEARBOX), ("lease.txt", LEASE)):
        add(client, name, body)
    assert ("embed", [GEARBOX.decode()]) in calls                                         # passages as they are
    assert [h["filename"] for h in client.get("/api/search?q=hydraulic oil capacity").json] == ["engine.txt"]
    monkeypatch.setenv("TOP_K", "2")
    r = client.post("/api/chat", json={"question": "hydraulic oil capacity"}, headers=H).json
    assert [(e["filename"], e["match"]) for e in r["excerpts"]] == [("engine.txt", "both"), ("gearbox.txt", "meaning")]
    assert "38.0 litres" in client.seen["msg"] and "900 euros" not in client.seen["msg"]
    assert calls[-1] == ("embed-query", [vectors.QUERY.format("hydraulic oil capacity")])  # questions get the instruction
    assert [r["kind"] for r in _usage_rows()][-2:] == ["embed-query", "chat"]             # logged with the answer
    # A question near nothing (cosine under MIN_SIMILARITY everywhere) is sent no passages.
    r = client.post("/api/chat", json={"question": "brake pedal"}, headers=H).json
    assert r["excerpts"] == [] and r["sources"] == [] and client.seen["msg"] == "brake pedal"
    monkeypatch.setattr(vectors, "MIN_SIMILARITY", -1.0)                                   # no floor: the nearest anyway
    assert len(client.post("/api/chat", json={"question": "brake pedal"}, headers=H).json["excerpts"]) == 2


def test_an_upload_is_indexed_for_search_by_meaning(local, monkeypatch):
    """The upload job embeds its chunks before saving them ("Indexing"), a batch per call;
    the vectors are stored with their model, and each call is in the usage log at no cost."""
    from app import db, ollama, vectors
    fake_embedder(monkeypatch)
    monkeypatch.setattr(vectors, "BATCH", 2)
    stages, real = [], db.update_job

    def spy(job_id, when=("running",), **fields):
        if "stage" in fields:
            stages.append((fields["stage"], fields.get("progress"), fields.get("total")))
        return real(job_id, when, **fields)

    monkeypatch.setattr(db, "update_job", spy)
    body = "\n\n".join(f"Part {i} is about the hydraulic fluid. " + "x" * 700 for i in range(5)).encode()
    job = add(local, "long.txt", body)
    assert job["status"] == "done" and job["result"]["chunks"] == 3 and job["result"]["embedded"] is True
    i = stages.index(("Indexing for search by meaning: 0 of 3 passages", 0, 3))
    assert stages[i + 1:i + 4] == [("Indexing for search by meaning: 2 of 3 passages", 2, 3),
                                   ("Indexing for search by meaning: 3 of 3 passages", 3, 3), ("Saving", None, None)]
    assert {(r["model"], r["size"]) for r in _chunk_rows()} == {(ollama.EMBED_MODEL, 4 * (len(MEANINGS) + 1))}
    assert [(r["kind"], r["model"], r["doc_id"], r["cost_usd"]) for r in _usage_rows()] == [
        ("analyze", ollama.MODEL, job["doc_id"], 0)] + [("embed", ollama.EMBED_MODEL, job["doc_id"], 0)] * 2


def test_without_the_embedding_model_uploads_and_questions_go_on(local, monkeypatch):
    """Ollama down, or the embedding model not pulled: the upload is added without vectors,
    questions are answered by keyword search, and embedding pauses instead of being tried
    for every question or upload. Later the worker's backfill embeds what is missing."""
    from app import ollama, vectors
    tries = []

    def missing(texts, model=None, kind="embed", on_usage=None):
        tries.append(kind)
        raise ollama.OllamaError(f"{ollama.EMBED_MODEL}: model not found, try pulling it first")

    monkeypatch.setattr(ollama, "embed", missing)
    job = add(local, "gearbox.txt", GEARBOX)
    assert job["status"] == "done" and job["result"]["embedded"] is False and tries == ["embed"]
    assert [r["model"] for r in _chunk_rows()] == [None]
    r = local.post("/api/chat", json={"question": "How many litres of transmission fluid?"})
    assert r.status_code == 200 and r.json["excerpts"][0]["match"] == "keywords"
    status = local.get("/api/search/status").json
    assert (status["model"], status["chunks"], status["embedded"]) == (ollama.EMBED_MODEL, 1, 0)
    assert "model not found" in status["error"]
    assert vectors.backfill() == 0 and add(local, "engine.txt", ENGINE)["result"]["embedded"] is False
    assert tries == ["embed"]                                                              # paused meanwhile
    fake_embedder(monkeypatch)
    monkeypatch.setattr(vectors, "_retry_at", 0.0)                                         # RETRY_SECONDS later
    assert vectors.backfill() == 2 and vectors.backfill() == 0
    status = local.get("/api/search/status").json
    assert (status["embedded"], status["error"]) == (2, None)
    assert [(r["kind"], r["doc_id"]) for r in _usage_rows() if r["kind"] == "embed"] == [("embed", 1), ("embed", 2)]
    r = local.post("/api/chat", json={"question": "hydraulic capacity"}).json              # no word in common
    assert [(e["filename"], e["match"]) for e in r["excerpts"]] == [("gearbox.txt", "meaning"), ("engine.txt", "meaning")]


def test_vectors_follow_rereads_and_deletes(local, monkeypatch):
    """A re-read's new chunks get new vectors (the old ones go with their chunks), a delete
    takes them away, and the vectors held in memory are reloaded after each: SQLite gives a
    new chunk the id of a deleted one, which must not keep the old one's vector."""
    from app import db, extract, jobs, vectors
    fake_embedder(monkeypatch)
    monkeypatch.setattr(extract, "pdf_pages", lambda data, **kw: [("Transmission fluid: 38 litres.", None)])
    doc_id = add(local, "spec.pdf", blank_pdf(1), model="ollama:qwen3.5:9b")["doc_id"]
    assert vectors._vectors()[2][0].argmax() == 0                                         # the fluid meaning
    monkeypatch.setattr(extract, "pdf_pages", lambda data, **kw: [("Brake pedal: 30 mm of free travel.", None)])
    local.post(f"/api/documents/{doc_id}/reread", json={"model": "ollama:qwen3.5:9b"})
    jobs.drain()
    assert [(r["id"], r["model"] is not None) for r in _chunk_rows()] == [(1, True)]      # the same chunk id ...
    assert vectors._vectors()[2][0].argmax() == 4                                         # ... with its own vector
    assert local.delete(f"/api/documents/{doc_id}").status_code == 204
    assert vectors._vectors() is None and _chunk_rows() == []
    with db.connect() as con:
        assert con.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] == 0


def test_a_vector_made_for_text_that_changed_meanwhile_is_not_kept(client):
    """The backfill embeds a chunk's text while uploads and re-reads go on: if the chunk
    has other text by the time the vector is stored (a re-read, or a new chunk given the
    id), the vector is dropped and the new text is embedded on the next round."""
    from app import db, ollama
    (row,) = _chunk_rows(add(client)["doc_id"])
    assert db.store_vectors(ollama.EMBED_MODEL, [(row["id"], "Some other text.", bytes(24))]) == 0
    assert db.store_vectors(ollama.EMBED_MODEL, [(row["id"], LEASE.decode(), bytes(24))]) == 1
    assert db.store_vectors(ollama.EMBED_MODEL, [(999, LEASE.decode(), bytes(24))]) == 0  # no such chunk


def test_vectors_of_another_embedding_model_are_made_again(client, monkeypatch):
    from app import db, ollama, vectors
    fake_embedder(monkeypatch)
    add(client, "gearbox.txt", GEARBOX)
    assert db.vector_counts(ollama.EMBED_MODEL) == (1, 1)
    monkeypatch.setattr(ollama, "EMBED_MODEL", "other-embedding:1b")                       # OLLAMA_EMBED_MODEL changed
    assert db.vector_counts("other-embedding:1b") == (1, 0) and vectors._vectors() is None  # never compared across models
    assert vectors.backfill() == 1 and [r["model"] for r in _chunk_rows()] == ["other-embedding:1b"]


def test_the_worker_embeds_chunks_that_have_no_vectors(client, monkeypatch):
    """At start, and whenever no job is queued, the worker thread embeds the chunks that
    have no vectors: the live library's, and uploads made while embedding was paused."""
    import time
    from app import db, jobs, ollama, vectors
    add(client)                                                                           # no embedding model yet
    assert db.vector_counts(ollama.EMBED_MODEL) == (1, 0)
    fake_embedder(monkeypatch)
    monkeypatch.setattr(vectors, "_retry_at", 0.0)
    monkeypatch.setattr(jobs, "BACKGROUND", True)
    monkeypatch.setattr(jobs, "_thread", None)
    try:
        jobs.start()
        for _ in range(100):
            if db.vector_counts(ollama.EMBED_MODEL) == (1, 1):
                break
            time.sleep(0.05)
    finally:
        jobs.stop()
        monkeypatch.setattr(jobs, "BACKGROUND", False)
    assert db.vector_counts(ollama.EMBED_MODEL) == (1, 1)


def test_a_question_can_be_limited_to_chosen_documents(client, monkeypatch):
    """doc_ids limits both searches, by keywords and by meaning, to those documents, and
    the model is told which they are."""
    from app import llm
    fake_embedder(monkeypatch)
    lease, gearbox, engine = (add(client, name, body)["doc_id"] for name, body in (
        ("lease.txt", LEASE), ("gearbox.txt", GEARBOX), ("engine.txt", ENGINE)))

    def ask(**kw):
        return client.post("/api/chat", json={"question": "rent and oil and fluid", **kw}, headers=H)

    assert {e["filename"] for e in ask().json["excerpts"]} == {"lease.txt", "gearbox.txt", "engine.txt"}
    r = ask(doc_ids=[gearbox, engine]).json
    assert r["scope"] == [gearbox, engine] and {e["doc_id"] for e in r["excerpts"]} == {gearbox, engine}
    assert {s["doc_id"] for s in r["sources"]} == {gearbox, engine}
    assert client.seen["msg"].startswith("<scope>Only these documents were searched: gearbox.txt, engine.txt</scope>\n")
    assert [(e["filename"], e["match"]) for e in ask(doc_ids=[lease]).json["excerpts"]] == [("lease.txt", "both")]
    assert ask(doc_ids=[]).json["scope"] is None and "<scope>" not in client.seen["msg"]   # [] is every document
    assert ask(doc_ids=[lease, 99999, lease]).json["scope"] == [lease]                    # one deleted meanwhile
    for bad in ("1", [True], ["1"], [1.5], {"ids": [1]}):
        assert ask(doc_ids=bad).status_code == 400
    r = ask(doc_ids=[99999, 2**70])
    assert r.status_code == 400 and "None of the chosen documents" in r.json["error"]
    names = [f"f{i}.pdf" for i in range(12)]
    assert llm.build_user_message("Why?", [], names) == (
        "<scope>Only these documents were searched: " + ", ".join(names[:10]) + " and 2 more</scope>\nWhy?")


@pytest.mark.parametrize("schema", OLD_SCHEMAS.values(), ids=OLD_SCHEMAS.keys())
def test_chunks_from_before_pages_and_vectors_still_work(client, tmp_path, monkeypatch, schema):
    """The live library's chunks have no pages and no vectors: they are cited by file name,
    found by keywords at first, and embedded by the backfill from their stored text (no
    model reads the documents again), after which search by meaning finds them too."""
    import sqlite3
    from app import db, ollama, vectors
    path = tmp_path / "live.db"
    con = sqlite3.connect(path)
    con.executescript(schema)
    con.close()
    monkeypatch.setattr(db, "DB_PATH", str(path))
    r = client.post("/api/chat", json={"question": "When do I change the hydraulic oil?"}, headers=H).json
    assert r["sources"] == [{"doc_id": 1, "filename": "manual.pdf", "title": "Tractor manual", "has_file": False,
                             "pdf": False, "pages": [], "excerpts": [0]}]
    assert '<excerpt file="manual.pdf">\nChange the hydraulic oil every 400 hours.' in client.seen["msg"]
    assert "page" not in client.seen["msg"].lower()                                       # nothing to make a page up from
    assert _chunk_rows() == [{"id": 1, "doc_id": 1, "idx": 0, "page_start": None, "page_end": None, "model": None,
                              "size": None}]
    assert client.get("/api/search/status").json["embedded"] == 0
    fake_embedder(monkeypatch)
    assert vectors.backfill() == 1 and _chunk_rows()[0]["model"] == ollama.EMBED_MODEL
    r = client.post("/api/chat", json={"question": "lubricant quantity"}, headers=H).json  # no word in common
    assert [(e["doc_id"], e["match"]) for e in r["excerpts"]] == [(1, "meaning")]
    (row,) = [u for u in _usage_rows() if u["kind"] == "embed"]
    assert row["doc_id"] == 1 and row["cost_usd"] == 0
