import base64
import hashlib
import io
import re

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import db, jobs, llm, main
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
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
    upload(client)
    r = client.post("/api/chat", json={"question": "How much is the rent?", "history": []}, headers=H)
    assert r.json["answer"] == "42" and r.json["sources"] == ["lease.txt"]
    assert "900 euros" in client.seen["msg"]
    assert r.json["turns"][0]["content"] == client.seen["msg"]


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
    assert job["status"] == "done" and job["result"]["title"] == "Scan" and job["total"] == 2
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

    def analyze(c, name, text, model=None, on_usage=None):
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


@pytest.mark.parametrize("schema", [SCHEMA_894B984, SCHEMA_894B984 + SCHEMA_CD8C125_EXTRA], ids=["894b984", "cd8c125"])
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
    assert client.delete("/api/documents/1").status_code == 204
    assert [d["filename"] for d in client.get("/api/documents").json] == ["manual.pdf.txt"]


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
    from app import db

    def locked(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "add_usage", locked)
    r = client.post("/api/chat", json={"question": "hi"}, headers=H)
    assert r.status_code == 200 and r.json["answer"] == "42"
    job = add(client)
    assert job["status"] == "done" and [d["id"] for d in client.get("/api/documents").json] == [job["doc_id"]]
    job = add(client, "blank.txt", b"   ")
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
