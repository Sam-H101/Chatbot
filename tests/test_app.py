import io

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import db, llm, main
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setattr(llm, "client_for", lambda key: object())
    monkeypatch.setattr(llm, "analyze", lambda c, name, text: (
        {"title": "Lease", "doc_type": "contract", "summary": "A flat lease.", "keywords": ["rent", "tenant"]}, "full"))
    seen = {}

    def fake_chat(c, history, msg):
        seen["history"], seen["msg"] = history, msg
        return "42", {"input": 1, "output": 1, "cache_read": 0, "cache_write": 0}

    monkeypatch.setattr(llm, "chat", fake_chat)
    c = main.app.test_client()
    c.seen = seen
    return c


H = {"X-API-Key": "k"}


def upload(client, name="lease.txt", body=b"The monthly rent is 900 euros.\n\nPets are allowed."):
    return client.post("/api/upload", data={"file": (io.BytesIO(body), name)}, headers=H)


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
