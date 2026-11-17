"""Moving a SQLite library into Postgres (app/migrate.py), as the live library is moved."""
import struct

import pytest

from test_app import H, OLD_SCHEMAS, add, client, fake_embedder  # noqa: F401  (client: the app's fixture)

# The schema the app had last on SQLite (folders, key details, conversations with folder_ids),
# with a library in it: two folders, documents in one of them, a vector, a chat limited to a
# document and a folder, an instruction turned off, and ids with gaps where rows were deleted.
LAST_SQLITE = OLD_SCHEMAS["42a0a88"] + """
ALTER TABLE documents ADD COLUMN folder_id INTEGER;
ALTER TABLE documents ADD COLUMN filed_by TEXT;
ALTER TABLE documents ADD COLUMN suggested_folder TEXT;
ALTER TABLE documents ADD COLUMN document_date TEXT;
ALTER TABLE documents ADD COLUMN vendor TEXT;
ALTER TABLE documents ADD COLUMN total_amount REAL;
ALTER TABLE documents ADD COLUMN currency TEXT;
ALTER TABLE documents ADD COLUMN due_date TEXT;
ALTER TABLE documents ADD COLUMN user_edited TEXT;
ALTER TABLE conversations ADD COLUMN folder_ids TEXT;
CREATE TABLE folders (id INTEGER PRIMARY KEY AUTOINCREMENT, parent_id INTEGER, name TEXT NOT NULL,
    name_key TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')));
INSERT INTO folders (id, parent_id, name, name_key) VALUES (3, NULL, 'Tractor', 'tractor'), (7, 3, 'Manuals', 'manuals');
UPDATE documents SET folder_id = 7, vendor = 'KIOTI', user_edited = '["vendor"]' WHERE id = 1;
INSERT INTO documents (id, filename, char_count, title, total_amount, currency) VALUES
    (9, 'receipt.txt', 30, 'Oil receipt', 42.5, 'USD');
INSERT INTO chunks (id, doc_id, idx, text) VALUES (12, 9, 0, 'Paid 42.50 for hydraulic oil.');
UPDATE instructions SET enabled = 0;
INSERT INTO conversations (id, title, model, doc_ids, folder_ids) VALUES (5, 'Oil', 'auto', '[9]', '[3]');
INSERT INTO messages (id, conversation_id, role, content, sent) VALUES (20, 5, 'user', 'Oil?', 'sent Oil?');
INSERT INTO messages (id, conversation_id, role, content, usage) VALUES (21, 5, 'assistant', 'Every 400 hours.', '{}');
INSERT INTO usage (kind, backend, model, input_tokens, cost_usd, doc_id) VALUES ('chat', 'claude', 'claude-opus-5-5', 10, 0.25, 9);
"""


def _vector(values):
    return struct.pack(f"<{len(values)}f", *values)


def _sqlite(tmp_path, extra=""):
    import sqlite3
    path = tmp_path / "metabase.db"
    con = sqlite3.connect(path)
    con.executescript(LAST_SQLITE + extra)
    con.commit()
    con.close()
    return str(path)


def test_a_library_is_moved_with_its_ids(client, tmp_path, monkeypatch):  # noqa: F811
    from app import db, migrate, ollama
    fake_embedder(monkeypatch)
    meaning = [3.0, 0.0, 0.0, 0.0, 0.0, 0.1]                                                # the fluid meaning
    path = _sqlite(tmp_path)
    import sqlite3
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO chunk_vectors (chunk_id, model, vector) VALUES (12, ?, ?)",
                    (ollama.EMBED_MODEL, _vector(meaning)))
    assert migrate.migrate(path, dry_run=True)["documents"] == 2
    assert db.list_documents() == []                                                        # a dry run copies nothing
    copied = migrate.migrate(path)
    assert {k: copied[k] for k in ("folders", "documents", "chunks", "chunk_vectors", "conversations", "messages")} == {
        "folders": 2, "documents": 2, "chunks": 2, "chunk_vectors": 1, "conversations": 1, "messages": 2}
    docs = {d["id"]: d for d in client.get("/api/documents").json}
    assert sorted(docs) == [1, 9]
    assert (docs[1]["folder_id"], docs[1]["vendor"], docs[1]["user_edited"]) == (7, "KIOTI", ["vendor"])
    assert (docs[9]["total_amount"], docs[9]["currency"]) == (42.5, "USD")
    assert [f["path"] for f in client.get("/api/folders").json] == [["Tractor"], ["Tractor", "Manuals"]]
    assert client.get("/api/instructions").json[0]["enabled"] is False
    c = client.get("/api/conversations/5").json
    assert (c["doc_ids"], c["folder_ids"], [m["content"] for m in c["messages"]]) == ([9], [3], ["Oil?", "Every 400 hours."])
    assert client.get("/api/usage?days=3650").json["total"]["cost_usd"] == pytest.approx(0.25)
    # Search: by keywords (the tsvector is made from the text), and by meaning (the vector came along).
    assert sorted(h["doc_id"] for h in client.get("/api/search?q=hydraulic").json) == [1, 9]
    assert db.vector_counts(ollama.EMBED_MODEL) == (2, 1)
    near = db.nearest_chunks(ollama.EMBED_MODEL, meaning, 5)
    assert near[0][0] == 12 and near[0][1] == pytest.approx(1.0)
    # New rows carry on after the highest id of each table.
    assert add(client, "new.txt", b"Grease the pins weekly.")["doc_id"] == 10
    assert client.post("/api/folders", json={"name": "Receipts"}).json["id"] == 8
    r = client.post("/api/chat", json={"question": "Grease?"}, headers=H).json
    assert r["conversation_id"] == 6


def test_a_library_is_moved_only_into_an_empty_one(client, tmp_path):  # noqa: F811
    from app import migrate
    add(client)
    with pytest.raises(migrate.MigrateError, match="already holds 1 document"):
        migrate.migrate(_sqlite(tmp_path))
    with pytest.raises(migrate.MigrateError, match="No SQLite library"):
        migrate.migrate(str(tmp_path / "missing.db"))


def test_text_postgres_cannot_hold_is_cleaned_on_the_way(client, tmp_path):  # noqa: F811
    """SQLite text may hold NUL characters (Postgres text can't) and, from before they were
    dropped at indexing, search's match marks: both are left out."""
    from app import db, migrate
    path = _sqlite(tmp_path)
    import sqlite3
    with sqlite3.connect(path) as con:
        con.execute("UPDATE chunks SET text = ? WHERE id = 12", (f"Paid\x00 42.50 for {db.MATCH_START}oil{db.MATCH_END}.",))
    migrate.migrate(path)
    assert db.document_text(9) == "Paid 42.50 for oil."


def test_the_migrate_command_line(pg, tmp_path):
    import os
    import subprocess
    import sys
    from app import db
    path = _sqlite(tmp_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "DATABASE_URL": db.DATABASE_URL, "DB_SCHEMA": db.DB_SCHEMA}

    def run(*args):
        return subprocess.run([sys.executable, "-m", "app.migrate", *args], cwd=root, env=env,
                              capture_output=True, text=True, timeout=120)

    r = run(path, "--dry-run")
    assert r.returncode == 0 and "Would copy:" in r.stdout and "documents" in r.stdout
    r = run(path)
    assert r.returncode == 0 and "Copied:" in r.stdout and "SQLite file is unchanged" in r.stdout
    r = run(path)
    assert r.returncode == 1 and "already holds 2 document(s)" in r.stderr and "Traceback" not in r.stderr
