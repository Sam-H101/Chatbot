"""Embedded metadata database: SQLite + FTS5. One file, no server."""
import os
import re
import sqlite3

DB_PATH = os.environ.get("DB_PATH", "data/metabase.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    filename    TEXT NOT NULL,
    uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
    char_count  INTEGER NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    doc_type    TEXT NOT NULL DEFAULT '',
    summary     TEXT NOT NULL DEFAULT '',
    keywords    TEXT NOT NULL DEFAULT '',
    analysis    TEXT NOT NULL DEFAULT '',  -- 'full', 'partial' or 'none: <reason>'
    model       TEXT NOT NULL DEFAULT ''   -- model that read the document, e.g. 'Claude Opus 5.5'
);
CREATE TABLE IF NOT EXISTS chunks (
    id     INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx    INTEGER NOT NULL,
    text   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS instructions (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    enabled    INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    filename, title, doc_type, summary, keywords, tokenize='porter unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, tokenize='porter unicode61'
);
"""


def connect():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    if "model" not in {r[1] for r in con.execute("PRAGMA table_info(documents)")}:  # older databases
        con.execute("ALTER TABLE documents ADD COLUMN model TEXT NOT NULL DEFAULT ''")
    return con


def add_document(filename, chunks, meta, analysis, model=""):
    with connect() as con:
        cur = con.execute(
            "INSERT INTO documents (filename, char_count, title, doc_type, summary, keywords, analysis, model)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (filename, sum(len(c) for c in chunks), meta.get("title", ""), meta.get("doc_type", ""),
             meta.get("summary", ""), ", ".join(meta.get("keywords", [])), analysis, model),
        )
        doc_id = cur.lastrowid
        con.execute(
            "INSERT INTO docs_fts (rowid, filename, title, doc_type, summary, keywords) VALUES (?,?,?,?,?,?)",
            (doc_id, filename, meta.get("title", ""), meta.get("doc_type", ""),
             meta.get("summary", ""), ", ".join(meta.get("keywords", []))),
        )
        for i, text in enumerate(chunks):
            cid = con.execute(
                "INSERT INTO chunks (doc_id, idx, text) VALUES (?,?,?)", (doc_id, i, text)
            ).lastrowid
            con.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?,?)", (cid, text))
    return doc_id


def list_documents():
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM documents ORDER BY id DESC")]


def delete_document(doc_id):
    return delete_documents([doc_id]) > 0


def delete_documents(ids):
    """Delete several documents in one transaction. Returns how many existed."""
    ids = list(dict.fromkeys(int(i) for i in ids))
    deleted = 0
    with connect() as con:
        for start in range(0, len(ids), 500):  # stay well under SQLite's bound-parameter limit
            part = ids[start:start + 500]
            marks = ",".join("?" * len(part))
            con.execute(f"DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id IN ({marks}))", part)
            con.execute(f"DELETE FROM docs_fts WHERE rowid IN ({marks})", part)
            deleted += con.execute(f"DELETE FROM documents WHERE id IN ({marks})", part).rowcount
    return deleted


def fts_query(text):
    """Turn free text into a safe FTS5 query: quoted terms OR-ed together."""
    terms = [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


def search_documents(text, limit=10):
    q = fts_query(text)
    if not q:
        return []
    with connect() as con:
        rows = con.execute(
            "SELECT d.* FROM docs_fts f JOIN documents d ON d.id = f.rowid"
            " WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts) LIMIT ?", (q, limit))
        return [dict(r) for r in rows]


def search_chunks(text, limit=5):
    q = fts_query(text)
    if not q:
        return []
    with connect() as con:
        rows = con.execute(
            "SELECT c.doc_id, c.idx, c.text, d.filename, d.title,"
            " snippet(chunks_fts, 0, '[', ']', ' … ', 24) AS snippet"
            " FROM chunks_fts f JOIN chunks c ON c.id = f.rowid JOIN documents d ON d.id = c.doc_id"
            " WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?", (q, limit))
        return [dict(r) for r in rows]


# ---- operator instructions added to the chat system prompt ----

def list_instructions():
    with connect() as con:
        return [{**dict(r), "enabled": bool(r["enabled"])}
                for r in con.execute("SELECT * FROM instructions ORDER BY id")]


def active_instructions():
    with connect() as con:
        return [r[0] for r in con.execute("SELECT text FROM instructions WHERE enabled ORDER BY id")]


def add_instruction(text, enabled=True):
    with connect() as con:
        return con.execute("INSERT INTO instructions (text, enabled) VALUES (?,?)", (text, int(enabled))).lastrowid


def update_instruction(rule_id, text=None, enabled=None):
    with connect() as con:
        if text is not None:
            con.execute("UPDATE instructions SET text=? WHERE id=?", (text, rule_id))
        if enabled is not None:
            con.execute("UPDATE instructions SET enabled=? WHERE id=?", (int(enabled), rule_id))
        return con.execute("SELECT 1 FROM instructions WHERE id=?", (rule_id,)).fetchone() is not None


def delete_instruction(rule_id):
    with connect() as con:
        return con.execute("DELETE FROM instructions WHERE id=?", (rule_id,)).rowcount > 0
