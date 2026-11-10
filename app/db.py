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
    analysis    TEXT NOT NULL DEFAULT ''   -- 'full', 'partial' or 'none: <reason>'
);
CREATE TABLE IF NOT EXISTS chunks (
    id     INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx    INTEGER NOT NULL,
    text   TEXT NOT NULL
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
    return con


def add_document(filename, chunks, meta, analysis):
    with connect() as con:
        cur = con.execute(
            "INSERT INTO documents (filename, char_count, title, doc_type, summary, keywords, analysis)"
            " VALUES (?,?,?,?,?,?,?)",
            (filename, sum(len(c) for c in chunks), meta.get("title", ""), meta.get("doc_type", ""),
             meta.get("summary", ""), ", ".join(meta.get("keywords", [])), analysis),
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
    with connect() as con:
        con.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id=?)", (doc_id,))
        con.execute("DELETE FROM docs_fts WHERE rowid=?", (doc_id,))
        return con.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount > 0


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
