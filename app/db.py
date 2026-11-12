"""Embedded metadata database: SQLite + FTS5. One file, no server.
Original uploads are kept as plain files next to it (see files_dir)."""
import hashlib
import os
import re
import sqlite3
import tempfile
import threading

DB_PATH = os.environ.get("DB_PATH", "data/metabase.db")
# Where original uploads are kept, one file per distinct content, named by its SHA-256.
# Empty means a "files" folder next to DB_PATH (/data/files in the container).
FILES_DIR = os.environ.get("FILES_DIR", "")
# Serialises writing and removing stored files with the rows that point at them, so a
# delete never removes a file that an upload is about to reference (one gunicorn worker).
_files_lock = threading.Lock()
# The schema is brought up to date once per database per process, under this lock: the
# page's first requests arrive together, and two threads that both saw a column missing
# would both run ALTER TABLE, failing the slower request.
_schema_lock = threading.Lock()
_schema_ready = set()

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
    model       TEXT NOT NULL DEFAULT '',  -- model that read the document, e.g. 'Claude Opus 5.5'
    sha256      TEXT,                      -- see COLUMNS below; older databases gain these on connect
    file_path   TEXT,
    file_size   INTEGER,
    mime        TEXT
);
CREATE TABLE IF NOT EXISTS usage (
    id                 INTEGER PRIMARY KEY,
    created_at         TEXT NOT NULL DEFAULT (datetime('now')),
    kind               TEXT NOT NULL,      -- 'transcribe', 'analyze', 'chat', ...
    backend            TEXT NOT NULL,      -- 'claude' or 'ollama'
    model              TEXT NOT NULL,
    input_tokens       INTEGER NOT NULL DEFAULT 0,  -- uncached input
    output_tokens      INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd           REAL NOT NULL DEFAULT 0,
    doc_id             INTEGER              -- no foreign key: the cost stays logged after a delete,
                                            -- with doc_id set to NULL (ids get reused)
);
CREATE INDEX IF NOT EXISTS usage_created ON usage(created_at);
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


# Columns added after the first release. Older databases gain them on connect; their
# rows keep NULL here (no stored file, no hash) and still list, search and answer.
COLUMNS = {
    "model": "TEXT NOT NULL DEFAULT ''",
    "sha256": "TEXT",      # hash of the uploaded bytes, for duplicate detection
    "file_path": "TEXT",   # stored original, relative to files_dir()
    "file_size": "INTEGER",
    "mime": "TEXT",
}


def connect():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    fresh = not os.path.exists(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    if fresh or DB_PATH not in _schema_ready:
        with _schema_lock:
            _migrate(con)
            _schema_ready.add(DB_PATH)
    return con


def _migrate(con):
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(documents)")}
    for name, decl in COLUMNS.items():
        if name not in have:
            try:
                con.execute(f"ALTER TABLE documents ADD COLUMN {name} {decl}")
            except sqlite3.OperationalError as e:  # another process added it first
                if "duplicate column name" not in str(e):
                    raise
    con.execute("CREATE INDEX IF NOT EXISTS documents_sha256 ON documents(sha256)")


def _doc(row):
    """A documents row for the API: the stored file's location stays internal."""
    d = dict(row)
    d["has_file"] = bool(d.pop("file_path", None))
    return d


# ---- original files ----

def files_dir():
    return FILES_DIR or os.path.join(os.path.dirname(DB_PATH) or ".", "files")


def file_hash(data):
    return hashlib.sha256(data).hexdigest()


def _store(sha, data):
    """Write the bytes once per content; True if this call wrote the file. A temporary
    file plus rename means a crash never leaves a half-written original under its name."""
    folder = files_dir()
    path = os.path.join(folder, sha)
    if os.path.exists(path):
        return False
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".upload-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)  # mkstemp makes it private to the writer
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
    return True


def _remove(file_path):
    try:
        os.remove(os.path.join(files_dir(), os.path.basename(file_path)))
    except FileNotFoundError:
        pass


def stored_file(doc_id):
    """The kept original of a document: {path, filename, mime, size}, or None for
    documents uploaded before originals were kept (or whose file has gone missing).
    Re-reading a document with another model starts from here."""
    with connect() as con:
        row = con.execute("SELECT filename, file_path, file_size, mime FROM documents WHERE id=?",
                          (doc_id,)).fetchone()
    if not row or not row["file_path"]:
        return None
    path = os.path.join(files_dir(), os.path.basename(row["file_path"]))
    if not os.path.isfile(path):
        return None
    return {"path": path, "filename": row["filename"], "mime": row["mime"] or "application/octet-stream",
            "size": row["file_size"]}


def find_duplicates(sha):
    """Documents holding a file with this hash, newest first. Documents from before
    hashing have none, so they never match."""
    with connect() as con:
        return [_doc(r) for r in con.execute("SELECT * FROM documents WHERE sha256=? ORDER BY id DESC", (sha,))]


def add_document(filename, chunks, meta, analysis, model="", data=None, mime=None, sha=None, replace=False):
    """Index a document. `data` is the uploaded file, kept under files_dir(); `sha` is
    its hash if the caller already has it. replace=True first deletes the documents
    that hold the same file, in the same transaction. Returns the new id."""
    if data is not None:
        sha = sha or file_hash(data)
    kept = data is not None
    with _files_lock:
        wrote = False
        try:
            with connect() as con:
                if replace and sha:
                    old = [r[0] for r in con.execute("SELECT id FROM documents WHERE sha256=?", (sha,))]
                    _delete_rows(con, old)  # the file stays: the new row points at it
                cur = con.execute(
                    "INSERT INTO documents (filename, char_count, title, doc_type, summary, keywords, analysis,"
                    " model, sha256, file_path, file_size, mime) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (filename, sum(len(c) for c in chunks), meta.get("title", ""), meta.get("doc_type", ""),
                     meta.get("summary", ""), ", ".join(meta.get("keywords", [])), analysis, model,
                     sha if kept else None, sha if kept else None, len(data) if kept else None,
                     mime if kept else None),
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
                if kept:  # last, so a failed insert leaves no file behind
                    wrote = _store(sha, data)
        except BaseException:
            if wrote:  # written for rows that never committed
                _remove(sha)
            raise
    return doc_id


def list_documents():
    with connect() as con:
        return [_doc(r) for r in con.execute("SELECT * FROM documents ORDER BY id DESC")]


def delete_document(doc_id):
    return delete_documents([doc_id]) > 0


def _delete_rows(con, ids):
    """Delete documents and their index rows. Returns (how many existed, the stored
    files they pointed at)."""
    deleted, files = 0, set()
    for start in range(0, len(ids), 500):  # stay well under SQLite's bound-parameter limit
        part = ids[start:start + 500]
        marks = ",".join("?" * len(part))
        files |= {r[0] for r in con.execute(
            f"SELECT file_path FROM documents WHERE id IN ({marks}) AND file_path IS NOT NULL", part)}
        con.execute(f"DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id IN ({marks}))", part)
        con.execute(f"DELETE FROM docs_fts WHERE rowid IN ({marks})", part)
        # SQLite hands a deleted id to the next document: keep the cost, drop the link.
        con.execute(f"UPDATE usage SET doc_id = NULL WHERE doc_id IN ({marks})", part)
        deleted += con.execute(f"DELETE FROM documents WHERE id IN ({marks})", part).rowcount
    return deleted, files


def delete_documents(ids):
    """Delete several documents in one transaction, and the stored files no other
    document still uses. Returns how many existed."""
    ids = [i for i in dict.fromkeys(int(i) for i in ids) if -2**63 <= i < 2**63]  # others can't exist
    with _files_lock:
        with connect() as con:
            deleted, files = _delete_rows(con, ids)
            unused = [p for p in files
                      if not con.execute("SELECT 1 FROM documents WHERE file_path=?", (p,)).fetchone()]
        for p in unused:  # only after the commit: a rollback must not lose a file
            _remove(p)
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
        return [_doc(r) for r in rows]


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


# ---- usage log: one row per model call ----

def add_usage(calls, doc_id=None):
    """calls: [(kind, backend, model, usage dict, cost_usd)], where usage has the keys
    input, output, cache_read and cache_write (as llm.chat returns it)."""
    if not calls:
        return
    with connect() as con:
        con.executemany(
            "INSERT INTO usage (kind, backend, model, input_tokens, output_tokens, cache_read_tokens,"
            " cache_write_tokens, cost_usd, doc_id) VALUES (?,?,?,?,?,?,?,?,?)",
            [(kind, backend, model, u.get("input") or 0, u.get("output") or 0, u.get("cache_read") or 0,
              u.get("cache_write") or 0, cost, doc_id) for kind, backend, model, u, cost in calls])


_SUMS = ("COUNT(*) AS calls, SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,"
         " SUM(cache_read_tokens) AS cache_read_tokens, SUM(cache_write_tokens) AS cache_write_tokens,"
         " SUM(cost_usd) AS cost_usd")


def usage_summary(days=30):
    """Totals over the last `days` days (today included, UTC): per model, per day, overall."""
    since = f"-{days - 1} days"
    where = "WHERE created_at >= date('now', ?)"
    with connect() as con:
        by_model = [dict(r) for r in con.execute(
            f"SELECT backend, model, {_SUMS} FROM usage {where} GROUP BY backend, model"
            " ORDER BY SUM(cost_usd) DESC, COUNT(*) DESC", (since,))]
        by_day = [dict(r) for r in con.execute(
            f"SELECT date(created_at) AS day, {_SUMS} FROM usage {where} GROUP BY day ORDER BY day DESC", (since,))]
        total = dict(con.execute(f"SELECT {_SUMS} FROM usage {where}", (since,)).fetchone())
        start = con.execute("SELECT date('now', ?)", (since,)).fetchone()[0]
    total = {k: v or 0 for k, v in total.items()}  # SUM over no rows is NULL
    return {"days": days, "since": start, "total": total, "by_model": by_model, "by_day": by_day}


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
