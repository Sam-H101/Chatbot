"""Embedded metadata database: SQLite + FTS5. One file, no server.
Original uploads are kept as plain files next to it (see files_dir), and their thumbnails
in a folder of their own (thumbs_dir, thumbs.py)."""
import datetime
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading

DB_PATH = os.environ.get("DB_PATH", "data/metabase.db")
# Where original uploads are kept, one file per distinct content, named by its SHA-256.
# Empty means a "files" folder next to DB_PATH (/data/files in the container).
FILES_DIR = os.environ.get("FILES_DIR", "")
# Where thumbnails of PDFs and photos are kept (thumbs.py), named by the original's hash.
# Empty means a "thumbs" folder next to DB_PATH (/data/thumbs). Not backed up: they can
# always be made again from the originals.
THUMBS_DIR = os.environ.get("THUMBS_DIR", "")
# Serialises writing and removing stored files with the rows that point at them, so a
# delete never removes a file that an upload is about to reference (one gunicorn worker).
# Reentrant: the upload route holds it across its duplicate check and add_job.
files_lock = threading.RLock()
# Jobs that still need their file: waiting, running, or failed (and so retryable).
ACTIVE_JOBS = ("queued", "awaiting_confirmation", "running")
HOLDING_JOBS = ACTIVE_JOBS + ("failed",)
# The schema is brought up to date once per database per process, under this lock: the
# page's first requests arrive together, and two threads that both saw a column missing
# would both run ALTER TABLE, failing the slower request.
_schema_lock = threading.Lock()
_schema_ready = set()
# Moves after every committed change to chunks or their vectors: vectors.py keeps the
# vectors in memory and reloads them when it moves (one process, see the Dockerfile).
_index_version = 0
_version_lock = threading.Lock()
# Seconds a statement waits for another connection's write to finish (sqlite3's default is
# 5). Saving a large upload (every chunk, its FTS row and its vector in one transaction)
# can hold the database longer: 7 s for a 44 MB text, when polls and questions failed with
# "database is locked".
BUSY_SECONDS = 30

# AUTOINCREMENT: a deleted conversation's id is never given out again, so a tab or device
# that still has it open gets 404 rather than writing into someone's new chat. The table
# as first released lacked it and is copied into this one on connect (_keep_conversation_ids).
CONVERSATIONS = """CREATE TABLE IF NOT EXISTS {table} (  -- saved chats: questions and their answers (messages)
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL DEFAULT '',    -- its first question, cut short, until renamed
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),  -- its last question
    model      TEXT NOT NULL DEFAULT '',    -- the model choice of its last question: 'auto', 'ollama:<m>', 'claude:<m>'
    doc_ids    TEXT                         -- JSON: the documents its last question was limited to (deleted
                                            -- ones taken out, so [] when all were); NULL: every document
);"""

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
    analysis    TEXT NOT NULL DEFAULT '',  -- 'full', 'partial: <how much>' or 'none: <reason>'
                                           -- ('partial' alone: the start only, before sections.py)
    model       TEXT NOT NULL DEFAULT '',  -- model that read the document, e.g. 'Claude Opus 5.5'
    sha256      TEXT,                      -- see COLUMNS below; older databases gain these on connect
    file_path   TEXT,
    file_size   INTEGER,
    mime        TEXT,
    unit_findings TEXT                     -- JSON: units.check() of its chunks; NULL until checked
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
    doc_id             INTEGER,             -- no foreign key: the cost stays logged after a delete,
                                            -- with doc_id set to NULL (ids get reused)
    job_id             INTEGER              -- the upload or re-read job that made the call
);
CREATE INDEX IF NOT EXISTS usage_created ON usage(created_at);
CREATE TABLE IF NOT EXISTS jobs (           -- uploads and re-reads, worked through by jobs.py
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,               -- 'upload' or 'reread'
    status     TEXT NOT NULL DEFAULT 'queued',  -- queued, awaiting_confirmation, running, done, failed, cancelled
    filename   TEXT NOT NULL,
    sha256     TEXT NOT NULL,               -- the original, kept as files_dir()/<file_path>
    file_path  TEXT NOT NULL,
    file_size  INTEGER,
    mime       TEXT,
    model      TEXT NOT NULL,               -- the choice: 'auto', 'ollama:<model>' or 'claude:<model>'
    options    TEXT NOT NULL DEFAULT '{}',  -- JSON: on_duplicate, claude_pages, confirmed, ...
    progress   INTEGER NOT NULL DEFAULT 0,  -- pages read so far, then sections analyzed (long documents),
                                            -- then passages indexed for search by meaning
    total      INTEGER NOT NULL DEFAULT 0,  -- pages to read, then sections, then passages; 0 while unknown
    stage      TEXT NOT NULL DEFAULT '',    -- e.g. 'Reading page 40 of 312', 'Analyzing section 3 of 12 with …'
    error      TEXT,
    estimate   TEXT,                        -- JSON: what reading with Claude should cost
    result     TEXT,                        -- JSON, once done: model, title, chunks, ...
    partial    TEXT,                        -- JSON: pages a model already read, and notes on the sections
                                            -- of a long document, kept for a retry
    doc_id     INTEGER,                     -- the document added (upload) or re-read (reread)
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_sha256 ON jobs(sha256);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    text       TEXT NOT NULL,
    page_start INTEGER,                     -- pages of a PDF's chunk (extract.page_ranges); NULL for
    page_end   INTEGER                      -- other files and chunks indexed before pages were kept
);
CREATE TABLE IF NOT EXISTS chunk_vectors (  -- search by meaning (vectors.py): one embedding per chunk
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,                 -- the embedding model that made it (OLLAMA_EMBED_MODEL)
    vector   BLOB NOT NULL                  -- float32, little-endian
);
CREATE TABLE IF NOT EXISTS instructions (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    enabled    INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
{conversations}
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,          -- 'user' or 'assistant'; a question and its answer are added together
                                            -- when it is asked, the answer filled in when it ends
    content         TEXT NOT NULL,          -- what the page shows: the question, or the answer
    sent            TEXT,                   -- a question as the model got it, excerpts and all: what later questions
                                            -- send as history (answers are sent as shown)
    sources         TEXT,                   -- JSON, answers: sources, excerpts, overview, scope, scope_names, left_out
    model           TEXT,                   -- answers: the model that wrote it, 'ollama:<m>' or 'claude:<m>'
    model_label     TEXT,
    switched        TEXT,                   -- why Auto switched to Claude, if it did
    usage           TEXT,                   -- JSON: its tokens (input, output, cache_read, cache_write)
    status          TEXT NOT NULL DEFAULT 'done',  -- answers: 'answering' (being written), 'done', 'stopped' (by
                                                   -- the user), 'broken' (the model failed part way) or 'failed'
                                                   -- (no words at all); error says why for the last two
    error           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id, id);
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    filename, title, doc_type, summary, keywords, tokenize='porter unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, tokenize='porter unicode61'
);
CREATE TABLE IF NOT EXISTS folders (            -- documents.folder_id; nested through parent_id
    id         INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reused: chat scopes and open tabs hold folder ids
    parent_id  INTEGER,                         -- NULL = top level. No REFERENCES: kept valid under files_lock
    name       TEXT NOT NULL,                   -- as typed, trimmed, inner whitespace collapsed (clean_folder_name)
    name_key   TEXT NOT NULL,                   -- name.casefold(): names are unique per parent, ignoring case
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS folders_unique ON folders(IFNULL(parent_id, 0), name_key);
CREATE INDEX IF NOT EXISTS folders_parent ON folders(parent_id);
""".replace("{conversations}", CONVERSATIONS.format(table="conversations"))


# Columns added after the first release. Older databases gain them on connect; their
# rows keep NULL here (no stored file, no hash) and still list, search and answer.
COLUMNS = {
    "model": "TEXT NOT NULL DEFAULT ''",
    "sha256": "TEXT",      # hash of the uploaded bytes, for duplicate detection
    "file_path": "TEXT",   # stored original, relative to files_dir()
    "file_size": "INTEGER",
    "mime": "TEXT",
    "unit_findings": "TEXT",  # litre/gallon misprints; checked at upload, older documents by jobs.check_units
    "folder_id": "INTEGER",   # its folder (folders.id); NULL: Unfiled
    "filed_by": "TEXT",       # 'auto' when auto-filing put it in its folder; NULL otherwise (any move clears it)
    "suggested_folder": "TEXT",  # a folder path the analysis suggested ("Household/Bills"), while it is Unfiled
    # Key details read from it (jobs._details), or typed in by the user (user_edited):
    "document_date": "TEXT",  # the date it bears, YYYY-MM-DD
    "vendor": "TEXT",         # receipts, bills, invoices: who issued it
    "total_amount": "REAL",   # ... the total to pay
    "currency": "TEXT",       # ... its ISO 4217 code (GBP, EUR, USD)
    "due_date": "TEXT",       # bills, invoices: when payment is due, YYYY-MM-DD
    "user_edited": "TEXT",    # JSON list of the fields (EDITABLE) the user changed: re-reads and Replace keep them
}
USAGE_COLUMNS = {"job_id": "INTEGER"}  # usage rows from before jobs have none
CHUNK_COLUMNS = {"page_start": "INTEGER", "page_end": "INTEGER"}  # older chunks have no pages: cited by file name
# Added after conversations were copied into a table with AUTOINCREMENT (_keep_conversation_ids,
# whose column list stays as it was): so added after that copy.
CONVERSATION_COLUMNS = {"folder_ids": "TEXT"}  # JSON: the folders its last question was limited to; NULL: none
# Key details a reading fills in (jobs._details), and what the user may change of a document.
DETAILS = ("document_date", "vendor", "total_amount", "currency", "due_date")
EDITABLE = ("title", "doc_type", *DETAILS)


def parse_date(value):
    """A date as documents keep it (YYYY-MM-DD), or None if `value` isn't a real date written
    so, from 1900 to 2100. (date.fromisoformat alone also takes 20261103 and 2026-W45-2.)"""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return None
    try:
        day = datetime.date.fromisoformat(value.strip())
    except ValueError:
        return None
    return day.isoformat() if 1900 <= day.year <= 2100 else None
# A document's columns as the API lists it. Its unit check's findings stay here (they have
# their own route, unit_findings()): listings read only how many there are (_doc_columns).
_DOC_FIELDS = ("id", "filename", "uploaded_at", "char_count", "title", "doc_type", "summary", "keywords", "analysis",
               *(c for c in COLUMNS if c != "unit_findings"))


def connect():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    fresh = not os.path.exists(DB_PATH)
    con = sqlite3.connect(DB_PATH, timeout=BUSY_SECONDS)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    if fresh or DB_PATH not in _schema_ready:
        with _schema_lock:
            _migrate(con)
            _schema_ready.add(DB_PATH)
    return con


def lock_library(exclusive=False, wait=True):
    """An flock on app.lock next to DB_PATH, held until the returned file is closed or the
    process ends. The app holds it shared while it runs (gunicorn_conf.py); a restore takes
    it exclusively without waiting (backup.py), so it can't replace the database under a
    running app, which keeps some of it in memory: the schema it migrated, the vectors, the
    job it is reading. BlockingIOError: someone else holds it (wait=False)."""
    import fcntl  # Linux only, like gunicorn
    folder = os.path.dirname(DB_PATH) or "."
    os.makedirs(folder, exist_ok=True)
    f = open(os.path.join(folder, "app.lock"), "a")
    try:
        fcntl.flock(f, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | (0 if wait else fcntl.LOCK_NB))
    except BaseException:
        f.close()
        raise
    return f


def _migrate(con):
    con.executescript(SCHEMA)
    _add_columns(con, (("documents", COLUMNS), ("usage", USAGE_COLUMNS), ("chunks", CHUNK_COLUMNS)))
    con.execute("CREATE INDEX IF NOT EXISTS documents_sha256 ON documents(sha256)")
    con.execute("CREATE INDEX IF NOT EXISTS documents_folder ON documents(folder_id)")
    con.execute("CREATE INDEX IF NOT EXISTS usage_job ON usage(job_id)")
    _keep_conversation_ids(con)
    _add_columns(con, (("conversations", CONVERSATION_COLUMNS),))
    con.commit()


def _add_columns(con, tables):
    for table, columns in tables:
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in have:
                try:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                except sqlite3.OperationalError as e:  # another process added it first
                    if "duplicate column name" not in str(e):
                        raise


def _autoincrement(con):
    sql = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='conversations'").fetchone()[0]
    return "AUTOINCREMENT" in sql.upper()


def _keep_conversation_ids(con):
    """A conversations table as first released (no AUTOINCREMENT, so SQLite gave a deleted
    conversation's id to the next one) is copied into one that never does. Foreign keys are
    off meanwhile: dropping the old table must not take the messages with it."""
    if _autoincrement(con):
        return
    con.commit()
    con.execute("PRAGMA foreign_keys = OFF")
    try:
        con.execute("BEGIN IMMEDIATE")
        if not _autoincrement(con):  # another process may have done it while this one waited
            con.execute(CONVERSATIONS.format(table="conversations_new"))
            con.execute("INSERT INTO conversations_new (id, title, created_at, updated_at, model, doc_ids)"
                        " SELECT id, title, created_at, updated_at, model, doc_ids FROM conversations")
            con.execute("DROP TABLE conversations")
            con.execute("ALTER TABLE conversations_new RENAME TO conversations")
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.execute("PRAGMA foreign_keys = ON")


def _doc_columns(t):
    """What a listing selects of documents (as `t`): _DOC_FIELDS, and unit_fixes, how many
    misprints its unit check found (NULL: not checked yet). A check from before the total
    was kept counts its list."""
    return ", ".join(f"{t}.{c}" for c in _DOC_FIELDS) + (
        f", COALESCE(json_extract({t}.unit_findings, '$.total'), json_array_length({t}.unit_findings, '$.findings'))"
        " AS unit_fixes")


def _doc(row, folders):
    """A documents row (_doc_columns) for the API: the stored file's location stays internal.
    Computed: user_edited as a list; suggestion_exists, whether the suggested folder is one
    of `folders` (_Folders) already (None without a suggestion); kind (extract.kind); and
    thumb, its thumbnail's URL, versioned by the file's hash since document ids are reused
    (None for documents with no kept file, of another kind, or whose thumbnail failed)."""
    from . import extract, thumbs  # thumbs imports this module
    d = dict(row)
    d["has_file"] = bool(d.pop("file_path", None))
    d["user_edited"] = _edited(d.get("user_edited"))
    d["suggestion_exists"] = None if not d.get("suggested_folder") else folders.resolve(d["suggested_folder"]) is not None
    d["kind"] = extract.kind(d["filename"])
    sha = d.get("sha256")
    d["thumb"] = (f"/api/documents/{d['id']}/thumb?v={sha[:12]}"
                  if d["has_file"] and sha and d["kind"] in thumbs.KINDS and not thumbs.failed(sha) else None)
    return d


def _docs(con, rows):
    folders = _Folders(con)
    return [_doc(r, folders) for r in rows]


def _edited(raw):
    try:
        got = json.loads(raw) if raw else []
    except ValueError:
        return []
    return [k for k in EDITABLE if k in got] if isinstance(got, list) else []


# ---- original files ----

def files_dir():
    return FILES_DIR or os.path.join(os.path.dirname(DB_PATH) or ".", "files")


def thumbs_dir():
    return THUMBS_DIR or os.path.join(os.path.dirname(DB_PATH) or ".", "thumbs")


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
    """Remove a stored original, and its thumbnails."""
    from . import thumbs
    name = os.path.basename(file_path)
    try:
        os.remove(os.path.join(files_dir(), name))
    except FileNotFoundError:
        pass
    thumbs.remove(name)


def stored_file(doc_id):
    """The kept original of a document: {path, filename, mime, size, sha256}, or None
    for documents uploaded before originals were kept (or whose file has gone missing).
    Re-reading a document with another model starts from here."""
    with connect() as con:
        row = con.execute("SELECT filename, sha256, file_path, file_size, mime FROM documents WHERE id=?",
                          (doc_id,)).fetchone()
    if not row or not row["file_path"]:
        return None
    path = os.path.join(files_dir(), os.path.basename(row["file_path"]))
    if not os.path.isfile(path):
        return None
    return {"path": path, "filename": row["filename"], "mime": row["mime"] or "application/octet-stream",
            "size": row["file_size"], "sha256": row["sha256"]}


def document_exists(doc_id):
    with connect() as con:
        return con.execute("SELECT 1 FROM documents WHERE id=?", (doc_id,)).fetchone() is not None


def find_duplicates(sha):
    """Documents holding a file with this hash, newest first. Documents from before
    hashing have none, so they never match."""
    with connect() as con:
        return _docs(con, con.execute(
            f"SELECT {_doc_columns('d')} FROM documents d WHERE sha256=? ORDER BY id DESC", (sha,)))


class JobGone(Exception):
    """A job's result can't be saved: it was cancelled, or its document deleted, meanwhile."""


def _meta(meta):
    return (meta.get("title", ""), meta.get("doc_type", ""), meta.get("summary", ""),
            ", ".join(meta.get("keywords", [])))


def index_version():
    return _index_version


def _index_changed():
    """After a commit that changed chunks or vectors (before it, a search could reload the
    old rows and keep them as current)."""
    global _index_version
    with _version_lock:
        _index_version += 1


def _insert_chunks(con, doc_id, chunks, pages=None, vectors=None):
    """A document's chunks and their FTS rows. `pages`: (first, last) page of each chunk of
    a paged text (extract.page_ranges), None for text without pages. `vectors`: (embedding
    model, one vector per chunk) for search by meaning; None leaves them to vectors.backfill."""
    for i, text in enumerate(chunks):
        first, last = pages[i] if pages else (None, None)
        cid = con.execute("INSERT INTO chunks (doc_id, idx, text, page_start, page_end) VALUES (?,?,?,?,?)",
                          (doc_id, i, text, first, last)).lastrowid
        con.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?,?)", (cid, text))
        if vectors:
            con.execute("INSERT OR REPLACE INTO chunk_vectors (chunk_id, model, vector) VALUES (?,?,?)",
                        (cid, vectors[0], vectors[1][i]))


def _finish_job(con, job, doc_id):
    """Mark job = (id, result dict) done in the caller's transaction, if it is still
    running, and link its model calls (earlier failed attempts too) to the document."""
    job_id, result = job
    result["doc_id"] = doc_id
    if not con.execute(
            "UPDATE jobs SET status='done', stage='Done', error=NULL, partial=NULL, doc_id=?, result=?,"
            " updated_at=datetime('now') WHERE id=? AND status='running'",
            (doc_id, json.dumps(result), job_id)).rowcount:
        raise JobGone("The job was cancelled")
    con.execute("UPDATE usage SET doc_id=? WHERE job_id=?", (doc_id, job_id))


def _units(check):
    return None if check is None else json.dumps(check, ensure_ascii=False)


def _index_doc(con, doc_id):
    """(Re)write a document's docs_fts row from its documents row: its vendor is searchable
    with its keywords."""
    con.execute("DELETE FROM docs_fts WHERE rowid=?", (doc_id,))
    con.execute("INSERT INTO docs_fts (rowid, filename, title, doc_type, summary, keywords)"
                " SELECT id, filename, title, doc_type, summary, CASE WHEN IFNULL(vendor, '') = '' THEN keywords"
                " WHEN keywords = '' THEN vendor ELSE keywords || ', ' || vendor END FROM documents WHERE id=?",
                (doc_id,))


FOLDER_GONE_NOTE = "Its folder was deleted, so it is in Unfiled"


def add_document(filename, chunks, meta, analysis, model="", data=None, mime=None, sha=None, replace=False,
                 size=None, job=None, units=None, pages=None, vectors=None, details=None, folder_id=None,
                 auto_file=False, suggestion=None, confidence="low"):
    """Index a document and return its id. Its original is `data`, stored here under
    files_dir(), or one already stored under its hash `sha` (by an upload job) with
    `size` bytes; with neither, the document has no kept file. replace=True first
    deletes the documents that hold the same file, in the same transaction. `job` =
    (job id, result dict) marks that job done in the same transaction too, the result
    gaining doc_id and replaced; if the job was cancelled meanwhile, nothing is added.
    `units`: its unit check (units.check); None leaves it to jobs.check_units. `pages`
    and `vectors`: its chunks', as _insert_chunks takes them. `details`: its key details
    ({DETAILS}, jobs._details).

    Where it goes, decided here, in the transaction, so a folder deleted meanwhile can't
    be missed: replacing copies, the folder of the newest of them, with the details the
    user edited on it ("kept"); else `folder_id` if that folder still exists ("chosen"),
    Unfiled if not (with a note); else, uploaded into Unfiled with `auto_file` on and a
    "high" `confidence`, the existing folder the path `suggestion` names ("auto"; folders
    are never made here); else Unfiled with the suggestion kept for the user to accept.
    With `job`, folder_id and auto_file are those its row holds now, not the caller's: a
    folder deleted while the job was being read redirected it (_retarget_uploads). The
    job's result gains folder_id, filed, suggested_folder, suggestion_exists and
    folder_note."""
    if data is not None:
        sha, size = sha or file_hash(data), len(data)
    kept = sha is not None and size is not None
    title, doc_type, summary, keywords = _meta(meta)
    values = {"title": title, "doc_type": doc_type, **{k: (details or {}).get(k) for k in DETAILS}}
    with files_lock:
        wrote = False
        try:
            with connect() as con:
                folders = _Folders(con)
                following, edited, filed, note = [], [], None, None
                place = {"folder_id": None, "filed_by": None, "suggested_folder": None}
                old = []
                if replace and sha:
                    copies = con.execute(f"SELECT id, folder_id, filed_by, user_edited, {', '.join(EDITABLE)}"
                                         " FROM documents WHERE sha256=? ORDER BY id DESC", (sha,)).fetchall()
                    old = [r["id"] for r in copies]
                    following = _scoped_to(con, old)  # before _delete_rows takes them out
                    _delete_rows(con, old)  # the file stays: the new row points at it
                    if job:
                        job[1]["replaced"] = old
                    if copies:  # the newest copy's place, and what the user changed of it
                        newest = copies[0]
                        edited = _edited(newest["user_edited"])
                        values.update({k: newest[k] for k in edited})
                        place.update(folder_id=newest["folder_id"], filed_by=newest["filed_by"])
                        if newest["folder_id"] not in folders.rows:  # Unfiled: a re-read's rule, never moved
                            place.update(folder_id=None, filed_by=None, suggested_folder=suggestion)
                        filed = "kept"
                if job:
                    stored = con.execute("SELECT options FROM jobs WHERE id=?", (job[0],)).fetchone()
                    opts = json.loads(stored[0]) if stored and stored[0] else {}
                    folder_id, auto_file = opts.get("folder_id", folder_id), bool(opts.get("auto_file", auto_file))
                if not old:
                    if folder_id is not None:
                        if folder_id in folders.rows:
                            place["folder_id"], filed = folder_id, "chosen"
                        else:
                            place["suggested_folder"], note = suggestion, FOLDER_GONE_NOTE
                    else:
                        target = folders.resolve(suggestion) if auto_file and confidence == "high" and suggestion else None
                        if target is not None:
                            place.update(folder_id=target, filed_by="auto")
                            filed = "auto"
                        else:
                            place["suggested_folder"] = suggestion
                doc_id = con.execute(
                    "INSERT INTO documents (filename, char_count, title, doc_type, summary, keywords, analysis,"
                    " model, sha256, file_path, file_size, mime, unit_findings, folder_id, filed_by, suggested_folder,"
                    " document_date, vendor, total_amount, currency, due_date, user_edited)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (filename, sum(len(c) for c in chunks), values["title"], values["doc_type"], summary, keywords,
                     analysis, model, sha if kept else None, sha if kept else None, size if kept else None,
                     mime if kept else None, _units(units), place["folder_id"], place["filed_by"],
                     place["suggested_folder"], *(values[k] for k in DETAILS), json.dumps(edited) if edited else None),
                ).lastrowid
                _index_doc(con, doc_id)
                _insert_chunks(con, doc_id, chunks, pages, vectors)
                if job:
                    if "title" in edited:
                        job[1]["title"] = values["title"]
                    job[1].update(_placed(folders, place, filed, note))
                for conv_id, ids in following:  # conversations limited to a replaced copy ask the new one
                    con.execute("UPDATE conversations SET doc_ids=? WHERE id=?",
                                (_ids(dict.fromkeys(doc_id if i in old else i for i in ids)), conv_id))
                if job:
                    _finish_job(con, job, doc_id)
                if data is not None:  # last, so a failed insert leaves no file behind
                    wrote = _store(sha, data)
        except BaseException:
            if wrote:  # written for rows that never committed
                _remove(sha)
            raise
    _index_changed()
    return doc_id


def _placed(folders, place, filed, note):
    """What a job's result says of where its document went."""
    suggested = place["suggested_folder"]
    return {"folder_id": place["folder_id"], "filed": filed, "suggested_folder": suggested,
            "suggestion_exists": None if not suggested else folders.resolve(suggested) is not None, "folder_note": note}


def replace_document(doc_id, chunks, meta, analysis, model, job=None, units=None, pages=None, vectors=None,
                     details=None, suggestion=None):
    """A re-read: new text, chunks, metadata, key details, model and unit check for a
    document, keeping its id, file, upload date and folder (a re-read never moves it), and
    the fields the user edited (user_edited). Its folder `suggestion` is kept only while it
    is Unfiled. `job`, `units`, `pages`, `vectors` and `details` as in add_document; the
    old chunks' vectors go with them (ON DELETE CASCADE)."""
    title, doc_type, summary, keywords = _meta(meta)
    new = {"title": title, "doc_type": doc_type, **{k: (details or {}).get(k) for k in DETAILS}}
    with connect() as con:
        row = con.execute(f"SELECT folder_id, user_edited, suggested_folder, {', '.join(EDITABLE)} FROM documents"
                          " WHERE id=?", (doc_id,)).fetchone()
        if not row:
            raise JobGone("The document was deleted while it was being read")
        folders = _Folders(con)
        edited = _edited(row["user_edited"])
        values = {k: row[k] if k in edited else new[k] for k in EDITABLE}
        unfiled = row["folder_id"] not in folders.rows
        place = {"folder_id": row["folder_id"], "suggested_folder": suggestion if unfiled else row["suggested_folder"]}
        con.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id=?)", (doc_id,))
        con.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        con.execute("UPDATE documents SET char_count=?, title=?, doc_type=?, summary=?, keywords=?, analysis=?, model=?,"
                    " unit_findings=?, suggested_folder=?, document_date=?, vendor=?, total_amount=?, currency=?,"
                    " due_date=? WHERE id=?",
                    (sum(len(c) for c in chunks), values["title"], values["doc_type"], summary, keywords, analysis,
                     model, _units(units), place["suggested_folder"], *(values[k] for k in DETAILS), doc_id))
        _index_doc(con, doc_id)
        _insert_chunks(con, doc_id, chunks, pages, vectors)
        if job:
            if "title" in edited:
                job[1]["title"] = values["title"]
            job[1].update(_placed(folders, place, "kept", None))
            _finish_job(con, job, doc_id)
    _index_changed()


def list_documents(doc_ids=None):
    """Every document, newest first; only those among `doc_ids` (folder_documents), if given."""
    scope, args = _only(doc_ids)
    with connect() as con:
        return _docs(con, con.execute(f"SELECT {_doc_columns('d')} FROM documents d WHERE 1{scope} ORDER BY id DESC",
                                      args))


def _only(doc_ids, column="d.id"):
    """A filter on `column` for documents among `doc_ids` (None: all of them), as _among."""
    if doc_ids is None:
        return "", ()
    return f" AND {column} IN (SELECT value FROM json_each(?))", (json.dumps(list(doc_ids)),)


def get_document(doc_id):
    """A document as listings have it, plus `pages` (the last page its chunks reach, None for
    text without pages) and `copies` ([{id, folder_id}] of other documents holding the same
    file). None if there is no such document."""
    with connect() as con:
        row = con.execute(f"SELECT {_doc_columns('d')} FROM documents d WHERE d.id=?", (doc_id,)).fetchone()
        if not row:
            return None
        doc = _docs(con, [row])[0]
        doc["pages"] = con.execute("SELECT MAX(page_end) FROM chunks WHERE doc_id=?", (doc_id,)).fetchone()[0]
        doc["copies"] = [dict(r) for r in con.execute(
            "SELECT id, folder_id FROM documents WHERE sha256=? AND id != ? ORDER BY id DESC",
            (doc["sha256"], doc_id))] if doc["sha256"] else []
    return doc


def document_text(doc_id):
    """A document's text as indexed: its chunks, in order (page breaks kept as "\\f")."""
    with connect() as con:
        return "\n\n".join(r[0] for r in con.execute("SELECT text FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,)))


def update_document(doc_id, fields, suggestion=None):
    """Change a document: `fields` (checked by the caller) of EDITABLE, each added to its
    user_edited so re-reads keep it, and folder_id (a move: Unfiled when None). `suggestion`:
    "dismiss" forgets its suggested folder; "apply" moves it there, making any folders on the
    path that are missing (the user's click is the consent). Returns (the document, ids of
    folders made), or None if there is no such document. MissingFolder: folder_id names none;
    NoSuggestion: "apply" with none to apply."""
    with files_lock:
        with connect() as con:
            row = con.execute("SELECT user_edited, suggested_folder FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                return None
            sets, created = {}, []
            edited = set(_edited(row["user_edited"]))
            for k in EDITABLE:
                if k in fields:
                    sets[k] = fields[k]
                    edited.add(k)
            if fields.keys() & set(EDITABLE):
                sets["user_edited"] = json.dumps([k for k in EDITABLE if k in edited])
            if "folder_id" in fields:
                if fields["folder_id"] is not None and not _folder_exists(con, fields["folder_id"]):
                    raise MissingFolder()
                sets.update(folder_id=fields["folder_id"], filed_by=None, suggested_folder=None)
            if suggestion == "dismiss":
                sets["suggested_folder"] = None
            elif suggestion == "apply":
                if not row["suggested_folder"]:
                    raise NoSuggestion("There is no suggestion to apply")
                folder_id, created = _ensure_path(con, row["suggested_folder"].split("/"))
                sets.update(folder_id=folder_id, filed_by=None, suggested_folder=None)
            if sets:
                con.execute(f"UPDATE documents SET {', '.join(f'{k}=?' for k in sets)} WHERE id=?",
                            (*sets.values(), doc_id))
            if sets.keys() & {"title", "doc_type", "vendor"}:
                _index_doc(con, doc_id)
            row = con.execute(f"SELECT {_doc_columns('d')} FROM documents d WHERE d.id=?", (doc_id,)).fetchone()
            return _docs(con, [row])[0], created


# ---- unit checks: litre/gallon misprints found in a document (units.check) ----
# Stored as {"version", "paged", "total", "findings"} (checks from before "paged" and
# "total" have just the list); a check made by another version of units.py is stale.

_STALE = "(unit_findings IS NULL OR json_extract(unit_findings, '$.version') IS NOT ?)"


def unit_findings(doc_id, version):
    """A document's unit check: {id, filename, title, has_file, pdf, findings, total},
    findings (at most units.MAX_FINDINGS of them, total counting all) None while it has no
    check from this `version`; None if there is no such document."""
    with connect() as con:
        row = con.execute(f"SELECT id, filename, title, file_path, mime, unit_findings, {_STALE} AS stale"
                          " FROM documents WHERE id=?", (version, doc_id)).fetchone()
    if not row:
        return None
    check = None if row["stale"] else json.loads(row["unit_findings"])
    return {"id": row["id"], "filename": row["filename"], "title": row["title"], "has_file": bool(row["file_path"]),
            "pdf": bool(row["file_path"]) and row["mime"] == "application/pdf",
            "findings": check and check["findings"], "total": check and check.get("total", len(check["findings"]))}


def unchecked_documents(version):
    """Documents with no unit check from this version: indexed before checks were made,
    or checked by an earlier units.py."""
    with connect() as con:
        return [r[0] for r in con.execute(f"SELECT id FROM documents WHERE {_STALE} ORDER BY id", (version,))]


def unit_source(doc_id):
    """What a unit check of a document reads: its chunks, and whether its last check was
    of a PDF's text (paged), which chunks without page breaks can't tell."""
    with connect() as con:
        row = con.execute("SELECT json_extract(unit_findings, '$.paged') FROM documents WHERE id=?",
                          (doc_id,)).fetchone()
        chunks = [r[0] for r in con.execute("SELECT text FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,))]
    return chunks, bool(row and row[0])


def set_unit_findings(doc_id, check):
    """Store a unit check, unless the document has one from this version by now (say a
    re-read just stored its own, for new chunks)."""
    with connect() as con:
        con.execute(f"UPDATE documents SET unit_findings=? WHERE id=? AND {_STALE}",
                    (_units(check), doc_id, check["version"]))


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
        # A re-read of a deleted document has nothing left to update; the pages it read go too.
        con.execute(f"UPDATE jobs SET status='cancelled', stage='The document was deleted', partial=NULL,"
                    f" updated_at=datetime('now') WHERE kind='reread' AND doc_id IN ({marks})"
                    f" AND status IN ({','.join('?' * len(HOLDING_JOBS))})", [*part, *HOLDING_JOBS])
        con.execute(f"UPDATE jobs SET doc_id = NULL WHERE doc_id IN ({marks})", part)
        # Conversations limited to it forget it: its id may go to the next upload.
        con.execute(f"UPDATE conversations SET doc_ids = (SELECT json_group_array(value) FROM json_each(doc_ids)"
                    f" WHERE value NOT IN ({marks})) WHERE doc_ids IS NOT NULL AND EXISTS"
                    f" (SELECT 1 FROM json_each(doc_ids) WHERE value IN ({marks}))", [*part, *part])
        deleted += con.execute(f"DELETE FROM documents WHERE id IN ({marks})", part).rowcount
    return deleted, files


def _scoped_to(con, ids):
    """[(id, doc_ids)] of the conversations limited to any of these documents."""
    return [(r[0], json.loads(r[1])) for r in con.execute(
        "SELECT id, doc_ids FROM conversations WHERE doc_ids IS NOT NULL AND EXISTS (SELECT 1 FROM json_each(doc_ids)"
        " WHERE value IN (SELECT value FROM json_each(?)))", (json.dumps(list(ids)),))]


def _unused(con, file_path):
    """No document and no job that may still read it uses this stored file."""
    return not con.execute(
        f"SELECT 1 FROM documents WHERE file_path=? UNION ALL SELECT 1 FROM jobs WHERE file_path=?"
        f" AND status IN ({','.join('?' * len(HOLDING_JOBS))}) LIMIT 1",
        (file_path, file_path, *HOLDING_JOBS)).fetchone()


def delete_documents(ids):
    """Delete several documents in one transaction, and the stored files nothing else
    still uses. Returns how many existed."""
    ids = [i for i in dict.fromkeys(int(i) for i in ids) if -2**63 <= i < 2**63]  # others can't exist
    with files_lock:
        with connect() as con:
            deleted, files = _delete_rows(con, ids)
            unused = [p for p in files if _unused(con, p)]
        _index_changed()  # their chunks' vectors went with them
        for p in unused:  # only after the commit: a rollback must not lose a file
            _remove(p)
    return deleted


# ---- folders: documents.folder_id, nested through folders.parent_id ----
# Every change is made under files_lock in one transaction, so the tree stays a tree (no
# folder inside itself, no document in a deleted folder) without foreign keys.

FOLDER_NAME_CHARS = 100
# Levels of folders, the top level counting as one: each listed folder carries its whole path,
# so an unbounded depth (one request with a path of thousands of levels) would make the folder
# listing grow with its square.
FOLDER_DEPTH = 20
_KEEP = object()  # update_folder: leave as it is


class FolderError(Exception):
    """A folder change that can't be made; the message is for the user."""


class BadFolderName(FolderError, ValueError):
    pass


class MissingFolder(FolderError, LookupError):
    def __init__(self, message="That folder was deleted"):
        super().__init__(message)


class FolderCycle(FolderError):
    def __init__(self, message="A folder can't go inside itself or one of its own folders"):
        super().__init__(message)


class TooDeep(FolderError):
    def __init__(self, message=f"Folders go {FOLDER_DEPTH} levels deep at most"):
        super().__init__(message)


class FolderConflict(FolderError):
    """A folder of that name (ignoring case) is already there: `existing` {id, name}."""
    def __init__(self, message, existing):
        super().__init__(message)
        self.existing = existing


class FolderChanged(FolderError):
    """A delete of a folder and its documents whose count of documents was not the one confirmed."""
    def __init__(self, counts):
        super().__init__(f"The folder changed: it now holds {counts['documents']} "
                         f"document{'' if counts['documents'] == 1 else 's'}")
        self.counts = counts


class NoSuggestion(FolderError):
    pass


def clean_folder_name(raw):
    """A folder name as kept: trimmed, inner whitespace collapsed. BadFolderName when it is
    empty, too long, or has a / (paths are written with it) or control characters."""
    name = " ".join(raw.split()) if isinstance(raw, str) else ""
    if not name:
        raise BadFolderName("A folder needs a name")
    if len(name) > FOLDER_NAME_CHARS:
        raise BadFolderName(f"Folder names are {FOLDER_NAME_CHARS} characters at most")
    if "/" in name:
        raise BadFolderName("Folder names can't contain /")
    if re.search(r"[\x00-\x1f\x7f]", name):
        raise BadFolderName("Folder names can't contain control characters")
    return name


def clean_path(raw, most=None):
    """A folder path ("Household/Bills") as a list of clean names. BadFolderName for any bad
    one, or for more than `most` levels."""
    segments = [clean_folder_name(s) for s in (raw.split("/") if isinstance(raw, str) else [None])]
    if most and len(segments) > most:
        raise BadFolderName(f"A folder path has {most} levels at most")
    return segments


class _Folders:
    """Every folder, from one query: by id, with their paths and lookups by name."""

    def __init__(self, con):
        self.rows = {r["id"]: dict(r) for r in con.execute(
            "SELECT id, parent_id, name, name_key, created_at, updated_at FROM folders")}
        self.by_key = {(r["parent_id"] or 0, r["name_key"]): i for i, r in self.rows.items()}

    def path(self, folder_id):
        """Names from the top level down to this folder."""
        names, seen = [], set()
        while folder_id in self.rows and folder_id not in seen:  # seen: a damaged tree can't loop forever
            seen.add(folder_id)
            names.append(self.rows[folder_id]["name"])
            folder_id = self.rows[folder_id]["parent_id"]
        return names[::-1]

    def resolve(self, path):
        """The id of the existing folder a path names (each level matched ignoring case), or None."""
        parent = 0
        for name in path.split("/") if isinstance(path, str) else path:
            parent = self.by_key.get((parent, " ".join(name.split()).casefold()))
            if parent is None:
                return None
        return parent or None


def _folder_exists(con, folder_id):
    return isinstance(folder_id, int) and con.execute("SELECT 1 FROM folders WHERE id=?", (folder_id,)).fetchone()


def _folder_list(con, only=None):
    """Folders as the API lists them, sorted by path: their path, how many documents each holds
    (doc_count) and holds with its subfolders' (total_count)."""
    folders = _Folders(con)
    direct = dict(con.execute("SELECT folder_id, count(*) FROM documents WHERE folder_id IS NOT NULL GROUP BY folder_id"))
    total = dict.fromkeys(folders.rows, 0)
    for folder_id, n in direct.items():
        seen = set()
        while folder_id in folders.rows and folder_id not in seen:
            seen.add(folder_id)
            total[folder_id] += n
            folder_id = folders.rows[folder_id]["parent_id"]
    out = []
    for i, r in folders.rows.items():
        if only is None or i == only:
            out.append({"id": i, "name": r["name"], "parent_id": r["parent_id"], "path": folders.path(i),
                        "created_at": r["created_at"], "updated_at": r["updated_at"],
                        "doc_count": direct.get(i, 0), "total_count": total[i]})
    return sorted(out, key=lambda f: ([n.casefold() for n in f["path"]], f["id"]))


def list_folders():
    """[{id, name, parent_id, path, created_at, updated_at, doc_count, total_count}], by path."""
    with connect() as con:
        return _folder_list(con)


def get_folder(folder_id):
    with connect() as con:
        return next(iter(_folder_list(con, folder_id)), None)


def folder_paths():
    """Every folder's path, levels joined by / ("Tractor/Manuals"), sorted: what an analysis
    may file a document into."""
    with connect() as con:
        folders = _Folders(con)
    return sorted(("/".join(folders.path(i)) for i in folders.rows), key=str.casefold)


def _conflict(con, parent_id, name, but=None):
    """FolderConflict if `parent_id` (None: the top level) already holds a folder named `name`."""
    row = con.execute("SELECT id, name FROM folders WHERE IFNULL(parent_id, 0)=? AND name_key=? AND id IS NOT ?",
                      (parent_id or 0, name.casefold(), but)).fetchone()
    if row:
        raise _conflict_error(con, parent_id, row)


def _conflict_error(con, parent_id, row):
    where = con.execute("SELECT name FROM folders WHERE id=?", (parent_id,)).fetchone() if parent_id else None
    message = (f"{where[0]} already has a folder named \"{row['name']}\"" if where
               else f"There is already a folder named \"{row['name']}\"")
    return FolderConflict(message, {"id": row["id"], "name": row["name"]})


def _insert_folder(con, name, parent_id):
    try:
        return con.execute("INSERT INTO folders (parent_id, name, name_key) VALUES (?,?,?)",
                           (parent_id, name, name.casefold())).lastrowid
    except sqlite3.IntegrityError:  # the unique index: a clash _conflict missed can't get past it
        _conflict(con, parent_id, name)
        raise


def add_folder(name, parent_id=None):
    """Make a folder in `parent_id` (None: the top level) and return it. BadFolderName,
    MissingFolder (the parent), TooDeep, FolderConflict."""
    name = clean_folder_name(name)
    with files_lock:
        with connect() as con:
            if parent_id is not None:
                if not _folder_exists(con, parent_id):
                    raise MissingFolder()
                if len(_ancestors(con, parent_id)) >= FOLDER_DEPTH:
                    raise TooDeep()
            _conflict(con, parent_id, name)
            folder_id = _insert_folder(con, name, parent_id)
            return _folder_list(con, folder_id)[0]


def _ancestors(con, folder_id):
    """This folder and every folder above it."""
    return {r[0] for r in con.execute(
        "WITH RECURSIVE up(id) AS (SELECT ? UNION SELECT f.parent_id FROM folders f JOIN up ON f.id = up.id"
        " WHERE f.parent_id IS NOT NULL) SELECT id FROM up", (folder_id,))}


def _height(con, folder_id):
    """Levels of folders from this one down to its deepest, itself counting as one (counted
    no further than one past FOLDER_DEPTH)."""
    return con.execute(
        "WITH RECURSIVE sub(id, depth) AS (SELECT ?, 1 UNION SELECT f.id, sub.depth + 1 FROM folders f"
        " JOIN sub ON f.parent_id = sub.id WHERE sub.depth <= ?) SELECT max(depth) FROM sub",
        (folder_id, FOLDER_DEPTH)).fetchone()[0]


def update_folder(folder_id, name=_KEEP, parent_id=_KEEP):
    """Rename a folder and/or move it into `parent_id` (None: the top level), and return it.
    A rename that only changes case is allowed. MissingFolder (it, or the new parent),
    FolderCycle (into itself or one of its own folders), TooDeep (its deepest folder would
    be more than FOLDER_DEPTH levels down), BadFolderName, FolderConflict."""
    with files_lock:
        with connect() as con:
            row = con.execute("SELECT name, parent_id FROM folders WHERE id=?", (folder_id,)).fetchone()
            if not row:
                raise MissingFolder()
            name = row["name"] if name is _KEEP else clean_folder_name(name)
            parent_id = row["parent_id"] if parent_id is _KEEP else parent_id
            if parent_id is not None:
                if not _folder_exists(con, parent_id):
                    raise MissingFolder()
                above = _ancestors(con, parent_id)
                if folder_id in above:
                    raise FolderCycle()
                if parent_id != row["parent_id"] and len(above) + _height(con, folder_id) > FOLDER_DEPTH:
                    raise TooDeep()
            _conflict(con, parent_id, name, but=folder_id)
            if (name, parent_id) != (row["name"], row["parent_id"]):
                try:
                    con.execute("UPDATE folders SET name=?, name_key=?, parent_id=?, updated_at=datetime('now')"
                                " WHERE id=?", (name, name.casefold(), parent_id, folder_id))
                except sqlite3.IntegrityError:
                    _conflict(con, parent_id, name, but=folder_id)
                    raise
            return _folder_list(con, folder_id)[0]


def subtree(con, ids):
    """Ids of these folders (those that exist) and of every folder inside them, at any depth."""
    return [r[0] for r in con.execute(
        "WITH RECURSIVE sub(id) AS (SELECT id FROM folders WHERE id IN (SELECT value FROM json_each(?))"
        " UNION SELECT f.id FROM folders f JOIN sub ON f.parent_id = sub.id) SELECT id FROM sub",
        (json.dumps(list(ids)),))]


def _folder_docs(con, ids, recursive=True):
    """Ids of the documents in these folders ("unfiled": in none), and in their subfolders."""
    unfiled = "unfiled" in ids
    ids = [i for i in ids if i != "unfiled"]
    found = subtree(con, ids) if recursive else ids
    where = "folder_id IN (SELECT value FROM json_each(?))" + (" OR folder_id IS NULL" if unfiled else "")
    return [r[0] for r in con.execute(f"SELECT id FROM documents WHERE {where} ORDER BY id DESC", (json.dumps(found),))]


def folder_documents(folder_ids, recursive=True):
    """Ids of the documents in these folders, and with `recursive` in every folder inside
    them; "unfiled" stands for the documents in no folder. MissingFolder for an id that
    names no folder."""
    with connect() as con:
        for i in folder_ids:
            if i != "unfiled" and not _folder_exists(con, i):
                raise MissingFolder()
        return _folder_docs(con, folder_ids, recursive)


def scope_folders(folder_ids):
    """The folders among these that exist, as a question limited to them searches them now:
    [{id, name, count, doc_ids}], the documents in each folder and its subfolders, looked up
    at the time of asking, so documents added since the folder was chosen are included."""
    with connect() as con:
        folders = _Folders(con)
        out = []
        for i in dict.fromkeys(folder_ids):
            if i in folders.rows:
                docs = _folder_docs(con, [i])
                out.append({"id": i, "name": folders.rows[i]["name"], "count": len(docs), "doc_ids": docs})
    return out


def _ensure_path(con, segments):
    """Walk a folder path from the top level, matching each level ignoring case, and make the
    folders missing from it. Returns (its folder id, ids of the folders made). TooDeep for
    a path of more than FOLDER_DEPTH levels."""
    if len(segments) > FOLDER_DEPTH:
        raise TooDeep()
    parent, created = None, []
    for raw in segments:
        name = clean_folder_name(raw)
        row = con.execute("SELECT id FROM folders WHERE IFNULL(parent_id, 0)=? AND name_key=?",
                          (parent or 0, name.casefold())).fetchone()
        if row:
            parent = row[0]
        else:
            parent = _insert_folder(con, name, parent)
            created.append(parent)
    if parent is None:
        raise BadFolderName("A folder needs a name")
    return parent, created


def ensure_path(segments):
    """_ensure_path, in a transaction of its own."""
    with files_lock:
        with connect() as con:
            return _ensure_path(con, segments)


def move_documents(ids, folder_id=None, path=None):
    """Move documents into a folder (None: Unfiled), or into the folder at `path` (a list of
    names), making the folders missing from it. Their filed_by and suggestion are cleared:
    the user put them there. Ids that name no document are skipped. Returns {moved,
    folder_id, created: [folder ids made], from: {document id: the folder it was in}}, which
    is what an Undo needs. MissingFolder: folder_id names none."""
    ids = [i for i in dict.fromkeys(ids) if -2**63 <= i < 2**63]
    with files_lock:
        with connect() as con:
            created = []
            if path is not None:
                folder_id, created = _ensure_path(con, path)
            elif folder_id is not None and not _folder_exists(con, folder_id):
                raise MissingFolder()
            arg = json.dumps(ids)
            before = dict(con.execute("SELECT id, folder_id FROM documents WHERE id IN (SELECT value FROM json_each(?))",
                                      (arg,)).fetchall())
            con.execute("UPDATE documents SET folder_id=?, filed_by=NULL, suggested_folder=NULL"
                        " WHERE id IN (SELECT value FROM json_each(?))", (folder_id, arg))
    return {"moved": len(before), "folder_id": folder_id, "created": created,
            "from": {i: before[i] for i in ids if i in before}}


def _free_name(con, parent_id, name):
    """`name`, or "name (2)", "name (3)" ... the lowest that `parent_id` doesn't hold yet."""
    n = 2
    while True:
        suffix = f" ({n})"
        candidate = name[:FOLDER_NAME_CHARS - len(suffix)].rstrip() + suffix
        if not con.execute("SELECT 1 FROM folders WHERE IFNULL(parent_id, 0)=? AND name_key=?",
                           (parent_id or 0, candidate.casefold())).fetchone():
            return candidate
        n += 1


def _forget_folders(con, ids):
    """Conversations limited to these folders forget them, as they forget deleted documents;
    uploads on their way into them are told to go to Unfiled instead."""
    arg = json.dumps(list(ids))
    con.execute("UPDATE conversations SET folder_ids = (SELECT json_group_array(value) FROM json_each(folder_ids)"
                " WHERE value NOT IN (SELECT value FROM json_each(?))) WHERE folder_ids IS NOT NULL AND EXISTS"
                " (SELECT 1 FROM json_each(folder_ids) WHERE value IN (SELECT value FROM json_each(?)))", (arg, arg))


def _retarget_uploads(con, ids, folder_id):
    """Uploads not saved yet (queued, waiting, running, or failed and so retryable) whose
    destination is one of these folders go to `folder_id` instead (None: Unfiled). They
    were made into a folder, so auto-filing is turned off for them: one sent on to Unfiled
    stays there, as the delete dialog says, rather than being filed somewhere else."""
    marks = ",".join("?" * len(HOLDING_JOBS))
    return con.execute(
        f"UPDATE jobs SET options=json_set(options, '$.folder_id', ?, '$.auto_file', json('false')),"
        f" updated_at=datetime('now')"
        f" WHERE kind='upload' AND status IN ({marks}) AND json_extract(options, '$.folder_id')"
        f" IN (SELECT value FROM json_each(?))", (folder_id, *HOLDING_JOBS, json.dumps(list(ids)))).rowcount


def delete_folder(folder_id, contents="move", confirm_documents=None):
    """Delete a folder. contents="move": its documents go to its parent (Unfiled at the top
    level) and its folders too, each renamed "Name (2)" and so on when the parent already
    has one of that name. contents="delete": it, every folder inside it, and every document
    in them go, but only when `confirm_documents` is how many documents that is now (none
    needs no confirmation): FolderChanged with the counts otherwise. Either way, chats
    forget it, and uploads on their way into it go to the parent (move) or Unfiled (delete).
    Files no document or job uses any more are removed after the commit. Returns
    {deleted_folders, moved_documents, moved_folders, deleted_documents, renamed: [{id,
    from, to}], parent_id}. MissingFolder if there is no such folder."""
    out = {"deleted_folders": 0, "moved_documents": 0, "moved_folders": 0, "deleted_documents": 0, "renamed": []}
    unused = []
    with files_lock:
        with connect() as con:
            row = con.execute("SELECT parent_id FROM folders WHERE id=?", (folder_id,)).fetchone()
            if not row:
                raise MissingFolder()
            parent = out["parent_id"] = row["parent_id"]
            if contents == "delete":
                gone = subtree(con, [folder_id])
                docs = _folder_docs(con, gone, recursive=False)
                if docs and confirm_documents != len(docs):
                    raise FolderChanged({"documents": len(docs), "folders": len(gone) - 1})
                out["deleted_documents"], files = _delete_rows(con, docs)
                con.execute("DELETE FROM folders WHERE id IN (SELECT value FROM json_each(?))", (json.dumps(gone),))
                out["deleted_folders"] = len(gone)
                _retarget_uploads(con, gone, None)
                _forget_folders(con, gone)
                unused = [p for p in files if _unused(con, p)]
            else:
                children = con.execute("SELECT id, name FROM folders WHERE parent_id=? ORDER BY id",
                                       (folder_id,)).fetchall()
                con.execute("DELETE FROM folders WHERE id=?", (folder_id,))  # first: a child may share its name
                for child in children:
                    name = child["name"]
                    if con.execute("SELECT 1 FROM folders WHERE IFNULL(parent_id, 0)=? AND name_key=?",
                                   (parent or 0, name.casefold())).fetchone():
                        name = _free_name(con, parent, name)
                        out["renamed"].append({"id": child["id"], "from": child["name"], "to": name})
                    con.execute("UPDATE folders SET parent_id=?, name=?, name_key=?, updated_at=datetime('now')"
                                " WHERE id=?", (parent, name, name.casefold(), child["id"]))
                out["moved_folders"] = len(children)
                out["moved_documents"] = con.execute(
                    "UPDATE documents SET folder_id=?, filed_by=NULL WHERE folder_id=?", (parent, folder_id)).rowcount
                out["deleted_folders"] = 1
                _retarget_uploads(con, [folder_id], parent)
                _forget_folders(con, [folder_id])
        if contents == "delete":
            _index_changed()  # the documents' chunks' vectors went with them
            for p in unused:  # only after the commit, as in delete_documents
                _remove(p)
    return out


# ---- jobs: uploads and re-reads waiting for, or in, the background worker ----

_JSON = ("options", "estimate", "result", "partial", "kept")


def _job(row):
    j = dict(row)
    for k in _JSON:
        if k in j:
            j[k] = json.loads(j[k]) if j[k] else None
    return j


def add_job(kind, filename, sha, model, options, data=None, size=None, mime=None, doc_id=None, on_insert=None):
    """Queue a job and return its id. An upload passes its `data`, stored here as the
    original (once per content); a re-read uses its document's stored file, `sha`.
    on_insert(id) runs before the job is committed, so before the worker can see it."""
    if data is not None:
        size = len(data)
    with files_lock:
        wrote = False
        try:
            with connect() as con:
                job_id = con.execute(
                    "INSERT INTO jobs (kind, filename, sha256, file_path, file_size, mime, model, options, doc_id,"
                    " stage) VALUES (?,?,?,?,?,?,?,?,?, 'Waiting to start')",
                    (kind, filename, sha, sha, size, mime, model, json.dumps(options), doc_id)).lastrowid
                if on_insert:
                    on_insert(job_id)
                if data is not None:  # last, as in add_document
                    wrote = _store(sha, data)
        except BaseException:
            if wrote:
                _remove(sha)
            raise
    return job_id


def get_job(job_id):
    with connect() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _job(row) if row else None


def list_jobs(hours=1):
    """Jobs still to finish, failed ones (until dismissed), and any others that changed
    in the last `hours`; newest first. The page polls this, so a failed job's kept pages
    are counted here, per reader ("kept": {reader: pages}), rather than loaded."""
    cols = "id, kind, status, filename, sha256, file_size, mime, model, options, progress, total, stage, " \
           "error, estimate, result, doc_id, created_at, updated_at, CASE WHEN status='failed' THEN " \
           "(SELECT json_group_object(r.key, (SELECT count(*) FROM json_each(r.value))) FROM json_each(partial) AS r)" \
           " END AS kept"
    marks = ",".join("?" * len(HOLDING_JOBS))
    with connect() as con:
        return [_job(r) for r in con.execute(
            f"SELECT {cols} FROM jobs WHERE status IN ({marks}) OR updated_at >= datetime('now', ?)"
            " ORDER BY id DESC LIMIT 200", (*HOLDING_JOBS, f"-{hours} hours"))]


def active_jobs(sha=None, doc_id=None):
    """Unfinished upload jobs for this file, or re-read jobs for this document."""
    marks = ",".join("?" * len(ACTIVE_JOBS))
    where, arg = ("kind='upload' AND sha256=?", sha) if sha else ("kind='reread' AND doc_id=?", doc_id)
    with connect() as con:
        return [_job(r) for r in con.execute(
            f"SELECT * FROM jobs WHERE {where} AND status IN ({marks}) ORDER BY id", (arg, *ACTIVE_JOBS))]


def claim_next_job():
    """The oldest queued job, now marked running; None if nothing is queued."""
    with connect() as con:
        rows = con.execute(
            "UPDATE jobs SET status='running', stage='Starting', updated_at=datetime('now') WHERE id ="
            " (SELECT id FROM jobs WHERE status='queued' ORDER BY id LIMIT 1) RETURNING *").fetchall()
    return _job(rows[0]) if rows else None


def update_job(job_id, when=("running",), **fields):
    """Set fields of a job whose status is one of `when`. False if it isn't (say it
    was cancelled meanwhile). Dicts are stored as JSON."""
    fields = {k: json.dumps(v) if k in _JSON and v is not None else v for k, v in fields.items()}
    sets = "".join(f"{k}=?, " for k in fields)
    marks = ",".join("?" * len(when))
    with connect() as con:
        return con.execute(f"UPDATE jobs SET {sets}updated_at=datetime('now') WHERE id=? AND status IN ({marks})",
                           (*fields.values(), job_id, *when)).rowcount > 0


def set_job_options(job_id, when=("running",), **values):
    """Set these keys of a job's options, leaving the others as the row holds them now (a
    folder delete may have changed its destination since the job was claimed). False if
    its status isn't one of `when`."""
    sets = "".join(", ?, json(?)" for _ in values)
    marks = ",".join("?" * len(when))
    args = [x for k, v in values.items() for x in (f"$.{k}", json.dumps(v))]
    with connect() as con:
        return con.execute(f"UPDATE jobs SET options=json_set(IFNULL(options, '{{}}'){sets}), updated_at=datetime('now')"
                           f" WHERE id=? AND status IN ({marks})", (*args, job_id, *when)).rowcount > 0


def job_status(job_id):
    with connect() as con:
        row = con.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
    return row[0] if row else None


def requeue_running():
    """Jobs left running by a restart go back in the queue; their saved pages are kept."""
    with connect() as con:
        return con.execute("UPDATE jobs SET status='queued', stage='Interrupted by a restart, waiting to resume',"
                           " updated_at=datetime('now') WHERE status='running'").rowcount


def cancel_job(job_id):
    """Cancel a job that hasn't finished, or dismiss a failed one; its stored file goes
    once nothing uses it. Returns the job (unchanged if it had already finished)."""
    with files_lock:
        with connect() as con:
            con.execute(f"UPDATE jobs SET status='cancelled', stage='Cancelled', partial=NULL, updated_at=datetime('now')"
                        f" WHERE id=? AND status IN ({','.join('?' * len(HOLDING_JOBS))})", (job_id, *HOLDING_JOBS))
            row = con.execute("SELECT status, file_path FROM jobs WHERE id=?", (job_id,)).fetchone()
            drop = row and row["status"] == "cancelled" and _unused(con, row["file_path"])
        if drop:
            _remove(row["file_path"])
    return get_job(job_id)


# Words too common to tell passages apart. A question's "what is the" matched nearly every
# chunk, which ranked them by how often they say "the", and in the blend with search by
# meaning (vectors.search) counted as being found by keywords too. Greetings and thanks
# too: "hi" found a manual's "Hi-Low range shift".
STOPWORDS = frozenset(
    "about an and are as at be by can could do does did for from had has have how i if in into is it its many me "
    "much my of on or our should so than that the their them then there these they this those to was we were what "
    "when where which who whom why will with would you your "
    "hi hello hey hiya thanks thank thx please ok okay yes yeah bye goodbye cheers".split())


def fts_query(text, question=False):
    """Turn free text into a safe FTS5 query: quoted terms OR-ed together, without the
    STOPWORDS. A question that is nothing else ("What is 2 + 2?") searches for nothing; in
    the library's search box they are kept when they are all there is, since "will", "IT"
    or "WHO" typed alone are names there (a will, an IT policy)."""
    terms = [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]
    rare = [t for t in terms if t not in STOPWORDS]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(rare if rare or question else terms))


def search_documents(text, limit=10, doc_ids=None):
    """Documents whose file name, title, type, summary, keywords or vendor have the words of
    `text`, best first (bm25); only among `doc_ids` (folder_documents), if given."""
    q = fts_query(text)
    if not q:
        return []
    scope, args = _only(doc_ids)
    with connect() as con:
        rows = con.execute(
            f"SELECT {_doc_columns('d')} FROM docs_fts f JOIN documents d ON d.id = f.rowid"
            f" WHERE docs_fts MATCH ?{scope} ORDER BY bm25(docs_fts) LIMIT ?", (q, *args, limit))
        return _docs(con, rows)


# A chunk as search returns it: where it is (document, place, pages) and whether its
# document's original can be opened (at a page, for a PDF).
_HIT = ("c.id AS chunk_id, c.doc_id, c.idx, c.text, c.page_start, c.page_end, d.filename, d.title,"
        " d.file_path IS NOT NULL AS has_file, d.mime")


def _hit(row):
    h = dict(row)
    mime = h.pop("mime")
    h["has_file"] = bool(h["has_file"])
    h["pdf"] = h["has_file"] and mime == "application/pdf"
    return h


def _among(doc_ids):
    """A filter on c.doc_id for a search limited to these documents (None: all of them).
    One JSON parameter, however many there are."""
    if doc_ids is None:
        return "", ()
    return " AND c.doc_id IN (SELECT value FROM json_each(?))", (json.dumps(list(doc_ids)),)


def search_chunks(text, limit=5, doc_ids=None, question=False):
    """Chunks with the words of `text`, best first (bm25), each with a snippet that marks
    them [like this]; only in the documents `doc_ids`, if given. `question`: as fts_query."""
    q = fts_query(text, question)
    if not q:
        return []
    scope, args = _among(doc_ids)
    with connect() as con:
        rows = con.execute(
            f"SELECT {_HIT}, snippet(chunks_fts, 0, '[', ']', ' … ', 24) AS snippet"
            " FROM chunks_fts f JOIN chunks c ON c.id = f.rowid JOIN documents d ON d.id = c.doc_id"
            f" WHERE chunks_fts MATCH ?{scope} ORDER BY bm25(chunks_fts) LIMIT ?", (q, *args, limit))
        return [_hit(r) for r in rows]


def chunks_by_id(ids):
    """{chunk id: chunk as search_chunks has it, without a snippet} for those that exist."""
    with connect() as con:
        return {r["chunk_id"]: _hit(r) for r in con.execute(
            f"SELECT {_HIT} FROM chunks c JOIN documents d ON d.id = c.doc_id"
            " WHERE c.id IN (SELECT value FROM json_each(?))", (json.dumps(list(ids)),))}


def opening_chunks(doc_ids, limit):
    """The first `limit` chunks of these documents, as search_chunks has them: each one's
    first chunk, then each one's second, and so on."""
    scope, args = _among(doc_ids)
    with connect() as con:
        return [_hit(r) for r in con.execute(
            f"SELECT {_HIT} FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE 1{scope}"
            " ORDER BY c.idx, c.doc_id LIMIT ?", (*args, limit))]


def library_overview(limit=60):
    """(the newest `limit` documents' file names, titles, types and summaries, how many
    documents there are): what a question that matched no passage is given instead."""
    with connect() as con:
        docs = [dict(r) for r in con.execute(
            "SELECT filename, title, doc_type, summary FROM documents ORDER BY id DESC LIMIT ?", (limit,))]
        return docs, con.execute("SELECT count(*) FROM documents").fetchone()[0]


def document_names(ids):
    """{id: filename} of the documents among `ids` that exist, in the order given."""
    with connect() as con:
        names = dict(con.execute("SELECT id, filename FROM documents WHERE id IN (SELECT value FROM json_each(?))",
                                 (json.dumps(list(ids)),)).fetchall())
    return {i: names[i] for i in ids if i in names}


# ---- vectors for search by meaning (vectors.py) ----

def all_vectors(model, min_chars=0):
    """The chunks this embedding model embedded, for vectors.py to load: yields first how many
    vectors the model has (the most rows that can follow), then (chunk id, document id, vector
    bytes) of each, one at a time, read in one transaction so the count holds. A chunk
    shorter than `min_chars` is left out, unless it is its document's only one."""
    with connect() as con:
        con.execute("BEGIN")
        yield con.execute("SELECT count(*) FROM chunk_vectors WHERE model=?", (model,)).fetchone()[0]
        yield from con.execute(
            "SELECT v.chunk_id, c.doc_id, v.vector FROM chunk_vectors v JOIN chunks c ON c.id = v.chunk_id"
            " WHERE v.model=? AND (length(trim(c.text, char(9, 10, 12, 13, 32))) >= ?"
            " OR c.doc_id IN (SELECT doc_id FROM chunks GROUP BY doc_id HAVING count(*) = 1))", (model, min_chars))


def missing_vectors(model, limit, skip=()):
    """Up to `limit` chunks, all of one document, that have no vector from this embedding
    model (indexed before search by meaning, while the model was unavailable, or by
    another model): [(chunk id, document id, text)], in order. Chunks `skip` are passed over."""
    with connect() as con:
        rows = con.execute("SELECT c.id, c.doc_id, c.text FROM chunks c LEFT JOIN chunk_vectors v"
                           " ON v.chunk_id = c.id AND v.model = ? WHERE v.chunk_id IS NULL"
                           " AND c.id NOT IN (SELECT value FROM json_each(?))"
                           " ORDER BY c.doc_id, c.idx LIMIT ?", (model, json.dumps(list(skip)), limit)).fetchall()
    return [tuple(r) for r in rows if r[1] == rows[0][1]]


def store_vectors(model, rows):
    """Keep chunks' vectors: rows [(chunk id, the text embedded, vector bytes)]. A chunk
    whose text is no longer that (its document was re-read, or deleted and the id given to
    a new chunk, while the text was being embedded) is skipped. Returns how many were kept."""
    with connect() as con:
        kept = sum(con.execute("INSERT OR REPLACE INTO chunk_vectors (chunk_id, model, vector)"
                               " SELECT id, ?, ? FROM chunks WHERE id=? AND text=?",
                               (model, blob, cid, text)).rowcount for cid, text, blob in rows)
    _index_changed()
    return kept


def vector_counts(model):
    """(chunks in the library, chunks with a vector from this embedding model)."""
    with connect() as con:
        return tuple(con.execute("SELECT (SELECT count(*) FROM chunks),"
                                 " (SELECT count(*) FROM chunk_vectors WHERE model=?)", (model,)).fetchone())


# ---- usage log: one row per model call ----

def add_usage(calls, doc_id=None, job_id=None):
    """calls: [(kind, backend, model, usage dict, cost_usd)], where usage has the keys
    input, output, cache_read and cache_write (as llm.chat returns it)."""
    if not calls:
        return
    with connect() as con:
        con.executemany(
            "INSERT INTO usage (kind, backend, model, input_tokens, output_tokens, cache_read_tokens,"
            " cache_write_tokens, cost_usd, doc_id, job_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(kind, backend, model, u.get("input") or 0, u.get("output") or 0, u.get("cache_read") or 0,
              u.get("cache_write") or 0, cost, doc_id, job_id) for kind, backend, model, u, cost in calls])


def link_usage(job_id, doc_id):
    """A job's model calls belong to this document (a re-read's, even when it failed)."""
    with connect() as con:
        con.execute("UPDATE usage SET doc_id=? WHERE job_id=?", (doc_id, job_id))


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


# ---- saved conversations: each question and its answer, kept as two messages ----

def _ids(doc_ids):
    return None if doc_ids is None else json.dumps(list(doc_ids))


def begin_exchange(conv_id, question, model="", doc_ids=None, title="", folder_ids=None):
    """Keep a question as it is asked, with an empty answer ('answering') that finish_exchange
    fills in: a reloaded page, another tab, or a restart still finds the question. A new
    conversation titled `title` when conv_id is None. The conversation takes the question's
    model choice, documents and folders (doc_ids and folder_ids, both None for all; those
    deleted meanwhile left out, so a delete while it is answered isn't undone; with folders,
    doc_ids is kept as a list, [] for none). Returns {conversation_id, title, question_id,
    answer_id}, or None if the conversation was deleted meanwhile (nothing is kept then)."""
    with connect() as con:
        if folder_ids is not None:
            folder_ids = [r[0] for r in con.execute(
                "SELECT value FROM json_each(?) WHERE value IN (SELECT id FROM folders)", (_ids(folder_ids),))]
            doc_ids = [] if doc_ids is None else doc_ids
        if doc_ids is not None:
            doc_ids = [r[0] for r in con.execute(
                "SELECT value FROM json_each(?) WHERE value IN (SELECT id FROM documents)", (_ids(doc_ids),))]
        if conv_id is None:
            conv_id = con.execute("INSERT INTO conversations (title, model, doc_ids, folder_ids) VALUES (?,?,?,?)",
                                  (title, model, _ids(doc_ids), _ids(folder_ids))).lastrowid
        elif not con.execute("UPDATE conversations SET updated_at=datetime('now'), model=?, doc_ids=?, folder_ids=?"
                             " WHERE id=?", (model, _ids(doc_ids), _ids(folder_ids), conv_id)).rowcount:
            return None
        title = con.execute("SELECT title FROM conversations WHERE id=?", (conv_id,)).fetchone()[0]
        question_id = con.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, 'user', ?)",
                                  (conv_id, question)).lastrowid
        answer_id = con.execute("INSERT INTO messages (conversation_id, role, content, status)"
                                " VALUES (?, 'assistant', '', 'answering')", (conv_id,)).lastrowid
    return {"conversation_id": conv_id, "title": title, "question_id": question_id, "answer_id": answer_id}


def finish_exchange(exchange, sent, answer):
    """Fill in the answer of a question begin_exchange kept. `sent`: the question as the model
    got it. `answer`: {content, sources, model, model_label, switched, usage, status, error}.
    Returns the conversation's title as it is now (it may have been renamed meanwhile), or
    None if it was deleted meanwhile. Rows are found by their conversation too: SQLite gives a
    deleted message's id to the next one, never a deleted conversation's."""
    conv_id = exchange["conversation_id"]
    with connect() as con:
        con.execute("UPDATE messages SET sent=? WHERE id=? AND conversation_id=?",
                    (sent, exchange["question_id"], conv_id))
        if not con.execute(
                "UPDATE messages SET content=?, sources=?, model=?, model_label=?, switched=?, usage=?, status=?, error=?"
                " WHERE id=? AND conversation_id=?",
                (answer["content"], json.dumps(answer.get("sources"), ensure_ascii=False), answer.get("model"),
                 answer.get("model_label"), answer.get("switched"), json.dumps(answer.get("usage")),
                 answer.get("status", "done"), answer.get("error"), exchange["answer_id"], conv_id)).rowcount:
            return None
        return con.execute("SELECT title FROM conversations WHERE id=?", (conv_id,)).fetchone()[0]


def cut_answer(conv_id, answer_id, shown):
    """An answer the page stopped after showing `shown` characters of it, kept already: cut
    there and marked stopped. One still being written ('answering') is left to whoever writes
    it. False if there is no such answer."""
    with connect() as con:
        row = con.execute("SELECT status FROM messages WHERE id=? AND conversation_id=? AND role='assistant'",
                          (answer_id, conv_id)).fetchone()
        if row and row[0] != "answering":
            con.execute("UPDATE messages SET content=substr(content, 1, ?), status='stopped', error=NULL WHERE id=?",
                        (shown, answer_id))
    return row is not None


def end_interrupted_answers(before):
    """Answers begun before `before` (this process's start, as UTC 'YYYY-MM-DD HH:MM:SS') and
    never finished: a restart cut them off. They fail, saying so, and pages stop waiting."""
    with connect() as con:
        return con.execute("UPDATE messages SET status='failed', error='The server restarted before this was answered'"
                           " WHERE status='answering' AND created_at < ?", (before,)).rowcount


def history(conv_id):
    """A conversation's questions and answers as the model is sent them, oldest first:
    [{role, content}], each question as it was sent (excerpts and all, byte for byte, so
    Claude's prompt cache still hits). An empty answer (still being written, stopped before
    its first words, failed) is left out with its question: Claude refuses an empty turn."""
    with connect() as con:
        rows = con.execute("SELECT role, content, sent FROM messages WHERE conversation_id=? ORDER BY id",
                           (conv_id,)).fetchall()
    turns = []
    for q, a in zip(rows[::2], rows[1::2]):  # saved in pairs
        if q["role"] == "user" and a["role"] == "assistant" and a["content"].strip():
            turns += [{"role": "user", "content": q["sent"] or q["content"]},
                      {"role": "assistant", "content": a["content"]}]
    return turns


_CONVERSATION = ("c.id, c.title, c.created_at, c.updated_at, c.model, c.doc_ids, c.folder_ids, (SELECT count(*)"
                 " FROM messages m WHERE m.conversation_id = c.id AND m.role = 'user') AS questions")


def _conversation(row):
    c = dict(row)
    for k in ("doc_ids", "folder_ids"):
        c[k] = None if c[k] is None else json.loads(c[k])
    return c


def list_conversations(limit=500):
    """The latest conversations, the one asked last first, with how many questions each has."""
    with connect() as con:
        return [_conversation(r) for r in con.execute(
            f"SELECT {_CONVERSATION} FROM conversations c ORDER BY c.updated_at DESC, c.id DESC LIMIT ?", (limit,))]


def get_conversation(conv_id, messages=True):
    """A conversation as list_conversations has it, with its messages in order (questions:
    id, role, content, created_at; answers also their sources, model, usage, status and
    error). None if there is no such conversation."""
    with connect() as con:
        row = con.execute(f"SELECT {_CONVERSATION} FROM conversations c WHERE c.id=?", (conv_id,)).fetchone()
        if not row:
            return None
        c = _conversation(row)
        if messages:
            c["messages"] = [_message(r) for r in con.execute(
                "SELECT id, role, content, sources, model, model_label, switched, usage, status, error, created_at"
                " FROM messages WHERE conversation_id=? ORDER BY id", (conv_id,))]
            _mark_gone(con, c["messages"])
    return c


def _mark_gone(con, messages):
    """Answers' sources whose document has been deleted since, or whose id a later upload was
    given (SQLite reuses the highest one), are marked gone: their file can't be opened from
    here any more. The passages the answer was given are kept with it all the same."""
    ids = {s["doc_id"] for m in messages for s in m.get("sources") or ()}
    if not ids:
        return
    now = dict(con.execute("SELECT id, filename FROM documents WHERE id IN (SELECT value FROM json_each(?))",
                           (json.dumps(list(ids)),)).fetchall())
    for m in messages:
        for item in [*(m.get("sources") or ()), *(m.get("excerpts") or ())]:
            if now.get(item["doc_id"]) != item["filename"]:
                item.update(has_file=False, pdf=False, gone=True)


def _message(row):
    m = dict(row)
    if m["role"] == "user":
        return {k: m[k] for k in ("id", "role", "content", "created_at")}
    m["usage"] = json.loads(m["usage"]) if m["usage"] else None
    found = json.loads(m.pop("sources") or "null") or {}
    return {**m, **{k: found.get(k) for k in ("sources", "excerpts", "overview", "scope", "scope_names", "scope_folders",
                                              "scope_count", "left_out")}}


def rename_conversation(conv_id, title):
    with connect() as con:
        return con.execute("UPDATE conversations SET title=? WHERE id=?", (title, conv_id)).rowcount > 0


def delete_conversation(conv_id):
    """Delete a conversation and its messages. Its model calls stay in the usage log."""
    with connect() as con:
        return con.execute("DELETE FROM conversations WHERE id=?", (conv_id,)).rowcount > 0


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
