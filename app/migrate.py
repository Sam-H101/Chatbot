"""Move a library from the SQLite database this app used before Postgres into Postgres.

    python -m app.migrate /data/metabase.db              copy it (refused if Postgres has documents)
    python -m app.migrate /data/metabase.db --dry-run    say what it would copy

Every row keeps its id, so open tabs, saved chats limited to documents or folders, and the
usage log still point at the same rows; each table's id sequence then carries on after the
highest. Vectors become pgvector vectors (float32 bytes, read back as floats) and each size
gets its HNSW index. FTS5's tables are not copied: the tsvector columns are made from the
text. The SQLite file is only read, never changed: keep it until the new library has been
checked, then remove it. Originals stay where they are (FILES_DIR); with STORAGE_BACKEND s3 or
azure, `python -m app.storage --copy-local` copies them into the bucket.
"""
import argparse
import logging
import os
import sqlite3
import struct
import sys
from contextlib import closing
from urllib.parse import quote

from . import db

log = logging.getLogger(__name__)

# What each table copies, besides its id, from the columns the SQLite file has (an older one
# lacks some: Postgres gives those their default, or NULL).
BOOLEAN = {("instructions", "enabled")}
BATCH = 500


class MigrateError(Exception):
    pass


def _open(path):
    if not os.path.isfile(path):
        raise MigrateError(f"No SQLite library at {path}")
    con = sqlite3.connect(f"file:{quote(os.path.abspath(path))}?mode=ro", uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    return con


def _columns_sqlite(src, table):
    try:
        return [r[1] for r in src.execute(f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []


def _columns_pg(con, table):
    return [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = ?"
        " AND is_generated = 'NEVER' ORDER BY ordinal_position", (table,))]


def _floats(blob):
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def plan(src):
    """{table: rows} the SQLite library holds, of the tables Postgres has."""
    out = {}
    for table in db.TABLES:
        if _columns_sqlite(src, table):
            out[table] = src.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return out


def _copy_table(src, con, table, sizes):
    have = _columns_sqlite(src, table)
    if not have:
        return 0
    if table == "chunk_vectors":
        return _copy_vectors(src, con, sizes)
    cols = [c for c in _columns_pg(con, table) if c in have]
    marks = ",".join("?" * len(cols))
    insert = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({marks})"
    bools = [i for i, c in enumerate(cols) if (table, c) in BOOLEAN]
    rows = src.execute(f"SELECT {', '.join(cols)} FROM {table} ORDER BY rowid")
    n = 0
    while True:
        batch = rows.fetchmany(BATCH)
        if not batch:
            break
        values = [list(r) for r in batch]
        for v in values:
            for i in bools:
                v[i] = bool(v[i])
            for i, x in enumerate(v):
                if isinstance(x, str):  # Postgres text holds no NUL; chunks hold no match marks either
                    v[i] = x.translate(db._NO_MARKS)
        con.executemany(insert, values)
        n += len(values)
    return n


def _copy_vectors(src, con, sizes):
    """chunk_vectors' float32 bytes as pgvector vectors; `sizes` gains each vector size."""
    n = 0
    rows = src.execute("SELECT chunk_id, model, vector FROM chunk_vectors ORDER BY chunk_id")
    while True:
        batch = rows.fetchmany(BATCH)
        if not batch:
            break
        values = []
        for cid, model, blob in batch:
            v = _floats(blob)
            sizes.add(len(v))
            values.append((model, len(v), db._vec(v), cid))
        con.executemany(db.VECTOR_FROM_CHUNK, values)  # its document, length and so on from its chunk
        n += len(values)
    return n


def _reset_sequences(con):
    for table in db.TABLES:
        if table == "chunk_vectors":
            continue
        con.execute(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), coalesce((SELECT max(id) FROM {table}), 0) + 1,"
                    " false)")


def migrate(path, dry_run=False):
    """Copy the SQLite library at `path` into Postgres, in one transaction. Returns {table: rows}."""
    with closing(_open(path)) as src:
        counts = plan(src)
        if dry_run:
            return counts
        with db.connect() as con:
            existing = con.execute("SELECT count(*) FROM documents").fetchone()[0]
            if existing:
                raise MigrateError(f"Postgres already holds {existing} document(s): migrate into an empty library")
            sizes, copied = set(), {}
            for table in db.TABLES:
                copied[table] = _copy_table(src, con, table, sizes)
            # Saved chats point at documents and folders by id: rows that no longer exist were
            # already taken out by the old app, so nothing to fix. Folders' parents likewise.
            _reset_sequences(con)
    for size in sizes:
        db._vector_index(size)
    with db.connect() as con:
        for table, n in copied.items():
            got = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            if got != n:
                raise MigrateError(f"{table}: copied {n} rows but Postgres has {got}")
    return copied


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m app.migrate", description="Copy a SQLite library into Postgres.")
    p.add_argument("sqlite", help="the SQLite library (in the container: /data/metabase.db)")
    p.add_argument("--dry-run", action="store_true", help="only say what would be copied")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    try:
        counts = migrate(args.sqlite, args.dry_run)
    except (MigrateError, db.Error) as e:
        print(e, file=sys.stderr)
        return 1
    width = max(map(len, counts), default=0)
    print(("Would copy:" if args.dry_run else "Copied:"))
    for table, n in counts.items():
        print(f"  {table:<{width}}  {n:>8,}")
    if not args.dry_run:
        print("The SQLite file is unchanged: keep it until you have checked the library, then remove it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
