"""Backups of the library, run by the `backup` service in docker-compose.yml.

Every night at BACKUP_HOUR it writes a consistent snapshot of the Postgres database into
BACKUP_DIR as metabase-YYYYMMDD-HHMM.dump (pg_dump's custom format), keeps the newest
BACKUP_KEEP, and copies the kept originals that aren't there yet into BACKUP_DIR/files, from
whichever store keeps them (storage.py: a local folder, S3 or Azure). It only reads the
library.

    python -m app.backup                run nightly (the service)
    python -m app.backup --now          take one now
    python -m app.backup --list         list them
    python -m app.backup --restore X    put snapshot X ('latest': the newest) back; stop the app first
"""
import argparse
import datetime
import hashlib
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

from psycopg.conninfo import conninfo_to_dict

from . import db, storage

log = logging.getLogger(__name__)

BACKUP_DIR = os.environ.get("BACKUP_DIR") or "/backups"
NAME = re.compile(r"^metabase-(\d{8}-\d{4})\.dump$")  # sorts by time: the newest is the greatest
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TMP = ".backup-"                       # files being written; a crash leaves them, the next run clears them
RETRY = datetime.timedelta(minutes=30)  # after a failed run (say the database was down throughout)
CHECK_SECONDS = 60                     # the loop looks at the clock this often, so a night the machine
                                       # slept through is made up soon after it wakes


class BackupError(Exception):
    """An expected failure, logged as its message alone."""


def settings():
    """BACKUP_HOUR and BACKUP_KEEP, checked: a mistyped value ends the service with a message
    (Docker then restarts it, again and again, until it is fixed) rather than backing up at
    another hour or keeping nothing."""
    try:
        hour = int(os.environ.get("BACKUP_HOUR") or 3)
        keep = int(os.environ.get("BACKUP_KEEP") or 14)
    except ValueError as e:
        raise SystemExit(f"BACKUP_HOUR and BACKUP_KEEP must be whole numbers ({e})")
    if not 0 <= hour <= 23:
        raise SystemExit(f"BACKUP_HOUR must be an hour from 0 to 23, not {hour}")
    if keep < 1:
        raise SystemExit(f"BACKUP_KEEP must be 1 or more, not {keep}")
    return hour, keep


def _pg_env():
    """pg_dump's and pg_restore's connection, as environment variables (so the password is
    not on a command line)."""
    params = conninfo_to_dict(db.DATABASE_URL)
    names = {"host": "PGHOST", "port": "PGPORT", "user": "PGUSER", "password": "PGPASSWORD", "dbname": "PGDATABASE",
             "sslmode": "PGSSLMODE"}
    env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
    env.update({names[k]: str(v) for k, v in params.items() if k in names and v is not None})
    return env


def _schema_args():
    """Only the library's schema when it has one of its own (tests); else the whole database."""
    return ["--schema", db.DB_SCHEMA] if db.DB_SCHEMA else []


def _run(args, what):
    try:
        done = subprocess.run(args, env=_pg_env(), capture_output=True, text=True, timeout=3600)
    except FileNotFoundError:
        raise BackupError(f"{args[0]} is not installed (the image installs postgresql-client)") from None
    if done.returncode:
        detail = (done.stderr or done.stdout).strip().splitlines()
        raise BackupError(f"Could not {what}: {'; '.join(detail[-3:]) or f'{args[0]} failed'}")
    return done.stdout


def _verify(path, name=None):
    """Check a dump can be read (pg_restore lists its contents) and return how many tables
    it has data for."""
    out = _run(["pg_restore", "--list", path], f"read {name or os.path.basename(path)}")
    tables = sum(" TABLE DATA " in line for line in out.splitlines())
    if not tables:
        raise BackupError(f"{name or os.path.basename(path)} holds no library")
    return tables


def _clear_leftovers(folder):
    """Remove files a crashed run left half written. Only those over an hour old: a manual
    --now may be writing its own right now."""
    try:
        names = os.listdir(folder)
    except FileNotFoundError:
        return
    for name in names:
        path = os.path.join(folder, name)
        try:
            if name.startswith(TMP) and time.time() - os.path.getmtime(path) > 3600:
                os.remove(path)
        except FileNotFoundError:
            pass


def snapshots(folder=None):
    """Snapshot file names in `folder`, newest first."""
    try:
        return sorted((n for n in os.listdir(folder or BACKUP_DIR) if NAME.match(n)), reverse=True)
    except FileNotFoundError:
        return []


def newest(folder=None):
    """When the newest snapshot was taken (from its name, local time), or None."""
    names = snapshots(folder)
    return datetime.datetime.strptime(NAME.match(names[0])[1], "%Y%m%d-%H%M") if names else None


def due(now, last, hour):
    """True when no snapshot was taken since the latest hour:00 (local time), or none at all.
    So a night the machine was off is made up once it is back, and a new install gets its
    first snapshot at once."""
    if last is None:
        return True
    mark = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if mark > now:
        mark -= datetime.timedelta(days=1)
    return last < mark


def prune(folder, keep):
    """Remove all but the newest `keep` snapshots; returns the names removed. Other files
    (a copy you renamed, say) are left alone."""
    removed = snapshots(folder)[keep:]
    for name in removed:
        try:
            os.remove(os.path.join(folder, name))
        except FileNotFoundError:
            pass
    return removed


def _dump(path):
    """pg_dump the library into `path`, from one exported snapshot: the dump and the count of
    documents returned see the same moment, and a write under way is in it whole or not at
    all. Returns how many documents it holds."""
    try:
        con = db._dedicated(autocommit=False)
    except db.Error as e:
        raise BackupError(f"Could not reach the database: {e}") from e
    try:
        con.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        snap = con.execute("SELECT pg_export_snapshot()").fetchone()[0]
        try:
            documents = con.execute("SELECT count(*) FROM documents").fetchone()[0]
        except db.Error:
            raise BackupError("No library in the database yet: nothing to back up") from None
        _run(["pg_dump", "--format=custom", "--no-owner", "--no-privileges", f"--snapshot={snap}",
              *_schema_args(), "--file", path], "dump the database")
    finally:
        con.close()
    return documents


def snapshot(folder=None, keep=None, now=None):
    """Dump the database into `folder`, check the dump, then give it its name and remove all
    but the newest `keep` snapshots. Returns {path, size, documents, removed}."""
    folder = folder or BACKUP_DIR
    keep = keep or settings()[1]
    os.makedirs(folder, exist_ok=True)
    _clear_leftovers(folder)
    path = os.path.join(folder, (now or datetime.datetime.now()).strftime("metabase-%Y%m%d-%H%M.dump"))
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=TMP, suffix=".dump")  # readable by its owner only (0600)
    os.close(fd)
    try:
        documents = _dump(tmp)
        _verify(tmp, os.path.basename(path))
        os.replace(tmp, path)  # a second one within the same minute replaces the first
    except BaseException:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise
    return {"path": path, "size": os.path.getsize(path), "documents": documents, "removed": prune(folder, keep)}


def _write(folder, name, chunks, stamp=None):
    """Write an original into `folder` under `name` (temporary file, then rename), checking it
    against the SHA-256 in its name. `chunks`: its bytes, in pieces. Returns its size."""
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=TMP)
    digest, size = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, "wb") as out:
            for block in chunks:
                digest.update(block)
                out.write(block)
                size += len(block)
        os.chmod(tmp, 0o644)
        if stamp:
            os.utime(tmp, stamp)
        os.replace(tmp, os.path.join(folder, name))
    except BaseException:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise
    if SHA256.match(name) and digest.hexdigest() != name:
        log.warning("Original %s doesn't match its SHA-256 (damaged?); copied as it is", name)
    return size


def _blocks(path):
    with open(path, "rb") as f:
        yield from iter(lambda: f.read(1 << 20), b"")


def mirror_files(src=None, dest=None, names=None):
    """Copy the originals in the store `src` (storage.py's, by default) that the folder `dest`
    doesn't have yet. Add only: an original whose documents were deleted stays in the backup,
    so a restored snapshot finds it. Originals are named by their SHA-256, so one already there
    has the same content. names: only these. Returns {copied, bytes, failed, missing}."""
    src = src or storage.store()
    dest = dest or os.path.join(BACKUP_DIR, "files")
    result = {"copied": 0, "bytes": 0, "failed": 0, "missing": 0}
    try:
        names = src.names() if names is None else names
    except Exception as e:
        log.error("Could not list the originals in %s: %s", src.describe(), e)
        result["failed"] += 1
        return result
    _clear_leftovers(dest)
    for name in sorted(names):
        if os.path.exists(os.path.join(dest, name)):
            continue
        try:
            os.makedirs(dest, exist_ok=True)
            path = src.local_path(name)
            if path:
                st = os.stat(path)
                result["bytes"] += _write(dest, name, _blocks(path), (st.st_atime, st.st_mtime))
            else:
                result["bytes"] += _write(dest, name, [src.get(name)])
            result["copied"] += 1
        except FileNotFoundError:
            result["missing"] += 1  # deleted meanwhile
        except Exception as e:
            log.error("Could not copy original %s: %s", name, e)
            result["failed"] += 1
    return result


def restore_files(folder, names, dest=None):
    """Put the originals `names` back into the store `dest` (storage.py's) from the backup's
    files `folder`, those it lacks. Returns {copied, bytes, failed, missing}."""
    dest = dest or storage.store()
    result = {"copied": 0, "bytes": 0, "failed": 0, "missing": 0}
    for name in sorted(names):
        path = os.path.join(folder, name)
        try:
            if dest.exists(name):
                continue
            if not os.path.isfile(path):
                result["missing"] += 1  # never backed up
                continue
            with open(path, "rb") as f:
                data = f.read()
            if dest.put(name, data):
                result["copied"] += 1
                result["bytes"] += len(data)
        except Exception as e:
            log.error("Could not put original %s back: %s", name, e)
            result["failed"] += 1
    return result


def _mb(n):
    return f"{n / 1e6:,.1f} MB"


def _count(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def run_once(folder=None, keep=None, now=None):
    """One backup: the snapshot, then the originals (after it, so it finds every file the
    snapshot uses). Returns {snapshot, files}."""
    folder = folder or BACKUP_DIR
    snap = snapshot(folder, keep, now)
    files = mirror_files(dest=os.path.join(folder, "files"))
    log.info("Backed up the library to %s (%s, %s)%s; %s copied (%s)%s",
             os.path.basename(snap["path"]), _mb(snap["size"]), _count(snap["documents"] or 0, "document"),
             f", removed {', '.join(snap['removed'])}" if snap["removed"] else "",
             _count(files["copied"], "new original"), _mb(files["bytes"]),
             f", {files['failed']} could not be copied" if files["failed"] else "")
    return {"snapshot": snap, "files": files}


def _referenced():
    """The originals the library uses now: its documents', and those of jobs that still need theirs."""
    marks = ",".join("?" * len(db.HOLDING_JOBS))
    with db.connect() as con:
        rows = con.execute(f"SELECT file_path FROM documents WHERE file_path IS NOT NULL UNION"
                           f" SELECT file_path FROM jobs WHERE status IN ({marks})", db.HOLDING_JOBS).fetchall()
    return {storage.name(r[0]) for r in rows if r[0]}


def _lock_out_the_app():
    """The library's advisory lock (db.lock_out_app), taken exclusively: the app holds it while
    it runs, and one starting meanwhile waits until it is released."""
    try:
        return db.lock_out_app()
    except db.AppRunning:
        raise BackupError("The app is running: stop it first (docker compose stop chatbot). It keeps some of the"
                          " library in memory, and would carry on as if the database it replaced were still"
                          " there") from None
    except db.Error as e:
        raise BackupError(f"Could not reach the database: {e}") from e


def restore(name, folder=None, now=None):
    """Put snapshot `name` ('latest': the newest) back as the library, and copy back the
    originals it uses that the store lacks. The library it replaces is dumped first, beside
    the snapshots, as before-restore-<time>.dump. Refuses while the app runs (it keeps some of
    the library's state in memory). Returns {snapshot, documents, kept, files}."""
    folder = folder or BACKUP_DIR
    names = snapshots(folder)
    if name == "latest":
        if not names:
            raise BackupError(f"No snapshots in {folder}")
        name = names[0]
    if name not in names:  # also keeps a name from reaching outside the folder
        raise BackupError(f"No snapshot named {name!r} in {folder} (--list shows them)")
    path = os.path.join(folder, name)
    _verify(path)
    lock = _lock_out_the_app()
    try:
        stamp = (now or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")
        kept = os.path.join(folder, f"before-restore-{stamp}.dump")
        try:
            _dump(kept)
        except BackupError as e:
            if "No library" not in str(e):
                raise
            kept = None  # an empty database: nothing to keep
        db.close_pool()
        _run(["pg_restore", "--clean", "--if-exists", "--no-owner", "--no-privileges", "--single-transaction",
              "--exit-on-error", *_schema_args(), "--dbname", conninfo_to_dict(db.DATABASE_URL).get("dbname", ""),
              path], f"restore {name}")
        db._schema_ready.clear()
        with db.connect() as con:  # brings a snapshot from an older version of the app up to date
            documents = con.execute("SELECT count(*) FROM documents").fetchone()[0]
        files = restore_files(os.path.join(folder, "files"), _referenced())
    finally:
        lock.close()
    log.info("Restored %s (%s)%s. %s copied back%s.", name, _count(documents or 0, "document"),
             f"; the library it replaced is kept as {os.path.basename(kept)}" if kept else "",
             _count(files["copied"], "original"),
             f"; {files['missing']} the library uses are in neither the backup nor the store: their documents"
             " keep their text but can't be opened or re-read" if files["missing"] else "")
    return {"snapshot": name, "documents": documents, "kept": kept, "files": files}


def listing(folder=None, keep=None):
    """The snapshots and originals in `folder`, as text."""
    folder = folder or BACKUP_DIR
    names = snapshots(folder)
    lines = [f"Snapshots in {folder}, newest first" + (f" (the newest {keep} are kept):" if keep else ":")]
    lines += [f"  {n}  {_mb(os.path.getsize(os.path.join(folder, n))):>12}" for n in names] or ["  none yet"]
    files_dir = os.path.join(folder, "files")
    try:
        files = [os.path.join(files_dir, n) for n in os.listdir(files_dir) if not n.startswith(".")]
    except FileNotFoundError:
        files = []
    lines.append(f"Originals in {files_dir}: {len(files)} ({_mb(sum(map(os.path.getsize, files)))})")
    return "\n".join(lines)


def _now():  # the loop's clock (tests set it)
    return datetime.datetime.now()


def loop(hour, keep, folder=None):
    """The service: look at the clock every minute, and back up when due() says so."""
    folder = folder or BACKUP_DIR
    log.info("Backing up the library every night at %02d:00 %s into %s, keeping the newest %d",
             hour, time.strftime("%Z"), folder, keep)
    tried = None
    while True:
        now = _now()
        if due(now, newest(folder), hour) and (tried is None or not now - RETRY < tried <= now):
            tried = now
            try:
                run_once(folder, keep, now)
            except BackupError as e:
                log.warning("%s; trying again in %d minutes", e, RETRY.seconds // 60)
            except Exception:
                log.exception("Backup failed; trying again in %d minutes", RETRY.seconds // 60)
        time.sleep(CHECK_SECONDS)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m app.backup", description="Back up the library (see README, Backups).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--now", action="store_true", help="take a backup now, then exit")
    g.add_argument("--list", action="store_true", help="list the backups, then exit")
    g.add_argument("--restore", nargs="?", const="", metavar="NAME",
                   help="put snapshot NAME ('latest' for the newest) back as the library; stop the app first")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    hour, keep = settings()
    if args.list or args.restore == "":
        print(listing(keep=keep if args.list else None))  # the restore service isn't given BACKUP_KEEP
        if args.restore == "":
            print("Name the snapshot to restore, or 'latest' for the newest.")
            return 2
        return 0
    try:
        if args.restore:
            restore(args.restore)
            print("Start the app again: docker compose start chatbot")
            return 0
        if args.now:
            return 1 if run_once(keep=keep)["files"]["failed"] else 0
    except BackupError as e:
        log.error("%s", e)
        return 1
    # PID 1 in a container ignores SIGTERM unless it handles it: without this, `docker compose
    # stop` would wait 10 s and then kill the service.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    loop(hour, keep)


if __name__ == "__main__":
    sys.exit(main())
