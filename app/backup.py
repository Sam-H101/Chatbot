"""Backups of the library, run by the `backup` service in docker-compose.yml.

Every night at BACKUP_HOUR it writes a consistent snapshot of the SQLite database into
BACKUP_DIR as metabase-YYYYMMDD-HHMM.db, keeps the newest BACKUP_KEEP, and copies the kept
originals that aren't there yet into BACKUP_DIR/files. It only reads the library, so the
data volume can be mounted read-only.

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
import sqlite3
import sys
import tempfile
import time
from contextlib import closing
from urllib.parse import quote

from . import db

log = logging.getLogger(__name__)

BACKUP_DIR = os.environ.get("BACKUP_DIR") or "/backups"
NAME = re.compile(r"^metabase-(\d{8}-\d{4})\.db$")  # sorts by time: the newest is the greatest
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TMP = ".backup-"                       # files being written; a crash leaves them, the next run clears them
RETRY = datetime.timedelta(minutes=30)  # after a failed run (say the library was locked throughout)
CHECK_SECONDS = 60                     # the loop looks at the clock this often, so a night the machine
                                       # slept through is made up soon after it wakes


class BackupError(Exception):
    """An expected failure, logged as its message alone."""


def settings():
    """BACKUP_HOUR and BACKUP_KEEP, checked: a mistyped value stops the service with a message
    rather than backing up at another hour or keeping nothing."""
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


def _open_ro(path):
    """Open a database read-only (mode=ro), so it works on a read-only mount and can never
    change the file. Waits for a writer to finish its transaction."""
    return sqlite3.connect(f"file:{quote(os.path.abspath(path))}?mode=ro", uri=True, timeout=60)


def _check(path, compact=False, name=None):
    """Check a copy (integrity_check) and return how many documents it holds. compact: VACUUM
    it first, since the live file keeps the space deleted documents freed (the live library
    was 92% free pages). Every table has an INTEGER PRIMARY KEY (FTS5's shadow tables too),
    so VACUUM keeps all ids. Without compact, the file is only read. name: the copy's name in
    messages (the snapshot it is to become)."""
    name = name or os.path.basename(path)
    try:
        with closing(sqlite3.connect(path, isolation_level=None) if compact else _open_ro(path)) as con:
            if compact:
                con.execute("VACUUM")
            result = [r[0] for r in con.execute("PRAGMA integrity_check")]
            if result == ["ok"]:
                try:
                    return con.execute("SELECT count(*) FROM documents").fetchone()[0]
                except sqlite3.OperationalError:  # not a library (no documents table)
                    return None
    except sqlite3.OperationalError as e:  # say a full disk while compacting
        raise BackupError(f"Could not check {name}: {e}") from e
    except sqlite3.DatabaseError as e:  # not a database at all ("file is not a database")
        result = [str(e)]
    raise BackupError(f"{name} is damaged: {'; '.join(result[:3])}")


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


def snapshot(folder=None, keep=None, now=None):
    """Copy the database into `folder` through SQLite's backup API, which reads it in one
    read transaction: a write under way is in the copy whole or not at all, where a file
    copy could catch it half done. The copy is compacted and checked before it gets its
    name, then all but the newest `keep` snapshots are removed.
    Returns {path, size, documents, removed}."""
    folder = folder or BACKUP_DIR
    keep = keep or settings()[1]
    if not os.path.isfile(db.DB_PATH):
        raise BackupError(f"No library at {db.DB_PATH} yet: nothing to back up")
    os.makedirs(folder, exist_ok=True)
    _clear_leftovers(folder)
    path = os.path.join(folder, (now or datetime.datetime.now()).strftime("metabase-%Y%m%d-%H%M.db"))
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=TMP, suffix=".db")  # readable by its owner only (0600)
    os.close(fd)
    try:
        try:
            with closing(_open_ro(db.DB_PATH)) as src, closing(sqlite3.connect(tmp)) as out:
                src.backup(out)
        except sqlite3.Error as e:  # say still locked after a minute, a crash's journal to roll back first, a full disk
            raise BackupError(f"Could not copy {db.DB_PATH}: {e}") from e
        documents = _check(tmp, compact=True, name=os.path.basename(path))
        os.replace(tmp, path)  # a second one within the same minute replaces the first
    except BaseException:
        for leftover in (tmp, tmp + "-journal"):
            try:
                os.remove(leftover)
            except FileNotFoundError:
                pass
        raise
    return {"path": path, "size": os.path.getsize(path), "documents": documents, "removed": prune(folder, keep)}


def _copy(path, folder, name):
    """Copy one original into `folder` under `name` (temporary file, then rename, as db._store
    writes them), checking it against the SHA-256 in its name. Returns its size."""
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=TMP)
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f, os.fdopen(fd, "wb") as out:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
                out.write(block)
        os.chmod(tmp, 0o644)
        st = os.stat(path)
        os.utime(tmp, (st.st_atime, st.st_mtime))
        os.replace(tmp, os.path.join(folder, name))
    except BaseException:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise
    if SHA256.match(name) and digest.hexdigest() != name:
        log.warning("Original %s doesn't match its SHA-256 (damaged?); copied as it is", name)
    return st.st_size


def mirror_files(src=None, dest=None, names=None):
    """Copy the originals in `src` (the library's files) that `dest` doesn't have yet. Add only:
    an original whose documents were deleted stays in the backup, so a restored snapshot finds
    it. Files are named by their SHA-256, so one already there has the same content.
    names: only these (a restore copies back what its snapshot uses).
    Returns {copied, bytes, failed, missing}."""
    src = src or db.files_dir()
    dest = dest or os.path.join(BACKUP_DIR, "files")
    result = {"copied": 0, "bytes": 0, "failed": 0, "missing": 0}
    if names is None:
        try:
            names = [n for n in os.listdir(src) if not n.startswith(".")]  # .upload-*: still being written
        except FileNotFoundError:
            return result  # no originals kept yet
    _clear_leftovers(dest)
    for name in sorted(names):
        path = os.path.join(src, name)
        if os.path.exists(os.path.join(dest, name)):
            continue
        if not os.path.isfile(path):
            result["missing"] += 1  # deleted meanwhile, or (restoring) never backed up
            continue
        try:
            os.makedirs(dest, exist_ok=True)
            result["bytes"] += _copy(path, dest, name)
            result["copied"] += 1
        except FileNotFoundError:
            result["missing"] += 1
        except OSError as e:
            log.error("Could not copy original %s: %s", name, e)
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


def _referenced(path):
    """The originals a library uses: its documents', and those of jobs that still need theirs.
    A snapshot from before originals were kept (or jobs existed) has none."""
    names = set()
    with closing(_open_ro(path)) as con:
        for sql, args in (("SELECT file_path FROM documents", ()),
                          (f"SELECT file_path FROM jobs WHERE status IN ({','.join('?' * len(db.HOLDING_JOBS))})",
                           db.HOLDING_JOBS)):
            try:
                names.update(os.path.basename(r[0]) for r in con.execute(sql, args) if r[0])
            except sqlite3.OperationalError:
                pass
    return names


def restore(name, folder=None, now=None):
    """Put snapshot `name` ('latest': the newest) back as the library, and copy back the
    originals it uses that the library's files folder lacks. The database it replaces is
    kept beside it as <DB_PATH>.before-restore-<time> (with its journal, if it had one).
    The app must be stopped first: it keeps the library's state in memory.
    Returns {snapshot, documents, kept, files}."""
    folder = folder or BACKUP_DIR
    names = snapshots(folder)
    if name == "latest":
        if not names:
            raise BackupError(f"No snapshots in {folder}")
        name = names[0]
    if name not in names:  # also keeps a name from reaching outside the folder
        raise BackupError(f"No snapshot named {name!r} in {folder} (--list shows them)")
    path = os.path.join(folder, name)
    documents = _check(path)
    target = os.path.abspath(db.DB_PATH)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), prefix=TMP, suffix=".db")
    os.close(fd)
    kept = None
    try:
        with closing(_open_ro(path)) as src, closing(sqlite3.connect(tmp)) as out:
            src.backup(out)
        os.chmod(tmp, 0o644)
        stamp = (now or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")
        # A journal left by a crash goes with its database: beside the restored one, SQLite
        # would take it for that one's and "roll it back" into it.
        for suffix in ("", "-journal", "-wal", "-shm"):
            if os.path.exists(target + suffix):
                os.replace(target + suffix, f"{target}.before-restore-{stamp}{suffix}")
                kept = kept or f"{target}.before-restore-{stamp}"
        os.replace(tmp, target)
    except BaseException:
        for leftover in (tmp, tmp + "-journal"):
            try:
                os.remove(leftover)
            except FileNotFoundError:
                pass
        raise
    files = mirror_files(src=os.path.join(folder, "files"), dest=db.files_dir(), names=_referenced(target))
    log.info("Restored %s (%s) to %s%s. %s copied back%s.", name, _count(documents or 0, "document"), target,
             f"; the database it replaced is kept as {kept}" if kept else "", _count(files["copied"], "original"),
             f"; {files['missing']} the library uses are in neither the backup nor the library: their documents"
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
    log.info("Backing up %s every night at %02d:00 %s into %s, keeping the newest %d",
             db.DB_PATH, hour, time.strftime("%Z"), folder, keep)
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
