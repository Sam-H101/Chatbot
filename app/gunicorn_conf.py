"""gunicorn hooks (the settings are flags in the Dockerfile's CMD)."""
_lock = None  # the connection holding the library's lock, for as long as the worker runs


def post_worker_init(worker):
    """Hold the library's advisory lock (db.hold_app_lock) while the worker runs, so a restore
    refuses to replace the library under it; waits while a restore holds it. Taken in the
    worker, not the master: a connection opened before the fork would be shared with it.
    Then start the job worker as soon as the app is up, so jobs a restart cut off carry on
    without waiting for someone to open the page. Requests also start it (main.py)."""
    global _lock
    from app import db, jobs
    try:
        _lock = db.hold_app_lock()
    except db.Error as e:  # the database is down: a restore then can't tell the app runs
        worker.log.warning("Could not take the library's lock (%s): a restore can't tell that the app is running", e)
    try:
        jobs.start()
    except Exception:  # say the database stays locked: the first request tries again
        worker.log.exception("Could not start the job worker")
    try:
        from app import auth
        auth.announce()  # no account yet: the setup code, in the log
    except Exception:  # the first request that finds no account says it instead
        worker.log.exception("Could not check for user accounts")
