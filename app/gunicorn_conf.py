"""gunicorn hooks (the settings are flags in the Dockerfile's CMD)."""
_lock = None  # the library's lock, held by the master for as long as the app runs


def on_starting(server):
    """Hold the library's lock (db.lock_library) while the app runs, so a restore refuses to
    swap the database under it. Waits while a restore holds it."""
    global _lock
    from app import db
    try:
        _lock = db.lock_library()
    except OSError as e:  # a filesystem without flock: a restore then can't tell the app runs
        server.log.warning("Could not lock the library (%s): a restore can't tell that the app is running", e)


def post_worker_init(worker):
    """Start the job worker as soon as the app is up, so jobs a restart cut off carry on
    without waiting for someone to open the page. Requests also start it (main.py)."""
    from app import jobs
    try:
        jobs.start()
    except Exception:  # say the database stays locked: the first request tries again
        worker.log.exception("Could not start the job worker")
