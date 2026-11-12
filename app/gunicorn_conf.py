"""gunicorn hooks (the settings are flags in the Dockerfile's CMD)."""


def post_worker_init(worker):
    """Start the job worker as soon as the app is up, so jobs a restart cut off carry on
    without waiting for someone to open the page. Requests also start it (main.py)."""
    from app import jobs
    try:
        jobs.start()
    except Exception:  # say the database stays locked: the first request tries again
        worker.log.exception("Could not start the job worker")
