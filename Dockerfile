FROM python:3.13-slim
WORKDIR /srv
# pg_dump and pg_restore for the backup service (app/backup.py): Debian 13 ships version 17,
# the same as the postgres service in docker-compose.yml.
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
# Local originals (STORAGE_BACKEND=local) and thumbnails; the database is in the postgres service.
ENV FILES_DIR=/data/files THUMBS_DIR=/data/thumbs
VOLUME /data
EXPOSE 8000
# One worker: uploads are read by one background thread in that process (app/jobs.py), which
# app/gunicorn_conf.py starts at boot, and a few locks (db.files_lock) are kept in memory.
# Threads handle concurrent requests (a streamed answer holds one while it is written); long
# timeout for Claude chat calls. On stop, answers being streamed get 8 s to end: within
# Docker's 10 s, so gunicorn shuts down cleanly instead of being killed (its default is 30 s).
# Cut-off answers are marked failed, and a job being read carries on at the next start.
CMD ["gunicorn", "--config", "app/gunicorn_conf.py", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "600", "--graceful-timeout", "8", "app.main:app"]
