# The app's image, and (target test) the test suite's, built on the same base so the tests
# run with the very packages the app has (docker-compose.test.yml).
FROM python:3.13-slim AS base
WORKDIR /srv
# pg_dump and pg_restore for the backup service (app/backup.py). They must be at least the
# server's version (the postgres service runs PostgreSQL 18), and Debian 13 ships 17, so the
# client comes from PostgreSQL's own apt repository.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && install -d /usr/share/postgresql-common/pgdg \
    && curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt trixie-pgdg main" \
       > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update && apt-get install -y --no-install-recommends postgresql-client-18 \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS test
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

FROM base
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
