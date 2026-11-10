FROM python:3.13-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
ENV DB_PATH=/data/metabase.db
VOLUME /data
EXPOSE 8000
# One worker: SQLite is a single file. Threads handle concurrent requests; long timeout for Claude calls.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "600", "app.main:app"]
