"""The database and file store every test runs against: a Postgres schema of its own (made
here, dropped after it) in the database TEST_DATABASE_URL names, and a local folder for
originals. docker-compose.test.yml provides the database; without TEST_DATABASE_URL the
tests that need it are skipped."""
import os
import uuid

import psycopg
import pytest

URL = os.environ.get("TEST_DATABASE_URL", "")


@pytest.fixture
def pg(monkeypatch, tmp_path):
    if not URL:
        pytest.skip("TEST_DATABASE_URL is not set: run the tests with docker-compose.test.yml")
    from app import db, storage
    schema = "t_" + uuid.uuid4().hex[:16]
    with psycopg.connect(URL, autocommit=True) as c:
        c.execute(f"CREATE SCHEMA {schema}")
    monkeypatch.setattr(db, "DATABASE_URL", URL)
    monkeypatch.setattr(db, "DB_SCHEMA", schema)
    monkeypatch.setattr(db, "THUMBS_DIR", str(tmp_path / "thumbs"))
    storage.use(storage.LocalStore(str(tmp_path / "files")))
    try:
        yield schema
    finally:
        storage.use(None)
        db.close_pool()
        db._schema_ready.discard((URL, schema))
        with psycopg.connect(URL, autocommit=True) as c:
            c.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.fixture
def s3(pg, monkeypatch):
    """Originals in an S3 bucket (moto's, in this process) instead of the local folder."""
    moto = pytest.importorskip("moto")
    import boto3
    from app import storage
    for k, v in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(k, v)
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="library")
        store = storage.S3Store("library", "originals", client=client)
        storage.use(store)
        yield store


AZURITE = os.environ.get("TEST_AZURITE", "")


@pytest.fixture
def azure(pg):
    """Originals in an Azure Blob container on Azurite (docker-compose.test.yml), a new one
    per test, removed after it."""
    if not AZURITE:
        pytest.skip("TEST_AZURITE is not set: run the tests with docker-compose.test.yml")
    from azure.storage.blob import BlobServiceClient
    from app import storage
    name = "t" + uuid.uuid4().hex[:20]
    service = BlobServiceClient.from_connection_string(AZURITE)
    service.create_container(name)
    store = storage.AzureStore(name, "originals", connection_string=AZURITE)
    storage.use(store)
    try:
        yield store
    finally:
        service.delete_container(name)


def sql(query, params=()):
    """Run one statement on the test's schema (as db.connect does) and return its rows."""
    from app import db
    with db.connect() as con:
        cur = con.execute(query, params)
        return cur.fetchall() if cur.description else cur.rowcount
