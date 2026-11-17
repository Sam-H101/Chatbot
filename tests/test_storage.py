"""Where originals are kept (app/storage.py): the same behaviour from the local folder, an S3
bucket (moto) and an Azure Blob container (Azurite), and the app working on each."""
import hashlib

import pytest

from test_app import H, LEASE, add, client, local, photo  # noqa: F401  (client, local: the app's fixtures)

STORE_KINDS = ("local", "s3", "azure")


@pytest.fixture(params=STORE_KINDS)
def store(request, pg, tmp_path):
    """Each kind of store in turn, made the one in use."""
    from app import storage
    if request.param == "local":
        s = storage.LocalStore(str(tmp_path / "files"))
        storage.use(s)
        return s
    return request.getfixturevalue(request.param)


def test_each_store_keeps_an_original_once_and_gives_it_back(store):
    data = b"The rent is 900 euros."
    key = hashlib.sha256(data).hexdigest()
    assert not store.exists(key) and store.names() == []
    assert store.put(key, data) is True
    assert store.put(key, data) is False                                                     # once per content
    assert store.exists(key) and store.get(key) == data and store.size(key) == len(data)
    assert store.get(key, limit=8) == data[:8]                                               # the start only
    assert store.names() == [key]
    store.delete(key)
    store.delete(key)                                                                        # gone already: no error
    assert not store.exists(key) and store.names() == []
    with pytest.raises(FileNotFoundError):
        store.get(key)


def test_only_the_local_store_gives_a_path(store, tmp_path):
    store.put("a" * 64, b"x")
    path = store.local_path("a" * 64)
    assert (path == str(tmp_path / "files" / ("a" * 64))) if store.kind == "local" else path is None


def test_a_bucket_shared_with_other_data_lists_only_the_originals(s3):
    s3.client.put_object(Bucket="library", Key="elsewhere.txt", Body=b"not ours")
    s3.client.put_object(Bucket="library", Key="originals/deeper/x", Body=b"not an original")
    s3.put("b" * 64, b"ours")
    assert s3.names() == ["b" * 64]
    assert s3.client.get_object(Bucket="library", Key="originals/" + "b" * 64)["Body"].read() == b"ours"


@pytest.mark.parametrize("kind", ("s3", "azure"))
def test_the_app_keeps_opens_rereads_and_deletes_originals_in_the_cloud(client, request, kind):  # noqa: F811
    """Upload, open (sent from the app, headers and all), thumbnail, text preview, re-read and
    delete, with the originals in a bucket or container: nothing reaches for a local file."""
    from app import jobs
    store = request.getfixturevalue(kind)
    job = add(client)
    sha = hashlib.sha256(LEASE).hexdigest()
    assert job["status"] == "done" and store.exists(sha) and store.get(sha) == LEASE
    doc_id = job["doc_id"]
    f = client.get(f"/api/documents/{doc_id}/file")
    assert f.status_code == 200 and f.data == LEASE and f.headers["X-Content-Type-Options"] == "nosniff"
    assert client.get(f"/api/documents/{doc_id}/text").json["source"] == "file"
    r = client.post(f"/api/documents/{doc_id}/reread", json={"model": "auto"}, headers=H)
    assert r.status_code == 202
    jobs.drain()
    assert client.get(f"/api/jobs/{r.json['job']['id']}").json["status"] == "done"
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert not store.exists(sha)                                                              # nothing else used it


def test_a_photo_in_the_cloud_gets_its_thumbnail(local, s3, tmp_path):  # noqa: F811
    """Made by the upload; made again from the bucket's copy when it has gone (thumbnails are
    kept locally, as a cache)."""
    import shutil
    job = add(local, "receipt.jpg", photo("JPEG", (800, 600)))
    assert job["status"] == "done"
    assert local.get(f"/api/documents/{job['doc_id']}/thumb").headers["Content-Type"] == "image/webp"
    shutil.rmtree(tmp_path / "thumbs")
    t = local.get(f"/api/documents/{job['doc_id']}/thumb?size=large")
    assert t.status_code == 200 and t.headers["Content-Type"] == "image/webp"


def test_originals_kept_locally_can_be_copied_into_a_bucket(pg, tmp_path, s3):
    from app import storage
    local = storage.LocalStore(str(tmp_path / "old-files"))
    local.put("c" * 64, b"one")
    local.put("d" * 64, b"two")
    (tmp_path / "old-files" / "notes.txt").write_bytes(b"not an original")
    assert storage.copy_local(local, s3) == {"copied": 2, "skipped": 1, "failed": 0}
    assert storage.copy_local(local, s3) == {"copied": 0, "skipped": 3, "failed": 0}       # run again: nothing new
    assert s3.names() == ["c" * 64, "d" * 64]


def test_the_store_check_writes_reads_and_cleans_up(store):
    from app import storage
    assert "writing, reading and deleting work" in storage.check(store)
    assert store.names() == []


def test_a_misconfigured_store_says_what_is_missing(monkeypatch):
    from app import storage
    with pytest.raises(storage.StorageError, match="S3_BUCKET"):
        storage.from_environment("s3")
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_URL", raising=False)
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER", "library")
    with pytest.raises(storage.StorageError, match="AZURE_STORAGE_CONNECTION_STRING"):
        storage.from_environment("azure")
    with pytest.raises(storage.StorageError, match="local, s3 or azure"):
        storage.from_environment("dropbox")
