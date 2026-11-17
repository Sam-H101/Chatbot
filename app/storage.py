"""Where original uploads are kept: one object per distinct content, named by its SHA-256.

STORAGE_BACKEND picks the store:
  local  (default) files in FILES_DIR (/data/files in the container)
  s3     an S3 bucket (S3_BUCKET), or any S3-compatible store with S3_ENDPOINT_URL (MinIO,
         Cloudflare R2, Backblaze B2). Credentials come from boto3's usual chain: the
         AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables, a profile, or an
         instance role. AWS_REGION / S3_REGION names the region.
  azure  an Azure Blob Storage container (AZURE_STORAGE_CONTAINER), reached with
         AZURE_STORAGE_CONNECTION_STRING, or AZURE_STORAGE_ACCOUNT_URL plus
         AZURE_STORAGE_KEY (or a SAS token in AZURE_STORAGE_SAS_TOKEN).
S3_PREFIX / AZURE_STORAGE_PREFIX put the objects under a folder of the bucket or container,
so one can be shared with other data.

Objects are written once and never changed: a name is its content's hash, so storing the
same bytes again is a no-op, and a half-written object never shows under its name (a local
file is written to a temporary name and renamed; S3 and Azure make an upload visible only
when it completes).

    python -m app.storage --check         say which store is in use, and that it can be written
    python -m app.storage --copy-local    copy the originals in FILES_DIR into the store in use
"""
import argparse
import logging
import os
import re
import sys
import tempfile
import threading

log = logging.getLogger(__name__)

BACKEND = (os.environ.get("STORAGE_BACKEND") or "local").strip().lower()
FILES_DIR = os.environ.get("FILES_DIR", "") or "/data/files"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
_store = None
_store_lock = threading.Lock()


class StorageError(Exception):
    """The store is misconfigured or can't be reached; the message says how."""


def name(file_path):
    """A stored original's name: its documents' and jobs' file_path is the name itself (older
    libraries kept a path ending in it)."""
    return os.path.basename(file_path or "")


class LocalStore:
    kind = "local"

    def __init__(self, root):
        self.root = root

    def describe(self):
        return f"the folder {self.root}"

    def _path(self, key):
        return os.path.join(self.root, name(key))

    def local_path(self, key):
        path = self._path(key)
        return path if os.path.isfile(path) else None

    def exists(self, key):
        return os.path.isfile(self._path(key))

    def put(self, key, data):
        path = self._path(key)
        if os.path.exists(path):
            return False
        os.makedirs(self.root, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".upload-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.chmod(tmp, 0o644)  # mkstemp makes it private to the writer
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
        return True

    def get(self, key, limit=None):
        with open(self._path(key), "rb") as f:
            return f.read() if limit is None else f.read(limit)

    def size(self, key):
        return os.path.getsize(self._path(key))

    def delete(self, key):
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass

    def names(self):
        try:
            return sorted(n for n in os.listdir(self.root) if not n.startswith("."))  # .upload-*: being written
        except FileNotFoundError:
            return []


class S3Store:
    kind = "s3"

    def __init__(self, bucket, prefix="", endpoint=None, region=None, client=None):
        if not bucket:
            raise StorageError("STORAGE_BACKEND=s3 needs S3_BUCKET")
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client("s3", endpoint_url=endpoint or None, region_name=region or None,
                                  config=Config(retries={"max_attempts": 5, "mode": "standard"}))
        self.client, self.bucket, self.prefix = client, bucket, _prefix(prefix)

    def describe(self):
        return f"the S3 bucket {self.bucket}" + (f" under {self.prefix}" if self.prefix else "")

    def _key(self, key):
        return self.prefix + name(key)

    def local_path(self, key):
        return None

    def exists(self, key):
        from botocore.exceptions import ClientError
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def put(self, key, data):
        if self.exists(key):
            return False
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)
        return True

    def get(self, key, limit=None):
        extra = {"Range": f"bytes=0-{limit - 1}"} if limit else {}
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(key), **extra)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            raise FileNotFoundError(key) from None

    def size(self, key):
        return self.client.head_object(Bucket=self.bucket, Key=self._key(key))["ContentLength"]

    def delete(self, key):
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def names(self):
        out = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self.prefix):
            out += [o["Key"][len(self.prefix):] for o in page.get("Contents", ())]
        return sorted(n for n in out if n and "/" not in n)


class AzureStore:
    kind = "azure"

    def __init__(self, container, prefix="", connection_string=None, account_url=None, credential=None, client=None):
        if not container:
            raise StorageError("STORAGE_BACKEND=azure needs AZURE_STORAGE_CONTAINER")
        if client is None:
            from azure.storage.blob import BlobServiceClient
            if connection_string:
                service = BlobServiceClient.from_connection_string(connection_string)
            elif account_url:
                service = BlobServiceClient(account_url, credential=credential)
            else:
                raise StorageError("STORAGE_BACKEND=azure needs AZURE_STORAGE_CONNECTION_STRING, or "
                                   "AZURE_STORAGE_ACCOUNT_URL with AZURE_STORAGE_KEY or AZURE_STORAGE_SAS_TOKEN")
            client = service.get_container_client(container)
        self.client, self.container, self.prefix = client, container, _prefix(prefix)

    def describe(self):
        return f"the Azure container {self.container}" + (f" under {self.prefix}" if self.prefix else "")

    def _blob(self, key):
        return self.client.get_blob_client(self.prefix + name(key))

    def local_path(self, key):
        return None

    def exists(self, key):
        return self._blob(key).exists()

    def put(self, key, data):
        from azure.core.exceptions import ResourceExistsError
        try:
            self._blob(key).upload_blob(data, overwrite=False)
        except ResourceExistsError:
            return False
        return True

    def get(self, key, limit=None):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            return self._blob(key).download_blob(offset=0 if limit else None, length=limit or None).readall()
        except ResourceNotFoundError:
            raise FileNotFoundError(key) from None

    def size(self, key):
        return self._blob(key).get_blob_properties().size

    def delete(self, key):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            self._blob(key).delete_blob()
        except ResourceNotFoundError:
            pass

    def names(self):
        out = [b.name[len(self.prefix):] for b in self.client.list_blobs(name_starts_with=self.prefix or None)]
        return sorted(n for n in out if n and "/" not in n)


def _prefix(raw):
    raw = (raw or "").strip().strip("/")
    return raw + "/" if raw else ""


def from_environment(backend=None):
    """The store the environment describes (BACKEND unless `backend` is given)."""
    backend = (backend or BACKEND).strip().lower()
    env = os.environ.get
    if backend == "local":
        return LocalStore(FILES_DIR)
    if backend == "s3":
        return S3Store(env("S3_BUCKET", ""), env("S3_PREFIX", ""), env("S3_ENDPOINT_URL"),
                       env("S3_REGION") or env("AWS_REGION") or env("AWS_DEFAULT_REGION"))
    if backend == "azure":
        return AzureStore(env("AZURE_STORAGE_CONTAINER", ""), env("AZURE_STORAGE_PREFIX", ""),
                          env("AZURE_STORAGE_CONNECTION_STRING"), env("AZURE_STORAGE_ACCOUNT_URL"),
                          env("AZURE_STORAGE_KEY") or env("AZURE_STORAGE_SAS_TOKEN") or None)
    raise StorageError(f"STORAGE_BACKEND must be local, s3 or azure, not {backend!r}")


def store():
    """The store in use, made when first needed."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = from_environment()
    return _store


def use(new):
    """Use this store from now on (tests; None: the environment's again)."""
    global _store
    _store = new


# The module's interface, on the store in use.
def exists(key):
    return store().exists(key)


def put(key, data):
    """Keep `data` as `key`, unless it is there already; True if this call stored it."""
    return store().put(key, data)


def get(key, limit=None):
    """The bytes of `key` (the first `limit` of them, if given). FileNotFoundError if missing."""
    return store().get(key, limit)


def size(key):
    return store().size(key)


def delete(key):
    """Remove `key`; nothing happens if it is missing."""
    store().delete(key)


def local_path(key):
    """`key` as a file on this machine (the local store), or None (S3, Azure, or missing)."""
    return store().local_path(key)


def names():
    """Every stored original's name, sorted."""
    return store().names()


def source(key):
    """What a reader of the original takes (pypdfium2, Pillow): its local path when there is
    one, else its bytes."""
    return local_path(key) or get(key)


def copy_local(src=None, dest=None):
    """Copy the originals in a local folder (FILES_DIR) into `dest` (the store in use) that it
    doesn't have yet: run once after switching STORAGE_BACKEND, so documents kept so far can
    still be opened and re-read. Returns {copied, skipped, failed}."""
    src = src or LocalStore(FILES_DIR)
    dest = dest or store()
    result = {"copied": 0, "skipped": 0, "failed": 0}
    for key in src.names():
        if not SHA256.match(key):
            result["skipped"] += 1
            continue
        try:
            if dest.put(key, src.get(key)):
                result["copied"] += 1
            else:
                result["skipped"] += 1
        except Exception as e:
            log.error("Could not copy %s: %s", key, e)
            result["failed"] += 1
    return result


def check(dest=None):
    """Write, read back and remove a small object, to show the store works. Returns a message."""
    dest = dest or store()
    key = ".storage-check"
    data = b"chatbot storage check\n"
    dest.delete(key)
    dest.put(key, data)
    got = dest.get(key)
    dest.delete(key)
    if got != data:
        raise StorageError(f"Read back something else from {dest.describe()}")
    return f"Originals are kept in {dest.describe()}: writing, reading and deleting work"


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m app.storage", description="Where original uploads are kept.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="show the store in use and check it can be written")
    g.add_argument("--copy-local", action="store_true", help="copy the originals in FILES_DIR into the store in use")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    try:
        if args.check:
            print(check())
            return 0
        if store().kind == "local":
            print(f"STORAGE_BACKEND is local: the originals are already in {FILES_DIR}")
            return 0
        r = copy_local()
        print(f"Copied {r['copied']} original(s) from {FILES_DIR} into {store().describe()};"
              f" {r['skipped']} already there or not originals, {r['failed']} failed")
        return 1 if r["failed"] else 0
    except StorageError as e:
        print(e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
