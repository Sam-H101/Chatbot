"""Thumbnails of kept originals: page 1 of a PDF, or a photo scaled down, as WebP. Made by
the job that reads an upload (jobs.py), or when first asked for (get: documents uploaded
before thumbnails, or whose thumbnail was removed). Kept in db.thumbs_dir() by the
original's hash, so copies of one file share them, and outside files_dir(), so backups
(which mirror the originals) skip them: they can always be made again. Removed with their
original (db._remove)."""
import io
import logging
import os
import tempfile
import threading

import pypdfium2
from PIL import Image, ImageOps

from . import db, extract

log = logging.getLogger(__name__)

# Target width of each size in pixels; the height is capped at twice the width (a long receipt).
SIZES = {"card": 400, "large": 1200}
KINDS = ("pdf", "image")  # documents that get one; others show a typed icon
# Renders one at a time: a page of old documents opened at once asks for every thumbnail
# together, and a large PDF page takes a moment and some memory to render.
thumbs_lock = threading.RLock()
_failed = set()  # hashes whose thumbnail could not be made, in this process: not tried again


def failed(sha):
    return sha in _failed


def path(sha, size):
    return os.path.join(db.thumbs_dir(), f"{sha}-{size}.webp")


def _render(source, kind):
    """The `large` image of a PDF's first page or a photo; source is bytes or a file path."""
    width = SIZES["large"]
    if kind == "pdf":
        pdf = pypdfium2.PdfDocument(source)
        try:
            page = pdf[0]
            w, h = page.get_size()
            scale = min(width / w, 2 * width / h, 4)  # fills the width, never past 1200x2400 px
            return page.render(scale=scale).to_pil().convert("RGB")
        finally:
            pdf.close()
    with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as im:
        if im.format == "JPEG":  # decode at a fraction of the size when that is enough
            im.draft("RGB", (width, 2 * width))
        im = extract.rgb(ImageOps.exif_transpose(im))
    im.thumbnail((width, 2 * width))
    return im


def _save(im, target):
    """WebP with no metadata, through a temporary file, so a half-written one never shows."""
    folder = os.path.dirname(target)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".thumb-")
    try:
        with os.fdopen(fd, "wb") as f:
            im.info = {}  # no EXIF, ICC profile or comments carried over
            im.save(f, format="WEBP", quality=75, method=4)
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except BaseException:
        os.unlink(tmp)
        raise


def make(sha, source, kind):
    """Make the thumbnails of an original (bytes or a file path) of this kind, unless they
    exist already (a re-read, or a copy of a file already in the library). Raises on a file
    that can't be rendered; the caller decides what that means."""
    if kind not in KINDS:
        return False
    with thumbs_lock:
        if all(os.path.exists(path(sha, s)) for s in SIZES):
            return False
        os.makedirs(db.thumbs_dir(), exist_ok=True)
        large = _render(source, kind)
        _save(large, path(sha, "large"))
        card = large.copy()
        card.thumbnail((SIZES["card"], 2 * SIZES["card"]))
        _save(card, path(sha, "card"))
    _failed.discard(sha)
    return True


def try_make(sha, source, kind):
    """make(), a failure logged and remembered for the life of the process (so it isn't tried
    on every page load, and listings show a typed icon) rather than raised. True when the
    thumbnails exist afterwards."""
    try:
        make(sha, source, kind)
    except Exception as e:
        log.info("No thumbnail for %s (%s): %s", sha[:12], kind, e)
        _failed.add(sha)
        return False
    return True


def get(sha, file_path, kind, size):
    """The path of a thumbnail, made now if it is missing; None when it can't be made."""
    target = path(sha, size)
    if os.path.exists(target):
        return target
    if sha in _failed or kind not in KINDS:
        return None
    with thumbs_lock:
        if not os.path.exists(target):  # else made while this request waited
            try_make(sha, os.path.join(db.files_dir(), os.path.basename(file_path)), kind)
    return target if os.path.exists(target) else None


def remove(sha):
    """The thumbnails of an original that is gone."""
    for size in SIZES:
        try:
            os.remove(path(sha, size))
        except FileNotFoundError:
            pass
