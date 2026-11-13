"""Turn an uploaded file into plain text, and plain text into chunks.

PDF text is kept per page: the readers return one string per page, and a document's
text joins its pages with a form feed ("\\f", see join_pages). Chunks keep those page
breaks (see chunk_text), which give each chunk's pages (page_ranges) for citations.
Photos (receipts, bills) are read by a vision model (jobs.py); here they are only checked
and made ready for it (check, prepare_image)."""
import io
import os
import re
import zipfile
from xml.etree import ElementTree

import pypdfium2
from PIL import Image, ImageOps

try:  # HEIC photos (iPhones): readable only with pillow-heif, which registers itself with Pillow
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC = True
except ImportError:
    HEIC = False

CHUNK_CHARS = 1500
# Photos of documents, read by a vision model. Uploads are told apart by their extension
# (kind); the bytes must then be one of IMAGE_FORMATS (check).
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")
MARKDOWN_EXTS = (".md", ".markdown")
# Pillow's name of each format accepted, and the type the original is stored and served as:
# that of the bytes, never what the browser said.
IMAGE_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "HEIF": "image/heic"}
# The most pixels a photo may have, read from its header before anything is decoded: a
# decompression bomb is refused at the door, and a phone's 48 MP photo still passes.
MAX_MEGAPIXELS = 50
# The longest side a photo is sent to a vision model at: Ollama's minicpm-v, and Claude,
# which scales larger images down to about this itself (and bills by the pixel).
OLLAMA_PHOTO_SIDE, CLAUDE_PHOTO_SIDE = 2000, 1568
PAGE_BREAK = "\f"
# Pages per Claude call. Each call returns its pages' text, so this bounds both
# the request size and the output length of a single call.
PDF_PAGES_PER_CALL = int(os.environ.get("PDF_PAGES_PER_CALL", "20"))
# Bytes per piece. Base64 adds a third, so 20 MB becomes about 27 MB: under the
# Claude API's 32 MB request limit.
PDF_MAX_BATCH_BYTES = int(float(os.environ.get("PDF_MAX_BATCH_MB", "20")) * 1024 * 1024)
# Pages whose text layer has fewer visible characters than this are treated as scans:
# only they go to the vision model (Ollama) or, by default, to Claude.
MIN_PAGE_CHARS = int(os.environ.get("MIN_PAGE_CHARS", "20"))
RENDER_SCALE = float(os.environ.get("RENDER_SCALE", "2"))  # 2 = 144 dpi


class Unsupported(ValueError):
    """Not a file this app can read: the user's mistake, not the server's."""


def kind(filename):
    """What sort of document a file is, by its extension: 'pdf', 'image' (a photo, read by a
    vision model), 'markdown', 'docx', or 'text' (anything else, read as UTF-8 text)."""
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return "pdf"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in MARKDOWN_EXTS:
        return "markdown"
    return "docx" if ext == ".docx" else "text"


def extract_text(filename, data):
    """Local extraction. PDFs are normally read by a model instead (see jobs.py). Only
    PDFs have pages: a form feed in other text (a printout, ^L in code) is a line break.
    Photos have no text to extract here: a vision model reads them (jobs.py)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return join_pages(pdf_texts(data))
    if kind(name) == "image":
        raise Unsupported("A photo is read by a vision model, not as text")
    if name.endswith(".docx"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                root = ElementTree.fromstring(z.read("word/document.xml"))
        except (zipfile.BadZipFile, KeyError, ElementTree.ParseError):
            raise Unsupported("Could not open this DOCX file: it is damaged or not a Word document") from None
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        text = "\n".join(
            "".join(t.text or "" for t in p.iter(ns + "t")) for p in root.iter(ns + "p")
        )
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise Unsupported("Unsupported file: expected PDF, DOCX or UTF-8 text") from None
    return text.replace(PAGE_BREAK, "\n")


def check(filename, data):
    """Raise Unsupported for a file that can never be read, before it is queued: a PDF
    that doesn't open or has no pages, a photo that isn't one (check_image), a damaged
    DOCX, text that isn't UTF-8, or no text at all. Quick: a PDF is only opened, as its
    pages may need a model. Returns the type to store a photo as (that of its bytes), else
    None."""
    if filename.lower().endswith(".pdf"):
        try:
            pages = page_count(data)
        except pypdfium2.PdfiumError as e:
            raise Unsupported(f"Could not open this PDF: {e}") from None
        if not pages:
            raise Unsupported("This PDF has no pages")
    elif kind(filename) == "image":
        return check_image(filename, data)
    elif not extract_text(filename, data).strip():  # all whitespace: chunk_text would find nothing
        raise Unsupported("No text found in file")
    return None


def check_image(filename, data):
    """The stored type of a photo (IMAGE_FORMATS) whose bytes are a JPEG, PNG, WebP or HEIC
    image of at most MAX_MEGAPIXELS; Unsupported otherwise. Its size comes from the header,
    before anything is decoded; verify() then reads the rest for damage."""
    if not HEIC and filename.lower().endswith((".heic", ".heif")):
        raise Unsupported("HEIC photos aren't supported on this server")
    damaged = "Could not open this photo: it is damaged or not a JPG, PNG, WebP or HEIC image"
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in IMAGE_FORMATS or (im.format == "HEIF" and not HEIC):
                raise Unsupported(damaged)
            megapixels = im.width * im.height / 1e6
            if megapixels > MAX_MEGAPIXELS:
                raise Unsupported(f"This photo is too large ({megapixels:.0f} megapixels); {MAX_MEGAPIXELS} at most")
            fmt = im.format
            im.verify()
    except Image.DecompressionBombError:  # Pillow's own limit, far above ours
        raise Unsupported(f"This photo is too large; {MAX_MEGAPIXELS} megapixels at most") from None
    except Unsupported:
        raise
    except Exception:  # Pillow raises many kinds for a file that isn't an image it reads
        raise Unsupported(damaged) from None
    return IMAGE_FORMATS[fmt]


def rgb(im):
    """An image as RGB, transparent parts on white (a PNG screenshot of a bill, say), not black."""
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        out = Image.new("RGB", im.size, "white")
        out.paste(im, mask=im.getchannel("A"))
        return out
    return im.convert("RGB")


def image_size(data):
    """(width, height) of a photo as a person sees it (EXIF rotation applied), from its header."""
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        orientation = im.getexif().get(0x0112, 1)  # 5-8: turned a quarter, so sides swap
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


def prepare_image(data, max_side):
    """A photo as a vision model is sent it: upright (EXIF), RGB, its longest side at most
    `max_side` pixels, as JPEG (quality 90) bytes."""
    with Image.open(io.BytesIO(data)) as im:
        if im.format == "JPEG":  # decode at a fraction of the size when that is enough
            im.draft("RGB", (max_side, max_side))
        im = rgb(ImageOps.exif_transpose(im))
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def join_pages(pages):
    """One text for a document from one string per page. A form feed inside a page
    would read as a page break, so it becomes a line break."""
    return PAGE_BREAK.join(p.replace(PAGE_BREAK, "\n") for p in pages)


def pdf_texts(data):
    """The embedded text layer of every page: no OCR, so scanned pages come back
    (nearly) empty. Quick, since nothing is rendered."""
    pdf = pypdfium2.PdfDocument(data)
    try:
        return [page.get_textpage().get_text_bounded() for page in pdf]
    finally:
        pdf.close()


def page_count(data):
    pdf = pypdfium2.PdfDocument(data)
    try:
        return len(pdf)
    finally:
        pdf.close()


def is_scan(text, min_chars=MIN_PAGE_CHARS):
    """A page whose text layer has fewer than `min_chars` visible characters."""
    return len("".join(text.split())) < min_chars


def pdf_pages(data, min_chars=MIN_PAGE_CHARS, scale=RENDER_SCALE, skip=()):
    """Per page, one at a time: (text_layer, png_bytes or None). A page is rendered to
    PNG only when it is a scan (is_scan) and its number (from 1) is not in `skip`.
    Rendering as the caller goes keeps a 300-page scan out of memory."""
    pdf = pypdfium2.PdfDocument(data)
    try:
        for n, page in enumerate(pdf, 1):
            text = page.get_textpage().get_text_bounded()
            png = None
            if n not in skip and is_scan(text, min_chars):
                buf = io.BytesIO()
                page.render(scale=scale).to_pil().save(buf, format="PNG")
                png = buf.getvalue()
            yield text, png
    finally:
        pdf.close()


def pdf_batches(data, size=PDF_PAGES_PER_CALL, max_bytes=None, pages=None):
    """Split a PDF into (page_numbers, total_pages, pdf_bytes) pieces of at most `size`
    pages and `max_bytes` bytes; page numbers count from 1 in the original. `pages`
    limits the pieces to those pages (say only the scanned ones): each piece is then a
    new PDF of just its pages. A piece that is too big is halved until it fits, so
    heavy scans still fit in one Claude request."""
    max_bytes = PDF_MAX_BATCH_BYTES if max_bytes is None else max_bytes
    src = pypdfium2.PdfDocument(data)
    try:
        total = len(src)
        every = list(range(1, total + 1))
        pages = every if pages is None else sorted(set(pages))
        if pages == every and total <= size and len(data) <= max_bytes:
            return [(pages, total, data)]
        batches = []

        def add(part):
            new = pypdfium2.PdfDocument.new()
            new.import_pages(src, [p - 1 for p in part])
            buf = io.BytesIO()
            new.save(buf)
            new.close()
            piece = buf.getvalue()
            if len(piece) <= max_bytes:
                batches.append((part, total, piece))
            elif len(part) > 1:
                mid = len(part) // 2
                add(part[:mid])
                add(part[mid:])
            else:
                raise ValueError(f"page {part[0]} alone is {len(piece) / 1e6:.0f} MB, "
                                 "too large to send to Claude in one request")

        for start in range(0, len(pages), size):
            add(pages[start:start + size])
        return batches
    finally:
        src.close()


def chunk_text(text, size=CHUNK_CHARS):
    """Split on paragraph boundaries into chunks of roughly `size` characters.

    A page break ("\\f") also ends a paragraph, and it is kept rather than dropped as
    whitespace: a chunk that starts a new page begins with "\\f", and one that runs onto
    the next page has "\\f" where it does. Counting the form feeds before a spot in a
    document's chunks therefore gives its page (page_ranges). Empty pages still count."""
    chunks, current, breaks = [], "", 0
    for n, page in enumerate(text.split(PAGE_BREAK)):
        breaks += n > 0
        for para in re.split(r"\n\s*\n", page):
            para = para.strip()
            if not para:
                continue
            para, breaks = PAGE_BREAK * breaks + para, 0
            while len(para) > size:  # a single huge paragraph: hard split
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(para[:size])
                para = para[size:]
            if current and len(current) + len(para) + 2 > size:
                chunks.append(current)
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def page_ranges(chunks):
    """(first page, last page) of each chunk of a paged text (a PDF's), from 1. chunk_text
    keeps every page break, and a chunk that opens a page starts with its breaks, so the
    breaks before a chunk and at its start give the page its text starts on, and those in
    it the page it ends on: a chunk running onto the next pages gets a range."""
    ranges, page = [], 1
    for c in chunks:
        first = page + len(c) - len(c.lstrip(PAGE_BREAK))
        page += c.count(PAGE_BREAK)
        ranges.append((first, page))
    return ranges


def plain(text):
    """Text for a model or a person to read: page breaks become paragraph breaks."""
    return re.sub(r"\s*\f\s*", "\n\n", text).strip() if PAGE_BREAK in text else text
