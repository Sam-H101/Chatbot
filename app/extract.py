"""Turn an uploaded file into plain text, and plain text into chunks.

PDF text is kept per page: the readers return one string per page, and a document's
text joins its pages with a form feed ("\\f", see join_pages). Chunks keep those page
breaks (see chunk_text), so a chunk's page numbers can be worked out later."""
import io
import os
import re
import zipfile
from xml.etree import ElementTree

import pypdfium2

CHUNK_CHARS = 1500
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


def extract_text(filename, data):
    """Local extraction. PDFs are normally read by a model instead (see jobs.py)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return join_pages(pdf_texts(data))
    if name.endswith(".docx"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        return "\n".join(
            "".join(t.text or "" for t in p.iter(ns + "t")) for p in root.iter(ns + "p")
        )
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise Unsupported("Unsupported file: expected PDF, DOCX or UTF-8 text") from None


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
    document's chunks therefore gives its page. Empty pages still count."""
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


def plain(text):
    """Text for a model or a person to read: page breaks become paragraph breaks."""
    return re.sub(r"\s*\f\s*", "\n\n", text).strip() if PAGE_BREAK in text else text
