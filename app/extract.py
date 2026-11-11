"""Turn an uploaded file into plain text, and plain text into chunks."""
import io
import os
import re
import zipfile
from xml.etree import ElementTree

import pypdfium2

CHUNK_CHARS = 1500
# Pages per Claude call. Each call returns its pages' text, so this bounds both
# the request size and the output length of a single call.
PDF_PAGES_PER_CALL = int(os.environ.get("PDF_PAGES_PER_CALL", "20"))
# Bytes per piece. Base64 adds a third, so 20 MB becomes about 27 MB: under the
# Claude API's 32 MB request limit.
PDF_MAX_BATCH_BYTES = int(float(os.environ.get("PDF_MAX_BATCH_MB", "20")) * 1024 * 1024)
# Local (Ollama) PDF reading: pages whose text layer has fewer visible characters
# than this are treated as scans and rendered to an image for the vision model.
MIN_PAGE_CHARS = int(os.environ.get("MIN_PAGE_CHARS", "20"))
RENDER_SCALE = float(os.environ.get("RENDER_SCALE", "2"))  # 2 = 144 dpi


def extract_text(filename, data):
    """Local extraction. PDFs normally go to Claude instead (see llm.transcribe_pdf)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return pdf_text_layer(data)
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
        raise ValueError("Unsupported file: expected PDF, DOCX or UTF-8 text")


def pdf_text_layer(data):
    """The PDF's embedded text only: no OCR, so scanned pages come back empty."""
    pdf = pypdfium2.PdfDocument(data)
    try:
        return "\n\n".join(page.get_textpage().get_text_bounded() for page in pdf)
    finally:
        pdf.close()


def pdf_pages(data, min_chars=MIN_PAGE_CHARS, scale=RENDER_SCALE):
    """Per page: (text_layer, png_bytes or None). A page is rendered to PNG only when
    its text layer has fewer than `min_chars` visible characters, i.e. it is a scan."""
    pdf = pypdfium2.PdfDocument(data)
    try:
        pages = []
        for page in pdf:
            text = page.get_textpage().get_text_bounded()
            png = None
            if len("".join(text.split())) < min_chars:
                buf = io.BytesIO()
                page.render(scale=scale).to_pil().save(buf, format="PNG")
                png = buf.getvalue()
            pages.append((text, png))
        return pages
    finally:
        pdf.close()


def pdf_batches(data, size=PDF_PAGES_PER_CALL, max_bytes=None):
    """Split a PDF into (first_page, last_page, total_pages, pdf_bytes) pieces of at most
    `size` pages and `max_bytes` bytes. A piece that is too big is halved until it fits,
    so heavy scans still fit in one Claude request."""
    max_bytes = PDF_MAX_BATCH_BYTES if max_bytes is None else max_bytes
    src = pypdfium2.PdfDocument(data)
    try:
        total = len(src)
        if total <= size and len(data) <= max_bytes:
            return [(1, total, total, data)]
        batches = []

        def add(start, end):  # pages start..end-1
            part = pypdfium2.PdfDocument.new()
            part.import_pages(src, list(range(start, end)))
            buf = io.BytesIO()
            part.save(buf)
            part.close()
            piece = buf.getvalue()
            if len(piece) <= max_bytes:
                batches.append((start + 1, end, total, piece))
            elif end - start > 1:
                mid = (start + end) // 2
                add(start, mid)
                add(mid, end)
            else:
                raise ValueError(f"page {start + 1} alone is {len(piece) / 1e6:.0f} MB, "
                                 "too large to send to Claude in one request")

        for start in range(0, total, size):
            add(start, min(start + size, total))
        return batches
    finally:
        src.close()


def chunk_text(text, size=CHUNK_CHARS):
    """Split on paragraph boundaries into chunks of roughly `size` characters."""
    chunks, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
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
