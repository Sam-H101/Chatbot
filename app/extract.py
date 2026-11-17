"""Turn an uploaded file into plain text, and plain text into chunks.

PDF text is kept per page: the readers return one string per page, and a document's
text joins its pages with a form feed ("\\f", see join_pages). Chunks keep those page
breaks (see chunk_text), which give each chunk's pages (page_ranges) for citations.
Photos (receipts, bills) are read by a vision model (jobs.py); here they are only checked
and made ready for it (check, prepare_image)."""
import html
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
# that of the bytes, never what the browser said. MPO is a JPEG carrying more pictures after
# the first (an iPhone's HDR gain map or depth map, many Android and camera JPEGs): browsers
# show it as the JPEG it is, and only its first picture is read.
IMAGE_FORMATS = {"JPEG": "image/jpeg", "MPO": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp",
                 "HEIF": "image/heic"}
JPEG_FORMATS = ("JPEG", "MPO")  # decoded at a fraction of their size when that is enough (draft)
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
            text = data.decode("utf-8-sig")  # a BOM (Windows PowerShell, Visual Studio) is not text
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
        if im.format in JPEG_FORMATS:  # decode at a fraction of the size when that is enough
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


# Control characters that aren't white space (form feeds, line breaks and tabs are).
_CONTROLS = {c: None for c in (*range(0x09), *range(0x0e, 0x1c), *range(0x7f, 0xa0)) if not chr(c).isspace()}
# pdfium joins a word that a hyphen split at the end of a line and gives "\x02" for that hyphen
# ("hy\x02draulic"). Indexed so, the word was two ("hy", "draulic") and search never found it.
# The hyphen goes, except after a one-letter part ("O-ring") or beside a digit ("DK45-series"):
# those are the word's own.
_OWN_HYPHEN = re.compile(r"(?<!\w)(\w)\x02|(?<=\d)\x02|\x02(?=\d)")


def _text_layer(page):
    """A page's text layer, its words whole again and no control characters."""
    text = page.get_textpage().get_text_bounded()
    return _OWN_HYPHEN.sub(lambda m: (m[1] or "") + "-", text).translate(_CONTROLS)


def pdf_texts(data):
    """The embedded text layer of every page: no OCR, so scanned pages come back
    (nearly) empty. Quick, since nothing is rendered."""
    pdf = pypdfium2.PdfDocument(data)
    try:
        return [_text_layer(page) for page in pdf]
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
            text = _text_layer(page)
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


# ---- search snippets: a passage as a person reads it in a list of results ----
# Models write the pages they read as Markdown, with HTML in table cells ("adjusted<br>- Brake
# disc worn"), and a snippet showed all of it. Cleaning works on the whole snippet with the
# matches marked by these two noncharacters (any in the text are dropped first), so a heading,
# a bullet or **bold** is still seen around a match.
_OPEN, _CLOSE = chr(0xFDD0), chr(0xFDD1)
# Dropped from the text with them: control characters that aren't white space, such as the \x02
# pdfium gave for a hyphen that split a word at the end of a line ("facto\x02ry-set"), which
# documents read before _text_layer still have.
_DROPPED = {ord(_OPEN): None, ord(_CLOSE): None, **_CONTROLS}
_MARKS = re.compile(f"([{_OPEN}{_CLOSE}])")
# Only these are tags; any other <...> is text: a manual's figure caption ("<Power flow diagram>",
# "<The Condition of not Engaged the Diff. Lock Gear>"), a key ("<ENTER>"), a grade ("<SAE 10W-30>").
# A name of one letter is a tag in lower case only, so a label such as "<A>" or "<B> side" stays.
_TAG_NAMES = r"(?:(?i:br|hr|div|span|font|table|thead|tbody|tr|td|th|ul|ol|li|em|strong|sup|sub|small|img|h[1-6])|[abipsu])"
_NAME_END = rf"(?=[\s/>{_OPEN}{_CLOSE}])"  # "<li>", not "<Li-ion battery>"
_LINE_TAG = re.compile(rf"<\s*/?\s*(?:(?i:br|hr|div|tr|li|h[1-6])|p){_NAME_END}[^<>]*>")  # tags that end a line
_TAG = re.compile(rf"<!--.*?-->|</?{_OPEN}?{_TAG_NAMES}{_CLOSE}?{_NAME_END}[^<>]*>", re.S)
# A tag the snippet's window cut in two: its end at the start ("br> Brake"), its start at the end.
_TAG_TAIL = re.compile(rf"^\s*/?{_OPEN}?{_TAG_NAMES}{_CLOSE}?(?:\s+[A-Za-z-]+=[^<>]*)?\s*/?>")
_TAG_HEAD = re.compile(rf"<{_OPEN}?/?{_OPEN}?{_TAG_NAMES}{_CLOSE}?(?:\s[^<>]*)?\s*$")
_CUT = re.compile(r"(\s*…)?(.*?)(…\s*)?", re.S)  # the ellipses snippet() adds where it cut the text
_FENCE = re.compile(r"^[^\S\n]*(?:```|~~~).*$", re.M)
_QUOTED = re.compile(r"^(?:[^\S\n]*>)+", re.M)
_HEADING = re.compile(r"^[^\S\n]*#{1,6}(?=\s|$)", re.M)
_BULLET = re.compile(r"^[^\S\n]*[-*+•](?:[^\S\n]+\[[ xX]\])?(?=\s|$)", re.M)
_RULE = re.compile(r"[\s|:=*_-]*[-=*_][\s|:=*_-]*")  # a table's |---|---| row, or ---, *** or ===
_LINK = re.compile(r"!?\[([^\]\n]*)\]\([^()\s]*\)")
_EMPHASIS = re.compile(r"\*+|_+|~~+")
_SEP = " · "  # between a table's cells and rows, and the lines of a cell


def _emphasis(m):
    """An emphasis mark gone (**, *, __, _, ~~), the match marks aside, unless it stands alone
    between spaces (5 * 3), joins a word (snake_case) or two numbers or words with one star
    (5*3, an approval number's e11*89/173*2000), or is a long run (a form's ______)."""
    s, a, b, mark = m.string, m.start(), m.end(), m.group()
    while a and s[a - 1] in (_OPEN, _CLOSE):
        a -= 1
    while b < len(s) and s[b] in (_OPEN, _CLOSE):
        b += 1
    before, after = s[a - 1] if a else " ", s[b] if b < len(s) else " "
    if len(mark) > 3 or (before.isspace() and after.isspace()) or (
            (mark[0] == "_" or mark == "*") and before.isalnum() and after.isalnum()):
        return mark
    return ""


def _rows(text):
    """Lines of a Markdown table as their cells, _SEP between them and between rows, and
    between the lines of a cell (<br>, a list's bullets dropped); its |---| rows and other
    rules (---, ***) are dropped."""
    out, row = [], False
    for line in text.split("\n"):
        if _RULE.fullmatch(line) and ("|" in line or len(line.strip()) >= 3):
            continue
        if "|" in line:
            items = [_BULLET.sub("", item).strip() for cell in line.strip().strip("|").split("|")
                     for item in _LINE_TAG.split(cell)]
            line = _SEP.join(i for i in items if i)
            if row and out and line:
                out[-1] += _SEP + line
                continue
            row = row or bool(line)
        else:
            row = False
        out.append(line)
    return "\n".join(out)


def _balanced(text):
    """The text with only match marks that open and close in turn."""
    out, inside = [], False
    for piece in _MARKS.split(text):
        if piece in (_OPEN, _CLOSE):
            if (piece == _OPEN) == inside:
                continue
            inside = not inside
        out.append(piece)
    return "".join(out) + (_CLOSE if inside else "")


def snippet_parts(snippet, start, end):
    """A search snippet that marks each match between `start` and `end` (db.search_chunks), as
    plain text: [text, match, text, match, ..., text], always starting and ending with text
    (perhaps ""), the matches at odd indexes, as JavaScript's str.split(/(re)/) gives them. No
    Markdown (headings, emphasis, inline code, table pipes and |---| rows, bullets, quotes,
    links), no HTML (tags become spaces or line breaks, entities their characters) and the
    white space collapsed, so a page can show the parts as text, the matches highlighted. A
    passage's own brackets ("[see figure 3]") are text like any other."""
    marked, inside = [], False
    for piece in re.split(f"({re.escape(start)}|{re.escape(end)})", snippet):
        if piece in (start, end):
            if (piece == start) != inside:
                marked.append(_OPEN if piece == start else _CLOSE)
                inside = not inside
        else:  # entities first, so none can make a mark (or a control character) either
            marked.append(html.unescape(piece).translate(_DROPPED))
    cut = _CUT.fullmatch(re.sub(r"\r\n?|\f", "\n", "".join(marked) + (_CLOSE if inside else "")))
    text = _TAG_HEAD.sub("", _TAG_TAIL.sub("", cut[2]))
    text = _BULLET.sub("", _HEADING.sub("", _QUOTED.sub("", _FENCE.sub("", text))))
    text = _BULLET.sub("", _LINE_TAG.sub("\n", _rows(text)))  # what a <br> starts outside a table, too
    text = _TAG.sub(" ", text)
    text = _EMPHASIS.sub(_emphasis, _LINK.sub(r"\1", text)).replace("`", "")
    text = _balanced(text)
    text = re.sub(rf"{_OPEN}(\s+)", rf"\1{_OPEN}", re.sub(rf"(\s+){_CLOSE}", rf"{_CLOSE}\1", text))
    text = " ".join(text.replace(_OPEN + _CLOSE, "").split())
    if text:
        text = ("… " if cut[1] else "") + text + (" …" if cut[3] else "")
    parts = [""]
    for piece in _MARKS.split(text):
        if piece in (_OPEN, _CLOSE):
            parts.append("")
        else:
            parts[-1] += piece
    return parts
