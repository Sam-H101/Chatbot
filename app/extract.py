"""Turn an uploaded file into plain text, and plain text into chunks."""
import io
import re
import zipfile
from xml.etree import ElementTree

from pypdf import PdfReader

CHUNK_CHARS = 1500


def extract_text(filename, data):
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
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
