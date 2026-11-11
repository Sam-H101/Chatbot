"""Local models through Ollama: the same jobs as llm.py, with no API key and
nothing leaving the machine. Talks to Ollama's HTTP API with the standard library."""
import base64
import json
import os
import urllib.error
import urllib.request

from . import extract
from .llm import META_SCHEMA, SYSTEM

URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
# Reads scanned PDF pages. qwen3.5 also accepts images, so setting this to MODEL
# runs everything on one model (less VRAM juggling, slower page reading).
VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "minicpm-v4.6")
# Context window per request. On an 11 GB card a 9B model leaves room for ~16K;
# raise it on bigger GPUs. Ollama silently drops the oldest tokens past this.
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
# Characters of a document sent for metadata extraction. 40,000 is about 11-13K tokens,
# which fits NUM_CTX with room for the reply. Raise both together: past NUM_CTX Ollama
# silently cuts the text instead of failing. Longer documents are marked "partial".
ANALYZE_CHARS = int(os.environ.get("OLLAMA_ANALYZE_CHARS", "40000"))
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))

PAGE_PROMPT = (
    "This is page {page} of a {total}-page document. Transcribe all of its text in "
    "reading order. Render tables as Markdown tables and headings as Markdown headings. "
    "For charts, diagrams and photos, write a one-line description in [square brackets], "
    "including any numbers they show. Output only the transcription."
)


class OllamaError(Exception):
    """Ollama answered with an error (e.g. model not pulled)."""


class OllamaUnavailable(OllamaError):
    """Ollama could not be reached at all."""


def _chat(model, messages, think=None, temperature=None, **extra):
    options = {"num_ctx": NUM_CTX}
    if temperature is not None:
        options["temperature"] = temperature
    body = {"model": model, "messages": messages, "stream": False,
            "keep_alive": KEEP_ALIVE, "options": options, **extra}
    if think is not None:  # only sent to models that support thinking
        body["think"] = think
    req = urllib.request.Request(f"{URL}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # before URLError: it is a subclass
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except ValueError:
            pass
        raise OllamaError(f"{model}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise OllamaUnavailable(f"Could not reach Ollama at {URL}") from e


def models():
    """Names of the models pulled into Ollama."""
    req = urllib.request.Request(f"{URL}/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return sorted(m["name"] for m in json.load(r).get("models", []))
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
        raise OllamaUnavailable(f"Could not reach Ollama at {URL}") from e


def fits(messages, reply_tokens=2000):
    """Rough check that a prompt fits the local context window (about 3 characters
    per token, erring on the side of switching to Claude)."""
    chars = sum(len(m.get("content") or "") for m in messages)
    return chars / 3 + reply_tokens <= NUM_CTX


def analyze(_client, filename, text, model=MODEL):
    """Return (metadata dict, 'full' | 'partial'). Same contract as llm.analyze."""
    excerpt = text[:ANALYZE_CHARS]
    r = _chat(model, [{
        "role": "user",
        "content": f"Extract metadata for the document below.\n\nFile name: {filename}\n\n"
                   f"<document>\n{excerpt}\n</document>",
    }], think=False, temperature=0, format=META_SCHEMA)
    meta = json.loads(r["message"]["content"])
    return meta, "full" if len(text) <= ANALYZE_CHARS else "partial"


def transcribe_pdf(data):
    """Embedded text where the PDF has it; the vision model reads only scanned pages."""
    pages = extract.pdf_pages(data)
    out = []
    for n, (text, png) in enumerate(pages, 1):
        if png is None:
            out.append(text)
            continue
        r = _chat(VISION_MODEL, [{
            "role": "user",
            "content": PAGE_PROMPT.format(page=n, total=len(pages)),
            "images": [base64.standard_b64encode(png).decode("ascii")],
        }], temperature=0)
        out.append(r["message"]["content"])
    return "\n\n".join(out)


def chat(_client, history, user_message, model=MODEL, system=SYSTEM):
    """Same contract as llm.chat. Ollama reuses the KV cache for an unchanged prefix
    on its own, so resending identical history is cheap here too."""
    r = _chat(model, [{"role": "system", "content": system}, *history,
                      {"role": "user", "content": user_message}], think=False)
    usage = {"input": r.get("prompt_eval_count", 0), "output": r.get("eval_count", 0),
             "cache_read": 0, "cache_write": 0}
    return r["message"]["content"], usage
