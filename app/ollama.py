"""Local models through Ollama: the same jobs as llm.py, with no API key and
nothing leaving the machine. Talks to Ollama's HTTP API with the standard library."""
import base64
import json
import os
import urllib.error
import urllib.request

from . import extract, sections
from .llm import SYSTEM

URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
# Reads scanned PDF pages. qwen3.5 also accepts images, so setting this to MODEL
# runs everything on one model (less VRAM juggling, slower page reading).
VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "minicpm-v4.6")
# Context window per request. On an 11 GB card a 9B model leaves room for ~16K;
# raise it on bigger GPUs. Ollama silently drops the oldest tokens past this.
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
# Characters of a document one analysis call sends. 40,000 is about 11-13K tokens, which
# fits NUM_CTX with room for the reply. Longer documents are analyzed section by section
# (sections.py). Past NUM_CTX Ollama would silently cut the text, so a smaller NUM_CTX
# also makes the sections smaller (analyze_chars); raise both together.
ANALYZE_CHARS = int(os.environ.get("OLLAMA_ANALYZE_CHARS", "40000"))
REPLY_TOKENS = 2000  # context kept free for an analysis reply, and the most it may write
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


def _chat(model, messages, think=None, temperature=None, num_predict=None, **extra):
    options = {"num_ctx": NUM_CTX}
    if temperature is not None:
        options["temperature"] = temperature
    if num_predict is not None:  # the most tokens the reply may have
        options["num_predict"] = num_predict
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


def _usage(r):
    """Token counts from an /api/chat reply. Ollama leaves prompt_eval_count out when
    the whole prompt came from its cache."""
    return {"input": r.get("prompt_eval_count") or 0, "output": r.get("eval_count") or 0,
            "cache_read": 0, "cache_write": 0}


def analyze_chars():
    """Characters of document text (or notes) one analysis call carries: ANALYZE_CHARS,
    or fewer when NUM_CTX can't hold that many next to the prompt and the reply."""
    return max(1000, min(ANALYZE_CHARS, (NUM_CTX - REPLY_TOKENS) * 3 - sections.PROMPT_CHARS))


def analyze(_client, filename, text, model=MODEL, on_usage=None, **hooks):
    """Same contract as llm.analyze. Every call fits the context window: sections.py
    keeps the text in each within analyze_chars(), and the reply within REPLY_TOKENS."""
    def ask(prompt, schema, kind):
        messages = [{"role": "user", "content": prompt}]
        if not fits(messages, REPLY_TOKENS):  # only with a NUM_CTX too small for any analysis
            raise OllamaError(f"{model}: OLLAMA_NUM_CTX={NUM_CTX} is too small for an analysis prompt")
        r = _chat(model, messages, think=False, temperature=0, num_predict=REPLY_TOKENS, format=schema)
        if on_usage:
            on_usage(kind, "ollama", model, _usage(r))
        return json.loads(r["message"]["content"])

    return sections.analyze(ask, filename, text, analyze_chars(), **hooks)


def transcribe_pdf(data, on_usage=None, done=None, on_page=None):
    """One string per page: the embedded text where the page has it, while the vision
    model reads only scanned pages. `done` maps page numbers (from 1) read in an earlier
    attempt to their text; those are not read again. on_page(number, total, text, read)
    follows every page, read=True when the vision model read it, and on_page(number,
    total, None, True) comes just before the vision model starts on one. It may raise
    to stop."""
    done = done or {}
    total = extract.page_count(data)
    out = []
    for n, (text, png) in enumerate(extract.pdf_pages(data, skip=done), 1):
        read = n not in done and png is not None
        if n in done:
            text = done[n]
        elif read:
            if on_page:
                on_page(n, total, None, True)
            r = _chat(VISION_MODEL, [{
                "role": "user",
                "content": PAGE_PROMPT.format(page=n, total=total),
                "images": [base64.standard_b64encode(png).decode("ascii")],
            }], temperature=0)
            if on_usage:
                on_usage("transcribe", "ollama", VISION_MODEL, _usage(r))
            text = r["message"]["content"]
        out.append(text)
        if on_page:
            on_page(n, total, text, read)
    return out


def chat(_client, history, user_message, model=MODEL, system=SYSTEM, on_usage=None):
    """Same contract as llm.chat. Ollama reuses the KV cache for an unchanged prefix
    on its own, so resending identical history is cheap here too."""
    r = _chat(model, [{"role": "system", "content": system}, *history,
                      {"role": "user", "content": user_message}], think=False)
    usage = _usage(r)
    if on_usage:
        on_usage("chat", "ollama", model, usage)
    return r["message"]["content"], usage
