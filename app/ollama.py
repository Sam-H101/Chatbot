"""Local models through Ollama: the same jobs as llm.py, with no API key and
nothing leaving the machine. Talks to Ollama's HTTP API with the standard library."""
import base64
import json
import os
import re
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
# Characters of a document one analysis call sends, at most. Longer documents are analyzed
# section by section (sections.py), and sections are also kept within the tokens NUM_CTX
# holds (analyze_tokens): 40,000 characters of prose fit, while tables of figures, at a
# token a digit, make sections smaller. Raise this together with NUM_CTX.
ANALYZE_CHARS = int(os.environ.get("OLLAMA_ANALYZE_CHARS", "40000"))
REPLY_TOKENS = 2000  # context kept free for an analysis reply, and the most it may write
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))
# Embeds chunks and questions for search by meaning (vectors.py). A small model (0.6B,
# about 640 MB) that sits on the GPU beside the chat model.
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
# Seconds an embedding call may take: a question waits for it, so far less than TIMEOUT.
EMBED_TIMEOUT = 60
# What an analysis prompt adds to the text it carries (its instructions, the file name and
# labels), in tokens, and the factor a token estimate (tokens) is raised by for what it misses.
PROMPT_TOKENS, MARGIN = 500, 1.1
# Bits of text that are about a token each for qwen3.5, whose tokenizer, like many, gives
# every digit a token of its own: a digit, a non-ASCII character, a word (with its leading
# space) or six letters of a long one, up to four punctuation marks, a line break with the
# spaces around it, spaces before a number. On the library's manuals, tables of figures
# and part lists this came within 7% under to about 20% over qwen3.5's own count (dot
# leaders and CJK text, more over); the plain 3 characters a token used before was up to
# 3 times under for tables, and Ollama then cut the start off the text it was given.
_TOKEN = re.compile(r"\d|[^\x00-\x7f]|[A-Za-z]{1,6}|[!-/:-@\[-`{-~]{1,4}|\s*\n\s*|[ \t]+(?=\d)")

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


class ContextExceeded(OllamaError, sections.TooLong):
    """A prompt and its reply that NUM_CTX can't hold. Analysis calls ask Ollama to refuse
    such a prompt rather than cut it (truncate) and to stop such a reply rather than drop
    the start of the context (shift); sections.py then reads the text in smaller pieces."""


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
    return _post("/api/chat", body)


def _post(path, body, timeout=TIMEOUT):
    model = body["model"]
    req = urllib.request.Request(f"{URL}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # before URLError: it is a subclass
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except ValueError:
            pass
        # "request (17012 tokens) exceeds the available context size (16384 tokens)": truncate=False
        error = ContextExceeded if re.search(r"exceeds .*context", str(detail), re.I) else OllamaError
        raise error(f"{model}: {detail}") from e
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


def tokens(text):
    """About how many tokens `text` is for a local model (see _TOKEN). Counted a megabyte
    at a time, so a huge text never becomes a list of millions of bits at once."""
    step = 1_000_000
    return sum(len(_TOKEN.findall(text, i, i + step)) for i in range(0, len(text), step))


def fits(messages, reply_tokens=2000):
    """Whether a prompt fits the local context window with room for the reply: its token
    estimate (tokens) with MARGIN to spare, erring on the side of switching to Claude."""
    return sum(tokens(m.get("content") or "") for m in messages) * MARGIN + reply_tokens <= NUM_CTX


def _usage(r):
    """Token counts from an /api/chat reply. Ollama leaves prompt_eval_count out when
    the whole prompt came from its cache."""
    return {"input": r.get("prompt_eval_count") or 0, "output": r.get("eval_count") or 0,
            "cache_read": 0, "cache_write": 0}


def embed(texts, model=None, kind="embed", on_usage=None):
    """One embedding per text (a list of floats; Ollama returns them unit length), in one
    call to /api/embed. Text past the model's context (a very long question) is cut rather
    than refused (truncate). on_usage hears of the call as `kind` ("embed" for document
    text, "embed-query" for a question)."""
    model = model or EMBED_MODEL
    r = _post("/api/embed", {"model": model, "input": list(texts), "truncate": True, "keep_alive": KEEP_ALIVE},
              timeout=EMBED_TIMEOUT)
    if on_usage:
        on_usage(kind, "ollama", model, {"input": r.get("prompt_eval_count") or 0, "output": 0,
                                         "cache_read": 0, "cache_write": 0})
    vectors = r.get("embeddings") or []
    if len(vectors) != len(texts):
        raise OllamaError(f"{model}: {len(vectors)} embeddings for {len(texts)} texts")
    return vectors


def analyze_tokens():
    """Tokens of document text (or notes) one analysis call may carry: what NUM_CTX holds
    beside the prompt's own words and the reply, with MARGIN to spare."""
    return max(100, (NUM_CTX - REPLY_TOKENS) / MARGIN - PROMPT_TOKENS)


def _analysis_size(text):
    """How much of one analysis call `text` takes, in the tokens of analyze_tokens(): its
    token estimate, or more for text so long in characters that ANALYZE_CHARS caps it."""
    return max(tokens(text), len(text) * analyze_tokens() / ANALYZE_CHARS)


def analyze(_client, filename, text, model=MODEL, on_usage=None, **hooks):
    """Same contract as llm.analyze. Every call fits the context window: sections.py keeps
    the text in each within analyze_tokens() (as _analysis_size reckons it), and the reply
    within REPLY_TOKENS. Text denser than reckoned can't be cut silently: Ollama refuses a
    prompt too long for NUM_CTX, and stops a reply that reaches its end (ContextExceeded),
    and sections.py then reads that text in smaller pieces."""
    def ask(prompt, schema, kind):
        messages = [{"role": "user", "content": prompt}]
        if not fits(messages, REPLY_TOKENS):
            raise ContextExceeded(f"{model}: a prompt of about {tokens(prompt):,} tokens and a {REPLY_TOKENS:,}-token "
                                  f"reply don't fit OLLAMA_NUM_CTX={NUM_CTX}")
        r = _chat(model, messages, think=False, temperature=0, num_predict=REPLY_TOKENS, format=schema,
                  truncate=False, shift=False)
        if on_usage:
            on_usage(kind, "ollama", model, _usage(r))
        if r.get("done_reason") == "length":
            if (r.get("eval_count") or 0) < REPLY_TOKENS:  # stopped by the end of the context, not num_predict
                raise ContextExceeded(f"{model}: the prompt and its reply ran past OLLAMA_NUM_CTX={NUM_CTX}")
            raise ValueError(f"the reply was cut off at {REPLY_TOKENS:,} tokens")
        return json.loads(r["message"]["content"])

    return sections.analyze(ask, filename, text, analyze_tokens(), size=_analysis_size, **hooks)


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
