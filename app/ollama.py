"""Local models through Ollama: the same jobs as llm.py, with no API key and
nothing leaving the machine. Talks to Ollama's HTTP API with the standard library."""
import base64
import http.client
import json
import os
import re
import socket
import threading
import urllib.error
import urllib.request

from . import extract, sections
from .llm import PHOTO_PROMPT, SYSTEM

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
# What an analysis prompt adds to the text it carries (its instructions, the key details
# guidance, the file name and labels; a list of folders comes off the text, sections.analyze),
# in tokens, and the factor a token estimate (tokens) is raised by for what it misses.
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


class OllamaTimeout(OllamaError):
    """Ollama took the request but sent no answer within the time allowed: busy with other
    requests, or stuck. Not OllamaUnavailable: it answers other requests meanwhile."""


class Aborted(OllamaError):
    """The call was cut off from another thread (abort): its job was cancelled."""


class ContextExceeded(OllamaError, sections.TooLong):
    """A prompt and its reply that NUM_CTX can't hold. Analysis calls ask Ollama to refuse
    such a prompt rather than cut it (truncate), and a reply that ran past the end of the
    context is caught after it; sections.py then reads the text in smaller pieces."""


# Calls under way, by thread: the connection each waits on, so abort() can cut one off.
_waiting, _aborted, _waiting_lock = {}, set(), threading.Lock()


class _Handler(urllib.request.HTTPHandler):
    """Opens connections that abort() can find by the thread waiting on them."""

    def http_open(self, req):
        def connection(host, **kw):
            conn = http.client.HTTPConnection(host, **kw)
            with _waiting_lock:
                _waiting[threading.get_ident()] = conn
            return conn
        return self.do_open(connection, req)


_opener = urllib.request.build_opener(_Handler)


def _urlopen(req, timeout):
    return _opener.open(req, timeout=timeout)


def _done(thread_id=None):
    """The call this thread (or `thread_id`) made is over."""
    thread_id = thread_id or threading.get_ident()
    with _waiting_lock:
        _waiting.pop(thread_id, None)
        _aborted.discard(thread_id)


def abort(thread_id):
    """Cut off the call to Ollama that thread `thread_id` is waiting on, if any: it raises
    Aborted at once, and Ollama, seeing the connection close, stops working on it. A
    cancelled job stops now rather than when the call returns (up to TIMEOUT later)."""
    with _waiting_lock:
        conn = _waiting.get(thread_id)
        if conn is None or conn.sock is None:
            return False
        _aborted.add(thread_id)
        try:
            conn.sock.shutdown(socket.SHUT_RDWR)
        except OSError:  # closed meanwhile
            pass
        return True


def _call_error(model, e, timeout):
    """The OllamaError a call that failed with `e` raises; None when it is none of these."""
    if threading.get_ident() in _aborted:
        return Aborted(f"{model}: stopped, as the job was cancelled")
    if isinstance(e, urllib.error.HTTPError):  # before URLError: it is a subclass
        return _http_error(model, e)
    if isinstance(e, TimeoutError):  # connected, then no answer (a refused connection is a URLError)
        setting = " (OLLAMA_TIMEOUT)" if timeout == TIMEOUT else ""
        return OllamaTimeout(f"{model}: no answer within {timeout} s{setting}; Ollama may be busy")
    if isinstance(e, (urllib.error.URLError, ConnectionError)):
        return OllamaUnavailable(f"Could not reach Ollama at {URL}")
    return None


def _chat_body(model, messages, think=None, temperature=None, num_predict=None, stream=False, **extra):
    # Settings Ollama starts a model's runner with (num_ctx; "shift", left at its default)
    # are the same in every call: Ollama restarts the model when they change. With
    # shift=False on analysis calls only, qwen3.5 restarted between each analysis call and
    # question (5-13 s each), and after a restart mid-answer such calls went unanswered.
    options = {"num_ctx": NUM_CTX}
    if temperature is not None:
        options["temperature"] = temperature
    if num_predict is not None:  # the most tokens the reply may have
        options["num_predict"] = num_predict
    body = {"model": model, "messages": messages, "stream": stream,
            "keep_alive": KEEP_ALIVE, "options": options, **extra}
    if think is not None:  # only sent to models that support thinking
        body["think"] = think
    return body


def _chat(model, messages, **kw):
    return _post("/api/chat", _chat_body(model, messages, **kw))


def _request(path, body):
    return urllib.request.Request(f"{URL}{path}", data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})


def _http_error(model, e):
    """The OllamaError for an error status from Ollama."""
    detail = e.read().decode(errors="replace")
    try:
        detail = json.loads(detail).get("error", detail)
    except ValueError:
        pass
    # "request (17012 tokens) exceeds the available context size (16384 tokens)": truncate=False
    error = ContextExceeded if re.search(r"exceeds .*context", str(detail), re.I) else OllamaError
    return error(f"{model}: {detail}")


def _post(path, body, timeout=TIMEOUT):
    try:
        with _urlopen(_request(path, body), timeout) as r:
            return json.load(r)
    except Exception as e:
        error = _call_error(body["model"], e, timeout)
        if error is None:
            raise
        raise error from e
    finally:
        _done()


def _stream(model, messages, **kw):
    """/api/chat streamed: Ollama's reply chunks as they come, the last one marked done
    with the token counts. An error before the first chunk (Ollama down, model not pulled)
    is raised as _post raises it; one after it as an OllamaError saying the answer broke off."""
    me = threading.get_ident()  # closing the generator may happen in another thread
    try:
        r = _urlopen(_request("/api/chat", _chat_body(model, messages, stream=True, **kw)), TIMEOUT)
    except Exception as e:
        error = _call_error(model, e, TIMEOUT)
        _done(me)
        if error is None:
            raise
        raise error from e
    try:
        with r:  # closing it (the generator closed: the user stopped the answer) stops Ollama writing
            while True:
                try:
                    line = r.readline()
                    chunk = json.loads(line) if line.strip() else None
                except (OSError, ValueError, http.client.HTTPException) as e:  # timeouts and resets are OSErrors
                    raise OllamaError(f"{model}: the answer broke off ({e or type(e).__name__})") from e
                if not line:
                    raise OllamaError(f"{model}: the answer broke off before its end")
                if chunk is None:
                    continue
                if chunk.get("error"):  # say the model ran out of memory part way
                    raise OllamaError(f"{model}: {chunk['error']}")
                yield chunk
                if chunk.get("done"):
                    return
    finally:
        _done(me)


def models():
    """Names of the models pulled into Ollama that can chat, so read and answer: an embedding
    model (OLLAMA_EMBED_MODEL) is left out, as it would fail at either."""
    try:
        with _urlopen(urllib.request.Request(f"{URL}/api/tags"), 5) as r:
            pulled = json.load(r).get("models", [])
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
        raise OllamaUnavailable(f"Could not reach Ollama at {URL}") from e
    finally:
        _done()
    return sorted(m["name"] for m in pulled if _chats(m))


_capabilities = {}  # a model's digest -> its capabilities, as /api/show lists them


def _chats(model):
    """Whether a model from /api/tags can chat ("completion" in the capabilities /api/show lists,
    asked once per model version). Offered when Ollama doesn't say, as before."""
    if model["name"] in (EMBED_MODEL, f"{EMBED_MODEL}:latest"):
        return False
    key = model.get("digest") or model["name"]
    if key not in _capabilities:
        try:
            _capabilities[key] = _post("/api/show", {"model": model["name"]}, timeout=5).get("capabilities")
        except (OllamaError, ValueError):
            return True
    return not _capabilities[key] or "completion" in _capabilities[key]


def tokens(text):
    """About how many tokens `text` is for a local model (see _TOKEN). Counted a megabyte
    at a time, so a huge text never becomes a list of millions of bits at once."""
    step = 1_000_000
    return sum(len(_TOKEN.findall(text, i, i + step)) for i in range(0, len(text), step))


def fits(messages, reply_tokens=2000):
    """Whether a prompt fits the local context window with room for the reply: its token
    estimate (tokens) with MARGIN to spare, erring on the side of switching to Claude."""
    return sum(tokens(m.get("content") or "") for m in messages) * MARGIN + reply_tokens <= NUM_CTX


def recent(history, *fixed, reply_tokens=2000):
    """What a local model is sent of a long conversation: the latest turns of `history` that
    fit the context window beside the `fixed` messages (system prompt, question) with room
    for the reply, as fits() counts, starting at a question. Left to itself, Ollama would cut
    the oldest text wherever it falls. None when the fixed messages alone don't fit."""
    room = (NUM_CTX - reply_tokens) / MARGIN
    used = sum(tokens(m.get("content") or "") for m in fixed)
    if used > room:
        return None
    start = len(history)
    for i in range(len(history) - 1, -1, -1):
        used += tokens(history[i].get("content") or "")
        if used > room:
            break
        start = i
    while start < len(history) and history[start].get("role") != "user":
        start += 1
    return history[start:]


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
    within REPLY_TOKENS. Text denser than reckoned is not cut silently: Ollama refuses a
    prompt too long for NUM_CTX (truncate=False), a reply that ran past the end of the
    context is caught after it (ContextExceeded), and sections.py then reads that text in
    smaller pieces."""
    def ask(prompt, schema, kind):
        messages = [{"role": "user", "content": prompt}]
        if not fits(messages, REPLY_TOKENS):
            raise ContextExceeded(f"{model}: a prompt of about {tokens(prompt):,} tokens and a {REPLY_TOKENS:,}-token "
                                  f"reply don't fit OLLAMA_NUM_CTX={NUM_CTX}")
        # truncate is checked per request; shift is not sent (see _chat_body).
        r = _chat(model, messages, think=False, temperature=0, num_predict=REPLY_TOKENS, format=schema, truncate=False)
        if on_usage:
            on_usage(kind, "ollama", model, _usage(r))
        # A reply that reached the end of the context either went on with the start of the context
        # dropped (Ollama's context shift: more tokens than NUM_CTX in all, as counted when the
        # prompt was read anew, not from Ollama's cache), or stopped there, short of num_predict.
        used, wrote = (r.get("prompt_eval_count") or 0) + (r.get("eval_count") or 0), r.get("eval_count") or 0
        if used > NUM_CTX or (r.get("done_reason") == "length" and wrote < REPLY_TOKENS):
            raise ContextExceeded(f"{model}: the prompt and its reply ran past OLLAMA_NUM_CTX={NUM_CTX}")
        if r.get("done_reason") == "length":
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


def transcribe_image(jpeg, on_usage=None):
    """The text of a photo of a document, read by the vision model: one call, given the photo
    as extract.prepare_image makes it (JPEG bytes)."""
    r = _chat(VISION_MODEL, [{"role": "user", "content": PHOTO_PROMPT,
                              "images": [base64.standard_b64encode(jpeg).decode("ascii")]}], temperature=0)
    if on_usage:
        on_usage("transcribe", "ollama", VISION_MODEL, _usage(r))
    return r["message"]["content"]


def chat(_client, history, user_message, model=MODEL, system=SYSTEM, on_usage=None):
    """Same contract as llm.chat. Ollama reuses the KV cache for an unchanged prefix
    on its own, so resending identical history is cheap here too."""
    r = _chat(model, [{"role": "system", "content": system}, *history,
                      {"role": "user", "content": user_message}], think=False)
    usage = _usage(r)
    if on_usage:
        on_usage("chat", "ollama", model, usage)
    return r["message"]["content"], usage


def chat_stream(_client, history, user_message, model=MODEL, system=SYSTEM, on_usage=None):
    """Same contract as llm.chat_stream: yields the answer's text as the model writes it,
    and returns its usage. Ollama counts tokens only in its last chunk, so an answer that
    ends early (stopped, or broken off) is reported with both counts estimated (tokens);
    one that failed before its first words (Ollama down) is not reported."""
    messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": user_message}]
    text, usage = [], None
    try:
        for chunk in _stream(model, messages, think=False):
            piece = (chunk.get("message") or {}).get("content") or ""
            if piece:
                text.append(piece)
                yield piece
            if chunk.get("done"):
                usage = _usage(chunk)
    finally:
        if usage is None and text:
            usage = {"input": sum(tokens(m["content"]) for m in messages), "output": tokens("".join(text)),
                     "cache_read": 0, "cache_write": 0}
        if usage and on_usage:
            on_usage("chat", "ollama", model, usage)
    return usage
