"""Claude API calls: document analysis and retrieval-grounded chat."""
import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from . import extract, sections, units
from .sections import META_SCHEMA

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("CLAUDE_EFFORT", "low")
# Characters of a document one analysis call sends to Claude (about 125-170K tokens at
# the default, well inside Claude's context). Longer documents are analyzed section by
# section, each section this long at most, and the sections' notes combined (sections.py).
# Local models have their own, smaller limit (OLLAMA_ANALYZE_CHARS in ollama.py).
ANALYZE_CHARS = int(os.environ.get("CLAUDE_ANALYZE_CHARS", "500000"))
PDF_CONCURRENCY = int(os.environ.get("PDF_CONCURRENCY", "4"))
TRANSCRIBE_PROMPT = (
    "This PDF holds {which} of a {total}-page document, in that order. Transcribe all of its "
    "text in reading order. Begin each page with a line containing only <<<PAGE n>>>, where n "
    "is that page's number in the whole document. Render tables as Markdown tables and headings "
    "as Markdown headings. For charts, diagrams and photos, write a one-line description in "
    "[square brackets], including any numbers they show. Leave a blank line between paragraphs. "
    "Output only the transcription, with no preamble or commentary."
)
_PAGE_MARK = re.compile(r"^[ \t]*<<<\s*PAGE\s+(\d+)\s*>>>[ \t]*$", re.M | re.I)
# A photo of a document (an upload of a JPG, PNG, WebP or HEIC), for Claude and the local
# vision model alike (ollama.transcribe_image).
PHOTO_PROMPT = (
    "Transcribe all the text in this photo of a document (a receipt, bill, letter or page), top to bottom. "
    "Keep line breaks, amounts, dates and the rows of tables as plain text. Output only the text."
)
# Models offered in the UI's model pickers.
MODELS = [
    ("claude-opus-5-5", "Claude Opus 5.5"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
    ("claude-haiku-5-5", "Claude Haiku 5.5"),
    ("claude-fable-5-2", "Claude Fable 5.2"),
]
if MODEL not in dict(MODELS):
    MODELS.append((MODEL, MODEL))
# List prices in USD per million tokens (input, output, cache read), for the usage log.
# Cache reads are priced per model (0.05x input on Opus and Sonnet 5.5, 0.025x on Fable 5.2,
# 0.1x on Haiku 5.5); cache writes (5-minute, what chat uses) cost 1.25x input on all four.
# Models missing here are logged at $0.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-fable-5-2": (8.00, 40.00, 0.20),
}
# Haiku charges more once a prompt passes 100K tokens: (threshold, prices as above).
LONG_PROMPT_PRICES = {"claude-haiku-5-5": (100_000, (0.50, 2.50, 0.05))}
# Routes a safety-classifier decline to a fallback model instead of failing.
# Only these models take server-side fallbacks; others are sent without it.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5"}


def _fallback(model):
    if model in FALLBACK_MODELS:
        return {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    return {}


def label(model):
    return dict(MODELS).get(model, model)


def cost(model, usage):
    """USD for one call, from a usage dict (input, output, cache_read, cache_write)."""
    get = lambda k: usage.get(k) or 0
    p_in, p_out, p_read = PRICES.get(model, (0, 0, 0))
    threshold, long_prices = LONG_PROMPT_PRICES.get(model, (None, None))
    if threshold and get("input") + get("cache_read") + get("cache_write") > threshold:
        p_in, p_out, p_read = long_prices
    return (get("input") * p_in + get("cache_write") * p_in * 1.25 + get("cache_read") * p_read
            + get("output") * p_out) / 1_000_000


def _usage(response):
    u = response.usage
    return {"input": u.input_tokens or 0, "output": u.output_tokens or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}


def _report(on_usage, kind, model, response):
    """Pass one call's token counts to the caller's on_usage(kind, backend, model, usage)."""
    if on_usage and getattr(response, "usage", None) is not None:
        on_usage(kind, "claude", model, _usage(response))

# Kept byte-stable (no timestamps, no per-request data) so it stays cacheable. It names no
# file as an example: with "(Chapter 7 Brake.pdf)" here and in PAGE_NOTE, qwen3.5:9b cited
# that file, not in the library, in 4 of 24 answers about a manual's brakes; with PAGE_NOTE's
# example made from a real excerpt, in none, and it cited the right page in 20 of 24, not 10.
# General knowledge is not for figures: told only to "answer from general knowledge" when the
# excerpts lack the answer, qwen3.5:9b gave the live library's tractor a "typical" hydraulic
# oil capacity of 11.0 litres, confidently and wrongly.
SYSTEM = os.environ.get("SYSTEM_PROMPT") or (
    "You answer questions for the user of a document chatbot. Each user message may "
    "start with <today>, today's date, <excerpts> retrieved from the user's uploaded documents "
    "by search, and a <scope> naming the only documents the user wants searched. Use the excerpts when "
    "relevant, and cite each one you use in parentheses with the file name from its file "
    "attribute. When no excerpt matched the question, a <library> lists the documents with "
    "their summaries instead. If none of this contains the answer, say so. If the answer is "
    "a figure or specification (a capacity, pressure, torque, clearance, dimension, interval, "
    "part number or the like), stop there rather than give a typical or estimated value; "
    "anything else you may answer from general knowledge, saying that you do. Be concise. "
    "Write plain text: the answer is shown as written, so no Markdown such as ** or #."
)
# Follows the excerpts only when some of them have pages, its example made from the first
# that has. In the system prompt, it had a local model (qwen3.5:9b) give excerpts without
# pages (documents read before pages were kept) a made-up page in 7 of 15 answers; here, in
# none of 15, and pages were cited right.
PAGE_NOTE = ("Excerpts with a page or pages attribute: cite the page too, like ({file}, p. {page}); "
             "a [Page n] line inside an excerpt marks where page n starts. Cite an excerpt without one by its "
             "file name alone.")
# Of the documents' titles and summaries, the most a question that matched no passage is given.
LIBRARY_CHARS = 6000

def system_prompt(rules=()):
    """SYSTEM plus the operator's instructions. Changes only when the rules change,
    so the prompt cache still hits between questions."""
    rules = [r.strip() for r in rules if r and r.strip()]
    if not rules:
        return SYSTEM
    return (SYSTEM + "\n\nAlways follow these instructions from the operator of this chatbot:\n"
            + "\n".join(f"- {r}" for r in rules))


class Refused(RuntimeError):
    """Claude's safety filter declined (after any server-side fallbacks)."""


class MissingKey(Exception):
    """No Claude API key: none from the browser and no ANTHROPIC_API_KEY."""


def _text(response):
    if response.stop_reason == "refusal":
        raise Refused("Claude declined this request")
    return "".join(b.text for b in response.content if b.type == "text")


def analyze(client, filename, text, model=MODEL, on_usage=None, **hooks):
    """Return (metadata dict, 'full' | 'partial: ...'): one call, or section by section
    for text longer than ANALYZE_CHARS. `hooks` are those of sections.analyze."""
    def ask(prompt, schema, kind):
        response = client.beta.messages.create(
            model=model,
            max_tokens=2000 if schema is META_SCHEMA else 8000,  # a section's notes run longer
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
            **_fallback(model),
        )
        _report(on_usage, kind, model, response)
        if response.stop_reason == "max_tokens":
            raise ValueError("the reply was cut off at its max_tokens")
        return json.loads(_text(response))

    return sections.analyze(ask, filename, text, ANALYZE_CHARS, skip=(ValueError, Refused), **hooks)


def transcribe_pdf(client, batches, model=MODEL, on_usage=None, on_batch=None):
    """Have Claude read PDF pages (as text and image). `batches` comes from
    extract.pdf_batches. Returns {page number: text} for every page in them.

    Pieces are read in parallel, so on_usage is called from worker threads. on_batch
    ({page: text}) is called in this thread as each piece finishes, so the caller can
    save it and skip it after a failure; it may raise to stop. The first failure stops
    pieces not yet started, waits for those under way (they are billed), and is raised."""
    pages, error, stopped = {}, None, False
    with ThreadPoolExecutor(max_workers=PDF_CONCURRENCY) as pool:
        futures = [pool.submit(_transcribe, client, model, *b, on_usage=on_usage) for b in batches]

        def fail(e):
            for other in futures:
                other.cancel()
            return error or e

        for f in as_completed(futures):
            if f.cancelled():
                continue
            try:
                got = f.result()
            except Exception as e:
                error = fail(e)
                continue
            pages.update(got)
            if on_batch and not stopped:  # also pieces that finish after another failed
                try:
                    on_batch(got)
                except Exception as e:  # the caller wants to stop (say the job was cancelled)
                    error, stopped = fail(e), True
    if error:
        raise error
    return pages


def page_names(pages):
    """[21, ..., 40] -> 'pages 21-40'; [3, 7, 8] -> 'pages 3, 7 and 8'; [5] -> 'page 5'."""
    if len(pages) == 1:
        return f"page {pages[0]}"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"pages {pages[0]}-{pages[-1]}"
    return "pages " + ", ".join(map(str, pages[:-1])) + f" and {pages[-1]}"


def split_pages(text, pages):
    """A transcription of `pages` -> {page: text}, split at its <<<PAGE n>>> lines. With
    one mark per page they are taken in order (whatever numbers they carry); otherwise
    by number, a mark that names none of the pages staying with the page before. Text
    before the first mark belongs to the first page, so nothing is ever dropped."""
    parts = _PAGE_MARK.split(text)
    head, marks = parts[0].strip(), list(zip(parts[1::2], parts[2::2]))
    out = dict.fromkeys(pages, "")
    if len(marks) == len(pages):
        for p, (_, body) in zip(pages, marks):
            out[p] = body.strip()
    else:
        current = pages[0]
        for number, body in marks:
            current = int(number) if int(number) in out else current
            out[current] = f"{out[current]}\n\n{body.strip()}".strip()
    out[pages[0]] = f"{head}\n\n{out[pages[0]]}".strip()
    return out


def _transcribe(client, model, pages, total, pdf_bytes, on_usage=None):
    with client.beta.messages.stream(
        model=model,
        max_tokens=64000,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": [
            {"type": "document", "source": {
                "type": "base64", "media_type": "application/pdf",
                "data": base64.standard_b64encode(pdf_bytes).decode("ascii")}},
            {"type": "text", "text": TRANSCRIBE_PROMPT.format(which=page_names(pages), total=total)},
        ]}],
        **_fallback(model),
    ) as stream:
        response = stream.get_final_message()
    _report(on_usage, "transcribe", model, response)  # billed even when the reply is unusable
    if response.stop_reason == "max_tokens":
        raise RuntimeError(f"{page_names(pages)} hold more text than one call can return; "
                           "lower PDF_PAGES_PER_CALL")
    return split_pages(_text(response), pages)


def transcribe_image(client, jpeg, model=MODEL, on_usage=None):
    """The text of a photo of a document (JPEG bytes, extract.prepare_image), read by Claude:
    one image block and PHOTO_PROMPT, as the local vision model is given."""
    response = client.beta.messages.create(
        model=model,
        max_tokens=8000,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(jpeg).decode("ascii")}},
            {"type": "text", "text": PHOTO_PROMPT},
        ]}],
        **_fallback(model),
    )
    _report(on_usage, "transcribe", model, response)  # billed even when the reply is unusable
    if response.stop_reason == "max_tokens":
        raise RuntimeError("the photo holds more text than one call can return")
    return _text(response)


def excerpt_text(hit):
    """A retrieved chunk as the model reads it (and the page shows it as a source): litre/
    gallon misprints corrected so the model doesn't repeat them, and where a chunk of a PDF
    runs onto another page, a "[Page n]" line where that page starts, so a fact is cited
    with its own page rather than the range's. units.fix reads the stored chunk, as the
    unit check at upload did (units.scan), so both find the same misprints."""
    text, first = units.fix(hit["text"]), hit.get("page_start")
    parts = text.lstrip(extract.PAGE_BREAK).split(extract.PAGE_BREAK)  # breaks before its text: in page_start
    if first and len(parts) > 1:
        text = parts[0].rstrip() + "".join(f"\n\n[Page {first + i}]\n{p.strip()}" for i, p in enumerate(parts)
                                           if i and p.strip())
    return extract.plain(text)


def library_text(docs, total):
    """A question that matched no passage ("Summarise everything in my library") gets the
    documents' titles and summaries instead (db.library_overview), up to LIBRARY_CHARS, each
    with its key details (the date it bears, who issued it, its total, when it is due): "How
    much have I spent this month?" is near no receipt in particular, and is answered from these."""
    lines, size = [], 0
    for d in docs:
        title = d["title"] if d["title"] and d["title"] != d["filename"] else ""
        summary = " ".join((d["summary"] or "").split())
        if len(summary) > 400:
            summary = summary[:400].rsplit(" ", 1)[0] + " …"
        amount = d.get("total_amount")
        details = ", ".join(filter(None, (
            d.get("document_date") and f"dated {d['document_date']}", d.get("vendor") and f"from {d['vendor']}",
            amount is not None and f"total {amount:.2f}" + (f" {d['currency']}" if d.get("currency") else ""),
            d.get("due_date") and f"due {d['due_date']}")))
        line = (f"- {d['filename']}" + (f": {title}" if title else "") + (f" ({d['doc_type']})" if d["doc_type"] else "")
                + (f" [{details}]" if details else "") + (f". {summary}" if summary else ""))
        if lines and size + len(line) > LIBRARY_CHARS:
            break
        lines.append(line)
        size += len(line)
    if total > len(lines):
        lines.append(f"- and {total - len(lines)} more")
    return ("<library>\nNo excerpt matched the question. The documents in the library, newest first:\n"
            + "\n".join(lines) + "\n</library>\n")


def build_user_message(question, hits, scope=None, library=None, folders=None, today=None):
    """The question, after the excerpts retrieved for it. Each names its file and, for a
    PDF read since pages were kept, its page or pages, which PAGE_NOTE asks the answer to
    cite. `scope`: the file names of the documents the user limited the question to.
    `folders`: the folders it was limited to ([{name, count}]: documents in each, its
    subfolders' included). `library`: db.library_overview(), given when no excerpt was found.
    `today`: the date (a datetime.date), without which "this month" or "is it overdue?"
    can't be answered; in this turn rather than the system prompt, which stays cacheable."""
    head = f"<today>{today:%A} {today.isoformat()}</today>\n" if today else ""
    head += _scope_note(scope, folders)
    if not hits:
        return head + (library_text(*library) if library and library[1] else "") + question
    parts = []
    for h in hits:
        attrs = 'file="{}"'.format(_attr(h["filename"]))
        if h.get("page_start"):
            first, last = h["page_start"], h["page_end"]
            attrs += f' page="{first}"' if first == last else f' pages="{first}-{last}"'
        parts.append(f"<excerpt {attrs}>\n{excerpt_text(h)}\n</excerpt>")
    paged = next((h for h in hits if h.get("page_start")), None)
    note = "\n\n" + PAGE_NOTE.format(file=_attr(paged["filename"]), page=paged["page_start"]) if paged else ""
    return head + "<excerpts>\n" + "\n".join(parts) + "\n</excerpts>" + note + "\n\n" + question


def _scope_note(scope, folders):
    """The <scope> line of a question limited to documents or folders ("" when it isn't)."""
    head = ""
    if folders:
        items = [f"the folder {f['name']} ({f['count']} document{'' if f['count'] == 1 else 's'}, including its "
                 "subfolders)" for f in folders[:10]]
        if len(folders) > 10:
            items.append(f"{len(folders) - 10} more folders")
        names = list(scope or [])
        items += names[:10] + ([f"{len(names) - 10} more documents"] if len(names) > 10 else [])
        head = f"<scope>Only these were searched: {'; '.join(items)}</scope>\n"
    elif scope:
        names = ", ".join(scope[:10]) + (f" and {len(scope) - 10} more" if len(scope) > 10 else "")
        head = f"<scope>Only these documents were searched: {names}</scope>\n"
    return head


def _attr(name):
    return name.replace('"', "'")


def _chat_request(client, history, user_message, model, system):
    """The chat request, streamed: chat() and chat_stream() send the same one, so either
    reads the other's prompt cache."""
    return client.beta.messages.stream(
        model=model,
        max_tokens=16000,
        system=system,
        cache_control={"type": "ephemeral"},  # caches system + conversation so far
        output_config={"effort": EFFORT},
        messages=[*history, {"role": "user", "content": user_message}],
        **_fallback(model),
    )


def chat(client, history, user_message, model=MODEL, system=SYSTEM, on_usage=None):
    """history: prior [{role, content}] turns exactly as previously sent, so the
    conversation prefix is byte-identical between requests and hits the cache.
    Returns (answer, usage); on_usage also hears of a refusal, which is billed."""
    with _chat_request(client, history, user_message, model, system) as stream:
        response = stream.get_final_message()
    _report(on_usage, "chat", model, response)
    return _text(response), _usage(response)


def chat_stream(client, history, user_message, model=MODEL, system=SYSTEM, on_usage=None):
    """chat(), a piece at a time: yields the answer's text as Claude writes it (after a
    server-side fallback, the fallback model's text carries on from the declined one's),
    and returns its usage. on_usage hears of the call however it ends. One that ends early
    (the generator closed because the user stopped it, or the stream broke off) is reported
    with the input Claude counted at its start and the output estimated from the text so
    far, about 4 characters a token; one that never started (say a bad key) is not. A
    refusal is reported, then raised after the text already yielded: the caller drops it."""
    text, response = [], None
    with _chat_request(client, history, user_message, model, system) as stream:
        try:
            for piece in stream.text_stream:
                text.append(piece)
                yield piece
            response = stream.get_final_message()
        finally:
            if response is None and on_usage and (usage := _usage_so_far(stream, "".join(text))):
                on_usage("chat", "claude", model, usage)
    _report(on_usage, "chat", model, response)
    _text(response)  # raises Refused
    return _usage(response)


def _usage_so_far(stream, text):
    """The usage of a stream that ended early, None if it had not started (no message_start:
    nothing billed). Output tokens are only counted at the end, so they are estimated."""
    try:
        usage = _usage(stream.current_message_snapshot)
    except Exception:  # no snapshot yet
        return None
    usage["output"] = max(usage["output"], round(len(text) / 4))
    return usage


def client_for(api_key):
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
