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

# Kept byte-stable (no timestamps, no per-request data) so it stays cacheable.
SYSTEM = os.environ.get("SYSTEM_PROMPT") or (
    "You answer questions for the user of a document chatbot. Each user message may "
    "start with <excerpts> retrieved from the user's uploaded documents by search, and a "
    "<scope> naming the only documents the user wants searched. Use the excerpts when "
    "relevant, and cite each one you use in parentheses with its file name, like "
    "(Chapter 7 Brake.pdf). If the excerpts do not contain the answer, say so and answer "
    "from general knowledge. Be concise."
)
# Follows the excerpts only when some of them have pages. In the system prompt, it had a
# local model (qwen3.5:9b) give excerpts without pages (documents read before pages were
# kept) a made-up page in 7 of 15 answers; here, in none of 15, and pages were cited right.
PAGE_NOTE = ("Excerpts with a page or pages attribute: cite the page too, like (Chapter 7 Brake.pdf, p. 12); "
             "a [Page n] line inside an excerpt marks where page n starts. Cite an excerpt without one by its "
             "file name alone.")

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


def build_user_message(question, hits, scope=None):
    """The question, after the excerpts retrieved for it. Each names its file and, for a
    PDF read since pages were kept, its page or pages, which PAGE_NOTE asks the answer to
    cite. `scope`: the file names of the documents the user limited the question to."""
    head = ""
    if scope:
        names = ", ".join(scope[:10]) + (f" and {len(scope) - 10} more" if len(scope) > 10 else "")
        head = f"<scope>Only these documents were searched: {names}</scope>\n"
    if not hits:
        return head + question
    parts = []
    for h in hits:
        attrs = 'file="{}"'.format(h["filename"].replace('"', "'"))
        if h.get("page_start"):
            first, last = h["page_start"], h["page_end"]
            attrs += f' page="{first}"' if first == last else f' pages="{first}-{last}"'
        parts.append(f"<excerpt {attrs}>\n{excerpt_text(h)}\n</excerpt>")
    note = f"\n\n{PAGE_NOTE}" if any(h.get("page_start") for h in hits) else ""
    return head + "<excerpts>\n" + "\n".join(parts) + "\n</excerpts>" + note + "\n\n" + question


def chat(client, history, user_message, model=MODEL, system=SYSTEM, on_usage=None):
    """history: prior [{role, content}] turns exactly as previously sent, so the
    conversation prefix is byte-identical between requests and hits the cache.
    Returns (answer, usage); on_usage also hears of a refusal, which is billed."""
    with client.beta.messages.stream(
        model=model,
        max_tokens=16000,
        system=system,
        cache_control={"type": "ephemeral"},  # caches system + conversation so far
        output_config={"effort": EFFORT},
        messages=[*history, {"role": "user", "content": user_message}],
        **_fallback(model),
    ) as stream:
        response = stream.get_final_message()
    _report(on_usage, "chat", model, response)
    return _text(response), _usage(response)


def client_for(api_key):
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
