"""Claude API calls: document analysis and retrieval-grounded chat."""
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor

import anthropic

from . import units

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("CLAUDE_EFFORT", "low")
# Characters of a document sent to Claude for metadata extraction (about 125-170K
# tokens at the default, well inside Claude's context). The full text is always stored
# and searchable; this only bounds the cost of the analysis call. Longer documents are
# analyzed from their start and marked "partial". Local models have their own,
# smaller limit (OLLAMA_ANALYZE_CHARS in ollama.py).
ANALYZE_CHARS = int(os.environ.get("CLAUDE_ANALYZE_CHARS", "500000"))
PDF_CONCURRENCY = int(os.environ.get("PDF_CONCURRENCY", "4"))
TRANSCRIBE_PROMPT = (
    "This is pages {first}-{last} of a {total}-page PDF. Transcribe all of its text in "
    "reading order. Render tables as Markdown tables and headings as Markdown headings. "
    "For charts, diagrams and photos, write a one-line description in [square brackets], "
    "including any numbers they show. Leave a blank line between paragraphs. Output only "
    "the transcription, with no preamble or commentary."
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
# Routes a safety-classifier decline to a fallback model instead of failing.
# Only these models take server-side fallbacks; others are sent without it.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5"}


def _fallback(model):
    if model in FALLBACK_MODELS:
        return {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    return {}


def label(model):
    return dict(MODELS).get(model, model)

# Kept byte-stable (no timestamps, no per-request data) so it stays cacheable.
SYSTEM = os.environ.get("SYSTEM_PROMPT") or (
    "You answer questions for the user of a document chatbot. Each user message may "
    "start with <excerpts> retrieved from the user's uploaded documents by full-text "
    "search. Use them when relevant and cite the file name in parentheses. If the "
    "excerpts do not contain the answer, say so and answer from general knowledge. "
    "Be concise."
)

META_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "doc_type": {"type": "string", "description": "e.g. invoice, contract, paper, notes, code"},
        "summary": {"type": "string", "description": "2-3 sentences"},
        "keywords": {"type": "array", "items": {"type": "string"},
                     "description": "5-12 search terms: topics, names, entities"},
    },
    "required": ["title", "doc_type", "summary", "keywords"],
    "additionalProperties": False,
}


def system_prompt(rules=()):
    """SYSTEM plus the operator's instructions. Changes only when the rules change,
    so the prompt cache still hits between questions."""
    rules = [r.strip() for r in rules if r and r.strip()]
    if not rules:
        return SYSTEM
    return (SYSTEM + "\n\nAlways follow these instructions from the operator of this chatbot:\n"
            + "\n".join(f"- {r}" for r in rules))


def _text(response):
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request")
    return "".join(b.text for b in response.content if b.type == "text")


def analyze(client, filename, text, model=MODEL):
    """Return (metadata dict, 'full' | 'partial')."""
    excerpt = text[:ANALYZE_CHARS]
    response = client.beta.messages.create(
        model=model,
        max_tokens=2000,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": META_SCHEMA}},
        messages=[{
            "role": "user",
            "content": f"Extract metadata for the document below.\n\nFile name: {filename}\n\n"
                       f"<document>\n{excerpt}\n</document>",
        }],
        **_fallback(model),
    )
    return json.loads(_text(response)), "full" if len(text) <= ANALYZE_CHARS else "partial"


def transcribe_pdf(client, batches, model=MODEL):
    """Have Claude read the PDF pages (as text and image) and return their text.
    `batches` comes from extract.pdf_batches; pieces are read in parallel."""
    with ThreadPoolExecutor(max_workers=PDF_CONCURRENCY) as pool:
        return "\n\n".join(pool.map(lambda b: _transcribe(client, model, *b), batches))


def _transcribe(client, model, first, last, total, pdf_bytes):
    with client.beta.messages.stream(
        model=model,
        max_tokens=64000,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": [
            {"type": "document", "source": {
                "type": "base64", "media_type": "application/pdf",
                "data": base64.standard_b64encode(pdf_bytes).decode("ascii")}},
            {"type": "text", "text": TRANSCRIBE_PROMPT.format(first=first, last=last, total=total)},
        ]}],
        **_fallback(model),
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "max_tokens":
        raise RuntimeError(f"pages {first}-{last} hold more text than one call can return; "
                           "lower PDF_PAGES_PER_CALL")
    return _text(response)


def build_user_message(question, hits):
    if not hits:
        return question
    # units.fix corrects litre/gallon misprints so the model doesn't repeat them.
    parts = [f'<excerpt file="{h["filename"]}" part="{h["idx"]}">\n{units.fix(h["text"])}\n</excerpt>'
             for h in hits]
    return "<excerpts>\n" + "\n".join(parts) + "\n</excerpts>\n\n" + question


def chat(client, history, user_message, model=MODEL, system=SYSTEM):
    """history: prior [{role, content}] turns exactly as previously sent, so the
    conversation prefix is byte-identical between requests and hits the cache."""
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
    u = response.usage
    usage = {
        "input": u.input_tokens,
        "output": u.output_tokens,
        "cache_read": u.cache_read_input_tokens or 0,
        "cache_write": u.cache_creation_input_tokens or 0,
    }
    return _text(response), usage


def client_for(api_key):
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
