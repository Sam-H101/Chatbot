"""Claude API calls: document analysis and retrieval-grounded chat."""
import json
import os

import anthropic

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("CLAUDE_EFFORT", "low")
# Characters of a document sent to Claude for metadata extraction. The full text
# is always stored and searchable; this only bounds the cost of the analysis call.
ANALYZE_CHARS = int(os.environ.get("ANALYZE_CHARS", "40000"))
# Routes a safety-classifier decline to a fallback model instead of failing.
FALLBACK = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

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


def _text(response):
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request")
    return "".join(b.text for b in response.content if b.type == "text")


def analyze(client, filename, text):
    """Return (metadata dict, 'full' | 'partial')."""
    excerpt = text[:ANALYZE_CHARS]
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=2000,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": META_SCHEMA}},
        messages=[{
            "role": "user",
            "content": f"Extract metadata for the document below.\n\nFile name: {filename}\n\n"
                       f"<document>\n{excerpt}\n</document>",
        }],
        **FALLBACK,
    )
    return json.loads(_text(response)), "full" if len(text) <= ANALYZE_CHARS else "partial"


def build_user_message(question, hits):
    if not hits:
        return question
    parts = [f'<excerpt file="{h["filename"]}" part="{h["idx"]}">\n{h["text"]}\n</excerpt>' for h in hits]
    return "<excerpts>\n" + "\n".join(parts) + "\n</excerpts>\n\n" + question


def chat(client, history, user_message):
    """history: prior [{role, content}] turns exactly as previously sent, so the
    conversation prefix is byte-identical between requests and hits the cache."""
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        cache_control={"type": "ephemeral"},  # caches system + conversation so far
        output_config={"effort": EFFORT},
        messages=[*history, {"role": "user", "content": user_message}],
        **FALLBACK,
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
