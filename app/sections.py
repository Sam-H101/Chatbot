"""Document analysis: a document's metadata from one model call, or, for a text longer than
one call may carry, section by section (map-reduce). The same model takes structured notes
on each section (key topics, specifications and values, procedures, warnings), and the notes
are combined into the metadata, so a long document is described as a whole rather than by
its start. Backend-neutral: llm.py and ollama.py pass `ask`, which makes one structured-output
call to their model."""
import hashlib
import logging
import math
import os
import re

from .extract import PAGE_BREAK, plain

log = logging.getLogger(__name__)

# The most section calls one analysis makes. A document needing more is sampled: this many
# sections, spread evenly from its first to its last, and its analysis is marked partial.
MAX_SECTIONS = max(1, int(os.environ.get("ANALYZE_MAX_SECTIONS") or "24"))
# Room for what a prompt adds around the text it carries (instructions, file name, labels).
PROMPT_CHARS = 1500

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
NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {"type": "array", "items": {"type": "string"}, "description": "key topics of this part"},
        "specifications": {"type": "array", "items": {"type": "string"},
                           "description": "values with their units exactly as printed"},
        "procedures": {"type": "array", "items": {"type": "string"},
                       "description": "procedures, each named with its main steps in a few words"},
        "warnings": {"type": "array", "items": {"type": "string"}, "description": "warnings and cautions"},
    },
    "required": ["topics", "specifications", "procedures", "warnings"],
    "additionalProperties": False,
}
_HEADINGS = (("topics", "Key topics"), ("specifications", "Specifications and values"),
             ("procedures", "Procedures"), ("warnings", "Warnings"))


def _meta_prompt(filename, text):
    return (f"Extract metadata for the document below.\n\nFile name: {filename}\n\n"
            f"<document>\n{text}\n</document>")


def _notes_prompt(filename, label, text):
    return (f"The document below is too long to read at once, so it is read in parts. This is {label}. "
            "Take notes on this part, for a description of the whole document: its key topics; specifications "
            "and values (capacities, sizes, settings, part numbers, dates, amounts), each with its units exactly "
            "as printed; procedures, each named with its main steps in a few words; warnings and cautions. Keep "
            "each item to one short line, at most about 15 items per list, and note only what this part says."
            f"\n\nFile name: {filename}\n\n<part>\n{text}\n</part>")


def _merge_prompt(filename, label, notes):
    return (f"Below are notes taken on {label} of a document too long to read at once. Merge them into one set "
            "of notes on those parts, in the same four lists: drop repeats, keep every specification, value and "
            "warning that matters, one short line per item, at most about 25 items per list."
            f"\n\nFile name: {filename}\n\n<notes>\n{notes}\n</notes>")


def _combine_prompt(filename, notes, total, read):
    sample = "" if read == total else (
        f" Only {read} of its {total} parts were read, spread evenly through it, so the notes cover part of it.")
    return (f"Extract metadata for the document below. It was too long to read at once, so it was read in {total} "
            f"parts and notes were taken on each.{sample} Describe the whole document from these notes."
            f"\n\nFile name: {filename}\n\n<notes>\n{notes}\n</notes>")


def _pieces(page, limit):
    """A page as one piece if it fits, else its paragraphs. A paragraph longer than
    `limit` is cut into even pieces, each at a line break or space if there is one."""
    page = page.strip()
    if len(page) <= limit:
        return [page] if page else []
    out = []
    for para in re.split(r"\n\s*\n", page):
        para = para.strip()
        size = math.ceil(len(para) / math.ceil(len(para) / limit))
        while len(para) > limit:
            cut = para.rfind("\n", size // 2, size)
            cut = cut if cut > 0 else para.rfind(" ", size // 2, size)
            cut = cut if cut > 0 else size
            out.append(para[:cut].rstrip())
            para = para[cut:].lstrip()
        if para:
            out.append(para)
    return out


def split(text, limit):
    """A document's text in sections of at most `limit` characters, for analysis: whole
    pages where they fit, else paragraphs, else cuts. Sections come out about the same
    size (each aims at an even share of what is left), so the last is not a scrap.
    [(first page, last page, plain text)]: pages count from 1 in a text with page
    breaks ("\\f"), and are None in one without."""
    pages = text.split(PAGE_BREAK)
    paged = len(pages) > 1
    pieces = [(n if paged else None, p) for n, page in enumerate(pages, 1) for p in _pieces(page, limit)]
    if not pieces:
        return []
    left = sum(len(p) for _, p in pieces) + 2 * (len(pieces) - 1)
    count = math.ceil(left / limit)
    sections, current, size = [], [], 0
    for page, piece in pieces:
        target = left / max(1, count - len(sections))
        # Next section once this one is full, or the piece would end further past the
        # target than it starts before it.
        if current and (size + 2 + len(piece) > limit or size + 1 + len(piece) / 2 > target):
            sections.append(current)
            left -= size + 2
            current, size = [], 0
        size += len(piece) + (2 if current else 0)
        current.append((page, piece))
    sections.append(current)
    return [(s[0][0], s[-1][0], "\n\n".join(p for _, p in s)) for s in sections]


def plan(chars, limit, cap=None):
    """(sections, sections read) for a text of `chars` characters: for estimates."""
    n = max(1, math.ceil(chars / limit))
    return n, min(n, cap or MAX_SECTIONS)


def pick(n, cap):
    """Which of n sections to read when at most `cap` may be: all, or `cap` of them spread
    evenly, first and last included."""
    if n <= cap:
        return list(range(n))
    if cap == 1:
        return [0]
    return [round(i * (n - 1) / (cap - 1)) for i in range(cap)]


def _label(a, b, n, first, last):
    which = f"part {a + 1}" if a == b else f"parts {a + 1}-{b + 1}"
    pages = "" if first is None else f" (page {first})" if first == last else f" (pages {first}-{last})"
    return f"{which} of {n}{pages}"


def _block(a, b, n, first, last, notes):
    """Notes on parts a-b as text for the next call."""
    lines = [f"## {_label(a, b, n, first, last).capitalize()}"]
    notes = notes if isinstance(notes, dict) else {}
    for key, heading in _HEADINGS:
        items = [str(i).strip() for i in notes.get(key) or [] if str(i).strip()]
        if items:
            lines += [f"{heading}:", *(f"- {i}" for i in items)]
    return {"from": a, "to": b, "first": first, "last": last, "text": "\n".join(lines)}


def _joined(blocks):
    return "\n\n".join(b["text"] for b in blocks)


def _fit(ask, filename, blocks, n, limit, say):
    """Notes that fit one call. While they don't, neighbouring blocks are merged by the
    model, as many to a call as fit, round after round (a local model's context holds
    the notes of about ten sections)."""
    while len(_joined(blocks)) > limit:
        groups, size = [], 0
        for b in blocks:
            if groups and size + 2 + len(b["text"]) <= limit:
                groups[-1].append(b)
                size += 2 + len(b["text"])
            else:
                groups.append([b])
                size = len(b["text"])
        if len(groups) == len(blocks):  # not even two fit together: shorten each (never with sane settings)
            share = limit // len(blocks) - 2
            return [{**b, "text": b["text"][:share]} for b in blocks]
        merged = []
        for i, group in enumerate(groups, 1):
            if len(group) == 1:
                merged.append(group[0])
                continue
            say(f"Combining notes, group {i} of {len(groups)}")
            a, b, first, last = group[0]["from"], group[-1]["to"], group[0]["first"], group[-1]["last"]
            notes = ask(_merge_prompt(filename, _label(a, b, n, first, last), _joined(group)),
                        NOTES_SCHEMA, "analyze-merge")
            merged.append(_block(a, b, n, first, last, notes))
        blocks = merged
    return blocks


def analyze(ask, filename, text, limit, skip=(ValueError,), on_progress=None, kept=None, on_notes=None):
    """(metadata, analysis) for a document. ask(prompt, schema, kind) makes one call and
    returns its parsed JSON; `limit` is how many characters of the document (or of notes)
    one call may carry. Text within it takes one call, as always. Longer text is read in
    sections (split), at most MAX_SECTIONS of them (pick), and analysis is 'full' when
    every section was read, else 'partial: <read> of <sections> sections, <share> of the
    text'.

    on_progress(stage, done, total) hears of each step. on_notes(key, notes) gets each
    section's notes as they come, and `kept` ({key: notes}, from an earlier attempt)
    saves asking for them again. A section whose call fails with one of `skip` (a reply
    that isn't JSON, a refusal) is left out and logged; other errors are raised, as is
    the last one if no section could be read."""
    body = plain(text)
    parts = split(text, limit) if len(body) > limit else []
    if len(parts) <= 1:
        return ask(_meta_prompt(filename, parts[0][2] if parts else body), META_SCHEMA, "analyze"), "full"

    def say(stage, done=None, total=None):
        if on_progress:
            on_progress(stage, done, total)

    n, chosen = len(parts), pick(len(parts), MAX_SECTIONS)
    read, error = [], None
    for i, k in enumerate(chosen, 1):
        first, last, part = parts[k]
        say(f"Analyzing section {i} of {len(chosen)}" + (f" (a sample of {n})" if len(chosen) < n else ""),
            i - 1, len(chosen))
        key = hashlib.sha256(part.encode()).hexdigest()[:24]
        notes = (kept or {}).get(key)
        if notes is None:
            try:
                notes = ask(_notes_prompt(filename, _label(k, k, n, first, last), part), NOTES_SCHEMA,
                            "analyze-section")
            except skip as e:
                log.warning("Section %d of %d of %s left out of its analysis: %s", k + 1, n, filename, e)
                error = e
                continue
            if on_notes:
                on_notes(key, notes)
        read.append(_block(k, k, n, first, last, notes))
    if not read:
        raise error
    say(f"Combining the notes of {len(read)} sections", len(chosen), len(chosen))
    blocks = _fit(ask, filename, read, n, limit, say)
    meta = ask(_combine_prompt(filename, _joined(blocks), n, len(read)), META_SCHEMA, "analyze")
    if len(read) == n:
        return meta, "full"
    share = sum(len(parts[b["from"]][2]) for b in read) / sum(len(p[2]) for p in parts)
    return meta, f"partial: {len(read)} of {n} sections, {math.floor(share * 100)}% of the text"
