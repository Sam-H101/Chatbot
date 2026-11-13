"""Document analysis: a document's metadata from one model call, or, for a text longer than
one call may carry, section by section (map-reduce). The same model takes structured notes
on each section (key topics, specifications and values, procedures, warnings), and the notes
are combined into the metadata, so a long document is described as a whole rather than by
its start. Backend-neutral: llm.py and ollama.py pass `ask`, which makes one structured-output
call to their model, and how they measure what a call may carry (`size` and `limit`)."""
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
# File names are cut to this many characters in prompts: a name can be any length, and the
# room it takes is the document's.
NAME_CHARS = 200


class TooLong(Exception):
    """A prompt the model can't take in whole, though `size` said it would fit (dense text,
    say tables of figures). Raised by `ask` (ollama.ContextExceeded); the text is then
    read in smaller pieces."""


# Key details and the folder are strings, "" when unknown: no null unions, which both Ollama's
# `format` and Claude's json_schema take. jobs._details checks and converts them.
META_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "doc_type": {"type": "string", "description": "e.g. invoice, contract, paper, notes, code"},
        "summary": {"type": "string", "description": "2-3 sentences"},
        "keywords": {"type": "array", "items": {"type": "string"},
                     "description": "5-12 search terms: topics, names, entities"},
        "document_date": {"type": "string", "description": "The date the document itself bears (issue, invoice or "
                          "receipt date) as YYYY-MM-DD; \"\" if it has none or the date is ambiguous"},
        "vendor": {"type": "string", "description": "Receipts, bills and invoices: the shop or company that issued "
                   "it; otherwise \"\""},
        "total_amount": {"type": "string", "description": "Receipts, bills and invoices: the total to pay, digits "
                         "with a dot for decimals, no currency sign (e.g. 1234.50); otherwise \"\""},
        "currency": {"type": "string", "description": "ISO 4217 code of total_amount (GBP, EUR, USD...), or \"\""},
        "due_date": {"type": "string", "description": "Bills and invoices: the payment due date as YYYY-MM-DD; "
                     "otherwise \"\""},
        "folder": {"type": "string", "description": "Where to file it: one of the listed folder paths if it clearly "
                   "belongs there, otherwise a short new category (1-3 words, plural noun, e.g. Receipts), never the "
                   "document's own title; \"\" if unsure"},
        "folder_confidence": {"type": "string", "enum": ["high", "low"]},
    },
    "required": ["title", "doc_type", "summary", "keywords", "document_date", "vendor", "total_amount", "currency",
                 "due_date", "folder", "folder_confidence"],
    "additionalProperties": False,
}
# The existing folders a metadata prompt lists, at most: this many paths and about this many
# characters (a library with hundreds of folders must not crowd out the document).
PROMPT_FOLDERS, PROMPT_FOLDER_CHARS = 200, 6000
DETAILS_GUIDANCE = (
    "Fill document_date, vendor, total_amount, currency and due_date only with values the document itself states; "
    "leave a field empty if unsure, and never guess an ambiguous date (03/04/2026). Manuals, letters and papers have "
    "no vendor or total.")
# The schema's descriptions say this too, but Ollama turns a schema into a grammar and never
# shows them to the model: told only there, qwen3.5:9b wrote "3 November 2026" for a date and
# filed a receipt as "Household/Bills/Receipts" beside an existing "Receipts". Told here, it
# wrote 2026-11-03, chose the listed folder that fit (the deepest one, Receipts/2026, where
# there was one), and gave "" for a manual with no folder to fit.
FIELD_RULES = (
    "Write dates as YYYY-MM-DD (3 November 2026 is 2026-11-03) and total_amount as digits with a dot for decimals "
    "(1234.50).\nfolder: where to file it. If one of the existing folders below clearly fits, copy its path exactly as "
    "listed. Otherwise give a short new category (1-3 words, plural noun, e.g. Receipts), never the document's own "
    "title; \"\" if unsure. folder_confidence: high only when you are sure.")
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
_SPACE = re.compile(r"\s*")


def _folder_lines(folders):
    """The existing folders' paths as a metadata prompt lists them, "" for none."""
    folders, lines, used = list(folders), [], 0
    for path in folders[:PROMPT_FOLDERS]:
        if lines and used + len(path) > PROMPT_FOLDER_CHARS:
            break
        lines.append(f"- {path}")
        used += len(path)
    return "\n".join(lines) + (", …" if len(lines) < len(folders) else "") if lines else ""


def folders_text(folders):
    """What a metadata prompt adds: the key details guidance, how to write them and the
    folder, and the existing folders, for the model to file the document into one (or
    suggest a new one)."""
    return (f"\n\n{DETAILS_GUIDANCE}\n{FIELD_RULES}\nExisting folders (levels separated by /):\n"
            f"{_folder_lines(folders) or '(none yet)'}")


def _meta_prompt(filename, text, folders=()):
    return (f"Extract metadata for the document below.\n\nFile name: {filename}\n\n"
            f"<document>\n{text}\n</document>{folders_text(folders)}")


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


def _combine_prompt(filename, notes, total, read, folders=()):
    sample = "" if read == total else (
        f" Only {read} of its {total} parts were read, spread evenly through it, so the notes cover part of it.")
    return (f"Extract metadata for the document below. It was too long to read at once, so it was read in {total} "
            f"parts and notes were taken on each.{sample} Describe the whole document from these notes."
            f"\n\nFile name: {filename}\n\n<notes>\n{notes}\n</notes>{folders_text(folders)}")


def _pieces(page, limit, size):
    """A page as one piece if it fits, else its paragraphs. A paragraph over `limit` is
    cut into about even pieces, each at a line break or space if there is one."""
    page = page.strip()
    if size(page) <= limit:
        return [page] if page else []
    out = []
    for para in re.split(r"\n\s*\n", page):
        para = para.strip()
        # A CSV can be one paragraph of megabytes: it is read from `at` on rather than copied
        # after each cut, and what is left of it measured again only once it may fit.
        left, at = size(para), 0
        step = math.ceil(len(para) / max(1, math.ceil(left / limit)))  # characters a piece, at the average density
        while left > limit and at < len(para):
            end = min(len(para) - at, step)
            while end > 1 and size(para[at:at + end]) > limit:  # a denser stretch: shorter
                end = int(end * 0.9)
            cut = para.rfind("\n", at + end // 2, at + end)
            cut = cut if cut > at else para.rfind(" ", at + end // 2, at + end)
            cut = cut if cut > at else at + end
            out.append(para[at:cut].rstrip())
            left -= size(out[-1])
            at = _SPACE.match(para, cut).end()
            if left <= limit * 1.5:
                left = size(para[at:])
        if at < len(para):
            out.append(para[at:])
    return out


def split(text, limit, size=len):
    """A document's text in sections of at most `limit` (as `size` measures it: characters
    by default), for analysis: whole pages where they fit, else paragraphs, else cuts.
    Sections come out about the same size (each aims at an even share of what is left),
    so the last is not a scrap. [(first page, last page, plain text)]: pages count from 1
    in a text with page breaks ("\\f"), and are None in one without."""
    pages = text.split(PAGE_BREAK)
    paged = len(pages) > 1
    pieces = [(n if paged else None, p) for n, page in enumerate(pages, 1) for p in _pieces(page, limit, size)]
    if not pieces:
        return []
    sizes, sep = [size(p) for _, p in pieces], size("\n\n")
    left = sum(sizes) + sep * (len(pieces) - 1)
    count = math.ceil(left / limit)
    sections, current, used = [], [], 0
    for (page, piece), cost in zip(pieces, sizes):
        target = left / max(1, count - len(sections))
        # Next section once this one is full, or the piece would end further past the
        # target than it starts before it.
        if current and (used + sep + cost > limit or used + (sep + cost) / 2 > target):
            sections.append(current)
            left -= used + sep
            current, used = [], 0
        used += cost + (sep if current else 0)
        current.append((page, piece))
    sections.append(current)
    return [(s[0][0], s[-1][0], "\n\n".join(p for _, p in s)) for s in sections]


def plan(text, limit, cap=None, size=len):
    """(sections, sections read) for analyzing `text`, as analyze() will split it: for estimates."""
    n = max(1, len(split(text, limit, size))) if size(plain(text)) > limit else 1
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


def _shorten(text, budget, size):
    """The lines of `text` that fit `budget`, from the top."""
    out, used = [], 0
    for line in text.split("\n"):
        used += size(line) + size("\n")
        if used > budget:
            break
        out.append(line)
    return "\n".join(out)


def _fit(ask, filename, blocks, n, limit, size, say, skip):
    """Notes that fit one call. While they don't, neighbouring blocks are merged by the
    model, as many to a call as fit, round after round (a local model's context holds
    the notes of about ten sections). A merge that fails with one of `skip` (a reply cut
    off, or not JSON) or TooLong leaves its blocks as they were, and later rounds merge
    half as many at a time: a shorter merge writes a shorter reply. The analysis still
    gets to its final call."""
    sep, pack = size("\n\n"), limit
    while size(_joined(blocks)) > limit:
        groups, used = [], 0
        for b in blocks:
            cost = size(b["text"])
            if groups and used + sep + cost <= pack:
                groups[-1].append(b)
                used += sep + cost
            else:
                groups.append([b])
                used = cost
        if len(groups) == len(blocks):  # not even two merge together: shorten each (never with sane settings)
            return [{**b, "text": _shorten(b["text"], limit / len(blocks) - sep, size)} for b in blocks]
        merged, failed = [], False
        for i, group in enumerate(groups, 1):
            if len(group) == 1:
                merged.append(group[0])
                continue
            say(f"Combining notes, group {i} of {len(groups)}")
            a, b, first, last = group[0]["from"], group[-1]["to"], group[0]["first"], group[-1]["last"]
            try:
                notes = ask(_merge_prompt(filename, _label(a, b, n, first, last), _joined(group)),
                            NOTES_SCHEMA, "analyze-merge")
            except (TooLong, *skip) as e:
                log.warning("Notes on %s of %s left unmerged: %s", _label(a, b, n, first, last), filename, e)
                merged += group
                failed = True
                continue
            merged.append(_block(a, b, n, first, last, notes))
        blocks = merged
        if failed:
            pack /= 2
    return blocks


def _notes(ask, filename, label, text, size, depth=0):
    """Notes on one section. One the model can't take in whole (TooLong) is read in two or
    three pieces, again if need be (down to about an eighth), and their notes put together."""
    try:
        return ask(_notes_prompt(filename, label, text), NOTES_SCHEMA, "analyze-section")
    except TooLong:
        pieces = split(text, size(text) / 2, size) if depth < 3 else []
        if len(pieces) < 2:
            raise
        notes = [_notes(ask, filename, f"{label}, piece {i} of {len(pieces)}", t, size, depth + 1)
                 for i, (_, _, t) in enumerate(pieces, 1)]
        return {key: [x for got in notes if isinstance(got, dict) for x in got.get(key) or []] for key, _ in _HEADINGS}


def analyze(ask, filename, text, limit, size=len, skip=(ValueError,), on_progress=None, kept=None,
            on_notes=None, folders=()):
    """(metadata, analysis) for a document. ask(prompt, schema, kind) makes one call and
    returns its parsed JSON; `limit` is how much of the document (or of notes) one call may
    carry, as size(text) measures it: characters by default, estimated tokens for a local
    model. Text within it takes one call, as always. Longer text is read in sections
    (split), at most MAX_SECTIONS of them (pick), and analysis is 'full' when every
    section was read, else 'partial: <read> of <sections> sections, <share> of the text'.
    A prompt that `ask` finds too long after all (TooLong) is read in smaller pieces.

    The metadata calls (the one call, or the last, which combines the notes) are also given
    the existing folders' paths (`folders`, db.folder_paths), to suggest where to file the
    document, and asked for its key details (META_SCHEMA); the room the list takes comes
    off what they carry of the document. Notes on sections are asked for as before.

    on_progress(stage, done, total) hears of each step. on_notes(key, notes) gets each
    section's notes as they come, and `kept` ({key: notes}, from an earlier attempt)
    saves asking for them again. A section whose call fails with one of `skip` (a reply
    that isn't JSON, a refusal) is left out and logged; other errors are raised, as is
    the last one if no section could be read."""
    filename = filename if len(filename) <= NAME_CHARS else filename[:NAME_CHARS - 1] + "…"
    body = plain(text)
    # What a metadata call carries of the document beside the folder list. The guidance around
    # it is short, like the rest of a prompt's own words (which `limit` leaves room for).
    room = max(limit / 2, limit - size(_folder_lines(folders)))
    parts = split(text, limit, size) if size(body) > room else []
    if len(parts) == 1 and size(body) > room:  # fits a call of notes, not one with the folders: smaller
        parts = split(text, room, size)
    if len(parts) <= 1:
        try:
            return ask(_meta_prompt(filename, parts[0][2] if parts else body, folders), META_SCHEMA, "analyze"), "full"
        except TooLong:  # denser than `size` reckoned: in sections after all
            parts = split(text, size(body) / 2, size)
            if len(parts) < 2:
                raise

    def say(stage, done=None, total=None):
        if on_progress:
            on_progress(stage, done, total)

    n, chosen = len(parts), pick(len(parts), MAX_SECTIONS)
    read, error = [], None
    for i, k in enumerate(chosen, 1):
        first, last, part = parts[k]
        say(f"Analyzing section {i} of {len(chosen)}" + (f" (sampled from {n})" if len(chosen) < n else ""),
            i - 1, len(chosen))
        key = hashlib.sha256(part.encode()).hexdigest()[:24]
        notes = (kept or {}).get(key)
        if notes is None:
            try:
                notes = _notes(ask, filename, _label(k, k, n, first, last), part, size)
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
    blocks, budget = read, room
    while True:
        fitted = _fit(ask, filename, blocks, n, budget, size, say, skip)
        if fitted is not blocks:  # merging showed steps of its own
            say(f"Combining the notes of {len(read)} sections", len(chosen), len(chosen))
        blocks = fitted
        try:
            meta = ask(_combine_prompt(filename, _joined(blocks), n, len(read), folders), META_SCHEMA, "analyze")
            break
        except TooLong:  # denser notes than `size` reckoned: fit them into less
            if budget < limit / 8:
                raise
            budget /= 2
    if len(read) == n:
        return meta, "full"
    share = sum(len(parts[b["from"]][2]) for b in read) / sum(len(p[2]) for p in parts)
    return meta, f"partial: {len(read)} of {n} sections, {math.floor(share * 100)}% of the text"
