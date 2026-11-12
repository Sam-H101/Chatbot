"""Make litre/gallon figures in document excerpts consistent before a model reads them.

Manuals often print a volume in both units, and misprints happen: the KIOTI manual has
"38.0 ℓ / 40.1 U.S.gal" where 40.1 is really U.S. quarts (38.0 L = 10.0 U.S. gal).
Models, small ones above all, copy the printed figure, so the fix is done in code:
when a litre figure and a gallon figure next to it disagree, the litre figure is
trusted, the gallon figure is replaced with the correct one, and a note says what
the document printed. Only what is sent to the model changes; stored text does not.

fix() corrects a text; find() lists what fix() would correct in it, for the upload's
unit check. Both run the same code (_run), so they can't disagree.
"""
import bisect
import hashlib
import re

from .extract import PAGE_BREAK  # a document's pages are joined with form feeds

# Changes whenever this file does. Documents keep the version their unit check was made
# with, and a check from another version is run again (jobs.check_units): the list on a
# document always matches what fix() does now.
with open(__file__, "rb") as _f:
    VERSION = hashlib.sha256(_f.read()).hexdigest()[:12]

US_GAL = 3.785411784
IMP_GAL = 4.54609
US_QT = US_GAL / 4
# Findings kept with a document (all of them are counted): a table of figures that all
# disagree would otherwise store, and send to the page, a list as long as the document.
MAX_FINDINGS = 500

_NUM = r"(\d+(?:\.\d+)?)"
_LITRE = r"(?:ℓ|L|lit(?:re|er)s?)(?![A-Za-z])"
_GAL = r"((?:imp(?:erial)?\.?\s*|U\.?\s?S\.?\s*)?gal(?:lon)?s?\.?)"
# What may separate the two figures: spaces, slashes, brackets, <br>, and Markdown table
# syntax, since PDF tables often put the two units in separate rows ("| 38.0 ℓ |\n|---|\n| 40.1 gal |").
_SEP = r"(?:\s|/|\||-|,|\(|\)|\[|<br\s*/?>){1,40}"
_GAP = rf"({_SEP})"

LITRES_THEN_GALLONS = re.compile(rf"(?<![\d.]){_NUM}\s*({_LITRE}){_GAP}{_NUM}\s*{_GAL}", re.I)
GALLONS_THEN_LITRES = re.compile(rf"(?<![\d.]){_NUM}\s*{_GAL}{_GAP}{_NUM}\s*({_LITRE})", re.I)
# A PDF text layer can drop the litre symbol ("38.0 \n40.1 U.S.gal."). A bare number followed
# by a "gallon" figure 1.057 times larger is the quarts misprint: litres -> U.S. quarts.
BARE_THEN_GALLONS = re.compile(rf"(?<![\d.]){_NUM}(?!\s*{_LITRE}|\d|\.\d){_GAP}{_NUM}\s*{_GAL}", re.I)
SCRIPT_L = re.compile(r"(?<=\d)\s*ℓ")
# A litre or gallon figure (and a gap after it, as a pair has) right before a spot: tables
# print pairs side by side ("| 8.9 ℓ, 2.4 U.S.gal | 8.3 ℓ, 2.2 U.S.gal |"), and 2.4 gal
# belongs to the 8.9 ℓ before it, not to the 8.3 ℓ after it (see _taken).
LITRES_BEFORE = re.compile(rf"(?<![\d.])\d+(?:\.\d+)?\s*{_LITRE}{_SEP}$", re.I)
GALLONS_BEFORE = re.compile(rf"(?<![\d.])\d+(?:\.\d+)?\s*{_GAL}{_SEP}$", re.I)
_CLOSE = re.compile(r"[ \t]*\)")  # the bracket a gap opened closes after the pair: "3.0 US gal (2.0 L)"

# What find() calls each kind of misprint.
QUARTS, IMPERIAL, INCONSISTENT = "quarts-printed-as-gallons", "imperial-printed-as-us-gallons", "inconsistent-pair"


def _expected(litres, unit):
    return litres / (IMP_GAL if re.match(r"imp", unit, re.I) else US_GAL)


def _wrong(litres, gallons, unit):
    expect = _expected(litres, unit)
    return expect if abs(gallons - expect) > max(0.06, 0.03 * expect) else None


def _misprint(litres, printed):
    """(kind, note) for a gallon figure that disagrees with its litres. The note follows
    the corrected figure; the misprinted number is never shown next to the word
    "gallons" again: models copy whatever figure sits in the text."""
    if abs(printed - litres / US_QT) <= max(0.1, 0.02 * printed):
        return QUARTS, f" ({printed:g} U.S. qt)"  # the usual misprint: quarts labelled as gallons
    if abs(printed - litres / IMP_GAL) <= max(0.06, 0.02 * printed):
        return IMPERIAL, f" ({printed:g} imp. gal)"
    return INCONSISTENT, " [gallon figure recalculated from the litres: the document's own figure was a misprint]"


def _taken(m, before, other):
    """Is the first figure of this match already the second half of the pair before it?
    It is when it follows a figure in the other unit (`before`) that doesn't itself follow
    one in this match's unit (`other`): in "| 8.9 ℓ, 2.4 U.S.gal | 8.3 ℓ, 2.2 U.S.gal |"
    2.4 gal is 8.9 ℓ's, while in "| 2.4 U.S.gal, 8.9 ℓ | 2.2 U.S.gal, 8.3 ℓ |" 8.9 ℓ is
    2.4 gal's and 2.2 gal is free to pair with 8.3 ℓ."""
    prev = before.search(m.string, max(0, m.start() - 120), m.start())
    return bool(prev) and not other.search(m.string, max(0, prev.start() - 120), prev.start())


# Each rule takes a match and returns (replacement, kind), or None when the pair agrees
# (or isn't a pair).

def _litres_first(m):
    litres, lunit, gap, printed, gunit = float(m[1]), m[2], m[3], float(m[4]), m[5]
    expect = None if _taken(m, GALLONS_BEFORE, LITRES_BEFORE) else _wrong(litres, printed, gunit)
    if expect is None:
        return None
    kind, note = _misprint(litres, printed)
    return f"{m[1]} {lunit}{gap}{expect:.1f} {gunit.strip()}{note}", kind


def _gallons_first(m):
    printed, gunit, gap, litres, lunit = float(m[1]), m[2], m[3], float(m[4]), m[5]
    expect = None if _taken(m, LITRES_BEFORE, GALLONS_BEFORE) else _wrong(litres, printed, gunit)
    if expect is None:
        return None
    kind, note = _misprint(litres, printed)
    return f"{expect:.1f} {gunit.strip()}{gap}{m[4]} {lunit}{note}", kind


def _bare(m):
    litres, gap, printed, gunit = float(m[1]), m[2], float(m[3]), m[4]
    if re.match(r"imp", gunit, re.I) or abs(printed - litres / US_QT) > max(0.1, 0.01 * printed):
        return None  # only the unmistakable quarts misprint; any other bare number is left alone
    return f"{m[1]} L{gap}{litres / US_GAL:.1f} {gunit.strip()} ({printed:g} U.S. qt)", QUARTS


_RULES = ((LITRES_THEN_GALLONS, _litres_first), (GALLONS_THEN_LITRES, _gallons_first), (BARE_THEN_GALLONS, _bare))


def _clean(s):
    """Text for a person: table syntax and line breaks become single spaces."""
    return " ".join(re.sub(r"<br\s*/?>|\||-{3,}", " ", s, flags=re.I).split())


def _context(before):
    """The last words before a pair on its page, say the row label of a table."""
    text = _clean(before.rsplit(PAGE_BREAK, 1)[-1][-200:])
    return text if len(text) <= 80 else "…" + text[-80:].split(" ", 1)[-1]


class _Found(list):
    """Findings as they are made: the first `cap` of them (all, without one), and how
    many there were in all (total)."""

    def __init__(self, cap=None):
        super().__init__()
        self.cap, self.total = cap, 0


def _origin(pos, passes):
    """A spot in the text the rules' passes so far made -> the same spot in the text they
    were given. Each pass: (where its corrections start in its output, [(start, end in its
    input, length, start in its output)])."""
    for starts, edits in reversed(passes):
        i = bisect.bisect_right(starts, pos) - 1
        if i >= 0:
            start, end, size, out = edits[i]
            pos = min(end, start + pos - out) if pos < out + size else end + pos - out - size
    return pos


def _finding(text, start, end, new, kind, page, lead):
    """The pair at text[start:end], corrected to `new`, as a person reads it."""
    printed = text[start:end]
    tail = _CLOSE.match(text, end) if printed.count("(") > printed.count(")") else None
    if tail:
        printed, new = printed + tail[0], new + tail[0]
    before = text[max(0, start - 200):start]
    return {"page": None if page is None else page + text.count(PAGE_BREAK, 0, start), "kind": kind,
            "printed": _clean(printed), "corrected": _clean(new),
            "context": _context(f"{lead[-200:]}\n{before}" if start < 200 else before)}


def _run(text, page=None, found=None, lead=""):
    """The rules applied in turn, as fix() needs them: each pass reads the text the
    passes before it corrected. With `found` (a _Found), each correction is also
    recorded, as the text given has it (its page, words before it and printed figures,
    never what an earlier pass wrote): its page counted from `page` (the page the text
    starts on, or None when unknown) and the form feeds before it. `lead` is the text
    before this one (say the previous chunk), only for the context."""
    given, passes = text, []
    for pattern, rule in _RULES:
        edits = []

        def sub(m):
            hit = rule(m)
            if hit is None:
                return m[0]
            new, kind = hit
            _, end, size, out = edits[-1] if edits else (0, 0, 0, 0)
            edits.append((m.start(), m.end(), len(new), m.start() + out + size - end))  # + how much longer so far
            if found is not None:
                found.total += 1
                if found.cap is None or len(found) < found.cap:
                    found.append(_finding(given, _origin(m.start(), passes), _origin(m.end(), passes), new, kind,
                                          page, lead))
            return new

        text = pattern.sub(sub, text)
        passes.append(([e[3] for e in edits], edits))
    return text


def fix(text):
    return SCRIPT_L.sub(" L", _run(text))  # "38.0 ℓ" -> "38.0 L": not every model reads ℓ as litres


def find(text, page=None, lead=""):
    """The litre/gallon misprints fix() corrects in this text: [{page, kind, printed,
    corrected, context}], where context is the text just before the figures, on their
    page. `page` is the page the text starts on; without one, pages are counted from 1
    if the text has page breaks ("\\f"), and are None otherwise (text without pages, or
    read before pages were kept). `lead`: the text before this one, for the context."""
    found = _Found()
    _run(text, page if page is not None else 1 if PAGE_BREAK in text else None, found, lead)
    return found


def scan(chunks, paged=False, cap=None):
    """find() over a document's chunks, in order: what fix() corrects in the excerpts a
    question can retrieve, since it corrects one chunk at a time. Pages carry on from
    chunk to chunk (a chunk keeps the form feeds of the page breaks in it), counted from
    1 if the text has page breaks or is `paged` (a PDF's: one with text on its first page
    only has none). The first `cap` findings, their .total counting them all."""
    page = 1 if paged or any(PAGE_BREAK in c for c in chunks) else None
    found, lead = _Found(cap), ""
    for c in chunks:
        _run(c, page, found, lead)
        lead = c
        if page is not None:
            page += c.count(PAGE_BREAK)
    return found


def check(chunks, paged=False):
    """A document's unit check as it is stored: the first MAX_FINDINGS findings of scan(),
    how many it found in all, whether the text is `paged`, and this VERSION."""
    found = scan(chunks, paged, MAX_FINDINGS)
    return {"version": VERSION, "paged": bool(paged), "total": found.total, "findings": list(found)}
