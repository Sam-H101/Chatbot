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
import hashlib
import re

# Changes whenever this file does. Documents keep the version their unit check was made
# with, and a check from another version is run again (jobs.check_units): the list on a
# document always matches what fix() does now.
with open(__file__, "rb") as _f:
    VERSION = hashlib.sha256(_f.read()).hexdigest()[:12]

US_GAL = 3.785411784
IMP_GAL = 4.54609
US_QT = US_GAL / 4
PAGE_BREAK = "\f"  # as in extract.py: a document's pages are joined with form feeds

_NUM = r"(\d+(?:\.\d+)?)"
_LITRE = r"(?:ℓ|L|lit(?:re|er)s?)(?![A-Za-z])"
_GAL = r"((?:imp(?:erial)?\.?\s*|U\.?\s?S\.?\s*)?gal(?:lon)?s?\.?)"
# What may separate the two figures: spaces, slashes, brackets, <br>, and Markdown table
# syntax, since PDF tables often put the two units in separate rows ("| 38.0 ℓ |\n|---|\n| 40.1 gal |").
_GAP = r"((?:\s|/|\||-|,|\(|\)|\[|<br\s*/?>){1,40})"

LITRES_THEN_GALLONS = re.compile(rf"(?<![\d.]){_NUM}\s*({_LITRE}){_GAP}{_NUM}\s*{_GAL}", re.I)
GALLONS_THEN_LITRES = re.compile(rf"(?<![\d.]){_NUM}\s*{_GAL}{_GAP}{_NUM}\s*({_LITRE})", re.I)
# A PDF text layer can drop the litre symbol ("38.0 \n40.1 U.S.gal."). A bare number followed
# by a "gallon" figure 1.057 times larger is the quarts misprint: litres -> U.S. quarts.
BARE_THEN_GALLONS = re.compile(rf"(?<![\d.]){_NUM}(?!\s*{_LITRE}|\d|\.\d){_GAP}{_NUM}\s*{_GAL}", re.I)
SCRIPT_L = re.compile(r"(?<=\d)\s*ℓ")
# A litre or gallon figure (and the gap after it) right before a spot: tables print pairs
# side by side ("| 8.9 ℓ, 2.4 U.S.gal | 8.3 ℓ, 2.2 U.S.gal |"), and 2.4 gal belongs to the
# 8.9 ℓ before it, not to the 8.3 ℓ after it (see _taken).
_SEP = r"(?:\s|/|\||-|,|\(|\)|\[|<br\s*/?>){1,40}"
LITRES_BEFORE = re.compile(rf"(?<![\d.])\d+(?:\.\d+)?\s*{_LITRE}{_SEP}$", re.I)
GALLONS_BEFORE = re.compile(rf"(?<![\d.])\d+(?:\.\d+)?\s*{_GAL}{_SEP}$", re.I)

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


def _run(text, page=None, found=None, lead=""):
    """The rules applied in turn, as fix() needs them. With a `found` list, each
    correction is also recorded there, its page counted from `page` (the page the
    text starts on, or None when unknown) and the form feeds before it. `lead` is the
    text before this one (say the previous chunk), only for the context."""
    for pattern, rule in _RULES:
        def sub(m):
            hit = rule(m)
            if hit is None:
                return m[0]
            new, kind = hit
            if found is not None:  # m.string is this pass's text: every form feed is still there
                before = m.string[:m.start()]
                found.append({"page": None if page is None else page + before.count(PAGE_BREAK), "kind": kind,
                              "printed": _clean(m[0]), "corrected": _clean(new),
                              "context": _context(f"{lead[-200:]}\n{before}" if len(before) < 200 else before)})
            return new
        text = pattern.sub(sub, text)
    return text


def fix(text):
    return SCRIPT_L.sub(" L", _run(text))  # "38.0 ℓ" -> "38.0 L": not every model reads ℓ as litres


def find(text, page=None, lead=""):
    """The litre/gallon misprints fix() corrects in this text: [{page, kind, printed,
    corrected, context}], where context is the text just before the figures, on their
    page. `page` is the page the text starts on; without one, pages are counted from 1
    if the text has page breaks ("\\f"), and are None otherwise (text without pages, or
    read before pages were kept). `lead`: the text before this one, for the context."""
    found = []
    _run(text, page if page is not None else 1 if PAGE_BREAK in text else None, found, lead)
    return found


def scan(chunks):
    """find() over a document's chunks, in order: what fix() corrects in the excerpts a
    question can retrieve, since it corrects one chunk at a time. Pages carry on from
    chunk to chunk (a chunk keeps the form feeds of the page breaks in it)."""
    page = 1 if any(PAGE_BREAK in c for c in chunks) else None
    out, lead = [], ""
    for c in chunks:
        out += find(c, page, lead)
        lead = c
        if page is not None:
            page += c.count(PAGE_BREAK)
    return out


def check(chunks):
    """A document's unit check as it is stored: what scan() finds, and this VERSION."""
    return {"version": VERSION, "findings": scan(chunks)}
