"""Make litre/gallon figures in document excerpts consistent before a model reads them.

Manuals often print a volume in both units, and misprints happen: the KIOTI manual has
"38.0 ℓ / 40.1 U.S.gal" where 40.1 is really U.S. quarts (38.0 L = 10.0 U.S. gal).
Models, small ones above all, copy the printed figure, so the fix is done in code:
when a litre figure and a gallon figure next to it disagree, the litre figure is
trusted, the gallon figure is replaced with the correct one, and a note says what
the document printed. Only what is sent to the model changes; stored text does not.
"""
import re

US_GAL = 3.785411784
IMP_GAL = 4.54609
US_QT = US_GAL / 4

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


def _expected(litres, unit):
    return litres / (IMP_GAL if re.match(r"imp", unit, re.I) else US_GAL)


def _wrong(litres, gallons, unit):
    expect = _expected(litres, unit)
    return expect if abs(gallons - expect) > max(0.06, 0.03 * expect) else None


def _note(litres, printed):
    """What follows the corrected gallon figure. The misprinted number is never shown
    next to the word "gallons" again: models copy whatever figure sits in the text."""
    if abs(printed - litres / US_QT) <= max(0.1, 0.02 * printed):
        return f" ({printed:g} U.S. qt)"  # the usual misprint: quarts labelled as gallons
    if abs(printed - litres / IMP_GAL) <= max(0.06, 0.02 * printed):
        return f" ({printed:g} imp. gal)"
    return " [gallon figure recalculated from the litres: the document's own figure was a misprint]"


def _fix_lg(m):
    litres, lunit, gap, printed, gunit = float(m[1]), m[2], m[3], float(m[4]), m[5]
    expect = _wrong(litres, printed, gunit)
    if expect is None:
        return m[0]
    return f"{m[1]} {lunit}{gap}{expect:.1f} {gunit.strip()}{_note(litres, printed)}"


def _fix_gl(m):
    printed, gunit, gap, litres, lunit = float(m[1]), m[2], m[3], float(m[4]), m[5]
    expect = _wrong(litres, printed, gunit)
    if expect is None:
        return m[0]
    return f"{expect:.1f} {gunit.strip()}{gap}{m[4]} {lunit}{_note(litres, printed)}"


def _fix_bare(m):
    litres, gap, printed, gunit = float(m[1]), m[2], float(m[3]), m[4]
    if re.match(r"imp", gunit, re.I) or abs(printed - litres / US_QT) > max(0.1, 0.01 * printed):
        return m[0]  # only the unmistakable quarts misprint; any other bare number is left alone
    return f"{m[1]} L{gap}{litres / US_GAL:.1f} {gunit.strip()} ({printed:g} U.S. qt)"


def fix(text):
    text = LITRES_THEN_GALLONS.sub(_fix_lg, text)
    text = GALLONS_THEN_LITRES.sub(_fix_gl, text)
    text = BARE_THEN_GALLONS.sub(_fix_bare, text)
    return SCRIPT_L.sub(" L", text)  # "38.0 ℓ" -> "38.0 L": not every model reads ℓ as litres
