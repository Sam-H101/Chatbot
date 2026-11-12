import re

from app.units import SCRIPT_L, find, fix, scan


def test_kioti_quarts_misprint_is_corrected():
    out = fix("| Capacity | Transmission fluid | 38.0 ℓ / 40.1 U.S.gal |")
    assert out == "| Capacity | Transmission fluid | 38.0 L / 10.0 U.S.gal (40.1 U.S. qt) |"


def test_misprint_never_leaves_the_wrong_figure_next_to_gallons():
    out = fix("38.0 ℓ<br>40.1 U.S.gal.")
    assert out == "38.0 L<br>10.0 U.S.gal. (40.1 U.S. qt)"
    assert not re.search(r"40\.1\s*U\.?S\.?\s*gal", out, re.I)


def test_misprint_split_across_table_rows():  # real text from Chapter 6, read by Claude
    out = fix("| Capacity | Transmission fluid | 38.0 ℓ |\n|---|---|---|\n| | | 40.1 U.S.gal. |")
    assert out == "| Capacity | Transmission fluid | 38.0 L |\n|---|---|---|\n| | | 10.0 U.S.gal. (40.1 U.S. qt) |"


def test_misprint_with_the_litre_symbol_lost():  # real text from Chapter 6's PDF text layer
    out = fix("Capacity Transmission\r\nfluid\r\n38.0 \r\n40.1 U.S.gal.\r\nIMPORTANT")
    assert out == "Capacity Transmission\r\nfluid\r\n38.0 L \r\n10.0 U.S.gal. (40.1 U.S. qt)\r\nIMPORTANT"


def test_bare_numbers_without_the_quarts_signature_are_left_alone():
    for s in ["Model 2\n12 US gal tank", "Step 4 - 5 gallons", "38.0 \n10.0 U.S.gal.", "2024 / 40 gal"]:
        assert fix(s) == s


def test_unexplained_misprint_is_recalculated():
    out = fix("Tank 50 L / 31 US gal")
    assert out.startswith("Tank 50 L / 13.2 US gal [") and "31" not in out


def test_consistent_pairs_are_left_alone():
    assert fix("38.0 ℓ<br>10.0 U.S.gal.") == "38.0 L<br>10.0 U.S.gal."
    assert fix("Fuel tank: 50 L (13.2 US gal)") == "Fuel tank: 50 L (13.2 US gal)"
    assert fix("10.0 U.S. gallons / 38 litres") == "10.0 U.S. gallons / 38 litres"
    assert fix("DEF: 15.5 L / 3.4 imp. gal") == "DEF: 15.5 L / 3.4 imp. gal"


def test_gallons_first_misprint():
    out = fix("Capacity 40.1 US gal (38.0 L)")
    assert out.startswith("Capacity 10.0 US gal (38.0 L") and "40.1 U.S. qt" in out


UNRELATED = ["R2 bulbs 12 volt and 40/45 watt", "adjust the length (L) of feedback rod", "M12 x 40 (four places)",
             "25 ~ 40 mm<br>0.984 ~ 1.574 in.", "4-40 L. SHAFT 2", "Torque 40 lbf·ft"]


def test_unrelated_numbers_untouched():
    for s in UNRELATED:
        assert fix(s) == s


# ---- find(): the misprints fix() corrects, listed at upload ----

KIOTI = [  # the real KIOTI texts above: one quarts misprint each
    "| Capacity | Transmission fluid | 38.0 ℓ / 40.1 U.S.gal |",
    "38.0 ℓ<br>40.1 U.S.gal.",
    "| Capacity | Transmission fluid | 38.0 ℓ |\n|---|---|---|\n| | | 40.1 U.S.gal. |",
    "Capacity Transmission\r\nfluid\r\n38.0 \r\n40.1 U.S.gal.\r\nIMPORTANT",
    "Capacity 40.1 US gal (38.0 L)",
]
FINE = ["38.0 ℓ<br>10.0 U.S.gal.", "Fuel tank: 50 L (13.2 US gal)", "10.0 U.S. gallons / 38 litres",
        "DEF: 15.5 L / 3.4 imp. gal", "Model 2\n12 US gal tank", "Step 4 - 5 gallons", "38.0 \n10.0 U.S.gal.",
        "2024 / 40 gal", *UNRELATED]


def test_find_lists_the_kioti_misprints():
    assert find(KIOTI[0]) == [{"page": None, "kind": "quarts-printed-as-gallons", "printed": "38.0 ℓ / 40.1 U.S.gal",
                               "corrected": "38.0 ℓ / 10.0 U.S.gal (40.1 U.S. qt)",
                               "context": "Capacity Transmission fluid"}]
    (split,) = find(KIOTI[2])                                                            # table rows apart
    assert (split["printed"], split["corrected"]) == ("38.0 ℓ 40.1 U.S.gal.", "38.0 ℓ 10.0 U.S.gal. (40.1 U.S. qt)")
    (lost,) = find(KIOTI[3])                                                             # the text layer lost the ℓ
    assert (lost["printed"], lost["corrected"]) == ("38.0 40.1 U.S.gal.", "38.0 L 10.0 U.S.gal. (40.1 U.S. qt)")
    assert lost["context"] == "Capacity Transmission fluid"
    for s in KIOTI:
        assert [f["kind"] for f in find(s)] == ["quarts-printed-as-gallons"], s


def test_find_and_fix_cannot_disagree():
    """They run the same code: a text has findings exactly when fix() changes more than
    the ℓ symbol, and fix() writes what the findings say it corrects to."""
    for s in KIOTI + FINE + ["Tank 50 L / 31 US gal", "Tank 50 L / 11.0 US gal"]:
        found = find(s)
        assert bool(found) == (fix(s) != SCRIPT_L.sub(" L", s)), s
        for f in found:
            assert " ".join(SCRIPT_L.sub(" L", f["corrected"]).split()) in " ".join(re.sub(r"[|]|-{3,}|<br>", " ", fix(s)).split())
    assert find("") == [] and all(find(s) == [] for s in FINE)


def test_each_kind_of_misprint():
    assert [f["kind"] for f in find("Tank 50 L / 31 US gal")] == ["inconsistent-pair"]
    assert [f["kind"] for f in find("Tank 50 L / 11.0 US gal")] == ["imperial-printed-as-us-gallons"]
    assert fix("Tank 50 L / 11.0 US gal") == "Tank 50 L / 13.2 US gal (11 imp. gal)"


def test_findings_carry_their_page_when_the_text_has_pages():
    text = "Contents\fFuel tank: 50 L (13.2 US gal)\f\fTransmission fluid 38.0 ℓ / 40.1 U.S.gal"
    (f,) = find(text)
    assert f["page"] == 4 and f["context"] == "Transmission fluid"                       # the context stays on its page
    assert find(KIOTI[0], page=7)[0]["page"] == 7 and find(KIOTI[0])[0]["page"] is None
    assert fix(text).count("\f") == 3                                                     # page breaks are kept


def test_pairs_side_by_side_in_a_table_are_not_mixed_up():
    """Real text from Chapter 9 (two models, one cell each): 2.4 gal belongs to 8.9 ℓ.
    Before, the gallons-first rule paired it with the next cell's 8.3 ℓ and "corrected"
    it to 2.2 gal, a misprint note included."""
    cabin = "| | Coolant Capacity | 8.9 ℓ, 2.4 U.S.gal | 8.3 ℓ, 2.2 U.S.gal |"
    assert fix(cabin) == SCRIPT_L.sub(" L", cabin) and find(cabin) == []
    mirrored = "| | Coolant Capacity | 2.4 U.S.gal, 8.9 ℓ | 2.2 U.S.gal, 8.3 ℓ |"
    assert fix(mirrored) == SCRIPT_L.sub(" L", mirrored) and find(mirrored) == []
    # A misprint in the second cell is still found, whichever unit comes first.
    assert [f["printed"] for f in find("| 8.9 ℓ, 2.4 U.S.gal | 8.3 ℓ, 8.8 U.S.gal |")] == ["8.3 ℓ, 8.8 U.S.gal"]
    assert [f["printed"] for f in find("| 2.4 U.S.gal, 8.9 ℓ | 8.8 U.S.gal, 8.3 ℓ |")] == ["8.8 U.S.gal, 8.3 ℓ"]
    assert [f["printed"] for f in find("| 38.0 ℓ / 40.1 U.S.gal | 30.0 ℓ / 7.9 U.S.gal |")] == ["38.0 ℓ / 40.1 U.S.gal"]


def test_the_context_reaches_into_the_chunk_before():
    """A long paragraph is cut into chunks anywhere: the context then starts in the chunk
    before (on the same page), as the KIOTI-like staging manual showed."""
    (f,) = scan(["Engine notes.\n\nFuel tank (export model", ") ..... 50 L / 11.0 US gal"])
    assert f["context"] == "Engine notes. Fuel tank (export model ) ....."
    (f,) = scan(["Fuel tank", "\f50 L / 11.0 US gal"])                                  # not across a page break
    assert f["context"] == "" and f["page"] == 2


def test_scan_counts_pages_across_a_documents_chunks():
    """Chunks keep the page breaks inside them, so pages carry on from chunk to chunk."""
    chunks = ["Contents", "\fIntro.\n\n\fCapacity 38.0 ℓ / 40.1 U.S.gal", "More: 19.0 L / 20.05 U.S.gal", "\fx"]
    assert [(f["page"], f["printed"]) for f in scan(chunks)] == [(3, "38.0 ℓ / 40.1 U.S.gal"),
                                                                 (3, "19.0 L / 20.05 U.S.gal")]
    assert [f["page"] for f in scan([KIOTI[0], KIOTI[1]])] == [None, None]               # no pages to count
    assert scan([]) == []
