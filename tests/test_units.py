import re

from app.units import fix


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


def test_unrelated_numbers_untouched():
    for s in ["R2 bulbs 12 volt and 40/45 watt", "adjust the length (L) of feedback rod",
              "M12 x 40 (four places)", "25 ~ 40 mm<br>0.984 ~ 1.574 in.", "4-40 L. SHAFT 2",
              "Torque 40 lbf·ft"]:
        assert fix(s) == s
