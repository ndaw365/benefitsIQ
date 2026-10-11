"""Text cleaning applied to every extracted PDF page."""

from src.rag.load import clean


def test_dot_leaders_become_colon():
    assert clean("Paraplegia .................. 50% of Amount") == "Paraplegia: 50% of Amount"


def test_dot_leader_without_spaces():
    assert clean("Definitions..........3") == "Definitions: 3"


def test_short_dot_runs_kept():
    assert clean("90 days ; or... the date. Ends.") == "90 days ; or... the date. Ends."


def test_whitespace_collapsed():
    assert clean("  line one\n  line\ttwo  ") == "line one line two"
