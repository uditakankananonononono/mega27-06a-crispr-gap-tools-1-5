import os


def test_interactive_html():
    p = "docs/transfer_matrix_interactive.html"
    assert os.path.getsize(p) > 4_000_000
    html = open(p).read()
    for name in ("fcres", "dc_hek293t", "horlbeck_a"):
        assert name in html
    assert "Cross-dataset gRNA efficacy transfer" in html
