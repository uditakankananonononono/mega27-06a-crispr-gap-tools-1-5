

def test_crisproff_scorer_sanity():
    from crisprgap.crisproff import crisproff_score
    on = crisproff_score("GCCTCTTTCCCACCCACCTTGGG", "GCCTCTTTCCCACCCACCTTGGG")
    off = crisproff_score("GCCTCTTTCCCACCCACCTTGGG", "GTCTCTTTCCCAGCGACCTGGGG")
    assert on > 0 > off  # perfect match binds, 3-mismatch pair does not
