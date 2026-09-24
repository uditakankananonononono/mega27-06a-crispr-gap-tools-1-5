from crisprgap.baselines import mit_score


def test_mit_perfect_match_is_one():
    assert mit_score("ACGT" * 5, "ACGT" * 5) == 1.0


def test_mit_more_mismatches_lower_score():
    g = "A" * 20
    one_mm = "A" * 10 + "C" + "A" * 9
    three_mm = "A" * 8 + "C" + "A" * 3 + "C" + "A" * 6 + "C"  # mismatches at 8, 12, 19
    assert 0 < mit_score(g, three_mm) < mit_score(g, one_mm) < 1.0


def test_mit_seed_region_mismatch_penalized_more():
    g = "A" * 20
    pam_proximal = "A" * 19 + "C"        # position 19: high Hsu weight
    pam_distal = "C" + "A" * 19          # position 0: zero weight
    assert mit_score(g, pam_proximal) < mit_score(g, pam_distal)
