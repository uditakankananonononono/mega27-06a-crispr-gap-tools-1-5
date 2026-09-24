import pytest
from crispr_gap_tools.baselines.cfd import cfd_score, _mm_pam_scores

WT = "GAGTCCGAGCAGAAGAAGAAGGG"  # guide GAGTCCGAGCAGAAGAAGAA + GGG PAM context


def test_identity_is_one():
    assert cfd_score(WT, WT) == pytest.approx(1.0)


def test_single_mismatch_below_one():
    off = WT[:18] + "T" + WT[19:]  # mismatch at guide position 19, PAM intact
    s = cfd_score(WT, off)
    assert 0.0 < s < 1.0


def test_pam_proximal_hurts_more_than_distal():
    mm, pam = _mm_pam_scores()
    # Doench 2016: activity drops as mismatches approach the PAM (position 20).
    assert mm["rA:dC,20"] < mm["rA:dC,1"]


def test_canonical_pam_is_max():
    _, pam = _mm_pam_scores()
    assert pam["GG"] == 1.0


def test_two_mismatches_compound():
    off1 = WT[:18] + "T" + WT[19:]
    off2 = WT[:2] + "T" + WT[3:18] + "T" + WT[19:]
    assert cfd_score(WT, off2) < cfd_score(WT, off1)


def test_rejects_non_23mer():
    with pytest.raises(ValueError):
        cfd_score("ACGT", "ACGT")
