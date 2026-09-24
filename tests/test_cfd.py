import pytest
from crisprgap.cfd import cfd_score, _mm_pam_scores

WT = "GAGTCCGAGCAGAAGAAGAAGGG"  # guide GAGTCCGAGCAGAAGAAGAA + GGG PAM context


def test_identity_is_one():
    assert cfd_score(WT, WT) == pytest.approx(1.0)


def test_single_mismatch_below_one():
    off = WT[:18] + "T" + WT[19:]
    assert 0.0 < cfd_score(WT, off) < 1.0


def test_pam_proximal_hurts_more_than_distal():
    mm, pam = _mm_pam_scores()
    assert mm["rA:dC,20"] < mm["rA:dC,1"]  # Doench 2016 position effect


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


def test_cfd_applicable_rejects_indels_and_bad_alphabet():
    from crisprgap.cfd import cfd_applicable
    good_wt = "ACGT" * 5 + "AGG"   # 23-mer, NGG PAM
    good_off = "ACGT" * 5 + "AGG"
    assert cfd_applicable(good_wt, good_off)
    assert not cfd_applicable(good_wt, "ACGT-ACGTACGTACGTACG" + "AGG"[:1] + "GG")
    assert not cfd_applicable(good_wt, "N" * 23)
    assert not cfd_applicable("ACGT", good_off)  # wrong length
