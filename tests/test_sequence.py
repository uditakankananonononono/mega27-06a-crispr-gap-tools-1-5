import numpy as np

from crisprgap.sequence import (reverse_complement, one_hot, mismatch_mask,
                                gc_content, dinucleotide_features)


def test_reverse_complement():
    assert reverse_complement("ACGT") == "ACGT"
    assert reverse_complement("AAAA") == "TTTT"
    assert reverse_complement("aCgT") == "ACGT"


def test_one_hot_shape_and_values():
    arr = one_hot("ACGTN")
    assert arr.shape == (4, 5)
    assert arr[:, 0].tolist() == [1, 0, 0, 0]
    assert arr[:, 3].tolist() == [0, 0, 0, 1]
    assert arr[:, 4].sum() == 0.0  # ambiguous base


def test_one_hot_padding():
    arr = one_hot("AC", length=5)
    assert arr.shape == (4, 5)
    assert arr[:, 2:].sum() == 0.0


def test_mismatch_mask():
    m = mismatch_mask("AAAA", "AATA")
    assert m.tolist() == [0, 0, 1, 0]


def test_gc_content():
    assert gc_content("GGCC") == 1.0
    assert gc_content("AATT") == 0.0
    assert gc_content("") == 0.0


def test_dinucleotide_features_sum_to_one():
    f = dinucleotide_features("ACGTACGT")
    assert f.shape == (16,)
    assert abs(f.sum() - 1.0) < 1e-6
