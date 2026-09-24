import numpy as np
from crispr_gap_tools import encoding as enc


def test_revcom():
    assert enc.revcom("ACGT") == "ACGT"
    assert enc.revcom("AAAA") == "TTTT"
    assert enc.revcom("GAGTCCGAGCAGAAGAAGAA") == "TTCTTCTTCTGCTCGGACTC"


def test_one_hot_shape_and_rows():
    x = enc.one_hot("ACGT")
    assert x.shape == (4, 4)
    assert np.allclose(x.sum(axis=1), 1.0)
    assert x[0, 0] == 1 and x[1, 1] == 1 and x[2, 2] == 1 and x[3, 3] == 1


def test_rna_equals_dna():
    assert np.array_equal(enc.one_hot("ACGU"), enc.one_hot("ACGT"))


def test_gc():
    assert enc.gc_content("GGCC") == 1.0
    assert enc.gc_content("AT") == 0.0


def test_mismatch_mask_and_pair():
    m = enc.pair_mismatch_mask("AAAA", "AATA")
    assert m.tolist() == [False, False, True, False]
    assert enc.encode_pair("AAAA", "AATA").shape == (4, 8)
