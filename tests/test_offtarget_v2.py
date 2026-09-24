import numpy as np
import torch
from crisprgap.models.offtarget_v2 import OffTargetGNNv2, pam_one_hot, global_features
from crisprgap.models.offtarget_gnn import duplex_to_graph, collate_graphs


def test_pam_one_hot():
    assert pam_one_hot("AGG").sum() == 3
    assert pam_one_hot("").sum() == 0
    assert pam_one_hot("AGG").shape == (12,)
    assert not np.array_equal(pam_one_hot("AGG"), pam_one_hot("TGG"))


def test_model_with_and_without_globals():
    x, ei = duplex_to_graph("ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGA")
    xb, eib, batch = collate_graphs([(x, ei), (x, ei)])
    m0 = OffTargetGNNv2(n_global=0)
    assert m0(xb, eib, batch, torch.zeros(2, 0)).shape == (2,)
    m5 = OffTargetGNNv2(n_global=17)
    assert m5(xb, eib, batch, torch.zeros(2, 17)).shape == (2,)
